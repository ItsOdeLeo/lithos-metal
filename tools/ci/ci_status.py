#!/usr/bin/env python3
"""Set a self-hosted lane's commit statuses from its runners' results (the report jobs of mac-gpu.yml and
release-gate.yml).

    python3 tools/ci/ci_status.py --context mac-gpu --sha SHA --runners '["m5_max_40c"]' --results DIR \
        --artifact-prefix mac-gpu- --run-url URL

Each runner's job uploads ``result.json`` as the artifact ``<prefix><runner>``; the report job downloads them all
under DIR. One status per runner, ``<context> (<runner>)``: success when the lane passed, failure when a test failed,
error when the lane could not run or left no result (its plan or job failed, was cancelled, never ran). This runs on a
hosted runner, so the token that writes statuses never reaches the Mac that ran the code under test.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from github_api import GitHub  # noqa: E402

STATES = {"passed": "success", "failed": "failure"}   # anything else (could not run, inconclusive): error


def status_for(result: Optional[dict]) -> Tuple[str, str]:
    """(state, description) of one runner's result; description fits GitHub's 140 characters."""
    if result is None:
        return "error", "no result: the plan or the Mac job failed, was cancelled, timed out or never started"
    state = STATES.get(result.get("status"), "error")
    if result.get("error"):
        text = f"could not run: {result['error']}"
    elif result.get("summary"):
        text = result["summary"]
    else:
        problems = [f"{t['tier']}: {p}" for t in result.get("tiers", []) for p in t.get("problems", [])]
        if problems:
            text = problems[0] + (f" (+{len(problems) - 1} more)" if len(problems) > 1 else "")
        else:
            ran = sum(t["tests"] - t["skipped"] for t in result.get("tiers", []))
            text = f"{ran} tests passed in {len(result.get('tiers', []))} tiers"
    return state, text[:140]


def superseded(gh: GitHub, sha: str, context: str, started: str) -> bool:
    """Whether ``context`` on ``sha`` was set after this run started (``started``, ISO 8601): a newer `/mac-test`
    marked it pending, or a newer run reported. Asked right before each write, since a cancellation reaches a running
    step late."""
    statuses = gh.get(f"/commits/{sha}/statuses?per_page=100")   # newest first
    newest = next((s for s in statuses if s["context"] == context), None)
    return newest is not None and newest["created_at"] > started


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--context", required=True)
    ap.add_argument("--sha", required=True)
    ap.add_argument("--runners", required=True, help="JSON list of runner labels")
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--artifact-prefix", required=True)
    ap.add_argument("--run-url", required=True)
    args = ap.parse_args(argv)
    gh = GitHub(os.environ["GITHUB_REPOSITORY"], os.environ["GH_TOKEN"],
                os.environ.get("GITHUB_API_URL", "https://api.github.com"))
    started = ""
    if os.environ.get("GITHUB_RUN_ID"):
        try:
            started = gh.get(f"/actions/runs/{os.environ['GITHUB_RUN_ID']}")["run_started_at"]
        except Exception as e:   # report anyway; only the check against newer runs is lost
            print(f"::warning::could not read this run's start time: {e}")
    worst = 0
    for runner in json.loads(args.runners):
        context = f"{args.context} ({runner})"
        path = args.results / f"{args.artifact_prefix}{runner}" / "result.json"
        result = json.loads(path.read_text()) if path.exists() else None
        state, description = status_for(result)
        if started and superseded(gh, args.sha, context, started):
            print(f"{context}: {state}: not set, a newer run owns the status")
            continue
        print(f"{context}: {state}: {description}")
        gh.post(f"/statuses/{args.sha}", {"state": state, "context": context,
                                          "description": description, "target_url": args.run_url})
        worst = max(worst, 0 if state == "success" else 1)
    return worst


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
