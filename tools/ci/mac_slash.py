#!/usr/bin/env python3
"""Pull request comments that start a self-hosted Mac lane (.github/workflows/mac-slash.yml, docs/ci.md).

    /mac-test [--shader-validation] [RUNNER ...] [COMMIT]          the test tiers on bare-metal Macs (mac-gpu.yml)
    /release-gate [--expect-output-change] [RUNNER ...] [COMMIT]   against the last release (release-gate.yml)

RUNNER is a label from the repository variable ``MAC_CI_RUNNERS`` (a JSON list, e.g. ``["m5_max_40c"]``); without one
the lane runs on every listed runner. The workflow that calls this runs from the default branch, never from the pull
request, because an ``issue_comment`` job holds write permissions. The commenter needs write access. COMMIT (the full
40-digit SHA: a short prefix can be matched by a commit made to match it) is the head the commenter read, and it is
required when the pull request's author cannot write here: this job reads the head after the comment, and a push in
between must not run unread code on a Mac. The command pins that head commit, sets a pending ``<lane> (<runner>)``
commit status on it per runner, and dispatches the lane's workflow on the default branch with that commit as input; the
lane's report job sets the final status. Everything it reads from the event arrives through the environment.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from github_api import GitHub  # noqa: E402

COMMANDS = {
    "/mac-test": {"workflow": "mac-gpu.yml", "context": "mac-gpu",
                  "flags": {"--shader-validation": "shader_validation"}},
    "/release-gate": {"workflow": "release-gate.yml", "context": "release-gate",
                      "flags": {"--expect-output-change": "expect_output_change"}},
}
WRITE_ROLES = ("admin", "maintain", "write")


@dataclass
class Command:
    name: str
    workflow: str
    context: str
    flags: Dict[str, str] = field(default_factory=dict)
    runners: List[str] = field(default_factory=list)
    commit: str = ""


def parse(body: str) -> Optional[Command]:
    """The command on the comment's first line, or None. Raises ValueError for an unknown flag."""
    line = next((l.strip() for l in body.splitlines() if l.strip()), "")
    words = line.split()
    if not words or words[0] not in COMMANDS:
        return None
    spec = COMMANDS[words[0]]
    cmd = Command(words[0], spec["workflow"], spec["context"])
    for word in words[1:]:
        if word.startswith("-"):
            if word not in spec["flags"]:
                raise ValueError(f"unknown option `{word}` (known: {', '.join(spec['flags']) or 'none'})")
            cmd.flags[spec["flags"][word]] = "true"
        elif re.fullmatch(r"[0-9a-f]{7,40}", word):
            if len(word) != 40:
                raise ValueError(f"name the commit by its full 40-digit SHA, not `{word}`")
            if cmd.commit and cmd.commit != word:
                raise ValueError("name one commit")
            cmd.commit = word
        elif word not in cmd.runners:
            cmd.runners.append(word)
    return cmd


def select_runners(cmd: Command, configured: List[str]) -> List[str]:
    """The runners to dispatch to. Raises ValueError when none is configured or one named is unknown."""
    if not configured:
        raise ValueError("no Mac runner is configured (repository variable MAC_CI_RUNNERS)")
    unknown = [r for r in cmd.runners if r not in configured]
    if unknown:
        raise ValueError(f"unknown runner {', '.join(unknown)} (configured: {', '.join(configured)})")
    return cmd.runners or list(configured)


def can_write(gh: GitHub, login: str) -> bool:
    role = gh.get(f"/collaborators/{login}/permission")
    return role.get("role_name") in WRITE_ROLES or role.get("permission") in ("admin", "write")


def handle(gh: GitHub, env: Dict[str, str]) -> str:
    """Act on one comment; returns the reply posted on the pull request ('' when the comment is not a command)."""
    number, actor = env["PR_NUMBER"], env["ACTOR"]
    try:
        cmd = parse(env.get("COMMENT_BODY", ""))
    except ValueError as e:
        return reply(gh, number, f"@{actor}: {e}.")
    if cmd is None:
        return ""
    if not can_write(gh, actor):
        return reply(gh, number, f"@{actor}: `{cmd.name}` needs write access to this repository.")
    pr = gh.get(f"/pulls/{number}")
    if pr.get("state") != "open" or pr.get("draft"):
        return reply(gh, number, f"@{actor}: `{cmd.name}` runs on open pull requests that are ready for review.")
    try:
        runners = select_runners(cmd, json.loads(env.get("MAC_CI_RUNNERS") or "[]"))
    except ValueError as e:
        return reply(gh, number, f"@{actor}: {e}.")
    sha = pr["head"]["sha"]
    if cmd.commit and sha != cmd.commit:
        return reply(gh, number, f"@{actor}: the head is {sha[:10]}, not {cmd.commit[:10]}; read it and comment again.")
    if not cmd.commit and not can_write(gh, pr["user"]["login"]):
        return reply(gh, number, f"@{actor}: the author of this pull request cannot write here, so name the commit you "
                                 f"read by its full SHA: `{cmd.name} <sha>` (the head is {sha[:10]}).")
    url = f"{env.get('GITHUB_SERVER_URL', 'https://github.com')}/{gh.repo}/actions/workflows/{cmd.workflow}"
    for runner in runners:
        context = f"{cmd.context} ({runner})"
        gh.post(f"/statuses/{sha}", {"state": "pending", "context": context,
                                     "description": f"queued by @{actor}", "target_url": url})
        try:
            gh.post(f"/actions/workflows/{cmd.workflow}/dispatches", {
                "ref": env["DEFAULT_BRANCH"],
                "inputs": {"git_ref": sha, "runner": runner, "pr": str(number), **cmd.flags}})
        except Exception as e:
            # No run will report this context: it must not stay pending.
            gh.post(f"/statuses/{sha}", {"state": "error", "context": context,
                                         "description": f"could not start the run: {e}"[:140], "target_url": url})
            raise
    if env.get("COMMENT_ID"):
        gh.post(f"/issues/comments/{env['COMMENT_ID']}/reactions", {"content": "rocket"})
    options = "".join(f" `{k}`" for k in cmd.flags)
    return reply(gh, number, f"`{cmd.name}`{options} dispatched for {sha[:10]} on {', '.join(runners)}: the result "
                             f"appears as the `{cmd.context} (<runner>)` checks ([runs]({url})).")


def reply(gh: GitHub, number: str, text: str) -> str:
    gh.post(f"/issues/{number}/comments", {"body": text})
    return text


def main() -> int:
    env = dict(os.environ)
    gh = GitHub(env["GITHUB_REPOSITORY"], env["GH_TOKEN"], env.get("GITHUB_API_URL", "https://api.github.com"))
    try:
        print(handle(gh, env) or "not a command")
    except urllib.error.HTTPError as e:
        print(f"GitHub API: {e.code} {e.reason}: {e.read()[:500]!r}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
