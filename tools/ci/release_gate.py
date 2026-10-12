#!/usr/bin/env python3
"""The release gate: this build against the previous release on real models, on a bare-metal Mac (docs/ci.md).

    python3 tools/ci/release_gate.py --root <dir> [--baseline-ref auto|vX.Y.Z] [--candidate-bundle <dir>]
                                     [--only CASE ...] [--expect-output-change] [--report-dir <dir>]

Installs two release bundles the way their Homebrew formula does, each in its own environment under ``<root>/gate``: the
candidate (the bundle release-gate.yml built on hosted macOS for this commit, the bytes a tag of it publishes; without
``--candidate-bundle`` one built here with tools/release/build_macos.py) and the baseline (``auto``: the newest ``v*``
tag behind this commit that has a published release, downloaded from that release and checked against its formula). Then
for every case in ``tools/ci/release_cases.json`` that fits this machine's memory it serves each build in turn, ABBA
(baseline, candidate, candidate, baseline), through the public ``lithos-metal serve`` and its OpenAI API, and sends the
prompt set greedily, streaming. A case passes when

* every completion is identical between the two runs of one build (otherwise the measurement is inconclusive) and,
  unless ``--expect-output-change``, between the builds;
* decode speed (the tokens generated between the first and last streamed text, per second, as both servers count
  them per chunk; a server that does not, such as a release before per-chunk usage, leaves decode uncompared, which
  the summary says) and time to first token on the longest prompt regress by at most the larger of
  ``--max-regression`` and twice the runs' own spread, and that spread is within ``--max-spread`` (otherwise
  inconclusive).

Models download from the Hub at pinned revisions into ``<root>/gate/hub``, and each build packs them with its own packer
into ``packs`` beside its environment, all afresh every gate. The measurement runs inside ``$LITHOS_CI_GPU_WRAPPER``
when set.
Writes ``<report-dir>/result.json`` (with each case's numbers) and the servers' logs. Exit status: 0 passed;
1 an output changed or a metric regressed; 3 could not run or inconclusive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mac_gpu import fresh_dir  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CASES = Path(__file__).resolve().with_name("release_cases.json")
MIN_FREE_GB = 130          # the gate downloads every case's checkpoints, and both builds pack them
BOOT_TIMEOUT = 3600          # a first boot downloads and packs the checkpoint
EXIT_FAIL, EXIT_COULD_NOT_RUN = 1, 3


class CouldNotRun(Exception):
    pass


def sh(cmd: List[str], **kw) -> str:
    print("+ " + shlex.join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode:
        raise CouldNotRun(f"{shlex.join(cmd[:4])} … exited {r.returncode}: {r.stderr[-800:]}")
    return r.stdout


# ---- builds -----------------------------------------------------------------------------------------------------

def published(tag: str) -> bool:
    """Whether ``tag`` has a published GitHub release: a tag release.yml refused published nothing, so it is no
    baseline. Reads the public API (with ``$GH_TOKEN`` when set, for the rate limit)."""
    repo = os.environ.get("GITHUB_REPOSITORY", "lithos-ai/lithos-metal")
    api = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    req = urllib.request.Request(f"{api}/repos/{repo}/releases/tags/{tag}", headers={
        "Accept": "application/vnd.github+json", **({"Authorization": f"Bearer {token}"} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return not json.load(r).get("draft")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise CouldNotRun(f"could not read the release of {tag}: HTTP {e.code}")
    except urllib.error.URLError as e:
        raise CouldNotRun(f"could not read the release of {tag}: {e.reason}")


def resolve_baseline(repo: Path, ref: str, is_published=published) -> str:
    """``ref``, or for ``auto`` the newest published release tag behind HEAD (not HEAD's own)."""
    if ref != "auto":
        return ref
    head = sh(["git", "rev-parse", "HEAD"], cwd=repo).strip()
    for tag in sh(["git", "tag", "--merged", "HEAD", "--sort=-v:refname", "--list", "v*"], cwd=repo).split():
        if sh(["git", "rev-list", "-n", "1", tag], cwd=repo).strip() != head and is_published(tag):
            return tag
    raise CouldNotRun("no published release behind this commit to compare with (fetch the tags or pass --baseline-ref)")


def download(url: str, dest: Path) -> Path:
    try:
        with urllib.request.urlopen(url, timeout=300) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
    except urllib.error.URLError as e:
        raise CouldNotRun(f"could not download {url}: {e}")
    return dest


def bundle_digest(archive: Path) -> str:
    """The first 12 hex digits of a bundle's SHA-256: the commit status names the bytes the gate measured."""
    return hashlib.sha256(archive.read_bytes()).hexdigest()[:12]


def formula_bundle(formula: str) -> Tuple[str, str]:
    """(archive file name, SHA-256) of a Homebrew formula written by tools/release/build_macos.py."""
    url = re.search(r'^\s*url "([^"]+)"', formula, re.M)
    digest = re.search(r'^\s*sha256 "([0-9a-f]{64})"', formula, re.M)
    if not url or not digest:
        raise CouldNotRun("the release's formula names no bundle and SHA-256")
    return url[1].rsplit("/", 1)[-1], digest[1]


def baseline_bundle(tag: str, gate: Path) -> Path:
    """The previous release as users install it: its published bundle, checked against its published formula."""
    base = f"https://github.com/{os.environ.get('GITHUB_REPOSITORY', 'lithos-ai/lithos-metal')}/releases/download/{tag}"
    home = gate / "baseline-dist"
    home.mkdir()
    name, digest = formula_bundle(download(f"{base}/lithos-metal.rb", home / "lithos-metal.rb").read_text())
    archive = download(f"{base}/{name}", home / name)
    got = hashlib.sha256(archive.read_bytes()).hexdigest()
    if got != digest:
        raise CouldNotRun(f"{tag}'s bundle has SHA-256 {got}, its formula says {digest}")
    return archive


def candidate_bundle(repo: Path, given: Optional[Path], gate: Path, python: str) -> Path:
    """The bundle a tag of this commit publishes: the one release-gate.yml built on hosted macOS
    (``--candidate-bundle``), else one built here with tools/release/build_macos.py."""
    if given is None:
        given = gate / "candidate-dist"
        sh([python, str(repo / "tools/release/build_macos.py"), "--out", str(given)], cwd=str(repo))
    archives = sorted(Path(given).glob("lithos-metal-*-macos-arm64.tar.gz"))
    if len(archives) != 1:
        raise CouldNotRun(f"expected one lithos-metal-*-macos-arm64.tar.gz in {given}, found {len(archives)}")
    return archives[0]


def install_bundle(archive: Path, home: Path, python: str) -> Path:
    """A bundle installed as its Homebrew formula installs it: a new environment, the project wheel and its serving
    dependencies from the bundle's own wheels (no index, no compiler)."""
    home.mkdir()
    with tarfile.open(archive) as tar:
        tar.extractall(home, filter="data")
    wheels = next(home.glob("lithos-metal-*/wheels"))
    venv, wheel = home / "venv", next(wheels.glob("lithos_metal-*.whl"))
    sh([python, "-m", "venv", str(venv)])
    sh([str(venv / "bin" / "python"), "-m", "pip", "install", "-q", "--no-index", f"--find-links={wheels}",
        f"{wheel}[serve]"])
    sh([str(venv / "bin" / "python"), "-c", "from monolith.runtime import is_available; assert is_available()"],
       cwd=str(home))
    return venv / "bin" / "lithos-metal"


# ---- one served run ---------------------------------------------------------------------------------------------

def render_prompt(prompt: dict) -> List[dict]:
    rep = prompt.get("repeat")
    filler = "".join(rep["text"].replace("{i}", str(i)) for i in range(1, rep["times"] + 1)) if rep else ""
    return [{**m, "content": m["content"].replace("{repeat}", filler)} for m in prompt["messages"]]


def available_gb() -> float:
    """Memory macOS can hand out without paging: free, inactive, speculative and purgeable pages."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+) bytes", out).group(1))
    pages = 0
    for key in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
        m = re.search(rf"{key}:\s+(\d+)", out)
        pages += int(m.group(1)) if m else 0
    return pages * page / 1e9


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http_json(url: str, body: Optional[dict] = None, timeout: float = 10.0) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def stream_chat(base: str, model: str, messages: List[dict], max_tokens: int, continuous: bool = False) -> dict:
    """One greedy streamed completion. With ``continuous`` (vLLM's continuous_usage_stats) every chunk carries the
    tokens so far, so ``interval_tokens`` counts exactly the tokens generated between the first and last text chunks
    (``decode_s``): a first chunk can hold several tokens (one publish per few steps, or a whole speculative round)."""
    options = {"include_usage": True, **({"continuous_usage_stats": True} if continuous else {})}
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": options}
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = last = None
    parts, usage, counts = [], None, []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            event = json.loads(line[5:])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                text = (choice.get("delta") or {}).get("content")
                if text:
                    now = time.perf_counter()
                    first = first or now
                    last = now
                    parts.append(text)
                    if continuous and event.get("usage"):
                        counts.append(int(event["usage"]["completion_tokens"]))
    if first is None or not usage:
        raise CouldNotRun("a completion streamed no text or no usage")
    tokens = int(usage["completion_tokens"])
    return {"text": "".join(parts), "tokens": tokens, "ttft_s": first - t0,
            "decode_s": last - first, "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "interval_tokens": counts[-1] - counts[0] if continuous and counts else None}


def serve_and_measure(exe: Path, case: dict, prompts: List[dict], packs: Path, log: Path) -> dict:
    need = case.get("min_memory_gb", 0)
    for _ in range(60):                              # the previous server's memory returns gradually
        if available_gb() >= need:
            break
        time.sleep(5)
    else:
        raise CouldNotRun(f"{case['name']}: {available_gb():.0f} GB available, the case needs {need}")
    port = free_port()
    cmd = [str(exe), "serve", *case["args"], "--pack", str(packs), "--port", str(port)]
    print("+ " + shlex.join(cmd), flush=True)
    with open(log, "w") as out:
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, cwd=log.parent, start_new_session=True)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + BOOT_TIMEOUT
        while True:
            if proc.poll() is not None:
                raise CouldNotRun(f"{case['name']}: the server exited {proc.returncode} while starting ({log.name})")
            try:
                model = http_json(f"{base}/v1/models")["data"][0]["id"]
                break
            except (urllib.error.URLError, OSError, KeyError, IndexError, ValueError):
                if time.time() > deadline:
                    raise CouldNotRun(f"{case['name']}: the server did not start in {BOOT_TIMEOUT} s")
                time.sleep(5)
        hello = [{"role": "user", "content": "Say hello."}]
        try:                                         # warm-up, not measured; also asks for per-chunk usage
            stream_chat(base, model, hello, 16, continuous=True)
            continuous = True
        except urllib.error.HTTPError:               # a release from before continuous_usage_stats
            stream_chat(base, model, hello, 16)
            continuous = False
        results = {p["id"]: stream_chat(base, model, render_prompt(p), p["max_tokens"], continuous) for p in prompts}
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    return results


# ---- the verdict ------------------------------------------------------------------------------------------------

def metrics(run: Dict[str, dict]) -> Dict[str, float]:
    """Decode speed over the tokens streamed after the first chunk (when the server counted them per chunk) and the
    longest prompt's time to first token."""
    out = {"ttft_long_s": max(run.values(), key=lambda r: r["prompt_tokens"])["ttft_s"]}
    if all(r.get("interval_tokens") is not None for r in run.values()):
        seconds = sum(r["decode_s"] for r in run.values())
        out["decode_tok_s"] = sum(r["interval_tokens"] for r in run.values()) / seconds if seconds > 0 else 0.0
    return out


HIGHER_IS_BETTER = {"decode_tok_s": True, "ttft_long_s": False}


def judge_case(runs: Dict[str, List[Dict[str, dict]]], expect_output_change: bool, max_regression: float,
               max_spread: float) -> dict:
    """runs = {"baseline": [run, run], "candidate": [run, run]}; each run maps prompt id -> measurement."""
    nondeterministic = [f"{build} is not deterministic on '{pid}'" for build, pair in runs.items() for pid in pair[0]
                        if len({r[pid]["text"] for r in pair}) > 1]
    problems, inconclusive = [], list(nondeterministic)
    # A build whose two runs disagree gives nothing to compare the other build's outputs or speed with.
    changed = [] if nondeterministic else [pid for pid in runs["baseline"][0]
                                           if runs["candidate"][0][pid]["text"] != runs["baseline"][0][pid]["text"]]
    if not expect_output_change:
        problems += [f"output changed on '{pid}'" for pid in changed]
    measured = {b: [metrics(r) for r in pair] for b, pair in runs.items()}
    table = {}
    for name, higher in HIGHER_IS_BETTER.items():
        if not all(name in m for ms in measured.values() for m in ms):
            continue    # decode needs per-chunk counts from both builds; an estimate would depend on chunk sizes
        values = {b: [m[name] for m in ms] for b, ms in measured.items()}
        mean = {b: sum(v) / len(v) for b, v in values.items()}
        spread = max((max(v) - min(v)) / mean[b] if mean[b] else 0.0 for b, v in values.items())
        change = (mean["candidate"] - mean["baseline"]) / mean["baseline"]
        regression = -change if higher else change
        allowed = max(max_regression, 2 * spread)
        table[name] = {"baseline": mean["baseline"], "candidate": mean["candidate"], "change": change,
                       "spread": spread, "allowed_regression": allowed, "runs": values}
        if spread > max_spread:
            inconclusive.append(f"{name} varies {spread:.1%} between runs of one build (limit {max_spread:.0%})")
        elif regression > allowed:
            problems.append(f"{name} regressed {regression:.1%} (allowed {allowed:.1%})")
    verdict = ("inconclusive" if nondeterministic else "failed" if problems
               else "inconclusive" if inconclusive else "passed")
    return {"verdict": verdict, "problems": problems + inconclusive, "metrics": table, "changed": changed}


def measure(args, exes: Dict[str, Path], cases: dict, report: Path) -> dict:
    # <home>/venv/bin/lithos-metal: each build serves the packs its own packer wrote, never the other build's
    packs = {build: Path(exe).parents[2] / "packs" for build, exe in exes.items()}
    for root in packs.values():
        root.mkdir(parents=True, exist_ok=True)
    total_gb = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 1e9
    out = []
    for case in cases["cases"]:
        if args.only and case["name"] not in args.only:
            continue
        if case.get("min_memory_gb", 0) > total_gb * 0.9:
            out.append({"name": case["name"], "verdict": "skipped",
                        "problems": [f"needs {case['min_memory_gb']} GB, this Mac has {total_gb:.0f}"]})
            continue
        runs: Dict[str, List[Dict[str, dict]]] = {"baseline": [], "candidate": []}
        for i, build in enumerate(["baseline", "candidate", "candidate", "baseline"]):
            log = report / f"{case['name']}-{i}-{build}.log"
            runs[build].append(serve_and_measure(exes[build], case, cases["prompts"], packs[build], log))
        verdict = judge_case(runs, args.expect_output_change, args.max_regression, args.max_spread)
        out.append({"name": case["name"], **verdict, "runs": runs})
        print(f"{case['name']}: {verdict['verdict']} {verdict['problems']}", flush=True)
    return {"cases": out}


def overall(cases: List[dict]) -> tuple:
    ran = [c for c in cases if c["verdict"] != "skipped"]
    if not ran:
        return "could not run", "no case fits this machine"
    for verdict in ("failed", "inconclusive"):
        bad = [c for c in ran if c["verdict"] == verdict]
        if bad:
            more = f" (+{len(bad) - 1} cases)" if len(bad) > 1 else ""
            return verdict, f"{bad[0]['name']}: {bad[0]['problems'][0]}{more}"
    speed = ", ".join(f"{c['name']} decode {c['metrics']['decode_tok_s']['change']:+.1%}" if "decode_tok_s" in
                      c["metrics"] else f"{c['name']} decode not compared (no per-chunk counts)" for c in ran)
    changed = [c["name"] for c in ran if c.get("changed")]
    if changed:                                  # --expect-output-change
        return "passed", f"{len(ran)} cases passed, outputs changed in {', '.join(changed)}; {speed}"
    return "passed", f"{len(ran)} cases identical; {speed}"


def release_refusal(statuses: List[dict], previous: str, bundle: Optional[str] = None) -> Optional[str]:
    """Why a tag may not publish this commit (release.yml), or None: it needs a ``release-gate (<runner>)`` status,
    and every one a success against ``previous``, the newest release behind it, and (given ``bundle``, the digest of
    the bundle about to be published) of that bundle. The description names both: ``vs <tag> bundle <digest>:``."""
    gates = [s for s in statuses if s["context"].startswith("release-gate (")]
    if not gates:
        return "no release-gate status on this commit"
    for s in gates:
        if s["state"] != "success":
            return f"{s['context']} is {s['state']}"
        ran = re.match(r"vs (\S+) bundle ([0-9a-f]{12}):", s.get("description") or "")
        if not ran or ran[1] != previous:
            return f"{s['context']} did not compare with {previous}"
        if bundle is not None and ran[2] != bundle:
            return f"{s['context']} gated bundle {ran[2]}, not {bundle}"
    return None


def uncovered(results: List[dict], cases: dict) -> Optional[str]:
    """Why the runners' results do not cover the release (release.yml), or None: every case of ``cases``
    (release_cases.json) passed on at least one runner. A runner skips a case its memory cannot hold, so the 27B cases
    need a runner large enough for them."""
    for name in [c["name"] for c in cases["cases"]]:
        if not any(c.get("name") == name and c.get("verdict") == "passed" for r in results for c in r.get("cases", [])):
            return f"no runner passed the case {name}"
    return None


def summary_markdown(result: dict) -> str:
    lines = [f"### release-gate on {result.get('chip', '?')}: {result['status']}",
             f"candidate {str(result.get('commit'))[:10]} vs baseline {result.get('baseline')}", "",
             "| case | verdict | decode tok/s (base → cand) | TTFT long s (base → cand) | problems |",
             "|---|---|---|---|---|"]
    def cell(m: dict, key: str, fmt: str) -> str:
        if key not in m:
            return ""
        return f"{m[key]['baseline']:{fmt}} → {m[key]['candidate']:{fmt}} ({m[key]['change']:+.1%})"

    for c in result.get("cases", []):
        m = c.get("metrics", {})
        lines.append(f"| {c['name']} | {c['verdict']} | {cell(m, 'decode_tok_s', '.1f')} | "
                     f"{cell(m, 'ttft_long_s', '.3f')} | {'; '.join(c.get('problems', [])) or 'ok'} |")
    if result.get("error"):
        lines += ["", f"Could not run: {result['error']}"]
    return "\n".join(lines) + "\n"


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="persistent CI directory on this machine")
    ap.add_argument("--repo", type=Path, default=ROOT, help="the checkout under test (with its tags)")
    ap.add_argument("--baseline-ref", default="auto", help="the release to compare with (default: the newest behind)")
    ap.add_argument("--candidate-bundle", type=Path, help="directory holding the bundle to gate "
                    "(release-gate.yml builds it on hosted macOS); default: build one here")
    ap.add_argument("--cases", type=Path, default=CASES)
    ap.add_argument("--only", nargs="*", help="run only these cases")
    ap.add_argument("--expect-output-change", action="store_true", help="outputs may differ from the baseline's")
    ap.add_argument("--max-regression", type=float, default=0.02)
    ap.add_argument("--max-spread", type=float, default=0.05)
    ap.add_argument("--report-dir", type=Path, default=None)
    ap.add_argument("--python", default=os.environ.get("LITHOS_CI_PYTHON", sys.executable))
    ap.add_argument("--measure", type=Path, help=argparse.SUPPRESS)      # internal: the locked measurement phase
    args = ap.parse_args(argv)
    root_is_link = args.root.is_symlink()     # before resolve(), which would follow it
    args.root = args.root.resolve()
    report = (args.report_dir or args.root / "reports" / time.strftime("gate-%Y%m%dT%H%M%S")).resolve()
    report.mkdir(parents=True, exist_ok=True)
    cases = json.loads(args.cases.read_text())
    if args.measure:                     # the child: measure with the builds the parent prepared
        exes = {k: Path(v) for k, v in json.loads(args.measure.read_text()).items()}
        try:
            measured = measure(args, exes, cases, report)
        except CouldNotRun as e:
            measured = {"cases": [], "error": str(e)}
        (report / "measure.json").write_text(json.dumps(measured, indent=1))
        return 0
    repo = args.repo.resolve()
    result: dict = {"status": "could not run", "cases": [], "chip": None, "baseline": None, "commit": None}
    try:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise CouldNotRun("the release gate runs on Apple silicon macOS only")
        result["chip"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                        text=True).stdout.strip()
        result["commit"] = sh(["git", "rev-parse", "HEAD"], cwd=repo).strip()
        result["baseline"] = resolve_baseline(repo, args.baseline_ref)
        if root_is_link:                # what the gate deletes and rebuilds under it must not land elsewhere
            raise CouldNotRun(f"{args.root} is a symlink; set LITHOS_CI_ROOT to the real directory")
        # Every gate starts from an empty directory, packs and Hub downloads included: the builds under test ran as this
        # user and could have changed anything an earlier gate left.
        gate = fresh_dir(args.root, "gate")
        os.environ["HF_HUB_CACHE"] = str(gate / "hub")          # where both builds' servers download the checkpoints
        free_gb = shutil.disk_usage(gate).free / 1e9
        if free_gb < MIN_FREE_GB:
            raise CouldNotRun(f"only {free_gb:.0f} GB free under {gate}; the gate needs {MIN_FREE_GB} GB")
        version = sh([args.python, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"]).strip()
        if version != "3.12":
            raise CouldNotRun(f"{args.python} is Python {version}; the bundles are built for 3.12 (LITHOS_CI_PYTHON)")
        candidate = candidate_bundle(repo, args.candidate_bundle, gate, args.python)
        result["bundle"] = bundle_digest(candidate)
        exes = {"baseline": install_bundle(baseline_bundle(result["baseline"], gate), gate / "baseline", args.python),
                "candidate": install_bundle(candidate, gate / "candidate", args.python)}
        plan = report / "builds.json"
        plan.write_text(json.dumps({k: str(v) for k, v in exes.items()}))
        child = [sys.executable, str(Path(__file__).resolve()), *argv, "--report-dir", str(report),
                 "--measure", str(plan)]
        wrapper = os.environ.get("LITHOS_CI_GPU_WRAPPER", "")
        rc = subprocess.run(shlex.split(wrapper) + child if wrapper else child, cwd=report).returncode
        measured = report / "measure.json"
        if rc or not measured.exists():
            raise CouldNotRun(f"the measurement exited {rc}")
        measured = json.loads(measured.read_text())
        if measured.get("error"):
            raise CouldNotRun(measured["error"])
        result["cases"] = measured["cases"]
        result["status"], summary = overall(result["cases"])
        # The status names the baseline and the bundle; release.yml checks both before it publishes that bundle.
        result["summary"] = f"vs {result['baseline']} bundle {result['bundle']}: {summary}"
    except CouldNotRun as e:
        result["error"] = str(e)
        result["summary"] = f"could not run: {e}"
    (report / "result.json").write_text(json.dumps(result, indent=1))
    text = summary_markdown(result)
    print("\n" + text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(text)
    return {"passed": 0, "failed": EXIT_FAIL}.get(result["status"], EXIT_COULD_NOT_RUN)


if __name__ == "__main__":
    sys.exit(main())
