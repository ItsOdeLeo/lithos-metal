"""tools/ci/release_gate.py: the verdict on a release candidate against the previous release."""

import hashlib
import json
import subprocess

import pytest

from tools.ci import ci_status, release_gate


def _run(texts, tok_s, ttft, interval=100):
    # one served run: prompt id -> measurement; "long" has the most prompt tokens
    return {pid: {"text": text, "tokens": 101, "decode_s": 100 / tok_s, "ttft_s": ttft if pid == "long" else 0.05,
                  "prompt_tokens": 2000 if pid == "long" else 20, "interval_tokens": interval}
            for pid, text in texts.items()}


def test_release_gate_passes_identical_outputs_within_noise():
    texts = {"math": "a", "long": "b"}
    runs = {"baseline": [_run(texts, 50.0, 1.0), _run(texts, 50.5, 1.01)],
            "candidate": [_run(texts, 51.0, 0.99), _run(texts, 51.2, 1.0)]}
    verdict = release_gate.judge_case(runs, False, 0.02, 0.05)
    assert verdict["verdict"] == "passed" and verdict["problems"] == []
    assert verdict["metrics"]["decode_tok_s"]["change"] > 0


def test_release_gate_fails_changed_output_and_regressions():
    base = {"math": "a", "long": "b"}
    runs = {"baseline": [_run(base, 50.0, 1.0), _run(base, 50.0, 1.0)],
            "candidate": [_run({"math": "x", "long": "b"}, 45.0, 1.1), _run({"math": "x", "long": "b"}, 45.0, 1.1)]}
    verdict = release_gate.judge_case(runs, False, 0.02, 0.05)
    assert verdict["verdict"] == "failed"
    assert verdict["problems"][0] == "output changed on 'math'"
    assert any(p.startswith("decode_tok_s regressed 10.0%") for p in verdict["problems"])
    assert any(p.startswith("ttft_long_s regressed 10.0%") for p in verdict["problems"])
    allowed = release_gate.judge_case(runs, True, 0.02, 0.05)
    assert allowed["problems"][0].startswith("decode_tok_s") and allowed["changed"] == ["math"]


def test_release_gate_is_inconclusive_on_noise_or_nondeterminism():
    texts = {"math": "a", "long": "b"}
    noisy = {"baseline": [_run(texts, 50.0, 1.0), _run(texts, 40.0, 1.0)],
             "candidate": [_run(texts, 50.0, 1.0), _run(texts, 50.0, 1.0)]}
    assert release_gate.judge_case(noisy, False, 0.02, 0.05)["verdict"] == "inconclusive"
    flaky = {"baseline": [_run(texts, 50.0, 1.0), _run({"math": "z", "long": "b"}, 50.0, 1.0)],
             "candidate": [_run(texts, 50.0, 1.0), _run(texts, 50.0, 1.0)]}
    verdict = release_gate.judge_case(flaky, True, 0.02, 0.05)
    assert verdict["verdict"] == "inconclusive" and "not deterministic on 'math'" in verdict["problems"][0]
    # the baseline's runs disagree and its second matches the candidate: no evidence of a change, nor of a regression
    flaky = {"baseline": [_run({"math": "a", "long": "b"}, 50.0, 1.0), _run({"math": "x", "long": "b"}, 40.0, 1.0)],
             "candidate": [_run({"math": "x", "long": "b"}, 45.0, 1.1), _run({"math": "x", "long": "b"}, 45.0, 1.1)]}
    verdict = release_gate.judge_case(flaky, False, 0.02, 0.05)
    assert verdict["verdict"] == "inconclusive"
    assert not any(p.startswith("output changed") for p in verdict["problems"])


def test_release_gate_overall_and_prompts():
    assert release_gate.overall([{"name": "a", "verdict": "skipped", "problems": ["needs 40 GB"]}])[0] == "could not run"
    status, summary = release_gate.overall([
        {"name": "a", "verdict": "passed", "problems": [], "metrics": {"decode_tok_s": {"change": 0.031}}},
        {"name": "b", "verdict": "failed", "problems": ["output changed on 'code'"]}])
    assert (status, summary) == ("failed", "b: output changed on 'code'")
    passed = {"name": "a", "verdict": "passed", "problems": [], "metrics": {"decode_tok_s": {"change": 0.031}}}
    assert release_gate.overall([passed])[1] == "1 cases identical; a decode +3.1%"
    assert release_gate.overall([{**passed, "changed": ["code"]}])[1] == (
        "1 cases passed, outputs changed in a; a decode +3.1%")
    cases = json.loads((release_gate.CASES).read_text())
    long = next(p for p in cases["prompts"] if p.get("repeat"))
    text = release_gate.render_prompt(long)[0]["content"]
    assert "{repeat}" not in text and "Record 48:" in text and "{i}" not in text
    assert ci_status.status_for({"status": "inconclusive", "summary": "x"}) == ("error", "x")


def test_release_gate_baseline_is_the_newest_tag_behind_the_commit(tmp_path):

    def git(*a):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=tmp_path, check=True,
                       capture_output=True)
    git("init", "-q")
    for tag in ("v0.1.9", "v0.1.10", None):
        git("commit", "-q", "--allow-empty", "-m", str(tag))
        if tag:
            git("tag", tag)
    def every(tag):
        return True
    assert release_gate.resolve_baseline(tmp_path, "auto", every) == "v0.1.10"
    git("tag", "v0.2.0")                                  # releasing this very commit: compare with the one before
    assert release_gate.resolve_baseline(tmp_path, "auto", every) == "v0.1.10"
    # a tag release.yml refused has no release: the one before it is the baseline
    assert release_gate.resolve_baseline(tmp_path, "auto", lambda tag: tag != "v0.1.10") == "v0.1.9"
    assert release_gate.resolve_baseline(tmp_path, "v0.1.9") == "v0.1.9"


def test_release_gate_takes_the_bundle_a_tag_publishes(tmp_path):
    formula = ('class LithosMetal < Formula\n  url "https://github.com/o/r/releases/download/v0.1.2/'
               'lithos-metal-0.1.2-macos-arm64.tar.gz"\n  version "0.1.2"\n  sha256 "' + "ab" * 32 + '"\nend\n')
    assert release_gate.formula_bundle(formula) == ("lithos-metal-0.1.2-macos-arm64.tar.gz", "ab" * 32)
    with pytest.raises(release_gate.CouldNotRun):
        release_gate.formula_bundle("class LithosMetal < Formula\nend\n")
    dist = tmp_path / "dist"
    dist.mkdir()
    with pytest.raises(release_gate.CouldNotRun, match="found 0"):
        release_gate.candidate_bundle(tmp_path, dist, tmp_path, "python3")
    (dist / "lithos-metal-0.1.3-macos-arm64.tar.gz").write_bytes(b"bundle")
    (dist / "lithos-metal.rb").write_text(formula)
    archive = release_gate.candidate_bundle(tmp_path, dist, tmp_path, "python3")
    assert archive.name == "lithos-metal-0.1.3-macos-arm64.tar.gz"
    assert release_gate.bundle_digest(archive) == hashlib.sha256(b"bundle").hexdigest()[:12]


def test_release_gate_counts_the_timed_tokens_when_every_run_reports_them():
    texts = {"math": "a", "long": "b"}
    # the candidate publishes 9 tokens in its first chunk instead of 1: 50 tok/s either way, timed over 92 tokens
    runs = {"baseline": [_run(texts, 50.0, 1.0, 100), _run(texts, 50.0, 1.0, 100)],
            "candidate": [_run(texts, 50.0 * 100 / 92, 1.0, 92), _run(texts, 50.0 * 100 / 92, 1.0, 92)]}
    verdict = release_gate.judge_case(runs, False, 0.02, 0.05)
    assert verdict["verdict"] == "passed" and abs(verdict["metrics"]["decode_tok_s"]["change"]) < 1e-9
    runs["baseline"][0]["math"]["interval_tokens"] = None   # an older baseline counts no tokens per chunk: an estimate
    verdict = release_gate.judge_case(runs, False, 0.02, 0.05)       # would credit the candidate with 8.7 %
    assert verdict["verdict"] == "passed" and "decode_tok_s" not in verdict["metrics"]
    summary = release_gate.overall([{"name": "a", **verdict}])[1]
    assert summary == "1 cases identical; a decode not compared (no per-chunk counts)"


def test_release_needs_passing_gates_against_the_previous_release():
    ok = {"context": "release-gate (m4_pro)", "state": "success",
          "description": "vs v0.1.2 bundle 0123456789ab: 3 cases identical"}
    other = {"context": "mac-gpu (m4_pro)", "state": "failure", "description": "kernels: 1 failed"}
    assert release_gate.release_refusal([ok, other], "v0.1.2") is None
    assert "did not compare with v0.1.3" in release_gate.release_refusal([ok], "v0.1.3")
    assert "is pending" in release_gate.release_refusal([ok, {**ok, "context": "release-gate (m5_max_40c)",
                                                              "state": "pending"}], "v0.1.2")
    assert release_gate.release_refusal([other], "v0.1.2") == "no release-gate status on this commit"
    assert release_gate.release_refusal([ok], "v0.1.2", bundle="0123456789ab") is None
    assert "gated bundle 0123456789ab, not fedcba987654" in release_gate.release_refusal([ok], "v0.1.2", "fedcba987654")


def test_release_needs_every_case_passed_on_some_runner():
    cases = {"cases": [{"name": "27b-dspark"}, {"name": "27b-plain"}, {"name": "8b-plain"}]}
    small = {"cases": [{"name": "27b-dspark", "verdict": "skipped"}, {"name": "27b-plain", "verdict": "skipped"},
                       {"name": "8b-plain", "verdict": "passed"}]}
    large = {"cases": [{"name": n, "verdict": "passed"} for n in ("27b-dspark", "27b-plain", "8b-plain")]}
    assert release_gate.uncovered([small], cases) == "no runner passed the case 27b-dspark"
    assert release_gate.uncovered([small, large], cases) is None
