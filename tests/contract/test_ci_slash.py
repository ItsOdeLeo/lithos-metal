"""tools/ci/mac_slash.py: the pull request commands that start the Mac lanes."""

import pytest

from tools.ci import github_api, mac_slash


def test_slash_command_parsing():
    cmd = mac_slash.parse("\n/mac-test --shader-validation m5_max_40c m5_max_40c m4_pro\nthanks")
    assert (cmd.workflow, cmd.flags, cmd.runners) == ("mac-gpu.yml", {"shader_validation": "true"},
                                                      ["m5_max_40c", "m4_pro"])
    assert mac_slash.parse("looks good, /mac-test later") is None
    with pytest.raises(ValueError, match="unknown option"):
        mac_slash.parse("/mac-test --fast")
    assert mac_slash.select_runners(mac_slash.parse("/mac-test"), ["a", "b"]) == ["a", "b"]
    with pytest.raises(ValueError, match="unknown runner"):
        mac_slash.select_runners(mac_slash.parse("/mac-test c"), ["a", "b"])
    with pytest.raises(ValueError, match="MAC_CI_RUNNERS"):
        mac_slash.select_runners(mac_slash.parse("/mac-test"), [])
    cmd = mac_slash.parse("/mac-test m4_pro " + "0abc123" * 5 + "01234")
    assert (cmd.runners, cmd.commit) == (["m4_pro"], "0abc123" * 5 + "01234")
    with pytest.raises(ValueError, match="full 40-digit SHA"):
        mac_slash.parse("/mac-test 0abc123")      # a prefix a forged commit can share
    with pytest.raises(ValueError, match="one commit"):
        mac_slash.parse("/mac-test " + "a" * 40 + " " + "f" * 40)


class FakeGitHub(github_api.GitHub):
    def __init__(self, role="write", draft=False, author_role="write"):
        super().__init__("o/r", "t", send=self.record)
        self.calls, self.role, self.draft, self.author_role = [], role, draft, author_role

    def record(self, method, path, body):
        self.calls.append((method, path, body))
        if path.endswith("/permission"):   # alice comments, bob wrote the pull request
            role = self.author_role if "/collaborators/bob/" in path else self.role
            return {"role_name": role, "permission": "read" if role in ("read", "triage") else role}
        if path.endswith("/pulls/7"):
            return {"state": "open", "draft": self.draft, "head": {"sha": "f" * 40}, "user": {"login": "bob"}}
        return {}


ENV = {"PR_NUMBER": "7", "ACTOR": "alice", "COMMENT_ID": "99", "DEFAULT_BRANCH": "main",
       "MAC_CI_RUNNERS": '["m5_max_40c", "m4_pro"]'}


def test_slash_command_dispatches_the_pinned_head_per_runner():
    gh = FakeGitHub()
    text = mac_slash.handle(gh, {**ENV, "COMMENT_BODY": "/mac-test m4_pro"})
    posts = [(p, b) for m, p, b in gh.calls if m == "POST"]
    assert posts[0] == ("/repos/o/r/statuses/" + "f" * 40, {
        "state": "pending", "context": "mac-gpu (m4_pro)", "description": "queued by @alice",
        "target_url": "https://github.com/o/r/actions/workflows/mac-gpu.yml"})
    assert posts[1] == ("/repos/o/r/actions/workflows/mac-gpu.yml/dispatches", {
        "ref": "main", "inputs": {"git_ref": "f" * 40, "runner": "m4_pro", "pr": "7"}})
    assert posts[2][0] == "/repos/o/r/issues/comments/99/reactions"
    assert posts[3][0] == "/repos/o/r/issues/7/comments" and "dispatched" in text


@pytest.mark.parametrize("role,draft,expect", [("read", False, "needs write access"),
                                                ("triage", False, "needs write access"),
                                                ("write", True, "ready for review")])


def test_slash_command_refuses_without_dispatching(role, draft, expect):
    gh = FakeGitHub(role=role, draft=draft)
    assert expect in mac_slash.handle(gh, {**ENV, "COMMENT_BODY": "/mac-test"})
    assert not any("dispatches" in p or "statuses" in p for _, p, _ in gh.calls)


def test_slash_command_does_not_leave_a_status_pending_when_dispatch_fails():
    gh = FakeGitHub()
    send = gh.send

    def refuse_dispatch(method, path, body):
        if path.endswith("/dispatches"):
            raise RuntimeError("HTTP 422")
        return send(method, path, body)

    gh.send = refuse_dispatch
    with pytest.raises(RuntimeError):
        mac_slash.handle(gh, {**ENV, "COMMENT_BODY": "/mac-test m4_pro"})
    statuses = [b for m, p, b in gh.calls if "/statuses/" in p]
    assert [s["state"] for s in statuses] == ["pending", "error"]
    assert statuses[1]["context"] == "mac-gpu (m4_pro)" and "HTTP 422" in statuses[1]["description"]


def test_slash_command_ignores_a_comment_that_only_mentions_it():
    gh = FakeGitHub()
    assert mac_slash.handle(gh, {**ENV, "COMMENT_BODY": "looks good, /mac-test later"}) == ""
    assert gh.calls == []


@pytest.mark.parametrize("body,dispatched,expect", [
    ("/mac-test", False, "by its full SHA: `/mac-test <sha>` (the head is ffffffffff)"),
    ("/mac-test " + "f" * 39 + "e", False, "the head is ffffffffff, not ffffffffff"),
    ("/mac-test " + "f" * 40, True, "dispatched for ffffffffff")])
def test_slash_command_binds_a_forks_pull_request_to_the_commit_read(body, dispatched, expect):
    # the author cannot write here: a push after the comment must not reach a Mac unread
    gh = FakeGitHub(author_role="read")
    assert expect in mac_slash.handle(gh, {**ENV, "COMMENT_BODY": body})
    assert any("dispatches" in p for _, p, _ in gh.calls) == dispatched

