import subprocess

import pytest

from scene_classifier.utils.git_util import resolve_git_sha


def test_configured_sha_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("git should not be called")

    monkeypatch.setattr(subprocess, "run", fail)
    assert resolve_git_sha("abc123") == "abc123"


def test_falls_back_to_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "deadbeef\n")
    )
    assert resolve_git_sha(None) == "deadbeef"


def test_raises_when_git_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", missing_git)
    with pytest.raises(RuntimeError, match="GIT_COMMIT_SHA"):
        resolve_git_sha(None)
