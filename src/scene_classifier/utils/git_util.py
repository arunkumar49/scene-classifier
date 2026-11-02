import subprocess


def resolve_git_sha(configured: str | None) -> str:
    """Return the configured commit SHA, or ask git for the current one."""
    if configured:
        return configured
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError) as e:
        # A Training Job has no .git folder, so the launcher must set GIT_COMMIT_SHA
        raise RuntimeError("GIT_COMMIT_SHA is not set and git can't report the commit") from e
    return result.stdout.strip()
