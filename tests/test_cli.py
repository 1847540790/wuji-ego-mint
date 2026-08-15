from __future__ import annotations

import subprocess
import sys


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "mint", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_top_level_help_lists_public_commands() -> None:
    result = run_cli("--help")
    assert result.returncode == 0
    for command in ("pipeline", "train", "infer", "viewer", "doctor"):
        assert command in result.stdout


def test_inference_help_is_dispatched_to_subcommand() -> None:
    result = run_cli("infer", "--help")
    assert result.returncode == 0
    assert "prediction-only inference" in result.stdout
    assert "--checkpoint" in result.stdout


def test_viewer_help_is_dispatched_to_subcommand() -> None:
    result = run_cli("viewer", "--help")
    assert result.returncode == 0
    assert "prediction viewer" in result.stdout
    assert "--samples" in result.stdout
