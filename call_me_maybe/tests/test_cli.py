"""The program must exit cleanly on bad input, never with a traceback."""

import sys
from pathlib import Path

import pytest

from src.__main__ import main


def test_missing_input_files_exit_with_code_1(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Missing files -> clear error and exit code 1."""
    monkeypatch.setattr(sys, "argv", [
        "src",
        "--functions_definition", str(tmp_path / "a.json"),
        "--input", str(tmp_path / "b.json"),
        "--output", str(tmp_path / "o.json"),
    ])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1