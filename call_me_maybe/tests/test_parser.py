"""Parser edge cases: missing, broken and malformed input files."""

import json
from pathlib import Path
from typing import Any

import pytest

from src.models import FunctionCallResult
from src.parser import Parser

VALID_FN = {
    "name": "fn_x", "description": "d",
    "parameters": {"a": {"type": "number"}},
    "returns": {"type": "number"},
}


def write(path: Path, content: Any) -> str:
    """Write text or JSON to a file and return its path."""
    text = content if isinstance(content, str) else json.dumps(content)
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_missing_file(tmp_path: Path) -> None:
    """A missing file gives an empty list, not a crash."""
    assert Parser.load_functions(str(tmp_path / "nope.json")) == []
    assert Parser.load_tests(str(tmp_path / "nope.json")) == []


def test_directory_instead_of_file(tmp_path: Path) -> None:
    """A directory is not a readable file."""
    assert Parser.load_tests(str(tmp_path)) == []


@pytest.mark.parametrize("content", [
    "{not json", "", "[1, 2,", '{"a": 1}', "42", "null",
    [1, 2], [{"name": "x"}], ["text"],
])
def test_bad_functions_file(tmp_path: Path, content: Any) -> None:
    """Every kind of broken definitions file yields []."""
    assert Parser.load_functions(write(tmp_path / "f.json", content)) == []


@pytest.mark.parametrize("content", [
    "{bad", '{"prompt": "x"}', [{"nope": 1}], [{"prompt": 5}], [3],
])
def test_bad_tests_file(tmp_path: Path, content: Any) -> None:
    """Every kind of broken tests file yields []."""
    assert Parser.load_tests(write(tmp_path / "t.json", content)) == []


def test_valid_files(tmp_path: Path) -> None:
    """Well-formed inputs load correctly."""
    fns = Parser.load_functions(write(tmp_path / "f.json", [VALID_FN]))
    tests = Parser.load_tests(write(tmp_path / "t.json",
                                    [{"prompt": "hello"}, {"prompt": ""}]))
    assert fns[0].name == "fn_x"
    assert [t.prompt for t in tests] == ["hello", ""]


def test_save_results_roundtrip(tmp_path: Path) -> None:
    """Saved results have exactly prompt/name/parameters."""
    out = tmp_path / "out.json"
    res = [FunctionCallResult(prompt="p", name="fn_x",
                              parameters={"a": 1.0})]
    assert Parser.save_results(res, str(out)) is True
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data == [{"prompt": "p", "name": "fn_x",
                     "parameters": {"a": 1.0}}]


def test_save_results_bad_path(tmp_path: Path) -> None:
    """Writing into a missing directory reports failure."""
    bad = tmp_path / "no" / "dir" / "out.json"
    assert Parser.save_results([], str(bad)) is False
