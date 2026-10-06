"""Nested arguments: arrays and objects."""

from typing import Any

from src.constrained_decoder import ConstrainedDecoder
from src.models import FunctionCallResult, FunctionDefinition
from tests.fakes import FakeTokenizer, ScriptedModel

PROMPT = "hi"


def fn(params: dict[str, Any]) -> FunctionDefinition:
    """One function 'fn_x' with the given parameters."""
    return FunctionDefinition.model_validate({
        "name": "fn_x", "description": "x", "parameters": params,
        "returns": {"type": "number"}})


def decode(params: dict[str, Any], target: str) -> FunctionCallResult:
    """Decode with a model that wants to write `target`."""
    tok = FakeTokenizer()
    prefix_len = len(tok.encode(PROMPT + ' {"name": "'))
    model = ScriptedModel(tok, prefix_len, target)
    decoder = ConstrainedDecoder(model, tok, [fn(params)])
    result = decoder.generate(PROMPT, PROMPT)
    assert decoder.validate(result) is None
    return result


HEAD = 'fn_x", "parameters": {'
STR_ARR = {"tags": {"type": "array", "items": {"type": "string"}}}
NUM_ARR = {"v": {"type": "array", "items": {"type": "number"}}}


def test_array_of_strings() -> None:
    """Several string items."""
    r = decode(STR_ARR, HEAD + '"tags": ["a", "bc"]}}')
    assert r.parameters == {"tags": ["a", "bc"]}


def test_array_of_numbers() -> None:
    """Numbers inside arrays end at ',' or ']'."""
    r = decode(NUM_ARR, HEAD + '"v": [1, 22, 3]}}')
    assert r.parameters == {"v": [1.0, 22.0, 3.0]}


def test_empty_array() -> None:
    """The model may close the array immediately."""
    r = decode(NUM_ARR, HEAD + '"v": []}}')
    assert r.parameters == {"v": []}


def test_nested_object() -> None:
    """Object keys come from the schema."""
    params = {"point": {"type": "object", "properties": {
        "x": {"type": "number"}, "y": {"type": "number"}}}}
    r = decode(params, HEAD + '"point": {"x": 1, "y": 2}}}')
    assert r.parameters == {"point": {"x": 1.0, "y": 2.0}}


def test_array_of_objects() -> None:
    """Recursion: an array whose items are objects."""
    params = {"pts": {"type": "array", "items": {
        "type": "object",
        "properties": {"n": {"type": "string"}}}}}
    r = decode(params, HEAD + '"pts": [{"n": "a"}, {"n": "b"}]}}')
    assert r.parameters == {"pts": [{"n": "a"}, {"n": "b"}]}


def test_missing_items_raises() -> None:
    """An array without 'items' is reported, not guessed."""
    import pytest
    from src.constrained_decoder import DecodingError
    tok = FakeTokenizer()
    model = ScriptedModel(tok, 0, HEAD + '"v": []}}')
    decoder = ConstrainedDecoder(
        model, tok, [fn({"v": {"type": "array"}})])
    with pytest.raises(DecodingError):
        decoder.generate(PROMPT, PROMPT)