"""Constrained decoder: structure, types, mask and recovery."""

from typing import Any

import pytest

from src.constrained_decoder import (
    ATTEMPTS, ConstrainedDecoder, DecodingError)
from src.models import FunctionCallResult, FunctionDefinition
from tests.fakes import ConstantModel, FakeTokenizer, ScriptedModel

PROMPT = "hi"
HEAD = 'fn_{}", "parameters": {{'


def make_functions() -> list[FunctionDefinition]:
    """Three functions: numbers, a string, a boolean."""
    raw: list[dict[str, Any]] = [
        {"name": "fn_add", "description": "Add",
         "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
         "returns": {"type": "number"}},
        {"name": "fn_greet", "description": "Greet",
         "parameters": {"name": {"type": "string"}},
         "returns": {"type": "string"}},
        {"name": "fn_flag", "description": "Flag",
         "parameters": {"on": {"type": "boolean"}},
         "returns": {"type": "boolean"}},
    ]
    return [FunctionDefinition.model_validate(r) for r in raw]


def decode(target: str) -> FunctionCallResult:
    """Run the decoder against a model that wants to write `target`."""
    tok = FakeTokenizer()
    prefix_len = len(tok.encode(PROMPT + ' {"name": "'))
    model = ScriptedModel(tok, prefix_len, target)
    decoder = ConstrainedDecoder(model, tok, make_functions())
    return decoder.generate(PROMPT, PROMPT)


def test_two_numbers() -> None:
    """Multiple parameters, in schema order, as floats."""
    r = decode('fn_add", "parameters": {"a": 42, "b": 7}}')
    assert r.name == "fn_add"
    assert r.parameters == {"a": 42.0, "b": 7.0}
    assert all(isinstance(v, float) for v in r.parameters.values())


def test_negative_and_decimal() -> None:
    """Negative and decimal numbers are allowed by the number mask."""
    r = decode('fn_add", "parameters": {"a": -5, "b": 3.5}}')
    assert r.parameters == {"a": -5.0, "b": 3.5}


def test_large_number() -> None:
    """Large numbers survive."""
    r = decode('fn_add", "parameters": {"a": 123456789, "b": 0}}')
    assert r.parameters["a"] == 123456789.0


def test_string_with_special_characters() -> None:
    """Spaces and punctuation inside strings."""
    r = decode('fn_greet", "parameters": {"name": "a b-c_d"}}')
    assert r.parameters == {"name": "a b-c_d"}


def test_empty_string() -> None:
    """An empty string value is valid."""
    r = decode('fn_greet", "parameters": {"name": ""}}')
    assert r.parameters == {"name": ""}


def test_escape_sequence_is_decoded() -> None:
    """A JSON escape (\\n) becomes a real newline."""
    r = decode('fn_greet", "parameters": {"name": "x\\ny"}}')
    assert r.parameters == {"name": "x\ny"}


def test_escaped_quotes_do_not_close_the_string() -> None:
    """Only an unescaped double quote ends a string."""
    r = decode('fn_greet", "parameters": {"name": "say \\"hi\\""}}')
    assert r.parameters == {"name": 'say "hi"'}


def test_boolean() -> None:
    """Booleans come out as Python bool."""
    r = decode('fn_flag", "parameters": {"on": true}}')
    assert r.parameters == {"on": True}


def test_mask_blocks_invalid_tokens() -> None:
    """The model loves 'z', but only digits are allowed: 'z' loses."""
    tok = FakeTokenizer()
    z = tok.token_to_id["z"]
    digits = [tok.token_to_id[c] for c in "0123456789"]
    decoder = ConstrainedDecoder(
        ConstantModel(tok.vocab_size, z), tok, make_functions())
    assert decoder._pick([1, 2, 3], digits) in digits


def test_single_option_skips_the_model() -> None:
    """With one valid token the LLM is not even called."""
    tok = FakeTokenizer()
    model = ConstantModel(tok.vocab_size, 0)
    decoder = ConstrainedDecoder(model, tok, make_functions())
    assert decoder._pick([1], [5]) == 5
    assert model.calls == 0


def test_no_valid_token_raises() -> None:
    """Dead end -> DecodingError, not garbage."""
    tok = FakeTokenizer()
    decoder = ConstrainedDecoder(
        ConstantModel(tok.vocab_size, 0), tok, make_functions())
    with pytest.raises(DecodingError):
        decoder._pick([1], [])


def test_unsupported_type_raises() -> None:
    """A type the decoder does not know is reported, not ignored."""
    tok = FakeTokenizer()
    fn = FunctionDefinition.model_validate({
        "name": "fn_add", "description": "x",
        "parameters": {"a": {"type": "banana"}},
        "returns": {"type": "number"}})
    prefix_len = len(tok.encode(PROMPT + ' {"name": "'))
    model = ScriptedModel(tok, prefix_len, 'fn_add", "parameters": {"a": 1}}')
    decoder = ConstrainedDecoder(model, tok, [fn])
    with pytest.raises(DecodingError):
        decoder.generate(PROMPT, PROMPT)


def test_is_stuck() -> None:
    """Repeating windows are detected."""
    assert ConstrainedDecoder._is_stuck([1, 2, 3] * 4)
    assert not ConstrainedDecoder._is_stuck(list(range(20)))
    assert not ConstrainedDecoder._is_stuck([1, 2])


def test_validate_catches_wrong_types() -> None:
    """validate() rejects results that break the schema."""
    tok = FakeTokenizer()
    decoder = ConstrainedDecoder(
        ConstantModel(tok.vocab_size, 0), tok, make_functions())
    good = FunctionCallResult(prompt="p", name="fn_add",
                              parameters={"a": 1.0, "b": 2.0})
    bad_type = FunctionCallResult(prompt="p", name="fn_add",
                                  parameters={"a": "1", "b": 2.0})
    missing = FunctionCallResult(prompt="p", name="fn_add",
                                 parameters={"a": 1.0})
    unknown = FunctionCallResult(prompt="p", name="nope", parameters={})
    assert decoder.validate_result(good) is None
    assert decoder.validate_result(bad_type) is not None
    assert decoder.validate_result(missing) is not None
    assert decoder.validate_result(unknown) is not None


def make_decoder() -> ConstrainedDecoder:
    """Decoder with a dummy model (generate gets monkeypatched)."""
    tok = FakeTokenizer()
    return ConstrainedDecoder(
        ConstantModel(tok.vocab_size, 0), tok, make_functions())


def test_recovery_second_attempt_succeeds(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Attempt 1 fails, attempt 2 returns a valid result."""
    decoder = make_decoder()
    good = FunctionCallResult(prompt="p", name="fn_add",
                              parameters={"a": 1.0, "b": 2.0})
    outcomes: list[Any] = [DecodingError("boom"), good]

    def fake_generate(prompt: str, original: str) -> FunctionCallResult:
        item = outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return item  # type: ignore[no-any-return]

    monkeypatch.setattr(decoder, "generate", fake_generate)
    assert decoder.generate_with_recovery("p", "p") == good


def test_recovery_all_attempts_fail(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """If every strategy fails, a clear DecodingError is raised."""
    decoder = make_decoder()

    def always_fail(prompt: str, original: str) -> FunctionCallResult:
        raise DecodingError("boom")

    monkeypatch.setattr(decoder, "generate", always_fail)
    with pytest.raises(DecodingError):
        decoder.generate_with_recovery("p", "p")
    assert len(ATTEMPTS) == 3
