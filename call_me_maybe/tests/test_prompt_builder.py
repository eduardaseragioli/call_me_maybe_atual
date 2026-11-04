"""Prompt construction."""

from src.prompt_builder import PromptBuilder
from tests.test_decoder import make_functions


def test_prompt_contains_catalog_and_request() -> None:
    """Names, the request and the final 'Answer:' cue are present."""
    prompt = PromptBuilder.build_prompt(make_functions(), "do the thing")
    assert "fn_add" in prompt and "fn_greet" in prompt
    assert "Request: do the thing" in prompt
    assert prompt.endswith("Answer:")


def test_catalog_is_compact() -> None:
    """No indentation: fewer tokens, faster inference."""
    catalog = PromptBuilder.format_functions_catalog(make_functions())
    assert "\n" not in catalog
    assert '"name":"fn_add"' in catalog
