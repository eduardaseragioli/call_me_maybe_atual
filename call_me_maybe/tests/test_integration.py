"""End-to-end check with the real model. Slow: needs the weights."""

from pathlib import Path

import pytest

from src.constrained_decoder import ConstrainedDecoder
from src.custom_tokenizer import CustomTokenizer
from src.parser import Parser
from src.prompt_builder import PromptBuilder

FUNCS = Path("data/input/functions_definition.json")


@pytest.mark.slow
@pytest.mark.parametrize("text", [
    "What is the sum of 2 and 3?",
    "Greet john",
    "",
    "asdf qwerty ???",
])
def test_every_output_matches_the_schema(text: str) -> None:
    """Even for weird prompts the output is schema-valid."""
    from llm_sdk import Small_LLM_Model

    if not FUNCS.exists():
        pytest.skip("functions_definition.json not found")
    functions = Parser.load_functions(str(FUNCS))
    model = Small_LLM_Model()
    tok = CustomTokenizer(model.get_path_to_vocab_file(),
                          model.get_path_to_merges_file())
    decoder = ConstrainedDecoder(model, tok, functions)
    result = decoder.generate_with_recovery(
        PromptBuilder.build_prompt(functions, text), text)
    assert decoder.validate(result) is None