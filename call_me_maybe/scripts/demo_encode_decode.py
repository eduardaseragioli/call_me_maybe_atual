"""Show how encode/decode plug into constrained decoding."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "llm_sdk"))

from llm_sdk import Small_LLM_Model  # noqa: E402

from src.constrained_decoder import ConstrainedDecoder  # noqa: E402
from src.custom_tokenizer import CustomTokenizer  # noqa: E402
from src.parser import Parser  # noqa: E402
from src.prompt_builder import PromptBuilder  # noqa: E402


def show_top(tok: CustomTokenizer, logits: np.ndarray,
             allowed: set[int] | None) -> str:
    """Top-5 tokens; with a mask, everything else is -inf."""
    scores = logits.copy()
    if allowed is not None:
        keep = np.full_like(scores, -np.inf)
        idx = np.fromiter(allowed, dtype=np.int64)
        keep[idx] = scores[idx]
        scores = keep
    top = np.argsort(-scores)[:5]
    return "  ".join(f"{tok.token_text(int(i))!r}({scores[i]:.1f})"
                     for i in top)


def main() -> None:
    """Walk through one decoding step and a full round trip."""
    functions = Parser.load_functions(
        "data/input/functions_definition.json")
    model = Small_LLM_Model()
    tok = CustomTokenizer(model.get_path_to_vocab_file(),
                          model.get_path_to_merges_file())
    decoder = ConstrainedDecoder(model, tok, functions)

    request = "Greet john"
    prompt = PromptBuilder.build_prompt(functions, request)
    ids = tok.encode(prompt + ' {"name": "')
    print(f"1. ENCODE: prompt -> {len(ids)} ids")
    print("   last 6 ids:   ", ids[-6:])
    print("   as text:      ", [tok.token_text(i) for i in ids[-6:]])

    logits = np.asarray(model.get_logits_from_input_ids(ids),
                        dtype=np.float32)
    print("2. LOGITS (no mask):")
    print("  ", show_top(tok, logits, None))

    names = list(decoder.functions)
    allowed = {i for i in decoder.name_ids
               if any(n.startswith(decoder.texts[i]) for n in names)}
    print(f"3. MASK: only {len(allowed)} of {tok.vocab_size} tokens "
          "can start a function name:")
    print("  ", show_top(tok, logits, allowed))

    result = decoder.generate(prompt, request)
    print("4. RESULT:", result.name, result.parameters)

    text = f'{{"name": "{result.name}"}}'
    back = tok.decode(tok.encode(text))
    print(f"5. DECODE round trip OK: {back == text} -> {back}")


if __name__ == "__main__":
    main()