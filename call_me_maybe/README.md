*This project has been created as part of the 42 curriculum by eseragio.*

# call me maybe

Function calling with a small LLM (Qwen3-0.6B), using constrained decoding.

## Description

Small language models are bad at producing JSON: asked for it, they often return broken or off-schema text. This project turns a natural-language request into a structured function call, and guarantees the result is valid.

For example, given `"What is the sum of 40 and 2?"` the program does not answer `42`. It returns the call that would solve it:

```json
{"prompt": "What is the sum of 40 and 2?", "name": "fn_add_numbers", "parameters": {"a": 40.0, "b": 2.0}}
```

The program only decides which function to call and with which arguments. It never runs the functions. The available functions come from a JSON file, so nothing is hardcoded.

The idea is to make the output valid by construction: the code writes the fixed parts of the JSON, and at every choice point the tokens that would break the structure or the schema are removed before the model picks the next one.

How it works, in short:

1. `Parser` loads the function definitions and the prompts (validated with pydantic).
2. `PromptBuilder` builds the prompt: the function catalog, a few rules and generic examples.
3. `CustomTokenizer` converts text to token ids (my own byte-level BPE implementation).
4. `ConstrainedDecoder` generates the call token by token, masking invalid tokens.
5. The result is saved to `data/output/function_calling_results.json`.

Project layout:

```
src/        main code (decoder, tokenizer, parser, prompt builder, models)
llm_sdk/    provided SDK
data/input/ example inputs
tests/      pytest suite
scripts/    demo of encode / decode inside constrained decoding
```

## Instructions

You need Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
make install     # uv sync
make run         # run with the default files
make debug       # run under pdb
make lint        # flake8 + mypy
make test        # fast tests (no LLM)
make test-all    # also runs the tests that load the real model
make clean       # remove caches
```

To choose the files:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Extra options: `--verbose` shows, step by step, what the model wanted and what the mask blocked. `--model NAME` uses another Hugging Face model (it needs a Qwen-style tokenizer).

The first run downloads the model (about 1.5 GB).

### Example usage

```bash
uv run python -m src
```

```
Prompt: Greet john
Function: fn_greet
Parameters: {'name': 'john'}
```

Output file:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {"s": "hello"}
  }
]
```

With `--verbose` you can see the mask working:

```
allowed=3 model wants: '_with' '",'[MASKED] 'With'[MASKED] -> '_with'
```

The model wanted to close the function name early, but `fn_substitute_string` does not exist, so the mask blocked it.

## Algorithm explanation

**Tokenizer.** `CustomTokenizer` rebuilds the Qwen tokenizer from `vocab.json` and `merges.txt`, using the SDK's public methods to find the files. Text is split into pre-tokens, turned into bytes, mapped to printable characters (that is why spaces show up as `Ġ`), and merged with BPE by lowest rank. `decode` goes back through bytes to UTF-8. Its output matches the SDK's `encode` and `decode(encode(x)) == x`.

**Constrained decoding.** The code writes the fixed structure (`{"name": "`, the parameter names, quotes, commas, braces) and the model only decides at the choice points. At each one, the decoder:

1. works out which token ids are allowed in the current state;
2. gets the logits with `get_logits_from_input_ids`;
3. ignores everything not allowed (same as setting it to `-inf`);
4. takes the best remaining token and repeats.

What is allowed at each choice point:

| Choice point | Allowed tokens |
|---|---|
| Function name | Only tokens that keep the text a prefix of a real function name. The closing quote is allowed only once the name is complete |
| `number` / `integer` | Only digits, `-` and `.`, while the text still looks like a number. `,` or `}` only after a complete number |
| `string` | Any token without control characters. An unescaped `"` closes the string, including tokens like `",` that fuse the quote with punctuation |
| `boolean` | Only prefixes of `true` or `false` |
| `array` | Either an item start, or `,` / `]` |
| `object` | Keys come from the schema, the model fills the values |

Because the structure is written by the code, all parameters are always present and types always match the schema. The result is a Python dict written with `json.dump`, so the file is always valid JSON. Arrays and objects are handled recursively.

**Error recovery.** `generate_with_recovery` tries up to three settings (normal, with a repetition penalty, with a repetition penalty and a shorter limit). Each result goes through `validate`, which checks the function, the parameter names and the types. Dead ends and loops raise a `DecodingError`. If everything fails, only that prompt is skipped.

## Design decisions

- **The code writes the structure; the model only fills the choice points.** Just counting braces cannot guarantee a valid schema.
- **The function is chosen by the LLM.** The mask only removes names that do not exist. No heuristics.
- **Tokens like `",` can close a string.** Forbidding them made long strings loop until the limit.
- **No hardcoded examples.** The few-shot examples in the prompt use functions that do not exist in the test files.
- **Short prompt.** The catalog is written without indentation, which makes every model call faster.
- **Shortcut for a single option.** If only one token is valid, the model is not called.
- **Own tokenizer.** The main code does not use the SDK's `encode` / `decode`.
- **Pydantic** for data and settings (`FunctionDefinition`, `TypeInfo`, `FunctionCallResult`, `DecoderSettings`, `DecodingConfig`, `Parser`, `PromptBuilder`). `CustomTokenizer` and `ConstrainedDecoder` are normal classes, since they mostly hold heavy internal state (vocabulary, arrays, the model).
- **Bad input stops early.** Missing or invalid files give a clear error before the slow model loading.

## Performance analysis

Tested on a MacBook Air (Apple silicon) with Qwen3-0.6B and the 11 example prompts:

| Metric | Result |
|---|---|
| Valid JSON | 100% (by construction) |
| Correct function | 11 / 11 |
| Correct arguments | 10 / 11 |
| Total time | about 70 to 82 s (limit: 5 min) |
| Time per model call | about 0.4 to 0.6 s |

Making the catalog compact cut the total time from about 207 s to about 70 s. Every call reprocesses the whole prompt (the public SDK has no KV-cache), so a shorter prompt helps every call. Timing varies from run to run on this laptop, probably because of heat or memory pressure; a single prompt run alone took about the same time as inside the full run.

`Qwen3-1.7B` also works with the same tokenizer, but took 469 s here and was not more accurate.

Known limits of the 0.6B model (the structure is always valid, but values can be imperfect):

- For "replace vowels with asterisks" it gets the regex right (`([aeiou])`) but returns `' *'` with a leading space.
- With nested arguments it sometimes closes an array right away (`"tags": []`).
- An empty prompt still gives a valid call, because the model copies a pattern from the context.
- No KV-cache or batching, since that would need private SDK methods, which the subject forbids.

## Challenges faced

- **Empty results at first.** My first tokenizer split on spaces and looked up whole words, so the model saw almost no prompt. I replaced it with a real byte-level BPE.
- **Masks that guaranteed nothing.** Counting braces does not give valid JSON or correct types. I moved to generating the structure from the schema.
- **Strings that never ended.** Tokens like `",` were forbidden, so the model could not close long strings. Letting any unescaped quote close the string fixed it.
- **Repetition loops** in regex values. Solved with loop detection, a repetition penalty and retries.
- **Slow runs.** Solved with a shorter prompt and by skipping the model when only one token is valid.
- **Environment.** A corrupted uv cache in `/tmp` broke a build; cleaning it fixed it.

## Testing strategy

`make test` runs 47 fast tests without loading the LLM. They use a fake tokenizer and a fake model that "wants" to write a given text, which makes the decoder easy to test:

- **Decoder:** several parameters, negative / decimal / large numbers, empty strings, special characters, escapes, escaped quotes, booleans, the mask blocking a favorite token, the single-option shortcut, dead ends, unsupported types, loop detection, validation and recovery.
- **Nested arguments:** arrays, empty arrays, objects, arrays of objects, missing `items`.
- **Parser:** missing files, invalid JSON, empty files, wrong structure, saving results.
- **Tokenizer:** merges and round trip (accents, newlines, CJK) on a small synthetic vocabulary.
- **Prompt builder** and **CLI** (missing files exit with code 1).

`make test-all` adds 4 tests with the real model (including an empty prompt and nonsense text) that check every output against the schema.

I also checked the tokenizer against the SDK, read `--verbose` traces, forced a failure of the first attempt to confirm the retry, and ran everything from a clean copy with `rm -rf .venv && uv sync && uv run python -m src`.

## Bonus features

- Other models via `--model` (tested with Qwen3-1.7B).
- Own tokenizer with public `encode` / `decode`.
- Error recovery with retries and validation.
- Visualization of the generation (`--verbose`).
- Test suite (51 tests).
- Nested arguments (arrays and objects).
- Demo of encode / decode inside constrained decoding: `uv run python scripts/demo_encode_decode.py`.
- Partial: performance optimizations (short prompt, single-option shortcut, NumPy, small logits cache).

## Resources

- Hugging Face: Qwen3 model card and `tokenizers` documentation.
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2 byte mapping).
- Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units* (BPE).
- Willard and Louf, *Efficient Guided Generation for Large Language Models*.
- Documentation of pydantic, NumPy, pytest, flake8 and mypy.

**How AI was used.** I used Claude as a tutor and reviewer: to explain constrained decoding and byte-level BPE, to review my code and find bugs (tokenizer, string closing, mask logic), to help draft parts of the decoder, tokenizer, tests and this README, and to suggest generic prompt rules and the recovery strategy. I ran and tested everything myself and reviewed the code so I can explain it.