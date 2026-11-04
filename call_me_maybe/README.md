*This project has been created as part of the 42 curriculum by eseragio.*

# call me maybe
 
Function calling with a small LLM (Qwen3-0.6B), using constrained decoding.
 
## Description
 
Small language models are bad at producing JSON: asked for it, they often return broken or off-schema text. This project turns a natural-language request into a structured function call and guarantees that the result is valid.
 
For example, given `"What is the sum of 40 and 2?"` the program does not answer `42`. It returns the call that would solve it:
 
```json
{"prompt": "What is the sum of 40 and 2?", "name": "fn_add_numbers", "parameters": {"a": 40.0, "b": 2.0}}
```
 
The program only decides which function to call and with which arguments. It never runs the functions, and the available functions come from a JSON file, so nothing is hardcoded.
 
The output is valid by construction: the code writes the fixed parts of the JSON, and at every choice point the tokens that would break the structure or the schema are removed before the model picks the next one.
 
Pipeline: `Parser` loads and validates the inputs, `PromptBuilder` builds the prompt, `CustomTokenizer` converts text to token ids (own byte-level BPE), `ConstrainedDecoder` generates the call token by token, and the result is saved to `data/output/function_calling_results.json`.
 
```
src/        decoder, tokenizer, parser, prompt builder, models
llm_sdk/    provided SDK
data/input/ example inputs
tests/      pytest suite
scripts/    demo of encode / decode inside constrained decoding
```
 
## Instructions
 
Requirements: Python 3.10+ and [uv](https://docs.astral.sh/uv/).
 
```bash
make install     # uv sync
make run         # run with the default files
make debug       # run under pdb
make lint        # flake8 + mypy
make test        # fast tests (no LLM)
make test-all    # also runs the tests that use the real model
make clean       # remove caches
```
 
Choosing the files:
 
```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```
 
Extra options: `--verbose` shows what the model wanted and what the mask blocked at each step; `--model NAME` uses another Hugging Face model (it needs a Qwen-style byte-level BPE tokenizer). The first run downloads the model (about 1.5 GB).
 
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
 
**Tokenizer.** `CustomTokenizer` rebuilds the Qwen tokenizer from `vocab.json` and `merges.txt` (found with the SDK's public methods). Text is split into pre-tokens, turned into bytes, mapped to printable characters and merged with BPE by lowest rank. Its output matches the SDK's `encode`, and `decode(encode(x)) == x`.
 
**Constrained decoding.** The code writes the fixed structure (`{"name": "`, parameter names, quotes, commas, braces) and the model only decides at the choice points. At each one the decoder works out which token ids are allowed, gets the logits with `get_logits_from_input_ids`, ignores everything not allowed (same as setting it to `-inf`), and takes the best remaining token.
 
| Choice point | Allowed tokens |
|---|---|
| Function name | Tokens that keep the text a prefix of a real function name; the closing quote only when the name is complete |
| `number` / `integer` | Digits, `-` and `.` while the text still looks like a number; `,` or `}` only after a complete number |
| `string` | Any token without control characters; an unescaped `"` closes the string (also tokens like `",`) |
| `boolean` | Prefixes of `true` or `false` |
| `array` | An item start, or `,` / `]` (recursive) |
| `object` | Keys come from the schema; the model fills the values (recursive) |
 
Since the structure is written by the code, all parameters are present and types match the schema. The result is a Python dict written with `json.dump`, so the file is always valid JSON.
 
**Natural tokenization when opening values.** The tokenizer does not split text the way a naive decoder would: `{"path": "/home/user` becomes `{"`, `path`, `":`, ` "/`, `home`, `/user`, and `-5` uses a single ` -` token. My first version forced a standalone ` "` (or ` `) before letting the model write, a sequence the model almost never sees, and it dropped characters (`/home` became `home`, `-5` became `5`). Now, at the start of a string or number, the model chooses among the natural opening tokens (` "`, ` "/`, ` "C`, ` `, ` -`), and the rest of the chosen token becomes the start of the value.
 
**Error recovery.** `generate_with_recovery` tries up to three settings (normal; with a repetition penalty; with a repetition penalty and a shorter limit). Each result goes through `validate_result`, which checks the function, the parameter names and the types. Dead ends and loops raise a `DecodingError`; if every attempt fails, only that prompt is skipped.
 
## Design decisions
 
- The code writes the structure; the model only fills the choice points. Counting braces cannot guarantee a valid schema.
- The function is chosen by the LLM; the mask only removes names that do not exist. No heuristics and no hardcoded examples (the few-shot examples use functions that are not in the test files).
- Strings and numbers open with a natural token (see above).
- The prompt is short (compact catalog), the model is skipped when only one token is valid, and the own tokenizer replaces the SDK's `encode` / `decode`.
- Pydantic is used for data and settings (`FunctionDefinition`, `TypeInfo`, `FunctionCallResult`, `DecoderSettings`, `DecodingConfig`, `Parser`, `PromptBuilder`). `CustomTokenizer` and `ConstrainedDecoder` also inherit from `BaseModel` (arbitrary types and extra attributes allowed), because they hold heavy internal state: the vocabulary, NumPy arrays and the model.
- Bad input stops early: missing or invalid files, and an invalid output path, give a clear error before the slow model loading.
- The program always picks one of the given functions, even for prompts that match none, because the mask only allows real names and the subject defines no "no match" output.
## Performance analysis
 
Tested on a MacBook Air (Apple silicon) with Qwen3-0.6B:
 
| Metric | Public examples (11) | Moulinette private set (11) |
|---|---|---|
| Valid JSON | 100% (by construction) | 100% |
| Correct function | 11 / 11 | 11 / 11 |
| Correct arguments | 11 / 11 | 10 / 11 (90.9%) |
| Total time | about 70 to 194 s | about 109 to 212 s |
 
The private set has 6 functions I had never seen, including an `integer` parameter and long decimal numbers. The only mistake is `Format template: Say "hello" to {name}`, where the model skips the part between the escaped quotes.
 
Making the catalog compact cut the total time from about 207 s to about 70 s, because every call reprocesses the whole prompt (the public SDK has no KV-cache). Timing varies a lot on this laptop (the same prompt can take 10 s or 90 s, and one full run took 468 s), probably because of heat or memory pressure; a prompt run alone took the same time as a fast run, so the variation is not in the code. On a slower machine the full run could get close to the 5-minute limit.
 
`Qwen3-1.7B` also works with the same tokenizer, but took 469 s here and was not more accurate.
 
Limits of the 0.6B model (the structure is always valid, but values can be imperfect):
 
- It can skip text between escaped quotes (see above).
- With nested arguments it sometimes closes an array right away (`"tags": []`).
- Empty or meaningless prompts still produce a valid call with invented values.
- There is no KV-cache or batching, since that would need private SDK methods.
## Challenges faced
 
- **Empty results at first.** My first tokenizer split on spaces and looked up whole words, so the model saw almost no prompt. A real byte-level BPE fixed it.
- **Masks that guaranteed nothing.** Counting braces gives neither valid JSON nor correct types; generating the structure from the schema did.
- **Strings that never ended.** Tokens like `",` were forbidden, so long strings looped. Letting any unescaped quote close the string fixed it.
- **Repetition loops** in regex values: solved with loop detection, a repetition penalty and retries.
- **Lost characters.** The moulinette showed `home/user/data.json` instead of `/home/user/data.json`, and `-5` came out as `5.0`. The `--verbose` trace and the natural tokenization of the text showed that my decoder forced an unusual token sequence. Letting the model choose the opening token fixed both and raised the private score from 9/11 to 10/11.
- **Slow runs:** solved with a shorter prompt and by skipping the model when only one token is valid.
