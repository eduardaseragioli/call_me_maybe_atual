import argparse
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "llm_sdk"))

from llm_sdk import Small_LLM_Model

from src.constrained_decoder import ConstrainedDecoder
from src.custom_tokenizer import CustomTokenizer
from src.models import FunctionCallResult, DecoderSettings
from src.parser import Parser
from src.prompt_builder import PromptBuilder

def parse_arguments() -> argparse.Namespace:
    """Configures and processes command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Call Me Maybe - Function Calling using Constrained Decoding"
    )

    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the JSON file containing function definitions.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the input test/prompt JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help=" Path to the output JSON file containing the results.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show, token by token, what the model wanted vs. what the "
             "mask allowed.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Hugging Face model id (must use a Qwen-style byte-level "
             "BPE tokenizer: vocab.json + merges.txt).",
    )

    return parser.parse_args()

def main() -> None:
    args = parse_arguments()
    settings = DecoderSettings(verbose=args.verbose, model_name=args.model)

    functions = Parser.load_functions(args.functions_definition)
    tests = Parser.load_tests(args.input)
    if not functions or not tests:
        print("Error: invalid or empty input files.", file=sys.stderr)
        sys.exit(1)

    print(f"Functions loaded: {len(functions)}")
    print(f"Tests loaded: {len(tests)}")

    print("\nLoading model and tokenizer...")
    try:
        model = Small_LLM_Model(model_name=settings.model_name)
        tokenizer = CustomTokenizer(
            model.get_path_to_vocab_file(),
            model.get_path_to_merges_file(),
        )
        decoder = ConstrainedDecoder(
            model_llm=model,
            tokenizer=tokenizer,
            functions=functions,
            verbose=settings.verbose,
        )
    except Exception as e:
        print(f"Error: could not initialize the model: {e}",
              file=sys.stderr)
        sys.exit(1)

    results: list[FunctionCallResult] = []
    print("\nStarting constrained decoding...\n")
    start = time.perf_counter()

    for idx, test in enumerate(tests, 1):
        print(f"--- [Test {idx}/{len(tests)}] ---")
        print(f"Prompt: {test.prompt}")
        t0 = time.perf_counter()
        try:
            formatted_prompt = PromptBuilder.build_prompt(
                functions, test.prompt)
            result = decoder.generate_with_recovery(
                formatted_prompt, test.prompt)
        except Exception as e:
            print(f"Error on this prompt, skipping: {e}\n",
                  file=sys.stderr)
            continue

        print(f"Function: {result.name}")
        print(f"Parameters: {result.parameters}")
        print(f"Time: {time.perf_counter() - t0:.1f}s\n")
        if settings.verbose:
            avg = decoder.forward_time / max(decoder.forward_calls, 1)
            print(f"Model calls: {decoder.forward_calls}, avg {avg:.2f}s")
        decoder.forward_calls = 0
        decoder.forward_time = 0.0
        results.append(result)

    total = time.perf_counter() - start
    print(f"Total generation time: {total:.1f}s")
    if total > 300:
        print("Warning: exceeded the 5-minute limit.", file=sys.stderr)

    if not results:
        print("Error: no results were generated.", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error: cannot create output directory: {e}",
              file=sys.stderr)
        sys.exit(1)

    if Parser.save_results(results, str(output_path)):
        print(f"Done! Results saved to: {output_path}")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
