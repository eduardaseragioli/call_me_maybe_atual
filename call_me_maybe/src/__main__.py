import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "llm_sdk" / "llm_sdk"))
sys.path.insert(0, str(ROOT_DIR / "llm_sdk"))

from custom_tokenizer import CustomTokenizer
from src.parser import Parser
from prompt_builder import PromptBuilder

try:
    from llm_sdk import Small_LLM_Model
except ImportError:
    from llm_sdk.llm_sdk import Small_LLM_Model



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

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    functions = Parser.load_functions(args.functions_definition)
    tests = Parser.load_tests(args.input)

    print(f"Funções carregadas: {len(functions)}")
    for fn in functions:
        print(f" - {fn.name}: {fn.description}")

    print(f"\nTestes carregados: {len(tests)}")
    for test in tests:
        print(f" - Prompt: {test.prompt}")


    if tests and functions:
        sample_prompt = PromptBuilder.build_prompt(functions, tests[0].prompt)
        print("\n--- Exemplo de Prompt Gerado ---")
        print(sample_prompt)


    #model = Small_LLM_Model()
    #vocab_path = model.get_path_to_vocab_file()

    #tokenizer = CustomTokenizer(vocab_path)

    #texto_original = "Hello world"
    #ids = tokenizer.encode(texto_original)
   # texto_recuperado = tokenizer.decode(ids)

    #print(f"Original: {texto_original}")
    #print(f"IDs: {ids}")
    #print(f"Recuperado: {texto_recuperado}")


if __name__ == "__main__":
    main()
