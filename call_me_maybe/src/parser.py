import json
from typing import List
from src.models import FunctionDefinition, TestCase, FunctionCallResult
import pydantic

class Parser:

    @staticmethod
    def load_functions(file_path: str) -> list[FunctionDefinition]:
        try:
              with open(file_path, 'r') as file:
                  data = json.load(file)
                  return [FunctionDefinition(**item) for item in data]

        except FileNotFoundError:
            print("File not found")
            return []
        except json.JSONDecodeError:
            print("JSON file: incorrect sintax")
            return []
        except pydantic.ValidationError:
            print("JSON struct is not in the model")
            return []

    @staticmethod
    def load_tests(file_path: str) -> list[TestCase]:
        try:
            with open(file_path, 'r') as file:
                  data = json.load(file)
                  return [TestCase(**item) for item in data]

        except FileNotFoundError:
            print("File not found")
            return []
        except json.JSONDecodeError:
            print("JSON file: incorrect sintax")
            return []
        except pydantic.ValidationError:
            print("JSON struct is not in the model")
            return []

    @staticmethod
    def save_results(results: List[FunctionCallResult], file_path: str) -> bool:
        try:
            data = [result.model_dump() for result in results]
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=2, ensure_ascii=False)
            return True
        except IOError as e:
            print(f"Error writing output file: {e}")
            return False
