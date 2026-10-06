import json
import sys
from typing import Any
from pydantic import BaseModel, ValidationError
from src.models import FunctionDefinition, TestCase, FunctionCallResult

class Parser(BaseModel):

    @staticmethod
    def _read_list(file_path: str) -> list[Any] | None:
        try:
            with open(file_path, encoding="utf-8") as file:
                data = json.load(file)
        except FileNotFoundError:
            print(f"Error: file not found: {file_path}", file=sys.stderr)
            return None
        except json.JSONDecodeError as exc:
            print(f"Error: invalid JSON in {file_path}: {exc}",
                  file=sys.stderr)
            return None
        except (OSError, UnicodeDecodeError) as exc:
            print(f"Error: cannot read {file_path}: {exc}",
                  file=sys.stderr)
            return None
        if not isinstance(data, list):
            print(f"Error: {file_path} must contain a JSON list",
                  file=sys.stderr)
            return None
        return data

    @staticmethod
    def load_functions(file_path: str) -> list[FunctionDefinition]:
        data = Parser._read_list(file_path)
        if data is None:
            return []
        try:
            return [FunctionDefinition(**item) for item in data]
        except (ValidationError, TypeError) as exc:
            print(f"Error: bad function definition in {file_path}: {exc}",
                  file=sys.stderr)
            return []

    @staticmethod
    def load_tests(file_path: str) -> list[TestCase]:
        data = Parser._read_list(file_path)
        if data is None:
            return []
        try:
            return [TestCase(**item) for item in data]
        except (ValidationError, TypeError) as exc:
            print(f"Error: bad test case in {file_path}: {exc}",
                  file=sys.stderr)
            return []

    @staticmethod
    def save_results(results: list[FunctionCallResult],
                     file_path: str) -> bool:
        try:
            data = [result.model_dump() for result in results]
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=2, ensure_ascii=False)
        except OSError as exc:
            print(f"Error writing output file: {exc}", file=sys.stderr)
            return False
        return True
