import json
from typing import Any

from pydantic import BaseModel

from src.models import FunctionDefinition

SYSTEM_PROMPT_TEMPLATE = """You are a function calling assistant.
Given the available functions, output ONE JSON object with the function \
name and its parameters.

Rules:
- Pick the function whose description best matches the request.
- Copy string values exactly as written by the user.
- For regex parameters, write a regex that matches the CONCEPT described, \
not the words used to describe it (a class of characters goes inside [...]).
- For replacement parameters, write the literal replacement characters \
(the symbol itself, not its name).
- Keep every value as short as possible.
- Escape double quotes inside string values with a backslash.

Available Functions:
{functions_catalog}

Examples of the expected output format:
Request: Multiply 4 by 7
Answer: {{"name": "fn_multiply", "parameters": {{"a": 4, "b": 7}}}}
Request: Replace all spaces in 'a b c' with dashes
Answer: {{"name": "fn_replace", "parameters": {{"text": "a b c", \
"pattern": " ", "replacement": "-"}}}}
Request: Replace all uppercase letters in 'aBcD' with hashes
Answer: {{"name": "fn_replace", "parameters": {{"text": "aBcD", \
"pattern": "[A-Z]", "replacement": "#"}}}}

Request: {user_prompt}
Answer:"""


class PromptBuilder(BaseModel):

    @staticmethod
    def format_functions_catalog(
        list_of_functions: list[FunctionDefinition],
    ) -> str:
        catalog: list[dict[str, Any]] = []
        for function in list_of_functions:
            parameters_dict = {
                name: info.model_dump(exclude_none=True)
                for name, info in function.parameters.items()
            }
            catalog.append({
                "name": function.name,
                "description": function.description,
                "parameters": parameters_dict,
            })
        return json.dumps(catalog, ensure_ascii=False,
                          separators=(",", ":"))

    @staticmethod
    def build_prompt(
        list_of_functions: list[FunctionDefinition], user_prompt: str
    ) -> str:
        return SYSTEM_PROMPT_TEMPLATE.format(
            functions_catalog=PromptBuilder.format_functions_catalog(
                list_of_functions),
            user_prompt=user_prompt,
        )