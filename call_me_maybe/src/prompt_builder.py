import json
from typing import Any, List, Dict

from pydantic import BaseModel

from src.models import FunctionDefinition

SYSTEM_PROMPT_TEMPLATE = """You are a function calling assistant.
Given the available functions, output ONE JSON object with the function \
name and its parameters.

Rules:
- Pick the function whose description best matches the request.
- Copy string values exactly as written by the user.
- Copy string values character by character, including leading slashes, \
quotes and punctuation. Do not add or remove anything.
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
Request: Open the file /var/log/app.log
Answer: {{"name": "fn_open", "parameters": {{"path": "/var/log/app.log"}}}}
Request: Show the message: He said "stop" now
Answer: {{"name": "fn_show", "parameters": {{"text": \
"He said \\"stop\\" now"}}}}

Reminder: copy every string value exactly from the request, \
character by character.

Request: {user_prompt}
Answer:"""


class PromptBuilder(BaseModel):
    """Builds and formats system prompts containing function definitions and user requests."""

    @staticmethod
    def format_functions_catalog(
        list_of_functions: list[FunctionDefinition],
    ) -> str:
        """Formats a list of function definitions into a compact JSON string catalog."""

        catalog: list[dict[str, Any]] = []
        for function in list_of_functions:
            parameters_dict: dict[str, Any] = {
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
        """Builds the final prompt string by filling the system prompt template."""
        
        return SYSTEM_PROMPT_TEMPLATE.format(
            functions_catalog=PromptBuilder.format_functions_catalog(
                list_of_functions),
            user_prompt=user_prompt,
        )
