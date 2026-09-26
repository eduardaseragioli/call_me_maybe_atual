import json
from typing import List
from src.models import FunctionDefinition

SYSTEM_PROMPT_TEMPLATE = """You are a function calling assistant.
Given the following available functions, respond ONLY with a valid JSON object matching the required structure.

Available Functions:
{functions_catalog}

Output JSON format:
{{
  "name": "<function_name>",
  "parameters": {{
    "<parameter_name>": <value>
  }}
}}

User prompt: {user_prompt}
JSON response:"""


class PromptBuilder:

    @staticmethod
    def format_functions_catalog(list_of_functions: List[FunctionDefinition]) -> str:
        list_of_dit: list[dict] = []

        for function in list_of_functions:
            parameters_dict = {
                param_name: param_info.model_dump()
                for param_name, param_info in function.parameters.items()
            }

            dictionary = {
                "name": function.name,
                "description": function.description,
                "parameters": parameters_dict,
            }
            list_of_dit.append(dictionary)
        return json.dumps(list_of_dit, indent=2)

    @staticmethod
    def build_prompt(list_of_functions: List[FunctionDefinition], user_prompt: str) -> str:
        catalog_string = PromptBuilder.format_functions_catalog(list_of_functions)

        final_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            functions_catalog=catalog_string,
            user_prompt=user_prompt
        )

        return (final_prompt)