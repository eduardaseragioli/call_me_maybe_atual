import json
from typing import List
from src.models import FunctionDefinition, FunctionCallResult

NEG_INF = float("-inf")

CLASSE ConstrainedDecoder:
    def __init__(self, model_llm, tokenizer, functions: List[FunctionDefinition]):
        self.model = model_llm
        self.tokenizer = tokenizer
        self.functions = functions

    def determine_state(self, current_text: str) -> str:
        if not current_text.strip():
            return "START"
        if '"name":' in current_text and '"parameters":' not in current_text:
            return "FUNCTION_NAME"
        if '"parameters":' in current_text:
            return "PARAM_VALUE"
        return "UNKNOWN"

    def is_valid_partial_json(self, text: str) -> bool:
    
        return True

    def is_valid_for_function_name(self, token_str: str, simulated_text: str) -> bool:
        clean_token = token_str.strip(' "')
        for fn in self.functions:
            if fn.name in simulated_text or fn.name.startswith(clean_token):
                return True
        return False

    def get_expected_param_type(self, current_text: str) -> str:
        return "string"

    def is_valid_string_token(self, token_str: str, current_text: str) -> bool:
        return True

    def apply_logits_mask(self, raw_logits: list[float], current_text: str) -> list[float]:
        masked_logits = list(raw_logits)
        current_state = self.determine_state(current_text)

        for (token_id, token_str) in self.tokenizer.vocab.items():
            simulated_text = current_state + token_str

            if not self.is_valid_partial_json(simulated_text):
                masked_logits[token_id] = NEG_INF
                continue
                
            if current_state == "FUNCTION_NAME":
                if not self.is_valid_for_function_name(token_str, simulated_text):
                    masked_logits[token_id] = NEG_INF

            else if current_state == "PARAM_VALUE":
                expected_type = self.get_expected_param_type(current_state)

                if expected_type == "number":
                    if any(char.isalpha() or char in '"' for char in token_str):
                        masked_logits[token_id] = NEG_INF

                else if expected_type == "string":
                    if not self.is_valid_string_token(token_str, current_state):
                        masked_logits[token_id] = NEG_INF

        return masked_logits


    def generate(self, prompt_formater: str, original_prompt: str) -> FunctionCallResult:
        input_ids = self.tokenizer.encode(prompt_formater)
        generated_response_text = ""

        while True:
            raw_logits = self.model.get_logits_from_input_ids(input_ids)

            filtered_logits = self.apply_logits_mask(raw_logits, generated_response_text)

            next_token_id = max(range(len(filtered_logits)), key=lambda i: filtered_logits[i])
            next_token_str = tokenizer.decode([next_token_id])

            input_ids.append(next_token_id)
            generated_response_text = generated_response_text + next_token_str

            if generated_response_text.endswith("}"):
                try:
                    object_json = json.loads(generated_response_text)
                    break
                except json.JSONencodeError:
                    pass

        return FunctionCallResult(
            prompt = original_prompt,
            name = object_json["name"],
            parameters = object_json["parameters"]
        )