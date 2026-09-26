import json

class CustomTokenizer:

    def __init__(self, vocab_path) -> None:
        self.vocab_path = vocab_path
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

        try:
            with open(vocab_path, 'r') as file:
                content = json.load(file)

                for token, token_id in content.items():
                    self.token_to_id[str(token)] = int(token_id)
                    self.id_to_token[int(token_id)] = str(token)

        except Exception as e:
                   print(f"An unexpected error occurred {e}.")

    def encode(self, text: str) -> list[int]:
        ids_list: list[int] = []

        words = text.split(" ")
        for word in words:
            token_id = self.token_to_id.get(word)

            if token_id is not None:
                ids_list.append(token_id)
        return (ids_list)

    def decode(self, ids_list: list[int]) -> str:
        str_list: list[str] = []

        for token_id in ids_list:
            token = self.id_to_token.get(token_id)

            if token is not None:
                str_list.append(token)
        return " ".join(str_list)
