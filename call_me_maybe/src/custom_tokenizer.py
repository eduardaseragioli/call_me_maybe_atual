import json
import re
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, ConfigDict

PAT: re.Pattern[str] = re.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|(?:(?![\r\n])[\W_])?[^\W\d_]+"
    r"|\d"
    r"| ?(?:[^\s\w]|_)+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def _bytes_to_unicode() -> dict[int, str]:
    """Map every byte to a printable character (GPT-2 scheme)."""
    bs: list[int] = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs: list[int] = bs[:]
    n: int = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class CustomTokenizer(BaseModel):
    """Byte-level BPE tokenizer (Qwen compatible)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(self, vocab_path: str, merges_path: str) -> None:
        """Initializes the tokenizer by loading vocabulary, merges, and setting up encoder mappings."""

        super().__init__()
        with open(vocab_path, encoding="utf-8") as f:
            self.token_to_id: dict[str, int] = json.load(f)

        self.id_to_token: dict[int, str] = {
            v: k for k, v in self.token_to_id.items()}
        self.vocab_size: int = max(self.id_to_token) + 1

        self.byte_encoder: dict[int, str] = _bytes_to_unicode()
        self.byte_decoder: dict[str, int] = {v: k for k, v in self.byte_encoder.items()}

        self.ranks: dict[tuple[str, str], int] = {}
        with open(merges_path, encoding="utf-8") as f:
            lines: list[str] = [ln for ln in f.read().split("\n")
                     if ln and not ln.startswith("#version")]
        for rank, line in enumerate(lines):
            a, b = line.split(" ")
            self.ranks[(a, b)] = rank
        self._cache: dict[str, list[str]] = {}

    def _bpe(self, word: str) -> list[str]:
        """Applies Byte-Pair Encoding (BPE) tokenization to a mapped word string."""

        if word in self._cache:
            return self._cache[word]
        parts: list[str] = list(word)
        while len(parts) > 1:
            pairs: list[tuple[str, str]] = [(parts[i], parts[i + 1])
                     for i in range(len(parts) - 1)]
            best: tuple[str, str] = min(pairs, key=lambda p: self.ranks.get(p, 10**9))
            if best not in self.ranks:
                break
            merged: list[str] = []

            i: int = 0
            while i < len(parts):
                if (i < len(parts) - 1
                        and (parts[i], parts[i + 1]) == best):
                    merged.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = merged
        self._cache[word] = parts
        return parts

    def encode(self, text: str) -> list[int]:
        """Encodes an input text string into a list of token integer IDs."""

        ids: list[int] = []
        for piece in PAT.findall(text):
            mapped: str = "".join(self.byte_encoder[b]
                             for b in piece.encode("utf-8"))
            for tok in self._bpe(mapped):
                ids.append(self.token_to_id[tok])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decodes a list of token integer IDs back into a text string."""

        raw: bytearray = bytearray()
        for i in ids:
            for c in self.id_to_token.get(i, ""):
                raw.append(self.byte_decoder[c])
        return raw.decode("utf-8", errors="replace")

    def token_text(self, token_id: int) -> str:
        """Retrieves the literal text representation of a single token ID safely."""
        
        try:
            return bytes(self.byte_decoder[c]
                         for c in self.id_to_token.get(token_id, "")
                         ).decode("utf-8")
        except (KeyError, UnicodeDecodeError):
            return ""
