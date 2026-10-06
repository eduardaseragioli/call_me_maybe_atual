import json
import re

PAT = re.compile(
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
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}

class CustomTokenizer:

    def __init__(self, vocab_path: str, merges_path: str) -> None:
        with open(vocab_path, encoding="utf-8") as f:
            self.token_to_id: dict[str, int] = json.load(f)
        self.id_to_token: dict[int, str] = {
            v: k for k, v in self.token_to_id.items()}
        self.vocab_size: int = max(self.id_to_token) + 1

        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        self.ranks: dict[tuple[str, str], int] = {}
        with open(merges_path, encoding="utf-8") as f:
            lines = [ln for ln in f.read().split("\n")
                     if ln and not ln.startswith("#version")]
        for rank, line in enumerate(lines):
            a, b = line.split(" ")
            self.ranks[(a, b)] = rank
        self._cache: dict[str, list[str]] = {}

    def _bpe(self, word: str) -> list[str]:
        if word in self._cache:
            return self._cache[word]
        parts = list(word)
        while len(parts) > 1:
            pairs = [(parts[i], parts[i + 1])
                     for i in range(len(parts) - 1)]
            best = min(pairs, key=lambda p: self.ranks.get(p, 10**9))
            if best not in self.ranks:
                break
            merged: list[str] = []
            i = 0
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
        ids: list[int] = []
        for piece in PAT.findall(text):
            mapped = "".join(self.byte_encoder[b]
                             for b in piece.encode("utf-8"))
            for tok in self._bpe(mapped):
                ids.append(self.token_to_id[tok])
        return ids

    def decode(self, ids: list[int]) -> str:
        raw = bytearray()
        for i in ids:
            for c in self.id_to_token.get(i, ""):
                raw.append(self.byte_decoder[c])
        return raw.decode("utf-8", errors="replace")

    def token_text(self, token_id: int) -> str:
        try:
            return bytes(self.byte_decoder[c]
                         for c in self.id_to_token.get(token_id, "")
                         ).decode("utf-8")
        except (KeyError, UnicodeDecodeError):
            return ""