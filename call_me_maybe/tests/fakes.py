"""Tiny fake tokenizer and models so decoder tests run without the LLM."""

import string

CHARS = string.ascii_letters + string.digits + " _.,-:{}\"'\\()[]+*?|^$/"
EXTRAS = ["fn_add", "fn_greet", "fn_flag", "true", "false", '",', '"}']


class FakeTokenizer:
    """Greedy longest-match tokenizer over a ~100 token vocabulary."""

    def __init__(self) -> None:
        """Build the vocabulary."""
        self.vocab: list[str] = list(CHARS) + EXTRAS
        self.vocab_size = len(self.vocab)
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}

    def token_text(self, token_id: int) -> str:
        """Text of one token."""
        return self.vocab[token_id]

    def encode(self, text: str) -> list[int]:
        """Greedy longest-match encoding."""
        ids: list[int] = []
        pos = 0
        while pos < len(text):
            best = max((t for t in self.vocab if text.startswith(t, pos)),
                       key=len)
            ids.append(self.token_to_id[best])
            pos += len(best)
        return ids


class ScriptedModel:
    """Fake LLM that always 'wants' to continue writing `target`."""

    def __init__(self, tok: FakeTokenizer, prefix_len: int,
                 target: str) -> None:
        """Remember the target text and how long the prompt is."""
        self.tok = tok
        self.prefix_len = prefix_len
        self.target = target
        self.calls = 0

    def get_logits_from_input_ids(self, ids: list[int]) -> list[float]:
        """High logit on the longest token that continues the target."""
        self.calls += 1
        done = "".join(self.tok.vocab[i] for i in ids[self.prefix_len:])
        rest = self.target[len(done):]
        logits = [0.0] * self.tok.vocab_size
        fits = [i for i, t in enumerate(self.tok.vocab)
                if rest.startswith(t)]
        if fits:
            logits[max(fits, key=lambda i: len(self.tok.vocab[i]))] = 10.0
        return logits


class ConstantModel:
    """Fake LLM that always wants one specific token."""

    def __init__(self, vocab_size: int, wanted: int) -> None:
        """Store the favourite token id."""
        self.vocab_size = vocab_size
        self.wanted = wanted
        self.calls = 0

    def get_logits_from_input_ids(self, ids: list[int]) -> list[float]:
        """Return logits with a single peak."""
        self.calls += 1
        logits = [0.0] * self.vocab_size
        logits[self.wanted] = 10.0
        return logits