"""CustomTokenizer on a tiny synthetic byte-level BPE vocabulary."""

import json
from pathlib import Path

from src.custom_tokenizer import CustomTokenizer, _bytes_to_unicode


def build(tmp_path: Path) -> tuple[CustomTokenizer, dict[str, int]]:
    """256 byte tokens plus one merge: 'a' + 'b' -> 'ab'."""
    vocab = {ch: i for i, ch in enumerate(_bytes_to_unicode().values())}
    vocab["ab"] = len(vocab)
    v = tmp_path / "vocab.json"
    m = tmp_path / "merges.txt"
    v.write_text(json.dumps(vocab), encoding="utf-8")
    m.write_text("#version: 0.2\na b\n", encoding="utf-8")
    return CustomTokenizer(str(v), str(m)), vocab


def test_merge_is_applied(tmp_path: Path) -> None:
    """'ab' becomes a single token."""
    tok, vocab = build(tmp_path)
    assert tok.encode("ab") == [vocab["ab"]]


def test_roundtrip(tmp_path: Path) -> None:
    """decode(encode(x)) == x, including accents and newlines."""
    tok, _ = build(tmp_path)
    for text in ["", "hello world", "olá, ação\nnova linha", "42 + 3.5",
                 "tab\there", "日本語"]:
        assert tok.decode(tok.encode(text)) == text


def test_partial_utf8_token_has_empty_text(tmp_path: Path) -> None:
    """A lone first byte of 'á' is not valid text on its own."""
    tok, vocab = build(tmp_path)
    first_byte = _bytes_to_unicode()[0xC3]
    assert tok.token_text(vocab[first_byte]) == ""