import json
import sys
import re
import time
from typing import Any, Sequence
from collections import OrderedDict
import numpy as np
from src.models import FunctionDefinition, FunctionCallResult, TypeInfo
from pydantic import BaseModel, ConfigDict

FLOAT_PREFIX = re.compile(r"-?\d*\.?\d*")
FLOAT_FULL = re.compile(r"-?\d+(\.\d+)?")
INT_PREFIX = re.compile(r"-?\d*")
INT_FULL = re.compile(r"-?\d+")
CACHE_SIZE = 16
STUCK_WINDOW = 6
MAX_ITEMS = 16


class DecodingError(Exception):
    """Raised when decoding reaches a dead end."""
    pass


class DecodingConfig(BaseModel):
    """Knobs that change between recovery attempts."""
    max_value_tokens: int = 64
    repeat_penalty: float = 0.0
    strict: bool = True


ATTEMPTS = [
    DecodingConfig(),
    DecodingConfig(repeat_penalty=5.0),
    DecodingConfig(repeat_penalty=5.0, max_value_tokens=24, strict=False),
]


class ConstrainedDecoder(BaseModel):
    """Decoder that masks logits so only schema-valid tokens survive."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(self, model_llm: Any, tokenizer: Any,
                 functions: list[FunctionDefinition],
                 verbose: bool = False) -> None:
        """Precompute the text of every token and the static masks."""
        super().__init__()
        self.model = model_llm
        self.tokenizer = tokenizer
        self.functions = {fn.name: fn for fn in functions}
        self.verbose = verbose
        self.config = DecodingConfig()
        self.forward_calls = 0
        self.forward_time = 0.0
        self._cache: OrderedDict[tuple[int, ...], np.ndarray] = \
            OrderedDict()
        self.texts = [tokenizer.token_text(i)
                      for i in range(tokenizer.vocab_size)]
        self.quote_id: int = tokenizer.token_to_id['"']
        self.number_ids = [i for i, t in enumerate(self.texts)
                           if t and all(c in "0123456789.-" for c in t)]
        self.end_ids = [i for i, t in enumerate(self.texts)
                        if t and len(t) <= 2 and t.strip() in (",", "}")]
        self.end_set = set(self.end_ids)
        self.string_allowed = np.array(
            [i for i, t in enumerate(self.texts)
             if t and all(ord(c) >= 32 for c in t)],
            dtype=np.int64)
        self.name_ids = self._candidate_ids(list(self.functions))
        self.bool_ids = self._candidate_ids(["true", "false"])
        self._opener_cache: dict[str, list[int]] = {}
        self.array_end_ids = [
            i for i, t in enumerate(self.texts)
            if t and len(t) <= 2 and t.strip() in (",", "]")]
        ids_of = tokenizer.token_to_id
        self.close_id: int = ids_of["]"]
        self.comma_id: int = ids_of[","]
        self.open_obj_id: int = ids_of["{"]
        self.open_arr_id: int = ids_of["["]
        self.num_start_ids = [i for i in self.number_ids
                              if self.texts[i][0] in "0123456789-"]
        self.bool_start_ids = [
            i for i in self.bool_ids
            if "true".startswith(self.texts[i])
            or "false".startswith(self.texts[i])]

    def _candidate_ids(self, options: list[str]) -> list[int]:
        return [i for i, t in enumerate(self.texts)
                if t and any(t in o for o in options)]

    def _logits(self, ids: list[int]) -> np.ndarray:
        key = tuple(ids)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        t = time.perf_counter()
        result = np.asarray(self.model.get_logits_from_input_ids(ids),
                            dtype=np.float32)
        self.forward_time += time.perf_counter() - t
        self.forward_calls += 1
        self._cache[key] = result
        if len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return result

    def _pick(self, ids: list[int], allowed: Sequence[int] | np.ndarray,
              recent: Sequence[int] = ()) -> int:
        if len(allowed) == 0:
            raise DecodingError("no valid token to choose from")
        if len(allowed) == 1:
            if self.verbose:
                print(f"  forced: {self.texts[int(allowed[0])]!r}")
            return int(allowed[0])

        logits = self._logits(ids)
        cand = np.asarray(allowed, dtype=np.int64)
        cand = cand[cand < logits.shape[0]]

        if cand.size == 0:
            raise DecodingError("allowed tokens outside vocabulary")

        scores = logits[cand].copy()
        if self.config.repeat_penalty and recent:
            scores[np.isin(cand, list(set(recent)))] -= \
                self.config.repeat_penalty
        chosen = int(cand[np.argmax(scores)])
        if self.verbose:
            self._trace(logits, cand, chosen)
        return chosen

    def _trace(self, logits: np.ndarray, cand: np.ndarray,
               chosen: int) -> None:
        top = np.argpartition(logits, -3)[-3:]
        top = top[np.argsort(-logits[top])]
        ok = set(cand.tolist())
        wants = " ".join(
            f"{self.texts[t]!r}{'' if t in ok else '[MASKED]'}"
            for t in top.tolist())
        print(f"  allowed={cand.size} model wants: {wants} "
              f"-> {self.texts[chosen]!r}")

    def _choose(self, ids: list[int], options: list[str],
                cands: list[int], allow_quote_end: bool) -> str:
        cur = ""
        while True:
            longer = [o for o in options if o.startswith(cur) and o != cur]
            if cur in options and not longer:
                return cur
            allowed = [i for i in cands if any(
                o.startswith(cur + self.texts[i]) for o in options)]
            if allow_quote_end and cur in options:
                allowed.append(self.quote_id)
            tok = self._pick(ids, allowed)
            if allow_quote_end and cur in options and tok == self.quote_id:
                return cur
            cur += self.texts[tok]
            ids.append(tok)

    def _gen_number(self, ids: list[int], integer: bool,
                    enders: list[int] | None = None,
                    start: str = "") -> float | int:
        ender_ids = self.end_ids if enders is None else enders
        ender_set = set(ender_ids)
        prefix_re = INT_PREFIX if integer else FLOAT_PREFIX
        full_re = INT_FULL if integer else FLOAT_FULL
        text = start
        for _ in range(self.config.max_value_tokens):
            allowed = [i for i in self.number_ids
                       if prefix_re.fullmatch(text + self.texts[i])]
            if full_re.fullmatch(text):
                allowed = allowed + ender_ids
            tok = self._pick(ids, allowed)
            if tok in ender_set:
                break
            text += self.texts[tok]
            ids.append(tok)
        try:
            return int(text) if integer else float(text)
        except ValueError as exc:
            raise DecodingError(f"invalid number {text!r}") from exc

    @staticmethod
    def _is_stuck(tokens: list[int]) -> bool:
        w = STUCK_WINDOW
        return len(tokens) >= 2 * w and tokens[-w:] == tokens[-2 * w:-w]

    @staticmethod
    def _closing_quote(prefix: str, text: str) -> int:
        escaped = (len(prefix) - len(prefix.rstrip("\\"))) % 2 == 1
        for i, ch in enumerate(text):
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                return i
        return -1

    def _opener_ids(self, opener: str) -> list[int]:
        """Tokens that start with 'opener' (cached, built once)."""
        cached = self._opener_cache.get(opener)
        if cached is None:
            cached = [i for i, t in enumerate(self.texts)
                      if t.startswith(opener)
                      and all(ord(c) >= 32 for c in t)]
            self._opener_cache[opener] = cached
        return cached

    def _close_string(self, ids: list[int], raw: str) -> str:
        """Write the closing quote and decode JSON escapes."""
        ids.extend(self.tokenizer.encode('"'))
        try:
            return str(json.loads(f'"{raw}"'))
        except json.JSONDecodeError:
            return raw

    def _gen_quoted(self, ids: list[int], lead: str) -> str:
        """Open a string letting the model pick a natural first token."""
        enc = self.tokenizer.encode
        head = lead.rstrip()
        opener = lead[len(head):] + '"'
        options = self._opener_ids(opener) if opener != '"' else []
        if not options:
            ids.extend(enc(lead + '"'))
            return self._gen_string(ids)
        if head:
            ids.extend(enc(head))
        tok = self._pick(ids, options)
        start = self.texts[tok][len(opener):]
        cut = self._closing_quote("", start)
        if cut >= 0:
            raw = start[:cut]
            ids.extend(enc(opener + raw))
            return self._close_string(ids, raw)
        ids.append(tok)
        return self._gen_string(ids, start)

    def _gen_string(self, ids: list[int], start: str = "") -> str:
        """Generate a JSON string; an unescaped '"' closes it."""
        gen: list[int] = []
        pieces: list[str] = [start] if start else []
        for _ in range(self.config.max_value_tokens):
            tok = self._pick(ids, self.string_allowed, gen[-20:])
            text = self.texts[tok]
            cut = self._closing_quote("".join(pieces), text)
            if cut >= 0:
                before = text[:cut]
                pieces.append(before)
                if before:
                    ids.extend(self.tokenizer.encode(before))
                break
            ids.append(tok)
            gen.append(tok)
            pieces.append(text)
            if self._is_stuck(gen):
                if self.config.strict:
                    raise DecodingError("model stuck repeating tokens")
                del pieces[-STUCK_WINDOW:]
                break
        return self._close_string(ids, "".join(pieces))

    def generate(self, prompt_formatted: str, original_prompt: str) -> FunctionCallResult:
        enc = self.tokenizer.encode
        ids: list[int] = enc(prompt_formatted + ' {"name": "')

        name = self._choose(ids, list(self.functions),
                            self.name_ids, allow_quote_end=True)
        ids += enc('", "parameters": {')

        params: dict[str, Any] = {}
        for n, (pname, info) in enumerate(
                self.functions[name].parameters.items()):
            sep = ", " if n else ""
            params[pname] = self._gen_value(ids, info,
                                            f'{sep}"{pname}": ')
        return FunctionCallResult(prompt=original_prompt, name=name,
                                  parameters=params)

    def _matches(self, value: Any, info: TypeInfo) -> bool:
        kind = info.type
        if kind == "number":
            return isinstance(value, float)
        if kind == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if kind == "string":
            return isinstance(value, str)
        if kind == "boolean":
            return isinstance(value, bool)
        if kind == "array":
            return (isinstance(value, list) and info.items is not None
                    and all(self._matches(v, info.items) for v in value))
        if kind == "object":
            props = info.properties or {}
            return (isinstance(value, dict) and set(value) == set(props)
                    and all(self._matches(value[k], p)
                            for k, p in props.items()))
        return False

    def validate_result(self, result: FunctionCallResult) -> str | None:
        fn = self.functions.get(result.name)
        if fn is None:
            return f"unknown function {result.name!r}"
        if set(result.parameters) != set(fn.parameters):
            return "parameter names do not match the schema"
        for key, info in fn.parameters.items():
            if not self._matches(result.parameters[key], info):
                return f"parameter {key!r} has the wrong type"
        return None

    def _start_ids(self, info: TypeInfo) -> list[int]:
        starts: dict[str, list[int]] = {
            "string": [self.quote_id],
            "number": self.num_start_ids,
            "integer": self.num_start_ids,
            "boolean": self.bool_start_ids,
            "object": [self.open_obj_id],
            "array": [self.open_arr_id],
        }
        if info.type not in starts:
            raise DecodingError(f"unsupported type: {info.type}")
        return starts[info.type]

    def _gen_signed(self, ids: list[int], lead: str, integer: bool,
                    enders: list[int] | None) -> float | int:
        """Open a number letting the model pick ' ' or ' -' naturally."""
        enc = self.tokenizer.encode
        head = lead.rstrip()
        gap = lead[len(head):]
        key = "num:" + gap
        options = self._opener_cache.get(key)
        if options is None:
            options = [i for i, t in enumerate(self.texts)
                       if gap and t in (gap, gap + "-")]
            self._opener_cache[key] = options
        if len(options) < 2:
            if lead:
                ids.extend(enc(lead))
            return self._gen_number(ids, integer, enders)
        if head:
            ids.extend(enc(head))
        tok = self._pick(ids, options)
        ids.append(tok)
        return self._gen_number(ids, integer, enders,
                                self.texts[tok][len(gap):])

    def _gen_value(self, ids: list[int], info: TypeInfo, lead: str = "",
                   enders: list[int] | None = None) -> Any:
        enc = self.tokenizer.encode
        kind = info.type
        if kind == "string":
            return self._gen_quoted(ids, lead)
        if kind in ("number", "integer"):
            value = self._gen_signed(ids, lead, kind == "integer", enders)
            return int(value) if kind == "integer" else float(value)
        if kind == "boolean":
            if lead:
                ids.extend(enc(lead))
            val = self._choose(ids, ["true", "false"], self.bool_ids,
                               allow_quote_end=False)
            return val == "true"
        if kind == "array":
            return self._gen_array(ids, info, lead)
        if kind == "object":
            return self._gen_object(ids, info, lead)
        raise DecodingError(f"unsupported type: {kind}")

    def _gen_array(self, ids: list[int], info: TypeInfo,
                   lead: str) -> list[Any]:
        if info.items is None:
            raise DecodingError("array type without 'items'")
        enc = self.tokenizer.encode
        ids.extend(enc(lead + "["))
        starts = self._start_ids(info.items)
        items: list[Any] = []
        while len(items) < MAX_ITEMS:
            if items:
                allowed = [self.comma_id, self.close_id]
            else:
                allowed = starts + [self.close_id]
            if self._pick(ids, allowed) == self.close_id:
                break
            items.append(self._gen_value(
                ids, info.items, ", " if items else "",
                self.array_end_ids))
        ids.extend(enc("]"))
        return items

    def _gen_object(self, ids: list[int], info: TypeInfo,
                    lead: str) -> dict[str, Any]:
        if not info.properties:
            raise DecodingError("object type without 'properties'")
        ids.extend(self.tokenizer.encode(lead + "{"))
        out: dict[str, Any] = {}
        for n, (key, sub) in enumerate(info.properties.items()):
            sep = ", " if n else ""
            out[key] = self._gen_value(ids, sub, f'{sep}"{key}": ')
        ids.extend(self.tokenizer.encode("}"))
        return out

    def generate_with_recovery(self, prompt_formatted: str,
                               original_prompt: str) -> FunctionCallResult:
        best: FunctionCallResult | None = None
        reason = "unknown"
        for attempt, cfg in enumerate(ATTEMPTS, 1):
            self.config = cfg
            try:
                result = self.generate(prompt_formatted, original_prompt)
            except DecodingError as exc:
                reason = str(exc)
            else:
                problem = self.validate_result(result)
                if problem is None:
                    return result
                best, reason = result, problem
            print(f"  attempt {attempt} failed: {reason}", file=sys.stderr)
        self.config = DecodingConfig()
        if best is not None:
            return best
        raise DecodingError(f"all attempts failed: {reason}")
