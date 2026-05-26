# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-process per-request ISL / requested-OSL recorder.

Used to validate that an aiperf run actually generates the requested ISL / OSL
distribution on the wire. Enabled by `--record-requests PATH`; tokenizes each
incoming request inline with the configured tokenizer, appends one JSONL line
per request, and writes a per-endpoint distribution summary on shutdown.

In addition to the resolved `requested_osl` (= max_completion_tokens or
max_tokens), each record also captures the raw OSL-shaping fields that came
in on the request — max_tokens, max_completion_tokens, min_tokens, ignore_eos,
reasoning_effort — so the JSONL is a complete fingerprint of what the client
asked the server to do.

The recorder reuses the tokenizer name configured for corpus loading, which is
why `--record-requests` requires that a tokenizer is loaded (i.e. it conflicts
with `--no-tokenizer`). Chat request tokenization follows trtllm-serve's
precedence: supplied `prompt_token_ids` win, otherwise a tokenizer chat
template is applied when one exists, with a role-preserving fallback for
tokenizers that do not implement chat templates.
"""

import logging
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

import orjson
from aiperf_mock_server.models import (
    ChatCompletionRequest,
    CohereRerankRequest,
    CompletionRequest,
    EmbeddingRequest,
    HFTEIRerankRequest,
    ImageGenerationRequest,
    RankingRequest,
    RequestT,
    SolidoRAGRequest,
    TGIGenerateRequest,
)

logger = logging.getLogger(__name__)

# Histogram bucketing rule: at least _HISTOGRAM_MIN_BINS bins, and bin width
# never exceeds _HISTOGRAM_MAX_BIN_WIDTH. Floor keeps narrow ranges informative;
# cap keeps wide ranges from collapsing 10 bins onto a 1500-token spread.
_HISTOGRAM_MIN_BINS = 10
_HISTOGRAM_MAX_BIN_WIDTH = 100.0


class RequestRecorder:
    """Tokenizes each request and writes one JSONL record per call.

    The configured tokenizer is loaded once in `open()`; subsequent `record()`
    calls run on the FastAPI event loop. With `--workers=1` (enforced when
    recording is enabled) there is exactly one producer, so no locking is
    required around the file handle or the stats dicts.
    """

    def __init__(
        self,
        path: str,
        tokenizer_name: str,
        tokenizer_revision: str,
        trust_remote_code: bool,
    ) -> None:
        self.path = path
        self.tokenizer_name = tokenizer_name
        self.tokenizer_revision = tokenizer_revision
        self.trust_remote_code = trust_remote_code
        self._tokenizer: Any = None
        self._file: IO[bytes] | None = None
        self._isls: dict[str, list[int]] = defaultdict(list)
        self._vocab_counts: dict[str, Counter[int]] = defaultdict(Counter)
        self._vocab_size: int | None = None
        self._vocab_size_source: str = "tokenizer"
        self._osls: dict[str, list[int]] = defaultdict(list)
        self._min_tokens: dict[str, list[int]] = defaultdict(list)
        self._streamed: dict[str, int] = defaultdict(int)
        self._ignore_eos: dict[str, int] = defaultdict(int)
        self._reasoning_efforts: dict[str, Counter[str]] = defaultdict(Counter)
        self._total: int = 0

    def open(self) -> None:
        from aiperf.common.tokenizer import Tokenizer

        self._tokenizer = Tokenizer.from_pretrained(
            self.tokenizer_name,
            revision=self.tokenizer_revision,
            trust_remote_code=self.trust_remote_code,
        )
        try:
            # aiperf.common.tokenizer.Tokenizer wraps the underlying tokenizer
            # in a _tokenizer attribute; tiktoken exposes n_vocab, HF exposes
            # vocab_size. Fall through to observed derivation at summary time
            # if neither is available.
            inner = getattr(self._tokenizer, "_tokenizer", self._tokenizer)
            vocab_size = getattr(inner, "vocab_size", None)
            if vocab_size is None:
                enc = getattr(inner, "_encoding", None)
                vocab_size = getattr(enc, "n_vocab", None)
            if vocab_size is None:
                vocab_size = len(self._tokenizer)
            self._vocab_size = int(vocab_size)
            self._vocab_size_source = "tokenizer"
        except (TypeError, AttributeError):
            # Tokenizer doesn't expose vocab size; we'll derive from observed ids
            # at summary time.
            self._vocab_size = None
            self._vocab_size_source = "observed"
        # Write-binary (truncate on open) so each run's JSONL stays consistent
        # with its `.summary.json` — the per-process stats accumulators start
        # empty, so an `ab`-mode file would mix records from prior runs that
        # the summary never sees. The per-record flush below still keeps the
        # on-disk file in sync with in-process state, so SIGKILL / OOM only
        # loses the in-flight record.
        self._file = open(self.path, "wb")  # noqa: SIM115 — lifetime is the recorder's open/close pair
        logger.info(
            "Request recorder writing to %s (tokenizer=%s)",
            self.path,
            self.tokenizer_name,
        )

    def record(
        self,
        ts: float,
        endpoint: str,
        request_id: str,
        model: str,
        text: str,
        stream: bool | None,
        osl_fingerprint: dict[str, Any],
    ) -> None:
        if self._tokenizer is None or self._file is None:
            return
        try:
            ids = self._tokenizer.encode(text)
        except Exception:
            logger.exception(
                "recorder: tokenization failed for %s %s", endpoint, request_id
            )
            return
        self._record_ids(
            ts=ts,
            endpoint=endpoint,
            request_id=request_id,
            model=model,
            ids=ids,
            stream=stream,
            osl_fingerprint=osl_fingerprint,
            tokenization_mode="plain_text_encode",
        )

    def record_request(
        self,
        ts: float,
        endpoint: str,
        request_id: str,
        model: str,
        request: RequestT,
        stream: bool | None,
        osl_fingerprint: dict[str, Any],
    ) -> None:
        """Tokenize and record a parsed request using framework-server semantics."""
        if self._tokenizer is None or self._file is None:
            return
        try:
            ids, tokenization_mode = _encode_request_prompt_ids(
                self._tokenizer, request
            )
        except Exception:
            logger.exception(
                "recorder: tokenization failed for %s %s", endpoint, request_id
            )
            return
        self._record_ids(
            ts=ts,
            endpoint=endpoint,
            request_id=request_id,
            model=model,
            ids=ids,
            stream=stream,
            osl_fingerprint=osl_fingerprint,
            tokenization_mode=tokenization_mode,
        )

    def _record_ids(
        self,
        *,
        ts: float,
        endpoint: str,
        request_id: str,
        model: str,
        ids: list[int],
        stream: bool | None,
        osl_fingerprint: dict[str, Any],
        tokenization_mode: str,
    ) -> None:
        isl = len(ids)
        self._vocab_counts[endpoint].update(ids)
        max_tokens = osl_fingerprint.get("max_tokens")
        max_completion_tokens = osl_fingerprint.get("max_completion_tokens")
        min_tokens = osl_fingerprint.get("min_tokens")
        ignore_eos = osl_fingerprint.get("ignore_eos")
        reasoning_effort = osl_fingerprint.get("reasoning_effort")
        # Resolved cap: matches `request.max_output_tokens` for chat and
        # `request.max_tokens` everywhere else, but derived here from the raw
        # fields so the recorder doesn't depend on extra request properties.
        requested_osl = (
            max_completion_tokens if max_completion_tokens is not None else max_tokens
        )

        self._isls[endpoint].append(isl)
        if requested_osl is not None:
            self._osls[endpoint].append(int(requested_osl))
        if min_tokens is not None:
            self._min_tokens[endpoint].append(int(min_tokens))
        if stream:
            self._streamed[endpoint] += 1
        if ignore_eos:
            self._ignore_eos[endpoint] += 1
        if reasoning_effort is not None:
            self._reasoning_efforts[endpoint][str(reasoning_effort)] += 1
        self._total += 1

        self._file.write(
            orjson.dumps(
                {
                    "ts": ts,
                    "request_id": request_id,
                    "endpoint": endpoint,
                    "model": model,
                    "isl": isl,
                    "requested_osl": requested_osl,
                    "max_tokens": max_tokens,
                    "max_completion_tokens": max_completion_tokens,
                    "min_tokens": min_tokens,
                    "ignore_eos": ignore_eos,
                    "reasoning_effort": reasoning_effort,
                    "stream": stream,
                    "tokenization_mode": tokenization_mode,
                }
            )
        )
        self._file.write(b"\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
        decode_fn: Callable[[int], str] | None
        if self._tokenizer is not None:

            def decode_fn(token_id: int) -> str:
                return self._tokenizer.decode([token_id])
        else:
            decode_fn = None
        summary = _build_summary(
            total=self._total,
            isls=self._isls,
            osls=self._osls,
            min_tokens=self._min_tokens,
            streamed=self._streamed,
            ignore_eos=self._ignore_eos,
            reasoning_efforts=self._reasoning_efforts,
            vocab_counts=self._vocab_counts,
            vocab_size=self._vocab_size,
            vocab_size_source=self._vocab_size_source,
            decode_fn=decode_fn,
        )
        Path(self.path + ".summary.json").write_bytes(
            orjson.dumps(summary, option=orjson.OPT_INDENT_2)
        )
        _print_summary(summary)


def _encode_request_prompt_ids(
    tokenizer: Any, request: RequestT
) -> tuple[list[int], str]:
    """Return prompt token IDs using the same request shape a framework sees."""
    prompt_token_ids = getattr(request, "prompt_token_ids", None)
    if prompt_token_ids is not None:
        return _flatten_token_ids(prompt_token_ids), "prompt_token_ids"

    if isinstance(request, ChatCompletionRequest):
        return _encode_chat_prompt_ids(tokenizer, request)
    if isinstance(request, CompletionRequest):
        return _encode_completion_prompt_ids(tokenizer, request.prompt)
    if isinstance(request, TGIGenerateRequest):
        return _encode_texts_with_tokenizer_call(tokenizer, [request.prompt_text])
    if isinstance(request, EmbeddingRequest):
        return _encode_texts_with_tokenizer_call(tokenizer, request.inputs)
    if isinstance(request, (RankingRequest, HFTEIRerankRequest, CohereRerankRequest)):
        return _encode_texts_without_special_tokens(
            tokenizer, [request.query_text, *request.passage_texts]
        )
    if isinstance(request, ImageGenerationRequest):
        return _encode_texts_without_special_tokens(tokenizer, [request.prompt])
    if isinstance(request, SolidoRAGRequest):
        return _encode_texts_without_special_tokens(tokenizer, request.query)
    return [], "unsupported_request"


def _encode_chat_prompt_ids(
    tokenizer: Any, request: ChatCompletionRequest
) -> tuple[list[int], str]:
    """Mirror trtllm-serve chat prompt tokenization when a chat template exists."""
    messages = [_message_to_dict(msg) for msg in request.messages]
    add_generation_prompt = bool(getattr(request, "add_generation_prompt", True))
    inner = _unwrap_tokenizer(tokenizer)
    apply_chat_template = getattr(inner, "apply_chat_template", None)
    if callable(apply_chat_template):
        result = _call_chat_template(
            apply_chat_template, messages, add_generation_prompt
        )
        if result is not None:
            if isinstance(result, str):
                return (
                    _encode_without_special_tokens(tokenizer, result),
                    "chat_template_string",
                )
            return _flatten_token_ids(result), "chat_template"

    rendered = _render_chat_template_fallback(messages, add_generation_prompt)
    return (
        _encode_without_special_tokens(tokenizer, rendered),
        "chat_template_fallback",
    )


def _call_chat_template(
    apply_chat_template: Callable[..., Any],
    messages: list[dict[str, Any]],
    add_generation_prompt: bool,
) -> Any | None:
    """Invoke a tokenizer's ``apply_chat_template``, retrying with the
    ``conversation=`` kwarg on TypeError (older HF where the parameter was
    keyword-only). Returns ``None`` when the tokenizer reports no chat
    template defined; other ``ValueError`` types propagate so callers see
    real failures.

    Pulled out of `_encode_chat_prompt_ids` so the result is handled outside
    the try/except: with ``else:``, an exception handler that rebinds
    ``result`` would not trigger ``else``, and the rebound value would be
    silently dropped.
    """
    kwargs: dict[str, Any] = {
        "add_generation_prompt": add_generation_prompt,
        "tokenize": True,
        "return_dict": False,
    }
    try:
        return apply_chat_template(messages, **kwargs)
    except TypeError:
        pass
    except ValueError as exc:
        if "chat template" not in str(exc).lower():
            raise
        return None
    try:
        return apply_chat_template(conversation=messages, **kwargs)
    except ValueError as exc:
        if "chat template" not in str(exc).lower():
            raise
        return None


def _encode_completion_prompt_ids(tokenizer: Any, prompt: Any) -> tuple[list[int], str]:
    if isinstance(prompt, str):
        return _encode_texts_with_tokenizer_call(tokenizer, [prompt])
    if _is_token_id_sequence(prompt):
        return _flatten_token_ids(prompt), "prompt_token_ids"
    if (
        isinstance(prompt, list)
        and prompt
        and all(_is_token_id_sequence(p) for p in prompt)
    ):
        return _flatten_token_ids(prompt), "prompt_token_ids"
    if isinstance(prompt, list):
        return _encode_texts_with_tokenizer_call(tokenizer, [str(p) for p in prompt])
    return _encode_texts_with_tokenizer_call(tokenizer, [str(prompt)])


def _encode_texts_with_tokenizer_call(
    tokenizer: Any, texts: list[str]
) -> tuple[list[int], str]:
    ids: list[int] = []
    for text in texts:
        ids.extend(_tokenizer_call_ids(tokenizer, text))
    return ids, "tokenizer_call"


def _encode_texts_without_special_tokens(
    tokenizer: Any, texts: list[str]
) -> tuple[list[int], str]:
    ids: list[int] = []
    for text in texts:
        ids.extend(_encode_without_special_tokens(tokenizer, text))
    return ids, "encode_without_special_tokens"


def _tokenizer_call_ids(tokenizer: Any, text: str) -> list[int]:
    # Prefer the wrapper's `__call__` when it has one — AIPerf's `Tokenizer`
    # wrapper sets `add_special_tokens=False` there; unwrapping to the
    # backend would silently re-introduce BOS/EOS tokens.
    if callable(tokenizer):
        return _extract_input_ids(tokenizer(text))
    inner = _unwrap_tokenizer(tokenizer)
    if callable(inner):
        return _extract_input_ids(inner(text))
    return _encode_without_special_tokens(tokenizer, text)


def _encode_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    inner = _unwrap_tokenizer(tokenizer)
    encode = getattr(inner, "encode", None)
    if callable(encode):
        try:
            return _flatten_token_ids(encode(text, add_special_tokens=False))
        except TypeError:
            return _flatten_token_ids(encode(text))
    return _flatten_token_ids(tokenizer.encode(text))


def _extract_input_ids(result: Any) -> list[int]:
    if isinstance(result, dict):
        return _flatten_token_ids(result.get("input_ids", []))
    input_ids = getattr(result, "input_ids", None)
    if input_ids is not None:
        return _flatten_token_ids(input_ids)
    return _flatten_token_ids(result)


def _flatten_token_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return _flatten_token_ids(value.get("input_ids", []))
    input_ids = getattr(value, "input_ids", None)
    if input_ids is not None:
        return _flatten_token_ids(input_ids)
    if isinstance(value, str | bytes):
        raise TypeError("token id sequence must not be text")

    ids: list[int] = []
    for item in value:
        if hasattr(item, "tolist"):
            item = item.tolist()
        if isinstance(item, list | tuple):
            ids.extend(_flatten_token_ids(item))
        else:
            ids.append(int(item))
    return ids


def _is_token_id_sequence(value: Any) -> bool:
    return isinstance(value, list | tuple) and all(isinstance(v, int) for v in value)


def _message_to_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return dict(message)
    model_dump = getattr(message, "model_dump", None)
    if callable(model_dump):
        return model_dump(exclude_none=True)
    return {
        "role": getattr(message, "role", "user"),
        "content": getattr(message, "content", ""),
    }


def _render_chat_template_fallback(
    messages: list[dict[str, Any]], add_generation_prompt: bool
) -> str:
    rendered: list[str] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = _content_to_text(message.get("content", ""))
        rendered.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    if add_generation_prompt:
        rendered.append("<|im_start|>assistant\n")
    return "\n".join(rendered)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            item_type = str(item.get("type", ""))
            if item_type == "text":
                parts.append(str(item.get("text", "")))
            elif "image" in item_type:
                parts.append("<image>")
            elif "audio" in item_type:
                parts.append("<audio>")
            elif "video" in item_type:
                parts.append("<video>")
        return "".join(parts)
    return "" if content is None else str(content)


def _unwrap_tokenizer(tokenizer: Any) -> Any:
    return getattr(tokenizer, "_tokenizer", tokenizer)


def _histogram(values: list[int]) -> dict[str, list[float] | list[int]] | None:
    """Equal-width histogram with the max_bin_width / min_bins rule.

    Returns ``None`` for an empty input, ``{"bin_edges": [v, v], "counts": [n]}``
    when all values are equal, and otherwise a dict with ``len(bin_edges) ==
    len(counts) + 1``. The last bin is closed on both ends so the observed
    maximum lands in it instead of just past the last edge.
    """
    if not values:
        return None
    lo = float(min(values))
    hi = float(max(values))
    if lo == hi:
        return {"bin_edges": [lo, hi], "counts": [len(values)]}
    span = hi - lo
    num_bins = max(_HISTOGRAM_MIN_BINS, math.ceil(span / _HISTOGRAM_MAX_BIN_WIDTH))
    width = span / num_bins
    edges = [lo + i * width for i in range(num_bins + 1)]
    edges[-1] = hi  # pin last edge exactly to max to avoid float drift
    counts = [0] * num_bins
    for v in values:
        if v >= hi:
            idx = num_bins - 1
        else:
            idx = int((v - lo) / width)
            # Float-drift guard: int((v-lo)/width) can round to num_bins when v is very close to hi.
            if idx >= num_bins:
                idx = num_bins - 1
        counts[idx] += 1
    return {"bin_edges": edges, "counts": counts}


def _render_histogram(
    metric: str,
    hist: dict[str, list[float] | list[int]],
    count: int,
    unique: int,
) -> list[str]:
    """Render a histogram as 4-/6-space-indented stdout lines (header + bin rows).

    Bars are 20 chars wide, scaled so the tallest bin is full width. Bin range
    labels and the count column align within the histogram.
    """
    edges = hist["bin_edges"]
    counts = hist["counts"]
    num_bins = len(counts)
    header = f"    {metric} histogram ({num_bins} bins, n={count}, {unique} unique)"
    if not counts:
        return [header]
    max_count = max(counts) or 1
    bar_width = 20
    label_width = max(len(str(round(e))) for e in edges)
    count_width = max(3, len(str(max_count)))
    lines = [header]
    for i, c in enumerate(counts):
        filled = round(bar_width * c / max_count)
        bar = "█" * filled + "░" * (bar_width - filled)
        lo = round(edges[i])
        hi = round(edges[i + 1])
        lines.append(
            f"      {lo:>{label_width}d}- {hi:>{label_width}d}"
            f"  {c:>{count_width}d} {bar}"
        )
    return lines


def _compute_shape_80(counts: Counter[int], vocab_size: int) -> list[int]:
    """Sum counts into 80 equal-width buckets over [0, vocab_size).

    Each bucket spans `vocab_size / 80` token ids. The last bucket is closed
    on its upper end so `vocab_size - 1` lands in bucket 79 (instead of just
    past it). Ids >= `vocab_size` are dropped — defensive only; should not
    occur with a well-behaved tokenizer.
    """
    shape = [0] * 80
    if vocab_size <= 0:
        return shape
    width = vocab_size / 80
    for token_id, count in counts.items():
        if token_id < 0 or token_id >= vocab_size:
            continue
        idx = int(token_id / width)
        if idx >= 80:
            idx = 79  # float-drift guard, mirrors `_histogram`
        shape[idx] += count
    return shape


def _vocab_distribution(
    counts: Counter[int],
    vocab_size: int,
    source: str,
    decode_fn: Callable[[int], str],
) -> dict[str, Any] | None:
    """Build the vocab_distribution JSON block, or None if there are no observations.

    `decode_fn` maps a token id to its text representation. If `decode_fn`
    raises for a given id, that entry in `top_tokens` falls back to
    ``"<id=N>"``.
    """
    total = sum(counts.values())
    if total == 0:
        return None

    sorted_items = counts.most_common(10)
    top_tokens: list[dict[str, Any]] = []
    for token_id, count in sorted_items:
        try:
            text = decode_fn(token_id)
        except Exception:
            text = f"<id={token_id}>"
        else:
            if not text.isprintable():
                text = f"<id={token_id}>"
        top_tokens.append({"id": int(token_id), "text": text, "count": int(count)})

    top_10_count = sum(count for _, count in sorted_items)
    top_10_concentration_pct = round(top_10_count / total * 100, 4)

    entropy_bits = 0.0
    for count in counts.values():
        p = count / total
        entropy_bits -= p * math.log2(p)
    max_entropy_bits = math.log2(vocab_size) if vocab_size > 1 else 0.0

    shape_80 = _compute_shape_80(counts, vocab_size)
    shape_stats = _quantiles(shape_80)

    return {
        "vocab_size": int(vocab_size),
        "vocab_size_source": source,
        "unique_ids": len(counts),
        "coverage_pct": round(len(counts) / vocab_size * 100, 4) if vocab_size else 0.0,
        "total_tokens": int(total),
        "top_10_concentration_pct": top_10_concentration_pct,
        "entropy_bits": round(entropy_bits, 4),
        "max_entropy_bits": round(max_entropy_bits, 4),
        "top_tokens": top_tokens,
        "shape_80": shape_80,
        "shape_80_stats": shape_stats,
        "frequencies": {str(tid): int(c) for tid, c in counts.items()},
    }


def _quantiles(values: list[int]) -> dict[str, float] | None:
    if not values:
        return None
    if len(values) == 1:
        only = float(values[0])
        return {
            "min": only,
            "max": only,
            "mean": only,
            "stdev": 0.0,
            "p50": only,
            "p90": only,
            "p95": only,
            "p99": only,
        }
    qs = statistics.quantiles(values, n=100, method="inclusive")
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values),
        "p50": qs[49],
        "p90": qs[89],
        "p95": qs[94],
        "p99": qs[98],
    }


def _stat_block(values: list[int]) -> dict[str, Any] | None:
    """Build the percentiles + histogram + unique_values block, or None when empty."""
    if not values:
        return None
    block = _quantiles(values)
    assert block is not None  # `_quantiles` only returns None for empty input
    block["unique_values"] = len(set(values))
    block["histogram"] = _histogram(values)
    return block


def _build_summary(
    total: int,
    isls: dict[str, list[int]],
    osls: dict[str, list[int]],
    min_tokens: dict[str, list[int]],
    streamed: dict[str, int],
    ignore_eos: dict[str, int],
    reasoning_efforts: dict[str, Counter[str]],
    vocab_counts: dict[str, Counter[int]] | None = None,
    vocab_size: int | None = None,
    vocab_size_source: str = "tokenizer",
    decode_fn: Callable[[int], str] | None = None,
) -> dict[str, Any]:
    per_endpoint: dict[str, Any] = {}
    vocab_counts = vocab_counts or {}
    for ep in sorted(isls.keys()):
        isl_vals = isls[ep]
        osl_vals = osls.get(ep, [])
        ep_vocab_counter = vocab_counts.get(ep, Counter())
        if decode_fn is not None and (
            vocab_size is not None or vocab_size_source == "observed"
        ):
            resolved_size = _resolve_vocab_size(
                vocab_size, vocab_size_source, ep_vocab_counter
            )
            vd = _vocab_distribution(
                ep_vocab_counter,
                resolved_size,
                vocab_size_source,
                decode_fn,
            )
        else:
            vd = None
        per_endpoint[ep] = {
            "count": len(isl_vals),
            "streamed_count": streamed.get(ep, 0),
            "ignore_eos_count": ignore_eos.get(ep, 0),
            "reasoning_effort_counts": dict(reasoning_efforts.get(ep, Counter()))
            or None,
            "isl": _stat_block(isl_vals),
            "requested_osl": _stat_block(osl_vals),
            "min_tokens": _quantiles(min_tokens.get(ep, [])),
            "vocab_distribution": vd,
        }
    return {"total_requests": total, "per_endpoint": per_endpoint}


def _resolve_vocab_size(declared: int | None, source: str, counts: Counter[int]) -> int:
    """Return vocab size for the per-endpoint distribution.

    For the `"tokenizer"` source we trust the declared value. For the
    `"observed"` source we use `max_observed_id + 1` (or the declared value,
    whichever is greater) so coverage_pct stays sane when the tokenizer
    doesn't expose len().
    """
    if not counts:
        return declared or 0
    observed_max = max(counts.keys())
    if source == "observed":
        return max(declared or 0, observed_max + 1)
    return declared or (observed_max + 1)


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"\nRequest distribution ({summary['total_requests']} requests)")
    print("─" * 46)
    if _summary_has_vocab_distribution(summary):
        for line in _render_description_box():
            print(line)
        print("")
    for ep, stats in summary["per_endpoint"].items():
        print(f"  {ep}  n={stats['count']}")
        token_stats = (
            ("ISL", stats["isl"]),
            ("Requested OSL", stats["requested_osl"]),
        )
        label_width = max(len(label) for label, _ in token_stats)
        for label, s in token_stats:
            if s is None:
                print(f"    {label:<{label_width}}  n/a")
            else:
                print(
                    f"    {label:<{label_width}}  mean {s['mean']:7.1f}"
                    f"   min {s['min']:5.0f}   max {s['max']:5.0f}"
                    f"   p50 {s['p50']:5.0f}   p99 {s['p99']:5.0f}"
                )
        rendered_histogram = False
        for label, s in token_stats:
            if s is None or s.get("histogram") is None:
                continue
            hist = s["histogram"]
            n = sum(hist["counts"])
            print("")  # blank line before each histogram block
            for line in _render_histogram(label, hist, n, s["unique_values"]):
                print(line)
            rendered_histogram = True
        vd = stats.get("vocab_distribution")
        if vd is not None:
            print("")  # blank line before vocab block
            if rendered_histogram:
                print("")  # extra visual gap after histogram blocks
            for line in _render_vocab_lines(vd):
                print(line)
        mn = stats["min_tokens"]
        if mn is not None:
            print("")  # blank line before misc lines
            print(f"    min_tokens  mean {mn['mean']:7.1f}   p50 {mn['p50']:5.0f}")
        if stats["ignore_eos_count"]:
            print(f"    ignore_eos=true: {stats['ignore_eos_count']}")
        if stats["reasoning_effort_counts"]:
            print(f"    reasoning_effort: {stats['reasoning_effort_counts']}")


def _summary_has_vocab_distribution(summary: dict[str, Any]) -> bool:
    """Return true when any endpoint has vocab-distribution details."""
    return any(
        stats.get("vocab_distribution") is not None
        for stats in summary["per_endpoint"].values()
    )


def _render_description_box() -> list[str]:
    """Small stdout glossary for less obvious request-recorder metrics."""
    return [
        "  Definitions",
        "    ISL/OSL: input/requested output sequence length in tokens; OSL is the request cap, not generated output.",
        "    Vocab used: unique token IDs observed / tokenizer vocab size.",
        "    top-10 cover: share of prompt tokens from the 10 most common token IDs.",
        "    entropy: token-id diversity; higher means broader prompt vocabulary use.",
        "    top decoded tokens: most frequent token IDs decoded for sanity checks; tokens are not words.",
        "    vocab shape: log-scaled 80-bucket view across token-id space.",
        "    vocab shape stats: mean/percentiles of prompt-token counts per bucket, including empty buckets.",
    ]


_BLOCK_CHARS = "▁▂▃▄▅▆▇█"


def _format_top_tokens_line(top_tokens: list[dict[str, Any]]) -> str:
    """Format the decoded-token line of the vocab stdout block."""
    pieces: list[str] = []
    for entry in top_tokens[:5]:
        text = entry["text"]
        count = entry["count"]
        if isinstance(text, str) and text.startswith("<id=") and text.endswith(">"):
            pieces.append(f"{text} {count}")
        else:
            pieces.append(f'"{text}" {count}')
    return "      top decoded tokens: " + ", ".join(pieces)


def _format_tick(value: int) -> str:
    """Right-side axis tick formatting: '0' / '38K' / '152K' (rounded, no decimals)."""
    if value < 1000:
        return str(value)
    return f"{round(value / 1000)}K"


def _format_shape_stats_line(stats: dict[str, Any]) -> str:
    """Format the per-bucket vocab-shape stats line."""
    return (
        f"      bucket tokens mean {stats['mean']:7.1f}"
        f"   p50 {stats['p50']:5.0f}"
        f"   p90 {stats['p90']:5.0f}"
        f"   p95 {stats['p95']:5.0f}"
        f"   p99 {stats['p99']:5.0f}"
    )


def _render_vocab_lines(vd: dict[str, Any]) -> list[str]:
    """Return the stdout block for one endpoint's vocab_distribution.

    Layout (4-space indent on top-level rows, 6-space indent on token details):
        ``    Vocab  used N/V (P%)  top-10 cover X%  entropy E/M bits``
        ``      top decoded tokens: "tok1" c1, "tok2" c2, ...``
        ``    ``
        ``    vocab shape  (80 buckets over id 0..V-1, log-y)``
        ``    ``
        ``      bucket tokens mean M  p50 P50  p90 P90  p95 P95  p99 P99``
        ``    ``
        ``    [80-char sparkline]``
        ``    0 ... K_q1 ... K_q2 ... K_q3 ... K_max``
    """
    headline = (
        f"    Vocab  used {vd['unique_ids']}/{vd['vocab_size']}"
        f" ({vd['coverage_pct']:.1f}%)"
        f"  top-10 cover {vd['top_10_concentration_pct']:.0f}%"
        f"  entropy {vd['entropy_bits']:.1f}/{vd['max_entropy_bits']:.1f} bits"
    )
    top_line = _format_top_tokens_line(vd["top_tokens"])
    shape_header = (
        f"    vocab shape  (80 buckets over id 0..{vd['vocab_size'] - 1}, log-y)"
    )

    shape = vd["shape_80"]
    shape_stats = vd.get("shape_80_stats") or _quantiles(shape) or _quantiles([0])
    max_count = max(shape) if shape else 0
    if max_count <= 0:
        sparkline = " " * 80
    else:
        log_max = math.log1p(max_count)
        sparkline_chars: list[str] = []
        for count in shape:
            if count <= 0:
                sparkline_chars.append(" ")
                continue
            ratio = math.log1p(count) / log_max
            # Map (0, 1] -> index [0, 7]; ratio==1.0 must give index 7 (full block).
            idx = min(7, max(0, math.ceil(ratio * 8) - 1))
            sparkline_chars.append(_BLOCK_CHARS[idx])
        sparkline = "".join(sparkline_chars)

    vocab_size = vd["vocab_size"]
    tick_positions = (
        0,
        vocab_size // 4,
        vocab_size // 2,
        (3 * vocab_size) // 4,
        vocab_size,
    )
    tick_labels = [_format_tick(p) for p in tick_positions]
    # Each tick sits at the column index where its bucket starts (80-char line).
    columns = (0, 20, 40, 60, 79)
    tick_line = list(" " * 80)
    for col, label in zip(columns, tick_labels, strict=True):
        start = min(col, 80 - len(label))
        for i, ch in enumerate(label):
            tick_line[start + i] = ch

    return [
        headline,
        top_line,
        "",
        shape_header,
        "",
        _format_shape_stats_line(shape_stats),
        "",
        "    " + sparkline,
        "    " + "".join(tick_line).rstrip(),
    ]


_GLOBAL_RECORDER: RequestRecorder | None = None


def set_global_recorder(rec: RequestRecorder | None) -> None:
    """Install (or clear) the per-process recorder that `make_ctx` reads."""
    global _GLOBAL_RECORDER
    _GLOBAL_RECORDER = rec


def get_global_recorder() -> RequestRecorder | None:
    return _GLOBAL_RECORDER
