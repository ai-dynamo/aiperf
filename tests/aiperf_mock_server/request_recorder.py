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
with `--no-tokenizer`).
"""

import logging
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

import orjson

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
                }
            )
        )
        self._file.write(b"\n")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
        summary = _build_summary(
            total=self._total,
            isls=self._isls,
            osls=self._osls,
            min_tokens=self._min_tokens,
            streamed=self._streamed,
            ignore_eos=self._ignore_eos,
            reasoning_efforts=self._reasoning_efforts,
        )
        Path(self.path + ".summary.json").write_bytes(
            orjson.dumps(summary, option=orjson.OPT_INDENT_2)
        )
        _print_summary(summary)


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
        top_tokens.append({"id": int(token_id), "text": text, "count": int(count)})

    top_10_count = sum(count for _, count in sorted_items)
    top_10_concentration_pct = round(top_10_count / total * 100, 4)

    entropy_bits = 0.0
    for count in counts.values():
        p = count / total
        entropy_bits -= p * math.log2(p)
    max_entropy_bits = math.log2(vocab_size) if vocab_size > 1 else 0.0

    return {
        "vocab_size": int(vocab_size),
        "vocab_size_source": source,
        "unique_ids": len(counts),
        "coverage_pct": round(len(counts) / vocab_size * 100, 4) if vocab_size else 0.0,
        "total_tokens": int(total),
        "top_10_concentration_pct": top_10_concentration_pct,
        "entropy_bits": round(entropy_bits, 4),
        "max_entropy_bits": max_entropy_bits,
        "top_tokens": top_tokens,
        "shape_80": _compute_shape_80(counts, vocab_size),
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
) -> dict[str, Any]:
    per_endpoint: dict[str, Any] = {}
    for ep in sorted(isls.keys()):
        isl_vals = isls[ep]
        osl_vals = osls.get(ep, [])
        per_endpoint[ep] = {
            "count": len(isl_vals),
            "streamed_count": streamed.get(ep, 0),
            "ignore_eos_count": ignore_eos.get(ep, 0),
            "reasoning_effort_counts": dict(reasoning_efforts.get(ep, Counter()))
            or None,
            "isl": _stat_block(isl_vals),
            "requested_osl": _stat_block(osl_vals),
            "min_tokens": _quantiles(min_tokens.get(ep, [])),
        }
    return {"total_requests": total, "per_endpoint": per_endpoint}


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"\nRequest distribution ({summary['total_requests']} requests)")
    print("─" * 46)
    for ep, stats in summary["per_endpoint"].items():
        print(f"  {ep}  n={stats['count']}")
        for label, s in (("ISL", stats["isl"]), ("OSL", stats["requested_osl"])):
            if s is None:
                print(f"    {label}    n/a")
            else:
                print(
                    f"    {label}    mean {s['mean']:7.1f}"
                    f"   p50 {s['p50']:5.0f}   p99 {s['p99']:5.0f}"
                )
        for label, s in (("ISL", stats["isl"]), ("OSL", stats["requested_osl"])):
            if s is None or s.get("histogram") is None:
                continue
            hist = s["histogram"]
            n = sum(hist["counts"])
            for line in _render_histogram(label, hist, n, s["unique_values"]):
                print(line)
        mn = stats["min_tokens"]
        if mn is not None:
            print(f"    min_tokens  mean {mn['mean']:7.1f}   p50 {mn['p50']:5.0f}")
        if stats["ignore_eos_count"]:
            print(f"    ignore_eos=true: {stats['ignore_eos_count']}")
        if stats["reasoning_effort_counts"]:
            print(f"    reasoning_effort: {stats['reasoning_effort_counts']}")


_BLOCK_CHARS = "▁▂▃▄▅▆▇█"


def _format_top_tokens_line(top_tokens: list[dict[str, Any]]) -> str:
    """Format the `top:` line of the vocab stdout block (first 5 entries)."""
    pieces: list[str] = []
    for entry in top_tokens[:5]:
        text = entry["text"]
        count = entry["count"]
        if isinstance(text, str) and text.startswith("<id=") and text.endswith(">"):
            pieces.append(f"{text} {count}")
        else:
            pieces.append(f'"{text}" {count}')
    return "      top: " + ", ".join(pieces)


def _format_tick(value: int) -> str:
    """Right-side axis tick formatting: '0' / '38K' / '152K' (rounded, no decimals)."""
    if value < 1000:
        return str(value)
    return f"{round(value / 1000)}K"


def _render_vocab_lines(vd: dict[str, Any]) -> list[str]:
    """Return the 6-line stdout block for one endpoint's vocab_distribution.

    Layout (4-space indent on top-level rows, 6-space indent on `top:`):
        ``    Vocab  used N/V (P%)  top-10 cover X%  entropy E/M bits``
        ``      top: "tok1" c1, "tok2" c2, ...``
        ``    ``
        ``    vocab shape  (80 buckets over id 0..V-1, log-y)``
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
