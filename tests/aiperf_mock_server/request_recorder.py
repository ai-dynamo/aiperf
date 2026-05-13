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
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import IO, Any

import orjson

logger = logging.getLogger(__name__)


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
            isl = len(self._tokenizer.encode(text))
        except Exception:
            logger.exception(
                "recorder: tokenization failed for %s %s", endpoint, request_id
            )
            return
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
        per_endpoint[ep] = {
            "count": len(isls[ep]),
            "streamed_count": streamed.get(ep, 0),
            "ignore_eos_count": ignore_eos.get(ep, 0),
            "reasoning_effort_counts": dict(reasoning_efforts.get(ep, Counter()))
            or None,
            "isl": _quantiles(isls[ep]),
            "requested_osl": _quantiles(osls.get(ep, [])),
            "min_tokens": _quantiles(min_tokens.get(ep, [])),
        }
    return {"total_requests": total, "per_endpoint": per_endpoint}


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"\nRequest distribution summary ({summary['total_requests']} requests):")
    print("-" * 88)
    for ep, stats in summary["per_endpoint"].items():
        isl = stats["isl"]
        osl = stats["requested_osl"]
        mn = stats["min_tokens"]
        isl_str = (
            f"ISL mean={isl['mean']:7.1f} p50={isl['p50']:6.0f} p99={isl['p99']:6.0f}"
            if isl
            else "ISL n/a                              "
        )
        osl_str = (
            f"OSL mean={osl['mean']:7.1f} p50={osl['p50']:6.0f} p99={osl['p99']:6.0f}"
            if osl
            else "OSL n/a"
        )
        print(f"  {ep:32s}  n={stats['count']:6d}  {isl_str}  {osl_str}")
        if mn is not None:
            print(f"      min_tokens mean={mn['mean']:7.1f} p50={mn['p50']:6.0f}")
        if stats["ignore_eos_count"]:
            print(f"      ignore_eos=true: {stats['ignore_eos_count']}")
        if stats["reasoning_effort_counts"]:
            print(f"      reasoning_effort: {stats['reasoning_effort_counts']}")


_GLOBAL_RECORDER: RequestRecorder | None = None


def set_global_recorder(rec: RequestRecorder | None) -> None:
    """Install (or clear) the per-process recorder that `make_ctx` reads."""
    global _GLOBAL_RECORDER
    _GLOBAL_RECORDER = rec


def get_global_recorder() -> RequestRecorder | None:
    return _GLOBAL_RECORDER
