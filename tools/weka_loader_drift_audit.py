# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-turn drift audit for the weka trace loader (sampled subset by default).

Runs ``WekaTraceLoader`` directly against a directory of recorded ``.json``
traces and reports per-turn drift between three quantities:

  * ``C`` = recorded ``in[k]`` from the trace.
  * ``A`` = ``sum(len(seg.tokens))`` — the canonical synthesized ISL the
    reconstructor produces (algorithm fidelity; post-P17 this is
    block-aligned and should equal ``C`` byte-for-byte).
  * ``B`` = ``len(tokenizer.encode(" ".join(content)))`` — what aiperf
    actually reports as ISL via the standard message-join formula.

Each turn is classified by the structural relationship between its
``hash_ids`` and the previous turn's:

  * ``init``: ``k == 0``.
  * ``A``: LCP == ``M_prev``                (append-only).
  * ``B``: LCP == ``M_prev - 1``            (trailing-block churn).
  * ``C``: LCP <  ``M_prev - 1``            (pull-back).

This is a fast pre-flight diagnostic after loader changes — distinct from
``tools/weka_byte_exact_verify.py`` which consumes a post-run
``profile_export.jsonl``. This tool exercises only the loader, so it
re-runs in seconds.

Memory model (P24): each trace is processed in a forked child process.
The parent forks once per trace, the child does the heavy
``WekaTraceLoader`` work, ships back a small list of drift rows over a
``multiprocessing.Queue``, and exits. On exit the child's entire address
space — including glibc/Rust native-allocator high-water marks that
``gc.collect`` cannot release — is reclaimed by the OS. Parent RSS
plateaus near the post-warmup baseline regardless of corpus size.
``--max-traces`` therefore defaults to 0 (disabled); the only reason to
cap is wall time. For full-corpus structural assertions use
``tests/unit/dataset/loader/test_weka_trace_byte_exact_corpus.py`` which
streams trace-by-trace at flat memory.

Usage:
  python tools/weka_loader_drift_audit.py \\
      --traces /path/to/trace/dir \\
      [--tokenizer Qwen/Qwen3-0.6B] \\
      [--top-n 25] \\
      [--max-traces 0]

Always exits 0 on a successful audit regardless of drift findings — this
is a diagnostic tool, not a contract enforcer. Exits nonzero only on
missing inputs or invalid arguments.
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import resource
import shutil
import signal
import statistics
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from transformers import AutoTokenizer

from aiperf.common.config import PrefixPromptConfig, PromptConfig
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.generator.prompt import PromptGenerator
from aiperf.dataset.loader import weka_synth_buf
from aiperf.dataset.loader.weka_trace import WekaTraceLoader

# Aggressive thresholds vs CPython default (700, 10, 10): collect gen0 ~7x
# more often and sweep older generations after far fewer gen0 cycles. Tool
# runs on memory-constrained machines under an aggressive memory protector;
# the wall-time hit is negligible vs avoiding a SIGTERM.
gc.set_threshold(100, 5, 5)
gc.enable()

DEFAULT_TOKENIZER = "Qwen/Qwen3-0.6B"
DEFAULT_BLOCK_SIZE = 64


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--traces",
        type=Path,
        required=True,
        help="Directory containing the recorded .json trace files.",
    )
    p.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        help=f"HF tokenizer name to load locally. Default: {DEFAULT_TOKENIZER}.",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="How many trickiest turns to print per ranking section. Default 25.",
    )
    p.add_argument(
        "--max-traces",
        type=int,
        default=0,
        help=(
            "Process at most this many traces (sorted alphabetically). Default 0 "
            "(disabled): fork-per-trace bounds parent RSS regardless of corpus "
            "size, so the cap exists only for wall-time control. Pass a positive "
            "value to truncate. For full-corpus structural assertions use "
            "tests/unit/dataset/loader/test_weka_trace_byte_exact_corpus.py "
            "(streaming, flat memory)."
        ),
    )
    return p.parse_args()


def _make_user_config(model_names: set[str], tokenizer_name: str) -> MagicMock:
    uc = MagicMock()
    uc.input.random_seed = 0
    uc.input.fixed_schedule_start_offset = None
    uc.input.fixed_schedule_end_offset = None
    uc.input.ignore_trace_delays = False
    uc.input.use_think_time_only = False
    uc.input.synthesis.max_isl = None
    uc.input.synthesis.max_osl = None
    uc.input.synthesis.should_synthesize.return_value = False
    uc.input.prompt.input_tokens.block_size = None
    uc.tokenizer.trust_remote_code = False
    uc.tokenizer.revision = None
    uc.tokenizer.name = tokenizer_name
    uc.endpoint.model_names = sorted(model_names)
    return uc


def _build_tokenizer(name: str) -> Tokenizer:
    auto = AutoTokenizer.from_pretrained(name, local_files_only=True)
    tk = Tokenizer()
    tk._tokenizer = auto
    tk._resolved_name = name
    tk._apply_kwarg_overrides()
    return tk


def _classify_turns(
    traces_dir: Path,
) -> tuple[dict[tuple[str, int], str], dict[str, list[int]], set[str]]:
    """Build per-(trace_id, k) structural pattern map and recorded ISL list."""
    trace_patterns: dict[tuple[str, int], str] = {}
    recorded: dict[str, list[int]] = {}
    models: set[str] = set()
    for path in sorted(traces_dir.glob("*.json")):
        blob = json.loads(path.read_text())
        tid = blob["id"]
        ns_reqs = [r for r in blob["requests"] if r.get("type") in ("n", "s")]
        recorded[tid] = [r["in"] for r in ns_reqs]
        for r in blob["requests"]:
            if "model" in r:
                models.add(r["model"])
        prev: list[int] | None = None
        for k, r in enumerate(ns_reqs):
            curr = r["hash_ids"]
            if k == 0 or prev is None:
                trace_patterns[(tid, k)] = "init"
            else:
                lcp = 0
                while lcp < min(len(prev), len(curr)) and prev[lcp] == curr[lcp]:
                    lcp += 1
                m_prev = len(prev)
                if lcp == m_prev:
                    trace_patterns[(tid, k)] = "A"
                elif lcp == m_prev - 1:
                    trace_patterns[(tid, k)] = "B"
                else:
                    trace_patterns[(tid, k)] = "C"
            prev = curr
    return trace_patterns, recorded, models


def _stats(label: str, vals: list[int | float]) -> None:
    if not vals:
        return
    print(
        f"  {label}: median={statistics.median(vals)} "
        f"mean={statistics.mean(vals):+.2f} "
        f"max={max(vals, key=abs)} "
        f"(abs max={max(abs(v) for v in vals)})"
    )


def _build_rows_for_trace(
    convs: list[Any],
    captured: list[list[int]],
    recorded: dict[str, list[int]],
    trace_patterns: dict[tuple[str, int], str],
    tk: Tokenizer,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    capture_idx = 0
    for conv in convs:
        # Subagent conversations share the user's hash space but aren't part
        # of the per-turn ISL recorded contract; advance the snapshot index
        # to keep alignment with the loader's emit order.
        if "::sa:" in conv.session_id:
            capture_idx += len(conv.turns)
            continue
        ins = recorded.get(conv.session_id, [])
        for k, turn in enumerate(conv.turns):
            if capture_idx >= len(captured):
                break
            seg_lens = captured[capture_idx]
            capture_idx += 1
            if k >= len(ins):
                continue
            messages = turn.raw_messages or []
            n_msgs = len(messages)
            a = sum(seg_lens)
            joined = " ".join(m["content"] for m in messages)
            b = len(tk.encode(joined))
            c = ins[k]
            pat = trace_patterns.get((conv.session_id, k), "?")
            rows.append(
                {
                    "trace": conv.session_id,
                    "k": k,
                    "n_msgs": n_msgs,
                    "pattern": pat,
                    "A": a,
                    "B": b,
                    "C": c,
                    "bpe_on_join": b - a,
                    "synth_vs_rec": a - c,
                    "obs_vs_rec": b - c,
                    "abs_obs": abs(b - c),
                    "abs_synth": abs(a - c),
                    "per_msg": abs(b - c) / max(n_msgs, 1),
                }
            )
    return rows


def _print_report(rows: list[dict[str, Any]], top_n: int) -> None:
    print(f"Total turns analyzed: {len(rows)}")
    print(f"Total convs: {len({r['trace'] for r in rows})}")
    by_pat: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_pat.setdefault(r["pattern"], []).append(r)
    print("\nPer-pattern counts:")
    for p, rs in sorted(by_pat.items()):
        print(f"  {p}: {len(rs)}")

    print("\n=== Aggregate drift (all turns) ===")
    _stats("synth_vs_rec (A-C)", [r["synth_vs_rec"] for r in rows])
    _stats("bpe_on_join  (B-A)", [r["bpe_on_join"] for r in rows])
    _stats("obs_vs_rec   (B-C)", [r["obs_vs_rec"] for r in rows])
    _stats("per_msg drift (|B-C|/n)", [r["per_msg"] for r in rows])

    print("\n=== Per-pattern aggregate drift ===")
    for p, rs in sorted(by_pat.items()):
        print(f"\n  Pattern {p} ({len(rs)} turns):")
        _stats("    synth_vs_rec", [r["synth_vs_rec"] for r in rs])
        _stats("    bpe_on_join ", [r["bpe_on_join"] for r in rs])
        _stats("    obs_vs_rec  ", [r["obs_vs_rec"] for r in rs])

    print(f"\n=== Top {top_n} trickiest turns (by |B-C|, the user-observed drift) ===")
    print(
        f"{'trace':<14}{'k':>3}{'pat':>5}{'n_msgs':>8}{'C=rec':>8}{'A=synth':>10}"
        f"{'B=join':>8}{'A-C':>6}{'B-A':>6}{'B-C':>6}{'per_msg':>9}"
    )
    print("-" * 110)
    for r in sorted(rows, key=lambda x: -x["abs_obs"])[:top_n]:
        print(
            f"{r['trace']:<14}{r['k']:>3}{r['pattern']:>5}{r['n_msgs']:>8}"
            f"{r['C']:>8}{r['A']:>10}{r['B']:>8}"
            f"{r['synth_vs_rec']:>+6}{r['bpe_on_join']:>+6}{r['obs_vs_rec']:>+6}"
            f"{r['per_msg']:>9.2f}"
        )

    print(f"\n=== Top {top_n} by |A-C| (algorithm fidelity, BPE-noise-stripped) ===")
    print(
        f"{'trace':<14}{'k':>3}{'pat':>5}{'n_msgs':>8}{'C=rec':>8}{'A=synth':>10}{'A-C':>6}"
    )
    print("-" * 70)
    for r in sorted(rows, key=lambda x: -x["abs_synth"])[:top_n]:
        print(
            f"{r['trace']:<14}{r['k']:>3}{r['pattern']:>5}{r['n_msgs']:>8}"
            f"{r['C']:>8}{r['A']:>10}{r['synth_vs_rec']:>+6}"
        )

    print("\n=== Top 5 trickiest of each pattern ===")
    for p in ("init", "A", "B", "C"):
        rs = by_pat.get(p, [])
        if not rs:
            continue
        print(f"\n  Pattern {p}:")
        print(
            f"  {'trace':<14}{'k':>3}{'n_msgs':>8}{'C':>8}{'A':>8}{'B':>8}"
            f"{'A-C':>6}{'B-A':>6}{'B-C':>6}"
        )
        for r in sorted(rs, key=lambda x: -x["abs_obs"])[:5]:
            print(
                f"  {r['trace']:<14}{r['k']:>3}{r['n_msgs']:>8}"
                f"{r['C']:>8}{r['A']:>8}{r['B']:>8}"
                f"{r['synth_vs_rec']:>+6}{r['bpe_on_join']:>+6}{r['obs_vs_rec']:>+6}"
            )

    print("\n=== Deepest conversations (most turns) — top 10 ===")
    by_trace: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_trace.setdefault(r["trace"], []).append(r)
    for tid in sorted(by_trace, key=lambda t: -len(by_trace[t]))[:10]:
        rs = by_trace[tid]
        max_turn = max(rs, key=lambda x: x["abs_obs"])
        sum_drift = sum(abs(r["obs_vs_rec"]) for r in rs)
        print(
            f"  {tid}: {len(rs)} turns | sum|B-C|={sum_drift} | "
            f"worst turn k={max_turn['k']} pat={max_turn['pattern']} "
            f"drift={max_turn['obs_vs_rec']:+}"
        )

    perfect = [r for r in rows if r["synth_vs_rec"] == 0]
    pct = 100 * len(perfect) / len(rows) if rows else 0.0
    print(
        f"\n=== Algorithm-perfect turns (A == C, exact byte match): "
        f"{len(perfect)} / {len(rows)} = {pct:.1f}% ==="
    )


# Module-level globals are populated in the parent before any forking
# happens. Children inherit them via fork's copy-on-write — they are NOT
# passed as Process args (which would force pickling and defeat the COW
# share of the heavy tokenized-corpus state inside _PARENT_PG).
_PARENT_TK: Tokenizer | None = None
_PARENT_PG: PromptGenerator | None = None
_CAPTURED: list[list[int]] = []


def _seq_decode(seqs, name, **_):  # type: ignore[no-untyped-def]
    # Sequential decode keeps wall time bounded on small corpora and avoids
    # forking a worker pool from inside an already-forked child.
    assert _PARENT_TK is not None
    return [_PARENT_TK.decode(s) for s in seqs]


def _process_one_trace_in_child(
    trace_path: Path,
    models: set[str],
    tokenizer_name: str,
    recorded: dict[str, list[int]],
    trace_patterns: dict[tuple[str, int], str],
    out_queue: mp.Queue[Any],
) -> None:
    try:
        assert _PARENT_PG is not None and _PARENT_TK is not None
        with tempfile.TemporaryDirectory() as td:
            single_dir = Path(td)
            shutil.copy(trace_path, single_dir / trace_path.name)
            uc = _make_user_config(models, tokenizer_name)
            loader = WekaTraceLoader(
                filename=str(single_dir),
                user_config=uc,
                prompt_generator=_PARENT_PG,
            )
            loader._block_size = DEFAULT_BLOCK_SIZE
            with patch(
                "aiperf.dataset.loader.hash_ids_synthesis.parallel_decode",
                _seq_decode,
            ):
                convs = loader.convert_to_conversations(loader.load_dataset())
            new_rows = _build_rows_for_trace(
                convs, list(_CAPTURED), recorded, trace_patterns, _PARENT_TK
            )
        out_queue.put(("ok", new_rows))
    except BaseException as e:  # noqa: BLE001
        # Catch SystemExit / KeyboardInterrupt too: if we don't always send,
        # the parent's q.get(timeout=...) is the only fallback and we'd
        # waste 5 minutes on what's actually a fast crash.
        out_queue.put(("err", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


def main() -> int:
    args = parse_args()

    print(
        f"  tuning: gc.threshold={gc.get_threshold()} sys.maxsize={sys.maxsize}",
        file=sys.stderr,
    )

    if not args.traces.is_dir():
        print(f"error: --traces {args.traces} is not a directory", file=sys.stderr)
        return 2

    trace_files = sorted(args.traces.glob("*.json"))
    if not trace_files:
        print(f"error: no .json traces found under {args.traces}", file=sys.stderr)
        return 2

    total_available = len(trace_files)
    if args.max_traces > 0 and total_available > args.max_traces:
        trace_files = trace_files[: args.max_traces]
        print(
            f"  capped to first {len(trace_files)} of {total_available} traces "
            f"(--max-traces={args.max_traces}). Pass --max-traces 0 to disable.",
            file=sys.stderr,
        )

    global _PARENT_TK, _PARENT_PG
    _PARENT_TK = _build_tokenizer(args.tokenizer)
    trace_patterns, recorded, models = _classify_turns(args.traces)
    if not models:
        print(
            f"error: could not infer any model_names from traces in {args.traces}",
            file=sys.stderr,
        )
        return 2

    # Built once in the parent; the tokenized prompt corpus (~MB) is
    # COW-shared with every child via fork.
    _PARENT_PG = PromptGenerator(
        PromptConfig(
            mean=200,
            stddev=0,
            block_size=DEFAULT_BLOCK_SIZE,
            prefix_prompt=PrefixPromptConfig(pool_size=0, length=0),
        ),
        _PARENT_TK,
    )

    orig_snapshot = weka_synth_buf.ConversationReconstructor.snapshot_messages

    def patched_snapshot(self):  # type: ignore[no-untyped-def]
        _CAPTURED.append([len(s.tokens) for s in self._segments])
        return orig_snapshot(self)

    weka_synth_buf.ConversationReconstructor.snapshot_messages = patched_snapshot

    # Force fork explicitly: Python 3.14+ may default to spawn on Linux,
    # which would re-import everything per child and lose the inherited
    # tokenizer + tokenized corpus + capture-list COW share.
    mp_ctx = mp.get_context("fork")

    rows: list[dict[str, Any]] = []
    current_proc: mp.process.BaseProcess | None = None

    def _on_sigint(signum, frame):  # type: ignore[no-untyped-def]
        if current_proc is not None and current_proc.is_alive():
            current_proc.terminate()
        # Restore default and re-raise so the parent unwinds cleanly.
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        raise KeyboardInterrupt

    prev_sigint = signal.signal(signal.SIGINT, _on_sigint)

    try:
        for i, trace_path in enumerate(trace_files):
            # Reset in the parent before fork so the child's COW snapshot
            # of _CAPTURED starts empty.
            _CAPTURED.clear()
            q: mp.Queue[Any] = mp_ctx.Queue()
            p = mp_ctx.Process(
                target=_process_one_trace_in_child,
                args=(trace_path, models, args.tokenizer, recorded, trace_patterns, q),
            )
            current_proc = p
            p.start()
            try:
                try:
                    status, payload = q.get(timeout=300)
                except Exception as e:  # noqa: BLE001
                    p.terminate()
                    p.join(5)
                    print(
                        f"  [{i + 1}/{len(trace_files)}] WARN: "
                        f"{trace_path.name} timed out: {e}",
                        file=sys.stderr,
                    )
                    continue
                p.join(timeout=10)
                if p.is_alive():
                    p.terminate()
                    p.join(5)
                if status == "ok":
                    rows.extend(payload)
                else:
                    print(
                        f"  [{i + 1}/{len(trace_files)}] WARN: "
                        f"{trace_path.name}: {payload}",
                        file=sys.stderr,
                    )
            finally:
                current_proc = None
                if p.is_alive():
                    p.terminate()
                    p.join(5)

            if (i + 1) % 25 == 0 or i + 1 == len(trace_files):
                rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                n_objects = len(gc.get_objects())
                print(
                    f"  [{i + 1}/{len(trace_files)}] parent RSS = {rss_mib:.1f} MiB, "
                    f"objects = {n_objects}, rows accumulated = {len(rows)}",
                    file=sys.stderr,
                )
    finally:
        signal.signal(signal.SIGINT, prev_sigint)
        weka_synth_buf.ConversationReconstructor.snapshot_messages = orig_snapshot

    if not rows:
        print("no comparable rows produced — corpus may be empty or all subagent")
        return 0

    _print_report(rows, args.top_n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
