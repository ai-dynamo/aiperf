# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-trace per-turn inspection of weka loader output.

Runs ``WekaTraceLoader`` against a single ``.json`` trace and dumps, for
every turn:

  * Turn index, structural pattern (init / A / B / C) with LCP/M_prev info.
  * Recorded ``in[k]`` and ``out[k]``.
  * Hash-IDs summary (head + tail when ``M`` is large).
  * Per-message: role, ``encode``-token count, head/tail content preview.
  * ``sum(re-encode per msg)`` vs recorded — algorithm fidelity check.
  * ``encode(' '.join(content))`` vs recorded — what aiperf reports as ISL.
  * Drift vs recorded for both quantities.

Useful for explaining loader behavior in PR/audit prose, debugging a
single trace flagged by ``tools/weka_loader_drift_audit.py``, or as a
regression smoke test against a reference trace.

Usage:
  python tools/weka_trace_inspect.py \\
      --trace /path/to/trace_0103.json \\
      [--tokenizer Qwen/Qwen3-0.6B] \\
      [--turn N] \\
      [--show-content-chars 60]

When ``--turn N`` is provided, only that turn is printed but with extra
detail: full per-segment ``block_start`` / ``block_count`` and a less
abbreviated hash_ids list.

Always exits 0 on a successful run; exits nonzero only on missing inputs
or invalid arguments.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from transformers import AutoTokenizer

from aiperf.common.config import PrefixPromptConfig, PromptConfig
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.generator.prompt import PromptGenerator
from aiperf.dataset.loader import weka_synth_buf
from aiperf.dataset.loader.weka_trace import WekaTraceLoader

DEFAULT_TOKENIZER = "Qwen/Qwen3-0.6B"
DEFAULT_BLOCK_SIZE = 64


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--trace",
        type=Path,
        required=True,
        help="Path to a single recorded weka trace .json file.",
    )
    p.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        help=f"HF tokenizer name to load locally. Default: {DEFAULT_TOKENIZER}.",
    )
    p.add_argument(
        "--turn",
        type=int,
        default=None,
        help="If set, print only this turn index (with extra per-segment detail).",
    )
    p.add_argument(
        "--show-content-chars",
        type=int,
        default=60,
        help="How many head/tail characters of each message to preview. Default 60.",
    )
    return p.parse_args()


def _classify(prev: list[int] | None, curr: list[int]) -> str:
    if prev is None:
        return "init"
    lcp = 0
    while lcp < min(len(prev), len(curr)) and prev[lcp] == curr[lcp]:
        lcp += 1
    if lcp == len(prev):
        return f"A (LCP={lcp}=M_prev)"
    if lcp == len(prev) - 1:
        return f"B (LCP={lcp}, M_prev={len(prev)})"
    return f"C (LCP={lcp}, M_prev={len(prev)} -> M_curr={len(curr)})"


def _build_tokenizer(name: str) -> Tokenizer:
    auto = AutoTokenizer.from_pretrained(name, local_files_only=True)
    tk = Tokenizer()
    tk._tokenizer = auto
    tk._resolved_name = name
    tk._apply_kwarg_overrides()
    return tk


def _make_user_config(model_names: set[str], tokenizer_name: str) -> MagicMock:
    uc = MagicMock()
    uc.input.random_seed = 0
    uc.input.fixed_schedule_start_offset = None
    uc.input.fixed_schedule_end_offset = None
    uc.input.synthesis.max_isl = None
    uc.input.synthesis.max_osl = None
    uc.input.synthesis.should_synthesize.return_value = False
    uc.input.prompt.input_tokens.block_size = None
    uc.tokenizer.trust_remote_code = False
    uc.tokenizer.revision = None
    uc.tokenizer.name = tokenizer_name
    uc.endpoint.model_names = sorted(model_names)
    return uc


def _format_hash_ids(curr: list[int], detailed: bool) -> str:
    threshold = 30 if detailed else 12
    if len(curr) <= threshold:
        return f"{curr}"
    if detailed:
        head = ", ".join(str(x) for x in curr[:6])
        tail = ", ".join(str(x) for x in curr[-6:])
        return f"[{head}, ..., {tail}] (M={len(curr)})"
    return f"[{curr[0]}, {curr[1]}, ..., {curr[-2]}, {curr[-1]}] (M={len(curr)})"


def _format_preview(content: str, n: int) -> tuple[str, str]:
    head = content[:n].replace("\n", "\\n")
    tail = ""
    if len(content) > 2 * n:
        tail = content[-n:].replace("\n", "\\n")
    return head, tail


def _print_turn(
    *,
    k: int,
    pat: str,
    curr_hash: list[int],
    recorded_in: int,
    recorded_out: int,
    msgs: list[dict[str, Any]],
    seg_lens: list[int] | None,
    seg_block_info: list[tuple[int, int]] | None,
    tk: Tokenizer,
    detailed: bool,
    show_chars: int,
) -> None:
    print(f"=== Turn {k}  pattern={pat} ===")
    print(f"  hash_ids: {_format_hash_ids(curr_hash, detailed)}")
    print(f"  recorded in[{k}]={recorded_in}, out[{k}]={recorded_out}")

    if detailed and seg_lens is not None:
        print(f"  segments ({len(seg_lens)} total):")
        if seg_block_info is not None:
            for i, ((bs, bc), n_tok) in enumerate(
                zip(seg_block_info, seg_lens, strict=False)
            ):
                print(
                    f"    seg[{i}] block_start={bs} block_count={bc} "
                    f"len(tokens)={n_tok}"
                )
        else:
            for i, n_tok in enumerate(seg_lens):
                print(f"    seg[{i}] len(tokens)={n_tok}")
        print(f"  sum(len(seg.tokens)) = {sum(seg_lens)} (canonical synth ISL)")

    print(f"  Loader emitted {len(msgs)} messages:")
    sum_emitted_tokens = 0
    for i, m in enumerate(msgs):
        ntok = len(tk.encode(m["content"]))
        sum_emitted_tokens += ntok
        head, tail = _format_preview(m["content"], show_chars)
        if tail:
            print(
                f"    [{i}] role={m['role']:<10} encode_tokens={ntok:>5}  "
                f"'{head}...' ... '{tail}'"
            )
        else:
            print(f"    [{i}] role={m['role']:<10} encode_tokens={ntok:>5}  '{head}'")
    print(f"  sum(re-encode per msg) = {sum_emitted_tokens}")
    print(f"  drift sum_per_msg vs recorded: {sum_emitted_tokens - recorded_in:+d}")

    joined = " ".join(m["content"] for m in msgs)
    joined_enc = len(tk.encode(joined))
    print(
        f"  encode(' '.join(content)) = {joined_enc}  (this is what aiperf reports as ISL)"
    )
    print(f"  drift join vs recorded: {joined_enc - recorded_in:+d}")
    print()


def main() -> int:
    args = parse_args()

    if not args.trace.is_file():
        print(f"error: --trace {args.trace} is not a file", file=sys.stderr)
        return 2

    blob = json.loads(args.trace.read_text())
    if "id" not in blob or "requests" not in blob:
        print(
            f"error: {args.trace} does not look like a weka trace "
            "(missing 'id' or 'requests')",
            file=sys.stderr,
        )
        return 2

    ns_reqs = [r for r in blob["requests"] if r.get("type") in ("n", "s")]
    if not ns_reqs:
        print(f"error: {args.trace} has no n/s requests", file=sys.stderr)
        return 2

    if args.turn is not None and not (0 <= args.turn < len(ns_reqs)):
        print(
            f"error: --turn {args.turn} out of range (0..{len(ns_reqs) - 1})",
            file=sys.stderr,
        )
        return 2

    tk = _build_tokenizer(args.tokenizer)
    pg = PromptGenerator(
        PromptConfig(
            mean=200,
            stddev=0,
            block_size=DEFAULT_BLOCK_SIZE,
            prefix_prompt=PrefixPromptConfig(pool_size=0, length=0),
        ),
        tk,
    )
    models = {r.get("model") for r in blob["requests"] if r.get("model")}
    uc = _make_user_config({m for m in models if m}, args.tokenizer)

    captured_lens: list[list[int]] = []
    captured_blocks: list[list[tuple[int, int]]] = []
    orig_snapshot = weka_synth_buf.ConversationReconstructor.snapshot_messages

    def patched_snapshot(self):  # type: ignore[no-untyped-def]
        captured_lens.append([len(s.tokens) for s in self._segments])
        # Segment objects expose block_start/block_count after P17; fall
        # back to (-1, -1) on older variants so older corpora still load.
        block_info: list[tuple[int, int]] = []
        for s in self._segments:
            bs = getattr(s, "block_start", -1)
            bc = getattr(s, "block_count", -1)
            block_info.append((bs, bc))
        captured_blocks.append(block_info)
        return orig_snapshot(self)

    def _seq_decode(seqs, name, **_):  # type: ignore[no-untyped-def]
        return [tk.decode(s) for s in seqs]

    weka_synth_buf.ConversationReconstructor.snapshot_messages = patched_snapshot
    try:
        # The loader scans a directory; copy the single trace into a temp
        # dir so we don't pull in unrelated traces from the parent.
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            shutil.copy(args.trace, td_path / args.trace.name)
            loader = WekaTraceLoader(
                filename=str(td_path),
                user_config=uc,
                prompt_generator=pg,
            )
            loader._block_size = DEFAULT_BLOCK_SIZE
            with patch(
                "aiperf.dataset.loader.hash_ids_synthesis.parallel_decode",
                _seq_decode,
            ):
                convs = loader.convert_to_conversations(loader.load_dataset())
    finally:
        weka_synth_buf.ConversationReconstructor.snapshot_messages = orig_snapshot

    target = next(
        (
            c
            for c in convs
            if c.session_id == blob["id"] and "::sa:" not in c.session_id
        ),
        None,
    )
    if target is None:
        print(
            f"error: loader did not emit a non-subagent conversation for id={blob['id']}",
            file=sys.stderr,
        )
        return 2

    recorded_ins = [r["in"] for r in ns_reqs]
    recorded_outs = [r["out"] for r in ns_reqs]

    # Capture index for the target conv: the loader emits in trace order, so
    # the first len(target.turns) snapshots belong to it (subagents follow).
    capture_idx_base = 0
    for c in convs:
        if c is target:
            break
        capture_idx_base += len(c.turns)

    print(f"Trace: {blob['id']} ({args.trace.name})")
    print(f"Turns: {len(target.turns)} (recorded: {len(ns_reqs)})")
    print(f"Tokenizer: {args.tokenizer}")
    if args.turn is not None:
        print(f"Focused turn: k={args.turn} (extra detail mode)")
    print()

    prev_h: list[int] | None = None
    for k, turn in enumerate(target.turns):
        if k >= len(ns_reqs):
            break
        curr_h = ns_reqs[k]["hash_ids"]
        pat = _classify(prev_h, curr_h)
        prev_h = curr_h

        if args.turn is not None and k != args.turn:
            continue

        idx = capture_idx_base + k
        seg_lens = captured_lens[idx] if idx < len(captured_lens) else None
        seg_blocks = captured_blocks[idx] if idx < len(captured_blocks) else None
        msgs = turn.raw_messages or []
        _print_turn(
            k=k,
            pat=pat,
            curr_hash=curr_h,
            recorded_in=recorded_ins[k],
            recorded_out=recorded_outs[k],
            msgs=msgs,
            seg_lens=seg_lens,
            seg_block_info=seg_blocks,
            tk=tk,
            detailed=(args.turn is not None),
            show_chars=args.show_content_chars,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
