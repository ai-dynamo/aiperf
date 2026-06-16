# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-contained collapsible viewer of the per-turn INPUT messages a run sent.

Reads ``--export-level raw`` output (``raw_records/*.jsonl`` shards or a single
``profile_export_raw.jsonl``) under a run directory and renders a nested
dropdown tree::

    conversation  ->  turn (turn_index)  ->  message (by role)

Each raw record is one turn's request; ``payload.messages`` is the accumulated
chat array sent for that turn. Conversations are grouped by ``conversation_id``;
subagent conversations (``agent_depth >= 1``) are labelled with the parent
conversation resolved via ``parent_correlation_id``.

The data is NOT baked into HTML markup (that produced >100 MB files that choke
the browser). Instead the conversation/turn/message tree is serialised to
compact JSON, zstd-compressed with a 64 MB long-distance window, base64-embedded
as a single string literal, and inflated client-side by an inlined pure-JS zstd
decoder (``fzstd.umd.js``) -- no network, no WASM, no ``DecompressionStream``.

Because every turn re-sends the full accumulated history, identical message
items repeat constantly, so each unique ``(role, content)`` is interned ONCE
into a ``msgs`` lookup table and every turn references it by integer id. That is
a lossless dedup of the byte-identical repeats; the leftover redundancy is
*near*-duplicate large bodies scattered across the payload, which zstd's 64 MB
window catches but gzip's 32 KB window cannot.

The DOM is built LAZILY: a conversation's turns materialise only when its
``<details>`` is first opened, and a turn's messages only when that turn is
opened, so the live DOM stays tiny regardless of run size.
"""

from __future__ import annotations

import base64
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import orjson
import zstandard

from aiperf.common.config.config_defaults import OutputDefaults

VIEWER_TEMPLATE = Path(__file__).with_name("turn_messages_viewer.html")
FZSTD_JS = Path(__file__).with_name("fzstd.umd.js")

# The inlined fzstd 0.1.1 decoder silently corrupts frames with windowLog >= 27
# (128 MiB) -- correct length, wrong bytes. 26 (64 MiB) is the largest safe
# window and captures essentially all the long-range redundancy. Do NOT raise
# without re-verifying fzstd round-trips; write_turn_messages_html self-checks
# every payload (and caps the decoder window) before writing.
ZSTD_WINDOW_LOG = 26

# Generator defaults, shared with the CLI so they document once.
DEFAULT_LIMIT_CONVERSATIONS = 40
DEFAULT_MAX_TURNS = 60
DEFAULT_CONTENT_CAP = 8000


class TurnMessagesError(Exception):
    """Raised when a run directory has no usable raw records to render."""


def _text_of(content: Any) -> str:
    """OpenAI message content is a str or a list of typed parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for p in content:
            if isinstance(p, dict):
                out.append(p.get("text") or f"<{p.get('type', 'part')}>")
            else:
                out.append(str(p))
        return "\n".join(out)
    return str(content)


def _short(s: str | None, n: int = 14) -> str:
    """Truncate ``s`` to ``n`` chars, mapping a falsy value to ``"?"``."""
    return (s or "?")[:n]


def _disp_id(conv: str) -> str:
    """Display id that keeps the DISTINGUISHING part of a conversation id visible.

    Conversation ids in a split corpus share one long trace-id prefix (root
    ``<trace>``, subagents ``<trace>::sa:<id>``, flat chains ``<trace>::fa:NNN``)
    -- naive prefix truncation renders every panel identically. Shorten only the
    trace-id component and keep the full suffix.
    """
    base, sep, suffix = conv.partition("::")
    if sep:
        return f"{_short(base, 10)}…::{suffix}"
    return f"{_short(base, 20)}…"


def _find_raw_records(run_dir: Path) -> list[Path]:
    """Locate raw-record jsonl for ``run_dir``.

    Accepts either a direct jsonl file or a run directory (a ``raw_records/``
    shard folder, a single ``profile_export_raw.jsonl``, or a recursive
    fallback for custom export-base names).
    """
    if run_dir.is_file():
        return [run_dir]
    folder = run_dir / OutputDefaults.RAW_RECORDS_FOLDER
    if folder.is_dir():
        shards = sorted(folder.glob("*.jsonl"))
        if shards:
            return shards
    single = run_dir / OutputDefaults.PROFILE_EXPORT_RAW_JSONL_FILE
    if single.is_file():
        return [single]
    return sorted(run_dir.glob("**/raw_records/*.jsonl")) + sorted(
        run_dir.glob("**/*_raw.jsonl")
    )


def _flatten_record(r: dict) -> dict:
    """Flatten one parsed raw record into the fields ``build_payload`` consumes.

    Tolerant of malformed records: ``metadata``/``payload`` that are missing or
    not objects fall back to empty, and ``messages`` that is not a list is
    dropped (a non-list would otherwise be iterated character-by-character
    downstream).
    """
    md = r.get("metadata")
    md = md if isinstance(md, dict) else {}
    pl = r.get("payload")
    pl = pl if isinstance(pl, dict) else {}
    msgs = pl.get("messages")
    return {
        "conv": md.get("conversation_id") or "(none)",
        "turn": md.get("turn_index"),
        "depth": md.get("agent_depth") or 0,
        "parent_corr": md.get("parent_correlation_id"),
        "x_corr": md.get("x_correlation_id"),
        "model": pl.get("model"),
        "max_tokens": pl.get("max_completion_tokens") or pl.get("max_tokens"),
        "phase": md.get("benchmark_phase"),
        "cancelled": md.get("was_cancelled"),
        "error": r.get("error"),
        "messages": msgs if isinstance(msgs, list) else [],
        "start_ns": md.get("request_start_ns") or 0,
    }


def load_records(files: list[Path]) -> list[dict]:
    """Parse and flatten the per-turn raw records from the given jsonl files."""
    recs: list[dict] = []
    for f in files:
        with open(f, "rb") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                if isinstance(r, dict):
                    recs.append(_flatten_record(r))
    return recs


def build_payload(
    recs: list[dict], n_files: int, cap: int, limit: int, max_turns: int
) -> dict:
    """Compute the compact, interned render tree (data only -- no HTML)."""
    cap, limit, max_turns = max(cap, 0), max(limit, 0), max(max_turns, 0)
    corr2conv = {r["x_corr"]: r["conv"] for r in recs if r["x_corr"]}
    # A conversation can be replayed by multiple concurrent lanes in one run
    # (e.g. --concurrency N recycling a small dataset, trajectory wrap-fill).
    # x_correlation_id is stable across every turn of ONE session, so it is the
    # true lane identity; number a conversation's lanes by first-arrival order.
    # Fall back to repeated-(turn_index) arrival order only when correlation ids
    # are absent from the export.
    lanes_by_conv: dict[str, dict[str, int]] = defaultdict(dict)
    seen_ti: dict[tuple[str, int], int] = defaultdict(int)
    for r in sorted(recs, key=lambda r: r["start_ns"] or 0):
        if r.get("x_corr"):
            lanes = lanes_by_conv[r["conv"]]
            if r["x_corr"] not in lanes:
                lanes[r["x_corr"]] = len(lanes)
            r["replay"] = lanes[r["x_corr"]]
        else:
            key = (r["conv"], r["turn"])
            r["replay"] = seen_ti[key]
            seen_ti[key] += 1
    by_conv: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        replay = r.get("replay", 0)
        conv_key = r["conv"] if replay == 0 else f"{r['conv']} · replay {replay + 1}"
        by_conv[conv_key].append(r)

    # Order: roots first, then by earliest request time; cap conversation count.
    conv_order = sorted(
        by_conv,
        key=lambda c: (
            min(r["depth"] for r in by_conv[c]),
            min(r["start_ns"] or 0 for r in by_conv[c]),
            c,
        ),
    )
    total_convs = len(conv_order)
    shown = conv_order[:limit]

    n_turns = sum(len(v) for v in by_conv.values())
    n_msgs = sum(len(r["messages"]) for r in recs)
    role_counts = Counter(
        (m.get("role", "?") if isinstance(m, dict) else "?")
        for r in recs
        for m in r["messages"]
    )

    # Intern unique messages: every (role, content) stored once in `table`; each
    # turn references them by integer id. Lossless dedup of the repeated history.
    intern: dict[tuple[str, str], int] = {}
    table: list[dict] = []

    convs: list[dict] = []
    for conv in shown:
        turns = sorted(
            by_conv[conv],
            key=lambda r: (r["turn"] if r["turn"] is not None else 0, r["start_ns"]),
        )
        depth = min(r["depth"] for r in turns)
        kind = "root" if depth == 0 else f"subagent (depth {depth})"
        parent_corr = next((r["parent_corr"] for r in turns if r["parent_corr"]), None)
        parent_conv = corr2conv.get(parent_corr)
        parent_lbl = _disp_id(parent_conv) if parent_conv else None
        models = ",".join(sorted({r["model"] or "?" for r in turns}))
        n_t = len(turns)
        more_t = 0
        if n_t > max_turns:
            more_t = n_t - max_turns
            turns = turns[:max_turns]
        lane = next((r["x_corr"] for r in turns if r["x_corr"]), None)
        base_id, _, replay_tag = conv.partition(" · ")

        turns_payload: list[dict] = []
        for r in turns:
            ids: list[int] = []
            for m in r["messages"]:
                if isinstance(m, dict):
                    role = m.get("role", "?")
                    content = _text_of(m.get("content"))
                else:
                    role = "?"
                    content = _text_of(m)
                key = (role, content)
                mid = intern.get(key)
                if mid is None:
                    mid = len(table)
                    intern[key] = mid
                    clen = len(content)
                    table.append(
                        {
                            "role": role,
                            "len": clen,
                            "body": content[:cap],
                            "trunc": max(0, clen - cap),
                        }
                    )
                ids.append(mid)
            turns_payload.append(
                {
                    "ti": r["turn"],
                    "phase": r["phase"],
                    "maxTokens": r["max_tokens"],
                    "cancelled": bool(r["cancelled"]),
                    "error": bool(r["error"]),
                    "ids": ids,
                }
            )
        convs.append(
            {
                "id": base_id,
                "disp": _disp_id(base_id),
                "kind": kind,
                "isRoot": depth == 0,
                "parent": parent_lbl,
                "lane": _short(lane, 8) if lane else None,
                "models": models,
                "nt": n_t,
                "moreTurns": more_t,
                "replayTag": replay_tag or "",
                "turns": turns_payload,
            }
        )

    stat = (
        f"{total_convs:,} conversations ({len(shown):,} shown) · {n_turns:,} turns · "
        f"{n_msgs:,} message refs ({len(table):,} unique) · from {n_files} raw shard(s)"
    )
    return {
        "stat": stat,
        "legend": role_counts.most_common(),
        "msgs": table,
        "convs": convs,
    }


def write_turn_messages_html(
    run_dir: Path,
    out: Path | None = None,
    *,
    limit_conversations: int = DEFAULT_LIMIT_CONVERSATIONS,
    max_turns: int = DEFAULT_MAX_TURNS,
    content_cap: int = DEFAULT_CONTENT_CAP,
) -> Path:
    """Render the interactive turn-messages viewer for ``run_dir``.

    Reads the run's ``--export-level raw`` records, interns the messages, zstd
    compresses the payload (pinned to a fzstd-safe ``windowLog=26`` window and
    round-trip-verified), embeds it with the inlined decoder, and writes a
    single self-contained HTML file. ``run_dir`` may be a run directory or a
    direct raw jsonl file; output defaults to ``turn_messages.html`` beside the
    records. Returns the output path.

    Raises:
        TurnMessagesError: if ``run_dir`` has no raw records, none parse, the
            compressed payload fails its round-trip self-check, or the output
            file cannot be written.
    """
    files = _find_raw_records(run_dir)
    if not files:
        raise TurnMessagesError(
            f"no raw records under {run_dir}; run aiperf with --export-level raw first"
        )
    recs = load_records(files)
    if not recs:
        raise TurnMessagesError(f"no valid raw records in {run_dir}")

    payload = build_payload(
        recs, len(files), content_cap, limit_conversations, max_turns
    )
    raw = orjson.dumps(payload)

    params = zstandard.ZstdCompressionParameters.from_level(
        19, enable_ldm=True, window_log=ZSTD_WINDOW_LOG
    )
    compressed = zstandard.ZstdCompressor(compression_params=params).compress(raw)

    # Self-check before writing: a windowLog > 26 frame would decode to the
    # right length with WRONG bytes under the inlined fzstd decoder, silently.
    # Verify the payload round-trips, capping the decoder at the safe window.
    dctx = zstandard.ZstdDecompressor(max_window_size=1 << ZSTD_WINDOW_LOG)
    if dctx.decompress(compressed) != raw:
        raise TurnMessagesError("zstd round-trip mismatch — refusing to write")

    payload_b64 = base64.b64encode(compressed).decode()
    html = (
        VIEWER_TEMPLATE.read_text()
        .replace("__FZSTD_JS__", FZSTD_JS.read_text())
        .replace("__PAYLOAD_B64__", payload_b64)
    )
    default_dir = run_dir.parent if run_dir.is_file() else run_dir
    out_path = out or (default_dir / "turn_messages.html")
    try:
        out_path.write_text(html)
    except OSError as e:
        raise TurnMessagesError(f"could not write {out_path}: {e}") from e
    return out_path
