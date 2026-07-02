# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command for rendering the AIPerf turn-messages HTML viewer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

app = App(
    name="turn-messages",
    help="Render a collapsible HTML viewer of the per-turn input messages a run sent.",
)


@app.default
def turn_messages(
    run_dirs: list[Path],
    out: Annotated[Path | None, Parameter(name=["-o", "--out"])] = None,
    limit_conversations: Annotated[
        int, Parameter(name=["-n", "--limit-conversations"])
    ] = 40,
    max_turns: Annotated[int, Parameter(name=["--max-turns"])] = 60,
    content_cap: Annotated[int, Parameter(name=["--content-cap"])] = 8000,
) -> None:
    """Render an interactive turn-messages HTML viewer for AIPerf run directories.

    Each run directory must contain ``--export-level raw`` output
    (``raw_records/*.jsonl`` shards or a single ``profile_export_raw.jsonl``).
    The viewer is a single self-contained file: conversation -> turn -> message,
    built lazily in the browser from a zstd+base64 payload with an inlined
    decoder (no network). Output defaults to ``<run_dir>/turn_messages.html``.

    Examples:
        # Render the newest run (writes <run_dir>/turn_messages.html)
        aiperf analyze turn-messages ./artifacts/my-run/

        # Multiple runs in one invocation
        aiperf analyze turn-messages ./artifacts/run_a/ ./artifacts/run_b/

        # Single run, explicit output path, show more conversations, full bodies
        aiperf analyze turn-messages ./artifacts/my-run/ -o /tmp/msgs.html \
            -n 1000 --content-cap 1000000

    Args:
        run_dirs: One or more AIPerf run directories.
        out: Output HTML path. Only valid when a single run directory is given.
        limit_conversations: Max conversations to render (roots first, then by
            earliest request time).
        max_turns: Max turns rendered per conversation; the rest are summarized
            as a hidden count.
        content_cap: Max characters kept per unique message body; longer bodies
            are truncated with a remaining-chars note. Raise for full fidelity.
    """
    from aiperf.analysis.turn_messages import (
        TurnMessagesError,
        write_turn_messages_html,
    )

    if out is not None and len(run_dirs) > 1:
        print("error: --out only valid with a single run dir", file=sys.stderr)
        sys.exit(2)

    failures = 0
    for run_dir in run_dirs:
        try:
            saved = write_turn_messages_html(
                run_dir,
                out=out,
                limit_conversations=limit_conversations,
                max_turns=max_turns,
                content_cap=content_cap,
            )
            print(f"saved {saved} ({saved.stat().st_size // 1024:,} KB)")
        except TurnMessagesError as e:
            print(f"skip {run_dir}: {e}", file=sys.stderr)
            failures += 1
    if failures and failures == len(run_dirs):
        sys.exit(1)
