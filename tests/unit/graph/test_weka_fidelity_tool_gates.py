# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The offline fidelity tool's content knobs are threaded, not hardcoded.

``tools/weka_trace_fidelity.py`` used to hardcode ``gpt2`` / ``"coding"`` /
no-seed into its trie rebuild, so any run built with different knobs (or even
the bare live-run defaults, which use the builtin tokenizer) spuriously failed
every content comparison. These tests lock the fix: :func:`build_recorded_trace`
defaults to the live-run knobs (builtin / coding / no seed) and honors explicit
``tokenizer_name`` / ``prompt_corpus`` / ``root_seed`` overrides, and the CLI
``--tokenizer`` / ``--corpus`` / ``--seed`` flags reach the rebuild.

These live in the UNIT lane on purpose: the component-integration package
patches ``Tokenizer.from_pretrained`` to a FakeTokenizer, which flattens
tokenizer/seed-dependent content and makes the knobs indistinguishable. The
unit lane's fake-tokenizer fixture is opt-in, so the real builtin synthesizer
runs here.
"""

from __future__ import annotations

import json
from pathlib import Path

from tools.weka_trace_fidelity import _main, build_recorded_trace

_MODEL = "M"


def _linear_trace() -> dict:
    """A two-turn linear trace: r_1 extends r_0's hash prefix 1.0s after it ends."""
    return {
        "id": "knobs",
        "models": [_MODEL],
        "block_size": 64,
        "hash_id_scope": "local",
        "requests": [
            {"t": 0.0, "type": "n", "model": _MODEL, "in": 64, "out": 8,
             "hash_ids": [1], "api_time": 1.0},
            {"t": 2.0, "type": "n", "model": _MODEL, "in": 128, "out": 8,
             "hash_ids": [1, 2], "api_time": 1.0},
        ],
    }  # fmt: skip


def _write_trace(tmp_path: Path) -> Path:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    path = trace_dir / "knobs.json"
    path.write_text(json.dumps(_linear_trace()))
    return path


def _faithful_raw(tmp_path: Path, trace_file: Path) -> Path:
    """A faithful default-knob export: genuine materialized prompts, warped timing.

    ``r_0`` dispatches at an arbitrary origin; ``r_1`` at origin + its warped
    edge delay off ``r_0`` (the zero-latency-mock collapse of end-to-start onto
    start-to-start), so both criteria pass when the proof rebuilds with the SAME
    content knobs.
    """
    recorded = build_recorded_trace(trace_file)
    t0 = 1_000_000_000_000
    delay_s = recorded.nodes["knobs:1"].pred_delay_us["knobs:0"] / 1e6
    dispatch_ns = {"knobs:0": t0, "knobs:1": t0 + int(delay_s * 1e9)}
    lines = []
    for nid, ns in dispatch_ns.items():
        lines.append(
            json.dumps(
                {
                    "metadata": {
                        "conversation_id": "knobs#0.0",
                        "x_request_id": f"{nid}::deadbeefdeadbeefdeadbeefdeadbeef",
                        "benchmark_phase": "profiling",
                        "request_start_ns": ns,
                        "credit_issued_ns": ns - 1_000_000,
                    },
                    "payload": {"messages": recorded.nodes[nid].messages},
                }
            )
        )
    raw = tmp_path / "profile_export_raw.jsonl"
    raw.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return raw


def test_build_recorded_trace_content_knobs_thread_into_rebuild(
    tmp_path: Path,
) -> None:
    """Explicit seed/corpus knobs change the rebuilt content; the default equals
    the explicit live-run knobs (builtin / coding / no seed)."""
    trace_file = _write_trace(tmp_path)

    default = build_recorded_trace(trace_file)
    explicit = build_recorded_trace(
        trace_file, tokenizer_name="builtin", prompt_corpus="coding", root_seed=None
    )
    seeded = build_recorded_trace(trace_file, root_seed=7)
    sonnet = build_recorded_trace(trace_file, prompt_corpus="sonnet")

    assert {n: default.nodes[n].messages for n in default.nodes} == {
        n: explicit.nodes[n].messages for n in explicit.nodes
    }
    assert any(
        default.nodes[n].messages != seeded.nodes[n].messages for n in default.nodes
    )
    assert any(
        default.nodes[n].messages != sonnet.nodes[n].messages for n in default.nodes
    )


def test_main_content_knob_flags_thread_into_proof(tmp_path: Path) -> None:
    """The ``--tokenizer`` / ``--corpus`` / ``--seed`` CLI flags reach
    ``build_recorded_trace``: a default-knob export passes under explicit default
    flags but fails content under a different ``--seed``."""
    trace_file = _write_trace(tmp_path)
    raw = _faithful_raw(tmp_path, trace_file)
    base_argv = ["--raw", str(raw), "--trace-dir", str(trace_file.parent)]

    assert _main([*base_argv, "--tokenizer", "builtin", "--corpus", "coding"]) == 0
    assert _main([*base_argv, "--seed", "7"]) == 1
