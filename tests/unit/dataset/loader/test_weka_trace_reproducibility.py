# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-process reproducibility test for the weka byte-exact loader.

Spawns two subprocesses with different PYTHONHASHSEED, runs the loader on
the same fixture trace, and asserts byte-identical outputs. Verifies the
sha256-keyed determinism contract from spec §4.6 — Python's builtin
hash() is salted per-process via PYTHONHASHSEED, and any path that
depends on it would diverge across runs (kv-cache-tester audit H3).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parents[3] / "fixtures" / "weka_traces"


# Inline runner script used in the subprocess. Walks one Weka trace through
# the loader, dumps a deterministic representation of every emitted Turn's
# raw_messages to stdout (sorted JSON). The hash of stdout is then compared
# across PYTHONHASHSEED variants to detect any per-process nondeterminism.
RUNNER = textwrap.dedent("""
    import json
    import sys
    from unittest.mock import MagicMock

    from aiperf.dataset.loader.weka_trace import WekaTraceLoader

    fixture_path = sys.argv[1]

    uc = MagicMock()
    uc.input.random_seed = 0
    uc.input.fixed_schedule_start_offset = None
    uc.input.fixed_schedule_end_offset = None
    uc.input.ignore_trace_delays = False
    uc.input.use_think_time_only = False
    uc.input.use_end_to_start_delays = False
    uc.loadgen.inter_turn_delay_cap_seconds = None
    uc.loadgen.trace_idle_gap_cap_seconds = None
    uc.input.synthesis.max_isl = None
    uc.input.synthesis.max_osl = None
    uc.input.max_context_length = None
    uc.input.synthesis.should_synthesize.return_value = False
    uc.input.prompt.input_tokens.block_size = None
    uc.tokenizer.trust_remote_code = False
    uc.tokenizer.revision = None
    uc.tokenizer.name = "test-tok"
    uc.endpoint.model_names = [
        "claude-opus-4-5-20251101",
        "claude-haiku-4-5-20251001",
        "m",
    ]

    loader = WekaTraceLoader(filename=fixture_path, user_config=uc)

    pg = MagicMock()
    pg._cache = {}
    pg._sample_tokens.side_effect = lambda n: [0] * n
    pg._tokenized_corpus = list(range(10000, 11000))
    pg._corpus_size = 1000
    state = {"h": 0}
    def _reseed(h):
        state["h"] = h
    pg._hash_id_corpus_rng.reseed_for_hash_id.side_effect = _reseed
    pg._hash_id_corpus_rng.randrange.side_effect = lambda n: state["h"] % n
    pg.tokenizer.decode.side_effect = lambda toks: "x" * len(toks)
    loader.prompt_generator = pg
    loader._tokenizer_name = "test-tok"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64
    loader.synthesize_prompts_from_hash_ids = (
        lambda reqs: {r.key: f"prompt-{r.key}" for r in reqs}
    )

    convs = loader.convert_to_conversations(loader.load_dataset())
    out = []
    for c in sorted(convs, key=lambda c: c.session_id):
        for k, t in enumerate(c.turns):
            msgs = []
            for m in (t.raw_messages or []):
                # Project only the load-bearing keys to insulate against
                # any incidental MagicMock leakage / repr-id drift.
                msgs.append({
                    "role": m.get("role"),
                    "content": m.get("content"),
                })
            out.append({
                "sid": c.session_id,
                "k": k,
                "msgs": msgs,
            })
    sys.stdout.write(json.dumps(out, sort_keys=True))
""")


def _run_with_seed(seed: str | int, fixture_path: Path) -> bytes:
    """Run the loader script in a subprocess with a fixed PYTHONHASHSEED."""
    env = {**os.environ, "PYTHONHASHSEED": str(seed)}
    return subprocess.check_output(
        [sys.executable, "-c", RUNNER, str(fixture_path)],
        env=env,
        timeout=120,
    )


@pytest.mark.parametrize(
    "fixture_name",
    ["simple.json", "one_subagent.json", "multi_model.json"],
)
def test_loader_byte_identical_across_processes(fixture_name: str) -> None:
    """Run the loader twice with different PYTHONHASHSEEDs; outputs must match.

    Covers parent-only (simple.json), parent + one subagent (one_subagent.json),
    and multi-model (multi_model.json) fixtures.
    """
    fixture = FIXTURES / fixture_name
    if not fixture.exists():
        pytest.skip(f"Fixture {fixture} not present")

    a = _run_with_seed(0, fixture)
    b = _run_with_seed(42, fixture)
    c = _run_with_seed("random", fixture)

    sha_a = hashlib.sha256(a).hexdigest()
    sha_b = hashlib.sha256(b).hexdigest()
    sha_c = hashlib.sha256(c).hexdigest()

    assert sha_a == sha_b, (
        f"PYTHONHASHSEED=0 vs 42 diverged for {fixture_name}: {sha_a} != {sha_b}"
    )
    assert sha_a == sha_c, (
        f"PYTHONHASHSEED=0 vs 'random' diverged for {fixture_name}: {sha_a} != {sha_c}"
    )
    # Sanity: non-empty output (catches silent skips where the loader
    # produced nothing and every seed produced the same empty string).
    assert len(a) > 2, f"Loader produced empty output for {fixture_name}"
