# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-mp E2E: the #1106 graph dataset-selection contract, scenarios 1-3.

Runs the full multiprocess ``aiperf`` stack (subprocess ``python -m aiperf
profile`` + live mock server + real Worker + HTTP) to lock the ai-dynamo/aiperf
#1106 dataset-selection contract end-to-end:

* **Scenario 1 (fail loud).** ``--max-context-length`` rejects most traces,
  leaving far fewer distinct traces than the requested ``--concurrency 100``;
  with cache-bust OFF and wrapping unset the run FAILS LOUDLY -- the wrap-guard's
  ``ConfigurationError`` surfaces and no records are produced, instead of the old
  silent clone-to-fill. (The guard fires in the AWAITED ``setup_phase``, so it
  propagates through the phase-failure path to a non-zero exit; it does not hang.)
* **Scenario 2 (cache-bust rescue).** The SAME over-subscribing shape SUCCEEDS
  once ``--cache-bust first_turn_prefix`` turns wrapping on: the finite eligible
  pool is intentionally recycled to fill the requested concurrency and the run
  completes with real records.
* **Scenario 3 (exact-N selection).** A corpus of >= 100 eligible traces with
  ``--num-dataset-entries 100`` dispatches EXACTLY 100 distinct traces -- not the
  whole corpus, not fewer.

The determinism env (t* window pinned to ``[0, 0]`` = full replay) mirrors the
sibling weka/dynamo e2e tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
import pytest

from tests.harness.dynamo_synth_corpus import write_synthetic_dynamo_capture
from tests.harness.utils import AIPerfCLI, AIPerfMockServer

_MODEL = "m"

# --max-context-length threshold: small eligible traces (peak 144) pass, big
# traces (peak 8200) are rejected by the graph filter-then-cap selection.
_MAX_CTX = 1000
_ELIGIBLE = 3  # small traces surviving the peak-context filter
_REJECTED = 5  # oversized traces the filter drops
_NUM_CONVERSATIONS = 6  # session cap: forces the eligible pool to wrap (> _ELIGIBLE)


def _weka_trace(trace_id: str, *, in_tokens: int, out_tokens: int) -> dict[str, Any]:
    """One minimal single-turn weka trace (peak context == ``in + out``)."""
    return {
        "id": trace_id,
        "models": [_MODEL],
        "block_size": 64,
        "hash_id_scope": "local",
        "requests": [
            {
                "t": 0.0,
                "type": "n",
                "model": _MODEL,
                "in": in_tokens,
                "out": out_tokens,
                "hash_ids": [1, 2],
                "input_types": ["text"],
                "output_types": ["text"],
                "stop": "end_turn",
                "api_time": 0.5,
                "think_time": 0.0,
            }
        ],
    }


def _write_1106_weka_corpus(directory: Path) -> None:
    """Write a #1106-shaped weka dir: a few small traces + many oversized ones."""
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(_ELIGIBLE):
        (directory / f"small_{i:03d}.json").write_bytes(
            orjson.dumps(_weka_trace(f"small-{i}", in_tokens=128, out_tokens=16))
        )
    for i in range(_REJECTED):
        (directory / f"big_{i:03d}.json").write_bytes(
            orjson.dumps(_weka_trace(f"big-{i}", in_tokens=8000, out_tokens=200))
        )


def _template_ids(records: list[Any]) -> set[str]:
    """Distinct template trace ids over records.

    ``conversation_id`` is the nonce-less trajectory TEMPLATE (``{trace}`` for
    the root scope, ``{trace}::{scope}`` for children); the base trace id is
    its first ``::`` segment.
    """
    templates: set[str] = set()
    for rec in records:
        conv = rec.metadata.conversation_id
        assert conv is not None, f"record missing conversation_id: {rec.metadata}"
        templates.add(conv.split("::", 1)[0])
    return templates


def _instance_ids(records: list[Any]) -> set[str]:
    """Distinct per-INSTANCE ids over records.

    Instance identity rides ``x_correlation_id`` (``{conversation}::{nonce}``,
    fresh per trajectory instance): every wrap/recycle of one template mints a
    new corr, so distinct corrs > distinct templates proves cloning happened.
    """
    return {
        rec.metadata.x_correlation_id
        for rec in records
        if rec.metadata.x_correlation_id is not None
    }


@pytest.mark.integration
@pytest.mark.asyncio
class TestIssue1106Selection:
    async def test_oversubscription_without_cache_bust_fails_loud(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Scenario 1: filter drops most traces, then concurrency 100 FAILS LOUD.

        ``--max-context-length 1000`` rejects the 5 oversized traces (peak 8200),
        leaving 3 eligible; ``--num-dataset-entries 100`` does not cap below that;
        resolved concurrency 100 exceeds the 3 distinct loaded traces with
        cache-bust OFF and wrapping unset. The wrap-guard raises a
        ``ConfigurationError`` in the awaited ``setup_phase``, so the run exits
        NON-ZERO with the guard's message surfaced -- NOT a silent clone-to-fill,
        and NOT a hang (no ``--benchmark-duration`` is set; the old orphaned-task
        behavior would have waited on the sending event forever).
        """

        corpus_dir = tmp_path / "corpus"
        _write_1106_weka_corpus(corpus_dir)

        result = await cli.run(
            f"""
            aiperf profile \
                --model {_MODEL} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --input-file {corpus_dir} \
                --tokenizer builtin \
                --random-seed 1234 \
                --max-context-length {_MAX_CTX} \
                --num-dataset-entries 100 \
                --concurrency 100 \
                --cache-bust none \
                --workers-max 2 \
                --ui simple
            """,
            timeout=90.0,
            assert_success=False,
        )
        # Fails loudly: non-zero exit (the guard's ConfigurationError propagates
        # through setup_phase -> the phase-failure path), NOT a clean success.
        assert result.exit_code != 0, "over-subscription must fail, not succeed"

        combined = (result.stderr + result.stdout + (result.log or "")).lower()
        # The guard's actionable message surfaced -- the operator sees WHY it
        # failed, not a generic timeout / no-records error.
        assert (
            f"concurrency 100 exceeds {_ELIGIBLE} distinct loaded traces" in combined
        ), (
            f"expected the wrap-guard message in the run output; got tail: "
            f"{combined[-1500:]}"
        )
        assert "dataset wrapping is disabled" in combined
        assert "--cache-bust first_turn_prefix" in combined
        # NOT a silent clone-to-fill: no profiling records were produced.
        assert result.request_count == 0, (
            "the guard must fail BEFORE dispatch, not clone traces to fill lanes"
        )
        assert not result.jsonl, "no records should be produced when the guard fires"

    async def test_cache_bust_rescues_oversubscribed_corpus(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Scenario 2: over-subscription + ``--cache-bust`` wraps and SUCCEEDS.

        ``--max-context-length 1000`` rejects the 5 oversized traces, leaving 3
        eligible; ``--num-dataset-entries 100`` keeps all 3; resolved concurrency
        100 exceeds those 3. With cache-bust ON, wrapping is allowed (the guard
        does not fire), the eligible pool recycles to fill the lanes, and the run
        completes with real records. ``--num-conversations 6`` bounds the recycle.
        """

        corpus_dir = tmp_path / "corpus"
        _write_1106_weka_corpus(corpus_dir)

        result = await cli.run(
            f"""
            aiperf profile \
                --model {_MODEL} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --input-file {corpus_dir} \
                --tokenizer builtin \
                --random-seed 1234 \
                --max-context-length {_MAX_CTX} \
                --num-dataset-entries 100 \
                --concurrency 100 \
                --num-conversations {_NUM_CONVERSATIONS} \
                --cache-bust first_turn_prefix \
                --workers-max 2 \
                --export-level raw \
                --ui simple
            """,
            timeout=300.0,
            assert_success=True,
        )
        assert result.exit_code == 0, result.stderr[-2000:]
        assert result.request_count > 0, "over-subscribed run produced no records"
        assert result.jsonl, "no per-record metrics captured"

        # Selection kept exactly the 3 eligible traces (the 5 oversized ones were
        # filtered out by --max-context-length before the build).
        templates = _template_ids(result.jsonl)
        assert templates == {f"small-{i}" for i in range(_ELIGIBLE)}, (
            f"expected exactly the {_ELIGIBLE} small eligible templates, got {templates}"
        )
        # Wrapping happened: more distinct instances than distinct templates means
        # the finite eligible pool was cloned/recycled to fill the lanes (the
        # duplication cache-bust made safe), which is the whole point of the
        # cache-bust rescue vs. scenario 1's fail-loud.
        instances = _instance_ids(result.jsonl)
        assert len(instances) > len(templates), (
            f"expected wrapping to clone the {len(templates)} eligible templates "
            f"into more instances, got {len(instances)} instances"
        )

    async def test_num_dataset_entries_selects_exactly_100(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Scenario 3: >=100 eligible traces + ``--num-dataset-entries 100`` -> 100 distinct.

        A 130-session dynamo capture (each session one tiny single-turn trace, all
        well under any context cap) is capped to 100 by ``--num-dataset-entries``.

        The discrimination is that MORE conversations run than the cap:
        ``--num-conversations 130`` with ``--allow-dataset-wrap`` dispatches 130
        sessions over the loaded pool. A WORKING cap loads 100 traces, so the 130
        sessions wrap (sequential draw ``x % 100``) and cover exactly 100 distinct
        templates; a BROKEN cap (a no-op that loaded all 130) would draw ``x % 130``
        over 130 sessions and produce 130 distinct templates. Asserting exactly 100
        therefore fails if the cap did not bind -- unlike a run whose conversation
        count equals the cap, where the draw yields 100 distinct regardless.
        """

        capture = tmp_path / "dynamo_130.jsonl"
        write_synthetic_dynamo_capture(
            capture,
            sessions=130,
            turns_per_session=1,
            new_blocks_per_turn=2,
            block_size=16,
            seed=1234,
        )

        result = await cli.run(
            f"""
            aiperf profile \
                --model {_MODEL} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --input-file {capture} \
                --tokenizer builtin \
                --random-seed 1234 \
                --num-dataset-entries 100 \
                --num-conversations 130 \
                --allow-dataset-wrap \
                --concurrency 10 \
                --workers-max 2 \
                --ui simple
            """,
            timeout=300.0,
            assert_success=True,
        )
        assert result.exit_code == 0, result.stderr[-2000:]
        assert result.jsonl, "no per-record metrics captured"

        # 130 conversations ran (more than the cap), but the cap bound the loaded
        # corpus to 100 of the 130 sessions, so the wrapped draw covers EXACTLY 100
        # distinct templates -- a no-op cap (all 130 loaded) would yield 130.
        templates = _template_ids(result.jsonl)
        assert len(templates) == 100, (
            f"expected exactly 100 distinct dispatched templates from the "
            f"--num-dataset-entries cap over a 130-session corpus, got "
            f"{len(templates)} (130 => the cap was a no-op)"
        )
        # Sanity: more distinct INSTANCES than templates confirms wrapping actually
        # happened (so the 100 is a wrapped-cover, not a short 100-session pass).
        assert len(_instance_ids(result.jsonl)) > len(templates), (
            "expected wrapping (distinct instances > distinct templates) so the "
            "100-distinct-templates result is a genuine capped-then-wrapped cover"
        )
