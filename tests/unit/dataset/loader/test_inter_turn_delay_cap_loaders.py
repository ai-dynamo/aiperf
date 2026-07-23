# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cap behavior across non-weka trace loaders."""

import json
import logging
from pathlib import Path
from unittest.mock import Mock

import pytest

from aiperf.dataset.loader.bailian_trace import BailianTraceDatasetLoader
from aiperf.dataset.loader.burst_gpt import BurstGPTTraceDatasetLoader
from aiperf.dataset.loader.dag_jsonl import DagJsonlLoader
from aiperf.dataset.loader.models import BailianTrace, BurstGPTTrace
from aiperf.dataset.loader.mooncake_trace import MooncakeTraceDatasetLoader
from aiperf.dataset.loader.multi_turn import MultiTurnDatasetLoader
from tests.unit.dataset.loader.conftest import make_weka_run


@pytest.fixture
def cap_run():
    """A v2 BenchmarkRun whose FileDataset carries a 1s inter-turn delay cap.

    Replaces the v1 ``UserConfig`` + ``loadgen.inter_turn_delay_cap_seconds``
    pattern; loaders now read the cap off ``run.cfg``'s active FileDataset.
    """
    return make_weka_run(inter_turn_delay_cap_seconds=1.0)  # 1000 ms


@pytest.fixture
def prompt_generator_factory():
    """Factory producing a deterministic mock prompt_generator.

    Mirrors the inline pattern used by ``test_trace.py`` /
    ``test_burst_gpt_trace.py`` so this test file does not depend on a
    shared conftest fixture.
    """

    def _make() -> Mock:
        gen = Mock()
        gen.generate.return_value = "Generated prompt"
        gen._build_token_sequence.return_value = [1, 2, 3, 4, 5]
        return gen

    return _make


def _write_jsonl(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    p = tmp_path / name
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


def test_mooncake_loader_clamps_inter_turn_delay(
    tmp_path: Path,
    cap_run,
    prompt_generator_factory,
) -> None:
    rows = [
        {"session_id": "s1", "input_length": 10, "output_length": 5},
        {
            "session_id": "s1",
            "delay": 5_000,
            "input_length": 10,
            "output_length": 5,
        },
    ]
    path = _write_jsonl(tmp_path, "mc.jsonl", rows)

    loader = MooncakeTraceDatasetLoader(
        filename=str(path),
        prompt_generator=prompt_generator_factory(),
        run=cap_run,
    )
    data = loader.load_dataset()
    convs = loader.convert_to_conversations(data)

    assert len(convs) == 1
    assert convs[0].turns[1].delay == 1000.0  # clamped to cap


def test_mooncake_payload_mode_clamps_inter_turn_delay(
    tmp_path: Path,
    cap_run,
    prompt_generator_factory,
) -> None:
    """Verbatim ``payload`` turns must honor the cap too (not just the
    synthesized-prompt branch). The payload/messages branches once used the raw
    recorded delay, silently ignoring --inter-turn-delay-cap-seconds."""
    rows = [
        {"session_id": "s1", "payload": {"prompt": "p0"}},
        {"session_id": "s1", "payload": {"prompt": "p1"}, "delay": 5_000},
    ]
    path = _write_jsonl(tmp_path, "mc_payload.jsonl", rows)
    loader = MooncakeTraceDatasetLoader(
        filename=str(path),
        prompt_generator=prompt_generator_factory(),
        run=cap_run,
    )
    convs = loader.convert_to_conversations(loader.load_dataset())
    assert convs[0].turns[1].delay == 1000.0  # clamped to cap
    assert convs[0].turns[1].raw_payload == {"prompt": "p1"}


def test_mooncake_messages_mode_clamps_inter_turn_delay(
    tmp_path: Path,
    cap_run,
    prompt_generator_factory,
) -> None:
    """Verbatim ``messages`` turns must honor the cap too."""
    rows = [
        {"session_id": "s1", "messages": [{"role": "user", "content": "m0"}]},
        {
            "session_id": "s1",
            "messages": [{"role": "user", "content": "m1"}],
            "delay": 5_000,
        },
    ]
    path = _write_jsonl(tmp_path, "mc_messages.jsonl", rows)
    loader = MooncakeTraceDatasetLoader(
        filename=str(path),
        prompt_generator=prompt_generator_factory(),
        run=cap_run,
    )
    convs = loader.convert_to_conversations(loader.load_dataset())
    assert convs[0].turns[1].delay == 1000.0  # clamped to cap
    assert convs[0].turns[1].raw_messages == [{"role": "user", "content": "m1"}]


def test_burst_gpt_loader_clamps_inter_turn_delay(
    tmp_path: Path,
    cap_run,
    prompt_generator_factory,
) -> None:
    """BurstGPT's CSV schema has no ``delay`` column today, but the base
    loader's ``_build_turn`` is shared with mooncake/bailian and must clamp
    any ``delay`` attribute that lands on the trace object. This test feeds
    a synthetic trace through ``_build_turn`` to assert the cap path is
    wired regardless of how the loader populates ``delay``.
    """
    # Empty CSV satisfies BurstGPTTraceDatasetLoader.__init__ requirements
    # (we exercise _build_turn directly, not the CSV-parse path).
    csv_path = tmp_path / "burst.csv"
    csv_path.write_text("Timestamp,Request tokens,Response tokens\n")

    loader = BurstGPTTraceDatasetLoader(
        filename=str(csv_path),
        prompt_generator=prompt_generator_factory(),
        run=cap_run,
    )
    # AIPerfBaseModel is configured with ``extra="allow"`` so an extra
    # ``delay`` attribute is preserved on the trace.
    trace = BurstGPTTrace.model_validate(
        {
            "timestamp": 1.0,
            "input_length": 5,
            "output_length": 5,
            "delay": 5_000,
        }
    )
    turn = loader._build_turn(trace, "prompt")
    assert turn.delay == 1000.0


def test_bailian_loader_clamps_inter_turn_delay(
    tmp_path: Path,
    cap_run,
    prompt_generator_factory,
) -> None:
    """Bailian's schema also lacks a first-class ``delay`` field; the base
    loader's ``_build_turn`` reads ``delay`` via ``getattr``. We verify the
    cap path on the loader's ``_build_turn`` using a Bailian trace that
    carries ``delay`` as a ``extra="allow"`` attribute.
    """
    # Minimal valid file so __init__ + load_dataset can run later if needed.
    rows = [
        {
            "chat_id": 1,
            "parent_chat_id": -1,
            "timestamp": 0.0,
            "input_length": 5,
            "output_length": 5,
        }
    ]
    path = _write_jsonl(tmp_path, "bailian.jsonl", rows)
    loader = BailianTraceDatasetLoader(
        filename=str(path),
        prompt_generator=prompt_generator_factory(),
        run=cap_run,
    )
    trace = BailianTrace.model_validate(
        {
            "chat_id": 1,
            "parent_chat_id": -1,
            "timestamp": 0.0,
            "input_length": 5,
            "output_length": 5,
            "delay": 5_000,
        }
    )
    turn = loader._build_turn(trace, "prompt")
    assert turn.delay == 1000.0


@pytest.mark.parametrize(
    "delay_in, cap_seconds, expected",
    [
        (5_000, 1.0, 1000.0),  # delay > cap_ms -> clamped
        (500, 1.0, 500.0),  # delay < cap_ms -> unchanged
        (1_000, 1.0, 1000.0),  # delay == cap_ms -> unchanged (boundary inclusive)
        (1_000_000_000, None, 1_000_000_000.0),  # cap None -> never clamps
        (5_000, 0.0, 0.0),  # cap == 0 -> always clamp to 0
    ],
)
def test_multi_turn_loader_clamps_inter_turn_delay(
    tmp_path: Path,
    delay_in: int,
    cap_seconds: float | None,
    expected: float,
) -> None:
    cfg = make_weka_run(inter_turn_delay_cap_seconds=cap_seconds)
    rows = [
        {
            "session_id": "s1",
            "turns": [
                {"text": "hello"},
                {"text": "world", "delay": delay_in},
            ],
        }
    ]
    path = _write_jsonl(tmp_path, "mt.jsonl", rows)
    loader = MultiTurnDatasetLoader(filename=str(path), run=cfg)
    data = loader.load_dataset()
    convs = loader.convert_to_conversations(data)
    delays = [t.delay for t in convs[0].turns]
    assert delays[1] == expected


def test_multi_turn_loader_logs_cap_summary(
    tmp_path: Path,
    cap_run,
    caplog,
) -> None:
    rows = [
        {
            "session_id": "s1",
            "turns": [
                {"text": "a", "delay": 5_000},
                {"text": "b", "delay": 4_000},
                {"text": "c", "delay": 500},
            ],
        }
    ]
    path = _write_jsonl(tmp_path, "mt.jsonl", rows)
    loader = MultiTurnDatasetLoader(filename=str(path), run=cap_run)
    data = loader.load_dataset()
    with caplog.at_level(logging.INFO, logger="aiperf"):
        loader.convert_to_conversations(data)
    assert any("Capped 2 inter-turn" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    "delay_in, cap_seconds, expected",
    [
        (5_000, 1.0, 1000.0),
        (500, 1.0, 500.0),
        (1_000, 1.0, 1000.0),
        (1_000_000_000, None, 1_000_000_000.0),
        (5_000, 0.0, 0.0),
    ],
)
def test_dag_jsonl_loader_clamps_inter_turn_delay(
    tmp_path: Path,
    delay_in: int,
    cap_seconds: float | None,
    expected: float,
) -> None:
    cfg = make_weka_run(inter_turn_delay_cap_seconds=cap_seconds)
    row = {
        "session_id": "s1",
        "turns": [
            {"messages": [{"role": "user", "content": "hi"}], "delay": delay_in},
        ],
    }
    path = _write_jsonl(tmp_path, "dag.jsonl", [row])
    loader = DagJsonlLoader(filename=str(path), run=cfg)
    data = loader.load_dataset()
    convs = loader.convert_to_conversations(data)
    assert convs[0].turns[0].delay == expected


def test_dag_jsonl_loader_logs_cap_summary(
    tmp_path: Path,
    cap_run,
    caplog,
) -> None:
    rows = [
        {
            "session_id": "s1",
            "turns": [
                {"messages": [{"role": "user", "content": "a"}], "delay": 5_000},
                {"messages": [{"role": "user", "content": "b"}], "delay": 4_000},
                {"messages": [{"role": "user", "content": "c"}], "delay": 500},
            ],
        }
    ]
    path = _write_jsonl(tmp_path, "dag.jsonl", rows)
    loader = DagJsonlLoader(filename=str(path), run=cap_run)
    with caplog.at_level(logging.INFO, logger="aiperf"):
        data = loader.load_dataset()
        loader.convert_to_conversations(data)
    assert any("Capped 2 inter-turn" in r.message for r in caplog.records)


def test_base_trace_loader_logs_cap_summary(
    tmp_path: Path,
    cap_run,
    prompt_generator_factory,
    caplog,
) -> None:
    rows = [
        {"session_id": "s1", "input_length": 5, "output_length": 5},
        {"session_id": "s1", "delay": 5_000, "input_length": 5, "output_length": 5},
        {"session_id": "s1", "delay": 4_000, "input_length": 5, "output_length": 5},
    ]
    path = _write_jsonl(tmp_path, "mc.jsonl", rows)
    loader = MooncakeTraceDatasetLoader(
        filename=str(path),
        prompt_generator=prompt_generator_factory(),
        run=cap_run,
    )
    data = loader.load_dataset()
    with caplog.at_level(logging.INFO, logger="aiperf"):
        loader.convert_to_conversations(data)
    assert any("Capped 2 inter-turn" in r.message for r in caplog.records)
