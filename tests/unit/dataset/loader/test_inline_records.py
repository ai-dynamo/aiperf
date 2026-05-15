# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper + parity tests for inline-records loader path."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from aiperf.dataset.loader.base_loader import BaseFileLoader
from aiperf.dataset.loader.mooncake_trace import MooncakeTraceDatasetLoader
from aiperf.dataset.loader.multi_turn import MultiTurnDatasetLoader
from aiperf.dataset.loader.random_pool import RandomPoolDatasetLoader
from aiperf.dataset.loader.sagemaker_data_capture import SageMakerDataCaptureLoader
from aiperf.dataset.loader.single_turn import SingleTurnDatasetLoader
from tests.unit.conftest import make_run_from_cli


class _DummyLoader(BaseFileLoader):
    """Concrete BaseFileLoader for helper-method testing."""

    def load_dataset(self):
        return {}

    def convert_to_conversations(self, custom_data):
        return []


class TestBaseFileLoaderInlineRecords:
    def test_inline_records_flat_list_iterates(self):
        records = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
        loader = _DummyLoader(inline_records=list(records))
        assert list(loader._iter_record_dicts()) == records

    def test_inline_records_multi_pool_requires_source(self):
        loader = _DummyLoader(inline_records={"pool_a": [{"text": "a"}]})
        with pytest.raises(ValueError) as exc:
            list(loader._iter_record_dicts())
        assert "source" in str(exc.value).lower()

    def test_inline_records_multi_pool_iterates_named_pool(self):
        loader = _DummyLoader(
            inline_records={"pool_a": [{"text": "a1"}], "pool_b": [{"text": "b1"}]}
        )
        assert list(loader._iter_record_dicts(source="pool_a")) == [{"text": "a1"}]
        assert list(loader._iter_record_dicts(source="pool_b")) == [{"text": "b1"}]

    def test_inline_records_multi_pool_unknown_source_raises(self):
        loader = _DummyLoader(inline_records={"pool_a": [{"text": "a"}]})
        with pytest.raises(KeyError):
            list(loader._iter_record_dicts(source="nope"))

    def test_file_records_iterates_via_path(self, tmp_path):
        f = tmp_path / "x.jsonl"
        f.write_text('{"text": "a"}\n\n{"text": "b"}\n')
        loader = _DummyLoader(filename=f)
        assert list(loader._iter_record_dicts()) == [{"text": "a"}, {"text": "b"}]

    def test_neither_filename_nor_inline_rejected(self):
        with pytest.raises(ValueError) as exc:
            _DummyLoader()
        msg = str(exc.value).lower()
        assert "filename" in msg or "inline_records" in msg

    def test_both_filename_and_inline_rejected(self, tmp_path):
        f = tmp_path / "x.jsonl"
        f.write_text('{"text": "a"}\n')
        with pytest.raises(ValueError) as exc:
            _DummyLoader(filename=f, inline_records=[{"text": "a"}])
        msg = str(exc.value).lower()
        assert "exactly one" in msg or "both" in msg

    def test_inline_records_non_dict_entries_pass_through_unchanged(self):
        """`_iter_record_dicts` is a yield helper; type validation is the caller's
        responsibility (Pydantic model_validate runs in the subclass). Verify the
        helper does not pre-filter or pre-validate.
        """
        weird_records = [{"text": "ok"}, "string entry", 42, None]
        loader = _DummyLoader(inline_records=list(weird_records))
        assert list(loader._iter_record_dicts()) == weird_records


# ---------- Parity: inline vs JSONL file ----------


@pytest.fixture
def jsonl_file(tmp_path):
    def _make(records: list[dict]) -> Path:
        f = tmp_path / "data.jsonl"
        f.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        return f

    return _make


class TestSingleTurnInlineParity:
    def test_inline_matches_file(self, jsonl_file, default_cfg):
        records = [
            {"text": "What is ML?"},
            {"text": "Explain GANs.", "output_length": 200},
            {"text": "Define AI."},
        ]
        run = make_run_from_cli(default_cfg)

        file_loader = SingleTurnDatasetLoader(filename=jsonl_file(records), run=run)
        file_data = file_loader.load_dataset()

        run2 = make_run_from_cli(default_cfg)
        inline_loader = SingleTurnDatasetLoader(inline_records=records, run=run2)
        inline_data = inline_loader.load_dataset()

        assert (
            sum(len(v) for v in file_data.values())
            == sum(len(v) for v in inline_data.values())
            == 3
        )
        file_texts = sorted(t.text for sessions in file_data.values() for t in sessions)
        inline_texts = sorted(
            t.text for sessions in inline_data.values() for t in sessions
        )
        assert file_texts == inline_texts

    def test_inline_three_records_load_correctly(self, default_cfg):
        records = [{"text": "one"}, {"text": "two"}]
        run = make_run_from_cli(default_cfg)
        loader = SingleTurnDatasetLoader(inline_records=records, run=run)
        data = loader.load_dataset()
        assert sum(len(v) for v in data.values()) == 2


class TestMultiTurnInlineParity:
    def test_inline_matches_file(self, jsonl_file, default_cfg):
        records = [
            {
                "session_id": "chat_1",
                "turns": [{"text": "Hi"}, {"text": "How are you?"}],
            },
            {
                "session_id": "chat_2",
                "turns": [
                    {"text": "Hello"},
                    {"text": "Tell me more"},
                    {"text": "Cool"},
                ],
            },
        ]
        run = make_run_from_cli(default_cfg)

        file_loader = MultiTurnDatasetLoader(filename=jsonl_file(records), run=run)
        file_data = file_loader.load_dataset()

        run2 = make_run_from_cli(default_cfg)
        inline_loader = MultiTurnDatasetLoader(inline_records=records, run=run2)
        inline_data = inline_loader.load_dataset()

        assert set(file_data.keys()) == set(inline_data.keys())
        for sid in file_data:
            file_turns = [t.text for mt in file_data[sid] for t in mt.turns]
            inline_turns = [t.text for mt in inline_data[sid] for t in mt.turns]
            assert file_turns == inline_turns


# ---------- Parity: mooncake_trace ----------


def _make_mock_prompt_generator() -> Mock:
    """Mirror the mock_prompt_generator fixture in test_trace.py."""
    generator = Mock()
    generator.generate.return_value = "Generated prompt text"
    generator._decoded_cache = {}
    generator._build_token_sequence.return_value = [1, 2, 3, 4, 5]
    # No resolved_name attribute on the inner tokenizer mock — fall through
    # to BenchmarkConfig.tokenizer / model name.
    generator.tokenizer = Mock(spec=[])
    return generator


class TestMooncakeTraceInlineParity:
    def test_inline_matches_file(self, jsonl_file, default_cfg):
        records = [
            {
                "timestamp": 0,
                "input_length": 16,
                "output_length": 8,
                "hash_ids": [1, 2],
            },
            {
                "timestamp": 100,
                "input_length": 32,
                "output_length": 16,
                "hash_ids": [3, 4],
            },
            {
                "timestamp": 250,
                "input_length": 24,
                "output_length": 12,
                "hash_ids": [1, 2],
            },
        ]

        run = make_run_from_cli(default_cfg)
        file_loader = MooncakeTraceDatasetLoader(
            filename=jsonl_file(records),
            run=run,
            prompt_generator=_make_mock_prompt_generator(),
        )
        file_data = file_loader.load_dataset()

        run2 = make_run_from_cli(default_cfg)
        inline_loader = MooncakeTraceDatasetLoader(
            inline_records=records,
            run=run2,
            prompt_generator=_make_mock_prompt_generator(),
        )
        inline_data = inline_loader.load_dataset()

        file_lengths = sorted(
            t.input_length for traces in file_data.values() for t in traces
        )
        inline_lengths = sorted(
            t.input_length for traces in inline_data.values() for t in traces
        )
        assert file_lengths == inline_lengths
        assert file_lengths == [16, 24, 32]


# ---------- Parity: random_pool ----------


class TestRandomPoolInlineParity:
    def test_inline_single_pool_matches_single_file(self, jsonl_file, default_cfg):
        records = [
            {"text": "What is ML?", "type": "random_pool"},
            {"text": "Explain GANs.", "type": "random_pool"},
            {"text": "Define AI.", "type": "random_pool"},
        ]
        run = make_run_from_cli(default_cfg)
        file_loader = RandomPoolDatasetLoader(filename=jsonl_file(records), run=run)
        file_data = file_loader.load_dataset()

        run2 = make_run_from_cli(default_cfg)
        inline_loader = RandomPoolDatasetLoader(inline_records=records, run=run2)
        inline_data = inline_loader.load_dataset()

        # Single-pool inline keys to "<inline>"; file keys to filename basename.
        assert len(file_data) == 1
        assert len(inline_data) == 1
        assert "<inline>" in inline_data
        file_pool = next(iter(file_data.values()))
        inline_pool = next(iter(inline_data.values()))
        assert len(file_pool) == len(inline_pool) == 3

    def test_inline_multi_pool_matches_directory(self, tmp_path, default_cfg):
        d = tmp_path / "pools"
        d.mkdir()
        (d / "queries.jsonl").write_text(
            '{"text": "Q1", "type": "random_pool"}\n'
            '{"text": "Q2", "type": "random_pool"}\n'
        )
        (d / "passages.jsonl").write_text(
            '{"text": "P1", "type": "random_pool"}\n'
            '{"text": "P2", "type": "random_pool"}\n'
        )
        run = make_run_from_cli(default_cfg)
        file_loader = RandomPoolDatasetLoader(filename=d, run=run)
        file_data = file_loader.load_dataset()

        inline = {
            "queries.jsonl": [
                {"text": "Q1", "type": "random_pool"},
                {"text": "Q2", "type": "random_pool"},
            ],
            "passages.jsonl": [
                {"text": "P1", "type": "random_pool"},
                {"text": "P2", "type": "random_pool"},
            ],
        }
        run2 = make_run_from_cli(default_cfg)
        inline_loader = RandomPoolDatasetLoader(inline_records=inline, run=run2)
        inline_data = inline_loader.load_dataset()

        assert (
            set(file_data.keys())
            == set(inline_data.keys())
            == {"queries.jsonl", "passages.jsonl"}
        )
        for pool_name in file_data:
            assert len(file_data[pool_name]) == len(inline_data[pool_name])


# ---------- Parity: sagemaker_data_capture ----------


def _make_sagemaker_record(
    inference_time: str,
    event_id: str,
    user_message: str,
    max_tokens: int = 50,
    prompt_tokens: int = 16,
) -> dict:
    """Build a minimal SageMaker Data Capture record (JSON-encoded payloads)."""
    input_payload = {
        "messages": [{"role": "user", "content": user_message}],
        "max_tokens": max_tokens,
    }
    output_payload = {
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 8,
            "total_tokens": prompt_tokens + 8,
        }
    }
    return {
        "captureData": {
            "endpointInput": {
                "data": json.dumps(input_payload),
                "encoding": "JSON",
            },
            "endpointOutput": {
                "data": json.dumps(output_payload),
                "encoding": "JSON",
            },
        },
        "eventMetadata": {
            "eventId": event_id,
            "inferenceTime": inference_time,
        },
        "eventVersion": "0",
    }


class TestSageMakerInlineParity:
    def test_inline_matches_file(self, jsonl_file, default_cfg):
        records = [
            _make_sagemaker_record(
                inference_time="2026-04-29T00:00:01Z",
                event_id="evt-1",
                user_message="hello one",
                prompt_tokens=12,
            ),
            _make_sagemaker_record(
                inference_time="2026-04-29T00:00:03Z",
                event_id="evt-2",
                user_message="hello two",
                prompt_tokens=20,
            ),
            _make_sagemaker_record(
                inference_time="2026-04-29T00:00:05Z",
                event_id="evt-3",
                user_message="hello three",
                prompt_tokens=24,
            ),
        ]

        run = make_run_from_cli(default_cfg)
        file_loader = SageMakerDataCaptureLoader(
            filename=jsonl_file(records),
            run=run,
            prompt_generator=_make_mock_prompt_generator(),
        )
        file_data = file_loader.load_dataset()

        run2 = make_run_from_cli(default_cfg)
        inline_loader = SageMakerDataCaptureLoader(
            inline_records=records,
            run=run2,
            prompt_generator=_make_mock_prompt_generator(),
        )
        inline_data = inline_loader.load_dataset()

        # Each captured record becomes its own session with one trace.
        assert (
            sum(len(v) for v in file_data.values())
            == sum(len(v) for v in inline_data.values())
            == 3
        )

        file_lengths = sorted(
            t.input_length for traces in file_data.values() for t in traces
        )
        inline_lengths = sorted(
            t.input_length for traces in inline_data.values() for t in traces
        )
        assert file_lengths == inline_lengths == [12, 20, 24]

        # Timestamps are zero-aligned identically for both paths.
        file_ts = sorted(t.timestamp for traces in file_data.values() for t in traces)
        inline_ts = sorted(
            t.timestamp for traces in inline_data.values() for t in traces
        )
        assert file_ts == inline_ts
        assert file_ts[0] == 0.0

    def test_inline_records_does_not_crash_on_none_filename(self, default_cfg):
        """Regression: SageMaker load_dataset must not call filename.is_dir()
        when inline_records is set (filename is None)."""
        records = [
            _make_sagemaker_record(
                inference_time="2026-04-29T00:00:01Z",
                event_id="evt-1",
                user_message="just one",
            )
        ]
        run = make_run_from_cli(default_cfg)
        loader = SageMakerDataCaptureLoader(
            inline_records=records,
            run=run,
            prompt_generator=_make_mock_prompt_generator(),
        )
        assert loader.filename is None
        # Must not raise AttributeError on None.is_dir().
        data = loader.load_dataset()
        assert sum(len(v) for v in data.values()) == 1
