# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-row endpoint routing in the Mooncake trace loader.

A row may name a registered endpoint plugin via ``endpoint_type``, sending that
request to a different endpoint of the same server than the run-level
``--endpoint-type``.
"""

import logging
from pathlib import Path

import orjson
import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.dataset.loader.models import MooncakeTrace
from aiperf.dataset.loader.mooncake_trace import MooncakeTraceDatasetLoader


class TestEndpointTypeValidation:
    """Validation of the endpoint_type field on MooncakeTrace."""

    def test_registered_endpoint_accepted(self):
        trace = MooncakeTrace(input_length=10, endpoint_type="embeddings")
        assert trace.endpoint_type == "embeddings"

    @pytest.mark.parametrize(
        "spelling",
        [
            param("EMBEDDINGS", id="uppercase"),
            param("Embeddings", id="mixed-case"),
        ],
    )  # fmt: skip
    def test_endpoint_type_normalized_to_registered_spelling(self, spelling: str):
        """The endpoint enum lookup is case-insensitive; the stored value is not.

        Downstream code compares this string against the run-level endpoint type,
        so it must be normalized at load time rather than at every comparison.
        """
        assert MooncakeTrace(input_length=10, endpoint_type=spelling).endpoint_type == (
            "embeddings"
        )

    def test_unknown_endpoint_rejected_with_valid_names(self):
        """An unregistered name fails at load time, listing what is valid."""
        with pytest.raises(ValidationError, match="not a registered endpoint"):
            MooncakeTrace(input_length=10, endpoint_type="embedings")

        with pytest.raises(ValidationError, match="Valid endpoints:.*embeddings"):
            MooncakeTrace(input_length=10, endpoint_type="embedings")

    def test_endpoint_type_with_session_id_rejected(self):
        """Per-row routing is single-turn only.

        Multi-turn dispatch replays conversation history on every turn, which
        endpoints like embeddings reject (they accept exactly one turn).
        """
        with pytest.raises(
            ValidationError, match="cannot be combined with 'session_id'"
        ):
            MooncakeTrace(input_length=10, endpoint_type="embeddings", session_id="s1")

    def test_endpoint_type_with_payload_rejected(self):
        """A verbatim payload row is not really sent "to" an endpoint.

        payload bypasses format_payload entirely, so naming an endpoint would
        apply its URL path while silently ignoring its request format.
        """
        with pytest.raises(ValidationError, match="cannot be combined with 'payload'"):
            MooncakeTrace(
                payload={"input": ["hi"], "model": "m"}, endpoint_type="embeddings"
            )

    def test_pathless_endpoint_rejected(self):
        """`raw` declares endpoint_path: null, so it is not a routing target.

        Routing to it would send the row to the bare base URL with no path.
        """
        with pytest.raises(ValidationError, match="declares no endpoint path"):
            MooncakeTrace(input_length=10, endpoint_type="raw")

    def test_absent_by_default(self):
        """A native Mooncake row carries no endpoint_type."""
        native = MooncakeTrace(
            timestamp=0, input_length=6755, output_length=500, hash_ids=[0, 1]
        )
        assert native.endpoint_type is None

    def test_can_load_still_accepts_native_rows(self):
        """Adding an optional field must not change loader auto-detection."""
        assert MooncakeTraceDatasetLoader.can_load(
            {
                "timestamp": 0,
                "input_length": 6755,
                "output_length": 500,
                "hash_ids": [0],
            }
        )
        assert MooncakeTraceDatasetLoader.can_load(
            {"timestamp": 0, "input_length": 300, "endpoint_type": "embeddings"}
        )


def _load_single_turn(tmp_path: Path, row: dict, default_cfg, mock_prompt_generator):
    """Write ``row`` as a one-line trace, load it, and return the built Turn."""
    file = tmp_path / "trace.jsonl"
    with open(file, "wb") as f:
        f.write(orjson.dumps(row))
        f.write(b"\n")

    loader = MooncakeTraceDatasetLoader(
        filename=file,
        cfg=default_cfg,
        prompt_generator=mock_prompt_generator,
    )
    conversations = loader.convert_to_conversations(loader.load_dataset())
    return conversations[0].turns[0]


class TestEndpointTypePropagation:
    """endpoint_type must reach the Turn in every input mode."""

    def test_propagates_in_synthesized_mode(
        self, tmp_path: Path, default_cfg, mock_prompt_generator
    ):
        turn = _load_single_turn(
            tmp_path,
            {"input_length": 10, "endpoint_type": "embeddings"},
            default_cfg,
            mock_prompt_generator,
        )
        assert turn.endpoint_type == "embeddings"

    def test_propagates_in_messages_mode(
        self, tmp_path: Path, default_cfg, mock_prompt_generator
    ):
        turn = _load_single_turn(
            tmp_path,
            {
                "messages": [{"role": "user", "content": "hi"}],
                "endpoint_type": "embeddings",
            },
            default_cfg,
            mock_prompt_generator,
        )
        assert turn.endpoint_type == "embeddings"
        assert turn.raw_messages == [{"role": "user", "content": "hi"}]

    def test_absent_leaves_turn_unrouted(
        self, tmp_path: Path, default_cfg, mock_prompt_generator
    ):
        turn = _load_single_turn(
            tmp_path,
            {"timestamp": 0, "input_length": 10, "output_length": 4},
            default_cfg,
            mock_prompt_generator,
        )
        assert turn.endpoint_type is None


class TestOutputLengthOnRoutedRows:
    """output_length is one of the four native Mooncake fields, so real traces
    carry it on every row -- including rows routed to an endpoint that generates
    no tokens. Those rows are accepted; the value is dropped rather than handed
    to an endpoint that would reject it once per request."""

    def test_dropped_for_endpoint_that_produces_no_tokens(
        self, tmp_path: Path, default_cfg, mock_prompt_generator
    ):
        turn = _load_single_turn(
            tmp_path,
            {"input_length": 10, "output_length": 40, "endpoint_type": "embeddings"},
            default_cfg,
            mock_prompt_generator,
        )
        assert turn.endpoint_type == "embeddings"
        assert turn.max_tokens is None

    def test_kept_for_endpoint_that_produces_tokens(
        self, tmp_path: Path, default_cfg, mock_prompt_generator
    ):
        """Routing alone must not discard output_length."""
        turn = _load_single_turn(
            tmp_path,
            {"input_length": 10, "output_length": 40, "endpoint_type": "completions"},
            default_cfg,
            mock_prompt_generator,
        )
        assert turn.endpoint_type == "completions"
        assert turn.max_tokens == 40

    def test_kept_on_unrouted_rows(
        self, tmp_path: Path, default_cfg, mock_prompt_generator
    ):
        turn = _load_single_turn(
            tmp_path,
            {"input_length": 10, "output_length": 40},
            default_cfg,
            mock_prompt_generator,
        )
        assert turn.max_tokens == 40

    def test_dropped_in_messages_mode(
        self, tmp_path: Path, default_cfg, mock_prompt_generator
    ):
        turn = _load_single_turn(
            tmp_path,
            {
                "messages": [{"role": "user", "content": "hi"}],
                "output_length": 40,
                "endpoint_type": "embeddings",
            },
            default_cfg,
            mock_prompt_generator,
        )
        assert turn.max_tokens is None

    def test_warns_once_with_row_count(
        self, tmp_path: Path, default_cfg, mock_prompt_generator, caplog
    ):
        """One aggregated warning per endpoint, not one message per row."""
        file = tmp_path / "trace.jsonl"
        with open(file, "wb") as f:
            for _ in range(3):
                f.write(
                    orjson.dumps(
                        {
                            "input_length": 10,
                            "output_length": 40,
                            "endpoint_type": "embeddings",
                        }
                    )
                )
                f.write(b"\n")

        loader = MooncakeTraceDatasetLoader(
            filename=file,
            cfg=default_cfg,
            prompt_generator=mock_prompt_generator,
        )
        with caplog.at_level(logging.WARNING):
            loader.load_dataset()

        warnings = [
            r.message for r in caplog.records if "output_length" in str(r.message)
        ]
        assert len(warnings) == 1
        assert "3 row(s) routed to 'embeddings'" in warnings[0]

    def test_no_warning_when_endpoint_produces_tokens(
        self, tmp_path: Path, default_cfg, mock_prompt_generator, caplog
    ):
        file = tmp_path / "trace.jsonl"
        with open(file, "wb") as f:
            f.write(
                orjson.dumps(
                    {
                        "input_length": 10,
                        "output_length": 40,
                        "endpoint_type": "completions",
                    }
                )
            )
            f.write(b"\n")

        loader = MooncakeTraceDatasetLoader(
            filename=file,
            cfg=default_cfg,
            prompt_generator=mock_prompt_generator,
        )
        with caplog.at_level(logging.WARNING):
            loader.load_dataset()

        assert not [r for r in caplog.records if "output_length" in str(r.message)]
