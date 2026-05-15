# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component integration coverage for verbatim payload replay dataset modes."""

from pathlib import Path

import orjson
import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI


def _write_json(path: Path, data: dict) -> None:
    path.write_bytes(orjson.dumps(data))


@pytest.mark.component_integration
class TestRawPayloadReplay:
    def test_raw_payload_cli_replays_authored_body_verbatim(
        self, cli: AIPerfCLI, tmp_path: Path
    ) -> None:
        payload = {
            "messages": [{"role": "user", "content": "raw-payload body"}],
            "model": defaults.model,
            "stream": False,
            "max_tokens": 7,
            "temperature": 0.01,
            "vendor_flag": {"preserve": True},
        }
        input_file = tmp_path / "payloads.jsonl"
        input_file.write_bytes(orjson.dumps(payload) + b"\n")

        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --custom-dataset-type raw_payload \
                --input-file {input_file} \
                --concurrency 1 \
                --num-conversations 1 \
                --workers-max {defaults.workers_max} \
                --export-level raw \
                --ui {defaults.ui}
            """,
            timeout=60.0,
        )

        assert result.raw_records is not None
        assert len(result.raw_records) == 1
        assert result.raw_records[0].payload == payload

    def test_inputs_json_cli_replays_stored_payload_verbatim(
        self, cli: AIPerfCLI, tmp_path: Path
    ) -> None:
        payload = {
            "messages": [{"role": "user", "content": "inputs-json body"}],
            "model": defaults.model,
            "stream": False,
            "max_tokens": 9,
            "temperature": 0.02,
            "vendor_flag": {"preserve": "inputs"},
        }
        input_file = tmp_path / "inputs.json"
        _write_json(
            input_file,
            {
                "data": [
                    {
                        "session_id": "session-raw-replay",
                        "payloads": [payload],
                    }
                ]
            },
        )

        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --custom-dataset-type inputs_json \
                --input-file {input_file} \
                --concurrency 1 \
                --num-conversations 1 \
                --workers-max {defaults.workers_max} \
                --export-level raw \
                --ui {defaults.ui}
            """,
            timeout=60.0,
        )

        assert result.raw_records is not None
        assert len(result.raw_records) == 1
        assert result.raw_records[0].payload == payload
