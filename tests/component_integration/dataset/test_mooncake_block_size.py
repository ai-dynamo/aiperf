# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component integration coverage for the trace block-size override (AIP-1016).

``--isl-block-size`` overrides a trace loader's ``default_block_size`` plugin
metadata. This restores behavior regressed in the v2 config transition: a
Mooncake-format trace recorded at a block size other than the loader default
(512) could not be replayed, because the hash-based prompt reconstruction
raises ``ConfigurationError`` when the recorded ``input_length`` is not
consistent with ``(len(hash_ids) - 1) * block_size``.

This is the end-to-end proof that the override reaches the real reconstruction
path through the full CLI. The precise "16 succeeds / default 512 raises
ConfigurationError" behavior is covered faster at the loader level in
tests/unit/dataset/loader/test_trace.py.
"""

import json
from pathlib import Path

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI

# input_length 48 with 3 hash blocks is consistent with block_size 16
# (final block = 48 - 2*16 = 16, in (0, 16]) but not with the default 512
# (final block = 48 - 2*512 < 0 -> ConfigurationError).
_TRACE_ROWS = [
    {"input_length": 48, "output_length": 8, "hash_ids": [1, 2, 3], "timestamp": 100},
    {"input_length": 48, "output_length": 8, "hash_ids": [4, 5, 6], "timestamp": 200},
]
_RECORDED_BLOCK_SIZE = 16


def _write_mooncake(path: Path) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in _TRACE_ROWS) + "\n")
    return path


@pytest.mark.component_integration
class TestMooncakeTraceBlockSize:
    def test_block_size_override_replays_trace(
        self, cli: AIPerfCLI, tmp_path: Path
    ) -> None:
        """With --isl-block-size matching the recording, the trace replays and
        produces one record per line (real reconstruction, no error)."""
        input_file = _write_mooncake(tmp_path / "mooncake.jsonl")

        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --endpoint-type chat \
                --custom-dataset-type mooncake_trace \
                --input-file {input_file} \
                --isl-block-size {_RECORDED_BLOCK_SIZE} \
                --request-count {len(_TRACE_ROWS)} \
                --concurrency 1 \
                --workers-max {defaults.workers_max} \
                --export-level raw \
                --ui {defaults.ui}
            """,
            timeout=60.0,
        )

        assert result.raw_records is not None
        assert len(result.raw_records) == len(_TRACE_ROWS)
