# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def test_channel_codecs_import_before_common_models() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from aiperf.common.channel_codecs import RAW_INFERENCE_CODEC; print(RAW_INFERENCE_CODEC.cache_key)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "raw-inference-msgpack" in result.stdout


def test_metric_record_metadata_lazy_reexport() -> None:
    from aiperf.common.models import MetricRecordMetadata

    assert MetricRecordMetadata.__name__ == "MetricRecordMetadata"
