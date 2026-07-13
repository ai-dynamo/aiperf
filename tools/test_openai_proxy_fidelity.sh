#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture_root="$(mktemp -d "${TMPDIR:-/tmp}/aiperf-openai-proxy-fidelity.XXXXXX")"
trap 'rm -rf "${fixture_root}"' EXIT
nemo_root="${fixture_root}/providers/nemo"
openbench_root="${fixture_root}/providers/openbench"
harness_root="${fixture_root}/harness"

env UV_PROJECT_ENVIRONMENT="${nemo_root}" \
  uv sync --frozen --no-dev --python 3.12.10 \
    --project "${repo_root}/tools/stock_evaluators/nemo"
env UV_PROJECT_ENVIRONMENT="${openbench_root}" \
  uv sync --frozen --no-dev --python 3.12.10 \
    --project "${repo_root}/tools/stock_evaluators/openbench"

uv venv --python 3.12.10 "${harness_root}"
uv pip install \
  --python "${harness_root}/bin/python" \
  'blake3==1.0.9' \
  'orjson==3.11.9' \
  'packaging==26.2' \
  'pytest==9.1.1' \
  'pytest-asyncio==1.4.0'

cd "${repo_root}"
env PYTHONPATH="${repo_root}/src" \
  "${harness_root}/bin/python" tools/generate_stock_evaluator_manifest.py \
    --check \
    --nemo-root "${nemo_root}" \
    --openbench-root "${openbench_root}"

env \
  -u RUSTC_WRAPPER \
  AIPERF_PINNED_OPENAI_PYTHON="${openbench_root}/bin/python" \
  cargo test -p aiperf --lib \
  'evaluation::proxy::tests::pinned_openai_sdk_uses_uds_and_keeps_one_faithful_terminal_envelope' \
  -- --exact --nocapture

env -u RUSTC_WRAPPER cargo build -p aiperf-runner

env \
  -u RUSTC_WRAPPER \
  AIPERF_EVALUATOR_PROVIDER_ROOTS="${nemo_root}:${openbench_root}" \
  cargo test -p aiperf-runner --test evaluation_process \
    both_stock_providers_execute_through_rust_owned_http_and_sse \
    -- --exact --ignored --nocapture

env \
  PYTHONPATH="${repo_root}/src:${openbench_root}/lib/python3.12/site-packages" \
  "${harness_root}/bin/python" -c '
import sys
from pathlib import Path

from aiperf.orchestrator.runner_installation import RunnerInstallation

installation = RunnerInstallation.resolve(
    Path(sys.argv[1]),
    provider_roots=(Path(sys.argv[3]), Path(sys.argv[4])),
)
assert ["online_http", "evaluation"] in installation.capabilities["supported_pairs"]
combinations = installation.capabilities["supported_evaluation_combinations"]
assert any(item["provider"] == "nemo_evaluator" for item in combinations)
assert any(item["provider"] == "openbench" for item in combinations)
request = {
    "protocol_version": 2,
    "operation": "validate",
    "expected_distribution_id": installation.distribution_id,
    "run": {
        "identity": {"benchmark_id": "openbench-product-validation", "label": "validation", "trial": 0},
        "artifact_target": str(Path(sys.argv[2]) / "validation-artifacts"),
        "backend": {"type": "online_http", "config": {}},
        "workload": {
            "type": "evaluation",
            "config": {
                "provider": {"type": "openbench", "distribution": "groq_openbench_0_5_3_inspect_0_3_141_locked"},
                "evaluation": {"task": "gsm8k", "task_args": {}, "epochs": 1, "limit": 1},
                "routes": {"candidate": {"model": "candidate", "endpoint_profile": "default", "purpose": "primary"}},
                "resources": {},
                "unit_concurrency": 1,
            },
        },
        "resources": {
            "models": {"items": [{"name": "candidate"}], "strategy": "round_robin"},
            "endpoints": {"profiles": [{
                "id": "default", "type": "chat", "urls": ["http://127.0.0.1:9"],
                "streaming": True, "headers": {}, "extra": {},
                "ssl_verify": True, "http2": False, "use_legacy_max_tokens": False,
                "use_server_token_count": False, "download_video_content": False,
                "timeout_seconds": 21600.0,
                "connection_limit": 2500, "connection_reuse": "pooled",
                "keepalive_timeout": 300.0, "wait_for_model_timeout": 0.0,
                "wait_for_model_interval": 5.0, "wait_for_model_mode": "inference",
            }]},
            "metrics": {"slos": {}},
            "artifacts": {"records_path": "profile_export.jsonl", "trace": False},
        },
    },
}
response = installation.validate_authored_request(
    request, benchmark_id="openbench-product-validation"
)
assert response["success"] is True
' "${repo_root}/target/debug/aiperf-runner" "${fixture_root}" \
  "${nemo_root}" "${openbench_root}"

env \
  PYTHONPATH="${repo_root}/src" \
  AIPERF_REQUIRE_STOCK_PROVIDER_PROOF=1 \
  AIPERF_TEST_NEMO_PROVIDER_ROOT="${nemo_root}" \
  AIPERF_TEST_OPENBENCH_PROVIDER_ROOT="${openbench_root}" \
  "${harness_root}/bin/python" -m pytest -q \
  --noconftest \
  'tests/unit/accuracy/test_evaluation_provider_worker_v2.py::test_stock_provider_over_dedicated_fds'
