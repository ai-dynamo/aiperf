#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate Task 2's literal candidate inventory from the pinned base tree."""

import subprocess
import sys
from pathlib import Path

import blake3

BASE = "057d116850cd059bcfa8e259c1e929e913e6ef07"
ROOT = Path(sys.argv[1]).resolve()
IS_CHECK_MODE = "--check" in sys.argv[2:]
RUST = ROOT / "rust"

rows = []


def source(path, package, owner, classification="implementation_leaf", is_planned=False):
    path = Path(path).as_posix()
    if path.startswith("runtime/src/"):
        candidate = f"plugins/{package}/src/{path.removeprefix('runtime/src/')}"
    elif path.startswith("runtime/tests/"):
        candidate = f"plugins/{package}/tests/baseline/{path.removeprefix('runtime/tests/')}"
    elif path.startswith("dry-run-tests/tests/"):
        candidate = f"plugins/{package}/tests/baseline/{path.removeprefix('dry-run-tests/tests/')}"
    else:
        raise ValueError(path)
    rows.append((path, candidate, owner, classification, is_planned))


def leaves(package, owner, paths):
    for path in paths:
        source(path, package, owner)


def tree(package, owner, prefix, classification="implementation_leaf"):
    output = subprocess.check_output(
        ["git", "ls-tree", "-r", "-z", BASE, "--", f"rust/{prefix}"], cwd=ROOT
    )
    for record in output.split(b"\0"):
        if not record:
            continue
        metadata, path = record.split(b"\t", 1)
        mode = metadata.split()[0]
        if mode not in {b"100644", b"100755"}:
            continue
        source(path.decode().removeprefix("rust/"), package, owner, classification)


leaves("export-basic", 24, [
    "runtime/src/export/accuracy_csv.rs", "runtime/src/export/accuracy_csv/tests.rs",
    "runtime/src/export/analysis_html.rs", "runtime/src/export/analysis_txt.rs",
    "runtime/src/export/console_txt.rs", "runtime/src/export/console_txt/cell_widths.rs",
    "runtime/src/export/console_txt/tests.rs", "runtime/src/export/dataset_analysis.rs",
    "runtime/src/export/genai_perf.rs", "runtime/src/export/genai_perf/tests.rs",
    "runtime/src/export/server_metrics/mod.rs", "runtime/src/export/server_metrics/tests.rs",
    "runtime/src/export/timeslice.rs", "runtime/src/export/timeslice/tests.rs",
])
tree("export-basic", 24, "runtime/src/export/console_txt/golden", "asset")
leaves("export-parquet", 25, [
    "runtime/src/export/parquet.rs", "runtime/src/export/parquet/tests.rs",
    "runtime/src/export/parquet/units.rs", "runtime/src/export/parquet_util.rs",
    "runtime/src/export/per_record_parquet.rs",
])
leaves("export-mlflow", 26, ["runtime/src/export/mlflow.rs", "runtime/src/export/mlflow/tests.rs"])
leaves("export-wandb", 27, [
    "runtime/src/export/wandb/mod.rs", "runtime/src/export/wandb/datastore.rs",
    "runtime/src/export/wandb/proto.rs", "runtime/src/export/wandb/tests.rs",
])
leaves("export-otel", 28, ["runtime/src/export/otel.rs", "runtime/src/export/otel/tests.rs"])
leaves("endpoints", 29, [
    "runtime/src/endpoints/anthropic.rs", "runtime/src/endpoints/chat.rs",
    "runtime/src/endpoints/chat_chunk.rs", "runtime/src/endpoints/extraction.rs",
    "runtime/src/endpoints/implementation.rs", "runtime/src/endpoints/kserve.rs",
    "runtime/src/endpoints/riva.rs", "runtime/src/endpoints/sagemaker.rs",
    "runtime/src/endpoints/spec_decode.rs", "runtime/src/endpoints/tier2.rs",
    "runtime/src/endpoints/tier2/flexible.rs", "runtime/src/endpoints/usage.rs",
    "runtime/src/endpoints/vllm_generate.rs",
])
source("runtime/src/endpoints/mod.rs", "endpoints", 29, "facade")
leaves("endpoints", 29, [
    "runtime/tests/endpoints_anthropic_messages.rs", "runtime/tests/endpoints_endpoints.rs",
    "runtime/tests/endpoints_kserve.rs", "runtime/tests/endpoints_registry.rs",
    "runtime/tests/endpoints_riva.rs", "runtime/tests/endpoints_tier2.rs",
    "runtime/tests/endpoints_vllm_generate.rs", "runtime/tests/tier2_endpoints_online.rs",
])
tree("transport-http", 31, "runtime/src/transport/http")
tree("transport-http", 31, "runtime/tests/transport_http")
for path in [
    "runtime/src/transport/grpc/kserve_binding.rs", "runtime/src/transport/grpc/proto.rs",
    "runtime/src/transport/grpc/riva_binding.rs", "runtime/src/transport/grpc/riva_codec.rs",
    "runtime/src/transport/grpc/riva_proto.rs",
]:
    source(path, "endpoints", 30, is_planned=path.endswith("kserve_binding.rs"))
leaves("transport-grpc", 32, [
    "runtime/src/transport/grpc/codec.rs", "runtime/src/transport/grpc/models.rs",
    "runtime/src/transport/grpc/raw_codec.rs", "runtime/src/transport/grpc/sink.rs",
    "runtime/src/transport/grpc/transport.rs",
])
source("runtime/tests/proto/grpc_predict_v2.proto", "transport-grpc", 32, "asset")
source("runtime/src/transport/grpc/mod.rs", "transport-grpc", 32, "facade")
leaves("transport-grpc", 32, [
    "runtime/tests/transport_grpc_codec.rs", "runtime/tests/transport_grpc_riva.rs",
    "runtime/tests/transport_grpc_riva_transport.rs", "runtime/tests/transport_grpc_transport.rs",
])
for path in [
    "runtime/src/transport/ws.rs", "runtime/src/transport/ws/connector.rs",
    "runtime/src/transport/ws/dialect.rs", "runtime/src/transport/ws/driver.rs",
    "runtime/src/transport/ws/sink.rs",
]:
    source(path, "transport-websocket", 33, is_planned=path.endswith("ws/sink.rs"))
source("runtime/tests/websocket_transport_config.rs", "transport-websocket", 33)
source("runtime/src/transport/dry_run.rs", "transport-dry-run", 33, is_planned=True)
leaves("transport-dry-run", 33, [
    "dry-run-tests/tests/common/mod.rs", "dry-run-tests/tests/component_packages.rs",
    "dry-run-tests/tests/dry_run.rs", "dry-run-tests/tests/random_pool_batches.rs",
    "dry-run-tests/tests/timing.rs", "dry-run-tests/tests/timing_extended.rs",
    "dry-run-tests/tests/tracelab.rs", "dry-run-tests/tests/virtual_workers.rs",
])
source("runtime/src/dynosim/direct.rs", "transport-dynosim", 34, is_planned=True)
source("runtime/src/endpoints/dynosim.rs", "transport-dynosim", 34)

if len(rows) != 126:
    raise SystemExit(f"expected 126 rows, generated {len(rows)}")
if len({row[0] for row in rows}) != len(rows):
    raise SystemExit("duplicate source paths")
if len({row[1] for row in rows}) != len(rows):
    raise SystemExit("duplicate candidate paths")

output = [
    "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
    "# SPDX-License-Identifier: Apache-2.0",
    "# Generated mechanically from the immutable Task-1 base tree; do not edit individual rows.",
    f'base_commit = "{BASE}"',
    "",
]
for path, candidate, owner, classification, is_planned in rows:
    output.extend([
        "[[source]]",
        f'source_path = "{path}"',
        f'candidate_path = "{candidate}"',
        f"owner_task = {owner}",
        f'classification = "{classification}"',
        f'state = "{("planned" if is_planned else "present")}"',
    ])
    if is_planned:
        source_path = ROOT / "rust" / path
        if source_path.exists():
            raise SystemExit(f"planned source exists in worktree: {path}")
        base_probe = subprocess.run(
            ["git", "cat-file", "-e", f"{BASE}:rust/{path}"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if base_probe.returncode == 0:
            raise SystemExit(f"planned source exists in pinned base: {path}")
        if base_probe.returncode != 128:
            raise SystemExit(f"cannot inspect planned source in pinned base: {path}")
        output.append("producer_task = 6")
    else:
        source_path = ROOT / "rust" / path
        expected_blob = subprocess.check_output(["git", "show", f"{BASE}:rust/{path}"], cwd=ROOT)
        actual_blob = source_path.read_bytes()
        if actual_blob != expected_blob:
            raise SystemExit(f"worktree differs from pinned base: {path}")
        output.append(f'blake3 = "{blake3.blake3(actual_blob).hexdigest()}"')
    output.append("")

destination = RUST / "plugin-conformance/candidate-source-inventory.toml"
rendered = "\n".join(output)
if IS_CHECK_MODE:
    if destination.read_text() != rendered:
        raise SystemExit("candidate inventory is not the generator fixed point")
else:
    destination.write_text(rendered)
