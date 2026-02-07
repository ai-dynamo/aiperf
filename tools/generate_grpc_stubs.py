#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate Python gRPC stubs from the KServe V2 proto definition.

Usage:
    uv run python tools/generate_grpc_stubs.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROTO_DIR = REPO_ROOT / "src" / "aiperf" / "transports" / "grpc" / "proto"
PROTO_FILE = PROTO_DIR / "grpc_predict_v2.proto"


def main() -> None:
    """Generate protobuf and gRPC stubs."""
    if not PROTO_FILE.exists():
        print(f"Proto file not found: {PROTO_FILE}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"--proto_path={PROTO_DIR}",
        f"--python_out={PROTO_DIR}",
        f"--grpc_python_out={PROTO_DIR}",
        str(PROTO_FILE),
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print("Failed to generate stubs", file=sys.stderr)
        sys.exit(result.returncode)

    # Fix the import in the generated gRPC stub to use relative import
    grpc_stub = PROTO_DIR / "grpc_predict_v2_pb2_grpc.py"
    if grpc_stub.exists():
        content = grpc_stub.read_text()
        content = content.replace(
            "import grpc_predict_v2_pb2 as grpc__predict__v2__pb2",
            "from . import grpc_predict_v2_pb2 as grpc__predict__v2__pb2",
        )
        grpc_stub.write_text(content)
        print(f"Fixed relative import in {grpc_stub}")

    print("Stubs generated successfully.")


if __name__ == "__main__":
    main()
