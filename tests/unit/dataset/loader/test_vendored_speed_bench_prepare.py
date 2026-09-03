# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guards the vendored SPEED-Bench prepare script against local edits.

The file's value is being byte-identical to a known upstream commit: benchmark
prompts are reproducible only if the resolution code is. A local "small fix"
silently changes published numbers, so pin the hash and route fixes upstream.
"""

import hashlib
from pathlib import Path

VENDORED = (
    Path(__file__).resolve().parents[4]
    / "src/aiperf/dataset/loader/vendor/speed_bench_prepare.py"
)

UPSTREAM_COMMIT = "5ac8609a56ac941540b10c92e68d556e6343cd4c"
UPSTREAM_SHA256 = "a551be4df541474e54e21b480022b0cbb66c2da068fda61b2a64bd3223bbbed2"


def test_vendored_prepare_script_is_unmodified():
    digest = hashlib.sha256(VENDORED.read_bytes()).hexdigest()

    normalized = hashlib.sha256(
        VENDORED.read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()
    if digest != UPSTREAM_SHA256 and normalized == UPSTREAM_SHA256:
        raise AssertionError(
            f"{VENDORED.name} content is intact but its line endings were "
            "rewritten on checkout. .gitattributes marks the vendor directory "
            "`-text` to prevent exactly this; confirm it is present and that "
            "the working copy was checked out after it was added."
        )

    assert digest == UPSTREAM_SHA256, (
        f"{VENDORED.name} no longer matches upstream commit {UPSTREAM_COMMIT}. "
        "Do not edit vendored files: fix it upstream in NVIDIA-NeMo/Skills, "
        "then re-vendor and update UPSTREAM_COMMIT/UPSTREAM_SHA256 together."
    )


def test_vendored_prepare_script_retains_its_licence_header():
    header = VENDORED.read_text(encoding="utf-8")[:600]

    assert "Copyright (c) 2026, NVIDIA CORPORATION" in header
    assert "Apache License, Version 2.0" in header
