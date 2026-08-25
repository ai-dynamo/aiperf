# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for the FFmpeg source and attribution contract."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_ffmpeg_source_and_attribution_lock_the_security_release() -> None:
    """The shipped source archive and attribution name FFmpeg 8.1.2."""
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    attributions = (ROOT / "ATTRIBUTIONS-container.md").read_text(encoding="utf-8")

    version_match = re.search(r"^ARG FFMPEG_VERSION=(\S+)$", dockerfile, re.MULTILINE)
    assert version_match is not None
    version = version_match.group(1)
    assert version == "8.1.2"
    assert f"- **Version**: {version}" in attributions
    assert f"ffmpeg-{version}.tar.xz" in attributions
