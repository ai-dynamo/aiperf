# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A tag that looks like a docs-e2e tag must actually be one.

``MarkdownParser`` only recognizes tags ending in ``endpoint-server``. A tag
that nearly matches -- ``...-endpoint-server-asr`` -- is silently ignored, so
the guide reads as covered in review and is never executed. That is strictly
worse than leaving it untagged, because nothing reports it missing.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TAG_LIKE = re.compile(r"<!--\s*/?((?:setup|setup-file|health-check|aiperf-run)-\S+)")
VALID_SUFFIX = "endpoint-server"


def test_every_tag_like_comment_is_a_valid_tag() -> None:
    broken: list[str] = []
    for doc in sorted((REPO / "docs").rglob("*.md")):
        for lineno, line in enumerate(
            doc.read_text(encoding="utf-8").splitlines(), start=1
        ):
            match = TAG_LIKE.search(line)
            if match and not match.group(1).rstrip("-").endswith(VALID_SUFFIX):
                rel = doc.relative_to(REPO)
                broken.append(f"{rel}:{lineno}: {match.group(1)}")

    assert not broken, (
        "docs-e2e tags that the parser silently ignores, so the guide is "
        "never executed:\n  " + "\n  ".join(broken)
    )
