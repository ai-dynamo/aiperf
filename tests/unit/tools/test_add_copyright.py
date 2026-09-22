# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SPDX header generation across every enforced source format."""

from pathlib import Path

import pytest

from tools import add_copyright
from tools import check_spdx_headers as checker

LICENSE_TEXT = (Path(__file__).parents[3] / "tools/COPYRIGHT").read_text().strip()


@pytest.mark.parametrize(
    "filename",
    [
        *(f"example{suffix}" for suffix in sorted(checker.SOURCE_SUFFIXES)),
        *sorted(checker.SOURCE_FILENAMES),
    ],
)
def test_fixer_generates_every_required_header(tmp_path: Path, filename: str) -> None:
    """Every file classified by the checker has a compatible fixer handler."""
    path = tmp_path / filename
    path.write_text("content\n", encoding="utf-8")

    changed, status = add_copyright.process_file(path, LICENSE_TEXT)

    assert (changed, status) == (True, "added copyright")
    assert checker.validate_file(tmp_path, Path(filename)) == []


def test_fixer_preserves_mdc_frontmatter(tmp_path: Path) -> None:
    """Cursor rule metadata remains the first construct in an MDC file."""
    path = tmp_path / "rule.mdc"
    path.write_text(
        "---\ndescription: Example rule\nalwaysApply: true\n---\n\n# Rule\n",
        encoding="utf-8",
    )

    changed, status = add_copyright.process_file(path, LICENSE_TEXT)

    assert (changed, status) == (True, "added copyright")
    assert path.read_text(encoding="utf-8").startswith("---\n")
    assert checker.validate_file(tmp_path, Path("rule.mdc")) == []


def test_fixer_normalizes_malformed_nvidia_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fixer repairs casing and a reversed year range in an NVIDIA header."""
    path = tmp_path / "example.py"
    path.write_text(
        "# SPDX-FileCopyrightText: Copyright (c) 2026-2025 Nvidia Corporation "
        "& affiliates. All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(add_copyright, "was_modified_this_year", lambda _: True)
    changed, status = add_copyright.process_file(path, LICENSE_TEXT)

    assert (changed, status) == (True, "updated year")
    assert checker.validate_file(tmp_path, Path("example.py")) == []
