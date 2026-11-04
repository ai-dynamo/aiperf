# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from choreographer.browsers import Chromium
from pytest import param

from tests.integration.test_plot_integration import browser_available_for_png_export


@pytest.mark.parametrize(
    "kind,expected",
    [
        param("missing", False, id="missing-browser"),
        param("invalid", False, id="invalid-browser-path"),
        param("directory", False, id="directory-not-browser"),
        param("file", True, id="existing-browser-file"),
    ],
)  # fmt: skip
def test_browser_available_for_png_export(
    kind: str, expected: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "browser"
    if kind == "file":
        path.touch()
    elif kind == "directory":
        path.mkdir()
    monkeypatch.setattr(
        Chromium,
        "find_browser",
        lambda **kwargs: None if kind == "missing" else str(path),
    )
    assert browser_available_for_png_export() is expected
