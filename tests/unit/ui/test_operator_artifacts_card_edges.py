# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ARTIFACTS_CARD_JS = (
    _REPO_ROOT
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "components"
    / "artifacts-card.js"
)


def _artifacts_card_source() -> str:
    return _ARTIFACTS_CARD_JS.read_text()


def _function_body(source: str, function_name: str) -> str:
    match = re.search(
        rf"function {re.escape(function_name)}\([^)]*\) \{{(?P<body>.*?)\n\}}",
        source,
        re.DOTALL,
    )
    assert match is not None, f"{function_name} must remain statically testable"
    return match.group("body")


def test_result_file_url_preserves_nested_artifact_paths_while_encoding_segments() -> (
    None
):
    """Nested artifact filenames must reach FastAPI's ``{filename:path}`` route."""
    body = _function_body(_artifacts_card_source(), "resultFileUrl")

    assert "encodeURIComponent(namespace)" in body
    assert "encodeURIComponent(name)" in body
    assert "encodeURIComponent(epoch)" in body
    assert "encodeURIComponent(fileName)" not in body
    assert ".split('/')" in body
    assert ".map(encodeURIComponent)" in body
    assert ".join('/')" in body


def test_default_quick_export_and_bundle_urls_use_resolved_epoch_not_raw_epoch() -> (
    None
):
    """Artifact actions should use the same pinned epoch as the empty-state gate."""
    src = _artifacts_card_source()

    assert (
        "const emptyKey = selectedEmptyKey({ resolvedEpoch, isCompleted, isRunning });"
        in src
    )
    assert "api.resultBundleUrl(namespace, name, resolvedEpoch)" in src
    assert "encodeURIComponent(resolvedEpoch)}/profile_export?format=json" in src
    assert "api.resultBundleUrl(namespace, name, epoch)" not in src
    assert "encodeURIComponent(epoch)}/profile_export?format=json" not in src


def test_individual_download_all_is_hidden_when_file_urls_cannot_be_built() -> None:
    """Avoid rendering a button that silently no-ops when neither epoch nor fileUrl exists."""
    src = _artifacts_card_source()

    assert "const canBuildFileUrls = fileUrl != null || resolvedEpoch != null;" in src
    assert "showIndividualDownloadAll && canBuildFileUrls" in src


def test_file_viewer_modal_has_dialog_accessibility_semantics() -> None:
    body = _function_body(_artifacts_card_source(), "ModalChrome")

    assert 'role="dialog"' in body
    assert 'aria-modal="true"' in body
    assert 'aria-labelledby="artifact-preview-title"' in body
    assert 'id="artifact-preview-title"' in body
    assert 'aria-label="Close preview"' in body
