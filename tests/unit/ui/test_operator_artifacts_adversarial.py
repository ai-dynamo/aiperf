# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UI_ROOT = _REPO_ROOT / "src" / "aiperf" / "operator" / "ui"
_ARTIFACTS_CARD_JS = _UI_ROOT / "components" / "artifacts-card.js"
_API_JS = _UI_ROOT / "lib" / "api.js"
_SWEEP_DETAIL_JS = _UI_ROOT / "pages" / "sweep-detail.js"


def _source(path: Path) -> str:
    return path.read_text()


def _function_body(source: str, function_name: str) -> str:
    match = re.search(
        rf"function {re.escape(function_name)}\([^)]*\) \{{(?P<body>.*?)\n\}}",
        source,
        re.DOTALL,
    )
    assert match is not None, f"{function_name} must remain statically testable"
    return match.group("body")


def _method_body(source: str, method_name: str) -> str:
    match = re.search(
        rf"\n  {re.escape(method_name)}\([^)]*\) \{{(?P<body>.*?)\n  \}},",
        source,
        re.DOTALL,
    )
    assert match is not None, f"api.{method_name} must remain statically testable"
    return match.group("body")


def test_job_artifact_file_urls_preserve_nested_paths_and_encode_each_segment() -> None:
    body = _function_body(_source(_ARTIFACTS_CARD_JS), "resultFileUrl")

    assert "fileName.split('/').map(encodeURIComponent).join('/')" in body
    assert "encodeURIComponent(fileName)" not in body
    assert "/runs/${encodeURIComponent(epoch)}/${encodedFileName}`" in body


def test_sweep_artifact_file_urls_preserve_nested_paths_like_job_artifacts() -> None:
    body = _method_body(_source(_API_JS), "sweepArtifactFileUrl")

    assert "filename.split('/').map(encodeURIComponent).join('/')" in body
    assert "encodeURIComponent(filename)" not in body
    assert "/artifacts/${fileSeg}`" in body


def test_artifact_url_builders_never_decode_or_normalize_adversarial_names() -> None:
    combined = _source(_ARTIFACTS_CARD_JS) + "\n" + _source(_API_JS)

    for forbidden in [
        "decodeURIComponent",
        "new URL(",
        ".pathname",
        ".normalize(",
        "replace('../'",
        'replace("../"',
    ]:
        assert forbidden not in combined


def test_artifact_urls_encode_namespace_name_epoch_and_spaces_unicode_segments() -> (
    None
):
    card = _source(_ARTIFACTS_CARD_JS)
    api = _source(_API_JS)

    assert "encodeURIComponent(namespace)" in card
    assert "encodeURIComponent(name)" in card
    assert "encodeURIComponent(resolvedEpoch)" in card
    assert "encodeURIComponent(ns)" in api
    assert "encodeURIComponent(jobId)" in api
    assert "encodeURIComponent(sweepName)" in api
    assert "encodeURIComponent(epoch)" in api
    assert ".map(encodeURIComponent).join('/')" in card


def test_missing_resolved_epoch_blocks_previews_and_uses_waiting_empty_state() -> None:
    src = _source(_ARTIFACTS_CARD_JS)
    selected_empty = _function_body(src, "selectedEmptyKey")
    resolve_file_url = _function_body(src, "resolveFileUrl")

    assert "if (resolvedEpoch == null) return 'waiting';" in selected_empty
    assert "if (resolvedEpoch == null) return null;" in resolve_file_url
    assert "resultFileUrl(namespace, name, resolvedEpoch, fileName)" in resolve_file_url
    assert "resultFileUrl(namespace, name, epoch, fileName)" not in resolve_file_url
    assert "encodeURIComponent(resolvedEpoch)}/profile_export?format=json" in src
    assert "encodeURIComponent(epoch)}/profile_export?format=json" not in src


def test_preview_choice_is_extension_based_not_server_mime_based() -> None:
    src = _source(_ARTIFACTS_CARD_JS)
    modal = _function_body(src, "FileViewerModal")

    assert "const PREVIEWABLE = new Set(['json', 'csv', 'txt', 'ansi']);" in src
    assert "const ext = filename.split('.').pop().toLowerCase();" in modal
    assert "if (ext === 'json')" in modal
    assert "await response.json()" in modal
    assert "await response.text()" in modal
    assert "Content-Type" not in modal
    assert "headers.get" not in modal


def test_large_file_lists_are_not_silently_truncated_in_summary_or_rows() -> None:
    src = _source(_ARTIFACTS_CARD_JS)

    assert "`${files.length} file${files.length === 1 ? '' : 's'}`" in src
    assert "files.map(f =>" in src
    assert "files.slice(" not in src
    assert "MAX_FILES" not in src


def test_download_all_urls_prefer_single_archive_and_do_not_require_file_urls() -> None:
    src = _source(_ARTIFACTS_CARD_JS)

    assert (
        "const downloadAllUrl = bundleUrl ?? (resolvedEpoch != null && api?.resultBundleUrl ? api.resultBundleUrl(namespace, name, resolvedEpoch) : null);"
        in src
    )
    assert "${downloadAllUrl && html`" in src
    assert "href=${downloadAllUrl}" in src
    assert "Download .zip" in src
    assert "const canBuildFileUrls = fileUrl != null || resolvedEpoch != null;" in src
    assert "showIndividualDownloadAll && canBuildFileUrls" in src


def test_job_archives_and_sweep_aggregate_artifacts_use_distinct_routes() -> None:
    api = _source(_API_JS)
    sweep_detail = _source(_SWEEP_DETAIL_JS)

    job_bundle = _method_body(api, "resultBundleUrl")
    sweep_bundle = _method_body(api, "sweepArtifactBundleUrl")
    sweep_file = _method_body(api, "sweepArtifactFileUrl")

    assert (
        "`${BASE}/results/${nsSeg}/${idSeg}/runs/${encodeURIComponent(epoch)}.zip`"
        in job_bundle
    )
    assert (
        "`${BASE}/sweeps/${nsSeg}/${sweepSeg}/epochs/${epSeg}/artifacts.zip`"
        in sweep_bundle
    )
    assert (
        "`${BASE}/sweeps/${nsSeg}/${sweepSeg}/epochs/${epSeg}/artifacts/${fileSeg}`"
        in sweep_file
    )
    assert "api.sweepArtifactBundleUrl(namespace, name, resolvedEpoch)" in sweep_detail
    assert (
        "api.sweepArtifactFileUrl(namespace, name, resolvedEpoch, fileName)"
        in sweep_detail
    )
    assert 'title="Aggregate artifacts"' in sweep_detail
