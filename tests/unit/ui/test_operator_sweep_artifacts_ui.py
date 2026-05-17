# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_API_JS = _REPO_ROOT / "src" / "aiperf" / "operator" / "ui" / "lib" / "api.js"
_ARTIFACTS_CARD_JS = (
    _REPO_ROOT
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "components"
    / "artifacts-card.js"
)


def _source(path: Path) -> str:
    return path.read_text()


def test_api_exports_encoded_sweep_artifact_url_helpers() -> None:
    src = _source(_API_JS)

    expected_helpers = {
        "sweepArtifactListUrl": "`${BASE}/sweeps/${nsSeg}/${sweepSeg}/epochs/${epSeg}/artifacts`",
        "sweepArtifactBundleUrl": "`${BASE}/sweeps/${nsSeg}/${sweepSeg}/epochs/${epSeg}/artifacts.zip`",
        "sweepArtifactFileUrl": "`${BASE}/sweeps/${nsSeg}/${sweepSeg}/epochs/${epSeg}/artifacts/${fileSeg}`",
        "sweepProfileExportUrl": "`${BASE}/sweeps/${nsSeg}/${sweepSeg}/epochs/${epSeg}/artifacts/profile_export?format=${formatSeg}`",
    }
    for helper, expected_return in expected_helpers.items():
        assert f"{helper}(" in src, f"api must export {helper}"
        assert expected_return in src, f"{helper} must build the sweep artifact route"

    assert src.count("const nsSeg = encodeURIComponent(ns);") >= 4
    assert src.count("const sweepSeg = encodeURIComponent(sweepName);") >= 4
    assert src.count("const epSeg = encodeURIComponent(epoch);") >= 4
    assert "const fileSeg = encodeURIComponent(filename);" in src
    assert "const formatSeg = encodeURIComponent(format);" in src


def test_artifacts_card_accepts_endpoint_agnostic_props_with_job_defaults() -> None:
    src = _source(_ARTIFACTS_CARD_JS)

    signature = re.search(
        r"export function ArtifactsCard\(\{(?P<body>.*?)\}\) \{", src, re.DOTALL
    )
    assert signature is not None, "ArtifactsCard must keep destructured props"
    props = signature.group("body")

    for prop in [
        "title = 'Artifacts'",
        "testIdPrefix = 'job-detail-artifacts'",
        "bundleUrl = null",
        "quickExportUrl = null",
        "emptyMessages = null",
    ]:
        assert prop in props

    assert (
        "const downloadAllUrl = bundleUrl ?? (epoch != null ? api.resultBundleUrl(namespace, name, epoch) : null);"
        in src
    )
    assert "const resolvedQuickExportUrl = quickExportUrl ?? (epoch != null" in src
    assert "data-testid=${testIdPrefix}" in src
    assert "data-testid=${`${testIdPrefix}-download-all`}" in src
    assert "data-testid=${`${testIdPrefix}-quick-export`}" in src
    assert '<h3 class="artifacts-card__title">${title}</h3>' in src


def test_artifacts_card_empty_messages_are_overrideable_without_changing_job_text() -> (
    None
):
    src = _source(_ARTIFACTS_CARD_JS)

    assert "const defaultEmptyMessages = {" in src
    assert "waiting: 'Waiting for a run epoch before showing result files.'" in src
    assert "completed: 'No result files persisted for this run.'" in src
    assert "running: 'No result files yet.'" in src
    assert "unavailable: 'No result files available.'" in src
    assert (
        "const messages = { ...defaultEmptyMessages, ...(emptyMessages ?? {}) };" in src
    )
    assert "? messages.waiting" in src
    assert "? messages.completed" in src
    assert "? messages.running" in src
    assert ": messages.unavailable" in src


def test_sweep_detail_fetches_and_renders_aggregate_artifacts_card() -> None:
    src = _source(
        _REPO_ROOT / "src" / "aiperf" / "operator" / "ui" / "pages" / "sweep-detail.js"
    )

    assert "import { ArtifactsCard }" in src
    assert "const [artifactFiles, setArtifactFiles] = useState([]);" in src
    assert (
        "const [artifactFilesLoaded, setArtifactFilesLoaded] = useState(false);" in src
    )
    assert (
        "const latestPersistedSweepEpoch = epochs.find(e => e?.isLatest)?.epoch ?? epochs[0]?.epoch;"
        in src
    )
    assert "const resolvedEpoch = epoch" in src
    assert "?? (status.runEpoch != null ? String(status.runEpoch) : null)" in src
    assert (
        "?? (latestPersistedSweepEpoch != null ? String(latestPersistedSweepEpoch) : null);"
        in src
    )
    assert "api.sweepArtifactListUrl(namespace, name, resolvedEpoch)" in src
    assert 'title="Aggregate Artifacts"' in src
    assert 'testIdPrefix="sweep-detail-aggregate-artifacts"' in src
    assert "Waiting for a sweep epoch before showing aggregate artifacts." in src
    assert "No aggregate artifacts available for this sweep epoch." in src
    assert "No aggregate artifacts yet." in src
    assert "api.sweepArtifactBundleUrl(namespace, name, resolvedEpoch)" in src
    assert "api.sweepProfileExportUrl(namespace, name, resolvedEpoch, 'json')" in src
    assert "api.sweepArtifactFileUrl(namespace, name, resolvedEpoch, fileName)" in src
