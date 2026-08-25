# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for job-detail state helpers."""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.ui.node_utils import run_node

UI_ROOT = Path(__file__).resolve().parents[3] / "src" / "aiperf" / "operator" / "ui"
JOB_DETAIL_STATE_PATH = UI_ROOT / "pages" / "job-detail-state.js"


def test_archived_phase_is_terminal_and_not_live() -> None:
    script = f"""
        import {{ deriveJobRunState }} from {JOB_DETAIL_STATE_PATH.as_uri()!r};
        const state = deriveJobRunState({{
          phase: 'Archived',
          epoch: '1779050863',
          runEpoch: null,
        }});
        console.log(JSON.stringify(state));
    """

    state = json.loads(run_node(script))

    assert state == {
        "phaseLower": "archived",
        "isRunning": False,
        "isCompleted": False,
        "isCancelled": False,
        "isPartiallyFailed": False,
        "isArchived": True,
        "isTerminal": True,
        "viewingCurrentRun": False,
        "pollingDone": True,
        "showLiveRunPanels": False,
    }


def test_job_detail_uses_named_freshness_and_terminal_stop() -> None:
    source = (UI_ROOT / "pages" / "job-detail.js").read_text(encoding="utf-8")

    assert (
        "import { FreshnessPill, StaleBanner } from '../components/freshness.js';"
        in source
    )
    assert "freshness.value['job-detail']" in source
    assert "source: 'job-detail'" in source
    assert "stopFreshness('terminal')" in source
    assert '<${StaleBanner} source=${jobFreshness} label="Job detail" />' in source


def test_job_detail_clears_route_identity_before_first_poll() -> None:
    source = (UI_ROOT / "pages" / "job-detail.js").read_text(encoding="utf-8")

    catch_block = source.split("} catch (err) {", 1)[1].split("throw err;", 1)[0]
    effect_block = source.split("useEffect(() => {", 3)[3].split("poll(", 1)[0]

    assert "clearFreshnessSource('job-detail')" in source
    assert "setJob(null)" in effect_block
    assert "let firstLoadDone = false" in effect_block
    assert "let firstLoadDone = job != null" not in source
    assert "firstLoadDone = true" in source
    assert "if (!firstLoadDone)" in catch_block
    assert "setError(err?.message ?? String(err))" in catch_block
    assert catch_block.index("if (!firstLoadDone)") < catch_block.index(
        "setError(err?.message ?? String(err))"
    )


def test_job_detail_empty_response_uses_first_load_guard_and_throws() -> None:
    source = (UI_ROOT / "pages" / "job-detail.js").read_text(encoding="utf-8")

    empty_block = source.split("if (data == null) {", 1)[1].split("setJob(data);", 1)[0]

    assert "const emptyError = new Error('Empty response from operator')" in empty_block
    assert "if (!firstLoadDone)" in empty_block
    assert "setError(emptyError.message)" in empty_block
    assert "throw emptyError" in empty_block
    assert empty_block.index("if (!firstLoadDone)") < empty_block.index(
        "setError(emptyError.message)"
    )
    assert empty_block.index("setError(emptyError.message)") < empty_block.index(
        "throw emptyError"
    )


def test_job_detail_poll_effect_uses_stopped_guard_for_route_cleanup() -> None:
    source = (UI_ROOT / "pages" / "job-detail.js").read_text(encoding="utf-8")

    # Anchored on the effect's own dependency list rather than on its ordinal
    # position: the page has since grown a websocket effect ahead of this one,
    # and counting `useEffect(` occurrences silently sliced the wrong block.
    effect_end = source.index("}, [namespace, name, epoch, resultsBase]);")
    effect_block = source[
        source.rindex("useEffect(() => {", 0, effect_end) : effect_end
    ]
    poll_setup_block = effect_block.split("// Fetch job config", 1)[0]
    poll_body = poll_setup_block.split("async ({ stopFreshness }) => {", 1)[1].split(
        "},\n      3000,", 1
    )[0]
    cleanup_block = effect_block.split("return () => {", 1)[1].split("};", 1)[0]
    catch_block = poll_body.split("} catch (err) {", 1)[1].split("throw err;", 1)[0]
    after_get_job = poll_body.split(
        "data = await api.getJob(namespace, name, epoch);", 1
    )[1]

    assert "let stopped = false" in poll_setup_block
    assert "stopped = true" in cleanup_block
    assert "if (stopped) return" in after_get_job
    assert "if (stopped) return" in catch_block
    assert after_get_job.index("if (stopped) return") < after_get_job.index(
        "setJob(data)"
    )
    assert catch_block.index("if (stopped) return") < catch_block.index(
        "setError(err?.message ?? String(err))"
    )


def test_job_detail_artifact_fetches_ignore_route_cleanup() -> None:
    source = (UI_ROOT / "pages" / "job-detail.js").read_text(encoding="utf-8")

    artifacts_block = source.split("fetch(resultsBase, { signal: ac.signal })", 1)[
        1
    ].split("return () => {", 1)[0]
    files_then = artifacts_block.split(".then(d => {", 1)[1].split(
        "const fileList = d?.files ?? [];", 1
    )[0]
    files_catch = artifacts_block.rsplit(".catch(() => {", 1)[1].split("});", 1)[0]

    assert "if (ac.signal.aborted) return" in files_then
    assert "if (ac.signal.aborted) return" in files_catch
    assert files_then.index("if (ac.signal.aborted) return") < files_then.index(
        "if (!d) {"
    )
    assert files_catch.index("if (ac.signal.aborted) return") < files_catch.index(
        "setFilesLoaded(true)"
    )
