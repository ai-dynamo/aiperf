# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge-case tests for operator UI global state and poll error handling."""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.ui.node_utils import run_node

UI_ROOT = Path(__file__).resolve().parents[3] / "src" / "aiperf" / "operator" / "ui"
STATE_PATH = UI_ROOT / "lib" / "state.js"
API_PATH = UI_ROOT / "lib" / "api.js"
APP_PATH = UI_ROOT / "app.js"


def _state_import_script() -> str:
    return f"""
        import {{ readFileSync }} from 'node:fs';
        let stateSource = readFileSync({str(STATE_PATH)!r}, 'utf8');
        stateSource = stateSource.replace(
          "import {{ signal, computed }} from '@preact/signals';",
          "const signal = (value) => ({{ value }}); const computed = (fn) => ({{ get value() {{ return fn(); }} }});",
        );
        const stateUrl = 'data:text/javascript;base64,' + Buffer.from(stateSource).toString('base64');
        const state = await import(stateUrl);
    """


def _api_import_script() -> str:
    return f"""
        {_state_import_script()}
        let apiSource = readFileSync({str(API_PATH)!r}, 'utf8');
        apiSource = apiSource.replace(
          "import {{ setError }} from './state.js';",
          `import {{ setError }} from '${{stateUrl}}';`,
        );
        const apiUrl = 'data:text/javascript;base64,' + Buffer.from(apiSource).toString('base64');
        const apiModule = await import(apiUrl);
    """


def test_global_state_defaults_and_error_set_clear_are_explicit() -> None:
    script = f"""
        {_state_import_script()}
        const initial = {{
          jobs: state.jobs.value,
          sweeps: state.sweeps.value,
          selectedJob: state.selectedJob.value,
          clusterInfo: state.clusterInfo.value,
          globalError: state.globalError.value,
          loading: state.loading.value,
          jobsById: state.jobsById.value,
          runningJobs: state.runningJobs.value,
          completedJobs: state.completedJobs.value,
          failedJobs: state.failedJobs.value,
        }};
        state.setError('operator unavailable');
        const setMessage = state.globalError.value;
        state.setError(null);
        state.setLoading('jobs', true);
        state.setLoading('cluster', false);
        console.log(JSON.stringify({{
          initial,
          setMessage,
          cleared: state.globalError.value,
          loading: state.loading.value,
        }}));
    """

    assert json.loads(run_node(script)) == {
        "initial": {
            "jobs": [],
            "sweeps": [],
            "selectedJob": None,
            "clusterInfo": None,
            "globalError": None,
            "loading": {"jobs": False, "cluster": False, "leaderboard": False, "history": False},
            "jobsById": {},
            "runningJobs": [],
            "completedJobs": [],
            "failedJobs": [],
        },
        "setMessage": "operator unavailable",
        "cleared": None,
        "loading": {"jobs": True, "cluster": False, "leaderboard": False, "history": False},
    }


def test_jobs_and_sweeps_signals_keep_independent_defaults_and_phase_buckets() -> None:
    script = f"""
        {_state_import_script()}
        state.sweeps.value = [{{ namespace: 'ns', name: 'sweep-a', phase: 'Running' }}];
        state.jobs.value = [
          {{ namespace: 'prod', name: 'run-a', phase: 'Running' }},
          {{ name: 'run-b', phase: 'Initializing' }},
          {{ namespace: 'prod', name: 'run-c', phase: 'Completed' }},
          {{ namespace: 'prod', name: 'run-d', phase: 'Succeeded' }},
          {{ namespace: 'prod', name: 'run-e', phase: 'Failed' }},
          {{ namespace: 'prod', name: 'run-f', phase: 'Error' }},
          {{ namespace: 'prod', name: 'run-g', phase: 'Cancelled' }},
        ];
        console.log(JSON.stringify({{
          sweeps: state.sweeps.value,
          jobKeys: Object.keys(state.jobsById.value).sort(),
          running: state.runningJobs.value.map((j) => j.name),
          completed: state.completedJobs.value.map((j) => j.name),
          failed: state.failedJobs.value.map((j) => j.name),
        }}));
    """

    assert json.loads(run_node(script)) == {
        "sweeps": [{"namespace": "ns", "name": "sweep-a", "phase": "Running"}],
        "jobKeys": [
            "default/run-b",
            "prod/run-a",
            "prod/run-c",
            "prod/run-d",
            "prod/run-e",
            "prod/run-f",
            "prod/run-g",
        ],
        "running": ["run-a", "run-b"],
        "completed": ["run-c", "run-d"],
        "failed": ["run-e", "run-f", "run-g"],
    }


def test_app_shell_renders_global_error_banner_from_state_signal() -> None:
    source = APP_PATH.read_text()

    assert "import { globalError } from './lib/state.js';" in source
    assert "const error = globalError.value;" in source
    assert "${error && html`" in source
    assert '<div class="error-banner">' in source
    assert "<strong>Error:</strong> ${error}" in source


def test_poll_sets_error_only_after_threshold_and_clears_after_recovery() -> None:
    script = f"""
        {_api_import_script()}
        const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
        async function waitFor(predicate) {{
          for (let i = 0; i < 100; i += 1) {{
            if (predicate()) return;
            await wait(2);
          }}
          throw new Error('condition timed out');
        }}

        let calls = 0;
        const ac = new AbortController();
        apiModule.poll(async () => {{
          calls += 1;
          if (calls <= 2) throw new Error('down');
          await wait(20);
        }}, 1, ac.signal);

        await waitFor(() => calls === 1);
        const afterFirstFailure = state.globalError.value;
        await waitFor(() => calls === 3);
        const afterThreshold = state.globalError.value;
        await waitFor(() => state.globalError.value === null);
        ac.abort();
        console.log(JSON.stringify({{ afterFirstFailure, afterThreshold, afterRecovery: state.globalError.value }}));
    """

    assert json.loads(run_node(script)) == {
        "afterFirstFailure": None,
        "afterThreshold": "Operator API unreachable — live data is paused. Retrying…",
        "afterRecovery": None,
    }


def test_poll_unhealthy_slots_survive_other_pollers_and_clear_on_navigation_abort() -> None:
    script = f"""
        {_api_import_script()}
        const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
        async function waitFor(predicate) {{
          for (let i = 0; i < 100; i += 1) {{
            if (predicate()) return;
            await wait(2);
          }}
          throw new Error('condition timed out');
        }}

        const first = new AbortController();
        const second = new AbortController();
        let firstCalls = 0;
        let secondCalls = 0;
        apiModule.poll(async () => {{ firstCalls += 1; throw new Error('first down'); }}, 1, first.signal);
        apiModule.poll(async () => {{ secondCalls += 1; throw new Error('second down'); }}, 1, second.signal);

        await waitFor(() => firstCalls >= 2 && secondCalls >= 2 && state.globalError.value !== null);
        first.abort();
        await wait(0);
        const afterFirstNavigation = state.globalError.value;
        second.abort();
        await wait(0);
        console.log(JSON.stringify({{
          afterFirstNavigation,
          afterSecondNavigation: state.globalError.value,
        }}));
    """

    assert json.loads(run_node(script)) == {
        "afterFirstNavigation": "Operator API unreachable — live data is paused. Retrying…",
        "afterSecondNavigation": None,
    }
