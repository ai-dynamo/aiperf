# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavior edge tests for operator UI API helpers."""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.ui.node_utils import VISIBLE_DOCUMENT_STUB_JS, run_node

_API_JS_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "lib"
    / "api.js"
)


def _run_api_script(body: str, set_error: str = "function setError(_) {}") -> object:
    script = f"""
        import fs from 'node:fs';
        {VISIBLE_DOCUMENT_STUB_JS}

        const sourcePath = {_API_JS_PATH.as_posix()!r};
        const stateImport = `import {{
  clearFreshnessSource,
  markFreshnessAttempt,
  markFreshnessFailure,
  markFreshnessStopped,
  markFreshnessSuccess,
  setError,
}} from './state.js';`;
        const sourceText = fs.readFileSync(sourcePath, 'utf8');
        const source = sourceText.includes(stateImport)
          ? sourceText.replace(stateImport, {set_error!r})
          : sourceText.replace("import {{ setError }} from './state.js';", {set_error!r});
        const moduleUrl = `data:text/javascript;base64,${{Buffer.from(source).toString('base64')}}`;
        const {{ api, poll }} = await import(moduleUrl);

        {body}
    """
    return json.loads(run_node(script))


def test_api_fetch_surfaces_error_body_and_status_text_fallback() -> None:
    result = _run_api_script(
        """
        const responses = [
          {
            ok: false,
            status: 422,
            statusText: 'Unprocessable Entity',
            text: async () => JSON.stringify({detail: 'bad manifest'}),
          },
          {
            ok: false,
            status: 503,
            statusText: 'Service Unavailable',
            text: async () => { throw new Error('body stream failed'); },
          },
        ];
        globalThis.fetch = async () => responses.shift();

        const messages = [];
        for (const call of [() => api.listJobs(), () => api.getCluster()]) {
          try {
            await call();
          } catch (err) {
            messages.push(err.message);
          }
        }

        console.log(JSON.stringify(messages));
        """
    )

    assert result == [
        'API 422: {"detail":"bad manifest"}',
        "API 503: Service Unavailable",
    ]


def test_api_fetch_returns_null_for_204_without_reading_json_body() -> None:
    result = _run_api_script(
        """
        const calls = [];
        globalThis.fetch = async (url, opts = {}) => {
          calls.push({url, method: opts.method});
          return {
            ok: true,
            status: 204,
            text: async () => { throw new Error('text should not be read'); },
            json: async () => { throw new Error('json should not be read'); },
          };
        };

        const value = await api.getCluster();
        console.log(JSON.stringify({value, calls}));
        """
    )

    assert result == {
        "value": None,
        "calls": [
            {
                "url": "/api/v1/cluster",
            }
        ],
    }


def test_poll_threshold_shared_unhealthy_state_and_abort_cleanup() -> None:
    result = _run_api_script(
        """
        const timers = [];
        globalThis.setTimeout = (cb, ms) => {
          const timer = {cb, ms, cleared: false};
          timers.push(timer);
          return timer;
        };
        globalThis.clearTimeout = (timer) => { timer.cleared = true; };
        const flush = async () => {
          await new Promise((resolve) => queueMicrotask(resolve));
          await new Promise((resolve) => queueMicrotask(resolve));
        };

        let modeA = 'fail';
        let modeB = 'fail';
        const abortA = new AbortController();
        const abortB = new AbortController();
        const calls = {a: 0, b: 0};

        poll(async () => {
          calls.a += 1;
          if (modeA === 'fail') throw new Error('a down');
        }, 123, abortA.signal);
        poll(async () => {
          calls.b += 1;
          if (modeB === 'fail') throw new Error('b down');
        }, 456, abortB.signal);
        await flush();
        const afterFirstFailures = globalThis.errorEvents.slice();

        await timers[0].cb();
        await flush();
        const afterAThreshold = globalThis.errorEvents.slice();

        await timers[1].cb();
        await flush();
        const afterBThreshold = globalThis.errorEvents.slice();

        modeA = 'ok';
        await timers[2].cb();
        await flush();
        const afterAHealthy = globalThis.errorEvents.slice();

        abortB.abort();
        const afterBAbort = globalThis.errorEvents.slice();

        console.log(JSON.stringify({
          calls,
          timerIntervals: timers.map((timer) => timer.ms),
          afterFirstFailures,
          afterAThreshold,
          afterBThreshold,
          afterAHealthy,
          afterBAbort,
          bTimerCleared: timers[3].cleared,
        }));
        """,
        set_error=(
            "globalThis.errorEvents = []; "
            "function setError(message) { globalThis.errorEvents.push(message); }"
        ),
    )

    # The banner now carries the failing poller's error so a 403 and a dead
    # pod are not the same sentence; each poller stamps its own detail.
    prefix = "Operator API unreachable — live data is paused. Retrying…"
    unreachable_a = f"{prefix} (last error: a down)"
    unreachable_b = f"{prefix} (last error: b down)"
    assert result == {
        "calls": {"a": 3, "b": 2},
        "timerIntervals": [123, 456, 123, 456, 123],
        "afterFirstFailures": [],
        "afterAThreshold": [unreachable_a],
        "afterBThreshold": [unreachable_a, unreachable_b],
        "afterAHealthy": [unreachable_a, unreachable_b],
        "afterBAbort": [unreachable_a, unreachable_b, None],
        "bTimerCleared": True,
    }


def test_poll_aborted_signal_prevents_initial_tick() -> None:
    result = _run_api_script(
        """
        let calls = 0;
        let timers = 0;
        globalThis.setTimeout = () => { timers += 1; };
        const abort = new AbortController();
        abort.abort();

        poll(async () => { calls += 1; }, 50, abort.signal);
        await new Promise((resolve) => queueMicrotask(resolve));

        console.log(JSON.stringify({calls, timers}));
        """
    )

    assert result == {"calls": 0, "timers": 0}


def test_poll_ignores_rejection_after_abort_cleanup() -> None:
    result = _run_api_script(
        """
        const timers = [];
        globalThis.setTimeout = (cb, ms) => {
          timers.push({cb, ms});
          return timers[timers.length - 1];
        };
        globalThis.clearTimeout = () => {};
        const flush = async () => {
          await new Promise((resolve) => queueMicrotask(resolve));
          await new Promise((resolve) => queueMicrotask(resolve));
        };
        const events = [];
        globalThis.markFreshnessAttempt = (source, intervalMs, at) => events.push(['attempt', source, intervalMs, typeof at]);
        globalThis.markFreshnessSuccess = (source, at) => events.push(['success', source, typeof at]);
        globalThis.markFreshnessFailure = (source, error, at, retrying) => events.push(['failure', source, error, typeof at, retrying]);
        globalThis.markFreshnessStopped = (source, reason, at) => events.push(['stopped', source, reason, typeof at]);
        globalThis.clearFreshnessSource = (source) => events.push(['clear', source]);

        let rejectInFlight;
        const inFlight = new Promise((_resolve, reject) => { rejectInFlight = reject; });
        const abort = new AbortController();
        poll(async () => {
          abort.abort();
          await inFlight;
        }, 5000, abort.signal, {source: 'jobs'});
        await flush();
        rejectInFlight(new Error('late abort failure'));
        await flush();

        console.log(JSON.stringify({events, errorEvents: globalThis.errorEvents, timers: timers.length}));
        """,
        set_error=(
            "globalThis.errorEvents = []; "
            "function setError(message) { globalThis.errorEvents.push(message); } "
            "function markFreshnessAttempt(...args) { globalThis.markFreshnessAttempt(...args); } "
            "function markFreshnessSuccess(...args) { globalThis.markFreshnessSuccess(...args); } "
            "function markFreshnessFailure(...args) { globalThis.markFreshnessFailure(...args); } "
            "function markFreshnessStopped(...args) { globalThis.markFreshnessStopped(...args); } "
            "function clearFreshnessSource(...args) { globalThis.clearFreshnessSource(...args); }"
        ),
    )

    assert result == {
        "events": [["attempt", "jobs", 5000, "number"], ["clear", "jobs"]],
        "errorEvents": [],
        "timers": 0,
    }


def test_poll_named_source_records_freshness_attempt_success_retry_and_abort() -> None:
    result = _run_api_script(
        """
        const timers = [];
        globalThis.setTimeout = (cb, ms) => {
          const timer = {cb, ms, cleared: false};
          timers.push(timer);
          return timer;
        };
        globalThis.clearTimeout = (timer) => { timer.cleared = true; };
        const flush = async () => {
          await new Promise((resolve) => queueMicrotask(resolve));
          await new Promise((resolve) => queueMicrotask(resolve));
        };

        const events = [];
        globalThis.markFreshnessAttempt = (source, intervalMs, at) => events.push(['attempt', source, intervalMs, typeof at]);
        globalThis.markFreshnessSuccess = (source, at) => events.push(['success', source, typeof at]);
        globalThis.markFreshnessFailure = (source, error, at, retrying) => events.push(['failure', source, error, typeof at, retrying]);
        globalThis.markFreshnessStopped = (source, reason, at) => events.push(['stopped', source, reason, typeof at]);
        globalThis.clearFreshnessSource = (source) => events.push(['clear', source]);

        let mode = 'fail';
        const abort = new AbortController();
        poll(async () => {
          if (mode === 'fail') throw new Error('jobs down');
        }, 5000, abort.signal, {source: 'jobs'});
        await flush();
        await timers[0].cb();
        await flush();
        mode = 'ok';
        await timers[1].cb();
        await flush();
        abort.abort();

        console.log(JSON.stringify({events, timerIntervals: timers.map(t => t.ms)}));
        """,
        set_error=(
            "globalThis.errorEvents = []; "
            "function setError(message) { globalThis.errorEvents.push(message); } "
            "function markFreshnessAttempt(...args) { globalThis.markFreshnessAttempt(...args); } "
            "function markFreshnessSuccess(...args) { globalThis.markFreshnessSuccess(...args); } "
            "function markFreshnessFailure(...args) { globalThis.markFreshnessFailure(...args); } "
            "function markFreshnessStopped(...args) { globalThis.markFreshnessStopped(...args); } "
            "function clearFreshnessSource(...args) { globalThis.clearFreshnessSource(...args); }"
        ),
    )

    assert result == {
        "events": [
            ["attempt", "jobs", 5000, "number"],
            ["failure", "jobs", "jobs down", "number", False],
            ["attempt", "jobs", 5000, "number"],
            ["failure", "jobs", "jobs down", "number", True],
            ["attempt", "jobs", 5000, "number"],
            ["success", "jobs", "number"],
            ["clear", "jobs"],
        ],
        "timerIntervals": [5000, 5000, 5000],
    }


def test_poll_can_mark_named_source_stopped_from_callback() -> None:
    result = _run_api_script(
        """
        const timers = [];
        globalThis.setTimeout = (cb, ms) => {
          const timer = {cb, ms, cleared: false};
          timers.push(timer);
          return timer;
        };
        const flush = async () => {
          await new Promise((resolve) => queueMicrotask(resolve));
          await new Promise((resolve) => queueMicrotask(resolve));
        };
        const events = [];
        globalThis.markFreshnessAttempt = (source, intervalMs, at) => events.push(['attempt', source, intervalMs, typeof at]);
        globalThis.markFreshnessSuccess = (source, at) => events.push(['success', source, typeof at]);
        globalThis.markFreshnessFailure = (source, error, at, retrying) => events.push(['failure', source, error, typeof at, retrying]);
        globalThis.markFreshnessStopped = (source, reason, at) => events.push(['stopped', source, reason, typeof at]);
        globalThis.clearFreshnessSource = (source) => events.push(['clear', source]);

        const abort = new AbortController();
        poll(async ({stopFreshness}) => {
          stopFreshness('terminal');
          abort.abort();
        }, 5000, abort.signal, {source: 'job-detail'});
        await flush();

        console.log(JSON.stringify({events, timers: timers.length}));
        """,
        set_error=(
            "function setError(_) {} "
            "function markFreshnessAttempt(...args) { globalThis.markFreshnessAttempt(...args); } "
            "function markFreshnessSuccess(...args) { globalThis.markFreshnessSuccess(...args); } "
            "function markFreshnessFailure(...args) { globalThis.markFreshnessFailure(...args); } "
            "function markFreshnessStopped(...args) { globalThis.markFreshnessStopped(...args); } "
            "function clearFreshnessSource(...args) { globalThis.clearFreshnessSource(...args); }"
        ),
    )

    assert result == {
        "events": [
            ["attempt", "job-detail", 5000, "number"],
            ["stopped", "job-detail", "terminal", "number"],
        ],
        "timers": 0,
    }


def test_poll_stopped_success_clears_prior_global_unhealthy_state() -> None:
    result = _run_api_script(
        """
        const timers = [];
        globalThis.setTimeout = (cb, ms) => {
          const timer = {cb, ms, cleared: false};
          timers.push(timer);
          return timer;
        };
        globalThis.clearTimeout = (timer) => { timer.cleared = true; };
        const flush = async () => {
          await new Promise((resolve) => queueMicrotask(resolve));
          await new Promise((resolve) => queueMicrotask(resolve));
        };

        const events = [];
        globalThis.markFreshnessAttempt = (source, intervalMs, at) => events.push(['attempt', source, intervalMs, typeof at]);
        globalThis.markFreshnessSuccess = (source, at) => events.push(['success', source, typeof at]);
        globalThis.markFreshnessFailure = (source, error, at, retrying) => events.push(['failure', source, error, typeof at, retrying]);
        globalThis.markFreshnessStopped = (source, reason, at) => events.push(['stopped', source, reason, typeof at]);
        globalThis.clearFreshnessSource = (source) => events.push(['clear', source]);

        let mode = 'fail';
        let calls = 0;
        const abort = new AbortController();
        poll(async ({stopFreshness}) => {
          calls += 1;
          if (mode === 'fail') throw new Error('jobs down');
          stopFreshness('terminal');
        }, 5000, abort.signal, {source: 'jobs'});
        await flush();
        await timers[0].cb();
        await flush();
        mode = 'ok';
        await timers[1].cb();
        await flush();

        console.log(JSON.stringify({events, errorEvents: globalThis.errorEvents, calls, timers: timers.length}));
        """,
        set_error=(
            "globalThis.errorEvents = []; "
            "function setError(message) { globalThis.errorEvents.push(message); } "
            "function markFreshnessAttempt(...args) { globalThis.markFreshnessAttempt(...args); } "
            "function markFreshnessSuccess(...args) { globalThis.markFreshnessSuccess(...args); } "
            "function markFreshnessFailure(...args) { globalThis.markFreshnessFailure(...args); } "
            "function markFreshnessStopped(...args) { globalThis.markFreshnessStopped(...args); } "
            "function clearFreshnessSource(...args) { globalThis.clearFreshnessSource(...args); }"
        ),
    )

    unreachable = (
        "Operator API unreachable — live data is paused. Retrying… "
        "(last error: jobs down)"
    )
    assert result == {
        "events": [
            ["attempt", "jobs", 5000, "number"],
            ["failure", "jobs", "jobs down", "number", False],
            ["attempt", "jobs", 5000, "number"],
            ["failure", "jobs", "jobs down", "number", True],
            ["attempt", "jobs", 5000, "number"],
            ["stopped", "jobs", "terminal", "number"],
        ],
        "errorEvents": [unreachable, None],
        "calls": 3,
        "timers": 3,
    }


def test_poll_stop_freshness_after_external_abort_does_not_recreate_source() -> None:
    result = _run_api_script(
        """
        const timers = [];
        globalThis.setTimeout = (cb, ms) => {
          const timer = {cb, ms, cleared: false};
          timers.push(timer);
          return timer;
        };
        globalThis.clearTimeout = (timer) => { timer.cleared = true; };
        const flush = async () => {
          await new Promise((resolve) => queueMicrotask(resolve));
          await new Promise((resolve) => queueMicrotask(resolve));
        };

        const events = [];
        globalThis.markFreshnessAttempt = (source, intervalMs, at) => events.push(['attempt', source, intervalMs, typeof at]);
        globalThis.markFreshnessSuccess = (source, at) => events.push(['success', source, typeof at]);
        globalThis.markFreshnessFailure = (source, error, at, retrying) => events.push(['failure', source, error, typeof at, retrying]);
        globalThis.markFreshnessStopped = (source, reason, at) => events.push(['stopped', source, reason, typeof at]);
        globalThis.clearFreshnessSource = (source) => events.push(['clear', source]);

        let capturedStopFreshness;
        const inFlight = new Promise(() => {});
        const abort = new AbortController();
        poll(async ({stopFreshness}) => {
          capturedStopFreshness = stopFreshness;
          await inFlight;
        }, 5000, abort.signal, {source: 'jobs'});
        await flush();
        abort.abort();
        const afterAbort = events.slice();
        capturedStopFreshness('terminal');
        await flush();

        console.log(JSON.stringify({events, afterAbort, timers: timers.length}));
        """,
        set_error=(
            "function setError(_) {} "
            "function markFreshnessAttempt(...args) { globalThis.markFreshnessAttempt(...args); } "
            "function markFreshnessSuccess(...args) { globalThis.markFreshnessSuccess(...args); } "
            "function markFreshnessFailure(...args) { globalThis.markFreshnessFailure(...args); } "
            "function markFreshnessStopped(...args) { globalThis.markFreshnessStopped(...args); } "
            "function clearFreshnessSource(...args) { globalThis.clearFreshnessSource(...args); }"
        ),
    )

    assert result == {
        "events": [
            ["attempt", "jobs", 5000, "number"],
            ["clear", "jobs"],
        ],
        "afterAbort": [
            ["attempt", "jobs", 5000, "number"],
            ["clear", "jobs"],
        ],
        "timers": 0,
    }


def test_compare_jobs_uses_repeated_query_params_without_request_body() -> None:
    result = _run_api_script(
        """
        const calls = [];
        globalThis.fetch = async (url, opts = {}) => {
          calls.push({url, method: opts.method ?? null, body: opts.body ?? null});
          return {
            ok: true,
            status: 200,
            json: async () => ({ok: true}),
            text: async () => 'ok',
          };
        };

        await api.compareJobs(['default/bench-a', 'team space/bench?b', 'α/β']);
        console.log(JSON.stringify(calls));
        """
    )

    assert result == [
        {
            "url": (
                "/api/v1/analytics/compare?jobs=default%2Fbench-a"
                "&jobs=team+space%2Fbench%3Fb&jobs=%CE%B1%2F%CE%B2"
            ),
            "method": None,
            "body": None,
        }
    ]


def test_logs_urls_handle_empty_false_and_zero_query_options() -> None:
    result = _run_api_script(
        """
        const calls = [];
        globalThis.fetch = async (url, opts = {}) => {
          calls.push({url, accept: opts.headers.Accept});
          return {
            ok: true,
            status: 200,
            text: async () => 'logs',
          };
        };

        await api.getJobLogs('team/ns', 'bench/name', {});
        await api.getSweepLogs('team/ns', 'sweep/name', {
          pod: '',
          container: '',
          follow: false,
          tailLines: 0,
        });

        console.log(JSON.stringify(calls));
        """
    )

    assert result == [
        {
            "url": "/api/v1/jobs/team%2Fns/bench%2Fname/logs?",
            "accept": "text/plain",
        },
        {
            "url": "/api/v1/sweeps/team%2Fns/sweep%2Fname/logs?tail_lines=0",
            "accept": "text/plain",
        },
    ]


def test_log_fetch_passes_abort_signal_and_propagates_abort_error() -> None:
    result = _run_api_script(
        """
        const abort = new AbortController();
        abort.abort();
        let signalWasForwarded = false;
        globalThis.fetch = async (_url, opts = {}) => {
          signalWasForwarded = opts.signal === abort.signal;
          if (opts.signal.aborted) {
            throw new DOMException('aborted by test', 'AbortError');
          }
          throw new Error('expected aborted signal');
        };

        try {
          await api.getJobLogs('default', 'bench', {signal: abort.signal});
        } catch (err) {
          console.log(JSON.stringify({signalWasForwarded, name: err.name, message: err.message}));
        }
        """
    )

    assert result == {
        "signalWasForwarded": True,
        "name": "AbortError",
        "message": "aborted by test",
    }
