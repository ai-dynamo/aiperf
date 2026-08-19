# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for operator UI API helpers."""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.ui.node_utils import run_node

_API_JS_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "lib"
    / "api.js"
)


def _run_api_script(body: str) -> object:
    script = f"""
        import fs from 'node:fs';

        const sourcePath = {_API_JS_PATH.as_posix()!r};
        const source = fs.readFileSync(sourcePath, 'utf8').replace(
          /import \{{[\s\S]*?\}} from '\.\/state\.js';/,
          'function setError(_) {{}}',
        );
        const moduleUrl = `data:text/javascript;base64,${{Buffer.from(source).toString('base64')}}`;
        const {{ api }} = await import(moduleUrl);

        {body}
    """
    return json.loads(run_node(script))


def test_path_segments_encode_slashes_dots_spaces_and_unicode() -> None:
    result = _run_api_script(
        """
        const calls = [];
        globalThis.fetch = async (url, opts = {}) => {
          calls.push({url, accept: opts.headers?.Accept ?? null});
          return {
            ok: true,
            status: 200,
            json: async () => ({ok: true}),
            text: async () => 'ok',
            headers: {get: () => null},
          };
        };

        const ns = 'team/../space.δ';
        const job = 'bench.name/phase..Ω';
        const epoch = '2026.05/iter..β';
        await api.getJob(ns, job, epoch);
        await api.fetchRunSummary(ns, job, epoch);
        await api.fetchRunRequests(ns, job, epoch);
        console.log(JSON.stringify(calls));
        """
    )

    resource = "team%2F..%2Fspace.%CE%B4/bench.name%2Fphase..%CE%A9"
    epoch_query = "epoch=2026.05%2Fiter..%CE%B2"
    epoch_segment = "2026.05%2Fiter..%CE%B2"
    assert result == [
        {
            "url": f"/api/v1/jobs/{resource}?{epoch_query}",
            "accept": None,
        },
        {
            "url": (f"/api/v1/results/{resource}/runs/{epoch_segment}/profile_export"),
            "accept": None,
        },
        {
            "url": (
                f"/api/v1/results/{resource}/runs/{epoch_segment}/profile_export.jsonl"
            ),
            "accept": "application/x-ndjson, text/plain",
        },
    ]


def test_artifact_filenames_with_parent_components_remain_single_url_segment() -> None:
    result = _run_api_script(
        """
        console.log(JSON.stringify({
          file: api.sweepArtifactFileUrl(
            'ns/α',
            'sweep/β',
            'epoch/γ',
            '../nested/../../secret.json',
          ),
          bundle: api.resultBundleUrl('ns/α', '../job/β', '../epoch/γ'),
        }));
        """
    )

    assert result == {
        "file": (
            "/api/v1/sweeps/ns%2F%CE%B1/sweep%2F%CE%B2/epochs/epoch%2F%CE%B3"
            "/artifacts/../nested/../../secret.json"
        ),
        "bundle": "/api/v1/results/ns%2F%CE%B1/..%2Fjob%2F%CE%B2/runs/..%2Fepoch%2F%CE%B3.zip",
    }


def test_log_query_params_encode_injection_characters() -> None:
    result = _run_api_script(
        """
        const calls = [];
        globalThis.fetch = async (url, opts = {}) => {
          calls.push({url, accept: opts.headers.Accept});
          return {ok: true, status: 200, text: async () => 'logs'};
        };

        await api.getJobLogs('ns/α', 'job/β', {
          pod: 'pod&follow=1?x=/../\\nnext',
          container: 'main&tail_lines=999#frag',
          follow: true,
          tailLines: '0&follow=0',
        });
        await api.getSweepLogs('ns/α', 'sweep/β', {
          pod: 'controller=1&container=bad',
          container: 'sidecar/../logs?tail=all',
          tailLines: -1,
        });

        console.log(JSON.stringify(calls));
        """
    )

    assert result == [
        {
            "url": (
                "/api/v1/jobs/ns%2F%CE%B1/job%2F%CE%B2/logs?"
                "pod=pod%26follow%3D1%3Fx%3D%2F..%2F%0Anext"
                "&container=main%26tail_lines%3D999%23frag"
                "&follow=1&tail_lines=0%26follow%3D0"
            ),
            "accept": "text/plain",
        },
        {
            "url": (
                "/api/v1/sweeps/ns%2F%CE%B1/sweep%2F%CE%B2/logs?"
                "pod=controller%3D1%26container%3Dbad"
                "&container=sidecar%2F..%2Flogs%3Ftail%3Dall&tail_lines=-1"
            ),
            "accept": "text/plain",
        },
    ]


def test_fetch_error_body_rejection_falls_back_to_status_text() -> None:
    result = _run_api_script(
        """
        const responses = [
          {
            ok: false,
            status: 500,
            statusText: 'Internal Server Error',
            text: async () => { throw new TypeError('stream already read'); },
          },
          {
            ok: false,
            status: 418,
            statusText: "I'm a teapot",
            text: async () => { throw 'not an Error'; },
          },
        ];
        globalThis.fetch = async () => responses.shift();

        const messages = [];
        for (const call of [() => api.listJobs(), () => api.getJobLogs('ns', 'job', {})]) {
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
        "API 500: Internal Server Error",
        "API 418: I'm a teapot",
    ]


def test_invalid_json_responses_are_not_silently_accepted() -> None:
    result = _run_api_script(
        """
        globalThis.fetch = async () => ({
          ok: true,
          status: 200,
          json: async () => { throw new SyntaxError('Unexpected token < in JSON'); },
          text: async () => '<html>not json</html>',
          headers: {get: () => null},
        });

        try {
          await api.getCluster();
        } catch (err) {
          console.log(JSON.stringify({name: err.name, message: err.message}));
        }
        """
    )

    assert result == {
        "name": "SyntaxError",
        "message": "Unexpected token < in JSON",
    }


def test_aborted_fetches_propagate_or_return_skipped_by_api_contract() -> None:
    result = _run_api_script(
        """
        const abortError = new DOMException('user cancelled', 'AbortError');
        const calls = [];
        globalThis.fetch = async (url) => {
          calls.push(url);
          throw abortError;
        };

        const outcomes = [];
        try {
          await api.fetchRunSummary('ns', 'job', 'epoch');
        } catch (err) {
          outcomes.push({method: 'summary', name: err.name, message: err.message});
        }
        const requests = await api.fetchRunRequests('ns', 'job', 'epoch');
        outcomes.push({method: 'requests', skipped: requests.skipped, records: requests.records});
        try {
          await api.getJobLogs('ns', 'job', {signal: new AbortController().signal});
        } catch (err) {
          outcomes.push({method: 'logs', name: err.name, message: err.message});
        }

        console.log(JSON.stringify({calls, outcomes}));
        """
    )

    assert result == {
        "calls": [
            "/api/v1/results/ns/job/runs/epoch/profile_export",
            "/api/v1/results/ns/job/runs/epoch/profile_export.jsonl",
            "/api/v1/jobs/ns/job/logs?",
        ],
        "outcomes": [
            {"method": "summary", "name": "AbortError", "message": "user cancelled"},
            {
                "method": "requests",
                "skipped": "fetch failed: user cancelled",
                "records": [],
            },
            {"method": "logs", "name": "AbortError", "message": "user cancelled"},
        ],
    }


def test_compare_jobs_keeps_repeated_params_but_slash_ids_are_identity_ambiguous() -> (
    None
):
    result = _run_api_script(
        """
        const calls = [];
        globalThis.fetch = async (url, opts = {}) => {
          calls.push({url, body: opts.body ?? null});
          return {ok: true, status: 200, json: async () => ({ok: true})};
        };

        const logicallyDifferentSelections = [
          ['team/a', 'job/b'],
          ['team', 'a/job/b'],
        ];
        const ids = logicallyDifferentSelections.map(([ns, name]) => `${ns}/${name}`);
        await api.compareJobs(ids);
        const parsed = new URL(`http://example.invalid${calls[0].url}`)
          .searchParams
          .getAll('jobs');
        console.log(JSON.stringify({ids, parsed, calls}));
        """
    )

    assert result == {
        "ids": ["team/a/job/b", "team/a/job/b"],
        "parsed": ["team/a/job/b", "team/a/job/b"],
        "calls": [
            {
                "url": "/api/v1/analytics/compare?jobs=team%2Fa%2Fjob%2Fb&jobs=team%2Fa%2Fjob%2Fb",
                "body": None,
            }
        ],
    }
