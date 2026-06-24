# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge-case tests for operator UI API URL builders."""

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

        const calls = [];
        globalThis.fetch = async (url, opts = {{}}) => {{
          calls.push({{ url, opts }});
          return {{
            ok: true,
            status: 200,
            json: async () => ({{ ok: true }}),
            text: async () => 'ok',
            headers: {{ get: () => null }},
          }};
        }};

        {body}
    """
    return json.loads(run_node(script))


def test_job_and_sweep_fetch_urls_encode_segments_and_epoch_query() -> None:
    result = _run_api_script(
        """
        const ns = 'team space/ns';
        const name = 'bench/name α';
        const epoch = 'epoch/with space';

        await api.getJob(ns, name, epoch);
        await api.getJobConfig(ns, name, epoch);
        await api.getJobEpochs(ns, name);
        await api.getJobEvents(ns, name);
        await api.getSweep(ns, name, epoch);
        await api.getSweepEpochs(ns, name);
        await api.getSweepCells(ns, name, epoch);
        await api.getSweepChildren(ns, name, epoch);
        await api.getSweepEvents(ns, name);

        console.log(JSON.stringify(calls.map(call => call.url)));
        """
    )

    encoded_resource = "team%20space%2Fns/bench%2Fname%20%CE%B1"
    encoded_epoch_query = "epoch=epoch%2Fwith%20space"
    assert result == [
        f"/api/v1/jobs/{encoded_resource}?{encoded_epoch_query}",
        f"/api/v1/config/{encoded_resource}?{encoded_epoch_query}",
        f"/api/v1/jobs/{encoded_resource}/epochs",
        f"/api/v1/jobs/{encoded_resource}/events",
        f"/api/v1/sweeps/{encoded_resource}?{encoded_epoch_query}",
        f"/api/v1/sweeps/{encoded_resource}/epochs",
        f"/api/v1/sweeps/{encoded_resource}/cells?{encoded_epoch_query}",
        f"/api/v1/sweeps/{encoded_resource}/children?{encoded_epoch_query}",
        f"/api/v1/sweeps/{encoded_resource}/events",
    ]


def test_result_bundle_and_sweep_artifact_builders_encode_paths() -> None:
    result = _run_api_script(
        """
        const ns = 'team space/ns';
        const name = 'sweep/name β';
        const epoch = 'epoch/with space';
        const filename = 'nested dir/report file.json';
        let latestBundleError = null;
        try {
          api.resultBundleUrl(ns, name);
        } catch (err) {
          latestBundleError = err.message;
        }
        console.log(JSON.stringify({
          latestBundleError,
          pinnedBundle: api.resultBundleUrl(ns, name, epoch),
          artifactList: api.sweepArtifactListUrl(ns, name, epoch),
          artifactBundle: api.sweepArtifactBundleUrl(ns, name, epoch),
          artifactFile: api.sweepArtifactFileUrl(ns, name, epoch, filename),
          profileJsonl: api.sweepProfileExportUrl(ns, name, epoch, 'json lines/pretty'),
        }));
        """
    )

    encoded_resource = "team%20space%2Fns/sweep%2Fname%20%CE%B2"
    encoded_epoch = "epoch%2Fwith%20space"
    assert result == {
        "latestBundleError": "resultBundleUrl requires a concrete run epoch",
        "pinnedBundle": f"/api/v1/results/{encoded_resource}/runs/{encoded_epoch}.zip",
        "artifactList": f"/api/v1/sweeps/{encoded_resource}/epochs/{encoded_epoch}/artifacts",
        "artifactBundle": f"/api/v1/sweeps/{encoded_resource}/epochs/{encoded_epoch}/artifacts.zip",
        "artifactFile": (
            f"/api/v1/sweeps/{encoded_resource}/epochs/{encoded_epoch}"
            "/artifacts/nested%20dir/report%20file.json"
        ),
        "profileJsonl": (
            f"/api/v1/sweeps/{encoded_resource}/epochs/{encoded_epoch}"
            "/artifacts/profile_export?format=json%20lines%2Fpretty"
        ),
    }


def test_log_fetch_urls_encode_query_params_and_preserve_zero_tail() -> None:
    result = _run_api_script(
        """
        const ns = 'team space/ns';
        const name = 'bench/name α';

        await api.getJobLogs(ns, name, {
          pod: 'pod/name one',
          container: 'results/server',
          follow: true,
          tailLines: 0,
        });
        await api.getSweepLogs(ns, name, {
          container: 'sweep/controller',
          tailLines: 25,
        });

        console.log(JSON.stringify(calls.map(call => ({
          url: call.url,
          accept: call.opts.headers.Accept,
        }))));
        """
    )

    encoded_resource = "team%20space%2Fns/bench%2Fname%20%CE%B1"
    assert result == [
        {
            "url": (
                f"/api/v1/jobs/{encoded_resource}/logs?"
                "pod=pod%2Fname+one&container=results%2Fserver&follow=1&tail_lines=0"
            ),
            "accept": "text/plain",
        },
        {
            "url": (
                f"/api/v1/sweeps/{encoded_resource}/logs?"
                "container=sweep%2Fcontroller&tail_lines=25"
            ),
            "accept": "text/plain",
        },
    ]
