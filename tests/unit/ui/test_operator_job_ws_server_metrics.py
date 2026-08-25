# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drives lib/job-ws.js with a fake WebSocket implementation under node and
asserts the per-KPI rolling buffer gains samples for each `realtime_server_metrics`
frame, keyed by curator KPI ids.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
JOB_WS = (REPO / "src" / "aiperf" / "operator" / "ui" / "lib" / "job-ws.js").as_uri()


def _run_node(script: str) -> str:
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)
    return result.stdout.strip()


def test_job_ws_accumulates_server_metric_samples() -> None:
    """Two `realtime_server_metrics` frames produce two-point buffers per KPI."""
    script = f"""
        // Stub the global WebSocket so openJobWs's connect() is exercised
        // synchronously; we drive onmessage manually.
        const fakeSockets = [];
        globalThis.WebSocket = class {{
          constructor(url) {{ this.url = url; fakeSockets.push(this); }}
          send() {{}}
          close() {{ this.onclose && this.onclose(); }}
        }};
        globalThis.window = {{ location: {{ protocol: 'http:', host: 'x' }} }};
        const {{ openJobWs }} = await import({JOB_WS!r});
        let last = null;
        const handle = openJobWs('ns', 'name', (snap) => {{ last = snap; }});
        const sock = fakeSockets[0];
        sock.onopen && sock.onopen();
        // Hand two minimal realtime_server_metrics frames, ~1 s apart.
        const frame1 = {{
          type: 'realtime_server_metrics',
          endpoint_summaries: {{
            'h1:9090': {{
              endpoint_url: 'http://h1:9090/metrics',
              metrics: {{
                dynamo_frontend_requests: {{
                  type: 'counter',
                  series: [{{ endpoint_url: 'http://h1:9090/metrics', labels: {{}}, stats: {{ rate: 4 }} }}],
                }},
              }},
            }},
          }},
        }};
        const frame2 = JSON.parse(JSON.stringify(frame1));
        frame2.endpoint_summaries['h1:9090'].metrics.dynamo_frontend_requests
              .series[0].stats.rate = 6;
        sock.onmessage({{ data: JSON.stringify(frame1) }});
        sock.onmessage({{ data: JSON.stringify(frame2) }});
        const ts = last.serverTimeseries['request-rate'];
        console.log(JSON.stringify({{
          hasSummary: last.serverSummary != null && typeof last.serverSummary === 'object',
          n: ts.length,
          values: ts.map(p => p.v),
          monotonic: ts[0].t <= ts[1].t,
        }}));
        handle.close();
    """
    assert (
        _run_node(script) == '{"hasSummary":true,"n":2,"values":[4,6],"monotonic":true}'
    )
