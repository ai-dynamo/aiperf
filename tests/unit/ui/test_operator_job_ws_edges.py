# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge tests for the operator job WebSocket client."""

from __future__ import annotations

from pathlib import Path

from tests.unit.ui.node_utils import run_node

REPO = Path(__file__).resolve().parents[3]
JOB_WS = (REPO / "src" / "aiperf" / "operator" / "ui" / "lib" / "job-ws.js").as_uri()


def test_job_ws_encodes_namespace_and_name_in_websocket_url() -> None:
    script = f"""
        const sockets = [];
        globalThis.WebSocket = class {{
          constructor(url) {{ this.url = url; sockets.push(this); }}
          send() {{}}
          close() {{}}
        }};
        globalThis.window = {{ location: {{ protocol: 'https:', host: 'dash.example.test' }} }};
        const {{ openJobWs }} = await import({JOB_WS!r});
        const handle = openJobWs('team/a b', 'job/name?x y', () => {{}});
        console.log(JSON.stringify({{ url: sockets[0].url }}));
        handle.close();
    """

    assert run_node(script) == (
        '{"url":"wss://dash.example.test/api/v1/jobs/team%2Fa%20b/'
        'job%2Fname%3Fx%20y/ws"}'
    )


def test_job_ws_ignores_malformed_json_and_parses_message_type_alias() -> None:
    script = f"""
        Date.now = () => 0;
        const sockets = [];
        globalThis.WebSocket = class {{
          constructor(url) {{ this.url = url; sockets.push(this); }}
          send() {{}}
          close() {{}}
        }};
        globalThis.window = {{ location: {{ protocol: 'http:', host: 'dash.example.test' }} }};
        const {{ openJobWs }} = await import({JOB_WS!r});
        const updates = [];
        const handle = openJobWs('ns', 'job', snap => updates.push(snap));
        const sock = sockets[0];
        sock.onopen();
        updates.length = 0;
        sock.onmessage({{ data: '{{not json' }});
        sock.onmessage({{ data: JSON.stringify({{ type: 'other', metrics: [] }}) }});
        sock.onmessage({{
          data: JSON.stringify({{
            message_type: 'realtime_metrics',
            metrics: [{{ tag: 'latency', avg: 10, unit: 'ms', bad: 'skip', inf: Infinity }}],
          }}),
        }});
        console.log(JSON.stringify({{
          updates: updates.length,
          summary: updates.at(-1).summary,
          point: updates.at(-1).timeseries.latency[0],
        }}));
        handle.close();
    """

    assert run_node(script) == (
        '{"updates":1,"summary":{"latency":{"avg":10,"unit":"ms"}},'
        '"point":{"t":0,"values":{"avg":10}}}'
    )


def test_job_ws_merges_live_snapshots_without_mutating_previous_snapshot() -> None:
    script = f"""
        let now = 1000;
        Date.now = () => now;
        const sockets = [];
        globalThis.WebSocket = class {{
          constructor(url) {{ this.url = url; sockets.push(this); }}
          send() {{}}
          close() {{}}
        }};
        globalThis.window = {{ location: {{ protocol: 'http:', host: 'dash.example.test' }} }};
        const {{ openJobWs }} = await import({JOB_WS!r});
        const updates = [];
        const handle = openJobWs('ns', 'job', snap => updates.push(snap));
        const sock = sockets[0];
        updates.length = 0;
        sock.onmessage({{
          data: JSON.stringify({{
            type: 'realtime_metrics',
            metrics: [{{ tag: 'latency', avg: 10, unit: 'ms' }}],
          }}),
        }});
        const first = updates.at(-1);
        now = 2000;
        sock.onmessage({{
          data: JSON.stringify({{
            type: 'realtime_metrics',
            metrics: [{{ tag: 'latency', p99: 25 }}],
          }}),
        }});
        const second = updates.at(-1);
        console.log(JSON.stringify({{
          firstSummary: first.summary,
          secondSummary: second.summary,
          firstCount: first.timeseries.latency.length,
          secondPoints: second.timeseries.latency,
          topLevelMapsDiffer: first.summary !== second.summary && first.timeseries !== second.timeseries,
        }}));
        handle.close();
    """

    assert run_node(script) == (
        '{"firstSummary":{"latency":{"avg":10,"unit":"ms"}},'
        '"secondSummary":{"latency":{"avg":10,"unit":"ms","p99":25}},'
        '"firstCount":1,"secondPoints":[{"t":1000,"values":{"avg":10}},'
        '{"t":2000,"values":{"p99":25}}],"topLevelMapsDiffer":true}'
    )


def test_job_ws_merges_server_summary_and_timeseries_snapshot_shape() -> None:
    script = f"""
        let now = 1000;
        Date.now = () => now;
        const sockets = [];
        globalThis.WebSocket = class {{
          constructor(url) {{ this.url = url; sockets.push(this); }}
          send() {{}}
          close() {{}}
        }};
        globalThis.window = {{ location: {{ protocol: 'http:', host: 'dash.example.test' }} }};
        const {{ openJobWs }} = await import({JOB_WS!r});
        let last = null;
        const handle = openJobWs('ns', 'job', snap => {{ last = snap; }});
        const sock = sockets[0];
        const payload = rate => ({{
          type: 'realtime_server_metrics',
          payload: {{
            endpoint_summaries: {{
              'h1:9090': {{
                endpoint_url: 'http://h1:9090/metrics',
                metrics: {{
                  dynamo_frontend_requests: {{
                    type: 'counter',
                    series: [{{ endpoint_url: 'http://h1:9090/metrics', labels: {{}}, stats: {{ rate }} }}],
                  }},
                }},
              }},
            }},
          }},
        }});
        sock.onmessage({{ data: JSON.stringify(payload(4)) }});
        now = 2000;
        sock.onmessage({{ data: JSON.stringify(payload(6)) }});
        console.log(JSON.stringify({{
          serverSummaryKeys: Object.keys(last.serverSummary.endpoint_summaries),
          requestRate: last.serverTimeseries['request-rate'],
          connected: last.connected,
        }}));
        handle.close();
    """

    assert run_node(script) == (
        '{"serverSummaryKeys":["h1:9090"],'
        '"requestRate":[{"t":1000,"v":4},{"t":2000,"v":6}],'
        '"connected":true}'
    )


def test_job_ws_schedules_reconnect_and_close_cancels_pending_timer() -> None:
    script = f"""
        const timers = [];
        globalThis.setTimeout = (fn, delay) => {{
          timers.push({{ fn, delay, cleared: false }});
          return timers.length - 1;
        }};
        globalThis.clearTimeout = id => {{ timers[id].cleared = true; }};
        const sockets = [];
        globalThis.WebSocket = class {{
          constructor(url) {{ this.url = url; sockets.push(this); }}
          send(msg) {{ this.sent = msg; }}
          close(code, reason) {{ this.closedWith = [code, reason]; this.onclose(); }}
        }};
        globalThis.window = {{ location: {{ protocol: 'http:', host: 'dash.example.test' }} }};
        const {{ openJobWs }} = await import({JOB_WS!r});
        const updates = [];
        const handle = openJobWs('ns', 'job', snap => updates.push(snap.connected));
        const sock = sockets[0];
        sock.onopen();
        sock.onclose();
        handle.close();
        timers[0].fn();
        console.log(JSON.stringify({{
          sent: JSON.parse(sock.sent),
          updates,
          timerDelay: timers[0].delay,
          timerCleared: timers[0].cleared,
          socketCount: sockets.length,
        }}));
    """

    assert run_node(script) == (
        '{"sent":{"type":"subscribe","message_types":["realtime_metrics",'
        '"realtime_server_metrics"]},"updates":[true,false],"timerDelay":2000,'
        '"timerCleared":true,"socketCount":1}'
    )


def test_job_ws_close_live_socket_does_not_schedule_reconnect() -> None:
    script = f"""
        const timers = [];
        globalThis.setTimeout = (fn, delay) => {{ timers.push({{ fn, delay }}); return timers.length - 1; }};
        globalThis.clearTimeout = () => {{}};
        const sockets = [];
        globalThis.WebSocket = class {{
          constructor(url) {{ this.url = url; sockets.push(this); }}
          send() {{}}
          close(code, reason) {{ this.closedWith = [code, reason]; this.onclose(); }}
        }};
        globalThis.window = {{ location: {{ protocol: 'http:', host: 'dash.example.test' }} }};
        const {{ openJobWs }} = await import({JOB_WS!r});
        const updates = [];
        const handle = openJobWs('ns', 'job', snap => updates.push(snap.connected));
        handle.close();
        console.log(JSON.stringify({{
          closeArgs: sockets[0].closedWith,
          updates,
          timerCount: timers.length,
        }}));
    """

    assert run_node(script) == (
        '{"closeArgs":[1000,"page leaving"],"updates":[],"timerCount":0}'
    )
