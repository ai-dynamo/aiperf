# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from pathlib import Path

HELPERS_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "components"
    / "server-metrics"
    / "helpers.js"
)


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


def test_normalize_server_metrics_flattens_live_cr_endpoint_summaries() -> None:
    script = f"""
        import {{ normalizeServerMetrics }} from {HELPERS_PATH.as_uri()!r};
        const normalized = normalizeServerMetrics({{
          endpoint_summaries: {{
            '10.0.0.1:9090': {{
              endpoint_url: 'http://10.0.0.1:9090/metrics',
              metrics: {{
                dynamo_frontend_requests: {{
                  type: 'counter',
                  series: [{{ labels: {{ component: 'frontend-a' }}, stats: {{ rate: 7 }} }}],
                }},
              }},
            }},
            '10.0.0.2:9090': {{
              endpoint_url: 'http://10.0.0.2:9090/metrics',
              metrics: {{
                dynamo_frontend_requests: {{
                  type: 'counter',
                  series: [{{ labels: {{ component: 'frontend-b' }}, stats: {{ rate: 11 }} }}],
                }},
              }},
            }},
          }},
        }});
        console.log(JSON.stringify({{
          configured: normalized.summary.endpoints_configured.length,
          successful: normalized.summary.endpoints_successful.length,
          series: normalized.metrics.dynamo_frontend_requests.series.map(s => [s.endpoint_url, s.stats.rate]),
        }}));
    """
    assert _run_node(script) == (
        '{"configured":2,"successful":2,'
        '"series":[["http://10.0.0.1:9090/metrics",7],'
        '["http://10.0.0.2:9090/metrics",11]]}'
    )
