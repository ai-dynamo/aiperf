# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CI guard for the dashboard-v2 GPU telemetry hero tiles.

The tile contract is JS-to-wire: ``components/gpu-telemetry.js`` matches on the
telemetry field names that ``GPUTelemetryAccumulator.generate_metric_results``
embeds in ``MetricResult.tag``, which come verbatim from
``GPU_TELEMETRY_METRICS_CONFIG``. A backend rename silently blanks all four
tiles, which is exactly what happened when the fields gained vendor prefixes.

Two layers:

* ``TestPrimaryTagsAliases`` parses the ``PRIMARY_TAGS`` literal out of the JS
  and asserts every alias still exists in ``GPU_TELEMETRY_METRICS_CONFIG``.
  No runtime needed, so it always runs.
* ``TestPrimaryTagsAgainstRealFields`` executes the real JS predicate under node
  against the real emitted tags, and skips when ``node`` is unavailable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from pytest import param

from aiperf.gpu_telemetry.constants import (
    AMD_GPU_TELEMETRY_PLATFORM,
    GPU_TELEMETRY_METRICS_CONFIG,
    NVIDIA_GPU_TELEMETRY_PLATFORM,
)
from tests.unit.api.test_dashboard_js import (
    _NODE_REASON,
    _node_binary,
    _run_v2_node_script,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_GPU_TELEMETRY_JS = (
    _REPO_ROOT
    / "src"
    / "aiperf"
    / "api"
    / "static-v2"
    / "components"
    / "gpu-telemetry.js"
)

_PRIMARY_TAGS_RE = re.compile(
    r"export const PRIMARY_TAGS = \[(?P<body>.*?)^\];", re.DOTALL | re.MULTILINE
)
_TILE_RE = re.compile(
    r"\{\s*label:\s*'(?P<label>[^']+)'\s*,\s*aliases:\s*\[(?P<aliases>[^\]]*)\]\s*\}"
)

_EXPECTED_LABELS = ["Power", "Utilization", "Temp", "Memory"]


def _parse_primary_tags() -> list[tuple[str, list[str]]]:
    """Extract ``[(label, [alias, ...]), ...]`` from the JS source."""
    source = _GPU_TELEMETRY_JS.read_text(encoding="utf-8")
    block = _PRIMARY_TAGS_RE.search(source)
    assert block is not None, (
        f"could not find the PRIMARY_TAGS array literal in {_GPU_TELEMETRY_JS}. "
        "If its shape changed, update this parser - do not delete the guard."
    )
    tiles = [
        (m.group("label"), re.findall(r"'([^']+)'", m.group("aliases")))
        for m in _TILE_RE.finditer(block.group("body"))
    ]
    assert tiles, "PRIMARY_TAGS parsed as empty; the guard would assert nothing"
    return tiles


_CONFIG_FIELDS = {field for _display, field, _unit in GPU_TELEMETRY_METRICS_CONFIG}


class TestPrimaryTagsAliases:
    """Every dashboard tile alias must be a field the backend actually emits."""

    def test_parse_primary_tags_finds_expected_labels(self) -> None:
        assert [label for label, _ in _parse_primary_tags()] == _EXPECTED_LABELS

    def test_primary_tags_aliases_exist_in_metrics_config(self) -> None:
        unknown = {
            f"{label}:{alias}"
            for label, aliases in _parse_primary_tags()
            for alias in aliases
            if alias not in _CONFIG_FIELDS
        }

        assert not unknown, (
            "dashboard-v2 GPU telemetry tiles reference telemetry fields that "
            "GPU_TELEMETRY_METRICS_CONFIG no longer emits: "
            f"{sorted(unknown)}. Update PRIMARY_TAGS in "
            "src/aiperf/api/static-v2/components/gpu-telemetry.js, or the tiles "
            "will render blank."
        )

    @pytest.mark.parametrize(
        "platform",
        [
            param(NVIDIA_GPU_TELEMETRY_PLATFORM, id="nvidia"),
            param(AMD_GPU_TELEMETRY_PLATFORM, id="amd"),
        ],
    )  # fmt: skip
    def test_every_tile_covers_platform(self, platform: str) -> None:
        """Each tile needs at least one alias per supported vendor, or that
        vendor's hosts get a permanently blank tile."""
        missing = [
            label
            for label, aliases in _parse_primary_tags()
            if not any(alias.startswith(f"{platform}_") for alias in aliases)
        ]

        assert not missing, (
            f"GPU telemetry tiles {missing} have no {platform} alias; "
            f"{platform} hosts would see them blank."
        )

    def test_primary_tags_aliases_are_unique(self) -> None:
        aliases = [alias for _, aliases in _parse_primary_tags() for alias in aliases]

        assert len(aliases) == len(set(aliases)), (
            f"duplicate alias across GPU telemetry tiles: {sorted(aliases)}"
        )


def _emitted_metrics() -> list[dict[str, str]]:
    """The MetricResult shape the accumulator pushes over the WebSocket.

    Mirrors ``GPUTelemetryAccumulator.generate_metric_results``:
    ``tag = f"{field}_dcgm_{source}_gpu{index}_{uuid[:12]}"``.
    """
    return [
        {"tag": f"{field}_dcgm_http___localhost_9401_metrics_gpu0_GPU-abcdef012345"}
        for _display, field, _unit in GPU_TELEMETRY_METRICS_CONFIG
    ]


@pytest.mark.skipif(_node_binary() is None, reason=_NODE_REASON)
class TestPrimaryTagsAgainstRealFields:
    """Run the real JS predicate over the real emitted tags."""

    @pytest.mark.parametrize(
        ("platform", "expected"),
        [
            param(
                NVIDIA_GPU_TELEMETRY_PLATFORM,
                {
                    "Power": "nvidia_power_usage",
                    "Utilization": "nvidia_gpu_utilization",
                    "Temp": "nvidia_temperature",
                    "Memory": "nvidia_memory_used",
                },
                id="nvidia",
            ),
            param(
                AMD_GPU_TELEMETRY_PLATFORM,
                {
                    "Power": "amd_power",
                    "Utilization": "amd_gfx_activity",
                    "Temp": "amd_temperature",
                    "Memory": "amd_memory_used",
                },
                id="amd",
            ),
        ],
    )  # fmt: skip
    def test_partition_resolves_every_tile(
        self, tmp_path: Path, platform: str, expected: dict[str, str]
    ) -> None:
        metrics = [m for m in _emitted_metrics() if m["tag"].startswith(f"{platform}_")]
        result = _run_v2_node_script(
            tmp_path,
            "import { partitionGpuMetrics, baseName } from "
            "'./components/gpu-telemetry.js';\n"
            f"const metrics = {json.dumps(metrics)};\n"
            "const { tiles, others } = partitionGpuMetrics(metrics);\n"
            "console.log(JSON.stringify({\n"
            "  tiles: Object.fromEntries(tiles.map(t => "
            "[t.label, t.metric ? baseName(t.metric.tag) : null])),\n"
            "  others: others.map(m => baseName(m.tag)),\n"
            "}));\n",
        )

        assert result["tiles"] == expected

    def test_partition_never_renders_a_metric_twice(self, tmp_path: Path) -> None:
        metrics = _emitted_metrics()
        result = _run_v2_node_script(
            tmp_path,
            "import { partitionGpuMetrics, baseName } from "
            "'./components/gpu-telemetry.js';\n"
            f"const metrics = {json.dumps(metrics)};\n"
            "const { tiles, others } = partitionGpuMetrics(metrics);\n"
            "console.log(JSON.stringify({\n"
            "  tiles: tiles.filter(t => t.metric).map(t => baseName(t.metric.tag)),\n"
            "  others: others.map(m => baseName(m.tag)),\n"
            "}));\n",
        )

        overlap = set(result["tiles"]) & set(result["others"])
        assert not overlap, (
            "GPU telemetry metrics rendered both as a hero tile and in the "
            f"'other metrics' table: {sorted(overlap)}"
        )
        assert set(result["tiles"]) | set(result["others"]) == _CONFIG_FIELDS

    def test_unprefixed_vendor_field_still_lands_on_its_tile(
        self, tmp_path: Path
    ) -> None:
        """The vendor-agnostic suffix fallback is what survives the next rename."""
        result = _run_v2_node_script(
            tmp_path,
            "import { partitionGpuMetrics, baseName } from "
            "'./components/gpu-telemetry.js';\n"
            "const metrics = [{ tag: 'intel_temperature_dcgm_x_gpu0_abc' }];\n"
            "const { tiles } = partitionGpuMetrics(metrics);\n"
            "console.log(JSON.stringify(Object.fromEntries(tiles.map(t => "
            "[t.label, t.metric ? baseName(t.metric.tag) : null]))));\n",
        )

        assert result["Temp"] == "intel_temperature"
        assert result["Power"] is None
