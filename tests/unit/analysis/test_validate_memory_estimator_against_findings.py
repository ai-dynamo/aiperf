# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the estimator-vs-findings validator script.

This script's testable surface is small — most logic is the inline
``main()`` print loop. The pure helpers are:
- ``_build_params``: maps (concurrency, isl, osl, requests) →
  ``MemoryEstimationParams`` using the sweep's topology constants.
- The ``_FINDINGS`` and ``_HIGH_ERROR_CELLS`` module-level data tables —
  smoke-checked for shape consistency.
- ``main``: end-to-end smoke (small finding set, small max_error_ratio).

Out of scope: there is no findings-doc parser in this script; the
measurement table is hard-coded.
"""

from __future__ import annotations

import math
import sys
from typing import Any

import pytest

from aiperf.analysis.validate_memory_estimator_against_findings import (
    _FINDINGS,
    _HIGH_ERROR_CELLS,
    _SWEEP_CONNECTIONS_PER_WORKER,
    _SWEEP_DURATION_S,
    _SWEEP_RP_PER_POD,
    _SWEEP_WORKERS_PER_POD,
    _build_params,
    main,
)

# ============================================================
# _build_params
# ============================================================


class TestBuildParams:
    def test_build_params_basic_fields(self) -> None:
        params = _build_params(conc=5000, isl=128, osl=64, reqs=50_000)
        assert params.max_concurrency == 5000
        assert params.avg_isl_tokens == 128
        assert params.avg_osl_tokens == 64
        assert params.total_requests == 50_000

    def test_build_params_uses_sweep_topology_constants(self) -> None:
        params = _build_params(conc=2500, isl=128, osl=64, reqs=10_000)
        assert params.connections_per_worker == _SWEEP_CONNECTIONS_PER_WORKER
        assert params.record_processors_per_pod == _SWEEP_RP_PER_POD
        assert params.total_benchmark_duration_s == _SWEEP_DURATION_S

    def test_build_params_workers_count_is_ceil_conc_over_connections(self) -> None:
        # 5000 / 250 = 20 workers
        params = _build_params(conc=5000, isl=128, osl=64, reqs=50_000)
        expected = math.ceil(5000 / _SWEEP_CONNECTIONS_PER_WORKER)
        assert params.total_workers == expected

    def test_build_params_pod_count_is_ceil_workers_over_wpp(self) -> None:
        params = _build_params(conc=10_000, isl=128, osl=64, reqs=50_000)
        workers = math.ceil(10_000 / _SWEEP_CONNECTIONS_PER_WORKER)
        expected_pods = math.ceil(workers / _SWEEP_WORKERS_PER_POD)
        assert params.num_worker_pods == expected_pods

    def test_build_params_workers_per_pod_clamps_to_workers_when_low(self) -> None:
        # 100 conc / 250 connections = 1 worker. workers_per_pod must clamp
        # down to 1 (min(workers, _SWEEP_WORKERS_PER_POD)).
        params = _build_params(conc=100, isl=128, osl=64, reqs=1000)
        assert params.workers_per_pod == 1

    def test_build_params_streaming_is_true(self) -> None:
        # The script forces streaming=True since the sweep tests SSE only.
        params = _build_params(conc=5000, isl=128, osl=64, reqs=50_000)
        assert params.streaming is True

    def test_build_params_no_gpus(self) -> None:
        params = _build_params(conc=5000, isl=128, osl=64, reqs=50_000)
        assert params.num_gpus == 0


# ============================================================
# Module data tables
# ============================================================


class TestFindingsTable:
    def test_findings_non_empty(self) -> None:
        assert len(_FINDINGS) > 0

    def test_findings_rows_have_six_fields(self) -> None:
        for row in _FINDINGS:
            assert len(row) == 6
            conc, isl, osl, reqs, wp_meas, ct_meas = row
            assert isinstance(conc, int) and conc > 0
            assert isinstance(isl, int) and isl > 0
            assert isinstance(osl, int) and osl > 0
            assert isinstance(reqs, int) and reqs > 0
            assert wp_meas > 0
            assert ct_meas > 0

    def test_high_error_cells_subset_of_findings_keys(self) -> None:
        # Every high-error cell must reference a row that actually exists,
        # else the exclusion is silently no-op.
        finding_keys = {(c, i, o) for c, i, o, _, _, _ in _FINDINGS}
        assert _HIGH_ERROR_CELLS.issubset(finding_keys)


# ============================================================
# main entry point — smoke
# ============================================================


class TestMain:
    def test_main_succeeds_with_loose_tolerance(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Set a very loose tolerance so all cells pass; main returns 0.
        monkeypatch.setattr(sys, "argv", ["validate", "--max-error-ratio", "100.0"])
        rc = main()
        assert rc == 0
        out = capsys.readouterr().out
        assert "K8s memory estimator vs measured findings" in out

    def test_main_returns_nonzero_when_cells_fail_tolerance(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Inject a synthetic finding row whose measured values guarantee a
        # ratio outside [1/1.001, 1.001]. We monkeypatch _FINDINGS to a
        # one-row list with a clearly wrong measured WP value.
        import aiperf.analysis.validate_memory_estimator_against_findings as mod

        # Build a row that will not fall in the tight tolerance band.
        synthetic: list[tuple[Any, ...]] = [
            (5000, 128, 128, 50_000, 1, 1),  # absurdly low measured RSS
        ]
        monkeypatch.setattr(mod, "_FINDINGS", synthetic)
        monkeypatch.setattr(mod, "_HIGH_ERROR_CELLS", set())
        monkeypatch.setattr(sys, "argv", ["validate", "--max-error-ratio", "1.001"])

        rc = main()
        assert rc == 1
        out = capsys.readouterr().out
        assert "outside tolerance" in out
