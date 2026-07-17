# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""V2 wiring tests for ``SmoothIsotonicSLAPlanner._mutate_base``.

Covers the three knobs whose values land on ``AdaptiveSearchSweep`` but are
consumed inside ``_mutate_base`` per probe:

* ``sla_precision`` -> ``phases.profiling.requests`` mapping.
* ``sla_warmup_seconds`` -> prepended ``warmup`` phase.
* First-probe-at-new-x extra warmup vs. replicate-at-existing-x floor.

Tests build a real ``BenchmarkConfig`` via ``model_validate`` (planner calls
``model_dump`` + ``model_validate`` -- ``SimpleNamespace`` would fail), then
assert on the fields of the ``BenchmarkConfig`` returned by ``_mutate_base``.
The full ask/tell loop is bypassed so each test isolates one mutation.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.config.config import BenchmarkConfig
from aiperf.config.sweep import AdaptiveSearchSweep, Objective
from aiperf.config.sweep.adaptive import SearchSpaceDimension, SLAFilter
from aiperf.orchestrator.aggregation.sweep import OptimizationDirection
from aiperf.orchestrator.search_planner.smooth_isotonic import (
    SmoothIsotonicSLAPlanner,
)


def _base_config(*, profiling_requests: int | None = None) -> BenchmarkConfig:
    """Real ``BenchmarkConfig`` with one ``profiling`` concurrency phase."""
    profiling: dict[str, object] = {
        "name": "profiling",
        "type": "concurrency",
        "concurrency": 1,
    }
    if profiling_requests is not None:
        profiling["requests"] = profiling_requests
    else:
        # Stop-condition required: use duration when requests is unset.
        profiling["duration"] = 60.0
    return BenchmarkConfig.model_validate(
        {
            "models": ["m"],
            "endpoint": {"urls": ["http://x"], "type": "chat"},
            "datasets": [{"name": "default", "type": "synthetic"}],
            "phases": [profiling],
        }
    )


def _adaptive_cfg(
    *,
    sla_precision: str = "normal",
    sla_warmup_seconds: float | None = None,
    sla_replicates: int = 0,
) -> AdaptiveSearchSweep:
    return AdaptiveSearchSweep(
        planner="smooth_isotonic",
        search_space=[
            SearchSpaceDimension(
                path="phases.profiling.concurrency",
                lo=1,
                hi=1000,
                kind="int",
            )
        ],
        objectives=[
            Objective(
                metric="output_token_throughput",
                stat="avg",
                direction=OptimizationDirection.MAXIMIZE,
            )
        ],
        max_iterations=30,
        n_initial_points=1,
        sla_filters=[
            SLAFilter(
                metric_tag="time_to_first_token",
                stat="p95",
                op="lt",
                threshold=200.0,
            )
        ],
        sla_precision=sla_precision,  # type: ignore[arg-type]
        sla_warmup_seconds=sla_warmup_seconds,
        sla_replicates=sla_replicates,
    )


def _make_planner(
    *,
    profiling_requests: int | None = None,
    sla_precision: str = "normal",
    sla_warmup_seconds: float | None = None,
) -> SmoothIsotonicSLAPlanner:
    return SmoothIsotonicSLAPlanner(
        _base_config(profiling_requests=profiling_requests),
        _adaptive_cfg(
            sla_precision=sla_precision,
            sla_warmup_seconds=sla_warmup_seconds,
        ),
    )


def _profiling_phase(cfg: BenchmarkConfig) -> object:
    """Return the phase named ``profiling`` from ``cfg.phases``."""
    for phase in cfg.phases:
        if phase.name == "profiling":
            return phase
    raise AssertionError("profiling phase missing from mutated config")


def _warmup_phase(cfg: BenchmarkConfig) -> object | None:
    for phase in cfg.phases:
        if phase.name == "warmup":
            return phase
    return None


# ---------------------------------------------------------------------------
# Item 2 — sla_precision -> phases.profiling.requests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("precision", "expected_requests"),
    [
        param("tight", 10000, id="tight"),
        param("normal", 1000, id="normal"),
        param("coarse", 300, id="coarse"),
    ],
)  # fmt: skip
def test_sla_precision_sets_requests_mapping(
    precision: str, expected_requests: int
) -> None:
    """Each precision tier maps to its documented per-probe request count."""
    planner = _make_planner(sla_precision=precision, sla_warmup_seconds=0)
    mutated = planner._mutate_base(42)
    profiling = _profiling_phase(mutated)
    assert profiling.requests == expected_requests  # type: ignore[attr-defined]


def test_sla_precision_normal_sets_requests_to_1000() -> None:
    planner = _make_planner(sla_precision="normal", sla_warmup_seconds=0)
    mutated = planner._mutate_base(7)
    assert _profiling_phase(mutated).requests == 1000  # type: ignore[attr-defined]


def test_sla_precision_tight_sets_requests_to_10000() -> None:
    planner = _make_planner(sla_precision="tight", sla_warmup_seconds=0)
    mutated = planner._mutate_base(7)
    assert _profiling_phase(mutated).requests == 10000  # type: ignore[attr-defined]


def test_sla_precision_coarse_sets_requests_to_300() -> None:
    planner = _make_planner(sla_precision="coarse", sla_warmup_seconds=0)
    mutated = planner._mutate_base(7)
    assert _profiling_phase(mutated).requests == 300  # type: ignore[attr-defined]


def test_sla_precision_does_not_override_user_set_requests() -> None:
    """Explicit per-phase requests beats the precision default."""
    planner = _make_planner(
        profiling_requests=42,
        sla_precision="tight",
        sla_warmup_seconds=0,
    )
    mutated = planner._mutate_base(7)
    assert _profiling_phase(mutated).requests == 42  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Item 1 — sla_warmup_seconds -> prepended warmup phase
# ---------------------------------------------------------------------------


def test_sla_warmup_seconds_zero_skips_warmup_phase() -> None:
    """Explicit ``--sla-warmup-seconds 0`` -> no warmup phase prepended."""
    planner = _make_planner(sla_warmup_seconds=0)
    mutated = planner._mutate_base(50)
    assert _warmup_phase(mutated) is None
    # Profiling is still the first (and only) phase.
    assert mutated.phases[0].name == "profiling"


def test_sla_warmup_seconds_default_auto_30s() -> None:
    """``None`` (default) -> floor at 30s, but first-probe lifts to 60s."""
    planner = _make_planner(sla_warmup_seconds=None)
    mutated = planner._mutate_base(50)
    warmup = _warmup_phase(mutated)
    assert warmup is not None
    # First probe at this x lifts the floor to 60s. Subsequent test asserts
    # the 30s floor on a replicate.
    assert warmup.duration == 60.0  # type: ignore[attr-defined]


def test_sla_warmup_seconds_default_replicate_uses_30s_floor() -> None:
    """After first probe at x, the per-replicate floor is 15s but the
    auto-default lifts the user-effective floor to 30s."""
    planner = _make_planner(sla_warmup_seconds=None)
    planner._mutate_base(50)  # first probe at 50
    mutated = planner._mutate_base(50)  # replicate at 50
    warmup = _warmup_phase(mutated)
    assert warmup is not None
    assert warmup.duration == 30.0  # type: ignore[attr-defined]


def test_sla_warmup_seconds_custom_value() -> None:
    """Explicit warmup seconds is honored when above the per-state floor."""
    planner = _make_planner(sla_warmup_seconds=120.0)
    mutated = planner._mutate_base(50)
    warmup = _warmup_phase(mutated)
    assert warmup is not None
    # 120s > 60s first-probe floor, so the explicit value wins.
    assert warmup.duration == 120.0  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Item 3 — first-probe-at-new-x extra warmup vs. replicate floor
# ---------------------------------------------------------------------------


def test_first_probe_at_new_x_uses_60s_min_warmup() -> None:
    planner = _make_planner(sla_warmup_seconds=10.0)
    mutated = planner._mutate_base(100)
    warmup = _warmup_phase(mutated)
    assert warmup is not None
    # 10s < 60s first-probe floor -> floor wins.
    assert warmup.duration == 60.0  # type: ignore[attr-defined]


def test_replicate_at_existing_x_uses_15s_min_warmup() -> None:
    """Second probe at the same x downshifts to the 15s replicate floor."""
    planner = _make_planner(sla_warmup_seconds=10.0)
    planner._mutate_base(100)  # first probe at 100
    mutated = planner._mutate_base(100)  # replicate at 100
    warmup = _warmup_phase(mutated)
    assert warmup is not None
    # 10s < 15s replicate floor -> floor wins; <60s confirms downshift.
    assert warmup.duration == 15.0  # type: ignore[attr-defined]
    assert warmup.duration < 60.0  # type: ignore[attr-defined]


def test_distinct_x_values_each_get_first_probe_floor() -> None:
    """Each unique x is its own first-probe; the planner does not collapse."""
    planner = _make_planner(sla_warmup_seconds=10.0)
    m1 = planner._mutate_base(50)
    m2 = planner._mutate_base(100)
    assert _warmup_phase(m1).duration == 60.0  # type: ignore[union-attr]
    assert _warmup_phase(m2).duration == 60.0  # type: ignore[union-attr]


def test_warmup_phase_has_exclude_from_results_true() -> None:
    planner = _make_planner(sla_warmup_seconds=30.0)
    mutated = planner._mutate_base(75)
    warmup = _warmup_phase(mutated)
    assert warmup is not None
    assert warmup.exclude_from_results is True  # type: ignore[attr-defined]


def test_warmup_phase_concurrency_matches_swept_value() -> None:
    """Warmup runs at the same concurrency as the probe so cache state
    matches what profiling will measure."""
    planner = _make_planner(sla_warmup_seconds=30.0)
    mutated = planner._mutate_base(123)
    warmup = _warmup_phase(mutated)
    assert warmup is not None
    assert warmup.concurrency == 123  # type: ignore[attr-defined]
    # And profiling itself still picks up the swept value.
    assert _profiling_phase(mutated).concurrency == 123  # type: ignore[attr-defined]


def test_warmup_phase_is_first_in_phase_order() -> None:
    """Warmup must be prepended, not appended -- it has to run before profiling."""
    planner = _make_planner(sla_warmup_seconds=30.0)
    mutated = planner._mutate_base(50)
    assert mutated.phases[0].name == "warmup"
    assert mutated.phases[1].name == "profiling"


# ---------------------------------------------------------------------------
# Credential preservation across _mutate_base
# ---------------------------------------------------------------------------


def _base_config_with_credentials() -> BenchmarkConfig:
    """Base config carrying credential-bearing fields the JSON serializers
    would redact (used to lock in the credential-preservation regression)."""
    return BenchmarkConfig.model_validate(
        {
            "models": ["m"],
            "endpoint": {
                "urls": ["http://x"],
                "type": "chat",
                "api_key": "sk-real-prod-key",
                "headers": {
                    "Authorization": "Api-Key real-secret-value",
                    "X-Trace-Id": "trace-001",
                },
            },
            "datasets": [{"name": "default", "type": "synthetic"}],
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "concurrency": 1,
                    "duration": 60.0,
                }
            ],
        }
    )


def test_mutate_base_preserves_api_key() -> None:
    """REGRESSION-LOCK: ``_mutate_base`` previously dumped with mode="json",
    which fired the EndpointConfig.api_key redactor and baked "<redacted>"
    into every iteration's config. Authenticated endpoints would then 401.
    """
    planner = SmoothIsotonicSLAPlanner(_base_config_with_credentials(), _adaptive_cfg())
    mutated = planner._mutate_base(42)
    assert mutated.endpoint.api_key == "sk-real-prod-key"


def test_mutate_base_preserves_sensitive_headers() -> None:
    """REGRESSION-LOCK: companion to ``preserves_api_key``. The headers
    serializer is what bit endpoints using non-Bearer auth schemes
    (e.g. ``Authorization: Api-Key …``, ``Authorization: Token …``) —
    a sweep run would see ``Authorization: <redacted>`` and 403 every request.
    """
    planner = SmoothIsotonicSLAPlanner(_base_config_with_credentials(), _adaptive_cfg())
    mutated = planner._mutate_base(42)
    assert mutated.endpoint.headers["Authorization"] == "Api-Key real-secret-value"
    # Non-sensitive header must round-trip too.
    assert mutated.endpoint.headers["X-Trace-Id"] == "trace-001"


def test_mutate_base_preserves_url_userinfo() -> None:
    """REGRESSION-LOCK (PR #982 dynamo-ops): ``EndpointConfig.urls`` has an
    unconditional ``_redact_urls`` serializer (no ``when_used="json"`` guard),
    so even ``mode="python"`` dumps strip ``user:pass@`` userinfo. The fix
    pairs ``mode="python"`` with ``context={"include_secrets": True}`` so
    the urls serializer's context-aware bypass fires for the planner's
    in-pipeline dump too. URL-credentialed endpoints (e.g. database URIs,
    proxy URLs) would otherwise lose their userinfo in every iteration's
    config and fail to authenticate.
    """
    cfg_dict = _base_config_with_credentials().model_dump(
        mode="python", exclude_none=True, context={"include_secrets": True}
    )
    cfg_dict["endpoint"]["urls"] = [
        "http://alice:s3cret@host1.example.com/v1/chat/completions"
    ]
    base = BenchmarkConfig.model_validate(cfg_dict)
    planner = SmoothIsotonicSLAPlanner(base, _adaptive_cfg())
    mutated = planner._mutate_base(42)
    assert mutated.endpoint.urls == [
        "http://alice:s3cret@host1.example.com/v1/chat/completions"
    ]
