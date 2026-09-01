# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for operator models module."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.kubernetes.crd_models import (
    AIPerfJobSpec,
    K8sEndpointConfig,
    MetricsSummary,
)

# =============================================================================
# Test MetricsSummary
# =============================================================================


class TestMetricsSummary:
    """Tests for MetricsSummary projection."""

    def test_creates_empty_summary(self) -> None:
        """Empty default has an empty data dict."""
        summary = MetricsSummary()
        assert summary.data == {}
        assert summary.to_status_dict() == {}

    def test_from_metrics_with_empty_dict(self) -> None:
        """Empty input metrics produces empty summary."""
        summary = MetricsSummary.from_metrics({})
        assert summary.data == {}

    def test_from_metrics_with_none(self) -> None:
        """None input is handled."""
        summary = MetricsSummary.from_metrics(None)
        assert summary.data == {}

    def test_from_metrics_passthrough_list_form(self) -> None:
        """Legacy list-of-dicts ``metrics`` form is normalized to tag-keyed dict."""
        metrics = {
            "metrics": [
                {"tag": "request_throughput", "avg": 100.5, "unit": "requests/sec"},
                {"tag": "request_latency", "avg": 50.0, "p50": 45.0, "p99": 120.0},
            ]
        }
        result = MetricsSummary.from_metrics(metrics).to_status_dict()
        assert result["request_throughput"]["avg"] == 100.5
        assert result["request_latency"]["p99"] == 120.0

    def test_from_metrics_passthrough_dict_form(self) -> None:
        """Live ``metrics`` dict form passes through verbatim for known tags."""
        metrics = {
            "metrics": {
                "output_token_throughput": {"avg": 500.0, "unit": "tokens/sec"},
                "time_to_first_token": {"avg": 100.0, "p50": 90.0, "p99": 200.0},
            }
        }
        result = MetricsSummary.from_metrics(metrics).to_status_dict()
        assert result["output_token_throughput"]["avg"] == 500.0
        assert result["time_to_first_token"]["p99"] == 200.0

    def test_from_metrics_unwrapped_top_level_form(self) -> None:
        """``profile_export_aiperf.json`` top-level tag dict is also accepted."""
        metrics = {
            "request_throughput": {"avg": 200.0, "unit": "requests/sec"},
            "request_latency": {"avg": 30.0, "p99": 80.0},
            "request_count": 1000,
            "error_rate": 0.05,
        }
        result = MetricsSummary.from_metrics(metrics).to_status_dict()
        assert result["request_throughput"]["avg"] == 200.0
        assert result["request_latency"]["p99"] == 80.0
        assert result["total_requests"] == 1000
        assert result["error_rate"] == 0.05

    def test_from_metrics_drops_unknown_tags(self) -> None:
        """Tags not on the curated allowlist are dropped — keeps summary tight."""
        metrics = {
            "metrics": {
                "request_throughput": {"avg": 100.0},
                "internal_diagnostic_xyz": {"avg": 999.0},
            }
        }
        result = MetricsSummary.from_metrics(metrics).to_status_dict()
        assert "request_throughput" in result
        assert "internal_diagnostic_xyz" not in result

    def test_from_metrics_does_not_shadow_with_e2e_variant(self) -> None:
        """``e2e_output_token_throughput`` is on the summary allowlist (it has its
        own UI panel) and must coexist with ``output_token_throughput`` without
        either shadowing the other — distinct keys, distinct values.
        """
        metrics = {
            "metrics": {
                "e2e_output_token_throughput": {"avg": 83.5, "unit": "tokens/sec/user"},
                "output_token_throughput": {"avg": 80105.4, "unit": "tokens/sec"},
            }
        }
        result = MetricsSummary.from_metrics(metrics).to_status_dict()
        assert result["output_token_throughput"]["avg"] == 80105.4
        assert result["e2e_output_token_throughput"]["avg"] == 83.5
        assert result["output_token_throughput"]["unit"] == "tokens/sec"
        assert result["e2e_output_token_throughput"]["unit"] == "tokens/sec/user"

    def test_from_metrics_derives_error_rate_and_total_requests(self) -> None:
        """``total_requests`` is successes + errors; ``error_rate`` is errors / total.

        ``request_count`` counts successful requests only, so the grand total
        is ``request_count + error_request_count`` (1000 + 50 = 1050) and the
        error rate is ``50 / 1050`` — matching the authoritative
        RequestErrorRateMetric, not the old ``errors / successes`` arithmetic.
        """
        metrics = {
            "metrics": {
                "request_count": {"avg": 1000},
                "error_request_count": {"avg": 50},
            }
        }
        result = MetricsSummary.from_metrics(metrics).to_status_dict()
        assert result["total_requests"] == 1050
        assert result["error_rate"] == pytest.approx(50 / 1050)

    def test_from_metrics_mirrors_authoritative_request_error_rate(self) -> None:
        """When the ``request_error_rate`` metric (a percent) is present it is
        mirrored verbatim as ``error_rate = rate / 100`` so status.summary
        agrees with the export, and ``total_requests`` stays successes + errors.
        """
        metrics = {
            "metrics": {
                "request_count": {"avg": 30},
                "error_request_count": {"avg": 30},
                "request_error_rate": {"unit": "%", "avg": 50.0},
            }
        }
        result = MetricsSummary.from_metrics(metrics).to_status_dict()
        assert result["total_requests"] == 60
        assert result["error_rate"] == pytest.approx(0.5)

    def test_from_metrics_export_parity_30_success_30_error(self) -> None:
        """A 30-success / 30-error run reports total=60 and error_rate=0.5,
        matching what the console table and profile export show (the export's
        RequestErrorRateMetric = 100 * 30 / 60 = 50%)."""
        metrics = {
            "request_count": {"unit": "requests", "avg": 30.0},
            "error_request_count": {"unit": "requests", "avg": 30.0},
        }
        result = MetricsSummary.from_metrics(metrics).to_status_dict()
        assert result["total_requests"] == 60
        assert result["error_rate"] == pytest.approx(0.5)

    def test_from_metrics_all_errors_no_request_count(self) -> None:
        """A fully-failed run (``request_count`` absent, only
        ``error_request_count`` present) still reports its total and a 1.0
        error rate rather than dropping both scalars."""
        metrics = {"error_request_count": {"unit": "requests", "avg": 9.0}}
        result = MetricsSummary.from_metrics(metrics).to_status_dict()
        assert result["total_requests"] == 9
        assert result["error_rate"] == pytest.approx(1.0)

    def test_from_metrics_handles_zero_request_count(self) -> None:
        """Zero requests means no derived scalars (avoids ZeroDivisionError)."""
        metrics = {
            "metrics": {
                "request_count": {"avg": 0},
                "error_request_count": {"avg": 0},
            }
        }
        result = MetricsSummary.from_metrics(metrics).to_status_dict()
        assert "total_requests" not in result
        assert "error_rate" not in result

    def test_from_metrics_drops_infinite_scalar_request_count(self) -> None:
        """An invalid archived scalar request count has no safe summary value."""
        assert MetricsSummary.from_metrics({"request_count": float("inf")}).data == {}

    def test_to_status_dict_returns_projection(self) -> None:
        """``to_status_dict`` returns the same nested dict that's written to CR status."""
        metrics = {"metrics": {"request_throughput": {"avg": 100.0, "unit": "rps"}}}
        summary = MetricsSummary.from_metrics(metrics)
        result = summary.to_status_dict()
        assert result == {"request_throughput": {"avg": 100.0, "unit": "rps"}}


# =============================================================================
# Test K8sEndpointConfig
# =============================================================================


class TestEndpointConfig:
    """Tests for K8sEndpointConfig model."""

    def test_creates_with_valid_url(self) -> None:
        """Verify creates with valid HTTP URL."""
        config = K8sEndpointConfig(url="http://localhost:8000")
        assert config.url == "http://localhost:8000"

    def test_creates_with_https_url(self) -> None:
        """Verify creates with HTTPS URL."""
        config = K8sEndpointConfig(url="https://api.example.com/v1")
        assert config.url == "https://api.example.com/v1"

    def test_creates_with_model_and_api_type(self) -> None:
        """Verify creates with optional fields."""
        config = K8sEndpointConfig(
            url="http://localhost:8000",
            model="gpt-4",
            api_type="openai",
        )
        assert config.model == "gpt-4"
        assert config.api_type == "openai"

    def test_default_api_type(self) -> None:
        """Verify default api_type is openai."""
        config = K8sEndpointConfig(url="http://localhost:8000")
        assert config.api_type == "openai"

    @pytest.mark.parametrize(
        "url,error_msg",
        [
            param("", "Endpoint URL is required", id="empty"),
            param("localhost:8000", "must start with http://", id="no_scheme"),
            param("ftp://example.com", "must start with http://", id="wrong_scheme"),
        ],
    )  # fmt: skip
    def test_rejects_invalid_url(self, url: str, error_msg: str) -> None:
        """Verify rejects invalid URLs."""
        with pytest.raises(ValidationError) as exc_info:
            K8sEndpointConfig(url=url)
        assert error_msg in str(exc_info.value)


# =============================================================================
# Test AIPerfJobSpec
# =============================================================================


class TestAIPerfJobSpec:
    """Tests for AIPerfJobSpec model."""

    @staticmethod
    def _benchmark(endpoint: dict[str, Any]) -> dict[str, Any]:
        """Build a minimal valid AIPerfConfig dict around the given endpoint shape."""
        return {
            "models": ["test-model"],
            "endpoint": endpoint,
            "datasets": [
                {
                    "name": "default",
                    "type": "synthetic",
                    "entries": 1,
                    "prompts": {"isl": 8, "osl": 8},
                }
            ],
            "phases": [
                {
                    "name": "default",
                    "type": "concurrency",
                    "kind": "profiling",
                    "requests": 1,
                    "concurrency": 1,
                }
            ],
        }

    @pytest.fixture
    def valid_spec(self) -> dict[str, Any]:
        """Create a valid nested spec dict."""
        return {
            "image": "aiperf:latest",
            "benchmark": self._benchmark({"url": "http://localhost:8000"}),
        }

    def test_creates_with_minimal_config(self, valid_spec: dict[str, Any]) -> None:
        """Verify creates with minimal valid configuration."""
        spec = AIPerfJobSpec.from_crd_spec(valid_spec)
        assert spec.image == "aiperf:latest"
        assert spec.image_pull_policy is None
        assert spec.cancel is False

    def test_default_resource_mode_is_burstable(
        self, valid_spec: dict[str, Any]
    ) -> None:
        """Default resourceMode is burstable.

        A spec that omits resourceMode must produce burstable pods (requests
        only, no limits) so the controller can grow during aggregation
        without hitting cgroup OOMKill. See test_jobset.py for the manifest-
        level assertion that no `resources.limits` is set in this mode.
        """
        spec = AIPerfJobSpec.from_crd_spec(valid_spec)
        assert spec.resource_mode == "burstable"

    def test_creates_with_full_config(self) -> None:
        """Verify creates with all optional fields."""
        spec = AIPerfJobSpec.from_crd_spec(
            {
                "image": "aiperf:v1.0",
                "imagePullPolicy": "Always",
                "ttlSecondsAfterFinished": 3600,
                "resultsTtlDays": 30,
                "cancel": True,
                "benchmark": self._benchmark({"url": "http://localhost:8000"}),
            }
        )
        assert spec.image == "aiperf:v1.0"
        assert spec.image_pull_policy == "Always"
        assert spec.ttl_seconds_after_finished == 3600
        assert spec.results_ttl_days == 30
        assert spec.cancel is True

    @pytest.mark.parametrize(
        "image",
        [
            param("", id="empty"),
            param("   ", id="whitespace"),
        ],
    )  # fmt: skip
    def test_rejects_empty_image(self, image: str) -> None:
        """Verify rejects empty image."""
        with pytest.raises(ValidationError) as exc_info:
            AIPerfJobSpec.from_crd_spec(
                {
                    "image": image,
                    "benchmark": self._benchmark({"url": "http://localhost:8000"}),
                }
            )
        assert "Image is required" in str(exc_info.value)

    @pytest.mark.parametrize(
        "policy",
        [
            param("invalid", id="invalid"),
            param("", id="empty"),
        ],
    )  # fmt: skip
    def test_rejects_invalid_pull_policy(self, policy: str) -> None:
        """Verify rejects invalid imagePullPolicy."""
        with pytest.raises(ValidationError) as exc_info:
            AIPerfJobSpec.from_crd_spec(
                {
                    "image": "aiperf:latest",
                    "imagePullPolicy": policy,
                    "benchmark": self._benchmark({"url": "http://localhost:8000"}),
                }
            )
        assert "image_pull_policy" in str(exc_info.value) or "imagePullPolicy" in str(
            exc_info.value
        )

    def test_rejects_missing_endpoint(self) -> None:
        """Verify rejects spec with no endpoint in benchmark."""
        bench = self._benchmark({"url": "http://localhost:8000"})
        bench.pop("endpoint")
        with pytest.raises(ValidationError):
            AIPerfJobSpec.from_crd_spec(
                {
                    "image": "aiperf:latest",
                    "benchmark": bench,
                }
            )

    def test_rejects_missing_endpoint_url(self) -> None:
        """Verify rejects endpoint without url or urls."""
        with pytest.raises(ValidationError):
            AIPerfJobSpec.from_crd_spec(
                {
                    "image": "aiperf:latest",
                    "benchmark": self._benchmark({"type": "openai"}),
                }
            )

    def test_accepts_urls_array(self) -> None:
        """Verify accepts urls array instead of url."""
        spec = AIPerfJobSpec.from_crd_spec(
            {
                "image": "aiperf:latest",
                "benchmark": self._benchmark(
                    {"urls": ["http://localhost:8000", "http://localhost:8001"]}
                ),
            }
        )
        assert spec.get_endpoint_url() == "http://localhost:8000"

    def test_get_endpoint_url_from_url(self, valid_spec: dict[str, Any]) -> None:
        """Verify get_endpoint_url extracts URL."""
        spec = AIPerfJobSpec.from_crd_spec(valid_spec)
        assert spec.get_endpoint_url() == "http://localhost:8000"

    def test_get_endpoint_url_from_urls_array(self) -> None:
        """Verify get_endpoint_url extracts first URL from array."""
        spec = AIPerfJobSpec.from_crd_spec(
            {
                "image": "aiperf:latest",
                "benchmark": self._benchmark(
                    {"urls": ["http://first:8000", "http://second:8000"]}
                ),
            }
        )
        assert spec.get_endpoint_url() == "http://first:8000"

    @pytest.mark.parametrize(
        "pull_policy",
        [
            param("Always", id="always"),
            param("IfNotPresent", id="if_not_present"),
            param("Never", id="never"),
        ],
    )  # fmt: skip
    def test_accepts_valid_pull_policies(self, pull_policy: str) -> None:
        """Verify accepts all valid pull policies."""
        spec = AIPerfJobSpec.from_crd_spec(
            {
                "image": "aiperf:latest",
                "imagePullPolicy": pull_policy,
                "benchmark": self._benchmark({"url": "http://localhost:8000"}),
            }
        )
        assert spec.image_pull_policy == pull_policy

    # =========================================================================
    # Full-CRD-spec coverage tests (Task 1 of typesafety plan)
    # =========================================================================

    def test_aiperf_job_spec_validates_full_crd_dict_via_model_validate(self) -> None:
        """A complete CRD spec dict (camelCase, with benchmark) round-trips through model_validate."""
        crd_spec = {
            "image": "nvcr.io/nvidia/aiperf:latest",
            "imagePullPolicy": "IfNotPresent",
            "timeoutSeconds": 600,
            "skipEndpointCheck": True,
            "benchmark": self._benchmark({"url": "http://example:8000"}),
        }
        spec = AIPerfJobSpec.model_validate(crd_spec)
        assert spec.image == "nvcr.io/nvidia/aiperf:latest"
        assert spec.skip_endpoint_check is True
        assert spec.timeout_seconds == 600
        assert spec.benchmark.endpoint.urls == ["http://example:8000"]

    def test_aiperf_job_spec_rejects_unknown_top_level_keys(self) -> None:
        """Unknown camelCase keys at the spec top level must be rejected."""
        crd_spec = {
            "image": "nvcr.io/nvidia/aiperf:latest",
            "benchmark": self._benchmark({"url": "http://example:8000"}),
            "bogusField": "nope",
        }
        with pytest.raises(ValueError, match="bogusField|extra"):
            AIPerfJobSpec.model_validate(crd_spec)

    def test_aiperf_job_spec_get_endpoint_url_reads_from_benchmark(self) -> None:
        """get_endpoint_url() reads benchmark.endpoint.url after restructure."""
        spec = AIPerfJobSpec.model_validate(
            {
                "benchmark": self._benchmark({"url": "http://example:8000"}),
            }
        )
        assert spec.get_endpoint_url() == "http://example:8000"


class TestAIPerfJobSpecRejectsSweep:
    """spec.sweep must be null on AIPerfJob; only AIPerfSweep accepts it."""

    def _base_spec(self) -> dict:
        return {
            "image": "test:0",
            "benchmark": {
                "models": ["test/m"],
                "endpoint": {
                    "type": "chat",
                    "urls": ["http://x:8000/v1/chat/completions"],
                },
                "datasets": [{"name": "d", "type": "synthetic", "entries": 10}],
                "phases": [
                    {
                        "name": "p",
                        "type": "concurrency",
                        "kind": "profiling",
                        "requests": 1,
                        "concurrency": 1,
                    }
                ],
            },
        }

    def test_aiperfjob_accepts_no_sweep(self):
        from aiperf.kubernetes.crd_models import AIPerfJobSpec

        AIPerfJobSpec.model_validate(self._base_spec())

    @pytest.mark.parametrize(
        "multi_run",
        [
            param({"numRuns": 2}, id="multiple-runs"),
            param(
                {
                    "numRuns": 2,
                    "convergence": {"metric": "ttft", "minRuns": 2},
                },
                id="convergence",
            ),
        ],
    )
    def test_aiperfjob_rejects_multi_run_orchestration(
        self, multi_run: dict[str, Any]
    ) -> None:
        from aiperf.kubernetes.crd_models import AIPerfJobSpec

        spec = self._base_spec()
        spec["multiRun"] = multi_run
        with pytest.raises(ValueError, match="AIPerfSweep"):
            AIPerfJobSpec.model_validate(spec)

    def test_aiperfjob_rejects_grid_sweep_with_clear_message(self):
        import pytest

        from aiperf.kubernetes.crd_models import AIPerfJobSpec

        spec = self._base_spec()
        spec["sweep"] = {
            "type": "grid",
            "parameters": {"phases.p.concurrency": [1, 2]},
        }
        with pytest.raises(ValueError) as excinfo:
            AIPerfJobSpec.model_validate(spec)
        assert "sweep must be null" in str(excinfo.value)
        assert "AIPerfSweep" in str(excinfo.value)

    def test_aiperfjob_rejects_scenarios_sweep_with_clear_message(self):
        import pytest

        from aiperf.kubernetes.crd_models import AIPerfJobSpec

        spec = self._base_spec()
        spec["sweep"] = {"type": "scenarios", "runs": [{"variables": {"x": 1}}]}
        with pytest.raises(ValueError) as excinfo:
            AIPerfJobSpec.model_validate(spec)
        assert "sweep must be null" in str(excinfo.value)


class TestAIPerfSweepSpecRequiresSweep:
    """spec.sweep is required on AIPerfSweep."""

    def _base_spec(self) -> dict:
        return {
            "image": "test:0",
            "benchmark": {
                "models": ["test/m"],
                "endpoint": {
                    "type": "chat",
                    "urls": ["http://x:8000/v1/chat/completions"],
                },
                "datasets": [{"name": "d", "type": "synthetic", "entries": 10}],
                "phases": [
                    {
                        "name": "p",
                        "type": "concurrency",
                        "kind": "profiling",
                        "requests": 1,
                        "concurrency": 1,
                    }
                ],
            },
        }

    def test_aiperfsweep_accepts_grid_sweep(self):
        from aiperf.kubernetes.crd_models import AIPerfSweepSpec

        spec = self._base_spec()
        spec["sweep"] = {
            "type": "grid",
            "parameters": {"phases.p.concurrency": [1, 2]},
        }
        AIPerfSweepSpec.model_validate(spec)

    def test_aiperfsweep_accepts_scenarios_sweep(self):
        from aiperf.kubernetes.crd_models import AIPerfSweepSpec

        spec = self._base_spec()
        spec["sweep"] = {"type": "scenarios", "runs": [{"variables": {"isl": 128}}]}
        AIPerfSweepSpec.model_validate(spec)

    def test_aiperfsweep_rejects_missing_sweep_with_clear_message(self):
        import pytest

        from aiperf.kubernetes.crd_models import AIPerfSweepSpec

        with pytest.raises(ValueError) as excinfo:
            AIPerfSweepSpec.model_validate(self._base_spec())
        assert "sweep is required" in str(excinfo.value)
        assert "AIPerfJob" in str(excinfo.value)

    def test_aiperfsweep_rejects_explicit_null_sweep_with_clear_message(self):
        import pytest

        from aiperf.kubernetes.crd_models import AIPerfSweepSpec

        spec = self._base_spec()
        spec["sweep"] = None
        with pytest.raises(ValueError) as excinfo:
            AIPerfSweepSpec.model_validate(spec)
        assert "sweep is required" in str(excinfo.value)

    def test_aiperfsweep_rejects_repeated_iteration_with_convergence(self):
        import pytest

        from aiperf.kubernetes.crd_models import AIPerfSweepSpec

        spec = self._base_spec()
        spec["sweep"] = {
            "type": "grid",
            "iterationOrder": "repeated",
            "parameters": {"phases.p.concurrency": [1, 2]},
        }
        spec["multiRun"] = {
            "numRuns": 3,
            "convergence": {"metric": "ttft", "minRuns": 2},
        }
        with pytest.raises(ValueError) as excinfo:
            AIPerfSweepSpec.model_validate(spec)
        assert "iteration_order='repeated'" in str(excinfo.value)
        assert "multi_run.convergence" in str(excinfo.value)


class TestAIPerfSweepSpecChildMetadata:
    """spec.childMetadata is the passthrough for child labels/annotations."""

    def _sweep_spec(self) -> dict:
        return {
            "image": "test:0",
            "benchmark": {
                "models": ["test/m"],
                "endpoint": {
                    "type": "chat",
                    "urls": ["http://x:8000/v1/chat/completions"],
                },
                "datasets": [{"name": "d", "type": "synthetic", "entries": 10}],
                "phases": [
                    {
                        "name": "p",
                        "type": "concurrency",
                        "kind": "profiling",
                        "requests": 1,
                        "concurrency": 1,
                    }
                ],
            },
            "sweep": {
                "type": "grid",
                "parameters": {"phases.p.concurrency": [1, 2]},
            },
        }

    def test_child_metadata_defaults_to_none(self):
        from aiperf.kubernetes.crd_models import AIPerfSweepSpec

        s = AIPerfSweepSpec.model_validate(self._sweep_spec())
        assert s.child_metadata is None

    def test_child_metadata_round_trips_camelcase(self):
        from aiperf.kubernetes.crd_models import AIPerfSweepSpec

        spec = self._sweep_spec()
        spec["childMetadata"] = {
            "labels": {"team": "perf"},
            "annotations": {"runbook": "https://wiki/runbook"},
        }
        s = AIPerfSweepSpec.model_validate(spec)
        assert s.child_metadata is not None
        assert s.child_metadata.labels == {"team": "perf"}
        assert s.child_metadata.annotations == {"runbook": "https://wiki/runbook"}
        # Round-trip: dump uses camelCase via inherited BaseConfig alias_generator.
        dumped = s.model_dump(by_alias=True, exclude_none=True)
        assert dumped["childMetadata"]["labels"] == {"team": "perf"}

    def test_child_metadata_round_trips_snake_case(self):
        """populate_by_name: snake_case input also accepted."""
        from aiperf.kubernetes.crd_models import AIPerfSweepSpec

        spec = self._sweep_spec()
        spec["child_metadata"] = {"labels": {"team": "perf"}}
        s = AIPerfSweepSpec.model_validate(spec)
        assert s.child_metadata is not None
        assert s.child_metadata.labels == {"team": "perf"}

    def test_child_metadata_rejects_typo_keys(self):
        """ObjectMetaPartial uses extra='forbid' to surface typos."""
        import pytest

        from aiperf.kubernetes.crd_models import AIPerfSweepSpec

        spec = self._sweep_spec()
        spec["childMetadata"] = {"unknownLabels": {"team": "perf"}}
        with pytest.raises(ValueError):
            AIPerfSweepSpec.model_validate(spec)

    def test_aiperfjob_rejects_child_metadata_field(self):
        """AIPerfJob has no children — child_metadata is meaningless and forbidden."""
        import pytest

        spec = {
            "image": "test:0",
            "benchmark": {
                "models": ["test/m"],
                "endpoint": {
                    "type": "chat",
                    "urls": ["http://x:8000/v1/chat/completions"],
                },
                "datasets": [{"name": "d", "type": "synthetic", "entries": 10}],
                "phases": [
                    {
                        "name": "p",
                        "type": "concurrency",
                        "kind": "profiling",
                        "requests": 1,
                        "concurrency": 1,
                    }
                ],
            },
            "childMetadata": {"labels": {"team": "perf"}},
        }
        with pytest.raises(ValueError):
            AIPerfJobSpec.model_validate(spec)
