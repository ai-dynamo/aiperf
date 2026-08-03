# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for GPU telemetry collection and reporting."""

import platform

import orjson
import pytest
from pytest import param

from aiperf.common.models.telemetry_models import TelemetryRecord
from aiperf.gpu_telemetry.constants import AMDSMI_SOURCE_IDENTIFIER
from tests.harness.utils import AIPerfCLI, AIPerfMockServer


@pytest.mark.skipif(
    platform.system() in ("Darwin", "Windows"),
    reason="Requires NVIDIA GPUs for DCGM telemetry (only available on Linux CI; DCGM is Linux-only).",
)
@pytest.mark.integration
@pytest.mark.asyncio
class TestGpuTelemetry:
    """Tests for GPU telemetry collection and reporting."""

    async def test_gpu_telemetry(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """GPU telemetry collection with DCGM endpoint."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer builtin \
                --endpoint-type chat \
                --gpu-telemetry {" ".join(aiperf_mock_server.dcgm_urls)} \
                --streaming \
                --request-count 100 \
                --concurrency 2 \
                --workers-max 2 \
                --ui dashboard
            """
        )
        assert result.request_count == 100
        assert result.has_gpu_telemetry
        assert result.json.telemetry_data.endpoints is not None
        assert len(result.json.telemetry_data.endpoints) > 0

        for telemetry_source_url in result.json.telemetry_data.endpoints:
            assert (
                result.json.telemetry_data.endpoints[telemetry_source_url].gpus
                is not None
            )
            assert (
                len(result.json.telemetry_data.endpoints[telemetry_source_url].gpus) > 0
            )

            for gpu_data in result.json.telemetry_data.endpoints[
                telemetry_source_url
            ].gpus.values():
                assert gpu_data.metrics is not None
                assert len(gpu_data.metrics) > 0

                # Counter metrics only have avg (delta), not min/max
                counter_metrics = {
                    "nvidia_energy_consumption",
                    "nvidia_xid_errors",
                    "nvidia_power_violation",
                }
                for metric_name, metric_value in gpu_data.metrics.items():
                    assert metric_value is not None
                    assert metric_value.avg is not None
                    assert metric_value.unit is not None
                    # Gauge metrics should have min/max; counter metrics only have avg
                    if metric_name not in counter_metrics:
                        assert metric_value.min is not None
                        assert metric_value.max is not None

    async def test_gpu_telemetry_export(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test GPU telemetry export to JSONL file with validation."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer builtin \
                --endpoint-type chat \
                --gpu-telemetry {" ".join(aiperf_mock_server.dcgm_urls)} \
                --streaming \
                --request-count 50 \
                --concurrency 2 \
                --workers-max 2
            """
        )
        assert result.request_count == 50
        assert result.has_gpu_telemetry

        # Verify GPU telemetry export JSONL file exists
        export_file = result.artifacts_dir / "gpu_telemetry_export.jsonl"
        assert export_file.exists(), "GPU telemetry export file should exist"

        # Read and validate JSONL content
        content = export_file.read_text(encoding="utf-8")
        lines = content.splitlines()
        assert len(lines) > 0, "Export file should contain telemetry records"

        # Collect GPU data for validation
        gpu_uuids = set()
        timestamps = []

        # Validate each line is valid JSON and can be parsed as TelemetryRecord
        for line in lines:
            record_dict = orjson.loads(line)
            record = TelemetryRecord.model_validate(record_dict)

            # Verify required fields are present
            assert record.timestamp_ns > 0
            assert record.telemetry_source_url is not None
            assert record.gpu_index >= 0
            assert record.gpu_uuid is not None
            assert record.gpu_model_name is not None
            assert record.telemetry_data is not None

            # Collect data for validation
            gpu_uuids.add(record.gpu_uuid)
            timestamps.append(record.timestamp_ns)

        # Verify we captured data from GPUs
        assert len(gpu_uuids) >= 2, "Should have records from at least two GPUs"

        # NOTE: Records are not necessarily in timestamp order because of the asynchronous
        # nature of the telemetry collection.

    async def test_gpu_telemetry_export_with_custom_prefix(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test GPU telemetry export with custom filename prefix."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer builtin \
                --endpoint-type chat \
                --gpu-telemetry {" ".join(aiperf_mock_server.dcgm_urls)} \
                --streaming \
                --request-count 25 \
                --concurrency 1 \
                --workers-max 1 \
                --profile-export-prefix custom_test
            """
        )

        # Verify custom filename is used
        export_file = result.artifacts_dir / "custom_test_gpu_telemetry.jsonl"
        if export_file.exists():
            # Verify content is valid
            content = export_file.read_text(encoding="utf-8")
            lines = content.splitlines()
            assert len(lines) > 0, "Export file should contain telemetry records"

            # Validate first record
            first_record = TelemetryRecord.model_validate_json(lines[0])
            assert first_record.timestamp_ns > 0
            assert first_record.telemetry_source_url is not None

    async def test_gpu_telemetry_disabled(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """GPU telemetry collection is disabled with --no-gpu-telemetry flag.

        When --no-gpu-telemetry is provided, no GPU telemetry files should be
        created and no telemetry should be collected, even if DCGM endpoints
        would otherwise be reachable.
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer builtin \
                --endpoint-type chat \
                --streaming \
                --request-count 25 \
                --concurrency 1 \
                --workers-max 1 \
                --no-gpu-telemetry
            """
        )
        assert result.request_count == 25

        # GPU telemetry should NOT be collected when disabled
        assert not result.has_gpu_telemetry, "GPU telemetry should not be collected"

        # Verify no GPU telemetry files were created
        jsonl_files = list(result.artifacts_dir.glob("*gpu_telemetry*.jsonl"))
        assert len(jsonl_files) == 0, f"Unexpected GPU telemetry files: {jsonl_files}"


@pytest.mark.integration
@pytest.mark.asyncio
class TestAMDSMITelemetry:
    """Tests for AMD GPU telemetry collection using the fake amdsmi bindings.

    These tests activate ``aiperf-mock-amdsmi`` via ``AIPERF_MOCK_AMDSMI=1`` so
    the full AMD telemetry path exercises on any host — no AMD hardware required.
    """

    @pytest.fixture(autouse=True)
    def _activate_mock_amdsmi(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AIPERF_MOCK_AMDSMI", "1")

    async def test_amd_gpu_telemetry(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ) -> None:
        """AMD telemetry collection populates amd_* metrics end-to-end."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer builtin \
                --endpoint-type chat \
                --gpu-telemetry amdsmi \
                --streaming \
                --request-count 10 \
                --concurrency 2 \
                --workers-max 2
            """
        )
        assert result.request_count == 10
        assert result.has_gpu_telemetry
        assert result.json.telemetry_data.endpoints is not None
        assert len(result.json.telemetry_data.endpoints) > 0

        # amd_energy_consumption and amd_ecc_uncorrectable are counters (delta only)
        counter_metrics = {"amd_energy_consumption", "amd_ecc_uncorrectable"}
        for endpoint_data in result.json.telemetry_data.endpoints.values():
            assert endpoint_data.gpus is not None
            assert len(endpoint_data.gpus) > 0
            for gpu_data in endpoint_data.gpus.values():
                assert gpu_data.metrics is not None
                assert len(gpu_data.metrics) > 0
                amd_metrics = {k for k in gpu_data.metrics if k.startswith("amd_")}
                assert amd_metrics, "No amd_* metrics collected"
                # amd_mm_activity returns 'N/A' on Instinct GPUs — the collector
                # must drop it rather than surfacing it as 0.0 or None.
                assert "amd_mm_activity" not in gpu_data.metrics, (
                    "amd_mm_activity should be absent: mock returns 'N/A' for "
                    "Instinct GPUs and the collector must filter it out"
                )
                for metric_name, metric_value in gpu_data.metrics.items():
                    assert metric_value.avg is not None
                    assert metric_value.unit is not None
                    if metric_name not in counter_metrics:
                        assert metric_value.min is not None
                        assert metric_value.max is not None

    async def test_amd_gpu_telemetry_export(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ) -> None:
        """AMD telemetry JSONL export contains valid TelemetryRecord entries."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer builtin \
                --endpoint-type chat \
                --gpu-telemetry amdsmi \
                --streaming \
                --request-count 10 \
                --concurrency 2 \
                --workers-max 2
            """
        )
        assert result.request_count == 10
        assert result.has_gpu_telemetry

        export_file = result.artifacts_dir / "gpu_telemetry_export.jsonl"
        assert export_file.exists(), "GPU telemetry export file should exist"

        lines = export_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) > 0, "Export file should contain telemetry records"

        gpu_uuids: set[str] = set()
        for line in lines:
            record = TelemetryRecord.model_validate(orjson.loads(line))
            assert record.timestamp_ns > 0
            assert record.telemetry_source_url == AMDSMI_SOURCE_IDENTIFIER
            assert record.gpu_index >= 0
            assert record.gpu_uuid is not None
            assert record.gpu_model_name is not None
            assert record.telemetry_data is not None
            gpu_uuids.add(record.gpu_uuid)

        # Default mock exposes 2 GPUs
        assert len(gpu_uuids) >= 2, "Should have records from at least two GPUs"

    async def test_amd_gpu_telemetry_export_with_custom_prefix(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ) -> None:
        """AMD telemetry JSONL export respects --profile-export-prefix."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer builtin \
                --endpoint-type chat \
                --gpu-telemetry amdsmi \
                --streaming \
                --request-count 10 \
                --concurrency 1 \
                --workers-max 1 \
                --profile-export-prefix custom_amd
            """
        )

        export_file = result.artifacts_dir / "custom_amd_gpu_telemetry.jsonl"
        assert export_file.exists(), (
            "Custom-prefix GPU telemetry export file should exist"
        )
        lines = export_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) > 0, "Export file should contain telemetry records"
        first = TelemetryRecord.model_validate_json(lines[0])
        assert first.timestamp_ns > 0
        assert first.telemetry_source_url == AMDSMI_SOURCE_IDENTIFIER

    async def test_amd_gpu_telemetry_disabled(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ) -> None:
        """--no-gpu-telemetry suppresses AMD collection even when mock is active."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer builtin \
                --endpoint-type chat \
                --streaming \
                --request-count 10 \
                --concurrency 1 \
                --workers-max 1 \
                --no-gpu-telemetry
            """
        )
        assert result.request_count == 10
        assert not result.has_gpu_telemetry, "GPU telemetry should not be collected"
        jsonl_files = list(result.artifacts_dir.glob("*gpu_telemetry*.jsonl"))
        assert len(jsonl_files) == 0, f"Unexpected GPU telemetry files: {jsonl_files}"

    async def test_amd_derived_efficiency_metrics(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ) -> None:
        """AMD energy-efficiency derived metrics are present and internally consistent.

        Validates three cross-metric relationships that must hold by construction:
          energy_per_output_token (mJ/tok) = total_energy (J) * 1000 / total_osl
          energy_per_request (J/req)       = total_energy (J) / request_count
          output_tokens_per_joule (tok/J)  = total_osl / total_energy (J)
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer builtin \
                --endpoint-type chat \
                --gpu-telemetry amdsmi \
                --streaming \
                --request-count 10 \
                --concurrency 1 \
                --workers-max 1
            """
        )
        assert result.request_count == 10
        assert result.has_gpu_telemetry

        from aiperf.common.models.export_models import JsonMetricResult

        j = result.json
        extra = j.model_extra or {}

        def _m(name: str) -> JsonMetricResult | None:
            val = getattr(j, name, None) or extra.get(name)
            if val is None:
                return None
            if isinstance(val, dict):
                return JsonMetricResult.model_validate(val)
            return val

        total_energy = _m("amd_total_gpu_energy")
        energy_per_output_token = _m("amd_energy_per_output_token")
        energy_per_request = _m("amd_energy_per_request")
        output_tokens_per_joule = _m("amd_output_tokens_per_joule")

        assert total_energy is not None, "amd_total_gpu_energy missing from JSON export"
        assert energy_per_output_token is not None, (
            "amd_energy_per_output_token missing"
        )
        assert energy_per_request is not None, "amd_energy_per_request missing"

        total_energy_j: float = total_energy.avg
        assert total_energy_j > 0, "Total AMD GPU energy must be positive"
        assert energy_per_output_token.avg > 0
        assert energy_per_request.avg > 0

        # energy_per_output_token (mJ/tok) = total_energy_j * 1000 / total_osl
        if j.total_osl and j.total_osl.avg and j.total_osl.avg > 0:
            expected_mj_per_tok = total_energy_j * 1000.0 / j.total_osl.avg
            assert energy_per_output_token.avg == pytest.approx(
                expected_mj_per_tok, rel=1e-3
            ), (
                f"energy_per_output_token={energy_per_output_token.avg:.4f} mJ/tok "
                f"!= total_energy*1000/total_osl={expected_mj_per_tok:.4f} mJ/tok"
            )

            # output_tokens_per_joule = 1 / (energy_per_output_token / 1000)
            if output_tokens_per_joule is not None:
                expected_tpj = j.total_osl.avg / total_energy_j
                assert output_tokens_per_joule.avg == pytest.approx(
                    expected_tpj, rel=1e-3
                )

        # energy_per_request (J/req) = total_energy_j / request_count
        if j.request_count and j.request_count.avg and j.request_count.avg > 0:
            expected_j_per_req = total_energy_j / j.request_count.avg
            assert energy_per_request.avg == pytest.approx(
                expected_j_per_req, rel=1e-3
            ), (
                f"energy_per_request={energy_per_request.avg:.4f} J/req "
                f"!= total_energy/request_count={expected_j_per_req:.4f} J/req"
            )

        # energy_per_total_token <= energy_per_output_token (more tokens in denominator)
        energy_per_total_token = _m("amd_energy_per_total_token")
        if (
            energy_per_total_token is not None
            and energy_per_total_token.avg is not None
        ):
            assert energy_per_total_token.avg <= energy_per_output_token.avg, (
                "energy_per_total_token should be <= energy_per_output_token "
                "(total tokens >= output tokens)"
            )


_PLATFORM_EXPECTED_PREFIX: dict[str, str] = {
    "nvidia": "nvidia_",
    "amd": "amd_",
}
_PLATFORM_FORBIDDEN_PREFIX: dict[str, str] = {
    "nvidia": "amd_",
    "amd": "nvidia_",
}


@pytest.mark.integration
@pytest.mark.asyncio
class TestTelemetryVendorIsolation:
    """Per-GPU vendor namespace isolation.

    Each GPU's metrics must only contain the prefix matching its platform tag.
    This design is forward-compatible with heterogeneous systems: validation is
    localised to each GPU using GpuSummary.platform so a mixed NVIDIA+AMD node
    passes as long as every individual GPU is internally consistent.
    """

    @pytest.mark.parametrize(
        "collector",
        [
            param(
                "dcgm",
                marks=pytest.mark.skipif(
                    platform.system() in ("Darwin", "Windows"),
                    reason="DCGM telemetry requires Linux",
                ),
                id="dcgm",
            ),
            param("amdsmi", id="amdsmi"),
        ],
    )  # fmt: skip
    async def test_vendor_metric_isolation(
        self,
        collector: str,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Each GPU only carries metrics prefixed for its own platform.

        For every GPU in the result:
        - at least one metric with the expected vendor prefix is present
          (the vendor URL produced data)
        - no metric with the other vendor's prefix is present
          (namespaces are not crossed)
        """
        if collector == "amdsmi":
            monkeypatch.setenv("AIPERF_MOCK_AMDSMI", "1")
            telemetry_arg = "amdsmi"
        else:
            telemetry_arg = " ".join(aiperf_mock_server.dcgm_urls)

        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer builtin \
                --endpoint-type chat \
                --gpu-telemetry {telemetry_arg} \
                --streaming \
                --request-count 10 \
                --concurrency 1 \
                --workers-max 1
            """
        )
        assert result.has_gpu_telemetry
        assert result.json.telemetry_data.endpoints is not None

        for source_url, endpoint_data in result.json.telemetry_data.endpoints.items():
            assert endpoint_data.gpus, f"No GPUs reported for {source_url}"
            for gpu_uuid, gpu_data in endpoint_data.gpus.items():
                gpu_platform = gpu_data.platform
                expected_prefix = _PLATFORM_EXPECTED_PREFIX.get(gpu_platform)
                forbidden_prefix = _PLATFORM_FORBIDDEN_PREFIX.get(gpu_platform)

                if expected_prefix is not None:
                    present = [
                        k for k in gpu_data.metrics if k.startswith(expected_prefix)
                    ]
                    assert present, (
                        f"GPU {gpu_uuid[:12]} (platform={gpu_platform!r}) at "
                        f"{source_url!r} has no '{expected_prefix}' metrics — "
                        f"vendor URL present but no matching data collected"
                    )

                if forbidden_prefix is not None:
                    leaked = [
                        k for k in gpu_data.metrics if k.startswith(forbidden_prefix)
                    ]
                    assert not leaked, (
                        f"GPU {gpu_uuid[:12]} (platform={gpu_platform!r}) at "
                        f"{source_url!r} contains forbidden '{forbidden_prefix}' "
                        f"metrics: {leaked}"
                    )
