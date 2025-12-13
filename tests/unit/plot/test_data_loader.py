# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for data loading functionality.
"""

from pathlib import Path
from typing import Any

import orjson
import pandas as pd
import pytest

from aiperf.plot.core.data_loader import DataLoader, RunData, RunMetadata
from aiperf.plot.exceptions import DataLoadError


class TestDataLoaderLoadRun:
    """Tests for DataLoader.load_run method."""

    def test_load_single_run_success(self, single_run_dir: Path) -> None:
        """Test successfully loading a single run."""
        loader = DataLoader()
        run = loader.load_run(single_run_dir)

        assert isinstance(run, RunData)
        assert isinstance(run.metadata, RunMetadata)
        assert isinstance(run.requests, pd.DataFrame)
        assert isinstance(run.aggregated, dict)

        # Check that requests were loaded
        assert len(run.requests) == 10
        assert "time_to_first_token" in run.requests.columns
        assert "request_latency" in run.requests.columns

    def test_load_run_nonexistent_path(self) -> None:
        """Test loading from nonexistent path raises error."""
        loader = DataLoader()
        fake_path = Path("/nonexistent/path")

        with pytest.raises(DataLoadError, match="does not exist"):
            loader.load_run(fake_path)

    def test_load_run_file_path(self, tmp_path: Path) -> None:
        """Test loading from file path raises error."""
        loader = DataLoader()
        file_path = tmp_path / "test.txt"
        file_path.write_text("test")

        with pytest.raises(DataLoadError, match="not a directory"):
            loader.load_run(file_path)

    def test_load_run_missing_jsonl(self, tmp_path: Path) -> None:
        """Test loading run without JSONL file raises error."""
        loader = DataLoader()
        run_dir = tmp_path / "incomplete_run"
        run_dir.mkdir()

        with pytest.raises(DataLoadError, match="JSONL file not found"):
            loader.load_run(run_dir)

    def test_metadata_extraction(self, single_run_dir: Path) -> None:
        """Test that metadata is correctly extracted."""
        loader = DataLoader()
        run = loader.load_run(single_run_dir)

        assert run.metadata.run_name == single_run_dir.name
        assert run.metadata.run_path == single_run_dir
        assert run.metadata.model == "Qwen/Qwen3-0.6B"
        assert run.metadata.concurrency == 1
        assert run.metadata.request_count == 64
        assert run.metadata.endpoint_type == "chat"

    def test_timestamp_columns_remain_as_integers(self, single_run_dir: Path) -> None:
        """Test that timestamp columns remain as integer nanoseconds."""
        loader = DataLoader()
        run = loader.load_run(single_run_dir)

        # Check that timestamp columns are numeric integers (nanoseconds)
        assert pd.api.types.is_numeric_dtype(run.requests["request_start_ns"])
        assert pd.api.types.is_numeric_dtype(run.requests["request_end_ns"])
        # Verify they contain nanosecond values (large integers)
        assert run.requests["request_start_ns"].min() > 1e18


class TestDataLoaderLoadMultipleRuns:
    """Tests for DataLoader.load_multiple_runs method."""

    def test_load_multiple_runs_success(self, multiple_run_dirs: list[Path]) -> None:
        """Test successfully loading multiple runs."""
        loader = DataLoader()
        runs = loader.load_multiple_runs(multiple_run_dirs)

        assert len(runs) == 2
        assert all(isinstance(r, RunData) for r in runs)

    def test_load_empty_list_raises_error(self) -> None:
        """Test that empty run list raises error."""
        loader = DataLoader()

        with pytest.raises(DataLoadError, match="No run paths provided"):
            loader.load_multiple_runs([])

    def test_load_with_invalid_run_raises_error(
        self, single_run_dir: Path, tmp_path: Path
    ) -> None:
        """Test that invalid run in list raises error."""
        loader = DataLoader()
        invalid_dir = tmp_path / "invalid"
        invalid_dir.mkdir()

        with pytest.raises(DataLoadError):
            loader.load_multiple_runs([single_run_dir, invalid_dir])


class TestDataLoaderLoadJsonl:
    """Tests for DataLoader._load_jsonl method."""

    def test_load_valid_jsonl(self, single_run_dir: Path) -> None:
        """Test loading valid JSONL file."""
        loader = DataLoader()
        jsonl_path = single_run_dir / "profile_export.jsonl"
        df = loader._load_jsonl(jsonl_path)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert "time_to_first_token" in df.columns
        assert "session_num" in df.columns

    def test_load_jsonl_missing_file(self, tmp_path: Path) -> None:
        """Test loading nonexistent JSONL file raises error."""
        loader = DataLoader()
        fake_path = tmp_path / "nonexistent.jsonl"

        with pytest.raises(DataLoadError, match="JSONL file not found"):
            loader._load_jsonl(fake_path)

    def test_load_jsonl_with_corrupted_lines(self, tmp_path: Path) -> None:
        """Test loading JSONL with corrupted lines."""
        loader = DataLoader()
        jsonl_path = tmp_path / "corrupted.jsonl"

        # Write JSONL with two good lines and one bad line
        with open(jsonl_path, "w") as f:
            f.write(
                orjson.dumps(
                    {
                        "metadata": {
                            "session_num": 0,
                            "request_start_ns": 1000000000,
                            "request_end_ns": 1500000000,
                            "worker_id": "worker-0",
                            "record_processor_id": "processor-0",
                            "benchmark_phase": "profiling",
                        },
                        "metrics": {
                            "time_to_first_token": {"value": 45.0, "unit": "ms"}
                        },
                    }
                ).decode("utf-8")
                + "\n"
            )
            f.write("{ invalid json }\n")
            f.write(
                orjson.dumps(
                    {
                        "metadata": {
                            "session_num": 1,
                            "request_start_ns": 2000000000,
                            "request_end_ns": 2500000000,
                            "worker_id": "worker-0",
                            "record_processor_id": "processor-0",
                            "benchmark_phase": "profiling",
                        },
                        "metrics": {
                            "time_to_first_token": {"value": 50.0, "unit": "ms"}
                        },
                    }
                ).decode("utf-8")
                + "\n"
            )

        df = loader._load_jsonl(jsonl_path)
        # Should load 2 valid records, skip 1 corrupted
        assert len(df) == 2

    def test_load_jsonl_empty_file_raises_error(self, tmp_path: Path) -> None:
        """Test loading empty JSONL file raises error."""
        loader = DataLoader()
        jsonl_path = tmp_path / "empty.jsonl"
        jsonl_path.write_text("")

        with pytest.raises(DataLoadError, match="No valid records found"):
            loader._load_jsonl(jsonl_path)


class TestDataLoaderComputeInterChunkLatencyStats:
    """Tests for DataLoader._compute_inter_chunk_latency_stats method."""

    def test_compute_stats_with_valid_data(self) -> None:
        """Test computing statistics from valid inter_chunk_latency data."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = DataLoader._compute_inter_chunk_latency_stats(values)

        assert "inter_chunk_latency_avg" in stats
        assert "inter_chunk_latency_p50" in stats
        assert "inter_chunk_latency_p95" in stats
        assert "inter_chunk_latency_std" in stats
        assert "inter_chunk_latency_min" in stats
        assert "inter_chunk_latency_max" in stats
        assert "inter_chunk_latency_range" in stats

        # Verify computed values
        assert stats["inter_chunk_latency_avg"] == 30.0
        assert stats["inter_chunk_latency_p50"] == 30.0
        assert stats["inter_chunk_latency_min"] == 10.0
        assert stats["inter_chunk_latency_max"] == 50.0
        assert stats["inter_chunk_latency_range"] == 40.0

    def test_compute_stats_with_empty_array(self) -> None:
        """Test that empty array returns empty dict."""
        stats = DataLoader._compute_inter_chunk_latency_stats([])
        assert stats == {}

    def test_compute_stats_with_single_value(self) -> None:
        """Test computing statistics from single value."""
        values = [25.0]
        stats = DataLoader._compute_inter_chunk_latency_stats(values)

        assert stats["inter_chunk_latency_avg"] == 25.0
        assert stats["inter_chunk_latency_p50"] == 25.0
        assert stats["inter_chunk_latency_min"] == 25.0
        assert stats["inter_chunk_latency_max"] == 25.0
        assert stats["inter_chunk_latency_range"] == 0.0
        assert stats["inter_chunk_latency_std"] == 0.0

    def test_compute_stats_with_jitter(self) -> None:
        """Test statistics capture jitter/variance in stream health."""
        # Stable stream
        stable_values = [20.0, 20.0, 20.0, 20.0, 20.0]
        stable_stats = DataLoader._compute_inter_chunk_latency_stats(stable_values)

        # Jittery stream with spike
        jittery_values = [10.0, 10.0, 10.0, 50.0, 10.0]
        jittery_stats = DataLoader._compute_inter_chunk_latency_stats(jittery_values)

        # Stable stream should have zero std and range
        assert stable_stats["inter_chunk_latency_std"] == 0.0
        assert stable_stats["inter_chunk_latency_range"] == 0.0

        # Jittery stream should have higher std and range
        assert jittery_stats["inter_chunk_latency_std"] > 0.0
        assert jittery_stats["inter_chunk_latency_range"] == 40.0
        assert jittery_stats["inter_chunk_latency_max"] == 50.0


class TestDataLoaderLoadAggregatedJson:
    """Tests for DataLoader._load_aggregated_json method."""

    def test_load_valid_aggregated_json(self, single_run_dir: Path) -> None:
        """Test loading valid aggregated JSON."""
        loader = DataLoader()
        json_path = single_run_dir / "profile_export_aiperf.json"
        data = loader._load_aggregated_json(json_path)

        assert isinstance(data, dict)
        assert "input_config" in data

    def test_load_missing_aggregated_json(self, tmp_path: Path) -> None:
        """Test loading missing aggregated JSON raises DataLoadError."""
        loader = DataLoader()
        fake_path = tmp_path / "nonexistent.json"

        with pytest.raises(DataLoadError, match="JSON file not found"):
            loader._load_aggregated_json(fake_path)

    def test_load_invalid_json(self, tmp_path: Path) -> None:
        """Test loading invalid JSON raises DataLoadError."""
        loader = DataLoader()
        json_path = tmp_path / "invalid.json"
        json_path.write_text("{ invalid json }")

        with pytest.raises(DataLoadError, match="Failed to parse JSON"):
            loader._load_aggregated_json(json_path)


class TestDataLoaderExtractMetadata:
    """Tests for DataLoader._extract_metadata method."""

    def test_extract_metadata_with_all_data(
        self, tmp_path: Path, sample_aggregated_data: dict[str, Any]
    ) -> None:
        """Test metadata extraction with complete data."""
        loader = DataLoader()
        run_path = tmp_path / "test_run"
        requests_df = pd.DataFrame(
            {
                "request_start_ns": pd.to_datetime(
                    [1000000000, 2000000000], unit="ns", utc=True
                ),
                "request_end_ns": pd.to_datetime(
                    [1500000000, 2500000000], unit="ns", utc=True
                ),
            }
        )

        metadata = loader._extract_metadata(
            run_path, requests_df, sample_aggregated_data
        )

        assert metadata.run_name == "test_run"
        assert metadata.run_path == run_path
        assert metadata.model == "test-model"
        assert metadata.concurrency == 4
        assert metadata.request_count == 100
        assert metadata.endpoint_type == "chat"
        assert metadata.duration_seconds is not None

    def test_extract_metadata_missing_aggregated_data(self, tmp_path: Path) -> None:
        """Test metadata extraction without aggregated data."""
        loader = DataLoader()
        run_path = tmp_path / "test_run"
        requests_df = pd.DataFrame()

        metadata = loader._extract_metadata(run_path, requests_df, {})

        assert metadata.run_name == "test_run"
        assert metadata.model is None
        assert metadata.concurrency is None


class TestDataLoaderReloadWithDetails:
    """Tests for DataLoader.reload_with_details method."""

    def test_reload_adds_per_request_data(self, single_run_dir: Path) -> None:
        """Test that reload_with_details loads per-request data."""
        loader = DataLoader()
        run = loader.load_run(single_run_dir, load_per_request_data=False)

        assert run.requests is None

        reloaded_run = loader.reload_with_details(single_run_dir)

        assert reloaded_run.requests is not None
        assert not reloaded_run.requests.empty
        assert reloaded_run.metadata.run_name == run.metadata.run_name
        assert reloaded_run.metadata.model == run.metadata.model
        assert reloaded_run.metadata.concurrency == run.metadata.concurrency
        assert reloaded_run.aggregated == run.aggregated

    def test_reload_nonexistent_path_raises_error(self, tmp_path: Path) -> None:
        """Test reload_with_details with nonexistent path."""
        loader = DataLoader()
        fake_path = tmp_path / "nonexistent_run"

        with pytest.raises(DataLoadError, match="Run path does not exist"):
            loader.reload_with_details(fake_path)


class TestDataLoaderLoadPerRequestData:
    """Tests for DataLoader.load_run with load_per_request_data parameter."""

    def test_load_run_without_per_request_data(self, single_run_dir: Path) -> None:
        """Test loading run without per-request data."""
        loader = DataLoader()
        run = loader.load_run(single_run_dir, load_per_request_data=False)

        assert run.metadata is not None
        assert run.aggregated is not None
        assert run.requests is None

    def test_load_run_with_per_request_data(self, single_run_dir: Path) -> None:
        """Test loading run with per-request data."""
        loader = DataLoader()
        run = loader.load_run(single_run_dir, load_per_request_data=True)

        assert run.metadata is not None
        assert run.aggregated is not None
        assert run.requests is not None
        assert not run.requests.empty

    def test_load_without_per_request_data_missing_jsonl_raises_error(
        self, tmp_path: Path, sample_aggregated_data: dict[str, Any]
    ) -> None:
        """Test that load_per_request_data=False still validates JSONL exists."""
        run_dir = tmp_path / "test_run"
        run_dir.mkdir()

        json_path = run_dir / "profile_export_aiperf.json"
        json_path.write_bytes(orjson.dumps(sample_aggregated_data))

        loader = DataLoader()
        with pytest.raises(DataLoadError, match="Required JSONL file not found"):
            loader.load_run(run_dir, load_per_request_data=False)


class TestDataLoaderDurationCalculation:
    """Tests for duration calculation in metadata extraction."""

    def test_duration_calculated_from_requests(self, tmp_path: Path) -> None:
        """Test duration is calculated from request timestamps."""
        loader = DataLoader()
        run_path = tmp_path / "test_run"

        requests_df = pd.DataFrame(
            {
                "request_start_ns": pd.to_datetime(
                    [1000000000, 1500000000, 2000000000], unit="ns", utc=True
                ),
                "request_end_ns": pd.to_datetime(
                    [1200000000, 1700000000, 3000000000], unit="ns", utc=True
                ),
            }
        )

        metadata = loader._extract_metadata(run_path, requests_df, {})

        assert metadata.duration_seconds is not None
        assert metadata.duration_seconds == 2.0

    def test_duration_none_when_no_requests(self, tmp_path: Path) -> None:
        """Test duration is None when no request data available."""
        loader = DataLoader()
        run_path = tmp_path / "test_run"

        metadata = loader._extract_metadata(run_path, None, {})

        assert metadata.duration_seconds is None

    def test_duration_none_when_empty_dataframe(self, tmp_path: Path) -> None:
        """Test duration is None with empty DataFrame."""
        loader = DataLoader()
        run_path = tmp_path / "test_run"
        requests_df = pd.DataFrame()

        metadata = loader._extract_metadata(run_path, requests_df, {})

        assert metadata.duration_seconds is None

    def test_duration_with_missing_timestamp_columns(self, tmp_path: Path) -> None:
        """Test duration is None when timestamp columns are missing."""
        loader = DataLoader()
        run_path = tmp_path / "test_run"
        requests_df = pd.DataFrame({"other_column": [1, 2, 3]})

        metadata = loader._extract_metadata(run_path, requests_df, {})

        assert metadata.duration_seconds is None


class TestDataLoaderExtractTelemetry:
    """Tests for DataLoader.extract_telemetry_data method."""

    def test_extract_telemetry_with_valid_data(self) -> None:
        """Test extracting telemetry data from valid aggregated data."""
        loader = DataLoader()
        aggregated = {
            "telemetry_data": {
                "summary": {
                    "start_time": "2025-01-01T00:00:00",
                    "end_time": "2025-01-01T01:00:00",
                },
                "endpoints": {
                    "endpoint1": {"gpus": {"0": {}, "1": {}}},
                },
            }
        }

        result = loader.extract_telemetry_data(aggregated)

        assert result is not None
        assert "summary" in result
        assert "endpoints" in result
        assert len(result["endpoints"]) == 1

    def test_extract_telemetry_missing_data(self) -> None:
        """Test extracting telemetry when data is missing."""
        loader = DataLoader()
        aggregated = {"other_field": "value"}

        result = loader.extract_telemetry_data(aggregated)

        assert result is None

    def test_extract_telemetry_empty_aggregated(self) -> None:
        """Test extracting telemetry from empty aggregated dict."""
        loader = DataLoader()
        aggregated = {}

        result = loader.extract_telemetry_data(aggregated)

        assert result is None

    def test_extract_telemetry_wrong_structure(self) -> None:
        """Test extracting telemetry with wrong data structure."""
        loader = DataLoader()
        aggregated = {"telemetry_data": "not a dict"}

        result = loader.extract_telemetry_data(aggregated)

        assert result is None

    def test_extract_telemetry_missing_summary_key(self) -> None:
        """Test extracting telemetry when summary key is missing."""
        loader = DataLoader()
        aggregated = {
            "telemetry_data": {
                "endpoints": {"endpoint1": {}},
            }
        }

        result = loader.extract_telemetry_data(aggregated)

        assert result is None

    def test_extract_telemetry_missing_endpoints_key(self) -> None:
        """Test extracting telemetry when endpoints key is missing."""
        loader = DataLoader()
        aggregated = {
            "telemetry_data": {
                "summary": {"start_time": "2025-01-01T00:00:00"},
            }
        }

        result = loader.extract_telemetry_data(aggregated)

        assert result is None

    def test_extract_telemetry_empty_endpoints(self) -> None:
        """Test extracting telemetry with empty endpoints dict."""
        loader = DataLoader()
        aggregated = {
            "telemetry_data": {
                "summary": {"start_time": "2025-01-01T00:00:00"},
                "endpoints": {},
            }
        }

        result = loader.extract_telemetry_data(aggregated)

        # Should be valid even with empty endpoints
        assert result is not None
        assert result["endpoints"] == {}


class TestDataLoaderGetTelemetrySummary:
    """Tests for DataLoader.get_telemetry_summary method."""

    def test_get_telemetry_summary_valid(self) -> None:
        """Test getting telemetry summary from valid data."""
        loader = DataLoader()
        aggregated = {
            "telemetry_data": {
                "summary": {
                    "start_time": "2025-01-01T00:00:00",
                    "end_time": "2025-01-01T01:00:00",
                    "endpoints_configured": 2,
                    "endpoints_successful": 2,
                },
                "endpoints": {},
            }
        }

        result = loader.get_telemetry_summary(aggregated)

        assert result is not None
        assert result["start_time"] == "2025-01-01T00:00:00"
        assert result["end_time"] == "2025-01-01T01:00:00"

    def test_get_telemetry_summary_no_telemetry(self) -> None:
        """Test getting telemetry summary when no telemetry data exists."""
        loader = DataLoader()
        aggregated = {}

        result = loader.get_telemetry_summary(aggregated)

        assert result is None


class TestDataLoaderCalculateGPUCount:
    """Tests for DataLoader.calculate_gpu_count_from_telemetry method."""

    def test_calculate_gpu_count_single_endpoint(self) -> None:
        """Test calculating GPU count from single endpoint."""
        loader = DataLoader()
        aggregated = {
            "telemetry_data": {
                "summary": {},
                "endpoints": {
                    "endpoint1": {
                        "gpus": {
                            "0": {"gpu_index": 0},
                            "1": {"gpu_index": 1},
                            "2": {"gpu_index": 2},
                        }
                    },
                },
            }
        }

        result = loader.calculate_gpu_count_from_telemetry(aggregated)

        assert result == 3

    def test_calculate_gpu_count_multiple_endpoints(self) -> None:
        """Test calculating GPU count from multiple endpoints."""
        loader = DataLoader()
        aggregated = {
            "telemetry_data": {
                "summary": {},
                "endpoints": {
                    "endpoint1": {
                        "gpus": {
                            "0": {},
                            "1": {},
                        }
                    },
                    "endpoint2": {
                        "gpus": {
                            "0": {},
                            "1": {},
                            "2": {},
                        }
                    },
                },
            }
        }

        result = loader.calculate_gpu_count_from_telemetry(aggregated)

        assert result == 5

    def test_calculate_gpu_count_no_telemetry(self) -> None:
        """Test calculating GPU count when no telemetry data exists."""
        loader = DataLoader()
        aggregated = {}

        result = loader.calculate_gpu_count_from_telemetry(aggregated)

        assert result is None

    def test_calculate_gpu_count_zero_gpus(self) -> None:
        """Test calculating GPU count when endpoints have no GPUs."""
        loader = DataLoader()
        aggregated = {
            "telemetry_data": {
                "summary": {},
                "endpoints": {
                    "endpoint1": {"gpus": {}},
                },
            }
        }

        result = loader.calculate_gpu_count_from_telemetry(aggregated)

        assert result is None

    def test_calculate_gpu_count_invalid_endpoints_structure(self) -> None:
        """Test calculating GPU count with invalid endpoints structure."""
        loader = DataLoader()
        aggregated = {
            "telemetry_data": {
                "summary": {},
                "endpoints": "not a dict",
            }
        }

        result = loader.calculate_gpu_count_from_telemetry(aggregated)

        assert result is None

    def test_calculate_gpu_count_invalid_endpoint_data(self) -> None:
        """Test calculating GPU count when endpoint data is not a dict."""
        loader = DataLoader()
        aggregated = {
            "telemetry_data": {
                "summary": {},
                "endpoints": {
                    "endpoint1": "not a dict",
                    "endpoint2": {"gpus": {"0": {}, "1": {}}},
                },
            }
        }

        result = loader.calculate_gpu_count_from_telemetry(aggregated)

        # Should count GPUs from valid endpoint only
        assert result == 2

    def test_calculate_gpu_count_missing_gpus_key(self) -> None:
        """Test calculating GPU count when gpus key is missing."""
        loader = DataLoader()
        aggregated = {
            "telemetry_data": {
                "summary": {},
                "endpoints": {
                    "endpoint1": {"other_field": "value"},
                },
            }
        }

        result = loader.calculate_gpu_count_from_telemetry(aggregated)

        assert result is None

    def test_calculate_gpu_count_gpus_not_dict(self) -> None:
        """Test calculating GPU count when gpus field is not a dict."""
        loader = DataLoader()
        aggregated = {
            "telemetry_data": {
                "summary": {},
                "endpoints": {
                    "endpoint1": {"gpus": "not a dict"},
                },
            }
        }

        result = loader.calculate_gpu_count_from_telemetry(aggregated)

        assert result is None


class TestDataLoaderAddDerivedMetrics:
    """Tests for DataLoader._add_all_derived_metrics method."""

    def test_add_derived_metrics_with_telemetry(self) -> None:
        """Test adding derived metrics when telemetry data is available."""
        loader = DataLoader()
        aggregated = {
            "telemetry_data": {
                "summary": {},
                "endpoints": {
                    "endpoint1": {
                        "gpus": {"0": {}, "1": {}},
                    }
                },
            },
            "output_token_throughput": {"value": 1000.0, "unit": "tokens/s"},
        }

        loader._add_all_derived_metrics(aggregated)

        # Should have added per-GPU metric
        assert "output_token_throughput_per_gpu" in aggregated
        assert aggregated["output_token_throughput_per_gpu"]["value"] == 500.0
        assert aggregated["output_token_throughput_per_gpu"]["unit"] == "tokens/sec/gpu"

    def test_add_derived_metrics_no_telemetry(self) -> None:
        """Test adding derived metrics when no telemetry data exists."""
        loader = DataLoader()
        aggregated = {
            "output_token_throughput": {"value": 1000.0, "unit": "tokens/s"},
        }

        loader._add_all_derived_metrics(aggregated)

        # Should not add per-GPU metrics without telemetry
        assert "output_token_throughput_per_gpu" not in aggregated

    def test_add_derived_metrics_zero_gpus(self) -> None:
        """Test adding derived metrics when GPU count is zero."""
        loader = DataLoader()
        aggregated = {
            "telemetry_data": {
                "summary": {},
                "endpoints": {
                    "endpoint1": {"gpus": {}},
                },
            },
            "output_token_throughput": {"value": 1000.0, "unit": "tokens/s"},
        }

        loader._add_all_derived_metrics(aggregated)

        # Should not add per-GPU metrics with zero GPUs
        assert "output_token_throughput_per_gpu" not in aggregated

    def test_add_derived_metrics_missing_base_metric(self) -> None:
        """Test adding derived metrics when base metric is missing."""
        loader = DataLoader()
        aggregated = {
            "telemetry_data": {
                "summary": {},
                "endpoints": {
                    "endpoint1": {"gpus": {"0": {}, "1": {}}},
                },
            },
            # Missing output_token_throughput
        }

        loader._add_all_derived_metrics(aggregated)

        # Should handle gracefully - derived metric won't be added
        assert "output_token_throughput_per_gpu" not in aggregated


class TestDataLoaderGetAvailableMetrics:
    """Tests for DataLoader.get_available_metrics method."""

    def test_get_available_metrics_with_data(self, tmp_path: Path) -> None:
        """Test getting available metrics from loaded run."""
        loader = DataLoader()
        aggregated = {
            "time_to_first_token": {"value": 45.0, "unit": "ms"},
            "inter_token_latency": {"value": 20.0, "unit": "ms"},
            "request_latency": {"value": 500.0, "unit": "ms"},
        }

        run_data = RunData(
            metadata=RunMetadata(
                run_name="test", run_path=tmp_path, duration_seconds=None
            ),
            requests=None,
            aggregated=aggregated,
        )

        result = loader.get_available_metrics(run_data)

        assert "display_names" in result
        assert "units" in result
        assert len(result["display_names"]) == 3
        assert len(result["units"]) == 3
        assert "time_to_first_token" in result["display_names"]
        assert result["units"]["time_to_first_token"] == "ms"

    def test_get_available_metrics_no_aggregated_data(self, tmp_path: Path) -> None:
        """Test getting available metrics when no aggregated data exists."""
        loader = DataLoader()
        run_data = RunData(
            metadata=RunMetadata(
                run_name="test", run_path=tmp_path, duration_seconds=None
            ),
            requests=None,
            aggregated={},
        )

        result = loader.get_available_metrics(run_data)

        assert result["display_names"] == {}
        assert result["units"] == {}

    def test_get_available_metrics_filters_non_metrics(self, tmp_path: Path) -> None:
        """Test that non-metric keys are filtered out."""
        loader = DataLoader()
        aggregated = {
            "time_to_first_token": {"value": 45.0, "unit": "ms"},
            "input_config": {"some": "config"},  # Should be filtered
            "was_cancelled": False,  # Should be filtered
            "error_summary": [],  # Should be filtered
        }

        run_data = RunData(
            metadata=RunMetadata(
                run_name="test", run_path=tmp_path, duration_seconds=None
            ),
            requests=None,
            aggregated=aggregated,
        )

        result = loader.get_available_metrics(run_data)

        assert "time_to_first_token" in result["display_names"]
        assert "input_config" not in result["display_names"]
        assert "was_cancelled" not in result["display_names"]
        assert "error_summary" not in result["display_names"]

    def test_get_available_metrics_handles_non_dict_values(
        self, tmp_path: Path
    ) -> None:
        """Test that non-dict values are skipped."""
        loader = DataLoader()
        aggregated = {
            "time_to_first_token": {"value": 45.0, "unit": "ms"},
            "some_string": "value",
            "some_number": 123,
            "some_list": [1, 2, 3],
        }

        run_data = RunData(
            metadata=RunMetadata(
                run_name="test", run_path=tmp_path, duration_seconds=None
            ),
            requests=None,
            aggregated=aggregated,
        )

        result = loader.get_available_metrics(run_data)

        assert "time_to_first_token" in result["display_names"]
        assert "some_string" not in result["display_names"]
        assert "some_number" not in result["display_names"]
        assert "some_list" not in result["display_names"]

    def test_get_available_metrics_requires_unit_field(self, tmp_path: Path) -> None:
        """Test that metrics without unit field are skipped."""
        loader = DataLoader()
        aggregated = {
            "time_to_first_token": {"value": 45.0, "unit": "ms"},
            "metric_without_unit": {"value": 100.0},
        }

        run_data = RunData(
            metadata=RunMetadata(
                run_name="test", run_path=tmp_path, duration_seconds=None
            ),
            requests=None,
            aggregated=aggregated,
        )

        result = loader.get_available_metrics(run_data)

        assert "time_to_first_token" in result["display_names"]
        assert "metric_without_unit" not in result["display_names"]


class TestDataLoaderLoadRunWithGPUTelemetry:
    """Tests for DataLoader.load_run method with GPU telemetry data."""

    def test_load_run_includes_gpu_telemetry(self, single_run_dir: Path) -> None:
        """Test that load_run successfully loads GPU telemetry data from real fixtures."""
        loader = DataLoader()
        run = loader.load_run(single_run_dir)

        assert run.gpu_telemetry is not None
        assert isinstance(run.gpu_telemetry, pd.DataFrame)
        assert len(run.gpu_telemetry) > 0

        # Verify timestamp_s column exists with relative timestamps
        # Note: timestamps can be negative if GPU telemetry started before first request
        assert "timestamp_s" in run.gpu_telemetry.columns
        assert pd.api.types.is_numeric_dtype(run.gpu_telemetry["timestamp_s"])

        # Verify rich telemetry fields from real data
        expected_fields = [
            "gpu_index",
            "gpu_utilization",
            "gpu_power_usage",
            "gpu_memory_used",
            "gpu_temperature",
            "sm_clock_frequency",
            "memory_clock_frequency",
            "dcgm_url",
            "gpu_uuid",
            "hostname",
        ]
        for field in expected_fields:
            assert field in run.gpu_telemetry.columns, f"Missing field: {field}"


class TestDataLoaderLoadGPUTelemetryJSONL:
    """Tests for DataLoader._load_gpu_telemetry_jsonl method."""

    def test_load_gpu_telemetry_with_relative_timestamps(
        self, single_run_dir: Path
    ) -> None:
        """Test loading GPU telemetry with relative timestamp conversion using real data."""
        loader = DataLoader()
        jsonl_path = single_run_dir / "gpu_telemetry_export.jsonl"

        run_start_time_ns = 1762551530946074466

        df = loader._load_gpu_telemetry_jsonl(jsonl_path, run_start_time_ns)

        assert df is not None
        assert len(df) > 0
        assert "timestamp_s" in df.columns
        assert "gpu_utilization" in df.columns
        assert "gpu_power_usage" in df.columns
        assert "gpu_memory_used" in df.columns
        assert "gpu_temperature" in df.columns

        # Check relative timestamp conversion
        # Note: First timestamp can be negative if telemetry started before first request
        assert pd.api.types.is_numeric_dtype(df["timestamp_s"])
        # Timestamps should be monotonically increasing
        assert df["timestamp_s"].is_monotonic_increasing

    def test_load_gpu_telemetry_with_absolute_timestamps(
        self, single_run_dir: Path
    ) -> None:
        """Test loading GPU telemetry with absolute timestamps (no run start time) using real data."""
        loader = DataLoader()
        jsonl_path = single_run_dir / "gpu_telemetry_export.jsonl"

        df = loader._load_gpu_telemetry_jsonl(jsonl_path, run_start_time_ns=None)

        assert df is not None
        assert len(df) > 0
        assert "timestamp_s" in df.columns
        # Check absolute timestamp in seconds (should be large value)
        assert df["timestamp_s"].iloc[0] > 1e9

    def test_load_gpu_telemetry_missing_file(self, tmp_path: Path) -> None:
        """Test loading GPU telemetry when file doesn't exist."""
        loader = DataLoader()
        jsonl_path = tmp_path / "nonexistent.jsonl"

        df = loader._load_gpu_telemetry_jsonl(jsonl_path)

        assert df is None

    def test_load_gpu_telemetry_corrupted_lines(self, tmp_path: Path) -> None:
        """Test loading GPU telemetry with corrupted lines."""
        loader = DataLoader()
        jsonl_path = tmp_path / "gpu_telemetry_export.jsonl"

        with open(jsonl_path, "w") as f:
            f.write(
                orjson.dumps(
                    {
                        "timestamp_ns": 1000000000000,
                        "gpu_index": 0,
                        "telemetry_data": {"gpu_utilization": 80.5},
                    }
                ).decode("utf-8")
                + "\n"
            )
            f.write("{ invalid json }\n")
            f.write(
                orjson.dumps(
                    {
                        "timestamp_ns": 1000000100000,
                        "gpu_index": 1,
                        "telemetry_data": {"gpu_utilization": 75.0},
                    }
                ).decode("utf-8")
                + "\n"
            )

        df = loader._load_gpu_telemetry_jsonl(jsonl_path)

        # Should load 2 valid records, skip 1 corrupted
        assert df is not None
        assert len(df) == 2

    def test_load_gpu_telemetry_empty_file(self, tmp_path: Path) -> None:
        """Test loading GPU telemetry from empty file."""
        loader = DataLoader()
        jsonl_path = tmp_path / "gpu_telemetry_export.jsonl"
        jsonl_path.write_text("")

        df = loader._load_gpu_telemetry_jsonl(jsonl_path)

        assert df is None

    def test_load_gpu_telemetry_missing_timestamp_field(self, tmp_path: Path) -> None:
        """Test loading GPU telemetry when timestamp_ns field is missing."""
        loader = DataLoader()
        jsonl_path = tmp_path / "gpu_telemetry_export.jsonl"

        telemetry_data = [
            {
                "gpu_index": 0,
                "telemetry_data": {"gpu_utilization": 80.5},
                # Missing timestamp_ns
            },
        ]

        with open(jsonl_path, "w") as f:
            for record in telemetry_data:
                f.write(orjson.dumps(record).decode("utf-8") + "\n")

        df = loader._load_gpu_telemetry_jsonl(jsonl_path)

        # Should still load, but timestamp_s won't be present
        assert df is not None
        assert len(df) == 1
        assert "gpu_index" in df.columns

    def test_load_gpu_telemetry_flattens_nested_data(
        self, single_run_dir: Path
    ) -> None:
        """Test that telemetry_data dict is flattened into main record using real data."""
        loader = DataLoader()
        jsonl_path = single_run_dir / "gpu_telemetry_export.jsonl"

        df = loader._load_gpu_telemetry_jsonl(jsonl_path)

        assert df is not None
        assert len(df) > 0

        # Verify top-level metadata fields are present
        assert "timestamp_ns" in df.columns
        assert "gpu_index" in df.columns
        assert "dcgm_url" in df.columns
        assert "gpu_uuid" in df.columns
        assert "hostname" in df.columns

        # Verify telemetry_data fields are flattened to top level
        assert "gpu_utilization" in df.columns
        assert "gpu_power_usage" in df.columns
        assert "gpu_memory_used" in df.columns
        assert "gpu_temperature" in df.columns
        assert "sm_clock_frequency" in df.columns
        assert "memory_clock_frequency" in df.columns

        # telemetry_data should not be a nested column
        assert "telemetry_data" not in df.columns


class TestRunDataGetMetric:
    """Tests for RunData.get_metric method."""

    def test_get_metric_from_nested_metrics_dict(self, tmp_path: Path) -> None:
        """Test getting metric from nested 'metrics' structure with dict."""
        aggregated = {
            "metrics": {
                "time_to_first_token": {"avg": 45.0, "unit": "ms", "std": 5.0},
                "request_latency": {"avg": 500.0, "unit": "ms", "std": 50.0},
            }
        }

        run_data = RunData(
            metadata=RunMetadata(
                run_name="test", run_path=tmp_path, duration_seconds=None
            ),
            requests=None,
            aggregated=aggregated,
        )

        metric = run_data.get_metric("time_to_first_token")
        assert metric is not None
        assert metric["avg"] == 45.0
        assert metric["unit"] == "ms"
        assert metric["std"] == 5.0

    def test_get_metric_from_flat_structure_dict(self, tmp_path: Path) -> None:
        """Test getting metric from flat top-level structure with dict."""
        aggregated = {
            "time_to_first_token": {"avg": 45.0, "unit": "ms", "std": 5.0},
            "request_latency": {"avg": 500.0, "unit": "ms", "std": 50.0},
            "input_config": {"some": "config"},
        }

        run_data = RunData(
            metadata=RunMetadata(
                run_name="test", run_path=tmp_path, duration_seconds=None
            ),
            requests=None,
            aggregated=aggregated,
        )

        metric = run_data.get_metric("time_to_first_token")
        assert metric is not None
        assert metric["avg"] == 45.0
        assert metric["unit"] == "ms"

    def test_get_metric_returns_none_for_missing_metric(self, tmp_path: Path) -> None:
        """Test that get_metric returns None for metric that doesn't exist."""
        aggregated = {
            "time_to_first_token": {"avg": 45.0, "unit": "ms"},
        }

        run_data = RunData(
            metadata=RunMetadata(
                run_name="test", run_path=tmp_path, duration_seconds=None
            ),
            requests=None,
            aggregated=aggregated,
        )

        metric = run_data.get_metric("nonexistent_metric")
        assert metric is None

    def test_get_metric_returns_none_for_empty_aggregated(self, tmp_path: Path) -> None:
        """Test that get_metric returns None when aggregated is empty."""
        run_data = RunData(
            metadata=RunMetadata(
                run_name="test", run_path=tmp_path, duration_seconds=None
            ),
            requests=None,
            aggregated={},
        )

        metric = run_data.get_metric("time_to_first_token")
        assert metric is None

    def test_get_metric_prefers_nested_metrics_over_flat(self, tmp_path: Path) -> None:
        """Test that nested 'metrics' structure is preferred over flat when both exist."""
        aggregated = {
            "time_to_first_token": {"avg": 100.0, "unit": "ms"},
            "metrics": {
                "time_to_first_token": {"avg": 45.0, "unit": "ms", "std": 5.0},
            },
        }

        run_data = RunData(
            metadata=RunMetadata(
                run_name="test", run_path=tmp_path, duration_seconds=None
            ),
            requests=None,
            aggregated=aggregated,
        )

        metric = run_data.get_metric("time_to_first_token")
        assert metric is not None
        assert metric["avg"] == 45.0

    def test_get_metric_handles_metric_result_objects(
        self, single_run_dir: Path
    ) -> None:
        """Test get_metric with MetricResult objects from real data."""
        loader = DataLoader()
        run = loader.load_run(single_run_dir)

        if "metrics" in run.aggregated:
            metric = run.get_metric("time_to_first_token")
            assert metric is not None
            assert hasattr(metric, "avg") or "avg" in metric
