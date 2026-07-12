# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned projection from Config v2 into the native single-run contract.

Config v2 remains the user-facing and orchestration schema. Protocol v1 keeps
its resolved compatibility projection while protocol v2 has a separate,
side-effect-free authored projection. Raw Pydantic dumps are deliberately not
the outer process boundary in either version.
"""

from __future__ import annotations

import copy
import multiprocessing
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from aiperf.common.environment import Environment
from aiperf.config.dataset import FileDataset, PublicDataset, SyntheticDataset
from aiperf.config.phases import (
    ConcurrencyPhase,
    ConstantPhase,
    FixedSchedulePhase,
    GammaPhase,
    PoissonPhase,
    UserCentricPhase,
)

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


RUNNER_PROTOCOL_VERSION = 1
RUNNER_PROTOCOL_V2 = 2
SERVER_METRICS_PARQUET_WIRE_PATH = Path(".aiperf-server-metrics-parquet-wire.jsonl")
RunnerOperationV2 = Literal["validate", "execute"]
_DISTRIBUTION_ID_PREFIX = "blake3:"
_DISTRIBUTION_ID_HEX_LENGTH = 64


class RustWireError(ValueError):
    """Raised when a resolved Config v2 run cannot enter the native ABI."""


def build_authored_run_request(
    run: BenchmarkRun,
    *,
    operation: RunnerOperationV2,
    expected_distribution_id: str,
) -> dict[str, Any]:
    """Project one Config-v2 run into the strict protocol-v2 envelope.

    This projection reads authored/structurally normalized configuration only:
    it never reads ``BenchmarkRun.resolved``, loads a dataset, inspects the
    filesystem, warms a tokenizer, creates artifacts, or starts a worker. The
    Factory selection remains open, while every built-in policy object is
    explicitly lowered into its stable runner-owned shape.
    """
    if operation not in ("validate", "execute"):
        raise RustWireError(
            f"protocol-v2 operation must be 'validate' or 'execute', got {operation!r}"
        )
    if not _is_distribution_id(expected_distribution_id):
        raise RustWireError(
            "protocol-v2 expected_distribution_id must be 'blake3:' followed by "
            "exactly 64 lowercase hexadecimal characters"
        )

    return {
        "protocol_version": RUNNER_PROTOCOL_V2,
        "operation": operation,
        "expected_distribution_id": expected_distribution_id,
        "run": _authored_run_v2(run),
    }


def _is_distribution_id(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith(_DISTRIBUTION_ID_PREFIX):
        return False
    hexadecimal = value[len(_DISTRIBUTION_ID_PREFIX) :]
    return len(hexadecimal) == _DISTRIBUTION_ID_HEX_LENGTH and all(
        character in "0123456789abcdef" for character in hexadecimal
    )


def _authored_run_v2(run: BenchmarkRun) -> dict[str, Any]:
    """Assemble the provisional v2 run body in one alignment point."""
    cfg = run.cfg
    dataset = cfg.get_default_dataset()
    identity: dict[str, Any] = {
        "benchmark_id": run.benchmark_id,
        "label": run.label,
        "trial": run.trial,
    }
    _set_optional(identity, "random_seed", run.random_seed)
    if run.variation is not None:
        identity["variation"] = {
            "index": run.variation.index,
            "label": run.variation.label,
            "values": copy.deepcopy(run.variation.values),
        }

    sidecars = _authored_sidecars(run)
    endpoint_profiles = [
        {
            "id": "default",
            **_authored_endpoint(cfg.endpoint, include_readiness=True),
        }
    ]
    endpoint_profiles.extend(
        {
            "id": profile_id,
            **_authored_endpoint(profile, include_readiness=True),
        }
        for profile_id, profile in cfg.endpoint_profiles.items()
    )
    return {
        "identity": identity,
        "artifact_target": str(run.artifact_dir),
        "models": _authored_models(cfg),
        "endpoints": {"profiles": endpoint_profiles},
        "backend": {
            "type": str(cfg.backend.type),
            "config": copy.deepcopy(cfg.backend.config),
        },
        "workload": _authored_workload(run, dataset),
        "metrics": _authored_metrics(cfg),
        "artifacts": _authored_artifacts(run),
        "sidecars": sidecars,
    }


def _authored_models(cfg: Any) -> dict[str, Any]:
    """Project model selection without resolving tokenizers or model aliases."""
    return {
        "strategy": str(cfg.models.strategy),
        "items": [
            {
                "name": item.name,
                **({"weight": item.weight} if item.weight is not None else {}),
            }
            for item in cfg.models.items
        ],
    }


def _authored_endpoint(
    endpoint: Any, *, include_readiness: bool = False
) -> dict[str, Any]:
    """Project raw endpoint policy without consulting endpoint metadata."""
    result: dict[str, Any] = {
        "urls": list(endpoint.urls),
        "type": str(endpoint.type),
        "streaming": endpoint.streaming,
        "use_legacy_max_tokens": endpoint.use_legacy_max_tokens,
        "use_server_token_count": endpoint.use_server_token_count,
        "timeout_seconds": endpoint.timeout,
        "connection_reuse": str(endpoint.connection_reuse),
        "download_video_content": endpoint.download_video_content,
        "extra": copy.deepcopy(endpoint.extra),
        "headers": dict(endpoint.headers),
        "http2": False,
    }
    _set_optional(result, "path", endpoint.path)
    _set_optional(result, "api_key", endpoint.api_key)
    _set_optional(result, "session_header", endpoint.session_header)
    if endpoint.request_content_type is not None:
        result["request_content_type"] = {
            "application/json": "application_json",
            "multipart/form-data": "multipart_form_data",
        }[str(endpoint.request_content_type)]
    if endpoint.template is not None:
        result["template"] = endpoint.template.body
        result["response_field"] = endpoint.template.response_field
    if include_readiness:
        result["wait_for_model_timeout"] = endpoint.wait_for_model_timeout
        result["wait_for_model_interval"] = endpoint.wait_for_model_interval
        result["wait_for_model_mode"] = endpoint.wait_for_model_mode
    return result


def _authored_workload(run: BenchmarkRun, dataset: Any) -> dict[str, Any]:
    """Build an open workload selection around unprepared authored inputs."""
    cfg = run.cfg
    if cfg.workload is not None:
        workload_type = str(cfg.workload.type)
        workload_config = copy.deepcopy(cfg.workload.config)
    else:
        workload_type = _default_workload_type(cfg, dataset)
        workload_config = {}

    # Evaluation owns a deliberately different provider-neutral authored
    # shape. In particular, legacy dataset/tokenizer/phase fields are not
    # smuggled into its strict factory config, and Python never adds worker
    # launch coordinates. The selected runner provider factory is the only
    # authority that decodes ``evaluation``.
    if workload_type == "evaluation":
        return {"type": workload_type, "config": workload_config}

    # Explicit extension-owned keys remain intact. The current Config-v2
    # scheduled fields fill only missing keys during the compatibility window.
    current_fields: dict[str, Any] = {
        "worker_count": _worker_count(cfg),
        "dataset": _authored_dataset_v2(run, dataset),
        "tokenizer": _authored_tokenizer_v2(cfg),
        "phases": [_phase(phase) for phase in cfg.phases],
    }
    if cfg.accuracy is not None and cfg.accuracy.enabled:
        accuracy = _authored_model_dump(cfg.accuracy)
        accuracy["python_executable"] = _python_executable()
        accuracy["worker_module"] = "aiperf.accuracy.worker"
        current_fields["accuracy"] = accuracy
    for name, value in current_fields.items():
        workload_config.setdefault(name, value)
    return {"type": workload_type, "config": workload_config}


def _authored_dataset_v2(run: BenchmarkRun, dataset: Any) -> dict[str, Any]:
    """Project one dataset without acquiring or parsing its source.

    Named public datasets are part of the Python-owned Config catalog.  They
    are expanded here exactly once into explicit source coordinates and the
    selected native loader ID.  The Rust dataset adapter remains the sole
    downloader/parser and therefore receives no Python-materialized rows.
    """
    if isinstance(dataset, PublicDataset):
        return _public_dataset(run, dataset)

    if isinstance(dataset, FileDataset):
        native_format, options = _native_file_format(str(dataset.format))
        if native_format == "mooncake_trace":
            options.setdefault("block_size", 512)
        elif native_format == "bailian_trace":
            options.setdefault("block_size", 16)
        if dataset.inter_turn_delay_cap_seconds is not None:
            options["inter_turn_delay_cap_seconds"] = (
                dataset.inter_turn_delay_cap_seconds
            )
        result: dict[str, Any] = {
            "type": "file",
            "format": native_format,
            "sampling": str(dataset.sampling),
            "options": options,
        }
        _set_optional(result, "entries", dataset.entries)
        _set_optional(result, "random_seed", dataset.random_seed)
        if dataset.osl is not None:
            result["osl"] = _distribution(dataset.osl)
        if dataset.synthesis is not None:
            result["synthesis"] = dataset.synthesis.model_dump(
                mode="json", exclude_none=True
            )
        if dataset.path is not None:
            path = dataset.path.expanduser()
            result["path"] = str(
                path if path.is_absolute() else (Path.cwd() / path).absolute()
            )
        else:
            result["records"] = copy.deepcopy(dataset.records)
        return result

    result = _authored_model_dump(dataset)
    result.pop("name", None)
    if isinstance(dataset, SyntheticDataset) and "turn_delay" in result:
        result["turn_delay_ms"] = result.pop("turn_delay")
    return result


def _authored_tokenizer_v2(cfg: Any) -> dict[str, Any]:
    """Retain authored tokenizer acquisition policy with an explicit identity."""
    primary_model = cfg.models.items[0].name
    if cfg.tokenizer is None:
        from aiperf.common.tokenizer_fake_names import is_fake_model_name

        return {
            "name": "builtin" if is_fake_model_name(primary_model) else primary_model,
            "revision": "main",
            "trust_remote_code": False,
            "apply_chat_template": False,
        }
    result = _authored_model_dump(cfg.tokenizer)
    result["name"] = cfg.tokenizer.name or primary_model
    return result


def _default_workload_type(cfg: Any, dataset: Any) -> str:
    """Select the compatibility workload from exact normalized authored state."""
    if str(getattr(dataset, "format", "")) == "dag_jsonl":
        return "graph"
    if cfg.accuracy is not None and cfg.accuracy.enabled:
        return "static_accuracy"
    return "scheduled"


def _authored_model_dump(value: Any) -> dict[str, Any]:
    """Serialize one authored config object without resolution or aliases."""
    return value.model_dump(
        mode="json",
        by_alias=False,
        exclude_none=True,
        context={"include_secrets": True},
    )


def _authored_metrics(cfg: Any) -> dict[str, Any]:
    """Project authored metric policy shared with the v1 native path."""
    result: dict[str, Any] = {"slos": dict(cfg.slos or {})}
    if cfg.artifacts.slice_duration is not None:
        result["slice_duration_seconds"] = cfg.artifacts.slice_duration
    return result


def _authored_artifacts(run: BenchmarkRun) -> dict[str, Any]:
    """Project output names and once-rendered user-file bytes.

    Python remains the owner of Jinja, context injection, scalar coercion, and
    JSON/YAML serialization. Rust receives exact UTF-8 content and owns only
    path-safe materialization after complete run preparation.
    """
    from aiperf.config.user_files import (
        build_user_file_context,
        derive_run_meta,
        render_user_files,
    )

    cfg = run.cfg
    root = cfg.artifacts.dir
    result: dict[str, Any] = {"trace": cfg.artifacts.trace}
    if cfg.artifacts.records is not False or cfg.artifacts.raw:
        result["records_path"] = str(
            cfg.artifacts.profile_export_jsonl_file.relative_to(root)
        )
    if cfg.artifacts.export_outputs_json:
        result["outputs_path"] = str(cfg.artifacts.outputs_json_file.relative_to(root))
    if cfg.artifacts.raw:
        result["raw_path"] = str(
            cfg.artifacts.profile_export_raw_jsonl_file.relative_to(root)
        )
    if cfg.artifacts.user_files:
        context = build_user_file_context(
            cfg,
            derive_run_meta(run.artifact_dir),
            run_dir=run.artifact_dir,
            variables=run.variables,
        )
        result["user_files"] = [
            {
                "path": user_file.path,
                "format": user_file.format,
                "content": user_file.content,
            }
            for user_file in render_user_files(cfg.artifacts.user_files, context)
        ]
    return result


def _authored_sidecars(run: BenchmarkRun) -> dict[str, Any]:
    """Project direct native sidecar inputs without starting their resources.

    Protocol v1 and v2 share these protocol-neutral source policies, but v2
    never constructs a v1 run request or enters the resolver chain. Runtime
    acquisition, reachability, cadence, and worker startup remain owned by the
    selected Rust sidecar adapters during pair preparation.
    """
    result: dict[str, Any] = {}
    gpu_telemetry = _gpu_telemetry(run, include_resolved_custom_metrics=False)
    if gpu_telemetry is not None:
        result["gpu_telemetry"] = gpu_telemetry
    network_latency = _network_latency(run)
    if network_latency is not None:
        result["network_latency"] = network_latency
    server_metrics = _server_metrics(run)
    if server_metrics is not None:
        result["server_metrics"] = server_metrics
    live_streaming = _live_streaming(run)
    if live_streaming is not None:
        result["live_streaming"] = live_streaming
    return result


def build_run_request(run: BenchmarkRun) -> dict[str, Any]:
    """Build the complete protocol-v1 request for one native benchmark.

    Every accepted field is written explicitly.  That makes additions to
    Config v2 fail closed until this projection and the Rust DTO are updated in
    the same change.
    """
    cfg = run.cfg
    dataset = cfg.get_default_dataset()
    validate_v1_selection(cfg)

    variation = run.variation
    run_wire: dict[str, Any] = {
        "benchmark_id": run.benchmark_id,
        "label": run.label,
        "trial": run.trial,
        "workers": _worker_count(cfg),
        "artifact_dir": str(run.artifact_dir),
        "models": _authored_models(cfg),
        "endpoint": _authored_endpoint(cfg.endpoint),
        "dataset": _dataset(run, dataset),
        "tokenizer": {
            "name": _tokenizer_source(run),
            **(
                {"apply_chat_template": True}
                if cfg.tokenizer is not None and cfg.tokenizer.apply_chat_template
                else {}
            ),
        },
        "phases": [_phase(phase) for phase in cfg.phases],
        "metrics": {
            "slos": dict(cfg.slos or {}),
            **(
                {"slice_duration_seconds": cfg.artifacts.slice_duration}
                if cfg.artifacts.slice_duration is not None
                else {}
            ),
        },
        "artifacts": {
            **(
                {
                    "records_path": _artifact_relative_path(
                        run.artifact_dir,
                        cfg.artifacts.profile_export_jsonl_file,
                    )
                }
                if cfg.artifacts.records is not False or cfg.artifacts.raw
                else {}
            ),
            **(
                {
                    "outputs_path": _artifact_relative_path(
                        run.artifact_dir,
                        cfg.artifacts.outputs_json_file,
                    )
                }
                if cfg.artifacts.export_outputs_json
                else {}
            ),
            **(
                {
                    "raw_path": _artifact_relative_path(
                        run.artifact_dir,
                        cfg.artifacts.profile_export_raw_jsonl_file,
                    )
                }
                if cfg.artifacts.raw
                else {}
            ),
            "trace": cfg.artifacts.trace,
        },
    }
    _set_optional(run_wire, "sweep_id", run.sweep_id)
    _set_optional(run_wire, "random_seed", run.random_seed)
    if variation is not None:
        run_wire["variation"] = {
            "index": variation.index,
            "label": variation.label,
            "values": dict(variation.values),
        }
    if cfg.accuracy is not None and cfg.accuracy.enabled:
        accuracy: dict[str, Any] = {
            "benchmark": str(cfg.accuracy.benchmark),
            "python_executable": _python_executable(),
            "worker_module": "aiperf.accuracy.worker",
        }
        _set_optional(accuracy, "tasks", cfg.accuracy.tasks)
        _set_optional(accuracy, "n_shots", cfg.accuracy.n_shots)
        _set_optional(accuracy, "enable_cot", cfg.accuracy.enable_cot)
        _set_optional(
            accuracy,
            "grader",
            str(cfg.accuracy.grader) if cfg.accuracy.grader is not None else None,
        )
        _set_optional(accuracy, "system_prompt", cfg.accuracy.system_prompt)
        run_wire["accuracy"] = accuracy
    gpu_telemetry = _gpu_telemetry(run)
    if gpu_telemetry is not None:
        run_wire["gpu_telemetry"] = gpu_telemetry
    network_latency = _network_latency(run)
    if network_latency is not None:
        run_wire["network_latency"] = network_latency
    server_metrics = _server_metrics(run)
    if server_metrics is not None:
        run_wire["server_metrics"] = server_metrics
    live_streaming = _live_streaming(run)
    if live_streaming is not None:
        run_wire["live_streaming"] = live_streaming
    return {"protocol_version": RUNNER_PROTOCOL_VERSION, "run": run_wire}


def validate_v1_selection(cfg: Any) -> None:
    """Fail before v1 resolution when Config v2 selected a v2-only surface."""
    if str(cfg.backend.type) != "online_http" or cfg.backend.config:
        raise RustWireError(
            "native runner protocol v1 supports only backend type 'online_http' "
            "with an empty config; authored backend selections require protocol v2"
        )
    if cfg.workload is not None:
        raise RustWireError(
            "native runner protocol v1 does not represent an explicit workload "
            "selection; remove benchmark.workload or use protocol v2 once its strict "
            "execution DTO is available"
        )
    if cfg.endpoint_profiles:
        raise RustWireError(
            "native runner protocol v1 represents only benchmark.endpoint; named "
            "endpoint_profiles require protocol v2"
        )
    if cfg.endpoint.wait_for_model_timeout > 0:
        raise RustWireError(
            "native runner protocol v1 cannot honor endpoint.wait_for_model_timeout; "
            "readiness must be implemented by the selected runner before this run can "
            "execute. Disable readiness or use a protocol-v2-capable runner once its "
            "strict execution DTO is available."
        )


def _worker_count(cfg: Any) -> int:
    """Resolve Config-v2 worker policy for the Rust execution data plane.

    This preserves the established WorkerManager sizing rules while moving the
    actual workers into Rust: explicit totals win; automatic sizing uses the
    configured CPU fraction and cap; concurrency phases avoid idle excess
    workers; and ``workers_min`` remains the final lower bound.
    """
    workers = cfg.runtime.workers
    if workers is None:
        workers = max(
            1,
            min(
                int(
                    multiprocessing.cpu_count()
                    * Environment.WORKER.CPU_UTILIZATION_FACTOR
                )
                - 1,
                Environment.WORKER.MAX_WORKERS_CAP,
            ),
        )
    profiling_concurrency = [
        phase.concurrency
        for phase in cfg.get_profiling_phases()
        if getattr(phase, "concurrency", None) is not None
    ]
    if profiling_concurrency:
        workers = min(workers, max(profiling_concurrency))
    return max(workers, cfg.runtime.workers_min or 1)


def _gpu_telemetry(
    run: BenchmarkRun,
    *,
    include_resolved_custom_metrics: bool = True,
) -> dict[str, Any] | None:
    """Lower GPU sources while retaining canonical Python-only collectors."""
    config = run.cfg.gpu_telemetry
    if not config.enabled:
        return None

    from aiperf.common.environment import Environment

    collector = str(config.collector)
    metrics_file = (
        str(config.metrics_file.expanduser().resolve())
        if config.metrics_file is not None
        else None
    )
    sources: list[dict[str, Any]] = []
    if collector == "dcgm":
        authored = [*Environment.GPU.DEFAULT_DCGM_ENDPOINTS, *config.urls]
        urls = list(dict.fromkeys(_normalize_dcgm_url(url) for url in authored))
        for url in urls:
            if metrics_file is None:
                sources.append({"type": "dcgm", "url": url})
            else:
                sources.append(
                    {
                        "type": "python",
                        "collector": collector,
                        "url": url,
                        "metrics_file": metrics_file,
                        "python_executable": _python_executable(),
                        "worker_module": "aiperf.gpu_telemetry.worker",
                    }
                )
    else:
        source: dict[str, Any] = {
            "type": "python",
            "collector": collector,
            "python_executable": _python_executable(),
            "worker_module": "aiperf.gpu_telemetry.worker",
        }
        _set_optional(source, "metrics_file", metrics_file)
        sources.append(source)

    result: dict[str, Any] = {
        "collection_interval_ns": _positive_seconds_to_ns(
            Environment.GPU.COLLECTION_INTERVAL,
            "GPU collection interval",
        ),
        "request_timeout_ns": _positive_seconds_to_ns(
            Environment.GPU.REACHABILITY_TIMEOUT,
            "GPU reachability timeout",
        ),
        "records_path": _artifact_relative_path(
            run.artifact_dir,
            run.cfg.artifacts.profile_export_gpu_telemetry_jsonl_file,
        ),
        "sources": sources,
    }
    custom_metrics = (
        run.resolved.gpu_custom_metrics or [] if include_resolved_custom_metrics else []
    )
    if custom_metrics:
        result["custom_metrics"] = [
            {
                "header": header,
                "name": name,
                "unit": _native_gpu_unit(unit),
            }
            for header, name, unit in custom_metrics
        ]
    return result


def _network_latency(run: BenchmarkRun) -> dict[str, Any] | None:
    """Lower fixed or profiling-bounded RTT calibration into native policy."""
    config = run.cfg.network_latency
    if not config.enabled:
        return None
    if config.mean_ms is not None:
        return {
            "mean_rtt_ns": _nonnegative_seconds_to_ns(
                config.mean_ms / 1000.0,
                "network latency fixed mean",
            )
        }

    from aiperf.common.environment import Environment

    return {
        "probe": {
            "ping_interval_ns": _positive_seconds_to_ns(
                config.ping_interval,
                "network latency ping interval",
            ),
            "connect_timeout_ns": _positive_seconds_to_ns(
                Environment.NETWORK_LATENCY.CONNECT_TIMEOUT,
                "network latency connect timeout",
            ),
            "complete_topup_timeout_ns": _nonnegative_seconds_to_ns(
                Environment.NETWORK_LATENCY.COMPLETE_TOPUP_TIMEOUT,
                "network latency completion top-up timeout",
            ),
            "min_successful_samples": Environment.NETWORK_LATENCY.MIN_SAMPLES,
            "records_path": _artifact_relative_path(
                run.artifact_dir,
                run.cfg.artifacts.network_latency_export_jsonl_file,
            ),
        }
    }


def _server_metrics(run: BenchmarkRun) -> dict[str, Any] | None:
    """Lower server scraping while keeping Config-v2 discovery Python-owned."""
    config = run.cfg.server_metrics
    if not config.enabled:
        return None

    from aiperf.common.environment import Environment
    from aiperf.common.metric_utils import normalize_metrics_endpoint_url

    urls = list(
        dict.fromkeys(
            normalize_metrics_endpoint_url(url)
            for url in [*run.cfg.endpoint.urls, *config.urls]
        )
    )
    formats = [str(value) for value in config.formats]
    result: dict[str, Any] = {
        "collection_interval_ns": _positive_seconds_to_ns(
            Environment.SERVER_METRICS.COLLECTION_INTERVAL,
            "server metrics collection interval",
        ),
        "reachability_timeout_ns": _positive_seconds_to_ns(
            Environment.SERVER_METRICS.REACHABILITY_TIMEOUT,
            "server metrics reachability timeout",
        ),
        "urls": urls,
        "formats": formats,
    }
    if "jsonl" in formats:
        result["jsonl_path"] = _artifact_relative_path(
            run.artifact_dir,
            run.cfg.artifacts.server_metrics_export_jsonl_file,
        )
    if "parquet" in formats:
        result["parquet_wire_path"] = str(SERVER_METRICS_PARQUET_WIRE_PATH)
    return result


def _live_streaming(run: BenchmarkRun) -> dict[str, Any] | None:
    """Lower OTel/live-MLflow into the supervised Python extension ABI."""
    config = run.cfg
    if not config.otel.collector_enabled and not config.mlflow.enabled:
        return None

    from aiperf.common.environment import Environment
    from aiperf.config.otel import normalize_otel_metrics_url

    metrics_url = normalize_otel_metrics_url(config.otel.metrics_url)

    return {
        "python_executable": _python_executable(),
        "worker_module": "aiperf.post_processors.native_streaming_worker",
        "buffer_capacity": Environment.OTEL.MAX_BUFFERED_RECORDS,
        "otel": {
            "metrics_url": metrics_url,
            "stream_metrics_enabled": config.otel.stream_metrics_enabled,
            "stream_timing_enabled": config.otel.stream_timing_enabled,
            "custom_resource_attributes": dict(config.otel.custom_resource_attributes),
            "gen_ai_provider": config.otel.gen_ai_provider,
        },
        "mlflow": {
            "tracking_uri": config.mlflow.tracking_uri,
            "experiment": config.mlflow.experiment,
            "run_name": config.mlflow.run_name,
            "tags": (
                dict(config.mlflow.tags) if config.mlflow.tags is not None else None
            ),
            "parent_run_id": config.mlflow.parent_run_id,
            "artifact_globs": (
                list(config.mlflow.artifact_globs)
                if config.mlflow.artifact_globs is not None
                else None
            ),
        },
    }


def _normalize_dcgm_url(url: str) -> str:
    normalized = url.rstrip("/")
    return normalized if normalized.endswith("/metrics") else f"{normalized}/metrics"


def _positive_seconds_to_ns(value: float, label: str) -> int:
    if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
        raise RustWireError(f"{label} must be positive, got {value!r}")
    nanoseconds = round(float(value) * 1_000_000_000)
    if nanoseconds <= 0 or nanoseconds > 2**63 - 1:
        raise RustWireError(f"{label} is outside the native nanosecond range")
    return nanoseconds


def _nonnegative_seconds_to_ns(value: float, label: str) -> int:
    if not isinstance(value, int | float) or isinstance(value, bool) or value < 0:
        raise RustWireError(f"{label} must be non-negative, got {value!r}")
    nanoseconds = round(float(value) * 1_000_000_000)
    if nanoseconds < 0 or nanoseconds > 2**63 - 1:
        raise RustWireError(f"{label} is outside the native nanosecond range")
    return nanoseconds


def _python_executable() -> str:
    """Return an absolute interpreter path without dereferencing virtualenv links."""
    executable = Path(sys.executable).expanduser()
    if not executable.is_absolute():
        executable = Path.cwd() / executable
    return str(executable.absolute())


_GPU_UNIT_NAMES = {
    "COUNT": "count",
    "KILOBYTES": "kilobyte",
    "MEGABYTES": "megabyte",
    "GIGABYTES": "gigabyte",
    "MICROSECONDS": "microsecond",
    "MILLISECONDS": "millisecond",
    "SECONDS": "second",
    "PERCENT": "percent",
    "WATT": "watt",
    "JOULE": "joule",
    "MEGAJOULE": "megajoule",
    "MEGAHERTZ": "megahertz",
    "GIGAHERTZ": "gigahertz",
    "CELSIUS": "celsius",
}


def _native_gpu_unit(unit: Any) -> str:
    name = getattr(unit, "name", None)
    try:
        return _GPU_UNIT_NAMES[name]
    except (KeyError, TypeError) as error:
        raise RustWireError(f"unsupported custom GPU metric unit {unit!s}") from error


def _dataset(run: BenchmarkRun, dataset: Any) -> dict[str, Any]:
    if isinstance(dataset, SyntheticDataset):
        return _synthetic_dataset(dataset)
    if isinstance(dataset, FileDataset):
        return _file_dataset(run, dataset)
    if isinstance(dataset, PublicDataset):
        return _public_dataset(run, dataset)
    raise RustWireError(
        f"native runner protocol v1 does not accept dataset type {dataset.type!s}"
    )


def _synthetic_dataset(dataset: SyntheticDataset) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": "synthetic",
        "entries": dataset.entries,
        "sampling": str(dataset.sampling),
        "turns": _distribution(dataset.turns or 1),
        "turn_delay_ms": _distribution(dataset.turn_delay or 0),
        "turn_delay_ratio": dataset.turn_delay_ratio,
    }
    _set_optional(result, "random_seed", dataset.random_seed)
    if dataset.prompts is not None:
        prompts: dict[str, Any] = {"batch_size": dataset.prompts.batch_size}
        if dataset.prompts.isl is not None:
            prompts["isl"] = _distribution(dataset.prompts.isl)
        if dataset.prompts.osl is not None:
            prompts["osl"] = _distribution(dataset.prompts.osl)
        _set_optional(prompts, "block_size", dataset.prompts.block_size)
        if dataset.prompts.sequence_distribution is not None:
            prompts["sequence_distribution"] = [
                {
                    "isl": _distribution(entry.isl),
                    "osl": _distribution(entry.osl),
                    "probability": entry.probability,
                }
                for entry in dataset.prompts.sequence_distribution
            ]
        result["prompts"] = prompts
    if dataset.prefix_prompts is not None:
        result["prefix_prompts"] = dataset.prefix_prompts.model_dump(
            mode="json", exclude_none=True
        )
    if dataset.images is not None:
        source = dataset.images.source
        source_value = (
            str(source.expanduser().resolve())
            if isinstance(source, Path)
            else str(source)
        )
        result["images"] = {
            "batch_size": dataset.images.batch_size,
            "width": _distribution(dataset.images.width),
            "height": _distribution(dataset.images.height),
            "format": str(dataset.images.format),
            "source": source_value,
            "source_sampling": str(dataset.images.source_sampling),
        }
    if dataset.audio is not None:
        result["audio"] = {
            "batch_size": dataset.audio.batch_size,
            "length": _distribution(dataset.audio.length),
            "format": str(dataset.audio.format),
            "sample_rates": list(dataset.audio.sample_rates),
            "depths": list(dataset.audio.depths),
            "channels": dataset.audio.channels,
        }
    if dataset.video is not None:
        video: dict[str, Any] = {
            "batch_size": dataset.video.batch_size,
            "duration": dataset.video.duration,
            "fps": dataset.video.fps,
            "format": str(dataset.video.format),
            "codec": dataset.video.codec,
            "synth_type": str(dataset.video.synth_type),
            "audio": {
                "sample_rate": dataset.video.audio.sample_rate,
                "channels": dataset.video.audio.channels,
                "depth": dataset.video.audio.depth,
            },
        }
        _set_optional(video, "width", dataset.video.width)
        _set_optional(video, "height", dataset.video.height)
        _set_optional(video["audio"], "codec", dataset.video.audio.codec)
        result["video"] = video
    if dataset.rankings is not None:
        result["rankings"] = {
            "passages": _distribution(dataset.rankings.passages),
            "passage_tokens": _distribution(dataset.rankings.passage_tokens),
            "query_tokens": _distribution(dataset.rankings.query_tokens),
        }
    return result


def _file_dataset(run: BenchmarkRun, dataset: FileDataset) -> dict[str, Any]:
    resolved_types = run.resolved.dataset_types or {}
    resolved_sampling = run.resolved.dataset_sampling_strategies or {}
    format_name = str(resolved_types.get(dataset.name, dataset.format))
    native_format, format_options = _native_file_format(format_name)
    if native_format == "mooncake_trace":
        format_options.setdefault("block_size", 512)
    elif native_format == "bailian_trace":
        format_options.setdefault("block_size", 16)
    if dataset.inter_turn_delay_cap_seconds is not None:
        format_options["inter_turn_delay_cap_seconds"] = (
            dataset.inter_turn_delay_cap_seconds
        )
    result: dict[str, Any] = {
        "type": "file",
        "format": native_format,
        "sampling": str(resolved_sampling.get(dataset.name, dataset.sampling)),
        "options": format_options,
    }
    _set_optional(result, "entries", dataset.entries)
    _set_optional(result, "random_seed", dataset.random_seed)
    if dataset.osl is not None:
        result["osl"] = _distribution(dataset.osl)
    if dataset.synthesis is not None:
        result["synthesis"] = dataset.synthesis.model_dump(
            mode="json", exclude_none=True
        )
    if dataset.path is not None:
        resolved_paths = run.resolved.dataset_file_paths or {}
        path = Path(resolved_paths.get(dataset.name, dataset.path)).resolve()
        result["path"] = str(path)
    else:
        result["records"] = dataset.records
    return result


_PUBLIC_NATIVE_FORMATS = {
    "aiperf.dataset.loader.exgentic:ExgenticDatasetLoader": "exgentic",
    "aiperf.dataset.loader.exgentic_v2:ExgenticV2DatasetLoader": "exgentic_v2",
    "aiperf.dataset.loader.sharegpt:ShareGPTLoader": "sharegpt",
    "aiperf.dataset.loader.hf_instruction_response:HFInstructionResponseDatasetLoader": (
        "hf_instruction_response"
    ),
    "aiperf.dataset.loader.hf_conversation:HFConversationDatasetLoader": (
        "hf_conversation"
    ),
    "aiperf.dataset.loader.mt_bench:MTBenchDatasetLoader": "mt_bench",
    "aiperf.dataset.loader.mmvu:MMVUDatasetLoader": "mmvu",
    "aiperf.dataset.loader.spec_bench:SpecBenchLoader": "spec_bench",
    "aiperf.dataset.loader.hf_asr:HFASRDatasetLoader": "hf_asr",
}


def _public_dataset(run: BenchmarkRun, dataset: PublicDataset) -> dict[str, Any]:
    from aiperf.plugin import plugins
    from aiperf.plugin.enums import PluginType

    loader_class = plugins.get_class(PluginType.PUBLIC_DATASET_LOADER, dataset.dataset)
    class_key = f"{loader_class.__module__}:{loader_class.__name__}"
    try:
        native_format = _PUBLIC_NATIVE_FORMATS[class_key]
    except KeyError as error:
        raise RustWireError(
            f"public dataset {dataset.dataset!s} uses loader {class_key!r}, "
            "which has no native loader registration"
        ) from error
    metadata = plugins.get_public_dataset_loader_metadata(dataset.dataset)
    options: dict[str, Any] = {}
    for name in (
        "prompt_column",
        "image_column",
        "video_column",
        "audio_column",
        "prompt_template",
        "conversation_column",
    ):
        _set_optional(options, name, getattr(metadata, name))
    if metadata.conversation_column is not None:
        options["message_content_key"] = metadata.message_content_key
    if metadata.multi_turn:
        options["multi_turn"] = True
    if dataset.filters:
        if native_format not in {"exgentic", "exgentic_v2"}:
            raise RustWireError(
                f"public dataset {dataset.dataset!s} does not accept dataset filters"
            )
        options.update(dataset.filters)
    if native_format in {"exgentic", "exgentic_v2"}:
        options["fixed_schedule"] = any(
            isinstance(phase, FixedSchedulePhase) for phase in run.cfg.phases
        )

    max_conversations = _public_max_conversations(
        run,
        dataset,
        streaming=metadata.streaming,
        entries_first=native_format in {"exgentic", "exgentic_v2"},
    )
    if native_format in {"exgentic", "exgentic_v2"} and max_conversations is None:
        raise RustWireError(
            "Exgentic requires a finite entries or profiling request count"
        )
    if max_conversations is not None:
        options["max_conversations"] = max_conversations

    if metadata.hf_dataset_name is not None:
        source: dict[str, Any] = {
            "type": "hugging_face",
            "dataset": metadata.hf_dataset_name,
            "subset": dataset.hf_subset or metadata.hf_subset or "default",
            "split": metadata.hf_split,
        }
        _set_optional(source, "revision", getattr(loader_class, "hf_revision", None))
    else:
        url = getattr(loader_class, "url", None)
        if not isinstance(url, str) or not url:
            raise RustWireError(
                f"public dataset {dataset.dataset!s} has neither Hugging Face "
                "coordinates nor a loader URL"
            )
        source = {"type": "url", "url": url}

    result: dict[str, Any] = {
        "type": "public",
        "name": str(dataset.dataset),
        "format": native_format,
        "source": source,
        "sampling": str(dataset.sampling),
        "options": options,
    }
    _set_optional(result, "entries", dataset.entries)
    _set_optional(result, "random_seed", dataset.random_seed)
    return result


def _public_max_conversations(
    run: BenchmarkRun,
    dataset: PublicDataset,
    *,
    streaming: bool,
    entries_first: bool,
) -> int | None:
    request_counts = [
        phase.requests
        for phase in run.cfg.get_profiling_phases()
        if phase.requests is not None
    ]
    request_cap = max(request_counts) if request_counts else None
    if entries_first and dataset.entries is not None:
        return dataset.entries
    if streaming and request_cap is not None:
        return request_cap
    return dataset.entries


def _native_file_format(format_name: str) -> tuple[str, dict[str, Any]]:
    if format_name == "burst_gpt_trace":
        return "burst_gpt", {}
    if not format_name.startswith("speed_bench_"):
        return format_name, {}
    suffix = format_name.removeprefix("speed_bench_")
    category = None
    for candidate in (
        "low_entropy",
        "mixed",
        "high_entropy",
        "coding",
        "humanities",
        "math",
        "multilingual",
        "qa",
        "rag",
        "reasoning",
        "roleplay",
        "stem",
        "summarization",
        "writing",
    ):
        if suffix == candidate or suffix.endswith(f"_{candidate}"):
            category = candidate
            break
    return "speed_bench", ({"category": category} if category else {})


def _tokenizer_source(run: BenchmarkRun) -> str:
    cfg = run.cfg.tokenizer
    primary_model = run.cfg.models.items[0].name
    resolved = run.resolved.tokenizer_names or {}
    name = resolved.get(primary_model) or (cfg.name if cfg is not None else None)
    if name is None:
        from aiperf.common.tokenizer_fake_names import is_fake_model_name

        name = "builtin" if is_fake_model_name(primary_model) else primary_model
    normalized = name.lower().replace("-", "_")
    if normalized in {
        "builtin",
        "o200k_base",
        "o200k_harmony",
        "cl100k_base",
        "p50k_base",
        "p50k_edit",
        "r50k_base",
    }:
        return normalized
    path = Path(name).expanduser()
    if path.exists():
        return str(path.resolve())
    try:
        from huggingface_hub import try_to_load_from_cache

        tokenizer_file = try_to_load_from_cache(
            name,
            "tokenizer.json",
            revision=cfg.revision if cfg is not None else "main",
        )
    except (ImportError, OSError, ValueError) as error:
        raise RustWireError(
            f"cannot resolve native tokenizer.json for {name!r}: {error}"
        ) from error
    if not isinstance(tokenizer_file, str):
        raise RustWireError(
            f"Python resolved tokenizer {name!r}, but its tokenizer.json is not cached"
        )
    return str(Path(tokenizer_file).resolve().parent)


def _phase(phase: Any) -> dict[str, Any]:
    common: dict[str, Any] = {
        "name": phase.name,
        "exclude_from_results": phase.exclude_from_results,
        "seamless": phase.seamless,
    }
    for name in (
        "requests",
        "sessions",
        "duration",
        "prefill_concurrency",
        "grace_period",
    ):
        _set_optional(common, name, getattr(phase, name))
    _set_optional(common, "concurrency_ramp", _ramp(phase.concurrency_ramp))
    _set_optional(common, "prefill_ramp", _ramp(phase.prefill_ramp))
    _set_optional(common, "rate_ramp", _ramp(getattr(phase, "rate_ramp", None)))
    if phase.cancellation is not None:
        common["cancellation"] = {
            "rate": phase.cancellation.rate,
            "delay": phase.cancellation.delay,
        }
    adaptive_scale = _adaptive_scale(phase)
    if adaptive_scale is not None:
        common["adaptive_scale"] = adaptive_scale

    if isinstance(phase, ConcurrencyPhase):
        return {"type": "concurrency", **common, "concurrency": phase.concurrency}
    if isinstance(phase, PoissonPhase):
        return _rate_phase("poisson", phase, common)
    if isinstance(phase, GammaPhase):
        result = _rate_phase("gamma", phase, common)
        _set_optional(result, "smoothness", phase.smoothness)
        return result
    if isinstance(phase, ConstantPhase):
        return _rate_phase("constant", phase, common)
    if isinstance(phase, UserCentricPhase):
        result = {
            "type": "user_centric",
            **common,
            "rate": phase.rate,
            "users": phase.users,
        }
        _set_optional(result, "concurrency", phase.concurrency)
        return result
    if isinstance(phase, FixedSchedulePhase):
        result = {
            "type": "fixed_schedule",
            **common,
            "auto_offset": phase.auto_offset,
        }
        _set_optional(result, "start_offset", phase.start_offset)
        _set_optional(result, "end_offset", phase.end_offset)
        return result
    raise RustWireError(
        f"native runner protocol v1 does not accept phase type {phase.type!s}"
    )


def _rate_phase(kind: str, phase: Any, common: dict[str, Any]) -> dict[str, Any]:
    result = {"type": kind, **common, "rate": phase.rate}
    _set_optional(result, "concurrency", phase.concurrency)
    return result


def _adaptive_scale(phase: Any) -> dict[str, Any] | None:
    enabled = bool(getattr(phase, "adaptive_scale", False))
    sla_filters = list(getattr(phase, "sla", ()) or ())
    if not enabled:
        if sla_filters:
            raise RustWireError(
                f"phase {phase.name!r} defines adaptive SLA filters without "
                "enabling adaptive_scale"
            )
        return None
    if phase.name != "profiling":
        raise RustWireError("adaptive_scale is supported only on profiling phases")

    variable = str(phase.adaptive_control_variable)
    maximum = phase.adaptive_control_max
    if maximum is None:
        maximum = {
            "concurrency": phase.concurrency,
            "prefill_concurrency": phase.prefill_concurrency,
            "request_rate": getattr(phase, "rate", None),
            "users": getattr(phase, "users", None),
        }.get(variable)
    if maximum is None:
        raise RustWireError(
            f"adaptive_scale control.max could not be resolved for {variable!r}"
        )

    return {
        "control_variable": variable,
        "minimum": phase.adaptive_control_min,
        "maximum": maximum,
        "assessment_period_seconds": phase.adaptive_assessment_period or 30.0,
        "sustain_duration_seconds": phase.adaptive_sustain_duration,
        "min_completed_requests": phase.adaptive_min_completed_requests,
        "strategy_type": phase.adaptive_scale_strategy_type,
        "step_policy": phase.adaptive_scale_step_policy,
        "base_step": phase.adaptive_scale_base_step,
        "max_step_multiplier": phase.adaptive_scale_max_step_multiplier,
        "step_percent": phase.adaptive_scale_step_percent,
        "sla_filters": [
            {
                "metric_tag": sla.metric_tag,
                "stat": sla.stat,
                "op": sla.op,
                "threshold": sla.threshold,
            }
            for sla in sla_filters
        ],
    }


def _ramp(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return {"duration": value.duration, "strategy": str(value.strategy)}


def _distribution(value: Any) -> dict[str, Any]:
    if isinstance(value, int | float):
        return {"value": float(value)}
    dumped = value.model_dump(mode="json", exclude_none=True)
    if "peaks" in dumped:
        dumped["peaks"] = [
            {
                "distribution": _distribution(peak.distribution),
                "weight": peak.weight,
            }
            for peak in value.peaks
        ]
    return dumped


def _set_optional(target: dict[str, Any], name: str, value: Any) -> None:
    if value is not None:
        target[name] = value


def _artifact_relative_path(root: Path, output: Path) -> str:
    root_path = root.resolve()
    output_path = output.resolve()
    try:
        return str(output_path.relative_to(root_path))
    except ValueError as error:
        raise RustWireError(
            f"native artifact path {output_path} is outside run directory {root_path}"
        ) from error
