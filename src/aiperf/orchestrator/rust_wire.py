# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serialize BenchmarkRun into the strict protocol-v2 runner envelope.

The wire ``run`` body uses BenchmarkRun field names (``artifact_dir``, ``cfg``,
``resolved``, …). Nested factory inputs inside ``cfg`` (phases, datasets,
tokenizer) are lowered into the shapes linked runner factories already decode;
Python-only presentation sections are stripped via an explicit exclude set.
"""

from __future__ import annotations

import copy
import multiprocessing
import os
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


RUNNER_PROTOCOL_V2 = 2
SERVER_METRICS_PARQUET_WIRE_PATH = Path(".aiperf-server-metrics-parquet-wire.jsonl")
RunnerOperationV2 = Literal["validate", "execute"]

# Presentation / Python-orchestration sections that must not cross the wire.
_CFG_WIRE_EXCLUDE = frozenset(
    {
        "logging",
        "wandb",
        "otel",
        "mlflow",
    }
)


class RustWireError(ValueError):
    """Raised when a BenchmarkRun cannot enter the native wire envelope."""


def build_authored_run_request(
    run: BenchmarkRun,
    *,
    operation: RunnerOperationV2,
) -> dict[str, Any]:
    """Build the protocol-v2 envelope around a BenchmarkRun-shaped dump."""
    if operation not in ("validate", "execute"):
        raise RustWireError(
            f"protocol-v2 operation must be 'validate' or 'execute', got {operation!r}"
        )
    return {
        "protocol_version": RUNNER_PROTOCOL_V2,
        "operation": operation,
        "run": dump_benchmark_run(run),
    }


def dump_benchmark_run(run: BenchmarkRun) -> dict[str, Any]:
    """Serialize one BenchmarkRun for the runner with nested factory lowering."""
    payload = run.model_dump(
        mode="json",
        by_alias=False,
        exclude_none=False,
        context={"include_secrets": True},
    )
    cfg = payload.get("cfg")
    if not isinstance(cfg, dict):
        raise RustWireError("BenchmarkRun.cfg must dump to an object")

    for key in _CFG_WIRE_EXCLUDE:
        cfg.pop(key, None)
    if cfg.get("workload") is None:
        cfg.pop("workload", None)

    # Nested factory inputs keep their runner-owned shapes until PhaseSpec /
    # dataset adapters accept raw Config dumps directly.
    dataset = run.cfg.get_default_dataset()
    cfg["models"] = _authored_models(run.cfg)
    cfg["endpoint"] = _authored_endpoint(run.cfg.endpoint, include_readiness=True)
    cfg["endpoint_profiles"] = {
        profile_id: _authored_endpoint(profile, include_readiness=True)
        for profile_id, profile in run.cfg.endpoint_profiles.items()
    }
    cfg["datasets"] = [_authored_dataset_v2(run, dataset)]
    cfg.pop("dataset", None)
    cfg["phases"] = [_phase(phase) for phase in run.cfg.phases]
    cfg["tokenizer"] = _authored_tokenizer_v2(run.cfg)
    cfg["transport"] = _inline_transport(run.cfg.transport)
    cfg["runtime"] = {
        key: value
        for key, value in (cfg.get("runtime") or {}).items()
        if key in {"workers", "workers_max", "workers_min"}
    }
    cfg["metrics"] = _authored_metrics(run.cfg)
    cfg["artifacts"] = _authored_artifacts(run)
    if "sidecars" not in cfg:
        # Sidecar presence remains Config-section driven; the runner adapter
        # materializes them from these lowered blocks when present.
        cfg["sidecars"] = _authored_sidecars(run)
    return payload


_DYNOSIM_TRANSPORTS = frozenset({"dynosim_offline", "dynosim_online"})


def _is_dynosim(run: BenchmarkRun) -> bool:
    """Whether this run targets an in-process Dynamo co-simulation transport.

    The dynosim transports open no sockets and hard-reject online sidecars and
    common request/raw/output/user-file artifacts (they emit backend Dynamo
    artifacts instead). Callers use this to force those inputs off so a plain
    ``aiperf profile --config dynosim_*`` run validates without the author
    having to disable gpu/server telemetry and the per-record export by hand.
    """
    return str(run.cfg.transport.type) in _DYNOSIM_TRANSPORTS


def _inline_transport(transport: Any) -> dict[str, Any]:
    """Keep Config's inline discriminated transport object on the wire."""
    config = transport.model_dump(
        mode="json",
        by_alias=False,
        exclude_unset=True,
        exclude_none=True,
    )
    # Discriminator must always be present even when it is the model default.
    config["type"] = str(transport.type)
    if "required_features" in config:
        config["required_features"] = sorted(config["required_features"])
    return config


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
        "ssl_verify": endpoint.ssl_verify,
        "connection_limit": endpoint.connection_limit,
        "keepalive_timeout": endpoint.keepalive_timeout,
        "download_video_content": endpoint.download_video_content,
        "extra": copy.deepcopy(endpoint.extra),
        "headers": dict(endpoint.headers),
        "http2": endpoint.http2,
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
        workload_type = ""
        workload_config = {}

    authored_dataset = _authored_dataset_v2(run, dataset)
    if not workload_type:
        workload_type = _default_workload_type(cfg, authored_dataset)

    # Explicit extension-owned keys remain intact. The current Config-v2
    # scheduled fields fill only missing keys during the compatibility window.
    current_fields: dict[str, Any] = {
        "worker_count": _worker_count(cfg),
        "dataset": authored_dataset,
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
        native_format, options = _native_file_format(_resolved_file_format(dataset))
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
            synthesis = dataset.synthesis.model_dump(mode="json", exclude_none=True)
            # Null is semantic for recorded graphs: it disables idle-gap
            # compression, while an absent field selects the 60s default.
            synthesis["idle_gap_cap_seconds"] = dataset.synthesis.idle_gap_cap_seconds
            result["synthesis"] = synthesis
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
    from aiperf.common.tokenizer_fake_names import is_fake_model_name

    primary_model = cfg.models.items[0].name
    if cfg.tokenizer is None:
        return {
            "name": "builtin" if is_fake_model_name(primary_model) else primary_model,
            "revision": "main",
            "trust_remote_code": False,
            "apply_chat_template": False,
        }
    result = _authored_model_dump(cfg.tokenizer)
    result["name"] = cfg.tokenizer.name or (
        "builtin" if is_fake_model_name(primary_model) else primary_model
    )
    return result


def _default_workload_type(cfg: Any, dataset: Any) -> str:
    """Select the compatibility workload from exact normalized authored state."""
    format_name = (
        dataset.get("format", "")
        if isinstance(dataset, dict)
        else getattr(dataset, "format", "")
    )
    if str(format_name) in {
        "dag_jsonl",
        "weka_trace",
        "dynamo_trace",
    }:
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
    """Project authored metric policy into the v2 native request."""
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
    # The runner emits inputs.json (per-session formatted request payloads) on
    # the native path, replacing the legacy DatasetManager writer. Always
    # request it so introspection tooling and GenAI-Perf compatibility keep
    # working, mirroring the legacy always-on behavior.
    result["inputs_path"] = str(cfg.artifacts.inputs_json_file.relative_to(root))
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
    if _is_dynosim(run):
        # DynoSim emits backend Dynamo artifacts (report/per-request JSONL on the
        # transport block) and hard-rejects the common request/raw/output/
        # user-file projection. Force those off — the ``--export-level`` default
        # re-enables the per-record file even when the YAML disables it, so the
        # only reliable place to drop them is here at the wire choke point. The
        # runner permits ``inputs_path``, so introspection still works.
        for key in ("records_path", "raw_path", "outputs_path", "user_files"):
            result.pop(key, None)
        result["trace"] = False
    return result


def _authored_sidecars(run: BenchmarkRun) -> dict[str, Any]:
    """Project direct native sidecar inputs without starting their resources.

    Every sidecar input is projected directly into the v2 run request; the
    projection never enters a resolver chain. Runtime acquisition,
    reachability, cadence, and worker startup remain owned by the selected
    Rust sidecar adapters during pair preparation.
    """
    # DynoSim opens no sockets: force every online sidecar off (gpu telemetry
    # and server metrics both default on and would otherwise hard-fail native
    # validation). The co-simulation carries no live resources to scrape.
    if _is_dynosim(run):
        return {}
    result: dict[str, Any] = {}
    content_server = _content_server()
    if content_server is not None:
        result["content_server"] = content_server
    gpu_telemetry = _gpu_telemetry(run)
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


def _content_server() -> dict[str, Any] | None:
    """Project the public content-server environment into strict native policy.

    File externalization is active only when both ENABLED and CONTENT_DIR are
    set, exactly as in ``DatasetManager._get_content_server_kwargs`` on
    ``ajc/content-server``. An enabled server with no directory still receives
    a temporary serving root but synthetic media remains inline.
    """
    settings = Environment.CONTENT_SERVER
    if not settings.ENABLED:
        return None
    result: dict[str, Any] = {
        "host": settings.HOST,
        "port": settings.PORT,
        "max_tracked_records": settings.MAX_TRACKED_RECORDS,
    }
    if settings.CONTENT_DIR:
        content_dir = Path(settings.CONTENT_DIR).expanduser()
        result["content_dir"] = str(
            content_dir
            if content_dir.is_absolute()
            else (Path.cwd() / content_dir).absolute()
        )
    return result


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


def _gpu_telemetry(run: BenchmarkRun) -> dict[str, Any] | None:
    """Lower GPU sources while retaining canonical Python-only collectors.

    The resolved custom-metric catalog (validated from the ``--gpu-telemetry``
    CSV by ``GpuMetricsResolver``) must reach the runner: the Python collector
    worker scrapes those custom DCGM fields, but the runner's
    ``GpuTelemetryAccumulator`` only summarizes signals it has a registered
    spec for. Omitting the catalog silently drops every custom metric from the
    native-v2 report even though the values are scraped.
    """
    config = run.cfg.gpu_telemetry
    if not config.enabled:
        return None

    from aiperf.common.environment import Environment

    collector = str(config.collector)
    # Authored projection must not inspect the filesystem, so the metrics-file
    # path is made absolute without dereferencing symlinks (matching
    # ``_content_server`` / ``_python_executable``), not via ``Path.resolve``.
    if config.metrics_file is not None:
        metrics_path = config.metrics_file.expanduser()
        metrics_file: str | None = str(
            metrics_path
            if metrics_path.is_absolute()
            else (Path.cwd() / metrics_path).absolute()
        )
    else:
        metrics_file = None
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
            run.cfg.artifacts.dir,
            run.cfg.artifacts.profile_export_gpu_telemetry_jsonl_file,
        ),
        "sources": sources,
    }
    custom_metrics = run.resolved.gpu_custom_metrics or []
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
                run.cfg.artifacts.dir,
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
            run.cfg.artifacts.dir,
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


_PUBLIC_NATIVE_FORMATS = {
    "aiperf.dataset.loader.recorded_graph:WekaTraceNativeLoader": "weka_trace",
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
        entries_first=native_format in {"exgentic", "exgentic_v2", "weka_trace"},
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


def _resolved_file_format(dataset: FileDataset) -> str:
    """Return the loader-format name to ship for a file dataset.

    Pydantic defaults ``format`` to SINGLE_TURN, so a config that never set
    ``--custom-dataset-type`` still carries that value. Shipping it verbatim
    would send the wrong loader to the runner and make it JSON-parse a
    non-JSONL file (e.g. a BurstGPT CSV, whose header fails ``expected value
    at line 1 column 1``). When the user did not explicitly select a format
    and the source is a real file, run the same structural ``can_load``
    detection the composer would, and ship the detected format instead. An
    explicit format, inline records, or an undetected file all fall through
    to the authored value unchanged, so this is a no-op for every path that
    already worked.
    """
    if "format" in dataset.model_fields_set or dataset.path is None:
        return str(dataset.format)
    path = dataset.path.expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).absolute()
    if not path.is_file():
        return str(dataset.format)
    from aiperf.config.dataset.resolver import DatasetResolver

    detected, _ = DatasetResolver._detect_type(str(path))
    return str(detected) if detected is not None else str(dataset.format)


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
        f"native runner protocol v2 does not accept phase type {phase.type!s}"
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
    """Relative artifact path via pure normalization, never a filesystem probe.

    The authored projection is side-effect-free, so both paths are normalized
    with ``os.path.abspath`` (lexical ``..`` collapse plus cwd-anchoring, no
    ``lstat``/``readlink``) rather than ``Path.resolve``. ``relative_to`` is
    itself purely lexical. For ordinary non-symlink inputs this yields the
    identical string ``resolve`` would have produced.
    """
    root_path = Path(os.path.abspath(root))
    output_path = Path(os.path.abspath(output))
    try:
        return str(output_path.relative_to(root_path))
    except ValueError as error:
        raise RustWireError(
            f"native artifact path {output_path} is outside artifact directory {root_path}"
        ) from error
