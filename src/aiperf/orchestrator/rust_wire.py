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
        if key in {"workers", "workers_max", "workers_min", "cells"}
    }
    cfg["metrics"] = _authored_metrics(run.cfg)
    cfg["artifacts"] = _authored_artifacts(run)
    cfg["export"] = _export(run)
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
    _set_optional(result, "uds_path", endpoint.uds_path)
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
            _project_trajectory_knobs(synthesis, run)
            # Resolved dataset-sampling strategy governs WHICH recorded-graph
            # template a freed recycle lane serves in the native graph phase
            # runtime (`_draw_index`). It rides the synthesis block beside the
            # trajectory-start knobs because the runner binds every recorded knob
            # through `RecordedTraceInputConfig`. `sequential` (the trace default)
            # keeps the byte-unchanged cursor draw.
            synthesis["dataset_sampling_strategy"] = str(dataset.sampling)
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


def _project_trajectory_knobs(synthesis: dict[str, Any], run: BenchmarkRun) -> None:
    """Project the recorded-graph trajectory-start (t*) window and derived seed.

    The trajectory-start window is a per-run Config knob
    (``cfg.trajectory_start_min_ratio`` / ``cfg.trajectory_start_max_ratio``,
    auto-filled by the submission scenario). It is carried on the recorded
    dataset's ``synthesis`` block beside ``idle_gap_cap_seconds`` because the
    native runner binds every recorded-replay knob through
    ``RecordedTraceInputConfig`` (``rust/runner/src/graph_input.rs``), which reads
    the synthesis block. The t* sampling seed derives from the run seed
    (``BenchmarkRun.random_seed``); zero when unset selects the runner's
    run-root-derived default. ``getattr`` guards the Config attributes so this
    projection is a no-op on configs that predate the scenario fields (they stay
    at the runner's disabled 0.0/0.0/0 defaults).
    """
    synthesis["trajectory_start_min_ratio"] = float(
        getattr(run.cfg, "trajectory_start_min_ratio", 0.0) or 0.0
    )
    synthesis["trajectory_start_max_ratio"] = float(
        getattr(run.cfg, "trajectory_start_max_ratio", 0.0) or 0.0
    )
    synthesis["t_star_random_seed"] = int(run.random_seed or 0)


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


def _sketch_metrics_enabled() -> bool:
    """Whether the opt-in bounded-memory sketch metrics mode is active.

    Sourced from ``Environment.METRICS.SKETCH`` (env ``AIPERF_METRICS_SKETCH`` or
    the ``--sketch-metrics`` flag, which sets the same setting). In this mode the
    native runner streams each per-record value into a t-digest instead of
    retaining it, so per-record artifacts and per-row-only outputs are impossible.
    """
    return bool(Environment.METRICS.SKETCH)


def _authored_metrics(cfg: Any) -> dict[str, Any]:
    """Project authored metric policy into the v2 native request."""
    result: dict[str, Any] = {"slos": dict(cfg.slos or {})}
    if cfg.artifacts.slice_duration is not None:
        result["slice_duration_seconds"] = cfg.artifacts.slice_duration
    if _sketch_metrics_enabled():
        # Bounded-memory retention: exact counts/sums/min/max, approximate
        # percentiles, no per-record values. The runner rejects per-record
        # artifacts and per-record OTLP under this flag; the frontend drops them
        # from the projection below so the run stays consistent.
        result["sketch"] = True
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
    records_formats = cfg.artifacts.records if cfg.artifacts.records else []
    # `raw` requires the per-record base file, so it forces the JSONL on even when
    # the records formats list omits "jsonl".
    if "jsonl" in records_formats or cfg.artifacts.raw:
        result["records_path"] = str(
            cfg.artifacts.profile_export_jsonl_file.relative_to(root)
        )
    if "parquet" in records_formats:
        result["records_parquet_path"] = str(
            cfg.artifacts.profile_export_parquet_file.relative_to(root)
        )
    if "csv" in records_formats:
        result["records_csv_path"] = str(
            cfg.artifacts.profile_export_records_csv_file.relative_to(root)
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
        for key in (
            "records_path",
            "records_parquet_path",
            "records_csv_path",
            "raw_path",
            "outputs_path",
            "user_files",
        ):
            result.pop(key, None)
        result["trace"] = False
    if _sketch_metrics_enabled():
        # Sketch retention keeps no per-record values, so per-record artifacts are
        # impossible; fail closed by dropping them here (the runner's validate_plan
        # rejects them defensively). ``inputs_path`` stays — it is built from
        # dispatch-time request payloads, not retained response records.
        for key in (
            "records_path",
            "records_parquet_path",
            "records_csv_path",
            "raw_path",
            "outputs_path",
        ):
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


def _export(run: BenchmarkRun) -> dict[str, Any]:
    """Project the native-Rust post-report export policy onto ``cfg.export``.

    Toggle/config passthrough only — the runner's ``aiperf::export`` plane owns
    all emission and serialization. genai-perf v1 compat is enabled when
    ``"genai_perf"`` is present in ``artifacts.summary``; OTel/MLflow projection is
    added when those sinks are migrated to the native path (they currently emit
    via the supervised Python live-streaming extension, see
    :func:`_live_streaming`).

    When the v1 sink is enabled, we additionally project the frontend-owned data
    values the native report alone cannot reconstruct, so the native
    ``profile_export_aiperf.{json,csv}`` reproduce the Python exporters
    byte-for-byte. The sink still owns all assembly and serialization; Python
    only supplies the metric-registry-derived headers/filters and the exact
    envelope JSON values (see :func:`_genai_perf_frontend_projection`).

    The timeslice and server-metrics sinks follow the same discipline: their
    metric VALUES come from the native report, but the frontend-owned envelope
    (`input_config` for both, plus `benchmark_id` for server metrics) is projected
    here so the two native artifacts reproduce their Python exporters byte-for-byte
    (see :func:`_timeslice_frontend_projection` /
    :func:`_server_metrics_frontend_projection`).
    """
    # Legacy escape hatch: when the native export plane is disabled
    # (AIPERF_RUNTIME_NATIVE_EXPORT=0) the runner drives no export sinks — an
    # empty block decodes to all-disabled defaults (see
    # ``rust/aiperf/src/export/mod.rs``) so it writes only the authoritative
    # native-v2 report, and the Python ExporterManager + live-streaming sidecar
    # become the single emitter for every artifact (see
    # :func:`_live_streaming` and
    # ``native_report.export_python_compatibility_reports``).
    if not Environment.RUNTIME.NATIVE_EXPORT:
        return {}

    config = run.cfg
    summary = config.artifacts.summary
    # The native v1 summary sink writes profile_export_aiperf.{json,csv} — the
    # exact files the legacy MetricsJsonExporter (gated on ``"json" in summary``,
    # see metrics_json_exporter.py) + MetricsCsvExporter produced. It is the sole
    # emitter of those artifacts on the native path, so it enables on the same
    # ``"json"`` signal the Python JSON exporter used (default ``["json"]``).
    genai_perf_enabled = isinstance(summary, list) and "json" in summary
    genai_perf: dict[str, Any] = {"enabled": genai_perf_enabled}
    if genai_perf_enabled:
        genai_perf.update(_genai_perf_frontend_projection(run))
    result: dict[str, Any] = {"genai_perf": genai_perf}

    timeslice = _timeslice_frontend_projection(run)
    if timeslice is not None:
        result["timeslice"] = timeslice
    server_metrics = _server_metrics_frontend_projection(run)
    if server_metrics is not None:
        result["server_metrics"] = server_metrics
    parquet = _parquet_frontend_projection(run)
    if parquet is not None:
        result["parquet"] = parquet
    accuracy_csv = _accuracy_csv_frontend_projection(run)
    if accuracy_csv is not None:
        result["accuracy_csv"] = accuracy_csv
    result["console_txt"] = _console_txt_frontend_projection(run)

    # Network sinks (OTLP/HTTP, MLflow, W&B) are driven by the native export
    # plane by default. Each block is projected only when its config signal is
    # present (collector/tracking_uri/project); the native Rust sink is then the
    # single emitter and the Python streaming sidecar + post-run uploaders are
    # skipped (see :func:`_live_streaming` and
    # ``native_report.export_python_compatibility_reports``). The legacy Python
    # paths return under AIPERF_RUNTIME_NATIVE_EXPORT=0 above.
    otel = _otel_frontend_projection(run)
    if otel is not None and not _sketch_metrics_enabled():
        # The native OTLP sink emits per-record histograms whose bucket counts come
        # from retained per-record values; sketch retention has none, and the
        # runner's validate_plan rejects ``native_otel_enabled`` under sketch. Drop
        # the whole OTLP block so the run stays consistent (aggregate metrics still
        # export to the JSON/CSV/console sinks).
        result["otel"] = otel
    mlflow = _mlflow_frontend_projection(run)
    if mlflow is not None:
        result["mlflow"] = mlflow
    wandb = _wandb_frontend_projection(run)
    if wandb is not None:
        result["wandb"] = wandb
    return result


def _console_txt_frontend_projection(run: BenchmarkRun) -> dict[str, Any]:
    """Project the fixed-width console-artifact policy onto ``cfg.export.console_txt``.

    The runner's ``aiperf::export::console_txt`` sink owns the full render
    (Rich box geometry, the error-summary table, and the warning/insight panels).
    Metric VALUES come from the native report; but the grouped-metrics-table
    CONTENT — which metric lands in which ``MetricConsoleGroup``, its display
    header, its display order, the INTERNAL/EXPERIMENTAL filter, and the table
    titles — is owned by the Python ``MetricRegistry`` and cannot be reconstructed
    from the report. Projecting it here makes the native
    ``profile_export_console.txt`` reproduce the Python ``ConsoleMetricsExporter``
    byte-for-byte instead of grouping/naming from the divergent Rust
    ``metrics_core`` catalog.

    Wire fields (``ConsoleTxtExportConfig``):

    * ``enabled`` — the ``.txt`` artifact is always written
      (``ExporterManager._write_console_txt``).
    * ``width`` — the fixed render width (``Environment.UI.CONSOLE_EXPORT_WIDTH``,
      the width the Python recording ``Console`` is pinned to).
    * ``dev`` — INTERNAL/EXPERIMENTAL visibility; off for the standard end-of-run
      tables (the dev-only tables are not ported).
    * ``title`` — the base metrics title (``ConsoleMetricsExporter._get_title``):
      ``NVIDIA AIPerf | <endpoint metrics_title>``, degrading to ``NVIDIA AIPerf``
      for a runner-only endpoint dialect with no Python metadata.
    * ``metrics`` — per-registered-tag ``{header, group, display_order, internal,
      experimental, error_only}`` from ``MetricRegistry.all_classes()``. An
      unregistered (native-only) tag is deliberately omitted; the sink then
      renders it in the DEFAULT group under its raw tag with no display order and
      no flag filter, exactly as Python's ``MetricResult`` for an unregistered
      tag does (``console_metrics_exporter._record_group`` / ``_should_show`` /
      ``_display_order`` and ``native_report._metric_result``).
    """
    from aiperf.common.enums import MetricFlags
    from aiperf.metrics.metric_registry import MetricRegistry

    metrics: dict[str, Any] = {}
    for metric_class in MetricRegistry.all_classes():
        meta: dict[str, Any] = {
            "header": metric_class.header,
            "group": metric_class.console_group.value,
        }
        if metric_class.display_order is not None:
            meta["display_order"] = metric_class.display_order
        if metric_class.has_flags(MetricFlags.INTERNAL):
            meta["internal"] = True
        if metric_class.has_flags(MetricFlags.EXPERIMENTAL):
            meta["experimental"] = True
        if metric_class.has_flags(MetricFlags.ERROR_ONLY):
            meta["error_only"] = True
        metrics[str(metric_class.tag)] = meta

    return {
        "enabled": True,
        "width": Environment.UI.CONSOLE_EXPORT_WIDTH,
        "dev": False,
        "title": _console_metrics_title(run),
        "metrics": metrics,
    }


def _console_metrics_title(run: BenchmarkRun) -> str:
    """Reproduce ``ConsoleMetricsExporter._get_title`` for the runner path.

    The console title is ``NVIDIA AIPerf | <metrics_title>`` where the metrics
    title comes from the endpoint plugin metadata; a runner-only endpoint dialect
    (``dynosim``, ``vllm_generate``) has no Python metadata entry, so the title
    degrades to the product default ``NVIDIA AIPerf`` rather than failing.
    """
    from aiperf.plugin import plugins

    try:
        metadata = plugins.get_endpoint_metadata(run.cfg.endpoint.type)
    except Exception:
        return "NVIDIA AIPerf"
    return f"NVIDIA AIPerf | {metadata.metrics_title}"


def _otel_frontend_projection(run: BenchmarkRun) -> dict[str, Any] | None:
    """Project ``cfg.otel`` onto the native OTLP/HTTP metrics sink block.

    Mirrors the four wire fields the Rust emitter decodes (see
    ``aiperf::export::otel::OtelExportConfig``): ``enabled`` (true iff a metrics
    URL is configured), the already-normalized ``endpoint`` (``cfg.otel``'s
    ``BeforeValidator`` guarantees the ``…/v1/metrics`` suffix), the optional
    ``provider`` (``gen_ai.provider.name`` override), and the ``resource_attributes``
    map. ``service.name=aiperf`` is added by the sink, so the map here reproduces
    the rest of ``OtelMetricsResultsProcessor._build_resource_attributes``
    (``otel_metrics_results_processor.py:433``): ``service.instance.id`` (the
    records-manager default, matching the Python fanout's ``service_id`` fallback),
    ``aiperf.benchmark.id``, ``aiperf.endpoint.type``, ``aiperf.model.name``, then
    the user ``custom_resource_attributes``.
    """
    otel = run.cfg.otel
    if otel.metrics_url is None:
        return None

    resource_attributes: dict[str, str] = {"service.instance.id": "records-manager"}
    if run.benchmark_id is not None:
        resource_attributes["aiperf.benchmark.id"] = run.benchmark_id
    resource_attributes["aiperf.endpoint.type"] = str(run.cfg.endpoint.type)
    model_names = run.cfg.get_model_names()
    if model_names:
        resource_attributes["aiperf.model.name"] = model_names[0]
    resource_attributes.update(otel.custom_resource_attributes)

    projection: dict[str, Any] = {
        "enabled": True,
        "endpoint": otel.metrics_url,
        "resource_attributes": resource_attributes,
    }
    if otel.gen_ai_provider is not None:
        projection["provider"] = otel.gen_ai_provider
    return projection


def _mlflow_frontend_projection(run: BenchmarkRun) -> dict[str, Any] | None:
    """Project ``cfg.mlflow`` onto the native MLflow run-tracker block.

    Enabled iff a tracking URI is set (``MLflowConfig.enabled``). Reproduces the
    fields the Rust uploader decodes (``aiperf::export::mlflow::MlflowExportConfig``):
    the endpoint/experiment/run identity, the user ``tags``, the resolved
    ``artifact_globs``, the ``benchmark_id``, the pre-built ``params`` payload
    (``MLflowDataExporter._build_param_payload``, ``mlflow_data_exporter.py:357``),
    and the AIPerf package ``aiperf_version`` (for the ``aiperf.version`` tag; the
    native report carries only the Rust crate version) — pure frontend config the
    native report cannot reconstruct. Metric VALUES and the ``metric.tag`` /
    ``metric.tag.<stat>`` key scheme are assembled by the Rust sink from the
    report; only config-derived params/tags are projected here.
    """
    mlflow_cfg = run.cfg.mlflow
    if mlflow_cfg.tracking_uri is None:
        return None

    from aiperf import __version__ as aiperf_version

    projection: dict[str, Any] = {
        "enabled": True,
        "tracking_uri": mlflow_cfg.tracking_uri,
        "experiment": mlflow_cfg.experiment,
        "tags": mlflow_cfg.tags_dict,
        "artifact_globs": mlflow_cfg.resolved_artifact_globs,
        "params": _mlflow_param_payload(run),
        "aiperf_version": aiperf_version,
    }
    if mlflow_cfg.run_name is not None:
        projection["run_name"] = mlflow_cfg.run_name
    if mlflow_cfg.parent_run_id is not None:
        projection["parent_run_id"] = mlflow_cfg.parent_run_id
    if run.benchmark_id is not None:
        projection["benchmark_id"] = run.benchmark_id
    total_expected = _mlflow_total_expected(run)
    if total_expected is not None:
        projection["total_expected_requests"] = total_expected
    return projection


def _mlflow_param_payload(run: BenchmarkRun) -> dict[str, str]:
    """Reproduce ``MLflowDataExporter._build_param_payload`` (``mlflow_data_exporter.py:357``).

    Pure frontend config: endpoint type/models/urls (URLs redacted), the artifact
    directory, and — from the first profiling phase — the timing mode and any
    concurrency / request-rate / request-count / benchmark-duration loadgen axis,
    plus the redacted CLI command. The native report carries none of this, so it
    is assembled and redacted here and forwarded verbatim to the Rust sink.
    """
    from aiperf.common.redact import redact_cli_command, redact_url
    from aiperf.config.phases import get_phase_rate

    cfg = run.cfg
    params: dict[str, str] = {
        "endpoint.type": str(cfg.endpoint.type),
        "endpoint.models": ",".join(cfg.get_model_names()),
        "endpoint.urls": ",".join(redact_url(url) for url in cfg.endpoint.urls),
        "output.artifact_directory": str(cfg.artifacts.artifact_directory),
    }

    profiling_phases = cfg.get_profiling_phases()
    if profiling_phases:
        phase = profiling_phases[0]
        params["timing.mode"] = str(phase.type)
        if getattr(phase, "concurrency", None) is not None:
            params["loadgen.concurrency"] = str(phase.concurrency)
        rate = get_phase_rate(phase)
        if rate is not None:
            params["loadgen.request_rate"] = str(rate)
        if phase.requests is not None:
            params["loadgen.request_count"] = str(phase.requests)
        if phase.duration is not None:
            params["loadgen.benchmark_duration"] = str(phase.duration)

    cli_command = getattr(cfg, "cli_command", None) or run.cli_command
    if cli_command:
        params["aiperf.cli_command"] = redact_cli_command(cli_command)
    return params


def _mlflow_total_expected(run: BenchmarkRun) -> float | None:
    """Derive ``total_expected_requests`` for the ``aiperf.total_expected_requests`` metric.

    ``MLflowDataExporter`` reads this from ``ProfileResults.total_expected``; the
    native report has no such field, so it is derived from the first profiling
    phase's configured ``requests`` when present (``None`` otherwise, matching the
    exporter's skip-when-absent behavior).
    """
    profiling_phases = run.cfg.get_profiling_phases()
    if not profiling_phases:
        return None
    requests = profiling_phases[0].requests
    return float(requests) if requests is not None else None


def _wandb_frontend_projection(run: BenchmarkRun) -> dict[str, Any] | None:
    """Project ``cfg.wandb`` onto the native Weights & Biases sink block.

    Enabled iff a project is set (``WandbConfig.enabled``). Reproduces the fields
    the Rust offline-``.wandb`` writer decodes
    (``aiperf::export::wandb::WandbExportConfig``): project/entity/run-name, the
    user ``tags`` (the ``aiperf-<version>`` / ``benchmark-<id8>`` tags are derived
    by the sink from the projected ``aiperf_version`` — the AIPerf package version,
    since the native report carries only the Rust crate version — and the
    ``benchmark_id``), the ``benchmark_id``, the serialized redacted ``config_json``
    (``cfg.model_dump(mode="json", exclude_none=True)`` — the same object
    ``WandbDataExporter._build_config_payload`` logs, ``wandb_data_exporter.py:160``),
    and the redacted ``cli_command`` recorded under ``aiperf.cli_command``.
    """
    import orjson

    from aiperf import __version__ as aiperf_version
    from aiperf.common.redact import redact_cli_command

    wandb_cfg = run.cfg.wandb
    if wandb_cfg.project is None:
        return None

    config_payload = run.cfg.model_dump(mode="json", exclude_none=True)
    projection: dict[str, Any] = {
        "project": wandb_cfg.project,
        "config_json": orjson.dumps(config_payload).decode("utf-8"),
        "aiperf_version": aiperf_version,
    }
    if wandb_cfg.entity is not None:
        projection["entity"] = wandb_cfg.entity
    if wandb_cfg.run_name is not None:
        projection["run_name"] = wandb_cfg.run_name
    if wandb_cfg.tags:
        projection["tags"] = list(wandb_cfg.tags)
    if run.benchmark_id is not None:
        projection["benchmark_id"] = run.benchmark_id
    if run.cli_command:
        projection["cli_command"] = redact_cli_command(run.cli_command)
    return projection


def _timeslice_frontend_projection(run: BenchmarkRun) -> dict[str, Any] | None:
    """Project the frontend-owned timeslice export policy onto ``cfg.export``.

    The native runner emits per-slice timeslices only when the run configures a
    ``slice_duration``; mirror that gate. Both JSON and CSV timeslice files are
    always produced by the Python exporter suite (each gated only on the presence
    of timeslices), so the native sink emits both here too.

    Like the genai-perf v1 sink, the timeslice files also need the registry-derived
    metric identity the native report cannot reconstruct — ``header_map`` (CSV
    display names), ``filtered_tags`` (registered INTERNAL/EXPERIMENTAL drop set),
    and ``scalar_tags`` (AGGREGATE/DERIVED ``count``-drop set). These are projected
    via :func:`_metric_registry_projection` so the native sink names, filters, and
    scalar-suppresses exactly as the Python exporters do (native-runtime metrics
    Python never registered — ``active_*``/``effective_*``/``credit_*`` — are kept
    and named by their snake tag).

    The JSON ``input_config`` object is projected as the exact value
    :class:`TimesliceCollectionExportData` emits — ``model_dump(mode="json",
    exclude_unset=True, exclude_none=True)`` then ``scrub_non_finite`` — so the
    native sink wraps it verbatim after the ``timeslices`` array. This is the same
    ``input_config`` value the genai-perf envelope carries (identical field type,
    identical dump options); it is recomputed here through the timeslice model so
    the parity oracle is exact.
    """
    if run.cfg.artifacts.slice_duration is None:
        return None

    from aiperf.common.finite import scrub_non_finite
    from aiperf.common.models.export_models import TimesliceCollectionExportData

    collection = TimesliceCollectionExportData(timeslices=[], input_config=run.cfg)
    dumped = scrub_non_finite(
        collection.model_dump(mode="json", exclude_unset=True, exclude_none=True)
    )
    projection: dict[str, Any] = {
        "json": True,
        "csv": True,
        "input_config": dumped["input_config"],
    }
    projection.update(_metric_registry_projection())
    return projection


def _metric_registry_projection() -> dict[str, Any]:
    """Project the registry-derived metric identity shared by the file exporters.

    Returns ``header_map`` / ``filtered_tags`` / ``scalar_tags`` computed exactly
    as :func:`_genai_perf_frontend_projection` derives them (see that function for
    the ``native_report._metric_result`` / ``_prepare_metrics`` /
    ``to_json_result`` grounding). Factored out so the timeslice sink reproduces
    the genai-perf sink's naming/filtering byte-for-byte without re-deriving the
    catalog.
    """
    from aiperf.common.enums import MetricFlags, MetricType
    from aiperf.metrics.metric_registry import MetricRegistry

    show_internal = Environment.DEV.SHOW_INTERNAL_METRICS
    show_experimental = Environment.DEV.SHOW_EXPERIMENTAL_METRICS

    header_map: dict[str, str] = {}
    filtered_tags: list[str] = []
    scalar_tags: list[str] = []
    for metric_class in MetricRegistry.all_classes():
        tag = str(metric_class.tag)
        header_map[tag] = metric_class.header
        if (metric_class.has_flags(MetricFlags.INTERNAL) and not show_internal) or (
            metric_class.has_flags(MetricFlags.EXPERIMENTAL) and not show_experimental
        ):
            filtered_tags.append(tag)
        if metric_class.type in {MetricType.AGGREGATE, MetricType.DERIVED}:
            scalar_tags.append(tag)

    return {
        "header_map": header_map,
        "filtered_tags": sorted(filtered_tags),
        "scalar_tags": sorted(scalar_tags),
    }


def _server_metrics_frontend_projection(run: BenchmarkRun) -> dict[str, Any] | None:
    """Project the frontend-owned server-metrics export policy onto ``cfg.export``.

    Enabled when server-metrics collection is enabled and the JSON and/or CSV
    format is selected (``cfg.server_metrics.formats``); the ``jsonl`` / ``parquet``
    formats are handled by :func:`_server_metrics`, not this summary sink. The
    per-format toggles mirror the Python exporters' ``ServerMetricsFormat`` gate.

    Three frontend-owned envelope values are projected because the native report
    cannot reconstruct them:

    * ``aiperf_version`` — the AIPerf package version (``aiperf.__version__``); the
      native report carries only the Rust crate version, so the frontend supplies
      the authoritative value emitted in the JSON ``aiperf_version`` field and the
      CSV ``# aiperf_version:`` comment header.
    * ``benchmark_id`` — the run identity string; the Python exporters emit it in
      both the JSON ``benchmark_id`` field and the CSV ``# benchmark_id:`` comment
      header (``None`` when absent).
    * ``input_config`` — projected as the exact value
      :class:`ServerMetricsExportData` emits for its ``input_config`` field:
      ``cfg.model_dump(mode="json", exclude_unset=True)`` placed into the export
      model, then ``model_dump(mode="json", exclude_none=True)`` and
      ``scrub_non_finite``. It is reconstructed here through the real export model
      (with a throwaway minimal summary) so the outer ``exclude_none`` recursion is
      byte-exact regardless of Pydantic's dict-field semantics. Only the JSON file
      carries ``input_config``; the CSV does not.
    """
    server_metrics = run.cfg.server_metrics
    if not server_metrics.enabled:
        return None

    formats = {str(fmt) for fmt in server_metrics.formats}
    json_enabled = "json" in formats
    csv_enabled = "csv" in formats
    if not (json_enabled or csv_enabled):
        return None

    from aiperf import __version__ as aiperf_version

    projection: dict[str, Any] = {
        "json": json_enabled,
        "csv": csv_enabled,
        "aiperf_version": aiperf_version,
    }
    if run.benchmark_id is not None:
        projection["benchmark_id"] = run.benchmark_id
    if json_enabled:
        projection["input_config"] = _server_metrics_input_config(run)
    return projection


def _server_metrics_input_config(run: BenchmarkRun) -> Any:
    """Compute the exact ``input_config`` value the server-metrics JSON emits.

    Reproduces ``ServerMetricsJsonExporter._generate_content`` for the
    ``input_config`` field alone: dump the config with ``exclude_unset=True`` (no
    ``exclude_none``), place it into :class:`ServerMetricsExportData`, then apply
    the model's ``model_dump(mode="json", exclude_none=True)`` and
    ``scrub_non_finite``. Serialization of the ``input_config`` field is
    independent of the ``summary`` / ``metrics`` payloads, so a throwaway minimal
    summary yields the byte-exact value.
    """
    from datetime import datetime

    from aiperf.common.finite import scrub_non_finite
    from aiperf.common.models.server_metrics_models import (
        ServerMetricsExportData,
        ServerMetricsSummary,
    )

    input_config = run.cfg.model_dump(mode="json", exclude_unset=True)
    export_data = ServerMetricsExportData(
        aiperf_version=None,
        benchmark_id=None,
        summary=ServerMetricsSummary(
            endpoints_configured=[],
            endpoints_successful=[],
            start_time=datetime.fromtimestamp(0),
            end_time=datetime.fromtimestamp(0),
        ),
        input_config=input_config,
    )
    dumped = scrub_non_finite(export_data.model_dump(mode="json", exclude_none=True))
    return dumped["input_config"]


def _parquet_frontend_projection(run: BenchmarkRun) -> dict[str, Any] | None:
    """Project the server-metrics Parquet sink toggle onto ``cfg.export.parquet``.

    Config-only passthrough: the native ``aiperf::export::parquet`` sink owns all
    assembly and serialization. It reads the runner-emitted
    ``.aiperf-server-metrics-parquet-wire.jsonl`` wire file (whose path is lowered
    by :func:`_server_metrics` when ``parquet`` is in ``cfg.server_metrics.formats``)
    and the profiling boundary carried on the native report
    (``report.summary.server_metrics.profiling``); nothing else is frontend-owned,
    so ``enabled`` is the sole projected field
    (``aiperf::export::parquet::ParquetExportConfig``).

    Enabled iff server-metrics collection is on and the ``parquet`` format is
    selected — the same gate under which :func:`_server_metrics` writes the
    ``parquet_wire_path`` the sink consumes. Absent either, the block is omitted
    and the sink stays disabled.
    """
    server_metrics = run.cfg.server_metrics
    if not server_metrics.enabled:
        return None
    formats = {str(fmt) for fmt in server_metrics.formats}
    if "parquet" not in formats:
        return None
    return {"enabled": True}


def _accuracy_csv_frontend_projection(run: BenchmarkRun) -> dict[str, Any] | None:
    """Project the accuracy CSV sink toggle onto ``cfg.export.accuracy_csv``.

    Config-only passthrough: the native ``aiperf::export::accuracy_csv`` sink reads
    ``report.accuracy.summary`` (the overall + per-task rollups) directly and writes
    ``accuracy_results.csv`` byte-for-byte against the Python
    ``AccuracyDataExporter``; there is no frontend-owned envelope value, so
    ``enabled`` is the sole projected field
    (``aiperf::export::accuracy_csv::AccuracyCsvExportConfig``).

    Enabled iff accuracy benchmarking mode is on (``cfg.accuracy.enabled``), the
    same gate that selects the ``static_accuracy`` workload (see
    :func:`_workload_type`). The sink additionally self-skips when the report carries
    no accuracy analysis or an empty population, matching the Python exporter.
    """
    accuracy = run.cfg.accuracy
    if accuracy is None or not accuracy.enabled:
        return None
    return {"enabled": True}


def _genai_perf_frontend_projection(run: BenchmarkRun) -> dict[str, Any]:
    """Project frontend-owned genai-perf v1 data absent from the native report.

    Three families of value are computed here because only the Python frontend
    can derive them, and the native ``aiperf::export::genai_perf`` sink consumes
    them verbatim (it performs all assembly/serialization itself):

    * ``header_map`` — the display header for every registered metric tag, exactly
      as :func:`native_report._metric_result` derives it
      (``MetricRegistry.get_class_or_none(tag).header``). Unregistered tags are
      absent here; the sink falls back to the tag string, matching Python's
      ``else tag`` branch. Native-runtime metrics (``active_*``/``effective_*``/
      ``credit_*``) are unregistered, so Python emits their snake tag as the name.
    * ``filtered_tags`` / ``scalar_tags`` — the registered tags the Python file
      exporters drop (``metrics_base_exporter._prepare_metrics``: INTERNAL /
      EXPERIMENTAL, honoring the dev show-flags) and the registered scalar-tier
      tags whose ``count`` is dropped by ``record_models.to_json_result``
      (``MetricType.AGGREGATE`` / ``DERIVED``).
    * ``envelope`` — ``benchmark_id``, ``aiperf_version``, ``input_config``, and
      ``run_info`` serialized exactly as :class:`MetricsJsonExporter` emits them
      (``JsonExportData.model_dump(mode="json", exclude_unset=True,
      exclude_none=True)`` then ``scrub_non_finite``).
    """
    from aiperf import __version__ as aiperf_version
    from aiperf.common.enums import MetricFlags, MetricType
    from aiperf.common.finite import scrub_non_finite
    from aiperf.common.models.export_models import JsonExportData, RunInfo
    from aiperf.metrics.metric_registry import MetricRegistry

    show_internal = Environment.DEV.SHOW_INTERNAL_METRICS
    show_experimental = Environment.DEV.SHOW_EXPERIMENTAL_METRICS

    header_map: dict[str, str] = {}
    filtered_tags: list[str] = []
    scalar_tags: list[str] = []
    for metric_class in MetricRegistry.all_classes():
        tag = str(metric_class.tag)
        header_map[tag] = metric_class.header
        if (metric_class.has_flags(MetricFlags.INTERNAL) and not show_internal) or (
            metric_class.has_flags(MetricFlags.EXPERIMENTAL) and not show_experimental
        ):
            filtered_tags.append(tag)
        if metric_class.type in {MetricType.AGGREGATE, MetricType.DERIVED}:
            scalar_tags.append(tag)

    envelope_model = JsonExportData(
        aiperf_version=aiperf_version,
        benchmark_id=run.benchmark_id,
        input_config=run.cfg,
        run_info=RunInfo.from_run(run),
    )
    envelope = scrub_non_finite(
        envelope_model.model_dump(mode="json", exclude_unset=True, exclude_none=True)
    )

    return {
        "header_map": header_map,
        "filtered_tags": sorted(filtered_tags),
        "scalar_tags": sorted(scalar_tags),
        "envelope": envelope,
    }


def _live_streaming(run: BenchmarkRun) -> dict[str, Any] | None:
    """Lower OTel/live-MLflow into the supervised Python extension ABI."""
    config = run.cfg
    # By default the native Rust OTel/MLflow sinks are the single network emitter
    # (see :func:`_export`); suppress the Python streaming sidecar so those
    # destinations are not written twice. Reversible: AIPERF_RUNTIME_NATIVE_EXPORT=0
    # restores the legacy live-streaming path.
    if Environment.RUNTIME.NATIVE_EXPORT:
        return None
    if not config.otel.collector_enabled and not config.mlflow.enabled:
        return None

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
    # Agentic cache-warmup duration rides on the warmup phase; the runner lowers
    # it into the recorded-graph cache-pressure window. Absent on non-scenario
    # configs (getattr guard), leaving the pair's default cache-warmup policy.
    _set_optional(
        common,
        "agentic_cache_warmup_duration",
        getattr(phase, "agentic_cache_warmup_duration", None),
    )

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
