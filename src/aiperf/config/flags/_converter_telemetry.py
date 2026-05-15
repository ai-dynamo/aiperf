# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Telemetry section builders for the ``CLIConfig`` -> ``AIPerfConfig`` converter.

Builds ``gpu_telemetry``, ``server_metrics``, ``otel``, and ``mlflow`` sections
by reading top-level fields on the ``CLIConfig``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiperf.config.flags import CLIConfig


def _url(item: str) -> str:
    return item if item.startswith("http") else f"http://{item}"


def _is_localhost_url(url: str) -> bool:
    """True when ``url`` resolves to localhost (IPv4, IPv6, or hostname)."""
    from urllib.parse import urlparse

    # Handle IPv6 localhost without brackets (e.g. "::1:8000" or "http://::1:8000").
    url_without_scheme = url.removeprefix("http://").removeprefix("https://")
    if url_without_scheme.startswith("::1:") or url_without_scheme.startswith("[::1]"):
        return True

    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"

    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    return hostname.lower() in ("localhost", "127.0.0.1", "::1")


def _local_collector_keywords() -> dict[str, Any]:
    """CLI keyword (plugin name, lowercased) -> ``GPUTelemetryCollectorType``.

    Derived from plugin metadata: any ``gpu_telemetry_collector`` plugin whose
    metadata declares ``is_local: true`` becomes a valid keyword. Adding a new
    local collector therefore only requires editing ``plugins.yaml`` — no edits
    here.
    """
    from aiperf.plugin import plugins
    from aiperf.plugin.enums import GPUTelemetryCollectorType

    return {
        member.lower(): GPUTelemetryCollectorType(member)
        for member in GPUTelemetryCollectorType
        if plugins.get_gpu_telemetry_collector_metadata(member).is_local
    }


def build_gpu_telemetry(cli: CLIConfig) -> dict[str, Any]:
    """Translate ``--gpu-telemetry`` magic-list into the telemetry dict.

    Classifies each ``--gpu-telemetry`` item into a collector type, URLs,
    optional metrics file, or dashboard mode. Local collector keywords are
    discovered from plugin metadata (``is_local: true``) so adding a new
    local backend never touches this converter.

    Ports v1 ``_parse_gpu_telemetry_config``: rejects the mutex of
    ``--no-gpu-telemetry`` + ``--gpu-telemetry``, validates the ``.csv``
    metrics file exists at convert time, and warns when a local collector
    is paired with non-localhost server URLs.
    """
    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.common.enums import GPUTelemetryMode

    cli_set = cli.model_fields_set
    if "no_gpu_telemetry" in cli_set and "gpu_telemetry" in cli_set:
        raise ValueError(
            "Cannot use both --no-gpu-telemetry and --gpu-telemetry together. "
            "Use only one or the other."
        )
    if cli.no_gpu_telemetry:
        return {"enabled": False}
    if not cli.gpu_telemetry:
        return {"enabled": True}

    local_keywords = _local_collector_keywords()
    urls: list[str] = []
    metrics_file: Path | None = None
    collector_type = cli._gpu_telemetry_collector_type
    mode = cli._gpu_telemetry_mode

    from aiperf.plugin import plugins

    for item in cli.gpu_telemetry:
        lowered = item.lower()
        if lowered in local_keywords:
            selected = local_keywords[lowered]
            # Reject mixing two different local collectors in the same call.
            # "Local" is sourced from plugin metadata (is_local: true) so the
            # check stays correct when a new local backend is added.
            current_is_local = plugins.get_gpu_telemetry_collector_metadata(
                collector_type
            ).is_local
            if current_is_local and collector_type != selected:
                raise ValueError(
                    "Conflicting local GPU telemetry collectors: "
                    f"'{collector_type}' and '{selected}'. Choose exactly one."
                )
            collector_type = selected
        elif lowered == "dashboard":
            mode = GPUTelemetryMode.REALTIME_DASHBOARD
        elif item.endswith(".csv"):
            csv_path = Path(item)
            if not csv_path.exists():
                raise ValueError(f"GPU metrics file not found: {item}")
            metrics_file = csv_path
        elif item.startswith("http") or ":" in item:
            urls.append(_url(item))
        else:
            valid_kw = ", ".join(f"'{k}'" for k in sorted(local_keywords))
            raise ValueError(
                f"Invalid GPU telemetry item: {item}. Valid options are: "
                f"{valid_kw}, 'dashboard', '.csv' file, and URLs."
            )

    cli._gpu_telemetry_collector_type = collector_type
    cli._gpu_telemetry_mode = mode

    # Warn when a local collector is paired with non-localhost server URLs:
    # the local agent only measures the host machine, not the inference
    # server's GPUs. "Local" comes from plugin metadata (``is_local: true``),
    # same registry consulted by ``_local_collector_keywords``.
    is_local = plugins.get_gpu_telemetry_collector_metadata(collector_type).is_local
    if is_local and cli.urls:
        non_local = [u for u in cli.urls if not _is_localhost_url(u)]
        if non_local:
            AIPerfLogger(__name__).warning(
                f"Using {collector_type} for GPU telemetry with non-localhost "
                f"server URL(s): {non_local}. {collector_type} collects GPU "
                "metrics from the local machine only. If the inference server "
                "is running remotely, the GPU telemetry will not reflect the "
                "server's GPU usage. Consider using DCGM mode with the "
                "server's metrics endpoint instead."
            )

    gpu_telemetry: dict[str, Any] = {
        "enabled": True,
        "urls": urls,
        "collector": collector_type,
        "mode": mode,
    }
    if metrics_file is not None:
        gpu_telemetry["metrics_file"] = metrics_file
    return gpu_telemetry


def build_server_metrics(cli: CLIConfig) -> dict[str, Any]:
    """Translate ``--server-metrics`` flags into the server-metrics dict."""
    from aiperf.common.metric_utils import normalize_metrics_endpoint_url

    if cli.no_server_metrics:
        return {"enabled": False}
    sm_urls = [
        normalize_metrics_endpoint_url(_url(i))
        for i in cli.server_metrics or []
        if i.startswith("http") or ":" in i
    ]
    server_metrics: dict[str, Any] = {"enabled": True, "urls": sm_urls}
    if cli.server_metrics_formats:
        server_metrics["formats"] = list(cli.server_metrics_formats)
    return server_metrics


def _normalize_otel_metrics_url(url: str) -> str:
    """Normalize OTel collector URL to an OTLP/HTTP metrics endpoint.

    Ports v1 ``_normalize_otel_metrics_url``: validates scheme and host,
    auto-prefixes ``http://`` for bare ``host[:port]`` values, and ensures
    the path ends in ``/v1/metrics`` so users don't have to spell out the
    full OTLP/HTTP endpoint.
    """
    from urllib.parse import urlparse, urlunparse

    normalized_url = url.strip()
    if not normalized_url:
        raise ValueError("--otel-url cannot be empty.")

    if "://" not in normalized_url:
        normalized_url = f"http://{normalized_url}"

    parsed = urlparse(normalized_url)
    # ``urlparse("http://:4318")`` yields netloc=":4318" but hostname=None —
    # netloc truthiness alone is not enough. Require a non-empty hostname so
    # bare-port values don't slip through and produce a malformed endpoint.
    if not parsed.scheme or not parsed.netloc or not parsed.hostname:
        raise ValueError(
            f"Invalid --otel-url value: {url!r}. Expected host[:port] or a full URL."
        )
    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError(
            f"Invalid --otel-url value: {url!r}. "
            f"Only http and https schemes are supported (got {parsed.scheme!r}). "
            "OTLP/gRPC is not supported; use the OTLP/HTTP exporter endpoint."
        )

    path = parsed.path.rstrip("/")
    if path.endswith("/v1/metrics"):
        normalized_path = path
    elif not path:
        normalized_path = "/v1/metrics"
    else:
        normalized_path = f"{path}/v1/metrics"

    return urlunparse(parsed._replace(path=normalized_path))


def build_otel(cli: CLIConfig) -> dict[str, Any]:
    """Translate OTel CLI flags into the first-class OTel config dict."""
    otel: dict[str, Any] = {}
    cli_set = cli.model_fields_set

    if "otel_url" in cli_set and cli.otel_url is not None:
        otel["metrics_url"] = _normalize_otel_metrics_url(cli.otel_url)
    else:
        # ``--stream`` and ``--gen-ai-provider`` are OTel-only secondary
        # flags: they only take effect when ``--otel-url`` is set. Refuse
        # silently dropping them so the user discovers the missing primary.
        offenders: list[str] = []
        if "stream" in cli_set:
            offenders.append("--stream")
        if "gen_ai_provider" in cli_set:
            offenders.append("--gen-ai-provider")
        if offenders:
            raise ValueError(
                f"{', '.join(offenders)} requires --otel-url to be set; OTel "
                "streaming is disabled when no OTLP endpoint is configured."
            )

    if "stream" in cli_set:
        otel["stream_metrics_enabled"] = cli.stream in ("default", "metrics")
        otel["stream_timing_enabled"] = cli.stream in ("default", "timing")
    if "gen_ai_provider" in cli_set:
        otel["gen_ai_provider"] = cli.gen_ai_provider
    return otel


def build_mlflow(cli: CLIConfig) -> dict[str, Any]:
    """Translate MLflow CLI flags into the first-class MLflow config dict.

    Ports v1 ``_validate_mlflow_config``: refuses secondary MLflow flags
    without ``--mlflow-tracking-uri``, rejects empty strings on
    tracking_uri/experiment/artifact_glob entries, and normalizes
    whitespace on tracking_uri/experiment/run_name/artifact_globs.
    """
    cli_set = cli.model_fields_set

    # Normalize artifact-glob entries first so an "empty glob" error
    # surfaces before the missing-tracking-uri error.
    artifact_globs: list[str] | None = None
    if "mlflow_artifact_globs" in cli_set and cli.mlflow_artifact_globs is not None:
        normalized: list[str] = []
        for glob in cli.mlflow_artifact_globs:
            stripped = glob.strip()
            if not stripped:
                raise ValueError("--mlflow-artifact-glob entries cannot be empty.")
            normalized.append(stripped)
        artifact_globs = normalized

    tracking_uri: str | None = None
    if "mlflow_tracking_uri" in cli_set and cli.mlflow_tracking_uri is not None:
        stripped_uri = cli.mlflow_tracking_uri.strip()
        if not stripped_uri:
            raise ValueError("--mlflow-tracking-uri cannot be empty.")
        tracking_uri = stripped_uri

    # Secondary flags require --mlflow-tracking-uri.
    if tracking_uri is None:
        secondary_present = any(
            key in cli_set
            for key in (
                "mlflow_experiment",
                "mlflow_run_name",
                "mlflow_tags",
                "mlflow_parent_run_id",
                "mlflow_artifact_globs",
            )
        )
        if secondary_present:
            raise ValueError(
                "--mlflow-experiment, --mlflow-run-name, --mlflow-tag, "
                "--mlflow-artifact-glob, and --mlflow-parent-run-id require "
                "--mlflow-tracking-uri to be set."
            )
        return {}

    out: dict[str, Any] = {"tracking_uri": tracking_uri}

    if "mlflow_experiment" in cli_set and cli.mlflow_experiment is not None:
        experiment = cli.mlflow_experiment.strip()
        if not experiment:
            raise ValueError(
                "--mlflow-experiment cannot be empty when --mlflow-tracking-uri is set."
            )
        out["experiment"] = experiment

    if "mlflow_run_name" in cli_set and cli.mlflow_run_name is not None:
        run_name = cli.mlflow_run_name.strip()
        # Empty-string-after-strip collapses to None: matches v1 normalization.
        out["run_name"] = run_name or None

    if "mlflow_tags" in cli_set:
        out["tags"] = cli.mlflow_tags
    if "mlflow_parent_run_id" in cli_set:
        out["parent_run_id"] = cli.mlflow_parent_run_id
    if artifact_globs is not None:
        out["artifact_globs"] = artifact_globs

    return out
