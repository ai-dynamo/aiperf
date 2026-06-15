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


def _classify_gpu_telemetry_items(
    items: list[str],
    *,
    local_keywords: dict[str, Any],
    collector_type: Any,
    mode: Any,
) -> tuple[Any, Any, list[str], Path | None]:
    """Walk ``--gpu-telemetry`` items, classify each into collector/mode/url/csv.

    Returns the resolved ``(collector_type, mode, urls, metrics_file)``.
    """
    from aiperf.common.enums import GPUTelemetryMode
    from aiperf.plugin import plugins

    urls: list[str] = []
    metrics_file: Path | None = None

    for item in items:
        lowered = item.lower()
        if lowered in local_keywords:
            selected = local_keywords[lowered]
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

    return collector_type, mode, urls, metrics_file


def _warn_if_local_collector_with_remote_urls(
    collector_type: Any, server_urls: list[str]
) -> None:
    """Warn when a local collector is paired with non-localhost server URLs."""
    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.plugin import plugins

    if not plugins.get_gpu_telemetry_collector_metadata(collector_type).is_local:
        return
    non_local = [u for u in server_urls if not _is_localhost_url(u)]
    if not non_local:
        return
    AIPerfLogger(__name__).warning(
        f"Using {collector_type} for GPU telemetry with non-localhost "
        f"server URL(s): {non_local}. {collector_type} collects GPU "
        "metrics from the local machine only. If the inference server "
        "is running remotely, the GPU telemetry will not reflect the "
        "server's GPU usage. Consider using DCGM mode with the "
        "server's metrics endpoint instead."
    )


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

    collector_type, mode, urls, metrics_file = _classify_gpu_telemetry_items(
        cli.gpu_telemetry,
        local_keywords=_local_collector_keywords(),
        collector_type=cli._gpu_telemetry_collector_type,
        mode=cli._gpu_telemetry_mode,
    )

    cli._gpu_telemetry_collector_type = collector_type
    cli._gpu_telemetry_mode = mode

    _warn_if_local_collector_with_remote_urls(collector_type, cli.urls or [])

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

    if (
        "no_server_metrics" in cli.model_fields_set
        and "server_metrics" in cli.model_fields_set
    ):
        raise ValueError(
            "Cannot use both --no-server-metrics and --server-metrics together. "
            "Use only one or the other."
        )
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


def _parse_otel_resource_attributes(items: list[str] | None) -> dict[str, str]:
    if not items:
        return {}
    attrs: dict[str, str] = {}
    for item in items:
        for pair in item.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(
                    f"Invalid --otel-resource-attributes entry {pair!r}: expected key=value."
                )
            key, _, value = pair.partition("=")
            key = key.strip()
            value = value.strip()
            if not key:
                raise ValueError(
                    f"Invalid --otel-resource-attributes entry {pair!r}: key cannot be empty."
                )
            if not value:
                raise ValueError(
                    f"Invalid --otel-resource-attributes entry {pair!r}: value cannot be empty."
                )
            attrs[key] = value
    return attrs


def _resolve_stream_domains(value: Any) -> tuple[bool, bool]:
    """Resolve ``--stream`` into (stream_metrics_enabled, stream_timing_enabled)."""
    domains = value or []
    if isinstance(domains, str):
        domains = [domains]
    allowed = {"default", "metrics", "timing", "none"}
    invalid = [d for d in domains if d not in allowed]
    if invalid:
        raise ValueError(
            f"Invalid --stream value(s) {invalid!r}. Allowed: {sorted(allowed)}."
        )
    if "none" in domains:
        return False, False
    wants_default = not domains or "default" in domains
    return (
        wants_default or "metrics" in domains,
        wants_default or "timing" in domains,
    )


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
        if "otel_resource_attributes" in cli_set:
            offenders.append("--otel-resource-attributes")
        if "gen_ai_provider" in cli_set:
            offenders.append("--gen-ai-provider")
        if offenders:
            raise ValueError(
                f"{', '.join(offenders)} requires --otel-url to be set; OTel "
                "streaming is disabled when no OTLP endpoint is configured."
            )

    if "stream" in cli_set:
        metrics_enabled, timing_enabled = _resolve_stream_domains(cli.stream)
        otel["stream_metrics_enabled"] = metrics_enabled
        otel["stream_timing_enabled"] = timing_enabled
    if "otel_resource_attributes" in cli_set:
        otel["custom_resource_attributes"] = _parse_otel_resource_attributes(
            cli.otel_resource_attributes
        )
    if "gen_ai_provider" in cli_set:
        otel["gen_ai_provider"] = cli.gen_ai_provider
    return otel


_MLFLOW_SECONDARY_FIELDS = (
    "mlflow_experiment",
    "mlflow_run_name",
    "mlflow_tags",
    "mlflow_parent_run_id",
    "mlflow_artifact_globs",
)


def _normalize_mlflow_artifact_globs(cli: CLIConfig) -> list[str] | None:
    if "mlflow_artifact_globs" not in cli.model_fields_set:
        return None
    if cli.mlflow_artifact_globs is None:
        return None
    normalized: list[str] = []
    for glob in cli.mlflow_artifact_globs:
        stripped = glob.strip()
        if not stripped:
            raise ValueError("--mlflow-artifact-glob entries cannot be empty.")
        normalized.append(stripped)
    return normalized


def _normalize_mlflow_tracking_uri(cli: CLIConfig) -> str | None:
    if "mlflow_tracking_uri" not in cli.model_fields_set:
        return None
    if cli.mlflow_tracking_uri is None:
        return None
    stripped = cli.mlflow_tracking_uri.strip()
    if not stripped:
        raise ValueError("--mlflow-tracking-uri cannot be empty.")
    return stripped


def _apply_mlflow_secondary_fields(out: dict[str, Any], cli: CLIConfig) -> None:
    cli_set = cli.model_fields_set
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


def build_mlflow(cli: CLIConfig) -> dict[str, Any]:
    """Translate MLflow CLI flags into the first-class MLflow config dict.

    Ports v1 ``_validate_mlflow_config``: refuses secondary MLflow flags
    without ``--mlflow-tracking-uri``, rejects empty strings on
    tracking_uri/experiment/artifact_glob entries, and normalizes
    whitespace on tracking_uri/experiment/run_name/artifact_globs.
    """
    # Normalize artifact-glob entries first so an "empty glob" error
    # surfaces before the missing-tracking-uri error.
    artifact_globs = _normalize_mlflow_artifact_globs(cli)
    tracking_uri = _normalize_mlflow_tracking_uri(cli)

    if tracking_uri is None:
        secondary_present = any(
            key in cli.model_fields_set for key in _MLFLOW_SECONDARY_FIELDS
        )
        if secondary_present:
            raise ValueError(
                "--mlflow-experiment, --mlflow-run-name, --mlflow-tag, "
                "--mlflow-artifact-glob, and --mlflow-parent-run-id require "
                "--mlflow-tracking-uri to be set."
            )
        return {}

    out: dict[str, Any] = {"tracking_uri": tracking_uri}
    _apply_mlflow_secondary_fields(out, cli)
    if artifact_globs is not None:
        out["artifact_globs"] = artifact_globs
    return out


_WANDB_SECONDARY_FIELDS = (
    "wandb_entity",
    "wandb_run_name",
    "wandb_tags",
)


def build_wandb(cli: CLIConfig, *, base_enabled: bool = False) -> dict[str, Any]:
    """Translate Weights & Biases CLI flags into the wandb config dict.

    Refuses secondary wandb flags without ``--wandb-project`` and rejects an
    empty project name. ``base_enabled`` relaxes the project requirement for
    the YAML+CLI overlay path: when the base config already enables wandb,
    secondary flags alone emit a partial override dict (e.g.
    ``-f base.yaml --wandb-run-name rerun``).
    """
    cli_set = cli.model_fields_set
    out: dict[str, Any] = {}
    if "wandb_project" in cli_set and cli.wandb_project is not None:
        project = cli.wandb_project.strip()
        if not project:
            raise ValueError("--wandb-project cannot be empty.")
        out["project"] = project
    elif not base_enabled:
        if any(key in cli_set for key in _WANDB_SECONDARY_FIELDS):
            raise ValueError(
                "--wandb-entity, --wandb-run-name, and --wandb-tag require "
                "--wandb-project to be set."
            )
        return {}

    if "wandb_entity" in cli_set and cli.wandb_entity is not None:
        entity = cli.wandb_entity.strip()
        if not entity:
            raise ValueError("--wandb-entity cannot be empty when set.")
        out["entity"] = entity
    if "wandb_run_name" in cli_set and cli.wandb_run_name is not None:
        out["run_name"] = cli.wandb_run_name.strip() or None
    if "wandb_tags" in cli_set:
        out["tags"] = cli.wandb_tags
    return out
