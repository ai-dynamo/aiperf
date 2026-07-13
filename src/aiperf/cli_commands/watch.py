# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Human entry point for runner-owned durable telemetry collection.

Python expands the Config-v2 document, binds it to one exact runner image,
forwards process signals, and presents the typed terminal archive summary. The
native runner alone performs source IO, parsing, buffering, WAL/Parquet writes,
recovery, and remote publication.
"""

from __future__ import annotations

import copy
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import FrameType
from typing import Annotated, Any
from urllib.parse import unquote, urlsplit

import orjson
import yaml
from cyclopts import App, Parameter

from aiperf.common.redact import redact_string
from aiperf.config.loader import expand_config_dict
from aiperf.orchestrator.runner_installation import RunnerInstallation
from aiperf.orchestrator.rust_wire import RUNNER_PROTOCOL_V2

app = App(name="watch", help="Collect or finalize a durable telemetry archive.")

_ALLOWED_DOCUMENT_FIELDS = {"schema_version", "variables", "run"}
_ALLOWED_RUN_FIELDS = {
    "identity",
    "artifact_target",
    "transport",
    "workload",
    "resources",
}


@app.default
def watch(
    config_file: Annotated[
        Path,
        Parameter(
            name=["--config", "-c"],
            help="Config-v2 YAML containing one http/telemetry_watch run.",
        ),
    ],
    *,
    runner_bin: Annotated[
        Path | None,
        Parameter(
            name="--runner-bin",
            help="Exact aiperf-runner executable; otherwise use normal discovery.",
        ),
    ] = None,
) -> None:
    """Run ``telemetry_watch`` through the packaged native runner.

    The config accepts environment substitution and strict Jinja expansion in
    the same order as other Config-v2 files. It contains a single ``run``
    object; Python supplies only protocol operation and exact distribution
    identity.
    """
    try:
        installation = RunnerInstallation.resolve(runner_bin)
        request = build_watch_request(config_file, installation, operation="execute")
        validation = copy.deepcopy(request)
        validation["operation"] = "validate"
        benchmark_id = request["run"]["identity"]["benchmark_id"]
        installation.validate_authored_request(
            validation,
            benchmark_id=benchmark_id,
        )
        completed = _execute_with_signal_forwarding(installation, request)
        terminal = _parse_watch_terminal(
            completed,
            benchmark_id=benchmark_id,
            distribution_id=request["expected_distribution_id"],
        )
        if not terminal["success"]:
            raise RuntimeError(_terminal_failure(terminal, completed.stderr))
        archive = _load_archive_report(terminal, Path(request["run"]["artifact_target"]))
        _present_archive_summary(archive, terminal)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        print(f"Error running aiperf watch: {redact_string(str(error))}", file=sys.stderr)
        raise SystemExit(1) from error


def build_watch_request(
    config_file: Path,
    installation: RunnerInstallation,
    *,
    operation: str,
) -> dict[str, Any]:
    """Expand and bind one strict watch run without executing it."""
    if operation not in {"validate", "execute"}:
        raise ValueError("watch operation must be 'validate' or 'execute'")
    document = _load_watch_document(config_file)
    run = copy.deepcopy(document["run"])
    _validate_watch_run_shape(run)
    _normalize_local_paths(run, config_file.parent)

    distribution_id = installation.distribution_id
    versions = installation.capabilities.get("protocol_versions")
    if not isinstance(versions, list) or RUNNER_PROTOCOL_V2 not in versions:
        raise RuntimeError(
            f"selected aiperf-runner does not support protocol {RUNNER_PROTOCOL_V2}"
        )
    if not isinstance(distribution_id, str) or not distribution_id:
        raise RuntimeError("selected aiperf-runner omitted its exact distribution identity")
    request: dict[str, Any] = {
        "protocol_version": RUNNER_PROTOCOL_V2,
        "operation": operation,
        "expected_distribution_id": distribution_id,
        "run": run,
    }
    installation.preflight_request(request)
    return request


def _load_watch_document(config_file: Path) -> dict[str, Any]:
    path = Path(config_file).expanduser().resolve(strict=True)
    try:
        decoded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"cannot load watch config {path}: {error}") from error
    if not isinstance(decoded, dict):
        raise ValueError("watch config must be a YAML object")
    unknown = set(decoded) - _ALLOWED_DOCUMENT_FIELDS
    if unknown:
        raise ValueError(f"unknown watch config fields: {sorted(unknown)}")
    schema_version = decoded.get("schema_version", "2.0")
    if str(schema_version) != "2.0":
        raise ValueError("watch config schema_version must be '2.0'")
    expanded = expand_config_dict(decoded)
    run = expanded.get("run")
    if not isinstance(run, dict):
        raise ValueError("watch config requires one run object")
    return expanded


def _validate_watch_run_shape(run: dict[str, Any]) -> None:
    unknown = set(run) - _ALLOWED_RUN_FIELDS
    if unknown:
        raise ValueError(f"unknown watch run fields: {sorted(unknown)}")
    identity = run.get("identity")
    benchmark_id = identity.get("benchmark_id") if isinstance(identity, dict) else None
    if not isinstance(benchmark_id, str) or not benchmark_id.strip():
        raise ValueError("watch run identity.benchmark_id must be a non-empty string")
    if not isinstance(run.get("artifact_target"), (str, os.PathLike)):
        raise ValueError("watch run artifact_target must be a path")
    transport = run.get("transport")
    if not isinstance(transport, dict) or transport.get("type") != "http":
        raise ValueError("aiperf watch requires run.transport.type='http'")
    workload = run.get("workload")
    if not isinstance(workload, dict) or workload.get("type") != "telemetry_watch":
        raise ValueError("aiperf watch requires run.workload.type='telemetry_watch'")
    if not isinstance(workload.get("config"), dict):
        raise ValueError("watch workload.config must be an object")
    resources = run.setdefault("resources", {})
    if not isinstance(resources, dict):
        raise ValueError("watch run.resources must be an object")


def _normalize_local_paths(run: dict[str, Any], config_parent: Path) -> None:
    base = Path(config_parent).expanduser().resolve()
    run["artifact_target"] = str(_absolute_path(run["artifact_target"], base))
    archive = run["workload"]["config"].get("archive")
    if not isinstance(archive, dict):
        raise ValueError("watch workload config requires an archive object")
    local_spool = archive.get("local_spool")
    if not isinstance(local_spool, (str, os.PathLike)):
        raise ValueError("watch archive.local_spool must be a path")
    archive["local_spool"] = str(_absolute_path(local_spool, base))
    target = archive.get("target")
    if not isinstance(target, str) or not target:
        raise ValueError("watch archive.target must be a non-empty URI or local path")
    archive["target"] = _normalize_archive_target(target, base)


def _absolute_path(value: str | os.PathLike[str], base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _normalize_archive_target(value: str, base: Path) -> str:
    parsed = urlsplit(value)
    if not parsed.scheme:
        return _absolute_path(value, base).as_uri()
    if parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError("file archive targets cannot name a remote authority")
        return _absolute_path(unquote(parsed.path), base).as_uri()
    return value


def _execute_with_signal_forwarding(
    installation: RunnerInstallation,
    request: dict[str, Any],
) -> subprocess.CompletedProcess[bytes]:
    child = installation.spawn(request)
    previous: dict[signal.Signals, Any] = {}
    signal_count = 0

    def forward(signum: int, _frame: FrameType | None) -> None:
        nonlocal signal_count
        signal_count += 1
        try:
            if signal_count == 1:
                child.send_signal(signum)
            else:
                child.kill()
        except ProcessLookupError:
            return

    for selected in (signal.SIGINT, signal.SIGTERM):
        previous[selected] = signal.getsignal(selected)
        signal.signal(selected, forward)
    try:
        stdout, stderr = child.communicate(orjson.dumps(request))
    finally:
        for selected, handler in previous.items():
            signal.signal(selected, handler)
    if stderr:
        sys.stderr.buffer.write(stderr)
        sys.stderr.buffer.flush()
    return subprocess.CompletedProcess(
        child.args,
        child.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _parse_watch_terminal(
    completed: subprocess.CompletedProcess[bytes],
    *,
    benchmark_id: str,
    distribution_id: str,
) -> dict[str, Any]:
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(
            "native runner must emit exactly one terminal JSON line; "
            f"received {len(lines)} (exit {completed.returncode})"
        )
    try:
        terminal = orjson.loads(lines[0])
    except orjson.JSONDecodeError as error:
        raise ValueError(f"native runner returned invalid terminal JSON: {error}") from error
    if not isinstance(terminal, dict):
        raise ValueError("native runner terminal must be an object")
    expected = {
        "protocol_version": RUNNER_PROTOCOL_V2,
        "event": "run_terminal",
        "distribution_id": distribution_id,
        "benchmark_id": benchmark_id,
    }
    for field, value in expected.items():
        if terminal.get(field) != value:
            raise ValueError(
                f"native runner terminal {field}={terminal.get(field)!r}; expected {value!r}"
            )
    success = terminal.get("success")
    if not isinstance(success, bool):
        raise ValueError("native runner terminal success must be a boolean")
    if success != (completed.returncode == 0):
        raise ValueError("native runner terminal success disagrees with its exit code")
    return terminal


def _load_archive_report(
    terminal: dict[str, Any], artifact_target: Path
) -> dict[str, Any]:
    report_value = terminal.get("report_path")
    if not isinstance(report_value, str) or not report_value:
        raise ValueError("successful watch terminal omitted report_path")
    report_path = Path(report_value).resolve()
    expected = artifact_target.resolve() / "native-v2.json"
    if report_path != expected:
        raise ValueError(f"watch report escaped its artifact target: {report_path}")
    try:
        report = orjson.loads(report_path.read_bytes())
    except (OSError, orjson.JSONDecodeError) as error:
        raise ValueError(f"cannot read native watch report {report_path}: {error}") from error
    archive = report.get("telemetry_archive") if isinstance(report, dict) else None
    if not isinstance(archive, dict):
        raise ValueError("native watch report omitted typed telemetry_archive outcome")
    return archive


def _terminal_failure(terminal: dict[str, Any], stderr: bytes) -> str:
    errors = terminal.get("errors")
    messages = []
    if isinstance(errors, list):
        messages = [
            entry["message"]
            for entry in errors
            if isinstance(entry, dict)
            and isinstance(entry.get("message"), str)
            and entry["message"]
        ]
    detail = "; ".join(messages) or "native watch execution failed"
    artifacts = terminal.get("diagnostic_artifacts")
    if isinstance(artifacts, list):
        evidence = [
            f"{entry['kind']}={entry['relative_path']} ({entry['content_hash']})"
            for entry in artifacts
            if isinstance(entry, dict)
            and isinstance(entry.get("kind"), str)
            and isinstance(entry.get("relative_path"), str)
            and isinstance(entry.get("content_hash"), str)
        ]
        if evidence:
            detail = f"{detail}; diagnostic artifacts: {', '.join(evidence)}"
    diagnostic = stderr.decode(errors="replace").strip()
    if diagnostic:
        detail = f"{detail}; Rust stderr: {diagnostic[-4000:]}"
    return detail


def _present_archive_summary(
    archive: dict[str, Any], terminal: dict[str, Any]
) -> None:
    summary = {
        "archive_id": archive.get("archive_id"),
        "state": archive.get("state"),
        "finalized_local": archive.get("finalized_local"),
        "finalized_remote": archive.get("finalized_remote"),
        "lossy": archive.get("lossy"),
        "report_path": terminal.get("report_path"),
    }
    sys.stdout.write(orjson.dumps(summary, option=orjson.OPT_INDENT_2).decode())
    sys.stdout.write("\n")
