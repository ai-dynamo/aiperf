#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise an installed frontend and native runner companion end to end."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sysconfig
import threading
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import metadata
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import orjson

from aiperf.config import AIPerfConfig, BenchmarkRun
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.runner_installation import RunnerInstallation
from aiperf.orchestrator.rust_executor import RustSubprocessExecutor

RunnerProfile = Literal["online", "offline"]

_SSE = b"".join(
    [
        b'data: {"id":"release-smoke","object":"chat.completion.chunk","created":0,"model":"mock-model","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}\n\n',
        b'data: {"id":"release-smoke","object":"chat.completion.chunk","created":0,"model":"mock-model","choices":[],"usage":{"prompt_tokens":8,"completion_tokens":1}}\n\n',
        b"data: [DONE]\n\n",
    ]
)


class _LoopbackChatHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible SSE endpoint used by the release smoke."""

    protocol_version = "HTTP/1.1"
    request_count = 0
    lock = threading.Lock()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(length)
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        body = orjson.loads(payload)
        if not isinstance(body, dict) or body.get("model") != "mock-model":
            self.send_error(400)
            return
        with self.lock:
            type(self).request_count += 1
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(_SSE)))
        self.end_headers()
        self.wfile.write(_SSE)

    def log_message(self, format: str, *args: object) -> None:
        """Keep the verifier's output machine-readable."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("online", "offline"), required=True)
    return parser


def _installed_manifest(installation: RunnerInstallation) -> dict[str, Any]:
    scripts = Path(sysconfig.get_path("scripts")).resolve()
    if installation.binary.parent != scripts:
        raise RuntimeError(
            f"companion runner resolved to {installation.binary}, expected scripts "
            f"directory {scripts}"
        )
    if not _is_catalog_shape(installation.capabilities):
        raise RuntimeError(
            "companion runner did not return a plugins.yaml-shaped catalog with "
            "schema_version, endpoint, and transport maps"
        )

    distribution = metadata.distribution("aiperf-runner")
    manifest_text = distribution.read_text("extra_metadata/runner-build.json")
    if manifest_text is None:
        raise RuntimeError("companion wheel omitted runner-build.json metadata")
    manifest = orjson.loads(manifest_text)
    if not isinstance(manifest, dict):
        raise RuntimeError("companion runner-build.json must contain an object")
    return manifest


def _is_catalog_shape(capabilities: dict[str, Any]) -> bool:
    """Confirm the runner published a non-empty plugins.yaml-shaped catalog."""
    return (
        isinstance(capabilities.get("schema_version"), str)
        and bool(capabilities["schema_version"])
        and isinstance(capabilities.get("endpoint"), dict)
        and bool(capabilities["endpoint"])
        and isinstance(capabilities.get("transport"), dict)
        and bool(capabilities["transport"])
    )


def _verify_profile(
    profile: RunnerProfile,
    installation: RunnerInstallation,
    manifest: dict[str, Any],
) -> None:
    features = manifest.get("features")
    if not isinstance(features, list) or not all(
        isinstance(feature, str) for feature in features
    ):
        raise RuntimeError("companion manifest contains an invalid feature list")
    dependencies = manifest.get("dependency_revisions")
    if not isinstance(dependencies, dict):
        raise RuntimeError("companion manifest omitted dependency revisions")
    transports = installation.capabilities.get("transport")
    transports = transports if isinstance(transports, dict) else {}
    if profile == "offline":
        if "dynamo-offline" not in features:
            raise RuntimeError("offline companion omitted the dynamo-offline feature")
        if "dynamo-aiperf-native" not in dependencies:
            raise RuntimeError("offline companion omitted the Dynamo source revision")
        if "dynosim_offline" not in transports:
            raise RuntimeError(
                "offline companion omitted the dynosim_offline transport"
            )
        return
    if any(feature.startswith("dynamo-") for feature in features):
        raise RuntimeError("online companion unexpectedly contains a Dynamo feature")
    if dependencies:
        raise RuntimeError(
            "online companion unexpectedly contains external dependency revisions"
        )
    if "dynosim_offline" in transports:
        raise RuntimeError(
            "online companion unexpectedly advertises an offline transport"
        )


def _probe_malformed_stdin(installation: RunnerInstallation) -> int:
    malformed = subprocess.run(
        [os.fspath(installation.binary)],
        input=b"{}\n",
        capture_output=True,
        check=False,
    )
    lines = [line for line in malformed.stdout.splitlines() if line.strip()]
    if malformed.returncode != 2 or len(lines) != 1:
        raise RuntimeError(
            "companion runner bootstrap smoke expected exit 2 and one terminal line; "
            f"received exit {malformed.returncode} and {len(lines)} lines"
        )
    terminal = orjson.loads(lines[0])
    if not isinstance(terminal, dict) or terminal.get("event") != "run_terminal":
        raise RuntimeError(
            "companion runner bootstrap smoke returned an invalid terminal"
        )
    if terminal.get("success") is not False:
        raise RuntimeError("malformed companion request unexpectedly succeeded")
    return malformed.returncode


def _online_run(artifact_dir: Path, url: str) -> BenchmarkRun:
    envelope = AIPerfConfig.model_validate(
        {
            "benchmark": {
                "models": ["mock-model"],
                "endpoint": {
                    "urls": [url],
                    "type": "chat",
                    "streaming": True,
                    "use_server_token_count": True,
                },
                "dataset": {
                    "type": "synthetic",
                    "entries": 1,
                    "isl": 8,
                    "osl": 1,
                },
                "profiling": {
                    "type": "concurrency",
                    "requests": 1,
                    "concurrency": 1,
                },
                "tokenizer": {"name": "builtin"},
                "artifacts": {"dir": str(artifact_dir), "records": False},
                "gpu_telemetry": {"enabled": False},
                "server_metrics": {"enabled": False},
                "network_latency": {"enabled": False},
                "runtime": {"ui": "none"},
            }
        }
    )
    return BenchmarkRun(
        benchmark_id="installed-online-release-smoke",
        cfg=envelope.benchmark,
        artifact_dir=artifact_dir,
        label="installed-online-release-smoke",
        random_seed=11,
        cli_command=None,
    )


def _offline_run(artifact_dir: Path) -> BenchmarkRun:
    envelope = AIPerfConfig.model_validate(
        {
            "benchmark": {
                "models": ["mock-model"],
                "endpoint": {
                    "urls": ["http://127.0.0.1:9"],
                    "type": "chat",
                    "streaming": True,
                },
                "backend": {
                    "type": "dynamo_offline",
                    "config": {"topology": "aggregated", "workers": 1},
                },
                "dataset": {
                    "type": "synthetic",
                    "entries": 1,
                    "prompts": {"isl": 8, "osl": 1},
                },
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "requests": 1,
                        "concurrency": 1,
                    }
                ],
                "tokenizer": {"name": "builtin"},
                "artifacts": {"dir": str(artifact_dir), "records": False},
                "gpu_telemetry": {"enabled": False},
                "server_metrics": {"enabled": False},
                "network_latency": {"enabled": False},
                "runtime": {"workers": 1, "ui": "none"},
            }
        }
    )
    return BenchmarkRun(
        benchmark_id="installed-offline-release-smoke",
        cfg=envelope.benchmark,
        artifact_dir=artifact_dir,
        label="installed-offline-release-smoke",
        random_seed=13,
        cli_command=None,
    )


def _execute_quietly(
    executor: RustSubprocessExecutor,
    run: BenchmarkRun,
) -> RunResult:
    """Keep compatibility-table rendering and expected rejection logs out of CI."""
    executor_logger = logging.getLogger("aiperf.orchestrator.rust_executor")
    previous_level = executor_logger.level
    executor_logger.setLevel(logging.CRITICAL + 1)
    try:
        with redirect_stdout(StringIO()):
            return executor.execute_sync(run)
    finally:
        executor_logger.setLevel(previous_level)


def _run_online_smoke(
    installation: RunnerInstallation,
    root: Path,
) -> dict[str, object]:
    _LoopbackChatHandler.request_count = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _LoopbackChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        artifact_dir = root / "online"
        result = _execute_quietly(
            RustSubprocessExecutor(root, installation=installation),
            _online_run(
                artifact_dir,
                f"http://127.0.0.1:{port}/v1/chat/completions",
            ),
        )
        if not result.success:
            raise RuntimeError(f"installed online release smoke failed: {result.error}")
        request_count = result.summary_metrics["request_count"].avg
        if request_count != 1.0 or _LoopbackChatHandler.request_count != 1:
            raise RuntimeError(
                "installed online release smoke did not complete exactly one request"
            )
        if not (artifact_dir / "native-v2.json").is_file():
            raise RuntimeError("installed online release smoke omitted native-v2.json")
        return {"requests": 1, "report": os.fspath(artifact_dir / "native-v2.json")}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _run_offline_gate(
    profile: RunnerProfile,
    installation: RunnerInstallation,
    root: Path,
) -> dict[str, object]:
    artifact_dir = root / "offline"
    result = _execute_quietly(
        RustSubprocessExecutor(root, installation=installation),
        _offline_run(artifact_dir),
    )
    if profile == "online":
        if result.success:
            raise RuntimeError("online-only companion unexpectedly executed offline")
        expected = "executable protocol-v2 pair ('dynamo_offline', 'scheduled')"
        if expected not in (result.error or ""):
            raise RuntimeError(
                "online-only companion did not reject offline at exact-image pair preflight: "
                f"{result.error}"
            )
        if artifact_dir.exists():
            raise RuntimeError(
                "online-only companion touched offline artifacts before pair rejection"
            )
        return {"rejected": True, "artifact_created": False}

    if not result.success:
        raise RuntimeError(f"installed offline release smoke failed: {result.error}")
    report_path = artifact_dir / "native-v2.json"
    report = orjson.loads(report_path.read_bytes())
    if report.get("run", {}).get("mode") != "offline:scheduled":
        raise RuntimeError(
            "installed offline release smoke returned the wrong run mode"
        )
    request_count = result.summary_metrics["request_count"].avg
    if request_count != 1.0:
        raise RuntimeError("installed offline release smoke completed the wrong count")
    return {"requests": 1, "report": os.fspath(report_path)}


def main(argv: list[str] | None = None) -> int:
    """Discover the wheel-owned runner and exercise every packaging gate."""
    arguments = _parser().parse_args(argv)
    profile: RunnerProfile = arguments.profile
    os.environ.pop("AIPERF_RUNNER_BIN", None)
    # The absolute interpreter launching this script remains available. An
    # empty PATH proves discovery used wheel RECORD metadata, not tier four.
    os.environ["PATH"] = ""
    installation = RunnerInstallation.resolve()
    manifest = _installed_manifest(installation)
    _verify_profile(profile, installation, manifest)
    bootstrap_exit = _probe_malformed_stdin(installation)
    with TemporaryDirectory(prefix="aiperf-runner-release-") as temporary:
        root = Path(temporary)
        online = _run_online_smoke(installation, root)
        offline = _run_offline_gate(profile, installation, root)

    print(
        orjson.dumps(
            {
                "binary": os.fspath(installation.binary),
                "profile": profile,
                "protocol_versions": installation.capabilities.get("protocol_versions"),
                "source_revision": manifest.get("source_revision"),
                "cargo_lock_sha256": manifest.get("cargo_lock_sha256"),
                "features": manifest.get("features"),
                "dependency_revisions": manifest.get("dependency_revisions"),
                "bootstrap_exit": bootstrap_exit,
                "online": online,
                "offline": offline,
            }
        ).decode()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
