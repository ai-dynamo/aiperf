#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AIPerf local Kubernetes development CLI.

Single-file script for building, deploying, and running AIPerf in a local
cluster (Kind by default, Minikube for GPU passthrough). Invoke directly::

    uv run python dev/kube.py setup
    ./dev/kube.py setup
"""

from __future__ import annotations

import functools
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

# Ensure project root is on sys.path so `from dev.versions import ...` works
# regardless of how the script is invoked (uv run, ./dev/kube.py, python dev/kube.py).
_PROJECT_DIR = str(Path(__file__).resolve().parent.parent)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import orjson
from cyclopts import App, Group, Parameter
from pydantic import BaseModel, Field
from rich.console import Console
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from ruamel.yaml import YAML

from dev.versions import DEVICE_PLUGIN_VERSION as _DPV_DEFAULT
from dev.versions import DYNAMO_VERSION as _DYNAMO_VERSION_DEFAULT
from dev.versions import JOBSET_CRD_URL_TEMPLATE as _JOBSET_CRD_URL_TEMPLATE
from dev.versions import JOBSET_VERSION as _JSV_DEFAULT

_ANSI_ESCAPE = re.compile(r"\033\[[0-9;]*m")

_yaml = YAML()
_yaml.default_flow_style = False


def _yaml_dump(doc: dict) -> str:
    stream = io.StringIO()
    _yaml.dump(doc, stream)
    return stream.getvalue()


# ---------------------------------------------------------------------------
# Output mode: --json and --yes global flags
# ---------------------------------------------------------------------------


@dataclass
class _OutputMode:
    """Controls output format and interactivity."""

    json: bool = False
    yes: bool = False
    _command: str = ""
    _result: dict | None = None
    _errors: list[str] = field(default_factory=list)
    _streaming: bool = False
    _summary_emitted: bool = False
    _real_stdout_fd: int | None = None

    def set_result(self, result: dict) -> None:
        self._result = result

    def add_error(self, msg: str) -> None:
        self._errors.append(msg)

    def emit_event(self, event: dict) -> None:
        """Write one JSONL event line to the real stdout fd."""
        line = orjson.dumps(event).decode() + "\n"
        if self._real_stdout_fd is not None:
            os.write(self._real_stdout_fd, line.encode())
        else:
            sys.stdout.write(line)
            sys.stdout.flush()

    def emit_summary(self) -> None:
        """Emit the final summary event (for streaming commands). Idempotent."""
        if self._summary_emitted:
            return
        self._summary_emitted = True
        self.emit_event(
            {
                "type": "summary",
                "command": self._command,
                "status": "error" if self._errors else "ok",
                "result": self._result,
                "errors": self._errors,
            }
        )

    def emit(self) -> None:
        """Print JSON result to stdout. Called at the end of __main__."""
        if not self.json:
            return
        if self._streaming:
            self.emit_summary()
            return
        out = {
            "type": "summary",
            "command": self._command,
            "status": "error" if self._errors else "ok",
            "result": self._result,
            "errors": self._errors,
        }
        sys.stdout.write(orjson.dumps(out).decode() + "\n")
        sys.stdout.flush()


_mode = _OutputMode()


class _StreamEmitter:
    """Captures subprocess stdout/stderr and emits JSONL events.

    Used as a context manager around long-running steps in JSON mode.
    Redirects OS-level fds 1 and 2 to pipes, reads lines in background
    threads, and emits {"type": "output", ...} events to the real stdout.
    """

    def __init__(self, step: str) -> None:
        self.step = step
        self._saved_fd1: int | None = None
        self._saved_fd2: int | None = None
        self._threads: list[threading.Thread] = []
        self._pipe_fds: list[int] = []

    def __enter__(self) -> _StreamEmitter:
        if not _mode.json:
            return self

        # Save real stdout fd for writing events
        if _mode._real_stdout_fd is None:
            _mode._real_stdout_fd = os.dup(1)

        # Save current fds
        self._saved_fd1 = os.dup(1)
        self._saved_fd2 = os.dup(2)

        # Create pipes and redirect
        for fd, stream_name in [(1, "stdout"), (2, "stderr")]:
            r_fd, w_fd = os.pipe()
            self._pipe_fds.append(r_fd)
            os.dup2(w_fd, fd)
            os.close(w_fd)

            t = threading.Thread(
                target=self._reader,
                args=(r_fd, stream_name),
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        # Also redirect Python-level streams
        sys.stdout = io.TextIOWrapper(os.fdopen(os.dup(1), "wb", 0), write_through=True)
        sys.stderr = io.TextIOWrapper(os.fdopen(os.dup(2), "wb", 0), write_through=True)

        return self

    def __exit__(self, *exc: Any) -> None:
        if not _mode.json or self._saved_fd1 is None:
            return

        # Restore OS-level fds (this closes the write end of pipes)
        os.dup2(self._saved_fd1, 1)
        os.dup2(self._saved_fd2, 2)
        os.close(self._saved_fd1)
        os.close(self._saved_fd2)
        self._saved_fd1 = None
        self._saved_fd2 = None

        # Restore Python streams
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        # Wait for reader threads to drain
        for t in self._threads:
            t.join(timeout=5)
        self._threads.clear()

    def _reader(self, read_fd: int, stream_name: str) -> None:
        """Background thread: read lines from pipe and emit events."""
        try:
            with os.fdopen(read_fd, "r", errors="replace") as f:
                for raw_line in f:
                    line = _ANSI_ESCAPE.sub("", raw_line.rstrip("\n"))
                    if not line.strip():
                        continue
                    _mode.emit_event(
                        {
                            "type": "output",
                            "step": self.step,
                            "stream": stream_name,
                            "line": line,
                        }
                    )
        except (OSError, ValueError):
            pass


# ---------------------------------------------------------------------------
# Constants (overridable via env vars)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLUSTER_NAME = os.environ.get("CLUSTER_NAME") or "aiperf"
AIPERF_IMAGE = os.environ.get("AIPERF_IMAGE") or "aiperf:local"
MOCK_SERVER_IMAGE = os.environ.get("MOCK_SERVER_IMAGE") or "aiperf-mock-server:latest"
JOBSET_VERSION = os.environ.get("JOBSET_VERSION") or _JSV_DEFAULT
DEVICE_PLUGIN_VERSION = os.environ.get("DEVICE_PLUGIN_VERSION") or _DPV_DEFAULT
MOCK_SERVER_MANIFEST = PROJECT_ROOT / "dev" / "deploy" / "mock-server.yaml"
DEFAULT_BENCHMARK_CONFIG = (
    PROJECT_ROOT / "dev" / "deploy" / "test-benchmark-config.yaml"
)
KIND_GPU_CONFIG_PATH = PROJECT_ROOT / "dev" / "deploy" / "kind-gpu-cluster.yaml"
NVIDIA_RUNTIME_CLASS_PATH = (
    PROJECT_ROOT / "dev" / "deploy" / "nvidia-runtime-class.yaml"
)
NVIDIA_DEVICE_PLUGIN_TMPL = (
    PROJECT_ROOT / "dev" / "deploy" / "nvidia-device-plugin.yaml.tmpl"
)

VLLM_IMAGE = os.environ.get("VLLM_IMAGE") or "vllm/vllm-openai:latest"
VLLM_MODEL = os.environ.get("MODEL") or "Qwen/Qwen3-0.6B"
VLLM_GPUS = int(os.environ.get("GPUS") or "1")
VLLM_MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN") or "4096")
VLLM_GPU_MEM_UTIL = os.environ.get("GPU_MEM_UTIL") or "0.5"
VLLM_NAMESPACE = "vllm-server"

SGLANG_IMAGE = os.environ.get("SGLANG_IMAGE") or "lmsysorg/sglang:latest"
SGLANG_NAMESPACE = "sglang-server"

TRTLLM_IMAGE = (
    os.environ.get("TRTLLM_IMAGE") or "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc7"
)
TRTLLM_NAMESPACE = "trtllm-server"

DYNAMO_IMAGE = (
    os.environ.get("DYNAMO_IMAGE")
    or f"nvcr.io/nvidia/ai-dynamo/vllm-runtime:{_DYNAMO_VERSION_DEFAULT}"
)
DYNAMO_MODE = os.environ.get("DYNAMO_MODE") or "agg"
DYNAMO_NAMESPACE = "dynamo-server"
DYNAMO_VERSION = os.environ.get("DYNAMO_VERSION") or _DYNAMO_VERSION_DEFAULT
# Single-GPU disaggregated: both prefill+decode share 1 GPU at low memory util
DYNAMO_1GPU_MEM_UTIL = os.environ.get("DYNAMO_1GPU_MEM_UTIL") or "0.3"

# Kueue release: manifests.yaml ships all CRDs + the controller deployment.
KUEUE_VERSION = os.environ.get("KUEUE_VERSION") or "v0.9.1"
_KUEUE_MANIFEST_URL_TEMPLATE = (
    "https://github.com/kubernetes-sigs/kueue/releases/download/"
    "{version}/manifests.yaml"
)
KUEUE_NAMESPACE = "kueue-system"
# Default queue pair seeded into the aiperf-benchmarks namespace so test
# workloads can reference them without extra setup.
AIPERF_BENCHMARK_NAMESPACE = "aiperf-benchmarks"
KUEUE_CLUSTER_QUEUE_NAME = "aiperf-cluster-queue"
KUEUE_LOCAL_QUEUE_NAME = "aiperf-local-queue"
KUEUE_RESOURCE_FLAVOR_NAME = "aiperf-default-flavor"

# kube-prometheus-stack bundles Prometheus + Grafana + Alertmanager.
PROMETHEUS_STACK_VERSION = os.environ.get("PROMETHEUS_STACK_VERSION") or "65.5.1"
PROMETHEUS_NAMESPACE = "monitoring"
PROMETHEUS_RELEASE_NAME = "kube-prometheus-stack"
PROMETHEUS_HELM_REPO_NAME = "prometheus-community"
PROMETHEUS_HELM_REPO_URL = "https://prometheus-community.github.io/helm-charts"

# loki-stack bundles Loki + Promtail for log aggregation.
LOKI_STACK_VERSION = os.environ.get("LOKI_STACK_VERSION") or "2.10.2"
LOKI_NAMESPACE = "monitoring"
LOKI_RELEASE_NAME = "loki-stack"
LOKI_HELM_REPO_NAME = "grafana"
LOKI_HELM_REPO_URL = "https://grafana.github.io/helm-charts"
# KV router mode for frontend (e.g. "kv" for KV-aware routing, "round-robin")
DYNAMO_ROUTER_MODE: str | None = os.environ.get("DYNAMO_ROUTER_MODE") or None
# KVBM CPU cache size in GB for prefill workers (enables KV cache offloading)
DYNAMO_KVBM_CPU_CACHE_GB: int | None = (
    int(v) if (v := os.environ.get("DYNAMO_KVBM_CPU_CACHE_GB")) else None
)
# Connectors for prefill workers (comma-separated, e.g. "kvbm,nixl")
DYNAMO_CONNECTORS: list[str] = (
    [c.strip() for c in v.split(",") if c.strip()]
    if (v := os.environ.get("DYNAMO_CONNECTORS"))
    else []
)


@functools.lru_cache(maxsize=1)
def _detect_docker_cpus() -> int:
    """Query Docker daemon for CPU count. Falls back to host CPU count.

    On Colima/Docker Desktop the VM may have fewer CPUs than the host,
    so `os.cpu_count()` alone would over-provision.
    """
    try:
        r = subprocess.run(
            ["docker", "info", "--format", "{{.NCPU}}"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return int(r.stdout.strip()) or 2
    except Exception:
        return os.cpu_count() or 2


@functools.lru_cache(maxsize=1)
def _detect_docker_memory_mb() -> int:
    """Query Docker daemon for total memory (MB).  Falls back to host memory."""
    try:
        r = subprocess.run(
            ["docker", "info", "--format", "{{.MemTotal}}"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        mem_bytes = int(r.stdout.strip())
        return mem_bytes // (1024 * 1024)
    except Exception:
        pass
    try:
        mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return mem // (1024 * 1024)
    except Exception:
        return 4000


_docker_mem_mb = _detect_docker_memory_mb()
# Use 75% of Docker memory, capped at 16 GB. Don't exceed what Docker has.
_default_memory_mb = min(
    _docker_mem_mb, max(2000, min(_docker_mem_mb - _docker_mem_mb // 4, 16000))
)

MINIKUBE_MEMORY = os.environ.get("MINIKUBE_MEMORY") or f"{_default_memory_mb}mb"
_docker_cpus = _detect_docker_cpus()
MINIKUBE_CPUS = os.environ.get("MINIKUBE_CPUS") or str(min(_docker_cpus, 8))

# Cluster runtime: "kind" or "minikube".
# GPU systems default to minikube (simpler GPU passthrough via --gpus=all).
# CPU-only systems default to kind (lighter weight, no driver needed).
# Override with CLUSTER_RUNTIME=kind or CLUSTER_RUNTIME=minikube.
_CLUSTER_RUNTIME_OVERRIDE = os.environ.get("CLUSTER_RUNTIME", "").lower() or None


def _effective_runtime() -> str:
    """Return the active cluster runtime ('kind' or 'minikube').

    Defaults to 'minikube' when a GPU is detected (simpler passthrough),
    'kind' otherwise. Override with CLUSTER_RUNTIME env var.
    """
    if _CLUSTER_RUNTIME_OVERRIDE:
        return _CLUSTER_RUNTIME_OVERRIDE
    if shutil.which("nvidia-smi") is not None:
        return "minikube"
    return "kind"


def _kubectl_context() -> str:
    """Return the kubectl context name for the active cluster runtime."""
    if _effective_runtime() == "kind":
        return f"kind-{CLUSTER_NAME}"
    return CLUSTER_NAME


# fmt: off
_BANNER = (
    "\n\n"
    "       ░▒▓ █▀█ █ █▀█ █▀▀ █▀█ █▀▀ ▓▒░\n"
    "    ░░▒▒▓▓ █▀█ █ █▀▀ ██▄ █▀▄ █▀  ▓▓▒▒░░\n"
    "\n"
    "        From prefill to production.\n"
)
# fmt: on

DEFAULT_CONFIG: str | None = os.environ.get("CONFIG") or None
DEFAULT_WORKERS = int(os.environ.get("WORKERS") or str(min(_docker_cpus, 10)))
FOLLOW_DEFAULT = bool(os.environ.get("FOLLOW"))

# Scale timeouts on constrained systems: 2 CPUs → 2x, 1 CPU → 4x, 8+ → 1x (no change)
_TIMEOUT_SCALE = max(1, 4 // _docker_cpus)

# Reusable cyclopts type alias for --follow/-f flags
FollowFlag = Annotated[
    bool, Parameter(name=["--follow", "-f"], help="Follow log output.")
]

# ---------------------------------------------------------------------------
# CLI app + groups + shared option models
# ---------------------------------------------------------------------------

_ENV_HELP = """\
Environment variables:
  CLUSTER_NAME       Cluster name (default: aiperf)
  CLUSTER_RUNTIME    kind|minikube (default: minikube w/ GPU, kind w/o)
  AIPERF_IMAGE       AIPerf image (default: aiperf:local)
  CONFIG             Benchmark config file
  WORKERS            Number of workers (default: 10, --workers-max)
  MODEL              Model name (default: Qwen/Qwen3-0.6B)
  GPUS               GPUs per instance (default: 1)
  VLLM_IMAGE         vLLM image (default: vllm/vllm-openai:latest)
  MAX_MODEL_LEN      Max context length (default: 4096)
  GPU_MEM_UTIL       GPU memory utilization (default: 0.5)
  HF_TOKEN           Hugging Face token (for gated models)
  DYNAMO_IMAGE       Dynamo image (default: nvcr.io/.../vllm-runtime:0.9.0)
  DYNAMO_MODE        agg|disagg|disagg-1gpu (default: agg)
  DYNAMO_VERSION     Dynamo operator version (default: 0.9.0)
  PLATFORM           Override platform detection (arch|debian|fedora|linux|mac)
  ARCH               Override Linux binary arch (amd64|arm64)

Examples:
  ./dev/kube.py setup
  ./dev/kube.py deploy-dynamo
  ./dev/kube.py deploy-dynamo --mode disagg-1gpu
  ./dev/kube.py deploy-vllm --model facebook/opt-125m
  ./dev/kube.py reload && ./dev/kube.py run
  ./dev/kube.py run -c my.yaml -w 20
  ./dev/kube.py deploy-lora --name my-lora --base-model Qwen/Qwen3-0.6B --source hf://org/repo
"""

app = App(
    name="aiperf-kube",
    help="AIPerf local Kubernetes development CLI (Kind / Minikube).",
    help_format="plaintext",
    help_epilogue=_ENV_HELP,
)
app.register_install_completion_command()


# Allow --json/--yes before the subcommand name (e.g. `./dev/kube.py --json doctor`).
# Each command also accepts these flags via OutputOptions so both orderings work.
@app.meta.default
def _meta_handler(
    *tokens: Annotated[str, Parameter(allow_leading_hyphen=True)],
    json_output: Annotated[
        bool,
        Parameter(name=["--json"], negative="", show=False),
    ] = False,
    yes: Annotated[
        bool,
        Parameter(name=["--yes", "-y"], negative="", show=False),
    ] = False,
) -> None:
    if json_output:
        _mode.json = True
    if yes:
        _mode.yes = True
    app(tokens)


workflow = Group.create_ordered("Workflow")
server = Group.create_ordered("Server")
benchmark = Group.create_ordered("Benchmark")
lowlevel = Group.create_ordered("Low-level")


# fmt: off
@Parameter(name="*")
class OutputOptions(BaseModel):
    """Global output flags available on every command."""
    json_output: Annotated[bool, Parameter(name=["--json"], negative="")] = Field(default=False, description="Emit structured JSON to stdout.")
    yes:         Annotated[bool, Parameter(name=["--yes", "-y"], negative="")] = Field(default=False, description="Auto-accept all confirmations.")

@Parameter(name="*")
class ModelOptions(BaseModel):
    """Shared model configuration for deploy commands."""
    model:         Annotated[str, Parameter(name=["--model", "-m"])] = Field(default=VLLM_MODEL,         description="Model name.")
    gpus:          Annotated[int, Parameter(name=["--gpus", "-g"])]  = Field(default=VLLM_GPUS,          description="GPUs per instance.")
    max_model_len: int                                               = Field(default=VLLM_MAX_MODEL_LEN, description="Max context length.")

@Parameter(name="*")
class RunOptions(BaseModel):
    """Benchmark run configuration."""
    config:  Annotated[str | None, Parameter(name=["--config", "-c"])] = Field(default=DEFAULT_CONFIG,  description="Benchmark config file.")
    workers: Annotated[int,        Parameter(name=["--workers-max", "-w"])] = Field(default=DEFAULT_WORKERS, description="Number of workers.")

@Parameter(name="*")
class DynamoDeployOptions(BaseModel):
    """Dynamo-specific deploy configuration."""
    dynamo_image:      str              = Field(default=DYNAMO_IMAGE,              description="Dynamo container image.")
    mode:              str              = Field(default=DYNAMO_MODE,               description="Deployment mode: agg|disagg|disagg-1gpu.")
    router_mode:       str | None       = Field(default=DYNAMO_ROUTER_MODE,        description='KV router mode (e.g. "kv", "round-robin").')
    kvbm_cpu_cache_gb: int | None       = Field(default=DYNAMO_KVBM_CPU_CACHE_GB,  description="KVBM CPU cache GB for prefill workers.")
    connectors:        list[str] | None = Field(default=DYNAMO_CONNECTORS or None, description="Connectors for prefill workers (e.g. kvbm nixl).")
# fmt: on


def _apply_output(opts: OutputOptions) -> None:
    """Set global output mode from per-command OutputOptions.

    Uses OR logic so flags set by the meta handler (before the subcommand name)
    are preserved even if the command's own OutputOptions defaults to False.
    """
    _mode.json = _mode.json or opts.json_output
    _mode.yes = _mode.yes or opts.yes


# B008: module-level defaults for Pydantic models used in function signatures
_DEFAULT_OUTPUT_OPTS = OutputOptions()
_DEFAULT_MODEL_OPTS = ModelOptions()
_DEFAULT_RUN_OPTS = RunOptions()
_DEFAULT_DYNAMO_OPTS = DynamoDeployOptions()

# ---------------------------------------------------------------------------
# ANSI / formatting helpers
# ---------------------------------------------------------------------------

# fmt: off
_CYAN   = "\033[0;36m"
_GREEN  = "\033[0;32m"
_BLUE   = "\033[0;34m"
_YELLOW = "\033[1;33m"
_RED    = "\033[0;31m"
_NC     = "\033[0m"

def _log_dest() -> Any:
    """Return stderr in JSON mode so stdout stays clean for JSON output."""
    return sys.stderr if _mode.json else sys.stdout

def log_step(msg: str) -> None:    print(f"\n{_CYAN}--- {msg}{_NC}", file=_log_dest(), flush=True)
def log_info(msg: str) -> None:    print(f"  {_BLUE}[info]{_NC}  {msg}", file=_log_dest(), flush=True)
def log_success(msg: str) -> None: print(f"  {_GREEN}[ok]{_NC}    {msg}", file=_log_dest(), flush=True)
def log_warn(msg: str) -> None:    print(f"  {_YELLOW}[warn]{_NC}  {msg}", file=_log_dest(), flush=True)

def log_error(msg: str) -> None:
    print(f"  {_RED}[error]{_NC} {msg}", file=sys.stderr, flush=True)
    _mode.add_error(msg)

def _ok(msg: str) -> str:   return f"  {_GREEN}\u2713{_NC} {msg}"
def _miss(msg: str) -> str: return f"  {_YELLOW}\u25cb{_NC} {msg}"
def _fail(msg: str) -> str: return f"  {_RED}\u2717{_NC} {msg}"
# fmt: on


# ---------------------------------------------------------------------------
# Shell helpers
# ---------------------------------------------------------------------------


_SH_DEFAULT_TIMEOUT = 120  # seconds


def sh(
    *cmd: str,
    check: bool = True,
    capture: bool = False,
    timeout: int | float | None = _SH_DEFAULT_TIMEOUT,
    **kwargs,
) -> subprocess.CompletedProcess:
    """Run a command, forwarding stdout/stderr unless *capture* is set."""
    kwargs.setdefault("text", capture)
    return subprocess.run(
        cmd, check=check, capture_output=capture, timeout=timeout, **kwargs
    )


def kubectl(*args: str, **kwargs) -> subprocess.CompletedProcess:
    """Run kubectl against the active cluster."""
    return sh("kubectl", "--context", _kubectl_context(), *args, **kwargs)


def minikube(*args: str, **kwargs) -> subprocess.CompletedProcess:
    """Run minikube with the cluster profile."""
    return sh("minikube", "-p", CLUSTER_NAME, *args, **kwargs)


def kindcli(*args: str, **kwargs) -> subprocess.CompletedProcess:
    """Run kind with the cluster name."""
    return sh("kind", *args, **kwargs)


def _strip_localhost_proxy() -> None:
    """Remove localhost-bound proxy env vars that break Kind containers.

    Kind propagates HTTP_PROXY/HTTPS_PROXY into the node. If the proxy
    is on 127.0.0.1 (e.g. corporate proxies, claude_proxy), it's
    unreachable from inside the container and breaks image pulls,
    apt-get, and curl. Use --keep-proxy to disable this.
    """
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        val = os.environ.get(var, "")
        if "127.0.0.1" in val or "localhost" in val:
            log_info(f"Stripping {var}={val} (unreachable from Kind node)")
            os.environ.pop(var, None)


def cluster_exists() -> bool:
    if _effective_runtime() == "kind":
        r = sh("kind", "get", "clusters", capture=True, check=False)
        return r.returncode == 0 and CLUSTER_NAME in r.stdout.splitlines()
    r = minikube("status", "--format={{.Host}}", capture=True, check=False)
    return r.returncode == 0 and r.stdout.strip() in ("Running", "Stopped")


def cluster_running() -> bool:
    if _effective_runtime() == "kind":
        # Kind clusters are always running if they exist (Docker containers).
        return cluster_exists()
    r = minikube("status", "--format={{.Host}}", capture=True, check=False)
    return r.returncode == 0 and r.stdout.strip() == "Running"


def require(*cmds: str) -> None:
    missing = [c for c in cmds if shutil.which(c) is None]
    if missing:
        log_error(f"Required command(s) not found: {', '.join(missing)}")
        raise SystemExit(1)


def require_cluster() -> None:
    if not cluster_running():
        rt = _effective_runtime()
        log_error(
            f"Cluster {CLUSTER_NAME} ({rt}) is not running. "
            "Run 'setup' or 'cluster-create' first."
        )
        raise SystemExit(1)


def _require_kubectl_and_cluster() -> None:
    """Require kubectl and a running cluster. Exits on failure."""
    require("kubectl")
    require_cluster()


def _require_docker_running() -> None:
    """Require Docker daemon to be running. Exits on failure."""
    r = sh("docker", "info", capture=True, check=False)
    if r.returncode != 0:
        log_error("Docker is not running. Please start Docker first.")
        raise SystemExit(1)


def _require_gpu() -> None:
    """Require GPU (nvidia-smi). Exits with hint on CPU-only systems."""
    if shutil.which("nvidia-smi") is None:
        log_error("GPU required. deploy-vllm and deploy-dynamo need GPU passthrough.")
        log_info("Use 'deploy-mock' + 'run' for CPU-only benchmarking.")
        raise SystemExit(1)


def _skip_if_deployment_ready(
    name: str,
    namespace: str | None,
    message: str,
) -> bool:
    """Return True if deployment exists and has ready replicas (caller should return)."""
    ready = _deployment_ready_replicas(name, namespace)
    if ready is not None and ready > 0:
        log_success(message)
        return True
    return False


def _deployment_ready_replicas(name: str, namespace: str | None = None) -> int | None:
    """Return ready replicas for a deployment, or None if it doesn't exist."""
    ns_args = ("-n", namespace) if namespace else ()
    r = kubectl(
        "get",
        "deployment",
        name,
        *ns_args,
        "-o",
        "jsonpath={.status.readyReplicas}",
        capture=True,
        check=False,
    )
    if r.returncode != 0:
        return None
    return int(r.stdout or "0")


def _get_aiperf_namespaces(*, latest_only: bool = False) -> list[str]:
    """Return names of aiperf-managed namespaces."""
    extra = (
        [
            "--sort-by=.metadata.creationTimestamp",
            "-o",
            "jsonpath={.items[-1].metadata.name}",
        ]
        if latest_only
        else ["-o", "jsonpath={.items[*].metadata.name}"]
    )
    r = kubectl(
        "get", "namespaces", "-l", "app=aiperf", *extra, capture=True, check=False
    )
    return r.stdout.split() if r.stdout.strip() else []


def _has_buildx() -> bool:
    """Check if docker buildx is available."""
    r = sh("docker", "buildx", "version", capture=True, check=False)
    return r.returncode == 0


def _docker_build(image: str, dockerfile: str, *extra_args: str) -> None:
    """Build a Docker image from PROJECT_ROOT.

    Uses `docker buildx build --load` when buildx is available (required on
    macOS for reliable cross-platform builds), falls back to `docker build`.
    """
    log_step(f"Building {image}")
    cmd = (
        ("docker", "buildx", "build", "--load")
        if _has_buildx()
        else ("docker", "build")
    )
    sh(
        *cmd,
        *extra_args,
        "-t",
        image,
        "-f",
        dockerfile,
        ".",
        cwd=PROJECT_ROOT,
        timeout=600,
    )
    log_success(f"Built {image}")


def _confirm(
    prompt: str, default_yes: bool = False, console: Console | None = None
) -> bool:
    """Prompt for confirmation. default_yes: [Y/n] vs [y/N]. Auto-accepts with --yes."""
    if _mode.yes:
        return True
    con = console or Console()
    try:
        return Confirm.ask(prompt, default=default_yes, console=con)
    except (EOFError, KeyboardInterrupt):
        return default_yes


def _confirm_with_all(
    prompt: str,
    default_yes: bool = False,
    console: Console | None = None,
) -> tuple[bool, bool]:
    """Prompt for confirmation with y/n/a (a = yes to all). Returns (proceed, yes_to_all).

    Auto-accepts with --yes.
    """
    if _mode.yes:
        return (True, True)
    con = console or Console()
    default = "y" if default_yes else "n"
    try:
        choice = Prompt.ask(
            f"{prompt} [y/n/a] (a = yes to all)",
            choices=["y", "n", "a"],
            default=default,
            show_choices=False,
            show_default=True,
            console=con,
        )
        return (
            choice in ("y", "a"),
            choice == "a",
        )
    except (EOFError, KeyboardInterrupt):
        return (default_yes, False)


def _run_install(cmds: list[str], *, console: Console | None = None) -> bool:
    """Run a list of shell commands, returning True if all succeed."""
    for cmd in cmds:
        if console is not None:
            console.print(Syntax(cmd, "bash", theme="monokai", line_numbers=False))
        else:
            print(f"  $ {cmd}")
        r = subprocess.run(cmd, shell=True, check=False)
        if r.returncode != 0:
            log_error(f"Command failed (exit {r.returncode}): {cmd}")
            return False
    return True


def _ensure_hf_token_secret(namespace: str) -> str | None:
    """Create HF token secret if HF_TOKEN env var is set. Returns secret name or None."""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        return None
    log_info("Creating HF token secret...")
    kubectl("create", "namespace", namespace, check=False)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=True) as f:
        f.write(hf_token)
        f.flush()
        kubectl(
            "create",
            "secret",
            "generic",
            "hf-token",
            f"--from-file=token={f.name}",
            "-n",
            namespace,
            check=False,
        )
    return "hf-token"


def _remove_server(label: str, namespace: str) -> None:
    """Remove a server by deleting its namespace."""
    log_step(f"Removing {label}")
    _require_kubectl_and_cluster()
    kubectl("delete", "namespace", namespace, "--ignore-not-found")
    log_success(f"{label} removed")


def _poll(
    check: callable,
    *,
    attempts: int = 60,
    interval: int | float = 2,
    progress_every: int = 10,
    fail_msg: str = "Timed out",
) -> bool:
    """Poll *check()* up to *attempts* times, sleeping *interval* between.

    *check* should return True on success. Logs progress every *progress_every*
    iterations. Returns True on success, raises SystemExit with *fail_msg* on
    exhaustion.
    """
    for i in range(attempts):
        if check():
            return True
        if progress_every and (i + 1) % progress_every == 0:
            log_info(f"Still waiting... ({i + 1}/{attempts})")
        time.sleep(interval)
    log_error(fail_msg)
    raise SystemExit(1)


def _manifests_to_yaml(documents: list[dict]) -> str:
    """Serialize a list of K8s manifest dicts to multi-document YAML."""
    return "\n---\n".join(_yaml_dump(doc) for doc in documents)


def _namespace_manifest(name: str, **extra_labels: str) -> dict:
    """Build a Namespace manifest dict."""
    labels = {"app": "aiperf", **extra_labels} if extra_labels else {}
    meta: dict = {"name": name}
    if labels:
        meta["labels"] = labels
    return {"apiVersion": "v1", "kind": "Namespace", "metadata": meta}


def _helm_install(
    release: str,
    chart_url: str,
    namespace: str,
    *extra_args: str,
) -> None:
    """Fetch a Helm chart from *chart_url*, install it, and clean up the tgz."""
    tgz = chart_url.rsplit("/", 1)[-1]
    sh("helm", "fetch", chart_url)
    try:
        sh(
            "helm",
            "install",
            release,
            tgz,
            "--kube-context",
            _kubectl_context(),
            "--namespace",
            namespace,
            *extra_args,
        )
    finally:
        if os.path.exists(tgz):
            os.unlink(tgz)


def _format_bytes_iec(size_bytes: int) -> str:
    """Format byte count in IEC units (KiB, MiB, ...). Portable (no numfmt)."""
    if size_bytes < 0:
        return "?"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}" if unit != "B" else f"{size_bytes}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}PiB"


def _get_gpu_allocatable() -> str:
    """Return the nvidia.com/gpu allocatable count from the cluster, or empty string."""
    r = kubectl(
        "get",
        "nodes",
        "-o",
        r"jsonpath={.items[0].status.allocatable.nvidia\.com/gpu}",
        capture=True,
        check=False,
    )
    return r.stdout.strip() if r.stdout else ""


# ---------------------------------------------------------------------------
# Doctor: platform detection + install recipes
# ---------------------------------------------------------------------------


def _detect_platform() -> str:
    """Return a platform key: 'mac', 'arch', 'debian', 'fedora', or 'linux'.

    Override with PLATFORM=arch|debian|fedora|linux|mac env var.
    """
    override = os.environ.get("PLATFORM", "").lower()
    if override in ("mac", "arch", "debian", "fedora", "linux"):
        return override
    if sys.platform == "darwin":
        return "mac"
    if sys.platform.startswith("linux"):
        try:
            os_release = Path("/etc/os-release").read_text().lower()
        except OSError:
            return "linux"
        # Match on ID= or ID_LIKE= to avoid false positives from URLs/descriptions
        id_values: set[str] = set()
        for line in os_release.splitlines():
            if m := re.match(r"(?:id|id_like)\s*=\s*\"?([^\"]+)\"?", line):
                id_values.update(m.group(1).split())
        if "arch" in id_values:
            return "arch"
        if id_values & {"debian", "ubuntu", "mint", "pop"}:
            return "debian"
        if id_values & {"fedora", "rhel", "centos", "rocky", "alma"}:
            return "fedora"
    return "linux"


def _detect_linux_arch() -> str:
    """Return Linux binary arch: 'amd64' or 'arm64'. Override with ARCH=amd64|arm64."""
    override = os.environ.get("ARCH", "").lower()
    if override in ("amd64", "arm64"):
        return override
    r = subprocess.run(["uname", "-m"], capture_output=True, text=True, check=False)
    machine = (r.stdout or "").strip().lower()
    if machine in ("aarch64", "arm64"):
        return "arm64"
    if machine in ("x86_64", "amd64"):
        return "amd64"
    return "amd64"


def _resolve_recipe_cmds(cmds: list[str], platform: str) -> list[str]:
    """Substitute {arch} and {install_prefix} in Linux recipe commands."""
    if platform not in ("linux", "arch", "debian", "fedora"):
        return cmds
    arch = _detect_linux_arch()
    install_prefix = os.environ.get("INSTALL_PREFIX", "/usr/local")
    return [
        c.replace("{arch}", arch).replace("{install_prefix}", install_prefix)
        for c in cmds
    ]


_KIND_LINUX_RECIPE: dict[str, list[str]] = {
    "cmds": [
        "curl -Lo ./kind https://kind.sigs.k8s.io/dl/latest/kind-linux-{arch}",
        "sudo install kind {install_prefix}/bin/kind",
        "rm -f kind",
    ],
    "post": [],
}
_MINIKUBE_LINUX_RECIPE: dict[str, list[str]] = {
    "cmds": [
        "curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-{arch}",
        "sudo install minikube-linux-{arch} {install_prefix}/bin/minikube",
        "rm -f minikube-linux-{arch}",
    ],
    "post": [],
}
_KUBECTL_LINUX_RECIPE: dict[str, list[str]] = {
    "cmds": [
        'curl -LO "https://dl.k8s.io/release/$(curl -sL https://dl.k8s.io/release/stable.txt)/bin/linux/{arch}/kubectl"',
        "sudo install -o root -g root -m 0755 kubectl {install_prefix}/bin/kubectl",
        "rm -f kubectl",
    ],
    "post": [],
}
_HELM_LINUX_RECIPE: dict[str, list[str]] = {
    "cmds": [
        "curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash"
    ],
    "post": [],
}
_K9S_LINUX_RECIPE: dict[str, list[str]] = {
    "cmds": [
        "curl -sL https://github.com/derailed/k9s/releases/latest/download/k9s_Linux_{arch}.tar.gz -o k9s.tar.gz",
        "tar xzf k9s.tar.gz",
        "sudo install k9s {install_prefix}/bin/k9s",
        "rm -f k9s k9s.tar.gz",
    ],
    "post": [],
}
_INSTALL_RECIPES: dict[str, dict[str, dict[str, list[str]]]] = {
    "docker": {
        "mac": {
            "cmds": ["brew install --cask docker"],
            "post": ["Open Docker Desktop to finish setup."],
        },
        "arch": {
            "cmds": ["sudo pacman -S --noconfirm docker"],
            "post": ["sudo systemctl enable --now docker"],
        },
        "debian": {
            "cmds": ["curl -fsSL https://get.docker.com | sh"],
            "post": ["sudo systemctl enable --now docker"],
        },
        "fedora": {
            "cmds": ["curl -fsSL https://get.docker.com | sh"],
            "post": ["sudo systemctl enable --now docker"],
        },
        "linux": {"cmds": ["curl -fsSL https://get.docker.com | sh"], "post": []},
    },
    "kind": {
        "mac": {"cmds": ["brew install kind"], "post": []},
        "arch": {"cmds": ["sudo pacman -S --noconfirm kind"], "post": []},
        "debian": _KIND_LINUX_RECIPE,
        "fedora": _KIND_LINUX_RECIPE,
        "linux": _KIND_LINUX_RECIPE,
    },
    "minikube": {
        "mac": {"cmds": ["brew install minikube"], "post": []},
        "arch": {"cmds": ["sudo pacman -S --noconfirm minikube"], "post": []},
        "debian": _MINIKUBE_LINUX_RECIPE,
        "fedora": _MINIKUBE_LINUX_RECIPE,
        "linux": _MINIKUBE_LINUX_RECIPE,
    },
    "kubectl": {
        "mac": {"cmds": ["brew install kubectl"], "post": []},
        "arch": {"cmds": ["sudo pacman -S --noconfirm kubectl"], "post": []},
        "debian": _KUBECTL_LINUX_RECIPE,
        "fedora": _KUBECTL_LINUX_RECIPE,
        "linux": _KUBECTL_LINUX_RECIPE,
    },
    "helm": {
        "mac": {"cmds": ["brew install helm"], "post": []},
        "arch": {"cmds": ["sudo pacman -S --noconfirm helm"], "post": []},
        "debian": _HELM_LINUX_RECIPE,
        "fedora": _HELM_LINUX_RECIPE,
        "linux": _HELM_LINUX_RECIPE,
    },
    "k9s": {
        "mac": {"cmds": ["brew install k9s"], "post": []},
        "arch": {"cmds": ["sudo pacman -S --noconfirm k9s"], "post": []},
        "debian": _K9S_LINUX_RECIPE,
        "fedora": {"cmds": ["sudo dnf install -y k9s"], "post": []},
        "linux": _K9S_LINUX_RECIPE,
    },
}

_PLATFORM_LABELS = {
    "mac": "macOS",
    "arch": "Arch Linux",
    "debian": "Debian/Ubuntu",
    "fedora": "Fedora/RHEL",
    "linux": "Linux",
}

_VERSION_CMDS: dict[str, tuple[list[str], str]] = {
    "docker": (["docker", "version", "--format", "{{.Client.Version}}"], "v{}"),
    "kubectl": (["kubectl", "version", "--client"], "{after_colon}"),
    "kind": (["kind", "version"], "{}"),
    "minikube": (["minikube", "version", "--short"], "{}"),
    "helm": (["helm", "version", "--short"], "{}"),
    "k9s": (["k9s", "version"], "{}"),
}


def _tool_version(tool: str) -> str:
    """Get a short version string for a tool, or empty string."""
    cmd, fmt = _VERSION_CMDS.get(tool, ([tool, "version"], "{}"))
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    stdout = (r.stdout or "").strip()
    if not stdout:
        return ""
    if tool == "k9s":
        for line in stdout.splitlines():
            if "Version:" in line:
                ver = _ANSI_ESCAPE.sub("", line.split("Version:", 1)[-1]).strip()
                return ver
        return ""
    line = stdout.splitlines()[0]
    if "{after_colon}" in fmt:
        line = line.split(": ", 1)[-1] if ": " in line else line
        return line
    return fmt.format(line)


def _required_tools() -> list[str]:
    """Return required tools based on the effective cluster runtime."""
    rt = _effective_runtime()
    if rt == "kind":
        return ["docker", "kind", "kubectl", "helm"]
    return ["docker", "minikube", "kubectl", "helm"]


_OPTIONAL_TOOLS = ["k9s"]


def _tools_status_rows(tools: list[str]) -> list[tuple[str, bool, str]]:
    """Return (tool, found, version) for each tool."""
    rows: list[tuple[str, bool, str]] = []
    for tool in tools:
        found = shutil.which(tool) is not None
        version = _tool_version(tool) if found else ""
        rows.append((tool, found, version or ("—" if not found else "")))
    return rows


def _doctor_tools_panel(
    rows: list[tuple[str, bool, str]],
    *,
    title: str,
    border_style: str,
    header_style: str,
    found_mark: str = "[green]✓[/]",
    missing_mark: str = "[red]✗[/]",
    detail_column: str = "Version",
) -> Panel:
    """Build a Rich Panel with a tool status table."""
    table = Table(show_header=True, header_style=header_style, box=None)
    table.add_column("Tool", style="bold")
    table.add_column("Status", width=6)
    table.add_column(detail_column)
    for tool, found, detail in rows:
        status = found_mark if found else missing_mark
        table.add_row(tool, status, detail or "—")
    return Panel(table, title=title, border_style=border_style)


def _check_tools(
    tools: list[str],
    *,
    title: str,
    border_style: str,
    header_style: str,
    found_mark: str = "[green]✓[/]",
    missing_mark: str = "[red]✗[/]",
    optional: bool = False,
    verbose: bool = True,
    console: Any = None,
) -> list[str]:
    """Check tools and optionally render status. Returns list of missing tool names."""
    rows = _tools_status_rows(tools)
    missing = [t for t, found, _ in rows if not found]
    if verbose and console is not None:
        console.print(
            _doctor_tools_panel(
                rows,
                title=title,
                border_style=border_style,
                header_style=header_style,
                found_mark=found_mark,
                missing_mark=missing_mark,
            )
        )
    elif verbose:
        for tool, found, version in rows:
            if found:
                print(_ok(f"{tool:10s} {version or ''}"))
            else:
                print(
                    _miss(f"{tool:10s} not found (optional cluster UI)")
                    if optional
                    else _fail(f"{tool:10s} not found")
                )
    return missing


def _check_prerequisites(*, verbose: bool = True, console: Any = None) -> list[str]:
    """Check for required tools. Returns list of missing tool names."""
    return _check_tools(
        _required_tools(),
        title="[bold cyan]Required[/]",
        border_style="cyan",
        header_style="bold cyan",
        verbose=verbose,
        console=console,
    )


def _check_optional_tools(*, verbose: bool = True, console: Any = None) -> list[str]:
    """Check optional tools (e.g. k9s). Returns list of missing."""
    return _check_tools(
        _OPTIONAL_TOOLS,
        title="[bold yellow]Optional (cluster UI)[/]",
        border_style="yellow",
        header_style="bold yellow",
        found_mark="[green]✓[/]",
        missing_mark="[yellow]○[/] optional",
        optional=True,
        verbose=verbose,
        console=console,
    )


def _preflight() -> None:
    """Quick prerequisite check for cmd_setup. Exits with hint on failure."""
    missing = _check_prerequisites(verbose=False)
    if missing:
        log_error(f"Missing prerequisites: {', '.join(missing)}")
        log_info("Run './dev/kube.py doctor' to install them.")
        raise SystemExit(1)


def _doctor_gpu_section(*, console: Any = None) -> None:
    """Show GPU prerequisites in doctor output."""
    nvidia_smi_ok = shutil.which("nvidia-smi") is not None
    nvidia_ctk_ok = shutil.which("nvidia-ctk") is not None
    nvidia_smi_detail = ""
    nvidia_ctk_detail = ""
    if nvidia_smi_ok:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        nvidia_smi_detail = (
            r.stdout.strip().splitlines()[0]
            if r.returncode == 0 and r.stdout.strip()
            else "installed"
        )
    if nvidia_ctk_ok:
        r = subprocess.run(
            ["nvidia-ctk", "--version"], capture_output=True, text=True, check=False
        )
        nvidia_ctk_detail = (
            r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "installed"
        )

    # Docker default runtime (relevant for Kind GPU passthrough)
    docker_runtime = _get_docker_default_runtime() if nvidia_smi_ok else ""
    docker_runtime_ok = docker_runtime == "nvidia"

    if console is not None:
        gpu_rows: list[tuple[str, bool, str]] = [
            (
                "nvidia-smi",
                nvidia_smi_ok,
                nvidia_smi_detail
                or "not found (install NVIDIA drivers for GPU support)",
            ),
            (
                "nvidia-ctk",
                nvidia_ctk_ok,
                nvidia_ctk_detail
                or "not found (install nvidia-container-toolkit for GPU support)",
            ),
        ]
        if nvidia_smi_ok:
            gpu_rows.append(
                (
                    "docker runtime",
                    docker_runtime_ok,
                    f"{docker_runtime} (Kind GPU needs nvidia)"
                    if not docker_runtime_ok
                    else "nvidia",
                )
            )
        console.print(
            _doctor_tools_panel(
                gpu_rows,
                title="[bold blue]GPU[/]",
                border_style="blue",
                header_style="bold blue",
                missing_mark="[yellow]○[/]",
                detail_column="Details",
            )
        )
    else:
        print(f"\n{_CYAN}GPU Prerequisites:{_NC}")
        if nvidia_smi_ok:
            print(_ok(f"nvidia-smi  {nvidia_smi_detail}"))
        else:
            print(
                _miss("nvidia-smi  not found (install NVIDIA drivers for GPU support)")
            )
        if nvidia_ctk_ok:
            print(_ok(f"nvidia-ctk  {nvidia_ctk_detail}"))
        else:
            print(
                _miss(
                    "nvidia-ctk  not found (install nvidia-container-toolkit for GPU support)"
                )
            )
        if nvidia_smi_ok:
            if docker_runtime_ok:
                print(_ok("docker runtime: nvidia"))
            else:
                print(
                    _miss(
                        f"docker runtime: {docker_runtime} "
                        "(Kind GPU needs nvidia — setup will configure this)"
                    )
                )


def _doctor_install_prompt(
    tool: str,
    cmds: list[str],
    default_yes: bool,
    recipe: dict,
    console: Any,
    *,
    yes_to_all: bool = False,
    offer_all: bool = True,
) -> tuple[bool, bool]:
    """Show install prompt and run install if confirmed. Returns (installed_ok, set_yes_to_all)."""
    block = "\n".join(cmds)
    console.print(
        Panel(
            Syntax(block, "bash", theme="monokai", line_numbers=False),
            title=f"[bold green]Install {tool}?[/]",
            border_style="green",
        )
    )
    if yes_to_all:
        proceed, set_all = True, False
    elif offer_all:
        proceed, set_all = _confirm_with_all(
            "\nRun the above?", default_yes=default_yes, console=console
        )
    else:
        proceed, set_all = (
            _confirm("\nRun the above?", default_yes=default_yes, console=console),
            False,
        )
    if not proceed:
        return (False, False)
    if _run_install(cmds, console=console):
        log_success(f"{tool} installed")
        for hint in recipe.get("post", []):
            log_info(f"  Note: {hint}")
        return (True, set_all)
    log_error(f"Failed to install {tool}")
    return (False, set_all)


def _doctor_offer_installs(
    tools: list[str],
    platform: str,
    default_yes: bool,
    console: Console,
    *,
    warn_no_recipe: bool = False,
    yes_to_all: bool = False,
) -> tuple[list[str], list[str]]:
    """Offer to install each tool. Returns (installed, skipped)."""
    installed: list[str] = []
    skipped: list[str] = []
    for tool in tools:
        recipe = _INSTALL_RECIPES.get(tool, {}).get(platform, {})
        cmds = _resolve_recipe_cmds(recipe.get("cmds", []), platform)
        if not cmds:
            if warn_no_recipe:
                log_warn(f"No auto-install recipe for {tool} on this platform.")
                skipped.append(tool)
            continue
        ok, set_yes_to_all = _doctor_install_prompt(
            tool, cmds, default_yes, recipe, console, yes_to_all=yes_to_all
        )
        if set_yes_to_all:
            yes_to_all = True
        if ok:
            installed.append(tool)
        else:
            if warn_no_recipe:
                log_info(f"Skipping {tool}")
            skipped.append(tool)
        if warn_no_recipe:
            print()
    return (installed, skipped)


def _collect_doctor() -> dict:
    """Collect doctor diagnostic info as a structured dict."""
    platform = _detect_platform()
    required_rows = _tools_status_rows(_required_tools())
    optional_rows = _tools_status_rows(_OPTIONAL_TOOLS)

    result: dict[str, Any] = {
        "platform": platform,
        "platform_label": _PLATFORM_LABELS.get(platform, platform),
        "required": [
            {"tool": t, "found": f, "version": v} for t, f, v in required_rows
        ],
        "optional": [
            {"tool": t, "found": f, "version": v} for t, f, v in optional_rows
        ],
        "missing_required": [t for t, f, _ in required_rows if not f],
        "missing_optional": [t for t, f, _ in optional_rows if not f],
        "docker_running": sh("docker", "info", capture=True, check=False).returncode
        == 0,
        "gpu": {},
    }

    nvidia_smi_ok = shutil.which("nvidia-smi") is not None
    nvidia_ctk_ok = shutil.which("nvidia-ctk") is not None
    result["gpu"]["nvidia_smi"] = nvidia_smi_ok
    result["gpu"]["nvidia_ctk"] = nvidia_ctk_ok
    if nvidia_smi_ok:
        result["gpu"]["docker_runtime"] = _get_docker_default_runtime()
    return result


def cmd_doctor() -> None:
    """Check prerequisites interactively. Offer to install missing tools."""
    yes_to_all = _mode.yes

    if _mode.json:
        _mode.set_result(_collect_doctor())
        return

    con = Console()
    con.print(Rule("[bold cyan]Doctor — Prerequisites[/]", style="cyan"))
    platform = _detect_platform()
    con.print(
        Panel(
            _PLATFORM_LABELS.get(platform, platform),
            title="[bold]Platform[/]",
            border_style="dim",
        )
    )

    missing = _check_prerequisites(verbose=True, console=con)
    missing_optional = _check_optional_tools(verbose=True, console=con)

    if not missing:
        r = sh("docker", "info", capture=True, check=False)
        if r.returncode != 0:
            print()
            log_warn("Docker is installed but not running. Start it with:")
            if platform == "mac":
                print("  open -a Docker")
            elif shutil.which("systemctl"):
                print("  sudo systemctl start docker")
            else:
                print("  Start Docker (see your distro docs, e.g. systemd or service)")
        else:
            print()
            log_success(
                "All prerequisites met! Run './dev/kube.py setup' to get started."
            )
        _doctor_gpu_section(console=con)
        _doctor_offer_installs(
            missing_optional, platform, True, con, yes_to_all=yes_to_all
        )
        if r.returncode == 0 and _confirm(
            "Run setup now?", default_yes=True, console=con
        ):
            print()
            log_info("Running setup...")
            cmd_setup(continue_on_error=True)
        return

    # Interactively install missing tools
    print()
    installed, skipped = _doctor_offer_installs(
        missing, platform, False, con, warn_no_recipe=True, yes_to_all=yes_to_all
    )
    if installed:
        log_success(f"Installed: {', '.join(installed)}")
    if skipped:
        log_warn(f"Still missing: {', '.join(skipped)}")
        print("\nRun './dev/kube.py doctor' again after installing manually.")
    elif not skipped:
        log_success("All prerequisites met! Run './dev/kube.py setup' to get started.")

    _doctor_offer_installs(missing_optional, platform, True, con, yes_to_all=yes_to_all)
    if not skipped and _confirm("Run setup now?", default_yes=True, console=con):
        print()
        log_info("Running setup...")
        cmd_setup(continue_on_error=True)


# ---------------------------------------------------------------------------
# Cluster management
# ---------------------------------------------------------------------------


def _get_docker_default_runtime() -> str:
    """Return the Docker default runtime name (e.g. 'runc', 'nvidia')."""
    r = sh("docker", "info", capture=True, check=False)
    for line in (r.stdout or "").splitlines():
        if "Default Runtime" in line:
            return line.split()[-1]
    return "unknown"


def _nvidia_volume_mounts_enabled() -> bool:
    """Check if nvidia-container-runtime volume mounts are enabled."""
    config_path = Path("/etc/nvidia-container-runtime/config.toml")
    if not config_path.exists():
        return False
    return (
        "accept-nvidia-visible-devices-as-volume-mounts = true"
        in config_path.read_text()
    )


def _configure_docker_nvidia() -> None:
    """Ensure Docker is configured with nvidia as default runtime. May require sudo."""
    log_step("Checking Docker runtime configuration")

    default_runtime = _get_docker_default_runtime()
    needs_runtime = default_runtime != "nvidia"
    needs_volume = not _nvidia_volume_mounts_enabled()

    if needs_runtime:
        print(_miss(f"Docker default runtime: {default_runtime} (needs nvidia)"))
    else:
        print(_ok("Docker default runtime: nvidia"))

    if needs_volume:
        print(_miss("nvidia volume mounts: disabled"))
    else:
        print(_ok("nvidia volume mounts: enabled"))

    if not needs_runtime and not needs_volume:
        log_success("Docker configuration looks good")
        return

    print()
    print(f"  {_YELLOW}The following changes require sudo:{_NC}")
    if needs_runtime:
        print("    - Set Docker default runtime to nvidia")
    if needs_volume:
        print("    - Enable nvidia-ctk volume mounts")
    print("    - Restart Docker daemon")
    print()

    if not _confirm("Apply these changes?"):
        log_error("Cannot proceed without Docker configuration.")
        raise SystemExit(1)

    cmds: list[str] = []
    if needs_volume:
        cmds.append(
            "sudo nvidia-ctk config --in-place"
            " --set accept-nvidia-visible-devices-as-volume-mounts=true"
        )
    if needs_runtime:
        cmds.append(
            "sudo nvidia-ctk runtime configure --runtime=docker --set-as-default"
        )
    cmds.append("sudo systemctl restart docker")

    if not _run_install(cmds):
        log_error("Failed to configure Docker")
        raise SystemExit(1)

    log_info("Waiting for Docker to restart...")
    for _i in range(30):
        if sh("docker", "info", capture=True, check=False).returncode == 0:
            break
        time.sleep(1)
    else:
        log_error("Docker failed to restart within 30s")
        raise SystemExit(1)

    if _get_docker_default_runtime() != "nvidia":
        log_error("Docker default runtime still not nvidia after configuration")
        raise SystemExit(1)

    log_success("Docker configured for NVIDIA GPU support")


def _verify_gpu_in_docker() -> None:
    """Verify GPU is accessible inside Docker containers."""
    log_step("Verifying GPU access inside Docker")
    r = sh(
        "docker",
        "run",
        "--rm",
        "nvidia/cuda:12.6.3-base-ubuntu24.04",
        "nvidia-smi",
        "--query-gpu=name",
        "--format=csv,noheader",
        capture=True,
        check=False,
    )
    if r.returncode == 0 and r.stdout.strip():
        log_success(f"GPU accessible in Docker ({r.stdout.strip()})")
    else:
        log_warn("Could not verify GPU inside Docker container")
        if not _confirm("Continue anyway?"):
            raise SystemExit(1)


def _install_nvidia_in_kind_node() -> None:
    """Install NVIDIA Container Toolkit inside the Kind node and configure containerd."""
    log_step("Installing NVIDIA Container Toolkit inside Kind node")
    node = f"{CLUSTER_NAME}-control-plane"

    def node_exec(
        *args: str,
        check: bool = True,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess:
        return sh("docker", "exec", node, *args, check=check, **kwargs)

    log_info("Unmounting masked /proc/driver/nvidia...")
    node_exec(
        "bash", "-c", "umount -R /proc/driver/nvidia 2>/dev/null || true", check=False
    )

    log_info("Verifying NVIDIA libraries inside Kind node...")
    r = node_exec(
        "bash", "-c", "ldconfig -p 2>/dev/null | grep -q libnvidia-ml", check=False
    )
    if r.returncode == 0:
        log_success("NVIDIA libraries present (injected by Docker nvidia runtime)")
    else:
        r2 = node_exec("bash", "-c", "nvidia-smi", capture=True, check=False)
        if r2.returncode == 0:
            log_success("nvidia-smi works inside Kind node")
        else:
            log_warn("NVIDIA libraries not detected inside Kind node")
            if not _confirm("Continue anyway? (toolkit install may fix this)"):
                raise SystemExit(1)

    log_info("Installing nvidia-container-toolkit packages (this may take a minute)...")
    # Unset proxy env vars that may point to host-only proxies (e.g. 127.0.0.1)
    # which are unreachable from inside the Kind node container.
    install_cmd = (
        "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; "
        "apt-get update -qq >/dev/null 2>&1 && "
        "apt-get install -y -qq gpg curl >/dev/null 2>&1 && "
        "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | "
        "gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null && "
        "curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | "
        "sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | "
        "tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null && "
        "apt-get update -qq >/dev/null 2>&1 && "
        "apt-get install -y -qq nvidia-container-toolkit >/dev/null 2>&1"
    )
    r = node_exec("bash", "-c", install_cmd, check=False)
    if r.returncode != 0:
        log_error("Failed to install nvidia-container-toolkit inside Kind node")
        log_error(
            f"Try: docker exec {node} apt-get install -y nvidia-container-toolkit"
        )
        raise SystemExit(1)
    log_success("nvidia-container-toolkit installed inside Kind node")

    log_info("Configuring containerd for CDI inside Kind node...")
    node_exec(
        "nvidia-ctk",
        "config",
        "--set",
        "nvidia-container-runtime.modes.cdi.annotation-prefixes=nvidia.cdi.k8s.io/",
    )
    node_exec(
        "nvidia-ctk",
        "runtime",
        "configure",
        "--runtime=containerd",
        "--cdi.enabled",
        "--config-source=command",
    )
    log_success("containerd configured for NVIDIA CDI")

    log_info("Generating CDI spec inside Kind node...")
    node_exec(
        "bash",
        "-c",
        "mkdir -p /var/run/cdi && "
        "nvidia-ctk cdi generate --driver-root=/ --output=/var/run/cdi/nvidia.yaml",
    )
    log_success("CDI spec generated")

    log_info("Restarting containerd inside Kind node...")
    node_exec("bash", "-c", "systemctl restart containerd")

    _poll(
        lambda: node_exec("crictl", "info", capture=True, check=False).returncode == 0,
        attempts=30,
        interval=1,
        progress_every=0,
        fail_msg="containerd failed to restart inside Kind node",
    )
    log_success("containerd restarted")

    log_info("Waiting for node to become Ready...")

    def _node_ready() -> bool:
        r = kubectl("get", "nodes", "--no-headers", capture=True, check=False)
        parts = (r.stdout or "").strip().split()
        return len(parts) >= 2 and parts[1] == "Ready"

    _poll(
        _node_ready,
        attempts=60,
        interval=2,
        fail_msg="Node did not become Ready within 120s",
    )
    log_success("Node is Ready")


def _install_device_plugin() -> None:
    """Install NVIDIA GPU device plugin DaemonSet into the cluster."""
    log_step(f"Installing NVIDIA GPU device plugin ({DEVICE_PLUGIN_VERSION})")
    node = f"{CLUSTER_NAME}-control-plane"

    kubectl("label", "node", node, "--overwrite", "nvidia.com/gpu.present=true")
    log_info("Node labeled with nvidia.com/gpu.present=true")

    manifest = NVIDIA_DEVICE_PLUGIN_TMPL.read_text().replace(
        "${DEVICE_PLUGIN_VERSION}",
        DEVICE_PLUGIN_VERSION,
    )
    kubectl("apply", "-f", "-", input=manifest, text=True)

    log_info("Waiting for device plugin rollout...")
    kubectl(
        "-n",
        "kube-system",
        "rollout",
        "status",
        "daemonset/nvidia-device-plugin-daemonset",
        f"--timeout={120 * _TIMEOUT_SCALE}s",
    )
    log_success("Device plugin daemonset ready")

    log_info("Waiting for nvidia.com/gpu in node allocatable...")
    time.sleep(5)

    gpu_count = ""

    def _gpu_allocatable() -> bool:
        nonlocal gpu_count
        r = kubectl(
            "get",
            "nodes",
            "-o",
            r"jsonpath={.items[0].status.allocatable.nvidia\.com/gpu}",
            capture=True,
            check=False,
        )
        try:
            if int(r.stdout.strip()) > 0:
                gpu_count = r.stdout.strip()
                return True
        except (ValueError, AttributeError):
            pass
        return False

    try:
        _poll(
            _gpu_allocatable,
            attempts=60,
            interval=2,
            fail_msg="nvidia.com/gpu not found in node allocatable after 120s",
        )
    except SystemExit:
        kubectl(
            "-n",
            "kube-system",
            "logs",
            "-l",
            "name=nvidia-device-plugin-ds",
            "--tail=20",
            check=False,
        )
        raise
    log_success(f"nvidia.com/gpu: {gpu_count}")


def _kind_cluster_create() -> str:
    """Create a Kind cluster with optional GPU passthrough."""
    has_gpu = shutil.which("nvidia-smi") is not None

    if has_gpu:
        log_info("GPU detected — creating Kind cluster with GPU passthrough")
        _configure_docker_nvidia()
        _verify_gpu_in_docker()

        config = KIND_GPU_CONFIG_PATH.read_text()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config)
            config_path = f.name

        try:
            kindcli(
                "create",
                "cluster",
                "--name",
                CLUSTER_NAME,
                "--config",
                config_path,
                "--wait",
                f"{120 * _TIMEOUT_SCALE}s",
                timeout=600,
            )
        finally:
            os.unlink(config_path)

        kubectl("cluster-info", capture=True)

        # Verify GPU devices visible inside Kind node
        node = f"{CLUSTER_NAME}-control-plane"
        r = sh(
            "docker",
            "exec",
            node,
            "ls",
            "/dev/nvidia0",
            capture=True,
            check=False,
        )
        if r.returncode == 0:
            log_success("GPU devices visible inside Kind node")
        else:
            log_error("GPU devices NOT visible inside Kind node")
            log_error("The nvidia runtime isn't passing devices to the Kind container.")
            if _confirm("Delete the broken cluster?"):
                kindcli("delete", "cluster", "--name", CLUSTER_NAME)
            raise SystemExit(1)

        _install_nvidia_in_kind_node()

        kubectl("apply", "-f", str(NVIDIA_RUNTIME_CLASS_PATH))
        log_info("Created nvidia RuntimeClass")

        _install_device_plugin()

        log_success(f"Cluster {CLUSTER_NAME} created (context: kind-{CLUSTER_NAME})")
        return "created (Kind, GPU)"
    else:
        log_info("No GPU detected — creating CPU-only Kind cluster")
        kindcli(
            "create",
            "cluster",
            "--name",
            CLUSTER_NAME,
            "--wait",
            f"{120 * _TIMEOUT_SCALE}s",
            timeout=600,
        )
        kubectl("cluster-info", capture=True)
        log_success(f"Cluster {CLUSTER_NAME} created (context: kind-{CLUSTER_NAME})")
        return "created (Kind, CPU-only)"


def _minikube_cluster_create() -> str:
    """Create a Minikube cluster with optional GPU passthrough."""
    has_gpu = shutil.which("nvidia-smi") is not None
    gpu_args = ["--gpus", "all"] if has_gpu else []
    if has_gpu:
        log_info("GPU detected — enabling GPU passthrough")
    else:
        log_info("No GPU detected — creating CPU-only cluster")

    minikube(
        "start",
        "--driver",
        "docker",
        "--container-runtime",
        "docker",
        *gpu_args,
        "--memory",
        MINIKUBE_MEMORY,
        "--cpus",
        MINIKUBE_CPUS,
        timeout=600,
    )
    kubectl("cluster-info", capture=True)

    if has_gpu:
        kubectl("apply", "-f", str(NVIDIA_RUNTIME_CLASS_PATH))
        log_info("Created nvidia RuntimeClass")

    mode = "GPU" if has_gpu else "CPU-only"
    log_success(f"Cluster {CLUSTER_NAME} created (context: {CLUSTER_NAME})")
    return f"created (Minikube, {mode})"


def cmd_cluster_create() -> str:
    rt = _effective_runtime()
    log_step(f"Creating {rt} cluster: {CLUSTER_NAME}")
    require("kubectl", "docker")
    _require_docker_running()

    if cluster_running():
        log_success(f"Cluster {CLUSTER_NAME} already running")
        result = "already running"
    elif rt == "kind":
        require("kind")
        result = _kind_cluster_create()
    elif cluster_exists():
        require("minikube")
        log_info(f"Cluster {CLUSTER_NAME} exists but stopped, starting...")
        minikube("start", timeout=600)
        log_success(f"Cluster {CLUSTER_NAME} started")
        result = "started"
    else:
        require("minikube")
        result = _minikube_cluster_create()

    _mode.set_result({"cluster": CLUSTER_NAME, "runtime": rt, "message": result})
    return result


def _detect_cluster_runtime() -> str | None:
    """Detect which runtime the cluster actually exists in (Kind or Minikube)."""
    if shutil.which("kind"):
        r = sh("kind", "get", "clusters", capture=True, check=False)
        if r.returncode == 0 and CLUSTER_NAME in r.stdout.splitlines():
            return "kind"
    if shutil.which("minikube"):
        r = minikube("status", "--format={{.Host}}", capture=True, check=False)
        if r.returncode == 0 and r.stdout.strip() in ("Running", "Stopped"):
            return "minikube"
    return None


def cmd_cluster_delete() -> None:
    rt = _detect_cluster_runtime() or _effective_runtime()
    log_step(f"Deleting {rt} cluster: {CLUSTER_NAME}")

    if not _detect_cluster_runtime():
        log_warn(f"Cluster {CLUSTER_NAME} does not exist")
        return

    if rt == "kind":
        require("kind")
        kindcli("delete", "cluster", "--name", CLUSTER_NAME)
    else:
        require("minikube")
        minikube("delete")

    log_success(f"Cluster {CLUSTER_NAME} deleted")


# ---------------------------------------------------------------------------
# Build & load
# ---------------------------------------------------------------------------


def cmd_build(*, aiperf: bool = True, mock: bool = True) -> str:
    require("docker")
    built: list[str] = []
    if _mode.json:
        _mode._streaming = True
    if aiperf:
        if _mode.json:
            _mode.emit_event(
                {
                    "type": "step_started",
                    "step": f"build {AIPERF_IMAGE}",
                    "timestamp": time.time(),
                }
            )
        with _StreamEmitter(f"build {AIPERF_IMAGE}"):
            _docker_build(AIPERF_IMAGE, "Dockerfile", "--target", "runtime")
        built.append(AIPERF_IMAGE)
        if _mode.json:
            _mode.emit_event(
                {
                    "type": "step_completed",
                    "step": f"build {AIPERF_IMAGE}",
                    "detail": "built",
                }
            )
    if mock:
        if _mode.json:
            _mode.emit_event(
                {
                    "type": "step_started",
                    "step": f"build {MOCK_SERVER_IMAGE}",
                    "timestamp": time.time(),
                }
            )
        with _StreamEmitter(f"build {MOCK_SERVER_IMAGE}"):
            _docker_build(MOCK_SERVER_IMAGE, "dev/deploy/Dockerfile.mock-server")
        built.append(MOCK_SERVER_IMAGE)
        if _mode.json:
            _mode.emit_event(
                {
                    "type": "step_completed",
                    "step": f"build {MOCK_SERVER_IMAGE}",
                    "detail": "built",
                }
            )
    _mode.set_result({"images": built})
    return ", ".join(built)


def _local_image_id(image: str) -> str | None:
    """Get local Docker image ID (sha256:…), or None if not found."""
    r = sh("docker", "inspect", "--format={{.Id}}", image, capture=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else None


def _cluster_image_id(image: str) -> str | None:
    """Get image ID inside the cluster, or None if not loaded."""
    rt = _effective_runtime()
    if rt == "kind":
        # Check via crictl on the Kind control-plane node.
        node = f"{CLUSTER_NAME}-control-plane"
        r = sh(
            "docker",
            "exec",
            node,
            "crictl",
            "images",
            "--no-trunc",
            "-o",
            "json",
            capture=True,
            check=False,
        )
        if r.returncode != 0:
            return None
        needle = image if ":" in image else f"{image}:latest"
        data = orjson.loads(r.stdout)
        for img in data.get("images", []):
            for tag in img.get("repoTags", []):
                bare = tag.removeprefix("docker.io/library/")
                if bare == needle or tag == needle:
                    return img.get("id", "")
        return None

    # Minikube: inspect inside the VM's Docker daemon.
    r = minikube(
        "ssh",
        "--",
        "docker",
        "inspect",
        "--format={{.Id}}",
        image,
        capture=True,
        check=False,
    )
    return r.stdout.strip() if r.returncode == 0 else None


def cmd_load(*, aiperf: bool = True, mock: bool = True) -> str:
    rt = _effective_runtime()
    require("kind" if rt == "kind" else "minikube")
    require_cluster()

    loaded: list[str] = []
    skipped: list[str] = []

    def _load(image: str) -> None:
        local_id = _local_image_id(image)
        if not local_id:
            log_error(f"Image {image} not found. Run 'build' first.")
            raise SystemExit(1)
        if local_id == _cluster_image_id(image):
            log_success(f"{image} already loaded (up to date)")
            skipped.append(image)
            return
        log_step(f"Loading {image} into {rt}")
        if rt == "kind":
            kindcli("load", "docker-image", image, "--name", CLUSTER_NAME, timeout=300)
        else:
            minikube("image", "load", image, timeout=300)
        log_success(f"Loaded {image}")
        loaded.append(image)

    if aiperf:
        _load(AIPERF_IMAGE)
    if mock:
        _load(MOCK_SERVER_IMAGE)

    _mode.set_result({"loaded": loaded, "skipped": skipped, "runtime": rt})
    if not loaded:
        return "already loaded (up to date)"
    n = len(loaded)
    return f"{n} image{'s' if n > 1 else ''} -> {rt}"


# ---------------------------------------------------------------------------
# JobSet controller
# ---------------------------------------------------------------------------


def cmd_install_jobset() -> str:
    log_step(f"Installing JobSet controller {JOBSET_VERSION}")
    _require_kubectl_and_cluster()

    if _skip_if_deployment_ready(
        "jobset-controller-manager",
        "jobset-system",
        "JobSet controller already installed",
    ):
        return "already installed"

    log_info("Applying JobSet manifests...")
    kubectl(
        "apply",
        "--server-side",
        "-f",
        _JOBSET_CRD_URL_TEMPLATE.format(version=JOBSET_VERSION),
    )
    log_info("Waiting for JobSet controller to be ready...")
    kubectl(
        "wait",
        "--for=condition=available",
        f"--timeout={120 * _TIMEOUT_SCALE}s",
        "deployment/jobset-controller-manager",
        "-n",
        "jobset-system",
    )
    log_success(f"JobSet controller {JOBSET_VERSION} installed and ready")
    return f"{JOBSET_VERSION} installed"


# ---------------------------------------------------------------------------
# AIPerf operator (Helm from local chart)
# ---------------------------------------------------------------------------

AIPERF_OPERATOR_CHART = PROJECT_ROOT / "deploy" / "helm" / "aiperf-operator"
AIPERF_OPERATOR_NAMESPACE = "aiperf-system"


def cmd_install_aiperf_operator() -> str:
    """Install the AIPerf operator from the local Helm chart."""
    log_step("Installing AIPerf operator")
    require("helm")
    _require_kubectl_and_cluster()

    if _skip_if_deployment_ready(
        "aiperf-operator",
        AIPERF_OPERATOR_NAMESPACE,
        "AIPerf operator already installed",
    ):
        return "already installed"

    helm_args = [
        "install",
        "aiperf-operator",
        str(AIPERF_OPERATOR_CHART),
        "--namespace",
        AIPERF_OPERATOR_NAMESPACE,
        "--create-namespace",
    ]

    # For local clusters, use the locally built image
    repo, tag = (
        AIPERF_IMAGE.rsplit(":", 1) if ":" in AIPERF_IMAGE else (AIPERF_IMAGE, "latest")
    )
    helm_args += [
        "--set",
        f"image.repository={repo}",
        "--set",
        f"image.tag={tag}",
        "--set",
        "image.pullPolicy=Never",
    ]

    sh("helm", *helm_args)

    log_info("Waiting for AIPerf operator to be ready...")
    kubectl(
        "wait",
        "--for=condition=available",
        f"--timeout={120 * _TIMEOUT_SCALE}s",
        "deployment/aiperf-operator",
        "-n",
        AIPERF_OPERATOR_NAMESPACE,
    )
    log_success("AIPerf operator installed and ready")
    return "installed"


# ---------------------------------------------------------------------------
# Dynamo operator (Helm from NGC)
# ---------------------------------------------------------------------------


def cmd_install_dynamo() -> str:
    """Install Dynamo operator via Helm from NGC."""
    log_step(f"Installing Dynamo operator {DYNAMO_VERSION}")
    require("helm")
    _require_kubectl_and_cluster()

    # Check if already installed
    r = kubectl(
        "get", "crd", "dynamographdeployments.nvidia.com", capture=True, check=False
    )
    if r.returncode == 0:
        log_success("Dynamo operator already installed")
        return "already installed"

    ngc_base = "https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts"

    log_info("Fetching Dynamo CRDs chart...")
    _helm_install(
        "dynamo-crds", f"{ngc_base}/dynamo-crds-{DYNAMO_VERSION}.tgz", "default"
    )

    log_info("Fetching Dynamo platform chart...")
    _helm_install(
        "dynamo-platform",
        f"{ngc_base}/dynamo-platform-{DYNAMO_VERSION}.tgz",
        "dynamo-system",
        "--create-namespace",
        "--set",
        "dynamo-operator.webhook.enabled=false",
        "--set",
        "grove.enabled=false",
        "--set",
        "kai-scheduler.enabled=false",
    )

    # Wait for operator deployment
    log_info("Waiting for Dynamo operator to be ready...")

    def _operator_running() -> bool:
        r = kubectl(
            "get",
            "pods",
            "-n",
            "dynamo-system",
            "-l",
            "app.kubernetes.io/name=dynamo-operator",
            "--no-headers",
            capture=True,
            check=False,
        )
        return r.returncode == 0 and r.stdout.strip() and "Running" in r.stdout

    try:
        _poll(
            _operator_running,
            attempts=60,
            interval=5,
            fail_msg="Dynamo operator not ready within timeout",
        )
    except SystemExit:
        log_warn(
            "Dynamo operator may not be fully ready — "
            "check 'kubectl get pods -n dynamo-system'"
        )

    log_success(f"Dynamo operator {DYNAMO_VERSION} installed")
    return f"{DYNAMO_VERSION} installed"


# ---------------------------------------------------------------------------
# Kueue (batch job queuing)
# ---------------------------------------------------------------------------


def _kueue_default_queues_manifest() -> str:
    """Render ResourceFlavor + ClusterQueue + LocalQueue for the default namespace.

    Seeds a best-effort queue pair so benchmark jobs referencing
    `queue-name=aiperf-local-queue` Just Work without extra setup. The
    ClusterQueue is intentionally permissive (nominalQuota sized for dev
    clusters); users running real workloads should tune these numbers.
    """
    manifests: list[dict] = [
        _namespace_manifest(AIPERF_BENCHMARK_NAMESPACE),
        {
            "apiVersion": "kueue.x-k8s.io/v1beta1",
            "kind": "ResourceFlavor",
            "metadata": {"name": KUEUE_RESOURCE_FLAVOR_NAME},
        },
        {
            "apiVersion": "kueue.x-k8s.io/v1beta1",
            "kind": "ClusterQueue",
            "metadata": {"name": KUEUE_CLUSTER_QUEUE_NAME},
            "spec": {
                "namespaceSelector": {},
                "resourceGroups": [
                    {
                        "coveredResources": ["cpu", "memory", "nvidia.com/gpu"],
                        "flavors": [
                            {
                                "name": KUEUE_RESOURCE_FLAVOR_NAME,
                                "resources": [
                                    {"name": "cpu", "nominalQuota": "1000"},
                                    {"name": "memory", "nominalQuota": "4000Gi"},
                                    {"name": "nvidia.com/gpu", "nominalQuota": "64"},
                                ],
                            }
                        ],
                    }
                ],
            },
        },
        {
            "apiVersion": "kueue.x-k8s.io/v1beta1",
            "kind": "LocalQueue",
            "metadata": {
                "name": KUEUE_LOCAL_QUEUE_NAME,
                "namespace": AIPERF_BENCHMARK_NAMESPACE,
            },
            "spec": {"clusterQueue": KUEUE_CLUSTER_QUEUE_NAME},
        },
    ]
    return _manifests_to_yaml(manifests)


def cmd_install_kueue() -> str:
    """Install Kueue controller from the upstream release manifest."""
    log_step(f"Installing Kueue {KUEUE_VERSION}")
    _require_kubectl_and_cluster()

    if _skip_if_deployment_ready(
        "kueue-controller-manager",
        KUEUE_NAMESPACE,
        "Kueue already installed",
    ):
        # Still seed the default queues in case they were removed.
        log_info("Ensuring default ClusterQueue / LocalQueue are present...")
        kubectl("apply", "-f", "-", input=_kueue_default_queues_manifest(), text=True)
        return "already installed"

    log_info("Applying Kueue manifests...")
    kubectl(
        "apply",
        "--server-side",
        "-f",
        _KUEUE_MANIFEST_URL_TEMPLATE.format(version=KUEUE_VERSION),
    )
    log_info("Waiting for Kueue controller to be ready...")
    kubectl(
        "wait",
        "--for=condition=available",
        f"--timeout={180 * _TIMEOUT_SCALE}s",
        "deployment/kueue-controller-manager",
        "-n",
        KUEUE_NAMESPACE,
    )

    log_info("Seeding default ClusterQueue / LocalQueue for aiperf-benchmarks...")

    # Retry briefly — Kueue CRDs can take a moment to register even after the
    # controller deployment reports Available.
    def _apply_queues() -> bool:
        r = kubectl(
            "apply",
            "-f",
            "-",
            input=_kueue_default_queues_manifest(),
            text=True,
            capture=True,
            check=False,
        )
        return r.returncode == 0

    _poll(
        _apply_queues,
        attempts=30,
        interval=2,
        fail_msg="Failed to create default Kueue queues",
    )

    log_success(f"Kueue {KUEUE_VERSION} installed and ready")
    return f"{KUEUE_VERSION} installed"


def cmd_remove_kueue() -> None:
    """Remove Kueue and the default queue pair."""
    log_step("Removing Kueue")
    _require_kubectl_and_cluster()

    # Delete the queues first so Kueue can clean up any admission state.
    kubectl(
        "delete",
        "localqueue",
        KUEUE_LOCAL_QUEUE_NAME,
        "-n",
        AIPERF_BENCHMARK_NAMESPACE,
        "--ignore-not-found",
    )
    kubectl("delete", "clusterqueue", KUEUE_CLUSTER_QUEUE_NAME, "--ignore-not-found")
    kubectl(
        "delete",
        "resourceflavor",
        KUEUE_RESOURCE_FLAVOR_NAME,
        "--ignore-not-found",
    )

    kubectl(
        "delete",
        "-f",
        _KUEUE_MANIFEST_URL_TEMPLATE.format(version=KUEUE_VERSION),
        "--ignore-not-found",
    )
    log_success("Kueue removed")


# ---------------------------------------------------------------------------
# Observability: Prometheus + Grafana (kube-prometheus-stack) and Loki
# ---------------------------------------------------------------------------


def _helm_repo_add(name: str, url: str) -> None:
    """Idempotently add a Helm repo and refresh its index."""
    sh("helm", "repo", "add", name, url, "--force-update")
    sh("helm", "repo", "update", name)


def _helm_upgrade_install(
    release: str,
    chart: str,
    namespace: str,
    version: str,
    *extra_args: str,
) -> None:
    """`helm upgrade --install` with a pinned chart version and namespace."""
    sh(
        "helm",
        "upgrade",
        "--install",
        release,
        chart,
        "--kube-context",
        _kubectl_context(),
        "--namespace",
        namespace,
        "--create-namespace",
        "--version",
        version,
        *extra_args,
    )


def _print_grafana_port_forward_hint() -> None:
    """Tell the user how to access Grafana locally."""
    log_info(
        "Access Grafana:\n"
        f"    kubectl port-forward -n {PROMETHEUS_NAMESPACE} "
        f"svc/{PROMETHEUS_RELEASE_NAME}-grafana 3000:80\n"
        "    # then open http://localhost:3000 (user: admin, "
        f"pass: `kubectl get secret -n {PROMETHEUS_NAMESPACE} "
        f"{PROMETHEUS_RELEASE_NAME}-grafana "
        "-o jsonpath='{.data.admin-password}' | base64 -d`)"
    )


def cmd_install_prometheus() -> str:
    """Install kube-prometheus-stack (Prometheus + Grafana + Alertmanager)."""
    log_step(f"Installing kube-prometheus-stack {PROMETHEUS_STACK_VERSION}")
    require("helm")
    _require_kubectl_and_cluster()

    # Grafana is the deployment most users care about for port-forwarding; if
    # it's already up we short-circuit.
    grafana_deployment = f"{PROMETHEUS_RELEASE_NAME}-grafana"
    if _skip_if_deployment_ready(
        grafana_deployment,
        PROMETHEUS_NAMESPACE,
        "kube-prometheus-stack already installed",
    ):
        _print_grafana_port_forward_hint()
        return "already installed"

    log_info("Adding prometheus-community Helm repo...")
    _helm_repo_add(PROMETHEUS_HELM_REPO_NAME, PROMETHEUS_HELM_REPO_URL)

    log_info("Installing kube-prometheus-stack...")
    _helm_upgrade_install(
        PROMETHEUS_RELEASE_NAME,
        f"{PROMETHEUS_HELM_REPO_NAME}/kube-prometheus-stack",
        PROMETHEUS_NAMESPACE,
        PROMETHEUS_STACK_VERSION,
    )

    log_info("Waiting for Grafana to be ready...")
    kubectl(
        "wait",
        "--for=condition=available",
        f"--timeout={300 * _TIMEOUT_SCALE}s",
        f"deployment/{grafana_deployment}",
        "-n",
        PROMETHEUS_NAMESPACE,
    )
    log_success(f"kube-prometheus-stack {PROMETHEUS_STACK_VERSION} installed and ready")
    _print_grafana_port_forward_hint()
    return f"{PROMETHEUS_STACK_VERSION} installed"


def cmd_install_loki() -> str:
    """Install loki-stack (Loki + Promtail)."""
    log_step(f"Installing loki-stack {LOKI_STACK_VERSION}")
    require("helm")
    _require_kubectl_and_cluster()

    # loki-stack ships Loki as a StatefulSet, not a Deployment, so we use a
    # StatefulSet readiness check here instead of _skip_if_deployment_ready.
    r = kubectl(
        "get",
        "statefulset",
        LOKI_RELEASE_NAME,
        "-n",
        LOKI_NAMESPACE,
        "-o",
        "jsonpath={.status.readyReplicas}",
        capture=True,
        check=False,
    )
    if r.returncode == 0 and r.stdout.strip() and int(r.stdout) > 0:
        log_success("loki-stack already installed")
        return "already installed"

    log_info("Adding grafana Helm repo...")
    _helm_repo_add(LOKI_HELM_REPO_NAME, LOKI_HELM_REPO_URL)

    log_info("Installing loki-stack...")
    _helm_upgrade_install(
        LOKI_RELEASE_NAME,
        f"{LOKI_HELM_REPO_NAME}/loki-stack",
        LOKI_NAMESPACE,
        LOKI_STACK_VERSION,
    )

    log_info("Waiting for Loki StatefulSet to be ready...")
    kubectl(
        "rollout",
        "status",
        f"statefulset/{LOKI_RELEASE_NAME}",
        "-n",
        LOKI_NAMESPACE,
        f"--timeout={300 * _TIMEOUT_SCALE}s",
    )
    log_success(f"loki-stack {LOKI_STACK_VERSION} installed and ready")
    log_info(
        "Query Loki locally:\n"
        f"    kubectl port-forward -n {LOKI_NAMESPACE} "
        f"svc/{LOKI_RELEASE_NAME} 3100:3100"
    )
    return f"{LOKI_STACK_VERSION} installed"


# ---------------------------------------------------------------------------
# Image push (retag + docker push the aiperf-slim image)
# ---------------------------------------------------------------------------


def cmd_push(*, registry: str, tag: str | None = None) -> str:
    """Retag and push the locally built aiperf image to a remote registry.

    Authentication is the user's responsibility — run `docker login <registry>`
    or configure a credential helper before invoking this command.
    """
    require("docker")

    if not registry:
        log_error("--registry is required")
        raise SystemExit(1)

    local_id = _local_image_id(AIPERF_IMAGE)
    if not local_id:
        log_error(f"Image {AIPERF_IMAGE} not found locally. Run 'build' first.")
        raise SystemExit(1)

    # Allow callers to point at bare registry hosts ("nvcr.io/my-org") or full
    # image refs ("nvcr.io/my-org/aiperf"). Normalize to the former.
    registry = registry.rstrip("/")
    local_repo, local_tag = (
        AIPERF_IMAGE.rsplit(":", 1) if ":" in AIPERF_IMAGE else (AIPERF_IMAGE, "latest")
    )
    remote_tag = tag or local_tag
    # If the registry already looks like a full image path (contains a '/'
    # after the host), treat it as <host>/<repo>; otherwise append the local
    # short name.
    if "/" in registry:
        remote_image = f"{registry}:{remote_tag}"
    else:
        remote_image = f"{registry}/{local_repo}:{remote_tag}"

    log_step(f"Pushing {AIPERF_IMAGE} to {remote_image}")
    sh("docker", "tag", AIPERF_IMAGE, remote_image)
    sh("docker", "push", remote_image)
    log_success(f"Pushed {remote_image}")
    return remote_image


# ---------------------------------------------------------------------------
# Mock server
# ---------------------------------------------------------------------------


def cmd_deploy_mock() -> str:
    log_step("Deploying mock server")
    _require_kubectl_and_cluster()

    if _skip_if_deployment_ready(
        "aiperf-mock-server", None, "Mock server already deployed and ready"
    ):
        return "already deployed"

    if not MOCK_SERVER_MANIFEST.exists():
        log_error(f"Mock server manifest not found: {MOCK_SERVER_MANIFEST}")
        raise SystemExit(1)

    if _mode.json:
        _mode._streaming = True
        _mode.emit_event(
            {"type": "step_started", "step": "deploy-mock", "timestamp": time.time()}
        )

    log_info("Applying mock server manifest...")
    manifest = MOCK_SERVER_MANIFEST.read_text().replace(
        "image: aiperf-mock-server:latest",
        f"image: {MOCK_SERVER_IMAGE}",
    )
    kubectl("apply", "-f", "-", input=manifest, text=True)
    log_info("Waiting for mock server to be ready...")
    with _StreamEmitter("deploy-mock"):
        kubectl(
            "rollout",
            "status",
            "deployment/aiperf-mock-server",
            f"--timeout={120 * _TIMEOUT_SCALE}s",
        )
    log_success("Mock server deployed")
    kubectl("get", "pods", "-l", "app=aiperf-mock-server")
    endpoint = "http://aiperf-mock-server.default.svc.cluster.local:8000"
    log_info(f"Endpoint: {endpoint}")
    _mode.set_result({"endpoint": endpoint, "namespace": "default"})
    if _mode._streaming:
        _mode.emit_event(
            {"type": "step_completed", "step": "deploy-mock", "detail": "deployed"}
        )
    return "deployed"


def cmd_remove_mock() -> None:
    log_step("Removing mock server")
    _require_kubectl_and_cluster()
    kubectl("delete", "-f", str(MOCK_SERVER_MANIFEST), "--ignore-not-found")
    log_success("Mock server removed")


# ---------------------------------------------------------------------------
# vLLM deployment
# ---------------------------------------------------------------------------


def _generate_vllm_manifest(
    model: str,
    gpus: int,
    vllm_image: str,
    max_model_len: int,
    *,
    hf_token: bool,
) -> str:
    """Generate Namespace + Service + Deployment YAML for vLLM."""
    container: dict[str, Any] = {
        "name": "vllm",
        "image": vllm_image,
        "args": [
            "--model",
            model,
            "--port",
            "8000",
            "--max-model-len",
            str(max_model_len),
            "--dtype",
            "auto",
            "--tensor-parallel-size",
            str(gpus),
            "--gpu-memory-utilization",
            VLLM_GPU_MEM_UTIL,
            "--enforce-eager",
        ],
        "ports": [{"containerPort": 8000, "name": "http"}],
        "readinessProbe": {
            "httpGet": {"path": "/health", "port": 8000},
            "initialDelaySeconds": 30,
            "periodSeconds": 10,
            "timeoutSeconds": 5,
            "failureThreshold": 30,
        },
        "livenessProbe": {
            "httpGet": {"path": "/health", "port": 8000},
            "initialDelaySeconds": 60,
            "periodSeconds": 15,
            "timeoutSeconds": 5,
            "failureThreshold": 5,
        },
    }

    if gpus > 0:
        container["resources"] = {
            "limits": {"nvidia.com/gpu": str(gpus)},
            "requests": {"nvidia.com/gpu": str(gpus)},
        }

    if hf_token:
        container["env"] = [
            {
                "name": "HF_TOKEN",
                "valueFrom": {"secretKeyRef": {"name": "hf-token", "key": "token"}},
            }
        ]

    documents: list[dict] = [
        _namespace_manifest(VLLM_NAMESPACE),
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": "vllm-server", "namespace": VLLM_NAMESPACE},
            "spec": {
                "selector": {"app": "vllm-server"},
                "ports": [
                    {
                        "port": 8000,
                        "targetPort": 8000,
                        "protocol": "TCP",
                        "name": "http",
                    }
                ],
                "type": "ClusterIP",
            },
        },
        {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "vllm-server", "namespace": VLLM_NAMESPACE},
            "spec": {
                "replicas": 1,
                "selector": {"matchLabels": {"app": "vllm-server"}},
                "template": {
                    "metadata": {"labels": {"app": "vllm-server"}},
                    "spec": {
                        "runtimeClassName": "nvidia",
                        "containers": [container],
                    },
                },
            },
        },
    ]

    return _manifests_to_yaml(documents)


@app.command(name="deploy-vllm", group=server, help="Deploy standalone vLLM server.")
def cmd_deploy_vllm(
    *,
    output: OutputOptions = _DEFAULT_OUTPUT_OPTS,
    model_opts: ModelOptions = _DEFAULT_MODEL_OPTS,
    vllm_image: Annotated[str, Parameter(help="vLLM container image.")] = VLLM_IMAGE,
) -> None:
    """Deploy vLLM inference server."""
    _apply_output(output)
    log_step(
        f"Deploying vLLM server (model={model_opts.model}, gpus={model_opts.gpus})"
    )
    _require_kubectl_and_cluster()
    _require_gpu()

    if _skip_if_deployment_ready(
        "vllm-server", VLLM_NAMESPACE, "vLLM server already deployed and ready"
    ):
        return

    hf_secret = _ensure_hf_token_secret(VLLM_NAMESPACE)
    manifest = _generate_vllm_manifest(
        model_opts.model,
        model_opts.gpus,
        vllm_image,
        model_opts.max_model_len,
        hf_token=bool(hf_secret),
    )
    kubectl("apply", "-f", "-", input=manifest, text=True)

    if _mode.json:
        _mode._streaming = True
        _mode.emit_event(
            {"type": "step_started", "step": "deploy-vllm", "timestamp": time.time()}
        )

    log_info(
        "Waiting for vLLM to be ready (this may take several minutes for model download)..."
    )
    with _StreamEmitter("deploy-vllm"):
        kubectl(
            "rollout",
            "status",
            "deployment/vllm-server",
            "-n",
            VLLM_NAMESPACE,
            "--timeout=600s",
        )
    log_success("vLLM server deployed")
    kubectl("get", "pods", "-n", VLLM_NAMESPACE, "-l", "app=vllm-server")
    endpoint = f"http://vllm-server.{VLLM_NAMESPACE}.svc.cluster.local:8000/v1"
    log_info(f"Endpoint: {endpoint}")
    _mode.set_result(
        {
            "endpoint": endpoint,
            "namespace": VLLM_NAMESPACE,
            "model": model_opts.model,
            "gpus": model_opts.gpus,
        }
    )
    if _mode._streaming:
        _mode.emit_event(
            {"type": "step_completed", "step": "deploy-vllm", "detail": endpoint}
        )


def cmd_remove_vllm() -> None:
    _remove_server("vLLM server", VLLM_NAMESPACE)


@app.command(name="vllm-logs", group=server, help="View vLLM logs.")
def cmd_vllm_logs(
    *, output: OutputOptions = _DEFAULT_OUTPUT_OPTS, follow: FollowFlag = FOLLOW_DEFAULT
) -> None:
    """View vLLM server logs."""
    _apply_output(output)
    _require_kubectl_and_cluster()
    kubectl(
        "logs",
        "-n",
        VLLM_NAMESPACE,
        "-l",
        "app=vllm-server",
        *(["-f"] if follow else []),
    )


# ---------------------------------------------------------------------------
# SGLang deployment
# ---------------------------------------------------------------------------


def _generate_sglang_manifest(
    model: str,
    gpus: int,
    sglang_image: str,
    *,
    hf_token: bool,
) -> str:
    """Generate Namespace + Service + Deployment YAML for SGLang."""
    container: dict[str, Any] = {
        "name": "sglang",
        "image": sglang_image,
        "command": [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model,
            "--port",
            "8000",
            "--host",
            "0.0.0.0",
        ],
        "ports": [{"containerPort": 8000, "name": "http"}],
        "readinessProbe": {
            "httpGet": {"path": "/health", "port": 8000},
            "initialDelaySeconds": 30,
            "periodSeconds": 10,
            "timeoutSeconds": 5,
            "failureThreshold": 30,
        },
        "livenessProbe": {
            "httpGet": {"path": "/health", "port": 8000},
            "initialDelaySeconds": 60,
            "periodSeconds": 15,
            "timeoutSeconds": 5,
            "failureThreshold": 5,
        },
    }

    if gpus > 0:
        container["resources"] = {
            "limits": {"nvidia.com/gpu": str(gpus)},
            "requests": {"nvidia.com/gpu": str(gpus)},
        }

    if hf_token:
        container["env"] = [
            {
                "name": "HF_TOKEN",
                "valueFrom": {"secretKeyRef": {"name": "hf-token", "key": "token"}},
            }
        ]

    documents: list[dict] = [
        _namespace_manifest(SGLANG_NAMESPACE),
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": "sglang-server", "namespace": SGLANG_NAMESPACE},
            "spec": {
                "selector": {"app": "sglang-server"},
                "ports": [
                    {
                        "port": 8000,
                        "targetPort": 8000,
                        "protocol": "TCP",
                        "name": "http",
                    }
                ],
                "type": "ClusterIP",
            },
        },
        {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "sglang-server", "namespace": SGLANG_NAMESPACE},
            "spec": {
                "replicas": 1,
                "selector": {"matchLabels": {"app": "sglang-server"}},
                "template": {
                    "metadata": {"labels": {"app": "sglang-server"}},
                    "spec": {
                        "runtimeClassName": "nvidia",
                        "containers": [container],
                    },
                },
            },
        },
    ]

    return _manifests_to_yaml(documents)


@app.command(
    name="deploy-sglang", group=server, help="Deploy standalone SGLang server."
)
def cmd_deploy_sglang(
    *,
    output: OutputOptions = _DEFAULT_OUTPUT_OPTS,
    model_opts: ModelOptions = _DEFAULT_MODEL_OPTS,
    sglang_image: Annotated[
        str, Parameter(help="SGLang container image.")
    ] = SGLANG_IMAGE,
) -> None:
    """Deploy SGLang inference server."""
    _apply_output(output)
    log_step(
        f"Deploying SGLang server (model={model_opts.model}, gpus={model_opts.gpus})"
    )
    _require_kubectl_and_cluster()
    _require_gpu()

    if _skip_if_deployment_ready(
        "sglang-server",
        SGLANG_NAMESPACE,
        "SGLang server already deployed and ready",
    ):
        return

    hf_secret = _ensure_hf_token_secret(SGLANG_NAMESPACE)
    manifest = _generate_sglang_manifest(
        model_opts.model,
        model_opts.gpus,
        sglang_image,
        hf_token=bool(hf_secret),
    )
    kubectl("apply", "-f", "-", input=manifest, text=True)

    if _mode.json:
        _mode._streaming = True
        _mode.emit_event(
            {
                "type": "step_started",
                "step": "deploy-sglang",
                "timestamp": time.time(),
            }
        )

    log_info(
        "Waiting for SGLang to be ready (this may take several minutes for model download)..."
    )
    with _StreamEmitter("deploy-sglang"):
        kubectl(
            "rollout",
            "status",
            "deployment/sglang-server",
            "-n",
            SGLANG_NAMESPACE,
            "--timeout=600s",
        )
    log_success("SGLang server deployed")
    kubectl("get", "pods", "-n", SGLANG_NAMESPACE, "-l", "app=sglang-server")
    endpoint = f"http://sglang-server.{SGLANG_NAMESPACE}.svc.cluster.local:8000/v1"
    log_info(f"Endpoint: {endpoint}")
    _mode.set_result(
        {
            "endpoint": endpoint,
            "namespace": SGLANG_NAMESPACE,
            "model": model_opts.model,
            "gpus": model_opts.gpus,
        }
    )
    if _mode._streaming:
        _mode.emit_event(
            {"type": "step_completed", "step": "deploy-sglang", "detail": endpoint}
        )


def cmd_remove_sglang() -> None:
    _remove_server("SGLang server", SGLANG_NAMESPACE)


@app.command(name="sglang-logs", group=server, help="View SGLang logs.")
def cmd_sglang_logs(
    *, output: OutputOptions = _DEFAULT_OUTPUT_OPTS, follow: FollowFlag = FOLLOW_DEFAULT
) -> None:
    """View SGLang server logs."""
    _apply_output(output)
    _require_kubectl_and_cluster()
    kubectl(
        "logs",
        "-n",
        SGLANG_NAMESPACE,
        "-l",
        "app=sglang-server",
        *(["-f"] if follow else []),
    )


# ---------------------------------------------------------------------------
# TensorRT-LLM deployment (trtllm-serve with on-the-fly engine building)
# ---------------------------------------------------------------------------

_TRTLLM_LD_LIBRARY_PATH = (
    "/usr/local/tensorrt/lib:"
    "/usr/local/cuda/lib64:"
    "/usr/local/cuda/compat/lib.real:"
    "/usr/local/lib/python3.12/dist-packages/torch/lib:"
    "/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
)


def _generate_trtllm_manifest(
    model: str,
    gpus: int,
    trtllm_image: str,
    max_model_len: int,
    *,
    hf_token: bool,
) -> str:
    """Generate Namespace + Service + Deployment YAML for TensorRT-LLM."""
    env: list[dict] = [{"name": "LD_LIBRARY_PATH", "value": _TRTLLM_LD_LIBRARY_PATH}]
    if hf_token:
        env.append(
            {
                "name": "HF_TOKEN",
                "valueFrom": {"secretKeyRef": {"name": "hf-token", "key": "token"}},
            }
        )

    container: dict[str, Any] = {
        "name": "trtllm",
        "image": trtllm_image,
        "command": ["trtllm-serve"],
        "args": [
            "serve",
            model,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--max_seq_len",
            str(max_model_len),
            "--tensor_parallel_size",
            str(max(gpus, 1)),
        ],
        "ports": [{"containerPort": 8000, "name": "http"}],
        "env": env,
        "readinessProbe": {
            "httpGet": {"path": "/health", "port": 8000},
            "initialDelaySeconds": 30,
            "periodSeconds": 10,
            "timeoutSeconds": 10,
            "failureThreshold": 120,
        },
        "livenessProbe": {
            "httpGet": {"path": "/health", "port": 8000},
            "initialDelaySeconds": 60,
            "periodSeconds": 30,
            "timeoutSeconds": 10,
            "failureThreshold": 40,
        },
        "volumeMounts": [{"name": "shm", "mountPath": "/dev/shm"}],
    }

    if gpus > 0:
        container["resources"] = {
            "limits": {"nvidia.com/gpu": str(gpus)},
            "requests": {"nvidia.com/gpu": str(gpus)},
        }

    pod_spec: dict = {
        "runtimeClassName": "nvidia",
        "containers": [container],
        "volumes": [
            {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "4Gi"}},
        ],
    }

    documents: list[dict] = [
        _namespace_manifest(TRTLLM_NAMESPACE),
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": "trtllm-server", "namespace": TRTLLM_NAMESPACE},
            "spec": {
                "selector": {"app": "trtllm-server"},
                "ports": [
                    {
                        "port": 8000,
                        "targetPort": 8000,
                        "protocol": "TCP",
                        "name": "http",
                    }
                ],
                "type": "ClusterIP",
            },
        },
        {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "trtllm-server", "namespace": TRTLLM_NAMESPACE},
            "spec": {
                "replicas": 1,
                "selector": {"matchLabels": {"app": "trtllm-server"}},
                "template": {
                    "metadata": {"labels": {"app": "trtllm-server"}},
                    "spec": pod_spec,
                },
            },
        },
    ]

    return _manifests_to_yaml(documents)


@app.command(
    name="deploy-trtllm",
    group=server,
    help="Deploy standalone TensorRT-LLM server.",
)
def cmd_deploy_trtllm(
    *,
    output: OutputOptions = _DEFAULT_OUTPUT_OPTS,
    model_opts: ModelOptions = _DEFAULT_MODEL_OPTS,
    trtllm_image: Annotated[
        str, Parameter(help="TensorRT-LLM container image.")
    ] = TRTLLM_IMAGE,
) -> None:
    """Deploy TensorRT-LLM inference server."""
    _apply_output(output)
    log_step(
        f"Deploying TensorRT-LLM server (model={model_opts.model}, gpus={model_opts.gpus})"
    )
    _require_kubectl_and_cluster()
    _require_gpu()

    if _skip_if_deployment_ready(
        "trtllm-server",
        TRTLLM_NAMESPACE,
        "TensorRT-LLM server already deployed and ready",
    ):
        return

    hf_secret = _ensure_hf_token_secret(TRTLLM_NAMESPACE)
    manifest = _generate_trtllm_manifest(
        model_opts.model,
        model_opts.gpus,
        trtllm_image,
        model_opts.max_model_len,
        hf_token=bool(hf_secret),
    )
    kubectl("apply", "-f", "-", input=manifest, text=True)

    if _mode.json:
        _mode._streaming = True
        _mode.emit_event(
            {
                "type": "step_started",
                "step": "deploy-trtllm",
                "timestamp": time.time(),
            }
        )

    log_info(
        "Waiting for TensorRT-LLM to be ready (engine build can take 10+ minutes)..."
    )
    with _StreamEmitter("deploy-trtllm"):
        kubectl(
            "rollout",
            "status",
            "deployment/trtllm-server",
            "-n",
            TRTLLM_NAMESPACE,
            "--timeout=900s",
        )
    log_success("TensorRT-LLM server deployed")
    kubectl("get", "pods", "-n", TRTLLM_NAMESPACE, "-l", "app=trtllm-server")
    endpoint = f"http://trtllm-server.{TRTLLM_NAMESPACE}.svc.cluster.local:8000/v1"
    log_info(f"Endpoint: {endpoint}")
    _mode.set_result(
        {
            "endpoint": endpoint,
            "namespace": TRTLLM_NAMESPACE,
            "model": model_opts.model,
            "gpus": model_opts.gpus,
        }
    )
    if _mode._streaming:
        _mode.emit_event(
            {"type": "step_completed", "step": "deploy-trtllm", "detail": endpoint}
        )


def cmd_remove_trtllm() -> None:
    _remove_server("TensorRT-LLM server", TRTLLM_NAMESPACE)


@app.command(name="trtllm-logs", group=server, help="View TensorRT-LLM logs.")
def cmd_trtllm_logs(
    *, output: OutputOptions = _DEFAULT_OUTPUT_OPTS, follow: FollowFlag = FOLLOW_DEFAULT
) -> None:
    """View TensorRT-LLM server logs."""
    _apply_output(output)
    _require_kubectl_and_cluster()
    kubectl(
        "logs",
        "-n",
        TRTLLM_NAMESPACE,
        "-l",
        "app=trtllm-server",
        *(["-f"] if follow else []),
    )


# ---------------------------------------------------------------------------
# Dynamo deployment (DynamoGraphDeployment CRD via Dynamo operator)
# ---------------------------------------------------------------------------

_DYNAMO_HEALTH_PORT = 9090
_DYNAMO_METRICS_PORT = 8081  # DYN_SYSTEM_PORT: worker /metrics endpoint
_DYNAMO_1GPU_KVBM_CPU_CACHE_GB = (
    1  # conservative default for single-GPU (pinned memory)
)
_DYNAMO_WORKER_WORKING_DIR = "/workspace/examples/backends/vllm"
_DYNAMO_WORKER_COMMAND = ["python3", "-m", "dynamo.vllm"]


def _check_dynamo_operator() -> None:
    """Verify the Dynamo operator CRD is installed. Exits with hint on failure."""
    r = kubectl(
        "get", "crd", "dynamographdeployments.nvidia.com", capture=True, check=False
    )
    if r.returncode != 0:
        log_error(
            "Dynamo operator not found (CRD dynamographdeployments.nvidia.com missing)"
        )
        log_info("Run './dev/kube.py install-dynamo' to install the Dynamo operator.")
        raise SystemExit(1)


def _generate_dynamo_manifest(
    model: str,
    gpus: int,
    dynamo_image: str,
    max_model_len: int,
    *,
    mode: str,
    hf_token_secret: str | None,
    router_mode: str | None = None,
    kvbm_cpu_cache_gb: int | None = None,
    connectors: list[str] | None = None,
) -> str:
    """Generate Namespace + DynamoGraphDeployment CRD YAML.

    Modes: agg, disagg, disagg-1gpu.
    disagg-1gpu shares 1 physical GPU between prefill+decode workers
    by requesting 0 GPUs in K8s resources (runtimeClassName: nvidia gives access)
    and using low gpu_memory_utilization so both fit.

    Optional KV features:
      router_mode: sets DYN_ROUTER_MODE on frontend (e.g. "kv", "round-robin")
      kvbm_cpu_cache_gb: sets DYN_KVBM_CPU_CACHE_GB on prefill workers
      connectors: adds --connector flags to prefill workers (e.g. ["kvbm", "nixl"])
    """
    single_gpu = mode == "disagg-1gpu"
    deploy_name = f"dynamo-{mode}"

    # Single-GPU disagg: override to low mem util, force gpus=0 for K8s scheduling
    gpu_mem_util = DYNAMO_1GPU_MEM_UTIL if single_gpu else VLLM_GPU_MEM_UTIL
    gpu_request = 0 if single_gpu else gpus

    # Auto-default KVBM CPU cache for single-GPU when kvbm connector is used
    has_kvbm = connectors and "kvbm" in connectors
    if has_kvbm and kvbm_cpu_cache_gb is None and single_gpu:
        kvbm_cpu_cache_gb = _DYNAMO_1GPU_KVBM_CPU_CACHE_GB
        log_info(
            f"KVBM: defaulting CPU cache to {kvbm_cpu_cache_gb}GB "
            f"(pinned memory, conservative for single-GPU)"
        )

    worker_args = ["--model", model]
    if max_model_len is not None:
        worker_args.extend(["--max-model-len", str(max_model_len)])
    worker_args.extend(["--gpu-memory-utilization", gpu_mem_util, "--enforce-eager"])

    probes = {
        "startupProbe": {
            "httpGet": {"path": "/live", "port": _DYNAMO_HEALTH_PORT},
            "initialDelaySeconds": 60,
            "periodSeconds": 10,
            "failureThreshold": 60,
            "timeoutSeconds": 5,
        },
        "livenessProbe": {
            "httpGet": {"path": "/live", "port": _DYNAMO_HEALTH_PORT},
            "initialDelaySeconds": 0,
            "periodSeconds": 10,
            "failureThreshold": 15,
            "timeoutSeconds": 5,
        },
    }

    worker_container: dict = {
        "image": dynamo_image,
        "workingDir": _DYNAMO_WORKER_WORKING_DIR,
        "command": _DYNAMO_WORKER_COMMAND,
        "args": worker_args,
        **probes,
    }

    pod_uid_env = {
        "name": "POD_UID",
        "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}},
    }
    metrics_port_env = {
        "name": "DYN_SYSTEM_PORT",
        "value": str(_DYNAMO_METRICS_PORT),
    }

    frontend_envs: list[dict] = [pod_uid_env]
    if router_mode:
        frontend_envs.append({"name": "DYN_ROUTER_MODE", "value": router_mode})

    frontend: dict = {
        "dynamoNamespace": deploy_name,
        "componentType": "frontend",
        "replicas": 1,
        "extraPodSpec": {
            "mainContainer": {"image": dynamo_image, "env": frontend_envs}
        },
    }

    decode_envs: list[dict] = [pod_uid_env, metrics_port_env]

    decode_worker: dict = {
        "dynamoNamespace": deploy_name,
        "componentType": "worker",
        "replicas": 1,
        "resources": {"limits": {"gpu": str(gpu_request)}},
        "extraPodSpec": {
            "runtimeClassName": "nvidia",
            "mainContainer": {**worker_container, "env": decode_envs},
        },
    }

    if hf_token_secret:
        decode_worker["envFromSecret"] = hf_token_secret

    services: dict = {"Frontend": frontend, "VllmDecodeWorker": decode_worker}

    if mode in ("disagg", "disagg-1gpu"):
        decode_worker["subComponentType"] = "decode"
        decode_worker["extraPodSpec"]["mainContainer"]["args"] = [
            *worker_args,
            "--is-decode-worker",
        ]

        prefill_args = [*worker_args, "--is-prefill-worker"]
        for connector in connectors or []:
            prefill_args.extend(["--connector", connector])

        prefill_envs: list[dict] = [pod_uid_env, metrics_port_env]
        if kvbm_cpu_cache_gb is not None:
            prefill_envs.append(
                {"name": "DYN_KVBM_CPU_CACHE_GB", "value": str(kvbm_cpu_cache_gb)}
            )
            prefill_envs.append({"name": "DYN_KVBM_METRICS", "value": "true"})

        prefill_container = {
            **worker_container,
            "args": prefill_args,
            "env": prefill_envs,
        }
        prefill_worker: dict = {
            "dynamoNamespace": deploy_name,
            "componentType": "worker",
            "subComponentType": "prefill",
            "replicas": 1,
            "resources": {"limits": {"gpu": str(gpu_request)}},
            "extraPodSpec": {
                "runtimeClassName": "nvidia",
                "mainContainer": prefill_container,
            },
        }
        if hf_token_secret:
            prefill_worker["envFromSecret"] = hf_token_secret
        services["VllmPrefillWorker"] = prefill_worker

    sa_name = f"{deploy_name}-k8s-service-discovery"

    documents = [
        _namespace_manifest(DYNAMO_NAMESPACE),
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "Role",
            "metadata": {"name": "dynamo-discovery", "namespace": DYNAMO_NAMESPACE},
            "rules": [
                {
                    "apiGroups": ["nvidia.com"],
                    "resources": ["dynamoworkermetadatas"],
                    "verbs": [
                        "get",
                        "list",
                        "watch",
                        "create",
                        "update",
                        "patch",
                        "delete",
                    ],
                }
            ],
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "RoleBinding",
            "metadata": {"name": "dynamo-discovery", "namespace": DYNAMO_NAMESPACE},
            "subjects": [
                {
                    "kind": "ServiceAccount",
                    "name": sa_name,
                    "namespace": DYNAMO_NAMESPACE,
                }
            ],
            "roleRef": {
                "kind": "Role",
                "name": "dynamo-discovery",
                "apiGroup": "rbac.authorization.k8s.io",
            },
        },
        {
            "apiVersion": "nvidia.com/v1alpha1",
            "kind": "DynamoGraphDeployment",
            "metadata": {"name": deploy_name, "namespace": DYNAMO_NAMESPACE},
            "spec": {"services": services},
        },
    ]

    return _manifests_to_yaml(documents)


def _wait_for_dynamo_ready(mode: str, timeout: int = 600) -> None:
    """Wait for all Dynamo pods to be Ready."""
    deploy_name = f"dynamo-{mode}"
    log_info(f"Waiting for Dynamo pods to be ready (timeout={timeout}s)...")
    deadline = time.time() + timeout

    while time.time() < deadline:
        r = kubectl(
            "get",
            "pods",
            "-n",
            DYNAMO_NAMESPACE,
            "-o",
            "json",
            capture=True,
            check=False,
        )
        if r.returncode == 0 and r.stdout.strip():
            data = orjson.loads(r.stdout)
            pods = [
                p for p in data.get("items", []) if deploy_name in p["metadata"]["name"]
            ]
            if pods:
                ready_count = sum(
                    1
                    for p in pods
                    if p.get("status", {}).get("phase") == "Running"
                    and (css := p.get("status", {}).get("containerStatuses"))
                    and all(cs.get("ready", False) for cs in css)
                )
                if ready_count == len(pods):
                    log_success(f"All {len(pods)} Dynamo pod(s) ready")
                    return
                log_info(f"  {ready_count}/{len(pods)} pods ready...")
        time.sleep(10)

    log_error(f"Dynamo pods not ready within {timeout}s")
    kubectl("get", "pods", "-n", DYNAMO_NAMESPACE, "-o", "wide", check=False)
    raise SystemExit(1)


@app.command(
    name="deploy-dynamo",
    group=server,
    help="Deploy Dynamo inference server (agg/disagg/disagg-1gpu).",
)
def cmd_deploy_dynamo(
    *,
    output: OutputOptions = _DEFAULT_OUTPUT_OPTS,
    model_opts: ModelOptions = _DEFAULT_MODEL_OPTS,
    dynamo_opts: DynamoDeployOptions = _DEFAULT_DYNAMO_OPTS,
) -> None:
    """Deploy Dynamo inference server."""
    _apply_output(output)
    extras: list[str] = []
    if dynamo_opts.router_mode:
        extras.append(f"router={dynamo_opts.router_mode}")
    if dynamo_opts.kvbm_cpu_cache_gb is not None:
        extras.append(f"kvbm={dynamo_opts.kvbm_cpu_cache_gb}GB")
    if dynamo_opts.connectors:
        extras.append(f"connectors={','.join(dynamo_opts.connectors)}")
    extras_str = f", {', '.join(extras)}" if extras else ""

    if dynamo_opts.mode == "disagg-1gpu":
        log_step(
            f"Deploying Dynamo server ({dynamo_opts.mode}, model={model_opts.model}, "
            f"shared 1 GPU, mem_util={DYNAMO_1GPU_MEM_UTIL}{extras_str})"
        )
    else:
        log_step(
            f"Deploying Dynamo server ({dynamo_opts.mode}, model={model_opts.model}, "
            f"gpus={model_opts.gpus}{extras_str})"
        )
    _require_kubectl_and_cluster()
    _require_gpu()
    _check_dynamo_operator()

    hf_secret = _ensure_hf_token_secret(DYNAMO_NAMESPACE)
    manifest = _generate_dynamo_manifest(
        model_opts.model,
        model_opts.gpus,
        dynamo_opts.dynamo_image,
        model_opts.max_model_len,
        mode=dynamo_opts.mode,
        hf_token_secret=hf_secret,
        router_mode=dynamo_opts.router_mode,
        kvbm_cpu_cache_gb=dynamo_opts.kvbm_cpu_cache_gb,
        connectors=dynamo_opts.connectors,
    )
    kubectl("apply", "-f", "-", input=manifest, text=True)

    if _mode.json:
        _mode._streaming = True
        _mode.emit_event(
            {"type": "step_started", "step": "deploy-dynamo", "timestamp": time.time()}
        )

    with _StreamEmitter("deploy-dynamo"):
        _wait_for_dynamo_ready(dynamo_opts.mode, timeout=600)

    deploy_name = f"dynamo-{dynamo_opts.mode}"
    endpoint = (
        f"http://{deploy_name}-frontend.{DYNAMO_NAMESPACE}.svc.cluster.local:8000/v1"
    )
    log_success("Dynamo server deployed")
    kubectl("get", "pods", "-n", DYNAMO_NAMESPACE, "-o", "wide")
    log_info(f"Endpoint: {endpoint}")
    _mode.set_result(
        {
            "endpoint": endpoint,
            "namespace": DYNAMO_NAMESPACE,
            "mode": dynamo_opts.mode,
            "model": model_opts.model,
            "gpus": model_opts.gpus,
        }
    )
    if _mode._streaming:
        _mode.emit_event(
            {"type": "step_completed", "step": "deploy-dynamo", "detail": endpoint}
        )


def cmd_remove_dynamo() -> None:
    _remove_server("Dynamo server", DYNAMO_NAMESPACE)


@app.command(name="dynamo-logs", group=server, help="View Dynamo pod logs.")
def cmd_dynamo_logs(
    *, output: OutputOptions = _DEFAULT_OUTPUT_OPTS, follow: FollowFlag = FOLLOW_DEFAULT
) -> None:
    """View Dynamo server logs."""
    _apply_output(output)
    _require_kubectl_and_cluster()

    r = kubectl(
        "get", "pods", "-n", DYNAMO_NAMESPACE, "-o", "name", capture=True, check=False
    )
    pods = [p.strip() for p in (r.stdout or "").splitlines() if p.strip()]
    if not pods:
        log_warn("No pods found in dynamo-server namespace")
        return

    if follow:
        worker = next((p for p in pods if "worker" in p.lower()), pods[-1])
        kubectl("logs", "-n", DYNAMO_NAMESPACE, worker, "--all-containers", "-f")
    else:
        for pod in pods:
            print(f"\n{_CYAN}--- {pod} ---{_NC}")
            kubectl(
                "logs",
                "-n",
                DYNAMO_NAMESPACE,
                pod,
                "--all-containers",
                "--tail=50",
                check=False,
            )


# ---------------------------------------------------------------------------
# LoRA adapter deployment (DynamoModel CRD)
# ---------------------------------------------------------------------------


def _generate_lora_manifest(name: str, base_model: str, source: str) -> str:
    """Generate DynamoModel CRD YAML for a LoRA adapter."""
    doc = {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoModel",
        "metadata": {"name": name, "namespace": DYNAMO_NAMESPACE},
        "spec": {
            "modelName": name,
            "baseModelName": base_model,
            "modelType": "lora",
            "source": {"uri": source},
        },
    }
    return _yaml_dump(doc)


@app.command(
    name="deploy-lora",
    group=server,
    help="Deploy LoRA adapter on running Dynamo base model.",
)
def cmd_deploy_lora(
    *,
    output: OutputOptions = _DEFAULT_OUTPUT_OPTS,
    name: Annotated[str, Parameter(help="LoRA adapter name.")],
    base_model: Annotated[str, Parameter(help="Base model name.")],
    source: Annotated[str, Parameter(help="LoRA source URI (e.g. hf://org/repo).")],
) -> None:
    """Deploy a LoRA adapter on a running Dynamo base model."""
    _apply_output(output)
    log_step(f"Deploying LoRA adapter: {name}")
    _require_kubectl_and_cluster()
    _check_dynamo_operator()

    manifest = _generate_lora_manifest(name, base_model, source)
    kubectl("apply", "-f", "-", input=manifest, text=True)
    log_success(
        f"LoRA adapter '{name}' deployed (base: {base_model}, source: {source})"
    )


@app.command(name="remove-lora", group=server, help="Remove LoRA adapter.")
def cmd_remove_lora(
    *,
    output: OutputOptions = _DEFAULT_OUTPUT_OPTS,
    name: Annotated[str, Parameter(help="LoRA adapter name.")],
) -> None:
    """Remove a LoRA adapter."""
    _apply_output(output)
    log_step(f"Removing LoRA adapter: {name}")
    _require_kubectl_and_cluster()
    kubectl("delete", "dynamomodel", name, "-n", DYNAMO_NAMESPACE, "--ignore-not-found")
    log_success(f"LoRA adapter '{name}' removed")


# ---------------------------------------------------------------------------
# Benchmark run
# ---------------------------------------------------------------------------


def cmd_run(*, opts: RunOptions, detach: bool, dry_run: bool) -> None:
    import math

    from aiperf.cli_commands.kube.profile import generate_benchmark_name
    from aiperf.config import AIPerfConfig
    from aiperf.config.kube import KubeOptions
    from aiperf.config.loader import load_config
    from aiperf.kubernetes.constants import DEFAULT_BENCHMARK_NAMESPACE
    from aiperf.kubernetes.environment import K8sEnvironment
    from aiperf.kubernetes.resources import KubernetesDeployment
    from aiperf.operator.spec_converter import (
        apply_k8s_runtime_config,
        apply_worker_config,
    )

    if not dry_run:
        _require_kubectl_and_cluster()

    config_path = Path(opts.config) if opts.config else DEFAULT_BENCHMARK_CONFIG
    if not config_path.exists():
        log_error(f"Config file not found: {config_path}")
        raise SystemExit(1)

    config = load_config(config_path)
    kube_options = KubeOptions(
        image=AIPERF_IMAGE, image_pull_policy="Never", workers=opts.workers
    )

    name = generate_benchmark_name(config)
    namespace = kube_options.namespace or DEFAULT_BENCHMARK_NAMESPACE

    config_dict = config.model_dump(mode="json", exclude_none=True)
    apply_k8s_runtime_config(config_dict, name, namespace)
    config = AIPerfConfig.model_validate(config_dict)

    deploy_config = kube_options.to_deployment_config()
    deploy_config.ttl_seconds_after_finished = (
        K8sEnvironment.JOBSET.DIRECT_MODE_TTL_SECONDS
    )

    concurrency = max(
        (getattr(phase, "concurrency", 1) or 1 for phase in config.phases.values()),
        default=1,
    )
    total_workers = max(
        1, math.ceil(concurrency / deploy_config.connections_per_worker)
    )
    num_pods = apply_worker_config(config, total_workers)

    deployment = KubernetesDeployment(
        job_id=name,
        namespace=kube_options.namespace,
        worker_replicas=num_pods,
        config=config,
        deployment=deploy_config,
        model_names=config.get_model_names(),
        endpoint_url=config.endpoint.urls[0] if config.endpoint.urls else None,
    )

    manifests = deployment.get_all_manifests()
    manifest_yaml = _manifests_to_yaml(manifests)

    if dry_run:
        print(manifest_yaml, end="")
        return

    kubectl("apply", "-f", "-", input=manifest_yaml, text=True)
    log_success(f"Deployed to namespace: {namespace} (job_id: {name})")
    _mode.set_result(
        {"job_id": name, "namespace": namespace, "workers": num_pods, "detach": detach}
    )

    if detach or _mode.json:
        log_info("Detached mode — benchmark running in background")
        log_info(f"Check status: uv run aiperf kube status {name}")
        log_info(f"View logs:    uv run aiperf kube logs {name}")
    else:
        if shutil.which("uv") is None:
            log_error("uv not found on PATH. Install uv or activate the project venv.")
            raise SystemExit(1)
        log_info("Attaching to benchmark...")
        os.execvp(
            "uv",
            ["uv", "run", "aiperf", "kube", "attach", name, "--namespace", namespace],
        )


# ---------------------------------------------------------------------------
# Single-pod benchmark (run-local)
# ---------------------------------------------------------------------------


def _generate_single_pod_manifest(
    cli_args: list[str],
    namespace: str,
    image: str,
) -> str:
    """Generate Namespace + Job YAML for a single-pod `aiperf profile` run."""
    documents: list[dict] = [
        _namespace_manifest(namespace),
        {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": "aiperf-benchmark",
                "namespace": namespace,
                "labels": {"app": "aiperf"},
            },
            "spec": {
                "backoffLimit": 0,
                "ttlSecondsAfterFinished": 300,
                "template": {
                    "metadata": {"labels": {"app": "aiperf"}},
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [
                            {
                                "name": "aiperf",
                                "image": image,
                                "imagePullPolicy": "Never",
                                "command": ["aiperf"],
                                "args": ["profile", *cli_args],
                            }
                        ],
                    },
                },
            },
        },
    ]
    return _manifests_to_yaml(documents)


def cmd_run_local(*, detach: bool, dry_run: bool, extra_args: list[str]) -> None:
    """Run benchmark as a single pod via `aiperf profile` CLI flags.

    All aiperf arguments must be passed after `--`.
    """
    MOCK_SERVER_URL = "http://aiperf-mock-server.default.svc.cluster.local:8000"

    if not extra_args:
        log_error(
            "No aiperf arguments provided. Pass them after '--', e.g.:\n"
            f"  ./dev/kube.py run-local -- --model mock --url {MOCK_SERVER_URL} --endpoint-type chat"
        )
        raise SystemExit(1)

    cli_args = ["--url", MOCK_SERVER_URL, *extra_args]

    namespace = f"aiperf-local-{int(time.time())}"
    manifest = _generate_single_pod_manifest(cli_args, namespace, AIPERF_IMAGE)

    if dry_run:
        print(manifest, end="")
        return

    _require_kubectl_and_cluster()
    log_step("Deploying single-pod benchmark")
    kubectl("apply", "-f", "-", input=manifest, text=True)
    log_success(f"Deployed to namespace: {namespace}")

    if detach:
        log_info("Detached mode — benchmark running in background")
        log_info(f"View logs: kubectl logs -n {namespace} -l app=aiperf -f")
        log_info(f"Cleanup:   kubectl delete namespace {namespace}")
    else:
        log_info("Waiting for pod to start...")
        kubectl(
            "wait",
            "--for=condition=Ready",
            "pod",
            "-l",
            "app=aiperf",
            "-n",
            namespace,
            f"--timeout={120 * _TIMEOUT_SCALE}s",
            check=False,
        )
        log_info("Attaching to benchmark logs (Ctrl-C to detach)...")
        kubectl(
            "logs",
            "-n",
            namespace,
            "-l",
            "app=aiperf",
            "-f",
            "--all-containers",
            check=False,
        )


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


_STATUS_W = 20  # label column width for status output


def _sline(label: str, indicator: str, msg: str) -> str:
    """Format a single status line: '  Label          ✓ message'."""
    return f"  {_CYAN}{label:<{_STATUS_W}}{_NC}{indicator} {msg}"


def _s_ok(label: str, msg: str) -> str:
    return _sline(label, f"{_GREEN}\u2713{_NC}", msg)


def _s_miss(label: str, msg: str) -> str:
    return _sline(label, f"{_YELLOW}\u25cb{_NC}", msg)


def _s_fail(label: str, msg: str) -> str:
    return _sline(label, f"{_RED}\u2717{_NC}", msg)


def _s_deploy(label: str, name: str, namespace: str | None = None) -> str:
    """Return a status line for a Kubernetes deployment."""
    ready = _deployment_ready_replicas(name, namespace)
    if ready is not None:
        return _s_ok(label, f"{ready} ready")
    return _s_miss(label, "not found")


def _s_image(image: str) -> str:
    """Return a status line for a Docker image."""
    short = image.split(":")[0]
    r = sh(
        "docker",
        "image",
        "inspect",
        image,
        "--format",
        "{{.Size}}",
        capture=True,
        check=False,
    )
    if r.returncode != 0:
        return _s_miss(short, "not built")
    try:
        size = _format_bytes_iec(int(r.stdout.strip()))
    except (ValueError, TypeError):
        size = "?"
    return _s_ok(short, size)


def _collect_status() -> dict:
    """Collect cluster status as a structured dict."""
    rt = _effective_runtime()
    result: dict[str, Any] = {
        "cluster": {
            "name": CLUSTER_NAME,
            "runtime": rt,
            "exists": False,
            "running": False,
            "connected": False,
        },
        "images": [],
        "infra": {},
        "servers": {},
        "workloads": {},
    }

    if not cluster_exists():
        return result
    result["cluster"]["exists"] = True

    if not cluster_running():
        return result
    result["cluster"]["running"] = True

    r = kubectl("cluster-info", capture=True, check=False)
    result["cluster"]["connected"] = r.returncode == 0
    if r.returncode != 0:
        return result

    # images
    for image in (AIPERF_IMAGE, MOCK_SERVER_IMAGE):
        r = sh(
            "docker",
            "image",
            "inspect",
            image,
            "--format={{.Size}}",
            capture=True,
            check=False,
        )
        built = r.returncode == 0
        size_bytes = (
            int(r.stdout.strip()) if built and r.stdout.strip().isdigit() else None
        )
        result["images"].append(
            {"name": image, "built": built, "size_bytes": size_bytes}
        )

    # infra
    result["infra"]["jobset"] = {
        "ready_replicas": _deployment_ready_replicas(
            "jobset-controller-manager", "jobset-system"
        )
    }
    result["infra"]["mock_server"] = {
        "ready_replicas": _deployment_ready_replicas("aiperf-mock-server")
    }

    gpu_count = _get_gpu_allocatable()
    result["infra"]["gpu"] = {
        "count": int(gpu_count) if gpu_count and gpu_count.isdigit() else 0
    }

    r = kubectl(
        "get", "crd", "dynamographdeployments.nvidia.com", capture=True, check=False
    )
    result["infra"]["dynamo_operator"] = {"installed": r.returncode == 0}

    # servers
    result["servers"]["vllm"] = {
        "ready_replicas": _deployment_ready_replicas("vllm-server", VLLM_NAMESPACE)
    }

    r = kubectl(
        "get", "pods", "-n", DYNAMO_NAMESPACE, "--no-headers", capture=True, check=False
    )
    if r.returncode == 0 and r.stdout.strip():
        lines = r.stdout.strip().splitlines()
        running = sum(1 for ln in lines if "Running" in ln)
        mode = DYNAMO_MODE
        for known in ("disagg-1gpu", "disagg", "agg"):
            if any(f"dynamo-{known}" in ln for ln in lines):
                mode = known
                break
        result["servers"]["dynamo"] = {
            "deployed": True,
            "running": running,
            "total": len(lines),
            "mode": mode,
        }
    else:
        result["servers"]["dynamo"] = {"deployed": False}

    r = kubectl(
        "get",
        "dynamomodel",
        "-n",
        DYNAMO_NAMESPACE,
        "--no-headers",
        capture=True,
        check=False,
    )
    if r.returncode == 0 and r.stdout.strip():
        result["servers"]["lora"] = [
            ln.split()[0] for ln in r.stdout.strip().splitlines()
        ]
    else:
        result["servers"]["lora"] = []

    # workloads
    r = kubectl(
        "get",
        "jobsets",
        "--all-namespaces",
        "-o",
        "jsonpath={.items[*].metadata.name}",
        capture=True,
        check=False,
    )
    result["workloads"]["benchmarks"] = (
        r.stdout.strip().split() if r.stdout.strip() else []
    )
    result["workloads"]["namespaces"] = _get_aiperf_namespaces()

    return result


def cmd_status() -> None:
    require("kubectl")
    status = _collect_status()

    if _mode.json:
        _mode.set_result(status)
        return

    rt = status["cluster"]["runtime"]
    out = _log_dest()
    print(file=out)

    # -- cluster --
    if not status["cluster"]["exists"]:
        print(_s_miss("Cluster", f"{CLUSTER_NAME} not found"), file=out)
        print("\n  Run: ./dev/kube.py cluster-create", file=out)
        return
    if not status["cluster"]["running"]:
        print(_s_miss("Cluster", f"{CLUSTER_NAME} stopped"), file=out)
        hint = (
            "./dev/kube.py cluster-create"
            if rt == "kind"
            else f"minikube start -p {CLUSTER_NAME}"
        )
        print(f"\n  Run: {hint}", file=out)
        return
    if status["cluster"]["connected"]:
        print(_s_ok("Cluster", f"{CLUSTER_NAME} ({rt})"), file=out)
    else:
        print(_s_fail("Cluster", "kubectl cannot connect"), file=out)
        return

    # -- images --
    for img in status["images"]:
        short = img["name"].split(":")[0]
        if img["built"]:
            size = _format_bytes_iec(img["size_bytes"]) if img["size_bytes"] else "?"
            print(_s_ok(short, size), file=out)
        else:
            print(_s_miss(short, "not built"), file=out)
    print(file=out)

    # -- infra --
    for label, key in [("JobSet", "jobset"), ("Mock Server", "mock_server")]:
        ready = status["infra"][key]["ready_replicas"]
        print(
            _s_ok(label, f"{ready} ready")
            if ready is not None
            else _s_miss(label, "not found"),
            file=out,
        )

    gpu = status["infra"]["gpu"]["count"]
    print(
        _s_ok("GPU", f"{gpu} nvidia.com/gpu")
        if gpu
        else _s_miss("GPU", "none detected"),
        file=out,
    )
    print(
        _s_ok("Dynamo Operator", "installed")
        if status["infra"]["dynamo_operator"]["installed"]
        else _s_miss("Dynamo Operator", "not installed"),
        file=out,
    )
    print(file=out)

    # -- servers --
    vllm_ready = status["servers"]["vllm"]["ready_replicas"]
    print(
        _s_ok("vLLM", f"{vllm_ready} ready")
        if vllm_ready is not None
        else _s_miss("vLLM", "not found"),
        file=out,
    )

    dynamo = status["servers"]["dynamo"]
    if dynamo["deployed"]:
        print(
            _s_ok(
                "Dynamo Server",
                f"{dynamo['running']}/{dynamo['total']} pods, mode={dynamo['mode']}",
            ),
            file=out,
        )
    else:
        print(_s_miss("Dynamo Server", "not deployed"), file=out)

    lora = status["servers"]["lora"]
    print(_s_ok("LoRA", ", ".join(lora)) if lora else _s_miss("LoRA", "none"), file=out)
    print(file=out)

    # -- workloads --
    benchmarks = status["workloads"]["benchmarks"]
    if benchmarks:
        print(_s_ok("Benchmarks", f"{len(benchmarks)} active"), file=out)
        kubectl("get", "jobsets", "--all-namespaces", "-o", "wide", check=False)
    else:
        print(_s_miss("Benchmarks", "none"), file=out)

    namespaces = status["workloads"]["namespaces"]
    if namespaces:
        print(
            _s_ok("Namespaces", f"{len(namespaces)}: {', '.join(namespaces)}"), file=out
        )
    else:
        print(_s_miss("Namespaces", "none"), file=out)
    print(file=out)


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------


@app.command(name="logs", group=lowlevel, help="View AIPerf benchmark pod logs.")
def cmd_logs(
    *,
    output: OutputOptions = _DEFAULT_OUTPUT_OPTS,
    namespace: Annotated[
        str | None,
        Parameter(name=["--namespace", "-n"], help="Kubernetes namespace."),
    ] = os.environ.get("NS") or None,
    pod: Annotated[
        str, Parameter(name=["--pod", "-p"], help="Pod name filter.")
    ] = os.environ.get("POD") or "controller",
    follow: FollowFlag = FOLLOW_DEFAULT,
) -> None:
    _apply_output(output)
    _require_kubectl_and_cluster()

    if not namespace:
        ns_list = _get_aiperf_namespaces(latest_only=True)
        if not ns_list:
            log_error("No AIPerf namespace found. Run a benchmark first.")
            raise SystemExit(1)
        namespace = ns_list[0]
        log_info(f"Using namespace: {namespace}")

    follow_flag = ["-f"] if follow else []

    if pod == "all":
        kubectl(
            "logs",
            "-n",
            namespace,
            *follow_flag,
            "-l",
            "job-name",
            "--all-containers",
            "--prefix",
        )
        return

    r = kubectl(
        "get", "pods", "-n", namespace, "-l", "job-name", "-o", "name", capture=True
    )
    match = [p for p in r.stdout.strip().splitlines() if pod in p]

    if not match:
        log_error(f"Pod '{pod}' not found in {namespace}")
        raise SystemExit(1)

    extra = ["-c", "system-controller"] if pod == "controller" else []
    kubectl("logs", "-n", namespace, *follow_flag, match[0], *extra)


# ---------------------------------------------------------------------------
# Cleanup / teardown / reload
# ---------------------------------------------------------------------------


def cmd_cleanup() -> None:
    log_step("Cleaning up AIPerf resources")
    require("kubectl")

    if not cluster_running():
        log_warn(f"Cluster {CLUSTER_NAME} is not running")
        return

    log_info("Deleting all AIPerf namespaces...")
    namespaces = _get_aiperf_namespaces()
    if namespaces:
        for ns in namespaces:
            log_info(f"  Deleting: {ns}")
            kubectl("delete", "namespace", ns, "--ignore-not-found", check=False)
    else:
        log_info("  No AIPerf namespaces found")

    log_info("Deleting any orphaned JobSets...")
    kubectl(
        "delete",
        "jobsets",
        "--all",
        "--all-namespaces",
        "--ignore-not-found",
        check=False,
    )
    log_success("Cleanup complete")
    _mode.set_result({"deleted_namespaces": namespaces, "message": "cleanup complete"})


def cmd_teardown() -> None:
    if cluster_running():
        cmd_cleanup()
    cmd_cluster_delete()
    _mode.set_result({"message": f"cluster {CLUSTER_NAME} torn down"})


def cmd_reload() -> None:
    cmd_build(aiperf=True, mock=False)
    cmd_load(aiperf=True, mock=False)
    log_success("AIPerf image rebuilt and reloaded")
    _mode.set_result({"images": [AIPERF_IMAGE], "message": "rebuilt and reloaded"})


# ---------------------------------------------------------------------------
# Setup pipeline
# ---------------------------------------------------------------------------


@dataclass
class _SetupStep:
    """A single step in the setup pipeline with status tracking."""

    label: str
    func: Any  # Callable[[], str]
    optional: bool = False
    skip: bool = False
    skip_reason: str = ""
    status: str = ""
    failed: bool = False
    error: str = ""


class _StepProgress:
    """Live renderable: spinner + step label + elapsed stopwatch."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.t0 = time.monotonic()
        self._spinner = Spinner("dots", style="cyan")

    def __rich_console__(self, console: Console, options: Any) -> Any:
        elapsed = int(time.monotonic() - self.t0)
        line = Text.assemble(
            "  ",
            self._spinner.render(console.get_time()),
            " ",
            (f"{self.label:16s}", "bold"),
            f" {elapsed}s",
        )
        yield line


def _run_setup_steps(
    steps: list[_SetupStep], *, continue_on_error: bool, has_gpu: bool
) -> None:
    """Execute setup steps and display results."""
    if _mode.json:
        _run_setup_steps_json(steps, continue_on_error=continue_on_error)
        return

    # Console on a dup'd fd so it can write to the terminal even while
    # fd 1/2 are redirected to /dev/null during step execution.
    con_fd = os.dup(sys.stdout.fileno())
    con_file = os.fdopen(con_fd, "w", buffering=1)
    con = Console(file=con_file)

    try:
        _run_setup_steps_inner(
            steps, con=con, continue_on_error=continue_on_error, has_gpu=has_gpu
        )
    finally:
        con_file.close()


def _run_setup_steps_json(steps: list[_SetupStep], *, continue_on_error: bool) -> None:
    """Execute setup steps with JSONL streaming."""
    _mode._streaming = True
    step_results: list[dict] = []
    t0 = time.monotonic()

    for step in steps:
        if step.skip:
            evt = {"label": step.label, "status": "skipped", "detail": step.skip_reason}
            step_results.append(evt)
            _mode.emit_event(
                {"type": "step_skipped", "step": step.label, "reason": step.skip_reason}
            )
            continue

        _mode.emit_event(
            {"type": "step_started", "step": step.label, "timestamp": time.time()}
        )
        step_t0 = time.monotonic()
        try:
            with _StreamEmitter(step.label):
                step.status = step.func() or "done"
            secs = int(time.monotonic() - step_t0)
            evt = {
                "label": step.label,
                "status": "ok",
                "detail": step.status,
                "seconds": secs,
            }
            step_results.append(evt)
            _mode.emit_event(
                {
                    "type": "step_completed",
                    "step": step.label,
                    "detail": step.status,
                    "seconds": secs,
                }
            )
        except (SystemExit, Exception) as exc:
            step.failed = True
            step.error = str(exc) or repr(exc)
            secs = int(time.monotonic() - step_t0)
            evt = {
                "label": step.label,
                "status": "error",
                "detail": step.error,
                "seconds": secs,
            }
            step_results.append(evt)
            _mode.emit_event(
                {
                    "type": "step_failed",
                    "step": step.label,
                    "error": step.error,
                    "seconds": secs,
                }
            )
            if not (step.optional and continue_on_error):
                break

    elapsed = int(time.monotonic() - t0)
    _mode.set_result({"steps": step_results, "elapsed_seconds": elapsed})
    _mode.emit_summary()


def _run_setup_steps_inner(
    steps: list[_SetupStep],
    *,
    con: Console,
    continue_on_error: bool,
    has_gpu: bool,
) -> None:
    """Inner logic for _run_setup_steps (separated so con_file cleanup is guaranteed)."""
    rt = _effective_runtime()
    con.print(Rule(f"[bold cyan]Setup — aiperf {rt} cluster[/]", style="cyan"))

    # Environment summary
    parts: list[str] = []
    parts.append(rt)
    parts.append("GPU" if has_gpu else "CPU-only")
    parts.append(f"{_docker_cpus} CPUs")
    parts.append(f"{_docker_mem_mb // 1024} GB")
    if _TIMEOUT_SCALE > 1:
        parts.append(f"timeout \u00d7{_TIMEOUT_SCALE}")
    con.print(
        Panel(
            " \u00b7 ".join(parts),
            title="[bold]System[/]",
            border_style="dim",
        )
    )

    t0 = time.monotonic()

    for step in steps:
        if step.skip:
            con.print(
                f"  [dim]\u25cb[/] [bold]{step.label:16s}[/] [dim]{step.skip_reason}[/]"
            )
            continue

        progress = _StepProgress(step.label)

        # Redirect Python streams (captures log_error messages in captured_err)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        captured_err = io.StringIO()
        sys.stdout = io.StringIO()
        sys.stderr = captured_err

        # Redirect OS-level fds so subprocess output goes to /dev/null
        saved_fd1 = os.dup(1)
        saved_fd2 = os.dup(2)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)

        try:
            with Live(progress, console=con, transient=True, refresh_per_second=8):
                step.status = step.func() or "done"
            step_secs = int(time.monotonic() - progress.t0)
            elapsed_tag = f" ({step_secs}s)" if step_secs >= 2 else ""
            con.print(
                f"  [green]\u2713[/] [bold]{step.label:16s}[/] {step.status}{elapsed_tag}"
            )
        except (SystemExit, Exception) as exc:
            step.failed = True
            # Prefer captured log_error() text over bare str(SystemExit(1))=="1"
            stderr_text = _ANSI_ESCAPE.sub("", captured_err.getvalue()).strip()
            step.error = stderr_text or str(exc) or repr(exc)
            con.print(
                f"  [red]\u2717[/] [bold]{step.label:16s}[/] [red]{escape(step.error)}[/]"
            )
            if not (step.optional and continue_on_error):
                raise
        finally:
            # Restore OS-level fds first, then Python streams
            os.dup2(saved_fd1, 1)
            os.dup2(saved_fd2, 2)
            os.close(saved_fd1)
            os.close(saved_fd2)
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    elapsed = time.monotonic() - t0
    failed = [s for s in steps if s.failed]

    con.print()
    if failed:
        lines = "\n".join(
            f"  [red]\u2717[/] {s.label}: {escape(s.error)}" for s in failed
        )
        con.print(
            Panel(
                f"{lines}\n\n"
                "Re-run failing steps individually or fix the issue and run setup again.",
                title="[bold yellow]Setup completed with errors[/]",
                border_style="yellow",
            )
        )
    else:
        con.print(
            Panel(
                f"Done in {elapsed:.0f}s\n\n"
                "  ./dev/kube.py run              # benchmark mock server\n"
                "  ./dev/kube.py run-local        # single-pod benchmark\n"
                "  ./dev/kube.py deploy-vllm      # deploy real server\n"
                "  ./dev/kube.py deploy-dynamo    # deploy Dynamo server",
                title="[bold green]Setup complete[/]",
                border_style="green",
            )
        )


@app.command(
    name="setup",
    group=workflow,
    help="Full setup: cluster + GPU + Dynamo operator + JobSet + mock server.",
)
def cmd_setup(
    *,
    output: OutputOptions = _DEFAULT_OUTPUT_OPTS,
    no_dynamo: Annotated[
        bool,
        Parameter(
            name=["-D", "--no-dynamo"],
            negative="",
            help="Skip Dynamo operator install.",
        ),
    ] = False,
    no_jobset: Annotated[
        bool,
        Parameter(
            name=["-J", "--no-jobset"],
            negative="",
            help="Skip JobSet controller install.",
        ),
    ] = False,
    no_mock: Annotated[
        bool,
        Parameter(
            name=["-M", "--no-mock"], negative="", help="Skip mock server deploy."
        ),
    ] = False,
    continue_on_error: Annotated[
        bool,
        Parameter(
            name=["-k", "--continue-on-error"],
            negative="",
            help="Continue past failures in optional steps.",
        ),
    ] = False,
    strip_proxy: Annotated[
        bool,
        Parameter(
            name="--strip-proxy",
            negative="--keep-proxy",
            help="Strip localhost-bound HTTP_PROXY/HTTPS_PROXY vars that are unreachable from Kind containers (default: on).",
        ),
    ] = True,
) -> None:
    """Full setup: cluster + GPU + Dynamo operator + JobSet + images + servers."""
    _apply_output(output)
    if strip_proxy and _effective_runtime() == "kind":
        _strip_localhost_proxy()
    _preflight()

    has_gpu = shutil.which("nvidia-smi") is not None
    if not has_gpu and not no_dynamo:
        no_dynamo = True

    build_mock = not no_mock

    steps: list[_SetupStep] = [
        _SetupStep(label="Cluster", func=cmd_cluster_create),
        _SetupStep(
            label="Build images",
            func=lambda: cmd_build(aiperf=True, mock=build_mock),
        ),
        _SetupStep(
            label="Load images",
            func=lambda: cmd_load(aiperf=True, mock=build_mock),
        ),
        _SetupStep(
            label="Dynamo",
            func=cmd_install_dynamo,
            optional=True,
            skip=no_dynamo,
            skip_reason="skipped (no GPU)" if not has_gpu else "skipped (--no-dynamo)",
        ),
        _SetupStep(
            label="AIPerf operator",
            func=cmd_install_aiperf_operator,
        ),
        _SetupStep(
            label="JobSet",
            func=cmd_install_jobset,
            optional=True,
            skip=no_jobset,
            skip_reason="skipped (--no-jobset)",
        ),
        _SetupStep(
            label="Mock server",
            func=cmd_deploy_mock,
            optional=True,
            skip=no_mock,
            skip_reason="skipped (--no-mock)",
        ),
    ]

    _run_setup_steps(steps, continue_on_error=continue_on_error, has_gpu=has_gpu)


# ---------------------------------------------------------------------------
# Command registration — no-arg commands (table-driven)
# ---------------------------------------------------------------------------


def _with_output(func: Callable[[], Any]) -> Callable[..., Any]:
    """Wrap a no-arg command to accept OutputOptions and apply them."""

    def wrapper(*, output: OutputOptions = _DEFAULT_OUTPUT_OPTS) -> Any:
        _apply_output(output)
        return func()

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


# fmt: off
for _func, _name, _group, _help in [
    (cmd_doctor,         "doctor",         workflow, "Check prerequisites (kind/minikube, kubectl, helm, docker, nvidia, k9s)."),
    (cmd_teardown,       "teardown",       workflow, "Delete entire local cluster."),
    (cmd_status,         "status",         workflow, "Show cluster status (GPU, Dynamo, vLLM, benchmarks)."),
    (cmd_reload,         "reload",         workflow, "Rebuild + load AIPerf image."),
    (cmd_remove_dynamo,  "remove-dynamo",  server,   "Remove Dynamo server."),
    (cmd_remove_vllm,    "remove-vllm",    server,   "Remove vLLM server."),
    (cmd_remove_sglang,  "remove-sglang",  server,   "Remove SGLang server."),
    (cmd_remove_trtllm,  "remove-trtllm",  server,   "Remove TensorRT-LLM server."),
    (cmd_deploy_mock,    "deploy-mock",    server,   "Deploy mock LLM server."),
    (cmd_remove_mock,    "remove-mock",    server,   "Remove mock server."),
    (cmd_cluster_create, "cluster-create", lowlevel, "Create local cluster (Kind or Minikube)."),
    (cmd_cluster_delete, "cluster-delete", lowlevel, "Delete local cluster."),
    (cmd_install_dynamo,          "install-dynamo",          lowlevel, "Install Dynamo operator (Helm)."),
    (cmd_install_aiperf_operator, "install-aiperf-operator", lowlevel, "Install AIPerf operator (Helm)."),
    (cmd_install_jobset,          "install-jobset",          lowlevel, "Install JobSet controller."),
    (cmd_install_kueue,           "install-kueue",           lowlevel, "Install Kueue + default ClusterQueue/LocalQueue."),
    (cmd_remove_kueue,            "remove-kueue",            lowlevel, "Remove Kueue and default queues."),
    (cmd_install_prometheus,      "install-prometheus",      lowlevel, "Install kube-prometheus-stack (Prometheus + Grafana)."),
    (cmd_install_loki,            "install-loki",            lowlevel, "Install loki-stack (Loki + Promtail)."),
    (cmd_cleanup,        "cleanup",        lowlevel, "Remove benchmark namespaces."),
]:
    app.command(_with_output(_func), name=_name, group=_group, help=_help)
# fmt: on

# ---------------------------------------------------------------------------
# Thin wrappers — commands that delegate with hardcoded args
# ---------------------------------------------------------------------------


# fmt: off
@app.command(name="run",        group=benchmark, help="Run AIPerf benchmark (attach).")
def _cli_run(*, output: OutputOptions = _DEFAULT_OUTPUT_OPTS, opts: RunOptions = _DEFAULT_RUN_OPTS) -> None:
    _apply_output(output)
    cmd_run(opts=opts, detach=False, dry_run=False)

@app.command(name="run-detach", group=benchmark, help="Run benchmark (detach).")
def _cli_run_detach(*, output: OutputOptions = _DEFAULT_OUTPUT_OPTS, opts: RunOptions = _DEFAULT_RUN_OPTS) -> None:
    _apply_output(output)
    cmd_run(opts=opts, detach=True,  dry_run=False)

@app.command(name="dry-run", group=benchmark, help="Print manifest only.")
def _cli_dry_run(*, output: OutputOptions = _DEFAULT_OUTPUT_OPTS, opts: RunOptions = _DEFAULT_RUN_OPTS) -> None:
    _apply_output(output)
    cmd_run(opts=opts, detach=False, dry_run=True)

@app.command(name="run-local",        group=benchmark, help="Run single-pod benchmark (attach). Pass aiperf args after '--'.")
def _cli_run_local(*extra_args: str, output: OutputOptions = _DEFAULT_OUTPUT_OPTS) -> None:
    _apply_output(output)
    cmd_run_local(detach=False, dry_run=False, extra_args=list(extra_args))

@app.command(name="run-local-detach", group=benchmark, help="Run single-pod benchmark (detach). Pass aiperf args after '--'.")
def _cli_run_local_detach(*extra_args: str, output: OutputOptions = _DEFAULT_OUTPUT_OPTS) -> None:
    _apply_output(output)
    cmd_run_local(detach=True,  dry_run=False, extra_args=list(extra_args))

@app.command(name="dry-run-local",    group=benchmark, help="Print single-pod manifest only. Pass aiperf args after '--'.")
def _cli_dry_run_local(*extra_args: str, output: OutputOptions = _DEFAULT_OUTPUT_OPTS) -> None:
    _apply_output(output)
    cmd_run_local(detach=False, dry_run=True, extra_args=list(extra_args))

@app.command(name="build", group=lowlevel, help="Build Docker images.")
def _cli_build(*, output: OutputOptions = _DEFAULT_OUTPUT_OPTS) -> None:
    _apply_output(output)
    cmd_build(aiperf=True, mock=True)

@app.command(name="load",  group=lowlevel, help="Load images into cluster.")
def _cli_load(*, output: OutputOptions = _DEFAULT_OUTPUT_OPTS) -> None:
    _apply_output(output)
    cmd_load(aiperf=True, mock=True)

@app.command(name="push", group=lowlevel, help="Retag and docker push the aiperf image to a remote registry.")
def _cli_push(
    *,
    output: OutputOptions = _DEFAULT_OUTPUT_OPTS,
    registry: Annotated[str, Parameter(name=["--registry", "-r"], help="Target registry host or full image path (e.g. 'nvcr.io/my-org' or 'nvcr.io/my-org/aiperf').")],
    tag: Annotated[str | None, Parameter(name=["--tag", "-t"], help="Remote tag (default: same tag as the local AIPERF_IMAGE).")] = None,
) -> None:
    _apply_output(output)
    cmd_push(registry=registry, tag=tag)
# fmt: on


if __name__ == "__main__":
    # Detect --json early so we can suppress the banner
    _raw_args = sys.argv[1:]
    if "--json" not in _raw_args:
        print(_BANNER, flush=True)

    # Extract command name for JSON output
    for arg in _raw_args:
        if not arg.startswith("-"):
            _mode._command = arg
            break

    try:
        app.meta(_raw_args)
    except SystemExit:
        _mode.emit()
        raise
    except Exception as exc:
        _mode.add_error(str(exc))
        _mode.emit()
        raise
    else:
        _mode.emit()
