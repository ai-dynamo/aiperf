# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pytest fixtures for Kubernetes E2E tests.

All settings can be configured via CLI options (``--k8s-*``) or environment
variables (``K8S_TEST_*``).  CLI takes precedence over environment.

Usage::

    uv run pytest tests/kubernetes/ -v --ignore=tests/kubernetes/gpu
    uv run pytest tests/kubernetes/ -v --k8s-reuse-cluster --k8s-skip-build
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console

import pytest
import pytest_asyncio

from aiperf.common.aiperf_logger import AIPerfLogger
from dev.versions import JOBSET_CRD_URL_TEMPLATE, JOBSET_VERSION, KUEUE_VERSION
from tests.kubernetes.helpers.benchmark import (
    BenchmarkConfig,
    BenchmarkDeployer,
    BenchmarkResult,
)
from tests.kubernetes.helpers.cluster import ClusterConfig, ClusterRuntime, LocalCluster
from tests.kubernetes.helpers.helm import HelmClient, HelmDeployer, HelmValues
from tests.kubernetes.helpers.images import ImageConfig, ImageManager
from tests.kubernetes.helpers.kubectl import KubectlClient
from tests.kubernetes.helpers.operator import (
    AIPerfJobConfig,
    OperatorDeployer,
    OperatorJobResult,
)

logger = AIPerfLogger(__name__)


def _is_tty() -> bool:
    """Detect whether stderr is a TTY (use plain logging otherwise)."""
    return sys.stderr.isatty()


# Stash key for K8sTestSettings on pytest.Config
_SETTINGS_KEY = pytest.StashKey["K8sTestSettings"]()


# ============================================================================
# K8sTestSettings - single source of truth for all local test configuration
# ============================================================================


@dataclass(frozen=True)
class K8sTestSettings:
    """Resolved K8s test configuration (CLI > env > default)."""

    cluster: str = "aiperf-pytest"
    """Cluster name for identification and context."""

    runtime: str = "kind"
    """Cluster runtime backend (kind or minikube)."""

    quick: bool = False
    """Shorthand for reuse + skip build/load/cleanup/preflight."""

    skip_build: bool = False
    """Skip building Docker images."""

    skip_load: bool = False
    """Skip loading images into the cluster."""

    skip_cleanup: bool = False
    """Keep cluster and resources after tests."""

    reuse_cluster: bool = False
    """Reuse an existing cluster instead of creating a new one."""

    skip_preflight: bool = False
    """Skip preflight prerequisite checks."""

    stream_logs: bool = False
    """Stream pod logs in real time during deploys."""

    aiperf_image: str = "aiperf:local"
    """AIPerf container image name."""

    mock_server_image: str = "aiperf-mock-server:latest"
    """Mock server container image name."""

    jobset_version: str = JOBSET_VERSION
    """JobSet controller version to install."""

    kueue_version: str = KUEUE_VERSION
    """Kueue controller version to install."""

    benchmark_timeout: int = 600
    """Benchmark completion timeout in seconds."""

    @property
    def cluster_runtime(self) -> ClusterRuntime:
        """Get the cluster runtime enum value."""
        return ClusterRuntime(self.runtime)


def _get_settings(config: pytest.Config) -> K8sTestSettings:
    """Retrieve K8sTestSettings from pytest config stash."""
    return config.stash[_SETTINGS_KEY]


# ============================================================================
# CLI options
# ============================================================================

# Option definitions: (cli_flag, env_var, default, type, help)
_OPTIONS: list[tuple[str, str, str | None, str, str]] = [
    ("--k8s-cluster", "K8S_TEST_CLUSTER", None, "str", "Cluster name (default: aiperf-<uuid>)"),
    ("--k8s-runtime", "K8S_TEST_RUNTIME", "kind", "str", "Cluster runtime: kind or minikube (default: kind)"),
    ("--k8s-quick", "K8S_TEST_QUICK", None, "bool", "Reuse cluster, skip build/load/cleanup/preflight"),
    ("--k8s-skip-build", "K8S_TEST_SKIP_BUILD", None, "bool", "Skip building Docker images (use existing)"),
    ("--k8s-skip-load", "K8S_TEST_SKIP_LOAD", None, "bool", "Skip loading images into cluster"),
    ("--k8s-skip-cleanup", "K8S_TEST_SKIP_CLEANUP", None, "bool", "Keep cluster and resources after tests"),
    ("--k8s-reuse-cluster", "K8S_TEST_REUSE_CLUSTER", None, "bool", "Reuse existing cluster"),
    ("--k8s-skip-preflight", "K8S_TEST_SKIP_PREFLIGHT", None, "bool", "Skip preflight checks"),
    ("--k8s-stream-logs", "K8S_TEST_STREAM_LOGS", None, "bool", "Stream pod logs in real time during deploys"),
    ("--k8s-aiperf-image", "K8S_TEST_AIPERF_IMAGE", "aiperf:local", "str", "AIPerf container image"),
    ("--k8s-mock-server-image", "K8S_TEST_MOCK_SERVER_IMAGE", "aiperf-mock-server:latest", "str", "Mock server container image"),
    ("--k8s-jobset-version", "K8S_TEST_JOBSET_VERSION", JOBSET_VERSION, "str", "JobSet controller version"),
    ("--k8s-kueue-version", "K8S_TEST_KUEUE_VERSION", KUEUE_VERSION, "str", "Kueue controller version"),
    ("--k8s-benchmark-timeout", "K8S_TEST_BENCHMARK_TIMEOUT", "300", "int", "Benchmark timeout in seconds"),
]  # fmt: skip


def _generate_cluster_name() -> str:
    """Generate a unique cluster name using first segment of UUID4."""
    uuid_prefix = str(uuid.uuid4()).split("-")[0]
    return f"aiperf-{uuid_prefix}"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register --k8s-* CLI options."""
    group = parser.getgroup("k8s", "Kubernetes E2E test options")

    for cli_flag, _env_var, _default, typ, help_text in _OPTIONS:
        if typ == "bool":
            group.addoption(cli_flag, action="store_true", default=None, help=help_text)
        else:
            group.addoption(cli_flag, default=None, help=help_text)


def _resolve_settings(config: pytest.Config) -> K8sTestSettings:
    """Build K8sTestSettings from CLI options + env vars + defaults.

    Priority: CLI option > environment variable > default.
    """

    def _resolve(cli_flag: str, env_var: str, default: str | None, typ: str) -> object:
        attr = cli_flag.lstrip("-").replace("-", "_")
        cli_val = getattr(config.option, attr, None)

        if cli_val is not None:
            raw = cli_val
        else:
            env_val = os.environ.get(env_var, "")
            raw = env_val if env_val else default

        if raw is None:
            return None

        if typ == "bool":
            if isinstance(raw, bool):
                return raw
            return str(raw).lower() in ("1", "true", "yes")
        if typ == "int":
            return int(raw)
        return str(raw)

    resolved = {}
    for cli_flag, env_var, default, typ, _help in _OPTIONS:
        field_name = cli_flag.removeprefix("--k8s-").replace("-", "_")
        resolved[field_name] = _resolve(cli_flag, env_var, default, typ)

    # --k8s-quick implies reuse + skip build/load/cleanup/preflight
    if resolved.get("quick"):
        for key in (
            "reuse_cluster",
            "skip_build",
            "skip_load",
            "skip_cleanup",
            "skip_preflight",
        ):
            if resolved.get(key) is None:
                resolved[key] = True

    # Use stable name when reusing, random name otherwise
    if resolved.get("cluster") is None:
        if resolved.get("reuse_cluster"):
            resolved["cluster"] = "aiperf-pytest"
        else:
            resolved["cluster"] = _generate_cluster_name()

    # Booleans that resolved to None should be False
    for key in (
        "quick",
        "skip_build",
        "skip_load",
        "skip_cleanup",
        "reuse_cluster",
        "skip_preflight",
        "stream_logs",
    ):
        if resolved.get(key) is None:
            resolved[key] = False

    return K8sTestSettings(**resolved)


@asynccontextmanager
async def timed_operation(operation: str):
    """Context manager that logs timing information for an operation."""
    logger.info(f"[START] {operation}")
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"[DONE] {operation} ({elapsed:.2f}s)")


def _configure_rich_logging() -> None:
    """Attach a log handler to the ``tests.kubernetes`` logger hierarchy.

    When stderr is a TTY, uses Rich for colourised, structured output.
    Otherwise falls back to a plain ``StreamHandler`` so log lines render
    correctly in CI, piped output, and non-interactive terminals.
    """
    k8s_logger = logging.getLogger("tests.kubernetes")

    if k8s_logger.handlers:
        return

    if _is_tty():
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(
            level=logging.DEBUG,
            show_time=True,
            show_path=False,
            markup=False,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
        )
    else:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    k8s_logger.addHandler(handler)
    k8s_logger.setLevel(logging.DEBUG)
    k8s_logger.propagate = False


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest for kubernetes tests."""
    # Resolve and stash settings
    settings = _resolve_settings(config)
    config.stash[_SETTINGS_KEY] = settings

    # Skip local preflight when only GPU tests are being collected
    # (GPU tests run their own preflight in gpu/conftest.py).
    test_paths = [
        str(a) for a in config.invocation_params.args if Path(str(a)).exists()
    ]
    gpu_only = test_paths and all("kubernetes/gpu" in p for p in test_paths)
    if not gpu_only:
        _run_preflight_checks(settings)


def pytest_sessionstart(session: pytest.Session) -> None:
    """Set up Rich logging after pytest has configured its log handlers."""
    _configure_rich_logging()


def _run_preflight_checks(settings: K8sTestSettings) -> None:
    """Run preflight checks to verify all prerequisites are available.

    This runs at pytest collection time and will fail fast with helpful
    error messages if prerequisites are missing.
    """
    import json
    import shutil
    import subprocess

    from tests.kubernetes.helpers.preflight import PreflightChecker

    runtime = settings.cluster_runtime

    checker = PreflightChecker(
        title="KUBERNETES E2E TEST",
        skip_message=(
            "Use --k8s-skip-preflight or K8S_TEST_SKIP_PREFLIGHT=1 to skip "
            "(not recommended)."
        ),
    )
    checker.set_mode(runtime.value, cluster=settings.cluster)
    checker.start(total_steps=5)

    # Check for the runtime CLI (kind or minikube)
    if runtime is ClusterRuntime.KIND:
        c = checker.check("kind CLI")
        kind_path = shutil.which("kind")
        if kind_path:
            result = subprocess.run(
                ["kind", "version"],
                capture_output=True,
                text=True,
                check=False,
            )
            version = result.stdout.strip() if result.returncode == 0 else "unknown"
            c.pass_(f"{kind_path} ({version})")
        else:
            c.fail(
                "not found in PATH - install from https://kind.sigs.k8s.io/docs/user/quick-start/"
            )
    else:
        c = checker.check("minikube CLI")
        minikube_path = shutil.which("minikube")
        if minikube_path:
            result = subprocess.run(
                ["minikube", "version", "--short"],
                capture_output=True,
                text=True,
                check=False,
            )
            version = result.stdout.strip() if result.returncode == 0 else "unknown"
            c.pass_(f"{minikube_path} ({version})")
        else:
            c.fail(
                "not found in PATH - install from https://minikube.sigs.k8s.io/docs/start/"
            )

    c = checker.check("Docker daemon")
    result = subprocess.run(
        ["docker", "info"], capture_output=True, text=True, check=False
    )
    if result.returncode == 0:
        c.pass_("Docker daemon is responsive")
    else:
        c.fail("Docker daemon not running - start Docker first")

    c = checker.check("kubectl CLI")
    kubectl_path = shutil.which("kubectl")
    if kubectl_path:
        result = subprocess.run(
            ["kubectl", "version", "--client", "--output=yaml"],
            capture_output=True,
            text=True,
            check=False,
        )
        version = "unknown"
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "gitVersion:" in line:
                    version = line.split(":")[-1].strip()
                    break
        c.pass_(f"{kubectl_path} ({version})")
    else:
        c.fail("kubectl not found in PATH")

    c = checker.check(f"aiperf image ({settings.aiperf_image})")
    result = subprocess.run(
        ["docker", "image", "inspect", settings.aiperf_image],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        try:
            info = json.loads(result.stdout)
            size_mb = info[0].get("Size", 0) / (1024 * 1024)
            created = info[0].get("Created", "")
            # Check if image is stale (older than src/ directory)
            from datetime import datetime

            src_dir = Path(__file__).parent.parent.parent / "src"
            if src_dir.exists():
                newest_src = max(
                    f.stat().st_mtime
                    for f in src_dir.rglob("*.py")
                    if "__pycache__" not in str(f)
                )
                try:
                    image_time = datetime.fromisoformat(
                        created.replace("Z", "+00:00")
                    ).timestamp()
                    if newest_src > image_time:
                        c.pass_(
                            f"found ({size_mb:.1f} MB) but STALE — source files are newer than image. "
                            f"Image will be rebuilt (use --k8s-skip-build to skip)."
                        )
                    else:
                        c.pass_(f"found ({size_mb:.1f} MB, up to date)")
                except (ValueError, TypeError):
                    c.pass_(f"found ({size_mb:.1f} MB)")
            else:
                c.pass_(f"found ({size_mb:.1f} MB)")
        except (json.JSONDecodeError, IndexError, KeyError):
            c.pass_("found")
    else:
        c.fail(f"not found - run: docker build -t {settings.aiperf_image} .")

    c = checker.check(f"mock server image ({settings.mock_server_image})")
    result = subprocess.run(
        ["docker", "image", "inspect", settings.mock_server_image],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        try:
            info = json.loads(result.stdout)
            size_mb = info[0].get("Size", 0) / (1024 * 1024)
            c.pass_(f"found ({size_mb:.1f} MB)")
        except (json.JSONDecodeError, IndexError, KeyError):
            c.pass_("found")
    else:
        c.fail(
            f"not found - run: docker build -f dev/deploy/Dockerfile.mock-server "
            f"-t {settings.mock_server_image} ."
        )

    checker.finish(error_cls=pytest.UsageError if not settings.skip_preflight else None)


_TEST_START_TIMES: dict[str, float] = {}
_test_console: Console | None = None
_test_total: int = 0
_test_completed: int = 0


def _get_test_console() -> Console | None:
    """Lazily create a shared Rich Console for test banners.

    Returns ``None`` when stderr is not a TTY (plain mode).
    """
    global _test_console  # noqa: PLW0603
    if not _is_tty():
        return None
    if _test_console is None:
        from rich.console import Console

        _test_console = Console(stderr=True)
    return _test_console


def pytest_collection_finish(session: pytest.Session) -> None:
    """Capture total test count after collection."""
    global _test_total  # noqa: PLW0603
    _test_total = len(session.items)


def pytest_runtest_logstart(nodeid: str, location: tuple) -> None:
    """Log a banner when a test starts."""
    _TEST_START_TIMES[nodeid] = time.perf_counter()
    short = nodeid.split("::")[-1]
    console = _get_test_console()
    if console is not None:
        from rich.rule import Rule

        console.print()
        console.print(
            Rule(f"[bold cyan]START[/]  [bold white]{short}[/]", style="cyan")
        )
        console.print(f"  [dim]{nodeid}[/]")
    else:
        print(f"\n{'=' * 60}", file=sys.stderr)
        print(f"START  {short}", file=sys.stderr)
        print(f"  {nodeid}", file=sys.stderr)


def pytest_runtest_logfinish(nodeid: str, location: tuple) -> None:
    """Log a banner when a test finishes."""
    global _test_completed  # noqa: PLW0603
    _test_completed += 1
    elapsed = time.perf_counter() - _TEST_START_TIMES.pop(nodeid, 0)
    short = nodeid.split("::")[-1]
    pct = (_test_completed / _test_total * 100) if _test_total else 0
    progress = f"[{_test_completed}/{_test_total}] {pct:.0f}%"
    console = _get_test_console()
    if console is not None:
        from rich.rule import Rule

        console.print()
        console.print(
            Rule(
                f"[bold green]DONE[/]  [bold white]{short}[/]  "
                f"[bold yellow]{elapsed:.2f}s[/]  [bold magenta]{progress}[/]",
                style="green",
            )
        )
        console.print()
    else:
        print(f"\n{'=' * 60}", file=sys.stderr)
        print(f"DONE  {short}  {elapsed:.2f}s  {progress}", file=sys.stderr)
        print(f"{'=' * 60}\n", file=sys.stderr)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Add k8s marker to all tests in this directory."""
    for item in items:
        if "kubernetes" in str(item.fspath):
            item.add_marker(pytest.mark.k8s)


@pytest.fixture(scope="package")
def k8s_settings(request: pytest.FixtureRequest) -> K8sTestSettings:
    """Resolved K8s test settings (package-scoped)."""
    return _get_settings(request.config)


@pytest.fixture(scope="package")
def project_root() -> Path:
    """Get the project root directory."""
    # Navigate up from tests/kubernetes/conftest.py
    return Path(__file__).parent.parent.parent


@pytest.fixture(scope="package")
def cluster_config(k8s_settings: K8sTestSettings) -> ClusterConfig:
    """Create cluster configuration."""
    return ClusterConfig(
        name=k8s_settings.cluster,
        runtime=k8s_settings.cluster_runtime,
        wait_timeout=120,
    )


@pytest_asyncio.fixture(scope="package", loop_scope="package")
async def local_cluster(
    cluster_config: ClusterConfig,
    k8s_settings: K8sTestSettings,
) -> AsyncGenerator[LocalCluster, None]:
    """Create and manage a local K8s cluster for the test package.

    Uses Kind by default (faster) or Minikube via --k8s-runtime=minikube.
    Package-scoped so the cluster is shared across all tests in the package.
    Use --k8s-reuse-cluster to keep an existing cluster.
    """
    s = k8s_settings
    cluster = LocalCluster(config=cluster_config)

    # Create cluster (or reuse existing)
    if s.reuse_cluster and await cluster.exists():
        logger.info(f"Reusing existing cluster: {cluster.name}")
    else:
        await cluster.create(force=not s.reuse_cluster)

    yield cluster

    # Cleanup
    if not s.skip_cleanup and not s.reuse_cluster:
        await cluster.delete()
    else:
        logger.info(f"Keeping cluster: {cluster.name}")


@pytest.fixture(scope="package")
def kubectl(local_cluster: LocalCluster) -> KubectlClient:
    """Create kubectl client configured for the local cluster."""
    return KubectlClient(context=local_cluster.context)


@pytest.fixture(scope="package")
def image_manager(project_root: Path) -> ImageManager:
    """Create image manager with default images."""
    return ImageManager(
        project_root=project_root,
        images={
            "aiperf": ImageConfig(
                name="aiperf",
                tag="local",
                dockerfile=None,
                build_context=project_root,
            ),
            "mock-server": ImageConfig(
                name="aiperf-mock-server",
                tag="latest",
                dockerfile="dev/deploy/Dockerfile.mock-server",
                build_context=project_root,
            ),
        },
    )


@pytest_asyncio.fixture(scope="package", loop_scope="package")
async def built_images(
    image_manager: ImageManager,
    k8s_settings: K8sTestSettings,
) -> ImageManager:
    """Build all required Docker images.

    Always rebuilds to ensure the image matches the current source code.
    Use --k8s-skip-build to skip building (use existing images).
    """
    if k8s_settings.skip_build:
        missing = [
            key
            for key in image_manager.images
            if not await image_manager.image_exists(key)
        ]
        if missing:
            names = ", ".join(image_manager.get_image_name(k) for k in missing)
            logger.info(
                f"--k8s-skip-build set but missing images: {names}; building them"
            )
            for key in missing:
                await image_manager.build(key)
        else:
            logger.info("Skipping image build (--k8s-skip-build, all images exist)")
    else:
        await image_manager.build_all(force=True)

    return image_manager


@pytest_asyncio.fixture(scope="package", loop_scope="package")
async def loaded_images(
    local_cluster: LocalCluster,
    built_images: ImageManager,
    k8s_settings: K8sTestSettings,
) -> ImageManager:
    """Load all images into the local cluster.

    Use --k8s-skip-load to skip loading (images already in cluster).
    Falls back to loading any images not found in the cluster.
    """
    if k8s_settings.skip_load:
        missing = [
            key
            for key in built_images.images
            if not await local_cluster.image_loaded(built_images.get_image_name(key))
        ]
        if missing:
            names = ", ".join(built_images.get_image_name(k) for k in missing)
            logger.info(
                f"--k8s-skip-load set but missing in cluster: {names}; loading them"
            )
            for key in missing:
                await local_cluster.load_image(built_images.get_image_name(key))
        else:
            logger.info("Skipping image load (--k8s-skip-load, all images in cluster)")
    else:
        for image_key in built_images.images:
            await local_cluster.load_image(built_images.get_image_name(image_key))

    return built_images


@pytest_asyncio.fixture(scope="package", loop_scope="package")
async def jobset_controller(
    kubectl: KubectlClient,
    k8s_settings: K8sTestSettings,
) -> None:
    """Install the JobSet controller.

    This fixture is package-scoped and installs JobSet once per test package.
    """
    version = k8s_settings.jobset_version
    async with timed_operation(f"Installing JobSet controller {version}"):
        url = JOBSET_CRD_URL_TEMPLATE.format(version=version)
        await kubectl.apply_server_side(url)

        # Wait for controller to be ready
        await kubectl.wait_for_condition(
            "deployment",
            "jobset-controller-manager",
            "available",
            namespace="jobset-system",
            timeout=60,
        )


@pytest_asyncio.fixture(scope="package", loop_scope="package")
async def mock_server(
    kubectl: KubectlClient,
    loaded_images: ImageManager,
    project_root: Path,
    k8s_settings: K8sTestSettings,
) -> AsyncGenerator[None, None]:
    """Deploy the mock LLM server.

    This fixture is package-scoped and deploys the mock server once per package.
    """
    async with timed_operation("Deploying mock LLM server"):
        # Read and apply mock server manifest
        manifest_path = project_root / "dev" / "deploy" / "mock-server.yaml"
        with open(manifest_path) as f:
            manifest = f.read()

        await kubectl.apply(manifest)

        # Wait for rollout
        success = await kubectl.wait_for_rollout(
            "deployment",
            "aiperf-mock-server",
            namespace="default",
            timeout=120,
        )

        if not success:
            raise RuntimeError("Mock server deployment failed")

    yield

    # Cleanup mock server
    if not k8s_settings.skip_cleanup:
        async with timed_operation("Cleaning up mock server"):
            await kubectl.delete(
                "deployment", "aiperf-mock-server", namespace="default"
            )
            await kubectl.delete("service", "aiperf-mock-server", namespace="default")


@pytest.fixture(scope="package")
def k8s_ready(
    local_cluster: LocalCluster,
    loaded_images: ImageManager,
    jobset_controller: None,
    mock_server: None,
    operator_ready: None,
) -> LocalCluster:
    """Ensure all Kubernetes infrastructure is ready.

    This fixture combines all package-level setup:
    - Local cluster created (Kind or Minikube)
    - Images built and loaded
    - JobSet controller installed
    - Mock server deployed
    - AIPerf operator deployed (CRD + controller)

    Use this fixture when you need everything ready for benchmark tests.
    """
    return local_cluster


@pytest_asyncio.fixture(scope="package", loop_scope="package")
async def benchmark_deployer(
    kubectl: KubectlClient,
    project_root: Path,
    k8s_ready: LocalCluster,
    k8s_settings: K8sTestSettings,
) -> AsyncGenerator[BenchmarkDeployer, None]:
    """Create a benchmark deployer.

    This fixture is package-scoped and manages benchmark lifecycle.
    """
    deployer = BenchmarkDeployer(
        kubectl=kubectl,
        project_root=project_root,
        default_image=k8s_settings.aiperf_image,
    )

    yield deployer

    # Cleanup all deployments
    if not k8s_settings.skip_cleanup:
        await deployer.cleanup_all()


# ============================================================================
# Function-scoped fixtures for individual tests
# ============================================================================


@pytest.fixture
def benchmark_config(k8s_settings: K8sTestSettings) -> BenchmarkConfig:
    """Create a default benchmark configuration."""
    return BenchmarkConfig(
        concurrency=5,
        request_count=50,
        warmup_request_count=5,
        image=k8s_settings.aiperf_image,
    )


@pytest.fixture
def small_benchmark_config(k8s_settings: K8sTestSettings) -> BenchmarkConfig:
    """Create a small/fast benchmark configuration for quick tests."""
    return BenchmarkConfig(
        concurrency=2,
        request_count=10,
        warmup_request_count=2,
        image=k8s_settings.aiperf_image,
    )


@pytest.fixture
def large_benchmark_config(k8s_settings: K8sTestSettings) -> BenchmarkConfig:
    """Create a larger benchmark configuration for stress tests."""
    return BenchmarkConfig(
        concurrency=10,
        request_count=200,
        warmup_request_count=10,
        image=k8s_settings.aiperf_image,
    )


@pytest_asyncio.fixture
async def deployed_benchmark(
    benchmark_deployer: BenchmarkDeployer,
    benchmark_config: BenchmarkConfig,
    k8s_settings: K8sTestSettings,
) -> AsyncGenerator[BenchmarkResult, None]:
    """Deploy a benchmark and wait for completion (function-scoped).

    Use this when you need a fresh benchmark for each test.
    For read-only tests, use deployed_benchmark_module instead.
    """
    result = await benchmark_deployer.deploy(
        config=benchmark_config,
        wait_for_completion=True,
        timeout=k8s_settings.benchmark_timeout,
        stream_logs=k8s_settings.stream_logs,
    )

    yield result

    # Cleanup is handled by benchmark_deployer fixture


@pytest_asyncio.fixture
async def deployed_small_benchmark(
    benchmark_deployer: BenchmarkDeployer,
    small_benchmark_config: BenchmarkConfig,
    k8s_settings: K8sTestSettings,
) -> AsyncGenerator[BenchmarkResult, None]:
    """Deploy a small benchmark for quick tests (function-scoped).

    Use this when you need a fresh benchmark for each test.
    For read-only tests, use deployed_small_benchmark_module instead.
    """
    result = await benchmark_deployer.deploy(
        config=small_benchmark_config,
        wait_for_completion=True,
        timeout=k8s_settings.benchmark_timeout,
        stream_logs=k8s_settings.stream_logs,
    )

    yield result


# Module-scoped benchmark fixtures for faster test runs
# Use these when tests only READ from the benchmark result


@pytest.fixture(scope="module")
def small_benchmark_config_module(k8s_settings: K8sTestSettings) -> BenchmarkConfig:
    """Module-scoped small benchmark configuration."""
    return BenchmarkConfig(
        concurrency=2,
        request_count=10,
        warmup_request_count=2,
        image=k8s_settings.aiperf_image,
    )


@pytest_asyncio.fixture(scope="module", loop_scope="package")
async def deployed_small_benchmark_module(
    benchmark_deployer: BenchmarkDeployer,
    small_benchmark_config_module: BenchmarkConfig,
    k8s_settings: K8sTestSettings,
) -> AsyncGenerator[BenchmarkResult, None]:
    """Deploy a small benchmark shared across tests in a module.

    This is much faster for tests that only need to read benchmark results.
    The benchmark is deployed once per test module and reused.
    """
    result = await benchmark_deployer.deploy(
        config=small_benchmark_config_module,
        wait_for_completion=True,
        timeout=k8s_settings.benchmark_timeout,
        stream_logs=k8s_settings.stream_logs,
    )

    yield result


# ============================================================================
# Utility fixtures
# ============================================================================


@pytest.fixture
def assert_metrics():
    """Factory fixture for asserting benchmark metrics."""

    def _assert_metrics(
        result: BenchmarkResult,
        min_throughput: float | None = None,
        max_latency: float | None = None,
        expected_request_count: int | None = None,
        max_error_count: int = 0,
    ) -> None:
        """Assert benchmark metrics meet expectations.

        Args:
            result: Benchmark result to check.
            min_throughput: Minimum request throughput (req/s).
            max_latency: Maximum average latency (ms).
            expected_request_count: Expected number of completed requests.
            max_error_count: Maximum allowed error count.
        """
        assert result.success, f"Benchmark failed: {result.error_message}"
        assert result.metrics is not None, "No metrics collected"

        if min_throughput is not None:
            assert result.metrics.request_throughput is not None
            assert result.metrics.request_throughput >= min_throughput, (
                f"Throughput {result.metrics.request_throughput} < {min_throughput}"
            )

        if max_latency is not None:
            assert result.metrics.request_latency_avg is not None
            assert result.metrics.request_latency_avg <= max_latency, (
                f"Latency {result.metrics.request_latency_avg} > {max_latency}"
            )

        if expected_request_count is not None:
            assert result.metrics.request_count == expected_request_count, (
                f"Request count {result.metrics.request_count} != {expected_request_count}"
            )

        assert result.metrics.error_count <= max_error_count, (
            f"Error count {result.metrics.error_count} > {max_error_count}"
        )

    return _assert_metrics


@pytest.fixture
def get_pod_logs(kubectl: KubectlClient):
    """Factory fixture for getting pod logs."""

    async def _get_logs(
        result: BenchmarkResult,
        container: str = "control-plane",
        tail: int | None = 100,
    ) -> str:
        """Get logs from a benchmark pod.

        Args:
            result: Benchmark result.
            container: Container name (default: control-plane for new single-container arch).
            tail: Number of lines to tail.

        Returns:
            Log content.
        """
        controller_pod = result.controller_pod
        if not controller_pod:
            return ""

        return await kubectl.get_logs(
            controller_pod.name,
            container=container,
            namespace=result.namespace,
            tail=tail,
        )

    return _get_logs


# ============================================================================
# Operator-specific fixtures
# ============================================================================


@pytest_asyncio.fixture(scope="package", loop_scope="package")
async def operator_deployer(
    kubectl: KubectlClient,
    project_root: Path,
    loaded_images: ImageManager,
    jobset_controller: None,
    k8s_settings: K8sTestSettings,
) -> AsyncGenerator[OperatorDeployer, None]:
    """Create an operator deployer (package-scoped).

    This fixture sets up the CRD but does NOT deploy the operator automatically.
    Use operator_ready fixture for tests that need the operator running.
    """
    deployer = OperatorDeployer(
        kubectl=kubectl,
        project_root=project_root,
        operator_image=k8s_settings.aiperf_image,
    )

    # Install CRD (always needed)
    async with timed_operation("Installing AIPerfJob CRD"):
        await deployer.install_crd()

    yield deployer

    # Cleanup all jobs
    await deployer.cleanup_all()


@pytest_asyncio.fixture(scope="package", loop_scope="package")
async def operator_ready(
    operator_deployer: OperatorDeployer,
    mock_server: None,
    k8s_settings: K8sTestSettings,
) -> AsyncGenerator[OperatorDeployer, None]:
    """Ensure operator is deployed and ready (package-scoped).

    Use this fixture for tests that need the operator running.
    """
    async with timed_operation("Deploying AIPerf operator"):
        await operator_deployer.deploy_operator()

    yield operator_deployer

    # Cleanup operator
    if not k8s_settings.skip_cleanup:
        async with timed_operation("Uninstalling operator"):
            await operator_deployer.uninstall_operator()


@pytest.fixture
def small_operator_config(k8s_settings: K8sTestSettings) -> AIPerfJobConfig:
    """Small configuration for quick operator tests."""
    return AIPerfJobConfig(
        concurrency=2,
        request_count=10,
        warmup_request_count=2,
        image=k8s_settings.aiperf_image,
    )


@pytest.fixture
def large_operator_config(k8s_settings: K8sTestSettings) -> AIPerfJobConfig:
    """Larger configuration for cancellation and scaling tests."""
    return AIPerfJobConfig(
        concurrency=10,
        request_count=100,
        warmup_request_count=10,
        image=k8s_settings.aiperf_image,
    )


@pytest_asyncio.fixture
async def operator_deployed_job(
    operator_ready: OperatorDeployer,
    small_operator_config: AIPerfJobConfig,
    k8s_settings: K8sTestSettings,
) -> AsyncGenerator[OperatorJobResult, None]:
    """Deploy a job through the operator and wait for completion (function-scoped).

    Use this for tests that need a completed job.
    """
    result = await operator_ready.run_job(
        config=small_operator_config,
        timeout=k8s_settings.benchmark_timeout,
        stream_logs=k8s_settings.stream_logs,
    )

    yield result

    # Cleanup is handled by operator_deployer fixture


@pytest.fixture(scope="module")
def small_operator_config_module(k8s_settings: K8sTestSettings) -> AIPerfJobConfig:
    """Module-scoped small operator configuration."""
    return AIPerfJobConfig(
        concurrency=2,
        request_count=10,
        warmup_request_count=2,
        image=k8s_settings.aiperf_image,
    )


@pytest_asyncio.fixture(scope="module", loop_scope="package")
async def operator_deployed_job_module(
    operator_ready: OperatorDeployer,
    small_operator_config_module: AIPerfJobConfig,
    k8s_settings: K8sTestSettings,
) -> AsyncGenerator[OperatorJobResult, None]:
    """Deploy a job through the operator (module-scoped for speed).

    Use this for read-only tests that need a completed job.
    """
    result = await operator_ready.run_job(
        config=small_operator_config_module,
        timeout=k8s_settings.benchmark_timeout,
        stream_logs=k8s_settings.stream_logs,
    )

    yield result


# ============================================================================
# Helm-specific fixtures
# ============================================================================


@pytest.fixture(scope="module")
def helm_client(local_cluster: LocalCluster) -> HelmClient:
    """Create Helm client configured for the local cluster."""
    from tests.kubernetes.helpers.helm import HelmClient

    return HelmClient(kubecontext=local_cluster.context)


@pytest.fixture(scope="module")
def helm_values(k8s_settings: K8sTestSettings) -> HelmValues:
    """Default Helm values for testing."""
    from tests.kubernetes.helpers.helm import HelmValues

    return HelmValues(
        image_repository="aiperf",
        image_tag="local",
        image_pull_policy="Never",
        default_image=k8s_settings.aiperf_image,
        default_image_pull_policy="Never",
        storage_enabled=False,
    )


@pytest_asyncio.fixture(scope="module", loop_scope="package")
async def helm_deployer(
    kubectl: KubectlClient,
    helm_client: HelmClient,
    project_root: Path,
    loaded_images: ImageManager,
    jobset_controller: None,
    helm_values: HelmValues,
) -> AsyncGenerator[HelmDeployer, None]:
    """Create a Helm deployer (module-scoped).

    This fixture creates the HelmDeployer but does NOT install the chart automatically.
    Use helm_deployed fixture for tests that need the operator running.
    """
    from tests.kubernetes.helpers.helm import HelmDeployer

    deployer = HelmDeployer(
        kubectl=kubectl,
        helm=helm_client,
        project_root=project_root,
        values=helm_values,
    )

    yield deployer

    # Cleanup all jobs
    await deployer.cleanup_all()


@pytest_asyncio.fixture(scope="module", loop_scope="package")
async def helm_deployed(
    helm_deployer: HelmDeployer,
    mock_server: None,
    k8s_settings: K8sTestSettings,
) -> AsyncGenerator[HelmDeployer, None]:
    """Ensure operator is deployed via Helm and ready (module-scoped).

    Use this fixture for tests that need the Helm-deployed operator running.
    """
    async with timed_operation("Installing AIPerf operator via Helm"):
        await helm_deployer.install_chart()

    yield helm_deployer

    # Only clean up deployed jobs, NOT the Helm release.
    # The operator must persist across modules for tests that share
    # the package-scoped cluster.
    if not k8s_settings.skip_cleanup:
        try:
            await asyncio.wait_for(helm_deployer.cleanup_all(), timeout=120)
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"Job cleanup timed out: {e}")


@pytest.fixture
def small_helm_config(k8s_settings: K8sTestSettings) -> AIPerfJobConfig:
    """Small configuration for quick Helm tests."""
    return AIPerfJobConfig(
        concurrency=2,
        request_count=10,
        warmup_request_count=2,
        image=k8s_settings.aiperf_image,
    )


@pytest.fixture(scope="module")
def small_helm_config_module(k8s_settings: K8sTestSettings) -> AIPerfJobConfig:
    """Module-scoped small Helm configuration."""
    return AIPerfJobConfig(
        concurrency=2,
        request_count=10,
        warmup_request_count=2,
        image=k8s_settings.aiperf_image,
    )


@pytest_asyncio.fixture(scope="module", loop_scope="package")
async def helm_deployed_job_module(
    helm_deployed: HelmDeployer,
    small_helm_config_module: AIPerfJobConfig,
    k8s_settings: K8sTestSettings,
) -> AsyncGenerator[OperatorJobResult, None]:
    """Deploy a job through Helm-deployed operator (module-scoped for speed).

    Use this for read-only tests that need a completed job.
    """
    result = await helm_deployed.run_job(
        config=small_helm_config_module,
        timeout=k8s_settings.benchmark_timeout,
        stream_logs=k8s_settings.stream_logs,
    )

    yield result
