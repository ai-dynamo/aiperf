# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local Kubernetes cluster management for E2E tests.

Supports pluggable backends (Kind, Minikube) behind a unified ``LocalCluster``
facade.  Select the backend via ``ClusterConfig.runtime``.

GPU support differs by backend:

- **Minikube** uses native ``--gpus all`` passthrough (handles device plugin
  automatically) and only needs the nvidia RuntimeClass applied afterwards.
- **Kind** uses the nvidia-container-runtime sentinel mount to inject GPU
  devices into the node container, then runs ``KindGpuSetup`` to install
  the full NVIDIA stack (toolkit, CDI, RuntimeClass, device plugin) inside
  the node.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol

from aiperf.common.aiperf_logger import AIPerfLogger
from dev.versions import DEVICE_PLUGIN_VERSION

logger = AIPerfLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEPLOY_DIR = _PROJECT_ROOT / "dev" / "deploy"


def _strip_localhost_proxy() -> None:
    """Remove localhost-bound proxy env vars that break Kind containers.

    Kind propagates HTTP_PROXY/HTTPS_PROXY into the node. If the proxy
    is on 127.0.0.1 (e.g. corporate proxies), it's unreachable from
    inside the container and breaks image pulls, apt-get, and curl.
    """
    import os

    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        val = os.environ.get(var, "")
        if "127.0.0.1" in val or "localhost" in val:
            logger.info(f"Stripping {var}={val} (unreachable from Kind node)")
            os.environ.pop(var, None)


# ============================================================================
# Shared helpers
# ============================================================================


async def _run_streaming(
    cmd: list[str],
    prefix: str,
    error_msg: str,
) -> None:
    """Run a subprocess, streaming its combined output to the logger."""
    logger.info(f"Running: {' '.join(cmd)}")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    output_lines: list[str] = []
    assert proc.stdout is not None
    async for raw_line in proc.stdout:
        line = raw_line.decode().rstrip()
        output_lines.append(line)
        logger.info(f"[{prefix}] {line}")

    await proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"{error_msg}:\n" + "\n".join(output_lines))


async def _run_quiet(cmd: list[str]) -> tuple[int, str, str]:
    """Run a subprocess and capture output without streaming."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode, stdout.decode(), stderr.decode()


async def _kubectl(
    context: str, *args: str, check: bool = True
) -> tuple[int, str, str]:
    """Run kubectl with ``--context`` and return (rc, stdout, stderr)."""
    cmd = ["kubectl", "--context", context, *args]
    rc, stdout, stderr = await _run_quiet(cmd)
    if check and rc != 0:
        raise RuntimeError(f"kubectl {' '.join(args)} failed: {stderr}")
    return rc, stdout, stderr


async def _kubectl_apply_stdin(context: str, manifest: str) -> None:
    """Apply a YAML manifest via kubectl stdin."""
    proc = await asyncio.create_subprocess_exec(
        "kubectl",
        "apply",
        "--context",
        context,
        "-f",
        "-",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate(input=manifest.encode())
    if proc.returncode != 0:
        raise RuntimeError(f"kubectl apply failed: {stderr.decode()}")


async def _create_nvidia_runtime_class(context: str) -> None:
    """Apply the nvidia RuntimeClass to the cluster."""
    manifest = (_DEPLOY_DIR / "nvidia-runtime-class.yaml").read_text()
    await _kubectl_apply_stdin(context, manifest)
    logger.info("Created nvidia RuntimeClass")


async def _docker_exec(
    container: str, *args: str, check: bool = True
) -> tuple[int, str, str]:
    """Run a command inside a Docker container."""
    rc, stdout, stderr = await _run_quiet(["docker", "exec", container, *args])
    if check and rc != 0:
        raise RuntimeError(
            f"docker exec {container} {' '.join(args)} failed:\n{stdout}\n{stderr}"
        )
    return rc, stdout, stderr


@asynccontextmanager
async def timed_operation(operation: str) -> AsyncIterator[None]:
    """Context manager that logs timing information for an operation."""
    logger.info(f"[START] {operation}")
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"[DONE] {operation} ({elapsed:.2f}s)")


# ============================================================================
# Runtime enum and config
# ============================================================================


class ClusterRuntime(str, Enum):
    """Supported local Kubernetes cluster runtimes."""

    KIND = "kind"
    MINIKUBE = "minikube"


@dataclass
class ClusterConfig:
    """Configuration for a local Kubernetes cluster."""

    name: str = "aiperf-pytest"
    """Cluster name used for identification and context."""

    runtime: ClusterRuntime = ClusterRuntime.KIND
    """Kubernetes cluster runtime backend."""

    kubeconfig: Path | None = None
    """Path to kubeconfig file, or None for default."""

    wait_timeout: int = 120
    """Seconds to wait for cluster readiness."""

    node_image: str | None = None
    """Custom node image for the cluster, or None for default."""

    gpus: bool = False
    """Whether to enable GPU passthrough."""

    cache_images: list[str] = field(default_factory=list)
    """Images to cache in the cluster for offline use."""

    device_plugin_version: str = DEVICE_PLUGIN_VERSION
    """NVIDIA device plugin DaemonSet version."""


# ============================================================================
# Kind GPU setup
# ============================================================================


class KindGpuSetup:
    """Installs NVIDIA GPU infrastructure inside a Kind node container.

    Pipeline:
    1. Install nvidia-container-toolkit packages inside the node
    2. Configure containerd for CDI and generate the CDI spec
    3. Wait for the Kubernetes node to become Ready
    4. Create the nvidia RuntimeClass
    5. Deploy the NVIDIA device plugin DaemonSet
    """

    def __init__(
        self,
        context: str,
        node_container: str,
        device_plugin_version: str = DEVICE_PLUGIN_VERSION,
    ) -> None:
        self._context = context
        self._node = node_container
        self._dpv = device_plugin_version

    async def _exec(self, *args: str, check: bool = True) -> tuple[int, str, str]:
        """Run a command inside the Kind node container."""
        return await _docker_exec(self._node, *args, check=check)

    async def setup(self) -> None:
        """Run the complete GPU setup pipeline."""
        async with timed_operation("Kind GPU setup"):
            await self._install_toolkit()
            await self._configure_containerd()
            await self._wait_for_node_ready()
            await _create_nvidia_runtime_class(self._context)
            await self._install_device_plugin()

    async def ensure_ready(self) -> None:
        """Verify GPU infrastructure and re-install missing components.

        Idempotent: safe to call on both fresh and already-setup clusters.
        Checks RuntimeClass, device plugin, and GPU allocatable status,
        re-applying anything that is missing.
        """
        async with timed_operation("Ensuring GPU infrastructure ready"):
            # 1. RuntimeClass
            rc, _, _ = await _kubectl(
                self._context, "get", "runtimeclass", "nvidia", check=False
            )
            if rc != 0:
                logger.info("RuntimeClass 'nvidia' missing - creating")
                await _create_nvidia_runtime_class(self._context)
            else:
                logger.info("RuntimeClass 'nvidia' present")

            # 2. Device plugin DaemonSet
            rc, _, _ = await _kubectl(
                self._context,
                "get",
                "daemonset",
                "nvidia-device-plugin-daemonset",
                "-n",
                "kube-system",
                check=False,
            )
            if rc != 0:
                logger.info("NVIDIA device plugin missing - installing")
                await self._install_device_plugin()
            else:
                # Verify it's actually running
                rc2, stdout, _ = await _kubectl(
                    self._context,
                    "-n",
                    "kube-system",
                    "get",
                    "daemonset",
                    "nvidia-device-plugin-daemonset",
                    "-o",
                    "jsonpath={.status.numberReady}",
                    check=False,
                )
                ready = 0
                with contextlib.suppress(ValueError):
                    ready = int(stdout.strip()) if rc2 == 0 and stdout.strip() else 0

                if ready == 0:
                    logger.info(
                        "NVIDIA device plugin exists but not ready - reapplying"
                    )
                    await self._install_device_plugin()
                else:
                    logger.info(f"NVIDIA device plugin running ({ready} ready)")

            # 3. Verify GPU allocatable on node
            rc, stdout, _ = await _kubectl(
                self._context,
                "get",
                "nodes",
                "-o",
                r"jsonpath={.items[0].status.allocatable.nvidia\.com/gpu}",
                check=False,
            )
            gpu_count = 0
            with contextlib.suppress(ValueError):
                gpu_count = int(stdout.strip()) if rc == 0 and stdout.strip() else 0

            if gpu_count > 0:
                logger.info(f"GPU allocatable: {gpu_count}")
            else:
                logger.warning(
                    "No GPU allocatable yet - waiting for device plugin to register"
                )
                for i in range(60):
                    rc, stdout, _ = await _kubectl(
                        self._context,
                        "get",
                        "nodes",
                        "-o",
                        r"jsonpath={.items[0].status.allocatable.nvidia\.com/gpu}",
                        check=False,
                    )
                    try:
                        if rc == 0 and int(stdout.strip()) > 0:
                            logger.info(f"GPU allocatable: {stdout.strip()}")
                            return
                    except (ValueError, AttributeError):
                        pass
                    if (i + 1) % 10 == 0:
                        logger.info(f"Waiting for nvidia.com/gpu... ({i + 1}/60)")
                    await asyncio.sleep(2)

                raise RuntimeError(
                    "nvidia.com/gpu not found in node allocatable after 120s"
                )

    # -- private steps -------------------------------------------------------

    async def _install_toolkit(self) -> None:
        """Install nvidia-container-toolkit inside the Kind node."""
        async with timed_operation("Installing nvidia-container-toolkit in node"):
            await self._exec(
                "bash",
                "-c",
                "umount -R /proc/driver/nvidia 2>/dev/null || true",
                check=False,
            )
            # Unset proxy vars: Kind inherits host proxies pointing at
            # 127.0.0.1 which is unreachable inside the node container.
            install_cmd = (
                "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY; "
                "apt-get update -qq && "
                "apt-get install -y -qq gpg curl && "
                "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | "
                "gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null && "
                "curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | "
                "sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | "
                "tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null && "
                "apt-get update -qq && "
                "apt-get install -y -qq nvidia-container-toolkit"
            )
            rc, stdout, stderr = await self._exec(
                "bash", "-c", install_cmd, check=False
            )
            if rc != 0:
                raise RuntimeError(
                    f"Failed to install nvidia-container-toolkit in {self._node}:\n"
                    f"{stdout}\n{stderr}"
                )
            logger.info("nvidia-container-toolkit installed in node")

    async def _configure_containerd(self) -> None:
        """Configure containerd for CDI, generate CDI spec, and restart."""
        async with timed_operation("Configuring containerd for NVIDIA CDI"):
            await self._exec(
                "nvidia-ctk",
                "config",
                "--set",
                "nvidia-container-runtime.modes.cdi.annotation-prefixes=nvidia.cdi.k8s.io/",
            )
            await self._exec(
                "nvidia-ctk",
                "runtime",
                "configure",
                "--runtime=containerd",
                "--cdi.enabled",
                "--config-source=command",
            )
            await self._exec(
                "bash",
                "-c",
                "mkdir -p /var/run/cdi && "
                "nvidia-ctk cdi generate --driver-root=/ --output=/var/run/cdi/nvidia.yaml",
            )
            logger.info("CDI spec generated, restarting containerd")

            await self._exec("bash", "-c", "systemctl restart containerd")
            for _ in range(30):
                rc, _, _ = await self._exec("crictl", "info", check=False)
                if rc == 0:
                    break
                await asyncio.sleep(1)
            else:
                raise RuntimeError("containerd failed to restart in node")
            logger.info("containerd restarted with CDI support")

    async def _wait_for_node_ready(self) -> None:
        """Wait for the Kubernetes node to become Ready."""
        async with timed_operation("Waiting for node to become Ready"):
            for i in range(60):
                rc, stdout, _ = await _kubectl(
                    self._context, "get", "nodes", "--no-headers", check=False
                )
                if rc == 0 and stdout:
                    parts = stdout.strip().split()
                    if len(parts) >= 2 and parts[1] == "Ready":
                        return
                if (i + 1) % 10 == 0:
                    logger.info(f"Still waiting for node... ({i + 1}/60)")
                await asyncio.sleep(2)
            raise RuntimeError("Node did not become Ready within 120s")

    async def _install_device_plugin(self) -> None:
        """Deploy the NVIDIA device plugin DaemonSet and wait for GPU allocatable."""
        async with timed_operation("Installing NVIDIA device plugin"):
            await _kubectl(
                self._context,
                "label",
                "node",
                self._node,
                "--overwrite",
                "nvidia.com/gpu.present=true",
            )
            manifest = (
                (_DEPLOY_DIR / "nvidia-device-plugin.yaml.tmpl")
                .read_text()
                .replace(
                    "${DEVICE_PLUGIN_VERSION}",
                    self._dpv,
                )
            )
            await _kubectl_apply_stdin(self._context, manifest)

            await _kubectl(
                self._context,
                "-n",
                "kube-system",
                "rollout",
                "status",
                "daemonset/nvidia-device-plugin-daemonset",
                "--timeout=120s",
            )
            logger.info("Device plugin daemonset ready")

            await asyncio.sleep(5)
            for i in range(60):
                rc, stdout, _ = await _kubectl(
                    self._context,
                    "get",
                    "nodes",
                    "-o",
                    r"jsonpath={.items[0].status.allocatable.nvidia\.com/gpu}",
                    check=False,
                )
                try:
                    if rc == 0 and int(stdout.strip()) > 0:
                        logger.info(f"nvidia.com/gpu: {stdout.strip()}")
                        return
                except (ValueError, AttributeError):
                    pass
                if (i + 1) % 10 == 0:
                    logger.info(f"Waiting for nvidia.com/gpu... ({i + 1}/60)")
                await asyncio.sleep(2)

            raise RuntimeError(
                "nvidia.com/gpu not found in node allocatable after 120s"
            )


# ============================================================================
# Backend protocol
# ============================================================================


class ClusterBackend(Protocol):
    """Interface that each cluster runtime must implement."""

    @property
    def context_name(self) -> str:
        """Return the kubectl context name for this cluster."""
        ...

    async def exists(self) -> bool:
        """Check if the cluster exists and is running."""
        ...

    async def create(self) -> None:
        """Create the cluster (including GPU setup if configured)."""
        ...

    async def delete(self) -> None:
        """Delete the cluster."""
        ...

    async def image_loaded(self, image: str) -> bool:
        """Check if a Docker image is available inside the cluster."""
        ...

    async def load_image(self, image: str) -> None:
        """Load a Docker image into the cluster."""
        ...


# ============================================================================
# Kind backend
# ============================================================================

_KIND_GPU_CONFIG_PATH = _DEPLOY_DIR / "kind-gpu-cluster.yaml"


class KindBackend:
    """Kind (Kubernetes IN Docker) cluster backend.

    GPU support: creates the cluster with the nvidia-container-runtime
    sentinel mount, then delegates to ``KindGpuSetup`` for the full
    NVIDIA stack install inside the node.
    """

    def __init__(self, config: ClusterConfig) -> None:
        self._config = config

    @property
    def context_name(self) -> str:
        return f"kind-{self._config.name}"

    @property
    def _node_container(self) -> str:
        return f"{self._config.name}-control-plane"

    async def exists(self) -> bool:
        rc, stdout, _ = await _run_quiet(["kind", "get", "clusters"])
        if rc != 0:
            return False
        return self._config.name in stdout.splitlines()

    async def create(self) -> None:
        _strip_localhost_proxy()
        async with timed_operation(f"Creating Kind cluster '{self._config.name}'"):
            cmd = self._base_create_cmd()
            if self._config.gpus:
                await self._create_with_gpu(cmd)
            else:
                await _run_streaming(cmd, "KIND", "Failed to create cluster")

    async def delete(self) -> None:
        async with timed_operation(f"Deleting Kind cluster '{self._config.name}'"):
            cmd = ["kind", "delete", "cluster", "--name", self._config.name]
            if self._config.kubeconfig:
                cmd.extend(["--kubeconfig", str(self._config.kubeconfig)])
            await _run_streaming(cmd, "KIND", "Failed to delete cluster")

    async def image_loaded(self, image: str) -> bool:
        needle = image if ":" in image else f"{image}:latest"
        rc, stdout, _ = await _docker_exec(
            self._node_container,
            "crictl",
            "images",
            "-o",
            "json",
            check=False,
        )
        if rc != 0:
            return False

        import orjson

        data = orjson.loads(stdout)
        for img in data.get("images", []):
            for tag in img.get("repoTags", []):
                bare = tag.removeprefix("docker.io/library/")
                if bare == needle or tag == needle:
                    return True
        return False

    async def load_image(self, image: str) -> None:
        async with timed_operation(f"Loading image '{image}' into Kind cluster"):
            await _run_streaming(
                ["kind", "load", "docker-image", image, "--name", self._config.name],
                "KIND",
                f"Failed to load image {image}",
            )

    # -- private -------------------------------------------------------------

    def _base_create_cmd(self) -> list[str]:
        cmd = ["kind", "create", "cluster", "--name", self._config.name]
        if self._config.node_image:
            cmd.extend(["--image", self._config.node_image])
        if self._config.kubeconfig:
            cmd.extend(["--kubeconfig", str(self._config.kubeconfig)])
        cmd.extend(["--wait", f"{self._config.wait_timeout}s"])
        return cmd

    async def _create_with_gpu(self, base_cmd: list[str]) -> None:
        """Create cluster with GPU sentinel mount, then run GPU setup."""
        await _run_streaming(
            [*base_cmd, "--config", str(_KIND_GPU_CONFIG_PATH)],
            "KIND",
            "Failed to create GPU cluster",
        )

        # Verify GPU devices injected by nvidia runtime
        rc, _, _ = await _docker_exec(
            self._node_container, "ls", "/dev/nvidia0", check=False
        )
        if rc != 0:
            await self.delete()
            raise RuntimeError(
                "GPU devices not visible inside Kind node. "
                "Ensure Docker's default runtime is nvidia "
                "(nvidia-ctk runtime configure --runtime=docker --set-as-default)."
            )

        gpu = KindGpuSetup(
            context=self.context_name,
            node_container=self._node_container,
            device_plugin_version=self._config.device_plugin_version,
        )
        await gpu.setup()


# ============================================================================
# Minikube backend
# ============================================================================


class MinikubeBackend:
    """Minikube cluster backend.

    GPU support: uses native ``--gpus all`` which handles device injection
    and the device plugin automatically. Only the nvidia RuntimeClass needs
    to be created afterwards.
    """

    def __init__(self, config: ClusterConfig) -> None:
        self._config = config

    @property
    def context_name(self) -> str:
        return self._config.name

    async def exists(self) -> bool:
        rc, stdout, _ = await _run_quiet(
            ["minikube", "status", "-p", self._config.name]
        )
        return rc == 0 and "Running" in stdout

    async def create(self) -> None:
        async with timed_operation(f"Creating Minikube cluster '{self._config.name}'"):
            cmd = [
                "minikube",
                "start",
                "-p",
                self._config.name,
                "--driver=docker",
                "--wait=all",
                f"--wait-timeout={self._config.wait_timeout}s",
            ]
            if self._config.gpus:
                cmd.extend(["--gpus", "all"])
            if self._config.kubeconfig:
                cmd.extend(["--kubeconfig", str(self._config.kubeconfig)])
            await _run_streaming(cmd, "MINIKUBE", "Failed to create cluster")

            if self._config.gpus:
                await _create_nvidia_runtime_class(self.context_name)

            if self._config.cache_images:
                await self._cache_images()

    async def delete(self) -> None:
        async with timed_operation(f"Deleting Minikube cluster '{self._config.name}'"):
            cmd = ["minikube", "delete", "-p", self._config.name]
            if self._config.kubeconfig:
                cmd.extend(["--kubeconfig", str(self._config.kubeconfig)])
            await _run_streaming(cmd, "MINIKUBE", "Failed to delete cluster")

    async def image_loaded(self, image: str) -> bool:
        rc, stdout, _ = await _run_quiet(
            ["minikube", "image", "ls", "-p", self._config.name]
        )
        if rc != 0:
            return False

        needle = image if ":" in image else f"{image}:latest"
        prefixed = f"docker.io/library/{needle}"
        return any(line.strip() in (needle, prefixed) for line in stdout.splitlines())

    async def load_image(self, image: str) -> None:
        async with timed_operation(f"Loading image '{image}' into Minikube cluster"):
            await _run_streaming(
                ["minikube", "image", "load", image, "-p", self._config.name],
                "MINIKUBE",
                f"Failed to load image {image}",
            )

    async def _cache_images(self) -> None:
        """Add images to minikube's persistent cache."""
        for image in self._config.cache_images:
            async with timed_operation(f"Caching image '{image}' in minikube"):
                proc = await asyncio.create_subprocess_exec(
                    "minikube",
                    "cache",
                    "add",
                    image,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                output_lines: list[str] = []
                assert proc.stdout is not None
                async for raw_line in proc.stdout:
                    line = raw_line.decode().rstrip()
                    output_lines.append(line)
                    logger.info(f"[CACHE] {line}")
                await proc.wait()
                if proc.returncode != 0:
                    logger.warning(
                        f"Failed to cache {image} (non-fatal):\n"
                        + "\n".join(output_lines)
                    )


# ============================================================================
# Backend factory
# ============================================================================


def _create_backend(config: ClusterConfig) -> ClusterBackend:
    """Create the appropriate backend for the given config."""
    if config.runtime is ClusterRuntime.KIND:
        return KindBackend(config)
    return MinikubeBackend(config)


# ============================================================================
# LocalCluster facade
# ============================================================================


@dataclass
class LocalCluster:
    """Manages a local Kubernetes cluster for testing.

    Delegates to a runtime-specific backend (Kind or Minikube) based on
    ``config.runtime``.  Callers interact only with this facade.
    """

    config: ClusterConfig = field(default_factory=ClusterConfig)
    """Cluster configuration settings."""

    _backend: ClusterBackend = field(init=False, repr=False)
    """Runtime-specific backend implementation."""

    _created: bool = field(default=False, init=False)
    """Whether this instance created the cluster."""

    def __post_init__(self) -> None:
        self._backend = _create_backend(self.config)

    @property
    def name(self) -> str:
        """Get cluster name."""
        return self.config.name

    @property
    def context(self) -> str:
        """Get kubectl context name."""
        return self._backend.context_name

    @property
    def runtime(self) -> ClusterRuntime:
        """Get the active cluster runtime."""
        return self.config.runtime

    async def exists(self) -> bool:
        """Check if the cluster exists and is running."""
        return await self._backend.exists()

    async def create(self, force: bool = False) -> None:
        """Create the cluster.

        Args:
            force: If True, delete existing cluster first.
        """
        if await self.exists():
            if force:
                logger.info(f"Deleting existing cluster: {self.config.name}")
                await self.delete()
            else:
                logger.info(f"Cluster already exists: {self.config.name}")
                self._created = False
                return

        await self._backend.create()
        self._created = True

    async def delete(self) -> None:
        """Delete the cluster."""
        if not await self.exists():
            logger.info(f"Cluster does not exist: {self.config.name}")
            return

        await self._backend.delete()
        self._created = False

    async def image_loaded(self, image: str) -> bool:
        """Check if a Docker image is available inside the cluster."""
        return await self._backend.image_loaded(image)

    async def load_image(self, image: str) -> None:
        """Load a Docker image into the cluster."""
        await self._backend.load_image(image)

    async def get_nodes(self) -> list[str]:
        """Get list of cluster nodes."""
        rc, stdout, _ = await _kubectl(
            self.context, "get", "nodes", "-o", "name", check=False
        )
        if rc != 0:
            return []
        return [
            n.strip().removeprefix("node/") for n in stdout.splitlines() if n.strip()
        ]

    async def __aenter__(self) -> LocalCluster:
        """Context manager entry - create cluster."""
        await self.create()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - delete cluster if we created it."""
        if self._created:
            await self.delete()
