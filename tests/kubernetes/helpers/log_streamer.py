# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Background pod log streamer for Kubernetes E2E tests.

Streams ``kubectl logs -f`` output to the Python logger while tests run,
so operators can watch pod startup/runtime in real time.

Usage as an async context manager::

    async with PodLogStreamer(kubectl, "vllm-server", prefix="VLLM") as streamer:
        streamer.watch(name_filter="vllm-server")
        # ... poll for readiness ...
    # all streaming tasks are cancelled on exit

Multi-container pods are automatically handled: each container gets its
own stream with a ``pod/container`` label.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.helpers.kubectl import KubectlClient

logger = AIPerfLogger(__name__)


@dataclass
class PodLogStreamer:
    """Streams pod logs in the background via ``kubectl logs -f``.

    Discovers pods in *namespace* that match *name_filter*, starts a
    ``kubectl logs -f`` subprocess for each container, and pipes every
    line through the Python logger at INFO level.

    Designed to be used as an async context manager so that all background
    tasks and subprocesses are cleaned up automatically.
    """

    kubectl: KubectlClient
    """Kubectl client for cluster interaction."""

    namespace: str
    """Namespace to stream logs from."""

    prefix: str = ""
    """Log line prefix for identification."""

    tail: int = 20
    """Number of existing log lines to show on stream start."""

    _tasks: list[asyncio.Task] = field(default_factory=list, init=False, repr=False)
    """Background streaming tasks."""

    _processes: list[asyncio.subprocess.Process] = field(
        default_factory=list, init=False, repr=False
    )
    """Active kubectl log subprocess handles."""

    _known_streams: set[str] = field(default_factory=set, init=False, repr=False)
    """Tracks pod/container combos already being streamed."""

    _discovery_task: asyncio.Task | None = field(default=None, init=False, repr=False)
    """Background task that discovers new pods to stream."""

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> PodLogStreamer:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def watch(self, name_filter: str | None = None) -> None:
        """Start discovering pods and streaming their logs.

        Pods whose names contain *name_filter* (if given) are streamed.
        New pods that appear later are picked up automatically.

        For multi-container pods, each container gets its own stream.

        This is non-blocking; it spawns background asyncio tasks.
        """
        self._discovery_task = asyncio.create_task(
            self._discover_loop(name_filter),
            name=f"log-streamer-discovery-{self.namespace}",
        )

    async def stop(self) -> None:
        """Cancel all streaming tasks and terminate subprocesses."""
        if self._discovery_task and not self._discovery_task.done():
            self._discovery_task.cancel()

        for task in self._tasks:
            if not task.done():
                task.cancel()

        # Wait for all tasks to finish cancellation
        all_tasks = [t for t in [self._discovery_task, *self._tasks] if t is not None]
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)

        for proc in self._processes:
            try:
                proc.terminate()
                await proc.wait()
            except ProcessLookupError:
                pass

        self._tasks.clear()
        self._processes.clear()
        self._known_streams.clear()
        self._discovery_task = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _discover_loop(self, name_filter: str | None) -> None:
        """Periodically discover new pods and start streaming them."""
        try:
            while True:
                try:
                    pods = await self.kubectl.get_pods(self.namespace)
                except Exception:
                    await asyncio.sleep(5)
                    continue

                for pod in pods:
                    if name_filter and name_filter not in pod.name:
                        continue

                    containers = list(pod.containers.keys()) if pod.containers else []

                    if len(containers) <= 1:
                        # Single or no containers: stream the pod directly
                        stream_key = pod.name
                        if stream_key not in self._known_streams:
                            self._known_streams.add(stream_key)
                            task = asyncio.create_task(
                                self._stream_pod(pod.name),
                                name=f"log-stream-{pod.name}",
                            )
                            self._tasks.append(task)
                    else:
                        # Multi-container: stream each container separately
                        for container in containers:
                            stream_key = f"{pod.name}/{container}"
                            if stream_key not in self._known_streams:
                                self._known_streams.add(stream_key)
                                task = asyncio.create_task(
                                    self._stream_pod(pod.name, container=container),
                                    name=f"log-stream-{pod.name}-{container}",
                                )
                                self._tasks.append(task)

                await asyncio.sleep(5)
        except asyncio.CancelledError:
            pass

    def _make_label(self, pod_name: str, container: str | None = None) -> str:
        """Build a human-readable log label."""
        target = f"{pod_name}/{container}" if container else pod_name
        return f"[{self.prefix}:{target}]" if self.prefix else f"[{target}]"

    async def _stream_pod(self, pod_name: str, container: str | None = None) -> None:
        """Stream logs from a single pod (or container) via ``kubectl logs -f``."""
        args = [
            "logs",
            "-f",
            "--tail",
            str(self.tail),
        ]
        if container:
            args.extend(["-c", container])
        args.append(pod_name)

        cmd = self.kubectl._build_cmd(*args, namespace=self.namespace)
        label = self._make_label(pod_name, container)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self._processes.append(proc)

            if proc.stdout is None:
                return

            async for raw_line in proc.stdout:
                line = raw_line.decode(errors="replace").rstrip()
                if line:
                    logger.info(f"{label} {line}")

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.debug(lambda label=label, exc=exc: f"{label} stream error: {exc}")
