# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operator-side runner for sweep audit cases.

Submits an ``AIPerfSweep`` CR (no test-helper exists for sweep CRs, so the
manifest is built inline), waits for the parent's ``status.phase`` to reach a
terminal state, lists the owned child ``AIPerfJob`` CRs by the
``aiperf.nvidia.com/sweep=<name>`` label, then downloads each child's results
into a per-cell subdirectory ``v<varidx>-t<trialidx>/`` via
``aiperf kube results <child_name> --all``.

Notes on the parent download path
---------------------------------
The plan originally called for ``aiperf kube results <sweep-name>`` to fetch
the parent's ``children.json`` first. That CLI's ``resolve_job`` only looks
up ``AIPerfJob`` CRs (and falls back to JobSet), so it cannot resolve a
sweep-parent name. We instead enumerate children directly from the cluster:
the sweep-controller stamps ``aiperf.nvidia.com/variation-index`` and
``aiperf.nvidia.com/trial-index`` labels on every child AIPerfJob (see
``src/aiperf/sweep_controller/k8s_executor.py``), so we get the
``(variation_index, trial_index, child_name)`` mapping straight from the
child CR list. The authoritative ``children.json`` on the operator PVC is
preserved for post-mortem use; the audit just doesn't need it.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.audit.cases import AuditCase
from tests.kubernetes.audit.operator_runner import OperatorAuditConfig
from tests.kubernetes.helpers.kubectl import KubectlClient

logger = AIPerfLogger(__name__)

# Terminal phases for AIPerfSweep — mirror PARENT_TERMINAL_PHASES in
# src/aiperf/operator/handlers/sweep/child_rollup.py. ``Succeeded`` is the
# only success phase; everything else here is a hard failure for the audit.
_TERMINAL_PHASES = frozenset({"Succeeded", "Failed", "Cancelled", "PartiallyFailed"})
_SUCCESS_PHASES = frozenset({"Succeeded"})

# Labels stamped on child AIPerfJob CRs by the sweep-controller.
_SWEEP_LABEL = "aiperf.nvidia.com/sweep"
_VARIATION_INDEX_LABEL = "aiperf.nvidia.com/variation-index"
_TRIAL_INDEX_LABEL = "aiperf.nvidia.com/trial-index"


@dataclass(frozen=True)
class SweepCell:
    """One cell of the sweep x trials grid."""

    variation_index: int
    """Zero-based variation index within the sweep."""

    trial_index: int
    """Zero-based trial index within the variation; 0 when trials==1."""

    child_name: str
    """Name of the owned child AIPerfJob CR."""

    local_dir: Path
    """Local directory where this cell's artifacts were downloaded."""


class SweepAuditRunner:
    """Submits an AIPerfSweep, waits for terminal phase, downloads each child."""

    def __init__(
        self,
        kubectl: KubectlClient,
        config: OperatorAuditConfig | None = None,
    ) -> None:
        self.kubectl = kubectl
        self.config = config or OperatorAuditConfig()

    def _build_sweep_manifest(
        self, *, name: str, namespace: str, case: AuditCase
    ) -> str:
        """Build an AIPerfSweep CR manifest YAML.

        Mirrors ``AIPerfJobConfig.to_flat_spec`` for the per-child benchmark
        body, then adds the parent-level ``sweep`` (GridSweep) and
        ``multi_run`` blocks. Sweep-axis keys live at ``spec`` and are
        explicitly forbidden from ``template.spec.benchmark`` by
        ``AIPerfSweepSpec``'s ``_validate_axis_combination`` validator.
        """
        if case.sweep is None:
            raise ValueError("SweepAuditRunner requires case.sweep to be set")

        load: dict[str, Any] = {
            "profiling": {
                "type": "concurrency",
                "concurrency": case.concurrency,
                "requests": case.request_count,
            },
        }
        benchmark_spec: dict[str, Any] = {
            "models": {"items": [{"name": self.config.model_name}]},
            "endpoint": {"urls": [self.config.endpoint_url]},
            "datasets": [
                {
                    "name": "main",
                    "type": "synthetic",
                    "entries": max(case.request_count, 10),
                    "prompts": {"isl": {"mean": 550}},
                },
            ],
            "phases": load,
            "tokenizer": {"name": self.config.tokenizer_name},
            "runtime": {"ui": "none"},
        }

        # AuditCase.sweep is e.g. {"concurrency": [4, 8, 16]}. The CRD's
        # SweepConfig is a discriminated union; GridSweep takes a `variables`
        # mapping of dot-paths -> value lists. The path "concurrency" maps to
        # phases.profiling.concurrency via the magic-list detection in
        # aiperf.config.sweep, and the GridSweep variables shape lets us pin
        # to the explicit dot-path when needed.
        sweep_dim_name, sweep_values = next(iter(case.sweep.items()))
        sweep_block: dict[str, Any] = {
            "type": "grid",
            "variables": {sweep_dim_name: list(sweep_values)},
        }

        spec: dict[str, Any] = {
            "image": self.config.image,
            "imagePullPolicy": self.config.image_pull_policy,
            "sweep": sweep_block,
            "multiRun": {"trials": case.trials},
            "template": {"spec": {"benchmark": benchmark_spec}},
        }
        body = {
            "apiVersion": "aiperf.nvidia.com/v1alpha1",
            "kind": "AIPerfSweep",
            "metadata": {"name": name, "namespace": namespace},
            "spec": spec,
        }
        return yaml.safe_dump(body, sort_keys=False)

    async def _wait_for_terminal(self, name: str, namespace: str, timeout: int) -> str:
        """Poll AIPerfSweep ``.status.phase`` until terminal. Returns the phase."""
        deadline = asyncio.get_event_loop().time() + timeout
        last_phase = "<unknown>"
        while asyncio.get_event_loop().time() < deadline:
            result = await self.kubectl.run(
                "get",
                "aiperfsweep",
                name,
                "-n",
                namespace,
                "-o",
                "json",
                check=False,
            )
            if result.returncode == 0:
                payload = json.loads(result.stdout)
                phase = payload.get("status", {}).get("phase", "<pending>")
                last_phase = phase
                if phase in _TERMINAL_PHASES:
                    logger.info(
                        f"AIPerfSweep {namespace}/{name} reached terminal "
                        f"phase: {phase}"
                    )
                    return phase
                logger.info(f"AIPerfSweep {namespace}/{name} phase={phase}, waiting...")
            else:
                logger.debug(
                    lambda r=result: f"kubectl get aiperfsweep failed: {r.stderr}"
                )
            await asyncio.sleep(5)
        raise TimeoutError(
            f"AIPerfSweep {namespace}/{name} did not reach terminal state in "
            f"{timeout}s (last seen phase: {last_phase})"
        )

    async def _list_children(
        self, *, sweep_name: str, namespace: str, dest_root: Path
    ) -> list[SweepCell]:
        """List child AIPerfJob CRs by the ``aiperf.nvidia.com/sweep`` label.

        Reads variation/trial indices from the child labels stamped by the
        sweep-controller (see ``src/aiperf/sweep_controller/k8s_executor.py``
        ``VARIATION_INDEX_LABEL`` / ``TRIAL_INDEX_LABEL``). Children are
        sorted by ``(variation_index, trial_index)`` for deterministic diffs.
        """
        result = await self.kubectl.run(
            "get",
            "aiperfjob",
            "-n",
            namespace,
            "-l",
            f"{_SWEEP_LABEL}={sweep_name}",
            "-o",
            "json",
            check=True,
        )
        payload = json.loads(result.stdout)
        cells: list[SweepCell] = []
        for item in payload.get("items", []):
            metadata = item.get("metadata") or {}
            labels = metadata.get("labels") or {}
            child_name = metadata.get("name")
            if not child_name:
                continue
            try:
                v = int(labels.get(_VARIATION_INDEX_LABEL, "0"))
                t = int(labels.get(_TRIAL_INDEX_LABEL, "0"))
            except (TypeError, ValueError) as e:
                raise RuntimeError(
                    f"child AIPerfJob {namespace}/{child_name} has malformed "
                    f"variation/trial label (variation="
                    f"{labels.get(_VARIATION_INDEX_LABEL)!r}, trial="
                    f"{labels.get(_TRIAL_INDEX_LABEL)!r}); cannot map to a "
                    f"sweep cell."
                ) from e
            cells.append(
                SweepCell(
                    variation_index=v,
                    trial_index=t,
                    child_name=child_name,
                    local_dir=dest_root / f"v{v}-t{t}",
                )
            )
        if not cells:
            raise RuntimeError(
                f"No children found for AIPerfSweep {namespace}/{sweep_name} "
                f"using label selector {_SWEEP_LABEL}={sweep_name}; the "
                f"sweep-controller may not have created any child AIPerfJobs."
            )
        cells.sort(key=lambda c: (c.variation_index, c.trial_index))
        return cells

    async def _download_child(
        self,
        *,
        child_name: str,
        namespace: str,
        dest_dir: Path,
        kubeconfig: str | None,
    ) -> None:
        """Shell out to ``aiperf kube results <child_name> --all``."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "aiperf",
            "kube",
            "results",
            child_name,
            "--namespace",
            namespace,
            "--output",
            str(dest_dir),
            "--all",
        ]
        env = dict(os.environ)
        if kubeconfig:
            env["KUBECONFIG"] = kubeconfig

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"aiperf kube results {child_name} failed (rc={proc.returncode})\n"
                f"stdout:\n{stdout.decode(errors='replace')}\n"
                f"stderr:\n{stderr.decode(errors='replace')}"
            )

    async def run(
        self,
        *,
        case: AuditCase,
        namespace: str,
        dest_dir: Path,
        kubeconfig: str | None = None,
        timeout: int = 1800,
    ) -> list[SweepCell]:
        """Submit sweep, wait, download each child. Returns the SweepCell list.

        ``dest_dir`` will contain one subdirectory per child cell:
        ``v<i>-t<j>/`` with that cell's downloaded artifacts.
        """
        suffix = uuid.uuid4().hex[:6]
        sweep_name = f"audit-sweep-{case.case_id}-{suffix}"

        await self.kubectl.run("create", "namespace", namespace, check=False)

        manifest = self._build_sweep_manifest(
            name=sweep_name, namespace=namespace, case=case
        )
        await self.kubectl.apply(manifest, namespace=namespace)

        try:
            phase = await self._wait_for_terminal(sweep_name, namespace, timeout)
            if phase not in _SUCCESS_PHASES:
                raise RuntimeError(
                    f"AIPerfSweep {namespace}/{sweep_name} terminal phase = "
                    f"{phase}; expected one of {sorted(_SUCCESS_PHASES)}."
                )

            cells = await self._list_children(
                sweep_name=sweep_name, namespace=namespace, dest_root=dest_dir
            )
            for cell in cells:
                logger.info(f"Downloading child {cell.child_name} -> {cell.local_dir}")
                await self._download_child(
                    child_name=cell.child_name,
                    namespace=namespace,
                    dest_dir=cell.local_dir,
                    kubeconfig=kubeconfig,
                )
            return cells
        finally:
            await self.kubectl.run(
                "delete",
                "aiperfsweep",
                sweep_name,
                "-n",
                namespace,
                "--wait=false",
                check=False,
            )
