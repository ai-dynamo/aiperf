# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operator-side runner for sweep audit cases.

Submits an ``AIPerfSweep`` CR (no test-helper exists for sweep CRs, so the
manifest is built inline), waits for the parent's ``status.phase`` to reach a
terminal state, then shells ``aiperf kube results <sweep-name> --all`` to
download every child via R2's CLI fan-out path. Reads the resulting
``sweep_manifest.json`` (camelCase schema; see
``aiperf.kubernetes.results._fetch_children_manifest``) to enumerate the
``SweepCell`` list returned to the test.

Exercising the user-facing CLI is the whole point of the audit — until R2,
``aiperf kube results <sweep-name>`` errored on AIPerfSweep CRs, so the audit
had to bypass it via ``kubectl get aiperfjob -l aiperf.nvidia.com/sweep=...``.
R2 made the CLI sweep-aware, so we use it here.
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
from tests.kubernetes.audit import aiperf_cli
from tests.kubernetes.audit.cases import AuditCase
from tests.kubernetes.audit.operator_runner import OperatorAuditConfig
from tests.kubernetes.helpers.kubectl import KubectlClient

logger = AIPerfLogger(__name__)

# Terminal phases for AIPerfSweep — mirror PARENT_TERMINAL_PHASES in
# src/aiperf/operator/handlers/sweep/child_rollup.py. ``Succeeded`` is the
# only success phase; everything else here is a hard failure for the audit.
_TERMINAL_PHASES = frozenset({"Succeeded", "Failed", "Cancelled", "PartiallyFailed"})
_SUCCESS_PHASES = frozenset({"Succeeded"})


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
    """Submits an AIPerfSweep, waits for terminal phase, downloads via the CLI."""

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
        explicitly forbidden from ``spec.benchmark`` by
        ``AIPerfSweepSpec``'s ``_validate_axis_combination`` validator.
        """
        if case.sweep is None:
            raise ValueError("SweepAuditRunner requires case.sweep to be set")

        phases: list[dict[str, Any]] = [
            {
                "name": "profiling",
                "type": "concurrency",
                "concurrency": case.concurrency,
                "requests": case.request_count,
                **(
                    {"sessions": case.num_conversations}
                    if case.num_conversations is not None
                    else {}
                ),
            },
        ]
        benchmark_spec: dict[str, Any] = {
            "models": {"items": [{"name": self.config.model_name}]},
            "endpoint": {"urls": [self.config.endpoint_url]},
            "datasets": [
                {
                    "name": "main",
                    "type": "synthetic",
                    "entries": (
                        case.num_conversations
                        if case.num_conversations is not None
                        else max(case.request_count, 10)
                    ),
                    "prompts": {"isl": {"mean": 550}},
                },
            ],
            "phases": phases,
            "tokenizer": {"name": self.config.tokenizer_name},
            "runtime": {"ui": "none"},
        }

        # AuditCase.sweep is e.g. {"concurrency": [4, 8, 16]}. The CRD's
        # SweepConfig is a discriminated union; GridSweep takes a `parameters`
        # mapping of dot-paths -> value lists (it is required, and there is no
        # `variables` alias). The path "concurrency" maps to
        # phases.profiling.concurrency via the magic-list detection in
        # aiperf.config.sweep.
        sweep_dim_name, sweep_values = next(iter(case.sweep.items()))
        sweep_block: dict[str, Any] = {
            "type": "grid",
            "parameters": {sweep_dim_name: list(sweep_values)},
        }

        spec: dict[str, Any] = {
            "sweep": sweep_block,
            # Envelope-level, same as AIPerfJob: BenchmarkConfig forbids it, so
            # it cannot live under spec.benchmark. The bare-pod side pins the
            # same value via --random-seed, and without it here the operator
            # export has no run_info.random_seed to compare against.
            "randomSeed": case.seed,
            # MultiRunConfig calls this numRuns, not trials. The CRD decodes
            # strictly, so the stale key is rejected at admission.
            "multiRun": {"numRuns": case.trials},
            "image": self.config.image,
            "imagePullPolicy": self.config.image_pull_policy,
            "benchmark": benchmark_spec,
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

    async def _download_sweep_via_cli(
        self,
        *,
        sweep_name: str,
        namespace: str,
        dest_dir: Path,
        kubeconfig: str | None,
    ) -> None:
        """Shell out to ``aiperf kube results <sweep-name>`` (R2 fans out per child)."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            aiperf_cli(),
            "kube",
            "results",
            sweep_name,
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
                f"aiperf kube results {sweep_name} failed (rc={proc.returncode})\n"
                f"stdout:\n{stdout.decode(errors='replace')}\n"
                f"stderr:\n{stderr.decode(errors='replace')}"
            )

    def _enumerate_cells_from_manifest(self, dest_dir: Path) -> list[SweepCell]:
        """Read ``sweep_manifest.json`` produced by R2's CLI fan-out.

        Schema (camelCase, see ``aiperf.kubernetes.results._fetch_children_manifest``):
            {"sweepRunEpoch": int, "children": [{
                "namespace": str, "name": str,
                "variationIndex": int, "trialIndex": int | None,
                "variationLabel": str, "childRunEpoch": str,
            }]}
        """
        manifest_path = dest_dir / "sweep_manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(
                f"sweep_manifest.json missing in {dest_dir} — did `aiperf kube results "
                f"<sweep>` succeed? Run with `--all` and confirm R2's CLI is wired."
            )
        payload = json.loads(manifest_path.read_text())
        children = payload.get("children", [])
        if not children:
            raise RuntimeError(
                f"sweep_manifest.json at {manifest_path} has no children"
            )

        cells: list[SweepCell] = []
        for entry in children:
            v = int(entry["variationIndex"])
            # trialIndex may be None (single-trial); coerce to 0 to match the dir name.
            t_raw = entry.get("trialIndex")
            t = int(t_raw) if t_raw is not None else 0
            cells.append(
                SweepCell(
                    variation_index=v,
                    trial_index=t,
                    child_name=entry["name"],
                    local_dir=dest_dir / f"v{v}-t{t}",
                )
            )
        cells.sort(key=lambda c: (c.variation_index, c.trial_index))
        return cells

    async def run(
        self,
        *,
        case: AuditCase,
        namespace: str,
        dest_dir: Path,
        kubeconfig: str | None = None,
        timeout: int = 1800,
    ) -> list[SweepCell]:
        """Submit sweep, wait, download via CLI. Returns the SweepCell list.

        ``dest_dir`` will contain one subdirectory per child cell:
        ``v<i>-t<j>/`` with that cell's downloaded artifacts.
        """
        suffix = uuid.uuid4().hex[:6]
        sweep_name = f"sw-{case.case_id}-{suffix}"

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
            await self._download_sweep_via_cli(
                sweep_name=sweep_name,
                namespace=namespace,
                dest_dir=dest_dir,
                kubeconfig=kubeconfig,
            )
            cells = self._enumerate_cells_from_manifest(dest_dir)
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
