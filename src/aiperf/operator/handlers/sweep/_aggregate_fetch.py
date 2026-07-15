# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Harvest the cross-variation aggregate from a sweep-controller's sidecar.

The sweep-controller pod runs a results-sidecar (added in
``handlers/sweep/create.py``) that serves ``/results/`` over HTTP, gated by
the ``.aiperf_results_ready.json`` marker the controller writes after
aggregation. The pod itself uses ``emptyDir{}`` for ``/results``, so the
operator MUST pull the artifacts via the sidecar before the JobSet is
deleted — otherwise the per-sweep aggregate is lost on pod cleanup.

Symmetric with ``_completion_fetch.py`` (the AIPerfJob harvester) but
narrower in scope: there is exactly one host (the sweep-controller's
single replica), no checkpoint dance, and the destination is the
operator's PVC at ``<base>/<ns>/sweeps/<sweep>/<epoch>/`` so the
``getSweepEpochs`` / ``getSweepCells`` REST endpoints can read it.

The handler is idempotent: re-fetch over an already-populated dest-dir is
a no-op for the operator's reads (the `latest.txt` write at the end is
the only state-changing piece).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import aiohttp

from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.kubernetes.jobset import controller_dns_name
from aiperf.operator.progress_client import ProgressClient

__all__ = ["SweepAggregateFetchResult", "fetch_sweep_aggregate_to_disk"]

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class SweepAggregateFetchResult:
    """Outcome of one sidecar harvest attempt.

    ``downloaded < listed`` means the harvest is PARTIAL: the sidecar
    advertised files that never landed on the PVC (sidecar dying mid-stream,
    PVC write failure). Callers must NOT delete the sweep JobSet on a partial
    harvest — the controller pod's emptyDir holds the only other copy of the
    missing files.
    """

    downloaded: int
    """Count of files successfully written to the operator's PVC."""

    listed: int
    """Count of files the sidecar advertised via ``/api/results/list``."""

    @property
    def is_partial(self) -> bool:
        """True when at least one advertised file failed to download."""
        return self.downloaded < self.listed


def _ensure_sweep_root(base_dir: Path, namespace: str, sweep_name: str) -> Path:
    """Ensure ``<base>/<ns>/sweeps/<sweep>/`` exists; return it."""
    target = base_dir / namespace / "sweeps" / sweep_name
    target.mkdir(parents=True, exist_ok=True)
    return target


def _write_sweep_latest_pointer(
    base_dir: Path, namespace: str, sweep_name: str, epoch: str
) -> None:
    """Atomically point ``latest.txt`` at the just-fetched epoch.

    ``list_sweep_epochs`` reads this to flag ``isLatest`` on each entry; the
    SweepDetail page picks the latest as the default selection.
    """
    sweep_root = _ensure_sweep_root(base_dir, namespace, sweep_name)
    pointer = sweep_root / "latest.txt"
    tmp = pointer.with_suffix(".tmp")
    tmp.write_text(epoch)
    tmp.replace(pointer)


async def fetch_sweep_aggregate_to_disk(
    *,
    sweep_name: str,
    namespace: str,
    epoch: str,
    base_dir: Path,
) -> SweepAggregateFetchResult:
    """Download every artifact from the sweep-controller sidecar to the PVC.

    Args:
        sweep_name: AIPerfSweep CR name (also the JobSet's stem; the
            sweep-controller's JobSet is ``aiperf-<sweep_name>``).
        namespace: Sweep namespace.
        epoch: ``status.runEpoch`` from the parent CR — used as the per-sweep
            epoch directory name on the operator's PVC.
        base_dir: Operator-side results root (``OperatorEnvironment.RESULTS.DIR``,
            typically ``/data``).

    Returns:
        :class:`SweepAggregateFetchResult` with the downloaded and listed
        file counts. ``downloaded == 0`` indicates the sidecar was
        unreachable, the marker wasn't ready, or no files were listed;
        ``downloaded < listed`` indicates a partial harvest (some advertised
        files failed to land on disk).

    The handler does not raise on transient sidecar failures (the
    sweep-controller pod may be mid-shutdown); callers re-trigger on the
    next ``status.aggregation.phase`` reconcile if the count is zero or
    the harvest is partial.

    The sweep-controller writes its parent aggregate at
    ``/results/<ns>/sweeps/<sweep>/<epoch>/aggregate.json`` (and
    ``children.json``) — that path is already operator-mirror-shaped under
    its ``/results`` mount. Downloading to ``<base>`` (NOT the per-sweep
    epoch dir) re-roots the whole tree onto the operator's PVC at the
    expected ``<base>/<ns>/sweeps/<sweep>/<epoch>/aggregate.json`` location.
    Pointing dest_dir at the per-sweep dir would double-nest the path.
    """
    jobset_name = f"aiperf-{sweep_name}"
    host = controller_dns_name(jobset_name, namespace)
    sidecar_port = K8sEnvironment.PORTS.RESULTS_SIDECAR
    base_dir.mkdir(parents=True, exist_ok=True)

    try:
        async with ProgressClient(port=sidecar_port) as sidecar:
            # List first so callers can compare downloaded vs advertised.
            # download_all_results re-lists internally; the file set is
            # frozen once the controller writes the ready marker, so the
            # second list sees the same files (or fewer on sidecar death,
            # which the downloaded < listed comparison catches).
            listed_files = await sidecar.get_results_list(host)
            if not listed_files:
                logger.info(
                    f"sweep-aggregate fetch found no files for {namespace}/{sweep_name} "
                    f"(sidecar may be pre-marker)"
                )
                return SweepAggregateFetchResult(downloaded=0, listed=0)
            downloaded_files = await sidecar.download_all_results(host, base_dir)
    except (aiohttp.ClientError, ConnectionError, TimeoutError, OSError) as e:
        logger.warning(
            f"sweep-aggregate fetch failed for {namespace}/{sweep_name} "
            f"@ {host}:{sidecar_port}: {e}"
        )
        return SweepAggregateFetchResult(downloaded=0, listed=0)

    listed = len(listed_files)
    count = len(downloaded_files or [])
    if count == 0:
        logger.info(
            f"sweep-aggregate fetch returned no files for {namespace}/{sweep_name} "
            f"(sidecar may be pre-marker)"
        )
        return SweepAggregateFetchResult(downloaded=0, listed=listed)

    try:
        _write_sweep_latest_pointer(base_dir, namespace, sweep_name, epoch)
    except OSError as exc:
        # Pointer write is purely advisory (latest-epoch hint for `aiperf kube
        # sweeps`); a transient PVC hiccup must not make a successful fetch
        # look failed. Log and proceed.
        logger.warning(
            f"sweep-aggregate {namespace}/{sweep_name} epoch={epoch}: "
            f"latest-pointer write at {base_dir} failed "
            f"({type(exc).__name__}: {exc}); fetch already succeeded so this is advisory"
        )
    logger.info(
        f"sweep-aggregate fetch ok: {namespace}/{sweep_name} epoch={epoch} "
        f"files={count}/{listed} -> {base_dir}"
    )
    return SweepAggregateFetchResult(downloaded=count, listed=listed)
