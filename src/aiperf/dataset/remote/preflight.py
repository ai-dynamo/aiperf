# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Access and materialize stages for remote FileDataset references.

Access (before the endpoint probe): pin each ref's exact version, apply its manifest,
run compatibility checks, and finish fixed-schedule auto-promotion from the first record.
Materialize (after the endpoint probe): download into the cache so DatasetManager
never blocks on the network inside AIPERF_DATASET_CONFIGURATION_TIMEOUT.

Every run that names the same remote ref (sweep variations, --variant runs) shares
one probe, so all of them use the same version even if the remote changes meanwhile.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from aiperf import __version__
from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.record_files import (
    PARQUET_SUFFIX,
    PYARROW_MISSING,
    first_record,
    is_parquet_input,
    parquet_available,
    parquet_files,
)
from aiperf.config.dataset import FileDataset
from aiperf.config.dataset.remote_ref import RemoteRef
from aiperf.dataset.remote import service
from aiperf.dataset.remote.errors import (
    RemoteErrorKind,
    RemoteSourceError,
    RemoteStage,
    changed_after_pin,
)
from aiperf.dataset.remote.fixed_schedule import finalize_remote_fixed_schedule
from aiperf.dataset.remote.manifest import check_compatibility
from aiperf.dataset.remote.models import ProbeResult, ResolvedSource
from aiperf.dataset.remote.sniff import (
    COLUMNAR_SUFFIXES,
    remote_first_record_has_timestamp,
)

if TYPE_CHECKING:
    from aiperf.config.config import BenchmarkConfig
    from aiperf.plugin.enums import CustomDatasetType

_logger = AIPerfLogger(__name__)


class _ProbeCache:
    """Probes from the access stage, reused by the materialize stage in the same process."""

    def __init__(self) -> None:
        self._probes: dict[tuple[str, str | None], ProbeResult] = {}

    def add(self, probe: ProbeResult) -> None:
        self._probes[(probe.ref.canonical, probe.ref.resolved_version)] = probe

    def get(self, ref: RemoteRef) -> ProbeResult | None:
        return self._probes.get((ref.canonical, ref.resolved_version))

    def clear(self) -> None:
        self._probes.clear()


_probe_cache = _ProbeCache()


def reset_probe_cache() -> None:
    """Forget probes from earlier access stages."""
    _probe_cache.clear()


def _remote_groups(
    configs: Sequence[BenchmarkConfig],
) -> dict[str, list[BenchmarkConfig]]:
    """Configs whose default dataset is a remote file, grouped by canonical ref."""
    groups: dict[str, list[BenchmarkConfig]] = {}
    for cfg in configs:
        ds = cfg.get_default_dataset()
        if isinstance(ds, FileDataset) and isinstance(ds.path, RemoteRef):
            groups.setdefault(ds.path.canonical, []).append(cfg)
    return groups


def _declared_format(ds: FileDataset) -> CustomDatasetType | None:
    """The explicitly chosen loader, or None when the loader is auto-detected."""
    from aiperf.plugin.enums import CustomDatasetType

    if "format" not in ds.model_fields_set:
        return None
    return CustomDatasetType(str(ds.format))


def _probe_updates(
    cfg: BenchmarkConfig, ds: FileDataset, probe: ProbeResult
) -> dict[str, Any]:
    """Dataset fields set by the probe: the pinned ref plus manifest defaults."""
    from aiperf.plugin.enums import CustomDatasetType

    updates: dict[str, Any] = {"path": probe.ref}
    manifest = probe.manifest
    if manifest is None:
        return updates
    check_compatibility(
        manifest,
        ref_display=probe.ref.display,
        endpoint_type=str(cfg.endpoint.type),
        max_context_length=ds.max_context_length,
        aiperf_version=__version__,
    )
    if manifest.format is not None:
        # check_compatibility has confirmed the format is a registered loader.
        manifest_format = CustomDatasetType(manifest.format)
        declared = _declared_format(ds)
        if declared is not None and declared != manifest_format:
            raise RemoteSourceError(
                ref=probe.ref.display, kind=RemoteErrorKind.INCOMPATIBLE, stage=RemoteStage.ACCESS,
                what=f"its manifest declares format '{manifest_format}', but --custom-dataset-type is '{declared}'",
                hint="Drop --custom-dataset-type, or pass the manifest's format",
            )  # fmt: skip
        updates["format"] = manifest_format
    if manifest.block_size is not None and ds.block_size is None:
        updates["block_size"] = manifest.block_size
    return updates


def _apply_probe(cfg: BenchmarkConfig, probe: ProbeResult) -> FileDataset:
    """Replace the default dataset with one pinned to ``probe`` and marker-free.

    The dataset is re-validated so FileDataset's validators see manifest-supplied
    fields exactly as if the user had passed them.
    """
    ds = cfg.get_default_dataset()
    updates = _probe_updates(cfg, ds, probe)
    data = ds.model_dump(exclude_unset=True)
    data.pop("remote_fixed_schedule_check", None)
    data.update(updates)
    try:
        rebuilt = FileDataset.model_validate(data)
    except ValidationError as e:
        manifest_fields = ", ".join(
            f"{key}={value}" for key, value in updates.items() if key != "path"
        )
        raise RemoteSourceError(
            ref=probe.ref.display, kind=RemoteErrorKind.INCOMPATIBLE, stage=RemoteStage.ACCESS,
            what=f"its manifest ({manifest_fields}) conflicts with this run's dataset options: {e.errors()[0]['msg']}",
            hint="Drop the conflicting option",
        ) from e  # fmt: skip
    index = next(i for i, d in enumerate(cfg.datasets) if d is ds)
    cfg.datasets[index] = rebuilt
    return rebuilt


def _require_parquet_support(probe: ProbeResult) -> None:
    """Fail before any download when the source is parquet and pyarrow is unavailable."""
    is_parquet = (
        probe.data_files is not None
        or PurePosixPath(probe.filename).suffix.lower() == PARQUET_SUFFIX
    )
    if is_parquet and not parquet_available():
        raise RemoteSourceError(
            ref=probe.ref.display, kind=RemoteErrorKind.MISSING_DEPENDENCY, stage=RemoteStage.ACCESS,
            what=PYARROW_MISSING,
            hint="Use a JSONL copy of the dataset, or run on a platform where pyarrow installs",
        )  # fmt: skip


async def _access(groups: dict[str, list[BenchmarkConfig]]) -> None:
    from aiperf.plugin import plugins

    for cfgs in groups.values():
        probe = await service.probe_ref(cfgs[0].get_default_dataset().path)
        _require_parquet_support(probe)
        _probe_cache.add(probe)
        has_timestamp: bool | None = None
        for cfg in cfgs:
            check = cfg.get_default_dataset().remote_fixed_schedule_check
            ds = _apply_probe(cfg, probe)
            if check is None:
                continue
            fmt = _declared_format(ds)
            is_trace = fmt is not None and plugins.is_trace_dataset(str(fmt))
            if is_trace and has_timestamp is None:
                has_timestamp = await remote_first_record_has_timestamp(probe)
            finalize_remote_fixed_schedule(
                cfg, check, promote=is_trace and bool(has_timestamp)
            )


def preflight_remote_access(configs: Sequence[BenchmarkConfig]) -> None:
    """Pin, check, and settle fixed-schedule promotion for every remote dataset.

    Mutates ``configs`` in place.

    Raises:
        RemoteSourceError: The source is unreachable, or its manifest is invalid or
            incompatible with the run.
        ConfigurationError: The trace has timestamps, so it would be promoted to
            fixed_schedule, but the run selected an incompatible mode or a sweep.
    """
    groups = _remote_groups(configs)
    if groups:
        asyncio.run(_access(groups))


def _check_first_record_format(fmt: CustomDatasetType, source: ResolvedSource) -> None:
    """Reject a download whose first record the chosen loader cannot load."""
    from aiperf.config.dataset.resolver import DatasetResolver
    from aiperf.plugin import plugins
    from aiperf.plugin.enums import PluginType

    # A category variant loads a filtered subset, so its can_load says nothing
    # about whether the file itself has the right shape.
    if not plugins.is_autodetectable(
        plugins.get_entry(PluginType.CUSTOM_DATASET_LOADER, fmt)
    ):
        return
    path = source.local_path
    probe_name = str(path)
    if is_parquet_input(path):
        try:
            probe_name = str(parquet_files(path)[0])
            record = first_record(path)
        except (OSError, ValueError):
            return
    elif path.is_dir() or path.suffix.lower() in COLUMNAR_SUFFIXES:
        return
    else:
        try:
            record = DatasetResolver._read_first_jsonl_record(str(path))
        except OSError:
            return
    if record is None:
        return
    LoaderClass = plugins.get_class(PluginType.CUSTOM_DATASET_LOADER, fmt)
    if not LoaderClass.can_load(record, probe_name):
        raise RemoteSourceError(
            ref=source.probe.ref.display, kind=RemoteErrorKind.INCOMPATIBLE, stage=RemoteStage.MATERIALIZE,
            what=f"its first record does not look like '{fmt}' data",
            hint="Check the manifest's format, or pass the right --custom-dataset-type",
        )  # fmt: skip


async def _pinned_probe(ref: RemoteRef) -> ProbeResult:
    """The access-stage probe for ``ref``, or a fresh probe that must match its version."""
    if (probe := _probe_cache.get(ref)) is not None:
        return probe
    if ref.resolved_version is None:
        raise RuntimeError(
            f"{ref.display} was never resolved; the dataset access preflight must run first"
        )
    probe = await service.probe_ref(ref)
    if probe.ref.resolved_version != ref.resolved_version:
        raise changed_after_pin(ref.display, RemoteStage.MATERIALIZE, "the data")
    return probe


async def _materialize(groups: dict[str, list[BenchmarkConfig]]) -> None:
    for cfgs in groups.values():
        ref: RemoteRef = cfgs[0].get_default_dataset().path
        probe = await _pinned_probe(ref)
        source = await service.materialize_probe(probe)
        formats = {_declared_format(cfg.get_default_dataset()) for cfg in cfgs}
        for fmt in formats - {None}:
            await asyncio.to_thread(_check_first_record_format, fmt, source)
        total = sum(f.size_bytes for f in source.files)
        digest = (
            f", sha256 {source.files[0].sha256[:12]}" if len(source.files) == 1 else ""
        )
        _logger.info(
            f"Remote dataset {ref.display}: using {probe.pinned_ref} "
            f"(version {ref.resolved_version}, {len(source.files)} file(s), {total} bytes{digest})"
        )


def preflight_remote_materialize(configs: Sequence[BenchmarkConfig]) -> None:
    """Download every remote dataset at its pinned version into the local cache.

    Raises:
        RemoteSourceError: The download failed, does not match its manifest, or
            changed after the access stage pinned it.
    """
    groups = _remote_groups(configs)
    if groups:
        asyncio.run(_materialize(groups))
