# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""File-dataset resolver.

Split out from ``resolvers.py`` to keep that module under the file-size limit.
Imported and re-exported by ``resolvers`` so callers and test patches that
reference ``aiperf.config.resolvers.DatasetResolver`` continue to work.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiperf.config.benchmark import BenchmarkRun

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _DatasetResolution:
    """Accumulator for per-dataset resolution output."""

    paths: dict[str, object] = field(default_factory=dict)
    types: dict = field(default_factory=dict)
    sampling: dict = field(default_factory=dict)
    has_timing: dict[str, bool] = field(default_factory=dict)
    total_records: dict[str, int] = field(default_factory=dict)
    session_counts: dict[str, int] = field(default_factory=dict)


class DatasetResolver:
    """Resolve file-based dataset paths, detect types, timing, and sampling."""

    def resolve(self, run: BenchmarkRun) -> None:
        from aiperf.config.dataset import FileDataset

        acc = _DatasetResolution()
        format_map = self._build_format_map()

        for name, ds in run.cfg.datasets.items():
            if not isinstance(ds, FileDataset):
                continue
            self._resolve_one(name=name, ds=ds, format_map=format_map, acc=acc)

        self._publish(run, acc)

    @staticmethod
    def _publish(run: BenchmarkRun, acc: _DatasetResolution) -> None:
        if acc.paths:
            run.resolved.dataset_file_paths = acc.paths  # type: ignore[assignment]
        if acc.types:
            run.resolved.dataset_types = acc.types
            run.resolved.dataset_sampling_strategies = acc.sampling
            run.resolved.dataset_has_timing_data = acc.has_timing
        if acc.total_records:
            run.resolved.dataset_total_records = acc.total_records
            run.resolved.dataset_session_count = acc.session_counts
        if acc.paths or acc.types:
            logger.debug(
                "Resolved %d dataset paths, %d types",
                len(acc.paths),
                len(acc.types),
            )

    def _resolve_one(
        self,
        *,
        name: str,
        ds: object,
        format_map: dict[str, object],
        acc: _DatasetResolution,
    ) -> None:
        # 1. Resolve and validate path
        resolved = ds.path.resolve()  # type: ignore[attr-defined]
        if not resolved.exists():
            raise FileNotFoundError(f"Dataset '{name}' file not found: {resolved}")
        acc.paths[name] = resolved

        # 2. Detect dataset type from explicit format or via can_load
        fmt = ds.format  # type: ignore[attr-defined]
        first_record = None
        dataset_type = format_map.get(str(fmt)) if fmt else None
        if dataset_type is None:
            dataset_type, first_record = self._detect_type(str(resolved))

        if dataset_type is not None:
            acc.types[name] = dataset_type
            acc.sampling[name] = self._resolve_sampling(ds, dataset_type)
            acc.has_timing[name] = self._check_timing_data(str(resolved), first_record)

        # 3. Count records and sessions (for validation and fixed_schedule)
        if not resolved.is_dir():
            records, sessions = self._count_records_and_sessions(
                str(resolved), dataset_type
            )
            acc.total_records[name] = records
            acc.session_counts[name] = sessions

    @staticmethod
    def _resolve_sampling(ds: object, dataset_type: object) -> object:
        """Pick the loader's preferred sampling unless the user set an explicit one."""
        from aiperf.plugin.enums import DatasetSamplingStrategy

        loader_sampling = DatasetResolver._get_preferred_sampling(dataset_type)
        ds_sampling = ds.sampling  # type: ignore[attr-defined]
        if (
            ds_sampling == DatasetSamplingStrategy.SEQUENTIAL
            and loader_sampling != DatasetSamplingStrategy.SEQUENTIAL
        ):
            return loader_sampling
        return ds_sampling

    @staticmethod
    def _build_format_map() -> dict[str, object]:
        from aiperf.common.enums import DatasetFormat
        from aiperf.plugin.enums import CustomDatasetType

        return {
            str(DatasetFormat.SINGLE_TURN): CustomDatasetType.SINGLE_TURN,
            str(DatasetFormat.MULTI_TURN): CustomDatasetType.MULTI_TURN,
            str(DatasetFormat.MOONCAKE_TRACE): CustomDatasetType.MOONCAKE_TRACE,
            str(DatasetFormat.RANDOM_POOL): CustomDatasetType.RANDOM_POOL,
        }

    @staticmethod
    def _detect_type(
        file_path: str,
    ) -> tuple[object | None, dict | None]:
        """Auto-detect dataset type by querying registered loaders.

        Returns (detected_type, first_record) so the caller can reuse
        the already-parsed first line for timing data detection.
        """
        from pathlib import Path

        from aiperf.common.utils import load_json_str
        from aiperf.plugin import plugins
        from aiperf.plugin.enums import CustomDatasetType, PluginType

        path = Path(file_path)
        if path.is_dir():
            data = None
        else:
            try:
                with open(file_path) as f:
                    for line in f:
                        if line := line.strip():
                            data = load_json_str(line)
                            break
                    else:
                        return None, None
            except (OSError, ValueError):
                return None, None

        # Check explicit type field in data
        if data is not None and data.get("type") in CustomDatasetType:
            explicit_type = CustomDatasetType(data["type"])
            LoaderClass = plugins.get_class(
                PluginType.CUSTOM_DATASET_LOADER, explicit_type
            )
            if LoaderClass.can_load(data, file_path):
                return explicit_type, data

        # Structural detection
        detected = None
        for entry, LoaderClass in plugins.iter_all(PluginType.CUSTOM_DATASET_LOADER):
            if LoaderClass.can_load(data, file_path):
                if detected is not None:
                    logger.warning(
                        "Multiple loaders match dataset '%s', skipping auto-detection",
                        file_path,
                    )
                    return None, data
                detected = CustomDatasetType(entry.name)
        return detected, data

    @staticmethod
    def _check_timing_data(file_path: str, first_record: dict | None) -> bool:
        """Check whether the first record has timestamp or delay fields.

        Inspects the actual data rather than assuming from dataset type,
        because trace formats like mooncake may omit timing fields.
        """
        record = first_record
        if record is None:
            from pathlib import Path

            from aiperf.common.utils import load_json_str

            if Path(file_path).is_dir():
                return False
            try:
                with open(file_path) as f:
                    for line in f:
                        if line := line.strip():
                            record = load_json_str(line)
                            break
            except (OSError, ValueError):
                return False

        if record is None:
            return False
        return record.get("timestamp") is not None or record.get("delay") is not None

    @staticmethod
    def _count_records_and_sessions(
        file_path: str, dataset_type: object | None
    ) -> tuple[int, int]:
        """Count total non-empty records and unique sessions in a JSONL file.

        For multi-turn datasets, sessions are identified by session_id or
        chat_id fields. For single-turn, each record is its own session.
        """
        from aiperf.plugin.enums import CustomDatasetType

        is_multi_turn = dataset_type in (
            CustomDatasetType.MULTI_TURN,
            CustomDatasetType.BAILIAN_TRACE,
        )
        record_count = 0
        session_ids: set[str] = set()

        try:
            with open(file_path) as f:
                for line in f:
                    if not (line := line.strip()):
                        continue
                    record_count += 1
                    if is_multi_turn:
                        _add_session_id(line, session_ids)
        except OSError:
            return 0, 0

        if is_multi_turn and session_ids:
            return record_count, len(session_ids)
        return record_count, record_count

    @staticmethod
    def _get_preferred_sampling(dataset_type: object) -> object:
        """Get the loader's preferred sampling strategy."""
        from aiperf.plugin import plugins
        from aiperf.plugin.enums import DatasetSamplingStrategy, PluginType

        try:
            LoaderClass = plugins.get_class(
                PluginType.CUSTOM_DATASET_LOADER, dataset_type
            )
            if hasattr(LoaderClass, "get_preferred_sampling_strategy"):
                return LoaderClass.get_preferred_sampling_strategy()
        except (KeyError, ValueError):
            pass
        return DatasetSamplingStrategy.SEQUENTIAL


def _add_session_id(line: str, session_ids: set[str]) -> None:
    """Parse a JSONL line and add its session_id/chat_id to the set."""
    from aiperf.common.utils import load_json_str

    try:
        data = load_json_str(line)
    except (ValueError, TypeError):
        return
    sid = data.get("session_id") or data.get("chat_id")
    if sid is not None:
        session_ids.add(str(sid))
