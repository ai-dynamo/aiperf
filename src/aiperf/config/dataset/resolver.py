# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""File-dataset resolver.

Imported and re-exported by ``resolvers`` so callers and test patches that
reference ``aiperf.config.resolution.resolvers.DatasetResolver`` continue
to work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aiperf.common.aiperf_logger import AIPerfLogger

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

_logger = AIPerfLogger(__name__)

# FileDataset fields consumed only by the baseten_trace loader. The
# convert-time guard can reject them only when --custom-dataset-type is
# explicit; with an auto-detected format they would otherwise silently
# no-op (e.g. --replay-speedup on a mooncake JSONL replays at recorded
# speed), so warn once the resolved format is known.
_BASETEN_ONLY_REPLAY_FIELDS = (
    "trace_session_sample_ratio",
    "replay_speedup",
    "max_idle_gap_cap_seconds",
    "open_loop_replay",
    "open_loop_strict",
    "omit_kv_hints",
    "force_min_tokens",
)


def _warn_ignored_baseten_only_fields(
    name: str, ds: object, dataset_type: object
) -> None:
    """Warn when baseten_trace-only replay knobs are set on another loader."""
    from aiperf.plugin.enums import CustomDatasetType

    if dataset_type is None or dataset_type == CustomDatasetType.BASETEN_TRACE:
        return
    fields_set = getattr(ds, "model_fields_set", set())
    ignored = [
        f
        for f in _BASETEN_ONLY_REPLAY_FIELDS
        if f in fields_set and getattr(ds, f) is not None
    ]
    if ignored:
        _logger.warning(
            f"Dataset '{name}' resolved to {dataset_type}, which ignores the "
            f"baseten_trace-only replay option(s): {', '.join(ignored)}."
        )


@dataclass(slots=True)
class _DatasetResolution:
    """Accumulator for per-dataset resolution output."""

    paths: dict[str, object] = field(default_factory=dict)
    types: dict = field(default_factory=dict)
    sampling: dict = field(default_factory=dict)
    has_timing: dict[str, bool] = field(default_factory=dict)
    total_records: dict[str, int] = field(default_factory=dict)
    session_counts: dict[str, int] = field(default_factory=dict)
    root_counts: dict[str, int] = field(default_factory=dict)
    is_forking: dict[str, bool] = field(default_factory=dict)


class DatasetResolver:
    """Resolve file-based dataset paths, detect types, timing, and sampling."""

    def resolve(self, run: BenchmarkRun) -> None:
        """Populate dataset-derived fields on ``run.resolved``.

        Resolves file dataset paths, maps configured formats to loader dataset types,
        records loader-preferred sampling, detects first-record timing fields for
        ``fixed_schedule`` validation, and counts records/sessions. Raises
        ``FileNotFoundError`` when a file dataset path does not exist.
        """
        from aiperf.config.dataset import FileDataset, PublicDataset
        from aiperf.dataset.graph.workload_detect import resolve_graph_workload
        from aiperf.plugin import plugins

        acc = _DatasetResolution()
        format_map = self._build_format_map()

        for ds in run.cfg.datasets:
            if isinstance(ds, PublicDataset):
                metadata = plugins.get_public_dataset_loader_metadata(ds.dataset)
                acc.has_timing[ds.name] = metadata.has_timing_data
                continue
            if not isinstance(ds, FileDataset):
                continue
            self._resolve_one(
                run=run, name=ds.name, ds=ds, format_map=format_map, acc=acc
            )

        self._publish(run, acc)
        # Eagerly resolve-and-memoize graph-ness so single-run child processes
        # inherit it via the pickled run and never re-walk the registry. A
        # no-op when a `_resolve_one` site already populated it; for
        # synthetic/public/inline runs this stores None WITH the resolved
        # marker ("not a graph run", not "never checked").
        resolve_graph_workload(run)

    @staticmethod
    def _publish(run: BenchmarkRun, acc: _DatasetResolution) -> None:
        if acc.paths:
            run.resolved.dataset_file_paths = acc.paths  # type: ignore[assignment]
        if acc.types:
            run.resolved.dataset_types = acc.types
            run.resolved.dataset_sampling_strategies = acc.sampling
        if acc.has_timing:
            run.resolved.dataset_has_timing_data = acc.has_timing
        if acc.total_records:
            run.resolved.dataset_total_records = acc.total_records
            run.resolved.dataset_session_count = acc.session_counts
        if acc.is_forking:
            run.resolved.dataset_is_forking = acc.is_forking
        if acc.root_counts:
            run.resolved.dataset_root_count = acc.root_counts
        if acc.paths or acc.types:
            _logger.debug(
                f"Resolved {len(acc.paths)} dataset paths, {len(acc.types)} types"
            )

    def _resolve_one(
        self,
        *,
        run: BenchmarkRun,
        name: str,
        ds: object,
        format_map: dict[str, object],
        acc: _DatasetResolution,
    ) -> None:
        records = getattr(ds, "records", None)
        if records is not None:
            self._resolve_inline(name=name, ds=ds, format_map=format_map, acc=acc)
            return

        # A weka HuggingFace ``org/name`` repo id is a graph workload reference,
        # not a local path (resolved from the hub/cache by the weka adapter; the
        # graph-store planes own its record counting), so skip file resolution.
        # Populate the graph memo here too: this branch never reaches the main
        # graph check below, and the accessor stores the RAW ``org/name`` id
        # verbatim (never .resolve()d -- HF ids are not filesystem paths).
        from pathlib import Path

        from aiperf.dataset.graph.adapters.weka.trace import (
            _looks_like_hf_dataset_id,
        )
        from aiperf.dataset.graph.workload_detect import (
            is_graph_workload_path,
            resolve_graph_workload,
        )

        is_default_dataset = ds is run.cfg.get_default_dataset()
        raw_path = ds.path  # type: ignore[attr-defined]
        if raw_path is not None and _looks_like_hf_dataset_id(raw_path):
            acc.paths[name] = Path(str(raw_path))
            if is_default_dataset:
                resolve_graph_workload(run)
            return

        # 1. Resolve and validate path
        resolved = raw_path.resolve()  # type: ignore[attr-defined]
        if not resolved.exists():
            raise FileNotFoundError(f"Dataset '{name}' file not found: {resolved}")
        acc.paths[name] = resolved

        # A local graph workload (dynamo/weka trace file, segmented dir) is
        # owned by the graph-store build/schedule planes (parse_graph_workload),
        # not the custom-dataset-loader path. Skip type detection + record
        # counting, which text-mode read the file and crash on a .gz binary.
        # Forced via `--graph-format`, OR auto-detected as a trace adapter.
        # `graph_format` covers the `native`/ambiguous case that the path-level
        # content sniff (which excludes `native`) cannot see. For the default
        # dataset the check IS the accessor, so graph-ness is derived (and the
        # file sniffed) exactly once and memoized onto ``run.resolved``.
        if is_default_dataset:
            if resolve_graph_workload(run) is not None:
                return
        elif getattr(ds, "graph_format", None) is not None or is_graph_workload_path(
            resolved
        ):
            return

        # 2. Detect dataset type from explicit format or via can_load.
        # Pydantic defaults ``format`` to SINGLE_TURN, so a falsy check isn't
        # enough — when the user didn't *explicitly* set format, fall back to
        # structural detection so loaders like sagemaker_data_capture (whose
        # JSONL doesn't look like single-turn) are recognized here the same
        # way the composer recognizes them at load time.
        fmt = ds.format  # type: ignore[attr-defined]
        fields_set = getattr(ds, "model_fields_set", set())
        first_record = None
        explicit_format = "format" in fields_set
        dataset_type = format_map.get(str(fmt)) if explicit_format and fmt else None
        if dataset_type is None:
            dataset_type, first_record = self._detect_type(str(resolved))

        if dataset_type is not None:
            acc.types[name] = dataset_type
            acc.sampling[name] = self._resolve_sampling(ds, dataset_type)
            acc.has_timing[name] = self._check_timing_data(
                str(resolved), first_record, dataset_type
            )
        _warn_ignored_baseten_only_fields(name, ds, dataset_type)

        # 3. Count records and sessions (for validation and fixed_schedule)
        if not resolved.is_dir():
            records, sessions = self._count_records_and_sessions(
                str(resolved), dataset_type
            )
            acc.total_records[name] = records
            acc.session_counts[name] = sessions

        # 4. Forking-dataset analysis (DAG roots) — only dag_jsonl today.
        from aiperf.plugin.enums import CustomDatasetType

        is_forking = dataset_type == CustomDatasetType.DAG_JSONL
        acc.is_forking[name] = is_forking
        if is_forking and not resolved.is_dir():
            acc.root_counts[name] = self._count_dag_roots(str(resolved))

    @staticmethod
    def _resolve_inline(
        *,
        name: str,
        ds: object,
        format_map: dict[str, object],
        acc: _DatasetResolution,
    ) -> None:
        """Resolve dataset metadata for an inline records source.

        Inline mode relies on the ``format:`` field; Pydantic defaults to
        SINGLE_TURN, so every config has a value that lands in the format_map.
        No path is set.
        """
        from aiperf.plugin.enums import CustomDatasetType

        records = ds.records  # type: ignore[attr-defined]
        fmt = ds.format  # type: ignore[attr-defined]
        dataset_type = format_map.get(str(fmt))
        if dataset_type is not None:
            acc.types[name] = dataset_type
            acc.sampling[name] = DatasetResolver._resolve_sampling(ds, dataset_type)
        _warn_ignored_baseten_only_fields(name, ds, dataset_type)

        # Count records (sum across pools if multi-pool).
        if isinstance(records, dict):
            total = sum(len(v) for v in records.values())
            first_pool = next(iter(records.values()), None)
            first = first_pool[0] if first_pool else None
        else:
            total = len(records)
            first = records[0] if records else None
        acc.total_records[name] = total

        # Sessions: for multi_turn / bailian_trace, count session_ids; otherwise 1:1.
        is_multi_turn = dataset_type in (
            CustomDatasetType.MULTI_TURN,
            CustomDatasetType.BAILIAN_TRACE,
        )
        if is_multi_turn:
            sids: set[str] = set()
            iterables = records.values() if isinstance(records, dict) else [records]
            for items in iterables:
                for r in items:
                    sid = r.get("session_id") or r.get("chat_id")
                    if sid is not None:
                        sids.add(str(sid))
            acc.session_counts[name] = len(sids) if sids else total
        else:
            acc.session_counts[name] = total

        # Timing: detect from the first record's timestamp/delay fields.
        acc.has_timing[name] = bool(
            first is not None
            and (first.get("timestamp") is not None or first.get("delay") is not None)
        )

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

        mapping: dict[str, object] = {}
        for fmt in DatasetFormat:
            try:
                mapping[str(fmt)] = CustomDatasetType(str(fmt))
            except ValueError:
                continue
        return mapping

    @staticmethod
    def _read_first_jsonl_record(file_path: str) -> dict | None:
        """Return the first JSON-object line of ``file_path`` as a dict.

        Returns ``None`` for any of: an empty file, a binary or non-UTF-8
        file (the text iterator raises ``UnicodeDecodeError``), a
        non-JSON first line (e.g. BurstGPT's CSV header), or a first
        line that parses as valid JSON but isn't an object (a list,
        string, or number). All of these are expected probe outcomes —
        the caller falls through to structural detection so each
        loader's ``can_load`` gets a chance.

        Uses ``orjson.loads`` directly rather than ``load_json_str``
        because the non-JSON case is expected here; ``load_json_str``
        would log a misleading "Failed to parse JSON string" warning on
        every successful CSV auto-detect. Lets ``OSError`` propagate so
        callers can distinguish "can't read the file at all" from "read
        it but the first line isn't a JSON object".
        """
        import orjson

        try:
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    if line := line.strip():
                        try:
                            parsed = orjson.loads(line)
                        except orjson.JSONDecodeError:
                            return None
                        return parsed if isinstance(parsed, dict) else None
        except UnicodeDecodeError:
            return None
        return None

    @staticmethod
    def _detect_type(
        file_path: str,
    ) -> tuple[object | None, dict | None]:
        """Auto-detect dataset type by querying registered loaders.

        Returns (detected_type, first_record) so the caller can reuse
        the already-parsed first line for timing data detection.
        """
        from pathlib import Path

        from aiperf.plugin import plugins
        from aiperf.plugin.enums import CustomDatasetType, PluginType

        path = Path(file_path)
        data: dict | None = None
        if not path.is_dir():
            try:
                data = DatasetResolver._read_first_jsonl_record(file_path)
            except OSError:
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
                    _logger.warning(
                        f"Multiple loaders match dataset '{file_path}', skipping auto-detection"
                    )
                    return None, data
                detected = CustomDatasetType(entry.name)
        return detected, data

    @staticmethod
    def _check_timing_data(
        file_path: str,
        first_record: dict | None,
        dataset_type: object | None = None,
    ) -> bool:
        """Check whether the first record carries timing information.

        Most trace formats expose ``timestamp`` or ``delay`` at the top level,
        but sagemaker_data_capture nests its timing under
        ``eventMetadata.inferenceTime``. We branch per-loader so fixed_schedule
        validation accepts every format whose loader actually produces timing.
        """
        from aiperf.plugin.enums import CustomDatasetType

        if dataset_type == CustomDatasetType.SAGEMAKER_DATA_CAPTURE:
            return True
        if dataset_type == CustomDatasetType.BURST_GPT_TRACE:
            # BurstGPT is CSV; the loader enforces a ``Timestamp`` column at
            # load time (see ``BurstGPTTraceDatasetLoader._REQUIRED_COLUMNS``),
            # so the dataset cannot load without timing.
            return True
        if dataset_type == CustomDatasetType.BASETEN_TRACE:
            return True

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

        if dataset_type == CustomDatasetType.BASETEN_TRACE:
            from aiperf.dataset.loader.baseten_trace import (
                count_baseten_parquet_records_and_sessions,
            )

            return count_baseten_parquet_records_and_sessions(file_path)

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

    @staticmethod
    def _count_dag_roots(file_path: str) -> int:
        """Count root sessions (not referenced by any fork/spawn) in a dag_jsonl file.

        Roots are the entries the DAG loader actually samples standalone;
        non-root children are seeded into the orchestrator from their parent
        worker. Sizing ``num_conversations`` by total record count would
        over-run a file with deep fanout (e.g. 1 root + 2 children = 3
        records should default to 1 conversation, not 3).
        """
        try:
            all_ids, referenced = _collect_dag_session_and_fork_ids(file_path)
        except (OSError, FileNotFoundError) as err:
            _logger.error(
                f"Cannot read dag_jsonl file {file_path} for root counting: {err}"
            )
            return 0
        return len(all_ids - referenced)


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


def _collect_pre_session_refs(data: dict, into: set[str]) -> None:
    """Add ``pre_session_spawns`` child ids (bare strings only) into ``into``."""
    for child in data.get("pre_session_spawns", []) or []:
        if isinstance(child, str):
            into.add(child)


def _collect_turn_refs(turn: dict, into: set[str]) -> None:
    """Add child ids referenced from one turn's ``forks``/``spawns`` into ``into``.

    ``forks`` entries are a bare ``"<sid>"`` or a ``{"child": "<sid>", ...}``
    object. ``spawns`` entries are a bare ``"<sid>"`` or a
    ``{"children": [...], ...}`` object (DagSpawn form).
    """
    for fork_entry in turn.get("forks", []) or []:
        if isinstance(fork_entry, str):
            into.add(fork_entry)
        elif isinstance(fork_entry, dict):
            child = fork_entry.get("child")
            if isinstance(child, str):
                into.add(child)
    for spawn_entry in turn.get("spawns", []) or []:
        if isinstance(spawn_entry, str):
            into.add(spawn_entry)
        elif isinstance(spawn_entry, dict):
            for child in spawn_entry.get("children", []) or []:
                if isinstance(child, str):
                    into.add(child)


def _collect_dag_session_and_fork_ids(file_path: str) -> tuple[set[str], set[str]]:
    """Walk a dag_jsonl file once, returning ``(all_session_ids, referenced_ids)``.

    ``referenced_ids`` covers every id the orchestrator dispatches as a child
    of another conversation: bare-string and object-form ``forks`` entries,
    bare-string and ``DagSpawn``-object ``spawns`` entries, and top-level
    ``pre_session_spawns``. Anything in ``referenced_ids`` is NOT a root and
    must not be sampled standalone.
    """
    from aiperf.common.utils import load_json_str

    all_ids: set[str] = set()
    referenced_ids: set[str] = set()
    with open(file_path) as f:
        for raw in f:
            if not (line := raw.strip()):
                continue
            try:
                data = load_json_str(line)
            except (ValueError, TypeError):
                continue
            sid = data.get("session_id")
            if isinstance(sid, str):
                all_ids.add(sid)
            _collect_pre_session_refs(data, referenced_ids)
            for turn in data.get("turns", []) or []:
                if isinstance(turn, dict):
                    _collect_turn_refs(turn, referenced_ids)
    return all_ids, referenced_ids
