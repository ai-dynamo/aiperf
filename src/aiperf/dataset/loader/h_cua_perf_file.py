# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import orjson
import zstandard

from aiperf.common.exceptions import DatasetLoaderError
from aiperf.dataset.generator.prompt import PromptGenerator
from aiperf.dataset.loader.base_loader import LoaderProbeData
from aiperf.dataset.loader.h_cua_perf_processing import (
    HCuaPerfFilters,
    iter_selected_records,
    manifest_path,
    memory_shortfall,
    open_dataset,
    reject_ignore_trace_delays,
    verify_trace,
)
from aiperf.dataset.loader.mooncake_trace import MooncakeTraceDatasetLoader

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


class HCuaPerfFileLoader(MooncakeTraceDatasetLoader):
    """Replay an H CUA Perf build from disk.

    ``--input-file <name>.jsonl.zst --custom-dataset-type h_cua_perf`` reads the
    trace beside its ``<name>.meta.json`` manifest, which must carry the trace's
    sha256. The records are the Mooncake rows the ``h_cua_perf`` public dataset
    downloads from the Hub, so the replay is the same; a build derived with the
    dataset's ``trace_processor.py`` already carries its shape, which is why no
    ``--dataset-filter`` applies here.
    """

    tag: ClassVar[str] = "HCuaPerf"

    def __init__(
        self,
        *,
        filename: str | Path | None = None,
        prompt_generator: PromptGenerator,
        run: BenchmarkRun | None = None,
        default_block_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        if filename is None:
            raise DatasetLoaderError(
                f"{self.tag}: inline records are not supported; "
                "pass --input-file <build>.jsonl.zst"
            )
        super().__init__(
            filename=filename,
            prompt_generator=prompt_generator,
            run=run,
            default_block_size=default_block_size,
            **kwargs,
        )
        reject_ignore_trace_delays(self.tag, run)
        self.manifest = manifest_path(self.filename)
        if not self.manifest.is_file():
            raise DatasetLoaderError(
                f"{self.tag}: {self.manifest} not found beside {self.filename.name}"
            )
        self._plan: dict[str, int] = {}

    @classmethod
    def can_load(
        cls, data: LoaderProbeData | None = None, filename: str | Path | None = None
    ) -> bool:
        """Never claim a file: selected by name only.

        The records are valid Mooncake rows, so claiming them would make every
        Mooncake file ambiguous and break structural detection for
        ``mooncake_trace``.
        """
        return False

    def _init_trace_scope(self) -> None:
        """One read of the trace serves both the manifest check and the hash_ids scope.

        The base class would hash the file again for ``trace_id``; on a 2.5 GB
        build that is a second full pass before the decompressing read.
        """
        try:
            meta = orjson.loads(self.manifest.read_bytes())
            self._plan = meta["session_turns"]
            if shortfall := memory_shortfall(meta, self._plan):
                self.warning(f"{self.tag}: {shortfall}; derive a smaller build")
            self._trace_id = verify_trace(meta, self.filename)[:16]
        except (KeyError, ValueError) as e:
            raise DatasetLoaderError(f"{self.tag}: {e}") from e
        self.prompt_generator._hash_id_corpus_rng.set_trace_id(self._trace_id)
        self.prompt_generator._cache.clear()

    def _iter_record_dicts(
        self, source: str | Path | None = None
    ) -> Iterator[dict[str, Any]]:
        try:
            with open_dataset(self.filename) as lines:
                records = (orjson.loads(line) for line in lines if line.strip())
                yield from iter_selected_records(records, self._plan, HCuaPerfFilters())
        except (KeyError, ValueError, zstandard.ZstdError) as e:
            raise DatasetLoaderError(f"{self.tag}: {e}") from e
