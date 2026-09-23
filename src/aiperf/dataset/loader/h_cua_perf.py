# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import orjson
from pydantic import ValidationError

from aiperf.common.enums import ConversationContextMode
from aiperf.common.exceptions import DatasetLoaderError
from aiperf.common.models import Conversation
from aiperf.dataset.generator.prompt import PromptGenerator
from aiperf.dataset.loader.base_hf_dataset import BaseHFDatasetLoader
from aiperf.dataset.loader.h_cua_perf_processing import (
    HCuaPerfFilters,
    iter_selected_records,
    open_dataset,
    select_trace_lengths,
    supported_filter_keys,
)
from aiperf.dataset.loader.models import MooncakeTrace
from aiperf.dataset.loader.mooncake_trace import MooncakeTraceDatasetLoader
from aiperf.plugin.enums import DatasetSamplingStrategy

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


class HCuaPerfDatasetLoader(BaseHFDatasetLoader):
    """Download ``Hcompany/h_cua_perf`` and replay it as a Mooncake trace."""

    tag: ClassVar[str] = "HCuaPerf"
    hf_filename: ClassVar[str] = "h_cua.jsonl.zst"
    hf_manifest_filename: ClassVar[str] = "h_cua.meta.json"

    def __init__(
        self,
        *,
        run: BenchmarkRun | None = None,
        hf_dataset_name: str,
        hf_split: str = "train",
        hf_subset: str | None = None,
        prompt_generator: PromptGenerator,
        default_block_size: int | None = None,
        filters: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        # The corpus is one file fetched whole; HF streaming mode does not apply.
        kwargs.pop("streaming", None)
        super().__init__(
            run=run,
            hf_dataset_name=hf_dataset_name,
            hf_split=hf_split,
            hf_subset=hf_subset,
            streaming=False,
            **kwargs,
        )
        dataset = run.cfg.get_default_dataset() if run is not None else None
        if getattr(dataset, "ignore_trace_delays", False):
            raise DatasetLoaderError(
                f"{self.tag}: --ignore-trace-delays applies to the Weka loaders only; "
                "use --inter-turn-delay-cap-seconds 0 to send each session's turns "
                "back to back"
            )
        try:
            self.filters = HCuaPerfFilters.model_validate(filters or {})
        except ValidationError as e:
            raise DatasetLoaderError(
                f"Invalid h_cua_perf dataset filters: {e}; supported keys: "
                f"{supported_filter_keys()}"
            ) from e
        self._mooncake_kwargs = {
            "run": self.run,
            "prompt_generator": prompt_generator,
            "default_block_size": default_block_size,
        }
        self._mooncake: MooncakeTraceDatasetLoader | None = None

    def _load_hf_dataset(self) -> tuple[Path, Path]:
        """Fetch the manifest and the trace file into the Hub cache."""
        # hf_hub_download caches the 2.5 GB file across runs; datasets.load_dataset
        # would re-stream it every run (streaming) or write a 9 GB Arrow copy.
        from huggingface_hub import hf_hub_download

        manifest, trace = (
            Path(
                hf_hub_download(
                    repo_id=self.hf_dataset_name,
                    filename=filename,
                    repo_type="dataset",
                    revision=self.hf_revision,
                )
            )
            for filename in (self.hf_manifest_filename, self.hf_filename)
        )
        return manifest, trace

    def _explicit_entries(self) -> int | None:
        """``--num-dataset-entries`` when set explicitly; request-count fallbacks must not cap."""
        dataset = self.run.cfg.get_default_dataset()
        if not getattr(dataset, "entries_explicit", False):
            return None
        return getattr(dataset, "entries", None)

    def _read_records(self, trace: Path, plan: dict[str, int]) -> list[dict[str, Any]]:
        with open_dataset(trace) as lines:
            records = (orjson.loads(line) for line in lines if line.strip())
            return list(
                iter_selected_records(records, plan, self.filters.n_screenshots)
            )

    async def load_dataset(self) -> dict[str, list[MooncakeTrace]]:
        """Download, plan the trajectory selection, read that far, delegate the rest."""
        manifest, trace = (await super().load_dataset())["dataset"]
        loop = asyncio.get_running_loop()
        try:
            session_turns = orjson.loads(manifest.read_bytes())["session_turns"]
            plan = select_trace_lengths(
                self.filters, session_turns, first_n=self._explicit_entries()
            )
            records = await loop.run_in_executor(None, self._read_records, trace, plan)
        except (KeyError, ValueError) as e:
            raise DatasetLoaderError(f"{self.tag}: {e}") from e

        self._mooncake = MooncakeTraceDatasetLoader(
            inline_records=records, **self._mooncake_kwargs
        )
        data = await loop.run_in_executor(None, self._mooncake.load_dataset)
        self.info(
            f"Selected {len(data)}/{len(session_turns)} trajectories "
            f"({len(records):,} requests) from {self.hf_dataset_name}"
        )
        return data

    async def convert_to_conversations(
        self, data: dict[str, list[MooncakeTrace]]
    ) -> list[Conversation]:
        if self._mooncake is None:
            raise DatasetLoaderError(f"{self.tag}: load_dataset() must run first")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._mooncake.convert_to_conversations, data
        )

    @classmethod
    def get_default_context_mode(cls) -> ConversationContextMode | None:
        return MooncakeTraceDatasetLoader.get_default_context_mode()

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        return DatasetSamplingStrategy.SEQUENTIAL
