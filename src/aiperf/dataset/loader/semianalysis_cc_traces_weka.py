# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HF-backed Weka trace loader.

Pulls the SemiAnalysis cc-traces-weka-no-subagents-051226 dataset from
HuggingFace and delegates reconstruction to ``WekaTraceLoader`` so file-based
and HF-based replay use the EXACT same backing code (same serial + parallel
paths, same hash_id replay, same model mapping, same branch / spawn-join,
same delay capping). The public loader's only job is "download + parse rows
into WekaTrace + delegate".
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from pydantic import ValidationError

from aiperf.common.config.user_config import UserConfig
from aiperf.common.exceptions import DatasetLoaderError
from aiperf.common.models import Conversation
from aiperf.dataset.generator.prompt import PromptGenerator
from aiperf.dataset.loader.base_hf_dataset import BaseHFDatasetLoader
from aiperf.dataset.loader.weka_trace import WekaTraceLoader
from aiperf.dataset.loader.weka_trace_models import WekaTrace
from aiperf.plugin.enums import DatasetSamplingStrategy


class SemiAnalysisCCTracesWekaLoader(BaseHFDatasetLoader):
    """HF-backed Weka trace loader.

    Downloads ``semianalysisai/cc-traces-weka-no-subagents-051226`` (2.77 GB
    jsonl, 949 traces, 136k requests, public, no auth), validates each row as
    a ``WekaTrace``, and delegates conversation reconstruction to
    :class:`WekaTraceLoader`. File-based and HF-based replay are guaranteed
    byte-identical because they share one method body.
    """

    tag: ClassVar[str] = "SemiAnalysisCCTracesWeka"

    def __init__(
        self,
        *,
        user_config: UserConfig,
        hf_dataset_name: str,
        hf_split: str = "train",
        hf_subset: str | None = None,
        prompt_generator: PromptGenerator | None = None,
        default_block_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        # Hard-coded streaming=False: full corpus upfront. The dataset is
        # small enough for HF's local cache to make re-runs near-instant,
        # and trace replay is designed to be a whole-corpus benchmark.
        kwargs.pop("streaming", None)
        super().__init__(
            user_config=user_config,
            hf_dataset_name=hf_dataset_name,
            hf_split=hf_split,
            hf_subset=hf_subset,
            streaming=False,
            **kwargs,
        )
        self._weka = WekaTraceLoader(
            filename=None,
            user_config=user_config,
            prompt_generator=prompt_generator,
            default_block_size=default_block_size,
        )

    async def load_dataset(self) -> dict[str, list[WekaTrace]]:
        """Download the HF dataset and validate every row as a WekaTrace.

        Caps the number of rows to ``--num-dataset-entries`` (defaults to 100)
        to avoid reconstructing the full 949-trace corpus when the benchmark
        only needs a subset. Pass ``--num-dataset-entries 949`` (or higher)
        to load the full corpus. The no-subagents corpus has subagent entries
        stripped, so each row produces exactly one main-agent conversation
        downstream (no parent / child fan-out).
        """
        raw = await super().load_dataset()
        ds = raw["dataset"]
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._validate_rows, ds)

    def _validate_rows(self, ds: Any) -> dict[str, list[WekaTrace]]:
        total_rows = len(ds)
        cap = self.user_config.input.conversation.num_dataset_entries
        n_rows = min(cap, total_rows)
        if n_rows < total_rows:
            ds = ds.select(range(n_rows))
            self.info(
                f"Loading {n_rows}/{total_rows} traces "
                f"(--num-dataset-entries={cap}; pass a higher value to load "
                f"more, up to {total_rows})"
            )
        else:
            self.info(f"Loading all {total_rows} traces")

        out: dict[str, list[WekaTrace]] = {}
        for i, row in enumerate(ds):
            try:
                trace = WekaTrace.model_validate(row)
            except ValidationError as e:
                raise DatasetLoaderError(
                    f"Row {i} of {self.hf_dataset_name} failed WekaTrace "
                    f"validation: {e}"
                ) from e
            if trace.id in out:
                raise DatasetLoaderError(
                    f"Duplicate trace id '{trace.id}' at row {i} of "
                    f"{self.hf_dataset_name}"
                )
            out[trace.id] = [trace]
        return out

    async def convert_to_conversations(
        self, data: dict[str, list[WekaTrace]]
    ) -> list[Conversation]:
        """Delegate to the file-based loader's reconstruction (same code path)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._weka.convert_to_conversations, data
        )

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        return DatasetSamplingStrategy.SEQUENTIAL
