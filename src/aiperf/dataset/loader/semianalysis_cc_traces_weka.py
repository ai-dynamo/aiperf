# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HF-backed Weka trace loader.

Pulls a SemiAnalysis cc-traces-weka dataset variant from HuggingFace and
delegates reconstruction to ``WekaTraceLoader`` so file-based and HF-based
replay use the EXACT same backing code (same serial + parallel paths, same
hash_id replay, same model mapping, same branch / spawn-join, same delay
capping). The public loader's only job is "download + parse rows into
WekaTrace + delegate".

Many corpus variants are registered against this class in
``plugins.yaml`` (the ``semianalysis_cc_traces_weka*`` entries). For
example:

* ``semianalysis_cc_traces_weka`` and
  ``semianalysis_cc_traces_weka_no_subagents`` both map to
  ``semianalysisai/cc-traces-weka-no-subagents-051826`` (98 traces,
  v5-only, CC ≥ 2.1.139, subagent blocks stripped, ≥20 main-agent
  turns per trace).
* the date-pinned with-subagents variants (e.g.
  ``semianalysis_cc_traces_weka_062126`` and its ``_256k`` sibling)
  carry full parent + Task-tool subagent fan-out; ``062126`` is the
  current default AgentX corpus, and older date pins such as ``061526``
  remain as reproducibility aliases.

Which dataset is downloaded is governed by the ``hf_dataset_name``
plugin metadata field; the loader itself is variant-agnostic.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import ValidationError

from aiperf.common.exceptions import DatasetLoaderError
from aiperf.common.models import Conversation
from aiperf.dataset.generator.prompt import PromptGenerator
from aiperf.dataset.loader.base_hf_dataset import BaseHFDatasetLoader
from aiperf.dataset.loader.weka_trace import WekaTraceLoader
from aiperf.dataset.loader.weka_trace_models import WekaTrace
from aiperf.plugin.enums import DatasetSamplingStrategy

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


class SemiAnalysisCCTracesWekaLoader(BaseHFDatasetLoader):
    """HF-backed Weka trace loader.

    Downloads a ``semianalysisai/cc-traces-weka-*`` dataset (selected via
    the ``hf_dataset_name`` plugin metadata field), validates each row as
    a ``WekaTrace``, and delegates conversation reconstruction to
    :class:`WekaTraceLoader`. File-based and HF-based replay are
    guaranteed byte-identical because they share one method body.

    Many corpus variants are registered against this class (the
    ``semianalysis_cc_traces_weka*`` entries in plugins.yaml).
    ``semianalysis_cc_traces_weka`` and
    ``semianalysis_cc_traces_weka_no_subagents`` both map to the 051826
    no-subagents corpus (98 traces, v5-only + CC ≥ 2.1.139 filtered,
    main-agent linear streams only, ≥20 turns each); the date-pinned
    with-subagents variants carry full subagent fan-out. The loader
    code is identical for all of them — only ``hf_dataset_name`` differs.
    """

    tag: ClassVar[str] = "SemiAnalysisCCTracesWeka"

    def __init__(
        self,
        *,
        run: BenchmarkRun | None = None,
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
            run=run,
            hf_dataset_name=hf_dataset_name,
            hf_split=hf_split,
            hf_subset=hf_subset,
            streaming=False,
            **kwargs,
        )
        self._weka = WekaTraceLoader(
            filename=None,
            run=self.run,
            prompt_generator=prompt_generator,
            default_block_size=default_block_size,
        )

    async def load_dataset(self) -> dict[str, list[WekaTrace]]:
        """Download the HF dataset and validate every row as a WekaTrace.

        When ``--num-dataset-entries`` is not explicitly set, loads the
        full corpus. When it is set, caps at that value. For variants with
        subagents, each row produces 1 parent conversation plus 1 child
        conversation per subagent, so N rows typically yields 2-10x N
        conversations downstream; for the no-subagents variant the
        row-to-conversation ratio is ~1:1.
        """
        raw = await super().load_dataset()
        ds = raw["dataset"]
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._validate_rows, ds)

    def _validate_rows(self, ds: Any) -> dict[str, list[WekaTrace]]:
        total_rows = len(ds)
        # Auto-load full corpus unless --num-dataset-entries (-> dataset.entries)
        # was explicitly set. model_fields_set is the only reliable signal that
        # the user pinned a row cap (a default value is indistinguishable).
        dataset = self.run.cfg.get_default_dataset()
        cap = getattr(dataset, "entries", None)
        explicit_cap = "entries" in dataset.model_fields_set and cap is not None
        n_rows = min(cap, total_rows) if explicit_cap else total_rows
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
