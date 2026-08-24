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

    def _load_all_traces(self, ds: Any, total_rows: int) -> dict[str, list[WekaTrace]]:
        """Validate every HF row when no filter-then-cap constraints apply."""
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

    def _validate_rows(self, ds: Any) -> dict[str, list[WekaTrace]]:
        """Validate HF rows with filter-then-cap selection.

        Scans rows in order, drops traces whose peak context exceeds
        ``--max-context-length``, then keeps the first N *eligible* traces when
        ``--num-dataset-entries`` is explicit. Never caps the raw HF prefix
        before filtering (that was the silent 100→15 pool shrink).
        """
        from aiperf.dataset.loader.selection import (
            filter_then_cap,
            log_selection_summary,
        )
        from aiperf.dataset.loader.weka_trace import _trace_peak_context_length

        total_rows = len(ds)
        dataset = self.run.cfg.get_default_dataset()
        cap = getattr(dataset, "entries", None)
        # Key strictly off entries_explicit. --request-count / --num-conversations
        # fallbacks populate `entries` (so it lands in model_fields_set) but the
        # converter pins _entries_explicit=False for them; a direct YAML `entries`
        # is promoted to entries_explicit=True by _resolve_entries_explicit. The
        # old `or ("entries" in model_fields_set ...)` clause defeated that
        # distinction and silently capped the corpus to the request-count prefix.
        explicit_cap = bool(getattr(dataset, "entries_explicit", False))
        num_entries = cap if explicit_cap else None
        max_ctx = getattr(dataset, "max_context_length", None)
        synthesis = getattr(dataset, "synthesis", None)
        max_osl = getattr(synthesis, "max_osl", None) if synthesis else None

        def _candidates() -> Any:
            for i, row in enumerate(ds):
                try:
                    trace = WekaTrace.model_validate(row)
                except ValidationError as e:
                    raise DatasetLoaderError(
                        f"Row {i} of {self.hf_dataset_name} failed WekaTrace "
                        f"validation: {e}"
                    ) from e
                peak = _trace_peak_context_length(trace, max_osl=max_osl)
                yield (i, trace), peak

        if num_entries is None and max_ctx is None:
            return self._load_all_traces(ds, total_rows)

        kept_pairs, stats = filter_then_cap(
            _candidates(),
            num_dataset_entries=num_entries,
            max_context_length=max_ctx,
        )
        log_selection_summary(
            stats,
            source=self.hf_dataset_name,
            num_dataset_entries=num_entries,
            max_context_length=max_ctx,
        )
        if not kept_pairs:
            from aiperf.dataset.loader.selection import raise_on_empty_selection

            raise_on_empty_selection(
                stats,
                source=self.hf_dataset_name,
                num_dataset_entries=num_entries,
                max_context_length=max_ctx,
            )

        out = {}
        for i, trace in kept_pairs:
            if trace.id in out:
                raise DatasetLoaderError(
                    f"Duplicate trace id '{trace.id}' at row {i} of "
                    f"{self.hf_dataset_name}"
                )
            out[trace.id] = [trace]
        self.info(
            f"Loaded {len(out)}/{total_rows} eligible traces "
            f"(filter-then-cap; --num-dataset-entries={num_entries}, "
            f"--max-context-length={max_ctx})"
        )
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
