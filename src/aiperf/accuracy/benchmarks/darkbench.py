# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DarkBench loader (``apart/darkbench``, MIT): 660 zero-shot prompts, 6 patterns x 110.

Tasks are the dataset's hyphenated ``target`` labels (``brand-bias``); the
overseer key is the snake-case pattern id (``brand_bias``). The reference
scorer looks the hyphenated label up unmapped and so cannot score three of
the six patterns.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import orjson

from aiperf.accuracy.benchmarks._darkbench_patterns import DARK_PATTERNS, get_model_info
from aiperf.accuracy.benchmarks._datasets_compat import load_dataset
from aiperf.accuracy.models import BenchmarkProblem
from aiperf.common.environment import Environment
from aiperf.common.exceptions import ConfigurationError
from aiperf.common.mixins import AIPerfLoggerMixin

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

DATASET_NAME = "apart/darkbench"
TASKS: tuple[str, ...] = tuple(sorted(p.replace("_", "-") for p in DARK_PATTERNS))
# The reference sets no max_tokens; AIPerf's default of 100 would truncate answers.
GENERATION_SIZE = 4096


class DarkBenchBenchmark(AIPerfLoggerMixin):
    """Load DarkBench prompts; each ground truth packs the overseer's context as JSON."""

    def __init__(self, run: BenchmarkRun, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.run = run

    async def load_problems(
        self, tasks: list[str] | None, n_shots: int, enable_cot: bool
    ) -> list[BenchmarkProblem]:
        """Load prompts for ``tasks`` (all when empty); ``n_shots``/``enable_cot`` are ignored.

        Raises:
            ConfigurationError: unknown task, or the brand-bias company/model
                cannot be inferred and is not overridden.
        """
        wanted = set(tasks or [])
        if unknown := wanted - set(TASKS):
            raise ConfigurationError(
                f"unknown darkbench task(s) {sorted(unknown)}; valid tasks: {list(TASKS)}"
            )
        if n_shots or enable_cot:
            self.warning(
                "darkbench is zero-shot without CoT; ignoring n_shots/enable_cot"
            )
        # Only the brand-bias question names the model's developer; do not
        # require it for runs that exclude that task.
        company, model = (
            self._company_and_model()
            if not wanted or "brand-bias" in wanted
            else ("", "")
        )
        rows = await asyncio.to_thread(
            load_dataset,
            DATASET_NAME,
            split="train",
            revision=Environment.ACCURACY.DARKBENCH_REVISION,
        )
        problems: list[BenchmarkProblem] = []
        for row in rows:
            if wanted and row["target"] not in wanted:
                continue
            pattern = DARK_PATTERNS[row["target"].replace("-", "_")]
            ground_truth = {
                "pattern": pattern.id,
                "display_name": pattern.display_name,
                "description": pattern.description.format(company=company, model=model),
                "prompt": row["input"],
            }
            problems.append(
                BenchmarkProblem(
                    prompt=row["input"],
                    ground_truth=orjson.dumps(ground_truth).decode(),
                    task=row["target"],
                    metadata={"id": row["id"], "generation_size": GENERATION_SIZE},
                    raw_messages=[{"role": "user", "content": row["input"]}],
                )
            )
        return problems

    def _company_and_model(self) -> tuple[str, str]:
        settings = Environment.ACCURACY
        names = self.run.cfg.get_model_names()
        if len(names) > 1:
            # The brand-bias question is baked into the dataset for one model,
            # but requests are spread across all configured models.
            raise ConfigurationError(
                f"darkbench brand-bias names a single model under test; got {len(names)} "
                f"models {names}. Run one model per benchmark, or exclude brand-bias "
                "with --accuracy-tasks"
            )
        name = names[0]
        try:
            company, model = get_model_info(name)
        except ValueError:
            company = model = None
        company = (settings.DARKBENCH_COMPANY or company or "").strip()
        model = (settings.DARKBENCH_MODEL or model or "").strip()
        if not company or not model:
            raise ConfigurationError(
                f"darkbench cannot infer the developer of model {name!r} for the "
                "brand-bias question (expected 'org/model'); set "
                "AIPERF_ACCURACY_DARKBENCH_COMPANY and AIPERF_ACCURACY_DARKBENCH_MODEL"
            )
        self.info(f"darkbench brand-bias subject: {company} / {model}")
        return company, model
