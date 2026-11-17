# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DarkBench benchmark loader (``apart/darkbench``, MIT, 660 prompts = 6 x 110).

Aligned with the reference Inspect task apartresearch/DarkBench @ 7eef151: one
zero-shot user turn per prompt under the system prompt "You are a helpful
assistant." (set via ``default_system_prompt``), graded by
``darkbench_overseer``.

Tasks are the dataset's hyphenated ``target`` labels (``brand-bias``, ...),
so ``--accuracy-tasks`` takes the labels users see in the dataset. The
overseer's JSON key is the snake-case pattern id (``brand_bias``). The
reference scorer looks the hyphenated target up directly in a snake-case
table and so cannot score three of the six patterns; mapping ``-`` to ``_``
here fixes that.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import orjson

from aiperf.accuracy.benchmarks._darkbench_patterns import (
    DARK_PATTERNS,
    get_model_info,
)
from aiperf.accuracy.benchmarks._datasets_compat import load_dataset
from aiperf.accuracy.models import AccuracyChatMessage, BenchmarkProblem
from aiperf.common.environment import Environment
from aiperf.common.exceptions import ConfigurationError
from aiperf.common.mixins import AIPerfLoggerMixin

if TYPE_CHECKING:
    from datasets import Dataset

    from aiperf.config.resolution.plan import BenchmarkRun

DATASET_NAME = "apart/darkbench"

# The reference sets no max_tokens. AIPerf requires one (its default of 100
# would truncate open-ended answers and hide late-appearing patterns), so use
# a cap generous enough for full chat answers.
DEFAULT_GENERATION_SIZE = 4096
TASKS: tuple[str, ...] = tuple(
    sorted(pattern_id.replace("_", "-") for pattern_id in DARK_PATTERNS)
)


def pattern_id_for(target: str) -> str:
    """Map a dataset ``target`` label (``brand-bias``) to its pattern id (``brand_bias``)."""
    return target.replace("-", "_")


class DarkBenchBenchmark(AIPerfLoggerMixin):
    """Load DarkBench prompts and package each one's overseer context as ground truth."""

    def __init__(self, run: BenchmarkRun, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.run = run

    async def load_problems(
        self, tasks: list[str] | None, n_shots: int, enable_cot: bool
    ) -> list[BenchmarkProblem]:
        """Load DarkBench prompts, optionally filtered to ``tasks``.

        Args:
            tasks: Dark-pattern labels to keep (e.g. ``["sneaking"]``); all six
                when None or empty.
            n_shots: Ignored (zero-shot reference); a warning is logged if set.
            enable_cot: Ignored (the reference adds no CoT trigger); a warning
                is logged if set.

        Raises:
            ConfigurationError: an unknown task label, or the ``{company}`` /
                ``{model}`` of the model under test cannot be determined.
        """
        wanted = set(tasks or [])
        if unknown := wanted - set(TASKS):
            raise ConfigurationError(
                f"unknown darkbench task(s) {sorted(unknown)}; valid tasks: {list(TASKS)}"
            )
        if n_shots or enable_cot:
            self.warning(
                f"darkbench is zero-shot without CoT; ignoring n_shots={n_shots}, "
                f"enable_cot={enable_cot}"
            )
        company, model = self._resolve_company_model()
        ds: Dataset = await asyncio.to_thread(
            load_dataset,
            DATASET_NAME,
            split="train",
            revision=Environment.ACCURACY.DARKBENCH_REVISION,
        )
        return await asyncio.to_thread(self._build_problems, ds, wanted, company, model)

    def _resolve_company_model(self) -> tuple[str, str]:
        """Return ``(company, model)`` for brand bias: env overrides, else inferred from the model name."""
        settings = Environment.ACCURACY
        model_name = self.run.cfg.get_model_names()[0]
        inferred: tuple[str, str] | None
        try:
            inferred = get_model_info(model_name)
        except ValueError:
            inferred = None
        company = settings.DARKBENCH_COMPANY or (inferred[0] if inferred else None)
        model = settings.DARKBENCH_MODEL or (inferred[1] if inferred else None)
        if company is None or model is None:
            raise ConfigurationError(
                f"darkbench cannot infer the developer of model {model_name!r} for the "
                "brand-bias description (expected an 'org/model' name); set "
                "AIPERF_ACCURACY_DARKBENCH_COMPANY and AIPERF_ACCURACY_DARKBENCH_MODEL"
            )
        self.info(f"darkbench brand-bias subject: company={company!r} model={model!r}")
        return company, model

    def _build_problems(
        self, ds: Dataset, wanted: set[str], company: str, model: str
    ) -> list[BenchmarkProblem]:
        problems: list[BenchmarkProblem] = []
        for row in ds:
            target = row["target"]
            if wanted and target not in wanted:
                continue
            pattern = DARK_PATTERNS[pattern_id_for(target)]
            ground_truth = {
                "id": row["id"],
                "pattern": pattern.id,
                "display_name": pattern.display_name,
                "description": pattern.description.format(company=company, model=model),
                "prompt": row["input"],
            }
            messages: list[AccuracyChatMessage] = [
                {"role": "user", "content": row["input"]}
            ]
            problems.append(
                BenchmarkProblem(
                    prompt=row["input"],
                    ground_truth=orjson.dumps(ground_truth).decode(),
                    task=target,
                    metadata={
                        "id": row["id"],
                        "generation_size": DEFAULT_GENERATION_SIZE,
                    },
                    raw_messages=messages,
                )
            )
        return problems
