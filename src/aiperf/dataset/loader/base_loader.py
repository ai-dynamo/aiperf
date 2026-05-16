# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import orjson

from aiperf.common.enums import ConversationContextMode
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import Conversation
from aiperf.common.session_id_generator import SessionIDGenerator
from aiperf.dataset.loader.models import CustomDatasetT
from aiperf.plugin.enums import DatasetSamplingStrategy

LoaderProbeData = dict[str, Any]
"""First-line probe shape passed to ``can_load`` overrides.

Any of ``session_id``, ``turns``, ``messages``, ``data``, ``conversation_id``
may be present depending on the on-disk format. Loaders branch on which keys
exist to decide whether they recognise the file.
"""

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


def _default_test_run() -> BenchmarkRun:
    from aiperf.config import BenchmarkConfig, BenchmarkRun

    cfg = BenchmarkConfig.model_validate(
        {
            "models": ["test-model"],
            "endpoint": {
                "urls": ["http://localhost:8000/v1/chat/completions"],
                "wait_for_model_timeout": 0,
            },
            "datasets": [
                {
                    "name": "default",
                    "type": "synthetic",
                    "entries": 100,
                    "prompts": {"isl": 128, "osl": 64},
                }
            ],
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "requests": 100,
                    "concurrency": 1,
                }
            ],
            "runtime": {"ui": "simple"},
        }
    )
    return BenchmarkRun(
        benchmark_id="test-run",
        cfg=cfg,
        artifact_dir=cfg.artifacts.dir,
    )


class BaseLoader(AIPerfLoggerMixin, ABC):
    """Base class for loading data.

    This abstract class provides a base implementation for loading data.
    Subclasses must implement the load_dataset and convert_to_conversations methods.
    It includes a session ID generator that is used to generate unique session IDs
    for each conversation.

    Args:
        run: The benchmark run for the current iteration.
        **kwargs: Additional arguments to pass to the base class.
    """

    def __init__(self, *, run: BenchmarkRun | None = None, **kwargs: Any) -> None:
        self.run = run or _default_test_run()
        super().__init__(**kwargs)
        # Create session ID generator (deterministic when seed is set)
        # Per-dataset random_seed lives on the active dataset; envelope-level
        # seed lives on run.random_seed.
        dataset = self.run.cfg.get_default_dataset()
        seed = getattr(dataset, "random_seed", None) or self.run.random_seed
        self.session_id_generator = SessionIDGenerator(seed=seed)

    @classmethod
    def get_default_context_mode(cls) -> ConversationContextMode | None:
        """Dataset-level default context mode for conversations without an explicit one.

        Override in subclasses when the dataset format implies a specific mode.
        Returns None to fall through to the global DELTAS_WITHOUT_RESPONSES default.
        """
        return None

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        """Dataset-level preferred sampling strategy for downstream conversation selection.

        Override in subclasses when the dataset format implies a specific strategy
        (e.g. raw payload replay loaders prefer SEQUENTIAL to preserve recorded order).
        Defaults to SHUFFLE for general datasets.
        """
        return DatasetSamplingStrategy.SHUFFLE

    @abstractmethod
    def load_dataset(self) -> dict[str, list[CustomDatasetT]]: ...

    @abstractmethod
    def convert_to_conversations(
        self, custom_data: dict[str, list[CustomDatasetT]]
    ) -> list[Conversation]: ...


class BaseFileLoader(BaseLoader):
    """Base class for loading data from a file or from inline YAML records.

    Subclasses iterate parsed-record dicts via :meth:`_iter_record_dicts`,
    which abstracts away whether the source is a file on disk or an inline
    list/dict embedded in the YAML config.

    Args:
        filename: Path to a file or directory. Mutually exclusive with ``inline_records``.
        inline_records: List of record dicts (single source) or dict mapping pool
            name to list of record dicts (multi-pool). Mutually exclusive with
            ``filename``.
        run: The benchmark run for the current iteration.
        **kwargs: Additional arguments to pass to the base class.
    """

    def __init__(
        self,
        *,
        filename: str | Path | None = None,
        inline_records: list[dict[str, Any]]
        | dict[str, list[dict[str, Any]]]
        | None = None,
        run: BenchmarkRun | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(run=run, **kwargs)
        if (filename is None) == (inline_records is None):
            raise ValueError(
                "BaseFileLoader requires exactly one of `filename=` or "
                "`inline_records=`, not both/neither."
            )
        self.filename = Path(filename) if isinstance(filename, str) else filename
        self.inline_records = inline_records

    def _iter_record_dicts(
        self,
        source: str | Path | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield parsed-record dicts from either inline records or the file source.

        For inline mode:
          - Flat list: yields each entry; ``source`` is ignored if provided.
          - Dict-of-lists (multi-pool): ``source`` selects the pool name (required;
            raises ``ValueError`` if absent). Unknown pool names raise ``KeyError``.

        For file mode:
          - Yields ``orjson.loads`` of each non-empty stripped line.
          - ``source`` (if provided) is treated as an alternate file path
            (used by directory-walking loaders such as ``random_pool``).
        """
        if self.inline_records is not None:
            if isinstance(self.inline_records, dict):
                if source is None:
                    raise ValueError(
                        "Multi-pool inline dataset requires a `source` pool name; "
                        "loader iterated without specifying which pool."
                    )
                yield from self.inline_records[source]
                return
            yield from self.inline_records
            return

        target = source if source is not None else self.filename
        with open(target, encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                if line := line.strip():
                    try:
                        yield orjson.loads(line)
                    except orjson.JSONDecodeError as e:
                        raise ValueError(
                            f"Invalid JSON in dataset file {target} at line {lineno}: {e}"
                        ) from None


class BaseRawPayloadLoader(BaseFileLoader):
    """Base for loaders that produce verbatim raw_payload conversations.

    Provides shared defaults: MESSAGE_ARRAY_WITH_RESPONSES context mode and
    SEQUENTIAL sampling. Used by ``inputs_json`` and ``raw_payload`` loaders
    that replay pre-built API request payloads byte-for-byte.
    """

    @classmethod
    def get_default_context_mode(cls) -> ConversationContextMode | None:
        return ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        return DatasetSamplingStrategy.SEQUENTIAL
