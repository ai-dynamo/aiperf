# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import orjson

from aiperf.common.exceptions import ConfigurationError, DatasetLoaderError
from aiperf.common.models import Conversation, Text, Turn
from aiperf.dataset.loader.base_public_dataset import (
    AIPERF_DATASET_CACHE_DIR,
    BasePublicDatasetLoader,
)
from aiperf.dataset.loader.speed_bench import SpeedBenchRow
from aiperf.plugin.enums import DatasetSamplingStrategy

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

SPEED_BENCH_CACHE_DIR = AIPERF_DATASET_CACHE_DIR / "speed-bench"
HLE_ACCESS_URL = "https://huggingface.co/datasets/cais/hle"


class SpeedBenchPublicLoader(BasePublicDatasetLoader):
    """Auto-downloading loader for ``nvidia/SPEED-Bench``.

    SPEED-Bench publishes a placeholder in place of prompt text whose source
    dataset does not permit redistribution. This loader resolves those rows by
    running the vendored upstream prepare script (see ``vendor/README.md``),
    caching the resolved config, and serving categories as views over it.

    Two invariants make the result trustworthy:

    * **Whole-config resolution only.** Upstream consumes a module-global RNG
      while reconstructing HLE throughput prompts, so a row's text depends on
      how many rows preceded it. Resolving an entire config in upstream's own
      row order reproduces its output exactly; resolving a category subset does
      not. Categories are therefore filtered *after* resolution, never before.
    * **Complete data or an error.** A partially-resolved config is rejected
      rather than benchmarked, so a selector name always means the same rows.
    """

    tag = "SPEED-Bench"
    url = ""  # Resolved via the vendored prepare script, not a single URL.

    _GATED_SOURCE: ClassVar[str] = "cais/hle"

    def __init__(
        self,
        run: BenchmarkRun | None = None,
        *,
        hf_dataset_name: str = "nvidia/SPEED-Bench",
        hf_split: str = "test",
        hf_subset: str,
        category: str | None = None,
        multi_turn: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the loader.

        Args:
            run: The benchmark run this loader belongs to.
            hf_dataset_name: HuggingFace dataset identifier.
            hf_split: Split to load; SPEED-Bench publishes only ``test``.
            hf_subset: SPEED-Bench config to resolve (e.g. ``qualitative``).
            category: When set, only rows whose ``category`` matches are used.
            multi_turn: When True all turns in a row are used, else the first.
            kwargs: Forwarded to the base loader.
        """
        self.hf_dataset_name = hf_dataset_name
        self.hf_split = hf_split
        self.config = hf_subset
        self.category = category
        self.multi_turn = multi_turn
        super().__init__(run=run, **kwargs)

    @staticmethod
    def cache_path_for(config: str) -> Path:
        """Where the resolved rows for ``config`` are cached.

        A plain function of the config name so preflight can consult it without
        constructing a loader: the base class opens an aiohttp client in
        ``__init__``, which needs a running event loop that the synchronous
        preflight phase does not have.
        """
        return SPEED_BENCH_CACHE_DIR / f"{config}.jsonl"

    @property
    def cache_path(self) -> Path:
        """Where the resolved config for this loader is cached."""
        return self.cache_path_for(self.config)

    @classmethod
    def preflight_access(cls, **loader_kwargs: Any) -> None:
        """Check gated-source access before downloading anything.

        ``cais/hle`` supplies rows in every SPEED-Bench config, so an
        unauthorized account can never produce a complete dataset. Failing here
        costs one request and turns a multi-GB dead end into an immediate,
        actionable message.

        Raises:
            ConfigurationError: If the account cannot read the gated source.
        """
        from aiperf.config.loader.errors import ConfigurationError

        if cls.cache_path_for(loader_kwargs["hf_subset"]).exists():
            return

        try:
            from huggingface_hub import HfApi
            from huggingface_hub.errors import GatedRepoError

            HfApi().auth_check(cls._GATED_SOURCE, repo_type="dataset")
        except ImportError:
            # huggingface_hub absent: let the resolve step report the real
            # problem rather than inventing one here.
            return
        except GatedRepoError as e:
            raise ConfigurationError(cls._gate_message()) from e
        except Exception:
            # Network failure, HF outage, or an unexpected status. "I could not
            # tell" must not be reported as "you lack access" -- the resolve
            # step will surface the real error.
            return

    @classmethod
    def _gate_message(cls) -> str:
        """Guidance for an account that cannot read the gated source."""
        return (
            f"SPEED-Bench needs '{cls._GATED_SOURCE}', which is gated on "
            f"HuggingFace, and this account has not been granted access.\n\n"
            f"  1. Open {HLE_ACCESS_URL} and accept the terms.\n"
            f"     Approval is automatic -- no reviewer, no waiting period.\n"
            f"  2. Run: hf auth login\n\n"
            f"The request must be made from a browser: HuggingFace grants "
            f"access to individual users rather than organizations and provides "
            f"no API for it, so no tool can do this step for you. It is the "
            f"only gated source of the 14, and it appears in every SPEED-Bench "
            f"config."
        )

    @classmethod
    def preflight_materialize(cls, **loader_kwargs: Any) -> None:
        """Resolve and cache the config before services start.

        Doing this inside ``DatasetManager`` blocks the profiling handshake and
        trips ``AIPERF_DATASET_CONFIGURATION_TIMEOUT`` on any real download.

        Raises:
            ConfigurationError: If resolution fails.
        """
        from aiperf.config.loader.errors import ConfigurationError

        config = loader_kwargs["hf_subset"]
        if cls.cache_path_for(config).exists():
            return
        try:
            cls.resolve_config(config)
        except DatasetLoaderError as e:
            raise ConfigurationError(str(e)) from e

    async def load_dataset(self) -> dict[str, Any]:
        """Return resolved rows, resolving and caching the config if needed."""
        source = self.cache_path
        if not source.exists():
            await asyncio.get_running_loop().run_in_executor(
                None, self.resolve_config, self.config
            )
            source = self.cache_path

        with open(source, encoding="utf-8") as f:
            rows = [orjson.loads(line) for line in f if line.strip()]
        return {"dataset": rows}

    @classmethod
    def resolve_config(
        cls,
        config: str,
        hf_dataset_name: str = "nvidia/SPEED-Bench",
        hf_split: str = "test",
    ) -> Path:
        """Resolve ``config``'s prompt text and cache it; return the cache path.

        A classmethod so the synchronous preflight phase can call it: the base
        loader opens an aiohttp client in ``__init__`` and needs a running event
        loop, which preflight does not have.

        Raises:
            ConfigurationError: If the prepare script's dependencies are absent.
            DatasetLoaderError: If resolution fails or leaves rows unresolved.
        """
        import logging

        try:
            from aiperf.dataset.loader.vendor import speed_bench_prepare
        except ImportError as e:
            raise ConfigurationError(
                "Resolving SPEED-Bench requires the 'datasets', 'pandas', "
                "'numpy' and 'tiktoken' packages. They ship with AIPerf, so "
                "this usually means a partial install."
            ) from e

        cache_path = cls.cache_path_for(config)
        logging.getLogger(__name__).info(
            f"Resolving SPEED-Bench '{config}': downloading prompt text from "
            f"its 14 source datasets. This is several GB and can take tens of "
            f"minutes, but it happens once -- the result is cached to "
            f"{cache_path} and shared by every {config} category. Delete that "
            f"file to refetch."
        )
        SPEED_BENCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        try:
            from datasets import load_dataset as hf_load_dataset

            dataset = hf_load_dataset(hf_dataset_name, config, split=hf_split)
            # Whole-config, in upstream's row order -- see the class docstring.
            dataset = speed_bench_prepare._resolve_external_data(dataset, config)
            dataset = dataset.map(
                lambda example: {
                    "messages": [
                        {"role": "user", "content": turn} for turn in example["turns"]
                    ]
                },
                remove_columns=["turns"],
            )
        except Exception as e:
            raise DatasetLoaderError(cls._resolution_failed_message(config, e)) from e

        tmp_path = cache_path.with_suffix(".jsonl.partial")
        dataset.to_json(tmp_path)
        cls._reject_unresolved(config, tmp_path)
        # Publish atomically: a half-written cache must never look complete to
        # the next run, which would silently benchmark placeholder text.
        tmp_path.replace(cache_path)
        return cache_path

    @classmethod
    def _reject_unresolved(cls, config: str, path: Path) -> None:
        """Fail if any row still holds placeholder text.

        Upstream's source dispatch has no terminal ``else``, so an unrecognised
        source yields placeholder text and exits successfully. Exit status is
        therefore not evidence of success; the rows are.
        """
        with open(path, encoding="utf-8") as f:
            rows = [orjson.loads(line) for line in f if line.strip()]

        unresolved = sum(1 for row in rows if cls._has_placeholder(row))
        if unresolved:
            path.unlink(missing_ok=True)
            raise DatasetLoaderError(
                f"SPEED-Bench '{config}': {unresolved} of {len(rows)} rows "
                f"were left unresolved by the prepare step, which reports "
                f"success even when a source is unreachable. AIPerf will not "
                f"benchmark placeholder text. Re-run to retry; if it persists, "
                f"one of the source datasets is unavailable."
            )

    @staticmethod
    def _has_placeholder(row: dict[str, Any]) -> bool:
        return any(
            str(message.get("content", "")).startswith(SpeedBenchRow.TURNS_PLACEHOLDER)
            for message in row.get("messages", [])
        )

    @classmethod
    def _resolution_failed_message(cls, config: str, error: Exception) -> str:
        """Build a message that distinguishes the gated source from other faults."""
        text = str(error)
        gated = cls._GATED_SOURCE in text or "gated" in text.lower()
        if not gated:
            return (
                f"Failed to resolve SPEED-Bench '{config}' from its source "
                f"datasets: {error}"
            )
        return (
            f"SPEED-Bench needs '{cls._GATED_SOURCE}', which is gated on "
            f"HuggingFace. Accept its terms at {HLE_ACCESS_URL} -- approval is "
            f"automatic, with no reviewer and no waiting period -- then run "
            f"'hf auth login'. The request must be made from a browser: "
            f"HuggingFace grants access to individual users rather than "
            f"organizations and offers no API for it, so no tool can do this "
            f"step for you. It is the only gated source of the 14, and it "
            f"appears in every SPEED-Bench config. Underlying error: {error}"
        )

    async def convert_to_conversations(
        self, data: dict[str, Any]
    ) -> list[Conversation]:
        """Convert resolved rows into Conversations, applying the category filter.

        Raises:
            DatasetLoaderError: If the category matches no rows.
        """
        conversations: list[Conversation] = []
        total = 0

        for row in data["dataset"]:
            total += 1
            if self.category and row.get("category") != self.category:
                continue

            texts = [
                str(message.get("content", ""))
                for message in row.get("messages", [])
                if str(message.get("content", "")).strip()
            ]
            if not texts:
                continue
            if not self.multi_turn:
                texts = texts[:1]

            conversations.append(
                Conversation(
                    session_id=str(row.get("question_id"))
                    or self.session_id_generator.next(),
                    turns=[Turn(texts=[Text(contents=[text])]) for text in texts],
                )
            )

        if not conversations:
            raise DatasetLoaderError(
                f"SPEED-Bench category {self.category!r} matched none of the "
                f"{total} rows in config {self.config!r}. Verify the category "
                f"exists in this split -- the qualitative and throughput splits "
                f"have different category names."
            )
        return conversations

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        return DatasetSamplingStrategy.SEQUENTIAL
