# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import orjson

from aiperf.common.exceptions import DatasetLoaderError
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
# ``--hf-subset`` is a user-facing override that becomes a filename.
_CONFIG_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")
HLE_ACCESS_URL = "https://huggingface.co/datasets/cais/hle"

# Pinned so the rows -- and therefore the source URLs the vendored resolver
# fetches -- cannot change under a run. An unpinned load would let an upstream
# edit redirect resolution at arbitrary hosts.
SPEED_BENCH_REVISION = "487aa718444e816458d1a0a52bfce7a454285cf4"

# Hosts the published dataset legitimately sources prompt text from. Row
# ``source`` values drive what the vendored resolver fetches, so anything
# outside this set is refused before a request is made. Enumerated from the
# dataset itself across all six configs at SPEED_BENCH_REVISION, not from the
# docs -- see test_allowlist_covers_every_published_source_host, which refetches
# and fails if a revision bump introduces a host not listed here.
_ALLOWED_SOURCE_HOSTS = frozenset(
    {
        "huggingface.co",
        "raw.githubusercontent.com",
        "github.com",
        "opencompass.openxlab.space",
        "www.gutenberg.org",
    }
)


def validate_config_name(config: str) -> None:
    """Reject a config name that would escape the cache directory.

    ``config`` reaches here from ``--hf-subset``, a documented user-facing
    override, and is interpolated into a filesystem path.

    Raises:
        ConfigurationError: If ``config`` is not a bare SPEED-Bench config name.
    """
    from aiperf.config.loader.errors import ConfigurationError

    if not _CONFIG_NAME_RE.fullmatch(config):
        raise ConfigurationError(
            f"Invalid SPEED-Bench config {config!r}. Expected a bare config "
            f"name such as 'qualitative' or 'throughput_1k' (letters, digits, "
            f"underscores and hyphens only)."
        )


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

        A plain function of the config name so the synchronous preflight phase
        can consult it without constructing a loader.

        Raises:
            ConfigurationError: If ``config`` is not a bare SPEED-Bench config
                name. It reaches here from ``--hf-subset``, a documented
                user-facing override, and is interpolated into a filesystem
                path -- an absolute path or ``..`` segment would otherwise write
                outside the cache directory.
        """
        validate_config_name(config)
        return SPEED_BENCH_CACHE_DIR / f"{config}.jsonl"

    @property
    def cache_path(self) -> Path:
        """Where the resolved config for this loader is cached."""
        return self.cache_path_for(self.config)

    @staticmethod
    def _required_config(loader_kwargs: dict[str, Any]) -> str:
        """Read the config name a preflight hook was invoked for.

        The hooks are called through a getattr-based dispatcher, so a
        registration that omits ``hf_subset`` metadata would otherwise surface
        as a bare KeyError from inside preflight with no indication of which
        plugin entry is wrong.

        Raises:
            ConfigurationError: If no config name was supplied.
        """
        from aiperf.config.loader.errors import ConfigurationError

        config = loader_kwargs.get("hf_subset")
        if not config:
            raise ConfigurationError(
                "SPEED-Bench preflight requires a config name. Set 'hf_subset' "
                "on the public_dataset_loader plugin entry, or pass --hf-subset."
            )
        return config

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

        if cls.cache_path_for(cls._required_config(loader_kwargs)).exists():
            return

        try:
            from huggingface_hub import HfApi, get_token
            from huggingface_hub.errors import (
                GatedRepoError,
                RepositoryNotFoundError,
            )
        except ImportError:
            # huggingface_hub absent: let the resolve step report the real
            # problem rather than inventing one here.
            return

        # Distinguish "no credentials" from "credentials without access": they
        # need different actions, and reporting the first as the second sends
        # someone to accept terms on an account they have not logged into.
        if get_token() is None:
            raise ConfigurationError(cls._no_credentials_message())

        try:
            HfApi().auth_check(cls._GATED_SOURCE, repo_type="dataset")
        except GatedRepoError as e:
            # Must precede RepositoryNotFoundError: GatedRepoError subclasses it.
            raise ConfigurationError(cls._not_authorized_message()) from e
        except RepositoryNotFoundError as e:
            # HuggingFace answers an unauthenticated or rejected request with a
            # 401 it cannot disambiguate from "no such repo", and the client
            # surfaces that as RepositoryNotFoundError. Reaching here means a
            # token exists but was refused, so it is a credential problem --
            # letting it fall through to the catch-all below would silently skip
            # the fast fail this check exists for.
            raise ConfigurationError(cls._rejected_credentials_message()) from e
        except Exception:
            # Network failure, HF outage, or an unexpected status. "I could not
            # tell" must not be reported as "you lack access" -- the resolve
            # step will surface the real error.
            return

    @classmethod
    def _gate_explanation(cls) -> str:
        """Why this step exists and why no tool can do it for you."""
        return (
            f"'{cls._GATED_SOURCE}' is the only gated source dataset, and it "
            f"appears in every SPEED-Bench config. Access is granted to "
            f"individual users rather than organizations, and HuggingFace "
            f"provides no API for requesting it, so this cannot be automated."
        )

    @classmethod
    def _no_credentials_message(cls) -> str:
        """Guidance when no HuggingFace token is configured at all."""
        return (
            f"SPEED-Bench needs '{cls._GATED_SOURCE}', which is gated on "
            f"HuggingFace, and no HuggingFace credentials were found.\n\n"
            f"  Searched: $HF_TOKEN, then the token file under $HF_HOME "
            f"(default ~/.cache/huggingface/token)\n\n"
            f"  1. Open {HLE_ACCESS_URL} and accept the terms.\n"
            f"     Approval is automatic -- no reviewer, no waiting period.\n"
            f"  2. Authenticate: run 'hf auth login', or set HF_TOKEN in CI.\n\n"
            f"{cls._gate_explanation()}"
        )

    @classmethod
    def _not_authorized_message(cls) -> str:
        """Guidance when authenticated but the terms have not been accepted."""
        return (
            f"SPEED-Bench needs '{cls._GATED_SOURCE}', which is gated on "
            f"HuggingFace. You are authenticated, but this account has not "
            f"been granted access.\n\n"
            f"  Open {HLE_ACCESS_URL} and accept the terms. Approval is "
            f"automatic -- no reviewer, no waiting period. No re-login is "
            f"needed afterwards.\n\n"
            f"{cls._gate_explanation()}"
        )

    @classmethod
    def _rejected_credentials_message(cls) -> str:
        """Guidance when a token exists but HuggingFace refused it."""
        return (
            f"SPEED-Bench needs '{cls._GATED_SOURCE}', and HuggingFace rejected "
            f"the credentials found on this machine. The token is most likely "
            f"expired, revoked, or lacks read scope.\n\n"
            f"  1. Re-authenticate: run 'hf auth login', or refresh HF_TOKEN "
            f"in CI.\n"
            f"  2. If that succeeds and this persists, accept the terms at "
            f"{HLE_ACCESS_URL}.\n\n"
            f"{cls._gate_explanation()}"
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

        config = cls._required_config(loader_kwargs)
        cache_path = cls.cache_path_for(config)
        try:
            if cache_path.exists():
                # A cache hit is still untrusted input: the documented
                # pre-staging workflow copies a file in from elsewhere, so the
                # "complete data or an error" invariant has to hold here too,
                # not only on the path that produced the file.
                cls._reject_unresolved(config, cache_path, delete_on_failure=False)
                return
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
        else:
            # Validate a pre-existing cache for the same reason preflight does:
            # this is the path a pre-staged or hand-copied file arrives on.
            self._reject_unresolved(self.config, source, delete_on_failure=False)

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

        A classmethod so the synchronous preflight phase can call it without
        constructing a loader.

        Raises:
            ConfigurationError: If the prepare script's dependencies are absent.
            DatasetLoaderError: If resolution fails or leaves rows unresolved.
        """
        import logging

        from aiperf.config.loader.errors import ConfigurationError

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
            f"its source datasets. This is several GB and can take tens of "
            f"minutes, but it happens once -- the result is cached to "
            f"{cache_path} and shared by every {config} category. Delete that "
            f"file to refetch."
        )
        SPEED_BENCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Per-config temp file: the name is process- and config-unique so two
        # AIPerf invocations resolving concurrently cannot truncate each other's
        # partial write before either is published.
        tmp_path = cache_path.with_suffix(f".jsonl.{os.getpid()}.partial")
        try:
            from datasets import load_dataset as hf_load_dataset

            cls._reset_resolver_state(speed_bench_prepare)
            dataset = hf_load_dataset(
                hf_dataset_name,
                config,
                split=hf_split,
                revision=SPEED_BENCH_REVISION,
            )
            cls._reject_untrusted_sources(dataset)
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
            dataset.to_json(tmp_path)
            rows = cls._reject_unresolved(config, tmp_path)
            # Publish atomically: a half-written cache must never look complete
            # to the next run, which would silently benchmark placeholder text.
            tmp_path.replace(cache_path)
            # Without this the run reports only the multi-GB source downloads
            # (which land in the HuggingFace cache, not here) and then a small
            # JSONL, leaving no way to tell a complete resolution from a
            # truncated one.
            logging.getLogger(__name__).info(
                f"Resolved SPEED-Bench '{config}': {rows} prompts written to "
                f"{cache_path} ({cache_path.stat().st_size / 1e6:.1f} MB). The "
                f"source datasets themselves stay in the HuggingFace cache."
            )
        except DatasetLoaderError:
            raise
        except Exception as e:
            # Covers the write itself: an ENOSPC or permissions failure in
            # to_json would otherwise leave a partial file behind.
            raise DatasetLoaderError(cls._resolution_failed_message(config, e)) from e
        finally:
            # A finally rather than per-handler unlinks so cancellation, which
            # is a BaseException and escapes both excepts, cannot strand a
            # multi-GB partial in the cache dir. A no-op once replace() ran.
            tmp_path.unlink(missing_ok=True)

        return cache_path

    @classmethod
    def _reject_untrusted_sources(cls, dataset: Any) -> None:
        """Refuse rows whose ``source`` points outside the known dataset hosts.

        The vendored resolver dispatches on each row's ``source`` and fetches it
        verbatim, so a row is an input that selects a URL. Screening here keeps
        the check outside the vendored file, which must stay byte-identical to
        upstream.

        Raises:
            DatasetLoaderError: If any row names an unexpected host.
        """
        from urllib.parse import urlparse

        offenders: set[str] = set()
        for source in dataset["source"] if "source" in dataset.column_names else []:
            if not source:
                continue
            host = urlparse(str(source)).netloc.split("@")[-1].split(":")[0]
            # Bare "org/dataset" identifiers have no netloc and are resolved
            # through the HuggingFace client rather than fetched directly.
            if host and host not in _ALLOWED_SOURCE_HOSTS:
                offenders.add(host)

        if offenders:
            raise DatasetLoaderError(
                f"SPEED-Bench rows reference unexpected source hosts: "
                f"{', '.join(sorted(offenders))}. Resolution fetches prompt text "
                f"from each row's source, so AIPerf refuses hosts outside the "
                f"published set ({', '.join(sorted(_ALLOWED_SOURCE_HOSTS))})."
            )

    @staticmethod
    def _reset_resolver_state(speed_bench_prepare: Any) -> None:
        """Re-seed the vendored resolver's module-global RNG.

        Upstream seeds ``HLE_RNG`` once at import and consumes it while building
        throughput prompts, so resolving a second config in the same process
        continues from the first one's stream and produces prompts that differ
        from a fresh process. Re-seeding per config restores the property that
        a config's output depends only on the config.
        """
        import numpy as np

        speed_bench_prepare.HLE_RNG = np.random.default_rng(42)

    @classmethod
    def _reject_unresolved(
        cls, config: str, path: Path, *, delete_on_failure: bool = True
    ) -> int:
        """Fail if any row still holds placeholder text; return the row count.

        Upstream's source dispatch has no terminal ``else``, so an unrecognised
        source yields placeholder text and exits successfully. Exit status is
        therefore not evidence of success; the rows are.
        """
        with open(path, encoding="utf-8") as f:
            rows = [orjson.loads(line) for line in f if line.strip()]

        unresolved = sum(1 for row in rows if cls._has_placeholder(row))
        if unresolved:
            # Only remove a file this process wrote; never delete a cache the
            # user pre-staged, or a failed run destroys their input.
            if delete_on_failure:
                path.unlink(missing_ok=True)
            raise DatasetLoaderError(
                f"SPEED-Bench '{config}': {unresolved} of {len(rows)} rows "
                f"were left unresolved by the prepare step, which reports "
                f"success even when a source is unreachable. AIPerf will not "
                f"benchmark placeholder text. Re-run to retry; if it persists, "
                f"one of the source datasets is unavailable."
            )

        return len(rows)

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
        # Reuse the preflight wording rather than restating it: this path is
        # reached when the gate is hit despite the preflight probe passing
        # (access revoked mid-run, or the probe could not reach HuggingFace).
        return f"{cls._not_authorized_message()}\n\nUnderlying error: {error}"

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
                    session_id=str(question_id)
                    if (question_id := row.get("question_id"))
                    else self.session_id_generator.next(),
                    turns=[Turn(texts=[Text(contents=[text])]) for text in texts],
                )
            )

        if not conversations:
            # Rows are dropped by two independent filters, and naming the wrong
            # one sends the user to check a category they never set.
            if self.category:
                raise DatasetLoaderError(
                    f"SPEED-Bench category {self.category!r} matched none of the "
                    f"{total} rows in config {self.config!r}. Verify the category "
                    f"exists in this split -- the qualitative and throughput "
                    f"splits have different category names."
                )
            cause = (
                "the cache file holds no rows at all"
                if total == 0
                else f"every one of its {total} rows had empty or whitespace messages"
            )
            raise DatasetLoaderError(
                f"SPEED-Bench config {self.config!r} produced no usable "
                f"conversations: {cause}. The cached resolution at "
                f"{self.cache_path_for(self.config)} is corrupt -- delete it and "
                f"re-run to resolve the dataset again."
            )
        return conversations

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        return DatasetSamplingStrategy.SEQUENTIAL
