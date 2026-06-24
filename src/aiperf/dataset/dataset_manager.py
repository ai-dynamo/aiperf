# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import contextlib
import gc
import os
import tempfile
import time
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import aiohttp
import orjson
from PIL import Image as PILImage

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.config import OutputDefaults, ServiceConfig, UserConfig
from aiperf.common.enums import (
    CacheBustTarget,
    CommAddress,
    CommandType,
    ConversationContextMode,
    ImageFormat,
    MemoryMapFormat,
    MessageType,
)
from aiperf.common.environment import Environment
from aiperf.common.hooks import on_command, on_request, on_stop
from aiperf.common.messages import (
    ConversationRequestMessage,
    ConversationResponseMessage,
    ConversationTurnRequestMessage,
    ConversationTurnResponseMessage,
    DatasetConfigurationFailedNotification,
    DatasetConfiguredNotification,
    ProfileConfigureCommand,
)
from aiperf.common.mixins import ReplyClientMixin
from aiperf.common.models import (
    Conversation,
    DatasetClientMetadata,
    DatasetMetadata,
    InputsFile,
    ModelEndpointInfo,
    SessionPayloads,
)
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset import mmap_cache
from aiperf.dataset.payload_formatting import format_conversation_payloads
from aiperf.dataset.utils import encode_image
from aiperf.plugin import plugins
from aiperf.plugin.enums import (
    ComposerType,
    CustomDatasetType,
    DatasetBackingStoreType,
    PluginType,
    ServiceRunType,
)
from aiperf.transports.aiohttp_client import create_tcp_connector
from aiperf.transports.http_defaults import AioHttpDefaults

if TYPE_CHECKING:
    from aiperf.dataset.protocols import (
        DatasetBackingStoreProtocol,
        DatasetClientStoreProtocol,
    )
    from aiperf.plugin.schema.schemas import EndpointMetadata


class DatasetManager(ReplyClientMixin, BaseComponentService):
    """Manages dataset generation/acquisition and provides mmap access for workers.

    Primary responsibilities:
    - Generate synthetic prompts or load datasets from files/public sources
    - Write conversations to memory-mapped files via backing store
    - Publish DatasetConfiguredNotification with mmap paths for worker access

    Workers access conversations directly via mmap (zero-copy), eliminating the
    need for ZMQ request-response communication with DatasetManager at runtime.
    """

    def __init__(
        self,
        service_config: ServiceConfig,
        user_config: UserConfig,
        service_id: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            service_config=service_config,
            user_config=user_config,
            service_id=service_id,
            reply_client_address=CommAddress.DATASET_MANAGER_PROXY_BACKEND,
            reply_client_bind=False,
            **kwargs,
        )
        self.user_config = user_config
        self.tokenizer: Tokenizer | None = None
        self.dataset: dict[
            str, Conversation
        ] = {}  # conversation ID -> Conversation mapping
        self.dataset_metadata: DatasetMetadata | None = None
        self._conversation_ids_cache: list[str] = []
        self.dataset_configured = asyncio.Event()

        # In Kubernetes mode, use compress_only to stream directly to compressed files.
        # This avoids creating large uncompressed files on the control plane.
        # WorkerPodManagers will download compressed files and decompress locally.
        self._compress_only = (
            service_config.service_run_type == ServiceRunType.KUBERNETES
        )

        self._backing_store: DatasetBackingStoreProtocol | None = None
        self._dataset_client: DatasetClientStoreProtocol | None = None
        self._default_context_mode: ConversationContextMode | None = None
        # Whether every turn carried a source-loaded raw_payload BEFORE
        # _preformat_payloads ran. Used by the inputs.json skip decision so
        # synthesized payloads (preformatted at runtime) still get exported.
        self._all_turns_source_loaded_payloads: bool = False
        # Cache key for the current run; None on synthetic-only / accuracy /
        # cache-disabled. On MISS we keep the key so the post-run populate
        # writes under the same key the lookup would have used.
        self._cache_key_for_run: str | None = None
        self._cache_hit_used: bool = False

    @on_command(CommandType.PROFILE_CONFIGURE)
    async def _profile_configure_command(
        self, message: ProfileConfigureCommand
    ) -> None:
        """Configure the dataset.

        Wraps the entire configuration sequence so that any failure (synthetic
        prompt generation, custom dataset loading, mmap finalization, etc.) is
        broadcast as DatasetConfigurationFailedNotification before the
        exception propagates back to the command-handler. Without this fan-out,
        TimingManager's _profile_configure_command would block on its 300s
        dataset_configured_event timeout while the SystemController has already
        observed the CommandErrorResponse and is trying to shut down.
        """
        try:
            await self._do_profile_configure(message)
        except Exception as e:
            self.exception(f"Dataset configuration failed: {e!r}")
            try:
                await self.publish(
                    DatasetConfigurationFailedNotification(
                        service_id=self.service_id,
                        error=f"{type(e).__name__}: {e}",
                    )
                )
            except Exception as publish_exc:
                self.exception(
                    f"Failed to publish DatasetConfigurationFailedNotification: {publish_exc!r}"
                )
            raise

    async def _do_profile_configure(self, message: ProfileConfigureCommand) -> None:
        """Inner implementation of PROFILE_CONFIGURE handling.

        Fast path: cache HIT — restore mmap files and return.

        Slow path: cache MISS — acquire an exclusive per-key flock so
        concurrent processes targeting the same key share one tokenize +
        populate cycle. Re-check the cache under the lock so a waiter that
        wakes after the winner populates uses the cached entry instead of
        repeating the work.
        """
        cache_hit = self._try_cache_lookup()
        if cache_hit is not None:
            self.info(
                f"Memory-mapped dataset cache HIT (key={cache_hit.manifest.cache_key}); "
                "skipping tokenizer + composer."
            )
            await self._configure_from_cache_hit(cache_hit)
            await self._configure_dataset_client_and_free_memory()
            return

        # When a cache key was computed, serialize the populate path with a
        # file lock so concurrent jobs don't all repeat the expensive
        # tokenize. nullcontext when caching is disabled or no key.
        lock_ctx: contextlib.AbstractAsyncContextManager[Any]
        if self._cache_key_for_run is not None:
            lock_ctx = mmap_cache.acquire_cache_lock(self._cache_key_for_run)
        else:
            lock_ctx = contextlib.nullcontext()

        async with lock_ctx:
            await self._configure_dataset_locked()

    async def _configure_dataset_locked(self) -> None:
        """Run the cache-miss configure pipeline under the populate lock.

        Re-checks the cache (a concurrent process may have populated it
        while we were blocked on the lock acquire), then drives tokenizer
        configure + dataset configure + inputs.json + client init, and
        finally writes the result into the cache on the way out.
        """
        if self._cache_key_for_run is not None:
            hit_under_lock = self._lookup_under_lock()
            if hit_under_lock is not None:
                self.info(
                    f"Memory-mapped dataset cache HIT under lock "
                    f"(key={hit_under_lock.manifest.cache_key}); "
                    "another process populated while we waited."
                )
                await self._configure_from_cache_hit(hit_under_lock)
                await self._configure_dataset_client_and_free_memory()
                return

        endpoint_meta: EndpointMetadata = plugins.get_endpoint_metadata(
            self.user_config.endpoint.type
        )
        if endpoint_meta.tokenizes_input:
            self.info("Configuring tokenizer(s) for dataset manager")
            begin = time.perf_counter()
            await self._configure_tokenizer()
            duration = time.perf_counter() - begin
            self.info(lambda: f"Tokenizer(s) configured in {duration:.2f} seconds")
        else:
            self.info(
                "Tokenization is disabled for this endpoint, skipping tokenizer configuration"
            )

        self.info(lambda: f"Configuring dataset for {self.service_id}")
        begin = time.perf_counter()
        await self._configure_dataset()
        dataset_type = self.user_config.input.custom_dataset_type
        public_dataset = self.user_config.input.public_dataset
        is_mooncake_payload_mode = (
            dataset_type == CustomDatasetType.MOONCAKE_TRACE
            and self._all_turns_source_loaded_payloads
        )
        is_weka_format = (
            dataset_type == CustomDatasetType.WEKA_TRACE
            or self.user_config.input.detected_loader == "weka_trace"
            or public_dataset == "weka_hf"
            or (
                public_dataset is not None
                and str(public_dataset).startswith("semianalysis_cc_traces_weka")
            )
        )
        if (
            dataset_type
            in (CustomDatasetType.RAW_PAYLOAD, CustomDatasetType.INPUTS_JSON)
            or is_mooncake_payload_mode
            or is_weka_format
        ):
            self.info("Skipping inputs.json generation")
        else:
            await self._generate_inputs_json_file()
        await self._configure_dataset_client_and_free_memory()

        if self._cache_key_for_run is not None:
            self._populate_cache_after_run()

        duration = time.perf_counter() - begin
        self.info(lambda: f"Dataset configured in {duration:.2f} seconds")

    def _lookup_under_lock(self) -> mmap_cache.CacheHit | None:
        """Re-check the cache for a HIT after the populate lock is held."""
        assert self._cache_key_for_run is not None
        try:
            return mmap_cache.lookup(
                self._cache_key_for_run, compressed=self._compress_only
            )
        except (OSError, ValueError) as e:
            self.warning(f"Cache re-lookup under lock failed: {e!r}")
            return None

    async def _configure_dataset_client_and_free_memory(self) -> None:
        """Configure the dataset client for serving fallback requests, then free memory."""
        conversation_count = len(self.dataset)

        if not self._compress_only:
            client_metadata = self._backing_store.get_client_metadata()
            ClientStoreClass = plugins.get_class(
                PluginType.DATASET_CLIENT_STORE, client_metadata.client_type
            )
            self._dataset_client = ClientStoreClass(client_metadata=client_metadata)
            await self._dataset_client.initialize()

        self.dataset_configured.set()

        # Reassign to new empty containers (not .clear()) to release object references,
        # then run gc.collect() twice to ensure circular references are cleaned up.
        self.dataset = {}
        self._conversation_ids_cache = []
        gc.collect()
        gc.collect()

        if self._compress_only:
            self.info(
                f"Kubernetes mode: skipped local client, freed {conversation_count} "
                "conversations from memory (workers handle all requests)"
            )
        else:
            self.info(
                f"Dataset client initialized and freed {conversation_count} "
                "conversations from memory"
            )

    async def _configure_tokenizer(self) -> None:
        """Configure the tokenizer for the dataset manager."""
        model_name = self.user_config.endpoint.model_names[0]
        tokenizer_config = self.user_config.tokenizer
        tokenizer_name = tokenizer_config.get_tokenizer_name_for_model(model_name)

        # Let exceptions propagate - controller_utils will display the error panel
        self.tokenizer = await asyncio.to_thread(
            Tokenizer.from_pretrained,
            tokenizer_name,
            trust_remote_code=tokenizer_config.trust_remote_code,
            revision=tokenizer_config.revision,
            resolve_alias=tokenizer_config.should_resolve_alias,
        )

    async def _convert_media_urls_to_inline(self) -> None:
        """Download HTTP(S) image URLs and replace them with base64 data URLs.

        Collects unique URLs across all conversations/turns, downloads each once,
        and replaces all occurrences in-place. This is needed for endpoints that
        require inline media (e.g., NIM Image Retrieval).
        """
        url_to_locations: dict[str, list[tuple[list[str], int]]] = {}

        for conversation in self.dataset.values():
            for turn in conversation.turns:
                for image in turn.images:
                    for i, content in enumerate(image.contents):
                        parsed = urlparse(content)
                        if parsed.scheme in ("http", "https") and parsed.netloc:
                            url_to_locations.setdefault(content, []).append(
                                (image.contents, i)
                            )

        if not url_to_locations:
            return

        dataset_env = Environment.DATASET
        timeout = aiohttp.ClientTimeout(total=dataset_env.MEDIA_DOWNLOAD_TIMEOUT)
        max_concurrency = dataset_env.MEDIA_DOWNLOAD_MAX_CONCURRENCY

        self.info(
            f"Downloading {len(url_to_locations)} unique media URL(s) "
            f"for inline encoding (concurrency={max_concurrency})"
        )

        semaphore = asyncio.Semaphore(max_concurrency)
        url_to_data_url: dict[str, str] = {}

        async def _download_and_encode(
            session: aiohttp.ClientSession, url: str
        ) -> None:
            async with semaphore:
                async with session.get(url, timeout=timeout) as resp:
                    if resp.status != 200:
                        raise RuntimeError(
                            f"Failed to download media URL '{url}': HTTP {resp.status}"
                        )
                    data = await resp.read()

                img = PILImage.open(BytesIO(data))
                if img.format is None:
                    raise RuntimeError(
                        f"Failed to determine image format for URL '{url}'"
                    )
                if img.format.upper() not in list(ImageFormat):
                    raise RuntimeError(
                        f"'{img.format}' from URL '{url}' is not a supported "
                        f"image format: {', '.join(ImageFormat)}"
                    )
                url_to_data_url[url] = (
                    f"data:image/{img.format.lower()};base64,"
                    f"{encode_image(img, img.format)}"
                )

        connector = create_tcp_connector()
        async with aiohttp.ClientSession(
            connector=connector,
            trust_env=AioHttpDefaults.TRUST_ENV,
        ) as session:
            await asyncio.gather(
                *[_download_and_encode(session, url) for url in url_to_locations]
            )

        for url, locations in url_to_locations.items():
            data_url = url_to_data_url[url]
            for contents_list, index in locations:
                contents_list[index] = data_url

        self.info("Media URL download and inline encoding complete")

    def _preformat_payloads(self, conversations: list[Conversation]) -> None:
        """Pre-format API request payloads and store them on each turn.

        Must run after all content mutations (media rewriting, etc.) so the
        serialized payloads reflect final turn content.  Only preformats when
        every conversation is eligible: single-turn, or multi-turn with
        self-contained turns (MESSAGE_ARRAY_WITH_RESPONSES where each turn
        carries a complete message array).

        DELTAS_WITH_RESPONSES is NOT safe for preformatting because each turn
        is a delta — the worker accumulates prior turns at runtime, so the
        payload for turn N depends on turns 0..N-1.

        Conversations that already carry raw_payload on ALL turns are skipped.
        If ANY conversation cannot be preformatted, the entire batch is skipped
        to avoid mixed raw_payload state (which the mmap format check rejects).
        """
        if self.user_config is None:
            return

        # Cache-bust dispatch (worker.py `_process_credit_with_session`) mutates
        # `session_message`/`raw_messages` per credit; the PAYLOAD_BYTES fast
        # path early-returns before that dispatch, sending pre-encoded mmap
        # bytes to the wire verbatim. Pre-formatting under cache-bust would
        # silently no-op the marker injection. Bail to the structured-turns
        # path whenever cache-bust is enabled.
        if self.user_config.input.prompt.cache_bust.target != CacheBustTarget.NONE:
            return

        # DAG datasets (any FORK/SPAWN branch) are delta-compressed and
        # accumulate context across the tree: FORK children sticky-seed their
        # turn_list from the parent's live session, so the parent must be
        # created and pinned on the worker, and no turn's wire body is fully
        # determined at compose time. The PAYLOAD_BYTES fast path ships
        # pre-encoded per-turn bytes and skips session creation entirely, which
        # is fundamentally incompatible with that accumulation (a preformatted
        # FORK parent never gets a session, so every child fails the
        # sticky-routing invariant "parent session not found on this worker").
        # agentx is delta-compressed and must never use payload_bytes -- bail to
        # the structured-turns path whenever the dataset declares any branch.
        if any(conv.branches for conv in conversations):
            return

        needs_formatting = False
        for conv in conversations:
            if all(t.raw_payload is not None for t in conv.turns):
                continue
            needs_formatting = True
            is_single_turn = len(conv.turns) == 1
            is_self_contained = (
                conv.context_mode
                == ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
            )
            if not (is_single_turn or is_self_contained):
                return

        if not needs_formatting:
            return

        model_endpoint = ModelEndpointInfo.from_user_config(self.user_config)

        turn_lookup: dict[tuple[str, int], Any] = {}
        for conversation in conversations:
            for i, turn in enumerate(conversation.turns):
                turn_lookup[(conversation.session_id, i)] = turn

        try:
            count = 0
            for session_id, turn_idx, payload in format_conversation_payloads(
                conversations, model_endpoint
            ):
                turn_lookup[(session_id, turn_idx)].raw_payload = payload
                count += 1
        except NotImplementedError:
            self.info(
                "Skipping payload pre-formatting "
                "(endpoint does not support format_payload)"
            )
            return

        self.info(f"Pre-formatted {count} payloads for payload mmap fast path")

    def _select_mmap_format(self, conversations: list[Conversation]) -> MemoryMapFormat:
        """Pick the dataset mmap format and refuse PAYLOAD_BYTES under cache-bust.

        This is the earliest authoritative point in the loader where the
        run's ``MemoryMapFormat`` is finalized -- it runs once after dataset
        composition and before the backing store is initialized, so no
        per-fork / per-dataset preformat decisions have happened yet.

        PAYLOAD_BYTES is the mmap fast path: workers stream pre-encoded
        bytes verbatim and skip the cache-bust dispatch in
        ``_process_credit_with_session``. Loaders that natively populate
        ``Turn.raw_payload`` (RawPayloadDatasetLoader, InputsJsonPayloadLoader,
        and MooncakeTraceDatasetLoader entries with a ``payload`` field)
        would otherwise silently bypass the marker injection. Refuse here
        with a clear, actionable error rather than letting the worker
        discover the conflict at runtime.
        """
        has_payload_bytes = any(
            turn.raw_payload is not None
            for conv in conversations
            for turn in conv.turns
        )
        if has_payload_bytes and not all(
            turn.raw_payload is not None
            for conv in conversations
            for turn in conv.turns
        ):
            raise ValueError(
                "Mixed raw_payload state: all turns must have raw_payload "
                "when any turn does (PAYLOAD_BYTES format requires uniformity)"
            )
        if (
            has_payload_bytes
            and self.user_config is not None
            and self.user_config.input.prompt.cache_bust.target != CacheBustTarget.NONE
        ):
            raise ValueError(
                "--cache-bust is incompatible with the PAYLOAD_BYTES mmap "
                "fast path. The selected dataset (raw_payload / inputs_json "
                "/ mooncake_trace with payload field) ships pre-encoded bytes "
                "verbatim and bypasses the per-credit cache-bust marker "
                "injection. Either remove --cache-bust, or use a dataset "
                "type that produces structured turns "
                "(e.g. single_turn / multi_turn / dag_jsonl)."
            )
        if (
            has_payload_bytes
            and self.user_config is not None
            and self.user_config.endpoint.use_dynamo_conv_aware_routing
        ):
            raise ValueError(
                "--use-dynamo-conv-aware-routing is incompatible with the "
                "PAYLOAD_BYTES mmap fast path. The selected dataset (raw_payload "
                "/ inputs_json / mooncake_trace with payload field) ships "
                "pre-encoded bytes verbatim, so nvext.session_control cannot be "
                "injected. Either disable Dynamo conversation-aware routing, or "
                "use a dataset type that produces structured turns "
                "(e.g. single_turn / multi_turn / dag_jsonl)."
            )
        return (
            MemoryMapFormat.PAYLOAD_BYTES
            if has_payload_bytes
            else MemoryMapFormat.CONVERSATION
        )

    def _generate_input_payloads(
        self,
        model_endpoint: ModelEndpointInfo,
    ) -> InputsFile:
        """Generate input payloads from the dataset for use in the inputs.json file."""
        inputs = InputsFile()
        session_payloads_map: dict[str, list] = {}

        has_raw_payloads = any(
            turn.raw_payload is not None
            for conv in self.dataset.values()
            for turn in conv.turns
        )

        if has_raw_payloads:
            for conversation in self.dataset.values():
                raw_flags = [
                    turn.raw_payload is not None for turn in conversation.turns
                ]
                if any(raw_flags) and not all(raw_flags):
                    raw_indexes = [i for i, r in enumerate(raw_flags) if r]
                    missing_indexes = [i for i, r in enumerate(raw_flags) if not r]
                    raise ValueError(
                        f"conversation '{conversation.session_id}' has mixed "
                        f"raw_payload state: turns {raw_indexes} have "
                        f"raw_payload, turns {missing_indexes} do not; v1 "
                        "requires all-or-none per conversation"
                    )
            for conversation in self.dataset.values():
                payloads = [
                    turn.raw_payload
                    for turn in conversation.turns
                    if turn.raw_payload is not None
                ]
                if payloads:
                    session_payloads_map[conversation.session_id] = payloads
        else:
            from aiperf.dataset.payload_formatting import format_conversation_payloads

            for session_id, _turn_idx, payload in format_conversation_payloads(
                self.dataset.values(), model_endpoint
            ):
                if session_id not in session_payloads_map:
                    session_payloads_map[session_id] = []
                session_payloads_map[session_id].append(payload)

        for session_id, payloads in session_payloads_map.items():
            inputs.data.append(
                SessionPayloads(session_id=session_id, payloads=payloads)
            )
        return inputs

    async def _generate_inputs_json_file(self) -> None:
        """Generate inputs.json file in the artifact directory."""
        file_path = (
            self.user_config.output.artifact_directory / OutputDefaults.INPUTS_JSON_FILE
        )
        temp_file_path = file_path.with_suffix(".tmp")
        self.info(f"Generating inputs.json file at {file_path.resolve()}")

        try:
            start_time = time.perf_counter()
            file_path.parent.mkdir(parents=True, exist_ok=True)

            model_endpoint = ModelEndpointInfo.from_user_config(self.user_config)
            inputs = self._generate_input_payloads(model_endpoint)

            temp_file_path.write_bytes(
                orjson.dumps(
                    inputs.model_dump(exclude_none=True, mode="json"),
                    option=orjson.OPT_INDENT_2,
                )
            )
            temp_file_path.replace(file_path)

            duration = time.perf_counter() - start_time
            self.info(f"inputs.json file generated in {duration:.2f} seconds")

        except OSError as e:
            self.exception(
                f"Error generating inputs.json file at {file_path.resolve()}: {e!r}"
            )
            # NOTE: We don't raise an error here for OS related errors like writing to a file,
            # as this won't affect the benchmark execution.
        except Exception as e:
            # This is a fatal error, as later in the benchmark, errors will occur while trying to convert the payloads
            # on the worker side.
            self.exception(
                f"Error generating inputs.json file at {file_path.resolve()}: {e!r}"
            )
            raise
        finally:
            if temp_file_path.exists():
                temp_file_path.unlink()

    async def _load_public_dataset(self) -> list[Conversation]:
        ComposerClass = plugins.get_class(
            PluginType.DATASET_COMPOSER, ComposerType.PUBLIC
        )
        composer = ComposerClass(config=self.user_config, tokenizer=self.tokenizer)
        self._default_context_mode = composer.get_default_context_mode()
        return await composer.create_dataset_async()

    def _load_custom_dataset(self) -> list[Conversation]:
        ComposerClass = plugins.get_class(
            PluginType.DATASET_COMPOSER, ComposerType.CUSTOM
        )
        composer = ComposerClass(config=self.user_config, tokenizer=self.tokenizer)
        conversations = composer.create_dataset()
        self._default_context_mode = composer.get_default_context_mode()
        return conversations

    def _is_rankings_endpoint(self, endpoint_type: str) -> bool:
        return "rankings" in endpoint_type.lower()

    def _load_synthetic_dataset(self) -> list[Conversation]:
        endpoint_type = self.user_config.endpoint.type

        if self._is_rankings_endpoint(endpoint_type):
            composer_type = ComposerType.SYNTHETIC_RANKINGS
        else:
            composer_type = ComposerType.SYNTHETIC

        ComposerClass = plugins.get_class(PluginType.DATASET_COMPOSER, composer_type)
        composer = ComposerClass(config=self.user_config, tokenizer=self.tokenizer)
        conversations = composer.create_dataset()
        self._default_context_mode = composer.get_default_context_mode()
        return conversations

    async def _load_accuracy_dataset(self) -> list[Conversation]:
        from aiperf.dataset.loader.accuracy_dataset_loader import AccuracyDatasetLoader
        from aiperf.plugin.enums import DatasetSamplingStrategy, TimingMode

        if self.user_config.timing_mode == TimingMode.FIXED_SCHEDULE:
            raise self._service_error(
                "Accuracy mode requires sequential request order; "
                "fixed-schedule timing is not supported in accuracy mode."
            )

        if "dataset_sampling_strategy" not in self.user_config.input.model_fields_set:
            self.user_config.input.dataset_sampling_strategy = (
                DatasetSamplingStrategy.SEQUENTIAL
            )
        elif (
            self.user_config.input.dataset_sampling_strategy
            != DatasetSamplingStrategy.SEQUENTIAL
        ):
            raise self._service_error(
                f"Accuracy mode requires sequential request order; "
                f"'{self.user_config.input.dataset_sampling_strategy}' sampling is not supported. "
                f"Remove --dataset-sampling-strategy or set it to 'sequential'."
            )

        loader = AccuracyDatasetLoader(user_config=self.user_config)
        return await loader.load()

    async def _configure_dataset(self) -> None:
        if self.user_config is None:
            raise self._service_error("User config is required for dataset manager")

        self.dataset_configured.clear()
        self._default_context_mode = None

        if self.user_config.accuracy.enabled:
            conversations = await self._load_accuracy_dataset()
        elif self.user_config.input.public_dataset is not None:
            conversations = await self._load_public_dataset()
        elif (
            self.user_config.input.custom_dataset_type is not None
            or self.user_config.input.file is not None
        ):
            # Use CUSTOM composer if either:
            # 1. custom_dataset_type is explicitly set, OR
            # 2. input file is provided (composer will auto-infer type)
            conversations = self._load_custom_dataset()
        else:
            conversations = self._load_synthetic_dataset()

        self.dataset = {conv.session_id: conv for conv in conversations}
        self._conversation_ids_cache = [
            conversation.session_id for conversation in conversations
        ]

        # Capture pre-preformat raw_payload state. Once _preformat_payloads
        # runs, synthesized turns also gain raw_payload, which would falsely
        # trip the "payloads are pre-built" inputs.json skip in the caller.
        self._all_turns_source_loaded_payloads = bool(conversations) and all(
            turn.raw_payload is not None
            for conv in conversations
            for turn in conv.turns
        )

        endpoint_meta: EndpointMetadata = plugins.get_endpoint_metadata(
            self.user_config.endpoint.type
        )
        if endpoint_meta.requires_inline_media:
            await self._convert_media_urls_to_inline()

        # Pre-format payloads after all mutations (media rewriting, etc.) are
        # complete.  Safe only when every turn's payload is fully deterministic
        # at compose time: single-turn conversations, or multi-turn with
        # pre-canned assistant responses (WITH_RESPONSES context modes).
        self._preformat_payloads(conversations)

        mmap_format = self._select_mmap_format(conversations)

        # Initialize backing store and stream conversations to mmap files
        # Workers read directly from these files
        BackingStoreClass = plugins.get_class(
            PluginType.DATASET_BACKING_STORE, DatasetBackingStoreType.MEMORY_MAP
        )
        self._backing_store = BackingStoreClass(
            benchmark_id=self.user_config.benchmark_id,
            compress_only=self._compress_only,
            format=mmap_format,
        )
        await self._backing_store.initialize()
        await self._backing_store.add_conversations(self.dataset)
        await self._backing_store.finalize()
        # In Kubernetes mode (compress_only=True), files are already compressed
        # during finalize(). In local mode, uncompressed files are used directly.

        mmap_metadata = self._backing_store.get_client_metadata()
        self.info(f"Backing store finalized: {mmap_metadata}")

        # In Kubernetes mode, workers wait for DatasetDownloadedNotification from
        # WorkerPodManager which provides local file paths. We still send mmap_metadata
        # which has the control plane paths (ignored by workers in Kubernetes mode).
        client_metadata: DatasetClientMetadata = mmap_metadata
        if self.service_config.service_run_type == ServiceRunType.KUBERNETES:
            self.info(
                "Kubernetes mode: workers will wait for DatasetDownloadedNotification "
                "from WorkerPodManager before accessing dataset"
            )

        self.dataset_metadata = DatasetMetadata(
            conversations=[conversation.metadata() for conversation in conversations],
            sampling_strategy=self.user_config.input.dataset_sampling_strategy,
            default_context_mode=self._default_context_mode,
        )
        self.info(
            f"sampling strategy: {self.dataset_metadata.sampling_strategy}, "
            f"unique conversations: {len(self.dataset_metadata.conversations)}, "
            f"unique turn count: {self.dataset_metadata.total_turn_count}"
        )
        # Note: dataset_configured event is set in _configure_dataset_client_and_free_memory()
        # after the dataset client is initialized, to avoid a race condition where fallback
        # requests arrive before the client is ready.
        await self.publish(
            DatasetConfiguredNotification(
                service_id=self.service_id,
                metadata=self.dataset_metadata,
                client_metadata=client_metadata,
            )
        )

    def _run_mmap_paths(self) -> tuple[Path, Path]:
        """Return the (data, index) paths the backing store will write to."""
        base_path = Environment.DATASET.MMAP_BASE_PATH or Path(tempfile.gettempdir())
        mmap_dir = base_path / f"aiperf_mmap_{self.user_config.benchmark_id}"
        ext = ".dat.zst" if self._compress_only else ".dat"
        return mmap_dir / f"dataset{ext}", mmap_dir / f"index{ext}"

    def _try_cache_lookup(self) -> mmap_cache.CacheHit | None:
        """Return a CacheHit when the run can reuse a cached mmap, else None.

        Sets ``self._cache_key_for_run`` when caching is applicable so the
        post-run populate writes under the same key.
        """
        if not mmap_cache.cache_enabled():
            return None
        try:
            key = mmap_cache.compute_cache_key_from_user_config(self.user_config)
        except Exception as e:
            self.warning(f"Skipping mmap cache: failed to compute key: {e!r}")
            return None
        if key is None:
            return None
        self._cache_key_for_run = key
        try:
            return mmap_cache.lookup(key, compressed=self._compress_only)
        except Exception as e:
            self.warning(f"Skipping mmap cache lookup: {e!r}")
            return None

    async def _configure_from_cache_hit(self, hit: mmap_cache.CacheHit) -> None:
        """Restore mmap files + metadata from a cache HIT, then init backing store.

        Restores ``dataset.dat`` / ``index.dat`` into the run's mmap dir so the
        rest of the pipeline (backing-store cleanup, worker mmap reads, k8s
        download) sees byte-identical files to a non-cached run.
        """
        run_data_path, run_index_path = self._run_mmap_paths()
        mmap_cache.restore_to_run_dir(hit, run_data_path, run_index_path)

        manifest = hit.manifest
        try:
            self.dataset_metadata = DatasetMetadata.model_validate_json(
                manifest.dataset_metadata_json
            )
        except Exception as e:
            self.warning(
                f"Cache HIT manifest dataset_metadata_json invalid; treating as MISS: {e!r}"
            )
            self._cache_hit_used = False
            try:
                run_data_path.unlink(missing_ok=True)
                run_index_path.unlink(missing_ok=True)
            except OSError:
                pass
            return

        self._default_context_mode = self.dataset_metadata.default_context_mode
        self._all_turns_source_loaded_payloads = (
            manifest.all_turns_source_loaded_payloads
        )

        BackingStoreClass = plugins.get_class(
            PluginType.DATASET_BACKING_STORE, DatasetBackingStoreType.MEMORY_MAP
        )
        self._backing_store = BackingStoreClass(
            benchmark_id=self.user_config.benchmark_id,
            compress_only=self._compress_only,
            format=MemoryMapFormat(manifest.mmap_format),
        )
        # On-disk files already exist; adopt them without running the writer.
        # The on-stop cleanup hook still unlinks the run mmap dir at shutdown.
        session_ids = [c.conversation_id for c in self.dataset_metadata.conversations]
        self._backing_store.adopt_existing_files(
            session_ids=session_ids,
            total_size_bytes=manifest.total_size_bytes,
            compressed_size_bytes=manifest.compressed_size_bytes,
        )

        client_metadata = self._backing_store.get_client_metadata()
        self._cache_hit_used = True

        self.info(
            f"sampling strategy: {self.dataset_metadata.sampling_strategy}, "
            f"unique conversations: {len(self.dataset_metadata.conversations)}, "
            f"unique turn count: {self.dataset_metadata.total_turn_count}"
        )
        await self.publish(
            DatasetConfiguredNotification(
                service_id=self.service_id,
                metadata=self.dataset_metadata,
                client_metadata=client_metadata,
            )
        )

    def _populate_cache_after_run(self) -> None:
        """Write the just-finalized run's mmap files into the cache."""
        if self._cache_hit_used:
            return
        if self._cache_key_for_run is None or self._backing_store is None:
            return
        if self.dataset_metadata is None:
            return
        run_data_path, run_index_path = self._run_mmap_paths()
        if not run_data_path.exists() or not run_index_path.exists():
            return

        mmap_metadata = self._backing_store.get_client_metadata()
        manifest = mmap_cache.CacheManifest(
            cache_key=self._cache_key_for_run,
            created_at=time.time(),
            aiperf_version=os.environ.get("AIPERF_VERSION") or None,
            num_conversations=mmap_metadata.conversation_count,
            total_size_bytes=mmap_metadata.total_size_bytes,
            compressed=mmap_metadata.compressed,
            compressed_size_bytes=mmap_metadata.compressed_size_bytes,
            mmap_format=str(mmap_metadata.format),
            default_context_mode=(
                str(self._default_context_mode)
                if self._default_context_mode is not None
                else None
            ),
            all_turns_source_loaded_payloads=self._all_turns_source_loaded_payloads,
            dataset_metadata_json=self.dataset_metadata.model_dump_json(),
        )
        try:
            mmap_cache.populate(
                cache_key=self._cache_key_for_run,
                run_data_path=run_data_path,
                run_index_path=run_index_path,
                manifest=manifest,
            )
        except Exception as e:
            self.warning(f"Failed to populate mmap cache: {e!r}")

    @on_request(MessageType.CONVERSATION_REQUEST)
    async def _handle_conversation_request(
        self, message: ConversationRequestMessage
    ) -> ConversationResponseMessage:
        """Handle a conversation request using the dataset client."""
        self.debug(lambda: f"Handling conversation request: {message}")

        await self._wait_for_dataset_configuration()

        if self._dataset_client is None:
            if self._compress_only:
                raise self._service_error(
                    "DatasetManager cannot serve requests in Kubernetes mode. "
                    "Workers should handle all conversation requests.",
                )
            raise self._service_error(
                "Dataset client is not initialized. Dataset must be configured before handling requests.",
            )

        try:
            conversation = await self._dataset_client.get_conversation(
                message.conversation_id
            )
        except KeyError:
            raise self._service_error(
                f"Conversation {message.conversation_id} not found in dataset.",
            ) from None

        self.trace_or_debug(
            lambda: f"Sending conversation response: {conversation}",
            lambda: f"Sending conversation response with id: {conversation.session_id}",
        )
        return ConversationResponseMessage(
            service_id=self.service_id,
            request_id=message.request_id,
            conversation=conversation,
        )

    @on_request(MessageType.CONVERSATION_TURN_REQUEST)
    async def _handle_conversation_turn_request(
        self, message: ConversationTurnRequestMessage
    ) -> ConversationTurnResponseMessage:
        """Handle a turn request using the dataset client."""
        self.debug(lambda: f"Handling turn request: {message}")

        await self._wait_for_dataset_configuration()

        if self._dataset_client is None:
            if self._compress_only:
                raise self._service_error(
                    "DatasetManager cannot serve requests in Kubernetes mode. "
                    "Workers should handle all conversation requests.",
                )
            raise self._service_error(
                "Dataset client is not initialized. Dataset must be configured before handling requests.",
            )

        try:
            conversation = await self._dataset_client.get_conversation(
                message.conversation_id
            )
        except KeyError as e:
            raise self._service_error(
                f"Conversation {message.conversation_id} not found in dataset.",
            ) from e

        if message.turn_index >= len(conversation.turns):
            raise self._service_error(
                f"Turn index {message.turn_index} is out of range for conversation {message.conversation_id}.",
            )

        turn = conversation.turns[message.turn_index]

        self.trace_or_debug(
            lambda: f"Sending turn response: {turn}",
            "Sending turn response",
        )
        return ConversationTurnResponseMessage(
            service_id=self.service_id,
            request_id=message.request_id,
            turn=turn,
        )

    async def _wait_for_dataset_configuration(self) -> None:
        """Wait for the dataset to be configured if it is not already."""
        if not self.dataset_configured.is_set():
            self.debug(
                "Dataset not configured. Waiting for dataset to be configured..."
            )
            await asyncio.wait_for(
                self.dataset_configured.wait(),
                timeout=Environment.DATASET.CONFIGURATION_TIMEOUT,
            )

    @on_stop
    async def _cleanup(self) -> None:
        """Clean up the backing store, dataset client, and associated mmap files."""
        if self._dataset_client is not None:
            await self._dataset_client.stop()
            self.debug("Dataset client cleanup complete")
        if self._backing_store is not None:
            await self._backing_store.stop()
            self.debug("Backing store cleanup complete")


def main() -> None:
    """Main entry point for the dataset manager."""

    from aiperf.common.bootstrap import bootstrap_and_run_service
    from aiperf.plugin.enums import ServiceType

    bootstrap_and_run_service(ServiceType.DATASET_MANAGER)


if __name__ == "__main__":
    main()
