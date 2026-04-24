# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import gc
import time
from typing import TYPE_CHECKING

import msgspec
import orjson

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.control_structs import Command
from aiperf.common.enums import (
    CommAddress,
    CommandType,
    ConversationContextMode,
    MessageType,
)
from aiperf.common.environment import Environment
from aiperf.common.hooks import on_command, on_request, on_stop
from aiperf.common.messages import (
    ConversationRequestMessage,
    ConversationResponseMessage,
    ConversationTurnRequestMessage,
    ConversationTurnResponseMessage,
    DatasetConfiguredNotification,
)
from aiperf.common.mixins import ReplyClientMixin
from aiperf.common.models import (
    Conversation,
    DatasetClientMetadata,
    DatasetMetadata,
    InputsFile,
)
from aiperf.common.models.base_models import _msgspec_enc_hook
from aiperf.common.tokenizer import Tokenizer
from aiperf.config import OutputDefaults
from aiperf.dataset.composer_loader import (
    load_conversations_for_run,
    load_custom_dataset,
    load_public_dataset,
    load_synthetic_dataset,
)
from aiperf.dataset.inputs_file_builder import build_inputs_file
from aiperf.dataset.media_inline import (
    collect_http_image_urls,
    download_and_inline_urls,
)
from aiperf.dataset.tokenizer_loader import load_tokenizer_for_run
from aiperf.plugin import plugins
from aiperf.plugin.enums import (
    DatasetBackingStoreType,
    PluginType,
    ServiceRunType,
)

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun
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
        run: BenchmarkRun,
        service_id: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            run=run,
            service_id=service_id,
            reply_client_address=CommAddress.DATASET_MANAGER_PROXY_BACKEND,
            reply_client_bind=False,
            **kwargs,
        )
        self.tokenizer: Tokenizer | None = None
        self.dataset: dict[
            str, Conversation
        ] = {}  # conversation ID -> Conversation mapping
        self.dataset_metadata: DatasetMetadata | None = None
        self._conversation_ids_cache: list[str] = []
        self.dataset_configured = asyncio.Event()

        # In Kubernetes mode, use compress_only to stream directly to compressed files.
        # This avoids creating large uncompressed files on the control plane.
        # WorkerGroupManagers will download compressed files and decompress locally.
        self._compress_only = (
            run.cfg.runtime.service_run_type == ServiceRunType.KUBERNETES
        )

        BackingStoreClass = plugins.get_class(
            PluginType.DATASET_BACKING_STORE, DatasetBackingStoreType.MEMORY_MAP
        )
        self._backing_store: DatasetBackingStoreProtocol = BackingStoreClass(
            benchmark_id=self.run.cfg.artifacts.benchmark_id,
            compress_only=self._compress_only,
        )
        self._dataset_client: DatasetClientStoreProtocol | None = None
        self._rebroadcast_task: asyncio.Task | None = None
        self._default_context_mode: ConversationContextMode | None = None
        self._profile_configure_task: asyncio.Task[None] | None = None

    @on_command(CommandType.PROFILE_START)
    async def _on_profile_start(self, message: Command) -> None:
        """Stop rebroadcasting dataset notifications once profiling begins."""
        if self._rebroadcast_task is not None:
            self._rebroadcast_task.cancel()
            self._rebroadcast_task = None

    @on_command(CommandType.PROFILE_CONFIGURE)
    async def _profile_configure_command(self, message: Command) -> None:
        """Configure the dataset."""
        if self._profile_configure_task is not None:
            await self._profile_configure_task
            return

        self._profile_configure_task = asyncio.create_task(
            self._run_profile_configure()
        )
        try:
            await self._profile_configure_task
        finally:
            self._profile_configure_task = None

    async def _run_profile_configure(self) -> None:
        """Run dataset configuration once, coalescing concurrent configure commands."""
        endpoint_meta: EndpointMetadata = plugins.get_endpoint_metadata(
            self.run.cfg.endpoint.type
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
        await self._generate_inputs_json_file()
        await self._configure_dataset_client_and_free_memory()

        duration = time.perf_counter() - begin
        self.info(lambda: f"Dataset configured in {duration:.2f} seconds")

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
        self.tokenizer = await load_tokenizer_for_run(self.run)

    async def _convert_media_urls_to_inline(self) -> None:
        """Download HTTP(S) image URLs and replace them with base64 data URLs.

        Collects unique URLs across all conversations/turns, downloads each once,
        and replaces all occurrences in-place. This is needed for endpoints that
        require inline media (e.g., NIM Image Retrieval).
        """
        url_to_locations = collect_http_image_urls(self.dataset.values())
        if not url_to_locations:
            return

        max_concurrency = Environment.DATASET.MEDIA_DOWNLOAD_MAX_CONCURRENCY
        self.info(
            f"Downloading {len(url_to_locations)} unique media URL(s) "
            f"for inline encoding (concurrency={max_concurrency})"
        )
        await download_and_inline_urls(url_to_locations)
        self.info("Media URL download and inline encoding complete")

    def _generate_input_payloads(self) -> InputsFile:
        """Generate input payloads from the dataset for use in the inputs.json file."""
        self.debug(
            lambda: f"Building inputs.json payloads for endpoint {self.run.cfg.endpoint.type}"
        )
        return build_inputs_file(self.run, self.dataset)

    async def _generate_inputs_json_file(self) -> None:
        """Generate inputs.json file in the artifact directory.

        OSError is logged but not re-raised (failing to write this file does not
        affect benchmark execution). Any other exception is fatal because later
        stages rely on the payload format on the worker side.
        """
        file_path = self.run.cfg.artifacts.dir / OutputDefaults.INPUTS_JSON_FILE
        temp_file_path = file_path.with_suffix(".tmp")
        self.info(f"Generating inputs.json file at {file_path.resolve()}")

        try:
            start_time = time.perf_counter()
            file_path.parent.mkdir(parents=True, exist_ok=True)
            inputs = self._generate_input_payloads()
            temp_file_path.write_bytes(
                orjson.dumps(
                    msgspec.to_builtins(inputs, enc_hook=_msgspec_enc_hook),
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
        except Exception as e:
            self.exception(
                f"Error generating inputs.json file at {file_path.resolve()}: {e!r}"
            )
            raise
        finally:
            if temp_file_path.exists():
                temp_file_path.unlink()

    async def _load_public_dataset(self) -> list[Conversation]:
        conversations, self._default_context_mode = await load_public_dataset(
            self.run, self.tokenizer
        )
        return conversations

    def _load_custom_dataset(self) -> list[Conversation]:
        conversations, self._default_context_mode = load_custom_dataset(
            self.run, self.tokenizer
        )
        return conversations

    def _load_synthetic_dataset(self) -> list[Conversation]:
        conversations, self._default_context_mode = load_synthetic_dataset(
            self.run, self.tokenizer
        )
        return conversations

    async def _load_conversations(self) -> list[Conversation]:
        """Load conversations using the composer selected by the dataset config."""
        conversations, self._default_context_mode = await load_conversations_for_run(
            self.run, self.tokenizer
        )
        return conversations

    async def _persist_conversations_to_backing_store(
        self, conversations: list[Conversation]
    ) -> DatasetClientMetadata:
        """Stream conversations into the backing store and return client metadata.

        In Kubernetes mode (compress_only=True), files are compressed during finalize();
        in local mode, uncompressed files are used directly.
        """
        await self._backing_store.initialize()
        conversations_dict = {conv.session_id: conv for conv in conversations}
        await self._backing_store.add_conversations(conversations_dict)
        await self._backing_store.finalize()

        mmap_metadata = self._backing_store.get_client_metadata()
        self.info(f"Backing store finalized: {mmap_metadata}")

        if self.run.cfg.runtime.service_run_type == ServiceRunType.KUBERNETES:
            self.info(
                "Kubernetes mode: workers will wait for DatasetDownloadedNotification "
                "from WorkerGroupManager before accessing dataset"
            )
        return mmap_metadata

    def _build_dataset_metadata(
        self, conversations: list[Conversation]
    ) -> DatasetMetadata:
        """Build the DatasetMetadata describing loaded conversations."""
        from aiperf.config.resolved import (
            conversations_have_timing_data,
            get_sampling_strategy,
        )

        dataset_config = self.run.cfg.get_default_dataset()
        return DatasetMetadata(
            conversations=[conversation.metadata() for conversation in conversations],
            sampling_strategy=get_sampling_strategy(dataset_config),
            has_timing_data=conversations_have_timing_data(conversations),
            default_context_mode=self._default_context_mode,
        )

    async def _configure_dataset(self) -> None:
        self.dataset_configured.clear()
        self._default_context_mode = None

        conversations = await self._load_conversations()

        self.dataset = {conv.session_id: conv for conv in conversations}
        self._conversation_ids_cache = [
            conversation.session_id for conversation in conversations
        ]

        endpoint_meta: EndpointMetadata = plugins.get_endpoint_metadata(
            self.run.cfg.endpoint.type
        )
        if endpoint_meta.requires_inline_media:
            await self._convert_media_urls_to_inline()

        client_metadata = await self._persist_conversations_to_backing_store(
            conversations
        )

        self.dataset_metadata = self._build_dataset_metadata(conversations)
        self.info(
            f"sampling strategy: {self.dataset_metadata.sampling_strategy}, "
            f"unique conversations: {len(self.dataset_metadata.conversations)}, "
            f"unique turn count: {self.dataset_metadata.total_turn_count}"
        )
        # Note: dataset_configured event is set in _configure_dataset_client_and_free_memory()
        # after the dataset client is initialized, to avoid a race condition where fallback
        # requests arrive before the client is ready.
        notification = DatasetConfiguredNotification(
            service_id=self.service_id,
            metadata=self.dataset_metadata,
            client_metadata=client_metadata,
            benchmark_generation=self.run.cfg.artifacts.benchmark_id,
            dataset_generation=f"{self.run.cfg.artifacts.benchmark_id}:dataset",
        )
        await self.publish(notification)
        self._rebroadcast_task = asyncio.create_task(
            self._rebroadcast_dataset_notification(notification)
        )

    async def _rebroadcast_dataset_notification(
        self, notification: DatasetConfiguredNotification
    ) -> None:
        """Rebroadcast the dataset notification every second until profile_start."""
        try:
            while True:
                await asyncio.sleep(1.0)
                await self.publish(notification)
        except asyncio.CancelledError:
            pass

    async def _get_conversation_or_raise(self, conversation_id: str) -> Conversation:
        """Wait for configuration, validate client readiness, then fetch a conversation.

        Raises `_service_error` if the client is unavailable (Kubernetes mode or not
        initialized) or the conversation is missing.
        """
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
            return await self._dataset_client.get_conversation(conversation_id)
        except KeyError as e:
            raise self._service_error(
                f"Conversation {conversation_id} not found in dataset.",
            ) from e

    @on_request(MessageType.CONVERSATION_REQUEST)
    async def _handle_conversation_request(
        self, message: ConversationRequestMessage
    ) -> ConversationResponseMessage:
        """Handle a conversation request using the dataset client."""
        self.debug(lambda: f"Handling conversation request: {message}")
        conversation = await self._get_conversation_or_raise(message.conversation_id)

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
        conversation = await self._get_conversation_or_raise(message.conversation_id)

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
        if self._rebroadcast_task is not None:
            self._rebroadcast_task.cancel()
            self._rebroadcast_task = None
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
