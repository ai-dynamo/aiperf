# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import gc
import time
from typing import TYPE_CHECKING

import orjson

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.config import OutputDefaults, ServiceConfig, UserConfig
from aiperf.common.enums import (
    CommAddress,
    CommandType,
    CreditPhase,
    MessageType,
    PublicDatasetType,
)
from aiperf.common.environment import Environment
from aiperf.common.hooks import on_command, on_request, on_stop
from aiperf.common.messages import (
    ConversationRequestMessage,
    ConversationResponseMessage,
    ConversationTurnRequestMessage,
    ConversationTurnResponseMessage,
    DatasetConfiguredNotification,
    ProfileConfigureCommand,
)
from aiperf.common.mixins import ReplyClientMixin
from aiperf.common.models import (
    Conversation,
    DatasetMetadata,
    InputsFile,
    ModelEndpointInfo,
    RequestInfo,
    SessionPayloads,
)
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.loader import ShareGPTLoader
from aiperf.plugin import plugins
from aiperf.plugin.enums import (
    ComposerType,
    DatasetBackingStoreType,
    PluginType,
)

if TYPE_CHECKING:
    from aiperf.dataset.protocols import (
        DatasetBackingStoreProtocol,
        DatasetClientStoreProtocol,
    )
    from aiperf.endpoints.protocols import EndpointProtocol


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
    ) -> None:
        super().__init__(
            service_config=service_config,
            user_config=user_config,
            service_id=service_id,
            reply_client_address=CommAddress.DATASET_MANAGER_PROXY_BACKEND,
            reply_client_bind=False,
        )
        self.user_config = user_config
        self.tokenizer: Tokenizer | None = None
        self.dataset: dict[
            str, Conversation
        ] = {}  # conversation ID -> Conversation mapping
        self.dataset_metadata: DatasetMetadata | None = None
        self._conversation_ids_cache: list[str] = []
        self.dataset_configured = asyncio.Event()

        BackingStoreClass = plugins.get_class(
            PluginType.DATASET_BACKING_STORE, DatasetBackingStoreType.MEMORY_MAP
        )
        self._backing_store: DatasetBackingStoreProtocol = BackingStoreClass(
            benchmark_id=user_config.benchmark_id,
        )
        self._dataset_client: DatasetClientStoreProtocol | None = None

    @on_command(CommandType.PROFILE_CONFIGURE)
    async def _profile_configure_command(
        self, message: ProfileConfigureCommand
    ) -> None:
        """Configure the dataset."""

        self.info("Configuring tokenizer(s) for dataset manager")
        begin = time.perf_counter()
        await self._configure_tokenizer()
        duration = time.perf_counter() - begin
        self.info(lambda: f"Tokenizer(s) configured in {duration:.2f} seconds")

        self.info(lambda: f"Configuring dataset for {self.service_id}")
        begin = time.perf_counter()
        await self._configure_dataset()
        await self._generate_inputs_json_file()
        await self._configure_dataset_client_and_free_memory()

        duration = time.perf_counter() - begin
        self.info(lambda: f"Dataset configured in {duration:.2f} seconds")

    async def _configure_dataset_client_and_free_memory(self) -> None:
        """Configure the dataset client for serving fallback requests."""
        # Create dataset client for serving fallback requests, then free in-memory dataset
        client_metadata = self._backing_store.get_client_metadata()
        ClientStoreClass = plugins.get_class(
            PluginType.DATASET_CLIENT_STORE, client_metadata.client_type
        )
        self._dataset_client = ClientStoreClass(client_metadata=client_metadata)
        await self._dataset_client.initialize()
        # Now that the client is ready, signal that fallback requests can be served
        self.dataset_configured.set()
        # Free the in-memory dataset now that we have the client to serve fallback requests.
        # Reassign to new empty containers (not .clear()) to release object references,
        # then run gc.collect() twice to ensure circular references are cleaned up.
        conversation_count = len(self.dataset)
        self.dataset = {}
        self._conversation_ids_cache = []
        gc.collect()
        gc.collect()
        self.info(
            f"Dataset client initialized and freed {conversation_count} conversations from memory"
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

    def _generate_input_payloads(
        self,
        model_endpoint: ModelEndpointInfo,
    ) -> InputsFile:
        """Generate input payloads from the dataset for use in the inputs.json file."""
        inputs = InputsFile()

        EndpointClass = plugins.get_class(
            PluginType.ENDPOINT, model_endpoint.endpoint.type
        )
        endpoint: EndpointProtocol = EndpointClass(model_endpoint=model_endpoint)
        self.debug(
            lambda: f"Created endpoint protocol for {model_endpoint.endpoint.type}, "
            f"class: {endpoint.__class__.__name__}",
        )
        session_payloads_map: dict[str, list] = {}
        for conversation in self.dataset.values():
            session_id = conversation.session_id
            if session_id not in session_payloads_map:
                session_payloads_map[session_id] = []

            for i, turn in enumerate(conversation.turns):
                request_info = RequestInfo(
                    model_endpoint=model_endpoint,
                    turns=[turn],
                    turn_index=i,
                    credit_num=i,
                    credit_phase=CreditPhase.PROFILING,
                    x_request_id="",
                    x_correlation_id="",
                    conversation_id=conversation.session_id,
                )
                request_info.endpoint_headers = endpoint.get_endpoint_headers(
                    request_info
                )
                request_info.endpoint_params = endpoint.get_endpoint_params(
                    request_info
                )
                payload = endpoint.format_payload(request_info)
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
        public_dataset_type = self.user_config.input.public_dataset
        match public_dataset_type:
            case PublicDatasetType.SHAREGPT:
                loader = ShareGPTLoader(self.user_config, self.tokenizer)
            case _:
                raise ValueError(
                    f"Unsupported public dataset type: {public_dataset_type}"
                )

        dataset = await loader.load_dataset()
        # Only use loader's recommended strategy if user hasn't explicitly set one
        if "dataset_sampling_strategy" not in self.user_config.input.model_fields_set:
            self.user_config.input.dataset_sampling_strategy = (
                loader.get_recommended_sampling_strategy()
            )
        return await loader.convert_to_conversations(dataset)

    def _load_custom_dataset(self) -> list[Conversation]:
        ComposerClass = plugins.get_class(
            PluginType.DATASET_COMPOSER, ComposerType.CUSTOM
        )
        composer = ComposerClass(config=self.user_config, tokenizer=self.tokenizer)
        return composer.create_dataset()

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
        return composer.create_dataset()

    async def _configure_dataset(self) -> None:
        if self.user_config is None:
            raise self._service_error("User config is required for dataset manager")

        self.dataset_configured.clear()

        if self.user_config.input.public_dataset is not None:
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

        # Initialize backing store and stream conversations to mmap files
        # Workers read directly from these files
        await self._backing_store.initialize()
        conversations_dict = {conv.session_id: conv for conv in conversations}
        await self._backing_store.add_conversations(conversations_dict)
        await self._backing_store.finalize()
        client_metadata = self._backing_store.get_client_metadata()
        self.info(f"Backing store finalized: {client_metadata}")

        self.dataset_metadata = DatasetMetadata(
            conversations=[conversation.metadata() for conversation in conversations],
            sampling_strategy=self.user_config.input.dataset_sampling_strategy,
        )
        self.info(
            f"sampling strategy: {self.dataset_metadata.sampling_strategy}, "
            f"unique conversations: {len(self.dataset_metadata.conversations)}, "
            f"unique turn count: {self.dataset_metadata.total_turn_count}"
        )
        # Note: dataset_configured event is set in _profile_configure_command after
        # the dataset client is initialized, to avoid a race condition where fallback
        # requests arrive before the client is ready.
        await self.publish(
            DatasetConfiguredNotification(
                service_id=self.service_id,
                metadata=self.dataset_metadata,
                client_metadata=client_metadata,
            )
        )

    @on_request(MessageType.CONVERSATION_REQUEST)
    async def _handle_conversation_request(
        self, message: ConversationRequestMessage
    ) -> ConversationResponseMessage:
        """Handle a conversation request using the dataset client."""
        self.debug(lambda: f"Handling conversation request: {message}")

        await self._wait_for_dataset_configuration()

        if self._dataset_client is None:
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
