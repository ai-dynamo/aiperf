# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random
import uuid

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.enums import (
    CommAddress,
    MessageType,
    ModelSelectionStrategy,
    ServiceType,
)
from aiperf.common.factories import ServiceFactory
from aiperf.common.hooks import on_init, on_pull_message
from aiperf.common.messages import (
    ProcessDatasetResponseMessage,
    ProcessMooncakeTraceDatasetMessage,
    ProcessSyntheticDatasetMessage,
)
from aiperf.common.mixins import PullClientMixin
from aiperf.common.models import Audio, Conversation, Image, Text, Turn
from aiperf.common.protocols import PushClientProtocol
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset import AudioGenerator, ImageGenerator, PromptGenerator, utils
from aiperf.dataset.loader import MediaConversionMixin  # TODO: move to common.mixins


@ServiceFactory.register(ServiceType.DATASET_PROCESSOR)
class DatasetProcessor(PullClientMixin, BaseComponentService, MediaConversionMixin):
    """
    DatasetProcessor is responsible for generating dataset conversations in parallel.
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
            pull_client_address=CommAddress.DATASET_JOB,
            pull_client_bind=False,
        )
        self.debug("Dataset processor __init__")
        self.results_push_client: PushClientProtocol = self.comms.create_push_client(
            CommAddress.DATASET_RESULT
        )
        self.tokenizer: Tokenizer | None = None
        self.model_selection_counter: int | None = None

        self.user_config = user_config
        self._conversation_config = user_config.input.conversation
        self._prompt_config = user_config.input.prompt
        self._image_config = user_config.input.image
        self._audio_config = user_config.input.audio
        self._endpoint_config = user_config.endpoint

    @property
    def include_prompt(self) -> bool:
        return self._prompt_config.input_tokens.mean > 0

    @property
    def include_image(self) -> bool:
        return self._image_config.width.mean > 0 and self._image_config.height.mean > 0

    @property
    def include_audio(self) -> bool:
        return self._audio_config.length.mean > 0

    @property
    def prefix_prompt_enabled(self) -> bool:
        return self._prompt_config.prefix_prompt.length > 0

    @on_init
    async def _initialize(self) -> None:
        """Initialize dataset processor service-specific components."""
        self.debug("Initializing dataset processor service")
        tokenizer_name = self.user_config.tokenizer.name
        if tokenizer_name is None:
            tokenizer_name = self._endpoint_config.model_names[0]

        self.tokenizer = Tokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=self.user_config.tokenizer.trust_remote_code,
            revision=self.user_config.tokenizer.revision,
        )

        self.prompt_generator = PromptGenerator(self._prompt_config, self.tokenizer)
        self.image_generator = ImageGenerator(self._image_config)
        self.audio_generator = AudioGenerator(self._audio_config)

    @on_pull_message(MessageType.PROCESS_SYNTHETIC_DATASET)
    async def _on_process_synthetic_dataset(
        self,
        message: ProcessSyntheticDatasetMessage,
    ) -> None:
        """Handle a dataset generation job."""
        # TODO: change to debug log
        self.info(
            lambda: f"#### ({self.service_id}) Received synthetic dataset generation job to process {message.num_conversations} conversations"
        )

        if message.random_seed is not None:
            random.seed(message.random_seed)
            self.debug(lambda: f"Setting random seed to {message.random_seed}")

        self.model_selection_counter = 0

        conversations = []
        for _ in range(message.num_conversations):
            conversation = await self._create_conversation()
            conversations.append(conversation)

        await self.results_push_client.push(
            ProcessDatasetResponseMessage(
                service_id=self.service_id,
                generated_data=conversations,
            )
        )

    async def _create_conversation(self) -> Conversation:
        """Create a synthetic conversation from the given configuration.

        It generates a set of conversations with a varying number of turns,
        where each turn contains synthetic text, image, and audio payloads.

        Returns:
            Conversation: a conversation objects.
        """
        conversation = Conversation(session_id=str(uuid.uuid4()))

        num_turns = utils.sample_positive_normal_integer(
            self._conversation_config.turn.mean,
            self._conversation_config.turn.stddev,
        )
        self.debug("Creating conversation with %d turns", num_turns)

        for turn_idx in range(num_turns):
            turn = self._create_turn(is_first=(turn_idx == 0))
            conversation.turns.append(turn)
        return conversation

    def _create_turn(self, is_first: bool) -> Turn:
        """Create a turn object that contains synthetic payloads to send.

        It generates multi-modal data (e.g. text, image, audio) using synthetic
        generators and also the delay between turns.

        Args:
            is_first: Whether the turn is the first turn in the conversation.

        Returns:
            Turn: A dataset representation of a single turn.
        """
        turn = Turn()

        if self.include_prompt:
            turn.texts.append(self._generate_text_payloads(is_first))
        if self.include_image:
            turn.images.append(self._generate_image_payloads())
        if self.include_audio:
            turn.audios.append(self._generate_audio_payloads())

        # Add randomized delays between each turn. Skip if first turn.
        if not is_first:
            turn.delay = utils.sample_positive_normal_integer(
                self._conversation_config.turn.delay.mean,
                self._conversation_config.turn.delay.stddev,
            )

        if not turn.texts and not turn.images and not turn.audios:
            self.logger.warning(
                "There were no synthetic payloads generated. "
                "Please enable at least one of prompt, image, or audio by "
                "setting the mean to a positive value."
            )

        self._finalize_turn(turn)

        return turn

    def _generate_text_payloads(self, is_first: bool) -> Text:
        """Generate synthetic text payloads.

        If the turn is the first turn in the conversation, it could add a prefix prompt
        to the prompt.

        Args:
            is_first: Whether the turn is the first turn in the conversation.

        Returns:
            Text: A text payload object.
        """
        text = Text(name="text")
        for _ in range(self._prompt_config.batch_size):
            prompt = self.prompt_generator.generate(
                mean=self._prompt_config.input_tokens.mean,
                stddev=self._prompt_config.input_tokens.stddev,
            )

            if self.prefix_prompt_enabled and is_first:
                # TODO: Rename
                prefix_prompt = self.prompt_generator.get_random_prefix_prompt()
                prompt = f"{prefix_prompt} {prompt}"

            text.contents.append(prompt)
        return text

    def _generate_image_payloads(self) -> Image:
        """
        Generate synthetic images if the image width and height are specified.

        Returns:
            Image: An image payload object.
        """
        image = Image(name="image_url")
        for _ in range(self._image_config.batch_size):
            data = self.image_generator.generate()
            image.contents.append(data)
        return image

    def _generate_audio_payloads(self) -> Audio:
        """
        Generate synthetic audios if the audio length is specified.

        Returns:
            Audio: An audio payload object.
        """
        audio = Audio(name="input_audio")
        for _ in range(self._audio_config.batch_size):
            data = self.audio_generator.generate()
            audio.contents.append(data)
        return audio

    def _select_model_name(self) -> str:
        """Select a model name based on the model selection strategy.

        Returns:
            str: The selected model name.
        """
        strategy = self._endpoint_config.model_selection_strategy
        if strategy == ModelSelectionStrategy.RANDOM:
            return random.choice(self._endpoint_config.model_names)
        elif strategy == ModelSelectionStrategy.ROUND_ROBIN:
            index = self.model_selection_counter % len(
                self._endpoint_config.model_names
            )
            model_name = self._endpoint_config.model_names[index]
            self.model_selection_counter += 1
            return model_name

    def _set_max_tokens(self, turn: Turn) -> None:
        """Set max_tokens for the turn based on the output configuration.

        Args:
            turn: The turn object to finalize.
        """
        if self._prompt_config.output_tokens.mean is not None:
            turn.max_tokens = utils.sample_positive_normal_integer(
                mean=self._prompt_config.output_tokens.mean,
                stddev=self._prompt_config.output_tokens.stddev,
            )

    def _finalize_turn(self, turn: Turn) -> None:
        """Finalize a turn by populating all required metadata fields.

        This method handles:
        - Model name selection
        - Max tokens sampling based on output configuration
        - Any other turn-level metadata that needs to be set

        Args:
            turn: The turn object to finalize.
        """
        turn.model = self._select_model_name()
        self._set_max_tokens(turn)

    @on_pull_message(MessageType.PROCESS_MOONCAKE_TRACE_DATASET)
    async def _on_process_mooncake_trace_dataset(
        self,
        message: ProcessMooncakeTraceDatasetMessage,
    ) -> None:
        """Handle a mooncake trace dataset generation job."""
        self.debug(lambda: "Received mooncake trace dataset generation job")

        if message.random_seed is not None:
            random.seed(message.random_seed)
            self.info(f"{self.service_id} setting random seed to {message.random_seed}")

        # TODO: implement model selection strategy
        self.model_selection_counter = 0

        conversations = []
        for session_id, traces in message.dataset:
            conversation = Conversation(session_id=session_id)
            for trace in traces:
                prompt = self.prompt_generator.generate(
                    mean=trace["input_length"],
                    stddev=0,
                    hash_ids=trace["hash_ids"],
                )
                turn = Turn(
                    timestamp=trace["timestamp"],
                    texts=[Text(name="text", contents=[prompt])],
                )
                conversation.turns.append(turn)
            conversations.append(conversation)

        await self.results_push_client.push(
            ProcessDatasetResponseMessage(
                service_id=self.service_id,
                generated_data=conversations,
            )
        )

    #
    # @on_pull_message(MessageType.PROCESS_MULTI_TURN_DATASET)
    # async def _on_process_multi_turn_dataset(
    #    self, message: ProcessMultiTurnDatasetMessage,
    # ) -> None:
    #    """Handle a multi-turn dataset generation job."""
    #    self.debug(lambda: f"Received multi-turn dataset generation job: {message}")
    #
    #    if message.random_seed is not None:
    #        random.seed(message.random_seed)
    #        self.debug(lambda: f"Setting random seed to {message.random_seed}")
    #
    #    # TODO: implement model selection strategy
    #    self.model_selection_counter = 0
    #
    #    conversations = []
    #    for session_id, multi_turns in message.data.items():
    #        conversation = Conversation(session_id=session_id)
    #        for multi_turn in multi_turns:
    #            for single_turn in multi_turn.turns:
    #                media = self.convert_to_media_objects(single_turn)
    #                conversation.turns.append(
    #                    Turn(
    #                        texts=media[MediaType.TEXT],
    #                        images=media[MediaType.IMAGE],
    #                        audios=media[MediaType.AUDIO],
    #                        timestamp=single_turn.timestamp,
    #                        delay=single_turn.delay,
    #                        role=single_turn.role,
    #                    )
    #                )
    #        conversations.append(conversation)

    #    await self.results_push_client.push(
    #        DatasetResultMessage(
    #            service_id=self.service_id,
    #            success=True,
    #            generated_data=conversations,
    #        )
    #    )
    #
    # @on_pull_message(MessageType.PROCESS_SINGLE_TURN_DATASET)
    # async def _on_process_single_turn_dataset(
    #    self, message: ProcessSingleTurnDatasetMessage,
    # ) -> None:
    #    """Handle a single-turn dataset generation job."""
    #    self.debug(lambda: f"Received single-turn dataset generation job: {message}")
    #
    #    if message.random_seed is not None:
    #        random.seed(message.random_seed)
    #        self.debug(lambda: f"Setting random seed to {message.random_seed}")
    #
    #    # TODO: implement model selection strategy
    #    self.model_selection_counter = 0
    #
    #    conversations = []
    #    for session_id, single_turns in message.data.items():
    #        conversation = Conversation(session_id=session_id)
    #        for single_turn in single_turns:
    #            media = self.convert_to_media_objects(single_turn)
    #            conversation.turns.append(
    #                Turn(
    #                    texts=media[MediaType.TEXT],
    #                    images=media[MediaType.IMAGE],
    #                    audios=media[MediaType.AUDIO],
    #                    timestamp=single_turn.timestamp,
    #                    delay=single_turn.delay,
    #                    role=single_turn.role,
    #                )
    #            )
    #        conversations.append(conversation)
    #
    #    await self.results_push_client.push(
    #        DatasetResultMessage(
    #            service_id=self.service_id,
    #            job_id=message.job_id,
    #            success=True,
    #            generated_data=conversations,
    #        )
    #    )
    #
    # @on_pull_message(MessageType.PROCESS_RANDOM_POOL_DATASET)
    # async def _on_process_random_pool_dataset(
    #     self,
    #     message: ProcessRandomPoolDatasetMessage,
    # ) -> None:
    #     """Handle a random pool dataset generation job."""
    #     self.debug(lambda: f"Received random pool dataset generation job: {message}")

    #     if message.random_seed is not None:
    #         random.seed(message.random_seed)
    #         self.debug(lambda: f"Setting random seed to {seed}")

    #     # TODO: implement model selection strategy
    #     self.model_selection_counter = 0

    #     # TODO: add random pool dataset
    #     conversations = [
    #         Conversation(session_id=str(uuid.uuid4()))
    #         for _ in range(self.num_conversations)
    #     ]

    #     # F x N (F: num of files, N: num of conversations)
    #     sampled_dataset: dict[Filename, list[Turn]] = {}

    #     # Randomly sample (with replacement) from each dataset pool
    #     for filename, dataset_pool in data.items():
    #         samples = random.choices(dataset_pool, k=self.num_conversations)
    #         turns: list[Turn] = []
    #         for sample in samples:
    #             media = self.convert_to_media_objects(sample, name=Path(filename).stem)
    #             turns.append(
    #                 Turn(
    #                     texts=media[MediaType.TEXT],
    #                     images=media[MediaType.IMAGE],
    #                     audios=media[MediaType.AUDIO],
    #                 )
    #             )
    #         sampled_dataset[filename] = turns

    #     # Merge turns for each conversation
    #     for i, batched_turns in enumerate(zip(*sampled_dataset.values(), strict=False)):
    #         turn = self._merge_turns(batched_turns)
    #         conversations[i].turns.append(turn)

    #     await self.results_push_client.push(
    #         ProcessDatasetResponseMessage(
    #             service_id=self.service_id,
    #             job_id=message.job_id,
    #             success=True,
    #             generated_data=conversations,
    #         )
    #     )

    # def _merge_turns(self, turns: list[Turn]) -> Turn:
    #     """Merge turns into a single turn.

    #     Args:
    #         turns: A list of turns.

    #     Returns:
    #         A single turn.
    #     """
    #     merged_turn = Turn(
    #         texts=[text for turn in turns for text in turn.texts],
    #         images=[image for turn in turns for image in turn.images],
    #         audios=[audio for turn in turns for audio in turn.audios],
    #     )
    #     return merged_turn


def main() -> None:
    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(DatasetProcessor)


if __name__ == "__main__":
    main()
