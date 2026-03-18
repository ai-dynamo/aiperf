# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import Any

from aiperf.common.config import UserConfig
from aiperf.common.enums import ConversationContextMode
from aiperf.common.models import Conversation
from aiperf.common.tokenizer import Tokenizer
from aiperf.common.utils import load_json_str
from aiperf.dataset.composer.base import BaseDatasetComposer
from aiperf.dataset.loader.base_loader import BaseLoader
from aiperf.dataset.utils import check_file_exists
from aiperf.plugin import plugins
from aiperf.plugin.enums import CustomDatasetType, PluginType


class CustomDatasetComposer(BaseDatasetComposer):
    def __init__(self, config: UserConfig, tokenizer: Tokenizer | None):
        super().__init__(config, tokenizer)
        self.loader: BaseLoader | None = None

    def create_dataset(self) -> list[Conversation]:
        """Create conversations from a file or directory.

        Returns:
            list[Conversation]: A list of conversation objects.
        """
        # TODO: (future) for K8s, we need to transfer file data from SC (across node)
        check_file_exists(self.config.input.file)

        # Auto-infer dataset type if not provided
        dataset_type = self.config.input.custom_dataset_type
        if dataset_type is None:
            dataset_type = self._infer_dataset_type(self.config.input.file)
            self.info(f"Auto-detected dataset type: {dataset_type}")

        # Validate synthesis options are only used with mooncake_trace
        self._validate_synthesis_config(dataset_type)

        # Set dataset sampling strategy based on inferred type if not explicitly set
        self._set_sampling_strategy(dataset_type)

        self._create_loader_instance(dataset_type)
        dataset = self.loader.load_dataset()
        conversations = self.loader.convert_to_conversations(dataset)

        # Finalize all turns with metadata (custom datasets need this)
        for conversation in conversations:
            for turn in conversation.turns:
                self._finalize_turn(turn)

        # Finalize conversation-level context prompts
        self._finalize_conversations(conversations)
        return conversations

    def get_default_context_mode(self) -> ConversationContextMode | None:
        """Delegate to the loader's format-specific default, if a loader was created."""
        if self.loader is not None:
            return self.loader.get_default_context_mode()
        return None

    def _infer_dataset_type(self, file_path: str) -> CustomDatasetType:
        """Infer the custom dataset type from the input file.

        Queries all registered loaders to check if they can handle the data format.

        Args:
            file_path: Path to the JSONL file or directory

        Returns:
            CustomDatasetType if successfully inferred

        Raises:
            ValueError: If no loader can handle the data format
        """
        try:
            path = Path(file_path)

            # If it's a directory, use path-based detection only
            if path.is_dir():
                return self._infer_type(data=None, filename=file_path)

            # For files, read first non-empty line and use both content and path detection
            with open(file_path) as f:
                for line in f:
                    if not (line := line.strip()):
                        continue
                    data = load_json_str(line)
                    return self._infer_type(data=data, filename=file_path)

        except ValueError as e:
            self.exception(
                f"Error inferring dataset type from file: {file_path}: {e!r}"
            )
            raise

    def _infer_type(
        self, data: dict[str, Any] | None = None, filename: str | Path | None = None
    ) -> CustomDatasetType:
        """Infer the dataset type from data and/or filename.

        First checks for explicit 'type' field in the data, then falls back to
        structural detection by querying registered loaders via the factory.

        Args:
            data: Optional dictionary representing a single line from the JSONL file.
                  None indicates path-based detection only (e.g., for directories).
            filename: Optional path to the input file/directory for path-based detection

        Returns:
            CustomDatasetType if successfully inferred

        Raises:
            ValueError: If the type field is invalid or no loader can handle the data format
        """
        # Check for explicit type field first (most efficient).
        # Skip values that aren't known dataset types (e.g. Bailian's "type": "text"
        # is a request type, not a dataset type) and fall through to structural detection.
        if data is not None and data.get("type") in CustomDatasetType:
            explicit_type = CustomDatasetType(data["type"])
            LoaderClass = plugins.get_class(
                PluginType.CUSTOM_DATASET_LOADER, explicit_type
            )
            if not LoaderClass.can_load(data, filename):
                raise ValueError(
                    f"Explicit type field {explicit_type} specified, but loader {LoaderClass.__name__} "
                    "cannot handle the data format. Please specify --custom-dataset-type explicitly."
                )
            self.info(f"Using explicit type field: {explicit_type}")
            return explicit_type

        detected_type = None
        for entry, LoaderClass in plugins.iter_all(PluginType.CUSTOM_DATASET_LOADER):
            if LoaderClass.can_load(data, filename):
                self.info(
                    f"Loader {LoaderClass.__name__} can handle the input file data format."
                )
                dataset_type = CustomDatasetType(entry.name)
                if detected_type is not None:
                    raise ValueError(
                        f"Multiple loaders can handle the data format: {detected_type} and {dataset_type}. "
                        "Please specify --custom-dataset-type explicitly."
                    )
                detected_type = dataset_type

        if detected_type is None:
            raise ValueError(
                "No loader can handle the data format. Please specify --custom-dataset-type explicitly."
            )

        return detected_type

    def _set_sampling_strategy(self, dataset_type: CustomDatasetType) -> None:
        """Set the dataset sampling strategy based on the dataset type.

        If the user has not explicitly set a sampling strategy, use the loader's
        preferred strategy.

        Args:
            dataset_type: The type of custom dataset
        """
        if self.config.input.dataset_sampling_strategy is None:
            LoaderClass = plugins.get_class(
                PluginType.CUSTOM_DATASET_LOADER, dataset_type
            )
            preferred_strategy = LoaderClass.get_preferred_sampling_strategy()
            self.config.input.dataset_sampling_strategy = preferred_strategy
            self.info(
                f"Using preferred sampling strategy for {dataset_type}: {preferred_strategy}"
            )

    def _validate_synthesis_config(self, dataset_type: CustomDatasetType) -> None:
        """Validate that synthesis options are only used with trace datasets.

        Args:
            dataset_type: The determined dataset type.

        Raises:
            ValueError: If synthesis options are set but dataset type is not a trace format.
        """
        if (
            self.config.input.synthesis.should_synthesize()
            and not plugins.is_trace_dataset(dataset_type)
        ):
            raise ValueError(
                f"Synthesis options (--synthesis-speedup-ratio, --synthesis-prefix-len-multiplier, "
                f"--synthesis-prefix-root-multiplier, --synthesis-prompt-len-multiplier) "
                f"are only supported with trace datasets, "
                f"but got {dataset_type}. "
                f"Either remove synthesis options or use a trace dataset type."
            )

    def _create_loader_instance(self, dataset_type: CustomDatasetType) -> None:
        """Initializes the dataset loader based on the custom dataset type.

        Args:
            dataset_type: The type of custom dataset to create.
        """
        kwargs: dict[str, Any] = {}
        loader_metadata = plugins.get_dataset_loader_metadata(dataset_type)
        if loader_metadata.is_trace:
            if self.prompt_generator is None:
                raise ValueError(
                    "Trace datasets require a tokenizer for prompt synthesis. "
                    "Ensure the endpoint supports tokenization or provide a --tokenizer."
                )
            kwargs["prompt_generator"] = self.prompt_generator

            if loader_metadata.default_block_size is not None:
                kwargs["default_block_size"] = loader_metadata.default_block_size

        elif dataset_type == CustomDatasetType.RANDOM_POOL:
            kwargs["num_conversations"] = self.config.input.conversation.num

        LoaderClass = plugins.get_class(PluginType.CUSTOM_DATASET_LOADER, dataset_type)
        self.loader = LoaderClass(
            filename=self.config.input.file,
            user_config=self.config,
            **kwargs,
        )
