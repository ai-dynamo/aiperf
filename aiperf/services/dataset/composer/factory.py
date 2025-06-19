# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from aiperf.services.dataset.composer.base import BaseDatasetComposer
from aiperf.services.dataset.composer.custom import CustomDatasetComposer
from aiperf.services.dataset.composer.synthetic import SyntheticDatasetComposer
from aiperf.services.dataset.config import DatasetConfig, PromptConfig
from aiperf.services.dataset.enums import ComposerType

logger = logging.getLogger(__name__)


class ComposerFactory:
    """
    Factory class for creating dataset composer instances.

    This factory follows the Factory Method pattern and provides a centralized
    way to create different types of dataset composers based on the configuration.

    Example usage:
        # Basic usage
        config = DatasetConfig(...)
        composer = ComposerFactory.create_composer(ComposerType.SYNTHETIC, config)

        # Check available types
        available_types = ComposerFactory.get_available_types()
    """

    # Registry mapping composer types to their corresponding classes
    _registry: dict[ComposerType, type[BaseDatasetComposer]] = {
        ComposerType.SYNTHETIC: SyntheticDatasetComposer,
        ComposerType.CUSTOM: CustomDatasetComposer,
    }

    @classmethod
    def create_instance(
        cls,
        composer_type: ComposerType,
        config: "MockConfig",  # noqa: F821 - TODO: replace with real config
    ) -> BaseDatasetComposer:
        """Create a dataset composer instance based on the specified type.

        Args:
            composer_type: The type of composer to create
            config: Dataset configuration object

        Returns:
            An instance of the requested composer type

        Raises:
            ValueError: If the composer type is not supported
        """
        if composer_type not in cls._registry:
            available_types = list(cls._registry.keys())
            raise ValueError(
                f"Unsupported composer type: {composer_type}. "
                f"Available types: {available_types}"
            )

        composer_class = cls._registry[composer_type]
        logger.info("Creating %s instance", composer_class.__name__)

        try:
            # create config to pass to composer
            dataset_config = DatasetConfig(
                filename=config.filename,
                custom_dataset_type=config.custom_dataset_type,
                num_conversations=config.num_conversations,
                prompt=PromptConfig(
                    tokenizer=config.tokenizer,
                    mean=config.prompt.mean,
                    stddev=config.prompt.stddev,
                ),
            )
            return composer_class(dataset_config)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create composer of type {composer_type}: {e}"
            ) from e

    @classmethod
    def get_available_types(cls) -> list[ComposerType]:
        """
        Get a list of all available composer types.

        Returns:
            List of available composer types
        """
        return list(cls._registry.keys())

    @classmethod
    def is_type_supported(cls, composer_type: ComposerType) -> bool:
        """
        Check if a composer type is supported by the factory.

        Args:
            composer_type: The type to check

        Returns:
            True if the type is supported, False otherwise
        """
        return composer_type in cls._registry
