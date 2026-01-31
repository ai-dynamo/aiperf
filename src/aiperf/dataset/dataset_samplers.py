# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataset sampling strategy implementations."""

from abc import ABC, abstractmethod

from aiperf.common import random_generator as rng


class BaseDatasetSampler(ABC):
    """
    Base class for dataset samplers.

    Any class implementing this protocol will be provided a list of conversation ids, and must
    provide a `next_conversation_id` method that returns the next conversation id.

    Each sampler that requires randomness should create its own RNG in its __init__ method
    using rng.derive() with a unique identifier (e.g., "dataset.sampler.random_choice").
    """

    def __init__(self, conversation_ids: list[str], **kwargs) -> None:
        if not conversation_ids:
            raise ValueError("conversation_ids cannot be empty")
        self._conversation_ids = list(conversation_ids)

    @abstractmethod
    def next_conversation_id(self) -> str:
        """Return the next conversation id."""
        ...


class RandomSampler(BaseDatasetSampler):
    """
    Random sampler that randomly selects conversation IDs with replacement.
    Can return the same conversation ID multiple times before seeing all IDs.
    """

    def __init__(self, conversation_ids: list[str], **kwargs) -> None:
        super().__init__(conversation_ids, **kwargs)
        self._rng = rng.derive("dataset.sampler.random")

    def next_conversation_id(self) -> str:
        return self._rng.choice(self._conversation_ids)


class SequentialSampler(BaseDatasetSampler):
    """
    Sequential sampler that iterates through conversation IDs in order.
    When reaching the end, it wraps around to the beginning indefinitely.

    This sampler is completely deterministic and does not use any randomness.
    """

    def __init__(self, conversation_ids: list[str], **kwargs) -> None:
        super().__init__(conversation_ids, **kwargs)
        self._index: int = 0

    def next_conversation_id(self) -> str:
        if self._index >= len(self._conversation_ids):
            self._index = 0
        conversation_id = self._conversation_ids[self._index]
        self._index += 1
        return conversation_id


class ShuffleSampler(BaseDatasetSampler):
    """
    Shuffle sampler that randomly samples without replacement, then repeats.

    Shuffles all conversation IDs, iterates through them once, then reshuffles and repeats indefinitely.
    Similar to music shuffle - ensures all conversations are seen before any repetition.
    """

    def __init__(self, conversation_ids: list[str], **kwargs) -> None:
        super().__init__(conversation_ids, **kwargs)
        self._rng = rng.derive("dataset.sampler.shuffle")
        self._rng.shuffle(self._conversation_ids)
        self._index: int = 0

    def next_conversation_id(self) -> str:
        if self._index >= len(self._conversation_ids):
            self._rng.shuffle(self._conversation_ids)
            self._index = 0
        conversation_id = self._conversation_ids[self._index]
        self._index += 1
        return conversation_id
