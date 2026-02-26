# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.random_generator import RandomGenerator


def generate_noise_signal(
    rng: RandomGenerator,
    num_samples: int,
    channels: int,
) -> NDArray[np.floating]:
    """Generate a Gaussian noise signal clipped to [-1, 1].

    Args:
        rng: Random generator for reproducible output.
        num_samples: Number of audio samples to generate.
        channels: Number of audio channels (1=mono, 2=stereo).

    Returns:
        Float array of shape (num_samples,) for mono or (num_samples, channels) for stereo.
    """
    if channels < 1:
        raise ValueError(f"channels must be >= 1, got {channels}")
    shape = (num_samples, channels) if channels > 1 else num_samples
    signal = rng.normal(0, 0.3, shape)
    return np.clip(signal, -1, 1)


class BaseGenerator(AIPerfLoggerMixin, ABC):
    """Abstract base class for all data generators.

    Provides a consistent interface for generating synthetic data while allowing
    each generator type to use its own specific configuration and runtime parameters.

    Each class should create its own unique RNG in its __init__ method.
    """

    @abstractmethod
    def generate(self, *args, **kwargs) -> str:
        """Generate synthetic data.

        Args:
            *args: Variable length argument list (subclass-specific)
            **kwargs: Arbitrary keyword arguments (subclass-specific)

        Returns:
            Generated data as a string (could be text, base64 encoded media, etc.)
        """
        pass
