# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf

from aiperf.common import random_generator as rng
from aiperf.common.enums import AudioFormat
from aiperf.common.exceptions import AIPerfConfigurationError
from aiperf.dataset.generator.base import BaseGenerator, generate_noise_signal

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun

# MP3 supported sample rates in Hz
MP3_SUPPORTED_SAMPLE_RATES = {
    8000,
    11025,
    12000,
    16000,
    22050,
    24000,
    32000,
    44100,
    48000,
}

# Supported bit depths and their corresponding (numpy_type, subtype)
# Note: soundfile only accepts float32/64, int16, int32 as input arrays.
# For 8-bit output, we use int16 input and let soundfile convert to PCM_U8.
SUPPORTED_BIT_DEPTHS = {
    8: (np.int16, "PCM_U8"),
    16: (np.int16, "PCM_16"),
    24: (np.int32, "PCM_24"),  # soundfile handles 24-bit as 32-bit
    32: (np.int32, "PCM_32"),
}


class AudioGenerator(BaseGenerator):
    """
    A class for generating synthetic audio data.

    This class provides methods to create audio samples with specified
    characteristics such as format (WAV, MP3), length, sampling rate,
    bit depth, and number of channels. It supports validation of audio
    parameters to ensure compatibility with chosen formats.
    """

    def __init__(self, run: BenchmarkRun, **kwargs):
        super().__init__(run=run, **kwargs)
        # Extract audio config from dataset config
        dataset_config = run.cfg.get_default_dataset()
        self.audio_config = getattr(dataset_config, "audio", None)

        # Separate RNGs for independent concerns
        self._duration_rng = rng.derive("dataset.audio.duration")
        self._format_rng = rng.derive("dataset.audio.format")
        self._data_rng = rng.derive("dataset.audio.data")

    def _validate_sampling_rate(
        self, sampling_rate_hz: int, audio_format: AudioFormat
    ) -> None:
        """
        Validate sampling rate for the given output format.

        Args:
            sampling_rate_hz: Sampling rate in Hz
            audio_format: Audio format

        Raises:
            AIPerfConfigurationError: If sampling rate is not supported for the given format
        """
        if (
            audio_format == AudioFormat.MP3
            and sampling_rate_hz not in MP3_SUPPORTED_SAMPLE_RATES
        ):
            supported_rates = sorted(MP3_SUPPORTED_SAMPLE_RATES)
            raise AIPerfConfigurationError(
                f"MP3 format only supports the following sample rates (in Hz): {supported_rates}. "
                f"Got {sampling_rate_hz} Hz. Please choose a supported rate from the list."
            )

    def _validate_bit_depth(self, bit_depth: int) -> None:
        """
        Validate bit depth is supported.

        Args:
            bit_depth: Bit depth in bits

        Raises:
            AIPerfConfigurationError: If bit depth is not supported
        """
        if bit_depth not in SUPPORTED_BIT_DEPTHS:
            supported_depths = sorted(SUPPORTED_BIT_DEPTHS.keys())
            raise AIPerfConfigurationError(
                f"Unsupported bit depth: {bit_depth}. "
                f"Supported bit depths are: {supported_depths}"
            )

    def generate(self, *args, **kwargs) -> str:
        """Generate audio data with specified parameters.

        Returns:
            Data URI containing base64-encoded audio data with format specification

        Raises:
            AIPerfConfigurationError: If any of the following conditions are met:
                - audio length is less than 0.01 seconds
                - channels is not 1 (mono) or 2 (stereo)
                - sampling rate is not supported for MP3 format
                - bit depth is not supported (must be 8, 16, 24, or 32)
                - audio format is not supported (must be 'wav' or 'mp3')
        """
        if self.audio_config is None:
            raise ValueError("Audio config is not available in dataset config")

        num_channels = self.audio_config.channels
        if num_channels not in (1, 2):
            raise AIPerfConfigurationError(
                "Only mono (1) and stereo (2) channels are supported"
            )

        length_config = self.audio_config.length
        if length_config.mean < 0.01:
            raise AIPerfConfigurationError(
                "Audio length must be greater than 0.01 seconds"
            )

        audio_length = max(0.01, length_config.sample(self._duration_rng))

        # Randomly select sampling rate and bit depth
        sampling_rate_hz = int(
            self._format_rng.numpy_choice(self.audio_config.sample_rates) * 1000
        )  # Convert kHz to Hz
        bit_depth = self._format_rng.numpy_choice(self.audio_config.depths)

        # Validate sampling rate and bit depth
        audio_format = self.audio_config.format
        self._validate_sampling_rate(sampling_rate_hz, audio_format)
        self._validate_bit_depth(bit_depth)

        # Generate synthetic audio data (gaussian noise)
        num_samples = int(audio_length * sampling_rate_hz)
        signal = generate_noise_signal(self._data_rng, num_samples, num_channels)

        # Scale to the appropriate bit depth range
        # Note: For 8-bit, we use int16 input and let soundfile convert to PCM_U8
        numpy_type, _ = SUPPORTED_BIT_DEPTHS[bit_depth]
        scale_depth = 16 if bit_depth == 8 else bit_depth
        max_val = 2 ** (scale_depth - 1) - 1
        audio_data = (signal * max_val).astype(numpy_type)

        # Write audio using soundfile
        output_buffer = io.BytesIO()

        # Select appropriate subtype based on format
        if audio_format == AudioFormat.MP3:
            subtype = "MPEG_LAYER_III"
        elif audio_format == AudioFormat.WAV:
            _, subtype = SUPPORTED_BIT_DEPTHS[bit_depth]
        else:
            raise AIPerfConfigurationError(
                f"Unsupported audio format: {audio_format}. "
                f"Supported formats are: {AudioFormat.WAV.name}, {AudioFormat.MP3.name}"
            )

        sf.write(
            output_buffer,
            audio_data,
            sampling_rate_hz,
            format=audio_format,
            subtype=subtype,
        )
        audio_bytes = output_buffer.getvalue()

        # Encode to base64 with data URI scheme: "{format},{data}"
        base64_data = base64.b64encode(audio_bytes).decode("utf-8")
        return f"{audio_format.lower()},{base64_data}"
