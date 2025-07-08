# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io

import numpy as np
import soundfile as sf

from aiperf.common.config import AudioConfig
from aiperf.common.enums import AudioFormat
from aiperf.common.exceptions import GeneratorConfigurationError
from aiperf.services.dataset import utils
from aiperf.services.dataset.generator.base import BaseGenerator

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

# Supported bit depths and their corresponding numpy types
SUPPORTED_BIT_DEPTHS = {
    8: (np.int8, "PCM_S8"),
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

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config

    def _validate_sampling_rate(
        self, sampling_rate_hz: int, audio_format: AudioFormat
    ) -> None:
        """
        Validate sampling rate for the given output format.

        Args:
            sampling_rate_hz: Sampling rate in Hz
            audio_format: Audio format

        Raises:
            GeneratorConfigurationError: If sampling rate is not supported for the given format
        """
        if (
            audio_format == AudioFormat.MP3
            and sampling_rate_hz not in MP3_SUPPORTED_SAMPLE_RATES
        ):
            supported_rates = sorted(MP3_SUPPORTED_SAMPLE_RATES)
            raise GeneratorConfigurationError(
                f"MP3 format only supports the following sample rates (in Hz): {supported_rates}. "
                f"Got {sampling_rate_hz} Hz. Please choose a supported rate from the list."
            )

    def _validate_bit_depth(self, bit_depth: int) -> None:
        """
        Validate bit depth is supported.

        Args:
            bit_depth: Bit depth in bits

        Raises:
            GeneratorConfigurationError: If bit depth is not supported
        """
        if bit_depth not in SUPPORTED_BIT_DEPTHS:
            supported_depths = sorted(SUPPORTED_BIT_DEPTHS.keys())
            raise GeneratorConfigurationError(
                f"Unsupported bit depth: {bit_depth}. "
                f"Supported bit depths are: {supported_depths}"
            )

    def generate(self, *args, **kwargs) -> str:
        """Generate audio data with specified parameters.

        Returns:
            Data URI containing base64-encoded audio data with format specification

        Raises:
            GeneratorConfigurationError: If any of the following conditions are met:
                - audio length is less than 0.01 seconds
                - channels is not 1 (mono) or 2 (stereo)
                - sampling rate is not supported for MP3 format
                - bit depth is not supported (must be 8, 16, 24, or 32)
                - audio format is not supported (must be 'wav' or 'mp3')
        """
        if self.config.num_channels not in (1, 2):
            raise GeneratorConfigurationError(
                "Only mono (1) and stereo (2) channels are supported"
            )

        if self.config.length.mean < 0.01:
            raise GeneratorConfigurationError(
                "Audio length must be greater than 0.01 seconds"
            )

        # Sample audio length (in seconds) using rejection sampling
        audio_length = utils.sample_normal(
            self.config.length.mean, self.config.length.stddev, lower=0.01
        )

        # Randomly select sampling rate and bit depth
        sampling_rate_hz = int(
            np.random.choice(self.config.sample_rates) * 1000
        )  # Convert kHz to Hz
        bit_depth = np.random.choice(self.config.depths)

        # Validate sampling rate and bit depth
        self._validate_sampling_rate(sampling_rate_hz, self.config.format)
        self._validate_bit_depth(bit_depth)

        # Generate synthetic audio data (gaussian noise)
        num_samples = int(audio_length * sampling_rate_hz)
        audio_data = np.random.normal(
            0,
            0.3,
            (
                (num_samples, self.config.num_channels)
                if self.config.num_channels > 1
                else num_samples
            ),
        )

        # Ensure the signal is within [-1, 1] range
        audio_data = np.clip(audio_data, -1, 1)

        # Scale to the appropriate bit depth range
        max_val = 2 ** (bit_depth - 1) - 1
        numpy_type, _ = SUPPORTED_BIT_DEPTHS[bit_depth]
        audio_data = (audio_data * max_val).astype(numpy_type)

        # Write audio using soundfile
        output_buffer = io.BytesIO()

        # Select appropriate subtype based on format
        if self.config.format == AudioFormat.MP3:
            subtype = "MPEG_LAYER_III"
        elif self.config.format == AudioFormat.WAV:
            _, subtype = SUPPORTED_BIT_DEPTHS[bit_depth]
        else:
            raise GeneratorConfigurationError(
                f"Unsupported audio format: {self.config.format}. "
                f"Supported formats are: {AudioFormat.WAV.name}, {AudioFormat.MP3.name}"
            )

        sf.write(
            output_buffer,
            audio_data,
            sampling_rate_hz,
            format=self.config.format,
            subtype=subtype,
        )
        audio_bytes = output_buffer.getvalue()

        # Encode to base64 with data URI scheme: "{format},{data}"
        base64_data = base64.b64encode(audio_bytes).decode("utf-8")
        return f"{self.config.format.lower()},{base64_data}"
