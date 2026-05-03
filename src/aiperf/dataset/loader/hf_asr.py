# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
from typing import Any, TypedDict

import soundfile as sf
from datasets import Audio as HFAudio

from aiperf.common.config.user_config import UserConfig
from aiperf.common.models import Audio, Conversation, Text, Turn
from aiperf.dataset.loader.base_hf_dataset import BaseHFDatasetLoader

_ASR_PROMPT = "Transcribe this audio."
_MAX_DURATION_SECONDS = 30


class _HFAudioBytesRow(TypedDict, total=False):
    """Shape of an HF audio dict when loaded with HFAudio(decode=False)."""

    bytes: bytes | None
    path: str


class HFASRDatasetLoader(BaseHFDatasetLoader):
    """HuggingFace dataset loader for ASR (automatic speech recognition) datasets.

    Attaches audio from a configurable column and uses a fixed transcription
    prompt. Clips longer than 30 seconds are skipped to match standard ASR
    benchmark conventions.

    Uses HFAudio(decode=False) to receive raw audio bytes and decodes them
    with soundfile, avoiding the torchcodec dependency entirely.

    Example plugins.yaml entry::

        librispeech:
          class: aiperf.dataset.loader.hf_asr:HFASRDatasetLoader
          metadata:
            hf_dataset_name: openslr/librispeech_asr
            hf_split: test
            hf_subset: clean
            audio_column: audio
    """

    def __init__(
        self,
        user_config: UserConfig,
        audio_column: str = "audio",
        **kwargs,
    ) -> None:
        self.audio_column = audio_column
        super().__init__(user_config=user_config, **kwargs)

    def _audio_from_bytes(self, audio_value: _HFAudioBytesRow) -> list[Audio]:
        """Decode raw HF audio bytes into an AIPerf Audio object.

        When HFAudio(decode=False) is used, each audio value is a dict with
        'bytes' (raw file bytes) and 'path' keys. We decode with soundfile
        and re-encode as WAV base64 in the format expected by the chat endpoint.
        """
        raw_bytes = audio_value.get("bytes")
        if not raw_bytes:
            return []
        try:
            array, sr = sf.read(io.BytesIO(raw_bytes))
            buf = io.BytesIO()
            sf.write(buf, array, sr, format="WAV")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return [Audio(name="", contents=[f"wav,{b64}"])]
        except (OSError, ValueError, RuntimeError) as e:
            self.debug(
                lambda exc=e: f"Failed to decode audio bytes: {exc.__class__.__name__}: {exc}"
            )
            return []

    def _duration_seconds(self, audio_value: _HFAudioBytesRow) -> float | None:
        """Estimate clip duration in seconds from raw bytes via soundfile."""
        raw_bytes = audio_value.get("bytes")
        if not raw_bytes:
            return None
        try:
            info = sf.info(io.BytesIO(raw_bytes))
            return info.duration
        except (OSError, ValueError, RuntimeError) as e:
            self.debug(
                lambda exc=e: f"Failed to estimate audio duration: {exc.__class__.__name__}: {exc}"
            )
            return None

    async def convert_to_conversations(
        self, data: dict[str, Any]
    ) -> list[Conversation]:
        """Convert each ASR dataset row into a single-turn audio Conversation."""
        dataset = data["dataset"]
        # Disable HF audio decoding so we handle it ourselves with soundfile,
        # avoiding the torchcodec dependency.
        if hasattr(dataset, "cast_column"):
            dataset = dataset.cast_column(self.audio_column, HFAudio(decode=False))
        conversations = []
        skipped = 0
        max_conversations = self._max_conversations()

        for row in dataset:
            if (
                max_conversations is not None
                and len(conversations) >= max_conversations
            ):
                break

            audio_value = row.get(self.audio_column)
            if not isinstance(audio_value, dict):
                skipped += 1
                continue

            duration = self._duration_seconds(audio_value)
            if duration is not None and duration > _MAX_DURATION_SECONDS:
                skipped += 1
                continue

            audios = self._audio_from_bytes(audio_value)
            if not audios:
                skipped += 1
                continue

            conversations.append(
                Conversation(
                    session_id=self.session_id_generator.next(),
                    turns=[
                        Turn(
                            texts=[Text(contents=[_ASR_PROMPT])],
                            audios=audios,
                            audio_duration_seconds=duration,
                        )
                    ],
                )
            )

        if skipped > 0 and not conversations:
            self.warning(
                f"All {skipped} rows were skipped — no conversations loaded. "
                f"Check that '{self.audio_column}' contains valid audio data."
            )
        self.debug(
            lambda: f"Converted {len(conversations)} rows (skipped {skipped} empty/long)"
        )
        return conversations
