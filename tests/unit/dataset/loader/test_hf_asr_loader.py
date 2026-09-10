# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
from pathlib import Path
from unittest.mock import patch

import aiofiles
import numpy as np
import pytest
import soundfile as sf
from datasets import Audio as HFAudio
from datasets import Dataset

from aiperf.common.models import Conversation
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.loader.hf_asr import (
    _ASR_PROMPT,
    _MAX_DURATION_SECONDS,
    HFASRDatasetLoader,
)
from tests.unit.conftest import make_run_from_cli


@pytest.fixture
def cli_config() -> CLIConfig:
    return CLIConfig(model_names=["test-model"])


@pytest.fixture
async def loader(cli_config: CLIConfig) -> HFASRDatasetLoader:
    return HFASRDatasetLoader(
        run=make_run_from_cli(cli_config),
        hf_dataset_name="openslr/librispeech_asr",
        hf_split="test",
        hf_subset="clean",
        audio_column="audio",
    )


def _make_audio_bytes(duration_seconds: float, sr: int = 16000) -> bytes:
    """Build synthetic raw WAV bytes for the given duration."""
    num_samples = int(duration_seconds * sr)
    array = np.zeros(num_samples, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, array, sr, format="WAV")
    return buf.getvalue()


def _make_audio_row(duration_seconds: float, sr: int = 16000) -> dict:
    """Build a synthetic HF undecoded audio row (bytes + path)."""
    return {
        "audio": {"bytes": _make_audio_bytes(duration_seconds, sr), "path": "audio.wav"}
    }


@pytest.mark.asyncio
class TestAudioFromBytes:
    async def test_valid_bytes_returns_one_audio(self, loader):
        audio_value = {"bytes": _make_audio_bytes(1.0), "path": "audio.wav"}
        audios = loader._audio_from_bytes(audio_value)
        assert len(audios) == 1

    async def test_content_format_starts_with_wav(self, loader):
        audio_value = {"bytes": _make_audio_bytes(1.0), "path": "audio.wav"}
        audios = loader._audio_from_bytes(audio_value)
        assert audios[0].contents[0].startswith("wav,")

    async def test_content_decodes_to_valid_wav(self, loader):
        audio_value = {"bytes": _make_audio_bytes(0.5, sr=16000), "path": "audio.wav"}
        audios = loader._audio_from_bytes(audio_value)
        b64 = audios[0].contents[0][len("wav,") :]
        array, sr = sf.read(io.BytesIO(base64.b64decode(b64)))
        assert sr == 16000
        assert len(array) == pytest.approx(8000, rel=0.01)

    async def test_missing_bytes_returns_empty(self, loader):
        audio_value = {"bytes": None, "path": "audio.wav"}
        audios = loader._audio_from_bytes(audio_value)
        assert audios == []

    async def test_empty_dict_returns_empty(self, loader):
        audios = loader._audio_from_bytes({})
        assert audios == []

    async def test_invalid_bytes_returns_empty(self, loader):
        audio_value = {"bytes": b"not-valid-audio", "path": "audio.wav"}
        audios = loader._audio_from_bytes(audio_value)
        assert audios == []


@pytest.mark.asyncio
class TestDurationSeconds:
    async def test_returns_correct_duration(self, loader):
        audio_value = {"bytes": _make_audio_bytes(5.0, sr=16000)}
        duration = loader._duration_seconds(audio_value)
        assert duration == pytest.approx(5.0, rel=0.01)

    async def test_missing_bytes_returns_none(self, loader):
        duration = loader._duration_seconds({"bytes": None})
        assert duration is None

    async def test_invalid_bytes_returns_none(self, loader):
        duration = loader._duration_seconds({"bytes": b"not-audio"})
        assert duration is None


@pytest.mark.asyncio
class TestHFASRConvertToConversations:
    async def test_converts_rows_to_conversations(self, loader):
        data = {"dataset": [_make_audio_row(1.0)]}
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations) == 1
        assert isinstance(conversations[0], Conversation)

    async def test_prompt_is_fixed_transcription_text(self, loader):
        data = {"dataset": [_make_audio_row(1.0)]}
        conversations = await loader.convert_to_conversations(data)
        assert conversations[0].turns[0].texts[0].contents[0] == _ASR_PROMPT

    async def test_audio_attached_to_turn(self, loader):
        data = {"dataset": [_make_audio_row(1.0)]}
        conversations = await loader.convert_to_conversations(data)
        turn = conversations[0].turns[0]
        assert len(turn.audios) == 1
        assert turn.audios[0].contents[0].startswith("wav,")

    async def test_skips_clips_longer_than_max_duration(self, loader):
        data = {
            "dataset": [
                _make_audio_row(_MAX_DURATION_SECONDS + 1),
                _make_audio_row(1.0),
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations) == 1

    async def test_clip_exactly_at_max_duration_is_included(self, loader):
        data = {"dataset": [_make_audio_row(_MAX_DURATION_SECONDS)]}
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations) == 1

    async def test_skips_rows_with_none_audio(self, loader):
        data = {
            "dataset": [
                {"audio": None},
                _make_audio_row(1.0),
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations) == 1

    async def test_skips_rows_with_no_audio_column(self, loader):
        data = {
            "dataset": [
                {"transcript": "hello"},
                _make_audio_row(1.0),
            ]
        }
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations) == 1

    async def test_empty_dataset_returns_empty_list(self, loader):
        data = {"dataset": []}
        conversations = await loader.convert_to_conversations(data)
        assert conversations == []

    async def test_session_ids_are_unique(self, loader):
        data = {"dataset": [_make_audio_row(1.0) for _ in range(5)]}
        conversations = await loader.convert_to_conversations(data)
        session_ids = [c.session_id for c in conversations]
        assert len(set(session_ids)) == 5

    async def test_each_row_becomes_single_turn(self, loader):
        data = {"dataset": [_make_audio_row(1.0)]}
        conversations = await loader.convert_to_conversations(data)
        assert len(conversations[0].turns) == 1


@pytest.mark.asyncio
async def test_path_backed_hf_audio_is_loaded(
    loader: HFASRDatasetLoader, tmp_path: Path
) -> None:
    """HF keeps cached/local audio as paths when decoding is disabled."""
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(_make_audio_bytes(0.5))
    dataset = Dataset.from_dict({"audio": [str(audio_path)]}).cast_column(
        "audio", HFAudio(decode=False)
    )
    assert dataset[0]["audio"]["bytes"] is None
    conversations = await loader.convert_to_conversations({"dataset": dataset})
    assert len(conversations) == 1
    turn = conversations[0].turns[0]
    assert turn.audio_duration_seconds == pytest.approx(0.5)
    encoded = turn.audios[0].contents[0].split(",", 1)[1]
    samples, rate = sf.read(io.BytesIO(base64.b64decode(encoded)))
    assert rate == 16000
    assert len(samples) == 8000


@pytest.mark.asyncio
async def test_path_backed_long_and_missing_audio_are_skipped(
    loader: HFASRDatasetLoader, tmp_path: Path
) -> None:
    long_path = tmp_path / "long.wav"
    long_path.write_bytes(_make_audio_bytes(_MAX_DURATION_SECONDS + 1))
    rows = [
        {"audio": {"bytes": None, "path": str(long_path)}},
        {"audio": {"bytes": None, "path": str(tmp_path / "missing.wav")}},
        _make_audio_row(1.0),
    ]
    with patch(
        "aiperf.dataset.loader.hf_asr.aiofiles.open", wraps=aiofiles.open
    ) as open_audio:
        conversations = await loader.convert_to_conversations({"dataset": rows})
    assert str(long_path) not in [call.args[0] for call in open_audio.call_args_list]
    assert len(conversations) == 1
    assert conversations[0].turns[0].audio_duration_seconds == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_invalid_audio_path_does_not_abort_conversion(
    loader: HFASRDatasetLoader,
) -> None:
    rows = [
        {"audio": {"bytes": None, "path": "invalid\x00.wav"}},
        _make_audio_row(1.0),
    ]
    conversations = await loader.convert_to_conversations({"dataset": rows})
    assert len(conversations) == 1
    assert conversations[0].turns[0].audio_duration_seconds == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_embedded_audio_bytes_take_precedence_over_path(
    loader: HFASRDatasetLoader, tmp_path: Path
) -> None:
    """A metadata path must not replace an embedded recording."""
    audio_path = tmp_path / "different.wav"
    audio_path.write_bytes(_make_audio_bytes(2.0))
    row = {"audio": {"bytes": _make_audio_bytes(0.5), "path": str(audio_path)}}
    conversations = await loader.convert_to_conversations({"dataset": [row]})
    assert len(conversations) == 1
    assert conversations[0].turns[0].audio_duration_seconds == pytest.approx(0.5)
