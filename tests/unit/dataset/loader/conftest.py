# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.composer.custom import CustomDatasetComposer


@pytest.fixture(autouse=True)
def _disable_weka_parallel_reconstruction(monkeypatch):
    """Force WekaTraceLoader serial reconstruction in unit tests."""
    from aiperf.common import environment as env_mod

    monkeypatch.setattr(env_mod.Environment.DATASET, "WEKA_PARALLEL_WORKERS", 1)


@pytest.fixture(autouse=True)
def _disable_weka_aux_classification(monkeypatch):
    """Keep flattened-agent worker chains tagged ``::fa:`` in the loader suite."""
    from aiperf.common import environment as env_mod

    ds = env_mod.Environment.DATASET
    monkeypatch.setattr(ds, "WEKA_AUX_MAX_REQUESTS", 0)
    monkeypatch.setattr(ds, "WEKA_AUX_REDUCTION_OSL_MAX", 0)
    monkeypatch.setattr(ds, "WEKA_WORKER_GROUP_MIN", 0)


def make_weka_run(
    *,
    model_names=("test-model",),
    tokenizer_name: str | None = "test-tok",
    tokenizer_revision: str | None = None,
    trust_remote_code: bool = False,
    block_size: int | None = None,
    max_isl: int | None = None,
    max_osl: int | None = None,
    max_context_length: int | None = None,
    entries: int | None = None,
    ignore_trace_delays: bool = False,
    use_think_time_only: bool = False,
    inter_turn_delay_cap_seconds: float | None = None,
    trace_idle_gap_cap_seconds: float | None = None,
    fixed_schedule_start_offset: int | None = None,
    fixed_schedule_end_offset: int | None = None,
    random_seed: int | None = 0,
):
    """Build a real v2 ``BenchmarkRun`` for WekaTraceLoader unit tests."""
    from aiperf.config import BenchmarkConfig, BenchmarkRun

    models = list(model_names)
    synthesis: dict = {}
    if max_isl is not None:
        synthesis["max_isl"] = max_isl
    if max_osl is not None:
        synthesis["max_osl"] = max_osl

    dataset: dict = {
        "name": "default",
        "type": "file",
        "records": [{"text": "placeholder"}],
        "format": "weka_trace",
    }
    if synthesis:
        dataset["synthesis"] = synthesis
    if max_context_length is not None:
        dataset["max_context_length"] = max_context_length
    if entries is not None:
        dataset["entries"] = entries
    if ignore_trace_delays:
        dataset["ignore_trace_delays"] = ignore_trace_delays
    if use_think_time_only:
        dataset["use_think_time_only"] = use_think_time_only
    if inter_turn_delay_cap_seconds is not None:
        dataset["inter_turn_delay_cap_seconds"] = inter_turn_delay_cap_seconds
    if trace_idle_gap_cap_seconds is not None:
        dataset["trace_idle_gap_cap_seconds"] = trace_idle_gap_cap_seconds

    if fixed_schedule_start_offset is not None or fixed_schedule_end_offset is not None:
        phase: dict = {
            "name": "profiling",
            "type": "fixed_schedule",
            "auto_offset": False,
        }
        if fixed_schedule_start_offset is not None:
            phase["start_offset"] = fixed_schedule_start_offset
        if fixed_schedule_end_offset is not None:
            phase["end_offset"] = fixed_schedule_end_offset
    else:
        phase = {
            "name": "profiling",
            "type": "concurrency",
            "requests": 100,
            "concurrency": 1,
        }

    tokenizer: dict = {}
    if tokenizer_name is not None:
        tokenizer["name"] = tokenizer_name
    if tokenizer_revision is not None:
        tokenizer["revision"] = tokenizer_revision
    tokenizer["trust_remote_code"] = trust_remote_code

    cfg = BenchmarkConfig.model_validate(
        {
            "models": models,
            "endpoint": {
                "urls": ["http://localhost:8000/v1/chat/completions"],
                "wait_for_model_timeout": 0,
            },
            "datasets": [dataset],
            "phases": [phase],
            "tokenizer": tokenizer,
            "runtime": {"ui": "simple"},
        }
    )
    return BenchmarkRun(
        benchmark_id="test-weka-run",
        cfg=cfg,
        artifact_dir=cfg.artifacts.dir,
        random_seed=random_seed,
    )


def stub_hash_id_corpus_rng(prompt_generator) -> None:
    """Wire a deterministic stub for ``_hash_id_corpus_rng`` on a MagicMock pg."""
    state = {"h": 0}

    def _reseed(h):
        state["h"] = h

    prompt_generator._hash_id_corpus_rng.reseed_for_hash_id.side_effect = _reseed
    prompt_generator._hash_id_corpus_rng.randrange.side_effect = (
        lambda n: state["h"] % n
    )


@pytest.fixture
def create_jsonl_file():
    """Create a temporary JSONL file with custom content."""
    filename = None

    def _create_file(content_lines):
        nonlocal filename
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for line in content_lines:
                f.write(line + "\n")
            filename = f.name
        return filename

    yield _create_file

    if filename:
        Path(filename).unlink(missing_ok=True)


@pytest.fixture
def create_cfg_and_composer(mock_tokenizer_cls):
    """Create a CLIConfig and CustomDatasetComposer for testing."""

    from tests.unit.conftest import make_run_from_cli

    def _create():
        config = CLIConfig.model_construct(
            model_names=["test-model"],
            input_file="test_data.jsonl",
            conversation_num=5,
        )
        tokenizer = mock_tokenizer_cls.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        )
        run = make_run_from_cli(config)
        composer = CustomDatasetComposer(run=run, tokenizer=tokenizer)
        return config, composer

    return _create


@pytest.fixture
def default_cfg() -> CLIConfig:
    """Create a default CLIConfig for testing."""
    return CLIConfig(model_names=["test-model"])


@pytest.fixture
def mock_prompt_generator():
    """Create a mock prompt generator for loader tests."""
    generator = Mock()
    generator.generate.return_value = "Generated prompt text"
    generator._decoded_cache = {}
    generator._build_token_sequence.return_value = [1, 2, 3, 4, 5]
    return generator


@pytest.fixture
def test_images(tmp_path):
    """Create temporary test images copied from source assets."""
    source_images_dir = Path("src/aiperf/dataset/generator/assets/source_images")

    source_images = list(source_images_dir.glob("*.jpg"))[:4]

    if not source_images:
        from PIL import Image

        synthetic_path = tmp_path / "image1.jpg"
        img = Image.new("RGB", (1, 1), color="red")
        img.save(synthetic_path, format="JPEG")
        return {"image1.jpg": str(synthetic_path)}

    image_map = {}
    for i, source_img in enumerate(source_images, 1):
        dest_filename = f"image{i}{source_img.suffix}"
        dest_path = tmp_path / dest_filename
        shutil.copy(source_img, dest_path)
        image_map[dest_filename] = str(dest_path)

    return image_map


@pytest.fixture
def create_test_image(tmp_path):
    """Create a single test image copied from source assets."""
    source_images_dir = Path("src/aiperf/dataset/generator/assets/source_images")
    source_images = list(source_images_dir.glob("*.jpg"))

    def _create_image(name: str = "test_image.jpg"):
        from PIL import Image

        dest_path = tmp_path / name
        requested_ext = Path(name).suffix.lower()

        if source_images:
            img = Image.open(source_images[0])
            if requested_ext in [".jpg", ".jpeg"]:
                img.save(dest_path, format="JPEG")
            elif requested_ext == ".png":
                img.save(dest_path, format="PNG")
            else:
                img.save(dest_path, format="JPEG")
        else:
            img = Image.new("RGB", (1, 1), color="red")
            if requested_ext in [".jpg", ".jpeg"]:
                img.save(dest_path, format="JPEG")
            elif requested_ext == ".png":
                img.save(dest_path, format="PNG")
            else:
                img.save(dest_path, format="JPEG")

        return str(dest_path)

    return _create_image


@pytest.fixture
def create_test_audio(tmp_path):
    """Create test audio files (WAV and MP3)."""
    import wave

    import numpy as np

    def _create_audio(name: str = "test_audio.wav"):
        dest_path = tmp_path / name

        sample_rate = 16000
        duration = 0.1
        frequency = 440

        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * frequency * t)

        audio_data = (audio_data * 32767).astype(np.int16)

        with wave.open(str(dest_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        return str(dest_path)

    return _create_audio


@pytest.fixture
def create_test_video(tmp_path):
    """Create test video files (MP4)."""
    from PIL import Image, ImageDraw

    def _create_video(name: str = "test_video.mp4"):
        dest_path = tmp_path / name

        try:
            import tempfile

            import ffmpeg

            temp_frame_dir = tempfile.mkdtemp(prefix="video_frames_")
            for i in range(3):
                img = Image.new("RGB", (64, 64), (i * 80, 0, 0))
                draw = ImageDraw.Draw(img)
                draw.text((10, 25), f"F{i}", fill=(255, 255, 255))
                img.save(f"{temp_frame_dir}/frame_{i:03d}.png")

            (
                ffmpeg.input(f"{temp_frame_dir}/frame_%03d.png", framerate=1)
                .output(str(dest_path), vcodec="libx264", pix_fmt="yuv420p", t=1)
                .overwrite_output()
                .run(quiet=True)
            )

            for file in Path(temp_frame_dir).glob("*.png"):
                file.unlink()
            Path(temp_frame_dir).rmdir()

        except (ImportError, Exception):
            minimal_mp4 = bytes.fromhex(
                "000000186674797069736f6d0000020069736f6d69736f32617663310000"
                "0008667265650000002c6d6461740000001c6d6f6f7600000000006d7668"
                "6400000000000000000000000000000001000000"
            )
            with open(dest_path, "wb") as f:
                f.write(minimal_mp4)

        return str(dest_path)

    return _create_video
