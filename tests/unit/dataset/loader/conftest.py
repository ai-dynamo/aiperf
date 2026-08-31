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
    """Force WekaTraceLoader serial reconstruction in unit tests.

    The parallel path spawns worker processes that load a real tokenizer via
    ``Tokenizer.from_pretrained``, which most tests stub out with a MagicMock
    that doesn't survive process boundaries. Tests that specifically exercise
    the parallel path drive ``_process_task`` in-process or override this
    setting via ``monkeypatch``.

    Uses ``monkeypatch`` rather than manual save/restore so a per-test override
    of the same setting unwinds LIFO through the one function-scoped monkeypatch.
    A manual ``finally`` restore racing a per-test monkeypatch of the same global
    attribute leaked stale values across tests under xdist sharding.
    """
    from aiperf.common import environment as env_mod

    monkeypatch.setattr(env_mod.Environment.DATASET, "WEKA_PARALLEL_WORKERS", 1)


@pytest.fixture(autouse=True)
def _disable_weka_aux_classification(monkeypatch):
    """Keep flattened-agent worker chains tagged ``::fa:`` in the loader suite.

    Worker-chain sub-classification (``is_aux_chain`` size/cross-model arms,
    ``is_reduction_chain``, and ``worker_group_members``) relabels short
    one-shots as ``::aux:`` sidecars and shared-spawn fan-out as
    ``::wg:``. The flat-split mechanics tests assert ``::fa:`` session ids and
    byte-identical reconstruction and are agnostic to those tags, so default all
    three off here. Classification itself is covered directly by
    ``test_weka_aux_classification`` and end-to-end by the aux/reduction/
    worker-group cases in ``test_weka_flat_split_v1_contract_adv`` and
    ``test_weka_async_subagent`` (which re-enable them via monkeypatch).

    Uses ``monkeypatch`` rather than manual save/restore: a per-test override of
    these same globals (e.g. a test that re-enables aux) and this fixture then go
    through the SAME function-scoped monkeypatch and unwind LIFO, so neither can
    leak a stale value into another test. The previous manual ``finally`` restore
    raced per-test monkeypatches and leaked aux-enabled state into sibling loader
    tests under xdist sharding, flipping ``::fa:`` assertions to ``::aux:``.
    """
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
    """Build a real v2 ``BenchmarkRun`` for WekaTraceLoader unit tests.

    Replaces the v1 ``_mk_user_config()`` MagicMock pattern. The loader reads
    config off ``run.cfg.*`` (FileDataset trace knobs, synthesis max_isl/max_osl,
    tokenizer, models, fixed-schedule phase offsets), so a real BenchmarkRun is
    required -- a MagicMock no longer satisfies the loader's reads.

    ``block_size`` has no FileDataset home in v2 (only synthetic prompts carry
    it), so it is forwarded to the loader via ``default_block_size`` by callers,
    not through config. It is accepted here for signature compatibility but
    ignored at the config layer.
    """
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
        # Use inline records so the FileDataset source XOR is satisfied without
        # a path; the WekaTraceLoader is constructed directly with filename=,
        # so this dataset block only carries the trace-replay knobs the loader
        # reads via get_default_dataset().
        "records": [{"text": "placeholder"}],
        # weka_trace so max_context_length / weka-only knobs pass FileDataset
        # validation; the loader is still constructed with filename= directly.
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
    """Wire a deterministic stub for ``_hash_id_corpus_rng`` on a MagicMock pg.

    The Weka loader's ``_decode_block_tokens`` reseeds the hash-id RNG before
    every uncached block and slices the corpus at ``randrange(corpus_size)``.
    Tests that mock ``prompt_generator`` need stable, hash-id-derived offsets so
    cached blocks have non-empty content and per-(scope, hash_id) determinism.
    """
    state = {"h": 0}

    def _reseed(h):
        state["h"] = h

    prompt_generator._hash_id_corpus_rng.reseed_for_hash_id.side_effect = _reseed
    prompt_generator._hash_id_corpus_rng.randrange.side_effect = lambda n: (
        state["h"] % n
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

    # Cleanup all created files
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
    """Create temporary test images copied from source assets.

    Returns:
        A dictionary mapping image names to their temporary file paths.
    """
    # Get the source images directory
    source_images_dir = Path("src/aiperf/dataset/generator/assets/source_images")

    # Get some actual image files
    source_images = list(source_images_dir.glob("*.jpg"))[:4]

    if not source_images:
        # Create a minimal synthetic JPEG image if no source images found
        from PIL import Image

        synthetic_path = tmp_path / "image1.jpg"
        img = Image.new("RGB", (1, 1), color="red")
        img.save(synthetic_path, format="JPEG")
        return {"image1.jpg": str(synthetic_path)}

    # Create temporary copies preserving original file extensions
    image_map = {}
    for i, source_img in enumerate(source_images, 1):
        # Preserve the original file extension to avoid MIME/encoder mismatches
        dest_filename = f"image{i}{source_img.suffix}"
        dest_path = tmp_path / dest_filename
        shutil.copy(source_img, dest_path)
        image_map[dest_filename] = str(dest_path)

    return image_map


@pytest.fixture
def create_test_image(tmp_path):
    """Create a single test image copied from source assets.

    Returns:
        A function that creates a test image with the given name.
    """
    source_images_dir = Path("src/aiperf/dataset/generator/assets/source_images")
    source_images = list(source_images_dir.glob("*.jpg"))

    def _create_image(name: str = "test_image.jpg"):
        from PIL import Image

        dest_path = tmp_path / name
        requested_ext = Path(name).suffix.lower()

        if source_images:
            # Load the source image and save it in the requested format
            img = Image.open(source_images[0])
            if requested_ext in [".jpg", ".jpeg"]:
                img.save(dest_path, format="JPEG")
            elif requested_ext == ".png":
                img.save(dest_path, format="PNG")
            else:
                # Default to JPEG
                img.save(dest_path, format="JPEG")
        else:
            # Create a minimal synthetic image matching the requested format
            img = Image.new("RGB", (1, 1), color="red")
            if requested_ext in [".jpg", ".jpeg"]:
                img.save(dest_path, format="JPEG")
            elif requested_ext == ".png":
                img.save(dest_path, format="PNG")
            else:
                # Default to JPEG
                img.save(dest_path, format="JPEG")

        return str(dest_path)

    return _create_image


@pytest.fixture
def create_test_audio(tmp_path):
    """Create test audio files (WAV and MP3).

    Returns:
        A function that creates a test audio file with the given name.
    """
    import wave

    import numpy as np

    def _create_audio(name: str = "test_audio.wav"):
        dest_path = tmp_path / name

        # Generate simple sine wave audio
        sample_rate = 16000
        duration = 0.1  # 100ms
        frequency = 440  # A4 note

        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * frequency * t)

        # Convert to 16-bit PCM
        audio_data = (audio_data * 32767).astype(np.int16)

        # Write WAV file
        with wave.open(str(dest_path), "wb") as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        return str(dest_path)

    return _create_audio


@pytest.fixture
def create_test_video(tmp_path):
    """Create test video files (MP4).

    Returns:
        A function that creates a test video file with the given name.
    """
    from PIL import Image, ImageDraw

    def _create_video(name: str = "test_video.mp4"):
        dest_path = tmp_path / name

        # Try using ffmpeg-python if available, otherwise create a minimal MP4
        try:
            import tempfile

            import ffmpeg

            # Create a few simple frames
            temp_frame_dir = tempfile.mkdtemp(prefix="video_frames_")
            for i in range(3):
                img = Image.new("RGB", (64, 64), (i * 80, 0, 0))
                draw = ImageDraw.Draw(img)
                draw.text((10, 25), f"F{i}", fill=(255, 255, 255))
                img.save(f"{temp_frame_dir}/frame_{i:03d}.png")

            # Use ffmpeg to create video
            (
                ffmpeg.input(f"{temp_frame_dir}/frame_%03d.png", framerate=1)
                .output(str(dest_path), vcodec="libvpx-vp9", pix_fmt="yuv420p", t=1)
                .overwrite_output()
                .run(quiet=True)
            )

            for file in Path(temp_frame_dir).glob("*.png"):
                file.unlink()
            Path(temp_frame_dir).rmdir()

        except (ImportError, Exception):
            # Fallback: a minimal MP4 header stub with no encoded frames — it
            # won't play, and loader tests only need a file that base64-encodes.
            # Branded vp09 to match the libvpx-vp9 path above, not H.264.
            minimal_mp4 = bytes.fromhex(
                "000000186674797069736f6d0000020069736f6d69736f32767030390000"
                "0008667265650000002c6d6461740000001c6d6f6f7600000000006d7668"
                "6400000000000000000000000000000001000000"
            )
            with open(dest_path, "wb") as f:
                f.write(minimal_mp4)

        return str(dest_path)

    return _create_video
