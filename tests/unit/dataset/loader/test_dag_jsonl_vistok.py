# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random

import pytest
from pytest import param

from aiperf.dataset.loader.dag_jsonl_vistok import (
    VISION_SPECS,
    resolve_dimensions,
    smart_resize,
    visual_tokens,
)

# image_grid_thw / video_grid_thw products from the Qwen/Qwen3-VL-8B-Instruct and
# Qwen/Qwen2.5-VL-7B-Instruct processors (transformers 5.24.1, do_sample_frames=False).
HF_GOLDEN = [
    ("qwen3_vl", "image", 1024, 576, 1, 576),
    ("qwen3_vl", "image", 333, 777, 1, 240),
    ("qwen3_vl", "image", 4000, 3000, 1, 11750),
    ("qwen3_vl", "image", 50, 50, 1, 64),
    ("qwen3_vl", "image", 8000, 6000, 1, 16170),
    ("qwen3_vl", "video", 640, 384, 16, 1920),
    ("qwen3_vl", "video", 1280, 720, 64, 11648),
    ("qwen3_vl", "video", 98, 60, 3, 12),
    ("qwen3_vl", "video", 1920, 1080, 128, 11520),
    ("qwen3_vl", "video", 32, 16, 2, 2),
    ("qwen2_5_vl", "image", 1024, 576, 1, 777),
    ("qwen2_5_vl", "image", 333, 777, 1, 336),
    ("qwen2_5_vl", "image", 4000, 3000, 1, 15301),
    ("qwen2_5_vl", "image", 50, 50, 1, 4),
    ("qwen2_5_vl", "video", 640, 384, 16, 2576),
    ("qwen2_5_vl", "video", 1280, 720, 64, 38272),
    ("qwen2_5_vl", "video", 98, 60, 3, 16),
    ("qwen2_5_vl", "video", 1920, 1080, 128, 172224),
]


@pytest.mark.parametrize("model, kind, width, height, frames, expected", HF_GOLDEN)
def test_visual_tokens_match_hf_processors(
    model: str, kind: str, width: int, height: int, frames: int, expected: int
) -> None:
    assert (
        visual_tokens(kind, width, height, num_frames=frames, model=model) == expected
    )


@pytest.mark.parametrize(
    "model, kind, vis_tokens, frames, expected",
    [
        param("qwen3_vl", "image", 576, 1, (1024, 576), id="qwen3-image-16x9"),
        param("qwen3_vl", "image", 1369, 1, (1184, 1184), id="qwen3-image-square"),
        param("qwen3_vl", "video", 1920, 16, (640, 384), id="qwen3-video"),
        param("qwen3_vl", "video", 768, 3, (768, 512), id="qwen3-video-padded-frame"),
        param("qwen2_5_vl", "image", 576, 1, (896, 504), id="qwen25-image"),
        param("qwen2_5_vl", "video", 960, 16, (420, 224), id="qwen25-video"),
    ],
)  # fmt: skip
def test_resolve_dimensions_matches_hf_processors(
    model: str, kind: str, vis_tokens: int, frames: int, expected: tuple[int, int]
) -> None:
    assert (
        resolve_dimensions(kind, vis_tokens, num_frames=frames, model=model) == expected
    )


@pytest.mark.parametrize("model", list(VISION_SPECS))
def test_resolved_geometry_round_trips_or_raises(model: str) -> None:
    factor = VISION_SPECS[model].factor
    for tokens in range(1, 1500):
        try:
            w, h = resolve_dimensions("image", tokens, num_frames=1, model=model)
        except ValueError as e:
            assert "unreachable" in str(e)
            continue
        assert w % factor == 0 and h % factor == 0 and w % 2 == 0 and h % 2 == 0
        assert visual_tokens("image", w, h, num_frames=1, model=model) == tokens


@pytest.mark.parametrize(
    "model, kind, vis_tokens, frames, match",
    [
        param("qwen3_vl", "image", 63, 1, "unreachable", id="below-min-pixels"),
        param("qwen3_vl", "image", 577, 1, "unreachable", id="prime"),
        param("qwen3_vl", "video", 1921, 16, "not divisible by grid_t=8", id="grid-t"),
        param("qwen3_vl", "video", 120, 1, "num_frames >= 2", id="qwen3-single-frame"),
    ],
)  # fmt: skip
def test_resolve_dimensions_rejects_impossible_counts(
    model: str, kind: str, vis_tokens: int, frames: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        resolve_dimensions(kind, vis_tokens, num_frames=frames, model=model)


def test_smart_resize_matches_transformers() -> None:
    pytest.importorskip("torchvision")
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
        smart_resize as hf_image,
    )
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
        smart_resize as hf_video,
    )

    rng = random.Random(0)
    for _ in range(3000):
        h, w, frames = rng.randint(2, 9000), rng.randint(2, 9000), rng.randint(2, 300)
        if max(h, w) / min(h, w) > 200:
            continue
        for spec in VISION_SPECS.values():
            lo, hi = spec.image_pixels
            assert smart_resize(h, w, factor=spec.factor, pixels=(lo, hi)) == hf_image(
                h, w, factor=spec.factor, min_pixels=lo, max_pixels=hi
            )
        lo, hi = VISION_SPECS["qwen3_vl"].video_pixels
        ours = smart_resize(h, w, factor=32, pixels=(lo, hi), num_frames=frames)
        assert ours == hf_video(frames, h, w, 2, 32, lo, hi)
