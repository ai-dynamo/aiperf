# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
        param("qwen3_vl", "video", 16120, 3, (3552, 1888), id="qwen3-video-clamped-preimage"),
        param("qwen3_vl", "video", 16302, 3, (4000, 1600), id="qwen3-video-upscaled-grid-preimage"),
        param("qwen3_vl", "video", 13680, 7, (3296, 1952), id="qwen3-video-nearest-16x9-preimage"),
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


# Outputs of transformers 5.17.0 qwen2_vl image / qwen3_vl video ``smart_resize``.
SMART_RESIZE_HF_GOLDEN = [
    param("qwen3_vl", 576, 1024, None, (576, 1024), id="q3-img-16x9"),
    param("qwen3_vl", 50, 50, None, (256, 256), id="q3-img-min-clamp"),
    param("qwen3_vl", 6000, 8000, None, (3520, 4704), id="q3-img-max-clamp"),
    param("qwen3_vl", 777, 333, None, (768, 320), id="q3-img-portrait"),
    param("qwen3_vl", 40, 8000, None, (32, 8000), id="q3-img-aspect-200"),
    param("qwen2_5_vl", 576, 1024, None, (588, 1036), id="q25-img-16x9"),
    param("qwen2_5_vl", 30, 30, None, (56, 56), id="q25-img-min-clamp"),
    param("qwen2_5_vl", 3000, 4000, None, (2996, 4004), id="q25-img-max-clamp"),
    param("qwen2_5_vl", 42, 70, None, (56, 56), id="q25-img-round-half"),
    param("qwen3_vl", 384, 640, 16, (384, 640), id="q3-vid-16x9"),
    param("qwen3_vl", 2176, 4096, 3, (2080, 3968), id="q3-vid-max-clamp"),
    param("qwen3_vl", 1080, 1920, 128, (320, 576), id="q3-vid-many-frames"),
    param("qwen3_vl", 16, 32, 2, (32, 64), id="q3-vid-below-factor"),
    param("qwen3_vl", 60, 98, 3, (64, 96), id="q3-vid-min-clamp"),
    param("qwen3_vl", 720, 1280, 5, (704, 1280), id="q3-vid-odd-frames"),
]  # fmt: skip


@pytest.mark.parametrize(
    "model, height, width, frames, expected", SMART_RESIZE_HF_GOLDEN
)
def test_smart_resize_matches_transformers(
    model: str, height: int, width: int, frames: int | None, expected: tuple[int, int]
) -> None:
    spec = VISION_SPECS[model]
    pixels = spec.image_pixels if frames is None else spec.video_pixels
    assert (
        smart_resize(
            height, width, factor=spec.factor, pixels=pixels, num_frames=frames
        )
        == expected
    )


def test_resolve_dimensions_memoizes_preimage_scan() -> None:
    resolve_dimensions.cache_clear()
    first = resolve_dimensions("video", 16302, num_frames=3, model="qwen3_vl")
    second = resolve_dimensions("video", 16302, num_frames=3, model="qwen3_vl")
    info = resolve_dimensions.cache_info()
    assert first == second == (4000, 1600)
    assert info.misses == 1 and info.hits == 1
