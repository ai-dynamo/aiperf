# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for v1 InputConfig and its nested children.

Phase 1 of the v1 restoration: ensures the InputConfig DTO and all nested
child configs (conversation, prompt, image, audio, video, rankings, synthesis)
exist, accept the cyclopts-shaped input, and carry NO Pydantic field/model
validators.
"""

import inspect

from aiperf.config.v1._input import (
    AudioConfig,
    AudioLengthConfig,
    ConversationConfig,
    ImageConfig,
    ImageHeightConfig,
    ImageWidthConfig,
    InputConfig,
    InputTokensConfig,
    OutputTokensConfig,
    PrefixPromptConfig,
    PromptConfig,
    RankingsConfig,
    RankingsPassagesConfig,
    RankingsQueryConfig,
    SynthesisConfig,
    TurnConfig,
    TurnDelayConfig,
    VideoAudioConfig,
    VideoConfig,
)


def test_input_config_nested_round_trip():
    cfg = InputConfig.model_validate(
        {
            "conversation": {"num_turns": 3},
            "prompt": {"input_tokens": {"mean": 128, "stddev": 16}},
            "image": {"width": {"mean": 1024}, "height": {"mean": 768}},
        }
    )
    # `num_turns` is an alias-ignored extra; the actual field is `turn.mean`.
    # Just assert that the nested dump round-trips for the tokens we provided.
    assert cfg.prompt.input_tokens.mean == 128
    assert cfg.prompt.input_tokens.stddev == 16
    assert cfg.image.width.mean == 1024
    assert cfg.image.height.mean == 768


def test_input_config_default_construction():
    cfg = InputConfig()
    assert cfg.conversation is not None
    assert cfg.prompt is not None
    assert cfg.image is not None
    assert cfg.audio is not None
    assert cfg.video is not None
    assert cfg.rankings is not None
    assert cfg.synthesis is not None


def test_no_validators_on_input_config_or_children():
    classes = [
        InputConfig,
        ConversationConfig,
        TurnConfig,
        TurnDelayConfig,
        PromptConfig,
        InputTokensConfig,
        OutputTokensConfig,
        PrefixPromptConfig,
        ImageConfig,
        ImageWidthConfig,
        ImageHeightConfig,
        AudioConfig,
        AudioLengthConfig,
        VideoConfig,
        VideoAudioConfig,
        RankingsConfig,
        RankingsPassagesConfig,
        RankingsQueryConfig,
        SynthesisConfig,
    ]
    for cls in classes:
        bad = [
            m
            for m in inspect.getmembers(cls)
            if hasattr(m[1], "__pydantic_decorator_info__")
        ]
        assert not bad, f"{cls.__name__} must have NO validators (found: {bad})"
