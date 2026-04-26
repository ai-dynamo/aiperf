# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""v1 DTO contract: small nested configs have no validators and round-trip cleanly."""

import inspect

from aiperf.config.v1._accuracy import AccuracyConfig
from aiperf.config.v1._output import OutputConfig
from aiperf.config.v1._tokenizer import TokenizerConfig


def test_output_config_round_trip():
    cfg = OutputConfig.model_validate({"artifact_directory": "/tmp/x"})
    assert str(cfg.artifact_directory) == "/tmp/x" or cfg.artifact_directory == "/tmp/x"


def test_tokenizer_config_round_trip():
    cfg = TokenizerConfig.model_validate({"name": "gpt2"})
    assert cfg.name == "gpt2"


def test_accuracy_config_instantiates():
    cfg = AccuracyConfig.model_validate({})
    assert cfg is not None


def test_no_validators_on_small_configs():
    for cls in (OutputConfig, TokenizerConfig, AccuracyConfig):
        bad = [
            m
            for m in inspect.getmembers(cls)
            if hasattr(m[1], "__pydantic_decorator_info__")
        ]
        assert not bad, f"{cls.__name__} must have NO validators (found: {bad})"
