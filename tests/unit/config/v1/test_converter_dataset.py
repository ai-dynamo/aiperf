# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for v1 build_dataset converter.

Covers the four-way dataset-type discrimination (synthetic / file / public / composed)
and the field-mapping from nested ``user.input.*`` paths to the v2 dataset shape.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.config.v1 import UserConfig
from aiperf.config.v1._converter_dataset import build_dataset


@pytest.fixture
def existing_file(tmp_path) -> str:
    """Create a file ``parse_file`` validator will accept."""
    p = tmp_path / "data.jsonl"
    p.write_text("{}\n")
    return str(p)


@pytest.mark.parametrize(
    "input_cfg_factory,expected_type",
    [
        param(
            lambda _f: {"prompt": {"input_tokens": {"mean": 128}}},
            "synthetic",
            id="synthetic-with-prompt",
        ),
        param(lambda _f: {}, "synthetic", id="synthetic-default-empty"),
        param(lambda f: {"file": f}, "file", id="file"),
        param(
            lambda _f: {"public_dataset": "sharegpt"}, "public", id="public"
        ),
    ],
)  # fmt: skip
def test_build_dataset_picks_type(
    input_cfg_factory, expected_type: str, existing_file: str
) -> None:
    cfg = input_cfg_factory(existing_file)
    user = UserConfig.model_validate({"input": cfg} if cfg else {})
    out = build_dataset(user)
    assert out["type"] == expected_type


def test_build_dataset_public_uses_dataset_field() -> None:
    """Public datasets carry the source name in 'dataset' field, NOT 'name'.

    v2 PublicDataset.dataset is the source name (e.g. 'sharegpt'); the wrapper
    later sets .name = 'main' on the entry.
    """
    user = UserConfig.model_validate({"input": {"public_dataset": "sharegpt"}})
    out = build_dataset(user)
    assert out["type"] == "public"
    assert out["dataset"] == "sharegpt"


def test_build_dataset_file_with_augment_trigger_is_composed(
    existing_file: str,
) -> None:
    """File + augment-trigger field (osl) -> composed dataset.

    The original ``_cli_dataset.py`` chooses 'composed' when ``input_file`` is
    set AND any augment-trigger field (osl_mean, prefix, image-batch-size, ...)
    is also set. Here we trigger via OSL on the prompt's output_tokens.
    """
    user = UserConfig.model_validate(
        {
            "input": {
                "file": existing_file,
                "prompt": {"output_tokens": {"mean": 64}},
            },
        }
    )
    out = build_dataset(user)
    assert out["type"] == "composed"
    assert out["source"]["type"] == "file"
    assert out["augment"]["osl"]["mean"] == 64


def test_build_dataset_file_without_augment_is_plain_file(
    existing_file: str,
) -> None:
    user = UserConfig.model_validate({"input": {"file": existing_file}})
    out = build_dataset(user)
    assert out["type"] == "file"
    assert str(out["path"]) == existing_file


def test_build_dataset_synthetic_carries_prompt_distribution() -> None:
    user = UserConfig.model_validate(
        {"input": {"prompt": {"input_tokens": {"mean": 128, "stddev": 16}}}},
    )
    out = build_dataset(user)
    assert out["type"] == "synthetic"
    assert "prompts" in out
    assert out["prompts"]["isl"] == {"mean": 128, "stddev": 16}


def test_build_dataset_includes_random_seed_when_set() -> None:
    user = UserConfig.model_validate({"input": {"random_seed": 42}})
    out = build_dataset(user)
    assert out.get("random_seed") == 42


def test_build_dataset_synthetic_emits_default_isl_mean_when_no_prompt() -> None:
    """Synthetic must always have at least an ISL mean for downstream synthesis.

    Mirrors the ``_apply_dataset_type`` fallback ``setdefault(...["mean"], 550)``.
    """
    user = UserConfig.model_validate({})
    out = build_dataset(user)
    assert out["type"] == "synthetic"
    assert out["prompts"]["isl"]["mean"] == 550


def test_build_dataset_public_with_hf_subset() -> None:
    user = UserConfig.model_validate(
        {"input": {"public_dataset": "sharegpt", "hf_dataset_subset": "sharegpt4o"}}
    )
    out = build_dataset(user)
    assert out["type"] == "public"
    assert out["hf_subset"] == "sharegpt4o"


def test_build_dataset_synthetic_carries_image_batch() -> None:
    user = UserConfig.model_validate(
        {"input": {"image": {"batch_size": 2, "width": {"mean": 256.0}}}}
    )
    out = build_dataset(user)
    assert out["type"] == "synthetic"
    assert out["images"]["batch_size"] == 2
    assert out["images"]["width"]["mean"] == 256.0


def test_build_dataset_synthetic_carries_turns_and_delay() -> None:
    user = UserConfig.model_validate(
        {
            "input": {
                "conversation": {
                    "turn": {"mean": 3, "stddev": 1, "delay": {"mean": 100.0}},
                },
            },
        }
    )
    out = build_dataset(user)
    assert out["type"] == "synthetic"
    assert out["turns"] == {"mean": 3, "stddev": 1}
    assert out["turn_delay"]["mean"] == 100.0
