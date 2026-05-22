# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.common.enums import ImageSource
from aiperf.config.dataset import SyntheticDataset
from aiperf.config.flags._converter_dataset import build_dataset
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.converter import convert_cli_to_aiperf


def _synthetic_cli(**kwargs: object) -> CLIConfig:
    return CLIConfig(
        model_names=["test-model"],
        prompt_input_tokens_mean=16,
        prompt_output_tokens_mean=8,
        image_batch_size=1,
        image_width_mean=64,
        image_height_mean=64,
        **kwargs,
    )


def test_cli_image_source_noise_flows_to_synthetic_dataset() -> None:
    user = _synthetic_cli(image_source="noise")

    ds_dict = build_dataset(user)
    assert ds_dict["images"]["source"] is ImageSource.NOISE

    aiperf_config = convert_cli_to_aiperf(user)
    main_dataset = aiperf_config.benchmark.get_default_dataset()
    assert isinstance(main_dataset, SyntheticDataset)
    assert main_dataset.images.source is ImageSource.NOISE


def test_cli_image_source_path_flows_to_synthetic_dataset(tmp_path: Path) -> None:
    source_dir = tmp_path / "source_images"
    source_dir.mkdir()
    user = _synthetic_cli(image_source=str(source_dir))

    ds_dict = build_dataset(user)
    assert ds_dict["images"]["source"] == source_dir

    aiperf_config = convert_cli_to_aiperf(user)
    main_dataset = aiperf_config.benchmark.get_default_dataset()
    assert isinstance(main_dataset, SyntheticDataset)
    assert main_dataset.images.source == source_dir


def test_cli_image_source_rejected_for_file_dataset(tmp_path: Path) -> None:
    input_file = tmp_path / "inputs.jsonl"
    input_file.write_text("{}\n")
    user = CLIConfig(input_file=str(input_file), image_source="assets")

    with pytest.raises(ValueError, match="--image-source is only supported"):
        build_dataset(user)
