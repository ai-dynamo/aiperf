# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for grid-recipe expansion through the v1->v2 converter."""

from __future__ import annotations

import pytest

from aiperf.config.v1 import ServiceConfig, UserConfig
from aiperf.config.v1._endpoint import EndpointConfig
from aiperf.config.v1._loadgen import LoadGeneratorConfig
from aiperf.config.v1.converter import convert_user_to_aiperf


def _service() -> ServiceConfig:
    return ServiceConfig()


def _user(**loadgen_kwargs) -> UserConfig:
    return UserConfig(
        loadgen=LoadGeneratorConfig(**loadgen_kwargs),
        endpoint=EndpointConfig(streaming=True, model_names=["test-model"]),
    )


def test_grid_recipe_populates_sweep_variables_through_converter():
    user = _user(search_recipe="concurrency-ramp", degradation_threshold=0.25)
    aiperf = convert_user_to_aiperf(user, _service())
    # Grid recipes drive a sweep, so AIPerfConfig.sweep is populated.
    assert aiperf.sweep is not None
    assert aiperf.sweep.type == "grid"
    assert "phases.profiling.concurrency" in aiperf.sweep.variables
    values = aiperf.sweep.variables["phases.profiling.concurrency"]
    assert values[0] == 1
    assert values[-1] == 1000


def test_grid_recipe_threads_post_process_through_multi_run():
    user = _user(search_recipe="concurrency-ramp", degradation_threshold=0.25)
    aiperf = convert_user_to_aiperf(user, _service())
    assert aiperf.multi_run.post_process is not None
    assert aiperf.multi_run.post_process.handler == "degradation_knee_detect"
    assert aiperf.multi_run.post_process.params["threshold_pct"] == 0.25


def test_grid_recipe_does_not_set_adaptive_search():
    user = _user(search_recipe="concurrency-ramp")
    aiperf = convert_user_to_aiperf(user, _service())
    assert aiperf.multi_run.adaptive_search is None


def test_prefill_ttft_curve_through_converter():
    user = _user(search_recipe="prefill-ttft-curve", isl_min=512, isl_max=4096)
    aiperf = convert_user_to_aiperf(user, _service())
    isl_path = "datasets.main.prompts.isl"
    assert aiperf.sweep is not None
    assert isl_path in aiperf.sweep.variables
    assert aiperf.sweep.variables[isl_path][0] == 512
    assert aiperf.sweep.variables[isl_path][-1] == 4096
    assert aiperf.multi_run.post_process.handler == "ttft_curve_fit"
    assert aiperf.multi_run.post_process.output_filename == "prefill_curve.json"


def test_grid_recipe_with_magic_list_concurrency_raises():
    user = _user(search_recipe="concurrency-ramp", concurrency=[10, 20, 30])
    with pytest.raises(TypeError, match="mutually exclusive with magic-list"):
        convert_user_to_aiperf(user, _service())


def test_grid_recipe_dataset_path_targets_converter_actual_dataset_name():
    """Recipes that sweep ISL/OSL must target the dataset the v1->v2 converter
    actually creates. The converter materializes a single dataset named "main"
    from CLI input; recipes hardcode that same name (`_V1_DEFAULT_DATASET_NAME`).

    Locks in the contract: if the converter ever renames the default dataset
    or supports multiple datasets from CLI input, this test fires first.
    """
    user = _user(search_recipe="prefill-ttft-curve", isl_min=512, isl_max=4096)
    aiperf = convert_user_to_aiperf(user, _service())
    # Exactly one dataset, named "main".
    assert len(aiperf.datasets) == 1
    assert aiperf.datasets[0].name == "main"
    # And the sweep variable's dotted path resolves to that same dataset.
    expected_path = "datasets.main.prompts.isl"
    assert aiperf.sweep is not None
    assert expected_path in aiperf.sweep.variables
