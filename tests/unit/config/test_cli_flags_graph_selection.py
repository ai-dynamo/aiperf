# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``--max-context-length`` / ``--allow-dataset-wrap`` parse onto the CLI DTO and land on the resolved graph-plane config."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from pytest import param

from aiperf.config.dataset import FileDataset, SynthesisConfig
from aiperf.config.flags._converter_dataset import build_dataset
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.converter import convert_cli_to_aiperf
from aiperf.config.resolution.plan import ResolvedConfig


def _file_user(input_file: Path, **overrides: object) -> CLIConfig:
    """File-dataset CLIConfig with both selection knobs explicitly set."""
    kwargs: dict = {
        "model_names": ["test-model"],
        "input_file": str(input_file),
        "max_context_length": 131072,
        "allow_dataset_wrap": True,
    }
    kwargs.update(overrides)
    return CLIConfig(**kwargs)


@pytest.mark.parametrize(
    ("extra_argv", "expected_wrap", "expected_max"),
    [
        param(
            ["--max-context-length", "131072", "--allow-dataset-wrap"],
            True,
            131072,
            id="both-flags-set-wrap-true",
        ),
        param(
            ["--no-allow-dataset-wrap"], False, None, id="negated-flag-explicit-false"
        ),
    ],
)  # fmt: skip
def test_cyclopts_records_explicit_flag_values(
    parse_cli_args: Callable[[list[str]], CLIConfig],
    endpoint_cli_args: list[str],
    extra_argv: list[str],
    expected_wrap: bool,
    expected_max: int | None,
) -> None:
    """Explicitly passed selection flags land on the DTO and register in ``model_fields_set``."""
    uc = parse_cli_args([*endpoint_cli_args, *extra_argv])
    assert uc.allow_dataset_wrap is expected_wrap
    assert "allow_dataset_wrap" in uc.model_fields_set
    assert uc.max_context_length == expected_max
    assert ("max_context_length" in uc.model_fields_set) is (expected_max is not None)


def test_cli_defaults_are_none(
    parse_cli_args: Callable[[list[str]], CLIConfig], endpoint_cli_args: list[str]
) -> None:
    """Unset flags default to None and stay out of ``model_fields_set`` so the resolver can tell unset from explicit."""
    uc = parse_cli_args(endpoint_cli_args)
    assert uc.max_context_length is None
    assert uc.allow_dataset_wrap is None
    assert "max_context_length" not in uc.model_fields_set
    assert "allow_dataset_wrap" not in uc.model_fields_set


@pytest.mark.parametrize(
    "value",
    [param(None, id="none"), param(True, id="true"), param(False, id="false")],
)  # fmt: skip
def test_cli_allow_dataset_wrap_tristate(value: bool | None) -> None:
    """``CLIConfig.allow_dataset_wrap`` accepts None / True / False unchanged."""
    uc = CLIConfig(allow_dataset_wrap=value)
    assert uc.allow_dataset_wrap is value


def test_synthesis_config_defaults_none() -> None:
    """Both selection fields default to None on ``SynthesisConfig``."""
    cfg = SynthesisConfig()
    assert cfg.max_context_length is None
    assert cfg.allow_dataset_wrap is None


def test_synthesis_config_accepts_values() -> None:
    """``SynthesisConfig`` carries explicit selection values verbatim."""
    cfg = SynthesisConfig(max_context_length=131072, allow_dataset_wrap=True)
    assert cfg.max_context_length == 131072
    assert cfg.allow_dataset_wrap is True


def test_synthesis_config_rejects_non_positive_max_context_length() -> None:
    """A zero ``max_context_length`` is rejected."""
    with pytest.raises(ValueError):
        SynthesisConfig(max_context_length=0)


def test_resolved_config_allow_dataset_wrap_default_none() -> None:
    """The resolved field stays None until ``GraphDispatchResolver`` derives it."""
    assert ResolvedConfig().allow_dataset_wrap is None


def test_values_land_on_dataset_synthesis_dict(
    write_plain_trace_file: Callable[..., Path],
) -> None:
    """The converter routes both flags into the FileDataset ``synthesis`` sub-dict."""
    ds = build_dataset(_file_user(write_plain_trace_file()))
    assert ds["synthesis"]["max_context_length"] == 131072
    assert ds["synthesis"]["allow_dataset_wrap"] is True


def test_values_land_on_full_aiperf_config(
    write_plain_trace_file: Callable[..., Path],
) -> None:
    """The full CLI-to-AIPerfConfig path preserves both values on the FileDataset."""
    cfg = convert_cli_to_aiperf(_file_user(write_plain_trace_file()))
    main = cfg.benchmark.get_default_dataset()
    assert isinstance(main, FileDataset)
    assert main.synthesis is not None
    assert main.synthesis.max_context_length == 131072
    assert main.synthesis.allow_dataset_wrap is True


def test_explicit_false_wrap_is_carried_through(
    write_plain_trace_file: Callable[..., Path],
) -> None:
    """``allow_dataset_wrap=False`` is explicit, not unset, so it is carried through."""
    ds = build_dataset(_file_user(write_plain_trace_file(), allow_dataset_wrap=False))
    assert ds["synthesis"]["allow_dataset_wrap"] is False


def test_unset_wrap_omits_synthesis_key(
    write_plain_trace_file: Callable[..., Path],
) -> None:
    """Unset flags are not carried, so a downstream resolver sees None."""
    ds = build_dataset(
        CLIConfig(model_names=["test-model"], input_file=str(write_plain_trace_file()))
    )
    synthesis = ds.get("synthesis") or {}
    assert "max_context_length" not in synthesis
    assert "allow_dataset_wrap" not in synthesis


def test_graph_tool_image_description_mentions_pinchbench_exception() -> None:
    """The local-default help names PinchBench's recorded-image exception."""
    description = CLIConfig.model_fields["graph_tool_image"].description
    assert description is not None
    assert "PinchBench" in description
