# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI-flag plumbing + resolved-config landing for the graph-plane
dataset-selection knobs ``--max-context-length`` and ``--allow-dataset-wrap``.

These are the config FIELDS the resolver and dispatch layers wire to:

- ``CLIConfig.max_context_length`` / ``CLIConfig.allow_dataset_wrap``
  (flat cyclopts flags, INPUT group).
- ``SynthesisConfig.max_context_length`` / ``SynthesisConfig.allow_dataset_wrap``
  (raw explicit values carried through ``FileDataset.synthesis``; ``None``
  when unset so ``GraphDispatchResolver`` can distinguish unset from explicit).
- ``ResolvedConfig.allow_dataset_wrap`` (``GraphDispatchResolver`` derives the
  default here; ``GraphIRReplayStrategy`` reads
  ``run.resolved.allow_dataset_wrap``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest import param

from aiperf.config.dataset import FileDataset, SynthesisConfig
from aiperf.config.flags._converter_dataset import build_dataset
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.converter import convert_cli_to_aiperf
from aiperf.config.resolution.plan import ResolvedConfig

# --- cyclopts parse harness (mirrors test_auto_plot_fields) ------------------


def _parse_cli_args(argv: list[str]) -> CLIConfig:
    """Parse ``argv`` through cyclopts into a ``CLIConfig`` (no execution)."""
    from cyclopts import App

    captured: dict[str, CLIConfig] = {}
    app = App(name="test_profile")

    @app.default
    def _runner(*, cli_config: CLIConfig) -> None:  # pragma: no cover - capture only
        captured["uc"] = cli_config

    try:
        app(argv, exit_on_error=False)
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise
    return captured["uc"]


def _required_endpoint_args() -> list[str]:
    """Minimal endpoint flags needed for any CLIConfig parse to succeed."""
    return ["--url", "http://localhost:8000/test", "--model", "test-model"]


# --- CLI flag plumbing -------------------------------------------------------


def test_cyclopts_parses_max_context_length_and_allow_dataset_wrap() -> None:
    """``--max-context-length 131072 --allow-dataset-wrap`` populates the DTO."""
    uc = _parse_cli_args(
        [
            *_required_endpoint_args(),
            "--max-context-length",
            "131072",
            "--allow-dataset-wrap",
        ]
    )
    assert uc.max_context_length == 131072
    assert uc.allow_dataset_wrap is True
    assert "max_context_length" in uc.model_fields_set
    assert "allow_dataset_wrap" in uc.model_fields_set


def test_cyclopts_parses_no_allow_dataset_wrap() -> None:
    """``--no-allow-dataset-wrap`` records an explicit False (not unset)."""
    uc = _parse_cli_args([*_required_endpoint_args(), "--no-allow-dataset-wrap"])
    assert uc.allow_dataset_wrap is False
    assert "allow_dataset_wrap" in uc.model_fields_set


def test_cli_defaults_are_none() -> None:
    """Unset flags default to None and stay out of ``model_fields_set`` so the
    resolver can distinguish unset from explicit."""
    uc = _parse_cli_args(_required_endpoint_args())
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


# --- SynthesisConfig field ---------------------------------------------------


def test_synthesis_config_defaults_none() -> None:
    """Both new SynthesisConfig fields default to None."""
    cfg = SynthesisConfig()
    assert cfg.max_context_length is None
    assert cfg.allow_dataset_wrap is None


def test_synthesis_config_accepts_values() -> None:
    cfg = SynthesisConfig(max_context_length=131072, allow_dataset_wrap=True)
    assert cfg.max_context_length == 131072
    assert cfg.allow_dataset_wrap is True


def test_synthesis_config_rejects_non_positive_max_context_length() -> None:
    with pytest.raises(ValueError):
        SynthesisConfig(max_context_length=0)


# --- ResolvedConfig field ----------------------------------------------------


def test_resolved_config_allow_dataset_wrap_default_none() -> None:
    """The resolved field defaults to None until ``GraphDispatchResolver`` derives it."""
    assert ResolvedConfig().allow_dataset_wrap is None


# --- values land on the resolved config --------------------------------------


def _trace_file(tmp_path: Path) -> str:
    """CLIConfig.input_file validates existence, so materialize a trace file."""
    path = tmp_path / "trace.jsonl"
    path.write_text('{"timestamp": 0, "input_length": 8, "output_length": 4}\n')
    return str(path)


def _file_user(tmp_path: Path, **overrides) -> CLIConfig:
    kwargs: dict = {
        "model_names": ["test-model"],
        "input_file": _trace_file(tmp_path),
        "max_context_length": 131072,
        "allow_dataset_wrap": True,
    }
    kwargs.update(overrides)
    return CLIConfig(**kwargs)


def test_values_land_on_dataset_synthesis_dict(tmp_path: Path) -> None:
    """Converter routes both flags into the FileDataset ``synthesis`` sub-dict."""
    ds = build_dataset(_file_user(tmp_path))
    assert ds["synthesis"]["max_context_length"] == 131072
    assert ds["synthesis"]["allow_dataset_wrap"] is True


def test_values_land_on_full_aiperf_config(tmp_path: Path) -> None:
    """Full CLI -> AIPerfConfig path preserves the values on the FileDataset."""
    cfg = convert_cli_to_aiperf(_file_user(tmp_path))
    main = cfg.benchmark.get_default_dataset()
    assert isinstance(main, FileDataset)
    assert main.synthesis is not None
    assert main.synthesis.max_context_length == 131072
    assert main.synthesis.allow_dataset_wrap is True


def test_explicit_false_wrap_is_carried_through(tmp_path: Path) -> None:
    """``allow_dataset_wrap=False`` is a non-None explicit value and is carried."""
    ds = build_dataset(_file_user(tmp_path, allow_dataset_wrap=False))
    assert ds["synthesis"]["allow_dataset_wrap"] is False


def test_unset_wrap_omits_synthesis_key(tmp_path: Path) -> None:
    """Unset flags are not carried, so a resolver sees None (unset)."""
    ds = build_dataset(
        CLIConfig(model_names=["test-model"], input_file=_trace_file(tmp_path))
    )
    synthesis = ds.get("synthesis") or {}
    assert "max_context_length" not in synthesis
    assert "allow_dataset_wrap" not in synthesis
