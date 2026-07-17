# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Weights & Biases CLI conversion and config validation.

1. Secondary wandb flags (--wandb-entity / --wandb-run-name / --wandb-tag)
   require --wandb-project to be set.
2. --wandb-project / --wandb-entity empty-string rejection.
3. Whitespace normalization on project / entity / run_name.
4. YAML+CLI overlay: secondary flags alone are valid when the YAML base
   already enables wandb (``base_enabled`` / resolver path).
5. WandbConfig model-level normalization and secondary-requires-project rule.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.config import WandbConfig
from aiperf.config.flags._converter_telemetry import build_wandb
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.resolver import resolve_config


def _make_cli(**overrides: Any) -> CLIConfig:
    base = {
        "url": "http://localhost:8000/test",
        "model_names": ["test-model"],
    }
    base.update(overrides)
    return CLIConfig(**base)


class TestSecondaryFlagsRequireProject:
    def test_no_flags_returns_empty(self) -> None:
        assert build_wandb(_make_cli()) == {}

    @pytest.mark.parametrize(
        "field,value",
        [
            param("wandb_entity", "my-team", id="entity"),
            param("wandb_run_name", "my-run", id="run_name"),
            param("wandb_tags", ["a"], id="tags"),
        ],
    )  # fmt: skip
    def test_secondary_alone_raises(self, field: str, value: object) -> None:
        cli = _make_cli(**{field: value})
        with pytest.raises(ValueError, match="require"):
            build_wandb(cli)


class TestEmptyStringRejection:
    @pytest.mark.parametrize(
        "field,match",
        [
            param("wandb_project", "--wandb-project cannot be empty", id="project"),
            param("wandb_entity", "--wandb-entity cannot be empty", id="entity"),
        ],
    )  # fmt: skip
    def test_blank_value_raises(self, field: str, match: str) -> None:
        overrides = {"wandb_project": "proj", field: "  "}
        with pytest.raises(ValueError, match=match):
            build_wandb(_make_cli(**overrides))


class TestSuccessPaths:
    def test_project_only(self) -> None:
        assert build_wandb(_make_cli(wandb_project="proj")) == {"project": "proj"}

    def test_full_flags_normalize_whitespace(self) -> None:
        cli = _make_cli(
            wandb_project=" proj ",
            wandb_entity=" team ",
            wandb_run_name="  ",
            wandb_tags=["a", "b"],
        )
        assert build_wandb(cli) == {
            "project": "proj",
            "entity": "team",
            "run_name": None,
            "tags": ["a", "b"],
        }


_YAML_WANDB_BASE = textwrap.dedent("""\
benchmark:
  models:
    - test-model
  endpoint:
    urls:
      - http://localhost:8000/v1/chat/completions
  datasets:
    - name: default
      type: synthetic
      entries: 100
      prompts:
        isl: 128
        osl: 64
  phases:
    - name: profiling
      type: concurrency
      requests: 10
      concurrency: 1
  wandb:
    project: yaml-project
    entity: yaml-team
""")


class TestBaseEnabledOverlay:
    def test_secondary_alone_with_base_enabled_emits_partial(self) -> None:
        cli = _make_cli(wandb_run_name="rerun", wandb_tags=["t"])
        assert build_wandb(cli, base_enabled=True) == {
            "run_name": "rerun",
            "tags": ["t"],
        }

    def test_no_flags_with_base_enabled_returns_empty(self) -> None:
        assert build_wandb(_make_cli(), base_enabled=True) == {}

    def test_resolver_overlays_secondary_flags_on_yaml_project(
        self, tmp_path: Path
    ) -> None:
        """Regression: ``-f base.yaml --wandb-run-name rerun`` must succeed
        when the YAML already sets ``benchmark.wandb.project``."""
        cfg_file = tmp_path / "base.yaml"
        cfg_file.write_text(_YAML_WANDB_BASE)
        user = CLIConfig(wandb_run_name="rerun", wandb_tags=["overlay-tag"])

        config = resolve_config(user, cfg_file)

        assert config.benchmark.wandb.project == "yaml-project"
        assert config.benchmark.wandb.entity == "yaml-team"
        assert config.benchmark.wandb.run_name == "rerun"
        assert config.benchmark.wandb.tags == ["overlay-tag"]

    def test_resolver_still_rejects_secondary_without_yaml_project(
        self, tmp_path: Path
    ) -> None:
        cfg_file = tmp_path / "base.yaml"
        cfg_file.write_text(_YAML_WANDB_BASE[: _YAML_WANDB_BASE.index("  wandb:")])
        user = CLIConfig(wandb_run_name="rerun")

        with pytest.raises(ValueError, match="require"):
            resolve_config(user, cfg_file)


class TestWandbConfigModel:
    def test_whitespace_project_normalizes_to_disabled(self) -> None:
        cfg = WandbConfig(project="   ")
        assert cfg.project is None
        assert cfg.enabled is False

    def test_project_whitespace_is_stripped(self) -> None:
        cfg = WandbConfig(project=" proj ", entity=" team ", run_name=" run ")
        assert cfg.project == "proj"
        assert cfg.entity == "team"
        assert cfg.run_name == "run"

    @pytest.mark.parametrize(
        "field,value",
        [
            param("entity", "team", id="entity"),
            param("run_name", "run", id="run_name"),
            param("tags", ["a"], id="tags"),
        ],
    )  # fmt: skip
    def test_secondary_without_project_raises(self, field: str, value: object) -> None:
        with pytest.raises(ValidationError, match=r"wandb\.project"):
            WandbConfig(**{field: value})
