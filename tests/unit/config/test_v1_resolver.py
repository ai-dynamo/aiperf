# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial regression tests for ``aiperf.config.v1._resolver``.

These probe the YAML+CLI merge surface for shapes that broke during the
end-to-end recipe trial: deep_merge collisions, build_v1_overrides emit
gating, the endpoint url/urls collision, the artifacts cli_command-only
edge case, and the resolve_config code path. Each test names the bug it
locks in -- the merge path is small but its blast radius is every CLI
command, so silent regressions are expensive.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.config.v1 import ServiceConfig, UserConfig
from aiperf.config.v1._resolver import (
    build_v1_overrides,
    deep_merge,
    resolve_config,
)

# =====================================================================
# deep_merge adversarial
# =====================================================================


class TestDeepMerge:
    def test_empty_override_returns_base_unchanged(self) -> None:
        base = {"a": 1, "b": {"c": 2}}
        out = deep_merge(base, {})
        assert out == base
        # Result is a deep copy: mutating it does not touch base.
        out["b"]["c"] = 99
        assert base["b"]["c"] == 2

    def test_empty_base_returns_override_value(self) -> None:
        assert deep_merge({}, {"a": 1}) == {"a": 1}

    def test_nested_dicts_merge_recursively(self) -> None:
        base = {"endpoint": {"urls": ["http://x"], "type": "chat"}}
        ovr = {"endpoint": {"streaming": True}}
        out = deep_merge(base, ovr)
        assert out == {
            "endpoint": {"urls": ["http://x"], "type": "chat", "streaming": True}
        }

    def test_list_in_override_replaces_not_concatenates(self) -> None:
        base = {"endpoint": {"urls": ["http://yaml-one", "http://yaml-two"]}}
        ovr = {"endpoint": {"urls": ["http://cli-only"]}}
        out = deep_merge(base, ovr)
        # CLI list cleanly clobbers YAML list -- no append.
        assert out["endpoint"]["urls"] == ["http://cli-only"]

    def test_scalar_override_replaces_dict_in_base(self) -> None:
        # Type-mismatch isn't valid v2 input, but deep_merge must not crash:
        # the override scalar wins. (AIPerfConfig validation later catches it.)
        out = deep_merge({"x": {"nested": True}}, {"x": "scalar"})
        assert out == {"x": "scalar"}

    def test_dict_override_replaces_scalar_in_base(self) -> None:
        out = deep_merge({"x": "scalar"}, {"x": {"nested": True}})
        assert out == {"x": {"nested": True}}

    def test_three_level_nested_merge(self) -> None:
        base = {"a": {"b": {"c": 1, "d": 2}}}
        ovr = {"a": {"b": {"d": 99}}}
        out = deep_merge(base, ovr)
        assert out == {"a": {"b": {"c": 1, "d": 99}}}


# =====================================================================
# build_v1_overrides adversarial
# =====================================================================


class TestBuildV1Overrides:
    def test_default_user_config_emits_empty(self) -> None:
        """Pristine UserConfig (no CLI flags) must produce an empty override
        dict so callers short-circuit deep_merge and the YAML wins
        unchanged."""
        out = build_v1_overrides(UserConfig())
        assert out == {}

    def test_only_streaming_does_not_clobber_models(self) -> None:
        """Setting only --streaming must NOT include endpoint.urls or models;
        otherwise a YAML+CLI merge would zero out the YAML's models block.
        Locks in the model_fields_set gating that build_endpoint relies on.
        """
        user = UserConfig.model_validate({"endpoint": {"streaming": True}})
        out = build_v1_overrides(user)
        assert out == {"endpoint": {"streaming": True}}

    def test_only_model_promotes_to_v2_models_block(self) -> None:
        """v1 endpoint.model_names lives on EndpointConfig but maps to v2
        `models.items[*].name`. Verify the path translation."""
        user = UserConfig.model_validate({"endpoint": {"model_names": ["m1", "m2"]}})
        out = build_v1_overrides(user)
        assert out == {
            "models": {"items": [{"name": "m1"}, {"name": "m2"}]},
        }

    def test_recipe_emits_multi_run_adaptive_search(self) -> None:
        """A BO recipe with the required SLA flag produces the
        adaptive_search block (consumed by MultiRunConfig.model_validate)."""
        user = UserConfig.model_validate(
            {
                "endpoint": {"streaming": True},
                "loadgen": {
                    "search_recipe": "max-throughput-ttft-sla",
                    "ttft_sla_ms": 200.0,
                },
            }
        )
        out = build_v1_overrides(user)
        assert out["multi_run"]["adaptive_search"]["recipe_name"] == (
            "max-throughput-ttft-sla"
        )
        sla = out["multi_run"]["adaptive_search"]["sla_filters"]
        assert sla[0]["metric_tag"] == "time_to_first_token"
        assert sla[0]["threshold"] == 200.0

    def test_grid_recipe_emits_sweep_block(self) -> None:
        """A grid recipe lifts sweep_variables to a top-level sweep block."""
        user = UserConfig.model_validate(
            {
                "endpoint": {"streaming": True},
                "loadgen": {
                    "search_recipe": "prefill-ttft-curve",
                    "isl_min": 256,
                    "isl_max": 1024,
                },
            }
        )
        out = build_v1_overrides(user)
        assert out["sweep"]["type"] == "grid"
        assert "datasets.main.prompts.isl" in out["sweep"]["variables"]

    def test_service_config_ui_lands_on_runtime_block(self) -> None:
        """Service-level CLI flags go through build_logging_runtime."""
        user = UserConfig()
        service = ServiceConfig.model_validate({"ui_type": "none"})
        out = build_v1_overrides(user, service)
        assert out["runtime"]["ui"] == "none"

    def test_service_none_skips_runtime_logging_block(self) -> None:
        """Without ServiceConfig, runtime/logging stays untouched (otherwise
        we'd emit empty blocks that clobber the YAML's defaults)."""
        out = build_v1_overrides(UserConfig(), service=None)
        assert "runtime" not in out
        assert "logging" not in out

    def test_artifacts_skipped_when_only_cli_command_would_land(self) -> None:
        """build_artifacts always synthesizes cli_command from sys.argv. If
        the user passed no --output flag, we MUST omit the artifacts block
        entirely so a YAML `artifacts.dir` survives the merge."""
        out = build_v1_overrides(UserConfig())
        assert "artifacts" not in out

    def test_artifact_dir_flag_lands_on_artifacts_block(self, tmp_path: Path) -> None:
        """Locking in the --artifact-dir / output-flag propagation that
        broke `aiperf profile -f base.yaml --artifact-dir my/path` until
        the resolver started reading user.output.model_fields_set."""
        user = UserConfig.model_validate(
            {"output": {"artifact_directory": str(tmp_path / "out")}}
        )
        out = build_v1_overrides(user)
        assert "artifacts" in out
        assert str(out["artifacts"]["dir"]).endswith("out")


# =====================================================================
# Endpoint url+urls collision (the deep_merge fallout)
# =====================================================================


class TestEndpointUrlPluralization:
    def test_both_url_and_urls_drops_url_keeps_urls(self) -> None:
        """When YAML supplies singular `url` and CLI overlay supplies plural
        `urls`, the EndpointConfig before-validator must drop `url` so the
        config validates -- otherwise both reach Pydantic and trip
        `endpoint.url: Extra inputs are not permitted`. CLI wins (override
        semantics)."""
        from aiperf.config.endpoint import EndpointConfig

        cfg = EndpointConfig.model_validate(
            {"url": "http://yaml", "urls": ["http://cli"], "type": "chat"}
        )
        assert all("cli" in str(u) for u in cfg.urls)
        assert all("yaml" not in str(u) for u in cfg.urls)

    def test_only_url_promotes_to_urls(self) -> None:
        """Backwards-compat: lone singular shorthand still promotes."""
        from aiperf.config.endpoint import EndpointConfig

        cfg = EndpointConfig.model_validate({"url": "http://only", "type": "chat"})
        assert [str(u) for u in cfg.urls] == ["http://only"]

    def test_url_as_list_promoted_intact(self) -> None:
        """When `url` is itself a list (some tests pass that), don't double-
        wrap; the prior behavior is preserved."""
        from aiperf.config.endpoint import EndpointConfig

        cfg = EndpointConfig.model_validate(
            {"url": ["http://a", "http://b"], "type": "chat"}
        )
        assert len(cfg.urls) == 2


# =====================================================================
# resolve_config — full path
# =====================================================================


class TestResolveConfig:
    def _yaml(self, tmp_path: Path) -> Path:
        p = tmp_path / "base.yaml"
        p.write_text(
            """
model: m
endpoint:
  url: http://yaml
  type: chat
dataset:
  type: synthetic
  prompts: {isl: 64, osl: 32}
phases:
  type: concurrency
  concurrency: 4
  requests: 30
"""
        )
        return p

    def test_no_config_file_uses_cli_only_path(self) -> None:
        """Without --config, resolve_config must not import load_config_dict
        and instead drives the v1 -> v2 converter directly with whatever the
        UserConfig carries. Sparse UserConfig will fail v2 validation; we
        catch a ValidationError-equivalent rather than the wrong code path.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            resolve_config(UserConfig(), ServiceConfig(), config_file=None)

    def test_config_file_only_overlay_is_pure_yaml(self, tmp_path: Path) -> None:
        """Pristine UserConfig + a YAML file: no overrides applied, the YAML
        passes through to AIPerfConfig.model_validate verbatim."""
        cfg = resolve_config(UserConfig(), ServiceConfig(), self._yaml(tmp_path))
        assert any("yaml" in str(u) for u in cfg.endpoint.urls)

    def test_config_file_plus_cli_url_override(self, tmp_path: Path) -> None:
        """CLI --url wins over YAML's url. This is the original
        `--config + --flag` bug -- if the merge ever reverts to hard-override,
        this assertion flips."""
        user = UserConfig.model_validate({"endpoint": {"urls": ["http://cli-wins"]}})
        cfg = resolve_config(user, ServiceConfig(), self._yaml(tmp_path))
        assert any("cli-wins" in str(u) for u in cfg.endpoint.urls)

    def test_missing_config_file_raises_configuration_error(self) -> None:
        """Resolver delegates to load_config_dict, which surfaces a
        ConfigurationError for missing files. Don't swallow it."""
        from aiperf.config.loader.errors import ConfigurationError

        missing_path = Path("/tmp/aiperf-resolver-does-not-exist.yaml")
        with pytest.raises(ConfigurationError, match="not found"):
            resolve_config(UserConfig(), ServiceConfig(), missing_path)
