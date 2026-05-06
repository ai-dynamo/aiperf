# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the config module."""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from aiperf.dataset.agentic_code_gen import config
from aiperf.dataset.agentic_code_gen.config import (
    build_config_schema,
    list_bundled_configs,
    load_config,
)


class TestLoadConfig:
    def test_load_bundled_default(self) -> None:
        config = load_config("default")
        assert config.max_prompt_tokens == 167_000
        assert config.inter_turn_delay.agentic_delay.mean == 2_500
        assert config.inter_turn_delay.human_delay.mean == 40_000
        assert config.inter_turn_delay.agentic_delay.max != 1
        assert config.inter_turn_delay.human_delay.max != 1

    def test_list_bundled_configs(self) -> None:
        names = list_bundled_configs()
        assert "default" in names
        assert "spec" not in names
        assert names == sorted(names)

    def test_spec_json_matches_generated_schema(self) -> None:
        path = Path(config.__file__).parent / "configs" / "spec.json"
        schema = orjson.loads(path.read_bytes())
        assert schema == build_config_schema()
        assert "unknown fields are accepted" in schema["description"]

    def test_spec_json_is_not_loadable_config(self) -> None:
        with pytest.raises(ValueError, match="JSON Schema reference"):
            load_config("spec")

    def test_spec_json_copy_is_not_loadable_config(self, tmp_path: Path) -> None:
        path = tmp_path / "copied-spec.json"
        path.write_bytes(orjson.dumps(build_config_schema()))

        with pytest.raises(ValueError, match="JSON Schema reference"):
            load_config(str(path))

    def test_load_from_file_path(self, tmp_path: Path) -> None:
        data = {
            "new_tokens_per_turn": {"mean": 1000, "median": 500},
            "generation_length": {"mean": 300, "median": 200},
            "block_size": 512,
            "inter_turn_delay": {
                "agentic_fraction": 0.5,
                "agentic_delay": {"mean": 2000, "median": 1500},
                "human_delay": {"mean": 30000, "median": 20000},
            },
            "cache": {
                "layer1_tokens": 3000,
                "layer1_5_tokens": 2000,
                "layer2": {"mean": 5000, "median": 4000},
            },
        }
        path = tmp_path / "custom.json"
        path.write_bytes(orjson.dumps(data))

        config = load_config(str(path))
        assert config.cache.layer2.mean == 5000
        assert config.cache.layer2.mu is not None
        assert config.cache.layer2.sigma is not None

    def test_ignores_deprecated_system_prompt_tokens(self, tmp_path: Path) -> None:
        data = {
            "system_prompt_tokens": 5000,
            "new_tokens_per_turn": {"mean": 1000, "median": 500},
            "generation_length": {"mean": 300, "median": 200},
            "block_size": 512,
            "cache": {
                "layer1_tokens": 3000,
                "layer1_5_tokens": 2000,
                "layer2": {"mean": 5000, "median": 4000},
            },
        }
        path = tmp_path / "deprecated.json"
        path.write_bytes(orjson.dumps(data))

        config = load_config(str(path))
        assert "system_prompt_tokens" not in config.model_dump()

    def test_load_manifest_as_config(self, tmp_path: Path) -> None:
        """manifest.json wraps config under generation_params."""
        inner = {
            "new_tokens_per_turn": {"mean": 1234, "median": 640},
            "generation_length": {"mean": 600, "median": 350},
            "block_size": 512,
            "inter_turn_delay": {
                "agentic_fraction": 0.7,
                "agentic_delay": {"mean": 3000, "median": 2000},
                "human_delay": {"mean": 45000, "median": 30000},
            },
            "cache": {
                "layer1_tokens": 4096,
                "layer1_5_tokens": 1536,
                "layer2": {"mean": 6144, "median": 2048},
            },
        }
        manifest = {
            "seed": 42,
            "num_sessions": 100,
            "config_name": "custom",
            "generation_params": inner,
        }
        path = tmp_path / "manifest.json"
        path.write_bytes(orjson.dumps(manifest))

        config = load_config(str(path))
        assert config.generation_length.mean == 600

    def test_load_config_non_object_json_raises_value_error(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "array.json"
        path.write_bytes(orjson.dumps([{"not": "a config"}]))

        with pytest.raises(ValueError, match="must be a JSON object"):
            load_config(str(path))

    def test_unknown_path_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_config("nonexistent-config")
