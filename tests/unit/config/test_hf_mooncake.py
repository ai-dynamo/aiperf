# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HF trace source and mapping options must reach the loader and cache identity."""

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from pytest import param

from aiperf.config.dataset import PublicDataset
from aiperf.config.flags import CLIConfig
from aiperf.config.flags.resolver import resolve_config
from aiperf.config.loader.errors import ConfigurationError
from aiperf.dataset.composer.public import PublicDatasetComposer
from aiperf.dataset.loader.hf_mooncake_trace import HFMooncakeTraceDatasetLoader
from aiperf.dataset.mmap_cache import _settings_payload_from_run
from tests.unit.conftest import make_run_from_cli


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    config = tmp_path / "hf.yaml"
    config.write_text("""schemaVersion: "2.0"
benchmark:
  model: target
  endpoint: {type: chat, url: "http://localhost:8000"}
  dataset:
    type: public
    dataset: mooncake_hf
    hf_mooncake_dataset: original/repo
    hf_subset: original-subset
    mooncake_trace_to_hf_mapping: {messages: history, tools: functions}
    filters: {model: original-model}
  profiling: {type: concurrency, concurrency: 2, sessions: 3}
""")
    return config


def test_cli_overrides_hf_yaml_and_composer_arguments(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = make_run_from_cli(
        CLIConfig(
            config_file=config_file,
            hf_dataset_subset="new-subset",
            mooncake_trace_to_hf_mapping=[
                "messages=turns",
                "extra.temperature=params.temperature",
            ],
            dataset_filters=["model=new-model"],
        )
    )
    dataset = run.cfg.get_default_dataset()
    assert dataset.hf_mooncake_dataset == "original/repo"
    assert dataset.hf_subset == "new-subset"
    assert dataset.mooncake_trace_to_hf_mapping == {
        "messages": "turns",
        "tools": "functions",
        "extra.temperature": "params.temperature",
    }
    assert dataset.filters == {"model": "new-model"}
    composer = PublicDatasetComposer(run=run, tokenizer=None)
    composer.prompt_generator = Mock()
    monkeypatch.setattr(
        "aiperf.dataset.generator.corpus.resolve_prompt_generator",
        Mock(return_value=composer.prompt_generator),
    )
    kwargs = composer._build_loader_kwargs("mooncake_hf", HFMooncakeTraceDatasetLoader)
    assert kwargs["hf_dataset_name"] == "original/repo"
    assert kwargs["hf_subset"] == "new-subset"
    assert kwargs["hf_split"] == "test"
    assert "hf_revision" not in kwargs
    assert kwargs["field_mapping"] == dataset.mooncake_trace_to_hf_mapping
    assert kwargs["streaming"] is True


@pytest.mark.parametrize("flags", [
    param({"hf_mooncake_dataset": "other/repo"}, id="repo"),
    param({"hf_dataset_subset": "other-subset"}, id="subset"),
    param({"mooncake_trace_to_hf_mapping": ["messages=turns"]}, id="mapping"),
    param({"dataset_filters": ["model=other-model"]}, id="filter"),
])  # fmt: skip
def test_hf_options_change_cached_dataset_identity(flags: dict[str, Any]) -> None:
    base = {
        "model_names": ["target"],
        "public_dataset": "mooncake_hf",
        "hf_mooncake_dataset": "original/repo",
    }
    before = _settings_payload_from_run(make_run_from_cli(CLIConfig(**base)))
    after = _settings_payload_from_run(make_run_from_cli(CLIConfig(**(base | flags))))
    assert before != after


@pytest.mark.parametrize("flags,error", [
    param({"public_dataset": "mooncake_hf"}, "requires a non-empty hf_mooncake_dataset", id="missing_repo"),
    param({"public_dataset": "aimo", "hf_mooncake_dataset": "example/repo"}, "--hf-mooncake-dataset", id="wrong_plugin"),
    param({"mooncake_trace_to_hf_mapping": ["messages=turns"]}, "--mooncake-trace-to-hf-mapping", id="orphan_mapping"),
    param({"public_dataset": "aimo", "mooncake_trace_to_hf_mapping": ["messages=turns"]}, "--mooncake-trace-to-hf-mapping", id="mapping_wrong_public_loader"),
    param({"custom_dataset_type": "mooncake_trace", "mooncake_trace_to_hf_mapping": ["messages=turns"]}, "--mooncake-trace-to-hf-mapping", id="mapping_local_trace"),
    param({"public_dataset": "mooncake_hf", "hf_mooncake_dataset": "example/repo", "mooncake_trace_to_hf_mapping": ["messages=turns", "messages=other"]}, "Duplicate --mooncake-trace-to-hf-mapping", id="duplicate_mapping"),
    param({"public_dataset": "mooncake_hf", "hf_mooncake_dataset": "example/repo", "mooncake_trace_to_hf_mapping": ["messages"]}, "Invalid --mooncake-trace-to-hf-mapping", id="malformed_mapping"),
])  # fmt: skip
def test_invalid_hf_flags_fail_clearly(flags: dict[str, Any], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        make_run_from_cli(CLIConfig(model_names=["target"], **flags))


@pytest.mark.parametrize("mapping,error", [
    param({"timestamp": "time"}, "unsupported", id="unsupported_target"),
    param({"extra.": "temperature"}, "unsupported", id="empty_extra_key"),
    param({"messages": ""}, "invalid source path", id="empty_source"),
    param({"messages": "nested..messages"}, "invalid source path", id="empty_path_segment"),
])  # fmt: skip
def test_invalid_mapping_rejected_by_config(
    mapping: dict[str, str], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        PublicDataset(
            name="default",
            type="public",
            dataset="mooncake_hf",
            hf_mooncake_dataset="example/trajectories",
            mooncake_trace_to_hf_mapping=mapping,
        )


@pytest.mark.parametrize("explicit_loader", [False, True])
def test_repository_flag_selects_mooncake_loader(explicit_loader: bool) -> None:
    flags = {"public_dataset": "mooncake_hf"} if explicit_loader else {}
    run = make_run_from_cli(
        CLIConfig(
            model_names=["target"],
            hf_mooncake_dataset="  example/trajectories  ",
            mooncake_trace_to_hf_mapping=["messages=turns"],
            dataset_filters=["model=source/model"],
            **flags,
        )
    )
    dataset = run.cfg.get_default_dataset()
    assert dataset.type == "public"
    assert dataset.dataset == "mooncake_hf"
    assert dataset.hf_mooncake_dataset == "example/trajectories"
    assert dataset.mooncake_trace_to_hf_mapping == {"messages": "turns"}
    assert dataset.filters == {"model": "source/model"}


@pytest.mark.parametrize("conflict", [
    param({"hf_weka_dataset": "example/weka"}, id="weka_repo"),
    param({"custom_dataset_type": "mooncake_trace"}, id="custom_format"),
])  # fmt: skip
def test_repository_flag_rejects_conflicting_sources(conflict: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        make_run_from_cli(
            CLIConfig(
                model_names=["target"],
                hf_mooncake_dataset="example/trajectories",
                **conflict,
            )
        )


def test_repository_flag_with_config_is_rejected_like_weka(config_file: Path) -> None:
    with pytest.raises(ConfigurationError, match="--hf-mooncake-dataset"):
        resolve_config(
            CLIConfig(config_file=config_file, hf_mooncake_dataset="other/repo")
        )
