# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage for `--variant` CLI sweep emission."""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.converter import convert_cli_to_aiperf
from aiperf.config.flags.variant_parser import build_alias_table, parse_variant
from aiperf.config.loader import build_benchmark_plan
from aiperf.config.sweep import ScenarioSweep

# --- parser ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected_name", "expected_pairs"),
    [
        param(
            "isl=128, osl=64",
            None,
            {"isl": 128, "osl": 64},
            id="anonymous-int-pairs",
        ),
        param(
            "chat: isl=4096, osl=128, concurrency=128",
            "chat",
            {"isl": 4096, "osl": 128, "concurrency": 128},
            id="named-int-pairs",
        ),
        param(
            "rate=0.5, smoothing=1.5",
            None,
            {"rate": 0.5, "smoothing": 1.5},
            id="float-coercion",
        ),
        param(
            "kind='chat', model=foo",
            None,
            {"kind": "chat", "model": "foo"},
            id="quoted-and-bare-strings",
        ),
        param(
            "name : a=1 ,  b = 2",
            "name",
            {"a": 1, "b": 2},
            id="whitespace-tolerant",
        ),
    ],
)  # fmt: skip
def test_parse_variant_happy_paths(
    raw: str, expected_name: str | None, expected_pairs: dict[str, object]
) -> None:
    name, pairs = parse_variant(raw)
    assert name == expected_name
    assert pairs == expected_pairs


@pytest.mark.parametrize(
    "raw",
    [
        param("", id="empty"),
        param("   ", id="whitespace-only"),
        param("nokeys", id="no-equals"),
        param("isl=128, broken", id="malformed-token"),
        param("isl=1, isl=2", id="duplicate-key"),
        param(":only-name", id="empty-body-after-colon"),
        param(": isl=1", id="empty-name-prefix"),
    ],
)  # fmt: skip
def test_parse_variant_rejects_malformed(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_variant(raw)


# --- alias discovery ------------------------------------------------------


def test_alias_table_contains_known_short_aliases() -> None:
    table = build_alias_table()
    # Single-word aliases that exist on the v1 tree.
    assert "isl" in table
    assert "osl" in table
    assert "concurrency" in table
    # kebab-case + snake_case pairs both register.
    assert "request-rate" in table
    assert "request_rate" in table
    assert "benchmark-duration" in table
    assert "benchmark_duration" in table


def test_alias_table_paths_resolve_to_cli_config_subtree() -> None:
    table = build_alias_table()
    # After the de-nest, isl/osl resolve to flat ``prompt_*`` attrs on CLIConfig
    # (no nested holders). The path is just the top-level Python attr name.
    assert table["isl"] == "prompt_input_tokens_mean"
    assert table["osl"] == "prompt_output_tokens_mean"
    assert table["concurrency"] == "concurrency"
    # Each path should be non-empty and traversable by name.
    for path in table.values():
        assert path
        assert all(seg for seg in path.split("."))


# --- end-to-end converter -------------------------------------------------


def _user(*, variants: list[str], **sw_overrides: object) -> CLIConfig:
    return CLIConfig(
        model_names=["test-model"],
        streaming=True,
        **CLIConfig(
            concurrency=1,
            request_count=10,
        ).model_dump(exclude_unset=True),
        sweep_variants=variants,
        **sw_overrides,
    )


def test_two_variants_emit_scenario_sweep() -> None:
    user = _user(variants=["isl=128, osl=64", "isl=2048, osl=256"])
    config = convert_cli_to_aiperf(user)

    assert isinstance(config.sweep, ScenarioSweep)
    assert len(config.sweep.runs) == 2
    assert config.sweep.runs[0]["name"] == "v0"
    assert config.sweep.runs[1]["name"] == "v1"

    bench = config.sweep.runs[0]["benchmark"]
    main = next(d for d in bench["datasets"] if d["name"] == "main")
    assert main["prompts"]["isl"]["mean"] == 128
    assert main["prompts"]["osl"]["mean"] == 64


def test_named_variant_carries_label_through() -> None:
    user = _user(variants=["chat: isl=128, osl=64", "long: isl=4096, osl=512"])
    config = convert_cli_to_aiperf(user)

    assert isinstance(config.sweep, ScenarioSweep)
    labels = [r["name"] for r in config.sweep.runs]
    assert labels == ["chat", "long"]


def test_variants_round_trip_through_benchmark_plan() -> None:
    user = _user(
        variants=[
            "isl=128, osl=64",
            "isl=2048, osl=256, concurrency=64",
        ]
    )
    config = convert_cli_to_aiperf(user)
    plan = build_benchmark_plan(config)

    assert len(plan.configs) == 2
    assert [v.label for v in plan.variations] == ["v0", "v1"]

    bench0 = plan.configs[0]
    bench1 = plan.configs[1]
    assert bench0.datasets[0].prompts.isl.mean == 128
    assert bench1.datasets[0].prompts.isl.mean == 2048
    assert bench0.datasets[0].prompts.osl.mean == 64
    assert bench1.datasets[0].prompts.osl.mean == 256

    profiling0 = next(p for p in bench0.phases if p.name == "profiling")
    profiling1 = next(p for p in bench1.phases if p.name == "profiling")
    assert profiling0.concurrency == 1
    assert profiling1.concurrency == 64


# --- conflict rejection ---------------------------------------------------


def test_single_variant_rejected() -> None:
    user = _user(variants=["isl=128, osl=64"])
    with pytest.raises(TypeError, match="single occurrence is rejected"):
        convert_cli_to_aiperf(user)


def test_variant_plus_search_recipe_rejected() -> None:
    user = CLIConfig(
        model_names=["m"],
        streaming=True,
        **CLIConfig(
            concurrency=1,
            request_count=10,
        ).model_dump(exclude_unset=True),
        search_recipe="max-throughput-ttft-sla",
        ttft_sla_ms=200.0,
        sweep_variants=["isl=128, osl=64", "isl=2048, osl=256"],
    )
    with pytest.raises(TypeError, match="mutually exclusive with --search-recipe"):
        convert_cli_to_aiperf(user)


def test_variant_plus_magic_list_rejected() -> None:
    user = CLIConfig(
        model_names=["m"],
        streaming=True,
        **CLIConfig(
            concurrency=[1, 2, 4],
            request_count=10,
        ).model_dump(exclude_unset=True),
        sweep_variants=["isl=128, osl=64", "isl=2048, osl=256"],
    )
    with pytest.raises(TypeError, match="mutually exclusive with magic-list flags"):
        convert_cli_to_aiperf(user)


def test_variant_plus_yaml_sweep_rejected() -> None:
    user = CLIConfig(
        model_names=["m"],
        streaming=True,
        prompt_input_tokens_mean=100,
        prompt_output_tokens_mean=200,
        **CLIConfig(
            request_count=10,
        ).model_dump(exclude_unset=True),
        search_space=["phases.profiling.concurrency:1,1000:int"],
        search_metric="output_token_throughput",
        search_direction="maximize",
        search_max_iterations=10,
        sweep_variants=["isl=128, osl=64", "isl=2048, osl=256"],
    )
    # --search-space populates the top-level sweep block before --variant runs;
    # the variant emitter must reject the combo with the YAML/explicit-sweep
    # error message.
    with pytest.raises(
        TypeError,
        match="mutually exclusive with a YAML-declared sweep block",
    ):
        convert_cli_to_aiperf(user)


def test_unknown_variant_key_rejected() -> None:
    user = _user(variants=["bogus_key=42, isl=128", "isl=2048, osl=256"])
    with pytest.raises(TypeError, match="unknown key"):
        convert_cli_to_aiperf(user)
