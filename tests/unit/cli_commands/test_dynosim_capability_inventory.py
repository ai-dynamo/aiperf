# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drift gate from Dynamo's canonical sources to AIPerf's support manifest."""

from __future__ import annotations

import ast
import json
import os
import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

from aiperf.cli_commands.dynosim import SweepOperation, _capability_manifest

_AIPERF_ROOT = Path(__file__).resolve().parents[3]


def _dynamo_root() -> Path:
    configured = os.environ.get("DYNAMO_SOURCE_ROOT")
    candidate = (
        Path(configured).expanduser()
        if configured
        else _AIPERF_ROOT.parents[2] / "dynamo-aiperf-native"
    )
    if not candidate.joinpath("lib/mocker/Cargo.toml").is_file():
        pytest.skip(
            "DynoSim source audit requires the sibling dynamo-aiperf-native checkout"
        )
    return candidate


def _argparse_arguments(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    arguments: list[str] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            continue
        arguments.extend(
            argument.value
            for argument in node.args
            if isinstance(argument, ast.Constant)
            and isinstance(argument.value, str)
        )
    return arguments


def _argparse_choices(path: Path, argument_name: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and any(
                isinstance(argument, ast.Constant)
                and argument.value == argument_name
                for argument in node.args
            )
        ):
            continue
        for keyword in node.keywords:
            if keyword.arg == "choices":
                return list(ast.literal_eval(keyword.value))
    raise AssertionError(f"argument {argument_name!r} has no literal choices")


def _rust_struct_fields(path: Path, name: str) -> list[str]:
    source = path.read_text(encoding="utf-8")
    match = re.search(rf"struct {name} \{{(.*?)\n\}}", source, re.DOTALL)
    assert match is not None, f"missing Rust struct {name}"
    return re.findall(
        r"^\s+(?:pub\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*:",
        match.group(1),
        re.MULTILINE,
    )


def _python_model_fields(path: Path, classes: set[str]) -> dict[str, list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fields: dict[str, list[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name not in classes:
            continue
        fields[node.name] = [
            statement.target.id
            for statement in node.body
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id != "model_config"
        ]
    return fields


def _replay_exports(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    match = re.search(
        r"pub use entrypoints::\{(.*?)\n\};", source, re.DOTALL
    )
    assert match is not None
    return [
        name.strip()
        for name in match.group(1).replace("\n", " ").split(",")
        if name.strip()
    ]


def _trait_methods(path: Path, name: str) -> list[str]:
    source = path.read_text(encoding="utf-8")
    match = re.search(rf"pub trait {name} \{{(.*?)\n\}}", source, re.DOTALL)
    assert match is not None
    return re.findall(r"fn\s+(\w+)\s*\(", match.group(1))


def _items(manifest: dict[str, Any], requirement: str) -> Any:
    return manifest["requirements"][requirement]["items"]


def test_manifest_is_internally_total_and_duplicate_free() -> None:
    manifest = _capability_manifest()
    surfaces = manifest["surfaces"]
    assert manifest["schema_version"] == 1
    assert surfaces
    for name, requirement in manifest["requirements"].items():
        assert requirement["surface"], f"{name} lacks a support surface"
        items = requirement["items"]
        assert items, f"{name} is empty"
        if isinstance(items, list):
            assert len(items) == len(set(items)), f"{name} contains duplicates"


def test_canonical_cli_and_config_schemas_have_no_unmapped_capability() -> None:
    root = _dynamo_root()
    manifest = _capability_manifest()

    assert _argparse_arguments(
        root / "components/src/dynamo/replay/main.py"
    ) == _items(manifest, "replay_cli_arguments")
    assert _argparse_arguments(
        root / "components/src/dynamo/mocker/args.py"
    ) == _items(manifest, "mocker_cli_arguments")
    assert _rust_struct_fields(
        root / "lib/mocker/src/common/protocols.rs", "MockEngineArgsSerde"
    ) == _items(manifest, "engine_profile_fields")
    assert _rust_struct_fields(
        root / "lib/kv-router/src/scheduling/config.rs", "KvRouterConfigSerde"
    ) == _items(manifest, "router_profile_fields")


def test_canonical_workload_and_search_domains_have_no_unmapped_value() -> None:
    root = _dynamo_root()
    manifest = _capability_manifest()
    replay_main = root / "components/src/dynamo/replay/main.py"
    mocker_args = root / "components/src/dynamo/mocker/args.py"

    assert _argparse_choices(replay_main, "--trace-format") == _items(
        manifest, "trace_formats"
    )
    assert _argparse_choices(replay_main, "--replay-mode") == _items(
        manifest, "execution_modes"
    )
    assert _argparse_choices(replay_main, "--router-mode") == _items(
        manifest, "router_modes"
    )
    assert _argparse_choices(mocker_args, "--engine-type") == _items(
        manifest, "engine_types"
    )
    assert [operation.value for operation in SweepOperation] == _items(
        manifest, "sweep_operations"
    )

    expected_fields = _items(manifest, "sweep_spec_fields")
    assert _python_model_fields(
        root / "components/src/dynamo/profiler/utils/replay_optimize/specs.py",
        set(expected_fields),
    ) == expected_fields


def test_native_replay_contract_and_all_feature_flags_are_forwarded() -> None:
    root = _dynamo_root()
    manifest = _capability_manifest()

    assert _replay_exports(root / "lib/mocker/src/replay/mod.rs") == _items(
        manifest, "replay_rust_entrypoints"
    )
    assert _trait_methods(
        root / "lib/mocker/src/loadgen/steppable.rs", "SteppableReplay"
    ) == _items(manifest, "steppable_methods")

    mocker_toml = tomllib.loads(
        (root / "lib/mocker/Cargo.toml").read_text(encoding="utf-8")
    )
    mocker_features = set(mocker_toml["features"]) - {"default"}
    mappings = _items(manifest, "mocker_cargo_features")
    assert mocker_features == set(mappings)

    aiperf_toml = tomllib.loads(
        (_AIPERF_ROOT / "crates/aiperf/Cargo.toml").read_text(encoding="utf-8")
    )
    aiperf_features = aiperf_toml["features"]
    dependency = aiperf_toml["dependencies"]["dynamo-mocker"]
    assert dependency["optional"] is True
    assert dependency["default-features"] is False
    for mocker_feature, aiperf_feature in mappings.items():
        assert aiperf_feature in aiperf_features
        assert f"dynamo-mocker/{mocker_feature}" in aiperf_features[aiperf_feature]
        assert aiperf_feature in aiperf_features["dynamo-full"]

    cargo_config = tomllib.loads(
        (_AIPERF_ROOT / ".cargo/config.toml").read_text(encoding="utf-8")
    )
    rustflags = cargo_config["build"]["rustflags"]
    assert rustflags == ["--cfg", "tokio_unstable"]


def test_manifest_file_is_strict_json() -> None:
    path = _AIPERF_ROOT / "src/aiperf/dynosim_capabilities.json"
    decoded = json.loads(path.read_text(encoding="utf-8"))
    assert decoded == _capability_manifest()
