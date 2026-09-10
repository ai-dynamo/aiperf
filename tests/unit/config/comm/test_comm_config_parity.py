# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mechanical invariants pinning comm-config resolution to comm-config building.

Two distinct divergence classes are guarded here, both of which have already
bitten this codebase once:

1. ``CommConfigResolver`` re-implementing the user-facing-model to ZMQ-config
   mapping and drifting from ``build_comm_config`` (the resolved config dropped
   ``control_tcp_port`` and every proxy port).
2. A new port added to a user-facing input model but never plumbed through
   ``build_comm_config`` into the ZMQ config (the credit-return PUSH/PULL
   fan-in port, exposed on dual-bind only).

Both are silent: the run starts, and only the pods that connect to the
mis-defaulted port stall.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel
from pytest import param

from aiperf.config import BenchmarkConfig
from aiperf.config.comm import (
    DualBindCommunicationConfig,
    TcpCommunicationConfig,
    build_comm_config,
)
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.config.resolution.resolvers import CommConfigResolver

_COMM_CASES = [
    param(None, id="none"),
    param({"type": "ipc", "path": "/tmp/aiperf-parity"}, id="ipc"),
    param({"type": "ipc"}, id="ipc-defaults"),
    param({"type": "tcp"}, id="tcp-defaults"),
    param(
        {
            "type": "tcp",
            "host": "10.0.0.5",
            "records_port": 15557,
            "credit_router_port": 15564,
            "control_port": 15667,
            "credit_return_push_pull_port": 15669,
            "event_bus_proxy": {"frontend_port": 15550, "backend_port": 15551},
            "dataset_manager_proxy": {"frontend_port": 15552, "backend_port": 15553},
            "raw_inference_proxy": {"frontend_port": 15554, "backend_port": 15555},
        },
        id="tcp-explicit",
    ),
    param({"type": "dual"}, id="dual-defaults"),
    param(
        {
            "type": "dual",
            "tcp_host": "0.0.0.0",
            "controller_host": "controller.svc",
            "ipc_path": "/tmp/aiperf-parity-dual",
            "records_port": 25557,
            "credit_router_port": 25564,
            "control_port": 25667,
            "credit_return_push_pull_port": 25669,
            "event_bus_proxy": {"frontend_port": 25550, "backend_port": 25551},
            "dataset_manager_proxy": {"frontend_port": 25552, "backend_port": 25553},
            "raw_inference_proxy": {"frontend_port": 25554, "backend_port": 25555},
        },
        id="dual-explicit",
    ),
]


def _make_config(comm: dict[str, Any] | None) -> BenchmarkConfig:
    return BenchmarkConfig(
        models=["test-model"],
        endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
        datasets=[{"name": "profiling", "type": "synthetic"}],
        phases=[
            {
                "name": "profiling",
                "type": "concurrency",
                "concurrency": 1,
                "requests": 1,
            }
        ],
        runtime={"communication": comm} if comm is not None else {},
    )


def _normalize(dumped: dict[str, Any], strip_paths: bool) -> dict[str, Any]:
    """Blank out auto-generated IPC path values so two builds compare equal."""
    if not strip_paths:
        return dumped
    return {
        key: (
            "<auto>"
            if key.endswith("path")
            else _normalize(value, strip_paths)
            if isinstance(value, dict)
            else value
        )
        for key, value in dumped.items()
    }


@pytest.mark.parametrize("comm", _COMM_CASES)  # fmt: skip
def test_resolver_output_equals_builder_output(comm: dict[str, Any] | None) -> None:
    """CommConfigResolver must delegate to build_comm_config, not re-map by hand.

    Any hand-rolled mapping reintroduced in the resolver diverges here.
    """
    cfg = _make_config(comm)
    run = BenchmarkRun(
        benchmark_id="parity-run",
        cfg=cfg,
        artifact_dir=Path("/tmp/test-artifacts"),
    )

    CommConfigResolver().resolve(run)

    built = build_comm_config(cfg)
    resolved = run.resolved.comm_config
    assert type(resolved) is type(built)
    # An IPC config with no user-supplied path mints a fresh temp dir per
    # construction, so those paths can never compare equal across two builds.
    strip_paths = comm is None or "path" not in comm
    assert _normalize(resolved.model_dump(), strip_paths) == _normalize(
        built.model_dump(), strip_paths
    )


def _port_field_names(model: type[BaseModel], prefix: str = "") -> list[str]:
    """Recursively collect dotted paths of every ``*_port`` field on ``model``."""
    names: list[str] = []
    for name, field in model.model_fields.items():
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            names.extend(_port_field_names(annotation, f"{prefix}{name}."))
        elif name.endswith("_port"):
            names.append(f"{prefix}{name}")
    return names


def _set_path(target: dict[str, Any], path: str, value: Any) -> None:
    head, _, rest = path.partition(".")
    if rest:
        _set_path(target.setdefault(head, {}), rest, value)
    else:
        target[head] = value


def _all_ints(model: BaseModel) -> set[int]:
    """Every int reachable from ``model``, including nested proxy configs."""
    found: set[int] = set()
    for name in type(model).model_fields:
        value = getattr(model, name)
        if isinstance(value, BaseModel):
            found |= _all_ints(value)
        elif isinstance(value, bool):
            continue
        elif isinstance(value, int):
            found.add(value)
    return found


@pytest.mark.parametrize(
    "comm_type,input_model",
    [
        param("tcp", TcpCommunicationConfig, id="tcp"),
        param("dual", DualBindCommunicationConfig, id="dual"),
    ],
)  # fmt: skip
def test_every_user_facing_port_reaches_the_zmq_config(
    comm_type: str, input_model: type[BaseModel]
) -> None:
    """Every ``*_port`` on a user-facing comm model must be plumbed through.

    Mechanical guard: each port gets a unique sentinel value, and every sentinel
    must appear somewhere in the built ZMQ config. A newly added input port that
    ``build_comm_config`` forgets to pass through fails here by name.
    """
    paths = _port_field_names(input_model)
    assert paths, f"no *_port fields discovered on {input_model.__name__}"

    comm: dict[str, Any] = {"type": comm_type}
    sentinels = {path: 30000 + idx for idx, path in enumerate(paths)}
    for path, value in sentinels.items():
        _set_path(comm, path, value)

    built = build_comm_config(_make_config(comm))
    reachable = _all_ints(built)

    missing = sorted(
        path for path, value in sentinels.items() if value not in reachable
    )
    assert not missing, (
        f"{input_model.__name__} port(s) not plumbed through build_comm_config "
        f"into {type(built).__name__}: {missing}"
    )
