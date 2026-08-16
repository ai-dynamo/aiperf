# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The production tool-execution wiring: flag -> strategy -> dispatcher -> report.

The gap this closes: every other test built a ``TraceExecutor`` by hand and
handed it a dispatcher, so nothing checked that the ONE production
``TraceExecutor(...)`` ever gets one. Without it every trace with a tool step
errors out, and the tool-time measurement the mode exists for is never reported.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.graph.models import (
    GraphRecord,
    LlmNode,
    ParsedGraph,
    ToolNode,
    ToolSandboxSpec,
    TraceRecord,
)
from aiperf.graph.executor import TraceResult
from aiperf.graph.sandbox.docker import DockerSessionSandbox
from aiperf.graph.sandbox.local import LocalSessionSandbox
from aiperf.plugin.enums import TimingMode
from aiperf.timing.config import CreditPhaseConfig, TimingConfig
from aiperf.timing.strategies.agent_graph_replay import (
    GRAPH_TOOL_WORKSPACE_DIRNAME,
    AgentGraphReplayStrategy,
)
from tests.unit.conftest import make_run_from_cli
from tests.unit.dataset.graph.conftest import DYNAMO_NESTED_FIXTURE


class _Issuer:
    """Minimal stand-in; the strategy only stores it at construction here."""

    async def issue_graph_credit(self, *args, **kwargs) -> None:
        raise AssertionError("no credit should be issued by these tests")


def _parsed(*, with_tools: bool) -> ParsedGraph:
    nodes: dict[str, LlmNode | ToolNode] = {
        "n0": LlmNode(prompt=["hi"], output="n0_out")
    }
    if with_tools:
        nodes["t0"] = ToolNode(commands=["true"], output="t0_out")
    return ParsedGraph(
        graph=GraphRecord(nodes=nodes, edges=[], state={}),
        traces=[TraceRecord(id="t-1")],
    )


def _strategy(
    *,
    with_tools: bool,
    artifact_dir: Path | None = None,
    graph_tool_image: str | None = None,
    graph_tool_persistent_session: bool = False,
) -> AgentGraphReplayStrategy:
    return AgentGraphReplayStrategy(
        config=CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.AGENT_GRAPH,
            artifact_dir=artifact_dir,
        ),
        credit_issuer=_Issuer(),
        parsed_graph=_parsed(with_tools=with_tools),
        register_observer=lambda obs: None,
        register_first_token_observer=lambda obs: None,
        unregister_observer=lambda obs: None,
        unregister_first_token_observer=lambda obs: None,
        graph_tool_image=graph_tool_image,
        graph_tool_persistent_session=graph_tool_persistent_session,
    )


def test_graph_without_tool_nodes_builds_no_dispatcher(tmp_path: Path) -> None:
    """A plain replay must not construct a sandbox, or every run would pay for one."""
    strategy = _strategy(with_tools=False, artifact_dir=tmp_path)
    assert strategy._build_tool_dispatcher("t-1::abc", TraceRecord(id="t-1")) is None


def test_unset_image_selects_the_local_backend(tmp_path: Path) -> None:
    strategy = _strategy(with_tools=True, artifact_dir=tmp_path)

    dispatcher = strategy._build_tool_dispatcher("t-1::abc", TraceRecord(id="t-1"))

    assert dispatcher is not None
    sandbox = dispatcher._sandbox_factory("t-1")
    assert isinstance(sandbox, LocalSessionSandbox)
    assert not isinstance(sandbox, DockerSessionSandbox)


@pytest.mark.parametrize("image", ["", "   "])  # fmt: skip
def test_blank_image_is_treated_as_unset(tmp_path: Path, image: str) -> None:
    """An empty --graph-tool-image is 'no image', not a docker run of image ''."""
    strategy = _strategy(with_tools=True, artifact_dir=tmp_path, graph_tool_image=image)

    sandbox = strategy._build_tool_dispatcher(
        "t-1::abc", TraceRecord(id="t-1")
    )._sandbox_factory("t-1")

    assert not isinstance(sandbox, DockerSessionSandbox)


def test_non_empty_image_selects_docker_with_that_image(tmp_path: Path) -> None:
    strategy = _strategy(
        with_tools=True, artifact_dir=tmp_path, graph_tool_image="task:latest"
    )

    sandbox = strategy._build_tool_dispatcher(
        "t-1::abc", TraceRecord(id="t-1")
    )._sandbox_factory("t-1")

    assert isinstance(sandbox, DockerSessionSandbox)
    assert "task:latest" in sandbox.start_argv()
    # Networking stays off: a recorded trajectory's commands were captured
    # against a prepared workspace, not against the live internet.
    argv = sandbox.start_argv()
    assert argv[argv.index("--network") + 1] == "none"


def test_workspace_is_per_instance_under_the_artifact_dir(tmp_path: Path) -> None:
    """Two concurrent replays of the same template must not share a workspace."""
    strategy = _strategy(with_tools=True, artifact_dir=tmp_path)

    a = strategy._build_tool_dispatcher(
        "t-1::aaa", TraceRecord(id="t-1")
    )._sandbox_factory("t-1")
    b = strategy._build_tool_dispatcher(
        "t-1::bbb", TraceRecord(id="t-1")
    )._sandbox_factory("t-1")

    root = tmp_path / GRAPH_TOOL_WORKSPACE_DIRNAME
    assert a._workspace != b._workspace
    assert a._workspace.parent == root
    # `::` and any separator in a trace id must not escape into a path segment.
    assert a._workspace.name == "t-1-aaa"


def test_each_instance_gets_its_own_dispatcher(tmp_path: Path) -> None:
    """One dispatcher holds one sandbox and refuses a second trace, so lanes need one each."""
    strategy = _strategy(with_tools=True, artifact_dir=tmp_path)

    assert strategy._build_tool_dispatcher(
        "t-1::aaa", TraceRecord(id="t-1")
    ) is not strategy._build_tool_dispatcher("t-1::bbb", TraceRecord(id="t-1"))


def test_per_trace_container_overrides_global_image(tmp_path: Path) -> None:
    """tool_sandbox.container on the trace wins over --graph-tool-image."""
    strategy = _strategy(
        with_tools=True, artifact_dir=tmp_path, graph_tool_image="global:latest"
    )
    trace = TraceRecord(
        id="t-1", tool_sandbox=ToolSandboxSpec(container="per-trace:v1")
    )

    sandbox = strategy._build_tool_dispatcher("t-1::abc", trace)._sandbox_factory("t-1")

    assert isinstance(sandbox, DockerSessionSandbox)
    assert "per-trace:v1" in sandbox.start_argv()


def test_per_trace_container_none_falls_back_to_global(tmp_path: Path) -> None:
    """tool_sandbox with no container falls back to the run-level --graph-tool-image."""
    strategy = _strategy(
        with_tools=True, artifact_dir=tmp_path, graph_tool_image="global:latest"
    )
    trace = TraceRecord(id="t-1", tool_sandbox=ToolSandboxSpec())

    sandbox = strategy._build_tool_dispatcher("t-1::abc", trace)._sandbox_factory("t-1")

    assert isinstance(sandbox, DockerSessionSandbox)
    assert "global:latest" in sandbox.start_argv()


def test_docker_sandbox_defaults_to_fresh_exec_mode(tmp_path: Path) -> None:
    """Default (persistent_session=False) must produce fresh-exec-per-command sandboxes."""
    strategy = _strategy(
        with_tools=True, artifact_dir=tmp_path, graph_tool_image="task:latest"
    )
    sandbox = strategy._build_tool_dispatcher(
        "t-1::abc", TraceRecord(id="t-1")
    )._sandbox_factory("t-1")

    assert isinstance(sandbox, DockerSessionSandbox)
    assert not sandbox._persistent_session


def test_persistent_session_flag_is_forwarded_to_sandbox(tmp_path: Path) -> None:
    """--graph-tool-persistent-session must reach the sandbox layer."""
    strategy = _strategy(
        with_tools=True,
        artifact_dir=tmp_path,
        graph_tool_image="task:latest",
        graph_tool_persistent_session=True,
    )
    sandbox = strategy._build_tool_dispatcher(
        "t-1::abc", TraceRecord(id="t-1")
    )._sandbox_factory("t-1")

    assert isinstance(sandbox, DockerSessionSandbox)
    assert sandbox._persistent_session


def test_tool_durations_are_accumulated_and_reported(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The headline number must leave the run; a discarded TraceResult reports nothing."""
    strategy = _strategy(with_tools=True, artifact_dir=tmp_path)

    strategy._record_trace_timing(
        TraceResult(trace_id="t-1", channels={}, tool_durations_s=[0.5, 1.5])
    )
    strategy._record_trace_timing(
        TraceResult(trace_id="t-2", channels={}, tool_durations_s=[2.0])
    )

    assert strategy._tool_durations_s == [0.5, 1.5, 2.0]
    assert strategy._tool_traces == 2
    with caplog.at_level("INFO"):
        strategy.report_tool_execution()
    assert "3 commands" in caplog.text
    assert "4.000s total" in caplog.text
    assert "backend=local" in caplog.text


def test_report_is_silent_when_no_tool_ran(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    strategy = _strategy(with_tools=False, artifact_dir=tmp_path)
    with caplog.at_level("INFO"):
        strategy.report_tool_execution()
    assert "tool execution" not in caplog.text


def test_graph_tool_image_survives_the_cli_to_phase_config_chain() -> None:
    """The flag is worthless if it stops at CLIConfig: the strategy reads CreditPhaseConfig."""
    base = dict(
        model_names=["test-model"],
        input_file=str(DYNAMO_NESTED_FIXTURE),
        tokenizer_name="builtin",
    )
    run = make_run_from_cli(
        CLIConfig(
            **base,
            graph_execute_tools=True,
            graph_tool_image="task:latest",
            open_loop_replay=False,
        )
    )

    configs = TimingConfig.from_run(run).phase_configs

    assert configs
    assert all(c.graph_tool_image == "task:latest" for c in configs)


def test_graph_tool_image_defaults_to_none_on_the_phase_config() -> None:
    run = make_run_from_cli(
        CLIConfig(
            model_names=["test-model"],
            input_file=str(DYNAMO_NESTED_FIXTURE),
            tokenizer_name="builtin",
        )
    )

    configs = TimingConfig.from_run(run).phase_configs

    assert all(c.graph_tool_image is None for c in configs)
