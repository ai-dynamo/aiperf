// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in Docker daemon smoke coverage for recorded-agent tool execution.

use std::path::PathBuf;

use aiperf_runtime::clock::RealClock;
use aiperf_runtime::graph::driver::TraceIdentity;
use aiperf_runtime::graph::replay::ReplayRunIdentity;
use aiperf_runtime::graph::tools::{
    DockerSessionSandbox, EnvironmentRecipe, ResolvedTraceEnvironment, ToolSandbox, WorkspaceSpec,
};
use aiperf_runtime::rng::RngRoot;

#[tokio::test(flavor = "current_thread")]
#[ignore = "requires a running Docker daemon and aiperf-recorded-agent-pinchbench:v1"]
async fn docker_executes_a_pinch_recipe_with_network_disabled() {
    // This smoke test intentionally requires operator-provisioned Docker and
    // canonical image state. Run with:
    // cargo test -p aiperf-e2e-tests --test test_recorded_agent_docker -- --ignored
    let workspace = tempfile::tempdir().expect("temporary mounted workspace");
    let sandbox = DockerSessionSandbox::with_docker_cli(
        ResolvedTraceEnvironment {
            kind: EnvironmentRecipe::PinchBench,
            image: "aiperf-recorded-agent-pinchbench:v1".into(),
            workspace: WorkspaceSpec {
                files: Vec::new(),
                workdir: "/workspace".into(),
                interpreter: vec!["bash".into(), "-lc".into()],
                mount_workspace: true,
                command_timeout_ns: 30_000_000_000,
            },
        },
        Some(PathBuf::from(workspace.path())),
        TraceIdentity {
            run_id: "e2e-run".into(),
            trajectory_id: "e2e-trajectory".into(),
            trace_id: "docker-smoke".into(),
        },
        ReplayRunIdentity::mint(RngRoot::new(Some(8)), "e2e-docker-run"),
        RealClock::new(),
        4096,
    )
    .expect("Docker recipe is well formed");

    sandbox
        .open()
        .await
        .expect("canonical image starts in Docker");
    let result = sandbox
        .run("printf docker-sandbox", None)
        .await
        .expect("Docker exec returns a framed result");
    sandbox.close().await.expect("Docker container is removed");

    assert_eq!(result.output.as_ref(), b"docker-sandbox");
}
