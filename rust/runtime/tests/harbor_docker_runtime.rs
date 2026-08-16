// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Redacted structured Docker command contracts for Harbor execution.

use std::{
    cell::{Cell, RefCell},
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{self, Read},
    process::Command,
    sync::{Mutex, MutexGuard},
};

use aiperf_runtime::clock::SimClock;
use aiperf_runtime::eval::{
    DockerBuildRequest, DockerCopyRequest, DockerCreateRequest, DockerExecRequest,
    DockerProcessSandbox, DockerRemoveRequest, DockerRuntime, DockerStartRequest, EnvName,
    EvalExecutionError, HarborImporter, HarborSandboxRecipe, HarborSource, NativeSourceAcquirer,
    ProviderCapabilities, SecretProvider, SecretValue, preflight_docker,
};
use std::rc::Rc;

static DOCKER_RUNTIME_TEST_LOCK: Mutex<()> = Mutex::new(());

#[derive(Default)]
struct RecordingRuntime {
    build_calls: Cell<usize>,
    events: RefCell<Vec<String>>,
}

impl DockerRuntime for RecordingRuntime {
    fn capabilities(&self) -> ProviderCapabilities {
        self.events.borrow_mut().push("preflight".to_owned());
        ProviderCapabilities::none()
    }

    fn build(&self, _: &DockerBuildRequest) -> Result<(), EvalExecutionError> {
        self.build_calls.set(self.build_calls.get() + 1);
        self.events.borrow_mut().push("build".to_owned());
        Ok(())
    }

    fn create(&self, _: &DockerCreateRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("create".to_owned());
        Ok(())
    }

    fn start(&self, _: &DockerStartRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("start".to_owned());
        Ok(())
    }

    fn exec(&self, _: &DockerExecRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("exec".to_owned());
        Ok(())
    }

    fn copy(&self, _: &DockerCopyRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("copy".to_owned());
        Ok(())
    }

    fn remove(&self, _: &DockerRemoveRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("remove".to_owned());
        Ok(())
    }
}

#[test]
fn planned_lifecycle_preflights_before_build_and_health_before_agent() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(
        &temporary,
        r#"
[environment]
workdir = "/task"
user = "bench"
network = "no-network"

[environment.env]
BASE = "baseline"

[environment.healthcheck]
command = ["true"]
start_period_sec = 0.05
start_interval_sec = 0.1
interval_sec = 0.2
timeout_sec = 0.3
retries = 1

[agent]
user = "agent"
network = "public"

[agent.env]
PHASE = "agent"

[verifier]
user = "verifier"
network = "no-network"

[verifier.env]
PHASE = "verifier"
"#,
    );
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/ignored-by-plan",
    )
    .unwrap();
    let runtime = LifecycleRuntime::default();

    DockerProcessSandbox::new()
        .execute_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["true".to_owned()],
            &FixedSecret,
        )
        .unwrap();

    assert_eq!(
        runtime.events.into_inner(),
        vec![
            "preflight",
            "build:none",
            "create:none",
            "start",
            "healthcheck:bench:/task:none:BASE=baseline",
            "prepare:root:/task:none",
            "agent:agent:/task:aiperf-eval-public:BASE=baseline,PHASE=agent",
            "copy-tests",
            "prepare:root:/task:none",
            "verifier:verifier:/task:none:BASE=baseline,PHASE=verifier",
            "copy-reward",
            "copy-reward",
            "remove",
        ]
    );
}

#[test]
fn cli_recipe_workdir_overrides_the_manifest_without_mutating_the_plan() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(&temporary, "[environment]\nworkdir = \"/manifest-work\"\n");
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::for_standard_task(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        Some("/cli-work".to_owned()),
    )
    .unwrap();
    let runtime = LifecycleRuntime::default();

    DockerProcessSandbox::new()
        .execute_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["true".to_owned()],
            &FixedSecret,
        )
        .unwrap();

    assert_eq!(
        imported.package.execution_plan().environment().workdir(),
        Some("/manifest-work")
    );
    assert!(
        runtime
            .events
            .into_inner()
            .iter()
            .any(|event| event == "agent:root:/cli-work:aiperf-eval-public:"),
        "the explicit CLI workdir must be applied only at runtime"
    );
}

#[test]
fn each_execution_uses_distinct_image_and_container_names() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(&temporary, "");
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();
    let runtime = LifecycleRuntime::default();

    for _ in 0..2 {
        DockerProcessSandbox::new()
            .execute_with_runtime(
                &runtime,
                &recipe,
                &imported.package,
                imported.package.execution_plan(),
                &["true".to_owned()],
                &FixedSecret,
            )
            .unwrap();
    }

    let names = runtime.names.into_inner();
    assert_eq!(names.len(), 4);
    assert_ne!(names[0], names[2]);
    assert_ne!(names[1], names[3]);
    assert!(names[0].starts_with("aiperf-eval:"));
    assert!(names[1].starts_with("aiperf-eval-"));
}

#[test]
fn unhealthy_readiness_retries_then_prevents_agent_and_cleans_up() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(
        &temporary,
        r#"
[environment]
network = "no-network"

[environment.healthcheck]
command = ["false"]
start_period_sec = 0.05
start_interval_sec = 0.1
interval_sec = 0.2
timeout_sec = 0.3
retries = 3
"#,
    );
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();
    let runtime = UnhealthyRuntime::default();
    let clock = Rc::new(SimClock::new());

    let error = DockerProcessSandbox::with_clock(clock.clone())
        .execute_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["agent-must-not-run".to_owned()],
            &FixedSecret,
        )
        .expect_err("exhausted readiness must stop the lifecycle");

    assert!(matches!(error, EvalExecutionError::Unhealthy(_)));
    assert_eq!(
        runtime.events.into_inner(),
        vec![
            "preflight",
            "build",
            "create",
            "start",
            "health:1:300",
            "health:2:300",
            "health:3:300",
            "remove",
        ]
    );
    assert_eq!(clock.now_ns(), 350_000_000);
}

#[derive(Default)]
struct UnhealthyRuntime {
    events: RefCell<Vec<String>>,
}

impl DockerRuntime for UnhealthyRuntime {
    fn capabilities(&self) -> ProviderCapabilities {
        self.events.borrow_mut().push("preflight".to_owned());
        ProviderCapabilities::none()
            .with_docker()
            .with_image_source()
            .with_healthchecks()
            .with_no_network()
            .with_public_network()
    }

    fn build(&self, _: &DockerBuildRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("build".to_owned());
        Ok(())
    }

    fn create(&self, _: &DockerCreateRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("create".to_owned());
        Ok(())
    }

    fn start(&self, _: &DockerStartRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("start".to_owned());
        Ok(())
    }

    fn exec(&self, request: &DockerExecRequest) -> Result<(), EvalExecutionError> {
        assert_eq!(request.phase().to_string(), "healthcheck");
        let count = self
            .events
            .borrow()
            .iter()
            .filter(|event| event.starts_with("health:"))
            .count()
            + 1;
        self.events.borrow_mut().push(format!(
            "health:{count}:{}",
            request.deadline().expect("health deadline").as_millis()
        ));
        Err(EvalExecutionError::ProcessFailure("not ready".to_owned()))
    }

    fn copy(&self, _: &DockerCopyRequest) -> Result<(), EvalExecutionError> {
        panic!("unhealthy environment must not copy verifier files")
    }

    fn remove(&self, request: &DockerRemoveRequest) -> Result<(), EvalExecutionError> {
        assert_eq!(
            &request.public_arguments()[..3],
            ["rm", "--force", "--volumes"]
        );
        self.events.borrow_mut().push("remove".to_owned());
        Ok(())
    }
}

#[test]
fn agent_terminal_errors_prevent_verifier_and_remove_the_container() {
    for error in [
        EvalExecutionError::ProcessFailure("agent failed".to_owned()),
        EvalExecutionError::Timeout {
            phase: aiperf_runtime::eval::EvalExecutionPhase::Agent,
            timeout: std::time::Duration::from_secs(1),
        },
        EvalExecutionError::TerminalUncertainty {
            phase: aiperf_runtime::eval::EvalExecutionPhase::Agent,
            container: "task-container".to_owned(),
            reason: "docker client lost".to_owned(),
        },
    ] {
        let temporary = tempfile::tempdir().unwrap();
        let task_root = standard_task_root(&temporary, "");
        let imported = HarborImporter::new(&NativeSourceAcquirer)
            .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
            .unwrap();
        let recipe = HarborSandboxRecipe::new(
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "/work",
        )
        .unwrap();
        let runtime = AgentTerminalRuntime::new(error.clone());

        assert_eq!(
            DockerProcessSandbox::new()
                .execute_with_runtime(
                    &runtime,
                    &recipe,
                    &imported.package,
                    imported.package.execution_plan(),
                    &["agent-must-fail".to_owned()],
                    &FixedSecret,
                )
                .expect_err("a terminal agent error must stop the lifecycle"),
            error
        );
        assert_eq!(
            runtime.events.into_inner(),
            vec!["preflight", "build", "create", "start", "agent", "remove"]
        );
    }
}

#[test]
fn artifact_collection_failure_prevents_separate_verifier_setup() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_with_artifacts(
        &temporary,
        "[\"/work/missing-result.txt\"]",
        "[verifier]\nenvironment_mode = \"separate\"\n",
    );
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();
    let runtime = ArtifactFailureRuntime::default();

    let error = DockerProcessSandbox::new()
        .execute_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .expect_err("artifact collection must terminate before verifier setup");

    assert!(matches!(error, EvalExecutionError::ArtifactCollection(_)));
    assert_eq!(
        runtime.events.into_inner(),
        vec!["build", "create", "start", "agent", "collect", "remove"]
    );
}

#[derive(Default)]
struct ArtifactFailureRuntime {
    events: RefCell<Vec<String>>,
}

impl DockerRuntime for ArtifactFailureRuntime {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::none()
            .with_docker()
            .with_image_source()
            .with_separate_verifier()
            .with_no_network()
            .with_public_network()
    }

    fn build(&self, _: &DockerBuildRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("build".to_owned());
        Ok(())
    }

    fn create(&self, _: &DockerCreateRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("create".to_owned());
        Ok(())
    }

    fn start(&self, _: &DockerStartRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("start".to_owned());
        Ok(())
    }

    fn exec(&self, request: &DockerExecRequest) -> Result<(), EvalExecutionError> {
        assert_eq!(request.phase().to_string(), "agent");
        self.events.borrow_mut().push("agent".to_owned());
        Ok(())
    }

    fn copy(&self, _: &DockerCopyRequest) -> Result<(), EvalExecutionError> {
        panic!("artifact failure must prevent verifier files and reward copies")
    }

    fn copy_archive(&self, _: &str, _: &str) -> Result<Box<dyn Read>, EvalExecutionError> {
        self.events.borrow_mut().push("collect".to_owned());
        Err(EvalExecutionError::ArtifactCollection(
            "declared source is absent".to_owned(),
        ))
    }

    fn remove(&self, _: &DockerRemoveRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("remove".to_owned());
        Ok(())
    }
}

struct AgentTerminalRuntime {
    events: RefCell<Vec<String>>,
    error: EvalExecutionError,
}

impl AgentTerminalRuntime {
    fn new(error: EvalExecutionError) -> Self {
        Self {
            events: RefCell::new(Vec::new()),
            error,
        }
    }
}

impl DockerRuntime for AgentTerminalRuntime {
    fn capabilities(&self) -> ProviderCapabilities {
        self.events.borrow_mut().push("preflight".to_owned());
        ProviderCapabilities::none()
            .with_docker()
            .with_image_source()
            .with_no_network()
            .with_public_network()
    }

    fn build(&self, _: &DockerBuildRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("build".to_owned());
        Ok(())
    }

    fn create(&self, _: &DockerCreateRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("create".to_owned());
        Ok(())
    }

    fn start(&self, _: &DockerStartRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("start".to_owned());
        Ok(())
    }

    fn exec(&self, request: &DockerExecRequest) -> Result<(), EvalExecutionError> {
        assert_eq!(request.phase().to_string(), "agent");
        self.events.borrow_mut().push("agent".to_owned());
        Err(self.error.clone())
    }

    fn copy(&self, _: &DockerCopyRequest) -> Result<(), EvalExecutionError> {
        panic!("terminal agent errors must not copy verifier files")
    }

    fn remove(&self, _: &DockerRemoveRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("remove".to_owned());
        Ok(())
    }
}

#[test]
fn separate_verifier_failure_removes_both_container_leases() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(&temporary, "[verifier]\nenvironment_mode = \"separate\"\n");
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();
    let runtime = VerifierFailureRuntime::default();

    let error = DockerProcessSandbox::new()
        .execute_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .expect_err("a failed verifier must be terminal");

    assert!(matches!(error, EvalExecutionError::ProcessFailure(_)));
    assert_eq!(
        runtime.events.into_inner(),
        vec![
            "preflight",
            "build",
            "create",
            "start",
            "agent",
            "create",
            "start",
            "copy-tests",
            "verifier",
            "remove",
            "remove",
        ]
    );
}

#[test]
fn separate_verifier_start_failure_removes_registered_verifier_lease() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(&temporary, "[verifier]\nenvironment_mode = \"separate\"\n");
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();
    let runtime = SeparateStartFailureRuntime::default();

    let error = DockerProcessSandbox::new()
        .execute_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .expect_err("failed verifier start must be terminal");
    assert_eq!(
        error,
        EvalExecutionError::ProcessFailure("verifier start failed".to_owned())
    );
    assert_eq!(
        runtime.events.into_inner(),
        vec![
            "build", "create", "start", "agent", "create", "start", "remove", "remove"
        ]
    );
}

#[derive(Default)]
struct SeparateStartFailureRuntime {
    events: RefCell<Vec<String>>,
    starts: Cell<u8>,
    removals: Cell<u8>,
}

impl DockerRuntime for SeparateStartFailureRuntime {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::none()
            .with_docker()
            .with_image_source()
            .with_separate_verifier()
            .with_no_network()
            .with_public_network()
    }
    fn build(&self, _: &DockerBuildRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("build".to_owned());
        Ok(())
    }
    fn create(&self, _: &DockerCreateRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("create".to_owned());
        Ok(())
    }
    fn start(&self, _: &DockerStartRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("start".to_owned());
        self.starts.set(self.starts.get() + 1);
        if self.starts.get() == 2 {
            Err(EvalExecutionError::ProcessFailure(
                "verifier start failed".to_owned(),
            ))
        } else {
            Ok(())
        }
    }
    fn exec(&self, request: &DockerExecRequest) -> Result<(), EvalExecutionError> {
        assert_eq!(request.phase().to_string(), "agent");
        self.events.borrow_mut().push("agent".to_owned());
        Ok(())
    }
    fn copy(&self, _: &DockerCopyRequest) -> Result<(), EvalExecutionError> {
        panic!("verifier start failure must precede file copy")
    }
    fn remove(&self, _: &DockerRemoveRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("remove".to_owned());
        self.removals.set(self.removals.get() + 1);
        if self.removals.get() == 1 {
            Err(EvalExecutionError::ProcessFailure(
                "verifier removal failed".to_owned(),
            ))
        } else {
            Ok(())
        }
    }
}

#[test]
fn separate_verifier_health_failure_prevents_verifier_files_and_cleans_leases() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(
        &temporary,
        "[verifier]\nenvironment_mode = \"separate\"\n[verifier.environment.healthcheck]\ncommand = [\"false\"]\ntimeout_sec = 1\nretries = 1\n",
    );
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();
    let runtime = SeparateHealthFailureRuntime::default();

    assert!(matches!(
        DockerProcessSandbox::new()
            .execute_with_runtime(
                &runtime,
                &recipe,
                &imported.package,
                imported.package.execution_plan(),
                &["agent".to_owned()],
                &FixedSecret
            )
            .expect_err("unhealthy separate verifier must stop before verifier files"),
        EvalExecutionError::Unhealthy(_)
    ));
    assert_eq!(
        runtime.events.into_inner(),
        vec![
            "build",
            "create",
            "start",
            "agent",
            "create",
            "start",
            "healthcheck",
            "remove",
            "remove"
        ]
    );
}

#[derive(Default)]
struct SeparateHealthFailureRuntime {
    events: RefCell<Vec<String>>,
}
impl DockerRuntime for SeparateHealthFailureRuntime {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::none()
            .with_docker()
            .with_image_source()
            .with_separate_verifier()
            .with_healthchecks()
            .with_no_network()
            .with_public_network()
    }
    fn build(&self, _: &DockerBuildRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("build".to_owned());
        Ok(())
    }
    fn create(&self, _: &DockerCreateRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("create".to_owned());
        Ok(())
    }
    fn start(&self, _: &DockerStartRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("start".to_owned());
        Ok(())
    }
    fn exec(&self, request: &DockerExecRequest) -> Result<(), EvalExecutionError> {
        let phase = request.phase().to_string();
        self.events.borrow_mut().push(phase.clone());
        if phase == "healthcheck" {
            Err(EvalExecutionError::ProcessFailure("not ready".to_owned()))
        } else {
            Ok(())
        }
    }
    fn copy(&self, _: &DockerCopyRequest) -> Result<(), EvalExecutionError> {
        panic!("unhealthy verifier must not receive verifier files")
    }
    fn remove(&self, _: &DockerRemoveRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("remove".to_owned());
        Ok(())
    }
}

#[derive(Default)]
struct VerifierFailureRuntime {
    events: RefCell<Vec<String>>,
}

impl DockerRuntime for VerifierFailureRuntime {
    fn capabilities(&self) -> ProviderCapabilities {
        self.events.borrow_mut().push("preflight".to_owned());
        ProviderCapabilities::none()
            .with_docker()
            .with_image_source()
            .with_separate_verifier()
            .with_no_network()
            .with_public_network()
    }

    fn build(&self, _: &DockerBuildRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("build".to_owned());
        Ok(())
    }

    fn create(&self, _: &DockerCreateRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("create".to_owned());
        Ok(())
    }

    fn start(&self, _: &DockerStartRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("start".to_owned());
        Ok(())
    }

    fn exec(&self, request: &DockerExecRequest) -> Result<(), EvalExecutionError> {
        match request.phase().to_string().as_str() {
            "agent" => {
                self.events.borrow_mut().push("agent".to_owned());
                Ok(())
            }
            "verifier" => {
                self.events.borrow_mut().push("verifier".to_owned());
                Err(EvalExecutionError::ProcessFailure(
                    "verifier failed".to_owned(),
                ))
            }
            phase => panic!("unexpected {phase} phase"),
        }
    }

    fn copy(&self, request: &DockerCopyRequest) -> Result<(), EvalExecutionError> {
        assert!(request.public_arguments()[1].contains("/tests"));
        self.events.borrow_mut().push("copy-tests".to_owned());
        Ok(())
    }

    fn remove(&self, _: &DockerRemoveRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("remove".to_owned());
        Ok(())
    }
}

#[derive(Default)]
struct LifecycleRuntime {
    events: RefCell<Vec<String>>,
    names: RefCell<Vec<String>>,
}

impl DockerRuntime for LifecycleRuntime {
    fn capabilities(&self) -> ProviderCapabilities {
        self.events.borrow_mut().push("preflight".to_owned());
        ProviderCapabilities::none()
            .with_docker()
            .with_image_source()
            .with_users()
            .with_phase_env()
            .with_workdir()
            .with_healthchecks()
            .with_no_network()
            .with_public_network()
    }

    fn supports_phase_network_transitions(&self) -> bool {
        true
    }

    fn build(&self, request: &DockerBuildRequest) -> Result<(), EvalExecutionError> {
        assert!(request.public_arguments().windows(2).any(|arguments| {
            arguments[0] == "--network" && Some(arguments[1].as_str()) == request.network_lease()
        }));
        self.events.borrow_mut().push(format!(
            "build:{}",
            request.network_lease().expect("build network")
        ));
        self.names.borrow_mut().push(
            request
                .public_arguments()
                .windows(2)
                .find(|arguments| arguments[0] == "--tag")
                .expect("image tag")[1]
                .clone(),
        );
        Ok(())
    }

    fn create(&self, request: &DockerCreateRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push(format!(
            "create:{}",
            request.network_lease().expect("container network")
        ));
        self.names.borrow_mut().push(
            request
                .public_arguments()
                .windows(2)
                .find(|arguments| arguments[0] == "--name")
                .expect("container name")[1]
                .clone(),
        );
        Ok(())
    }

    fn start(&self, _: &DockerStartRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("start".to_owned());
        Ok(())
    }

    fn exec(&self, request: &DockerExecRequest) -> Result<(), EvalExecutionError> {
        if request
            .public_arguments()
            .iter()
            .any(|argument| argument.contains("mkdir -p"))
        {
            self.events.borrow_mut().push(format!(
                "prepare:root:{}:{}",
                request.workdir().unwrap_or("<image-workdir>"),
                request.network_lease(),
            ));
            return Ok(());
        }
        let environment = request
            .public_environment()
            .iter()
            .map(|(name, value)| format!("{name}={value}"))
            .collect::<Vec<_>>()
            .join(",");
        self.events.borrow_mut().push(format!(
            "{}:{}:{}:{}:{}",
            request.phase(),
            request.user().unwrap_or("root"),
            request.workdir().unwrap_or("<image-workdir>"),
            request.network_lease(),
            environment,
        ));
        Ok(())
    }

    fn copy(&self, request: &DockerCopyRequest) -> Result<(), EvalExecutionError> {
        assert_eq!(
            request.public_arguments().first().map(String::as_str),
            Some("cp")
        );
        let event = if request
            .public_arguments()
            .iter()
            .any(|argument| argument.contains("/tests"))
        {
            "copy-tests"
        } else {
            "copy-reward"
        };
        self.events.borrow_mut().push(event.to_owned());
        if let Some(destination) = request.public_arguments().last() {
            if destination.ends_with("reward.txt") {
                fs::write(destination, "1\n").unwrap();
            }
        }
        Ok(())
    }

    fn remove(&self, _: &DockerRemoveRequest) -> Result<(), EvalExecutionError> {
        self.events.borrow_mut().push("remove".to_owned());
        Ok(())
    }
}

#[test]
fn unsupported_plan_is_rejected_before_a_docker_build_is_possible() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(&temporary, "[environment]\ncpus = 1\nmemory_mb = 512\n");
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let runtime = RecordingRuntime::default();

    assert_eq!(
        preflight_docker(&runtime, imported.package.execution_plan()),
        Err(EvalExecutionError::UnsupportedEnforcement("docker"))
    );
    assert_eq!(runtime.build_calls.get(), 0);
}

struct FixedSecret;

impl SecretProvider for FixedSecret {
    fn resolve(&self, _: &EnvName) -> Result<SecretValue, EvalExecutionError> {
        Ok(SecretValue::new("unrenderable-secret".to_owned()))
    }
}

#[test]
fn docker_exec_request_redacts_secret_environment_values() {
    let request = DockerExecRequest::new(
        "task-container",
        ["/bin/sh", "-c", "true"],
        BTreeMap::from([("VISIBLE".to_owned(), "value".to_owned())]),
        BTreeMap::from([("TOKEN".to_owned(), SecretValue::new("unrenderable-secret"))]),
    );

    let rendering = format!("{request:?}");
    assert!(rendering.contains("VISIBLE"));
    assert!(rendering.contains("TOKEN"));
    assert!(!rendering.contains("unrenderable-secret"));
    assert_eq!(
        format!("{}", FixedSecret.resolve(&"TOKEN".to_owned()).unwrap()),
        "[REDACTED]"
    );
}

#[test]
fn multi_step_session_keeps_one_agent_and_injects_only_the_current_instruction() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = multi_step_task_root(&temporary, false);
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();
    let runtime = StepRecordingRuntime::default();

    let result = DockerProcessSandbox::new()
        .execute_multi_step_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .unwrap();

    assert_eq!(runtime.build_calls.get(), 1);
    assert_eq!(runtime.creates.borrow().len(), 1);
    assert_eq!(runtime.starts.get(), 1);
    assert_eq!(runtime.verifier_execs.get(), 2);
    assert_eq!(result.steps.len(), 2);
    assert_eq!(
        runtime.agent_environments.into_inner(),
        vec![
            BTreeMap::from([(
                "AIPERF_EVAL_INSTRUCTION".to_owned(),
                "First instruction.\n".to_owned(),
            )]),
            BTreeMap::from([(
                "AIPERF_EVAL_INSTRUCTION".to_owned(),
                "Second instruction.\n".to_owned(),
            )]),
        ]
    );
    assert!(
        runtime.creates.into_inner()[0]
            .arguments
            .iter()
            .all(|argument| !argument.contains("AIPERF_EVAL_INSTRUCTION")),
        "an instruction captured at container creation would become stale"
    );
}

#[test]
fn shared_verifier_resets_tests_before_each_selected_tree_copy() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = multi_step_task_root(&temporary, false);
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();
    let runtime = StepRecordingRuntime::default();

    DockerProcessSandbox::new()
        .execute_multi_step_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .unwrap();

    let events = runtime.events.into_inner();
    let reset_indices = events
        .iter()
        .enumerate()
        .filter_map(|(index, event)| event.starts_with("reset-tests:").then_some(index))
        .collect::<Vec<_>>();
    let copy_indices = events
        .iter()
        .enumerate()
        .filter_map(|(index, event)| event.starts_with("copy-tests:").then_some(index))
        .collect::<Vec<_>>();
    assert_eq!(reset_indices.len(), 4);
    assert_eq!(copy_indices.len(), 2);
    assert!(reset_indices[0] < copy_indices[0]);
    assert!(reset_indices[1] > copy_indices[0]);
    assert!(reset_indices[2] < copy_indices[1]);
    assert!(reset_indices[3] > copy_indices[1]);
    let second_agent = events.iter().position(|event| event == "agent:2").unwrap();
    assert!(reset_indices[1] < second_agent);
    assert!(events[copy_indices[0]].contains("/tests/."));
    assert!(events[copy_indices[1]].contains("/steps/two/tests/."));
    assert_eq!(
        runtime.reset_users.into_inner(),
        vec![
            Some("root".to_owned()),
            Some("root".to_owned()),
            Some("root".to_owned()),
            Some("root".to_owned()),
        ]
    );
}

#[test]
fn shared_verifier_failure_clears_hidden_state_and_beats_cleanup_error() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = multi_step_task_root(&temporary, false);
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();
    let runtime = StepRecordingRuntime::failing_shared_verifier_cleanup();

    let error = DockerProcessSandbox::new()
        .execute_multi_step_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .expect_err("the verifier failure must stop before the second agent");

    assert_eq!(
        error,
        EvalExecutionError::ProcessFailure("verifier 1 failed".to_owned())
    );
    assert_eq!(runtime.agent_execs.get(), 1);
    assert_eq!(runtime.reset_calls.get(), 2);
}

#[test]
fn separate_verifiers_use_fresh_staging_and_artifact_snapshots() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = multi_step_task_root(&temporary, true);
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();
    let runtime = StepRecordingRuntime::default();

    let result = DockerProcessSandbox::new()
        .execute_multi_step_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .unwrap();

    assert_eq!(
        result.steps[0].artifacts,
        vec![(
            "result.txt".to_owned(),
            aiperf_runtime::eval::ArtifactDigest::from_bytes(b"first snapshot"),
        )]
    );
    assert_eq!(
        result.steps[1].artifacts,
        vec![(
            "result.txt".to_owned(),
            aiperf_runtime::eval::ArtifactDigest::from_bytes(b"second snapshot"),
        )]
    );
    let creates = runtime.creates.into_inner();
    assert_eq!(creates.len(), 3);
    assert!(creates[0].workspace.is_some());
    assert_eq!(creates[1].workspace, None);
    assert_eq!(creates[2].workspace, None);
    assert!(creates[1].container.contains("verifier-one"));
    assert!(creates[2].container.contains("verifier-two"));
    let transfers = runtime.artifact_transfers.into_inner();
    assert_eq!(transfers.len(), 2);
    assert_ne!(transfers[0].0, transfers[1].0);
    assert!(transfers[0].1.contains("verifier-one:/work"));
    assert!(transfers[1].1.contains("verifier-two:/work"));
}

#[test]
fn separate_verifier_stages_artifacts_without_overriding_image_workdir() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = multi_step_task_root(&temporary, true);
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::for_standard_task(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        None,
    )
    .unwrap();
    let runtime = StepRecordingRuntime::default();

    let result = DockerProcessSandbox::new()
        .execute_multi_step_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .unwrap();

    assert_eq!(result.steps.len(), 2);
    let creates = runtime.creates.into_inner();
    assert_eq!(creates.len(), 3);
    assert_eq!(creates[0].workspace, None);
    assert_eq!(creates[1].workspace, None);
    assert_eq!(creates[2].workspace, None);
    assert_eq!(runtime.inspected_workdirs.borrow().len(), 2);
    let transfers = runtime.artifact_transfers.into_inner();
    assert_eq!(transfers.len(), 2);
    assert_ne!(transfers[0].0, transfers[1].0);
    assert!(transfers[0].1.ends_with(":/image-workdir"));
    assert!(transfers[1].1.ends_with(":/image-workdir"));
    assert_eq!(runtime.verifier_workdirs.into_inner(), vec![None, None]);
}

#[test]
fn separate_verifier_rejects_reserved_image_workdir_before_artifact_transfer() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = multi_step_task_root(&temporary, true);
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::for_standard_task(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        None,
    )
    .unwrap();
    let runtime = StepRecordingRuntime::with_image_workdir("/tests");

    let error = DockerProcessSandbox::new()
        .execute_multi_step_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .expect_err("the image workdir would stage an artifact below /tests");

    assert!(matches!(
        error,
        EvalExecutionError::InvalidWorkspace(reason)
            if reason.contains("reserved verifier path")
    ));
    assert_eq!(runtime.creates.borrow().len(), 2);
    assert_eq!(runtime.starts.get(), 2);
    let events = runtime.events.into_inner();
    assert!(
        events
            .iter()
            .any(|event| event.starts_with("inspect-workdir:"))
    );
    assert!(events.iter().all(|event| {
        !event.starts_with("copy-artifacts:")
            && !event.starts_with("reset-tests:")
            && !event.starts_with("copy-tests:")
            && !event.starts_with("verifier:")
    }));
}

#[test]
fn separate_verifier_rejects_reserved_cli_workdir_before_verifier_provisioning() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = multi_step_task_root(&temporary, true);
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::for_standard_task(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        Some("/logs/verifier".to_owned()),
    )
    .unwrap();
    let runtime = StepRecordingRuntime::default();

    let error = DockerProcessSandbox::new()
        .execute_multi_step_with_runtime(
            &runtime,
            &recipe,
            &imported.package,
            imported.package.execution_plan(),
            &["agent".to_owned()],
            &FixedSecret,
        )
        .expect_err("the CLI workdir would stage an artifact below evaluator reward paths");

    assert!(matches!(
        error,
        EvalExecutionError::InvalidWorkspace(reason)
            if reason.contains("reserved verifier path")
    ));
    assert_eq!(runtime.creates.borrow().len(), 1);
    assert_eq!(runtime.starts.get(), 1);
    assert!(runtime.events.into_inner().iter().all(|event| {
        !event.starts_with("inspect-workdir:")
            && !event.starts_with("copy-artifacts:")
            && !event.starts_with("reset-tests:")
            && !event.starts_with("copy-tests:")
            && !event.starts_with("verifier:")
    }));
}

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn separate_verifier_transfer_preserves_colliding_image_workdir_contents() {
    let _docker_lock = docker_runtime_test_lock();
    let temporary = tempfile::tempdir().unwrap();
    let task_root = multi_step_task_root(&temporary, true);
    fs::write(
        task_root.join("task.toml"),
        r#"schema_version = "1.0"
multi_step_reward_strategy = "mean"
artifacts = ["/aiperf-eval-artifacts/result.txt"]

[task]
name = "example/multi-step-colliding-image-workdir"

[[steps]]
name = "one"
[steps.verifier]
environment_mode = "separate"

[[steps]]
name = "two"
[steps.verifier]
environment_mode = "separate"
"#,
    )
    .unwrap();
    fs::write(
        task_root.join("environment/Dockerfile"),
        "FROM alpine:3.20\nRUN mkdir -p /logs/verifier /aiperf-eval-artifacts && printf image-sentinel > /aiperf-eval-artifacts/image.txt\nWORKDIR /aiperf-eval-artifacts\n",
    )
    .unwrap();
    let verifier = "test \"$(cat image.txt)\" = image-sentinel\ntest \"$(cat result.txt)\" = agent-artifact\nprintf '{\"reward\":1.0}' > /logs/verifier/reward.json\n";
    fs::write(task_root.join("tests/test.sh"), verifier).unwrap();
    fs::write(task_root.join("steps/two/tests/test.sh"), verifier).unwrap();
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::for_standard_task(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        None,
    )
    .unwrap();

    let result = DockerProcessSandbox::new()
        .execute_multi_step(
            &recipe,
            &imported.package,
            &[
                "/bin/sh".to_owned(),
                "-c".to_owned(),
                "printf agent-artifact > result.txt".to_owned(),
            ],
        )
        .unwrap();

    assert_eq!(result.steps.len(), 2);
    assert!(
        result
            .steps
            .iter()
            .all(|step| step.reward.metrics.get("reward") == Some(&1.0))
    );
}

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn separate_verifier_anonymous_volumes_are_removed_after_success() {
    let _docker_lock = docker_runtime_test_lock();
    let before = docker_resource_names();
    let temporary = tempfile::tempdir().unwrap();
    let result = run_multi_step_volume_task(&temporary, false).unwrap();

    assert_eq!(result.steps.len(), 2);
    assert_eq!(docker_resource_names(), before);
}

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn separate_verifier_anonymous_volumes_are_removed_after_timeout() {
    let _docker_lock = docker_runtime_test_lock();
    let before = docker_resource_names();
    let temporary = tempfile::tempdir().unwrap();
    let error = run_multi_step_volume_task(&temporary, true)
        .expect_err("the first separate verifier must time out");

    assert!(matches!(
        error,
        EvalExecutionError::Timeout {
            phase: aiperf_runtime::eval::EvalExecutionPhase::Verifier,
            ..
        }
    ));
    assert_eq!(docker_resource_names(), before);
}

#[test]
fn multi_step_failures_stop_successors_and_cleanup_every_acquired_lease() {
    for (failure, expected_counts, expected_error, fail_first_removal) in [
        (
            StepFailure::Agent(2),
            (2, 1, 1, 2),
            EvalExecutionError::ProcessFailure("agent 2 failed".to_owned()),
            false,
        ),
        (
            StepFailure::Collection(2),
            (2, 2, 1, 2),
            EvalExecutionError::ArtifactCollection("collection 2 failed".to_owned()),
            false,
        ),
        (
            StepFailure::Verifier(1),
            (1, 1, 1, 2),
            EvalExecutionError::ProcessFailure("verifier 1 failed".to_owned()),
            false,
        ),
        (
            StepFailure::Verifier(2),
            (2, 2, 2, 3),
            EvalExecutionError::ProcessFailure("verifier 2 failed".to_owned()),
            true,
        ),
    ] {
        let temporary = tempfile::tempdir().unwrap();
        let task_root = multi_step_task_root(&temporary, true);
        let imported = HarborImporter::new(&NativeSourceAcquirer)
            .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
            .unwrap();
        let recipe = HarborSandboxRecipe::new(
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "/work",
        )
        .unwrap();
        let runtime = StepRecordingRuntime::failing(failure, fail_first_removal);

        let error = DockerProcessSandbox::new()
            .execute_multi_step_with_runtime(
                &runtime,
                &recipe,
                &imported.package,
                imported.package.execution_plan(),
                &["agent".to_owned()],
                &FixedSecret,
            )
            .expect_err("the injected phase failure must be terminal");

        assert_eq!(error, expected_error);
        assert_eq!(runtime.agent_execs.get(), expected_counts.0);
        assert_eq!(runtime.collection_calls.get(), expected_counts.1);
        assert_eq!(runtime.verifier_execs.get(), expected_counts.2);
        assert_eq!(runtime.creates.borrow().len(), expected_counts.3);
        assert_eq!(runtime.removals.get(), expected_counts.3);
        assert!(runtime.removal_arguments.borrow().iter().all(|arguments| {
            arguments.len() == 4
                && arguments[0] == "rm"
                && arguments[1] == "--force"
                && arguments[2] == "--volumes"
        }));
    }
}

#[tokio::test]
async fn step_only_timeouts_refuse_nested_synchronous_docker_execution() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = multi_step_task_root(&temporary, false);
    fs::write(
        task_root.join("task.toml"),
        r#"schema_version = "1.0"
multi_step_reward_strategy = "mean"
[task]
name = "example/multi-step-timeouts"
[[steps]]
name = "one"
[[steps]]
name = "two"
[steps.agent]
timeout_sec = 2
[steps.verifier]
timeout_sec = 2
"#,
    )
    .unwrap();
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    assert_eq!(imported.package.timeouts(), None);
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();

    assert_eq!(
        DockerProcessSandbox::new().execute_multi_step(
            &recipe,
            &imported.package,
            &["agent".to_owned()],
        ),
        Err(EvalExecutionError::RuntimeContext(
            "synchronous Docker execution"
        ))
    );
}

#[tokio::test]
async fn timed_step_healthcheck_refuses_nested_synchronous_docker_execution() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = multi_step_task_root(&temporary, false);
    fs::write(
        task_root.join("task.toml"),
        r#"schema_version = "1.0"
multi_step_reward_strategy = "mean"
[task]
name = "example/multi-step-healthcheck"
[[steps]]
name = "one"
[[steps]]
name = "two"
[steps.verifier]
environment_mode = "separate"
[steps.verifier.environment.healthcheck]
command = ["true"]
start_period_sec = 1
retries = 1
"#,
    )
    .unwrap();
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    assert!(
        imported
            .package
            .execution_plan()
            .environment()
            .healthcheck()
            .is_none()
    );
    assert!(
        imported.package.execution_plan().steps()[1]
            .verifier()
            .environment()
            .healthcheck()
            .is_some()
    );
    let recipe = HarborSandboxRecipe::for_standard_task(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        None,
    )
    .unwrap();

    assert_eq!(
        DockerProcessSandbox::new().execute_multi_step(
            &recipe,
            &imported.package,
            &["agent".to_owned()],
        ),
        Err(EvalExecutionError::RuntimeContext(
            "synchronous Docker execution"
        ))
    );
}

#[derive(Clone, Copy)]
enum StepFailure {
    Agent(usize),
    Collection(usize),
    Verifier(usize),
}

struct RecordedCreate {
    container: String,
    workspace: Option<String>,
    arguments: Vec<String>,
}

#[derive(Default)]
struct StepRecordingRuntime {
    build_calls: Cell<usize>,
    starts: Cell<usize>,
    agent_execs: Cell<usize>,
    collection_calls: Cell<usize>,
    verifier_execs: Cell<usize>,
    reset_calls: Cell<usize>,
    removals: Cell<usize>,
    events: RefCell<Vec<String>>,
    creates: RefCell<Vec<RecordedCreate>>,
    agent_environments: RefCell<Vec<BTreeMap<String, String>>>,
    reset_users: RefCell<Vec<Option<String>>>,
    inspected_workdirs: RefCell<Vec<String>>,
    artifact_transfers: RefCell<Vec<(String, String)>>,
    verifier_workdirs: RefCell<Vec<Option<String>>>,
    removal_arguments: RefCell<Vec<Vec<String>>>,
    failure: Option<StepFailure>,
    fail_reset_call: Option<usize>,
    fail_first_removal: bool,
    image_workdir: Option<String>,
}

impl StepRecordingRuntime {
    fn with_image_workdir(workdir: &str) -> Self {
        Self {
            image_workdir: Some(workdir.to_owned()),
            ..Self::default()
        }
    }

    fn failing(failure: StepFailure, fail_first_removal: bool) -> Self {
        Self {
            failure: Some(failure),
            fail_first_removal,
            ..Self::default()
        }
    }

    fn failing_shared_verifier_cleanup() -> Self {
        Self {
            failure: Some(StepFailure::Verifier(1)),
            fail_reset_call: Some(2),
            ..Self::default()
        }
    }
}

impl DockerRuntime for StepRecordingRuntime {
    fn capabilities(&self) -> ProviderCapabilities {
        self.events.borrow_mut().push("preflight".to_owned());
        ProviderCapabilities::none()
            .with_docker()
            .with_image_source()
            .with_separate_verifier()
            .with_no_network()
            .with_public_network()
    }

    fn build(&self, _: &DockerBuildRequest) -> Result<(), EvalExecutionError> {
        self.build_calls.set(self.build_calls.get() + 1);
        self.events.borrow_mut().push("build".to_owned());
        Ok(())
    }

    fn create(&self, request: &DockerCreateRequest) -> Result<(), EvalExecutionError> {
        let arguments = request.public_arguments();
        let container = argument_after(arguments, "--name").to_owned();
        let workspace = argument_after(arguments, "--volume")
            .split_once(':')
            .map(|(host, _)| host.to_owned());
        self.events.borrow_mut().push(format!("create:{container}"));
        self.creates.borrow_mut().push(RecordedCreate {
            container,
            workspace,
            arguments: arguments.to_vec(),
        });
        Ok(())
    }

    fn start(&self, request: &DockerStartRequest) -> Result<(), EvalExecutionError> {
        self.starts.set(self.starts.get() + 1);
        self.events
            .borrow_mut()
            .push(format!("start:{}", request.container()));
        Ok(())
    }

    fn exec(&self, request: &DockerExecRequest) -> Result<(), EvalExecutionError> {
        if request
            .public_arguments()
            .iter()
            .any(|argument| argument.contains("rm -rf /tests"))
        {
            let call = self.reset_calls.get() + 1;
            self.reset_calls.set(call);
            self.reset_users
                .borrow_mut()
                .push(request.user().map(str::to_owned));
            self.events
                .borrow_mut()
                .push(format!("reset-tests:{}", request.container()));
            if self.fail_reset_call == Some(call) {
                return Err(EvalExecutionError::ProcessFailure(format!(
                    "reset {call} failed"
                )));
            }
            return Ok(());
        }
        if request.public_arguments().first().map(String::as_str) == Some("mkdir") {
            assert_eq!(request.user(), Some("root"));
            assert_eq!(request.workdir(), None);
            self.events
                .borrow_mut()
                .push(format!("prepare-artifacts:{}", request.container()));
            return Ok(());
        }
        match request.phase().to_string().as_str() {
            "agent" => {
                let call = self.agent_execs.get() + 1;
                self.agent_execs.set(call);
                self.agent_environments
                    .borrow_mut()
                    .push(request.public_environment().clone());
                self.events.borrow_mut().push(format!("agent:{call}"));
                if matches!(self.failure, Some(StepFailure::Agent(failed)) if failed == call) {
                    return Err(EvalExecutionError::ProcessFailure(format!(
                        "agent {call} failed"
                    )));
                }
                Ok(())
            }
            "verifier" => {
                let call = self.verifier_execs.get() + 1;
                self.verifier_execs.set(call);
                self.verifier_workdirs
                    .borrow_mut()
                    .push(request.workdir().map(str::to_owned));
                self.events.borrow_mut().push(format!("verifier:{call}"));
                if matches!(self.failure, Some(StepFailure::Verifier(failed)) if failed == call) {
                    return Err(EvalExecutionError::ProcessFailure(format!(
                        "verifier {call} failed"
                    )));
                }
                Ok(())
            }
            phase => panic!("unexpected {phase} phase"),
        }
    }

    fn copy(&self, request: &DockerCopyRequest) -> Result<(), EvalExecutionError> {
        let arguments = request.public_arguments();
        let source = &arguments[1];
        let destination = &arguments[2];
        if destination.ends_with(":/tests") {
            self.events
                .borrow_mut()
                .push(format!("copy-tests:{source}"));
            return Ok(());
        }
        if source.ends_with("/reward.json") {
            return Err(EvalExecutionError::ProcessFailure(
                "reward.json absent".to_owned(),
            ));
        }
        if source.ends_with("/reward.txt") {
            fs::write(destination, format!("{}\n", self.verifier_execs.get())).unwrap();
            return Ok(());
        }
        if !source.contains(':') && destination.contains(':') {
            self.artifact_transfers
                .borrow_mut()
                .push((source.to_owned(), destination.to_owned()));
            self.events
                .borrow_mut()
                .push(format!("copy-artifacts:{destination}"));
            return Ok(());
        }
        panic!("unexpected Docker copy: {arguments:?}")
    }

    fn container_workdir(&self, container: &str) -> Result<String, EvalExecutionError> {
        self.inspected_workdirs
            .borrow_mut()
            .push(container.to_owned());
        self.events
            .borrow_mut()
            .push(format!("inspect-workdir:{container}"));
        Ok(self
            .image_workdir
            .clone()
            .unwrap_or_else(|| "/image-workdir".to_owned()))
    }

    fn copy_archive(&self, _: &str, _: &str) -> Result<Box<dyn Read>, EvalExecutionError> {
        let call = self.collection_calls.get() + 1;
        self.collection_calls.set(call);
        self.events.borrow_mut().push(format!("collect:{call}"));
        if matches!(self.failure, Some(StepFailure::Collection(failed)) if failed == call) {
            return Err(EvalExecutionError::ArtifactCollection(format!(
                "collection {call} failed"
            )));
        }
        let contents = if call == 1 {
            b"first snapshot".as_slice()
        } else {
            b"second snapshot".as_slice()
        };
        Ok(Box::new(io::Cursor::new(test_tar_archive(
            "result.txt",
            contents,
        ))))
    }

    fn remove(&self, request: &DockerRemoveRequest) -> Result<(), EvalExecutionError> {
        let call = self.removals.get() + 1;
        self.removals.set(call);
        self.removal_arguments
            .borrow_mut()
            .push(request.public_arguments().to_vec());
        self.events.borrow_mut().push(format!(
            "remove:{}",
            request.public_arguments().last().unwrap()
        ));
        if self.fail_first_removal && call == 1 {
            Err(EvalExecutionError::ProcessFailure(
                "first removal failed".to_owned(),
            ))
        } else {
            Ok(())
        }
    }
}

fn argument_after<'a>(arguments: &'a [String], flag: &str) -> &'a str {
    arguments
        .windows(2)
        .find(|pair| pair[0] == flag)
        .map(|pair| pair[1].as_str())
        .unwrap_or("")
}

fn test_tar_archive(path: &str, contents: &[u8]) -> Vec<u8> {
    let mut header = [0_u8; 512];
    header[..path.len()].copy_from_slice(path.as_bytes());
    header[100..108].copy_from_slice(b"0000644\0");
    header[108..116].copy_from_slice(b"0000000\0");
    header[116..124].copy_from_slice(b"0000000\0");
    let size = format!("{:011o}\0", contents.len());
    header[124..136].copy_from_slice(size.as_bytes());
    header[136..148].copy_from_slice(b"00000000000\0");
    header[148..156].fill(b' ');
    header[156] = b'0';
    header[257..263].copy_from_slice(b"ustar\0");
    header[263..265].copy_from_slice(b"00");
    let checksum = header.iter().map(|byte| u32::from(*byte)).sum::<u32>();
    header[148..156].copy_from_slice(format!("{checksum:06o}\0 ").as_bytes());
    let mut archive = header.to_vec();
    archive.extend_from_slice(contents);
    archive.resize(archive.len().next_multiple_of(512) + 1024, 0);
    archive
}

fn docker_runtime_test_lock() -> MutexGuard<'static, ()> {
    DOCKER_RUNTIME_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn docker_resource_names() -> (BTreeSet<String>, BTreeSet<String>) {
    let containers = Command::new("docker")
        .args(["container", "ls", "--all", "--format", "{{.Names}}"])
        .output()
        .expect("inspect Docker containers");
    assert!(
        containers.status.success(),
        "docker container listing failed: {}",
        String::from_utf8_lossy(&containers.stderr)
    );
    let volumes = Command::new("docker")
        .args(["volume", "ls", "--quiet"])
        .output()
        .expect("inspect Docker volumes");
    assert!(
        volumes.status.success(),
        "docker volume listing failed: {}",
        String::from_utf8_lossy(&volumes.stderr)
    );
    (
        String::from_utf8(containers.stdout)
            .unwrap()
            .lines()
            .filter(|name| name.starts_with("aiperf-eval-"))
            .map(str::to_owned)
            .collect(),
        String::from_utf8(volumes.stdout)
            .unwrap()
            .lines()
            .map(str::to_owned)
            .collect(),
    )
}

fn multi_step_volume_task_root(
    temporary: &tempfile::TempDir,
    has_verifier_timeout: bool,
) -> std::path::PathBuf {
    let task_root = multi_step_task_root(temporary, true);
    let agent_timeout = has_verifier_timeout
        .then_some("[steps.agent]\ntimeout_sec = 5\n")
        .unwrap_or("");
    let verifier_timeout = has_verifier_timeout
        .then_some("timeout_sec = 1\n")
        .unwrap_or("");
    fs::write(
        task_root.join("task.toml"),
        format!(
            r#"schema_version = "1.0"
multi_step_reward_strategy = "mean"
artifacts = ["/aiperf-eval-artifacts/result.txt"]

[task]
name = "example/multi-step-volume-workdir"

[[steps]]
name = "one"
{agent_timeout}
[steps.verifier]
environment_mode = "separate"
{verifier_timeout}
[[steps]]
name = "two"
{agent_timeout}
[steps.verifier]
environment_mode = "separate"
{verifier_timeout}"#,
        ),
    )
    .unwrap();
    fs::write(
        task_root.join("environment/Dockerfile"),
        "FROM alpine:3.20\nRUN mkdir -p /logs/verifier /aiperf-eval-artifacts && printf image-sentinel > /aiperf-eval-artifacts/image.txt\nWORKDIR /aiperf-eval-artifacts\nVOLUME /aiperf-eval-artifacts\n",
    )
    .unwrap();
    let verifier = if has_verifier_timeout {
        "sleep 2\n"
    } else {
        "test \"$(cat image.txt)\" = image-sentinel\ntest \"$(cat result.txt)\" = agent-artifact\nprintf '{\"reward\":1.0}' > /logs/verifier/reward.json\n"
    };
    fs::write(task_root.join("tests/test.sh"), verifier).unwrap();
    fs::write(task_root.join("steps/two/tests/test.sh"), verifier).unwrap();
    task_root
}

fn run_multi_step_volume_task(
    temporary: &tempfile::TempDir,
    has_verifier_timeout: bool,
) -> Result<aiperf_runtime::eval::MultiStepExecutionResult, EvalExecutionError> {
    let task_root = multi_step_volume_task_root(temporary, has_verifier_timeout);
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::for_standard_task(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        None,
    )
    .unwrap();
    DockerProcessSandbox::new().execute_multi_step(
        &recipe,
        &imported.package,
        &[
            "/bin/sh".to_owned(),
            "-c".to_owned(),
            "printf agent-artifact > result.txt".to_owned(),
        ],
    )
}

fn multi_step_task_root(
    temporary: &tempfile::TempDir,
    has_separate_verifiers: bool,
) -> std::path::PathBuf {
    let task_root = temporary.path().join("multi-step-task");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::create_dir_all(task_root.join("steps/one")).unwrap();
    fs::create_dir_all(task_root.join("steps/two/tests")).unwrap();
    let verifier = has_separate_verifiers
        .then_some("\n[steps.verifier]\nenvironment_mode = \"separate\"\n")
        .unwrap_or("");
    fs::write(
        task_root.join("task.toml"),
        format!(
            r#"schema_version = "1.0"
multi_step_reward_strategy = "mean"
artifacts = ["/work/result.txt"]

[task]
name = "example/multi-step-runtime"

[[steps]]
name = "one"
{verifier}
[[steps]]
name = "two"
{verifier}"#,
        ),
    )
    .unwrap();
    fs::write(task_root.join("instruction.md"), "Legacy instruction.\n").unwrap();
    fs::write(task_root.join("environment/Dockerfile"), "FROM scratch\n").unwrap();
    fs::write(task_root.join("tests/test.sh"), "exit 0\n").unwrap();
    fs::write(
        task_root.join("steps/one/instruction.md"),
        "First instruction.\n",
    )
    .unwrap();
    fs::write(
        task_root.join("steps/two/instruction.md"),
        "Second instruction.\n",
    )
    .unwrap();
    fs::write(task_root.join("steps/two/tests/test.sh"), "exit 0\n").unwrap();
    task_root
}

fn standard_task_root(temporary: &tempfile::TempDir, manifest_suffix: &str) -> std::path::PathBuf {
    let task_root = temporary.path().join("task");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        format!("schema_version = \"1.0\"\n\n[task]\nname = \"example/task\"\n\n{manifest_suffix}"),
    )
    .unwrap();
    fs::write(task_root.join("instruction.md"), "Do work.\n").unwrap();
    fs::write(task_root.join("environment/Dockerfile"), "FROM scratch\n").unwrap();
    fs::write(task_root.join("tests/test.sh"), "exit 0\n").unwrap();
    task_root
}

fn standard_task_with_artifacts(
    temporary: &tempfile::TempDir,
    artifacts: &str,
    manifest_suffix: &str,
) -> std::path::PathBuf {
    let task_root = standard_task_root(temporary, "");
    fs::write(
        task_root.join("task.toml"),
        format!(
            "schema_version = \"1.0\"\nartifacts = {artifacts}\n\n[task]\nname = \"example/artifacts\"\n{manifest_suffix}"
        ),
    )
    .unwrap();
    task_root
}
