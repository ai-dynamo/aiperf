// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Redacted structured Docker command contracts for Harbor execution.

use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    fs,
    io::Read,
};

use aiperf_runtime::clock::SimClock;
use aiperf_runtime::eval::{
    DockerBuildRequest, DockerCopyRequest, DockerCreateRequest, DockerExecRequest,
    DockerProcessSandbox, DockerRemoveRequest, DockerRuntime, DockerStartRequest, EnvName,
    EvalExecutionError, HarborImporter, HarborSandboxRecipe, HarborSource, NativeSourceAcquirer,
    ProviderCapabilities, SecretProvider, SecretValue, preflight_docker,
};
use std::rc::Rc;

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

    fn remove(&self, _: &DockerRemoveRequest) -> Result<(), EvalExecutionError> {
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

    assert!(matches!(
        DockerProcessSandbox::new()
            .execute_with_runtime(
                &runtime,
                &recipe,
                &imported.package,
                imported.package.execution_plan(),
                &["agent".to_owned()],
                &FixedSecret,
            )
            .expect_err("failed verifier start must be terminal"),
        EvalExecutionError::ProcessFailure(_)
    ));
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
        Ok(())
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
