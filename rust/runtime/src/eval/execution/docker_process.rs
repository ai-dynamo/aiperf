// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Docker-backed execution for conventional native task directories.

use std::{
    cell::RefCell,
    fs,
    process::{Child, Command, Stdio},
    rc::Rc,
    time::Duration,
};

use tempfile::TempDir;

use crate::{
    clock::{Clock, RealClock},
    eval::{ArtifactDigest, HarborTaskPackage, RewardDocument, VerifierMode},
};

use super::{EvalExecutionError, EvalExecutionPhase, HarborSandboxRecipe, LocalExecutionResult};

/// Executes a conventional task in a task-built Docker environment.
pub struct DockerProcessSandbox {
    clock: Rc<dyn Clock>,
}

impl std::fmt::Debug for DockerProcessSandbox {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DockerProcessSandbox")
            .finish_non_exhaustive()
    }
}

impl Default for DockerProcessSandbox {
    fn default() -> Self {
        Self::new()
    }
}

impl DockerProcessSandbox {
    /// Creates a Docker-backed task executor.
    pub fn new() -> Self {
        Self::with_clock(RealClock::new())
    }

    /// Creates a Docker-backed task executor using the supplied execution clock.
    pub fn with_clock(clock: Rc<dyn Clock>) -> Self {
        Self { clock }
    }

    /// Builds the task environment, executes an external agent, and runs a shared verifier.
    pub fn execute(
        &self,
        recipe: &HarborSandboxRecipe,
        package: &HarborTaskPackage,
        agent_command: &[String],
        verifier_mode: VerifierMode,
    ) -> Result<LocalExecutionResult, EvalExecutionError> {
        if !package.is_standard_directory() {
            return Err(EvalExecutionError::Materialization(
                "Docker execution requires a standard task directory".to_owned(),
            ));
        }
        let source_root = package.source_root().ok_or_else(|| {
            EvalExecutionError::Materialization(
                "standard task directory was not retained after import".to_owned(),
            )
        })?;
        let environment = source_root.join("environment");
        if !environment.join("Dockerfile").is_file() {
            return Err(EvalExecutionError::Materialization(
                "standard task is missing environment/Dockerfile".to_owned(),
            ));
        }
        let workspace = tempfile::tempdir()
            .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
        let name_suffix = format!(
            "{}-{}",
            std::process::id(),
            package.source_digest().as_str()
        );
        let safe_suffix = name_suffix
            .chars()
            .filter(|character| character.is_ascii_alphanumeric())
            .take(32)
            .collect::<String>();
        let image = format!("aiperf-eval:{safe_suffix}");
        let container = format!("aiperf-eval-{safe_suffix}");
        let lease = DockerLease { container, image };
        docker(
            [
                "build",
                "--tag",
                &lease.image,
                environment.to_string_lossy().as_ref(),
            ],
            "build task environment",
        )?;
        create_container(
            &lease.container,
            &lease.image,
            workspace.path(),
            recipe,
            package,
            true,
        )?;
        docker(["start", &lease.container], "start task container")?;
        docker_exec(
            self.clock.clone(),
            &lease.container,
            agent_command,
            "run agent",
            package
                .timeouts()
                .map(|(agent_timeout, _)| (EvalExecutionPhase::Agent, agent_timeout)),
        )?;
        let artifacts = collect_workspace_artifacts(&workspace, recipe, package)?;
        let verifier_workspace = tempfile::tempdir()
            .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
        let verifier_container = if verifier_mode == VerifierMode::Separate {
            copy_workspace_artifacts(&workspace, &verifier_workspace, recipe, package)?;
            let container = format!("{}-verifier", lease.container);
            let verifier_lease = ContainerLease { container };
            create_container(
                &verifier_lease.container,
                &lease.image,
                verifier_workspace.path(),
                recipe,
                package,
                false,
            )?;
            docker(
                ["start", &verifier_lease.container],
                "start separate verifier container",
            )?;
            Some(verifier_lease)
        } else {
            None
        };
        let verifier = verifier_container
            .as_ref()
            .map_or(lease.container.as_str(), |lease| lease.container.as_str());
        docker(
            [
                "cp",
                &format!("{}/.", source_root.join("tests").display()),
                &format!("{verifier}:/tests"),
            ],
            "install verifier files",
        )?;
        docker(
            [
                "exec",
                "--user",
                "root",
                verifier,
                "/bin/sh",
                "-c",
                "mkdir -p /logs/verifier && chmod 0777 /logs /logs/verifier",
            ],
            "prepare verifier logs",
        )?;
        docker_exec(
            self.clock.clone(),
            verifier,
            &["/bin/sh".to_owned(), "/tests/test.sh".to_owned()],
            "run verifier",
            package
                .timeouts()
                .map(|(_, verifier_timeout)| (EvalExecutionPhase::Verifier, verifier_timeout)),
        )?;
        let reward = read_reward(
            verifier,
            if verifier_mode == VerifierMode::Separate {
                &verifier_workspace
            } else {
                &workspace
            },
        )?;
        Ok(LocalExecutionResult {
            artifacts,
            reward,
            verifier: package.source_digest(),
        })
    }
}

fn create_container(
    container: &str,
    image: &str,
    workspace: &std::path::Path,
    recipe: &HarborSandboxRecipe,
    package: &HarborTaskPackage,
    has_agent_instruction: bool,
) -> Result<(), EvalExecutionError> {
    let mut arguments = vec![
        "create".to_owned(),
        "--name".to_owned(),
        container.to_owned(),
        "--network".to_owned(),
        "none".to_owned(),
        "--workdir".to_owned(),
        recipe.workdir.clone(),
        "--volume".to_owned(),
        format!("{}:{}", workspace.display(), recipe.workdir),
    ];
    if let Some((cpus, memory_mb)) = package.container_resources() {
        arguments.extend([
            "--cpus".to_owned(),
            cpus.to_string(),
            "--memory".to_owned(),
            format!("{memory_mb}m"),
        ]);
    }
    if has_agent_instruction {
        arguments.extend([
            "--env".to_owned(),
            format!("AIPERF_EVAL_INSTRUCTION={}", package.instruction()),
        ]);
    }
    arguments.extend([image.to_owned(), "sleep".to_owned(), "infinity".to_owned()]);
    docker(
        arguments.iter().map(String::as_str),
        "create task container",
    )
    .map(|_| ())
}

#[derive(Debug)]
struct ContainerLease {
    container: String,
}

impl Drop for ContainerLease {
    fn drop(&mut self) {
        let _ = Command::new("docker")
            .args(["rm", "--force", &self.container])
            .output();
    }
}

#[derive(Debug)]
struct DockerLease {
    container: String,
    image: String,
}

impl Drop for DockerLease {
    fn drop(&mut self) {
        let _ = Command::new("docker")
            .args(["rm", "--force", &self.container])
            .output();
        let _ = Command::new("docker")
            .args(["image", "rm", &self.image])
            .output();
    }
}

fn docker<'a>(
    arguments: impl IntoIterator<Item = &'a str>,
    action: &str,
) -> Result<Vec<u8>, EvalExecutionError> {
    let arguments = arguments.into_iter().collect::<Vec<_>>();
    let output = Command::new("docker")
        .args(&arguments)
        .output()
        .map_err(|_| EvalExecutionError::ProcessSpawn(format!("docker {action}")))?;
    if output.status.success() {
        Ok(output.stdout)
    } else {
        Err(EvalExecutionError::ProcessFailure(format!(
            "docker {action}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )))
    }
}

fn docker_exec(
    clock: Rc<dyn Clock>,
    container: &str,
    command: &[String],
    action: &str,
    timeout: Option<(EvalExecutionPhase, Duration)>,
) -> Result<(), EvalExecutionError> {
    if command.is_empty() || command.iter().any(|part| part.trim().is_empty()) {
        return Err(EvalExecutionError::InvalidCommand);
    }
    let mut arguments = vec!["exec", container];
    arguments.extend(command.iter().map(String::as_str));
    let Some((phase, timeout)) = timeout else {
        return docker(arguments, action).map(|_| ());
    };
    docker_exec_bounded(clock, container, &arguments, action, phase, timeout)
}

fn docker_exec_bounded(
    clock: Rc<dyn Clock>,
    container: &str,
    arguments: &[&str],
    action: &str,
    phase: EvalExecutionPhase,
    timeout: Duration,
) -> Result<(), EvalExecutionError> {
    let child = Command::new("docker")
        .args(arguments)
        // docker exec output is not part of the evaluation contract. Redirecting both streams
        // prevents an unconsumed pipe from blocking the child past its phase deadline.
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|_| EvalExecutionError::ProcessSpawn(format!("docker {action}")))?;
    let mut process = DockerExecChild { child };
    let mut remove = remove_timed_out_container;
    drive_docker_exec(clock, &mut process, container, phase, timeout, &mut remove)
}

const DOCKER_EXEC_POLL_NS: i64 = 10_000_000;

#[derive(Clone, Debug, PartialEq, Eq)]
enum DockerExecState {
    Running,
    Succeeded,
    Failed(String),
}

trait DockerExecProcess {
    fn try_wait(&mut self) -> Result<DockerExecState, String>;
    fn kill(&mut self) -> Result<(), String>;
    fn wait(&mut self) -> Result<(), String>;
}

struct DockerExecChild {
    child: Child,
}

impl DockerExecProcess for DockerExecChild {
    fn try_wait(&mut self) -> Result<DockerExecState, String> {
        self.child
            .try_wait()
            .map_err(|error| error.to_string())?
            .map_or_else(
                || Ok(DockerExecState::Running),
                |status| {
                    if status.success() {
                        Ok(DockerExecState::Succeeded)
                    } else {
                        Ok(DockerExecState::Failed(status.to_string()))
                    }
                },
            )
    }

    fn kill(&mut self) -> Result<(), String> {
        self.child.kill().map_err(|error| error.to_string())
    }

    fn wait(&mut self) -> Result<(), String> {
        self.child
            .wait()
            .map(|_| ())
            .map_err(|error| error.to_string())
    }
}

fn drive_docker_exec<P, F>(
    clock: Rc<dyn Clock>,
    process: &mut P,
    container: &str,
    phase: EvalExecutionPhase,
    timeout: Duration,
    remove: &mut F,
) -> Result<(), EvalExecutionError>
where
    P: DockerExecProcess,
    F: for<'a> FnMut(&'a str) -> Result<(), EvalExecutionError>,
{
    let result = Rc::new(RefCell::new(None));
    let result_slot = result.clone();
    let outcome = clock.clone().drive(Box::pin(async {
        *result_slot.borrow_mut() =
            Some(wait_for_docker_exec(clock, process, container, phase, timeout, remove).await);
    }));
    if outcome.deadlocked {
        return Err(EvalExecutionError::TerminalUncertainty {
            phase,
            container: container.to_owned(),
            reason: "execution clock reached quiescence before Docker exec terminated".to_owned(),
        });
    }
    result
        .borrow_mut()
        .take()
        .ok_or_else(|| EvalExecutionError::TerminalUncertainty {
            phase,
            container: container.to_owned(),
            reason: "execution clock ended before Docker exec produced a terminal result"
                .to_owned(),
        })?
}

async fn wait_for_docker_exec<P, F>(
    clock: Rc<dyn Clock>,
    process: &mut P,
    container: &str,
    phase: EvalExecutionPhase,
    timeout: Duration,
    remove: &mut F,
) -> Result<(), EvalExecutionError>
where
    P: DockerExecProcess,
    F: for<'a> FnMut(&'a str) -> Result<(), EvalExecutionError>,
{
    let deadline = clock
        .now_ns()
        .saturating_add(timeout.as_nanos().min(i64::MAX as u128) as i64);
    loop {
        if clock.now_ns() >= deadline {
            return terminate_timed_out_exec(process, container, phase, timeout, remove);
        }
        let state = process.try_wait().map_err(|reason| {
            EvalExecutionError::ProcessFailure(format!("docker exec process check: {reason}"))
        })?;
        if clock.now_ns() >= deadline {
            return terminate_timed_out_exec(process, container, phase, timeout, remove);
        }
        match state {
            DockerExecState::Succeeded => {
                return Ok(());
            }
            DockerExecState::Failed(status) => {
                return Err(EvalExecutionError::ProcessFailure(format!(
                    "docker exec exited with {status}"
                )));
            }
            DockerExecState::Running => {
                let remaining_ns = deadline.saturating_sub(clock.now_ns());
                clock
                    .clone()
                    .sleep(remaining_ns.min(DOCKER_EXEC_POLL_NS))
                    .await;
            }
        }
    }
}

fn terminate_timed_out_exec<P, F>(
    process: &mut P,
    container: &str,
    phase: EvalExecutionPhase,
    timeout: Duration,
    remove: &mut F,
) -> Result<(), EvalExecutionError>
where
    P: DockerExecProcess,
    F: for<'a> FnMut(&'a str) -> Result<(), EvalExecutionError>,
{
    let kill = process.kill();
    let reap = process.wait();
    let removal = remove(container);
    if let Err(error) = removal {
        return Err(error);
    }
    let uncertainties = [
        kill.err()
            .map(|reason| format!("could not kill docker exec client: {reason}")),
        reap.err()
            .map(|reason| format!("could not reap docker exec client: {reason}")),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    if !uncertainties.is_empty() {
        return Err(EvalExecutionError::TerminalUncertainty {
            phase,
            container: container.to_owned(),
            reason: uncertainties.join("; "),
        });
    }
    Err(EvalExecutionError::Timeout { phase, timeout })
}

fn remove_timed_out_container(container: &str) -> Result<(), EvalExecutionError> {
    let removal = Command::new("docker")
        .args(["rm", "--force", container])
        .output()
        .map_err(|_| EvalExecutionError::ContainerTeardown {
            container: container.to_owned(),
            reason: "could not start docker rm --force".to_owned(),
        })?;
    if !removal.status.success() && !reports_absent_container(&removal.stderr) {
        return Err(EvalExecutionError::ContainerTeardown {
            container: container.to_owned(),
            reason: String::from_utf8_lossy(&removal.stderr).trim().to_owned(),
        });
    }
    let inspection = Command::new("docker")
        .args(["container", "inspect", container])
        .output()
        .map_err(|_| EvalExecutionError::ContainerTeardown {
            container: container.to_owned(),
            reason: "could not start docker container inspect".to_owned(),
        })?;
    if !inspection.status.success() && reports_absent_container(&inspection.stderr) {
        return Ok(());
    }
    let reason = if inspection.status.success() {
        "docker container inspect found the container after forced removal".to_owned()
    } else {
        String::from_utf8_lossy(&inspection.stderr)
            .trim()
            .to_owned()
    };
    Err(EvalExecutionError::ContainerTeardown {
        container: container.to_owned(),
        reason,
    })
}

fn reports_absent_container(stderr: &[u8]) -> bool {
    let diagnostic = String::from_utf8_lossy(stderr).to_ascii_lowercase();
    diagnostic.contains("no such container") || diagnostic.contains("no such object")
}

fn read_reward(container: &str, workspace: &TempDir) -> Result<RewardDocument, EvalExecutionError> {
    let json = copy_optional(
        container,
        "/logs/verifier/reward.json",
        workspace,
        "reward.json",
    )?;
    let text = copy_optional(
        container,
        "/logs/verifier/reward.txt",
        workspace,
        "reward.txt",
    )?;
    RewardDocument::parse(json.as_deref(), text.as_deref())
        .map_err(|error| EvalExecutionError::ProcessFailure(format!("verifier reward: {error}")))
}

fn copy_optional(
    container: &str,
    source: &str,
    workspace: &TempDir,
    destination: &str,
) -> Result<Option<Vec<u8>>, EvalExecutionError> {
    let destination_path = workspace.path().join(destination);
    let output = Command::new("docker")
        .args([
            "cp",
            &format!("{container}:{source}"),
            destination_path.to_string_lossy().as_ref(),
        ])
        .output()
        .map_err(|_| EvalExecutionError::ProcessSpawn("docker copy verifier reward".to_owned()))?;
    if output.status.success() {
        fs::read(destination_path)
            .map(Some)
            .map_err(|error| EvalExecutionError::Materialization(error.to_string()))
    } else {
        Ok(None)
    }
}

fn collect_workspace_artifacts(
    workspace: &TempDir,
    recipe: &HarborSandboxRecipe,
    package: &HarborTaskPackage,
) -> Result<Vec<(String, ArtifactDigest)>, EvalExecutionError> {
    package
        .declared_artifacts()
        .iter()
        .map(|path| {
            let relative = path
                .strip_prefix(&recipe.workdir)
                .and_then(|path| path.strip_prefix('/'))
                .ok_or_else(|| {
                    EvalExecutionError::Materialization(format!(
                        "Docker artifact must be under {}: {path}",
                        recipe.workdir
                    ))
                })?;
            let bytes = fs::read(workspace.path().join(relative))
                .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
            Ok((path.clone(), ArtifactDigest::from_bytes(&bytes)))
        })
        .collect()
}

fn copy_workspace_artifacts(
    source: &TempDir,
    destination: &TempDir,
    recipe: &HarborSandboxRecipe,
    package: &HarborTaskPackage,
) -> Result<(), EvalExecutionError> {
    for path in package.declared_artifacts() {
        let relative = path
            .strip_prefix(&recipe.workdir)
            .and_then(|path| path.strip_prefix('/'))
            .ok_or_else(|| {
                EvalExecutionError::Materialization(format!(
                    "Docker artifact must be under {}: {path}",
                    recipe.workdir
                ))
            })?;
        let destination_path = destination.path().join(relative);
        if let Some(parent) = destination_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
        }
        fs::copy(source.path().join(relative), destination_path)
            .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, collections::VecDeque, rc::Rc, time::Duration};

    use super::{
        DockerExecProcess, DockerExecState, EvalExecutionError, EvalExecutionPhase,
        drive_docker_exec,
    };
    use crate::clock::SimClock;

    #[test]
    fn completed_command_observed_after_deadline_times_out() {
        let clock = Rc::new(SimClock::new());
        let mut process = FakeDockerExec::new(clock.clone(), [DockerExecState::Succeeded])
            .advance_terminal_to(100);
        let was_removed = Cell::new(false);
        let mut remove = |_: &str| {
            was_removed.set(true);
            Ok(())
        };

        let result = drive_docker_exec(
            clock,
            &mut process,
            "agent-container",
            EvalExecutionPhase::Agent,
            Duration::from_nanos(100),
            &mut remove,
        );

        assert_eq!(
            result,
            Err(EvalExecutionError::Timeout {
                phase: EvalExecutionPhase::Agent,
                timeout: Duration::from_nanos(100),
            })
        );
        assert!(was_removed.get());
        assert!(process.was_killed.get());
        assert!(process.was_reaped.get());
    }

    #[test]
    fn failed_command_observed_after_deadline_also_times_out() {
        let clock = Rc::new(SimClock::new());
        let mut process = FakeDockerExec::new(
            clock.clone(),
            [DockerExecState::Failed("exit status: 1".to_owned())],
        )
        .advance_terminal_to(100);
        let was_removed = Cell::new(false);
        let mut remove = |_: &str| {
            was_removed.set(true);
            Ok(())
        };

        let result = drive_docker_exec(
            clock,
            &mut process,
            "agent-container",
            EvalExecutionPhase::Agent,
            Duration::from_nanos(100),
            &mut remove,
        );

        assert_eq!(
            result,
            Err(EvalExecutionError::Timeout {
                phase: EvalExecutionPhase::Agent,
                timeout: Duration::from_nanos(100),
            })
        );
        assert!(was_removed.get());
    }

    #[test]
    fn kill_failure_returns_typed_terminal_uncertainty_after_removal() {
        let clock = Rc::new(SimClock::new());
        let mut process =
            FakeDockerExec::new(clock.clone(), [DockerExecState::Running]).kill_fails();
        let was_removed = Cell::new(false);
        let mut remove = |_: &str| {
            was_removed.set(true);
            Ok(())
        };

        let result = drive_docker_exec(
            clock,
            &mut process,
            "agent-container",
            EvalExecutionPhase::Agent,
            Duration::from_nanos(10),
            &mut remove,
        );

        assert!(matches!(
            result,
            Err(EvalExecutionError::TerminalUncertainty {
                phase: EvalExecutionPhase::Agent,
                container,
                ..
            }) if container == "agent-container"
        ));
        assert!(was_removed.get());
        assert!(process.was_reaped.get());
    }

    #[test]
    fn reap_failure_returns_typed_terminal_uncertainty_after_removal() {
        let clock = Rc::new(SimClock::new());
        let mut process =
            FakeDockerExec::new(clock.clone(), [DockerExecState::Running]).wait_fails();
        let was_removed = Cell::new(false);
        let mut remove = |_: &str| {
            was_removed.set(true);
            Ok(())
        };

        let result = drive_docker_exec(
            clock,
            &mut process,
            "agent-container",
            EvalExecutionPhase::Agent,
            Duration::from_nanos(10),
            &mut remove,
        );

        assert!(matches!(
            result,
            Err(EvalExecutionError::TerminalUncertainty {
                phase: EvalExecutionPhase::Agent,
                container,
                ..
            }) if container == "agent-container"
        ));
        assert!(was_removed.get());
        assert!(process.was_killed.get());
        assert!(process.was_reaped.get());
    }

    struct FakeDockerExec {
        clock: Rc<SimClock>,
        states: VecDeque<DockerExecState>,
        advance_terminal_to: Option<i64>,
        kill_fails: bool,
        wait_fails: bool,
        was_killed: Cell<bool>,
        was_reaped: Cell<bool>,
    }

    impl FakeDockerExec {
        fn new(clock: Rc<SimClock>, states: impl IntoIterator<Item = DockerExecState>) -> Self {
            Self {
                clock,
                states: states.into_iter().collect(),
                advance_terminal_to: None,
                kill_fails: false,
                wait_fails: false,
                was_killed: Cell::new(false),
                was_reaped: Cell::new(false),
            }
        }

        fn advance_terminal_to(mut self, time_ns: i64) -> Self {
            self.advance_terminal_to = Some(time_ns);
            self
        }

        fn kill_fails(mut self) -> Self {
            self.kill_fails = true;
            self
        }

        fn wait_fails(mut self) -> Self {
            self.wait_fails = true;
            self
        }
    }

    impl DockerExecProcess for FakeDockerExec {
        fn try_wait(&mut self) -> Result<DockerExecState, String> {
            let state = self.states.pop_front().unwrap_or(DockerExecState::Running);
            if state != DockerExecState::Running {
                if let Some(time_ns) = self.advance_terminal_to {
                    self.clock.advance_to(time_ns);
                }
            }
            Ok(state)
        }

        fn kill(&mut self) -> Result<(), String> {
            self.was_killed.set(true);
            if self.kill_fails {
                Err("simulated kill failure".to_owned())
            } else {
                Ok(())
            }
        }

        fn wait(&mut self) -> Result<(), String> {
            self.was_reaped.set(true);
            if self.wait_fails {
                Err("simulated reap failure".to_owned())
            } else {
                Ok(())
            }
        }
    }
}
