// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Redacted structured Docker command contracts for Harbor execution.

use std::{cell::Cell, collections::BTreeMap, fs};

use aiperf_runtime::eval::{
    DockerBuildRequest, DockerCopyRequest, DockerCreateRequest, DockerExecRequest,
    DockerRemoveRequest, DockerRuntime, DockerStartRequest, EnvName, EvalExecutionError,
    HarborImporter, HarborSource, NativeSourceAcquirer, ProviderCapabilities, SecretProvider,
    SecretValue, preflight_docker,
};

#[derive(Default)]
struct RecordingRuntime {
    build_calls: Cell<usize>,
}

impl DockerRuntime for RecordingRuntime {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::none()
    }

    fn build(&self, _: &DockerBuildRequest) -> Result<(), EvalExecutionError> {
        self.build_calls.set(self.build_calls.get() + 1);
        Ok(())
    }

    fn create(&self, _: &DockerCreateRequest) -> Result<(), EvalExecutionError> {
        Ok(())
    }

    fn start(&self, _: &DockerStartRequest) -> Result<(), EvalExecutionError> {
        Ok(())
    }

    fn exec(&self, _: &DockerExecRequest) -> Result<(), EvalExecutionError> {
        Ok(())
    }

    fn copy(&self, _: &DockerCopyRequest) -> Result<(), EvalExecutionError> {
        Ok(())
    }

    fn remove(&self, _: &DockerRemoveRequest) -> Result<(), EvalExecutionError> {
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
