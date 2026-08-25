// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Process contracts for explicit Python utility delegation.

#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn recording_python(directory: &Path) -> PathBuf {
    let interpreter = directory.join("record-python");
    fs::write(
        &interpreter,
        "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$AIPERF_PYTHON_RECORD\"\nprintf delegated >&2\nexit 23\n",
    )
    .expect("write Python test double");
    let mut permissions = fs::metadata(&interpreter)
        .expect("read test double metadata")
        .permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&interpreter, permissions).expect("make test double executable");
    interpreter
}

fn run(arguments: &[&str], interpreter: &Path, record: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_aiperf"))
        .args(arguments)
        .env("AIPERF_PYTHON", interpreter)
        .env("AIPERF_PYTHON_RECORD", record)
        .output()
        .expect("run native aiperf")
}

fn recorded_arguments(record: &Path) -> Vec<String> {
    fs::read_to_string(record)
        .expect("test double was invoked")
        .lines()
        .map(str::to_owned)
        .collect()
}

#[test]
fn plot_delegates_to_the_python_utility_entry_point() {
    let directory = tempfile::tempdir().expect("create fixture directory");
    let record = directory.path().join("argv");
    let output = run(
        &["plot", "report.json"],
        &recording_python(directory.path()),
        &record,
    );

    assert_eq!(output.status.code(), Some(23));
    assert_eq!(
        recorded_arguments(&record),
        ["-m", "aiperf", "plot", "report.json"]
    );
    assert_eq!(output.stderr, b"delegated\n");
}

#[test]
fn slurm_generate_is_native_and_never_starts_python() {
    let directory = tempfile::tempdir().expect("create fixture directory");
    let record = directory.path().join("argv");
    let config = directory.path().join("benchmark.yaml");
    fs::write(&config, "benchmark: {}\n").expect("write config fixture");
    let absolute = fs::canonicalize(&config).expect("canonical config path");

    let output = run(
        &[
            "slurm",
            "generate",
            "--config",
            config.to_str().expect("utf-8 path"),
            "--cells",
            "2",
        ],
        &recording_python(directory.path()),
        &record,
    );

    assert_eq!(output.status.code(), Some(0));
    assert!(!record.exists(), "Python test double was invoked");
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        format!(
            "#!/bin/bash\n\
             #SBATCH --job-name=aiperf\n\
             #SBATCH --nodes=3\n\
             #SBATCH --ntasks=3\n\
             #SBATCH --ntasks-per-node=1\n\
             #SBATCH --time=01:00:00\n\
             \n\
             export AIPERF_CELL_LAUNCHER=slurm\n\
             export AIPERF_CONTROLLER_PORT=9500\n\
             \n\
             srun aiperf slurm run --config {}\n",
            absolute.display()
        )
    );
}

#[test]
fn help_and_completion_are_explicit_python_utility_routes() {
    let directory = tempfile::tempdir().expect("create fixture directory");
    let interpreter = recording_python(directory.path());
    for arguments in [
        ["--help"].as_slice(),
        ["--install-completion", "bash"].as_slice(),
    ] {
        let record = directory.path().join(format!("argv-{}", arguments[0]));
        let output = run(arguments, &interpreter, &record);

        assert_eq!(output.status.code(), Some(23));
        let mut expected = vec!["-m".to_string(), "aiperf".to_string()];
        expected.extend(arguments.iter().map(|argument| (*argument).to_string()));
        assert_eq!(recorded_arguments(&record), expected);
    }
}

#[test]
fn version_is_native_and_never_starts_python() {
    let directory = tempfile::tempdir().expect("create fixture directory");
    let record = directory.path().join("argv");
    let output = run(&["--version"], &recording_python(directory.path()), &record);

    assert_eq!(output.status.code(), Some(0));
    assert_eq!(output.stdout, format!("{}\n", env!("CARGO_PKG_VERSION")).as_bytes());
    assert!(!record.exists(), "Python test double was invoked");
}

#[test]
fn service_and_unknown_commands_refuse_without_starting_python() {
    let directory = tempfile::tempdir().expect("create fixture directory");
    let interpreter = recording_python(directory.path());
    for (arguments, expected_message) in [
        (
            ["service"].as_slice(),
            "aiperf service is unavailable from the native binary",
        ),
        (
            ["--help", "service"].as_slice(),
            "unsupported native aiperf command",
        ),
        (
            ["definitely-unknown"].as_slice(),
            "unsupported native aiperf command",
        ),
        (
            ["slurm", "other"].as_slice(),
            "unsupported native aiperf command",
        ),
    ] {
        let record = directory.path().join(format!("argv-{}", arguments[0]));
        let output = run(arguments, &interpreter, &record);

        assert_eq!(output.status.code(), Some(1));
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(expected_message),
            "stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            !record.exists(),
            "Python test double was invoked for {arguments:?}"
        );
    }
}
