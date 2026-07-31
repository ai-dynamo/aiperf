// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Shared harness for native dry-run component tests.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;

#[path = "../../../test-support/timing_helpers.rs"]
#[allow(dead_code)]
mod timing_helpers;

#[allow(unused_imports)]
pub use timing_helpers::*;

pub const MODEL: &str = "openai/gpt-oss-120b";
pub const TOKENIZER: &str = "cl100k_base";
pub const TTFT_MS: f64 = 10.0;
pub const ITL_MS: f64 = 2.0;
pub const ISL: u32 = 20;
pub const OSL: u32 = 4;

/// A completed native profile and its isolated artifact directory.
pub struct Run {
    pub output: Output,
    pub artifacts: Artifacts,
    _temporary_directory: tempfile::TempDir,
}

impl Run {
    /// Whether the child exited successfully.
    pub fn success(&self) -> bool {
        self.output.status.success()
    }

    /// Assert success with complete child diagnostics.
    pub fn assert_success(&self) {
        assert!(
            self.success(),
            "dry-run profile failed (status {}):\nstdout:\n{}\nstderr:\n{}",
            self.output.status,
            String::from_utf8_lossy(&self.output.stdout),
            String::from_utf8_lossy(&self.output.stderr),
        );
    }
}

impl ProfileExport for Run {
    fn profile_export_records(&self) -> Vec<serde_json::Value> {
        self.artifacts.jsonl()
    }
}

/// Artifact reader rooted at one profile's temporary output directory.
#[derive(Clone, Debug)]
pub struct Artifacts {
    pub dir: PathBuf,
}

impl Artifacts {
    pub fn jsonl(&self) -> Vec<serde_json::Value> {
        self.jsonl_file("profile_export.jsonl")
    }

    pub fn raw_jsonl(&self) -> Vec<serde_json::Value> {
        self.jsonl_file("profile_export_raw.jsonl")
    }

    pub fn summary(&self) -> serde_json::Value {
        self.json_file("profile_export_aiperf.json")
    }

    pub fn native_report(&self) -> serde_json::Value {
        self.json_file("native-v2.json")
    }

    pub fn metric(record: &serde_json::Value, name: &str) -> f64 {
        record["metrics"][name]["value"]
            .as_f64()
            .unwrap_or_else(|| panic!("missing numeric metric {name}: {record}"))
    }

    pub fn stable_record_projection(&self) -> Vec<String> {
        self.jsonl()
            .iter()
            .map(|record| {
                serde_json::json!({
                    "conversation_id": record["metadata"]["conversation_id"],
                    "turn_index": record["metadata"]["turn_index"],
                    "isl": record["metrics"]["input_sequence_length"]["value"],
                    "osl": record["metrics"]["output_sequence_length"]["value"],
                    "latency": record["metrics"]["request_latency"]["value"],
                    "error": record["error"],
                })
                .to_string()
            })
            .collect()
    }

    fn json_file(&self, name: &str) -> serde_json::Value {
        let path = self.dir.join(name);
        serde_json::from_slice(
            &std::fs::read(&path)
                .unwrap_or_else(|e| panic!("read artifact {}: {e}", path.display())),
        )
        .unwrap_or_else(|e| panic!("parse artifact {}: {e}", path.display()))
    }

    fn jsonl_file(&self, name: &str) -> Vec<serde_json::Value> {
        let path = self.dir.join(name);
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read artifact {}: {e}", path.display()))
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                serde_json::from_str(line)
                    .unwrap_or_else(|e| panic!("parse record in {}: {e}", path.display()))
            })
            .collect()
    }
}

/// Run the standard deterministic dry-run profile with caller-supplied flags.
pub fn run(extra: &[&str]) -> Run {
    let temporary_directory = tempfile::tempdir().expect("create artifact directory");
    let artifact_directory = temporary_directory
        .path()
        .to_str()
        .expect("artifact path is UTF-8");
    let mut args = vec![
        "profile",
        "--model",
        MODEL,
        "--tokenizer",
        TOKENIZER,
        "--endpoint-type",
        "chat",
        "--streaming",
        "--synthetic-input-tokens-mean",
        "20",
        "--synthetic-input-tokens-stddev",
        "0",
        "--output-tokens-mean",
        "4",
        "--output-tokens-stddev",
        "0",
        "--dry-run",
        "--dry-run-clock",
        "sim",
        "--dry-run-ttft-ms",
        "10",
        "--dry-run-itl-ms",
        "2",
        "--artifact-dir",
        artifact_directory,
    ];
    args.extend(extra.iter().copied());
    execute(
        args.into_iter().map(str::to_owned).collect(),
        temporary_directory,
    )
}

/// Run a complete Config-v2 YAML document after substituting its artifact path.
pub fn run_config(yaml: &str) -> Run {
    let temporary_directory = tempfile::tempdir().expect("create artifact directory");
    let artifact_directory = temporary_directory
        .path()
        .to_str()
        .expect("artifact path is UTF-8");
    let config_path = temporary_directory.path().join("benchmark.yaml");
    std::fs::write(
        &config_path,
        yaml.replace("$ARTIFACT_DIR", artifact_directory),
    )
    .expect("write dry-run config");
    execute(
        vec![
            "profile".to_string(),
            "--config".to_string(),
            config_path.display().to_string(),
        ],
        temporary_directory,
    )
}

/// Run a timing command built by the shared timing helper under the deterministic
/// dry-run transport. The builder supplies benchmark options; this harness adds
/// the tokenizer, synthetic input shape, dry-run model, and artifact directory.
pub fn run_timing(config: &TimingTestConfig, options: TimingCommandOptions<'_>) -> Run {
    let command = build_timing_command(config, options);
    let temporary_directory = tempfile::tempdir().expect("create artifact directory");
    let artifact_directory = temporary_directory
        .path()
        .to_str()
        .expect("artifact path is UTF-8");
    let mut args = vec!["profile".to_string()];
    args.extend(command.split_whitespace().map(str::to_owned));
    args.extend([
        "--tokenizer".to_string(),
        TOKENIZER.to_string(),
        "--endpoint-type".to_string(),
        "chat".to_string(),
        "--synthetic-input-tokens-mean".to_string(),
        "20".to_string(),
        "--synthetic-input-tokens-stddev".to_string(),
        "0".to_string(),
        "--output-tokens-stddev".to_string(),
        "0".to_string(),
        "--dry-run".to_string(),
        "--dry-run-clock".to_string(),
        "sim".to_string(),
        "--dry-run-ttft-ms".to_string(),
        "10".to_string(),
        "--dry-run-itl-ms".to_string(),
        "2".to_string(),
        "--artifact-dir".to_string(),
        artifact_directory.to_string(),
    ]);
    execute(args, temporary_directory)
}

fn execute(args: Vec<String>, temporary_directory: tempfile::TempDir) -> Run {
    let output = Command::new(binary())
        .args(args)
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("PYTHONUNBUFFERED", "1")
        .env("MALLOC_ARENA_MAX", "2")
        .output()
        .expect("spawn aiperf --dry-run");
    let artifacts = Artifacts {
        dir: temporary_directory.path().to_path_buf(),
    };
    Run {
        output,
        artifacts,
        _temporary_directory: temporary_directory,
    }
}

/// Path to the `aiperf` binary under test, from `AIPERF_DRY_RUN_BIN`.
///
/// `cargo test` cannot build it: this package declares no `[[bin]]`, so
/// `CARGO_BIN_EXE_aiperf` is unset, and `[[bin]]` targets of another package are
/// not built for a dependent. The binary must therefore be named explicitly.
/// Searching for one instead would silently pick some other build — a different
/// profile, or a stale artifact left in `target/` — and report passes for code
/// that was never compiled. That failure mode is a wrong answer, not an error, so
/// an unset variable is a hard panic.
fn binary() -> &'static str {
    static BINARY: OnceLock<String> = OnceLock::new();
    BINARY.get_or_init(|| {
        let path = match std::env::var("AIPERF_DRY_RUN_BIN") {
            Ok(path) if !path.is_empty() => path,
            _ => panic!(
                "AIPERF_DRY_RUN_BIN is not set.\n\
                 This suite drives a real `aiperf` binary and cannot build one itself.\n\
                 \x20 Use:    make test-dry-run-rust\n\
                 \x20 Or pin: cargo build --release -p aiperf-cli\n\
                 \x20         AIPERF_DRY_RUN_BIN=$PWD/rust/target/release/aiperf cargo test -p aiperf-dry-run-tests"
            ),
        };
        if !Path::new(&path).is_file() {
            panic!("AIPERF_DRY_RUN_BIN={path} is not a readable file");
        }
        path
    })
}

#[allow(dead_code)]
pub fn assert_file(artifacts: &Artifacts, name: &str) {
    let path: &Path = &artifacts.dir.join(name);
    assert!(path.is_file(), "missing artifact {}", path.display());
}
