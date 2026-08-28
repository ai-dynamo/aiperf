// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Socket-free product fixture for native streaming shadow replay.
//!
//! Owns the child `aiperf` process, the scratch source/checkpoint tree, and
//! bounded artifact readback for one row.
//!
//! Three rules this module exists to enforce:
//!
//! - **No implicit binary search.** [`product_binary`] reads the same
//!   `AIPERF_DRY_RUN_BIN` variable `common::binary` reads and panics when it is
//!   unset, so a row can never silently measure a stale `target/` artifact.
//! - **No sleep for correctness.** Every wait is a bounded poll over a durable
//!   predicate — the checkpoint manifest's committed epoch — whose deadline is
//!   a *failure* bound only. A slow machine is slow, never wrong.
//! - **No leaked child.** [`SupervisedChild`] delivers SIGINT, escalates to
//!   SIGKILL after a grace window, and joins in `Drop`, so a panicking row
//!   still reaps its process.
//!
//! The committed fixtures under `fixtures/streaming/` are rendered with an
//! explicit three-token set — `$ARTIFACT_DIR`, `$SOURCE_ROOT`, and
//! `$CHECKPOINT_ROOT`. `common::run_config` substitutes only `$ARTIFACT_DIR`;
//! widening that shared contract would change the meaning of every other
//! suite's inline YAML, so this module renders its own.

#![allow(dead_code)]

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// Longest a row waits for a committed checkpoint generation before failing.
const GENERATION_DEADLINE: Duration = Duration::from_secs(30);

/// Longest a row waits for a child to exit after SIGINT before SIGKILL.
const TERMINATE_GRACE: Duration = Duration::from_secs(10);

/// Longest a non-restarting invocation may run before the row fails.
const RUN_DEADLINE: Duration = Duration::from_secs(120);

/// Interval between durable-predicate reads inside a bounded poll.
///
/// This is a poll cadence, not a correctness delay: the loop re-reads a durable
/// file whose appearance *is* the event being waited on.
const POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Path to the `aiperf` binary under test, from `AIPERF_DRY_RUN_BIN`.
///
/// Deliberately identical in contract to `common::binary`: this suite drives a
/// real product binary and cannot build one itself. Searching `target/` instead
/// would report passes for code that was never compiled — a wrong answer rather
/// than an error — so an unset variable is a hard panic.
pub fn product_binary() -> &'static str {
    static BINARY: OnceLock<String> = OnceLock::new();
    BINARY.get_or_init(|| {
        let path = match std::env::var("AIPERF_DRY_RUN_BIN") {
            Ok(path) if !path.is_empty() => path,
            _ => panic!(
                "AIPERF_DRY_RUN_BIN is not set.\n\
                 The streaming product suite drives a real `aiperf` binary and cannot build one.\n\
                 \x20 Pin: cargo build --release -p aiperf-cli \\\n\
                 \x20        --features streaming-s3,cellular,parquet,grpc"
            ),
        };
        assert!(
            Path::new(&path).is_file(),
            "AIPERF_DRY_RUN_BIN={path} is not a readable file"
        );
        path
    })
}

/// Directory holding the committed Config-v2 streaming fixtures.
pub fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/streaming")
}

/// One socket-free streaming product scenario and its private scratch tree.
///
/// The scratch directory owns the source root, the checkpoint root, and the
/// artifact root, so rows never share filesystem state and the tree is removed
/// when the fixture drops.
pub struct StreamingProductFixture {
    scratch: tempfile::TempDir,
    fixture: &'static str,
    source_root: PathBuf,
    checkpoint_root: PathBuf,
    artifact_root: PathBuf,
}

impl StreamingProductFixture {
    /// Local follow source whose sessions span a partition boundary and whose
    /// session program is the incremental graph program, under a periodic
    /// `local` checkpoint backend.
    pub fn local_follow_cross_chunk_graph() -> Self {
        let fixture = Self::new("local_follow_cross_chunk_graph.yaml");
        fixture.write_partition("000-a.parquet", PARTITION_A);
        fixture
    }

    /// Local finite source replayed by the `conversation` session program with
    /// no checkpoint backend — the sealed single-pass reference shape.
    pub fn local_finite_conversation() -> Self {
        let fixture = Self::new("local_finite_conversation.yaml");
        fixture.write_partition("000-a.parquet", PARTITION_A);
        fixture.write_partition("001-b.parquet", PARTITION_B);
        fixture
    }

    /// Every streaming component identifier absent from the compiled inventory.
    pub fn unregistered_components() -> Self {
        Self::new("unregistered_components.yaml")
    }

    fn new(fixture: &'static str) -> Self {
        let scratch = tempfile::tempdir().expect("create streaming scratch directory");
        let source_root = scratch.path().join("source");
        let checkpoint_root = scratch.path().join("checkpoint");
        let artifact_root = scratch.path().join("artifacts");
        for directory in [&source_root, &checkpoint_root, &artifact_root] {
            std::fs::create_dir_all(directory).expect("create streaming scratch subdirectory");
        }
        Self {
            scratch,
            fixture,
            source_root,
            checkpoint_root,
            artifact_root,
        }
    }

    /// The acquired source root for this row.
    pub fn source_root(&self) -> &Path {
        &self.source_root
    }

    /// The durable checkpoint root for this row.
    pub fn checkpoint_root(&self) -> &Path {
        &self.checkpoint_root
    }

    /// The artifact root for this row.
    pub fn artifact_root(&self) -> &Path {
        &self.artifact_root
    }

    /// Publish one partition into the source root by atomic rename.
    ///
    /// Rename, not create-then-write: the local follow source accepts
    /// publish-by-rename only, so a partially written file must never be
    /// visible under its final name.
    pub fn write_partition(&self, name: &str, bytes: &[u8]) {
        let staging = self.scratch.path().join(format!(".staging-{name}"));
        std::fs::write(&staging, bytes).expect("stage source partition");
        std::fs::rename(&staging, self.source_root.join(name)).expect("publish source partition");
    }

    /// Publish the second partition, which the resumed run must consume.
    pub fn publish_partition_b(&self) {
        self.write_partition("001-b.parquet", PARTITION_B);
    }

    /// Publish the authored seal marker, ending follow discovery.
    pub fn publish_seal(&self) {
        self.write_partition("SEALED", b"");
    }

    /// Render the committed fixture into this row's scratch tree.
    ///
    /// The token set is exactly `$ARTIFACT_DIR`, `$SOURCE_ROOT`, and
    /// `$CHECKPOINT_ROOT`; an unsubstituted token is a fixture defect and
    /// panics rather than reaching the product as a literal path.
    pub fn render_config(&self) -> PathBuf {
        let source = fixture_root().join(self.fixture);
        let yaml = std::fs::read_to_string(&source)
            .unwrap_or_else(|e| panic!("read fixture {}: {e}", source.display()));
        let rendered = yaml
            .replace("$ARTIFACT_DIR", &self.artifact_root.display().to_string())
            .replace("$SOURCE_ROOT", &self.source_root.display().to_string())
            .replace(
                "$CHECKPOINT_ROOT",
                &self.checkpoint_root.display().to_string(),
            );
        assert!(
            !rendered.contains('$'),
            "fixture {} has an unsubstituted token",
            self.fixture
        );
        let path = self.scratch.path().join("benchmark.yaml");
        std::fs::write(&path, rendered).expect("write rendered streaming config");
        path
    }

    /// Run `aiperf config validate` over the rendered fixture.
    ///
    /// Validation is socket-free and dataset-free by construction, so this row
    /// proves the static registry stage without any execution effect.
    pub fn validate(&self) -> ProductRun {
        let config = self.render_config();
        self.invoke(&[
            "config".to_owned(),
            "validate".to_owned(),
            config.display().to_string(),
        ])
    }

    /// Run one uninterrupted `aiperf profile` over the complete source set.
    pub fn run_sealed_reference(&self) -> ProductRun {
        self.publish_seal();
        self.profile()
    }

    /// Run `aiperf profile --config <rendered>` to completion.
    pub fn profile(&self) -> ProductRun {
        let config = self.render_config();
        self.invoke(&[
            "profile".to_owned(),
            "--config".to_owned(),
            config.display().to_string(),
        ])
    }

    /// Launch a profile, wait for the first committed checkpoint generation,
    /// then terminate the child and return its completed invocation.
    ///
    /// Fails when the child exits before committing a generation: that is the
    /// signal the run never reached the streaming execution path at all, and
    /// silently returning a zero-generation result would let a restart row pass
    /// against a product that never started.
    pub fn run_until_checkpoint_then_kill(&self) -> Result<ProductRun, ProductError> {
        let config = self.render_config();
        let mut child = SupervisedChild::spawn(&[
            "profile".to_owned(),
            "--config".to_owned(),
            config.display().to_string(),
        ])?;
        let waited = wait_for_generation(&self.checkpoint_root, 1, GENERATION_DEADLINE, &mut child);
        match waited {
            Ok(_) => {
                let run = child.terminate(TERMINATE_GRACE, self.artifact_root.clone());
                Ok(run)
            }
            Err(error) => {
                let run = child.terminate(TERMINATE_GRACE, self.artifact_root.clone());
                Err(ProductError::NoCommittedGeneration {
                    detail: error,
                    exit_code: run.exit_code,
                    stderr: run.stderr,
                })
            }
        }
    }

    /// Resume the same checkpoint root under the exact logical replay run id.
    ///
    /// The locator is the only authored resume selector: `Resume` always names
    /// one exact prior run and never means "the latest arbitrary run".
    pub fn resume(&self, locator: &str) -> ProductRun {
        let config = self.render_config();
        self.invoke(&[
            "profile".to_owned(),
            "--config".to_owned(),
            config.display().to_string(),
            "--streaming-resume".to_owned(),
            locator.to_owned(),
        ])
    }

    fn invoke(&self, args: &[String]) -> ProductRun {
        let mut child = SupervisedChild::spawn(args).expect("spawn aiperf");
        let started = Instant::now();
        loop {
            match child.child.try_wait().expect("poll aiperf child") {
                Some(_) => break,
                None if started.elapsed() > RUN_DEADLINE => {
                    return child.terminate(TERMINATE_GRACE, self.artifact_root.clone());
                }
                None => std::thread::sleep(POLL_INTERVAL),
            }
        }
        child.join(self.artifact_root.clone())
    }
}

/// Why a supervised streaming invocation could not produce the state a row
/// needs.
#[derive(Debug)]
pub enum ProductError {
    /// The run exited or timed out without committing a checkpoint generation.
    NoCommittedGeneration {
        /// The bounded-poll diagnostic.
        detail: String,
        /// The child's exit code.
        exit_code: i32,
        /// The child's complete stderr.
        stderr: String,
    },
    /// The child could not be spawned at all.
    Spawn(String),
}

impl std::fmt::Display for ProductError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoCommittedGeneration {
                detail,
                exit_code,
                stderr,
            } => write!(
                f,
                "no committed checkpoint generation ({detail}); \
                 child exited {exit_code} with stderr:\n{stderr}"
            ),
            Self::Spawn(message) => write!(f, "spawn aiperf: {message}"),
        }
    }
}

/// One completed child invocation and its bounded artifact readback.
pub struct ProductRun {
    /// Process exit code, or `-1` when terminated by signal.
    pub exit_code: i32,
    /// Complete child stdout.
    pub stdout: String,
    /// Complete child stderr.
    pub stderr: String,
    /// Artifact root this invocation was pointed at.
    pub artifacts: PathBuf,
}

impl ProductRun {
    /// Whether the child exited successfully.
    pub fn success(&self) -> bool {
        self.exit_code == 0
    }

    /// Both child streams, for diagnostics and leak scanning.
    pub fn combined_output(&self) -> String {
        format!("{}\n{}", self.stdout, self.stderr)
    }

    /// Assert the run refused and its refusal names every expected fragment.
    pub fn assert_refused_naming(&self, fragments: &[&str]) {
        assert!(
            !self.success(),
            "expected a refusal, but the run exited 0:\n{}",
            self.combined_output()
        );
        let output = self.combined_output();
        for fragment in fragments {
            assert!(
                output.contains(fragment),
                "refusal must name {fragment:?}:\n{output}"
            );
        }
    }

    /// Every regular file under the artifact root, sorted by relative path.
    pub fn artifact_files(&self) -> Vec<PathBuf> {
        let mut found = Vec::new();
        collect_files(&self.artifacts, &self.artifacts, &mut found);
        found.sort();
        found
    }

    /// Latest committed generation epoch under a checkpoint root, if any.
    pub fn generation(checkpoint_root: &Path) -> Option<u64> {
        read_current_epoch(checkpoint_root)
    }

    /// Sorted stable action ids paired with their content digests.
    ///
    /// This is the comparison key a restart row uses: a resumed run and its
    /// sealed reference must agree on this multiset exactly, independent of the
    /// order records were written in.
    pub fn logical_record_multiset(&self) -> Vec<(String, String)> {
        let mut records: Vec<(String, String)> = self
            .jsonl("profile_export.jsonl")
            .iter()
            .map(|record| {
                (
                    record["metadata"]["stable_action_id"]
                        .as_str()
                        .unwrap_or_default()
                        .to_owned(),
                    record["metadata"]["content_digest"]
                        .as_str()
                        .unwrap_or_default()
                        .to_owned(),
                )
            })
            .collect();
        records.sort();
        records
    }

    /// Canonical projection of the compacted metric store.
    pub fn compacted_metric_store(&self) -> serde_json::Value {
        self.json("profile_export_aiperf.json")
    }

    /// Public execution status: `complete`, `degraded`, `export_incomplete`, or
    /// `failed`.
    pub fn public_status(&self) -> String {
        self.json("native-v2.json")["streaming"]["status"]
            .as_str()
            .unwrap_or(if self.success() { "complete" } else { "failed" })
            .to_owned()
    }

    /// Bounded issue-receipt projection, sorted by issue id.
    pub fn issue_receipts(&self) -> Vec<serde_json::Value> {
        let mut receipts = self.json("native-v2.json")["streaming"]["issues"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        receipts.sort_by_key(|receipt| receipt["id"].as_str().unwrap_or_default().to_owned());
        receipts
    }

    /// Incomplete derived sinks, sorted by `(generation, digest, sink_id)`.
    pub fn incomplete_sinks(&self) -> Vec<serde_json::Value> {
        let mut sinks = self.json("native-v2.json")["streaming"]["incomplete_sinks"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        sinks.sort_by_key(|sink| {
            (
                sink["generation"].as_u64().unwrap_or_default(),
                sink["digest"].as_str().unwrap_or_default().to_owned(),
                sink["sink_id"].as_str().unwrap_or_default().to_owned(),
            )
        });
        sinks
    }

    /// Assert no raw source byte and no authored secret reached any artifact.
    ///
    /// Source bytes are the product's private input: normalized request bodies
    /// may appear in raw records, but the acquired partition's own bytes must
    /// never be emitted, and neither must a credential the fixture authored.
    pub fn assert_no_raw_or_secret_leak(&self, needles: &[&str]) {
        let mut haystack = self.combined_output();
        for file in self.artifact_files() {
            let path = self.artifacts.join(&file);
            if let Ok(bytes) = std::fs::read(&path) {
                haystack.push('\n');
                haystack.push_str(&String::from_utf8_lossy(&bytes));
            }
        }
        for needle in needles {
            assert!(
                !haystack.contains(needle),
                "raw source bytes or an authored secret ({needle:?}) reached an artifact or the \
                 child's output"
            );
        }
    }

    fn json(&self, name: &str) -> serde_json::Value {
        match std::fs::read(self.artifacts.join(name)) {
            Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null),
            Err(_) => serde_json::Value::Null,
        }
    }

    fn jsonl(&self, name: &str) -> Vec<serde_json::Value> {
        match std::fs::read_to_string(self.artifacts.join(name)) {
            Ok(text) => text
                .lines()
                .filter(|line| !line.trim().is_empty())
                .filter_map(|line| serde_json::from_str(line).ok())
                .collect(),
            Err(_) => Vec::new(),
        }
    }
}

/// Assert the public status vocabulary and its exact preconditions.
///
/// `failed` requires a checked terminal-boundary invariant: no derived sink
/// failure and no ordinary data fault may reach it. `degraded` requires at
/// least one hole, quarantine, or failed terminal action *and* a readable
/// authoritative generation. `export_incomplete` requires a readable generation
/// plus at least one incomplete derived sink.
pub fn assert_public_status_vocabulary(
    status: &str,
    receipts: &[serde_json::Value],
    incomplete_sinks: &[serde_json::Value],
    generation: Option<u64>,
) {
    match status {
        "failed" => assert!(
            receipts
                .iter()
                .any(|receipt| receipt.get("terminal_invariant").is_some()),
            "failed without a checked terminal-boundary invariant: {receipts:?}"
        ),
        "degraded" => {
            assert!(
                generation.is_some(),
                "degraded must keep a readable authoritative generation"
            );
            assert!(
                receipts.iter().any(|receipt| matches!(
                    receipt["disposition"].as_str(),
                    Some("hole" | "quarantine" | "terminal_action_receipt")
                )),
                "degraded without a hole, quarantine, or failed terminal action: {receipts:?}"
            );
        }
        "export_incomplete" => {
            assert!(
                generation.is_some(),
                "export_incomplete must keep a readable authoritative generation"
            );
            assert!(
                !incomplete_sinks.is_empty(),
                "export_incomplete without an incomplete derived sink"
            );
        }
        "complete" => {
            assert!(receipts.is_empty(), "complete with issue receipts");
            assert!(
                incomplete_sinks.is_empty(),
                "complete with an incomplete derived sink"
            );
        }
        other => panic!("unknown public status {other:?}"),
    }
}

/// Poll the durable checkpoint manifest until `epoch >= wanted`.
///
/// The deadline is a *failure* bound, not a correctness bound: the loop reads a
/// durable file whose appearance is the event, so a slow machine is slow, never
/// wrong. The child is polled in the same loop so an early exit is reported
/// immediately instead of burning the whole deadline.
fn wait_for_generation(
    root: &Path,
    wanted: u64,
    deadline: Duration,
    child: &mut SupervisedChild,
) -> Result<u64, String> {
    let started = Instant::now();
    loop {
        if let Some(epoch) = read_current_epoch(root)
            && epoch >= wanted
        {
            return Ok(epoch);
        }
        if let Ok(Some(status)) = child.child.try_wait() {
            return Err(format!("child exited early with {status}"));
        }
        if started.elapsed() > deadline {
            return Err(format!(
                "deadline elapsed with no committed generation >= {wanted} under {}",
                root.display()
            ));
        }
        std::thread::sleep(POLL_INTERVAL);
    }
}

/// Read the committed epoch from a `local` checkpoint root.
///
/// The manifest is `<root>/<run-digest>/CURRENT`, a single JSON object with an
/// `epoch` field. A root with several run digests reports the highest epoch:
/// each row uses a private root, so more than one digest means the row itself
/// wrote more than one run identity.
fn read_current_epoch(root: &Path) -> Option<u64> {
    let entries = std::fs::read_dir(root).ok()?;
    let mut highest = None;
    for entry in entries.flatten() {
        let manifest = entry.path().join("CURRENT");
        let Ok(bytes) = std::fs::read(&manifest) else {
            continue;
        };
        let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
            continue;
        };
        if let Some(epoch) = value["epoch"].as_u64() {
            highest = Some(highest.map_or(epoch, |current: u64| current.max(epoch)));
        }
    }
    highest
}

fn collect_files(root: &Path, directory: &Path, found: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_files(root, &path, found);
        } else if let Ok(relative) = path.strip_prefix(root) {
            found.push(relative.to_path_buf());
        }
    }
}

/// A child `aiperf` process that is always joined, even when a row panics.
struct SupervisedChild {
    child: Child,
    stdout: Option<std::thread::JoinHandle<String>>,
    stderr: Option<std::thread::JoinHandle<String>>,
}

impl SupervisedChild {
    fn spawn(args: &[String]) -> Result<Self, ProductError> {
        let mut child = Command::new(product_binary())
            .args(args)
            .env("HF_HUB_OFFLINE", "1")
            .env("TRANSFORMERS_OFFLINE", "1")
            .env("PYTHONUNBUFFERED", "1")
            .env("MALLOC_ARENA_MAX", "2")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| ProductError::Spawn(error.to_string()))?;
        // Drain both pipes on their own threads: a child that fills a 64 KiB
        // pipe buffer while the row is parked on the manifest would otherwise
        // deadlock against a reader that never runs.
        let stdout = child.stdout.take().map(drain);
        let stderr = child.stderr.take().map(drain);
        Ok(Self {
            child,
            stdout,
            stderr,
        })
    }

    /// Deliver SIGINT, escalate to SIGKILL after the grace window, then join.
    fn terminate(mut self, grace: Duration, artifacts: PathBuf) -> ProductRun {
        signal_interrupt(&self.child);
        let started = Instant::now();
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) if started.elapsed() > grace => {
                    let _ = self.child.kill();
                    break;
                }
                Ok(None) => std::thread::sleep(POLL_INTERVAL),
                Err(_) => break,
            }
        }
        self.join(artifacts)
    }

    fn join(mut self, artifacts: PathBuf) -> ProductRun {
        let status = self.child.wait().ok();
        let stdout = self.stdout.take().map(join_drain).unwrap_or_default();
        let stderr = self.stderr.take().map(join_drain).unwrap_or_default();
        ProductRun {
            exit_code: status.as_ref().and_then(ExitStatus::code).unwrap_or(-1),
            stdout,
            stderr,
            artifacts,
        }
    }
}

impl Drop for SupervisedChild {
    fn drop(&mut self) {
        // Never leak a child, even when a row panicked mid-run.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn drain<R: Read + Send + 'static>(mut reader: R) -> std::thread::JoinHandle<String> {
    std::thread::spawn(move || {
        let mut buffer = Vec::new();
        let _ = reader.read_to_end(&mut buffer);
        String::from_utf8_lossy(&buffer).into_owned()
    })
}

fn join_drain(handle: std::thread::JoinHandle<String>) -> String {
    handle.join().unwrap_or_default()
}

/// Send SIGINT so the product's own handler can flush partial artifacts.
#[cfg(unix)]
fn signal_interrupt(child: &Child) {
    // SAFETY: `kill(2)` with a pid this process owns and a valid signal number.
    // The child is still referenced here, so the pid cannot have been reaped
    // and reused.
    unsafe {
        libc::kill(child.id() as libc::pid_t, libc::SIGINT);
    }
}

/// Non-unix stub: the caller falls through to the hard `kill()` escalation.
#[cfg(not(unix))]
fn signal_interrupt(_child: &Child) {}

/// Two committed source partitions whose sessions span the boundary.
///
/// The bytes are opaque to this harness on purpose: the row's job is to prove
/// they never reach an artifact, not to reimplement the format's decoder.
const PARTITION_A: &[u8] = b"aiperf-streaming-fixture-partition-a-SOURCESECRET\n";
const PARTITION_B: &[u8] = b"aiperf-streaming-fixture-partition-b-SOURCESECRET\n";

/// Byte sequences that must never appear in any artifact or child stream.
pub const LEAK_NEEDLES: &[&str] = &["SOURCESECRET"];
