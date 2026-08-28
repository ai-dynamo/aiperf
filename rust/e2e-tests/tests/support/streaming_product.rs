// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Server and cellular product fixture for native streaming shadow replay.
//!
//! The socket-free twin of this module lives in the dry-run suite
//! (`rust/dry-run-tests/tests/support/streaming_product.rs`) and owns the
//! `dry_run` rows. This module owns the rows that need a real listener: HTTP
//! and gRPC against the in-repo `aiperf-mock-server`, and the cellular
//! controller/cell topology.
//!
//! The same three rules apply here:
//!
//! - **No implicit binary search.** Every invocation runs `common::exec_binary`,
//!   which reads `AIPERF_E2E_BIN` and panics when it is unset, so a row can
//!   never silently measure a stale `target/` artifact.
//! - **No sleep for correctness.** `MockServer::start_with*` already polls
//!   `/health` (HTTP) or a raw TCP accept (gRPC) before returning, and the child
//!   is joined on its own exit rather than after a fixed delay. The only
//!   `sleep` in this module is the cadence inside a bounded predicate poll.
//! - **No leaked child.** [`SupervisedChild`] delivers SIGINT, escalates to
//!   SIGKILL after a grace window, and joins in `Drop`, so a panicking row still
//!   reaps its process; [`StreamingServerHarness`] owns the mock, every child,
//!   and the scratch tree, and drops them in that order.
//!
//! Fixtures under `tests/fixtures/streaming/` are rendered with an explicit
//! four-token set — `$ARTIFACT_DIR`, `$SOURCE_ROOT`, `$CHECKPOINT_ROOT`, and
//! `$ENDPOINT_URL`. An unsubstituted token is a fixture defect and panics rather
//! than reaching the product as a literal path.

#![allow(dead_code)]

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use aiperf_mock_server::config::MockServerConfig;

use crate::common::{MockServer, exec_binary};

/// Longest a row lets one invocation run before failing it.
const RUN_DEADLINE: Duration = Duration::from_secs(120);

/// Longest a row waits for a child to exit after SIGINT before SIGKILL.
const TERMINATE_GRACE: Duration = Duration::from_secs(10);

/// Interval between reads inside a bounded predicate poll.
///
/// A poll cadence, not a correctness delay: each iteration re-reads durable
/// process state whose change *is* the event being waited on.
const POLL_INTERVAL: Duration = Duration::from_millis(10);

/// The wire transport a row exercises.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamingTransport {
    /// HTTP/1 + SSE against the in-repo mock server.
    Http,
    /// KServe OIP v2 gRPC against the in-repo mock server.
    Grpc,
}

impl StreamingTransport {
    /// The committed fixture this transport authors.
    const fn fixture(self) -> &'static str {
        match self {
            Self::Http => "http_local_conversation.yaml",
            Self::Grpc => "grpc_local_conversation.yaml",
        }
    }
}

/// The execution topology a row exercises.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamingTopology {
    /// One `aiperf` process.
    SingleProcess,
    /// One controller plus `cells` cell processes, selected with `--cells`.
    Cellular {
        /// Number of cell processes the controller partitions across.
        cells: u32,
    },
}

/// One parameterized server/cellular row.
#[derive(Clone, Debug)]
pub struct StreamingServerCase {
    /// Row name, used in assertion messages.
    pub name: &'static str,
    /// Selected wire transport.
    pub transport: StreamingTransport,
    /// Selected execution topology.
    pub topology: StreamingTopology,
}

impl StreamingServerCase {
    /// A single-process row over the named transport.
    #[must_use]
    pub const fn single_process(name: &'static str, transport: StreamingTransport) -> Self {
        Self {
            name,
            transport,
            topology: StreamingTopology::SingleProcess,
        }
    }

    /// A cellular row over the named transport and cell count.
    #[must_use]
    pub const fn cellular(name: &'static str, transport: StreamingTransport, cells: u32) -> Self {
        Self {
            name,
            transport,
            topology: StreamingTopology::Cellular { cells },
        }
    }
}

/// Directory holding the committed Config-v2 streaming fixtures.
#[must_use]
pub fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/streaming")
}

/// RAII owner of the mock target, every child process, and the scratch tree for
/// one row.
///
/// Field order is drop order: the scratch tree is declared last so it is removed
/// only after every child that could still be writing into it has been reaped.
pub struct StreamingServerHarness {
    case: StreamingServerCase,
    mock: MockServer,
    source_root: PathBuf,
    checkpoint_root: PathBuf,
    artifact_root: PathBuf,
    scratch: tempfile::TempDir,
}

impl StreamingServerHarness {
    /// Start the mock target this case needs and prepare its scratch tree.
    ///
    /// Request capture is enabled so an HTTP row can inspect the exact bodies
    /// that reached the endpoint; the transport-neutral *count* comes from the
    /// mock's own Prometheus family, which both listeners record into.
    #[must_use]
    pub fn start(case: StreamingServerCase) -> Self {
        let cfg = MockServerConfig {
            fast: true,
            workers: 8,
            no_tokenizer: true,
            request_capture_capacity: 256,
            ..MockServerConfig::default()
        };
        let mock = match case.transport {
            StreamingTransport::Http => MockServer::start_with(cfg),
            StreamingTransport::Grpc => MockServer::start_with_grpc(cfg),
        };

        let scratch = tempfile::tempdir().expect("create streaming scratch directory");
        let source_root = scratch.path().join("source");
        let checkpoint_root = scratch.path().join("checkpoint");
        let artifact_root = scratch.path().join("artifacts");
        for directory in [&source_root, &checkpoint_root, &artifact_root] {
            std::fs::create_dir_all(directory).expect("create streaming scratch subdirectory");
        }

        let harness = Self {
            case,
            mock,
            source_root,
            checkpoint_root,
            artifact_root,
            scratch,
        };
        harness.write_partition("000-a.parquet", PARTITION_A);
        harness.write_partition("001-b.parquet", PARTITION_B);
        harness
    }

    /// The mock target this row benchmarks against.
    #[must_use]
    pub fn mock(&self) -> &MockServer {
        &self.mock
    }

    /// The acquired source root for this row.
    #[must_use]
    pub fn source_root(&self) -> &Path {
        &self.source_root
    }

    /// The durable checkpoint root for this row.
    #[must_use]
    pub fn checkpoint_root(&self) -> &Path {
        &self.checkpoint_root
    }

    /// Publish one partition into the source root by atomic rename.
    ///
    /// Rename, not create-then-write: the local source accepts publication by
    /// rename, so a partially written file must never be visible under its
    /// final name.
    pub fn write_partition(&self, name: &str, bytes: &[u8]) {
        let staging = self.scratch.path().join(format!(".staging-{name}"));
        std::fs::write(&staging, bytes).expect("stage source partition");
        std::fs::rename(&staging, self.source_root.join(name)).expect("publish source partition");
    }

    /// The endpoint URL the fixture is rendered against.
    fn endpoint_url(&self) -> String {
        match self.case.transport {
            StreamingTransport::Http => self.mock.url.clone(),
            StreamingTransport::Grpc => self
                .mock
                .grpc_url
                .clone()
                .expect("a grpc row starts the grpc listener"),
        }
    }

    /// Render the committed fixture into this row's scratch tree.
    fn render_config(&self) -> PathBuf {
        let source = fixture_root().join(self.case.transport.fixture());
        let yaml = std::fs::read_to_string(&source)
            .unwrap_or_else(|error| panic!("read fixture {}: {error}", source.display()));
        let rendered = yaml
            .replace("$ARTIFACT_DIR", &self.artifact_root.display().to_string())
            .replace("$SOURCE_ROOT", &self.source_root.display().to_string())
            .replace(
                "$CHECKPOINT_ROOT",
                &self.checkpoint_root.display().to_string(),
            )
            .replace("$ENDPOINT_URL", &self.endpoint_url());
        assert!(
            !rendered.contains('$'),
            "fixture {} has an unsubstituted token",
            self.case.transport.fixture()
        );
        let path = self.scratch.path().join("benchmark.yaml");
        std::fs::write(&path, rendered).expect("write rendered streaming config");
        path
    }

    /// Run `aiperf config validate` over the rendered fixture.
    #[must_use]
    pub fn validate(&self) -> StreamingServerOutcome {
        let config = self.render_config();
        self.invoke(vec![
            "config".to_owned(),
            "validate".to_owned(),
            config.display().to_string(),
        ])
    }

    /// Run `aiperf profile --config <rendered>` under this row's topology.
    #[must_use]
    pub fn profile(&self) -> StreamingServerOutcome {
        let config = self.render_config();
        let mut args = vec![
            "profile".to_owned(),
            "--config".to_owned(),
            config.display().to_string(),
        ];
        if let StreamingTopology::Cellular { cells } = self.case.topology {
            args.push("--cells".to_owned());
            args.push(cells.to_string());
        }
        self.invoke(args)
    }

    fn invoke(&self, args: Vec<String>) -> StreamingServerOutcome {
        let mut child = SupervisedChild::spawn(&args);
        let started = Instant::now();
        loop {
            match child.child.try_wait().expect("poll aiperf child") {
                Some(_) => break,
                None if started.elapsed() > RUN_DEADLINE => {
                    return self.finish(child.terminate(TERMINATE_GRACE));
                }
                None => std::thread::sleep(POLL_INTERVAL),
            }
        }
        self.finish(child.join())
    }

    fn finish(&self, completed: CompletedChild) -> StreamingServerOutcome {
        StreamingServerOutcome {
            exit_code: completed.exit_code,
            stdout: completed.stdout,
            stderr: completed.stderr,
            artifacts: self.artifact_root.clone(),
        }
    }

    /// Requests the mock target accepted, counted from its own metric family.
    ///
    /// Read after the child is fully reaped, so no request can still be in
    /// flight when the count is taken. The HTTP and gRPC handlers both record
    /// into `aiperf_mock_requests_by_model_total`, so this is the one
    /// transport-neutral endpoint-issue count available; the axum request
    /// capture store, by contrast, sees only the HTTP listener.
    pub async fn endpoint_issues(&self) -> u64 {
        // `no_proxy`: an ambient `HTTP_PROXY` would otherwise route this
        // loopback scrape through a proxy that answers 405.
        let body = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("build metrics scrape client")
            .get(format!("{}/metrics", self.mock.url))
            .send()
            .await
            .expect("scrape mock metrics")
            .text()
            .await
            .expect("read mock metrics body");
        sum_counter(&body, "aiperf_mock_requests_by_model_total")
    }

    /// Inference request bodies the mock retained, for the HTTP rows.
    ///
    /// The capture middleware sits in front of every route, including the
    /// harness's own `/metrics` scrape and the startup `/health` probe, so
    /// control routes are excluded by prefix — only a request the benchmark
    /// itself issued counts as an endpoint effect.
    #[must_use]
    pub fn captured_inference_requests(&self) -> usize {
        self.mock
            .state
            .request_captures()
            .iter()
            .filter(|capture| {
                !matches!(
                    capture.route.as_str(),
                    "/metrics" | "/health" | "/accuracy" | "/models" | "/v1/models"
                )
            })
            .count()
    }
}

/// Sum every sample of one Prometheus counter family in a scrape body.
///
/// The family is label-partitioned by model and endpoint, so a row that cares
/// only about "did anything reach the endpoint" must add the children rather
/// than read one series.
fn sum_counter(body: &str, family: &str) -> u64 {
    body.lines()
        .filter(|line| !line.starts_with('#'))
        .filter(|line| line.starts_with(family))
        .filter_map(|line| line.rsplit_once(' '))
        .filter_map(|(_, value)| value.trim().parse::<f64>().ok())
        .map(|value| value as u64)
        .sum()
}

/// The observable result of one server or cellular row.
pub struct StreamingServerOutcome {
    /// Process exit code, or `-1` when terminated by signal.
    pub exit_code: i32,
    /// Complete child stdout.
    pub stdout: String,
    /// Complete child stderr.
    pub stderr: String,
    /// Artifact root this invocation was pointed at.
    pub artifacts: PathBuf,
}

impl StreamingServerOutcome {
    /// Whether the child exited successfully.
    #[must_use]
    pub fn success(&self) -> bool {
        self.exit_code == 0
    }

    /// Both child streams, for diagnostics and leak scanning.
    #[must_use]
    pub fn combined_output(&self) -> String {
        format!("{}\n{}", self.stdout, self.stderr)
    }

    /// Assert the run refused and its refusal names every expected fragment.
    pub fn assert_refused_naming(&self, case: &str, fragments: &[&str]) {
        assert!(
            !self.success(),
            "{case}: expected a refusal, but the run exited 0:\n{}",
            self.combined_output()
        );
        let output = self.combined_output();
        for fragment in fragments {
            assert!(
                output.contains(fragment),
                "{case}: refusal must name {fragment:?}:\n{output}"
            );
        }
    }

    /// Artifact paths relative to the artifact root, sorted.
    #[must_use]
    pub fn artifact_files(&self) -> Vec<PathBuf> {
        let mut found = Vec::new();
        collect_files(&self.artifacts, &self.artifacts, &mut found);
        found.sort();
        found
    }

    /// Artifact paths excluding the run's own log tree.
    ///
    /// The logger is installed before the registry resolves anything, so a log
    /// file is present even for a run refused at capability agreement. It is
    /// not a measurement artifact and does not count as an execution effect.
    #[must_use]
    pub fn measurement_artifacts(&self) -> Vec<PathBuf> {
        self.artifact_files()
            .into_iter()
            .filter(|path| !path.starts_with("logs"))
            .collect()
    }

    /// The refusal message with volatile substrings removed.
    ///
    /// Timestamps, ports, and scratch paths differ between two rows that must
    /// nonetheless agree on *why* the product refused. Stripping them makes the
    /// transport-invariance comparison meaningful rather than trivially false.
    ///
    /// The timestamp strip is not cosmetic: the tracing prefix appears twice on
    /// a cellular line (once from the controller, once from the forwarded cell
    /// record), so without it two invocations of the *same* topology already
    /// compare unequal and the assertion proves nothing.
    #[must_use]
    pub fn stable_refusal(&self) -> String {
        static TIMESTAMP: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        // Wall-clock prefix emitted by the tracing subscriber: `HH:MM:SS.mmm`.
        let timestamp = TIMESTAMP.get_or_init(|| {
            #[allow(clippy::expect_used)]
            regex::Regex::new(r"\d{2}:\d{2}:\d{2}\.\d{3}\s*").expect("static timestamp pattern")
        });
        self.combined_output()
            .lines()
            .filter(|line| line.contains("Native AIPerf run failed") || line.contains("aiperf:"))
            .map(|line| match line.find("failed: ") {
                Some(index) => line[index + "failed: ".len()..].to_owned(),
                None => line.trim().to_owned(),
            })
            .map(|line| timestamp.replace_all(&line, "").into_owned())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Assert no raw source byte and no authored secret reached any artifact.
    pub fn assert_no_raw_or_secret_leak(&self, needles: &[&str]) {
        let mut haystack = self.combined_output();
        for file in self.artifact_files() {
            if let Ok(bytes) = std::fs::read(self.artifacts.join(&file)) {
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

/// One completed child invocation.
struct CompletedChild {
    exit_code: i32,
    stdout: String,
    stderr: String,
}

/// A child `aiperf` process that is always joined, even when a row panics.
struct SupervisedChild {
    child: Child,
    stdout: Option<std::thread::JoinHandle<String>>,
    stderr: Option<std::thread::JoinHandle<String>>,
}

impl SupervisedChild {
    fn spawn(args: &[String]) -> Self {
        let mut child = Command::new(exec_binary())
            .args(args)
            .env("HF_HUB_OFFLINE", "1")
            .env("TRANSFORMERS_OFFLINE", "1")
            .env("PYTHONUNBUFFERED", "1")
            .env("MALLOC_ARENA_MAX", "2")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn aiperf");
        // Drain both pipes on their own threads: a child that fills a 64 KiB
        // pipe buffer while the row is parked would otherwise deadlock against
        // a reader that never runs.
        let stdout = child.stdout.take().map(drain);
        let stderr = child.stderr.take().map(drain);
        Self {
            child,
            stdout,
            stderr,
        }
    }

    /// Deliver SIGINT, escalate to SIGKILL after the grace window, then join.
    fn terminate(mut self, grace: Duration) -> CompletedChild {
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
        self.join()
    }

    fn join(mut self) -> CompletedChild {
        let status = self.child.wait().ok();
        let stdout = self.stdout.take().map(join_drain).unwrap_or_default();
        let stderr = self.stderr.take().map(join_drain).unwrap_or_default();
        CompletedChild {
            exit_code: status.as_ref().and_then(ExitStatus::code).unwrap_or(-1),
            stdout,
            stderr,
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
    use nix::sys::signal::{Signal, kill};
    use nix::unistd::Pid;
    let _ = kill(Pid::from_raw(child.id() as i32), Signal::SIGINT);
}

/// Non-unix stub: the caller falls through to the hard `kill()` escalation.
#[cfg(not(unix))]
fn signal_interrupt(_child: &Child) {}

/// Two committed source partitions whose sessions span the boundary.
///
/// The bytes are opaque to this harness on purpose: the row's job is to prove
/// they never reach an artifact or the wire, not to reimplement the format's
/// decoder.
const PARTITION_A: &[u8] = b"aiperf-streaming-fixture-partition-a-SOURCESECRET\n";
const PARTITION_B: &[u8] = b"aiperf-streaming-fixture-partition-b-SOURCESECRET\n";

/// Byte sequences that must never appear in any artifact or child stream.
pub const LEAK_NEEDLES: &[&str] = &["SOURCESECRET"];
