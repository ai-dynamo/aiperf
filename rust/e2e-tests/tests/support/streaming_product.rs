// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Server and cellular product fixture for native streaming shadow replay.
//!
//! Owns child processes, ports, scratch trees, and bounded artifact readers
//! through RAII. It never searches `target/`, never sleeps for correctness, and
//! always joins children.
//!
//! The V4A normative Config-v2 fixtures are reused verbatim; this module adds
//! only the transport and cellular overlays — the endpoint URL a row targets
//! and the `runtime.cells` count it runs under.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::common::{self, MockServer, RunResult};

/// The transport a row exercises against the in-repo mock server.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamingTransport {
    /// HTTP/1 plus SSE.
    Http,
    /// gRPC (KServe OIP v2).
    Grpc,
}

/// The stream source a row exercises.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamingSourceKind {
    /// Local filesystem partitions.
    Local,
}

/// The execution topology a row exercises.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamingTopology {
    /// One process.
    SingleProcess,
    /// One controller plus `cells` cell processes.
    Cellular {
        /// Number of cell processes the controller partitions across.
        cells: u32,
    },
}

/// One parameterized server/cellular row.
#[derive(Clone, Debug)]
pub struct StreamingServerCase {
    /// Row name, used in failure messages and artifact subdirectories.
    pub name: &'static str,
    /// Transport overlay.
    pub transport: StreamingTransport,
    /// Source overlay.
    pub source: StreamingSourceKind,
    /// Topology overlay.
    pub topology: StreamingTopology,
    /// Expected public status: `complete`, `degraded`, `export_incomplete`, or
    /// `failed`.
    pub expected_status: &'static str,
}

/// RAII owner of the mock target, the scratch tree, and one row's artifacts.
///
/// `MockServer` owns its runtime and shuts the accept loop down on drop;
/// `tempfile::TempDir` removes the scratch tree. There is no separate join
/// step because every invocation this harness makes is a bounded, joined
/// `RunResult` — no child outlives its call.
pub struct StreamingServerHarness {
    mock: MockServer,
    scratch: tempfile::TempDir,
    source_root: PathBuf,
    checkpoint_root: PathBuf,
    artifact_root: PathBuf,
    case: StreamingServerCase,
}

impl StreamingServerHarness {
    /// Launch the mock target the case requires and prepare its scratch tree.
    ///
    /// `MockServer` binds `127.0.0.1:0` and polls `/health` (HTTP) or a raw TCP
    /// accept (gRPC) before returning, so readiness is a durable predicate and
    /// not a sleep.
    pub fn start(case: &StreamingServerCase) -> Self {
        let mock = match case.transport {
            StreamingTransport::Http => MockServer::start(),
            StreamingTransport::Grpc => {
                let mut cfg = aiperf_mock_server::config::MockServerConfig::default();
                cfg.fast = true;
                cfg.workers = 8;
                cfg.no_tokenizer = true;
                MockServer::start_with_grpc(cfg)
            }
        };
        let scratch = tempfile::TempDir::new().expect("create streaming scratch directory");
        let source_root = scratch.path().join("source");
        let checkpoint_root = scratch.path().join("checkpoint");
        let artifact_root = scratch.path().join("artifacts");
        for directory in [&source_root, &checkpoint_root, &artifact_root] {
            std::fs::create_dir_all(directory).expect("create streaming scratch subdirectory");
        }
        let harness = Self {
            mock,
            scratch,
            source_root,
            checkpoint_root,
            artifact_root,
            case: case.clone(),
        };
        harness.publish_partition("000-a.parquet", PARTITION_A);
        harness
    }

    /// The mock target's HTTP base URL.
    pub fn http_url(&self) -> &str {
        &self.mock.url
    }

    /// The mock target's gRPC URL, when the row enabled the gRPC listener.
    pub fn grpc_url(&self) -> Option<&str> {
        self.mock.grpc_url.as_deref()
    }

    /// The endpoint URL this row's transport overlay selects.
    pub fn endpoint_url(&self) -> String {
        match self.case.transport {
            StreamingTransport::Http => self.mock.url.clone(),
            StreamingTransport::Grpc => self
                .mock
                .grpc_url
                .clone()
                .expect("grpc row started the grpc listener"),
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

    /// Publish one partition into the source root by atomic rename.
    pub fn publish_partition(&self, name: &str, bytes: &[u8]) {
        let staging = self.scratch.path().join(format!(".staging-{name}"));
        std::fs::write(&staging, bytes).expect("stage source partition");
        std::fs::rename(&staging, self.source_root.join(name)).expect("publish source partition");
    }

    /// Render the shared V4A fixture with this row's transport and topology
    /// overlays applied.
    ///
    /// The token set is exactly `$ARTIFACT_DIR`, `$SOURCE_ROOT`,
    /// `$CHECKPOINT_ROOT`, and `$ENDPOINT_URL`; an unsubstituted token is a
    /// fixture defect and panics rather than reaching the product as a literal.
    pub fn render_config(&self, fixture: &str) -> PathBuf {
        let source = fixture_root().join(fixture);
        let yaml = std::fs::read_to_string(&source)
            .unwrap_or_else(|e| panic!("read fixture {}: {e}", source.display()));
        let mut rendered = yaml
            .replace("$ARTIFACT_DIR", &self.artifact_root.display().to_string())
            .replace("$SOURCE_ROOT", &self.source_root.display().to_string())
            .replace(
                "$CHECKPOINT_ROOT",
                &self.checkpoint_root.display().to_string(),
            )
            .replace("$ENDPOINT_URL", &self.endpoint_url());
        if let StreamingTopology::Cellular { cells } = self.case.topology {
            rendered = format!("runtime:\n  cells: {cells}\n{rendered}");
        }
        assert!(
            !rendered.contains('$'),
            "fixture {fixture} has an unsubstituted token"
        );
        let path = self.scratch.path().join("benchmark.yaml");
        std::fs::write(&path, rendered).expect("write rendered streaming config");
        path
    }

    /// Run `aiperf profile --config <rendered>` for this row.
    pub fn profile(&self, fixture: &str) -> StreamingServerOutcome {
        // The rendered config carries its own artifact directory, so the run is
        // driven through a direct binary invocation rather than
        // `AIPerfHarness::run`, which appends a second `--artifact-dir` flag.
        let config = self.render_config(fixture);
        let run = run_config(&config, self.artifact_root.as_path());
        StreamingServerOutcome {
            run,
            endpoint_issues: self.observed_endpoint_requests(),
            artifacts: self.artifact_root.clone(),
        }
    }

    /// Requests the mock target actually observed, from its own scrape.
    ///
    /// The evidence for "nothing was issued" has to come from the *server*, not
    /// from the client's own report: a client that refused before issuing and a
    /// client that issued and discarded the result look identical from the
    /// client side.
    fn observed_endpoint_requests(&self) -> u64 {
        let Some(body) = scrape(self.mock.port, "/metrics") else {
            return 0;
        };
        body.lines()
            .filter(|line| !line.starts_with('#'))
            .filter(|line| line.contains("requests_total"))
            .filter_map(|line| line.rsplit(' ').next())
            .filter_map(|value| value.trim().parse::<f64>().ok())
            .map(|value| value as u64)
            .sum()
    }
}

/// Fetch one loopback path with a minimal blocking HTTP/1.0 GET.
///
/// Deliberately not `reqwest`: the e2e crate links `reqwest` without its
/// `blocking` feature, and a loopback scrape must never consult ambient proxy
/// settings. Twenty lines of `std::net` avoids both problems.
fn scrape(port: u16, path: &str) -> Option<String> {
    use std::io::{Read, Write};
    let mut stream = std::net::TcpStream::connect(("127.0.0.1", port)).ok()?;
    stream
        .write_all(format!("GET {path} HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n").as_bytes())
        .ok()?;
    let mut response = String::new();
    stream.read_to_string(&mut response).ok()?;
    let body_start = response.find("\r\n\r\n")? + 4;
    Some(response[body_start..].to_owned())
}

/// Directory holding the committed Config-v2 streaming fixtures.
pub fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/streaming")
}

/// Invoke the pinned product binary on one rendered Config-v2 document.
fn run_config(config: &Path, artifact_dir: &Path) -> RunResult {
    let output = std::process::Command::new(common::exec_binary())
        .args([
            "profile",
            "--config",
            &config.display().to_string(),
        ])
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("PYTHONUNBUFFERED", "1")
        .env("MALLOC_ARENA_MAX", "2")
        .stdin(std::process::Stdio::null())
        .output()
        .expect("spawn aiperf profile");
    RunResult {
        exit_code: output.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        artifacts: common::ArtifactReader {
            dir: artifact_dir.to_path_buf(),
        },
    }
}

/// The observable result of one server/cellular row.
pub struct StreamingServerOutcome {
    run: RunResult,
    endpoint_issues: u64,
    artifacts: PathBuf,
}

impl StreamingServerOutcome {
    /// Both child streams, for diagnostics.
    pub fn combined_output(&self) -> String {
        format!("{}\n{}", self.run.stdout, self.run.stderr)
    }

    /// Whether the run exited successfully.
    pub fn success(&self) -> bool {
        self.run.success()
    }

    /// Public execution status.
    pub fn public_status(&self) -> String {
        self.run.artifacts.json()["streaming"]["status"]
            .as_str()
            .unwrap_or(if self.success() { "complete" } else { "failed" })
            .to_owned()
    }

    /// Sorted stable action ids paired with their content digests.
    pub fn logical_membership(&self) -> Vec<(String, String)> {
        let mut membership: Vec<(String, String)> = self
            .run
            .artifacts
            .jsonl()
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
        membership.sort();
        membership
    }

    /// The final report's metric row order, which must be topology-invariant.
    pub fn final_report_order(&self) -> Vec<String> {
        self.run.artifacts.json()["records"]
            .as_object()
            .map(|rows| rows.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Prepare acknowledgements the run committed.
    pub fn prepare_acknowledgements(&self) -> u64 {
        self.run.artifacts.json()["streaming"]["prepare_acknowledgements"]
            .as_u64()
            .unwrap_or(0)
    }

    /// Capacity releases the run committed.
    pub fn releases(&self) -> u64 {
        self.run.artifacts.json()["streaming"]["releases"]
            .as_u64()
            .unwrap_or(0)
    }

    /// Requests the mock target actually observed.
    pub fn endpoint_issues(&self) -> u64 {
        self.endpoint_issues
    }

    /// Every regular file under the artifact root, relative and sorted.
    pub fn artifact_files(&self) -> Vec<PathBuf> {
        let mut found = Vec::new();
        collect_files(&self.artifacts, &self.artifacts, &mut found);
        found.sort();
        found
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

/// Every transport × topology row the streaming server matrix covers.
pub fn server_matrix() -> Vec<StreamingServerCase> {
    vec![
        StreamingServerCase {
            name: "http_single_process",
            transport: StreamingTransport::Http,
            source: StreamingSourceKind::Local,
            topology: StreamingTopology::SingleProcess,
            expected_status: "failed",
        },
        StreamingServerCase {
            name: "grpc_single_process",
            transport: StreamingTransport::Grpc,
            source: StreamingSourceKind::Local,
            topology: StreamingTopology::SingleProcess,
            expected_status: "failed",
        },
        StreamingServerCase {
            name: "http_cellular_two_cells",
            transport: StreamingTransport::Http,
            source: StreamingSourceKind::Local,
            topology: StreamingTopology::Cellular { cells: 2 },
            expected_status: "failed",
        },
    ]
}

/// Unused today; retained so the matrix can key rows by name.
pub type CaseIndex = HashMap<&'static str, StreamingServerCase>;

/// One committed source partition. Opaque to this harness by design.
const PARTITION_A: &[u8] = b"aiperf-streaming-fixture-partition-a-SOURCESECRET\n";

/// Byte sequences that must never appear in an artifact or a child stream.
pub const LEAK_NEEDLES: &[&str] = &["SOURCESECRET"];
