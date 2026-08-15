// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Contract coverage for the worker-local persistent tool sandbox.

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::future::pending;
use std::rc::Rc;

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};

use aiperf_runtime::clock::{Clock, RealClock};
use aiperf_runtime::graph::tools::{
    LocalProcessRequest, LocalSessionSandbox, ProcessSession, ProcessSpawner, ToolSandbox,
    ToolSandboxError, WorkspaceSpec,
};

enum ReadStep {
    Bytes(Vec<u8>),
    Pending,
}

struct FakeProcess {
    output_prefix: Vec<u8>,
    reads: VecDeque<ReadStep>,
    termination_count: Rc<Cell<u8>>,
}

impl FakeProcess {
    fn new(output_prefix: impl Into<Vec<u8>>, reads: impl IntoIterator<Item = ReadStep>) -> Self {
        Self {
            output_prefix: output_prefix.into(),
            reads: reads.into_iter().collect(),
            termination_count: Rc::new(Cell::new(0)),
        }
    }

    fn with_termination_count(
        output_prefix: impl Into<Vec<u8>>,
        reads: impl IntoIterator<Item = ReadStep>,
        termination_count: Rc<Cell<u8>>,
    ) -> Self {
        Self {
            output_prefix: output_prefix.into(),
            reads: reads.into_iter().collect(),
            termination_count,
        }
    }
}

#[async_trait(?Send)]
impl ProcessSession for FakeProcess {
    async fn write_all(&mut self, command: &[u8]) -> Result<(), ToolSandboxError> {
        let command = std::str::from_utf8(command).map_err(|error| {
            ToolSandboxError::new(format!("fake command is not UTF-8: {error}"))
        })?;
        let marker = command
            .split("\\0aiperf-terminal:")
            .nth(1)
            .and_then(|tail| tail.split(":%d\\0").next())
            .ok_or_else(|| ToolSandboxError::new("fake cannot find terminal marker"))?;
        let mut frame = self.output_prefix.clone();
        frame.extend_from_slice(b"\0aiperf-terminal:");
        frame.extend_from_slice(marker.as_bytes());
        frame.extend_from_slice(b":0\0");
        self.reads.push_back(ReadStep::Bytes(frame));
        Ok(())
    }

    async fn read(&mut self, output: &mut BytesMut) -> Result<usize, ToolSandboxError> {
        match self.reads.pop_front().unwrap_or(ReadStep::Pending) {
            ReadStep::Bytes(bytes) => {
                let count = bytes.len();
                output.extend_from_slice(&bytes);
                Ok(count)
            }
            ReadStep::Pending => pending().await,
        }
    }

    async fn terminate(&mut self) -> Result<(), ToolSandboxError> {
        self.termination_count
            .set(self.termination_count.get().saturating_add(1));
        Ok(())
    }

    async fn kill(&mut self) -> Result<(), ToolSandboxError> {
        Ok(())
    }

    async fn wait(&mut self) -> Result<(), ToolSandboxError> {
        Ok(())
    }
}

struct FakeSpawner {
    processes: RefCell<VecDeque<Box<dyn ProcessSession>>>,
    requests: RefCell<Vec<LocalProcessRequest>>,
}

impl FakeSpawner {
    fn new(processes: impl IntoIterator<Item = Box<dyn ProcessSession>>) -> Self {
        Self {
            processes: RefCell::new(processes.into_iter().collect()),
            requests: RefCell::new(Vec::new()),
        }
    }
}

#[async_trait(?Send)]
impl ProcessSpawner for FakeSpawner {
    async fn spawn(
        &self,
        request: LocalProcessRequest,
    ) -> Result<Box<dyn ProcessSession>, ToolSandboxError> {
        self.requests.borrow_mut().push(request);
        self.processes
            .borrow_mut()
            .pop_front()
            .ok_or_else(|| ToolSandboxError::new("fake has no process left"))
    }
}

fn workspace() -> WorkspaceSpec {
    WorkspaceSpec {
        files: Vec::new(),
        workdir: ".".into(),
        interpreter: vec!["bash".into(), "-c".into()],
        mount_workspace: true,
        command_timeout_ns: 1_000_000,
    }
}

fn real_workspace(workdir: &str) -> WorkspaceSpec {
    WorkspaceSpec {
        workdir: workdir.into(),
        ..workspace()
    }
}

fn sandbox(spawner: Rc<dyn ProcessSpawner>) -> LocalSessionSandbox {
    let clock: Rc<dyn Clock> = RealClock::new();
    LocalSessionSandbox::new(workspace(), clock, spawner, 1024)
}

#[tokio::test(flavor = "current_thread")]
async fn timeout_kills_descendants_recycles_and_next_command_has_no_stale_output() {
    // This catches a timeout path that leaves the old session readable, causing
    // a following command to observe bytes emitted by its terminated predecessor.
    let termination_count = Rc::new(Cell::new(0));
    let spawner = Rc::new(FakeSpawner::new([
        Box::new(FakeProcess::with_termination_count(
            b"",
            [
                ReadStep::Bytes(b"first command output".to_vec()),
                ReadStep::Pending,
            ],
            termination_count.clone(),
        )) as Box<dyn ProcessSession>,
        Box::new(FakeProcess::new(b"second command output", [])),
    ]));
    let sandbox = sandbox(spawner.clone());

    sandbox.open().await.expect("first session opens");
    let first = sandbox
        .run("sleep forever", Some(10_000_000))
        .await
        .expect("timeout recovers its session");
    let second = sandbox
        .run("printf second", None)
        .await
        .expect("recycled session accepts the next command");

    assert!(first.is_timed_out);
    assert_eq!(first.output, Bytes::from_static(b"first command output"));
    assert_eq!(second.output, Bytes::from_static(b"second command output"));
    assert_eq!(termination_count.get(), 1);
    assert_eq!(spawner.requests.borrow().len(), 2);
}

#[tokio::test(flavor = "current_thread")]
async fn sentinel_like_output_is_not_terminal_frame() {
    // This catches a delimiter parser that accepts any sentinel-shaped bytes
    // instead of only the unique frame generated for this command.
    let spawner = Rc::new(FakeSpawner::new([Box::new(FakeProcess::new(
        b"before\0aiperf-terminal:not-this-command:0\0after",
        [],
    )) as Box<dyn ProcessSession>]));
    let sandbox = sandbox(spawner);

    let result = sandbox
        .run("printf output", None)
        .await
        .expect("unique terminal frame completes the command");

    assert_eq!(
        result.output,
        Bytes::from_static(b"before\0aiperf-terminal:not-this-command:0\0after")
    );
}

#[tokio::test(flavor = "current_thread")]
async fn persistent_workspace_uses_fresh_interpreters_and_reports_truncation() {
    // This catches a shell loop that loses filesystem state, leaks one command's
    // shell options into the next, or silently drops bytes after its output cap.
    let temporary = tempfile::tempdir().expect("temporary local workspace");
    let clock: Rc<dyn Clock> = RealClock::new();
    let sandbox = LocalSessionSandbox::with_tokio_processes(
        real_workspace(temporary.path().to_string_lossy().as_ref()),
        clock.clone(),
        1024,
    );

    let write = sandbox
        .run(
            "printf value > shared-state; set -o pipefail; false | true",
            Some(1_000_000_000),
        )
        .await
        .expect("first fresh interpreter finishes");
    let read = sandbox
        .run("cat shared-state; false | true", Some(1_000_000_000))
        .await
        .expect("second fresh interpreter sees workspace state");
    sandbox.close().await.expect("session closes idempotently");
    sandbox.close().await.expect("second close is a no-op");
    let bounded = LocalSessionSandbox::with_tokio_processes(
        real_workspace(temporary.path().to_string_lossy().as_ref()),
        clock,
        3,
    );
    let truncated = bounded
        .run("printf 12345", Some(1_000_000_000))
        .await
        .expect("bounded command output completes");
    bounded.close().await.expect("bounded session closes");

    assert_eq!(write.exit_code, 1);
    assert_eq!(read.exit_code, 0);
    assert_eq!(read.output, Bytes::from_static(b"value"));
    assert_eq!(truncated.output, Bytes::from_static(b"123"));
    assert!(truncated.is_output_truncated);
}
