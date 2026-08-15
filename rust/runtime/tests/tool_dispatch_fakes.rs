// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Contract coverage for deterministic environment-aware tool dispatch fakes.

use std::rc::Rc;

use async_trait::async_trait;
use bytes::Bytes;

use aiperf_runtime::graph::driver::TraceIdentity;
use aiperf_runtime::graph::tools::{
    CommandDisposition, EnvironmentToolDispatcher, ToolCommandPolicy, ToolCommandResult,
    ToolDispatchContext, ToolDispatchError, ToolDispatchRequest, ToolDispatcher, ToolSandbox,
    ToolSandboxError, TraceOpenContext, close_trace_preserving_primary,
};

struct AllowAll;

impl ToolCommandPolicy for AllowAll {
    fn evaluate(
        &self,
        _command: &str,
    ) -> Result<CommandDisposition, aiperf_runtime::graph::tools::TraceEnvironmentError> {
        Ok(CommandDisposition::Execute)
    }
}

struct CloseFailsSandbox;

#[async_trait(?Send)]
impl ToolSandbox for CloseFailsSandbox {
    async fn open(&self) -> Result<(), ToolSandboxError> {
        Ok(())
    }

    async fn run(
        &self,
        _command: &str,
        _timeout_ns: Option<u64>,
    ) -> Result<ToolCommandResult, ToolSandboxError> {
        Ok(ToolCommandResult::completed(0, Bytes::new()))
    }

    async fn recycle(&self) -> Result<(), ToolSandboxError> {
        Ok(())
    }

    async fn close(&self) -> Result<(), ToolSandboxError> {
        Err(ToolSandboxError::new("close failed"))
    }
}

struct FakeSandbox {
    events: std::cell::RefCell<Vec<String>>,
    results: std::cell::RefCell<Vec<ToolCommandResult>>,
}

#[tokio::test(flavor = "current_thread")]
async fn primary_dispatch_error_wins_over_a_close_error() {
    // This catches a lifecycle wrapper that hides an actionable command failure
    // behind a secondary cleanup failure.
    let dispatcher = EnvironmentToolDispatcher::new(Rc::new(CloseFailsSandbox), Rc::new(AllowAll));
    let trace = TraceIdentity {
        run_id: "run".into(),
        trajectory_id: "trajectory".into(),
        trace_id: "trace".into(),
    };
    let error = close_trace_preserving_primary(
        &dispatcher,
        &trace,
        Err(ToolDispatchError::new("primary dispatch failed")),
    )
    .await
    .expect_err("primary failure survives cleanup");
    assert_eq!(error.to_string(), "primary dispatch failed");
}

#[async_trait(?Send)]
impl ToolSandbox for FakeSandbox {
    async fn open(&self) -> Result<(), ToolSandboxError> {
        self.events.borrow_mut().push("open".into());
        Ok(())
    }

    async fn run(
        &self,
        command: &str,
        _timeout_ns: Option<u64>,
    ) -> Result<ToolCommandResult, ToolSandboxError> {
        self.events.borrow_mut().push(format!("run:{command}"));
        Ok(self.results.borrow_mut().remove(0))
    }

    async fn recycle(&self) -> Result<(), ToolSandboxError> {
        self.events.borrow_mut().push("recycle".into());
        Ok(())
    }

    async fn close(&self) -> Result<(), ToolSandboxError> {
        self.events.borrow_mut().push("close".into());
        Ok(())
    }
}

#[tokio::test(flavor = "current_thread")]
async fn dispatcher_continues_after_timeout_when_fake_recycles() {
    // This catches a dispatcher that treats a terminal timeout as infrastructure
    // failure, skips recycle, or permits the following command to overtake it.
    let sandbox = Rc::new(FakeSandbox {
        events: std::cell::RefCell::new(Vec::new()),
        results: std::cell::RefCell::new(vec![
            ToolCommandResult::timed_out(Bytes::from_static(b"first")),
            ToolCommandResult::completed(0, Bytes::from_static(b"second")),
        ]),
    });
    let dispatcher = EnvironmentToolDispatcher::new(sandbox.clone(), Rc::new(AllowAll));
    let trace = TraceIdentity {
        run_id: "run".into(),
        trajectory_id: "trajectory".into(),
        trace_id: "trace".into(),
    };
    dispatcher
        .open_trace(TraceOpenContext {
            trace: &trace,
            environment: None,
            workspace: None,
            run_label: "fake",
        })
        .await
        .expect("sandbox opens");

    let first = dispatcher
        .dispatch(
            ToolDispatchRequest::new("first", "slow"),
            &ToolDispatchContext::default(),
        )
        .await
        .expect("timeout is a terminal command result");
    let second = dispatcher
        .dispatch(
            ToolDispatchRequest::new("second", "fast"),
            &ToolDispatchContext::default(),
        )
        .await
        .expect("successful recycle permits the following command");
    dispatcher
        .close_trace(&trace)
        .await
        .expect("sandbox closes");

    assert!(first.is_timed_out);
    assert_eq!(second.output, Bytes::from_static(b"second"));
    assert_eq!(
        sandbox.events.borrow().clone(),
        ["open", "run:slow", "recycle", "run:fast", "close"]
    );
}
