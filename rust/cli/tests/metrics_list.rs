// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! `aiperf metrics list` / `metrics describe` command contract coverage.

use std::process::Command;

#[test]
fn metrics_list_includes_known_id() {
    let out = Command::new(env!("CARGO_BIN_EXE_aiperf"))
        .args(["metrics", "list"])
        .output()
        .expect("run");
    assert!(out.status.success(), "status: {:?}", out.status);
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(
        s.contains("aiperf.request_latency"),
        "list output missing known id; got:\n{s}"
    );
}

#[test]
fn metrics_list_markdown_emits_table() {
    let out = Command::new(env!("CARGO_BIN_EXE_aiperf"))
        .args(["metrics", "list", "--markdown"])
        .output()
        .expect("run");
    assert!(out.status.success(), "status: {:?}", out.status);
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("| id |"), "markdown header missing; got:\n{s}");
    assert!(
        s.contains("aiperf.request_latency"),
        "markdown missing known id; got:\n{s}"
    );
}

#[test]
fn metrics_describe_prints_header() {
    let out = Command::new(env!("CARGO_BIN_EXE_aiperf"))
        .args(["metrics", "describe", "aiperf.request_latency"])
        .output()
        .expect("run");
    assert!(out.status.success(), "status: {:?}", out.status);
    let s = String::from_utf8_lossy(&out.stdout);
    // The definition's header must appear in the describe output.
    assert!(
        s.contains("Request Latency"),
        "describe output missing header; got:\n{s}"
    );
    assert!(
        s.contains("aiperf.request_latency"),
        "describe output missing id; got:\n{s}"
    );
}

#[test]
fn metrics_describe_unknown_fails_cleanly() {
    let out = Command::new(env!("CARGO_BIN_EXE_aiperf"))
        .args(["metrics", "describe", "aiperf.not_a_real_metric"])
        .output()
        .expect("run");
    assert_eq!(out.status.code(), Some(1), "expected nonzero exit");
    let s = String::from_utf8_lossy(&out.stderr);
    assert!(
        s.contains("unknown metric"),
        "expected clean error; got stderr:\n{s}"
    );
}
