// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! `AIPERF_METRICS_SKETCH` env var honored by both surfaces (isolated in its own
//! test binary so mutating the process env cannot race the parity suite).

use aiperf_cli::flags::ProfileFlags;
use aiperf_cli::load;

#[test]
fn env_var_enables_sketch_without_flag() {
    // SAFETY: single-threaded test binary; no other test reads this env here.
    unsafe { std::env::set_var("AIPERF_METRICS_SKETCH", "1") };
    let args: Vec<String> = [
        "--model",
        "m",
        "--url",
        "127.0.0.1:8000",
        "--endpoint-type",
        "chat",
        "--concurrency",
        "1",
        "--request-count",
        "2",
        "--otel-url",
        "http://otel:4317",
        "--artifact-dir",
        "/tmp/x",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let flags = ProfileFlags::parse_from_args(&args).unwrap();
    let run = load::resolve(&flags).unwrap();
    let cfg = serde_json::to_value(&run).unwrap();
    let cfg = &cfg["cfg"];
    // Env sketch: metrics.sketch=true, per-record JSONL dropped, otel suppressed.
    assert_eq!(cfg["metrics"]["sketch"], serde_json::json!(true));
    assert!(cfg["artifacts"].get("records_path").is_none());
    assert!(cfg["export"]["otel"].is_null());
    unsafe { std::env::remove_var("AIPERF_METRICS_SKETCH") };
}
