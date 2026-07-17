// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

use std::path::Path;

use aiperf_mock_server::config::MockServerConfig;
use serde_json::Value;

fn load_jsonl(path: &Path) -> Vec<Value> {
    let text = std::fs::read_to_string(path).expect("read jsonl file");
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("parse jsonl line"))
        .collect()
}

#[tokio::test]
async fn test_adaptive_scale_subprocess_contract_with_deterministic_saturation() {
    let mut cfg = MockServerConfig::default();
    cfg.ttft = 15.0;
    cfg.itl = 0.0;
    cfg.ttft_concurrency_quad_ms = 3.0;
    cfg.ttft_jitter_cv = 0.0;
    cfg.itl_jitter_cv = 0.0;
    cfg.workers = 1;

    let h = AIPerfHarness::new_with(cfg).await;

    let config_dir = tempfile::TempDir::new().expect("config tempdir");
    let config_path = config_dir.path().join("adaptive_scale_subprocess.yaml");
    let config_body = format!(
        r#"schemaVersion: "2.0"

benchmark:
  model: {model}
  endpoint:
    url: {url}
    type: chat
    streaming: true
  dataset:
    type: synthetic
    entries: 1000
    prompts:
      isl: 32
      osl: 8
  phases:
    - name: profiling
      type: concurrency
      concurrency: 8
      duration: 8.0
      adaptive_scale:
        enabled: true
        control_variable: concurrency
        min_concurrency: 1
        assessment_period: 1.0
        min_completed_requests: 1
        sustain_duration: 1.0
        strategy:
          type: ramp_until_fail
          step_policy: sla_margin
          base_step: 3
          max_step_multiplier: 1
      sla:
        request_latency:
          p95:
            le: 100
        goodput:
          avg:
            ge: 0.1
"#,
        model = DEFAULT_MODEL,
        url = h.mock.url,
    );
    std::fs::write(&config_path, config_body).expect("write config file");

    let r = h.run_timeout(
        &format!(
            "--config {} --extra-inputs ignore_eos:true --workers-max 1 --ui none",
            config_path.display()
        ),
        900,
    );

    assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
    assert!(r.artifacts.request_count() > 0.0);
    assert!(!r.artifacts.json().is_null());
    assert_eq!(r.artifacts.was_cancelled(), false);

    let event_path = r
        .artifacts
        .find_file("**/adaptive_scale_events.jsonl")
        .expect("adaptive_scale_events.jsonl exists");
    let summary_path = r
        .artifacts
        .find_file("**/adaptive_scale_summary.json")
        .expect("adaptive_scale_summary.json exists");

    let events = load_jsonl(&event_path);
    assert!(!events.is_empty());

    let event_names: std::collections::HashSet<&str> =
        events.iter().filter_map(|e| e["event"].as_str()).collect();
    assert!(event_names.contains("adaptive_phase_started"));
    assert!(event_names.contains("adaptive_window"));
    assert!(event_names.contains("adaptive_decision"));
    assert!(event_names.contains("boundary_discovered"));
    assert!(event_names.contains("sustain_started"));
    assert!(event_names.contains("adaptive_complete"));

    let discover_windows: Vec<&Value> = events
        .iter()
        .filter(|e| e["event"] == "adaptive_window" && e["phase"] == "discover")
        .collect();
    assert!(discover_windows.len() >= 2);
    assert_eq!(discover_windows[0]["control_variable"], "concurrency");
    assert_eq!(discover_windows[0]["control_value"].as_f64(), Some(1.0));
    assert_eq!(discover_windows[0]["schema_version"].as_f64(), Some(2.0));
    assert_eq!(
        discover_windows[0]["timestamp_ns"],
        discover_windows[0]["timestamp"]
    );
    let ts_utc = discover_windows[0]["timestamp_utc"]
        .as_str()
        .expect("timestamp_utc string");
    let ts_re = regex::Regex::new(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$").unwrap();
    assert!(ts_re.is_match(ts_utc), "timestamp_utc: {ts_utc}");

    let discover_decisions: Vec<&Value> = events
        .iter()
        .filter(|e| e["event"] == "adaptive_decision" && e["phase"] == "discover")
        .collect();
    assert!(!discover_decisions.is_empty());
    assert!(
        discover_decisions[0]["control_value_after"]
            .as_f64()
            .unwrap()
            > 1.0
    );
    assert!(
        discover_decisions
            .iter()
            .all(|e| e["step_size"].as_f64().unwrap() >= 1.0)
    );

    let boundary_events: Vec<&Value> = events
        .iter()
        .filter(|e| e["event"] == "boundary_discovered")
        .collect();
    let boundary = boundary_events.last().expect("boundary event");
    assert_eq!(boundary["control_variable"], "concurrency");
    assert_eq!(boundary["last_passing_value"], boundary["boundary_value"]);
    assert!(
        boundary["first_failing_value"].as_f64().unwrap()
            > boundary["boundary_value"].as_f64().unwrap()
    );
    assert_eq!(boundary["sla_metric"], "request_latency");
    assert_eq!(boundary["sla_stat"], "p95");
    assert_eq!(boundary["sla_op"], "le");
    assert_eq!(boundary["sla_bound"].as_f64(), Some(100.0));
    assert!(boundary["sla_value"].as_f64().unwrap() > boundary["sla_bound"].as_f64().unwrap());

    let sustain_windows: Vec<&Value> = events
        .iter()
        .filter(|e| e["event"] == "adaptive_window" && e["phase"] == "sustain")
        .collect();
    assert!(!sustain_windows.is_empty());
    let boundary_value = boundary["boundary_value"].as_f64().unwrap();
    assert!(
        sustain_windows
            .iter()
            .all(|e| e["control_value"].as_f64().unwrap() <= boundary_value)
    );

    let summary = read_json_value(&summary_path);
    assert_eq!(summary["schema_version"].as_f64(), Some(2.0));
    assert_eq!(summary["status"], "completed");
    assert_eq!(summary["control_variable"], "concurrency");
    let sla = &summary["sla"];
    assert_eq!(sla["metric"], "request_latency");
    assert_eq!(sla["stat"], "p95");
    assert_eq!(sla["op"], "le");
    assert_eq!(sla["bound"].as_f64(), Some(100.0));
    assert_eq!(summary["boundary_value"], boundary["boundary_value"]);
    assert_eq!(
        summary["first_failing_value"],
        boundary["first_failing_value"]
    );
    assert!(summary["control_value"].as_f64().unwrap() <= boundary_value);
    assert_eq!(summary["completed_reason"], "sustain_duration_completed");
    assert_eq!(
        summary["result"]["last_passing_value"],
        summary["last_passing_value"]
    );
    assert_eq!(
        summary["result"]["first_failing_value"],
        boundary["first_failing_value"]
    );
    assert_eq!(
        summary["result"]["boundary_value"],
        boundary["boundary_value"]
    );
    assert!(
        summary["totals"]["sent"].as_f64().unwrap()
            >= summary["totals"]["completed"].as_f64().unwrap()
    );
    assert_eq!(summary["totals"]["cancelled"], 0);
    assert!(summary["sustain_windows"].as_f64().unwrap() > 0.0);
    assert_eq!(summary["strategy_type"], "ramp_until_fail");
    assert_eq!(summary["step_policy"], "sla_margin");
}

fn read_json_value(path: &Path) -> Value {
    let bytes = std::fs::read(path).expect("read json file");
    serde_json::from_slice(&bytes).expect("parse json")
}
