// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native Python evaluator integration coverage for request demux and process-group reap.

use std::ffi::OsString;
use std::path::Path;
use std::time::Duration;

use aiperf_runtime::accuracy_core::{
    AccuracyEvaluator, EvaluatorGradeItem, ProblemId, PythonEvaluator, WorkerProcessConfig,
};

fn python_program() -> OsString {
    std::env::var_os("PYTHON").unwrap_or_else(|| OsString::from("python3"))
}

fn worker_config(module_dir: &Path, module_name: &str, pid_file: &Path) -> WorkerProcessConfig {
    WorkerProcessConfig::new(python_program())
        .arg("-u")
        .arg("-m")
        .arg(module_name)
        .arg(pid_file.as_os_str())
        .env("PYTHONPATH", module_dir.as_os_str())
}

fn write_fixture_worker(module_dir: &Path, module_name: &str) {
    let script = r#"
import json, pathlib, subprocess, sys

pid_file = pathlib.Path(sys.argv[1])
pending = []
descendant = None

for line in sys.stdin:
    request = json.loads(line)
    op = request["op"]
    if op == "hello":
        descendant = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3600)"])
        pid_file.write_text(str(descendant.pid))
        result = {
            "protocol": 1,
            "worker_version": "fixture",
            "python_version": sys.version.split()[0],
            "python_executable": sys.executable,
            "packages": {"lighteval": "fixture"},
            "worker_source_sha256": "a" * 64,
            "dependency_lock_sha256": "b" * 64,
            "container_digest": None,
            "capabilities": ["load", "next_problems", "grade_batch", "shutdown"],
        }
        print(json.dumps({"id": request["id"], "ok": True, "result": result}), flush=True)
    elif op == "grade_batch":
        pending.append(request)
        if len(pending) == 2:
            for queued in reversed(pending):
                item = queued["items"][0]
                result = {
                    "items": [{
                        "problem_id": item["problem_id"],
                        "task": "lcb",
                        "correct": True,
                        "unparsed": False,
                        "confidence": float(queued["id"]),
                        "reasoning": "fixture",
                        "extracted_answer": item["response"],
                    }]
                }
                print(json.dumps({"id": queued["id"], "ok": True, "result": result}), flush=True)
            pending = []
    elif op == "shutdown":
        print(json.dumps({"id": request["id"], "ok": True, "result": {"shutdown": True}}), flush=True)
        break
"#;
    std::fs::write(
        module_dir.join(format!("{module_name}.py")),
        script.trim_start(),
    )
    .expect("write fixture worker module");
}

#[cfg(unix)]
fn process_exists(pid: i32) -> Result<bool, std::io::Error> {
    let result = unsafe { libc::kill(pid, 0) };
    if result == 0 {
        return Ok(true);
    }
    match std::io::Error::last_os_error().raw_os_error() {
        Some(libc::ESRCH) => Ok(false),
        Some(libc::EPERM) => Ok(true),
        _ => Err(std::io::Error::last_os_error()),
    }
}

async fn wait_for_text_file(path: &Path) -> String {
    for _ in 0..100 {
        if let Ok(text) = std::fs::read_to_string(path) {
            return text;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("timed out waiting for {}", path.display());
}

#[cfg(unix)]
#[tokio::test]
async fn native_worker_demuxes_out_of_order_grades_and_reaps_descendants() {
    let module_dir = tempfile::tempdir().expect("module tempdir");
    let pid_file = module_dir.path().join("descendant.pid");
    let module_name = "fixture_codegen_worker";
    write_fixture_worker(module_dir.path(), module_name);

    let mut evaluator =
        PythonEvaluator::spawn(worker_config(module_dir.path(), module_name, &pid_file))
            .await
            .expect("spawn native Python evaluator");
    let descendant_pid = wait_for_text_file(&pid_file)
        .await
        .trim()
        .parse::<i32>()
        .expect("parse descendant pid");

    let first_items = vec![EvaluatorGradeItem {
        problem_id: ProblemId::new("opaque-1").expect("problem id"),
        response: "first".to_string(),
    }];
    let second_items = vec![EvaluatorGradeItem {
        problem_id: ProblemId::new("opaque-2").expect("problem id"),
        response: "second".to_string(),
    }];

    let (first, second) = tokio::join!(
        evaluator.grade_batch_with_request_id_for_testing(101, &first_items),
        evaluator.grade_batch_with_request_id_for_testing(202, &second_items),
    );
    let first = first.expect("first grade batch");
    let second = second.expect("second grade batch");

    assert_eq!(first.items[0].problem_id.as_str(), "opaque-1");
    assert_eq!(second.items[0].problem_id.as_str(), "opaque-2");
    assert_eq!(first.items[0].confidence, 101.0);
    assert_eq!(second.items[0].confidence, 202.0);
    assert!(process_exists(descendant_pid).expect("inspect descendant before shutdown"));

    evaluator.shutdown().await.expect("shutdown evaluator");
    assert!(
        !process_exists(descendant_pid).expect("inspect descendant after shutdown"),
        "shutdown must reap the worker's descendant process group",
    );
}
