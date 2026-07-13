// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in process canaries for the pinned canonical agentic package locks.
//!
//! The proof environments are developer/CI fixtures rather than Cargo inputs,
//! so an absent environment skips its canary. When present, each test imports
//! the real packages and reads their owning registry without executing an
//! episode or allowing a Python model client to send inference traffic.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("runner crate must live below the repository root")
}

fn run_proof_environment(environment: &str, script: &str) {
    let root = repository_root();
    let python = root.join(environment).join("bin/python");
    if !python.is_file() {
        eprintln!("skipping {environment}: {} is absent", python.display());
        return;
    }
    let output = Command::new(&python)
        .arg("-c")
        .arg(script)
        .current_dir(&root)
        .env("PYTHONPATH", root.join("src"))
        .env("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        .output()
        .expect("starting pinned agentic proof interpreter");
    assert!(
        output.status.success(),
        "{environment} canary failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

#[test]
fn real_agentlab_browsergym_registry_imports_without_model_http() {
    run_proof_environment(
        ".venv-browser-proof",
        r#"
import importlib.metadata

import aiperf.accuracy.browsergym as adapter
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.experiments.loop import ExpArgs
from bgym import DEFAULT_BENCHMARKS

adapter._require_browsergym_environment()
assert importlib.metadata.version("agentlab") == "0.4.2"
assert importlib.metadata.version("browsergym-core") == "0.14.3"
assert adapter.AIPerfAgentLabModelArgs.make_model.__module__ == adapter.__name__
assert GenericAgentArgs.__module__.startswith("agentlab.")
assert ExpArgs.__module__.startswith("agentlab.")
benchmark = DEFAULT_BENCHMARKS["miniwob_tiny_test"]()
assert len(benchmark.env_args_list) == 4
assert benchmark.env_args_list[0].task_name.startswith("miniwob.")
"#,
    );
}

#[test]
fn real_mcpmark_registry_imports_without_model_http() {
    run_proof_environment(
        ".venv-mcpmark-proof",
        r#"
import importlib.metadata

import aiperf.accuracy.mcpmark as adapter
from src.evaluator import MCPEvaluator
from src.factory import MCPServiceFactory

adapter._require_mcpmark_environment()
assert importlib.metadata.version("MCPMark") == "0.0.1"
assert MCPEvaluator.__module__ == "src.evaluator"
services = MCPServiceFactory.get_supported_mcp_services()
assert "filesystem" in services
manager = MCPServiceFactory.create_task_manager("filesystem", task_suite="standard")
tasks = manager.discover_all_tasks()
assert len(tasks) == 30
assert all(task.task_instruction_path.is_file() for task in tasks)
assert adapter.litellm.acompletion.__module__ != adapter.__name__
"#,
    );
}
