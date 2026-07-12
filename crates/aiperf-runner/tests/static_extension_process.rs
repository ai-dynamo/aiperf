// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process proof for a statically linked product-registry extension.
//!
//! The spawned child runs the strict protocol-v2 coordinator with an injected
//! [`aiperf_extensions::AiperfRegistryFactory`]. The coordinator builds that factory exactly once
//! and threads the resulting frozen registry through validation, dataset
//! construction, scheduling, and execution. The custom sampler name appears
//! only in the authored request and extension registration: runner core has no
//! matching branch, and this test never reconstructs a private registry.

#![cfg(feature = "dynamo-offline")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use aiperf_graph::input::GraphInputAdapterRegistry;
use aiperf_runner::coordinator::{RunnerResponseV2, RunnerV2Coordinator};
use aiperf_runner::protocol_v2::RunnerEnvelopeV2;
use aiperf_runner::registry::BuiltinRunnerRegistryFactory;
use aiperf_test_static_extension::{
    EXTENSION_NAME, SAMPLER_NAME, StaticTestRegistryFactory, evidence, reset_evidence,
};

const CHILD_ENV: &str = "AIPERF_STATIC_EXTENSION_CHILD";
const CHILD_ROOT_ENV: &str = "AIPERF_STATIC_EXTENSION_ROOT";
const DISTRIBUTION_ID: &str =
    "blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

fn envelope(artifact_dir: &Path) -> RunnerEnvelopeV2 {
    serde_json::from_value(serde_json::json!({
        "protocol_version": 2,
        "operation": "execute",
        "expected_distribution_id": DISTRIBUTION_ID,
        "run": {
            "identity": {
                "benchmark_id": "static-extension-process-proof",
                "random_seed": 37
            },
            "artifact_target": artifact_dir,
            "models": {"items": [{"name": "extension-model"}]},
            "endpoints": {"profiles": [{
                "id": "default",
                "type": "chat",
                "urls": ["http://must-not-be-contacted.invalid"],
                "streaming": true,
                "use_server_token_count": true
            }]},
            "backend": {
                "type": "dynamo_offline",
                "config": {}
            },
            "workload": {
                "type": "scheduled",
                "config": {
                    "worker_count": 1,
                    "dataset": {
                        "type": "synthetic",
                        "entries": 3,
                        "sampling": SAMPLER_NAME,
                        "prompts": {
                            "isl": {"value": 4.0},
                            "osl": {"value": 1.0},
                            "batch_size": 1
                        },
                        "turns": {"value": 1.0},
                        "turn_delay_ms": {"value": 0.0},
                        "turn_delay_ratio": 1.0
                    },
                    "tokenizer": {
                        "name": "builtin",
                        "revision": "main",
                        "trust_remote_code": false,
                        "apply_chat_template": false
                    },
                    "phases": [{
                        "type": "concurrency",
                        "name": "profiling",
                        "exclude_from_results": false,
                        "requests": 3,
                        "concurrency": 2
                    }]
                }
            },
            "metrics": {},
            "artifacts": {},
            "sidecars": {}
        }
    }))
    .expect("fixture envelope is valid")
}

#[test]
fn statically_linked_extension_is_selected_in_a_fresh_runner_process() {
    let root = tempfile::tempdir().unwrap();
    let output = Command::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            "statically_linked_extension_child",
            "--nocapture",
        ])
        .env(CHILD_ENV, "1")
        .env(CHILD_ROOT_ENV, root.path())
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "child stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let proof: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.path().join("proof.json")).unwrap()).unwrap();
    assert_eq!(proof["registry_builds"], 1);
    assert_eq!(proof["sampler_creations"], 1);
    assert!(proof["sampler_next_calls"].as_u64().unwrap() >= 3);
    assert_eq!(proof["request_count"], 3.0);
    assert_eq!(proof["extension_name"], EXTENSION_NAME);
    assert_eq!(
        proof["advertised_extensions"],
        serde_json::json!([EXTENSION_NAME])
    );
    assert_eq!(proof["distribution_id"], DISTRIBUTION_ID);
    assert_eq!(proof["backend"], "dynamo_offline");
    assert_eq!(proof["workload"], "scheduled");
}

#[test]
#[ignore = "spawned explicitly by the parent process proof"]
fn statically_linked_extension_child() {
    assert_eq!(std::env::var(CHILD_ENV).as_deref(), Ok("1"));
    reset_evidence();
    let root = PathBuf::from(std::env::var_os(CHILD_ROOT_ENV).expect("child root is configured"));
    let artifact_dir = root.join("artifacts");
    let coordinator = RunnerV2Coordinator::new(
        DISTRIBUTION_ID,
        &BuiltinRunnerRegistryFactory,
        &StaticTestRegistryFactory,
        Arc::new(GraphInputAdapterRegistry::with_builtin_adapters()),
    )
    .unwrap();

    assert_eq!(evidence().registry_builds, 1);
    let capabilities = coordinator.capabilities();
    assert_eq!(capabilities.distribution_id, DISTRIBUTION_ID);
    assert_eq!(capabilities.extensions, [EXTENSION_NAME]);
    assert!(
        capabilities
            .supported_pairs
            .contains(&["dynamo_offline".to_owned(), "scheduled".to_owned()])
    );
    assert_eq!(evidence().registry_builds, 1);
    assert_eq!(
        coordinator
            .product_registry()
            .extension_names()
            .collect::<Vec<_>>(),
        [EXTENSION_NAME]
    );
    let result = coordinator.handle(envelope(&artifact_dir));
    assert_eq!(result.exit_code, 0);
    let RunnerResponseV2::Terminal(terminal) = result.response else {
        panic!("execute returned a validation response")
    };
    assert!(terminal.success, "terminal errors: {:?}", terminal.errors);
    assert_eq!(terminal.provenance["backend"], "dynamo_offline");
    assert_eq!(terminal.provenance["workload"], "scheduled");
    let extension_evidence = evidence();
    assert_eq!(extension_evidence.registry_builds, 1);
    assert_eq!(extension_evidence.sampler_creations, 1);
    assert!(extension_evidence.sampler_next_calls >= 3);

    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(artifact_dir.join("native-v2.json")).unwrap())
            .unwrap();
    let request_count = report["metrics"]["request_count"]["series"][0]["stats"]["total"]
        .as_f64()
        .unwrap();
    std::fs::write(
        root.join("proof.json"),
        serde_json::to_vec(&serde_json::json!({
            "registry_builds": extension_evidence.registry_builds,
            "sampler_creations": extension_evidence.sampler_creations,
            "sampler_next_calls": extension_evidence.sampler_next_calls,
            "request_count": request_count,
            "extension_name": EXTENSION_NAME,
            "advertised_extensions": capabilities.extensions,
            "distribution_id": capabilities.distribution_id,
            "backend": terminal.provenance["backend"],
            "workload": terminal.provenance["workload"]
        }))
        .unwrap(),
    )
    .unwrap();
}
