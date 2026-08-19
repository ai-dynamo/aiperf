// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Product acceptance for one terminal-only externally driven Harbor task.

mod common;

use std::{fs, path::Path, sync::Mutex};

use aiperf_runtime::eval::ArtifactDigest;
use common::AIPerfHarness;
use serde_json::Value;

static DOCKER_E2E_LOCK: Mutex<()> = Mutex::new(());

#[tokio::test]
async fn external_task_preserves_compatibility_fidelity_and_verifier_score_end_to_end() {
    let temporary = tempfile::tempdir().expect("create external compatibility fixture root");
    let task = temporary.path().join("external-compatibility-task");
    let lifecycle = temporary.path().join("lifecycle.json");
    write_external_task(&task);
    write_lifecycle(&lifecycle);
    let harness = AIPerfHarness::new().await;

    let _docker = DOCKER_E2E_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let result = harness.run_no_server(&format!(
        "eval --task {} --lifecycle-request {} --image sha256:{}",
        task.display(),
        lifecycle.display(),
        "a".repeat(64),
    ));

    assert!(
        result.success(),
        "external compatibility eval failed with {}\nstdout:\n{}\nstderr:\n{}",
        result.exit_code,
        result.stdout,
        result.stderr
    );
    let report_text = format!("{}\n{}", result.stdout, result.stderr);
    assert!(
        !report_text.contains("driver-private-output"),
        "raw Driver terminal data must not enter product output: {report_text}"
    );
    let report: Value = serde_json::from_str(
        result
            .stdout
            .lines()
            .last()
            .expect("scored eval prints a final summary"),
    )
    .expect("final external compatibility summary is JSON");

    assert_eq!(report["task"], "example/harbor-external-compatibility");
    assert_eq!(report["reward"]["reward"], 1.0);
    assert_eq!(report["score"]["state"], "verified");
    assert_eq!(report["score"]["reward"], 1.0);
    assert_eq!(report["fidelity"]["profile"], "externally_driven");
    assert_eq!(report["fidelity"]["capture"], "missing");
    let lifecycle_evidence = report["lifecycle_evidence"]
        .as_array()
        .expect("external result exports digest-only lifecycle evidence");
    assert_eq!(lifecycle_evidence.len(), 1);
    assert!(
        lifecycle_evidence[0]
            .as_str()
            .is_some_and(|digest| digest.starts_with("blake3:")),
        "compatibility lifecycle evidence must remain content-addressed: {lifecycle_evidence:?}"
    );
}

fn write_external_task(task: &Path) {
    fs::create_dir_all(task.join("environment")).expect("create environment directory");
    fs::create_dir_all(task.join("tests")).expect("create verifier directory");
    fs::create_dir_all(task.join("tools")).expect("create Driver directory");
    fs::write(
        task.join("task.toml"),
        r#"schema_version = "1.1"
artifacts = ["/work/result.txt"]

[task]
name = "example/harbor-external-compatibility"

[native_graph]
profile = "externally_driven"
adapter_manifest = "adapters.toml"
driver = "driver-adapter"
external_driver_factory_id = "terminal_v1"
"#,
    )
    .expect("write external task manifest");
    fs::write(
        task.join("instruction.md"),
        "Complete one supervised external episode.\n",
    )
    .expect("write instruction");
    fs::write(
        task.join("environment/Dockerfile"),
        r#"FROM alpine:3.20
COPY driver.sh /tools/driver.sh
RUN chmod 0755 /tools/driver.sh && mkdir -p /work /logs/verifier && chmod 0777 /work /logs/verifier
CMD ["sh", "-c", "while :; do sleep 3600; done"]
"#,
    )
    .expect("write Dockerfile");
    fs::write(
        task.join("tests/test.sh"),
        "test \"$(cat /work/result.txt)\" = external-complete\nprintf '{\"reward\":1.0}' > /logs/verifier/reward.json\n",
    )
    .expect("write verifier");
    fs::write(
        task.join("adapters.toml"),
        r#"[[adapters]]
id = "driver-adapter"
role = "driver"
argv = ["tools/driver.sh"]
executable = "tools/driver.sh"
"#,
    )
    .expect("write Driver manifest");
    let driver = r#"#!/bin/sh
sequence=0
episode=""

field() {
    printf '%s\n' "$1" | sed -n "s/.*\"$2\":\"\\([^\"]*\\)\".*/\\1/p"
}

emit() {
    printf '{"version":1,"episode":"%s","span":"%s","sequence":%s,"operation":"%s","message":%s}\n' \
        "$episode" "$1" "$sequence" "$2" "$3"
    sequence=$((sequence + 1))
}

while IFS= read -r line; do
    host_episode=$(field "$line" episode)
    host_span=$(field "$line" span)
    host_operation=$(field "$line" operation)
    case "$line" in
        *'"type":"hello"'*)
            episode=$host_episode
            emit "$host_span" "$host_operation" \
                '{"type":"ready","protocol_version":1,"capabilities":["driver"],"implementation_digest":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}'
            ;;
        *'"type":"request_episode_terminal"'*)
            printf 'external-complete' > /work/result.txt
            emit "external-driver-terminal" "$host_operation" \
                '{"type":"episode_terminal_candidate","output":{"private":"driver-private-output","status":"complete"}}'
            ;;
    esac
done
"#;
    fs::write(task.join("tools/driver.sh"), driver).expect("write immutable Driver executable");
    fs::write(task.join("environment/driver.sh"), driver)
        .expect("write Driver into the Docker build context");
}

fn write_lifecycle(path: &Path) {
    let policy = ArtifactDigest::from_bytes(b"external-compatibility-policy");
    fs::write(
        path,
        format!(
            r#"{{"version":1,"agent_variant":"external-driver","model":{{"provider":"external","model":"opaque"}},"seed":7,"policy":"{policy}","runtime":"external:v1","attempt":"attempt-1","budget":{{"execution_seconds":30.0,"verifier_seconds":30.0}},"agent_contract":"externally_driven","command":["tools/driver.sh"],"initial_score":{{"metric":"reward","rationale":"{policy}"}},"regrade":{{"metric":"reward","rationale":"{policy}"}}}}"#,
            policy = policy.as_str(),
        ),
    )
    .expect("write external lifecycle request");
}
