// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! NativeGraph rollout acceptance coverage against the in-repo mock server.

mod common;

use std::{fs, path::Path, sync::Mutex};

use aiperf_mock_server::accuracy::AccuracyFormat;
use common::{AIPerfHarness, MockServerConfig};
use serde_json::{Value, json};

const MODEL: &str = "harbor-policy-model";

static DOCKER_E2E_LOCK: Mutex<()> = Mutex::new(());

#[tokio::test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
async fn native_graph_rollout_uses_mock_selected_policy_decisions_end_to_end() {
    let temporary = tempfile::tempdir().expect("create rollout fixture root");
    let dataset = temporary.path().join("policy.jsonl");
    fs::write(
        &dataset,
        concat!(
            "{\"prompt\":\"state-zero\",\"match_key\":\"state-zero\",\"answer\":\"{\\\"kind\\\":\\\"move\\\",\\\"direction\\\":\\\"north\\\"}\",\"task\":\"policy-zero\"}\n",
            "{\"prompt\":\"state-one\",\"match_key\":\"state-one\",\"answer\":\"{\\\"kind\\\":\\\"move\\\",\\\"direction\\\":\\\"south\\\"}\",\"task\":\"policy-one\"}\n",
        ),
    )
    .expect("write policy accuracy dataset");
    let harness = AIPerfHarness::new_with(policy_mock_config(&dataset)).await;
    let task = temporary.path().join("native-graph-rollout-task");
    write_rollout_task(&task, &harness.mock.url);
    let model_runtime = temporary.path().join("model-runtime.toml");
    fs::write(&model_runtime, "version = 1\n").expect("write model runtime");
    let lifecycle = temporary.path().join("lifecycle.json");
    write_lifecycle(&lifecycle);

    let _docker = docker_e2e_lock();
    let result = harness.run_no_server(&format!(
        "eval --task {} --model-runtime {} --lifecycle-request {} --image sha256:{}",
        task.display(),
        model_runtime.display(),
        lifecycle.display(),
        "a".repeat(64),
    ));
    assert!(
        result.success(),
        "native rollout eval failed with {}\nstdout:\n{}\nstderr:\n{}",
        result.exit_code,
        result.stdout,
        result.stderr
    );

    let summary: Value = serde_json::from_str(
        result
            .stdout
            .lines()
            .last()
            .expect("scored eval prints a final summary"),
    )
    .expect("final scored summary is JSON");
    assert_eq!(summary["task"], "example/harbor-native-graph-rollout");
    assert_eq!(summary["reward"]["reward"], 0.75);
    assert_eq!(summary["episodes"], 1);

    let accuracy = harness.mock.state.accuracy_live.snapshot();
    assert_eq!(
        accuracy.matched, 2,
        "reset and transition must each drive a policy call"
    );
    assert_eq!(
        accuracy.unmatched, 0,
        "every policy prompt must contain the current observation"
    );
    assert_eq!(accuracy.tasks["policy-zero"].matched, 1);
    assert_eq!(accuracy.tasks["policy-one"].matched, 1);
}

#[tokio::test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
async fn selected_policy_actions_produce_distinct_verified_rewards_end_to_end() {
    let _docker = docker_e2e_lock();
    let north_reward = run_rollout_for_actions(
        r#"{"kind":"move","direction":"north"}"#,
        r#"{"kind":"move","direction":"north"}"#,
    )
    .await;
    let south_reward = run_rollout_for_actions(
        r#"{"kind":"move","direction":"south"}"#,
        r#"{"kind":"move","direction":"south"}"#,
    )
    .await;

    assert_eq!(north_reward, 0.25);
    assert_eq!(south_reward, 0.75);
}

#[tokio::test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
async fn malformed_mock_policy_decision_never_reaches_the_verifier() {
    let temporary = tempfile::tempdir().expect("create malformed-policy fixture root");
    let dataset = temporary.path().join("policy.jsonl");
    write_policy_dataset(
        &dataset,
        "not-json",
        r#"{"kind":"move","direction":"south"}"#,
    );
    let harness = AIPerfHarness::new_with(policy_mock_config(&dataset)).await;
    let task = temporary.path().join("native-graph-rollout-task");
    write_rollout_task(&task, &harness.mock.url);
    fs::write(task.join("tests/test.sh"), "exit 97\n").expect("make verifier failure observable");
    let model_runtime = temporary.path().join("model-runtime.toml");
    fs::write(&model_runtime, "version = 1\n").expect("write model runtime");
    let lifecycle = temporary.path().join("lifecycle.json");
    write_lifecycle(&lifecycle);

    let _docker = docker_e2e_lock();
    let result = harness.run_no_server(&eval_command(&task, &model_runtime, &lifecycle));

    assert!(
        !result.success(),
        "malformed policy decision must fail the episode"
    );
    assert!(
        !result.stderr.contains("97"),
        "the verifier must not run after policy admission fails: {}",
        result.stderr
    );
    let accuracy = harness.mock.state.accuracy_live.snapshot();
    assert_eq!(
        accuracy.matched, 1,
        "only the reset observation reaches the model"
    );
    assert_eq!(accuracy.unmatched, 0);
}

#[tokio::test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
async fn adapter_protocol_failure_never_reaches_policy_or_verifier() {
    let temporary = tempfile::tempdir().expect("create protocol-failure fixture root");
    let dataset = temporary.path().join("policy.jsonl");
    write_policy_dataset(
        &dataset,
        r#"{"kind":"move","direction":"north"}"#,
        r#"{"kind":"move","direction":"south"}"#,
    );
    let harness = AIPerfHarness::new_with(policy_mock_config(&dataset)).await;
    let task = temporary.path().join("native-graph-rollout-task");
    write_rollout_task(&task, &harness.mock.url);
    fs::write(
        task.join("environment/environment.sh"),
        "#!/bin/sh\nexit 0\n",
    )
    .expect("replace adapter with protocol failure");
    fs::write(task.join("tests/test.sh"), "exit 97\n").expect("make verifier failure observable");
    let model_runtime = temporary.path().join("model-runtime.toml");
    fs::write(&model_runtime, "version = 1\n").expect("write model runtime");
    let lifecycle = temporary.path().join("lifecycle.json");
    write_lifecycle(&lifecycle);

    let _docker = docker_e2e_lock();
    let result = harness.run_no_server(&eval_command(&task, &model_runtime, &lifecycle));

    assert!(!result.success(), "protocol failure must fail the episode");
    assert!(
        result.stderr.contains("adapter closed its protocol stream"),
        "adapter startup must fail before any later phase: {}",
        result.stderr
    );
    assert!(
        !result.stderr.contains("97"),
        "the verifier must not run after adapter admission fails: {}",
        result.stderr
    );
    let accuracy = harness.mock.state.accuracy_live.snapshot();
    assert_eq!(
        accuracy.matched, 0,
        "adapter startup failure must precede model dispatch"
    );
    assert_eq!(accuracy.unmatched, 0);
}

fn docker_e2e_lock() -> std::sync::MutexGuard<'static, ()> {
    DOCKER_E2E_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn write_policy_dataset(path: &Path, first: &str, second: &str) {
    fs::write(
        path,
        format!(
            "{{\"prompt\":\"state-zero\",\"match_key\":\"state-zero\",\"answer\":{},\"task\":\"policy-zero\"}}\n{{\"prompt\":\"state-one\",\"match_key\":\"state-one\",\"answer\":{},\"task\":\"policy-one\"}}\n",
            serde_json::to_string(first).expect("serialize first policy answer"),
            serde_json::to_string(second).expect("serialize second policy answer"),
        ),
    )
    .expect("write policy accuracy dataset");
}

fn eval_command(task: &Path, model_runtime: &Path, lifecycle: &Path) -> String {
    format!(
        "eval --task {} --model-runtime {} --lifecycle-request {} --image sha256:{}",
        task.display(),
        model_runtime.display(),
        lifecycle.display(),
        "a".repeat(64),
    )
}

async fn run_rollout_for_actions(first_action: &str, second_action: &str) -> f64 {
    let temporary = tempfile::tempdir().expect("create action-selected rollout fixture root");
    let dataset = temporary.path().join("policy.jsonl");
    write_policy_dataset(&dataset, first_action, second_action);
    let harness = AIPerfHarness::new_with(policy_mock_config(&dataset)).await;
    let task = temporary.path().join("native-graph-rollout-task");
    write_rollout_task(&task, &harness.mock.url);
    let model_runtime = temporary.path().join("model-runtime.toml");
    fs::write(&model_runtime, "version = 1\n").expect("write model runtime");
    let lifecycle = temporary.path().join("lifecycle.json");
    write_lifecycle(&lifecycle);

    let result = harness.run_no_server(&eval_command(&task, &model_runtime, &lifecycle));
    assert!(
        result.success(),
        "native rollout eval failed with {}\nstdout:\n{}\nstderr:\n{}",
        result.exit_code,
        result.stdout,
        result.stderr
    );
    let summary: Value = serde_json::from_str(
        result
            .stdout
            .lines()
            .last()
            .expect("scored eval prints a final summary"),
    )
    .expect("final scored summary is JSON");
    let accuracy = harness.mock.state.accuracy_live.snapshot();
    assert_eq!(accuracy.matched, 2);
    assert_eq!(accuracy.unmatched, 0);
    summary["reward"]["reward"]
        .as_f64()
        .expect("verifier reward is finite")
}

fn policy_mock_config(dataset: &Path) -> MockServerConfig {
    MockServerConfig {
        fast: true,
        workers: 2,
        no_tokenizer: true,
        accuracy_dataset: Some(dataset.to_string_lossy().into_owned()),
        accuracy_format: AccuracyFormat::Passthrough,
        accuracy_correct_rate: 1.0,
        accuracy_cot_rate: 0.0,
        accuracy_adversarial_rate: 0.0,
        ..MockServerConfig::default()
    }
}

fn write_rollout_task(task: &Path, endpoint: &str) {
    fs::create_dir_all(task.join("environment")).expect("create environment directory");
    fs::create_dir_all(task.join("tests")).expect("create verifier directory");
    fs::create_dir_all(task.join("rollout")).expect("create rollout directory");
    fs::write(
        task.join("task.toml"),
        r#"schema_version = "1.1"
artifacts = ["/work/result.txt"]

[task]
name = "example/harbor-native-graph-rollout"

[environment]
network = "no-network"

[agent]
network = "no-network"

[native_graph]
profile = "native_graph"
program = "agent_graph.json"
model_bindings = "models.toml"
adapter_manifest = "adapters.toml"
"#,
    )
    .expect("write task manifest");
    fs::write(task.join("instruction.md"), "Execute the sealed rollout.\n")
        .expect("write instruction");
    fs::write(
        task.join("environment/Dockerfile"),
        "FROM alpine:3.20\nCOPY environment.sh /environment/environment.sh\nRUN chmod 0755 /environment/environment.sh && mkdir -p /work /logs/verifier && printf rollout > /work/result.txt && chmod 0777 /work /logs/verifier\n",
    )
    .expect("write Dockerfile");
    fs::write(
        task.join("tests/test.sh"),
        r#"test -f /environment/environment.sh
case "$(cat /work/result.txt)" in
  north) reward=0.25 ;;
  south) reward=0.75 ;;
  *) exit 96 ;;
esac
printf '{"reward":%s}' "$reward" > /logs/verifier/reward.json
"#,
    )
    .expect("write verifier");
    fs::write(
        task.join("agent_graph.json"),
        r#"{
  "schema_version": "1.0", "trace_id": "harbor-native-graph-rollout", "stage_bound": 1,
  "channels": { "rollout": { "type": "text", "reducer": "overwrite" } },
  "nodes": [],
  "edges": [{ "source": "START", "target": "END" }],
  "terminal_outputs": []
}"#,
    )
    .expect("write closed graph");
    fs::write(
        task.join("models.toml"),
        format!(
            r#"[[model_bindings]]
id = "primary"
endpoint_profile_id = "provider-default"
endpoint_factory_id = "chat"
transport_factory_id = "http"
model = "{MODEL}"
urls = ["{endpoint}/v1"]
streaming = true
request_timeout_ms = 30000
capture = "metadata"

[model_bindings.tokenizer]
type = "local"
name = "builtin"
revision = "main"
apply_chat_template = false

[model_bindings.generation]
max_tokens = 17
temperature = 0.37
"#,
        ),
    )
    .expect("write model binding");
    fs::write(
        task.join("adapters.toml"),
        r#"[[adapters]]
id = "environment-adapter"
role = "environment"
argv = ["environment/environment.sh"]
executable = "environment/environment.sh"
"#,
    )
    .expect("write adapter manifest");
    fs::write(
        task.join("environment/environment.sh"),
        r#"#!/bin/sh
sequence=0
episode=""
parent=""
parent_span=""
stage=""
step_count=0
artifact_phase=""
current_bytes=""
output_operation=""
first_reference=""
pending_download=""
action_bytes=""

field() {
    printf '%s\n' "$1" | sed -n "s/.*\"$2\":\"\\([^\"]*\\)\".*/\\1/p"
}

emit() {
    printf '{"version":1,"episode":"%s","span":"%s","sequence":%s,"operation":"%s","message":%s}\n' \
        "$episode" "$1" "$sequence" "$2" "$3"
    sequence=$((sequence + 1))
}

request_upload() {
    output_operation="${parent}-output-${sequence}"
    byte_count=$(printf '%s' "$current_bytes" | wc -c | tr -d ' ')
    emit "$parent_span" "$output_operation" \
        "{\"type\":\"put_artifact_request\",\"parent_operation\":\"$parent\",\"declared_bytes\":$byte_count}"
}

while IFS= read -r line; do
    host_episode=$(field "$line" episode)
    host_span=$(field "$line" span)
    host_operation=$(field "$line" operation)
    case "$line" in
        *'"type":"hello"'*)
            episode=$host_episode
            emit "$host_span" "$host_operation" \
                '{"type":"ready","protocol_version":1,"capabilities":["environment","artifacts"],"implementation_digest":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}'
            ;;
        *'"type":"reset_environment"'*)
            parent=$host_operation
            parent_span=$host_span
            stage=reset
            current_bytes=state-zero
            artifact_phase=observation
            request_upload
            ;;
        *'"type":"step_environment"'*)
            parent=$host_operation
            parent_span=$host_span
            step_count=$((step_count + 1))
            stage=step
            action_reference=$(printf '%s\n' "$line" | sed -n 's/.*"action_ref":\(.*\)}}$/\1/p')
            [ -n "$action_reference" ] || exit 1
            emit "$parent_span" "${parent}-read" \
                "{\"type\":\"get_artifact_request\",\"parent_operation\":\"$parent\",\"request\":$action_reference}"
            ;;
        *'"type":"get_artifact_handle"'*)
            pending_download=$(field "$line" download)
            [ -n "$pending_download" ] || exit 1
            action_bytes=""
            ;;
        *'"type":"artifact_download_chunk"'*)
            download=$(field "$line" download)
            [ "$download" = "$pending_download" ] || exit 1
            chunk=$(field "$line" bytes_base64)
            action_bytes="${action_bytes}$(printf '%s' "$chunk" | base64 -d)"
            ;;
        *'"type":"artifact_download_complete"'*)
            download=$(field "$line" download)
            [ "$download" = "$pending_download" ] || exit 1
            pending_download=""
            case "$action_bytes" in
                *'"direction":"north"'*) printf north > /work/result.txt ;;
                *'"direction":"south"'*) printf south > /work/result.txt ;;
                *) exit 1 ;;
            esac
            artifact_phase=observation
            if [ "$step_count" -eq 1 ]; then
                current_bytes=state-one
            else
                current_bytes=state-two
            fi
            request_upload
            ;;
        *'"type":"put_artifact_handle"'*)
            upload=$(field "$line" upload)
            encoded=$(printf '%s' "$current_bytes" | base64 | tr -d '\n')
            emit "$host_span" "$host_operation" \
                "{\"type\":\"artifact_upload_chunk\",\"upload\":\"$upload\",\"bytes_base64\":\"$encoded\"}"
            emit "$host_span" "$host_operation" \
                "{\"type\":\"artifact_upload_complete\",\"upload\":\"$upload\"}"
            ;;
        *'"type":"artifact_committed"'*)
            reference=$(printf '%s\n' "$line" | sed -n 's/.*"reference":\(.*\)}}$/\1/p')
            if [ -z "$reference" ]; then
                exit 1
            fi
            if [ "$stage" = reset ]; then
                emit "$parent_span" "$parent" \
                    "{\"type\":\"environment_reset\",\"observation_ref\":$reference}"
                stage=ready
            elif [ "$artifact_phase" = observation ]; then
                first_reference=$reference
                artifact_phase=info
                current_bytes=transition-info
                request_upload
            else
                if [ "$step_count" -eq 1 ]; then
                    terminal=false
                else
                    terminal=true
                fi
                emit "$parent_span" "$parent" \
                    "{\"type\":\"transition\",\"observation_ref\":$first_reference,\"reward\":1.0,\"terminated\":$terminal,\"truncated\":false,\"info_ref\":$reference}"
                stage=ready
            fi
            ;;
    esac
done
"#,
    )
    .expect("write strict rollout adapter executable");
    fs::write(task.join("rollout/reset.json"), "{}\n").expect("write reset input");
    fs::write(task.join("rollout/policy.json"), "policy prompt\n").expect("write policy prompt");
    fs::write(
        task.join("rollout.toml"),
        r#"[environment]
adapter_id = "environment-adapter"
protocol_factory_id = "strict_jsonl"
runtime_provider_id = "strict_supervised"
stepper_factory_id = "supervised_environment"
action_encoder_id = "move_v1"
operation_deadline_ms = 5000
reset_source = "rollout/reset.json"
max_frame_bytes = 4096
max_identifier_bytes = 128
max_json_bytes = 2048
max_json_depth = 4
max_json_array_entries = 8
max_json_object_entries = 8
max_operation_ledger_entries = 16
max_model_call_lineage_entries = 4
max_session_model_call_lineage_entries = 16
max_session_model_call_lineage_bytes = 2048
max_artifact_handles = 4
max_artifact_bytes = 4096

[artifacts]
max_artifacts = 8
max_total_bytes = 16384
max_artifact_bytes = 3072
max_download_handles = 4

[policy]
environment = "counter-v1"
model_binding_id = "primary"
prompt_source = "rollout/policy.json"
max_decision_bytes = 256
horizon = 2
gamma = 0.75

[limits]
max_environment_bytes = 256
max_horizon = 8
max_prompt_bytes = 256
"#,
    )
    .expect("write rollout manifest");
}

fn write_lifecycle(path: &Path) {
    fs::write(
        path,
        serde_json::to_vec(&json!({
            "version": 1,
            "agent_variant": "native-graph",
            "model": {"provider": "provider-default", "model": MODEL},
            "seed": 11,
            "policy": format!("blake3:{}", "a".repeat(64)),
            "runtime": "native:e2e",
            "attempt": "harbor-native-graph-rollout-attempt",
            "budget": {"execution_seconds": 30.0, "verifier_seconds": 30.0},
            "agent_contract": "native_graph",
            "command": ["aiperf-native-graph"],
            "initial_score": {"metric": "reward", "rationale": format!("blake3:{}", "b".repeat(64))},
            "regrade": {"metric": "reward", "rationale": format!("blake3:{}", "c".repeat(64))}
        }))
        .expect("serialize lifecycle"),
    )
    .expect("write lifecycle");
}
