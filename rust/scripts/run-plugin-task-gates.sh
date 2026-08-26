#!/bin/sh
set -eu

task=${1-}
case "$task" in
  1) cargo test -p aiperf-runtime && cargo test -p aiperf-runtime --features engine && cargo test && cargo test -p aiperf-e2e-tests --test plugin_baseline_inventory --test plugin_task_gate_inventory && cargo clippy --all-targets ;;
  2) cargo test -p aiperf-plugin-api && cargo check --workspace && cargo clippy -p aiperf-plugin-api --all-targets ;;
  3) cargo test -p aiperf-bench-tools --all-targets && cargo clippy -p aiperf-bench-tools --all-targets ;;
  4) cargo test -p aiperf-core --all-targets && cargo test -p aiperf-runtime --lib && cargo clippy -p aiperf-core -p aiperf-runtime --all-targets ;;
  5) cargo test -p aiperf-plugin-api --all-targets && cargo clippy -p aiperf-plugin-api --all-targets ;;
  6) cargo test -p aiperf-core -p aiperf-plugin-api -p aiperf-endpoint-sdk -p aiperf-transport-sdk -p aiperf-export-sdk -p aiperf-plugin-test-support --all-targets && cargo test -p aiperf-runtime --features engine && cargo clippy -p aiperf-core -p aiperf-plugin-api -p aiperf-endpoint-sdk -p aiperf-transport-sdk -p aiperf-export-sdk -p aiperf-plugin-test-support -p aiperf-runtime --all-targets --features aiperf-runtime/engine ;;
  7) cargo test -p aiperf-allocator-provider -p aiperf-allocator-shim -p aiperf-plugin-conformance --test allocator && cargo test -p aiperf-cli && cargo build -p aiperf-cli --release && cargo clippy -p aiperf-allocator-provider -p aiperf-allocator-shim -p aiperf-plugin-conformance -p aiperf-cli --all-targets ;;
  8) cargo test -p aiperf-plugin-sdk --all-targets && cargo clippy -p aiperf-plugin-sdk --all-targets ;;
  9) cargo test -p aiperf-plugin-sdk -p aiperf-plugin-sdk-macros --all-targets && cargo clippy -p aiperf-plugin-sdk -p aiperf-plugin-sdk-macros --all-targets ;;
  10) cargo test -p aiperf-plugin-host --test manifest && cargo clippy -p aiperf-plugin-host --all-targets ;;
  11) cargo test -p aiperf-plugin-host --test acquisition --test acquisition_races && cargo clippy -p aiperf-plugin-host --all-targets ;;
  12) cargo test -p aiperf-plugin-host --test static_inspection && cargo clippy -p aiperf-plugin-host --all-targets ;;
  13) cargo test -p aiperf-plugin-host --test discovery --test priority --test authority && cargo clippy -p aiperf-plugin-host --all-targets ;;
  14) cargo test -p aiperf-plugin-host --test loader --test residency --test poison && cargo clippy -p aiperf-plugin-host --all-targets ;;
  15) cargo test -p aiperf-plugin-host --test registration && cargo test -p aiperf-runtime --features engine && cargo clippy -p aiperf-plugin-host -p aiperf-runtime --all-targets --features aiperf-runtime/engine ;;
  16) cargo test -p aiperf-plugin-host --test lock --test bundle --test lock_mismatch --test lock_input && cargo clippy -p aiperf-plugin-host --all-targets ;;
  17) cargo test -p aiperf-cli --test plugin_effect_order --test plugin_commands --test plugin_lock_input --test plugin_abort_contract --test plugin_route_census && cargo test -p aiperf-runtime --features engine && cargo clippy -p aiperf-cli -p aiperf-runtime --all-targets --features aiperf-runtime/engine ;;
  18) cargo test -p aiperf-cli --test plugin_config_open_selection && cargo test -p aiperf-runtime --test plugin_protocol_projection --features engine && cargo clippy -p aiperf-cli -p aiperf-runtime --all-targets --features aiperf-runtime/engine ;;
  19) cargo test -p aiperf-runtime --test plugin_capture_plan --test factory_validation_receipt --features engine && cargo clippy -p aiperf-runtime --all-targets --features engine ;;
  20) cargo test -p aiperf-cli --test plugin_reexec --test plugin_reexec_plan && cargo clippy -p aiperf-cli --all-targets ;;
  21) cargo test -p aiperf-cli --test plugin_cellular --test plugin_kube_slurm_projection --features cellular && cargo test -p aiperf-runtime --features engine,cellular && cargo clippy -p aiperf-cli -p aiperf-runtime --all-targets --features aiperf-cli/cellular,aiperf-runtime/engine,aiperf-runtime/cellular ;;
  22) cargo test -p aiperf-e2e-tests --test plugin_report_provenance && cargo test -p aiperf-runtime --features engine,cellular && cargo clippy -p aiperf-runtime -p aiperf-e2e-tests --all-targets --features aiperf-runtime/engine,aiperf-runtime/cellular ;;
  23) cargo test -p aiperf-e2e-tests --test plugin_cellular_capture --test plugin_exporter_outcomes && cargo test -p aiperf-runtime --features engine,cellular && cargo clippy -p aiperf-runtime -p aiperf-e2e-tests --all-targets --features aiperf-runtime/engine,aiperf-runtime/cellular ;;
  24) cargo test -p aiperf-plugin-export-basic --all-targets && cargo test -p aiperf-e2e-tests --test plugin_export_basic && cargo clippy -p aiperf-plugin-export-basic -p aiperf-e2e-tests --all-targets ;;
  25) cargo test -p aiperf-plugin-export-parquet --all-targets && cargo test -p aiperf-e2e-tests --test plugin_export_parquet && cargo clippy -p aiperf-plugin-export-parquet -p aiperf-e2e-tests --all-targets ;;
  26) cargo test -p aiperf-plugin-export-mlflow --all-targets && cargo test -p aiperf-e2e-tests --test plugin_export_mlflow && cargo clippy -p aiperf-plugin-export-mlflow -p aiperf-e2e-tests --all-targets ;;
  27) cargo test -p aiperf-plugin-export-wandb --all-targets && cargo test -p aiperf-e2e-tests --test plugin_export_wandb && cargo clippy -p aiperf-plugin-export-wandb -p aiperf-e2e-tests --all-targets ;;
  28) cargo test -p aiperf-plugin-export-otel --all-targets && cargo test -p aiperf-e2e-tests --test plugin_telemetry_capture && cargo clippy -p aiperf-plugin-export-otel -p aiperf-e2e-tests --all-targets ;;
  29) cargo test -p aiperf-plugin-endpoints --all-targets && cargo test -p aiperf-e2e-tests --test plugin_endpoints && cargo clippy -p aiperf-plugin-endpoints -p aiperf-e2e-tests --all-targets ;;
  30) cargo test -p aiperf-plugin-endpoints --test grpc_binding && cargo test -p aiperf-e2e-tests --test plugin_endpoint_grpc_override --features grpc && cargo clippy -p aiperf-plugin-endpoints -p aiperf-e2e-tests --all-targets --features aiperf-e2e-tests/grpc ;;
  31) cargo test -p aiperf-plugin-transport-http --all-targets && cargo test -p aiperf-e2e-tests --test plugin_transport_http && cargo clippy -p aiperf-plugin-transport-http -p aiperf-e2e-tests --all-targets ;;
  32) cargo test -p aiperf-plugin-transport-grpc --all-targets && cargo test -p aiperf-e2e-tests --test plugin_transport_grpc --features grpc && cargo clippy -p aiperf-plugin-transport-grpc -p aiperf-e2e-tests --all-targets --features aiperf-e2e-tests/grpc ;;
  33) cargo test -p aiperf-plugin-transport-websocket -p aiperf-plugin-transport-dry-run --all-targets && cargo test -p aiperf-e2e-tests --test plugin_transport_websocket --features websocket && cargo test -p aiperf-dry-run-tests && cargo clippy -p aiperf-plugin-transport-websocket -p aiperf-plugin-transport-dry-run -p aiperf-e2e-tests --all-targets --features aiperf-e2e-tests/websocket ;;
  34) cargo test -p aiperf-plugin-transport-dynosim --all-targets --features dynosim && cargo test -p aiperf-e2e-tests --test plugin_transport_dynosim --features dynosim && cargo clippy -p aiperf-plugin-transport-dynosim -p aiperf-e2e-tests --all-targets --features aiperf-plugin-transport-dynosim/dynosim,aiperf-e2e-tests/dynosim ;;
  35) cargo test -p aiperf-plugin-host --test discovery_authority --test atomic_generations && cargo test -p aiperf-plugin-packaging-tests --test distribution_lifecycle && cargo clippy -p aiperf-plugin-host -p aiperf-plugin-packaging-tests --all-targets ;;
  36) cargo test -p aiperf-plugin-conformance --all-targets && cargo clippy -p aiperf-plugin-conformance --all-targets ;;
  37) cargo test -p aiperf-plugin-packaging-tests --test distribution_census && cargo test -p aiperf-plugin-static-comparator --all-targets && make -C .. native-cli-candidate PLUGIN_RELEASE_CANDIDATE=1 AIPERF_CANDIDATE_FIXTURE=rust/plugin-packaging-tests/fixtures/candidate-generation/fixture.toml && make -C .. bundle-cli-candidate PLUGIN_RELEASE_CANDIDATE=1 AIPERF_CANDIDATE_FIXTURE=rust/plugin-packaging-tests/fixtures/candidate-generation/fixture.toml CLI_FEATURES='--features full' && make -C .. wheel-candidate PLUGIN_RELEASE_CANDIDATE=1 AIPERF_CANDIDATE_FIXTURE=rust/plugin-packaging-tests/fixtures/candidate-generation/fixture.toml && make -C .. native-cli-static-comparator AIPERF_CANDIDATE_FIXTURE=rust/plugin-packaging-tests/fixtures/candidate-generation/fixture.toml AIPERF_STATIC_COMPARATOR_OUTPUT=/cargo-target/plugin-static-fixture && cargo clippy -p aiperf-plugin-packaging-tests -p aiperf-plugin-static-comparator --all-targets ;;
  38) cargo test -p aiperf-plugin-perf --all-targets && cargo clippy -p aiperf-plugin-perf --all-targets ;;
  39) cargo test -p aiperf-plugin-conformance --test no_static_paths && cargo test && cargo test -p aiperf-runtime --features engine && cargo build -p aiperf-cli --features full && cargo clippy --all-targets ;;
  40) cargo test -p aiperf-plugin-sdk --test docs_examples && cargo test -p aiperf-cli --test plugin_commands && cargo test -p aiperf-plugin-conformance --test final_package_and_removal && cargo test && cargo test -p aiperf-runtime --features engine && cargo build -p aiperf-cli --features full && cargo clippy --all-targets ;;
  12-core) ./scripts/run-plugin-task-gates.sh 12 ;;
  12-elf) AIPERF_INSPECTOR_BACKEND=elf cargo test -p aiperf-plugin-host --test static_inspection && ./scripts/run-plugin-task-gates.sh 12 ;;
  12-macho) AIPERF_INSPECTOR_BACKEND=macho cargo test -p aiperf-plugin-host --test static_inspection && ./scripts/run-plugin-task-gates.sh 12 ;;
  12-pe) AIPERF_INSPECTOR_BACKEND=pe cargo test -p aiperf-plugin-host --test static_inspection && ./scripts/run-plugin-task-gates.sh 12 ;;
  33-websocket) ./scripts/run-plugin-task-gates.sh 33 ;;
  33-dry-run) ./scripts/run-plugin-task-gates.sh 33 ;;
  34-dynosim-offline) ./scripts/run-plugin-task-gates.sh 34 ;;
  34-dynosim-online) ./scripts/run-plugin-task-gates.sh 34 ;;
  37a-tooling) ./scripts/run-plugin-task-gates.sh 37 ;;
  37b-package) PLUGIN_RELEASE_CANDIDATE=1 AIPERF_CANDIDATE_WORKTREE=/work-pvc/paper-rig/aiperf-native-plugin-worktrees/unit-39a-dynosim-online AIPERF_CANDIDATE_OUTPUT=/cargo-target/plugin-release-candidate AIPERF_STATIC_COMPARATOR_OUTPUT=/cargo-target/plugin-static-baseline make -C .. native-cli-candidate bundle-cli-candidate wheel-candidate native-cli-static-comparator ;;
  38a-harness) ./scripts/run-plugin-task-gates.sh 38 ;;
  38b-benchmark) cargo run -p aiperf-plugin-perf --release --bin parity -- --inventory benchmarks/plugin-parity.yaml --candidate-root /cargo-target/plugin-release-candidate --baseline-root /cargo-target/plugin-static-baseline --pairs 30 --warmups 5 --bootstrap-resamples 100000 --output ../artifacts/native-plugin-parity ;;
  39a-basic) AIPERF_STATIC_PATH_COMPONENT=basic cargo test -p aiperf-plugin-conformance --test no_static_paths && ./scripts/run-plugin-task-gates.sh 24 ;;
  39a-parquet) AIPERF_STATIC_PATH_COMPONENT=parquet cargo test -p aiperf-plugin-conformance --test no_static_paths && ./scripts/run-plugin-task-gates.sh 25 ;;
  39a-mlflow) AIPERF_STATIC_PATH_COMPONENT=mlflow cargo test -p aiperf-plugin-conformance --test no_static_paths && ./scripts/run-plugin-task-gates.sh 26 ;;
  39a-wandb) AIPERF_STATIC_PATH_COMPONENT=wandb cargo test -p aiperf-plugin-conformance --test no_static_paths && ./scripts/run-plugin-task-gates.sh 27 ;;
  39a-otel) AIPERF_STATIC_PATH_COMPONENT=otel cargo test -p aiperf-plugin-conformance --test no_static_paths && ./scripts/run-plugin-task-gates.sh 28 ;;
  39a-endpoints-grpc-bindings) AIPERF_STATIC_PATH_COMPONENT=endpoints-grpc-bindings cargo test -p aiperf-plugin-conformance --test no_static_paths && ./scripts/run-plugin-task-gates.sh 29 && ./scripts/run-plugin-task-gates.sh 30 ;;
  39a-http) AIPERF_STATIC_PATH_COMPONENT=http cargo test -p aiperf-plugin-conformance --test no_static_paths && ./scripts/run-plugin-task-gates.sh 31 ;;
  39a-grpc) AIPERF_STATIC_PATH_COMPONENT=grpc cargo test -p aiperf-plugin-conformance --test no_static_paths && ./scripts/run-plugin-task-gates.sh 32 ;;
  39a-websocket) AIPERF_STATIC_PATH_COMPONENT=websocket cargo test -p aiperf-plugin-conformance --test no_static_paths && ./scripts/run-plugin-task-gates.sh 33 ;;
  39a-dry-run) AIPERF_STATIC_PATH_COMPONENT=dry-run cargo test -p aiperf-plugin-conformance --test no_static_paths && ./scripts/run-plugin-task-gates.sh 33 ;;
  39a-dynosim-offline) AIPERF_STATIC_PATH_COMPONENT=dynosim-offline cargo test -p aiperf-plugin-conformance --test no_static_paths && ./scripts/run-plugin-task-gates.sh 34 ;;
  39a-dynosim-online) AIPERF_STATIC_PATH_COMPONENT=dynosim-online cargo test -p aiperf-plugin-conformance --test no_static_paths && ./scripts/run-plugin-task-gates.sh 34 ;;
  39b) ./scripts/run-plugin-task-gates.sh 39 ;;
  *) echo "unknown plugin task gate: $task" >&2; exit 64 ;;
esac || exit $?

cargo fmt --check
