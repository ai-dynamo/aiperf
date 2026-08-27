// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Contract tests for the immutable native-plugin parity inventory.

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::{
    collections::BTreeSet,
    fs,
    path::{Component, Path, PathBuf},
    process::{Command, Stdio},
};

use serde::Deserialize;
use serde_json::Value as JsonValue;
use serde_yaml::Value;

const BASE_REVISION: &str = "caa3ff6fcf20ffe36a7704abe16274bedadbb9fb";
const ZERO_DIGEST: &str = "blake3:0000000000000000000000000000000000000000000000000000000000000000";
const TOP_LEVEL_FIELDS: &[&str] = &[
    "schema_version",
    "host_commit",
    "rustc",
    "target",
    "cargo_profile",
    "experiment_identity_json",
    "experiment_identity_digest",
    "source_projection_rule",
    "feature_sets",
    "build_commands",
    "runtime_scenarios",
    "allocation_probe",
    "artifacts",
    "invalid_capture_attempts",
    "raw_samples",
    "canonical_inventory_digest",
    "canonical_inventory_digest_rule",
];
const SCENARIOS: &[&str] = &[
    "http_non_streaming_c1",
    "http_non_streaming_c64",
    "http_streaming_c1",
    "http_streaming_c64",
    "grpc_unary_c1",
    "grpc_unary_c64",
    "grpc_streaming_c1",
    "grpc_streaming_c64",
    "http_streaming_workers4",
    "otlp_disabled_capture",
    "otlp_enabled_capture",
    "exporter_100k",
];
const METRICS: &[&str] = &[
    "successful_requests_per_second",
    "output_tokens_per_second",
    "cpu_nanoseconds_per_successful_request",
    "ttft_p50",
    "ttft_p90",
    "ttft_p99",
    "itl_p50",
    "itl_p90",
    "itl_p99",
];
const EXPORTER_METRICS: &[&str] = &[
    "exporter_nanoseconds_per_record",
    "allocated_bytes_per_successful_request",
    "allocation_count_per_successful_request",
    "ttft_p50",
    "ttft_p90",
    "ttft_p99",
    "itl_p50",
    "itl_p90",
    "itl_p99",
];
fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn validation_root() -> PathBuf {
    std::env::var_os("AIPERF_PLUGIN_BASELINE_VALIDATION_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(repository_root)
}

fn measurement_source_projection(root: &Path) -> Result<Vec<String>, String> {
    let contents = fs::read_to_string(
        root.join("rust/benchmarks/plugin-baseline-measurement-source-projection.txt"),
    )
    .map_err(|error| error.to_string())?;
    if contents.is_empty() || !contents.ends_with('\n') {
        return Err(
            "measurement projection list must be non-empty and newline terminated".to_owned(),
        );
    }
    let paths = contents.lines().map(str::to_owned).collect::<Vec<_>>();
    if paths.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err("measurement projection list must be sorted and unique".to_owned());
    }
    Ok(paths)
}

fn scenario_command_and_shape(name: &str) -> Option<(&'static str, &'static str)> {
    Some(match name {
        "http_non_streaming_c1" => (
            "taskset -c 4-7 aiperf profile --config configs/http-nonstream-c1.yaml --export-level summary",
            "one HTTP JSON response; ITL is zero by definition for a single-response transport",
        ),
        "http_non_streaming_c64" => (
            "taskset -c 4-71 aiperf profile --config configs/http-nonstream-c64.yaml --export-level summary",
            "one HTTP JSON response; ITL is zero by definition for a single-response transport",
        ),
        "http_streaming_c1" => (
            "taskset -c 4-7 aiperf profile --config configs/http-streaming-c1.yaml --export-level summary",
            "exactly 30 HTTP SSE content chunks plus two terminal transport frames",
        ),
        "http_streaming_c64" => (
            "taskset -c 4-71 aiperf profile --config configs/http-streaming-c64.yaml --export-level summary",
            "exactly 30 HTTP SSE content chunks plus two terminal transport frames",
        ),
        "grpc_unary_c1" => (
            "taskset -c 4-7 aiperf profile --config configs/grpc-unary-c1.yaml --export-level summary",
            "one KServe ModelInfer response; ITL is zero by definition for a unary transport",
        ),
        "grpc_unary_c64" => (
            "taskset -c 4-71 aiperf profile --config configs/grpc-unary-c64.yaml --export-level summary",
            "one KServe ModelInfer response; ITL is zero by definition for a unary transport",
        ),
        "grpc_streaming_c1" => (
            "taskset -c 4-7 aiperf profile --config configs/grpc-streaming-c1.yaml --export-level summary",
            "exactly 30 KServe ModelStreamInfer token messages",
        ),
        "grpc_streaming_c64" => (
            "taskset -c 4-71 aiperf profile --config configs/grpc-streaming-c64.yaml --export-level summary",
            "exactly 30 KServe ModelStreamInfer token messages",
        ),
        "http_streaming_workers4" => (
            "taskset -c 4-71 aiperf profile --config configs/http-streaming-workers4.yaml --export-level summary",
            "exactly 32 HTTP transport chunks across four global-dispatch workers",
        ),
        "otlp_disabled_capture" => (
            "taskset -c 4-71 aiperf profile --config configs/otlp-disabled-capture.yaml --export-level summary",
            "exactly 32 HTTP transport chunks; OTLP exporter disabled",
        ),
        "otlp_enabled_capture" => (
            "taskset -c 4-71 aiperf profile --config configs/otlp-enabled-capture.yaml --export-level summary --otel-url http://127.0.0.1:18080/v1/metrics",
            "exactly 32 HTTP transport chunks; OTLP exporter enabled against mock receiver",
        ),
        "exporter_100k" => (
            "taskset -c 4-71 cargo test --locked --release -p aiperf-runtime --features engine --lib engine::records::tests::exporter_capture_allocation_and_duration_baseline -- --exact --nocapture --test-threads=1",
            "one static-calibration member sample comprising 16 sequential repetitions of the same 100000-input-record deterministic corpus; one complete 100000-record artifact retained; processed_records=1600000; no sleep or padding",
        ),
        _ => return None,
    })
}

fn mapping<'a>(value: &'a Value, name: &str) -> Result<&'a serde_yaml::Mapping, String> {
    value
        .as_mapping()
        .ok_or_else(|| format!("{name} must be a mapping"))
}

fn field<'a>(mapping: &'a serde_yaml::Mapping, name: &str) -> Result<&'a Value, String> {
    mapping
        .get(Value::String(name.to_owned()))
        .ok_or_else(|| format!("missing required field `{name}`"))
}

fn text<'a>(mapping: &'a serde_yaml::Mapping, name: &str) -> Result<&'a str, String> {
    let value = field(mapping, name)?
        .as_str()
        .ok_or_else(|| format!("`{name}` must be text"))?;
    if value.is_empty() || value.contains("pending") || value.contains("placeholder") {
        return Err(format!("`{name}` must contain measured evidence"));
    }
    Ok(value)
}

fn digest_file(path: &Path) -> Result<(u64, String), String> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    Ok((
        bytes.len() as u64,
        format!("blake3:{}", blake3::hash(&bytes)),
    ))
}

fn assert_digest(value: &str, name: &str) -> Result<(), String> {
    let digest = value
        .strip_prefix("blake3:")
        .ok_or_else(|| format!("`{name}` lacks blake3 prefix"))?;
    if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("`{name}` is not a BLAKE3 digest"));
    }
    Ok(())
}

fn effective_source_tree_digest(root: &Path, archive: &[u8]) -> Result<String, String> {
    let directory = tempfile::tempdir().map_err(|error| error.to_string())?;
    let tree = directory.path().join("tree");
    fs::create_dir(&tree).map_err(|error| error.to_string())?;
    let archive_path = directory.path().join("base.tar");
    fs::write(&archive_path, archive).map_err(|error| error.to_string())?;
    let extraction = Command::new("tar")
        .args(["-xf"])
        .arg(&archive_path)
        .arg("-C")
        .arg(&tree)
        .output()
        .map_err(|error| format!("extracting baseline archive: {error}"))?;
    if !extraction.status.success() {
        return Err("could not extract baseline source archive".to_owned());
    }
    for relative in measurement_source_projection(root)? {
        let destination = tree.join(&relative);
        fs::create_dir_all(
            destination
                .parent()
                .ok_or_else(|| format!("projection path `{relative}` has no parent"))?,
        )
        .map_err(|error| error.to_string())?;
        fs::copy(root.join(&relative), &destination).map_err(|error| error.to_string())?;
    }
    normalize_source_modes(&tree)?;
    let effective = Command::new("tar")
        .args([
            "--sort=name",
            "--mtime=@0",
            "--owner=0",
            "--group=0",
            "--numeric-owner",
            "-C",
        ])
        .arg(&tree)
        .args(["-cf", "-", "."])
        .output()
        .map_err(|error| format!("archiving effective source tree: {error}"))?;
    if !effective.status.success() {
        return Err("could not archive effective source tree".to_owned());
    }
    Ok(format!("blake3:{}", blake3::hash(&effective.stdout)))
}

#[cfg(unix)]
fn normalize_source_modes(directory: &Path) -> Result<(), String> {
    fs::set_permissions(directory, fs::Permissions::from_mode(0o755))
        .map_err(|error| error.to_string())?;
    for entry in fs::read_dir(directory).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path).map_err(|error| error.to_string())?;
        if metadata.file_type().is_symlink() {
            continue;
        }
        if metadata.is_dir() {
            normalize_source_modes(&path)?;
        } else if metadata.is_file() {
            let mode = if metadata.permissions().mode() & 0o111 == 0 {
                0o644
            } else {
                0o755
            };
            fs::set_permissions(&path, fs::Permissions::from_mode(mode))
                .map_err(|error| error.to_string())?;
        }
    }
    Ok(())
}

#[cfg(not(unix))]
fn normalize_source_modes(_directory: &Path) -> Result<(), String> {
    Err("source-mode normalization requires Unix permissions".to_owned())
}

fn replace_digest_field(
    contents: &str,
    field_name: &str,
    replacement: &str,
    expected_count: usize,
) -> Result<String, String> {
    let mut found = 0;
    let mut output = String::with_capacity(contents.len());
    for line in contents.split_inclusive('\n') {
        let Some(field_index) = line.find(field_name) else {
            output.push_str(line);
            continue;
        };
        let Some(relative_start) = line[field_index..].find("blake3:") else {
            output.push_str(line);
            continue;
        };
        let value_start = field_index + relative_start;
        let value_end = value_start + ZERO_DIGEST.len();
        if value_end > line.len() {
            return Err(format!("`{field_name}` contains a truncated digest"));
        }
        assert_digest(&line[value_start..value_end], field_name)?;
        output.push_str(&line[..value_start]);
        output.push_str(replacement);
        output.push_str(&line[value_end..]);
        found += 1;
    }
    if found != expected_count {
        return Err(format!(
            "expected {expected_count} `{field_name}` digests, found {found}"
        ));
    }
    Ok(output)
}

fn zero_canonical_digests(contents: &str) -> Result<String, String> {
    let zeroed = replace_digest_field(
        contents,
        "canonical_inventory_digest",
        ZERO_DIGEST,
        SCENARIOS.len() + 2,
    )?;
    replace_digest_field(&zeroed, "experiment_identity_digest", ZERO_DIGEST, 1)
}

fn refresh_identity_digest(contents: &str) -> Result<String, String> {
    let document: Value = serde_yaml::from_str(contents).map_err(|error| error.to_string())?;
    let inventory = mapping(&document, "inventory")?;
    let identity_json = field(inventory, "experiment_identity_json")?
        .as_str()
        .ok_or_else(|| "experiment_identity_json must be text".to_owned())?;
    let identity_digest = format!("blake3:{}", blake3::hash(identity_json.as_bytes()));
    replace_digest_field(contents, "experiment_identity_digest", &identity_digest, 1)
}

fn refresh_canonical_digests(contents: &str) -> Result<String, String> {
    let zeroed = zero_canonical_digests(contents)?;
    let digest = format!("blake3:{}", blake3::hash(zeroed.as_bytes()));
    let with_canonical = replace_digest_field(
        &zeroed,
        "canonical_inventory_digest",
        &digest,
        SCENARIOS.len() + 2,
    )?;
    refresh_identity_digest(&with_canonical)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EvidenceManifest {
    schema_version: u8,
    files: Vec<EvidenceManifestFile>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EvidenceManifestFile {
    path: String,
    bytes: u64,
    blake3: String,
}

fn validate_manifest(manifest: &JsonValue) -> Result<usize, String> {
    let manifest: EvidenceManifest =
        serde_json::from_value(manifest.clone()).map_err(|error| error.to_string())?;
    if manifest.schema_version != 1 {
        return Err(format!(
            "unsupported evidence manifest schema {}",
            manifest.schema_version
        ));
    }
    if manifest.files.is_empty() {
        return Err("manifest is empty".to_owned());
    }
    let mut previous: Option<&str> = None;
    for entry in &manifest.files {
        let path = entry.path.as_str();
        let declared = Path::new(path);
        if path.is_empty()
            || declared.is_absolute()
            || declared
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
            || path
                .split('/')
                .any(|component| component.is_empty() || matches!(component, "." | ".."))
            || path.contains('\\')
        {
            return Err(format!("unsafe manifest path: {path}"));
        }
        assert_digest(&entry.blake3, path)?;
        let _ = entry.bytes;
        if previous.is_some_and(|previous| previous >= path) {
            return Err("manifest paths must be unique and sorted".to_owned());
        }
        previous = Some(path);
    }
    Ok(manifest.files.len())
}

fn validate_bundle_locator(locator: &JsonValue, raw: &serde_yaml::Mapping) -> Result<(), String> {
    let locator = locator
        .as_object()
        .ok_or_else(|| "bundle locator must be a JSON object".to_owned())?;
    if locator.get("schema_version").and_then(JsonValue::as_u64) != Some(1) {
        return Err("bundle locator schema must be 1".to_owned());
    }
    let locator_text = |name: &str| {
        locator
            .get(name)
            .and_then(JsonValue::as_str)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| format!("bundle locator lacks `{name}`"))
    };
    let raw_bundle = mapping(field(raw, "bundle")?, "bundle")?;
    let raw_manifest = mapping(field(raw, "manifest")?, "manifest")?;
    for (locator_name, raw_name) in [
        ("staged_path", "staged_path"),
        ("blake3", "blake3"),
        ("recommended_release_tag", "release_tag"),
        ("repository", "repository"),
    ] {
        if locator_text(locator_name)? != text(raw_bundle, raw_name)? {
            return Err(format!("bundle locator `{locator_name}` mismatch"));
        }
    }
    if locator.get("bytes").and_then(JsonValue::as_u64) != field(raw_bundle, "bytes")?.as_u64() {
        return Err("bundle locator length mismatch".to_owned());
    }
    if locator_text("manifest_path")? != text(raw_manifest, "path")?
        || locator_text("manifest_blake3")? != text(raw_manifest, "blake3")?
        || locator.get("manifest_bytes").and_then(JsonValue::as_u64)
            != field(raw_manifest, "bytes")?.as_u64()
    {
        return Err("bundle locator manifest mismatch".to_owned());
    }
    let asset_name = locator_text("asset_name")?;
    if Path::new(locator_text("staged_path")?)
        .file_name()
        .and_then(|name| name.to_str())
        != Some(asset_name)
    {
        return Err("bundle locator asset name mismatch".to_owned());
    }
    let stable_url = format!(
        "{}/releases/download/{}/{}",
        locator_text("repository")?,
        locator_text("recommended_release_tag")?,
        asset_name
    );
    if locator_text("stable_url")? != stable_url {
        return Err("bundle locator stable URL mismatch".to_owned());
    }
    match locator_text("publication_status")? {
        "published_and_verified" => {
            let generation = locator_text("recommended_release_tag")?
                .strip_prefix("native-plugin-baseline-caa3ff6f-")
                .and_then(|suffix| suffix.strip_suffix("-final"))
                .filter(|generation| {
                    !generation.is_empty()
                        && generation
                            .bytes()
                            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
                })
                .ok_or_else(|| "published locator release tag is not canonical".to_owned())?;
            if matches!(
                generation,
                "review1"
                    | "review1b"
                    | "review1c"
                    | "review1d"
                    | "review1e"
                    | "review1f"
                    | "review1g"
                    | "review1h"
            ) {
                return Err(format!(
                    "published locator generation `{generation}` is explicitly invalidated"
                ));
            }
            if text(raw, "admission_status")? != format!("published_verified_{generation}")
                || locator_text("archive_verification_status")?
                    != "downloaded_extracted_manifest_verified"
            {
                return Err("published locator lacks retrieval verification".to_owned());
            }
        }
        status => {
            return Err(format!(
                "canonical bundle must be published_and_verified, observed `{status}`"
            ));
        }
    }
    Ok(())
}

fn validate_inventory(contents: &str, root: &Path) -> Result<(), String> {
    let document: Value = serde_yaml::from_str(contents).map_err(|error| error.to_string())?;
    let inventory = mapping(&document, "inventory")?;
    let fields = inventory
        .keys()
        .map(|field| {
            field
                .as_str()
                .ok_or_else(|| "top-level field names must be text".to_owned())
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    if fields != TOP_LEVEL_FIELDS.iter().copied().collect() {
        return Err("top-level field set mismatch".to_owned());
    }
    if field(inventory, "schema_version")?.as_u64() != Some(1) {
        return Err("schema_version must be 1".to_owned());
    }
    if text(inventory, "host_commit")? != BASE_REVISION {
        return Err("host_commit is not the frozen baseline".to_owned());
    }
    let raw_samples = mapping(field(inventory, "raw_samples")?, "raw_samples")?;
    let is_prepublication =
        text(raw_samples, "admission_status")? == "prepublication_expected_failure";

    let identity_json = text(inventory, "experiment_identity_json")?;
    let identity: JsonValue =
        serde_json::from_str(identity_json).map_err(|error| error.to_string())?;
    let identity = identity
        .as_object()
        .ok_or_else(|| "experiment identity must be a JSON object".to_owned())?;
    for name in [
        "baseline_revision",
        "baseline_source_tree_blake3",
        "baseline_cargo_lock_blake3",
        "measurement_source_projection_blake3",
        "measurement_source_projection_list_blake3",
        "rustc",
        "cargo",
        "rustc_sysroot",
        "target",
        "cargo_profile",
        "cpu_model",
        "cpu_family",
        "cpu_model_number",
        "cpu_stepping",
        "effective_cargo_lock_blake3",
        "effective_source_tree_blake3",
        "kernel",
        "allocator_provider",
        "frequency_governor",
        "affinity_isolation",
        "firmware",
        "memory_topology",
        "microcode",
        "admitted_environment_blake3",
        "harness_blake3",
        "owned_command_helper_blake3",
        "exporter_observable_policy_blake3",
        "exporter_corpus_blake3",
        "exporter_receipt_schema_version",
        "exporter_observable_policy_schema_version",
        "exporter_corpus_records",
        "exporter_sample_repetitions",
        "exporter_processed_records",
        "exporter_retained_artifact_records",
        "exporter_pair_id",
        "exporter_member",
        "exporter_attempt_ordinal",
        "exporter_planned_schedule",
        "canonical_inventory_digest",
        "mock_server_blake3",
    ] {
        let value = identity.get(name).and_then(JsonValue::as_str);
        if value.is_none_or(str::is_empty) {
            return Err(format!("experiment identity is missing `{name}`"));
        }
    }
    if identity["baseline_revision"] != BASE_REVISION {
        return Err("identity baseline_revision mismatch".to_owned());
    }
    for name in ["rustc", "target", "cargo_profile"] {
        if text(inventory, name)? != identity[name] {
            return Err(format!(
                "top-level `{name}` does not match experiment identity"
            ));
        }
    }
    let identity_digest = text(inventory, "experiment_identity_digest")?;
    assert_digest(identity_digest, "experiment_identity_digest")?;
    let computed_identity = format!("blake3:{}", blake3::hash(identity_json.as_bytes()));
    if identity_digest != computed_identity {
        return Err(format!(
            "experiment identity digest mismatch: expected {identity_digest}, computed {computed_identity}"
        ));
    }

    let archive = Command::new("git")
        .args(["archive", "--format=tar", BASE_REVISION])
        .current_dir(repository_root())
        .output()
        .map_err(|error| format!("git archive: {error}"))?;
    if !archive.status.success() {
        return Err("could not recompute baseline source projection".to_owned());
    }
    let source_digest = format!("blake3:{}", blake3::hash(&archive.stdout));
    if identity["baseline_source_tree_blake3"] != source_digest {
        return Err("baseline source-tree digest mismatch".to_owned());
    }
    let effective_source_tree_digest = effective_source_tree_digest(root, &archive.stdout)?;
    if identity["effective_source_tree_blake3"] != effective_source_tree_digest {
        return Err(format!(
            "effective source-tree digest mismatch: expected {}, computed {effective_source_tree_digest}",
            identity["effective_source_tree_blake3"]
        ));
    }
    let lock = Command::new("git")
        .args(["show", &format!("{BASE_REVISION}:rust/Cargo.lock")])
        .current_dir(repository_root())
        .output()
        .map_err(|error| format!("git show Cargo.lock: {error}"))?;
    if !lock.status.success()
        || identity["baseline_cargo_lock_blake3"]
            != format!("blake3:{}", blake3::hash(&lock.stdout))
    {
        return Err("baseline Cargo.lock digest mismatch".to_owned());
    }
    let (_, effective_lock_digest) = digest_file(&root.join("rust/Cargo.lock"))?;
    if identity["effective_cargo_lock_blake3"] != effective_lock_digest {
        return Err("effective Cargo.lock digest mismatch".to_owned());
    }
    let projection_paths = measurement_source_projection(root)?;
    let projection_list = format!("{}\n", projection_paths.join("\n"));
    if identity["measurement_source_projection_list_blake3"]
        != format!("blake3:{}", blake3::hash(projection_list.as_bytes()))
    {
        return Err("measurement source projection list mismatch".to_owned());
    }
    let projection = Command::new("tar")
        .args([
            "--sort=name",
            "--mtime=@0",
            "--owner=0",
            "--group=0",
            "--numeric-owner",
            "-C",
        ])
        .arg(root)
        .args(["-cf", "-"])
        .args(&projection_paths)
        .output()
        .map_err(|error| format!("measurement projection tar: {error}"))?;
    let computed_projection = format!("blake3:{}", blake3::hash(&projection.stdout));
    if !projection.status.success()
        || identity["measurement_source_projection_blake3"] != computed_projection
    {
        return Err(format!(
            "measurement source projection digest mismatch: computed {computed_projection}"
        ));
    }

    let feature_sets = field(inventory, "feature_sets")?
        .as_sequence()
        .ok_or_else(|| "feature_sets must be a sequence".to_owned())?;
    let expected_feature_sets = [
        ("default", "aiperf-cli", &["default"][..]),
        ("engine", "aiperf-runtime", &["engine"][..]),
        ("grpc", "aiperf-cli", &["default", "grpc"][..]),
        ("parquet", "aiperf-cli", &["default", "parquet"][..]),
        ("dynosim", "aiperf-cli", &["default", "dynosim"][..]),
        ("full", "aiperf-cli", &["full"][..]),
    ];
    if feature_sets.len() != expected_feature_sets.len() {
        return Err("feature-set matrix is incomplete".to_owned());
    }
    for (value, (name, package, features)) in feature_sets.iter().zip(expected_feature_sets) {
        let feature_set = mapping(value, "feature set")?;
        let actual_features = field(feature_set, "features")?
            .as_sequence()
            .ok_or_else(|| format!("{name} features must be a sequence"))?
            .iter()
            .map(|feature| {
                feature
                    .as_str()
                    .ok_or_else(|| format!("{name} feature must be text"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if text(feature_set, "name")? != name
            || text(feature_set, "package")? != package
            || actual_features != features
        {
            return Err(format!("{name} feature-set definition mismatch"));
        }
    }

    let builds = mapping(field(inventory, "build_commands")?, "build_commands")?;
    let compared_artifacts = identity
        .get("compared_artifact_digests")
        .and_then(JsonValue::as_object)
        .ok_or_else(|| "experiment identity is missing `compared_artifact_digests`".to_owned())?;
    if compared_artifacts
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>()
        != ["default", "engine", "grpc", "parquet", "dynosim", "full"]
            .into_iter()
            .collect()
    {
        return Err("compared artifact set mismatch".to_owned());
    }
    for name in ["default", "engine", "grpc", "parquet", "dynosim", "full"] {
        let build = mapping(field(builds, name)?, name)?;
        let expected_command = match name {
            "default" => "cargo build --locked -p aiperf-cli --release",
            "engine" => "cargo build --locked -p aiperf-runtime --release --features engine",
            "grpc" => "cargo build --locked -p aiperf-cli --release --features grpc",
            "parquet" => "cargo build --locked -p aiperf-cli --release --features parquet",
            "dynosim" => "cargo build --locked -p aiperf-cli --release --features dynosim",
            "full" => "cargo build --locked -p aiperf-cli --release --features full",
            _ => unreachable!(),
        };
        if text(build, "command")? != expected_command {
            return Err(format!("{name} build command mismatch"));
        }
        if text(build, "first_build_kind")? != "isolated_clean_target" {
            return Err(format!("{name} first build was not isolated-clean"));
        }
        if !text(build, "target_dir")?.starts_with("/cargo-target/") {
            return Err(format!("{name} target is outside /cargo-target"));
        }
        for timing in ["first_build_nanoseconds", "second_build_nanoseconds"] {
            if field(build, timing)?
                .as_u64()
                .is_none_or(|value| value == 0)
            {
                return Err(format!("{name} `{timing}` must be positive"));
            }
        }
        if field(build, "artifact_bytes")?
            .as_u64()
            .is_none_or(|bytes| bytes == 0)
        {
            return Err(format!("{name} artifact length must be positive"));
        }
        for digest in ["artifact_digest", "first_log_digest", "second_log_digest"] {
            assert_digest(text(build, digest)?, digest)?;
        }
        if compared_artifacts.get(name).and_then(JsonValue::as_str)
            != Some(text(build, "artifact_digest")?)
        {
            return Err(format!("{name} compared artifact identity mismatch"));
        }
    }

    let scenarios = field(inventory, "runtime_scenarios")?
        .as_sequence()
        .ok_or_else(|| "runtime_scenarios must be a sequence".to_owned())?;
    let mut names = BTreeSet::new();
    for value in scenarios {
        let scenario = mapping(value, "scenario")?;
        let name = text(scenario, "name")?;
        if !names.insert(name) {
            return Err(format!("duplicate runtime scenario `{name}`"));
        }
        let (expected_command, expected_shape) = scenario_command_and_shape(name)
            .ok_or_else(|| format!("unknown runtime scenario `{name}`"))?;
        if text(scenario, "command")? != expected_command
            || text(scenario, "response_shape")? != expected_shape
        {
            return Err(format!("{name} command/response-shape mismatch"));
        }
        let is_c1 = name.ends_with("_c1");
        let expected_budget = if name == "exporter_100k" {
            1_600_000
        } else if is_c1 {
            1_000
        } else {
            64_000
        };
        if field(scenario, "request_budget")?.as_u64() != Some(expected_budget) {
            return Err(format!("{name} request budget mismatch"));
        }
        if name == "exporter_100k"
            && (field(scenario, "corpus_records")?.as_u64() != Some(100_000)
                || field(scenario, "sample_repetitions")?.as_u64() != Some(16)
                || field(scenario, "processed_records")?.as_u64() != Some(1_600_000)
                || field(scenario, "retained_artifact_records")?.as_u64() != Some(100_000)
                || text(scenario, "observable_kind")? != "artifact_tree"
                || text(scenario, "pair_id")? != "task1-static-calibration"
                || text(scenario, "member")? != "static"
                || field(scenario, "attempt_ordinal")?.as_u64() != Some(0))
        {
            return Err("exporter_100k repetition contract mismatch".to_owned());
        }
        if name == "exporter_100k" && !is_prepublication {
            let receipts = field(scenario, "repetition_receipts")?
                .as_sequence()
                .ok_or_else(|| "exporter repetition receipts must be a sequence".to_owned())?;
            if receipts.len() != 16 {
                return Err("exporter repetition receipt count mismatch".to_owned());
            }
            let expected_identity = text(scenario, "experiment_identity_blake3")?;
            assert_digest(expected_identity, "exporter pre-run experiment identity")?;
            let expected_corpus = identity["exporter_corpus_blake3"]
                .as_str()
                .ok_or_else(|| "identity exporter corpus digest is missing".to_owned())?;
            let engine_artifact = text(
                mapping(
                    field(
                        mapping(field(inventory, "build_commands")?, "build_commands")?,
                        "engine",
                    )?,
                    "engine build",
                )?,
                "artifact_digest",
            )?;
            let mut comparison = None;
            let mut active_duration_ns = 0_u64;
            for (ordinal, receipt) in receipts.iter().enumerate() {
                let receipt = mapping(receipt, "exporter repetition receipt")?;
                let fields = receipt
                    .keys()
                    .map(|key| {
                        key.as_str()
                            .ok_or_else(|| "exporter receipt field must be text".to_owned())
                    })
                    .collect::<Result<BTreeSet<_>, _>>()?;
                if fields
                    != [
                        "schema_version",
                        "experiment_identity_blake3",
                        "attempt_ordinal",
                        "scenario_id",
                        "pair_id",
                        "member",
                        "repetition_ordinal",
                        "corpus_blake3",
                        "processed_records",
                        "observable_kind",
                        "raw_observable_blake3",
                        "comparison_observable_blake3",
                        "provenance_receipt_blake3",
                        "active_duration_ns",
                        "build_artifact_blake3",
                        "build_receipt_blake3",
                    ]
                    .into_iter()
                    .collect()
                {
                    return Err(format!("exporter receipt {ordinal} field set mismatch"));
                }
                let duration = field(receipt, "active_duration_ns")?
                    .as_u64()
                    .filter(|duration| *duration > 0)
                    .ok_or_else(|| {
                        format!("exporter receipt {ordinal} duration is not positive")
                    })?;
                if field(receipt, "schema_version")?.as_u64() != Some(1)
                    || text(receipt, "experiment_identity_blake3")? != expected_identity
                    || field(receipt, "attempt_ordinal")?.as_u64() != Some(0)
                    || text(receipt, "scenario_id")? != "exporter_100k"
                    || text(receipt, "pair_id")? != "task1-static-calibration"
                    || text(receipt, "member")? != "static"
                    || field(receipt, "repetition_ordinal")?.as_u64() != Some(ordinal as u64)
                    || text(receipt, "corpus_blake3")? != expected_corpus
                    || field(receipt, "processed_records")?.as_u64() != Some(100_000)
                    || text(receipt, "observable_kind")? != "artifact_tree"
                    || text(receipt, "build_artifact_blake3")? != engine_artifact
                {
                    return Err(format!("exporter receipt {ordinal} binding mismatch"));
                }
                for digest in [
                    "raw_observable_blake3",
                    "comparison_observable_blake3",
                    "provenance_receipt_blake3",
                    "build_receipt_blake3",
                ] {
                    assert_digest(text(receipt, digest)?, digest)?;
                }
                let observed_comparison = text(receipt, "comparison_observable_blake3")?;
                if comparison.is_some_and(|expected| expected != observed_comparison) {
                    return Err("exporter comparison observable changed".to_owned());
                }
                comparison = Some(observed_comparison);
                active_duration_ns = active_duration_ns
                    .checked_add(duration)
                    .ok_or_else(|| "exporter duration sum overflow".to_owned())?;
            }
            if field(scenario, "active_duration_ns")?.as_u64() != Some(active_duration_ns) {
                return Err("exporter duration sum mismatch".to_owned());
            }
            let receipt_json = serde_json::to_value(receipts).map_err(|error| error.to_string())?;
            let mut receipt_bytes =
                serde_json::to_vec(receipt_json.as_array().expect("receipt JSON is an array"))
                    .map_err(|error| error.to_string())?;
            receipt_bytes.push(b'\n');
            if text(scenario, "repetition_receipts_blake3")?
                != format!("blake3:{}", blake3::hash(&receipt_bytes))
            {
                return Err("exporter receipt vector digest mismatch".to_owned());
            }
        }
        let expected_core_assignment = if name == "http_streaming_workers4" {
            "mock=0-3;four client workers=4-71"
        } else if name == "otlp_enabled_capture" {
            "mock+OTLP collector=0-3;client=4-71"
        } else if name == "exporter_100k" {
            "exporter probe=4-71;no mock process"
        } else if is_c1 {
            "mock=0-3;client=4-7"
        } else {
            "mock=0-3;client=4-71"
        };
        let expected_mock_placement = if name == "exporter_100k" {
            "not applicable; deterministic in-process exporter probe"
        } else {
            "co-located paper-rig process on a disjoint physical-core set"
        };
        if text(scenario, "core_assignment")? != expected_core_assignment
            || text(scenario, "mock_placement")? != expected_mock_placement
            || text(scenario, "estimator")? != "paired_hyndman_fan_type_7_max_degradation_bootstrap"
            || field(scenario, "bootstrap_seed")?.as_u64() != Some(20_260_826)
            || text(scenario, "invalidation_classifier")?
                != "host_reboot|affinity_loss|mock_death_unrelated_to_member;max_replacement_pairs=5"
            || text(scenario, "firmware")? != "Google Google 07/08/2026"
            || text(scenario, "memory_topology")?
                != "3 NUMA nodes;node0=0-23,72-95;node1=24-47,96-119;node2=48-71,120-143"
        {
            return Err(format!("{name} frozen comparison contract mismatch"));
        }
        for digest in [
            "artifact_digest",
            "process_log_digest",
            "harness_blake3",
            "mock_server_blake3",
        ] {
            assert_digest(text(scenario, digest)?, digest)?;
        }
        if text(scenario, "harness_blake3")? != identity["harness_blake3"]
            || text(scenario, "mock_server_blake3")? != identity["mock_server_blake3"]
        {
            return Err(format!("{name} harness/mock identity mismatch"));
        }
        let expected_primary =
            if name.starts_with("http_non_streaming") || name.starts_with("grpc_unary") {
                "successful_requests_per_second"
            } else if name.starts_with("otlp_") {
                "cpu_nanoseconds_per_successful_request"
            } else if name == "exporter_100k" {
                "exporter_nanoseconds_per_record"
            } else {
                "output_tokens_per_second"
            };
        if text(scenario, "primary_metric")? != expected_primary {
            return Err(format!("{name} primary metric mismatch"));
        }
        let expected_direction = if matches!(
            expected_primary,
            "successful_requests_per_second" | "output_tokens_per_second"
        ) {
            "dynamic_over_static"
        } else {
            "static_over_dynamic"
        };
        if text(scenario, "ratio_direction")? != expected_direction {
            return Err(format!("{name} ratio direction mismatch"));
        }
        let measured: BTreeSet<_> = field(scenario, "measured_metrics")?
            .as_sequence()
            .ok_or_else(|| format!("{name} measured_metrics must be a sequence"))?
            .iter()
            .map(|metric| {
                metric
                    .as_str()
                    .ok_or_else(|| format!("{name} measured metric must be text"))
            })
            .collect::<Result<_, _>>()?;
        let expected_measured: BTreeSet<_> = if name == "exporter_100k" {
            EXPORTER_METRICS.iter().copied().collect()
        } else {
            METRICS.iter().copied().collect()
        };
        if measured != expected_measured {
            return Err(format!("{name} measured metric set mismatch"));
        }
        if field(scenario, "warmups")?.as_u64() != Some(5)
            || field(scenario, "retained_pairs")?.as_u64() != Some(30)
        {
            return Err(format!("{name} pairing contract mismatch"));
        }
        let minimum = field(scenario, "minimum_duration_seconds")?
            .as_f64()
            .ok_or_else(|| format!("{name} minimum duration missing"))?;
        let observation = mapping(field(scenario, "baseline_observation")?, "observation")?;
        let duration = field(observation, "duration_seconds")?
            .as_f64()
            .ok_or_else(|| format!("{name} observed duration missing"))?;
        if minimum < 30.0 || duration < minimum || !duration.is_finite() {
            return Err(format!("{name} did not meet its duration contract"));
        }
        let expected_observation_metrics = if name == "exporter_100k" {
            EXPORTER_METRICS
        } else {
            METRICS
        };
        for metric in expected_observation_metrics {
            let value = field(observation, metric)?
                .as_f64()
                .ok_or_else(|| format!("{name} missing observation `{metric}`"))?;
            if !value.is_finite() {
                return Err(format!("{name} has non-finite `{metric}`"));
            }
        }
        if name.starts_with("http_streaming") || name.starts_with("otlp_") {
            if field(scenario, "deterministic_response_chunks")?.as_u64() != Some(32) {
                return Err(format!("{name} must freeze exactly 32 response chunks"));
            }
        }
    }
    if names != SCENARIOS.iter().copied().collect() {
        return Err("scenario matrix is incomplete".to_owned());
    }

    let allocations = mapping(field(inventory, "allocation_probe")?, "allocation_probe")?;
    for (path, expected_iterations) in [
        ("endpoint_formatting", 10_000),
        ("transport_dispatch", 10_000),
        ("response_reduction", 10_000),
        ("full_successful_request", 10_000),
        ("exporter_capture", 1_600_000),
    ] {
        let sample = mapping(field(allocations, path)?, path)?;
        if field(sample, "iterations")?.as_u64() != Some(expected_iterations) {
            return Err(format!("{path} iteration count mismatch"));
        }
        for metric in ["allocations_per_request", "allocated_bytes_per_request"] {
            if field(sample, metric)?
                .as_f64()
                .is_none_or(|value| !value.is_finite())
            {
                return Err(format!("{path} missing `{metric}`"));
            }
        }
    }
    if field(
        mapping(
            field(allocations, "response_reduction")?,
            "response_reduction",
        )?,
        "chunks_per_response",
    )?
    .as_u64()
        != Some(32)
    {
        return Err("response reduction must fold exactly 32 chunks per request".to_owned());
    }
    let exporter = mapping(field(allocations, "exporter_capture")?, "exporter_capture")?;
    for (field_name, expected) in [
        ("corpus_records", 100_000),
        ("sample_repetitions", 16),
        ("processed_records", 1_600_000),
        ("retained_artifact_records", 100_000),
    ] {
        if field(exporter, field_name)?.as_u64() != Some(expected) {
            return Err(format!("exporter allocation `{field_name}` mismatch"));
        }
    }
    if field(exporter, "exporter_interval_nanoseconds")?
        .as_u64()
        .is_none_or(|nanoseconds| nanoseconds < 30_000_000_000)
    {
        return Err("exporter-only allocation interval must last at least 30 seconds".to_owned());
    }

    let artifacts = mapping(field(inventory, "artifacts")?, "artifacts")?;
    for (_, value) in artifacts {
        let artifact = mapping(value, "artifact")?;
        let path = root.join(text(artifact, "path")?);
        let (bytes, digest) = digest_file(&path)?;
        if field(artifact, "bytes")?.as_u64() != Some(bytes) || text(artifact, "blake3")? != digest
        {
            return Err(format!("tracked artifact mismatch: {}", path.display()));
        }
    }

    let invalid_attempts = field(inventory, "invalid_capture_attempts")?
        .as_sequence()
        .ok_or_else(|| "invalid_capture_attempts must be a sequence".to_owned())?;
    if invalid_attempts.is_empty() {
        return Err("invalid capture attempt ledger is empty".to_owned());
    }
    for attempt in invalid_attempts {
        let attempt = mapping(attempt, "invalid capture attempt")?;
        for required in ["generation", "status", "reason"] {
            text(attempt, required)?;
        }
        if !matches!(text(attempt, "status")?, "invalid" | "superseded") {
            return Err("invalid capture attempt has an admissible status".to_owned());
        }
    }

    let canonical = format!(
        "blake3:{}",
        blake3::hash(zero_canonical_digests(contents)?.as_bytes())
    );
    if text(inventory, "canonical_inventory_digest")? != canonical {
        return Err(format!(
            "top-level canonical inventory digest mismatch: computed {canonical}"
        ));
    }
    if identity["canonical_inventory_digest"] != canonical {
        return Err("experiment identity canonical inventory digest mismatch".to_owned());
    }
    for scenario in scenarios {
        if text(mapping(scenario, "scenario")?, "canonical_inventory_digest")? != canonical {
            return Err("scenario canonical inventory digest mismatch".to_owned());
        }
    }

    let raw = mapping(field(inventory, "raw_samples")?, "raw_samples")?;
    if text(raw, "admission_status")? == "prepublication_expected_failure" {
        return Err("canonical raw evidence is not published_and_verified".to_owned());
    }
    for name in ["manifest", "bundle"] {
        let artifact = mapping(field(raw, name)?, name)?;
        assert_digest(text(artifact, "blake3")?, &format!("raw {name}"))?;
        if field(artifact, "bytes")?
            .as_u64()
            .is_none_or(|value| value == 0)
        {
            return Err(format!("raw {name} length missing"));
        }
    }
    let manifest_path = root.join(text(mapping(field(raw, "manifest")?, "manifest")?, "path")?);
    let manifest: JsonValue =
        serde_json::from_slice(&fs::read(&manifest_path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    let file_count = validate_manifest(&manifest)?;
    if field(mapping(field(raw, "manifest")?, "manifest")?, "file_count")?.as_u64()
        != Some(file_count as u64)
    {
        return Err("manifest file count mismatch".to_owned());
    }
    let locator_path = root.join(text(
        mapping(field(artifacts, "bundle_locator")?, "bundle_locator")?,
        "path",
    )?);
    let locator: JsonValue =
        serde_json::from_slice(&fs::read(locator_path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    validate_bundle_locator(&locator, raw)?;

    Ok(())
}

#[test]
fn baseline_inventory_is_complete_and_self_authenticating() {
    let root = validation_root();
    let contents = fs::read_to_string(root.join("rust/benchmarks/plugin-parity.yaml"))
        .expect("baseline inventory must be readable");
    let document: Value = serde_yaml::from_str(&contents).expect("inventory parses");
    let inventory = mapping(&document, "inventory").expect("inventory is a mapping");
    let raw = mapping(
        field(inventory, "raw_samples").expect("raw_samples is present"),
        "raw_samples",
    )
    .expect("raw_samples is a mapping");
    if text(raw, "admission_status") == Ok("prepublication_expected_failure") {
        assert_eq!(
            validate_inventory(&contents, &root)
                .expect_err("prepublication inventory must not satisfy canonical validation"),
            "canonical raw evidence is not published_and_verified"
        );
    } else {
        validate_inventory(&contents, &root).unwrap_or_else(|error| panic!("{error}"));
    }
}

#[test]
fn experiment_identity_contains_a_digest_not_a_prose_inventory_binding() {
    let contents = fs::read_to_string(repository_root().join("rust/benchmarks/plugin-parity.yaml"))
        .expect("baseline inventory must be readable");
    let document: Value = serde_yaml::from_str(&contents).expect("inventory parses");
    let inventory = mapping(&document, "inventory").expect("inventory is a mapping");
    let identity: JsonValue = serde_json::from_str(
        text(inventory, "experiment_identity_json").expect("identity JSON is present"),
    )
    .expect("identity JSON parses");
    let binding = identity["canonical_inventory_digest"]
        .as_str()
        .expect("identity contains canonical inventory digest");
    assert_digest(binding, "identity canonical inventory digest")
        .expect("identity inventory binding is a digest");
    assert!(identity.get("inventory_digest_binding").is_none());
}

#[test]
fn exporter_scenario_is_bound_to_the_authoritative_probe() {
    let contents = fs::read_to_string(repository_root().join("rust/benchmarks/plugin-parity.yaml"))
        .expect("baseline inventory must be readable");
    let document: Value = serde_yaml::from_str(&contents).expect("inventory parses");
    let inventory = mapping(&document, "inventory").expect("inventory is a mapping");
    let scenarios = field(inventory, "runtime_scenarios")
        .and_then(|value| {
            value
                .as_sequence()
                .ok_or_else(|| "runtime_scenarios must be a sequence".to_owned())
        })
        .expect("runtime scenarios are present");
    let exporter = scenarios
        .iter()
        .map(|value| mapping(value, "scenario").expect("scenario is a mapping"))
        .find(|scenario| text(scenario, "name") == Ok("exporter_100k"))
        .expect("exporter scenario is present");
    let (expected_command, expected_shape) =
        scenario_command_and_shape("exporter_100k").expect("exporter contract exists");

    assert_eq!(text(exporter, "command"), Ok(expected_command));
    assert_eq!(text(exporter, "response_shape"), Ok(expected_shape));
    assert_eq!(
        field(exporter, "request_budget").and_then(|value| {
            value
                .as_u64()
                .ok_or_else(|| "request_budget must be an integer".to_owned())
        }),
        Ok(1_600_000)
    );
    let allocation = mapping(
        field(inventory, "allocation_probe").expect("allocation probe exists"),
        "allocation probe",
    )
    .expect("allocation probe is a mapping");
    let exporter_allocation = mapping(
        field(allocation, "exporter_capture").expect("exporter allocation exists"),
        "exporter allocation",
    )
    .expect("exporter allocation is a mapping");
    assert_eq!(
        field(exporter_allocation, "iterations").and_then(|value| {
            value
                .as_u64()
                .ok_or_else(|| "iterations must be an integer".to_owned())
        }),
        Ok(1_600_000)
    );
    for (field_name, expected) in [
        ("corpus_records", 100_000),
        ("sample_repetitions", 16),
        ("processed_records", 1_600_000),
        ("retained_artifact_records", 100_000),
    ] {
        assert_eq!(
            field(exporter, field_name).and_then(|value| {
                value
                    .as_u64()
                    .ok_or_else(|| format!("{field_name} must be an integer"))
            }),
            Ok(expected)
        );
    }
}

#[test]
fn capture_harness_bounds_every_long_lived_command_site() {
    let capture =
        fs::read_to_string(repository_root().join("rust/scripts/capture-plugin-baseline.sh"))
            .expect("capture harness must be readable");
    let ownership =
        fs::read_to_string(repository_root().join("rust/scripts/plugin-baseline-owned-command.sh"))
            .expect("shared ownership helper must be readable");
    let script = format!("{ownership}\n{capture}");
    for required in [
        "run_owned()",
        "run_owned_with_stdin()",
        "terminate_owned_group()",
        "run_owned_from \"$timeout_seconds\" \"$label\" /dev/null \"$@\"",
        "setsid \"$@\" <\"$stdin_path\" &",
        "mktemp \"$(dirname \"$failure_ledger\")/.capture-stdin.XXXXXX\"",
        "chmod 0600 \"$owned_stdin_file\"",
        "run_owned 7200 \"build-$name-clean\"",
        "run_owned 7200 \"build-$name-second\"",
        "run_owned 7200 mock-server-build",
        "run_owned 600 cargo-metadata",
        "run_owned 600 cargo-tree-workspace",
        "run_owned 600 cargo-tree-cli",
        "run_owned_with_stdin 930 \"runtime-$name\"",
        "run_owned_with_stdin 60 \"runtime-report-$name\"",
        "setsid taskset -c 0-3 \"$mock\"",
        "run_owned_with_stdin 60 mock-readiness",
        "terminate_owned_group \"$mock_pid\" mock-server",
        "run_owned 3600 exporter-probe",
        "run_owned_with_stdin 60 exporter-observation",
        "run_owned 3600 allocation-probe",
        "run_owned 3600 response-reduction-probe",
        "run_owned 300 evidence-manifest",
        "run_owned 300 evidence-verify",
        "run_owned 1800 evidence-bundle",
        "run_owned 1800 bundle-extract",
        "run_owned 600 bundle-verify",
        "run_owned 300 locator",
        "run_owned 300 locator-verify",
        "failure_ledger=$(dirname \"$output_root\")/capture-failures.txt",
        ">>\"$failure_ledger\"",
        "subprocess.Popen",
        "timeout=900",
        "completed.terminate()",
        "completed.kill()",
        "kill -TERM \"-$pid\"",
        "kill -KILL \"-$pid\"",
        "kill -0 \"-$pid\"",
        "record_baseline_failure \"$label\" \"timeout after ${timeout_seconds}s\"",
        "record_baseline_failure \"$label\" \"leader exited while descendant survived\"",
        "require_output nonempty git-archive",
        "require_output nonempty extract-base",
        "require_output executable evidence-tool-build",
        "require_output nonempty effective-source-archive",
        "require_output nonempty cargo-metadata",
        "require_output nonempty cargo-tree-workspace",
        "require_output nonempty cargo-tree-cli",
        "require_output nonempty topology",
        "require_output nonempty \"build-$name-clean\"",
        "require_output nonempty \"build-$name-second\"",
        "require_output nonempty \"build-$name-artifact\"",
        "require_output executable mock-server-build",
        "require_output nonempty write-configs",
        "require_output nonempty \"runtime-$name-resource\"",
        "require_output file \"runtime-$name-process-log\"",
        "require_output nonempty \"runtime-$name-report\"",
        "require_output nonempty exporter-probe",
        "require_output nonempty exporter-observation",
        "require_output nonempty allocation-probe",
        "require_output nonempty response-reduction-probe",
        "require_output nonempty evidence-manifest",
        "require_output nonempty evidence-bundle",
        "require_output nonempty bundle-extract",
        "require_output nonempty locator",
    ] {
        assert!(
            script.contains(required),
            "capture harness lacks owned-process contract `{required}`"
        );
    }
    assert!(
        !script.contains("scenario_pid"),
        "runtime scenarios must use the same generic owned-process helper"
    );
    let cleanup = script
        .split("cleanup() {")
        .nth(1)
        .and_then(|suffix| suffix.split("}\ntrap cleanup EXIT").next())
        .expect("cleanup function has a stable boundary");
    assert!(
        cleanup
            .find("terminate_owned_group \"$owned_pid\"")
            .expect("cleanup terminates the active owned command")
            < cleanup
                .find("rm -f -- \"$owned_stdin_file\"")
                .expect("cleanup removes the owned stdin spool after teardown"),
        "owned stdin spool must remain until owned-command teardown completes"
    );
    assert!(
        cleanup
            .find("rm -f -- \"$owned_stdin_file\"")
            .expect("cleanup removes the owned stdin spool")
            < cleanup
                .find("release_baseline_lock")
                .expect("cleanup releases lock"),
        "capture lock must remain held until stdin-spool cleanup completes"
    );
}

#[test]
fn capture_and_refresh_share_exclusive_owned_commands_and_capacity_checked_tmp() {
    let root = repository_root();
    let capture = fs::read_to_string(root.join("rust/scripts/capture-plugin-baseline.sh"))
        .expect("capture harness must be readable");
    let refresh =
        fs::read_to_string(root.join("rust/scripts/refresh-plugin-baseline-inventory.sh"))
            .expect("refresh helper must be readable");
    let ownership = fs::read_to_string(root.join("rust/scripts/plugin-baseline-owned-command.sh"))
        .expect("shared ownership helper must be readable");

    for required in [
        "AIPERF_PLUGIN_BASELINE_LOCK",
        "acquire_baseline_lock",
        "run_owned()",
        "terminate_owned_group()",
        "record_baseline_failure()",
        "setsid \"$@\" <\"$stdin_path\" &",
    ] {
        assert!(
            ownership.contains(required),
            "ownership helper lacks {required}"
        );
    }
    for (name, script) in [("capture", &capture), ("refresh", &refresh)] {
        assert!(
            script.contains("plugin-baseline-owned-command.sh"),
            "{name} does not load the shared ownership contract"
        );
        assert!(
            script.contains("acquire_baseline_lock"),
            "{name} does not acquire the shared singleton"
        );
    }
    for required in [
        "required_free_bytes",
        "require_free_bytes",
        "AIPERF_PLUGIN_CAPTURE_TMPDIR",
        "export TMPDIR",
    ] {
        assert!(capture.contains(required), "capture lacks {required}");
    }
    for required in [
        "AIPERF_PLUGIN_REFRESH_TMPDIR",
        "require_free_bytes",
        "export TMPDIR",
    ] {
        assert!(refresh.contains(required), "refresh lacks {required}");
    }
}

#[test]
fn source_projection_rule_names_the_included_validator_boundary() {
    let contents = fs::read_to_string(repository_root().join("rust/benchmarks/plugin-parity.yaml"))
        .expect("baseline inventory must be readable");
    let document: Value = serde_yaml::from_str(&contents).expect("inventory parses");
    let inventory = mapping(&document, "inventory").expect("inventory is a mapping");
    let rule = text(inventory, "source_projection_rule").expect("projection rule is text");
    assert!(rule.contains("plugin_baseline_inventory.rs is included"));
    assert!(!rule.contains("validators, and generated evidence are excluded"));
}

#[test]
fn baseline_refresh_helper_has_explicit_truthful_modes_and_fixed_point() {
    let root = repository_root();
    let helper = fs::read_to_string(root.join("rust/scripts/refresh-plugin-baseline-inventory.sh"))
        .expect("tracked baseline refresh helper exists");
    for required in [
        "pre-capture",
        "post-capture",
        "postpublication",
        "AIPERF_PLUGIN_CAPTURE_ROOT",
        "AIPERF_PLUGIN_REFRESH_ROOT",
        "CARGO_TARGET_DIR",
        "refresh-contract",
        "mktemp -d",
        ".aiperf-plugin-refresh-owned",
        "cargo metadata --locked --format-version 1",
        "cargo tree --locked --workspace --edges normal,build --prefix depth",
        "cargo tree --locked -p aiperf-cli --edges normal,build --prefix depth",
        "generated-topology.json",
        "candidate_inventory",
        "AIPERF_PLUGIN_BASELINE_VALIDATION_ROOT",
        "publish-baseline",
        "cmp",
    ] {
        assert!(
            helper.contains(required),
            "refresh helper lacks `{required}`"
        );
    }
    assert!(
        !helper.contains("python"),
        "structured refresh must not use Python text surgery"
    );
    assert!(
        !helper.contains("refresh_root=$refresh_parent/task1-$generation-refresh"),
        "refresh cleanup must not recursively remove an env-derived fixed path"
    );
    let projection = fs::read_to_string(
        root.join("rust/benchmarks/plugin-baseline-measurement-source-projection.txt"),
    )
    .expect("tracked projection list exists");
    assert!(projection.contains("rust/scripts/refresh-plugin-baseline-inventory.sh\n"));
    assert!(projection.contains("rust/benchmarks/plugin-baseline-invalidations.tsv\n"));
    assert!(
        projection.contains("rust/benchmarks/plugin-baseline-measurement-source-projection.txt\n")
    );
}

#[test]
fn capture_harness_authors_machine_refresh_receipts() {
    let script =
        fs::read_to_string(repository_root().join("rust/scripts/capture-plugin-baseline.sh"))
            .expect("capture harness is readable");
    for required in [
        "plugin-baseline-measurement-source-projection.txt",
        "plugin-baseline-invalidations.tsv",
        "experiment-identity.json",
        "invalidations.tsv",
        "package-topology.json",
    ] {
        assert!(
            script.contains(required),
            "capture harness does not author `{required}`"
        );
    }
}

#[test]
fn capture_harness_preserves_owned_command_stdin_and_output_contract() {
    let directory = tempfile::tempdir().expect("temporary stdin self-test directory is created");
    let lock = directory.path().join("capture.lock");
    let output = directory.path().join("owned-output.txt");
    let status = Command::new("sh")
        .arg(repository_root().join("rust/scripts/capture-plugin-baseline.sh"))
        .arg("--stdin-self-test")
        .arg(&lock)
        .arg(&output)
        .status()
        .expect("capture stdin self-test starts");
    assert!(
        status.success(),
        "owned stdin/output contract failed: {status}"
    );
    assert_eq!(
        fs::read_to_string(&output).expect("owned output exists"),
        "owned-stdin-preserved\n"
    );
    assert_eq!(
        fs::read_to_string(output.with_extension("txt.empty")).expect("empty-stdin receipt exists"),
        "empty\n"
    );
    assert_eq!(
        fs::read_to_string(output.with_extension("txt.multiline"))
            .expect("multiline stdin output exists"),
        "first-line\nsecond-line\n"
    );
    assert!(
        fs::read_dir(directory.path())
            .expect("self-test directory is readable")
            .all(|entry| !entry
                .expect("self-test entry is readable")
                .file_name()
                .to_string_lossy()
                .starts_with(".capture-stdin.")),
        "stdin spool file survived owned-command teardown"
    );
    assert!(
        !lock.exists(),
        "stdin self-test releases the singleton lock"
    );
}

#[cfg(unix)]
#[test]
fn capture_harness_removes_stdin_spool_after_timeout() {
    let directory = tempfile::tempdir().expect("temporary timeout self-test directory is created");
    let lock = directory.path().join("capture.lock");
    let status = Command::new("sh")
        .arg(repository_root().join("rust/scripts/capture-plugin-baseline.sh"))
        .arg("--stdin-timeout-self-test")
        .arg(&lock)
        .status()
        .expect("capture stdin timeout self-test starts");
    assert_eq!(status.code(), Some(124), "timeout status is retained");
    assert!(!lock.exists(), "timeout releases the singleton lock");
    assert!(
        fs::read_dir(directory.path())
            .expect("timeout self-test directory is readable")
            .all(|entry| !entry
                .expect("timeout self-test entry is readable")
                .file_name()
                .to_string_lossy()
                .starts_with(".capture-stdin.")),
        "stdin spool file survived timeout teardown"
    );
    assert!(
        fs::read_to_string(directory.path().join("capture-failures.txt"))
            .expect("timeout failure ledger exists")
            .contains("stdin-timeout\ttimeout after 1s"),
        "timeout reason is retained"
    );
}

#[cfg(unix)]
#[test]
fn capture_harness_removes_stdin_spool_after_signal_and_group_teardown() {
    let directory = tempfile::tempdir().expect("temporary signal self-test directory is created");
    let lock = directory.path().join("capture.lock");
    let pidfile = directory.path().join("signal-child.pid");
    let mut child = Command::new("sh")
        .arg(repository_root().join("rust/scripts/capture-plugin-baseline.sh"))
        .arg("--stdin-signal-self-test")
        .arg(&lock)
        .arg(&pidfile)
        .spawn()
        .expect("capture stdin signal self-test starts");
    let observation_deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !pidfile.exists() && std::time::Instant::now() < observation_deadline {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let descendant = fs::read_to_string(&pidfile)
        .expect("signal child publishes its PID")
        .trim()
        .to_owned();
    assert!(lock.is_dir(), "signal child runs while singleton is held");
    assert!(
        fs::read_dir(directory.path())
            .expect("signal self-test directory is readable")
            .any(|entry| entry
                .expect("signal self-test entry is readable")
                .file_name()
                .to_string_lossy()
                .starts_with(".capture-stdin.")),
        "stdin spool must remain while the owned group is alive"
    );
    assert!(
        Command::new("kill")
            .args(["-TERM", &child.id().to_string()])
            .status()
            .expect("capture signal is delivered")
            .success(),
        "capture process accepts TERM"
    );
    let completion_deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    let status = loop {
        if let Some(status) = child.try_wait().expect("signal status can be queried") {
            break status;
        }
        assert!(
            std::time::Instant::now() < completion_deadline,
            "stdin signal self-test exceeded bounded teardown"
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    };
    assert_eq!(status.code(), Some(143), "TERM status is retained");
    assert!(!lock.exists(), "signal teardown releases singleton lock");
    assert!(
        !Command::new("kill")
            .args(["-0", descendant.as_str()])
            .status()
            .expect("post-signal liveness probe runs")
            .success(),
        "signal child is gone before lock release"
    );
    assert!(
        fs::read_dir(directory.path())
            .expect("signal self-test directory is readable")
            .all(|entry| !entry
                .expect("signal self-test entry is readable")
                .file_name()
                .to_string_lossy()
                .starts_with(".capture-stdin.")),
        "stdin spool file survived signal teardown"
    );
}

#[cfg(unix)]
#[test]
fn capture_harness_retains_lock_until_orphaned_group_is_gone() {
    let directory = tempfile::tempdir().expect("temporary self-test directory is created");
    let lock = directory.path().join("capture.lock");
    let pidfile = directory.path().join("descendant.pid");
    let script = repository_root().join("rust/scripts/capture-plugin-baseline.sh");
    let mut child = Command::new("sh")
        .arg(script)
        .arg("--ownership-self-test")
        .arg(&lock)
        .arg(&pidfile)
        .spawn()
        .expect("capture ownership self-test starts");

    let observation_deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !pidfile.exists() && std::time::Instant::now() < observation_deadline {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let descendant = fs::read_to_string(&pidfile)
        .expect("adversarial descendant publishes its PID")
        .trim()
        .to_owned();
    assert!(
        lock.is_dir(),
        "capture lock is held while descendant exists"
    );
    assert!(
        Command::new("kill")
            .args(["-0", descendant.as_str()])
            .status()
            .expect("descendant liveness probe runs")
            .success(),
        "adversarial descendant is alive while the lock is held"
    );

    let completion_deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    let status = loop {
        if let Some(status) = child.try_wait().expect("self-test status can be queried") {
            break status;
        }
        assert!(
            std::time::Instant::now() < completion_deadline,
            "capture ownership self-test exceeded its bounded teardown"
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    };
    assert!(
        status.success(),
        "capture ownership self-test failed: {status}"
    );
    assert!(
        !lock.exists(),
        "capture lock is released after full group teardown"
    );
    assert!(
        !Command::new("kill")
            .args(["-0", descendant.as_str()])
            .status()
            .expect("post-teardown liveness probe runs")
            .success(),
        "adversarial descendant is gone before lock release"
    );
}

#[cfg(unix)]
#[test]
fn capture_harness_retains_post_seal_failure_as_non_root() {
    let directory = tempfile::tempdir().expect("temporary self-test directory is created");
    fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o777))
        .expect("non-root self-test directory is accessible");
    let lock = directory.path().join("capture.lock");
    let pidfile = directory.path().join("descendant.pid");
    let sealed = directory.path().join("sealed-evidence");
    let ledger = directory.path().join("capture-failures.txt");
    let script = repository_root().join("rust/scripts/capture-plugin-baseline.sh");
    let uid = Command::new("id")
        .arg("-u")
        .output()
        .expect("current UID can be queried");
    let mut command = if uid.stdout == b"0\n" {
        let mut command = Command::new("setpriv");
        command.args(["--reuid=65534", "--regid=65534", "--clear-groups", "sh"]);
        command
    } else {
        Command::new("sh")
    };
    let mut child = command
        .arg(script)
        .arg("--post-seal-failure-self-test")
        .arg(&lock)
        .arg(&pidfile)
        .arg(&sealed)
        .arg(&ledger)
        .spawn()
        .expect("non-root post-seal self-test starts");

    let observation_deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !pidfile.exists() && std::time::Instant::now() < observation_deadline {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let descendant = fs::read_to_string(&pidfile)
        .expect("post-seal adversarial descendant publishes its PID")
        .trim()
        .to_owned();
    assert!(
        lock.is_dir(),
        "capture lock is held while descendant exists"
    );
    assert_eq!(
        fs::metadata(&sealed)
            .expect("sealed evidence exists")
            .permissions()
            .mode()
            & 0o222,
        0,
        "evidence is read-only before the helper failure"
    );
    assert!(
        Command::new("kill")
            .args(["-0", descendant.as_str()])
            .status()
            .expect("descendant liveness probe runs")
            .success(),
        "post-seal descendant is alive while the lock is held"
    );

    let completion_deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    let status = loop {
        if let Some(status) = child.try_wait().expect("self-test status can be queried") {
            break status;
        }
        assert!(
            std::time::Instant::now() < completion_deadline,
            "post-seal ownership self-test exceeded its bounded teardown"
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    };
    assert_eq!(
        status.code(),
        Some(42),
        "original helper failure is retained"
    );
    let failures = fs::read_to_string(&ledger).expect("writable sibling failure ledger exists");
    assert!(failures.contains("post-seal-helper\tleader exited while descendant survived"));
    assert!(failures.contains("post-seal-helper\texit status 42"));
    assert!(
        !lock.exists(),
        "capture lock is released after full group teardown"
    );
    assert!(
        !Command::new("kill")
            .args(["-0", descendant.as_str()])
            .status()
            .expect("post-teardown liveness probe runs")
            .success(),
        "post-seal descendant is gone before lock release"
    );
}

#[cfg(unix)]
#[test]
fn capture_harness_preserves_status_when_failure_ledger_is_unwritable() {
    let directory = tempfile::tempdir().expect("temporary self-test directory is created");
    fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o777))
        .expect("non-root self-test directory is accessible");
    let lock = directory.path().join("capture.lock");
    let pidfile = directory.path().join("descendant.pid");
    let sealed = directory.path().join("sealed-evidence");
    let ledger = sealed.join("uncreatable-control/capture-failures.txt");
    let stderr_path = directory.path().join("capture.stderr");
    let stderr = fs::File::create(&stderr_path).expect("stderr receipt is created");
    fs::set_permissions(&stderr_path, fs::Permissions::from_mode(0o666))
        .expect("non-root stderr receipt is writable");
    let script = repository_root().join("rust/scripts/capture-plugin-baseline.sh");
    let uid = Command::new("id")
        .arg("-u")
        .output()
        .expect("current UID can be queried");
    let mut command = if uid.stdout == b"0\n" {
        let mut command = Command::new("setpriv");
        command.args(["--reuid=65534", "--regid=65534", "--clear-groups", "sh"]);
        command
    } else {
        Command::new("sh")
    };
    let mut child = command
        .arg(script)
        .arg("--post-seal-failure-self-test")
        .arg(&lock)
        .arg(&pidfile)
        .arg(&sealed)
        .arg(&ledger)
        .stderr(Stdio::from(stderr))
        .spawn()
        .expect("unwritable-ledger self-test starts");

    let observation_deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !pidfile.exists() && std::time::Instant::now() < observation_deadline {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let descendant = fs::read_to_string(&pidfile)
        .expect("unwritable-ledger descendant publishes its PID")
        .trim()
        .to_owned();
    assert!(
        lock.is_dir(),
        "capture lock is held while descendant exists"
    );
    assert!(
        Command::new("kill")
            .args(["-0", descendant.as_str()])
            .status()
            .expect("descendant liveness probe runs")
            .success(),
        "unwritable-ledger descendant is alive while the lock is held"
    );

    let completion_deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    let status = loop {
        if let Some(status) = child.try_wait().expect("self-test status can be queried") {
            break status;
        }
        assert!(
            std::time::Instant::now() < completion_deadline,
            "unwritable-ledger self-test exceeded its bounded teardown"
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    };
    assert_eq!(
        status.code(),
        Some(42),
        "ledger I/O must not mask status 42"
    );
    let stderr = fs::read_to_string(&stderr_path).expect("fallback stderr is readable");
    assert!(stderr.contains(
        "post-seal-helper\tleader exited while descendant survived\tcould not persist capture failure ledger"
    ));
    assert!(
        stderr
            .contains("post-seal-helper\texit status 42\tcould not persist capture failure ledger")
    );
    assert!(
        !ledger.exists(),
        "the sealed-tree ledger path remains unwritable"
    );
    assert!(
        !lock.exists(),
        "capture lock is released only after group teardown"
    );
    assert!(
        !Command::new("kill")
            .args(["-0", descendant.as_str()])
            .status()
            .expect("post-teardown liveness probe runs")
            .success(),
        "unwritable-ledger descendant is gone before lock release"
    );
}

#[test]
fn baseline_inventory_rejects_contract_mutations() {
    let root = repository_root();
    let contents = fs::read_to_string(root.join("rust/benchmarks/plugin-parity.yaml"))
        .expect("baseline inventory must be readable");
    for (needle, replacement, expected_error) in [
        (
            "rustc: rustc 1.98.0",
            "removed_rustc: rustc 1.98.0",
            "top-level field set mismatch",
        ),
        (
            "target: x86_64-unknown-linux-gnu",
            "target: wrong-target",
            "top-level `target` does not match experiment identity",
        ),
        (
            "cargo_profile: release",
            "cargo_profile: debug",
            "top-level `cargo_profile` does not match experiment identity",
        ),
        (
            "\"rustc_sysroot\":",
            "\"removed_rustc_sysroot\":",
            "experiment identity is missing `rustc_sysroot`",
        ),
        (
            "\"microcode\":",
            "\"removed_microcode\":",
            "experiment identity is missing `microcode`",
        ),
        (
            "\"compared_artifact_digests\":",
            "\"removed_compared_artifact_digests\":",
            "experiment identity is missing `compared_artifact_digests`",
        ),
        (
            "first_build_nanoseconds: ",
            "first_build_nanoseconds: 0 # ",
            "`first_build_nanoseconds` must be positive",
        ),
        (
            "request_budget: 1000",
            "request_budget: 0",
            "request budget mismatch",
        ),
        (
            "core_assignment:",
            "removed_core_assignment:",
            "missing required field `core_assignment`",
        ),
        (
            "mock_placement:",
            "removed_mock_placement:",
            "missing required field `mock_placement`",
        ),
        (
            "estimator:",
            "removed_estimator:",
            "missing required field `estimator`",
        ),
        (
            "bootstrap_seed: 20260826",
            "bootstrap_seed: 1",
            "frozen comparison contract mismatch",
        ),
        (
            "primary_metric:",
            "removed_primary_metric:",
            "missing required field `primary_metric`",
        ),
        (
            "ratio_direction:",
            "removed_ratio_direction:",
            "missing required field `ratio_direction`",
        ),
        (
            "measured_metrics:",
            "removed_measured_metrics:",
            "missing required field `measured_metrics`",
        ),
        (
            "measured_metrics:\n    - successful_requests_per_second",
            "measured_metrics:\n    - unexpected_metric\n    - successful_requests_per_second",
            "measured metric set mismatch",
        ),
        (
            "invalidation_classifier:",
            "removed_invalidation_classifier:",
            "missing required field `invalidation_classifier`",
        ),
        (
            "harness_blake3:",
            "removed_harness_blake3:",
            "missing required field `harness_blake3`",
        ),
        (
            "mock_server_blake3:",
            "removed_mock_server_blake3:",
            "missing required field `mock_server_blake3`",
        ),
        (
            "firmware:",
            "removed_firmware:",
            "missing required field `firmware`",
        ),
        (
            "memory_topology:",
            "removed_memory_topology:",
            "missing required field `memory_topology`",
        ),
        (
            "response_shape:",
            "removed_response_shape:",
            "missing required field `response_shape`",
        ),
        ("warmups: 5", "warmups: 4", "pairing contract mismatch"),
        (
            "retained_pairs: 30",
            "retained_pairs: 29",
            "pairing contract mismatch",
        ),
        (
            "duration_seconds: 35.349796724",
            "duration_seconds: null",
            "observed duration missing",
        ),
        (
            "ttft_p50: 35.2508165",
            "ttft_p50: null",
            "missing observation `ttft_p50`",
        ),
        (
            "artifact_digest: blake3:",
            "artifact_digest: blake3:f",
            "is not a BLAKE3 digest",
        ),
        (
            "readme: {path: artifacts/native-plugin-baseline/README.md",
            "readme: {path: artifacts/native-plugin-baseline/missing-README.md",
            "missing-README.md",
        ),
    ] {
        let mutated = contents.replacen(needle, replacement, 1);
        assert_ne!(mutated, contents, "mutation needle `{needle}` was absent");
        let mutated = refresh_canonical_digests(&mutated)
            .unwrap_or_else(|error| panic!("refreshing `{needle}`: {error}"));
        let error = validate_inventory(&mutated, &root)
            .expect_err(&format!("mutation `{needle}` unexpectedly passed"));
        assert!(
            error.contains(expected_error),
            "mutation `{needle}` failed for `{error}`, expected `{expected_error}`"
        );
    }

    let stale_digest = contents.replacen(
        "canonical_inventory_digest: blake3:",
        "canonical_inventory_digest: blake3:f",
        1,
    );
    assert!(validate_inventory(&stale_digest, &root).is_err());

    let document: Value = serde_yaml::from_str(&contents).expect("inventory parses");
    let identity_json = text(
        mapping(&document, "inventory").expect("inventory is a mapping"),
        "experiment_identity_json",
    )
    .expect("identity JSON is present");
    let identity: JsonValue = serde_json::from_str(identity_json).expect("identity JSON parses");
    let bound = identity["canonical_inventory_digest"]
        .as_str()
        .expect("identity canonical digest is present");
    let mutated_identity = contents.replacen(
        &format!("\"canonical_inventory_digest\": \"{bound}\""),
        &format!("\"canonical_inventory_digest\": \"{}\"", ZERO_DIGEST),
        1,
    );
    assert_ne!(mutated_identity, contents);
    let mutated_identity =
        refresh_identity_digest(&mutated_identity).expect("refresh identity digest");
    let error = validate_inventory(&mutated_identity, &root)
        .expect_err("mutated nested canonical digest unexpectedly passed");
    assert_eq!(
        error,
        "experiment identity canonical inventory digest mismatch"
    );
}

#[test]
fn baseline_manifest_rejects_missing_length_and_bad_digest() {
    let path = repository_root().join("artifacts/native-plugin-baseline/evidence-manifest.json");
    let original: JsonValue =
        serde_json::from_slice(&fs::read(path).expect("manifest must be readable"))
            .expect("manifest must be JSON");
    assert!(validate_manifest(&original).is_ok());

    let mut missing_length = original.clone();
    missing_length["files"][0]
        .as_object_mut()
        .expect("manifest entry is an object")
        .remove("bytes");
    assert!(validate_manifest(&missing_length).is_err());

    let mut bad_digest = original;
    bad_digest["files"][0]["blake3"] = JsonValue::String("blake3:not-a-digest".to_owned());
    assert!(validate_manifest(&bad_digest).is_err());
}

#[test]
fn baseline_manifest_rejects_unsafe_or_noncanonical_schema() {
    let digest = "blake3:1111111111111111111111111111111111111111111111111111111111111111";
    let valid = serde_json::json!({
        "schema_version": 1,
        "files": [{"path": "identity/sample.txt", "bytes": 1, "blake3": digest}],
    });
    assert!(validate_manifest(&valid).is_ok());

    for invalid in [
        serde_json::json!({"schema_version": 2, "files": valid["files"]}),
        serde_json::json!({"schema_version": 1, "unexpected": true, "files": valid["files"]}),
        serde_json::json!({"schema_version": 1, "files": [{"path": "../outside", "bytes": 1, "blake3": digest}]}),
        serde_json::json!({"schema_version": 1, "files": [{"path": "/absolute", "bytes": 1, "blake3": digest}]}),
        serde_json::json!({"schema_version": 1, "files": [{"path": "identity//sample.txt", "bytes": 1, "blake3": digest}]}),
        serde_json::json!({"schema_version": 1, "files": [{"path": "identity/./sample.txt", "bytes": 1, "blake3": digest}]}),
        serde_json::json!({"schema_version": 1, "files": [{"path": "identity/sample.txt", "bytes": 1, "blake3": digest, "extra": true}]}),
        serde_json::json!({"schema_version": 1, "files": [valid["files"][0], valid["files"][0]]}),
    ] {
        assert!(
            validate_manifest(&invalid).is_err(),
            "malformed manifest unexpectedly passed: {invalid}"
        );
    }
}

#[test]
fn package_topology_dependency_claims_match_cargo_metadata() {
    let root = validation_root();
    let topology: JsonValue = serde_json::from_slice(
        &fs::read(root.join("artifacts/native-plugin-baseline/package-topology.json"))
            .expect("package topology must be readable"),
    )
    .expect("package topology must be JSON");
    let metadata = Command::new("cargo")
        .args(["metadata", "--locked", "--format-version", "1", "--no-deps"])
        .current_dir(root.join("rust"))
        .output()
        .expect("cargo metadata runs");
    assert!(metadata.status.success(), "cargo metadata must succeed");
    let metadata: JsonValue =
        serde_json::from_slice(&metadata.stdout).expect("cargo metadata must be JSON");
    assert_eq!(
        topology
            .as_object()
            .expect("topology is an object")
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>(),
        [
            "schema_version",
            "generation",
            "host_commit",
            "rustc",
            "target",
            "cargo_profile",
            "measurement",
            "workspace_packages",
        ]
        .into_iter()
        .collect()
    );
    assert_eq!(topology["schema_version"], 1);
    assert_eq!(topology["host_commit"], BASE_REVISION);
    assert_eq!(
        topology["measurement"]["commands"],
        serde_json::json!([
            "cargo metadata --locked --format-version 1",
            "cargo tree --locked --workspace --edges normal,build --prefix depth",
            "cargo tree --locked -p aiperf-cli --edges normal,build --prefix depth",
        ])
    );
    let packages = topology["workspace_packages"]
        .as_array()
        .expect("topology packages must be an array");
    let workspace_names = metadata["packages"]
        .as_array()
        .expect("metadata packages must be an array")
        .iter()
        .map(|package| package["name"].as_str().expect("package name is text"))
        .collect::<BTreeSet<_>>();
    assert_eq!(packages.len(), workspace_names.len());
    for package in metadata["packages"]
        .as_array()
        .expect("metadata packages must be an array")
    {
        let name = package["name"].as_str().expect("package name is text");
        let claim = packages
            .iter()
            .find(|claim| claim["name"] == name)
            .unwrap_or_else(|| panic!("topology omits `{name}`"));
        assert_eq!(
            claim["direct_dependency_count"].as_u64(),
            Some(
                package["dependencies"]
                    .as_array()
                    .expect("dependencies are an array")
                    .len() as u64
            ),
            "stale dependency census for `{name}`"
        );
        assert_eq!(
            claim["version"], package["version"],
            "stale version for `{name}`"
        );
        let mut expected_dependencies = package["dependencies"]
            .as_array()
            .expect("dependencies are an array")
            .iter()
            .map(|dependency| {
                (
                    dependency["name"]
                        .as_str()
                        .expect("dependency name is text"),
                    dependency["kind"].as_str().unwrap_or("normal"),
                    workspace_names.contains(
                        dependency["name"]
                            .as_str()
                            .expect("dependency name is text"),
                    ),
                )
            })
            .collect::<Vec<_>>();
        expected_dependencies.sort();
        let claimed_dependencies = claim["direct_dependencies"]
            .as_array()
            .expect("claimed dependencies are an array")
            .iter()
            .map(|dependency| {
                (
                    dependency["name"].as_str().expect("claimed name is text"),
                    dependency["kind"].as_str().expect("claimed kind is text"),
                    dependency["is_workspace"]
                        .as_bool()
                        .expect("claimed workspace marker is boolean"),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            claimed_dependencies, expected_dependencies,
            "stale dependencies for `{name}`"
        );
        let mut expected_kind_counts = std::collections::BTreeMap::new();
        let mut expected_workspace_dependencies = Vec::new();
        for (dependency, kind, is_workspace) in &expected_dependencies {
            *expected_kind_counts.entry(*kind).or_insert(0_u64) += 1;
            if *is_workspace {
                expected_workspace_dependencies.push(*dependency);
            }
        }
        let claimed_kind_counts = claim["dependency_kind_counts"]
            .as_object()
            .expect("claimed kind counts are an object")
            .iter()
            .map(|(kind, count)| {
                (
                    kind.as_str(),
                    count.as_u64().expect("claimed kind count is an integer"),
                )
            })
            .collect::<std::collections::BTreeMap<_, _>>();
        assert_eq!(
            claimed_kind_counts, expected_kind_counts,
            "stale dependency kinds for `{name}`"
        );
        let claimed_workspace_dependencies = claim["direct_workspace_dependencies"]
            .as_array()
            .expect("claimed workspace dependencies are an array")
            .iter()
            .map(|dependency| dependency.as_str().expect("workspace dependency is text"))
            .collect::<Vec<_>>();
        assert_eq!(
            claimed_workspace_dependencies, expected_workspace_dependencies,
            "stale workspace dependencies for `{name}`"
        );
        let expected_features = package["features"]
            .as_object()
            .expect("features are an object")
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let claimed_features = claim["features"]
            .as_array()
            .expect("claimed features are an array")
            .iter()
            .map(|feature| feature.as_str().expect("claimed feature is text"))
            .collect::<BTreeSet<_>>();
        assert_eq!(
            claimed_features, expected_features,
            "stale features for `{name}`"
        );
    }
}

fn published_locator_fixture(generation: &str) -> (Value, JsonValue) {
    let raw: Value = serde_yaml::from_str(&format!(
        r#"
admission_status: published_verified_{generation}
manifest:
  path: artifacts/native-plugin-baseline/evidence-manifest.json
  bytes: 10
  blake3: blake3:1111111111111111111111111111111111111111111111111111111111111111
bundle:
  staged_path: /work/evidence/aiperf-native-plugin-baseline-caa3ff6f-{generation}-final.tar.gz
  bytes: 20
  blake3: blake3:2222222222222222222222222222222222222222222222222222222222222222
  release_tag: native-plugin-baseline-caa3ff6f-{generation}-final
  repository: https://github.com/ajcasagrande/rust-native-plugin-lab
"#
    ))
    .expect("published raw-sample fixture parses");
    let locator = serde_json::json!({
        "schema_version": 1,
        "repository": "https://github.com/ajcasagrande/rust-native-plugin-lab",
        "recommended_release_tag": format!("native-plugin-baseline-caa3ff6f-{generation}-final"),
        "asset_name": format!("aiperf-native-plugin-baseline-caa3ff6f-{generation}-final.tar.gz"),
        "publication_status": "published_and_verified",
        "archive_verification_status": "downloaded_extracted_manifest_verified",
        "staged_path": format!("/work/evidence/aiperf-native-plugin-baseline-caa3ff6f-{generation}-final.tar.gz"),
        "bytes": 20,
        "blake3": "blake3:2222222222222222222222222222222222222222222222222222222222222222",
        "manifest_path": "artifacts/native-plugin-baseline/evidence-manifest.json",
        "manifest_bytes": 10,
        "manifest_blake3": "blake3:1111111111111111111111111111111111111111111111111111111111111111",
        "stable_url": format!("https://github.com/ajcasagrande/rust-native-plugin-lab/releases/download/native-plugin-baseline-caa3ff6f-{generation}-final/aiperf-native-plugin-baseline-caa3ff6f-{generation}-final.tar.gz")
    });
    (raw, locator)
}

#[test]
fn published_locator_requires_review1i_retrieval_admission() {
    let (raw, locator) = published_locator_fixture("review1i");

    validate_bundle_locator(
        &locator,
        raw.as_mapping().expect("raw-sample fixture is a mapping"),
    )
    .expect("review1i published locator must be admissible after retrieval verification");
}

#[test]
fn canonical_locator_rejects_superseded_evidence() {
    let (mut raw, mut locator) = published_locator_fixture("review1h");
    raw["admission_status"] = Value::String("superseded_rejected_review1e".to_owned());
    locator["publication_status"] = JsonValue::String("superseded_rejected".to_owned());
    locator["archive_verification_status"] =
        JsonValue::String("superseded_without_admission".to_owned());
    locator["invalidation_reason"] = JsonValue::String("superseded fixture".to_owned());

    let error = validate_bundle_locator(
        &locator,
        raw.as_mapping().expect("raw-sample fixture is a mapping"),
    )
    .expect_err("superseded evidence must never satisfy canonical admission");
    assert!(error.contains("published_and_verified"), "{error}");
}

#[test]
fn published_locator_rejects_invalidated_review1e_through_review1h() {
    for generation in ["review1e", "review1f", "review1g", "review1h"] {
        let (raw, locator) = published_locator_fixture(generation);
        let error = validate_bundle_locator(
            &locator,
            raw.as_mapping().expect("raw-sample fixture is a mapping"),
        )
        .expect_err("an explicitly invalidated generation must remain inadmissible");
        assert!(
            error.contains("explicitly invalidated"),
            "{generation} failed for unexpected reason: {error}"
        );
    }
}
