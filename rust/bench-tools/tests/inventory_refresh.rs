// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Behavioral contract for deterministic native-plugin inventory refreshes.

use std::{fs, fs::File, path::Path, process::Command};

use flate2::{Compression, write::GzEncoder};
use tar::{Builder, Header};

const CAPTURED_BUILD_DIGEST: &str =
    "blake3:1111111111111111111111111111111111111111111111111111111111111111";
const CAPTURED_CORPUS_DIGEST: &str =
    "blake3:2222222222222222222222222222222222222222222222222222222222222222";
const CAPTURED_RAW_OBSERVABLE_DIGEST: &str =
    "blake3:3333333333333333333333333333333333333333333333333333333333333333";
const CAPTURED_PROVENANCE_DIGEST: &str =
    "blake3:4444444444444444444444444444444444444444444444444444444444444444";

fn repository_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn prepare_repository_inventory(root: &Path) -> std::path::PathBuf {
    let inventory = root.join("rust/benchmarks/plugin-parity.yaml");
    let artifacts = root.join("artifacts/native-plugin-baseline");
    fs::create_dir_all(inventory.parent().expect("inventory parent exists"))
        .expect("inventory fixture directory is created");
    fs::create_dir_all(&artifacts).expect("artifact fixture directory is created");
    fs::copy(
        repository_root().join("rust/benchmarks/plugin-parity.yaml"),
        &inventory,
    )
    .expect("inventory fixture is copied");
    for name in ["README.md", "allocation-probe.json"] {
        fs::copy(
            repository_root()
                .join("artifacts/native-plugin-baseline")
                .join(name),
            artifacts.join(name),
        )
        .expect("derived artifact fixture is copied");
    }
    inventory
}

fn write_pre_capture_receipts(root: &Path, invalidations: &str, projection: &str) {
    fs::create_dir_all(root).expect("receipt root is created");
    let projection = if projection.contains("rust/e2e-tests/tests/plugin_baseline_inventory.rs\n") {
        projection.to_owned()
    } else {
        format!("rust/e2e-tests/tests/plugin_baseline_inventory.rs\n{projection}")
    };
    for (name, contents) in [
        ("source-tree.tar", b"source archive".as_slice()),
        ("baseline-Cargo.lock", b"baseline lock"),
        ("Cargo.lock", b"effective lock"),
        ("measurement-source-projection.txt", projection.as_bytes()),
        ("measurement-source-projection.tar", b"projection archive"),
        ("effective-source-tree.tar", b"effective archive"),
        ("capture-plugin-baseline.sh", b"capture harness"),
        ("plugin-baseline-owned-command.sh", b"owned command helper"),
        ("invalidations.tsv", invalidations.as_bytes()),
    ] {
        fs::write(root.join(name), contents).expect("receipt fixture is written");
    }
    for name in [
        "exporter-observable-policy.json",
        "exporter-static-calibration-corpus.json",
    ] {
        fs::copy(
            repository_root().join("rust/benchmarks").join(name),
            root.join(name),
        )
        .expect("pre-run exporter source configuration is copied");
    }
}

fn valid_invalidations() -> String {
    [
        "review1\tinvalid\tinterrupted before canonicalization",
        "review1b\tinvalid\toverlapped review1 compilation",
        "review1c\tinvalid\trequired timing utility was unavailable",
        "review1d\tsuperseded\ttransport probe was not repeated",
        "review1e\tsuperseded\tpublished before final allocation contract",
        "review1f\tinvalid\tincomplete measurement projection",
        "review1g\tinvalid\tstorage exhausted during build",
        "review1h\tinvalid\towned stdin was discarded",
    ]
    .join("\n")
        + "\n"
}

fn run_refresh(
    mode: &str,
    generation: &str,
    inventory: &Path,
    receipts: &Path,
    topology: Option<&Path>,
) -> std::process::Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_evidence_digest"));
    command
        .args(["refresh-contract", mode, generation])
        .arg(inventory)
        .arg(receipts)
        .env(
            "AIPERF_PLUGIN_REFRESH_TMPDIR",
            inventory.parent().expect("inventory has a parent"),
        );
    if let Some(topology) = topology {
        command.arg(topology);
    }
    command.output().expect("refresh-contract starts")
}

fn run_refresh_with_environment(
    mode: &str,
    generation: &str,
    inventory: &Path,
    receipts: &Path,
    topology: Option<&Path>,
    environment: &[(&str, &Path)],
) -> std::process::Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_evidence_digest"));
    command
        .args(["refresh-contract", mode, generation])
        .arg(inventory)
        .arg(receipts)
        .env(
            "AIPERF_PLUGIN_REFRESH_TMPDIR",
            inventory.parent().expect("inventory has a parent"),
        );
    if let Some(topology) = topology {
        command.arg(topology);
    }
    for (key, value) in environment {
        command.env(key, value);
    }
    command.output().expect("refresh-contract starts")
}

fn digest(path: &Path) -> String {
    format!(
        "blake3:{}",
        blake3::hash(&fs::read(path).expect("digest input is readable"))
    )
}

fn append_tar_file<W: std::io::Write>(builder: &mut Builder<W>, path: &str, contents: &[u8]) {
    let mut header = Header::new_gnu();
    header.set_mode(0o644);
    header.set_size(contents.len() as u64);
    header.set_cksum();
    builder
        .append_data(&mut header, path, contents)
        .expect("tar fixture member is appended");
}

fn append_tar_file_with_raw_name<W: std::io::Write>(
    builder: &mut Builder<W>,
    name: &[u8],
    contents: &[u8],
) {
    assert!(
        name.len() < 100,
        "raw tar fixture name fits the legacy field"
    );
    let mut header = Header::new_gnu();
    header.set_mode(0o644);
    header.set_size(contents.len() as u64);
    header.as_mut_bytes()[..100].fill(0);
    header.as_mut_bytes()[..name.len()].copy_from_slice(name);
    header.set_cksum();
    builder
        .append(&header, contents)
        .expect("raw-name tar fixture member is appended");
}

fn write_captured_source_trust_fixture(capture: &Path, archive: &Path) -> std::path::PathBuf {
    let mut trusted_identity = serde_json::Map::from_iter([
        (
            "canonical_inventory_digest".to_owned(),
            serde_json::json!(
                "blake3:0000000000000000000000000000000000000000000000000000000000000000"
            ),
        ),
        (
            "effective_source_tree_blake3".to_owned(),
            serde_json::json!(digest(archive)),
        ),
    ]);
    let mut captured_identity = trusted_identity.clone();
    captured_identity.remove("canonical_inventory_digest");
    let mut captured_identity_bytes =
        serde_json::to_vec_pretty(&captured_identity).expect("captured identity serializes");
    captured_identity_bytes.push(b'\n');
    let captured_identity_path = capture.join("evidence/identity/experiment-identity.json");
    fs::write(&captured_identity_path, &captured_identity_bytes)
        .expect("captured identity is written");
    let manifest = serde_json::json!({
        "schema_version": 1,
        "files": [
            {
                "path": "identity/effective-source-tree.tar",
                "bytes": fs::metadata(archive).expect("archive metadata").len(),
                "blake3": digest(archive),
            },
            {
                "path": "identity/experiment-identity.json",
                "bytes": captured_identity_bytes.len(),
                "blake3": format!("blake3:{}", blake3::hash(&captured_identity_bytes)),
            },
        ],
    });
    fs::write(
        capture.join("evidence-manifest.json"),
        serde_json::to_vec(&manifest).expect("manifest serializes"),
    )
    .expect("manifest is written");
    trusted_identity.insert(
        "canonical_inventory_digest".to_owned(),
        serde_json::json!(
            "blake3:0000000000000000000000000000000000000000000000000000000000000000"
        ),
    );
    let trusted_identity =
        serde_json::to_string_pretty(&trusted_identity).expect("trusted identity serializes");
    let inventory = capture.join("reviewed-plugin-parity.yaml");
    let indented = trusted_identity
        .lines()
        .map(|line| format!("  {line}\n"))
        .collect::<String>();
    fs::write(
        &inventory,
        format!("experiment_identity_json: |\n{indented}"),
    )
    .expect("reviewed inventory is written");
    inventory
}

fn write_captured_source_manifest(capture: &Path, archive: &Path) {
    let _ = write_captured_source_trust_fixture(capture, archive);
}

#[test]
fn captured_source_and_manifest_must_match_the_reviewed_pre_capture_identity() {
    let directory = tempfile::tempdir().expect("reviewed captured-source fixture root");
    let capture = directory.path().join("capture");
    let identity = capture.join("evidence/identity");
    let destination = directory.path().join("extracted");
    fs::create_dir_all(&identity).expect("capture identity root is created");
    fs::create_dir(&destination).expect("controlled extraction root is created");
    let archive = identity.join("effective-source-tree.tar");
    let mut builder = Builder::new(File::create(&archive).expect("source archive is created"));
    append_tar_file(&mut builder, "safe.txt", b"reviewed source");
    builder.finish().expect("reviewed archive is finished");
    let reviewed_inventory = write_captured_source_trust_fixture(&capture, &archive);
    let reviewed_inventory_bytes =
        fs::read(&reviewed_inventory).expect("reviewed inventory is retained");

    let mut builder = Builder::new(File::create(&archive).expect("archive is replaced"));
    append_tar_file(&mut builder, "build.rs", b"malicious replacement");
    builder.finish().expect("replacement archive is finished");
    write_captured_source_trust_fixture(&capture, &archive);
    fs::write(&reviewed_inventory, reviewed_inventory_bytes)
        .expect("reviewed inventory remains authoritative");

    let output = Command::new(env!("CARGO_BIN_EXE_evidence_digest"))
        .arg("extract-captured-source")
        .arg(&reviewed_inventory)
        .arg(&capture)
        .arg(&destination)
        .output()
        .expect("trusted extraction command starts");
    assert!(
        !output.status.success(),
        "caller-rebound source was accepted"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("reviewed pre-capture identity"),
        "unexpected trust refusal: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        fs::read_dir(&destination)
            .expect("destination remains readable")
            .count(),
        0,
        "unreviewed captured source was written"
    );
}

#[test]
fn trusted_tool_authenticates_captured_source_before_extraction() {
    let directory = tempfile::tempdir().expect("captured source fixture root");
    let capture = directory.path().join("capture");
    let identity = capture.join("evidence/identity");
    let destination = directory.path().join("extracted");
    fs::create_dir_all(&identity).expect("capture identity root is created");
    fs::create_dir(&destination).expect("controlled extraction root is created");
    let archive = identity.join("effective-source-tree.tar");
    let mut builder = Builder::new(File::create(&archive).expect("source archive is created"));
    append_tar_file(&mut builder, "rust/Cargo.toml", b"[workspace]\n");
    builder.finish().expect("source archive is finished");
    let reviewed_inventory = write_captured_source_trust_fixture(&capture, &archive);

    let output = Command::new(env!("CARGO_BIN_EXE_evidence_digest"))
        .arg("extract-captured-source")
        .arg(&reviewed_inventory)
        .arg(&capture)
        .arg(&destination)
        .output()
        .expect("trusted extraction command starts");
    assert!(
        output.status.success(),
        "trusted extraction failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        fs::read(destination.join("rust/Cargo.toml")).expect("source member is extracted"),
        b"[workspace]\n"
    );
}

#[test]
fn refresh_shell_executes_post_authentication_work_only_after_trusted_source_extraction() {
    let directory = tempfile::tempdir().expect("refresh shell captured-source fixture root");
    let script = repository_root().join("rust/scripts/refresh-plugin-baseline-inventory.sh");

    let valid_capture = directory.path().join("valid-capture");
    let valid_identity = valid_capture.join("evidence/identity");
    let valid_destination = directory.path().join("valid-extracted");
    fs::create_dir_all(&valid_identity).expect("valid identity root is created");
    fs::create_dir(&valid_destination).expect("valid destination is created");
    let valid_archive = valid_identity.join("effective-source-tree.tar");
    let mut builder =
        Builder::new(File::create(&valid_archive).expect("valid source archive is created"));
    append_tar_file(&mut builder, "rust/Cargo.toml", b"[workspace]\n");
    builder.finish().expect("valid source archive is finished");
    let valid_reviewed_inventory =
        write_captured_source_trust_fixture(&valid_capture, &valid_archive);
    let valid_marker = directory.path().join("valid-post-authentication-work");
    let valid = Command::new("sh")
        .arg(&script)
        .arg("--captured-source-verification-self-test")
        .arg(env!("CARGO_BIN_EXE_evidence_digest"))
        .arg(&valid_reviewed_inventory)
        .arg(&valid_capture)
        .arg(&valid_destination)
        .arg(&valid_marker)
        .env(
            "AIPERF_PLUGIN_BASELINE_LOCK",
            directory.path().join("valid-refresh.lock"),
        )
        .output()
        .expect("valid refresh shell verification starts");
    assert!(
        valid.status.success(),
        "valid captured source was refused: {}",
        String::from_utf8_lossy(&valid.stderr)
    );
    assert!(
        valid_marker.is_file(),
        "post-authentication work was not reached"
    );

    let tampered_capture = directory.path().join("tampered-capture");
    let tampered_identity = tampered_capture.join("evidence/identity");
    let tampered_destination = directory.path().join("tampered-extracted");
    fs::create_dir_all(&tampered_identity).expect("tampered identity root is created");
    fs::create_dir(&tampered_destination).expect("tampered destination is created");
    let tampered_archive = tampered_identity.join("effective-source-tree.tar");
    let mut builder =
        Builder::new(File::create(&tampered_archive).expect("tampered archive is created"));
    append_tar_file(&mut builder, "rust/Cargo.toml", b"malicious build input\n");
    builder
        .finish()
        .expect("tampered source archive is finished");
    let tampered_reviewed_inventory =
        write_captured_source_trust_fixture(&tampered_capture, &tampered_archive);
    fs::write(&tampered_archive, b"bytes changed after manifest")
        .expect("captured source is tampered after manifest");
    let tampered_marker = directory.path().join("unexpected-post-authentication-work");
    let rejected = Command::new("sh")
        .arg(script)
        .arg("--captured-source-verification-self-test")
        .arg(env!("CARGO_BIN_EXE_evidence_digest"))
        .arg(&tampered_reviewed_inventory)
        .arg(&tampered_capture)
        .arg(&tampered_destination)
        .arg(&tampered_marker)
        .env(
            "AIPERF_PLUGIN_BASELINE_LOCK",
            directory.path().join("tampered-refresh.lock"),
        )
        .output()
        .expect("tampered refresh shell verification starts");
    assert!(
        !rejected.status.success(),
        "tampered captured source passed"
    );
    assert!(
        !tampered_marker.exists(),
        "post-authentication work ran for a tampered captured source"
    );
    assert_eq!(
        fs::read_dir(&tampered_destination)
            .expect("tampered destination remains readable")
            .count(),
        0,
        "tampered source was extracted before authentication"
    );
}

#[test]
fn captured_source_archive_enforces_member_count_before_writing_destination() {
    let directory = tempfile::tempdir().expect("captured source limit fixture root");
    let capture = directory.path().join("capture");
    let identity = capture.join("evidence/identity");
    let destination = directory.path().join("extracted");
    fs::create_dir_all(&identity).expect("capture identity root is created");
    fs::create_dir(&destination).expect("controlled extraction root is created");
    let archive = identity.join("effective-source-tree.tar");
    let mut builder = Builder::new(File::create(&archive).expect("source archive is created"));
    append_tar_file(&mut builder, "one.txt", b"one");
    append_tar_file(&mut builder, "two.txt", b"two");
    builder.finish().expect("source archive is finished");
    let reviewed_inventory = write_captured_source_trust_fixture(&capture, &archive);

    let output = Command::new(env!("CARGO_BIN_EXE_evidence_digest"))
        .arg("extract-captured-source-test-limits")
        .arg(&reviewed_inventory)
        .arg(&capture)
        .arg(&destination)
        .env("AIPERF_TEST_CAPTURED_SOURCE_MAX_MEMBERS", "1")
        .output()
        .expect("trusted extraction command starts");
    assert!(
        !output.status.success(),
        "member-count overflow was accepted"
    );
    assert_eq!(
        fs::read_dir(&destination)
            .expect("destination remains readable")
            .count(),
        0,
        "archive validation wrote destination bytes before refusal"
    );
}

#[test]
fn production_captured_source_limits_ignore_ambient_test_overrides() {
    let directory = tempfile::tempdir().expect("production source-limit fixture root");
    let capture = directory.path().join("capture");
    let identity = capture.join("evidence/identity");
    let destination = directory.path().join("extracted");
    fs::create_dir_all(&identity).expect("capture identity root is created");
    fs::create_dir(&destination).expect("controlled extraction root is created");
    let archive = identity.join("effective-source-tree.tar");
    let mut builder = Builder::new(File::create(&archive).expect("source archive is created"));
    append_tar_file(&mut builder, "one.txt", b"one");
    append_tar_file(&mut builder, "two.txt", b"two");
    builder.finish().expect("source archive is finished");
    let reviewed_inventory = write_captured_source_trust_fixture(&capture, &archive);

    let output = Command::new(env!("CARGO_BIN_EXE_evidence_digest"))
        .arg("extract-captured-source")
        .arg(&reviewed_inventory)
        .arg(&capture)
        .arg(&destination)
        .env("AIPERF_TEST_CAPTURED_SOURCE_MAX_MEMBERS", "1")
        .output()
        .expect("production extraction starts");
    assert!(
        output.status.success(),
        "ambient test limit changed production policy: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(destination.join("one.txt").is_file());
    assert!(destination.join("two.txt").is_file());
}

#[test]
fn completed_bundle_is_verified_through_the_trusted_archive_entrypoint() {
    let directory = tempfile::tempdir().expect("completed bundle fixture root");
    let capture = directory.path().join("capture");
    let evidence = capture.join("evidence");
    let extraction_parent = directory.path().join("controlled");
    fs::create_dir_all(&evidence).expect("evidence root is created");
    fs::create_dir(&extraction_parent).expect("controlled root is created");
    fs::write(evidence.join("sample.txt"), b"sealed evidence").expect("evidence is written");
    let manifest = capture.join("evidence-manifest.json");
    fs::write(
        &manifest,
        serde_json::to_vec(&serde_json::json!({
            "schema_version": 1,
            "files": [{
                "path": "sample.txt",
                "bytes": 15,
                "blake3": format!("blake3:{}", blake3::hash(b"sealed evidence")),
            }],
        }))
        .expect("manifest serializes"),
    )
    .expect("manifest is written");
    let bundle = directory.path().join("bundle.tar.gz");
    create_bundle(&capture, &bundle);
    let receipt = directory.path().join("bundle-verification.json");

    let output = Command::new(env!("CARGO_BIN_EXE_evidence_digest"))
        .arg("verify-staged-bundle")
        .arg("review1i")
        .arg(&bundle)
        .arg(&manifest)
        .arg(&extraction_parent)
        .arg(&receipt)
        .output()
        .expect("trusted staged-bundle verifier starts");
    assert!(
        output.status.success(),
        "trusted staged-bundle verification failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let receipt: serde_json::Value = serde_json::from_slice(
        &fs::read(&receipt).expect("verification receipt is authored by the trusted command"),
    )
    .expect("verification receipt parses");
    assert_eq!(receipt["status"], "extracted_manifest_verified");
    assert_eq!(receipt["bundle_blake3"], digest(&bundle));
    assert_eq!(receipt["manifest_blake3"], digest(&manifest));
}

#[test]
fn staged_manifest_receipt_uses_the_owned_authenticated_snapshot() {
    let directory = tempfile::tempdir().expect("staged manifest snapshot fixture root");
    let capture = directory.path().join("capture");
    let evidence = capture.join("evidence");
    let extraction_parent = directory.path().join("controlled");
    fs::create_dir_all(&evidence).expect("evidence root is created");
    fs::create_dir(&extraction_parent).expect("controlled root is created");
    fs::write(evidence.join("sample.txt"), b"sealed evidence").expect("evidence is written");
    let manifest = capture.join("evidence-manifest.json");
    fs::write(
        &manifest,
        serde_json::to_vec(&serde_json::json!({
            "schema_version": 1,
            "files": [{
                "path": "sample.txt",
                "bytes": 15,
                "blake3": format!("blake3:{}", blake3::hash(b"sealed evidence")),
            }],
        }))
        .expect("manifest serializes"),
    )
    .expect("manifest is written");
    let authenticated_manifest = fs::read(&manifest).expect("authenticated manifest is readable");
    let bundle = directory.path().join("bundle.tar.gz");
    create_bundle(&capture, &bundle);
    let replacement = directory.path().join("replacement-manifest.json");
    fs::write(&replacement, b"{\"files\":[],\"schema_version\":1}\n")
        .expect("replacement manifest is written");
    let receipt = directory.path().join("bundle-verification.json");

    let output = Command::new(env!("CARGO_BIN_EXE_evidence_digest"))
        .arg("verify-staged-bundle")
        .arg("review1i")
        .arg(&bundle)
        .arg(&manifest)
        .arg(&extraction_parent)
        .arg(&receipt)
        .env(
            "AIPERF_TEST_REWRITE_STAGED_MANIFEST_AFTER_SNAPSHOT",
            &replacement,
        )
        .output()
        .expect("staged manifest snapshot verification starts");
    assert!(
        output.status.success(),
        "owned manifest snapshot was not verified: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        fs::read(&manifest).expect("caller manifest remains readable"),
        fs::read(&replacement).expect("replacement manifest remains readable"),
        "same-inode manifest mutation seam did not run"
    );
    let receipt: serde_json::Value =
        serde_json::from_slice(&fs::read(receipt).expect("snapshot receipt is readable"))
            .expect("snapshot receipt parses");
    assert_eq!(
        receipt["manifest_blake3"],
        format!("blake3:{}", blake3::hash(&authenticated_manifest))
    );
}

#[test]
fn staged_bundle_verifier_rejects_unsafe_archive_before_extraction() {
    for case in [
        "parent",
        "absolute",
        "backslash",
        "symlink",
        "hardlink",
        "duplicate",
        "duplicate_alias",
        "missing_evidence",
        "evidence_wrong_type",
        "manifest_wrong_type",
        "unlisted_evidence",
        "member_count",
        "member_size",
        "aggregate_size",
    ] {
        let directory = tempfile::tempdir().expect("staged bundle adversary root");
        let manifest = directory.path().join("evidence-manifest.json");
        let manifest_bytes = if case == "duplicate_alias" {
            serde_json::to_vec(&serde_json::json!({
                "schema_version": 1,
                "files": [{
                    "path": "same",
                    "bytes": 6,
                    "blake3": format!("blake3:{}", blake3::hash(b"second")),
                }],
            }))
            .expect("alias manifest serializes")
        } else {
            serde_json::to_vec(&serde_json::json!({
                "schema_version": 1,
                "files": [],
            }))
            .expect("manifest serializes")
        };
        fs::write(&manifest, &manifest_bytes).expect("external manifest is written");
        let bundle = directory.path().join("bundle.tar.gz");
        let encoder = GzEncoder::new(
            File::create(&bundle).expect("bundle is created"),
            Compression::default(),
        );
        let mut builder = Builder::new(encoder);
        if case == "evidence_wrong_type" {
            append_tar_file(&mut builder, "evidence", b"not a directory");
        } else if case != "missing_evidence" {
            let mut evidence_header = Header::new_gnu();
            evidence_header.set_entry_type(tar::EntryType::Directory);
            evidence_header.set_mode(0o755);
            evidence_header.set_size(0);
            evidence_header.set_cksum();
            builder
                .append_data(&mut evidence_header, "evidence", std::io::empty())
                .expect("evidence root is appended");
        }
        if case == "manifest_wrong_type" {
            let mut manifest_header = Header::new_gnu();
            manifest_header.set_entry_type(tar::EntryType::Directory);
            manifest_header.set_mode(0o755);
            manifest_header.set_size(0);
            manifest_header.set_cksum();
            builder
                .append_data(
                    &mut manifest_header,
                    "evidence-manifest.json",
                    std::io::empty(),
                )
                .expect("manifest directory is appended");
        } else {
            append_tar_file(&mut builder, "evidence-manifest.json", &manifest_bytes);
        }
        match case {
            "parent" => append_tar_file_with_raw_name(&mut builder, b"../outside", b"attack"),
            "absolute" => append_tar_file_with_raw_name(&mut builder, b"/outside", b"attack"),
            "backslash" => append_tar_file_with_raw_name(&mut builder, b"evidence\\bad", b"attack"),
            "symlink" | "hardlink" => {
                let mut header = Header::new_gnu();
                header.set_entry_type(if case == "symlink" {
                    tar::EntryType::Symlink
                } else {
                    tar::EntryType::Link
                });
                header.set_mode(0o777);
                header.set_size(0);
                header
                    .set_link_name("../outside")
                    .expect("link target is written");
                header.set_cksum();
                builder
                    .append_data(&mut header, "evidence/link", std::io::empty())
                    .expect("link fixture is appended");
            }
            "duplicate" => {
                append_tar_file(&mut builder, "evidence/same", b"first");
                append_tar_file(&mut builder, "evidence/same", b"second");
            }
            "duplicate_alias" => {
                append_tar_file(&mut builder, "evidence/same", b"first");
                append_tar_file_with_raw_name(&mut builder, b"evidence//same", b"second");
            }
            "unlisted_evidence" => {
                append_tar_file(&mut builder, "evidence/unlisted", b"not in manifest")
            }
            "member_count" => append_tar_file(&mut builder, "evidence/extra", b"one"),
            "member_size" => append_tar_file(
                &mut builder,
                "evidence/large",
                &vec![b'x'; manifest_bytes.len() + 1],
            ),
            "aggregate_size" => {
                append_tar_file(&mut builder, "evidence/one", b"three");
                append_tar_file(&mut builder, "evidence/two", b"three");
            }
            "missing_evidence" | "evidence_wrong_type" | "manifest_wrong_type" => {}
            _ => unreachable!(),
        }
        builder
            .into_inner()
            .expect("tar stream is finished")
            .finish()
            .expect("gzip stream is finished");
        let controlled = directory.path().join("controlled");
        fs::create_dir(&controlled).expect("controlled extraction root is created");
        let outside = directory.path().join("outside");
        fs::write(&outside, b"sentinel").expect("outside sentinel is written");
        let receipt = directory.path().join("bundle-verification.json");
        let mut command = Command::new(env!("CARGO_BIN_EXE_evidence_digest"));
        command
            .arg("verify-staged-bundle-test-limits")
            .arg("review1i")
            .arg(&bundle)
            .arg(&manifest)
            .arg(&controlled)
            .arg(&receipt);
        if case == "member_count" {
            command.env("AIPERF_TEST_DOWNLOADED_MAX_MEMBERS", "2");
        }
        if case == "member_size" {
            command.env(
                "AIPERF_TEST_DOWNLOADED_MAX_MEMBER_BYTES",
                manifest_bytes.len().to_string(),
            );
        }
        if case == "aggregate_size" {
            command.env(
                "AIPERF_TEST_DOWNLOADED_MAX_EXPANDED_BYTES",
                (manifest_bytes.len() + 5).to_string(),
            );
        }
        let output = command.output().expect("staged bundle verifier starts");
        assert!(
            !output.status.success(),
            "staged bundle adversary `{case}` was accepted"
        );
        assert_eq!(
            fs::read(&outside).expect("outside sentinel remains"),
            b"sentinel",
            "staged bundle adversary `{case}` escaped controlled storage"
        );
        assert_eq!(
            fs::read_dir(&controlled)
                .expect("controlled root remains readable")
                .count(),
            0,
            "staged bundle adversary `{case}` left extracted bytes"
        );
    }
}

#[test]
fn staged_bundle_verifier_rejects_uncontrolled_extraction_parents() {
    let directory = tempfile::tempdir().expect("staged bundle root fixture");
    let capture = directory.path().join("capture");
    let evidence = capture.join("evidence");
    fs::create_dir_all(&evidence).expect("evidence root is created");
    fs::write(evidence.join("sample.txt"), b"sealed evidence").expect("evidence is written");
    let manifest = capture.join("evidence-manifest.json");
    fs::write(
        &manifest,
        serde_json::to_vec(&serde_json::json!({
            "schema_version": 1,
            "files": [{
                "path": "sample.txt",
                "bytes": 15,
                "blake3": format!("blake3:{}", blake3::hash(b"sealed evidence")),
            }],
        }))
        .expect("manifest serializes"),
    )
    .expect("manifest is written");
    let bundle = directory.path().join("bundle.tar.gz");
    create_bundle(&capture, &bundle);

    for uncontrolled in [
        std::path::PathBuf::from("relative-extraction-root"),
        directory.path().join("missing-extraction-root"),
    ] {
        let receipt = directory.path().join("bundle-verification.json");
        let output = Command::new(env!("CARGO_BIN_EXE_evidence_digest"))
            .arg("verify-staged-bundle")
            .arg("review1i")
            .arg(&bundle)
            .arg(&manifest)
            .arg(&uncontrolled)
            .arg(receipt)
            .output()
            .expect("staged bundle verifier starts");
        assert!(
            !output.status.success(),
            "uncontrolled extraction parent {} was accepted",
            uncontrolled.display()
        );
    }
}

#[test]
fn captured_source_archive_rejects_unsafe_types_paths_and_sizes() {
    for case in [
        "parent",
        "absolute",
        "backslash",
        "newline",
        "non_utf8",
        "symlink",
        "hardlink",
        "duplicate",
        "member_size",
        "aggregate_size",
    ] {
        let directory = tempfile::tempdir().expect("captured archive adversary root");
        let capture = directory.path().join("capture");
        let identity = capture.join("evidence/identity");
        let destination = directory.path().join("extracted");
        fs::create_dir_all(&identity).expect("capture identity root is created");
        fs::create_dir(&destination).expect("controlled extraction root is created");
        let outside = directory.path().join("outside");
        fs::write(&outside, b"sentinel").expect("outside sentinel is written");
        let archive = identity.join("effective-source-tree.tar");
        let mut builder = Builder::new(File::create(&archive).expect("archive is created"));
        match case {
            "parent" => append_tar_file_with_raw_name(&mut builder, b"../outside", b"attack"),
            "absolute" => append_tar_file_with_raw_name(&mut builder, b"/outside", b"attack"),
            "backslash" => append_tar_file_with_raw_name(&mut builder, b"bad\\path", b"attack"),
            "newline" => append_tar_file_with_raw_name(&mut builder, b"bad\npath", b"attack"),
            "non_utf8" => append_tar_file_with_raw_name(&mut builder, b"bad\xffpath", b"attack"),
            "symlink" | "hardlink" => {
                let mut header = Header::new_gnu();
                header.set_entry_type(if case == "symlink" {
                    tar::EntryType::Symlink
                } else {
                    tar::EntryType::Link
                });
                header.set_mode(0o777);
                header.set_size(0);
                header
                    .set_link_name("../outside")
                    .expect("link target is written");
                header.set_cksum();
                builder
                    .append_data(&mut header, "link", std::io::empty())
                    .expect("link fixture is appended");
            }
            "duplicate" => {
                append_tar_file(&mut builder, "same", b"first");
                append_tar_file_with_raw_name(&mut builder, b"./same", b"second");
            }
            "member_size" => append_tar_file(&mut builder, "large", b"three"),
            "aggregate_size" => {
                append_tar_file(&mut builder, "one", b"three");
                append_tar_file(&mut builder, "two", b"three");
            }
            _ => unreachable!(),
        }
        builder.finish().expect("adversarial archive is finished");
        let reviewed_inventory = write_captured_source_trust_fixture(&capture, &archive);
        let mut command = Command::new(env!("CARGO_BIN_EXE_evidence_digest"));
        command
            .arg("extract-captured-source-test-limits")
            .arg(&reviewed_inventory)
            .arg(&capture)
            .arg(&destination);
        if case == "member_size" {
            command.env("AIPERF_TEST_CAPTURED_SOURCE_MAX_MEMBER_BYTES", "4");
        }
        if case == "aggregate_size" {
            command.env("AIPERF_TEST_CAPTURED_SOURCE_MAX_EXPANDED_BYTES", "9");
        }
        let output = command.output().expect("trusted archive verifier starts");
        assert!(
            !output.status.success(),
            "captured source adversary `{case}` was accepted"
        );
        assert_eq!(
            fs::read_dir(&destination)
                .expect("controlled destination is readable")
                .count(),
            0,
            "captured source adversary `{case}` wrote before refusal"
        );
        assert_eq!(
            fs::read(&outside).expect("outside sentinel remains"),
            b"sentinel",
            "captured source adversary `{case}` escaped controlled storage"
        );
    }
}

#[cfg(unix)]
#[test]
fn captured_source_authentication_precedes_extraction_and_rejects_symlinks() {
    use std::os::unix::fs::symlink;

    let directory = tempfile::tempdir().expect("captured source authentication root");
    let capture = directory.path().join("capture");
    let identity = capture.join("evidence/identity");
    let destination = directory.path().join("extracted");
    fs::create_dir_all(&identity).expect("capture identity root is created");
    fs::create_dir(&destination).expect("controlled extraction root is created");
    let archive = identity.join("effective-source-tree.tar");
    let mut builder = Builder::new(File::create(&archive).expect("archive is created"));
    append_tar_file(&mut builder, "safe.txt", b"safe");
    builder.finish().expect("archive is finished");
    let reviewed_inventory = write_captured_source_trust_fixture(&capture, &archive);
    let manifest_path = capture.join("evidence-manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).expect("manifest is readable"))
            .expect("manifest parses");
    manifest["files"][0]["blake3"] = serde_json::json!(
        "blake3:0000000000000000000000000000000000000000000000000000000000000000"
    );
    fs::write(
        &manifest_path,
        serde_json::to_vec(&manifest).expect("mutated manifest serializes"),
    )
    .expect("mutated manifest is written");
    let marker = directory.path().join("extraction-entered");
    let mismatched = Command::new(env!("CARGO_BIN_EXE_evidence_digest"))
        .arg("extract-captured-source")
        .arg(&reviewed_inventory)
        .arg(&capture)
        .arg(&destination)
        .env("AIPERF_CAPTURED_SOURCE_EXTRACTION_MARKER", &marker)
        .status()
        .expect("mismatched extraction starts");
    assert!(
        !mismatched.success(),
        "unauthenticated archive was accepted"
    );
    assert!(!marker.exists(), "extraction began before authentication");
    assert_eq!(
        fs::read_dir(&destination)
            .expect("destination remains readable")
            .count(),
        0
    );

    write_captured_source_manifest(&capture, &archive);
    let original = identity.join("original-source.tar");
    fs::rename(&archive, &original).expect("archive is moved behind a symlink");
    symlink(&original, &archive).expect("archive symlink is created");
    let symlinked = Command::new(env!("CARGO_BIN_EXE_evidence_digest"))
        .arg("extract-captured-source")
        .arg(&reviewed_inventory)
        .arg(&capture)
        .arg(&destination)
        .status()
        .expect("symlinked extraction starts");
    assert!(
        !symlinked.success(),
        "symlinked captured archive was accepted"
    );
}

#[test]
fn captured_source_path_replacement_after_authentication_is_refused() {
    let directory = tempfile::tempdir().expect("captured source replacement fixture root");
    let capture = directory.path().join("capture");
    let identity = capture.join("evidence/identity");
    let destination = directory.path().join("extracted");
    fs::create_dir_all(&identity).expect("capture identity root is created");
    fs::create_dir(&destination).expect("controlled extraction root is created");
    let archive = identity.join("effective-source-tree.tar");
    let mut builder = Builder::new(File::create(&archive).expect("source archive is created"));
    append_tar_file(&mut builder, "safe.txt", b"authenticated bytes");
    builder.finish().expect("source archive is finished");
    let reviewed_inventory = write_captured_source_trust_fixture(&capture, &archive);

    let replacement = directory.path().join("replacement-source.tar");
    let mut builder = Builder::new(File::create(&replacement).expect("replacement is created"));
    append_tar_file(&mut builder, "replacement.txt", b"different bytes");
    builder.finish().expect("replacement archive is finished");

    let output = Command::new(env!("CARGO_BIN_EXE_evidence_digest"))
        .arg("extract-captured-source")
        .arg(&reviewed_inventory)
        .arg(&capture)
        .arg(&destination)
        .env(
            "AIPERF_TEST_REPLACE_CAPTURED_SOURCE_AFTER_AUTHENTICATION",
            &replacement,
        )
        .output()
        .expect("replacement-resistant extraction starts");
    assert!(
        !output.status.success(),
        "captured source path replacement was accepted"
    );
}

#[test]
fn captured_source_extraction_uses_an_owned_immutable_snapshot() {
    let directory = tempfile::tempdir().expect("captured source snapshot fixture root");
    let capture = directory.path().join("capture");
    let identity = capture.join("evidence/identity");
    let destination = directory.path().join("extracted");
    fs::create_dir_all(&identity).expect("capture identity root is created");
    fs::create_dir(&destination).expect("controlled extraction root is created");
    let archive = identity.join("effective-source-tree.tar");
    let mut builder = Builder::new(File::create(&archive).expect("source archive is created"));
    append_tar_file(&mut builder, "safe.txt", b"reviewed bytes");
    builder.finish().expect("source archive is finished");
    let reviewed_inventory = write_captured_source_trust_fixture(&capture, &archive);

    let replacement = directory.path().join("same-inode-replacement.tar");
    let mut builder = Builder::new(File::create(&replacement).expect("replacement is created"));
    append_tar_file(&mut builder, "build.rs", b"unreviewed bytes");
    builder.finish().expect("replacement archive is finished");
    let replacement_bytes = fs::read(&replacement).expect("replacement bytes are readable");

    let output = Command::new(env!("CARGO_BIN_EXE_evidence_digest"))
        .arg("extract-captured-source")
        .arg(&reviewed_inventory)
        .arg(&capture)
        .arg(&destination)
        .env(
            "AIPERF_TEST_REWRITE_CAPTURED_SOURCE_AFTER_SNAPSHOT",
            &replacement,
        )
        .output()
        .expect("snapshot extraction starts");
    assert!(!output.status.success(), "same-inode mutation was accepted");
    assert_eq!(
        fs::read_dir(&destination)
            .expect("destination remains readable")
            .count(),
        0,
        "caller mutation caused source bytes to be written"
    );
    assert_eq!(
        fs::read(&archive).expect("caller archive remains readable"),
        replacement_bytes,
        "same-inode mutation seam did not run"
    );
}

#[test]
fn staged_bundle_path_replacement_after_authentication_is_refused() {
    let directory = tempfile::tempdir().expect("staged bundle replacement fixture root");
    let capture = directory.path().join("capture");
    let evidence = capture.join("evidence");
    let extraction_parent = directory.path().join("controlled");
    fs::create_dir_all(&evidence).expect("evidence root is created");
    fs::create_dir(&extraction_parent).expect("controlled root is created");
    fs::write(evidence.join("sample.txt"), b"sealed evidence").expect("evidence is written");
    let manifest = capture.join("evidence-manifest.json");
    fs::write(
        &manifest,
        serde_json::to_vec(&serde_json::json!({
            "schema_version": 1,
            "files": [{
                "path": "sample.txt",
                "bytes": 15,
                "blake3": format!("blake3:{}", blake3::hash(b"sealed evidence")),
            }],
        }))
        .expect("manifest serializes"),
    )
    .expect("manifest is written");
    let bundle = directory.path().join("bundle.tar.gz");
    create_bundle(&capture, &bundle);

    let replacement_capture = directory.path().join("replacement-capture");
    fs::create_dir_all(replacement_capture.join("evidence"))
        .expect("replacement evidence root is created");
    fs::write(
        replacement_capture.join("evidence/other.txt"),
        b"replacement evidence",
    )
    .expect("replacement evidence is written");
    fs::write(
        replacement_capture.join("evidence-manifest.json"),
        b"{\"schema_version\":1,\"files\":[]}",
    )
    .expect("replacement manifest is written");
    let replacement = directory.path().join("replacement.tar.gz");
    create_bundle(&replacement_capture, &replacement);
    let receipt = directory.path().join("bundle-verification.json");

    let output = Command::new(env!("CARGO_BIN_EXE_evidence_digest"))
        .arg("verify-staged-bundle")
        .arg("review1i")
        .arg(&bundle)
        .arg(&manifest)
        .arg(&extraction_parent)
        .arg(&receipt)
        .env(
            "AIPERF_TEST_REPLACE_STAGED_BUNDLE_AFTER_AUTHENTICATION",
            &replacement,
        )
        .output()
        .expect("replacement-resistant staged verifier starts");
    assert!(
        !output.status.success(),
        "staged bundle path replacement was accepted"
    );
}

#[test]
fn capture_shell_verifies_the_completed_bundle_before_authoring_its_receipt() {
    let directory = tempfile::tempdir().expect("capture bundle shell fixture root");
    let capture = directory.path().join("capture");
    let evidence = capture.join("evidence");
    fs::create_dir_all(&evidence).expect("evidence root is created");
    fs::write(evidence.join("sample.txt"), b"sealed evidence").expect("evidence is written");
    let manifest = capture.join("evidence-manifest.json");
    fs::write(
        &manifest,
        serde_json::to_vec(&serde_json::json!({
            "schema_version": 1,
            "files": [{
                "path": "sample.txt",
                "bytes": 15,
                "blake3": format!("blake3:{}", blake3::hash(b"sealed evidence")),
            }],
        }))
        .expect("manifest serializes"),
    )
    .expect("manifest is written");
    let bundle = directory.path().join("valid-bundle.tar.gz");
    create_bundle(&capture, &bundle);
    let valid_extraction = directory.path().join("valid-extraction");
    fs::create_dir(&valid_extraction).expect("valid controlled root is created");
    let valid_receipt = directory.path().join("valid-bundle-verification.json");
    let script = repository_root().join("rust/scripts/capture-plugin-baseline.sh");
    let valid = Command::new("sh")
        .arg(&script)
        .arg("--bundle-verification-self-test")
        .arg(directory.path().join("valid.lock"))
        .arg(env!("CARGO_BIN_EXE_evidence_digest"))
        .arg(&bundle)
        .arg(&manifest)
        .arg(&valid_extraction)
        .arg(&valid_receipt)
        .output()
        .expect("valid completed-bundle shell verification starts");
    assert!(
        valid.status.success(),
        "valid completed bundle was refused: {}",
        String::from_utf8_lossy(&valid.stderr)
    );
    assert!(valid_receipt.is_file(), "valid receipt was not authored");

    let malicious_bundle = directory.path().join("malicious-bundle.tar.gz");
    let encoder = GzEncoder::new(
        File::create(&malicious_bundle).expect("malicious bundle is created"),
        Compression::default(),
    );
    let mut builder = Builder::new(encoder);
    append_tar_file_with_raw_name(&mut builder, b"../outside", b"attack");
    builder
        .into_inner()
        .expect("malicious tar stream is finished")
        .finish()
        .expect("malicious gzip stream is finished");
    let malicious_extraction = directory.path().join("malicious-extraction");
    fs::create_dir(&malicious_extraction).expect("malicious controlled root is created");
    let malicious_receipt = directory.path().join("malicious-bundle-verification.json");
    let outside = directory.path().join("outside");
    fs::write(&outside, b"sentinel").expect("outside sentinel is written");
    let rejected = Command::new("sh")
        .arg(script)
        .arg("--bundle-verification-self-test")
        .arg(directory.path().join("malicious.lock"))
        .arg(env!("CARGO_BIN_EXE_evidence_digest"))
        .arg(&malicious_bundle)
        .arg(&manifest)
        .arg(&malicious_extraction)
        .arg(&malicious_receipt)
        .output()
        .expect("malicious completed-bundle shell verification starts");
    assert!(!rejected.status.success(), "malicious bundle was accepted");
    assert!(
        !malicious_receipt.exists(),
        "verification receipt was authored after archive refusal"
    );
    assert_eq!(
        fs::read(&outside).expect("outside sentinel remains"),
        b"sentinel"
    );
}

fn copy_tree(source: &Path, destination: &Path) {
    fs::create_dir_all(destination).expect("tree destination is created");
    for entry in fs::read_dir(source).expect("tree source is readable") {
        let entry = entry.expect("tree entry is readable");
        let target = destination.join(entry.file_name());
        if entry.path().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), target).expect("tree file is copied");
        }
    }
}

fn create_bundle(capture_root: &Path, bundle: &Path) {
    let status = Command::new("tar")
        .args([
            "--sort=name",
            "--mtime=@0",
            "--owner=0",
            "--group=0",
            "--numeric-owner",
            "-czf",
        ])
        .arg(bundle)
        .arg("-C")
        .arg(capture_root)
        .args(["evidence", "evidence-manifest.json"])
        .status()
        .expect("tar starts");
    assert!(
        status.success(),
        "deterministic evidence archive is created"
    );
}

fn current_identity_json() -> String {
    let inventory =
        fs::read_to_string(repository_root().join("rust/benchmarks/plugin-parity.yaml"))
            .expect("repository inventory is readable");
    let start_marker = "experiment_identity_json: |\n";
    let start = inventory
        .find(start_marker)
        .expect("identity marker exists")
        + start_marker.len();
    let end = inventory[start..]
        .find("experiment_identity_digest:")
        .map(|offset| start + offset)
        .expect("identity digest marker exists");
    inventory[start..end]
        .lines()
        .map(|line| line.strip_prefix("  ").expect("identity line is indented"))
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}

fn write_measurement_receipt(identity: &Path, generation: &str) {
    let experiment_identity_blake3 = digest(&identity.join("experiment-identity.json"));
    let exporter_corpus_blake3 = digest(&identity.join("exporter-static-calibration-corpus.json"));
    let inventory: serde_yaml::Value = serde_yaml::from_str(
        &fs::read_to_string(repository_root().join("rust/benchmarks/plugin-parity.yaml"))
            .expect("repository inventory is readable"),
    )
    .expect("repository inventory parses");
    let mut builds = serde_json::to_value(&inventory["build_commands"])
        .expect("build command fixture converts to JSON");
    let build_map = builds
        .as_object_mut()
        .expect("build command fixture is an object");
    for (index, name) in ["default", "engine", "grpc", "parquet", "dynosim", "full"]
        .into_iter()
        .enumerate()
    {
        let build = build_map[name]
            .as_object_mut()
            .expect("build fixture is an object");
        build.insert(
            "target_dir".to_owned(),
            serde_json::json!(format!(
                "/cargo-target/native-plugin-baseline/task1-{generation}/build-{name}"
            )),
        );
        build.insert(
            "first_build_nanoseconds".to_owned(),
            serde_json::json!(10_000_000_000_u64 + index as u64),
        );
        build.insert(
            "second_build_nanoseconds".to_owned(),
            serde_json::json!(1_000_000_000_u64 + index as u64),
        );
        build.insert(
            "artifact_digest".to_owned(),
            serde_json::json!(format!("blake3:{:064x}", 0x100 + index)),
        );
        build.insert(
            "artifact_bytes".to_owned(),
            serde_json::json!(50_000_000_u64 + index as u64),
        );
        build.insert(
            "first_log_digest".to_owned(),
            serde_json::json!(format!("blake3:{:064x}", 0x200 + index)),
        );
        build.insert(
            "second_log_digest".to_owned(),
            serde_json::json!(format!("blake3:{:064x}", 0x300 + index)),
        );
    }

    let mut scenarios = serde_json::to_value(&inventory["runtime_scenarios"])
        .expect("runtime scenario fixture converts to JSON");
    for (index, scenario) in scenarios
        .as_array_mut()
        .expect("runtime scenario fixture is an array")
        .iter_mut()
        .enumerate()
    {
        let scenario = scenario
            .as_object_mut()
            .expect("runtime scenario fixture is an object");
        let name = scenario["name"]
            .as_str()
            .expect("runtime scenario has a name")
            .to_owned();
        scenario.insert(
            "artifact_digest".to_owned(),
            serde_json::json!(format!("blake3:{:064x}", 0x400 + index)),
        );
        scenario.insert(
            "process_log_digest".to_owned(),
            serde_json::json!(format!("blake3:{:064x}", 0x500 + index)),
        );
        let observation = if name == "exporter_100k" {
            serde_json::json!({
                "duration_seconds": 32.0,
                "exporter_nanoseconds_per_record": 20_000.0,
                "allocation_count_per_successful_request": 700.0,
                "allocated_bytes_per_successful_request": 100_000.0,
                "ttft_p50": 5.0,
                "ttft_p90": 5.0,
                "ttft_p99": 5.0,
                "itl_p50": 2.5,
                "itl_p90": 2.5,
                "itl_p99": 2.5
            })
        } else {
            serde_json::json!({
                "duration_seconds": 31.0 + index as f64 / 10.0,
                "successful_requests_per_second": 1000.0 + index as f64,
                "output_tokens_per_second": 30_000.0 + index as f64,
                "cpu_nanoseconds_per_successful_request": 400_000.0 + index as f64,
                "ttft_p50": 5.0,
                "ttft_p90": 6.0,
                "ttft_p99": 7.0,
                "itl_p50": 1.0,
                "itl_p90": 1.1,
                "itl_p99": 1.2
            })
        };
        scenario.insert("baseline_observation".to_owned(), observation);
        if name == "exporter_100k" {
            scenario.insert("request_budget".to_owned(), serde_json::json!(1_600_000));
            scenario.insert("corpus_records".to_owned(), serde_json::json!(100_000));
            scenario.insert("sample_repetitions".to_owned(), serde_json::json!(16));
            scenario.insert("processed_records".to_owned(), serde_json::json!(1_600_000));
            scenario.insert(
                "retained_artifact_records".to_owned(),
                serde_json::json!(100_000),
            );
        }
    }
    let runtime_measurements = scenarios
        .as_array()
        .expect("runtime scenarios are an array")
        .iter()
        .map(|scenario| {
            let scenario = scenario.as_object().expect("runtime scenario is an object");
            let name = scenario["name"]
                .as_str()
                .expect("runtime scenario has a name")
                .to_owned();
            (
                name,
                serde_json::json!({
                    "artifact_digest": scenario["artifact_digest"],
                    "process_log_digest": scenario["process_log_digest"],
                    "baseline_observation": scenario["baseline_observation"],
                }),
            )
        })
        .collect::<serde_json::Map<_, _>>();

    let allocation = serde_json::json!({
        "endpoint_preparation": {"iterations": 10000, "allocations_per_request": 301.0, "allocated_bytes_per_request": 20001.0},
        "endpoint_formatting": {"iterations": 10000, "allocations_per_request": 31.0, "allocated_bytes_per_request": 4001.0},
        "transport_dispatch": {"iterations": 10000, "allocations_per_request": 51.0, "allocated_bytes_per_request": 41001.0},
        "response_reduction": {"iterations": 10000, "chunks_per_response": 32, "allocations_per_request": 12.0, "allocated_bytes_per_request": 3528.0},
        "full_successful_request": {"iterations": 10000, "allocations_per_request": 141.0, "allocated_bytes_per_request": 51001.0},
        "exporter_capture": {
            "iterations": 1600000,
            "corpus_records": 100000,
            "sample_repetitions": 16,
            "processed_records": 1600000,
            "retained_artifact_records": 100000,
            "allocations_per_request": 700.0,
            "allocated_bytes_per_request": 100000.0,
            "exporter_interval_nanoseconds": 32000000000_u64,
            "exporter_nanoseconds_per_record": 20000.0
        }
    });
    let allocation_sources = serde_json::json!({
        "allocation_log": {"path": "probes/allocation-probes.log", "bytes": 1234, "blake3": CAPTURED_BUILD_DIGEST},
        "exporter_log": {"path": "runtime/exporter-100k/process.log", "bytes": 5678, "blake3": CAPTURED_BUILD_DIGEST},
        "exporter_observation": {"path": "runtime/exporter-100k/observation.json", "bytes": 901, "blake3": CAPTURED_BUILD_DIGEST}
    });
    let engine_artifact_blake3 = builds["engine"]["artifact_digest"]
        .as_str()
        .expect("engine fixture has an artifact digest")
        .to_owned();
    let exporter_repetition_receipts = (0_u64..16)
        .map(|ordinal| {
            serde_json::json!({
                "schema_version": 1,
                "experiment_identity_blake3": experiment_identity_blake3,
                "attempt_ordinal": 0,
                "scenario_id": "exporter_100k",
                "pair_id": "task1-static-calibration",
                "member": "static",
                "repetition_ordinal": ordinal,
                "corpus_blake3": exporter_corpus_blake3,
                "processed_records": 100000,
                "observable_kind": "artifact_tree",
                "raw_observable_blake3": CAPTURED_RAW_OBSERVABLE_DIGEST,
                "comparison_observable_blake3": CAPTURED_RAW_OBSERVABLE_DIGEST,
                "provenance_receipt_blake3": CAPTURED_PROVENANCE_DIGEST,
                "active_duration_ns": 2000000000_u64,
                "build_artifact_blake3": engine_artifact_blake3,
                "build_receipt_blake3": CAPTURED_BUILD_DIGEST
            })
        })
        .collect::<Vec<_>>();
    let mut exporter_repetition_receipt_bytes =
        serde_json::to_vec(&exporter_repetition_receipts).expect("receipt vector serializes");
    exporter_repetition_receipt_bytes.push(b'\n');
    fs::write(
        identity.join("measurement-results.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema_version": 1,
            "generation": generation,
            "build_commands": builds,
            "runtime_measurements": runtime_measurements,
            "allocation_probe": allocation,
            "allocation_sources": allocation_sources,
            "exporter_observable_policy_blake3": digest(&identity.join("exporter-observable-policy.json")),
            "exporter_corpus_blake3": exporter_corpus_blake3,
            "exporter_build_receipt_blake3": CAPTURED_BUILD_DIGEST,
            "exporter_repetition_receipts_blake3": format!("blake3:{}", blake3::hash(&exporter_repetition_receipt_bytes)),
            "exporter_repetition_receipts": exporter_repetition_receipts
        }))
        .expect("measurement receipt serializes"),
    )
    .expect("capture-authored measurement receipt is written");
}

fn prepare_capture_layout_with_workspace_tree_command(
    root: &Path,
    generation: &str,
    workspace_tree_command: &str,
    sample: &[u8],
) -> std::path::PathBuf {
    let evidence = root.join("evidence");
    let identity = evidence.join("identity");
    write_pre_capture_receipts(
        &identity,
        &valid_invalidations(),
        "rust/scripts/capture-plugin-baseline.sh\n",
    );
    let mut identity_json: serde_json::Value =
        serde_json::from_str(&current_identity_json()).expect("identity fixture parses");
    identity_json["rustc"] = serde_json::json!("rustc captured 1.98.0;LLVM 22.1.8");
    fs::write(
        identity.join("experiment-identity.json"),
        serde_json::to_vec_pretty(&identity_json).expect("captured identity serializes"),
    )
    .expect("capture-authored identity is written");
    for name in [
        "exporter-observable-policy.json",
        "exporter-static-calibration-corpus.json",
    ] {
        fs::copy(
            repository_root().join("rust/benchmarks").join(name),
            identity.join(name),
        )
        .expect("pre-run exporter source configuration is copied");
    }
    for (name, contents) in [
        (
            "cargo-metadata.json",
            b"{\"packages\":[],\"workspace_members\":[]}".as_slice(),
        ),
        ("cargo-tree-workspace.txt", b"workspace tree\n"),
        ("cargo-tree-cli.txt", b"cli tree\n"),
    ] {
        fs::write(identity.join(name), contents).expect("captured topology input is written");
    }
    fs::write(
        identity.join("package-topology.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema_version": 1,
            "generation": generation,
            "host_commit": "caa3ff6fcf20ffe36a7704abe16274bedadbb9fb",
            "rustc": identity_json["rustc"],
            "target": identity_json["target"],
            "cargo_profile": identity_json["cargo_profile"],
            "measurement": {
                "commands": [
                    "cargo metadata --locked --format-version 1",
                    workspace_tree_command,
                    "cargo tree --locked -p aiperf-cli --edges normal,build --prefix depth",
                ],
                "cargo_lock_blake3": digest(&identity.join("Cargo.lock")),
                "cargo_metadata_blake3": digest(&identity.join("cargo-metadata.json")),
                "cargo_tree_blake3": digest(&identity.join("cargo-tree-workspace.txt")),
                "cargo_cli_tree_blake3": digest(&identity.join("cargo-tree-cli.txt")),
                "raw_metadata": "identity/cargo-metadata.json",
                "raw_tree": "identity/cargo-tree-workspace.txt",
                "raw_cli_tree": "identity/cargo-tree-cli.txt",
            },
            "workspace_packages": [],
        }))
        .expect("captured topology serializes"),
    )
    .expect("captured topology is written");
    write_measurement_receipt(&identity, generation);
    fs::write(evidence.join("sample.txt"), sample).expect("sample evidence is written");
    let manifest = root.join("evidence-manifest.json");
    fn collect(root: &Path, directory: &Path, files: &mut Vec<serde_json::Value>) {
        let mut entries = fs::read_dir(directory)
            .expect("evidence directory is readable")
            .collect::<Result<Vec<_>, _>>()
            .expect("evidence entries are readable");
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let path = entry.path();
            if path.is_dir() {
                collect(root, &path, files);
            } else {
                files.push(serde_json::json!({
                    "path": path.strip_prefix(root).expect("evidence path is relative").to_string_lossy(),
                    "bytes": fs::metadata(&path).expect("evidence metadata").len(),
                    "blake3": digest(&path),
                }));
            }
        }
    }
    let mut files = Vec::new();
    collect(&evidence, &evidence, &mut files);
    fs::write(
        &manifest,
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema_version": 1,
            "files": files,
        }))
        .expect("manifest serializes"),
    )
    .expect("manifest is written");
    let release_tag = format!("native-plugin-baseline-caa3ff6f-{generation}-final");
    let bundle = root.join(format!("aiperf-{release_tag}.tar.gz"));
    create_bundle(root, &bundle);
    let locator = root.join("bundle-locator.json");
    fs::write(
        &locator,
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema_version": 1,
            "repository": "https://github.com/ajcasagrande/rust-native-plugin-lab",
            "recommended_release_tag": release_tag,
            "asset_name": bundle.file_name().expect("bundle name").to_string_lossy(),
            "publication_status": "ready_for_controller_publication",
            "archive_verification_status": "extracted_manifest_verified",
            "staged_path": bundle.to_string_lossy(),
            "bytes": fs::metadata(&bundle).expect("bundle metadata").len(),
            "blake3": digest(&bundle),
            "manifest_path": "artifacts/native-plugin-baseline/evidence-manifest.json",
            "manifest_bytes": fs::metadata(&manifest).expect("manifest metadata").len(),
            "manifest_blake3": digest(&manifest),
            "stable_url": format!("https://github.com/ajcasagrande/rust-native-plugin-lab/releases/download/{release_tag}/{}", bundle.file_name().expect("bundle name").to_string_lossy()),
        }))
        .expect("locator serializes"),
    )
    .expect("staged locator is written");
    fs::write(
        root.join("bundle-verification.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema_version": 1,
            "generation": generation,
            "status": "extracted_manifest_verified",
            "bundle_bytes": fs::metadata(&bundle).expect("bundle metadata").len(),
            "bundle_blake3": digest(&bundle),
            "manifest_bytes": fs::metadata(&manifest).expect("manifest metadata").len(),
            "manifest_blake3": digest(&manifest),
        }))
        .expect("bundle verification serializes"),
    )
    .expect("bundle verification receipt is written");
    identity
}

fn prepare_capture_layout(root: &Path, generation: &str) -> std::path::PathBuf {
    prepare_capture_layout_with_workspace_tree_command(
        root,
        generation,
        "cargo tree --locked --workspace --edges normal,build --prefix depth",
        b"captured sample",
    )
}

fn staged_bundle(capture: &Path, generation: &str) -> std::path::PathBuf {
    capture.join(format!(
        "aiperf-native-plugin-baseline-caa3ff6f-{generation}-final.tar.gz"
    ))
}

fn run_publication_locator(
    generation: &str,
    capture: &Path,
    output: &Path,
) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_evidence_digest"))
        .arg("publication-locator")
        .arg(generation)
        .arg(capture)
        .arg(output)
        .output()
        .expect("publication-locator starts")
}

fn run_publication_verification(
    generation: &str,
    capture: &Path,
    publishable_locator: &Path,
    downloaded_bundle: &Path,
    downloaded_manifest: &Path,
    downloaded_locator: &Path,
    extraction_parent: &Path,
    output: &Path,
) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_evidence_digest"))
        .arg("verify-publication")
        .arg(generation)
        .arg(capture)
        .arg(publishable_locator)
        .arg(downloaded_bundle)
        .arg(downloaded_manifest)
        .arg(downloaded_locator)
        .arg(extraction_parent)
        .arg(output)
        .output()
        .expect("verify-publication starts")
}

fn prepare_independent_publication_download(
    directory: &Path,
    capture: &Path,
    generation: &str,
) -> (
    std::path::PathBuf,
    std::path::PathBuf,
    std::path::PathBuf,
    std::path::PathBuf,
) {
    let publication = directory.join("publication");
    let download = directory.join("independent-download");
    fs::create_dir(&publication).expect("publication directory is created");
    fs::create_dir(&download).expect("independent download directory is created");
    let publishable_locator = publication.join("bundle-locator.json");
    let output = run_publication_locator(generation, capture, &publishable_locator);
    assert!(
        output.status.success(),
        "publication locator failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let bundle = staged_bundle(capture, generation);
    let downloaded_bundle = download.join(bundle.file_name().expect("bundle name exists"));
    let downloaded_manifest = download.join("evidence-manifest.json");
    let downloaded_locator = download.join("bundle-locator.json");
    fs::copy(bundle, &downloaded_bundle).expect("bundle download is copied");
    fs::copy(capture.join("evidence-manifest.json"), &downloaded_manifest)
        .expect("manifest download is copied");
    fs::copy(&publishable_locator, &downloaded_locator).expect("locator download is copied");
    (
        publishable_locator,
        downloaded_bundle,
        downloaded_manifest,
        downloaded_locator,
    )
}

#[test]
fn publication_commands_author_verified_artifacts_consumed_by_postpublication() {
    let directory = tempfile::tempdir().expect("automatic publication fixture root");
    let capture = directory.path().join("task1-review1i-final");
    prepare_capture_layout(&capture, "review1i");
    let (publishable_locator, downloaded_bundle, downloaded_manifest, downloaded_locator) =
        prepare_independent_publication_download(directory.path(), &capture, "review1i");
    let extraction_parent = directory.path().join("extraction");
    fs::create_dir(&extraction_parent).expect("extraction parent is created");
    let receipt = capture.join("publication-verification.json");
    let output = run_publication_verification(
        "review1i",
        &capture,
        &publishable_locator,
        &downloaded_bundle,
        &downloaded_manifest,
        &downloaded_locator,
        &extraction_parent,
        &receipt,
    );
    assert!(
        output.status.success(),
        "publication verification failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let locator: serde_json::Value = serde_json::from_slice(
        &fs::read(&downloaded_locator).expect("downloaded locator is readable"),
    )
    .expect("downloaded locator parses");
    assert_eq!(locator["publication_status"], "published_and_verified");
    assert_eq!(
        locator["archive_verification_status"],
        "downloaded_extracted_manifest_verified"
    );
    assert_eq!(
        locator["manifest_path"],
        "artifacts/native-plugin-baseline/evidence-manifest.json"
    );
    let verification: serde_json::Value =
        serde_json::from_slice(&fs::read(&receipt).expect("publication receipt is readable"))
            .expect("publication receipt parses");
    assert_eq!(verification["schema_version"], 1);
    assert_eq!(verification["generation"], "review1i");
    assert_eq!(
        verification["status"],
        "independently_downloaded_extracted_manifest_verified"
    );
    assert_eq!(
        verification["downloaded_bundle_path"].as_str(),
        downloaded_bundle.to_str()
    );
    assert_eq!(
        verification["downloaded_manifest_path"].as_str(),
        downloaded_manifest.to_str()
    );
    assert_eq!(
        verification["published_locator_path"].as_str(),
        downloaded_locator.to_str()
    );

    let inventory = directory.path().join("plugin-parity.yaml");
    let topology = directory.path().join("compact/package-topology.json");
    fs::create_dir_all(topology.parent().expect("topology parent exists"))
        .expect("compact output directory is created");
    fs::copy(
        repository_root().join("rust/benchmarks/plugin-parity.yaml"),
        &inventory,
    )
    .expect("inventory fixture is copied");
    let consumed = run_refresh(
        "postpublication",
        "review1i",
        &inventory,
        &capture,
        Some(&topology),
    );
    assert!(
        consumed.status.success(),
        "postpublication refused authored receipt: {}",
        String::from_utf8_lossy(&consumed.stderr)
    );
}

#[test]
fn publication_verifier_rejects_tampered_download_without_partial_receipt() {
    let directory = tempfile::tempdir().expect("tampered publication fixture root");
    let capture = directory.path().join("task1-review1i-final");
    prepare_capture_layout(&capture, "review1i");
    let (publishable_locator, downloaded_bundle, downloaded_manifest, downloaded_locator) =
        prepare_independent_publication_download(directory.path(), &capture, "review1i");
    fs::write(&downloaded_bundle, b"tampered downloaded bundle")
        .expect("downloaded bundle is tampered");
    let extraction_parent = directory.path().join("extraction");
    fs::create_dir(&extraction_parent).expect("extraction parent is created");
    let receipt = capture.join("publication-verification.json");
    let output = run_publication_verification(
        "review1i",
        &capture,
        &publishable_locator,
        &downloaded_bundle,
        &downloaded_manifest,
        &downloaded_locator,
        &extraction_parent,
        &receipt,
    );
    assert!(!output.status.success(), "tampered download was admitted");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("exact staged capture bytes"),
        "unexpected tamper refusal: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!receipt.exists(), "failed verification left a receipt");
}

#[cfg(unix)]
#[test]
fn publication_verifier_rejects_staged_bundle_alias_without_receipt() {
    let directory = tempfile::tempdir().expect("publication alias fixture root");
    let capture = directory.path().join("task1-review1i-final");
    prepare_capture_layout(&capture, "review1i");
    let (publishable_locator, _, downloaded_manifest, downloaded_locator) =
        prepare_independent_publication_download(directory.path(), &capture, "review1i");
    let staged_bundle = staged_bundle(&capture, "review1i");
    let extraction_parent = directory.path().join("extraction");
    fs::create_dir(&extraction_parent).expect("extraction parent is created");
    let receipt = capture.join("publication-verification.json");
    let output = run_publication_verification(
        "review1i",
        &capture,
        &publishable_locator,
        &staged_bundle,
        &downloaded_manifest,
        &downloaded_locator,
        &extraction_parent,
        &receipt,
    );
    assert!(!output.status.success(), "staged bundle alias was admitted");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("staged bundle alias"),
        "unexpected alias refusal: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!receipt.exists(), "alias refusal left a receipt");
}

#[cfg(unix)]
#[test]
fn publication_verifier_rejects_publishable_locator_alias_without_receipt() {
    let directory = tempfile::tempdir().expect("publication locator alias fixture root");
    let capture = directory.path().join("task1-review1i-final");
    prepare_capture_layout(&capture, "review1i");
    let (publishable_locator, downloaded_bundle, downloaded_manifest, _) =
        prepare_independent_publication_download(directory.path(), &capture, "review1i");
    let extraction_parent = directory.path().join("extraction");
    fs::create_dir(&extraction_parent).expect("extraction parent is created");
    let receipt = capture.join("publication-verification.json");
    let output = run_publication_verification(
        "review1i",
        &capture,
        &publishable_locator,
        &downloaded_bundle,
        &downloaded_manifest,
        &publishable_locator,
        &extraction_parent,
        &receipt,
    );
    assert!(
        !output.status.success(),
        "publishable locator alias was admitted"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("downloaded locator alias"),
        "unexpected locator alias refusal: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!receipt.exists(), "locator alias refusal left a receipt");
}

#[test]
fn publication_verifier_rejects_mismatched_locator_identity() {
    for field in ["recommended_release_tag", "asset_name", "stable_url"] {
        let directory = tempfile::tempdir().expect("locator mismatch fixture root");
        let capture = directory.path().join("task1-review1i-final");
        prepare_capture_layout(&capture, "review1i");
        let (publishable_locator, downloaded_bundle, downloaded_manifest, downloaded_locator) =
            prepare_independent_publication_download(directory.path(), &capture, "review1i");
        let mut locator: serde_json::Value = serde_json::from_slice(
            &fs::read(&downloaded_locator).expect("downloaded locator is readable"),
        )
        .expect("downloaded locator parses");
        locator[field] = serde_json::json!("mismatched-publication-identity");
        fs::write(
            &downloaded_locator,
            serde_json::to_vec_pretty(&locator).expect("mismatched locator serializes"),
        )
        .expect("mismatched locator is written");
        let extraction_parent = directory.path().join("extraction");
        fs::create_dir(&extraction_parent).expect("extraction parent is created");
        let receipt = capture.join("publication-verification.json");
        let output = run_publication_verification(
            "review1i",
            &capture,
            &publishable_locator,
            &downloaded_bundle,
            &downloaded_manifest,
            &downloaded_locator,
            &extraction_parent,
            &receipt,
        );
        assert!(
            !output.status.success(),
            "mismatched locator field {field} was admitted"
        );
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("exact publishable locator bytes"),
            "unexpected {field} refusal: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!receipt.exists(), "locator refusal left a receipt");
    }
}

#[test]
fn publication_verifier_refuses_overwrite_before_extraction_without_residue() {
    let directory = tempfile::tempdir().expect("publication overwrite fixture root");
    let capture = directory.path().join("task1-review1i-final");
    prepare_capture_layout(&capture, "review1i");
    let (publishable_locator, downloaded_bundle, downloaded_manifest, downloaded_locator) =
        prepare_independent_publication_download(directory.path(), &capture, "review1i");
    let extraction_parent = directory.path().join("extraction");
    fs::create_dir(&extraction_parent).expect("extraction parent is created");
    let marker = directory.path().join("extraction-entered");
    let receipt = capture.join("publication-verification.json");
    fs::write(&receipt, b"do not overwrite\n").expect("existing output sentinel is written");
    let output = Command::new(env!("CARGO_BIN_EXE_evidence_digest"))
        .arg("verify-publication")
        .arg("review1i")
        .arg(&capture)
        .arg(&publishable_locator)
        .arg(&downloaded_bundle)
        .arg(&downloaded_manifest)
        .arg(&downloaded_locator)
        .arg(&extraction_parent)
        .arg(&receipt)
        .env("AIPERF_EVIDENCE_EXTRACTION_MARKER", &marker)
        .output()
        .expect("overwrite-refusing publication verification starts");
    assert!(!output.status.success(), "existing receipt was overwritten");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("already exists"),
        "unexpected overwrite refusal: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        fs::read(&receipt).expect("existing output remains readable"),
        b"do not overwrite\n"
    );
    assert!(!marker.exists(), "overwrite refusal entered extraction");
    assert!(
        fs::read_dir(&capture)
            .expect("capture root remains readable")
            .all(|entry| !entry
                .expect("capture entry is readable")
                .file_name()
                .to_string_lossy()
                .contains("publication-verification.json.tmp")),
        "overwrite refusal left a partial receipt"
    );
}

fn write_publication_verification(
    capture: &Path,
    downloaded_bundle: &Path,
    downloaded_manifest: &Path,
    verification_root: &Path,
) {
    fs::create_dir_all(verification_root).expect("verification directory is created");
    let release_tag = "native-plugin-baseline-caa3ff6f-review1i-final";
    let stable_url = format!(
        "https://github.com/ajcasagrande/rust-native-plugin-lab/releases/download/{release_tag}/{}",
        downloaded_bundle
            .file_name()
            .expect("downloaded bundle name")
            .to_string_lossy()
    );
    let published_locator = verification_root.join("bundle-locator.json");
    fs::write(
        &published_locator,
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema_version": 1,
            "repository": "https://github.com/ajcasagrande/rust-native-plugin-lab",
            "recommended_release_tag": release_tag,
            "asset_name": downloaded_bundle.file_name().expect("downloaded bundle name").to_string_lossy(),
            "publication_status": "published_and_verified",
            "archive_verification_status": "downloaded_extracted_manifest_verified",
            "staged_path": downloaded_bundle.to_string_lossy(),
            "bytes": fs::metadata(downloaded_bundle).expect("download metadata").len(),
            "blake3": digest(downloaded_bundle),
            "manifest_path": "artifacts/native-plugin-baseline/evidence-manifest.json",
            "manifest_bytes": fs::metadata(downloaded_manifest).expect("manifest metadata").len(),
            "manifest_blake3": digest(downloaded_manifest),
            "stable_url": stable_url,
        }))
        .expect("published locator serializes"),
    )
    .expect("published locator is written");
    fs::write(
        capture.join("publication-verification.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema_version": 1,
            "generation": "review1i",
            "status": "independently_downloaded_extracted_manifest_verified",
            "stable_url": stable_url,
            "downloaded_bundle_path": downloaded_bundle,
            "downloaded_bundle_bytes": fs::metadata(downloaded_bundle).expect("download metadata").len(),
            "downloaded_bundle_blake3": digest(downloaded_bundle),
            "downloaded_manifest_path": downloaded_manifest,
            "downloaded_manifest_bytes": fs::metadata(downloaded_manifest).expect("manifest metadata").len(),
            "downloaded_manifest_blake3": digest(downloaded_manifest),
            "published_locator_path": published_locator,
            "published_locator_bytes": fs::metadata(&published_locator).expect("locator metadata").len(),
            "published_locator_blake3": digest(&published_locator),
        }))
        .expect("publication verification serializes"),
    )
    .expect("publication verification receipt is written");
}

#[test]
fn post_capture_rejects_topology_not_derived_from_exact_receipts() {
    let directory = tempfile::tempdir().expect("temporary topology refusal directory");
    let inventory = directory.path().join("plugin-parity.yaml");
    let capture = directory.path().join("task1-review1i-final");
    let topology = directory.path().join("package-topology.json");
    fs::copy(
        repository_root().join("rust/benchmarks/plugin-parity.yaml"),
        &inventory,
    )
    .expect("inventory fixture is copied");
    prepare_capture_layout_with_workspace_tree_command(
        &capture,
        "review1i",
        "cargo tree --locked --workspace",
        b"captured sample",
    );

    let rejected = run_refresh(
        "post-capture",
        "review1i",
        &inventory,
        &capture,
        Some(&topology),
    );
    assert!(
        !rejected.status.success(),
        "topology with noncanonical commands passed"
    );
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("captured topology"));
}

#[test]
fn post_capture_rejects_topology_with_wrong_generation_or_profile() {
    for (field_name, wrong_value) in [
        ("generation", serde_json::json!("review1h")),
        ("cargo_profile", serde_json::json!("debug")),
    ] {
        let directory = tempfile::tempdir().expect("temporary topology identity directory");
        let inventory = directory.path().join("plugin-parity.yaml");
        let capture = directory.path().join("task1-review1i-final");
        let topology = directory.path().join("package-topology.json");
        fs::copy(
            repository_root().join("rust/benchmarks/plugin-parity.yaml"),
            &inventory,
        )
        .expect("inventory fixture is copied");
        let identity = prepare_capture_layout(&capture, "review1i");
        let topology_receipt = identity.join("package-topology.json");
        let mut value: serde_json::Value = serde_json::from_slice(
            &fs::read(&topology_receipt).expect("topology receipt is readable"),
        )
        .expect("topology receipt parses");
        value[field_name] = wrong_value;
        fs::write(
            &topology_receipt,
            serde_json::to_vec_pretty(&value).expect("topology serializes"),
        )
        .expect("mutated topology receipt is written");

        let rejected = run_refresh(
            "post-capture",
            "review1i",
            &inventory,
            &capture,
            Some(&topology),
        );
        assert!(
            !rejected.status.success(),
            "topology with wrong {field_name} was admitted"
        );
        assert!(
            String::from_utf8_lossy(&rejected.stderr).contains("captured topology"),
            "wrong {field_name} was not rejected by topology identity validation: {}",
            String::from_utf8_lossy(&rejected.stderr)
        );
    }
}

#[test]
fn pre_capture_repairs_stale_derived_fields_and_is_idempotent() {
    let directory = tempfile::tempdir().expect("temporary refresh directory");
    let inventory = prepare_repository_inventory(directory.path());
    let receipts = directory.path().join("receipts");
    write_pre_capture_receipts(
        &receipts,
        &valid_invalidations(),
        "rust/scripts/capture-plugin-baseline.sh\nrust/scripts/refresh-plugin-baseline-inventory.sh\n",
    );
    let stale = fs::read_to_string(&inventory).expect("inventory fixture is readable");

    let first = run_refresh("pre-capture", "review1i", &inventory, &receipts, None);
    assert!(
        first.status.success(),
        "pre-capture refresh failed: {}",
        String::from_utf8_lossy(&first.stderr)
    );
    let refreshed = fs::read(&inventory).expect("refreshed inventory is readable");
    assert_ne!(
        refreshed,
        stale.as_bytes(),
        "stale fields were not repaired"
    );
    let text = String::from_utf8(refreshed.clone()).expect("inventory remains UTF-8");
    assert!(text.contains("admission_status: prepublication_expected_failure"));
    assert!(text.contains("expected_generation: review1i"));
    assert!(text.contains(
        "rust/e2e-tests/tests/plugin_baseline_inventory.rs is included as the semantic validator"
    ));
    for (field, receipt) in [
        ("baseline_source_tree_blake3", "source-tree.tar"),
        ("baseline_cargo_lock_blake3", "baseline-Cargo.lock"),
        (
            "measurement_source_projection_blake3",
            "measurement-source-projection.tar",
        ),
        (
            "measurement_source_projection_list_blake3",
            "measurement-source-projection.txt",
        ),
        ("effective_cargo_lock_blake3", "Cargo.lock"),
        ("effective_source_tree_blake3", "effective-source-tree.tar"),
        ("harness_blake3", "capture-plugin-baseline.sh"),
    ] {
        assert!(
            text.contains(&format!(
                "\"{field}\": \"{}\"",
                digest(&receipts.join(receipt))
            )),
            "{field} was not repaired from exact bytes"
        );
    }
    assert!(text.contains(
        "generation: review1h\n    status: invalid\n    reason: \"owned stdin was discarded\""
    ));
    let document: serde_yaml::Value = serde_yaml::from_str(&text).expect("refreshed YAML parses");
    let exporter_allocation = &document["allocation_probe"]["exporter_capture"];
    for (field, expected) in [
        ("corpus_records", 100_000),
        ("sample_repetitions", 16),
        ("processed_records", 1_600_000),
        ("retained_artifact_records", 100_000),
    ] {
        assert_eq!(
            exporter_allocation[field].as_u64(),
            Some(expected),
            "exporter allocation contract was not regenerated: {field}"
        );
    }
    for (key, relative) in [
        ("readme", "artifacts/native-plugin-baseline/README.md"),
        (
            "allocation_probe",
            "artifacts/native-plugin-baseline/allocation-probe.json",
        ),
    ] {
        let artifact = directory.path().join(relative);
        assert!(
            text.contains(&format!(
                "{key}: {{path: {relative}, bytes: {}, blake3: {}}}",
                fs::metadata(&artifact)
                    .expect("artifact metadata is readable")
                    .len(),
                digest(&artifact)
            )),
            "{key} was not regenerated from exact tracked bytes"
        );
    }

    let second = run_refresh("pre-capture", "review1i", &inventory, &receipts, None);
    assert!(
        second.status.success(),
        "second pre-capture refresh failed: {}",
        String::from_utf8_lossy(&second.stderr)
    );
    assert_eq!(
        fs::read(&inventory).expect("twice-refreshed inventory is readable"),
        refreshed,
        "pre-capture refresh is not byte-idempotent"
    );
}

#[test]
fn post_capture_binds_completed_machine_receipts_and_is_idempotent() {
    let directory = tempfile::tempdir().expect("temporary post-capture directory");
    let inventory = directory.path().join("plugin-parity.yaml");
    let capture = directory.path().join("task1-review1i-final");
    let topology = directory.path().join("package-topology.json");
    fs::copy(
        repository_root().join("rust/benchmarks/plugin-parity.yaml"),
        &inventory,
    )
    .expect("inventory fixture is copied");
    let identity = prepare_capture_layout(&capture, "review1i");
    fs::write(
        directory.path().join("allocation-probe.json"),
        br#"{"schema_version":1,"raw_log":"provisional-local.log","samples":{"exporter_capture":{"exported_record_bytes":1740800000}}}"#,
    )
    .expect("legacy compact allocation artifact is seeded");

    let first = run_refresh(
        "post-capture",
        "review1i",
        &inventory,
        &capture,
        Some(&topology),
    );
    assert!(
        first.status.success(),
        "post-capture refresh failed: {}",
        String::from_utf8_lossy(&first.stderr)
    );
    let refreshed = fs::read(&inventory).expect("post-capture inventory is readable");
    let text = String::from_utf8(refreshed.clone()).expect("inventory remains UTF-8");
    assert!(text.contains("admission_status: prepublication_expected_failure"));
    assert!(text.contains("expected_generation: review1i"));
    assert!(text.contains(&format!(
        "blake3: {}",
        digest(&identity.join("package-topology.json"))
    )));
    assert!(text.contains("release_tag: native-plugin-baseline-caa3ff6f-review1i-final"));
    assert!(text.contains("repository: https://github.com/ajcasagrande/rust-native-plugin-lab"));
    assert!(text.contains("task1-review1i/build-default"));
    assert!(!text.contains("task1-review1d/build-default"));
    assert!(text.contains("first_build_nanoseconds: 10000000000"));
    assert!(text.contains("duration_seconds: 31.0"));
    assert!(text.contains("allocations_per_request: 301.0"));
    assert!(text.contains("corpus_records: 100000"));
    assert!(text.contains("sample_repetitions: 16"));
    assert!(text.contains("pair_id: task1-static-calibration"));
    assert!(text.contains("observable_kind: artifact_tree"));
    assert!(text.contains(&format!(
        "experiment_identity_blake3: {}",
        digest(&identity.join("experiment-identity.json"))
    )));
    assert!(text.contains(&format!(
        "corpus_blake3: {}",
        digest(&identity.join("exporter-static-calibration-corpus.json"))
    )));
    assert!(text.contains(&format!(
        "comparison_observable_blake3: {CAPTURED_RAW_OBSERVABLE_DIGEST}"
    )));
    assert!(!text.contains("original_static_baseline"));
    let compact_allocation = topology
        .parent()
        .expect("compact parent exists")
        .join("allocation-probe.json");
    let compact_text = fs::read_to_string(&compact_allocation)
        .expect("compact allocation receipt is mechanically published");
    assert!(compact_text.contains("\"generation\": \"review1i\""));
    assert!(compact_text.contains("\"allocation_count_per_request\": 301.0"));
    let compact_value: serde_json::Value =
        serde_json::from_str(&compact_text).expect("compact allocation receipt parses");
    assert_eq!(
        compact_value
            .as_object()
            .expect("compact allocation is an object")
            .keys()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>(),
        [
            "allocator",
            "generation",
            "measurement",
            "samples",
            "schema_version",
            "source_receipts",
        ]
        .into_iter()
        .collect()
    );
    assert!(compact_value.get("raw_log").is_none());
    assert!(compact_text.find("exported_record_bytes").is_none());
    assert!(text.starts_with("schema_version: 1\nhost_commit: caa3ff6fcf20ffe36a7704abe16274bedadbb9fb\nrustc: rustc captured 1.98.0;LLVM 22.1.8\n"));
    assert_eq!(
        fs::read(&topology).expect("topology output exists"),
        fs::read(identity.join("package-topology.json")).expect("captured topology exists")
    );

    let second = run_refresh(
        "post-capture",
        "review1i",
        &inventory,
        &capture,
        Some(&topology),
    );
    assert!(
        second.status.success(),
        "second post-capture refresh failed: {}",
        String::from_utf8_lossy(&second.stderr)
    );
    assert_eq!(
        fs::read(&inventory).expect("twice-refreshed inventory is readable"),
        refreshed,
        "post-capture refresh is not byte-idempotent"
    );
}

#[test]
fn post_capture_requires_exact_strict_measurement_receipt() {
    for mutation in [
        "missing",
        "unknown",
        "wrong_generation",
        "extra_build",
        "old_exporter_schema",
        "wrong_exporter_identity",
        "wrong_exporter_pair",
        "wrong_exporter_corpus",
        "wrong_exporter_class",
        "unequal_exporter_comparison",
        "zero_exporter_duration",
        "exporter_build_substitution",
        "reordered_exporter_receipts",
        "extra_exporter_receipt",
    ] {
        let directory = tempfile::tempdir().expect("temporary measurement refusal directory");
        let inventory = directory.path().join("plugin-parity.yaml");
        let capture = directory.path().join("task1-review1i-final");
        let topology = directory.path().join("package-topology.json");
        fs::copy(
            repository_root().join("rust/benchmarks/plugin-parity.yaml"),
            &inventory,
        )
        .expect("inventory fixture is copied");
        let identity = prepare_capture_layout(&capture, "review1i");
        let receipt = identity.join("measurement-results.json");
        match mutation {
            "missing" => fs::remove_file(&receipt).expect("measurement receipt is removed"),
            "unknown" => {
                let mut value: serde_json::Value =
                    serde_json::from_slice(&fs::read(&receipt).expect("receipt is readable"))
                        .expect("receipt parses");
                value["unexpected"] = serde_json::json!(true);
                fs::write(
                    &receipt,
                    serde_json::to_vec_pretty(&value).expect("receipt serializes"),
                )
                .expect("unknown receipt field is written");
            }
            "wrong_generation" => {
                let mut value: serde_json::Value =
                    serde_json::from_slice(&fs::read(&receipt).expect("receipt is readable"))
                        .expect("receipt parses");
                value["generation"] = serde_json::json!("review1h");
                fs::write(
                    &receipt,
                    serde_json::to_vec_pretty(&value).expect("receipt serializes"),
                )
                .expect("wrong generation is written");
            }
            "extra_build" => {
                let mut value: serde_json::Value =
                    serde_json::from_slice(&fs::read(&receipt).expect("receipt is readable"))
                        .expect("receipt parses");
                value["build_commands"]["extra"] = value["build_commands"]["default"].clone();
                fs::write(
                    &receipt,
                    serde_json::to_vec_pretty(&value).expect("receipt serializes"),
                )
                .expect("extra build receipt is written");
            }
            mutation => {
                let mut value: serde_json::Value =
                    serde_json::from_slice(&fs::read(&receipt).expect("receipt is readable"))
                        .expect("receipt parses");
                let repetitions = value["exporter_repetition_receipts"]
                    .as_array_mut()
                    .expect("exporter receipt vector is an array");
                match mutation {
                    "old_exporter_schema" => {
                        let first = repetitions[0]
                            .as_object_mut()
                            .expect("receipt is an object");
                        first.remove("processed_records");
                        first.insert("emitted_records".to_owned(), serde_json::json!(100000));
                    }
                    "wrong_exporter_identity" => {
                        repetitions[0]["experiment_identity_blake3"] =
                            serde_json::json!(CAPTURED_BUILD_DIGEST);
                    }
                    "wrong_exporter_pair" => {
                        repetitions[0]["pair_id"] = serde_json::json!("original_static_baseline");
                    }
                    "wrong_exporter_corpus" => {
                        repetitions[0]["corpus_blake3"] = serde_json::json!(CAPTURED_BUILD_DIGEST);
                    }
                    "wrong_exporter_class" => {
                        repetitions[0]["observable_kind"] = serde_json::json!("captured_stream");
                    }
                    "unequal_exporter_comparison" => {
                        repetitions[1]["comparison_observable_blake3"] =
                            serde_json::json!(CAPTURED_BUILD_DIGEST);
                    }
                    "zero_exporter_duration" => {
                        repetitions[0]["active_duration_ns"] = serde_json::json!(0);
                    }
                    "exporter_build_substitution" => {
                        repetitions[0]["build_artifact_blake3"] =
                            serde_json::json!(CAPTURED_CORPUS_DIGEST);
                    }
                    "reordered_exporter_receipts" => repetitions.swap(0, 1),
                    "extra_exporter_receipt" => repetitions.push(repetitions[15].clone()),
                    _ => unreachable!(),
                }
                fs::write(
                    &receipt,
                    serde_json::to_vec_pretty(&value).expect("receipt serializes"),
                )
                .expect("mutated exporter receipt is written");
            }
        }

        let rejected = run_refresh(
            "post-capture",
            "review1i",
            &inventory,
            &capture,
            Some(&topology),
        );
        assert!(
            !rejected.status.success(),
            "{mutation} receipt was admitted"
        );
        assert!(
            String::from_utf8_lossy(&rejected.stderr).contains("measurement"),
            "unexpected {mutation} refusal: {}",
            String::from_utf8_lossy(&rejected.stderr)
        );
    }
}

#[test]
fn refresh_transaction_rolls_back_every_output_after_commit_failure() {
    let directory = tempfile::tempdir().expect("temporary refresh rollback directory");
    let inventory = directory.path().join("plugin-parity.yaml");
    let capture = directory.path().join("task1-review1i-final");
    let compact = directory.path().join("compact");
    let topology = compact.join("package-topology.json");
    fs::create_dir_all(&compact).expect("compact root is created");
    fs::copy(
        repository_root().join("rust/benchmarks/plugin-parity.yaml"),
        &inventory,
    )
    .expect("inventory fixture is copied");
    prepare_capture_layout(&capture, "review1i");
    for (path, bytes) in [
        (&topology, b"old topology".as_slice()),
        (&compact.join("allocation-probe.json"), b"old allocation"),
        (&compact.join("evidence-manifest.json"), b"old manifest"),
        (&compact.join("bundle-locator.json"), b"old locator"),
    ] {
        fs::write(path, bytes).expect("old compact output is written");
    }
    let old_inventory = fs::read(&inventory).expect("old inventory is readable");

    let failed = run_refresh_with_environment(
        "post-capture",
        "review1i",
        &inventory,
        &capture,
        Some(&topology),
        &[("AIPERF_REFRESH_FAIL_AFTER_RENAMES", Path::new("2"))],
    );
    assert!(
        !failed.status.success(),
        "injected transactional failure passed"
    );
    assert_eq!(
        fs::read(&inventory).expect("inventory survives"),
        old_inventory
    );
    for (path, bytes) in [
        (&topology, b"old topology".as_slice()),
        (&compact.join("allocation-probe.json"), b"old allocation"),
        (&compact.join("evidence-manifest.json"), b"old manifest"),
        (&compact.join("bundle-locator.json"), b"old locator"),
    ] {
        assert_eq!(fs::read(path).expect("compact output survives"), bytes);
    }
    assert!(
        transaction_residue(directory.path()).is_empty(),
        "returned-error rollback left backup, stage, or journal residue"
    );
}

#[test]
fn refresh_contract_recovers_a_killed_candidate_transaction_before_reuse() {
    let directory = tempfile::tempdir().expect("killed refresh candidate fixture root");
    let inventory = directory.path().join("plugin-parity.yaml");
    let capture = directory.path().join("task1-review1i-final");
    let compact = directory.path().join("compact");
    let topology = compact.join("package-topology.json");
    fs::create_dir_all(&compact).expect("compact root is created");
    fs::copy(
        repository_root().join("rust/benchmarks/plugin-parity.yaml"),
        &inventory,
    )
    .expect("inventory fixture is copied");
    prepare_capture_layout(&capture, "review1i");
    for (path, bytes) in [
        (&topology, b"old topology".as_slice()),
        (&compact.join("allocation-probe.json"), b"old allocation"),
        (&compact.join("evidence-manifest.json"), b"old manifest"),
        (&compact.join("bundle-locator.json"), b"old locator"),
    ] {
        fs::write(path, bytes).expect("old compact output is written");
    }

    let killed = run_refresh_with_environment(
        "post-capture",
        "review1i",
        &inventory,
        &capture,
        Some(&topology),
        &[(
            "AIPERF_REFRESH_KILL_AFTER_BOUNDARY",
            Path::new("destination:2"),
        )],
    );
    assert!(
        !killed.status.success(),
        "candidate kill point did not kill"
    );
    assert!(
        !transaction_residue(directory.path()).is_empty(),
        "killed candidate transaction did not leave its recovery authority"
    );

    let recovered = run_refresh(
        "post-capture",
        "review1i",
        &inventory,
        &capture,
        Some(&topology),
    );
    assert!(
        recovered.status.success(),
        "next refresh did not recover before candidate reuse: {}",
        String::from_utf8_lossy(&recovered.stderr)
    );
    assert!(
        transaction_residue(directory.path()).is_empty(),
        "recovered candidate transaction left residue"
    );
}

#[test]
fn publish_baseline_commits_only_the_complete_validated_candidate_set() {
    let directory = tempfile::tempdir().expect("temporary publication directory");
    let candidate = directory.path().join("candidate");
    let repository = directory.path().join("repository");
    let paths = [
        "rust/benchmarks/plugin-parity.yaml",
        "artifacts/native-plugin-baseline/package-topology.json",
    ];
    for (index, relative) in paths.iter().enumerate() {
        let candidate_path = candidate.join(relative);
        let repository_path = repository.join(relative);
        fs::create_dir_all(candidate_path.parent().expect("candidate parent"))
            .expect("candidate parent is created");
        fs::create_dir_all(repository_path.parent().expect("repository parent"))
            .expect("repository parent is created");
        fs::write(&candidate_path, format!("candidate-{index}\n"))
            .expect("candidate output is written");
        fs::write(&repository_path, format!("original-{index}\n"))
            .expect("original output is written");
    }

    let missing = Command::new(env!("CARGO_BIN_EXE_evidence_digest"))
        .args(["publish-baseline", "post-capture"])
        .arg(&candidate)
        .arg(&repository)
        .output()
        .expect("incomplete publish starts");
    assert!(!missing.status.success(), "incomplete candidate set passed");
    for (index, relative) in paths.iter().enumerate() {
        assert_eq!(
            fs::read_to_string(repository.join(relative)).expect("original survives"),
            format!("original-{index}\n")
        );
    }

    let published = Command::new(env!("CARGO_BIN_EXE_evidence_digest"))
        .args(["publish-baseline", "pre-capture"])
        .arg(&candidate)
        .arg(&repository)
        .output()
        .expect("complete publish starts");
    assert!(
        published.status.success(),
        "complete publish failed: {}",
        String::from_utf8_lossy(&published.stderr)
    );
    for (index, relative) in paths.iter().enumerate() {
        assert_eq!(
            fs::read_to_string(repository.join(relative)).expect("candidate is published"),
            format!("candidate-{index}\n")
        );
    }
    assert!(
        transaction_residue(&repository).is_empty(),
        "successful publication left backup, stage, or journal residue"
    );
}

#[cfg(unix)]
#[test]
fn publish_baseline_rejects_candidate_symlink_before_any_destination_change() {
    use std::os::unix::fs::symlink;

    let directory = tempfile::tempdir().expect("candidate symlink fixture root");
    let candidate = directory.path().join("candidate");
    let repository = directory.path().join("repository");
    let paths = publication_paths("pre-capture");
    for (index, relative) in paths.iter().enumerate() {
        let candidate_path = candidate.join(relative);
        let repository_path = repository.join(relative);
        fs::create_dir_all(candidate_path.parent().expect("candidate parent"))
            .expect("candidate parent is created");
        fs::create_dir_all(repository_path.parent().expect("repository parent"))
            .expect("repository parent is created");
        fs::write(&candidate_path, format!("candidate-{index}\n"))
            .expect("candidate output is written");
        fs::write(&repository_path, format!("original-{index}\n"))
            .expect("original output is written");
    }
    let symlink_candidate = candidate.join(paths[0]);
    let regular_target = candidate.join("regular-target");
    fs::write(&regular_target, b"candidate-through-symlink\n").expect("symlink target is written");
    fs::remove_file(&symlink_candidate).expect("regular candidate is removed");
    symlink(&regular_target, &symlink_candidate).expect("candidate symlink is created");

    let rejected = Command::new(env!("CARGO_BIN_EXE_evidence_digest"))
        .args(["publish-baseline", "pre-capture"])
        .arg(&candidate)
        .arg(&repository)
        .output()
        .expect("symlink candidate publication starts");
    assert!(!rejected.status.success(), "candidate symlink was accepted");
    for (index, relative) in paths.iter().enumerate() {
        assert_eq!(
            fs::read_to_string(repository.join(relative)).expect("original remains readable"),
            format!("original-{index}\n")
        );
    }
    assert!(
        transaction_residue(&repository).is_empty(),
        "candidate rejection left publication residue"
    );
}

#[test]
fn publication_recovers_after_process_death_at_a_destination_rename() {
    let directory = tempfile::tempdir().expect("publication crash fixture root");
    let candidate = directory.path().join("candidate");
    let repository = directory.path().join("repository");
    let paths = [
        "rust/benchmarks/plugin-parity.yaml",
        "artifacts/native-plugin-baseline/package-topology.json",
    ];
    for (index, relative) in paths.iter().enumerate() {
        let candidate_path = candidate.join(relative);
        let repository_path = repository.join(relative);
        fs::create_dir_all(candidate_path.parent().expect("candidate parent"))
            .expect("candidate parent is created");
        fs::create_dir_all(repository_path.parent().expect("repository parent"))
            .expect("repository parent is created");
        fs::write(&candidate_path, format!("new-generation-{index}\n"))
            .expect("candidate output is written");
        fs::write(&repository_path, format!("old-generation-{index}\n"))
            .expect("original output is written");
    }

    let killed = Command::new(env!("CARGO_BIN_EXE_evidence_digest"))
        .args(["publish-baseline", "pre-capture"])
        .arg(&candidate)
        .arg(&repository)
        .env("AIPERF_REFRESH_KILL_AFTER_BOUNDARY", "destination:1")
        .status()
        .expect("kill-point publisher starts");
    assert!(!killed.success(), "kill-point publisher did not die");
    let refused_reader = Command::new(env!("CARGO_BIN_EXE_evidence_digest"))
        .args(["verify-baseline-publication", "pre-capture"])
        .arg(&repository)
        .output()
        .expect("publication reader starts");
    assert!(
        !refused_reader.status.success(),
        "reader accepted an interrupted publication"
    );

    let recovered = Command::new(env!("CARGO_BIN_EXE_evidence_digest"))
        .args(["publish-baseline", "pre-capture"])
        .arg(&candidate)
        .arg(&repository)
        .output()
        .expect("recovery publisher starts");
    assert!(
        recovered.status.success(),
        "mandatory publication recovery failed: {}",
        String::from_utf8_lossy(&recovered.stderr)
    );
    for (index, relative) in paths.iter().enumerate() {
        assert_eq!(
            fs::read_to_string(repository.join(relative)).expect("published output exists"),
            format!("new-generation-{index}\n")
        );
    }
    let verified_reader = Command::new(env!("CARGO_BIN_EXE_evidence_digest"))
        .args(["verify-baseline-publication", "pre-capture"])
        .arg(&repository)
        .output()
        .expect("recovered publication reader starts");
    assert!(
        verified_reader.status.success(),
        "reader rejected recovered publication: {}",
        String::from_utf8_lossy(&verified_reader.stderr)
    );
}

#[test]
fn publication_reader_rejects_a_generation_changed_during_the_whole_read() {
    let directory = tempfile::tempdir().expect("publication reader race fixture root");
    let candidate = directory.path().join("candidate");
    let repository = directory.path().join("repository");
    for (index, relative) in publication_paths("pre-capture").iter().enumerate() {
        let candidate_path = candidate.join(relative);
        let repository_path = repository.join(relative);
        fs::create_dir_all(candidate_path.parent().expect("candidate parent"))
            .expect("candidate parent is created");
        fs::create_dir_all(repository_path.parent().expect("repository parent"))
            .expect("repository parent is created");
        fs::write(candidate_path, format!("new-{index}\n")).expect("candidate is written");
        fs::write(repository_path, format!("old-{index}\n")).expect("original is written");
    }
    fs::write(
        repository.join("artifacts/native-plugin-baseline/.publication-generation.json"),
        b"{\"nonce\":1,\"schema_version\":1,\"status\":\"committed\"}\n",
    )
    .expect("initial generation marker is written");
    let pause = directory.path().join("reader-paused");
    let resume = directory.path().join("reader-paused.continue");
    let mut reader = Command::new(env!("CARGO_BIN_EXE_evidence_digest"))
        .args(["verify-baseline-publication", "pre-capture"])
        .arg(&repository)
        .env("AIPERF_TEST_BASELINE_READER_PAUSE_MARKER", &pause)
        .spawn()
        .expect("publication reader starts");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !pause.exists() && std::time::Instant::now() < deadline {
        if reader
            .try_wait()
            .expect("reader state is available")
            .is_some()
        {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(
        pause.is_file(),
        "reader did not enter the whole-read boundary"
    );

    let published = Command::new(env!("CARGO_BIN_EXE_evidence_digest"))
        .args(["publish-baseline", "pre-capture"])
        .arg(&candidate)
        .arg(&repository)
        .output()
        .expect("concurrent publication starts");
    assert!(
        published.status.success(),
        "concurrent publication failed: {}",
        String::from_utf8_lossy(&published.stderr)
    );
    fs::write(resume, b"continue\n").expect("reader is resumed");
    let status = reader.wait().expect("publication reader exits");
    assert!(!status.success(), "reader accepted a mixed-generation read");
}

fn publication_paths(mode: &str) -> Vec<&'static str> {
    let mut paths = vec![
        "rust/benchmarks/plugin-parity.yaml",
        "artifacts/native-plugin-baseline/package-topology.json",
    ];
    if mode != "pre-capture" {
        paths.extend([
            "artifacts/native-plugin-baseline/allocation-probe.json",
            "artifacts/native-plugin-baseline/evidence-manifest.json",
            "artifacts/native-plugin-baseline/bundle-locator.json",
        ]);
    }
    paths
}

fn transaction_residue(root: &Path) -> Vec<std::path::PathBuf> {
    fn visit(root: &Path, residue: &mut Vec<std::path::PathBuf>) {
        for entry in fs::read_dir(root).expect("transaction directory is readable") {
            let entry = entry.expect("transaction entry is readable");
            if entry.file_type().expect("transaction entry type").is_dir() {
                visit(&entry.path(), residue);
            } else {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.contains(".refresh-") || name == ".aiperf-baseline-refresh-transaction.json"
                {
                    residue.push(entry.path());
                }
            }
        }
    }
    let mut residue = Vec::new();
    visit(root, &mut residue);
    residue
}

#[test]
fn publication_is_crash_durable_at_every_commit_boundary() {
    for mode in ["pre-capture", "post-capture"] {
        let output_count = publication_paths(mode).len();
        let mut boundaries = vec!["journal".to_owned(), "commit-ready".to_owned()];
        for ordinal in 1..=output_count {
            boundaries.extend([
                format!("backup:{ordinal}"),
                format!("destination:{ordinal}"),
                format!("cleanup:{ordinal}"),
            ]);
        }
        boundaries.extend([
            "journal-removal-ready".to_owned(),
            "journal-removal".to_owned(),
        ]);
        for boundary in boundaries {
            let directory = tempfile::tempdir().expect("publication boundary fixture root");
            let candidate = directory.path().join("candidate");
            let repository = directory.path().join("repository");
            let paths = publication_paths(mode);
            for (index, relative) in paths.iter().enumerate() {
                let candidate_path = candidate.join(relative);
                let repository_path = repository.join(relative);
                fs::create_dir_all(candidate_path.parent().expect("candidate parent"))
                    .expect("candidate parent is created");
                fs::create_dir_all(repository_path.parent().expect("repository parent"))
                    .expect("repository parent is created");
                fs::write(&candidate_path, format!("new-{index}\n")).expect("candidate is written");
                if mode == "pre-capture" || index < 3 {
                    fs::write(&repository_path, format!("old-{index}\n"))
                        .expect("original is written");
                }
            }
            let killed = Command::new(env!("CARGO_BIN_EXE_evidence_digest"))
                .args(["publish-baseline", mode])
                .arg(&candidate)
                .arg(&repository)
                .env("AIPERF_REFRESH_KILL_AFTER_BOUNDARY", &boundary)
                .status()
                .expect("boundary publisher starts");
            assert!(!killed.success(), "{mode} boundary {boundary} did not kill");
            let reader = Command::new(env!("CARGO_BIN_EXE_evidence_digest"))
                .args(["verify-baseline-publication", mode])
                .arg(&repository)
                .status()
                .expect("interrupted reader starts");
            if boundary == "journal-removal" {
                assert!(
                    reader.success(),
                    "{mode} boundary {boundary} refused the complete committed generation"
                );
            } else {
                assert!(
                    !reader.success(),
                    "{mode} boundary {boundary} did not fail closed"
                );
            }
            let recovery = Command::new(env!("CARGO_BIN_EXE_evidence_digest"))
                .args(["recover-baseline-publication", mode])
                .arg(&repository)
                .output()
                .expect("publication recovery starts");
            assert!(
                recovery.status.success(),
                "{mode} boundary {boundary} recovery failed: {}",
                String::from_utf8_lossy(&recovery.stderr)
            );
            let recovered = paths
                .iter()
                .map(|relative| fs::read_to_string(repository.join(relative)).ok())
                .collect::<Vec<_>>();
            let is_old = recovered.iter().enumerate().all(|(index, bytes)| {
                if mode == "pre-capture" || index < 3 {
                    bytes.as_deref() == Some(format!("old-{index}\n").as_str())
                } else {
                    bytes.is_none()
                }
            });
            let is_new = recovered
                .iter()
                .enumerate()
                .all(|(index, bytes)| bytes.as_deref() == Some(format!("new-{index}\n").as_str()));
            assert!(
                is_old || is_new,
                "{mode} boundary {boundary} recovered a mixed generation: {recovered:?}"
            );
            assert!(
                transaction_residue(&repository).is_empty(),
                "{mode} boundary {boundary} left transaction residue"
            );
        }
    }
}

#[test]
fn staging_failure_cleans_all_created_temporary_outputs() {
    let directory = tempfile::tempdir().expect("publication staging fixture root");
    let candidate = directory.path().join("candidate");
    let repository = directory.path().join("repository");
    for (index, relative) in publication_paths("pre-capture").iter().enumerate() {
        let candidate_path = candidate.join(relative);
        let repository_path = repository.join(relative);
        fs::create_dir_all(candidate_path.parent().expect("candidate parent"))
            .expect("candidate parent is created");
        fs::create_dir_all(repository_path.parent().expect("repository parent"))
            .expect("repository parent is created");
        fs::write(candidate_path, format!("new-{index}\n")).expect("candidate is written");
        fs::write(repository_path, format!("old-{index}\n")).expect("original is written");
    }
    let failed = Command::new(env!("CARGO_BIN_EXE_evidence_digest"))
        .args(["publish-baseline", "pre-capture"])
        .arg(&candidate)
        .arg(&repository)
        .env("AIPERF_REFRESH_FAIL_STAGING_INDEX", "1")
        .status()
        .expect("staging-failure publisher starts");
    assert!(!failed.success(), "injected staging failure passed");
    assert!(
        transaction_residue(&repository).is_empty(),
        "staging failure left temporary or journal residue"
    );
    for (index, relative) in publication_paths("pre-capture").iter().enumerate() {
        assert_eq!(
            fs::read_to_string(repository.join(relative)).expect("original survives"),
            format!("old-{index}\n")
        );
    }
}

#[test]
fn refresh_rejects_invalid_generations_and_self_invalidation() {
    let directory = tempfile::tempdir().expect("temporary generation directory");
    let inventory = directory.path().join("plugin-parity.yaml");
    let receipts = directory.path().join("receipts");
    fs::copy(
        repository_root().join("rust/benchmarks/plugin-parity.yaml"),
        &inventory,
    )
    .expect("inventory fixture is copied");
    write_pre_capture_receipts(
        &receipts,
        &valid_invalidations(),
        "rust/scripts/capture-plugin-baseline.sh\n",
    );
    for generation in [
        "review1", "review1b", "review1c", "review1d", "review1e", "review1f", "review1g",
        "review1h",
    ] {
        let rejected = run_refresh("pre-capture", generation, &inventory, &receipts, None);
        assert!(!rejected.status.success(), "{generation} was re-admitted");
    }
    fs::write(
        receipts.join("invalidations.tsv"),
        format!(
            "{}review1i\tinvalid\tself invalidation\n",
            valid_invalidations()
        ),
    )
    .expect("self-invalidation fixture is written");
    let rejected = run_refresh("pre-capture", "review1i", &inventory, &receipts, None);
    assert!(
        !rejected.status.success(),
        "self-invalidated generation passed"
    );
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("invalidates itself"));

    for malformed in [
        valid_invalidations().replace("review1h\tinvalid\towned stdin was discarded\n", ""),
        valid_invalidations().replace("review1h\tinvalid", "review1h\tadmitted"),
        format!(
            "{}review1h\tinvalid\tduplicate row\n",
            valid_invalidations()
        ),
    ] {
        fs::write(receipts.join("invalidations.tsv"), malformed)
            .expect("malformed invalidation ledger is written");
        let rejected = run_refresh("pre-capture", "review1i", &inventory, &receipts, None);
        assert!(
            !rejected.status.success(),
            "malformed invalidation ledger passed"
        );
    }
}

#[test]
fn refresh_rejects_missing_receipts_and_unsafe_projection_paths() {
    let directory = tempfile::tempdir().expect("temporary refusal directory");
    let inventory = directory.path().join("plugin-parity.yaml");
    let receipts = directory.path().join("receipts");
    fs::copy(
        repository_root().join("rust/benchmarks/plugin-parity.yaml"),
        &inventory,
    )
    .expect("inventory fixture is copied");
    write_pre_capture_receipts(&receipts, &valid_invalidations(), "../escape\n");
    let unsafe_path = run_refresh("pre-capture", "review1i", &inventory, &receipts, None);
    assert!(!unsafe_path.status.success(), "unsafe projection passed");
    assert!(String::from_utf8_lossy(&unsafe_path.stderr).contains("unsafe projection path"));

    fs::write(
        receipts.join("measurement-source-projection.txt"),
        "rust/scripts/capture-plugin-baseline.sh\n",
    )
    .expect("safe projection is restored");
    fs::remove_file(receipts.join("effective-source-tree.tar"))
        .expect("required receipt is removed");
    let missing = run_refresh("pre-capture", "review1i", &inventory, &receipts, None);
    assert!(!missing.status.success(), "missing receipt passed");
    assert!(String::from_utf8_lossy(&missing.stderr).contains("missing required receipt"));
}

#[test]
fn postpublication_requires_independent_verification_facts() {
    let directory = tempfile::tempdir().expect("temporary publication directory");
    let inventory = directory.path().join("plugin-parity.yaml");
    let receipts = directory.path().join("receipts");
    let topology = directory.path().join("package-topology.json");
    fs::copy(
        repository_root().join("rust/benchmarks/plugin-parity.yaml"),
        &inventory,
    )
    .expect("inventory fixture is copied");
    write_pre_capture_receipts(
        &receipts,
        &valid_invalidations(),
        "rust/scripts/capture-plugin-baseline.sh\n",
    );
    let rejected = run_refresh(
        "postpublication",
        "review1i",
        &inventory,
        &receipts,
        Some(&topology),
    );
    assert!(
        !rejected.status.success(),
        "unverified publication was admitted"
    );
    assert!(
        String::from_utf8_lossy(&rejected.stderr)
            .contains("missing independent publication verification")
    );
}

#[test]
fn postpublication_admits_only_independently_retrieved_and_verified_bytes() {
    let directory = tempfile::tempdir().expect("temporary verified publication directory");
    let inventory = directory.path().join("plugin-parity.yaml");
    let capture = directory.path().join("task1-review1i-final");
    let topology = directory.path().join("compact/package-topology.json");
    fs::create_dir_all(topology.parent().expect("topology parent"))
        .expect("compact artifact directory is created");
    fs::copy(
        repository_root().join("rust/benchmarks/plugin-parity.yaml"),
        &inventory,
    )
    .expect("inventory fixture is copied");
    prepare_capture_layout(&capture, "review1i");
    let staged_bundle = fs::read_dir(&capture)
        .expect("capture root is readable")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| path.extension().and_then(|extension| extension.to_str()) == Some("gz"))
        .expect("staged bundle exists");
    let download_root = directory.path().join("independent-download");
    fs::create_dir_all(&download_root).expect("independent download directory is created");
    let downloaded_bundle = download_root.join(staged_bundle.file_name().expect("bundle name"));
    fs::copy(&staged_bundle, &downloaded_bundle).expect("independent bundle download is copied");
    let downloaded_manifest = download_root.join("evidence-manifest.json");
    fs::copy(capture.join("evidence-manifest.json"), &downloaded_manifest)
        .expect("independent manifest download is copied");
    let release_tag = "native-plugin-baseline-caa3ff6f-review1i-final";
    let stable_url = format!(
        "https://github.com/ajcasagrande/rust-native-plugin-lab/releases/download/{release_tag}/{}",
        downloaded_bundle
            .file_name()
            .expect("downloaded bundle name")
            .to_string_lossy()
    );
    let published_locator = download_root.join("bundle-locator.json");
    fs::write(
        &published_locator,
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema_version": 1,
            "repository": "https://github.com/ajcasagrande/rust-native-plugin-lab",
            "recommended_release_tag": release_tag,
            "asset_name": downloaded_bundle.file_name().expect("downloaded bundle name").to_string_lossy(),
            "publication_status": "published_and_verified",
            "archive_verification_status": "downloaded_extracted_manifest_verified",
            "staged_path": downloaded_bundle.to_string_lossy(),
            "bytes": fs::metadata(&downloaded_bundle).expect("download metadata").len(),
            "blake3": digest(&downloaded_bundle),
            "manifest_path": "artifacts/native-plugin-baseline/evidence-manifest.json",
            "manifest_bytes": fs::metadata(&downloaded_manifest).expect("manifest metadata").len(),
            "manifest_blake3": digest(&downloaded_manifest),
            "stable_url": stable_url,
        }))
        .expect("published locator serializes"),
    )
    .expect("published locator is written");
    fs::write(
        capture.join("publication-verification.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema_version": 1,
            "generation": "review1i",
            "status": "independently_downloaded_extracted_manifest_verified",
            "stable_url": stable_url,
            "downloaded_bundle_path": downloaded_bundle,
            "downloaded_bundle_bytes": fs::metadata(&downloaded_bundle).expect("download metadata").len(),
            "downloaded_bundle_blake3": digest(&downloaded_bundle),
            "downloaded_manifest_path": downloaded_manifest,
            "downloaded_manifest_bytes": fs::metadata(&downloaded_manifest).expect("manifest metadata").len(),
            "downloaded_manifest_blake3": digest(&downloaded_manifest),
            "published_locator_path": published_locator,
            "published_locator_bytes": fs::metadata(&published_locator).expect("locator metadata").len(),
            "published_locator_blake3": digest(&published_locator),
        }))
        .expect("publication verification serializes"),
    )
    .expect("publication verification receipt is written");

    let extraction_marker = directory.path().join("authorized-extraction.entered");
    let refreshed = run_refresh_with_environment(
        "postpublication",
        "review1i",
        &inventory,
        &capture,
        Some(&topology),
        &[("AIPERF_EVIDENCE_EXTRACTION_MARKER", &extraction_marker)],
    );
    assert!(
        refreshed.status.success(),
        "verified publication refresh failed: {}",
        String::from_utf8_lossy(&refreshed.stderr)
    );
    let text = fs::read_to_string(&inventory).expect("published inventory is readable");
    assert!(text.contains("admission_status: published_verified_review1i"));
    assert!(text.contains("task1-review1i/build-default"));
    assert!(text.contains("duration_seconds: 31.0"));
    assert!(text.contains("allocations_per_request: 301.0"));
    assert!(!text.contains("task1-review1d/build-default"));
    assert!(
        extraction_marker.is_file(),
        "authorized exact-byte archive did not enter extraction"
    );
    assert!(
        text.contains("publication_status: published_and_verified")
            || fs::read_to_string(directory.path().join("compact/bundle-locator.json"))
                .expect("compact locator exists")
                .contains("\"publication_status\": \"published_and_verified\"")
    );
    assert!(
        directory
            .path()
            .join("compact/evidence-manifest.json")
            .is_file()
    );
    let first_inventory = fs::read(&inventory).expect("published inventory is snapshotted");
    let first_topology = fs::read(&topology).expect("published topology is snapshotted");
    let first_manifest = fs::read(directory.path().join("compact/evidence-manifest.json"))
        .expect("compact manifest is snapshotted");
    let first_locator = fs::read(directory.path().join("compact/bundle-locator.json"))
        .expect("compact locator is snapshotted");
    let second = run_refresh(
        "postpublication",
        "review1i",
        &inventory,
        &capture,
        Some(&topology),
    );
    assert!(
        second.status.success(),
        "second verified publication refresh failed: {}",
        String::from_utf8_lossy(&second.stderr)
    );
    assert_eq!(
        fs::read(&inventory).expect("twice-published inventory exists"),
        first_inventory
    );
    assert_eq!(
        fs::read(&topology).expect("twice-published topology exists"),
        first_topology
    );
    assert_eq!(
        fs::read(directory.path().join("compact/evidence-manifest.json"))
            .expect("twice-published manifest exists"),
        first_manifest
    );
    assert_eq!(
        fs::read(directory.path().join("compact/bundle-locator.json"))
            .expect("twice-published locator exists"),
        first_locator
    );
}

#[test]
fn postpublication_rejects_different_internally_valid_capture_bytes() {
    let directory = tempfile::tempdir().expect("temporary staged binding directory");
    let inventory = directory.path().join("plugin-parity.yaml");
    let capture = directory.path().join("task1-review1i-final");
    let other_capture = directory.path().join("different-task1-review1i-final");
    let topology = directory.path().join("package-topology.json");
    fs::copy(
        repository_root().join("rust/benchmarks/plugin-parity.yaml"),
        &inventory,
    )
    .expect("inventory fixture is copied");
    prepare_capture_layout(&capture, "review1i");
    prepare_capture_layout_with_workspace_tree_command(
        &other_capture,
        "review1i",
        "cargo tree --locked --workspace --edges normal,build --prefix depth",
        b"different but internally valid captured evidence",
    );
    let other_bundle = fs::read_dir(&other_capture)
        .expect("different capture root is readable")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| path.extension().and_then(|extension| extension.to_str()) == Some("gz"))
        .expect("different staged bundle exists");
    let download_root = directory.path().join("independent-different-download");
    fs::create_dir_all(&download_root).expect("different download directory is created");
    let downloaded_bundle = download_root.join(other_bundle.file_name().expect("bundle name"));
    let downloaded_manifest = download_root.join("evidence-manifest.json");
    fs::copy(other_bundle, &downloaded_bundle).expect("different bundle is downloaded");
    fs::copy(
        other_capture.join("evidence-manifest.json"),
        &downloaded_manifest,
    )
    .expect("different manifest is downloaded");
    write_publication_verification(
        &capture,
        &downloaded_bundle,
        &downloaded_manifest,
        &download_root,
    );

    let extraction_marker = directory.path().join("mismatched-extraction.entered");
    let rejected = run_refresh_with_environment(
        "postpublication",
        "review1i",
        &inventory,
        &capture,
        Some(&topology),
        &[("AIPERF_EVIDENCE_EXTRACTION_MARKER", &extraction_marker)],
    );
    assert!(
        !rejected.status.success(),
        "different self-consistent capture was admitted"
    );
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("staged capture"));
    assert!(
        !extraction_marker.exists(),
        "mismatched archive entered extraction before authentication"
    );
}

#[cfg(unix)]
#[test]
fn postpublication_rejects_normalized_and_hardlink_aliases_of_staged_bundle() {
    use std::fs::hard_link;

    for alias_kind in ["normalized", "hardlink"] {
        let directory = tempfile::tempdir().expect("temporary alias refusal directory");
        let inventory = directory.path().join("plugin-parity.yaml");
        let capture = directory.path().join("task1-review1i-final");
        let topology = directory.path().join("package-topology.json");
        fs::copy(
            repository_root().join("rust/benchmarks/plugin-parity.yaml"),
            &inventory,
        )
        .expect("inventory fixture is copied");
        prepare_capture_layout(&capture, "review1i");
        let staged_bundle = fs::read_dir(&capture)
            .expect("capture root is readable")
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .find(|path| path.extension().and_then(|extension| extension.to_str()) == Some("gz"))
            .expect("staged bundle exists");
        let verification_root = directory.path().join("alias-verification");
        fs::create_dir_all(&verification_root).expect("alias verification root is created");
        let downloaded_manifest = verification_root.join("evidence-manifest.json");
        fs::copy(capture.join("evidence-manifest.json"), &downloaded_manifest)
            .expect("manifest download is copied");
        let downloaded_bundle = if alias_kind == "normalized" {
            capture
                .join("evidence")
                .join("..")
                .join(staged_bundle.file_name().expect("staged bundle name"))
        } else {
            let alias =
                verification_root.join(staged_bundle.file_name().expect("staged bundle name"));
            hard_link(&staged_bundle, &alias).expect("staged bundle hardlink is created");
            alias
        };
        write_publication_verification(
            &capture,
            &downloaded_bundle,
            &downloaded_manifest,
            &verification_root,
        );

        let rejected = run_refresh(
            "postpublication",
            "review1i",
            &inventory,
            &capture,
            Some(&topology),
        );
        assert!(
            !rejected.status.success(),
            "{alias_kind} staged-bundle alias was admitted"
        );
        assert!(String::from_utf8_lossy(&rejected.stderr).contains("staged bundle alias"));
    }
}

#[test]
fn postpublication_rejects_unrelated_archive_despite_separately_valid_tree() {
    let directory = tempfile::tempdir().expect("temporary unrelated archive directory");
    let inventory = directory.path().join("plugin-parity.yaml");
    let capture = directory.path().join("task1-review1i-final");
    let topology = directory.path().join("compact/package-topology.json");
    fs::create_dir_all(topology.parent().expect("topology parent"))
        .expect("compact artifact directory is created");
    fs::copy(
        repository_root().join("rust/benchmarks/plugin-parity.yaml"),
        &inventory,
    )
    .expect("inventory fixture is copied");
    prepare_capture_layout(&capture, "review1i");

    let download_root = directory.path().join("independent-download");
    let separately_valid = download_root.join("separately-valid-extracted-evidence");
    copy_tree(&capture.join("evidence"), &separately_valid);
    fs::create_dir_all(&download_root).expect("download directory is created");
    let downloaded_manifest = download_root.join("evidence-manifest.json");
    fs::copy(capture.join("evidence-manifest.json"), &downloaded_manifest)
        .expect("valid separate manifest is copied");

    let unrelated_root = directory.path().join("unrelated-archive");
    fs::create_dir_all(unrelated_root.join("evidence")).expect("unrelated tree is created");
    fs::write(
        unrelated_root.join("evidence/unrelated.txt"),
        b"unrelated archive bytes",
    )
    .expect("unrelated archive file is written");
    fs::copy(
        &downloaded_manifest,
        unrelated_root.join("evidence-manifest.json"),
    )
    .expect("separately valid manifest is embedded beside unrelated bytes");
    let release_tag = "native-plugin-baseline-caa3ff6f-review1i-final";
    let downloaded_bundle = download_root.join(format!("aiperf-{release_tag}.tar.gz"));
    create_bundle(&unrelated_root, &downloaded_bundle);
    let stable_url = format!(
        "https://github.com/ajcasagrande/rust-native-plugin-lab/releases/download/{release_tag}/{}",
        downloaded_bundle
            .file_name()
            .expect("downloaded bundle name")
            .to_string_lossy()
    );
    let published_locator = download_root.join("bundle-locator.json");
    fs::write(
        &published_locator,
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema_version": 1,
            "repository": "https://github.com/ajcasagrande/rust-native-plugin-lab",
            "recommended_release_tag": release_tag,
            "asset_name": downloaded_bundle.file_name().expect("downloaded bundle name").to_string_lossy(),
            "publication_status": "published_and_verified",
            "archive_verification_status": "downloaded_extracted_manifest_verified",
            "staged_path": downloaded_bundle.to_string_lossy(),
            "bytes": fs::metadata(&downloaded_bundle).expect("download metadata").len(),
            "blake3": digest(&downloaded_bundle),
            "manifest_path": "artifacts/native-plugin-baseline/evidence-manifest.json",
            "manifest_bytes": fs::metadata(&downloaded_manifest).expect("manifest metadata").len(),
            "manifest_blake3": digest(&downloaded_manifest),
            "stable_url": stable_url,
        }))
        .expect("published locator serializes"),
    )
    .expect("published locator is written");
    fs::write(
        capture.join("publication-verification.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema_version": 1,
            "generation": "review1i",
            "status": "independently_downloaded_extracted_manifest_verified",
            "stable_url": stable_url,
            "downloaded_bundle_path": downloaded_bundle,
            "downloaded_bundle_bytes": fs::metadata(&downloaded_bundle).expect("download metadata").len(),
            "downloaded_bundle_blake3": digest(&downloaded_bundle),
            "downloaded_manifest_path": downloaded_manifest,
            "downloaded_manifest_bytes": fs::metadata(&downloaded_manifest).expect("manifest metadata").len(),
            "downloaded_manifest_blake3": digest(&downloaded_manifest),
            "published_locator_path": published_locator,
            "published_locator_bytes": fs::metadata(&published_locator).expect("locator metadata").len(),
            "published_locator_blake3": digest(&published_locator),
        }))
        .expect("publication verification serializes"),
    )
    .expect("publication verification receipt is written");

    let rejected = run_refresh(
        "postpublication",
        "review1i",
        &inventory,
        &capture,
        Some(&topology),
    );
    assert!(
        !rejected.status.success(),
        "unrelated downloaded archive was trusted through a separately supplied extraction"
    );
    let stderr = String::from_utf8_lossy(&rejected.stderr);
    assert!(
        stderr.contains("published download does not match the exact staged capture bytes"),
        "unexpected refusal: {stderr}"
    );
}

#[test]
fn postpublication_rejects_malformed_unknown_and_mismatched_verification_receipts() {
    let directory = tempfile::tempdir().expect("temporary malformed publication directory");
    let inventory = directory.path().join("plugin-parity.yaml");
    let capture = directory.path().join("task1-review1i-final");
    let topology = directory.path().join("package-topology.json");
    fs::copy(
        repository_root().join("rust/benchmarks/plugin-parity.yaml"),
        &inventory,
    )
    .expect("inventory fixture is copied");
    prepare_capture_layout(&capture, "review1i");
    for malformed in [
        "",
        "{}",
        r#"{"schema_version":1,"generation":"review1i","status":"independently_downloaded_extracted_manifest_verified","unknown":true}"#,
        r#"{"schema_version":1,"generation":"review1i","status":"independently_downloaded_extracted_manifest_verified","downloaded_bundle_bytes":999,"downloaded_bundle_blake3":"blake3:0000000000000000000000000000000000000000000000000000000000000000"}"#,
    ] {
        fs::write(capture.join("publication-verification.json"), malformed)
            .expect("malformed publication receipt is written");
        let rejected = run_refresh(
            "postpublication",
            "review1i",
            &inventory,
            &capture,
            Some(&topology),
        );
        assert!(
            !rejected.status.success(),
            "malformed publication receipt passed: {malformed}"
        );
    }
}
