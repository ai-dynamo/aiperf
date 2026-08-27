// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! BLAKE3 file digests and deterministic evidence-tree manifests.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::process::Command;

use serde::de::{Error as _, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u8,
    files: Vec<ManifestFile>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ManifestFile {
    path: String,
    bytes: u64,
    blake3: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BundleVerification {
    schema_version: u8,
    generation: String,
    status: String,
    bundle_bytes: u64,
    bundle_blake3: String,
    manifest_bytes: u64,
    manifest_blake3: String,
}

fn write_bundle_verification(
    generation: &str,
    bundle_bytes: u64,
    bundle_digest: blake3::Hash,
    manifest_bytes: &[u8],
    output: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    validate_generation(generation)?;
    let manifest_digest = blake3::hash(manifest_bytes);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    serde_json::to_writer_pretty(
        &mut file,
        &BundleVerification {
            schema_version: 1,
            generation: generation.to_owned(),
            status: "extracted_manifest_verified".to_owned(),
            bundle_bytes,
            bundle_blake3: format!("blake3:{bundle_digest}"),
            manifest_bytes: manifest_bytes.len() as u64,
            manifest_blake3: format!("blake3:{manifest_digest}"),
        },
    )?;
    writeln!(file)?;
    Ok(())
}

fn extract_and_verify_staged_archive(
    bundle: &mut File,
    staged_manifest: &[u8],
    authenticated_bytes: u64,
    authenticated_digest: blake3::Hash,
    extraction_parent: &Path,
    limits: ArchiveLimits,
) -> Result<(Manifest, u64, blake3::Hash), Box<dyn std::error::Error>> {
    let initial_metadata = bundle.metadata()?;
    if initial_metadata.len() != authenticated_bytes {
        return Err("staged archive length changed after authentication".into());
    }
    if let Ok(marker) = std::env::var("AIPERF_EVIDENCE_EXTRACTION_MARKER") {
        fs::write(marker, b"authenticated archive extraction entered\n")?;
    }
    let mut members = BTreeSet::new();
    let mut has_evidence_directory = false;
    let mut has_manifest = false;
    let mut expanded_bytes = 0_u64;
    bundle.seek(SeekFrom::Start(0))?;
    {
        let decoder = flate2::read::GzDecoder::new(&mut *bundle);
        let mut archive = tar::Archive::new(decoder);
        for entry in archive.entries()? {
            let entry = entry?;
            if members.len() >= limits.max_members {
                return Err("staged archive exceeds member-count limit".into());
            }
            let entry_type = entry.header().entry_type();
            if !entry_type.is_dir() && !entry_type.is_file() {
                return Err(
                    "staged archive entries must be regular file or directory types".into(),
                );
            }
            let raw_path = entry.path_bytes();
            let raw_path = std::str::from_utf8(raw_path.as_ref())?;
            if raw_path.contains(['\n', '\r', '\\']) || raw_path.contains("//") {
                return Err(format!("staged archive contains unsafe member `{raw_path:?}`").into());
            }
            let member = raw_path.strip_suffix('/').unwrap_or(raw_path);
            let path = Path::new(member);
            if member.is_empty()
                || path.is_absolute()
                || path
                    .components()
                    .any(|component| !matches!(component, Component::Normal(_)))
                || (member != "evidence-manifest.json"
                    && member != "evidence"
                    && !member.starts_with("evidence/"))
            {
                return Err(format!("staged archive contains unsafe member `{member}`").into());
            }
            if !members.insert(member.to_owned()) {
                return Err(format!("staged archive contains duplicate member `{member}`").into());
            }
            let member_bytes = entry.header().size()?;
            if member_bytes > limits.max_member_bytes {
                return Err("staged archive member exceeds expanded-byte limit".into());
            }
            expanded_bytes = expanded_bytes
                .checked_add(member_bytes)
                .ok_or("staged archive expanded-byte count overflow")?;
            if expanded_bytes > limits.max_expanded_bytes {
                return Err("staged archive exceeds aggregate expanded-byte limit".into());
            }
            if member == "evidence" {
                if !entry_type.is_dir() {
                    return Err("staged archive `evidence` member must be a directory".into());
                }
                has_evidence_directory = true;
            } else if member == "evidence-manifest.json" {
                if !entry_type.is_file() {
                    return Err("staged archive manifest must be a regular file".into());
                }
                has_manifest = true;
            }
        }
    }
    if !has_evidence_directory || !has_manifest {
        return Err("staged archive lacks its evidence tree or embedded manifest".into());
    }

    let extracted = tempfile::Builder::new()
        .prefix("aiperf-evidence-extract-")
        .tempdir_in(extraction_parent)?;
    bundle.seek(SeekFrom::Start(0))?;
    {
        let decoder = flate2::read::GzDecoder::new(&mut *bundle);
        let mut archive = tar::Archive::new(decoder);
        for entry in archive.entries()? {
            if !entry?.unpack_in(extracted.path())? {
                return Err("staged archive member escaped the owned extraction directory".into());
            }
        }
    }
    let embedded_manifest = extracted.path().join("evidence-manifest.json");
    if !fs::symlink_metadata(&embedded_manifest)?
        .file_type()
        .is_file()
        || !fs::symlink_metadata(extracted.path().join("evidence"))?
            .file_type()
            .is_dir()
    {
        return Err(
            "staged archive extracted topology is not regular manifest plus evidence directory"
                .into(),
        );
    }
    if fs::read(&embedded_manifest)? != staged_manifest {
        return Err("staged archive embedded manifest does not match its manifest".into());
    }
    let evidence = extracted.path().join("evidence");
    verify_manifest(&embedded_manifest, &evidence)?;
    let manifest = serde_json::from_reader(File::open(embedded_manifest)?)?;
    let final_metadata = bundle.metadata()?;
    let (final_bytes, final_digest) = digest_open_file(bundle)?;
    if final_bytes != authenticated_bytes
        || final_digest != authenticated_digest
        || final_metadata.len() != initial_metadata.len()
        || final_metadata.modified().ok() != initial_metadata.modified().ok()
    {
        return Err("staged archive changed during descriptor-held verification".into());
    }
    Ok((manifest, authenticated_bytes, authenticated_digest))
}

fn verify_staged_bundle(
    generation: &str,
    bundle_path: &Path,
    manifest_path: &Path,
    extraction_parent: &Path,
    receipt_path: &Path,
    limits: ArchiveLimits,
) -> Result<(), Box<dyn std::error::Error>> {
    if !extraction_parent.is_absolute()
        || !fs::symlink_metadata(extraction_parent)?
            .file_type()
            .is_dir()
    {
        return Err("staged-bundle extraction parent must be an absolute directory".into());
    }
    let manifest = snapshot_regular_file(manifest_path)?;
    #[cfg(debug_assertions)]
    rewrite_authenticated_file_for_test(
        "AIPERF_TEST_REWRITE_STAGED_MANIFEST_AFTER_SNAPSHOT",
        manifest_path,
    )?;
    let mut bundle = open_regular_nofollow(bundle_path)?;
    let (authenticated_bytes, authenticated_digest) = digest_open_file(&mut bundle)?;
    #[cfg(debug_assertions)]
    replace_authenticated_path_for_test(
        "AIPERF_TEST_REPLACE_STAGED_BUNDLE_AFTER_AUTHENTICATION",
        bundle_path,
    )?;
    let (mut owned_bundle, snapshot_bytes, snapshot_digest) =
        immutable_owned_snapshot(&mut bundle, extraction_parent)?;
    if snapshot_bytes != authenticated_bytes || snapshot_digest != authenticated_digest {
        return Err("staged bundle changed while creating its immutable owned snapshot".into());
    }
    let reopened = open_regular_nofollow(bundle_path)?;
    if !same_file_identity(&bundle, &reopened)? {
        return Err("staged bundle path changed before immutable snapshot admission".into());
    }
    #[cfg(debug_assertions)]
    rewrite_authenticated_file_for_test(
        "AIPERF_TEST_REWRITE_STAGED_BUNDLE_AFTER_SNAPSHOT",
        bundle_path,
    )?;
    let (final_bytes, final_digest) = digest_open_file(&mut bundle)?;
    if final_bytes != authenticated_bytes || final_digest != authenticated_digest {
        return Err("staged bundle changed before immutable snapshot admission".into());
    }
    extract_and_verify_staged_archive(
        &mut owned_bundle,
        &manifest,
        authenticated_bytes,
        authenticated_digest,
        extraction_parent,
        limits,
    )?;
    write_bundle_verification(
        generation,
        authenticated_bytes,
        authenticated_digest,
        &manifest,
        receipt_path,
    )
}

fn open_regular_nofollow(path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;

        options.custom_flags(libc::O_NOFOLLOW);
    }
    let file = options.open(path).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!(
                "no-follow regular file acquisition failed for {}: {error}",
                path.display()
            ),
        )
    })?;
    if !file.metadata()?.file_type().is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "no-follow regular file acquisition rejected {}",
                path.display()
            ),
        ));
    }
    Ok(file)
}

fn snapshot_regular_file(path: &Path) -> io::Result<Vec<u8>> {
    let mut file = open_regular_nofollow(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn immutable_owned_snapshot(
    source: &mut File,
    parent: &Path,
) -> io::Result<(File, u64, blake3::Hash)> {
    source.seek(SeekFrom::Start(0))?;
    let mut snapshot = tempfile::tempfile_in(parent)?;
    io::copy(source, &mut snapshot)?;
    snapshot.sync_all()?;
    let (bytes, digest) = digest_open_file(&mut snapshot)?;
    Ok((snapshot, bytes, digest))
}

fn same_file_identity(left: &File, right: &File) -> io::Result<bool> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;

        let left = left.metadata()?;
        let right = right.metadata()?;
        Ok(left.dev() == right.dev() && left.ino() == right.ino())
    }
    #[cfg(not(unix))]
    {
        let _ = (left, right);
        Ok(false)
    }
}

#[cfg(debug_assertions)]
fn replace_authenticated_path_for_test(environment: &str, path: &Path) -> io::Result<()> {
    let Some(replacement) = std::env::var_os(environment) else {
        return Ok(());
    };
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "test path is not UTF-8"))?;
    let displaced = path.with_file_name(format!(".{name}.authenticated-test-original"));
    fs::rename(path, &displaced)?;
    if let Err(error) = fs::rename(PathBuf::from(replacement), path) {
        let _ = fs::rename(displaced, path);
        return Err(error);
    }
    Ok(())
}

#[cfg(debug_assertions)]
fn rewrite_authenticated_file_for_test(environment: &str, path: &Path) -> io::Result<()> {
    let Some(replacement) = std::env::var_os(environment) else {
        return Ok(());
    };
    let mut replacement = open_regular_nofollow(Path::new(&replacement))?;
    let mut options = OpenOptions::new();
    options.write(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;

        options.custom_flags(libc::O_NOFOLLOW);
    }
    let mut destination = options.open(path)?;
    io::copy(&mut replacement, &mut destination)?;
    destination.sync_all()
}

#[derive(Clone, Copy)]
struct ArchiveLimits {
    max_members: usize,
    max_member_bytes: u64,
    max_expanded_bytes: u64,
}

const CAPTURED_SOURCE_ARCHIVE_LIMITS: ArchiveLimits = ArchiveLimits {
    max_members: 100_000,
    max_member_bytes: 8 * 1024 * 1024 * 1024,
    max_expanded_bytes: 16 * 1024 * 1024 * 1024,
};

const STAGED_ARCHIVE_LIMITS: ArchiveLimits = ArchiveLimits {
    max_members: 10_000,
    max_member_bytes: 8 * 1024 * 1024 * 1024,
    max_expanded_bytes: 16 * 1024 * 1024 * 1024,
};

#[cfg(debug_assertions)]
fn archive_limits_from_test_environment(
    prefix: &str,
    defaults: ArchiveLimits,
) -> Result<ArchiveLimits, Box<dyn std::error::Error>> {
    let parse = |suffix: &str, default| -> Result<u64, Box<dyn std::error::Error>> {
        let name = format!("AIPERF_TEST_{prefix}_{suffix}");
        std::env::var_os(&name)
            .map(|value| {
                value
                    .to_str()
                    .ok_or_else(|| format!("{name} is not UTF-8"))?
                    .parse::<u64>()
                    .map_err(Into::into)
            })
            .unwrap_or(Ok(default))
    };
    Ok(ArchiveLimits {
        max_members: parse("MAX_MEMBERS", defaults.max_members as u64)?
            .try_into()
            .map_err(|_| "test archive member limit does not fit usize")?,
        max_member_bytes: parse("MAX_MEMBER_BYTES", defaults.max_member_bytes)?,
        max_expanded_bytes: parse("MAX_EXPANDED_BYTES", defaults.max_expanded_bytes)?,
    })
}

fn normalized_source_archive_member(
    raw_path: &str,
    is_directory: bool,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    if raw_path.contains(['\n', '\r', '\\', '\0']) {
        return Err(
            format!("captured source archive contains unsafe member `{raw_path:?}`").into(),
        );
    }
    if matches!(raw_path, "." | "./") {
        if !is_directory {
            return Err("captured source archive root member must be a directory".into());
        }
        return Ok(None);
    }
    let member = raw_path
        .strip_prefix("./")
        .unwrap_or(raw_path)
        .strip_suffix('/')
        .unwrap_or_else(|| raw_path.strip_prefix("./").unwrap_or(raw_path));
    let path = Path::new(member);
    if member.is_empty()
        || member.starts_with('/')
        || member.ends_with('/')
        || member.contains("//")
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!("captured source archive contains unsafe member `{raw_path}`").into());
    }
    Ok(Some(member.to_owned()))
}

fn reviewed_capture_identity(
    inventory_path: &Path,
) -> Result<(Vec<u8>, String), Box<dyn std::error::Error>> {
    let inventory = String::from_utf8(snapshot_regular_file(inventory_path)?)?;
    let mut identity = identity_from_inventory(&inventory)?;
    let effective_source_tree_blake3 = identity
        .get("effective_source_tree_blake3")
        .and_then(serde_json::Value::as_str)
        .filter(|digest| is_lower_blake3(digest))
        .ok_or("reviewed pre-capture identity lacks an effective source-tree digest")?
        .to_owned();
    if identity.remove("canonical_inventory_digest").is_none() {
        return Err("reviewed pre-capture identity lacks its canonical inventory binding".into());
    }
    let mut captured_identity = serde_json::to_vec_pretty(&identity)?;
    captured_identity.push(b'\n');
    Ok((captured_identity, effective_source_tree_blake3))
}

fn unique_manifest_file<'a>(
    manifest: &'a Manifest,
    path: &str,
) -> Result<&'a ManifestFile, Box<dyn std::error::Error>> {
    let matching = manifest
        .files
        .iter()
        .filter(|entry| entry.path == path)
        .collect::<Vec<_>>();
    if matching.len() != 1 {
        return Err(format!("captured evidence manifest must bind one `{path}`").into());
    }
    Ok(matching[0])
}

fn extract_authenticated_captured_source(
    reviewed_inventory: &Path,
    capture_root: &Path,
    destination: &Path,
    limits: ArchiveLimits,
) -> Result<(), Box<dyn std::error::Error>> {
    let (reviewed_identity, reviewed_source_digest) =
        reviewed_capture_identity(reviewed_inventory)?;
    let manifest_path = capture_root.join("evidence-manifest.json");
    let manifest_bytes = snapshot_regular_file(&manifest_path)?;
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
    if manifest.schema_version != 1 {
        return Err("captured evidence manifest schema must be 1".into());
    }
    let expected = unique_manifest_file(&manifest, "identity/effective-source-tree.tar")?;
    if !is_lower_blake3(&expected.blake3) {
        return Err("captured source archive manifest digest is not lower-case BLAKE3".into());
    }
    if expected.blake3 != reviewed_source_digest {
        return Err("captured source does not match the reviewed pre-capture identity".into());
    }
    let captured_identity_path = capture_root.join("evidence/identity/experiment-identity.json");
    let captured_identity = snapshot_regular_file(&captured_identity_path)?;
    let identity_entry = unique_manifest_file(&manifest, "identity/experiment-identity.json")?;
    if identity_entry.bytes != captured_identity.len() as u64
        || identity_entry.blake3 != format!("blake3:{}", blake3::hash(&captured_identity))
        || captured_identity != reviewed_identity
    {
        return Err(
            "captured manifest identity does not match the reviewed pre-capture identity".into(),
        );
    }
    let archive_path = capture_root.join("evidence/identity/effective-source-tree.tar");
    let mut archive_file = open_regular_nofollow(&archive_path)?;
    let initial_metadata = archive_file.metadata()?;
    let (authenticated_bytes, authenticated_digest) = digest_open_file(&mut archive_file)?;
    if authenticated_bytes != expected.bytes
        || format!("blake3:{authenticated_digest}") != reviewed_source_digest
    {
        return Err("captured source archive does not match its authenticated manifest".into());
    }
    #[cfg(debug_assertions)]
    replace_authenticated_path_for_test(
        "AIPERF_TEST_REPLACE_CAPTURED_SOURCE_AFTER_AUTHENTICATION",
        &archive_path,
    )?;

    let destination_metadata = fs::symlink_metadata(destination)?;
    if !destination_metadata.file_type().is_dir() || fs::read_dir(destination)?.next().is_some() {
        return Err("captured source destination must be an empty controlled directory".into());
    }
    let snapshot_parent = destination
        .parent()
        .ok_or("captured source destination has no controlled parent")?;
    let (mut owned_archive, snapshot_bytes, snapshot_digest) =
        immutable_owned_snapshot(&mut archive_file, snapshot_parent)?;
    if snapshot_bytes != authenticated_bytes || snapshot_digest != authenticated_digest {
        return Err("captured source changed while creating its immutable owned snapshot".into());
    }
    let reopened = open_regular_nofollow(&archive_path)?;
    if !same_file_identity(&archive_file, &reopened)? {
        return Err("captured source path changed before immutable snapshot admission".into());
    }
    #[cfg(debug_assertions)]
    rewrite_authenticated_file_for_test(
        "AIPERF_TEST_REWRITE_CAPTURED_SOURCE_AFTER_SNAPSHOT",
        &archive_path,
    )?;
    let final_metadata = archive_file.metadata()?;
    let (final_bytes, final_digest) = digest_open_file(&mut archive_file)?;
    if final_bytes != authenticated_bytes
        || final_digest != authenticated_digest
        || final_metadata.len() != initial_metadata.len()
        || final_metadata.modified().ok() != initial_metadata.modified().ok()
    {
        return Err("captured source changed before immutable snapshot admission".into());
    }
    if let Ok(marker) = std::env::var("AIPERF_CAPTURED_SOURCE_EXTRACTION_MARKER") {
        fs::write(
            marker,
            b"authenticated captured source extraction entered\n",
        )?;
    }

    let mut members = BTreeSet::new();
    let mut expanded_bytes = 0_u64;
    owned_archive.seek(SeekFrom::Start(0))?;
    {
        let mut archive = tar::Archive::new(&mut owned_archive);
        for entry in archive.entries()? {
            let entry = entry?;
            if members.len() >= limits.max_members {
                return Err("captured source archive exceeds member-count limit".into());
            }
            let entry_type = entry.header().entry_type();
            if !entry_type.is_dir() && !entry_type.is_file() {
                return Err(
                    "captured source archive entries must be regular files or directories".into(),
                );
            }
            let raw_path = entry.path_bytes();
            let raw_path = std::str::from_utf8(raw_path.as_ref())?;
            if let Some(member) = normalized_source_archive_member(raw_path, entry_type.is_dir())?
                && !members.insert(member)
            {
                return Err("captured source archive contains duplicate normalized member".into());
            }
            let member_bytes = entry.header().size()?;
            if member_bytes > limits.max_member_bytes {
                return Err("captured source archive member exceeds expanded-byte limit".into());
            }
            expanded_bytes = expanded_bytes
                .checked_add(member_bytes)
                .ok_or("captured source archive expanded-byte count overflow")?;
            if expanded_bytes > limits.max_expanded_bytes {
                return Err("captured source archive exceeds aggregate expanded-byte limit".into());
            }
        }
    }
    if members.is_empty() {
        return Err("captured source archive contains no source members".into());
    }

    owned_archive.seek(SeekFrom::Start(0))?;
    {
        let mut archive = tar::Archive::new(&mut owned_archive);
        for entry in archive.entries()? {
            if !entry?.unpack_in(destination)? {
                return Err("captured source archive member escaped controlled storage".into());
            }
        }
    }
    let mut extracted_paths = Vec::new();
    visit(destination, &mut extracted_paths)?;
    Ok(())
}

#[derive(Serialize)]
struct PackageTopology {
    schema_version: u8,
    generation: String,
    host_commit: String,
    rustc: String,
    target: String,
    cargo_profile: String,
    measurement: TopologyMeasurement,
    workspace_packages: Vec<TopologyPackage>,
    cargo_projection: Vec<CargoPackageProjection>,
}

#[derive(Serialize)]
struct TopologyMeasurement {
    commands: [&'static str; 3],
    cargo_lock_blake3: String,
    cargo_metadata_blake3: String,
    cargo_tree_blake3: String,
    cargo_cli_tree_blake3: String,
    raw_metadata: &'static str,
    raw_tree: &'static str,
    raw_cli_tree: &'static str,
}

#[derive(Serialize)]
struct TopologyPackage {
    name: String,
    version: String,
    direct_dependency_count: u64,
    dependency_kind_counts: BTreeMap<String, u64>,
    direct_dependencies: Vec<TopologyDependency>,
    direct_workspace_dependencies: Vec<String>,
    features: Vec<String>,
}

#[derive(Serialize)]
struct TopologyDependency {
    name: String,
    kind: String,
    is_workspace: bool,
}

#[derive(Ord, PartialOrd, Eq, PartialEq, Serialize)]
struct CargoDependencyProjection {
    package: String,
    local_name: String,
    kind: String,
    source: Option<String>,
    requirement: String,
    registry: Option<String>,
    path: Option<String>,
    target: Option<String>,
    is_optional: bool,
    uses_default_features: bool,
    features: Vec<String>,
    is_workspace: bool,
}

#[derive(Serialize)]
struct CargoPackageProjection {
    name: String,
    version: String,
    edition: String,
    dependencies: Vec<CargoDependencyProjection>,
    features: BTreeMap<String, Vec<String>>,
}

fn digest_file(path: &Path) -> io::Result<(u64, blake3::Hash)> {
    let mut file = File::open(path)?;
    digest_open_file(&mut file)
}

fn digest_open_file(file: &mut File) -> io::Result<(u64, blake3::Hash)> {
    file.seek(SeekFrom::Start(0))?;
    let mut hasher = blake3::Hasher::new();
    let bytes = io::copy(file, &mut hasher)?;
    file.seek(SeekFrom::Start(0))?;
    Ok((bytes, hasher.finalize()))
}

fn visit(directory: &Path, paths: &mut Vec<PathBuf>) -> io::Result<()> {
    let mut entries = fs::read_dir(directory)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("evidence tree contains symlink: {}", path.display()),
            ));
        }
        if file_type.is_dir() {
            visit(&path, paths)?;
        } else if file_type.is_file() {
            paths.push(path);
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "evidence tree contains unsupported entry: {}",
                    path.display()
                ),
            ));
        }
    }
    Ok(())
}

fn relative_utf8(root: &Path, path: &Path) -> io::Result<String> {
    path.strip_prefix(root)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "path escaped evidence root"))?
        .to_str()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "non-UTF-8 path"))
        .map(|path| path.replace(std::path::MAIN_SEPARATOR, "/"))
}

fn manifest(root: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let root = root.canonicalize()?;
    let mut paths = Vec::new();
    visit(&root, &mut paths)?;
    let files = paths
        .into_iter()
        .map(|path| {
            let relative = relative_utf8(&root, &path)?;
            let (bytes, digest) = digest_file(&path)?;
            Ok(ManifestFile {
                path: relative,
                bytes,
                blake3: format!("blake3:{digest}"),
            })
        })
        .collect::<io::Result<Vec<_>>>()?;
    serde_json::to_writer_pretty(
        io::stdout().lock(),
        &Manifest {
            schema_version: 1,
            files,
        },
    )?;
    println!();
    Ok(())
}

fn verify_manifest(manifest_path: &Path, root: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let manifest: Manifest = serde_json::from_reader(File::open(manifest_path)?)?;
    if manifest.schema_version != 1 {
        return Err(format!(
            "unsupported evidence manifest schema {}",
            manifest.schema_version
        )
        .into());
    }
    let root = root.canonicalize()?;
    let mut actual_paths = Vec::new();
    visit(&root, &mut actual_paths)?;
    if actual_paths.len() != manifest.files.len() {
        return Err(format!(
            "manifest lists {} files but evidence tree contains {}",
            manifest.files.len(),
            actual_paths.len()
        )
        .into());
    }

    for (entry, actual_path) in manifest.files.iter().zip(actual_paths) {
        let declared = Path::new(&entry.path);
        if entry.path.is_empty()
            || declared.is_absolute()
            || declared
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
        {
            return Err(format!("unsafe manifest path: {}", entry.path).into());
        }
        let actual_relative = relative_utf8(&root, &actual_path)?;
        if entry.path != actual_relative {
            return Err(format!(
                "manifest path mismatch: declared {}, actual {actual_relative}",
                entry.path
            )
            .into());
        }
        let (bytes, digest) = digest_file(&actual_path)?;
        let digest = format!("blake3:{digest}");
        if entry.bytes != bytes || entry.blake3 != digest {
            return Err(format!(
                "evidence mismatch for {}: expected {} bytes {}, observed {bytes} {digest}",
                entry.path, entry.bytes, entry.blake3
            )
            .into());
        }
    }
    Ok(())
}

fn write_topology(
    generation: &str,
    host_commit: &str,
    rustc: &str,
    target: &str,
    cargo_profile: &str,
    cargo_lock: &Path,
    metadata_path: &Path,
    workspace_tree: &Path,
    cli_tree: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let metadata: serde_json::Value = serde_json::from_reader(File::open(metadata_path)?)?;
    let members: BTreeSet<_> = metadata["workspace_members"]
        .as_array()
        .ok_or("cargo metadata lacks workspace_members")?
        .iter()
        .map(|member| {
            member
                .as_str()
                .map(str::to_owned)
                .ok_or("workspace member id is not text")
        })
        .collect::<Result<_, _>>()?;
    let all_packages = metadata["packages"]
        .as_array()
        .ok_or("cargo metadata lacks packages")?;
    let workspace_root = metadata["workspace_root"]
        .as_str()
        .ok_or("cargo metadata lacks workspace_root")?;
    let workspace_names: BTreeSet<_> = all_packages
        .iter()
        .filter(|package| {
            package["id"]
                .as_str()
                .is_some_and(|id| members.contains(id))
        })
        .filter_map(|package| package["name"].as_str().map(str::to_owned))
        .collect();
    let mut workspace_packages = Vec::new();
    let mut cargo_projection = Vec::new();
    for package in all_packages.iter().filter(|package| {
        package["id"]
            .as_str()
            .is_some_and(|id| members.contains(id))
    }) {
        let dependencies = package["dependencies"]
            .as_array()
            .ok_or("package dependencies are not an array")?;
        let mut direct_dependencies = Vec::new();
        let mut exact_dependencies = Vec::new();
        let mut dependency_kind_counts = BTreeMap::new();
        for dependency in dependencies {
            let name = dependency["name"]
                .as_str()
                .ok_or("dependency name is not text")?;
            let kind = dependency["kind"].as_str().unwrap_or("normal");
            *dependency_kind_counts.entry(kind.to_owned()).or_insert(0) += 1;
            direct_dependencies.push(TopologyDependency {
                name: name.to_owned(),
                kind: kind.to_owned(),
                is_workspace: workspace_names.contains(name),
            });
            let local_name = dependency["rename"].as_str().unwrap_or(name);
            let kind = dependency["kind"].as_str().unwrap_or("normal");
            let mut requested_features = dependency["features"]
                .as_array()
                .ok_or("dependency features are not an array")?
                .iter()
                .map(|feature| {
                    feature
                        .as_str()
                        .map(str::to_owned)
                        .ok_or("dependency feature is not text")
                })
                .collect::<Result<Vec<_>, _>>()?;
            requested_features.sort();
            let path = dependency["path"]
                .as_str()
                .map(|path| {
                    Path::new(path)
                        .strip_prefix(workspace_root)
                        .map_err(|_| "dependency path is outside the Cargo workspace root")?
                        .to_str()
                        .ok_or("dependency path is not UTF-8")
                        .map(|path| path.replace(std::path::MAIN_SEPARATOR, "/"))
                })
                .transpose()?;
            exact_dependencies.push(CargoDependencyProjection {
                package: name.to_owned(),
                local_name: local_name.to_owned(),
                kind: kind.to_owned(),
                source: dependency["source"].as_str().map(str::to_owned),
                requirement: dependency["req"]
                    .as_str()
                    .ok_or("dependency requirement is not text")?
                    .to_owned(),
                registry: dependency["registry"].as_str().map(str::to_owned),
                path,
                target: dependency["target"].as_str().map(str::to_owned),
                is_optional: dependency["optional"]
                    .as_bool()
                    .ok_or("dependency optional marker is not boolean")?,
                uses_default_features: dependency["uses_default_features"]
                    .as_bool()
                    .ok_or("dependency default-feature marker is not boolean")?,
                features: requested_features,
                is_workspace: workspace_names.contains(name),
            });
        }
        direct_dependencies.sort_by(|left, right| {
            (&left.name, &left.kind, left.is_workspace).cmp(&(
                &right.name,
                &right.kind,
                right.is_workspace,
            ))
        });
        let direct_workspace_dependencies = direct_dependencies
            .iter()
            .filter(|dependency| dependency.is_workspace)
            .map(|dependency| dependency.name.clone())
            .collect();
        let mut features = package["features"]
            .as_object()
            .ok_or("package features are not an object")?
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        features.sort();
        workspace_packages.push(TopologyPackage {
            name: package["name"]
                .as_str()
                .ok_or("package name is not text")?
                .to_owned(),
            version: package["version"]
                .as_str()
                .ok_or("package version is not text")?
                .to_owned(),
            direct_dependency_count: dependencies.len() as u64,
            dependency_kind_counts,
            direct_dependencies,
            direct_workspace_dependencies,
            features,
        });
        exact_dependencies.sort();
        let exact_features = package["features"]
            .as_object()
            .ok_or("package features are not an object")?
            .iter()
            .map(|(name, values)| {
                let mut values = values
                    .as_array()
                    .ok_or("package feature forwarding vector is not an array")?
                    .iter()
                    .map(|value| {
                        value
                            .as_str()
                            .map(str::to_owned)
                            .ok_or("package feature forwarding value is not text")
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                values.sort();
                Ok((name.clone(), values))
            })
            .collect::<Result<BTreeMap<_, _>, Box<dyn std::error::Error>>>()?;
        cargo_projection.push(CargoPackageProjection {
            name: package["name"]
                .as_str()
                .ok_or("package name is not text")?
                .to_owned(),
            version: package["version"]
                .as_str()
                .ok_or("package version is not text")?
                .to_owned(),
            edition: package["edition"]
                .as_str()
                .ok_or("package edition is not text")?
                .to_owned(),
            dependencies: exact_dependencies,
            features: exact_features,
        });
    }
    workspace_packages.sort_by(|left, right| left.name.cmp(&right.name));
    cargo_projection.sort_by(|left, right| left.name.cmp(&right.name));
    let digest = |path: &Path| -> Result<String, io::Error> {
        digest_file(path).map(|(_, digest)| format!("blake3:{digest}"))
    };
    let topology = PackageTopology {
        schema_version: 1,
        generation: generation.to_owned(),
        host_commit: host_commit.to_owned(),
        rustc: rustc.to_owned(),
        target: target.to_owned(),
        cargo_profile: cargo_profile.to_owned(),
        measurement: TopologyMeasurement {
            commands: [
                "cargo metadata --locked --format-version 1",
                "cargo tree --locked --workspace --edges normal,build --prefix depth",
                "cargo tree --locked -p aiperf-cli --edges normal,build --prefix depth",
            ],
            cargo_lock_blake3: digest(cargo_lock)?,
            cargo_metadata_blake3: digest(metadata_path)?,
            cargo_tree_blake3: digest(workspace_tree)?,
            cargo_cli_tree_blake3: digest(cli_tree)?,
            raw_metadata: "identity/cargo-metadata.json",
            raw_tree: "identity/cargo-tree-workspace.txt",
            raw_cli_tree: "identity/cargo-tree-cli.txt",
        },
        workspace_packages,
        cargo_projection,
    };
    serde_json::to_writer_pretty(io::stdout().lock(), &topology)?;
    println!();
    Ok(())
}

fn normalize_cargo_receipt(
    kind: &str,
    source_root: &Path,
    input: &Path,
    output: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let source_root = source_root
        .to_str()
        .filter(|root| source_root.is_absolute() && *root != "/")
        .ok_or("Cargo receipt source root must be an absolute non-root UTF-8 path")?;
    let normalized = match kind {
        "metadata" => {
            fn normalize_strings(value: &mut serde_json::Value, source_root: &str) {
                match value {
                    serde_json::Value::String(text) => {
                        *text = text.replace(source_root, "$MEASUREMENT_SOURCE");
                    }
                    serde_json::Value::Array(values) => {
                        for value in values {
                            normalize_strings(value, source_root);
                        }
                    }
                    serde_json::Value::Object(mapping) => {
                        for value in mapping.values_mut() {
                            normalize_strings(value, source_root);
                        }
                    }
                    _ => {}
                }
            }

            let mut metadata: serde_json::Value = serde_json::from_reader(File::open(input)?)?;
            normalize_strings(&mut metadata, source_root);
            let mut bytes = serde_json::to_vec(&metadata)?;
            bytes.push(b'\n');
            bytes
        }
        "tree" => fs::read_to_string(input)?
            .replace(source_root, "$MEASUREMENT_SOURCE")
            .into_bytes(),
        _ => return Err("Cargo receipt kind must be `metadata` or `tree`".into()),
    };
    atomic_write(output, &normalized)?;
    Ok(())
}

const ZERO_DIGEST: &str = "blake3:0000000000000000000000000000000000000000000000000000000000000000";

fn replace_inventory_digest_field(
    contents: &str,
    field_name: &str,
    replacement: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    fn json_key_count(value: &serde_json::Value, field_name: &str) -> usize {
        match value {
            serde_json::Value::Object(mapping) => mapping
                .iter()
                .map(|(key, value)| {
                    usize::from(key == field_name) + json_key_count(value, field_name)
                })
                .sum(),
            serde_json::Value::Array(values) => values
                .iter()
                .map(|value| json_key_count(value, field_name))
                .sum(),
            _ => 0,
        }
    }

    fn yaml_key_count(value: &serde_yaml::Value, field_name: &str) -> usize {
        match value {
            serde_yaml::Value::Mapping(mapping) => mapping
                .iter()
                .map(|(key, value)| {
                    let key = key.as_str();
                    let nested_identity = if key == Some("experiment_identity_json") {
                        value
                            .as_str()
                            .and_then(|identity| serde_json::from_str(identity).ok())
                            .map_or(0, |identity| json_key_count(&identity, field_name))
                    } else {
                        0
                    };
                    usize::from(key == Some(field_name))
                        + nested_identity
                        + yaml_key_count(value, field_name)
                })
                .sum(),
            serde_yaml::Value::Sequence(values) => values
                .iter()
                .map(|value| yaml_key_count(value, field_name))
                .sum(),
            _ => 0,
        }
    }

    let inventory: serde_yaml::Value = serde_yaml::from_str(contents)?;
    let expected_count = yaml_key_count(&inventory, field_name);
    if expected_count == 0 {
        return Err(format!("inventory structure lacks `{field_name}`").into());
    }
    let mut output = String::with_capacity(contents.len());
    let mut count = 0;
    for line in contents.split_inclusive('\n') {
        let Some(field) = line.find(field_name) else {
            output.push_str(line);
            continue;
        };
        let value_region = field + field_name.len();
        let Some(relative_digest) = line[value_region..].find("blake3:") else {
            output.push_str(line);
            continue;
        };
        let start = value_region + relative_digest;
        let end = start + ZERO_DIGEST.len();
        let digest = line.get(start..end).ok_or("truncated inventory digest")?;
        if !digest
            .strip_prefix("blake3:")
            .is_some_and(|hex| hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_hexdigit()))
        {
            return Err(format!(
                "invalid `{field_name}` digest `{digest}` in line `{}`",
                line.trim_end()
            )
            .into());
        }
        output.push_str(&line[..start]);
        output.push_str(replacement);
        output.push_str(&line[end..]);
        count += 1;
    }
    if count != expected_count {
        return Err(
            format!("expected {expected_count} `{field_name}` digests, found {count}").into(),
        );
    }
    Ok(output)
}

fn refreshed_inventory(contents: &str) -> Result<String, Box<dyn std::error::Error>> {
    let zeroed =
        replace_inventory_digest_field(contents, "canonical_inventory_digest", ZERO_DIGEST)?;
    let zeroed =
        replace_inventory_digest_field(&zeroed, "experiment_identity_digest", ZERO_DIGEST)?;
    let canonical = format!("blake3:{}", blake3::hash(zeroed.as_bytes()));
    let with_canonical =
        replace_inventory_digest_field(&zeroed, "canonical_inventory_digest", &canonical)?;
    let marker = "experiment_identity_json: |\n";
    let identity_start = with_canonical
        .find(marker)
        .map(|index| index + marker.len())
        .ok_or("inventory lacks experiment_identity_json block")?;
    let identity_end = with_canonical[identity_start..]
        .find("experiment_identity_digest:")
        .map(|offset| identity_start + offset)
        .ok_or("inventory lacks experiment_identity_digest")?;
    let mut identity = String::new();
    for line in with_canonical[identity_start..identity_end].split_inclusive('\n') {
        identity.push_str(
            line.strip_prefix("  ")
                .ok_or("experiment identity block is not indented by two spaces")?,
        );
    }
    let identity_digest = format!("blake3:{}", blake3::hash(identity.as_bytes()));
    replace_inventory_digest_field(
        &with_canonical,
        "experiment_identity_digest",
        &identity_digest,
    )
}

fn refresh_inventory(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let contents = fs::read_to_string(path)?;
    print!("{}", refreshed_inventory(&contents)?);
    Ok(())
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum InventoryRefreshMode {
    PreCapture,
    PostCapture,
}

impl InventoryRefreshMode {
    fn parse(value: &str) -> Result<Self, Box<dyn std::error::Error>> {
        match value {
            "pre-capture" => Ok(Self::PreCapture),
            "post-capture" => Ok(Self::PostCapture),
            _ => Err(format!("unknown inventory refresh mode: {value}").into()),
        }
    }
}

fn validate_generation(generation: &str) -> Result<(), Box<dyn std::error::Error>> {
    if generation.is_empty()
        || !generation
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
    {
        return Err("generation must contain only lowercase ASCII letters and digits".into());
    }
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
        return Err(format!("generation {generation} is permanently invalidated").into());
    }
    Ok(())
}

fn required_receipt(root: &Path, name: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let path = root.join(name);
    if !path.is_file() {
        return Err(format!("missing required receipt: {}", path.display()).into());
    }
    Ok(path)
}

fn validate_projection_list(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let contents = fs::read_to_string(path)?;
    if contents.is_empty() || !contents.ends_with('\n') {
        return Err("projection list must be non-empty and newline terminated".into());
    }
    let mut previous = None;
    for relative in contents.lines() {
        let path = Path::new(relative);
        if relative.is_empty()
            || path.is_absolute()
            || path
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
        {
            return Err(format!("unsafe projection path: {relative}").into());
        }
        if previous.is_some_and(|value| value >= relative) {
            return Err("projection list must be sorted and unique".into());
        }
        previous = Some(relative);
    }
    Ok(())
}

fn source_projection_rule(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let contents = fs::read_to_string(path)?;
    let validator = "rust/e2e-tests/tests/plugin_baseline_inventory.rs";
    if !contents.lines().any(|relative| relative == validator) {
        return Err("projection list omits its semantic inventory validator".into());
    }
    Ok(format!(
        "every build, runtime, metadata, allocation, and digest result uses one materialized tree formed from the complete git archive of host_commit overlaid by the exact allowlist in identity/measurement-source-projection.txt; its overlaid Cargo.lock is authoritative; the base archive/lock are provenance only; {validator} is included as the semantic validator; inventory, report, and generated evidence are excluded to prevent self-reference"
    ))
}

#[derive(Clone)]
struct Invalidation {
    generation: String,
    status: String,
    reason: String,
}

fn read_invalidations(
    path: &Path,
    generation: &str,
) -> Result<Vec<Invalidation>, Box<dyn std::error::Error>> {
    let contents = fs::read_to_string(path)?;
    let mut rows = Vec::new();
    let mut seen = BTreeSet::new();
    for line in contents.lines() {
        let mut fields = line.splitn(3, '\t');
        let invalid_generation = fields.next().unwrap_or_default();
        let status = fields.next().unwrap_or_default();
        let reason = fields.next().unwrap_or_default();
        if invalid_generation.is_empty()
            || !matches!(status, "invalid" | "superseded")
            || reason.is_empty()
            || !seen.insert(invalid_generation.to_owned())
        {
            return Err("invalid invalidations.tsv row".into());
        }
        if invalid_generation == generation {
            return Err(format!("generation {generation} invalidates itself").into());
        }
        rows.push(Invalidation {
            generation: invalid_generation.to_owned(),
            status: status.to_owned(),
            reason: reason.to_owned(),
        });
    }
    if rows.is_empty() {
        return Err("invalidations.tsv is empty".into());
    }
    for required in [
        "review1", "review1b", "review1c", "review1d", "review1e", "review1f", "review1g",
        "review1h",
    ] {
        if !seen.contains(required) {
            return Err(
                format!("invalidations.tsv omits permanently invalidated {required}").into(),
            );
        }
    }
    Ok(rows)
}

fn replace_section(
    contents: &str,
    start_marker: &str,
    end_marker: &str,
    replacement: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let start = contents
        .find(start_marker)
        .ok_or_else(|| format!("inventory lacks section marker {start_marker:?}"))?;
    let end = contents[start..]
        .find(end_marker)
        .map(|offset| start + offset)
        .ok_or_else(|| format!("inventory lacks section terminator {end_marker:?}"))?;
    let mut output = String::with_capacity(contents.len() + replacement.len());
    output.push_str(&contents[..start]);
    output.push_str(replacement);
    output.push_str(&contents[end..]);
    Ok(output)
}

fn replace_mapping_line(
    contents: &str,
    key: &str,
    replacement: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let needle = format!("{key}:");
    let mut count = 0;
    let mut output = String::with_capacity(contents.len() + replacement.len());
    for line in contents.split_inclusive('\n') {
        let trimmed = line.trim_start();
        if trimmed.starts_with(&needle) {
            let indentation = &line[..line.len() - trimmed.len()];
            output.push_str(indentation);
            output.push_str(replacement);
            output.push('\n');
            count += 1;
        } else {
            output.push_str(line);
        }
    }
    if count != 1 {
        return Err(format!("expected one `{key}` mapping line, found {count}").into());
    }
    Ok(output)
}

fn replace_top_level_scalar(
    contents: &str,
    key: &str,
    value: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    if value.is_empty() || value.contains(['\n', '\r']) {
        return Err(format!("top-level `{key}` identity must be one non-empty line").into());
    }
    let needle = format!("{key}:");
    let mut count = 0;
    let mut output = String::with_capacity(contents.len() + value.len());
    for line in contents.split_inclusive('\n') {
        if line.starts_with(&needle) {
            output.push_str(&format!("{key}: {value}\n"));
            count += 1;
        } else {
            output.push_str(line);
        }
    }
    if count != 1 {
        return Err(format!("expected one top-level `{key}` field, found {count}").into());
    }
    Ok(output)
}

fn identity_from_inventory(
    contents: &str,
) -> Result<BTreeMap<String, serde_json::Value>, Box<dyn std::error::Error>> {
    let document: serde_yaml::Value = serde_yaml::from_str(contents)?;
    let identity = document
        .get("experiment_identity_json")
        .and_then(serde_yaml::Value::as_str)
        .ok_or("inventory lacks string experiment_identity_json")?;
    Ok(serde_json::from_str(identity)?)
}

fn render_identity(
    identity: &BTreeMap<String, serde_json::Value>,
) -> Result<String, Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(identity)? + "\n";
    let mut output = String::from("experiment_identity_json: |\n");
    for line in json.split_inclusive('\n') {
        output.push_str("  ");
        output.push_str(line);
    }
    Ok(output)
}

fn atomic_write(path: &Path, contents: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let parent = path.parent().ok_or("output path has no parent")?;
    let temporary = parent.join(format!(
        ".{}.refresh.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .ok_or("non-UTF-8 output name")?
    ));
    fs::write(&temporary, contents)?;
    fs::rename(temporary, path)?;
    Ok(())
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PublicationJournal {
    schema_version: u8,
    phase: String,
    nonce: u64,
    entries: Vec<PublicationJournalEntry>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PublicationGeneration {
    nonce: u64,
    schema_version: u8,
    status: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PublicationJournalEntry {
    destination: PathBuf,
    temporary: PathBuf,
    backup: PathBuf,
    had_original: bool,
}

fn sync_directory(path: &Path) -> io::Result<()> {
    File::open(path)?.sync_all()
}

fn transaction_root(outputs: &[(PathBuf, Vec<u8>)]) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let first = outputs
        .first()
        .ok_or("transactional output set must not be empty")?
        .0
        .parent()
        .ok_or("transactional output has no parent")?;
    if !first.is_absolute() {
        return Err("transactional outputs must use absolute paths".into());
    }
    let mut root = first.to_path_buf();
    for (path, _) in &outputs[1..] {
        while !path.starts_with(&root) {
            if !root.pop() {
                return Err("transactional outputs do not share a safe root".into());
            }
        }
    }
    if root == Path::new("/") {
        return Err("transactional output root must not be the filesystem root".into());
    }
    Ok(root)
}

fn write_publication_journal(
    journal_path: &Path,
    journal: &PublicationJournal,
) -> Result<(), Box<dyn std::error::Error>> {
    let parent = journal_path
        .parent()
        .ok_or("publication journal has no parent")?;
    let temporary = parent.join(".aiperf-baseline-refresh-transaction.json.tmp");
    if temporary.exists() {
        fs::remove_file(&temporary)?;
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    serde_json::to_writer(&mut file, journal)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    fs::rename(&temporary, journal_path)?;
    sync_directory(parent)?;
    Ok(())
}

fn publication_generation_path(root: &Path) -> PathBuf {
    root.join("artifacts/native-plugin-baseline/.publication-generation.json")
}

fn read_publication_generation(root: &Path) -> Result<Option<Vec<u8>>, Box<dyn std::error::Error>> {
    let path = publication_generation_path(root);
    let bytes = match snapshot_regular_file(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let generation: PublicationGeneration = serde_json::from_slice(&bytes)?;
    if generation.schema_version != 1 || generation.status != "committed" {
        return Err("baseline publication generation is not committed".into());
    }
    let mut canonical = serde_json::to_vec(&generation)?;
    canonical.push(b'\n');
    if canonical != bytes {
        return Err("baseline publication generation is not canonical".into());
    }
    Ok(Some(bytes))
}

fn write_publication_generation(
    root: &Path,
    nonce: u64,
    status: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = publication_generation_path(root);
    let parent = path
        .parent()
        .ok_or("publication generation has no parent")?
        .to_path_buf();
    fs::create_dir_all(&parent)?;
    let temporary = parent.join(".publication-generation.json.tmp");
    if temporary.exists() {
        fs::remove_file(&temporary)?;
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    serde_json::to_writer(
        &mut file,
        &PublicationGeneration {
            nonce,
            schema_version: 1,
            status: status.to_owned(),
        },
    )?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    fs::rename(temporary, &path)?;
    sync_directory(&parent)?;
    Ok(())
}

fn maybe_kill_publication(boundary: &str) {
    if std::env::var("AIPERF_REFRESH_KILL_AFTER_BOUNDARY").as_deref() == Ok(boundary) {
        #[cfg(unix)]
        {
            // SAFETY: sending SIGKILL to the current process has no memory-safety preconditions.
            unsafe { libc::kill(libc::getpid(), libc::SIGKILL) };
        }
        #[cfg(not(unix))]
        std::process::abort();
    }
}

fn recover_publication(
    journal_path: &Path,
    expected_outputs: &[(PathBuf, Vec<u8>)],
) -> Result<(), Box<dyn std::error::Error>> {
    if !journal_path.exists() {
        return Ok(());
    }
    let journal: PublicationJournal =
        serde_json::from_slice(&snapshot_regular_file(journal_path)?)?;
    if journal.schema_version != 1
        || !matches!(journal.phase.as_str(), "staging" | "committing")
        || journal.entries.len() != expected_outputs.len()
        || journal
            .entries
            .iter()
            .zip(expected_outputs)
            .any(|(entry, output)| entry.destination != output.0)
    {
        return Err("publication journal does not match the mandatory recovery set".into());
    }
    let root = journal_path
        .parent()
        .ok_or("publication journal has no parent")?;
    for (index, entry) in journal.entries.iter().enumerate() {
        let parent = entry
            .destination
            .parent()
            .ok_or("journal destination has no parent")?;
        let name = entry
            .destination
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("journal destination name is not UTF-8")?;
        if !entry.destination.starts_with(root)
            || entry.temporary
                != parent.join(format!(".{name}.refresh-{}-{index}.tmp", journal.nonce))
            || entry.backup != parent.join(format!(".{name}.refresh-{}-{index}.bak", journal.nonce))
        {
            return Err("publication journal contains an unsafe path".into());
        }
    }

    let is_complete_new_generation = journal.phase == "committing"
        && journal
            .entries
            .iter()
            .all(|entry| !entry.temporary.exists())
        && journal
            .entries
            .iter()
            .all(|entry| entry.destination.is_file());
    if is_complete_new_generation {
        for entry in &journal.entries {
            if entry.backup.exists() {
                fs::remove_file(&entry.backup)?;
            }
        }
    } else {
        for entry in journal.entries.iter().rev() {
            if entry.backup.exists() {
                if entry.destination.exists() {
                    fs::remove_file(&entry.destination)?;
                }
                fs::rename(&entry.backup, &entry.destination)?;
            } else if !entry.had_original && !entry.temporary.exists() && entry.destination.exists()
            {
                fs::remove_file(&entry.destination)?;
            }
            if entry.temporary.exists() {
                fs::remove_file(&entry.temporary)?;
            }
        }
    }
    let mut parents = BTreeSet::new();
    for entry in &journal.entries {
        if let Some(parent) = entry.destination.parent() {
            parents.insert(parent);
        }
    }
    for parent in parents {
        sync_directory(parent)?;
    }
    write_publication_generation(root, journal.nonce, "committed")?;
    fs::remove_file(journal_path)?;
    sync_directory(root)?;
    Ok(())
}

fn transactional_write_all(
    outputs: &[(PathBuf, Vec<u8>)],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut unique = BTreeSet::new();
    for (path, _) in outputs {
        if !unique.insert(path) {
            return Err(format!("duplicate transactional output: {}", path.display()).into());
        }
        fs::create_dir_all(path.parent().ok_or("transactional output has no parent")?)?;
    }
    let root = transaction_root(outputs)?;
    let journal_path = root.join(".aiperf-baseline-refresh-transaction.json");
    recover_publication(&journal_path, outputs)?;

    let nonce = read_publication_generation(&root)?
        .map(|bytes| serde_json::from_slice::<PublicationGeneration>(&bytes))
        .transpose()?
        .map(|generation| generation.nonce)
        .unwrap_or(0)
        .checked_add(1)
        .ok_or("publication generation nonce overflow")?;
    let entries = outputs
        .iter()
        .enumerate()
        .map(|(index, (path, _))| {
            let parent = path.parent().ok_or("transactional output has no parent")?;
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or("transactional output name is not UTF-8")?;
            Ok(PublicationJournalEntry {
                destination: path.clone(),
                temporary: parent.join(format!(".{name}.refresh-{nonce}-{index}.tmp")),
                backup: parent.join(format!(".{name}.refresh-{nonce}-{index}.bak")),
                had_original: path.exists(),
            })
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    let mut journal = PublicationJournal {
        schema_version: 1,
        phase: "staging".to_owned(),
        nonce,
        entries,
    };
    write_publication_journal(&journal_path, &journal)?;
    write_publication_generation(&root, nonce, "publishing")?;
    maybe_kill_publication("journal");

    let stage_result = (|| -> Result<(), Box<dyn std::error::Error>> {
        for (index, ((_, bytes), entry)) in outputs.iter().zip(&journal.entries).enumerate() {
            if std::env::var("AIPERF_REFRESH_FAIL_STAGING_INDEX").as_deref()
                == Ok(index.to_string().as_str())
            {
                return Err(format!("injected staging failure at output {index}").into());
            }
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&entry.temporary)?;
            file.write_all(bytes)?;
            file.sync_all()?;
        }
        Ok(())
    })();
    if let Err(error) = stage_result {
        recover_publication(&journal_path, outputs)?;
        return Err(error);
    }
    journal.phase = "committing".to_owned();
    write_publication_journal(&journal_path, &journal)?;
    maybe_kill_publication("commit-ready");

    let fail_after = std::env::var("AIPERF_REFRESH_FAIL_AFTER_RENAMES")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?;
    let transaction = (|| -> Result<(), Box<dyn std::error::Error>> {
        for (index, entry) in journal.entries.iter().enumerate() {
            let ordinal = index + 1;
            let parent = entry
                .destination
                .parent()
                .ok_or("transactional destination has no parent")?;
            if entry.had_original {
                fs::rename(&entry.destination, &entry.backup)?;
                sync_directory(parent)?;
            }
            maybe_kill_publication(&format!("backup:{ordinal}"));
            fs::rename(&entry.temporary, &entry.destination)?;
            sync_directory(parent)?;
            maybe_kill_publication(&format!("destination:{ordinal}"));
            if fail_after == Some(ordinal) {
                return Err(
                    format!("injected transactional failure after {ordinal} renames").into(),
                );
            }
        }
        Ok(())
    })();
    if let Err(error) = transaction {
        recover_publication(&journal_path, outputs)?;
        return Err(error);
    }
    for (index, entry) in journal.entries.iter().enumerate() {
        if entry.backup.exists() {
            fs::remove_file(&entry.backup)?;
            sync_directory(
                entry
                    .backup
                    .parent()
                    .ok_or("transactional backup has no parent")?,
            )?;
        }
        maybe_kill_publication(&format!("cleanup:{}", index + 1));
    }
    write_publication_generation(&root, nonce, "committed")?;
    maybe_kill_publication("journal-removal-ready");
    fs::remove_file(&journal_path)?;
    maybe_kill_publication("journal-removal");
    sync_directory(&root)?;
    Ok(())
}

fn baseline_relative_paths(mode: InventoryRefreshMode) -> Vec<PathBuf> {
    let mut relative_paths = vec![
        PathBuf::from("rust/benchmarks/plugin-parity.yaml"),
        PathBuf::from("artifacts/native-plugin-baseline/package-topology.json"),
    ];
    if mode != InventoryRefreshMode::PreCapture {
        relative_paths.extend([
            PathBuf::from("artifacts/native-plugin-baseline/allocation-probe.json"),
            PathBuf::from("artifacts/native-plugin-baseline/evidence-manifest.json"),
        ]);
    }
    relative_paths
}

fn verify_baseline_publication(
    mode: InventoryRefreshMode,
    repository_root: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let journal = repository_root.join(".aiperf-baseline-refresh-transaction.json");
    if journal.exists() {
        return Err("baseline publication has an interrupted durable transaction".into());
    }
    let generation = read_publication_generation(repository_root)?;
    #[cfg(debug_assertions)]
    if let Some(marker) = std::env::var_os("AIPERF_TEST_BASELINE_READER_PAUSE_MARKER") {
        let marker = PathBuf::from(marker);
        fs::write(&marker, b"reader acquired generation\n")?;
        let resume = marker.with_file_name(format!(
            "{}.continue",
            marker
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or("reader test marker name is not UTF-8")?
        ));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !resume.exists() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        if !resume.exists() {
            return Err("reader test continuation was not provided".into());
        }
    }
    for relative in baseline_relative_paths(mode) {
        let path = repository_root.join(relative);
        if !fs::symlink_metadata(&path)?.file_type().is_file() {
            return Err(format!(
                "baseline publication is not a regular file: {}",
                path.display()
            )
            .into());
        }
    }
    if journal.exists() || read_publication_generation(repository_root)? != generation {
        return Err("baseline publication generation changed during the whole read".into());
    }
    Ok(())
}

fn recover_baseline_publication(
    mode: InventoryRefreshMode,
    repository_root: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let outputs = baseline_relative_paths(mode)
        .into_iter()
        .map(|relative| (repository_root.join(relative), Vec::new()))
        .collect::<Vec<_>>();
    let root = transaction_root(&outputs)?;
    recover_publication(
        &root.join(".aiperf-baseline-refresh-transaction.json"),
        &outputs,
    )
}

fn publish_baseline_candidates(
    mode: InventoryRefreshMode,
    candidate_root: &Path,
    repository_root: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let outputs = baseline_relative_paths(mode)
        .into_iter()
        .map(|relative| {
            let candidate = candidate_root.join(&relative);
            if !fs::symlink_metadata(&candidate)
                .is_ok_and(|metadata| metadata.file_type().is_file())
            {
                return Err(format!(
                    "missing validated baseline candidate: {}",
                    candidate.display()
                )
                .into());
            }
            Ok((repository_root.join(relative), fs::read(candidate)?))
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    transactional_write_all(&outputs)
}

fn upsert_artifact_line(
    contents: &str,
    key: &str,
    replacement: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let marker = "artifacts:\n";
    let start = contents
        .find(marker)
        .ok_or("inventory lacks artifacts section")?
        + marker.len();
    let end = contents[start..]
        .find("invalid_capture_attempts:\n")
        .map(|offset| start + offset)
        .ok_or("inventory lacks artifacts section terminator")?;
    let artifacts = &contents[start..end];
    let refreshed = if artifacts
        .lines()
        .any(|line| line.trim_start().starts_with(&format!("{key}:")))
    {
        replace_mapping_line(artifacts, key, replacement)?
    } else {
        format!("{artifacts}  {replacement}\n")
    };
    let mut output = String::with_capacity(contents.len() + replacement.len() + 3);
    output.push_str(&contents[..start]);
    output.push_str(&refreshed);
    output.push_str(&contents[end..]);
    Ok(output)
}

struct EvidenceFacts {
    staged_path: String,
    bundle_bytes: u64,
    bundle_blake3: String,
    manifest_bytes: u64,
    manifest_blake3: String,
    manifest_file_count: usize,
    compact_manifest: Vec<u8>,
}

fn staged_evidence_facts(
    capture_root: &Path,
    generation: &str,
) -> Result<EvidenceFacts, Box<dyn std::error::Error>> {
    let evidence = capture_root.join("evidence");
    let manifest_path = required_receipt(capture_root, "evidence-manifest.json")?;
    verify_manifest(&manifest_path, &evidence)?;
    let manifest: Manifest = serde_json::from_reader(File::open(&manifest_path)?)?;
    let bundle_path = required_receipt(
        capture_root,
        &format!("aiperf-native-plugin-baseline-{generation}.tar.gz"),
    )?;
    let verification: BundleVerification = serde_json::from_reader(File::open(required_receipt(
        capture_root,
        "bundle-verification.json",
    )?)?)?;
    let (bundle_bytes, bundle_digest) = digest_file(&bundle_path)?;
    let (manifest_bytes, manifest_digest) = digest_file(&manifest_path)?;
    if verification.schema_version != 1
        || verification.generation != generation
        || verification.status != "extracted_manifest_verified"
        || verification.bundle_bytes != bundle_bytes
        || verification.bundle_blake3 != format!("blake3:{bundle_digest}")
        || verification.manifest_bytes != manifest_bytes
        || verification.manifest_blake3 != format!("blake3:{manifest_digest}")
    {
        return Err("staged bundle verification receipt does not match captured bytes".into());
    }
    Ok(EvidenceFacts {
        staged_path: bundle_path.to_string_lossy().into_owned(),
        bundle_bytes,
        bundle_blake3: format!("blake3:{bundle_digest}"),
        manifest_bytes,
        manifest_blake3: format!("blake3:{manifest_digest}"),
        manifest_file_count: manifest.files.len(),
        compact_manifest: fs::read(manifest_path)?,
    })
}

fn render_raw_samples(
    generation: &str,
    admission_status: &str,
    facts: Option<&EvidenceFacts>,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut output = format!(
        "raw_samples:\n  admission_status: {admission_status}\n  expected_generation: {generation}\n"
    );
    if let Some(facts) = facts {
        output.push_str(&format!(
            "  manifest: {{path: artifacts/native-plugin-baseline/evidence-manifest.json, bytes: {}, blake3: {}, file_count: {}}}\n",
            facts.manifest_bytes, facts.manifest_blake3, facts.manifest_file_count
        ));
        output.push_str(&format!(
            "  bundle: {{staged_path: {}, bytes: {}, blake3: {}}}\n",
            serde_json::to_string(&facts.staged_path)?,
            facts.bundle_bytes,
            facts.bundle_blake3
        ));
    }
    Ok(output)
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct MeasurementResults {
    schema_version: u8,
    generation: String,
    build_commands: serde_json::Value,
    runtime_measurements: serde_json::Value,
    allocation_probe: serde_json::Value,
    allocation_sources: serde_json::Value,
    exporter_observable_policy_blake3: String,
    exporter_corpus_blake3: String,
    exporter_build_receipt_blake3: String,
    exporter_repetition_receipts_blake3: String,
    exporter_repetition_receipts: serde_json::Value,
}

fn is_blake3(value: &str) -> bool {
    value
        .strip_prefix("blake3:")
        .is_some_and(|hex| hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_hexdigit()))
}

fn exact_object_keys(
    value: &serde_json::Value,
    label: &str,
    expected: &[&str],
) -> Result<(), Box<dyn std::error::Error>> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("measurement `{label}` must be an object"))?;
    let observed = object.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let expected = expected.iter().copied().collect::<BTreeSet<_>>();
    if observed != expected {
        return Err(format!("measurement `{label}` field set mismatch").into());
    }
    Ok(())
}

fn validate_measurement_results(
    results: &MeasurementResults,
    generation: &str,
    experiment_identity_blake3: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if results.schema_version != 1 || results.generation != generation {
        return Err("measurement receipt schema or generation mismatch".into());
    }
    exact_object_keys(
        &results.build_commands,
        "build_commands",
        &["default", "engine", "grpc", "parquet", "dynosim", "full"],
    )?;
    for (name, build) in results
        .build_commands
        .as_object()
        .ok_or("measurement build_commands must be an object")?
    {
        exact_object_keys(
            build,
            &format!("build_commands.{name}"),
            &[
                "command",
                "target_dir",
                "first_build_kind",
                "first_build_nanoseconds",
                "second_build_nanoseconds",
                "artifact_digest",
                "artifact_bytes",
                "first_log_digest",
                "second_log_digest",
            ],
        )?;
    }
    let runtime_measurements = results
        .runtime_measurements
        .as_object()
        .ok_or("measurement runtime_measurements must be an object")?;
    let scenario_names = runtime_measurements
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let expected_scenarios = [
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
    ]
    .into_iter()
    .collect::<BTreeSet<_>>();
    if scenario_names != expected_scenarios {
        return Err("measurement runtime scenario set mismatch".into());
    }
    for (name, measurement) in runtime_measurements {
        exact_object_keys(
            measurement,
            &format!("runtime_measurements.{name}"),
            &[
                "artifact_digest",
                "process_log_digest",
                "baseline_observation",
            ],
        )?;
    }
    exact_object_keys(
        &results.allocation_probe,
        "allocation_probe",
        &[
            "endpoint_preparation",
            "endpoint_formatting",
            "transport_dispatch",
            "response_reduction",
            "full_successful_request",
            "exporter_capture",
        ],
    )?;
    exact_object_keys(
        &results.allocation_sources,
        "allocation_sources",
        &["allocation_log", "exporter_log", "exporter_observation"],
    )?;
    for (name, artifact) in results
        .allocation_sources
        .as_object()
        .ok_or("measurement allocation_sources must be an object")?
    {
        exact_object_keys(
            artifact,
            &format!("allocation_sources.{name}"),
            &["path", "bytes", "blake3"],
        )?;
    }
    let repetitions = results
        .exporter_repetition_receipts
        .as_array()
        .ok_or("measurement exporter_repetition_receipts must be an array")?;
    if repetitions.len() != 16 {
        return Err("measurement exporter repetition receipt count mismatch".into());
    }
    if !is_blake3(&results.exporter_observable_policy_blake3)
        || !is_blake3(&results.exporter_corpus_blake3)
        || !is_blake3(&results.exporter_build_receipt_blake3)
        || !is_blake3(&results.exporter_repetition_receipts_blake3)
    {
        return Err("measurement exporter evidence digest is malformed".into());
    }
    let mut repetition_bytes = serde_json::to_vec(repetitions)?;
    repetition_bytes.push(b'\n');
    if results.exporter_repetition_receipts_blake3
        != format!("blake3:{}", blake3::hash(&repetition_bytes))
    {
        return Err("measurement exporter repetition vector digest mismatch".into());
    }
    let engine_digest = results
        .build_commands
        .pointer("/engine/artifact_digest")
        .and_then(serde_json::Value::as_str)
        .ok_or("measurement engine build lacks artifact_digest")?;
    let mut comparison_binding = None;
    let mut corpus_binding = None;
    let mut active_duration = 0_u64;
    for (ordinal, receipt) in repetitions.iter().enumerate() {
        exact_object_keys(
            receipt,
            &format!("exporter_repetition_receipts[{ordinal}]"),
            &[
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
            ],
        )?;
        let object = receipt
            .as_object()
            .ok_or("measurement exporter repetition receipt must be an object")?;
        let raw_digest = object
            .get("raw_observable_blake3")
            .and_then(serde_json::Value::as_str)
            .ok_or("exporter repetition lacks raw_observable_blake3")?;
        let comparison_digest = object
            .get("comparison_observable_blake3")
            .and_then(serde_json::Value::as_str)
            .ok_or("exporter repetition lacks comparison_observable_blake3")?;
        let provenance_digest = object
            .get("provenance_receipt_blake3")
            .and_then(serde_json::Value::as_str)
            .ok_or("exporter repetition lacks provenance_receipt_blake3")?;
        let corpus = object
            .get("corpus_blake3")
            .and_then(serde_json::Value::as_str)
            .ok_or("exporter repetition lacks corpus_blake3")?;
        let duration = object
            .get("active_duration_ns")
            .and_then(serde_json::Value::as_u64)
            .filter(|value| *value > 0)
            .ok_or("measurement exporter repetition duration must be positive")?;
        if object
            .get("schema_version")
            .and_then(serde_json::Value::as_u64)
            != Some(1)
            || object
                .get("experiment_identity_blake3")
                .and_then(serde_json::Value::as_str)
                != Some(experiment_identity_blake3)
            || object
                .get("attempt_ordinal")
                .and_then(serde_json::Value::as_u64)
                != Some(0)
            || object
                .get("scenario_id")
                .and_then(serde_json::Value::as_str)
                != Some("exporter_100k")
            || object.get("pair_id").and_then(serde_json::Value::as_str)
                != Some("task1-static-calibration")
            || object.get("member").and_then(serde_json::Value::as_str) != Some("static")
            || object
                .get("repetition_ordinal")
                .and_then(serde_json::Value::as_u64)
                != Some(ordinal as u64)
            || object
                .get("processed_records")
                .and_then(serde_json::Value::as_u64)
                != Some(100_000)
            || object
                .get("observable_kind")
                .and_then(serde_json::Value::as_str)
                != Some("artifact_tree")
            || corpus != results.exporter_corpus_blake3
            || object
                .get("build_artifact_blake3")
                .and_then(serde_json::Value::as_str)
                != Some(engine_digest)
            || !is_blake3(raw_digest)
            || !is_blake3(comparison_digest)
            || !is_blake3(provenance_digest)
            || !is_blake3(corpus)
            || !object
                .get("build_receipt_blake3")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|value| value == results.exporter_build_receipt_blake3)
        {
            return Err("measurement exporter repetition binding mismatch".into());
        }
        if corpus_binding.is_some_and(|expected| expected != corpus) {
            return Err("measurement exporter repetition corpus changed".into());
        }
        corpus_binding = Some(corpus);
        if comparison_binding.is_some_and(|expected| expected != comparison_digest) {
            return Err("measurement exporter repetition comparison observable changed".into());
        }
        comparison_binding = Some(comparison_digest);
        active_duration = active_duration
            .checked_add(duration)
            .ok_or("measurement exporter duration overflow")?;
    }
    if active_duration < 30_000_000_000 {
        return Err("measurement static exporter sample is shorter than 30 seconds".into());
    }
    if results
        .allocation_probe
        .pointer("/exporter_capture/exporter_interval_nanoseconds")
        .and_then(serde_json::Value::as_u64)
        != Some(active_duration)
    {
        return Err("measurement exporter duration sum mismatch".into());
    }
    Ok(())
}

fn exporter_measurement_binding(
    results: &MeasurementResults,
) -> Result<serde_json::Map<String, serde_json::Value>, Box<dyn std::error::Error>> {
    let repetitions = results
        .exporter_repetition_receipts
        .as_array()
        .ok_or("measurement exporter repetition receipts must be an array")?;
    let first = repetitions
        .first()
        .and_then(serde_json::Value::as_object)
        .ok_or("measurement exporter repetition receipts are empty")?;
    let active_duration_ns = repetitions.iter().try_fold(0_u64, |sum, receipt| {
        let duration = receipt
            .get("active_duration_ns")
            .and_then(serde_json::Value::as_u64)
            .ok_or("measurement exporter repetition lacks duration")?;
        sum.checked_add(duration)
            .ok_or("measurement exporter duration overflow")
    })?;
    let fields = [
        "experiment_identity_blake3",
        "attempt_ordinal",
        "scenario_id",
        "pair_id",
        "member",
        "corpus_blake3",
        "observable_kind",
    ];
    let mut binding = serde_json::Map::new();
    for field in fields {
        binding.insert(
            field.to_owned(),
            first
                .get(field)
                .cloned()
                .ok_or_else(|| format!("measurement exporter receipt lacks `{field}`"))?,
        );
    }
    binding.insert("corpus_records".to_owned(), serde_json::json!(100_000));
    binding.insert("sample_repetitions".to_owned(), serde_json::json!(16));
    binding.insert("processed_records".to_owned(), serde_json::json!(1_600_000));
    binding.insert(
        "retained_artifact_records".to_owned(),
        serde_json::json!(100_000),
    );
    binding.insert(
        "active_duration_ns".to_owned(),
        serde_json::json!(active_duration_ns),
    );
    binding.insert(
        "repetition_receipts_blake3".to_owned(),
        serde_json::json!(results.exporter_repetition_receipts_blake3),
    );
    binding.insert(
        "observable_policy_blake3".to_owned(),
        serde_json::json!(results.exporter_observable_policy_blake3),
    );
    binding.insert(
        "repetition_receipts".to_owned(),
        results.exporter_repetition_receipts.clone(),
    );
    Ok(binding)
}

#[derive(Clone, Copy, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ExporterPolicyMode {
    Paired,
    StaticCalibration,
}

#[derive(Clone, Copy, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ExporterObservableKind {
    ArtifactTree,
    CapturedStream,
    ReceiverTranscript,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ExporterTransportFieldsRemoved {
    keys: Vec<String>,
    protocol: String,
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum ExporterOutputSelector {
    ArtifactContent { path: String },
    CapturedStream,
    TranscriptBody { sequence: u64 },
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum ExporterLocator {
    ByteRange { length: u64, offset: u64 },
    JsonPointer { pointer: String },
    WholeOutput,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ExporterExpectedValue {
    encoding: String,
    value: serde_json::Value,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ExporterProvenanceSlot {
    #[serde(skip_serializing_if = "Option::is_none")]
    dynamic_expected: Option<ExporterExpectedValue>,
    locator: ExporterLocator,
    output_selector: ExporterOutputSelector,
    replacement: ExporterExpectedValue,
    slot_id: String,
    static_expected: ExporterExpectedValue,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ExporterObservableScenario {
    allows_empty: bool,
    observable_kind: ExporterObservableKind,
    provenance_slots: Vec<ExporterProvenanceSlot>,
    scenario_id: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ExporterObservablePolicyV1 {
    mode: ExporterPolicyMode,
    receiver_transport_fields_removed: Vec<ExporterTransportFieldsRemoved>,
    scenarios: Vec<ExporterObservableScenario>,
    schema_version: u8,
}

fn is_policy_identifier(value: &str) -> bool {
    let bytes = value.as_bytes();
    (1..=128).contains(&bytes.len())
        && (bytes[0].is_ascii_lowercase() || bytes[0].is_ascii_digit())
        && bytes.iter().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'.' | b'-')
        })
}

fn validate_policy_json_value(value: &serde_json::Value) -> Result<(), Box<dyn std::error::Error>> {
    match value {
        serde_json::Value::String(value) if value.contains('\0') => {
            Err("exporter policy strings must not contain NUL".into())
        }
        serde_json::Value::Array(values) => {
            for value in values {
                validate_policy_json_value(value)?;
            }
            Ok(())
        }
        serde_json::Value::Object(values) => {
            for (key, value) in values {
                if key.contains('\0') {
                    return Err("exporter policy object keys must not contain NUL".into());
                }
                validate_policy_json_value(value)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn validate_artifact_policy_path(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let parsed = Path::new(path);
    if path.is_empty()
        || path.contains(['\0', '\n', '\r', '\\'])
        || path.starts_with('/')
        || path.ends_with('/')
        || path.contains("//")
        || parsed.is_absolute()
        || parsed
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err("exporter artifact selector path is not normalized".into());
    }
    Ok(())
}

fn validate_json_pointer(pointer: &str) -> Result<(), Box<dyn std::error::Error>> {
    if pointer.contains('\0') || (!pointer.is_empty() && !pointer.starts_with('/')) {
        return Err("exporter policy contains malformed JSON pointer".into());
    }
    let mut bytes = pointer.bytes();
    while let Some(byte) = bytes.next() {
        if byte == b'~'
            && !bytes
                .next()
                .is_some_and(|escaped| matches!(escaped, b'0' | b'1'))
        {
            return Err("exporter policy contains malformed JSON pointer".into());
        }
    }
    Ok(())
}

fn decoded_json_pointer(pointer: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    validate_json_pointer(pointer)?;
    if pointer.is_empty() {
        return Ok(Vec::new());
    }
    pointer[1..]
        .split('/')
        .map(|token| {
            let mut decoded = String::new();
            let mut characters = token.chars();
            while let Some(character) = characters.next() {
                if character == '~' {
                    decoded.push(match characters.next() {
                        Some('0') => '~',
                        Some('1') => '/',
                        _ => return Err("exporter policy contains malformed JSON pointer".into()),
                    });
                } else {
                    decoded.push(character);
                }
            }
            Ok(decoded)
        })
        .collect()
}

fn exporter_locators_overlap(
    left: &ExporterLocator,
    right: &ExporterLocator,
) -> Result<bool, Box<dyn std::error::Error>> {
    match (left, right) {
        (ExporterLocator::WholeOutput, _) | (_, ExporterLocator::WholeOutput) => Ok(true),
        (
            ExporterLocator::ByteRange {
                length: left_length,
                offset: left_offset,
            },
            ExporterLocator::ByteRange {
                length: right_length,
                offset: right_offset,
            },
        ) => {
            Ok(*left_offset < right_offset + right_length
                && *right_offset < left_offset + left_length)
        }
        (
            ExporterLocator::JsonPointer {
                pointer: left_pointer,
            },
            ExporterLocator::JsonPointer {
                pointer: right_pointer,
            },
        ) => {
            let left = decoded_json_pointer(left_pointer)?;
            let right = decoded_json_pointer(right_pointer)?;
            Ok(left.starts_with(&right) || right.starts_with(&left))
        }
        _ => Ok(false),
    }
}

fn validate_exporter_expected(
    value: &ExporterExpectedValue,
    locator: &ExporterLocator,
) -> Result<(), Box<dyn std::error::Error>> {
    validate_policy_json_value(&value.value)?;
    match (value.encoding.as_str(), locator) {
        ("canonical_json", ExporterLocator::JsonPointer { .. }) => Ok(()),
        ("hex_bytes", ExporterLocator::ByteRange { .. } | ExporterLocator::WholeOutput) => {
            let encoded = value
                .value
                .as_str()
                .ok_or("hex_bytes exporter policy value must be a string")?;
            if encoded.len() % 2 != 0
                || !encoded
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                return Err("hex_bytes exporter policy value must be even lower-case hex".into());
            }
            Ok(())
        }
        _ => Err("exporter policy encoding is incompatible with its locator".into()),
    }
}

fn validate_exporter_policy(
    policy: &ExporterObservablePolicyV1,
    authenticated_receiver_protocols: &BTreeSet<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    if policy.schema_version != 1 {
        return Err("exporter observable policy schema_version must be 1".into());
    }
    let mut previous_protocol = None;
    for removal in &policy.receiver_transport_fields_removed {
        if !is_policy_identifier(&removal.protocol)
            || previous_protocol.is_some_and(|previous| previous >= removal.protocol.as_str())
        {
            return Err("exporter policy protocols must be sorted and unique".into());
        }
        if !authenticated_receiver_protocols.contains(removal.protocol.as_str()) {
            return Err(
                "exporter policy contains a transport removal absent from the authenticated receiver protocols"
                    .into(),
            );
        }
        previous_protocol = Some(removal.protocol.as_str());
        let mut previous_key = None;
        for key in &removal.keys {
            if key.is_empty()
                || key.contains('\0')
                || key.bytes().any(|byte| byte.is_ascii_uppercase())
                || previous_key.is_some_and(|previous| previous >= key.as_str())
            {
                return Err(
                    "exporter policy metadata keys must be lower-case, sorted, and unique".into(),
                );
            }
            previous_key = Some(key.as_str());
        }
    }
    if !policy.receiver_transport_fields_removed.is_empty()
        && !policy
            .scenarios
            .iter()
            .any(|scenario| scenario.observable_kind == ExporterObservableKind::ReceiverTranscript)
    {
        return Err(
            "exporter policy contains a transport removal unused by every receiver scenario".into(),
        );
    }
    let mut previous_scenario = None;
    for scenario in &policy.scenarios {
        if !is_policy_identifier(&scenario.scenario_id)
            || previous_scenario.is_some_and(|previous| previous >= scenario.scenario_id.as_str())
        {
            return Err("exporter policy scenarios must be sorted and unique".into());
        }
        previous_scenario = Some(scenario.scenario_id.as_str());
        let mut previous_slot = None;
        let mut selector_locators = BTreeSet::new();
        let mut locators_by_selector = BTreeMap::<String, Vec<&ExporterLocator>>::new();
        for slot in &scenario.provenance_slots {
            if !is_policy_identifier(&slot.slot_id)
                || previous_slot.is_some_and(|previous| previous >= slot.slot_id.as_str())
            {
                return Err("exporter policy slots must be sorted and unique".into());
            }
            previous_slot = Some(slot.slot_id.as_str());
            match (&scenario.observable_kind, &slot.output_selector) {
                (
                    ExporterObservableKind::ArtifactTree,
                    ExporterOutputSelector::ArtifactContent { path },
                ) => validate_artifact_policy_path(path)?,
                (
                    ExporterObservableKind::CapturedStream,
                    ExporterOutputSelector::CapturedStream,
                )
                | (
                    ExporterObservableKind::ReceiverTranscript,
                    ExporterOutputSelector::TranscriptBody { .. },
                ) => {}
                _ => {
                    return Err(
                        "exporter policy selector is incompatible with observable kind".into(),
                    );
                }
            }
            match &slot.locator {
                ExporterLocator::ByteRange { length, offset } => {
                    if *length == 0 || offset.checked_add(*length).is_none() {
                        return Err("exporter byte_range must be nonempty and bounded".into());
                    }
                }
                ExporterLocator::JsonPointer { pointer } => validate_json_pointer(pointer)?,
                ExporterLocator::WholeOutput => {}
            }
            validate_exporter_expected(&slot.static_expected, &slot.locator)?;
            validate_exporter_expected(&slot.replacement, &slot.locator)?;
            if slot.static_expected.encoding != slot.replacement.encoding {
                return Err("exporter policy slot encodings disagree".into());
            }
            match (policy.mode, slot.dynamic_expected.as_ref()) {
                (ExporterPolicyMode::Paired, Some(dynamic)) => {
                    validate_exporter_expected(dynamic, &slot.locator)?;
                    if dynamic.encoding != slot.replacement.encoding {
                        return Err("exporter policy slot encodings disagree".into());
                    }
                }
                (ExporterPolicyMode::Paired, None) => {
                    return Err("paired exporter policy slot lacks dynamic_expected".into());
                }
                (ExporterPolicyMode::StaticCalibration, Some(_)) => {
                    return Err(
                        "static_calibration exporter policy slot contains dynamic_expected".into(),
                    );
                }
                _ => {}
            }
            let selector_locator = serde_json::to_string(&(&slot.output_selector, &slot.locator))?;
            if !selector_locators.insert(selector_locator) {
                return Err("exporter policy contains duplicate selector/locator pair".into());
            }
            let selector = serde_json::to_string(&slot.output_selector)?;
            let peer_locators = locators_by_selector.entry(selector).or_default();
            for peer in peer_locators.iter() {
                if exporter_locators_overlap(peer, &slot.locator)? {
                    return Err("exporter policy contains overlapping output slots".into());
                }
            }
            peer_locators.push(&slot.locator);
        }
    }
    Ok(())
}

struct DuplicateRejectingJson;

impl<'de> Deserialize<'de> for DuplicateRejectingJson {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct JsonVisitor;

        impl<'de> Visitor<'de> for JsonVisitor {
            type Value = DuplicateRejectingJson;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a JSON value without duplicate object keys")
            }

            fn visit_bool<E>(self, _value: bool) -> Result<Self::Value, E> {
                Ok(DuplicateRejectingJson)
            }

            fn visit_i64<E>(self, _value: i64) -> Result<Self::Value, E> {
                Ok(DuplicateRejectingJson)
            }

            fn visit_u64<E>(self, _value: u64) -> Result<Self::Value, E> {
                Ok(DuplicateRejectingJson)
            }

            fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E> {
                Ok(DuplicateRejectingJson)
            }

            fn visit_str<E>(self, _value: &str) -> Result<Self::Value, E> {
                Ok(DuplicateRejectingJson)
            }

            fn visit_none<E>(self) -> Result<Self::Value, E> {
                Ok(DuplicateRejectingJson)
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                Ok(DuplicateRejectingJson)
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                while sequence.next_element::<DuplicateRejectingJson>()?.is_some() {}
                Ok(DuplicateRejectingJson)
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut keys = BTreeSet::new();
                while let Some(key) = map.next_key::<String>()? {
                    if !keys.insert(key.clone()) {
                        return Err(A::Error::custom(format!(
                            "duplicate JSON object key `{key}`"
                        )));
                    }
                    map.next_value::<DuplicateRejectingJson>()?;
                }
                Ok(DuplicateRejectingJson)
            }
        }

        deserializer.deserialize_any(JsonVisitor)
    }
}

fn reject_duplicate_json_keys(bytes: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    DuplicateRejectingJson::deserialize(&mut deserializer)?;
    deserializer.end()?;
    Ok(())
}

fn parse_exporter_observable_policy(
    bytes: &[u8],
    authenticated_receiver_protocols: &BTreeSet<&str>,
) -> Result<ExporterObservablePolicyV1, Box<dyn std::error::Error>> {
    reject_duplicate_json_keys(bytes)?;
    let policy: ExporterObservablePolicyV1 = serde_json::from_slice(bytes)?;
    validate_exporter_policy(&policy, authenticated_receiver_protocols)?;
    let mut canonical = serde_json_canonicalizer::to_vec(&policy)?;
    canonical.push(b'\n');
    if canonical != bytes {
        return Err("exporter observable policy is not exact RFC 8785 JCS plus newline".into());
    }
    Ok(policy)
}

#[derive(Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ArtifactTreeKind {
    EmptyDirectory,
    RegularFile,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ArtifactTreeEntry {
    blake3: String,
    kind: ArtifactTreeKind,
    length: u64,
    path: String,
}

fn is_lower_blake3(value: &str) -> bool {
    value.strip_prefix("blake3:").is_some_and(|hex| {
        hex.len() == 64
            && hex
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}

fn parse_artifact_tree_observable(
    bytes: &[u8],
) -> Result<Vec<ArtifactTreeEntry>, Box<dyn std::error::Error>> {
    reject_duplicate_json_keys(bytes)?;
    let entries: Vec<ArtifactTreeEntry> = serde_json::from_slice(bytes)?;
    let empty_digest = format!("blake3:{}", blake3::hash(b""));
    let mut previous_path = None;
    for entry in &entries {
        validate_artifact_policy_path(&entry.path)?;
        if previous_path.is_some_and(|previous| previous >= entry.path.as_str()) {
            return Err("artifact-tree paths must be sorted and unique".into());
        }
        previous_path = Some(entry.path.as_str());
        if !is_lower_blake3(&entry.blake3) {
            return Err("artifact-tree digest must be lower-case BLAKE3".into());
        }
        if entry.kind == ArtifactTreeKind::EmptyDirectory
            && (entry.length != 0 || entry.blake3 != empty_digest)
        {
            return Err("empty artifact-tree directory has nonempty content identity".into());
        }
    }
    let mut canonical = serde_json_canonicalizer::to_vec(&entries)?;
    canonical.push(b'\n');
    if canonical != bytes {
        return Err("artifact-tree observable is not exact RFC 8785 JCS plus newline".into());
    }
    Ok(entries)
}

fn exporter_authored_contract(
    identity_root: &Path,
) -> Result<serde_json::Map<String, serde_json::Value>, Box<dyn std::error::Error>> {
    let policy_path = required_receipt(identity_root, "exporter-observable-policy.json")?;
    let policy =
        parse_exporter_observable_policy(&snapshot_regular_file(&policy_path)?, &BTreeSet::new())?;
    if policy.mode != ExporterPolicyMode::StaticCalibration
        || !policy.receiver_transport_fields_removed.is_empty()
        || policy.scenarios.len() != 1
        || policy.scenarios[0].scenario_id != "exporter_100k"
    {
        return Err("exporter observable policy is not Task-1 static calibration".into());
    }
    let scenario = &policy.scenarios[0];
    if scenario.observable_kind != ExporterObservableKind::ArtifactTree
        || scenario.allows_empty
        || !scenario.provenance_slots.is_empty()
    {
        return Err("Task-1 exporter observable policy scenario mismatch".into());
    }
    let corpus: serde_json::Value = serde_json::from_reader(File::open(required_receipt(
        identity_root,
        "exporter-static-calibration-corpus.json",
    )?)?)?;
    exact_object_keys(
        &corpus,
        "exporter static calibration corpus",
        &["schema_version", "corpus_records", "record"],
    )?;
    let corpus_records = corpus["corpus_records"]
        .as_u64()
        .filter(|records| *records == 100_000)
        .ok_or("exporter static calibration corpus must contain 100000 records")?;
    if corpus["schema_version"] != 1 || !corpus["record"].is_object() {
        return Err("exporter static calibration corpus schema mismatch".into());
    }
    let repetitions = 16_u64;
    let processed_records = corpus_records
        .checked_mul(repetitions)
        .ok_or("exporter processed-record count overflow")?;
    Ok(serde_json::Map::from_iter([
        (
            "request_budget".to_owned(),
            serde_json::json!(processed_records),
        ),
        (
            "corpus_records".to_owned(),
            serde_json::json!(corpus_records),
        ),
        (
            "sample_repetitions".to_owned(),
            serde_json::json!(repetitions),
        ),
        (
            "processed_records".to_owned(),
            serde_json::json!(processed_records),
        ),
        (
            "retained_artifact_records".to_owned(),
            serde_json::json!(corpus_records),
        ),
        (
            "observable_kind".to_owned(),
            serde_json::json!("artifact_tree"),
        ),
        (
            "pair_id".to_owned(),
            serde_json::json!("task1-static-calibration"),
        ),
        ("member".to_owned(), serde_json::json!("static")),
        ("attempt_ordinal".to_owned(), serde_json::json!(0)),
        (
            "response_shape".to_owned(),
            serde_json::json!(
                "one static-calibration member sample comprising 16 sequential repetitions of the same 100000-input-record deterministic corpus; one complete 100000-record artifact retained; processed_records=1600000; no sleep or padding"
            ),
        ),
    ]))
}

fn render_yaml_section(
    name: &str,
    value: &serde_json::Value,
) -> Result<String, Box<dyn std::error::Error>> {
    let yaml = serde_yaml::to_string(value)?;
    let mut output = format!("{name}:\n");
    for line in yaml.lines() {
        output.push_str("  ");
        output.push_str(line);
        output.push('\n');
    }
    Ok(output)
}

fn render_compact_allocation(
    generation: &str,
    results: &MeasurementResults,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut samples = results.allocation_probe.clone();
    for sample in samples
        .as_object_mut()
        .ok_or("measurement allocation_probe must be an object")?
        .values_mut()
    {
        let sample = sample
            .as_object_mut()
            .ok_or("measurement allocation sample must be an object")?;
        if let Some(value) = sample.remove("allocations_per_request") {
            sample.insert("allocation_count_per_request".to_owned(), value);
        }
    }
    let mut bytes = serde_json::to_vec_pretty(&serde_json::json!({
        "schema_version": 1,
        "generation": generation,
        "allocator": "mimalloc 0.1.52 / libmimalloc-sys 0.1.49",
        "measurement": "TLS-scoped GlobalAlloc counters around warmed single-thread request intervals",
        "samples": samples,
        "source_receipts": results.allocation_sources.clone(),
    }))?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn digest_string(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    Ok(format!("blake3:{}", digest_file(path)?.1))
}

fn required_json_f64(
    value: &serde_json::Value,
    pointer: &str,
) -> Result<f64, Box<dyn std::error::Error>> {
    value
        .pointer(pointer)
        .and_then(serde_json::Value::as_f64)
        .filter(|value| value.is_finite())
        .ok_or_else(|| format!("capture receipt lacks finite `{pointer}`").into())
}

fn read_build_summary(
    evidence_root: &Path,
    name: &str,
    command: &str,
    artifact_name: &str,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let directory = evidence_root.join("builds").join(name);
    let summary = required_receipt(&directory, "summary.txt")?;
    let contents = fs::read_to_string(&summary)?;
    let mut fields = BTreeMap::new();
    let mut digest_rows = 0_usize;
    for line in contents.lines() {
        if let Some((key, value)) = line.split_once('=') {
            if fields.insert(key.to_owned(), value.to_owned()).is_some() {
                return Err(format!("duplicate build summary field `{key}` for {name}").into());
            }
        } else if line.starts_with("blake3:") {
            digest_rows += 1;
        } else {
            return Err(format!("invalid build summary row for {name}: {line}").into());
        }
    }
    let expected_fields = [
        "command",
        "target_dir",
        "first_build_kind",
        "first_build_nanoseconds",
        "second_build_nanoseconds",
    ]
    .into_iter()
    .collect::<BTreeSet<_>>();
    if fields.keys().map(String::as_str).collect::<BTreeSet<_>>() != expected_fields
        || digest_rows != 3
        || fields.get("command").map(String::as_str) != Some(command)
        || fields.get("first_build_kind").map(String::as_str) != Some("isolated_clean_target")
    {
        return Err(format!("build summary contract mismatch for {name}").into());
    }
    let artifact = required_receipt(
        &evidence_root.join("builds/artifacts"),
        &format!("{name}-{artifact_name}"),
    )?;
    let clean_log = required_receipt(&directory, "clean.log")?;
    let second_log = required_receipt(&directory, "second.log")?;
    let (artifact_bytes, _) = digest_file(&artifact)?;
    Ok(serde_json::json!({
        "command": command,
        "target_dir": fields["target_dir"],
        "first_build_kind": "isolated_clean_target",
        "first_build_nanoseconds": fields["first_build_nanoseconds"].parse::<u64>()?,
        "second_build_nanoseconds": fields["second_build_nanoseconds"].parse::<u64>()?,
        "artifact_digest": digest_string(&artifact)?,
        "artifact_bytes": artifact_bytes,
        "first_log_digest": digest_string(&clean_log)?,
        "second_log_digest": digest_string(&second_log)?,
    }))
}

fn runtime_observation(
    scenario: &Path,
    expected_budget: u64,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let resource: serde_json::Value =
        serde_json::from_reader(File::open(required_receipt(scenario, "resource.json")?)?)?;
    exact_object_keys(
        &resource,
        "runtime resource",
        &[
            "command",
            "exit_status",
            "timed_out",
            "end_to_end_nanoseconds",
            "user_cpu_nanoseconds",
            "system_cpu_nanoseconds",
            "maximum_resident_set_kib",
        ],
    )?;
    if resource["exit_status"].as_i64() != Some(0) || resource["timed_out"].as_bool() != Some(false)
    {
        return Err("runtime resource receipt is not successful".into());
    }
    let report_path = required_receipt(&scenario.join("artifacts"), "profile_export_aiperf.json")?;
    let report: serde_json::Value = serde_json::from_reader(File::open(report_path)?)?;
    for metric in ["request_count", "completed_request_count"] {
        for statistic in ["avg", "min", "max"] {
            if required_json_f64(&report, &format!("/{metric}/{statistic}"))?
                != expected_budget as f64
            {
                return Err(
                    format!("runtime report `{metric}.{statistic}` budget mismatch").into(),
                );
            }
        }
    }
    let user = resource["user_cpu_nanoseconds"]
        .as_u64()
        .ok_or("runtime resource lacks user_cpu_nanoseconds")?;
    let system = resource["system_cpu_nanoseconds"]
        .as_u64()
        .ok_or("runtime resource lacks system_cpu_nanoseconds")?;
    let inter_token = report.get("inter_token_latency");
    let percentile = |metric: &str, statistic: &str| -> Result<f64, Box<dyn std::error::Error>> {
        if metric == "inter_token_latency" && inter_token.is_none_or(serde_json::Value::is_null) {
            Ok(0.0)
        } else {
            required_json_f64(&report, &format!("/{metric}/{statistic}"))
        }
    };
    Ok(serde_json::json!({
        "duration_seconds": required_json_f64(&report, "/benchmark_duration/avg")?,
        "successful_requests_per_second": required_json_f64(&report, "/request_throughput/avg")?,
        "output_tokens_per_second": required_json_f64(&report, "/output_token_throughput/avg")?,
        "cpu_nanoseconds_per_successful_request": (user + system) as f64 / expected_budget as f64,
        "ttft_p50": percentile("time_to_first_token", "p50")?,
        "ttft_p90": percentile("time_to_first_token", "p90")?,
        "ttft_p99": percentile("time_to_first_token", "p99")?,
        "itl_p50": percentile("inter_token_latency", "p50")?,
        "itl_p90": percentile("inter_token_latency", "p90")?,
        "itl_p99": percentile("inter_token_latency", "p99")?,
    }))
}

fn normalized_allocation_sample(
    sample: &serde_json::Value,
    exporter: bool,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let mut output = serde_json::Map::new();
    output.insert("iterations".to_owned(), sample["iterations"].clone());
    output.insert(
        "allocations_per_request".to_owned(),
        sample["allocation_count_per_request"].clone(),
    );
    output.insert(
        "allocated_bytes_per_request".to_owned(),
        sample["allocated_bytes_per_request"].clone(),
    );
    for field in if exporter {
        &[
            "corpus_records",
            "sample_repetitions",
            "processed_records",
            "retained_artifact_records",
            "exporter_interval_nanoseconds",
            "exporter_nanoseconds_per_record",
        ][..]
    } else {
        &["chunks_per_response", "nanoseconds_per_request"][..]
    } {
        if let Some(value) = sample.get(*field) {
            output.insert((*field).to_owned(), value.clone());
        }
    }
    Ok(serde_json::Value::Object(output))
}

fn write_measurement_results(
    generation: &str,
    evidence_root: &Path,
    output: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    validate_generation(generation)?;
    let build_specs = [
        (
            "default",
            "cargo build --locked -p aiperf-cli --release",
            "aiperf",
        ),
        (
            "engine",
            "cargo build --locked -p aiperf-runtime --release --features engine",
            "libaiperf_runtime.rlib",
        ),
        (
            "grpc",
            "cargo build --locked -p aiperf-cli --release --features grpc",
            "aiperf",
        ),
        (
            "parquet",
            "cargo build --locked -p aiperf-cli --release --features parquet",
            "aiperf",
        ),
        (
            "dynosim",
            "cargo build --locked -p aiperf-cli --release --features dynosim",
            "aiperf",
        ),
        (
            "full",
            "cargo build --locked -p aiperf-cli --release --features full",
            "aiperf",
        ),
    ];
    let mut builds = serde_json::Map::new();
    for (name, command, artifact) in build_specs {
        builds.insert(
            name.to_owned(),
            read_build_summary(evidence_root, name, command, artifact)?,
        );
    }
    let runtime_specs = [
        ("http_non_streaming_c1", "http-nonstream-c1", 1_000),
        ("http_non_streaming_c64", "http-nonstream-c64", 64_000),
        ("http_streaming_c1", "http-streaming-c1", 1_000),
        ("http_streaming_c64", "http-streaming-c64", 64_000),
        ("grpc_unary_c1", "grpc-unary-c1", 1_000),
        ("grpc_unary_c64", "grpc-unary-c64", 64_000),
        ("grpc_streaming_c1", "grpc-streaming-c1", 1_000),
        ("grpc_streaming_c64", "grpc-streaming-c64", 64_000),
        ("http_streaming_workers4", "http-streaming-workers4", 64_000),
        ("otlp_disabled_capture", "otlp-disabled-capture", 64_000),
        ("otlp_enabled_capture", "otlp-enabled-capture", 64_000),
    ];
    let mut runtime = serde_json::Map::new();
    for (inventory_name, directory_name, budget) in runtime_specs {
        let scenario = evidence_root.join("runtime").join(directory_name);
        runtime.insert(
            inventory_name.to_owned(),
            serde_json::json!({
                "artifact_digest": digest_string(&required_receipt(&scenario.join("artifacts"), "profile_export_aiperf.json")?)?,
                "process_log_digest": digest_string(&required_receipt(&scenario, "process.log")?)?,
                "baseline_observation": runtime_observation(&scenario, budget)?,
            }),
        );
    }

    let exporter = evidence_root.join("runtime/exporter-100k");
    let exporter_observation_path = required_receipt(&exporter, "observation.json")?;
    let exporter_log = required_receipt(&exporter, "process.log")?;
    let exporter_observation: serde_json::Value =
        serde_json::from_reader(File::open(&exporter_observation_path)?)?;
    exact_object_keys(
        &exporter_observation,
        "exporter observation",
        &[
            "path",
            "iterations",
            "corpus_records",
            "sample_repetitions",
            "processed_records",
            "retained_artifact_records",
            "allocation_count",
            "allocated_bytes",
            "allocation_count_per_request",
            "allocated_bytes_per_request",
            "exporter_interval_nanoseconds",
            "exporter_nanoseconds_per_record",
            "corpus_blake3",
            "observable_kind",
            "observable_policy_blake3",
            "experiment_identity_blake3",
            "attempt_ordinal",
            "pair_id",
            "member",
            "repetition_receipts_blake3",
            "retained_raw_observable_path",
            "retained_comparison_observable_path",
            "retained_provenance_receipt_path",
            "repetition_receipts_path",
            "repetition_receipts",
            "ttft_p50",
            "ttft_p90",
            "ttft_p99",
            "itl_p50",
            "itl_p90",
            "itl_p99",
        ],
    )?;
    let exact_exporter_path =
        |field: &str, expected: &str| -> Result<PathBuf, Box<dyn std::error::Error>> {
            if exporter_observation
                .get(field)
                .and_then(serde_json::Value::as_str)
                != Some(expected)
            {
                return Err(format!("exporter observation `{field}` path mismatch").into());
            }
            Ok(exporter.join(expected))
        };
    let repetition_path = exact_exporter_path(
        "repetition_receipts_path",
        "observable/member-repetition-receipts.json",
    )?;
    let repetition_file_bytes = fs::read(&repetition_path)?;
    let raw_repetitions = exporter_observation["repetition_receipts"]
        .as_array()
        .ok_or("exporter observation lacks repetition_receipts")?;
    let mut canonical_repetition_bytes = serde_json::to_vec(raw_repetitions)?;
    canonical_repetition_bytes.push(b'\n');
    let canonical_repetition_blake3 =
        format!("blake3:{}", blake3::hash(&canonical_repetition_bytes));
    if repetition_file_bytes != canonical_repetition_bytes
        || exporter_observation["repetition_receipts_blake3"].as_str()
            != Some(canonical_repetition_blake3.as_str())
    {
        return Err("exporter repetition vector file/digest mismatch".into());
    }
    let retained_raw = exact_exporter_path(
        "retained_raw_observable_path",
        "observable/retained-raw-observable.json",
    )?;
    let retained_comparison = exact_exporter_path(
        "retained_comparison_observable_path",
        "observable/retained-comparison-observable.json",
    )?;
    let retained_provenance = exact_exporter_path(
        "retained_provenance_receipt_path",
        "observable/retained-provenance-receipt.json",
    )?;
    let retained = raw_repetitions
        .last()
        .and_then(serde_json::Value::as_object)
        .ok_or("exporter repetition vector is empty")?;
    if digest_string(&retained_raw)?
        != retained["raw_observable_blake3"]
            .as_str()
            .ok_or("retained receipt lacks raw observable digest")?
        || digest_string(&retained_comparison)?
            != retained["comparison_observable_blake3"]
                .as_str()
                .ok_or("retained receipt lacks comparison observable digest")?
        || digest_string(&retained_provenance)?
            != retained["provenance_receipt_blake3"]
                .as_str()
                .ok_or("retained receipt lacks provenance digest")?
        || fs::read(&retained_raw)? != fs::read(&retained_comparison)?
        || fs::read(&retained_provenance)? != b"[]\n"
    {
        return Err("retained exporter observable evidence mismatch".into());
    }
    let raw_tree_bytes = snapshot_regular_file(&retained_raw)?;
    let raw_entries = parse_artifact_tree_observable(&raw_tree_bytes)?;
    if raw_entries.len() != 1 {
        return Err("retained exporter artifact tree must contain one entry".into());
    }
    let raw_entry = &raw_entries[0];
    let retained_output = required_receipt(&exporter, "exporter-corpus.jsonl")?;
    let retained_output_blake3 = digest_string(&retained_output)?;
    if raw_entry.path != "exporter-corpus.jsonl"
        || raw_entry.kind != ArtifactTreeKind::RegularFile
        || raw_entry.length != fs::metadata(&retained_output)?.len()
        || raw_entry.blake3 != retained_output_blake3
    {
        return Err("retained exporter artifact tree does not bind retained output".into());
    }
    let duration_ns = exporter_observation["exporter_interval_nanoseconds"]
        .as_u64()
        .ok_or("exporter observation lacks exporter_interval_nanoseconds")?;
    runtime.insert(
        "exporter_100k".to_owned(),
        serde_json::json!({
            "artifact_digest": digest_string(&exporter_observation_path)?,
            "process_log_digest": digest_string(&exporter_log)?,
            "baseline_observation": {
                "duration_seconds": duration_ns as f64 / 1_000_000_000.0,
                "exporter_nanoseconds_per_record": exporter_observation["exporter_nanoseconds_per_record"],
                "allocation_count_per_successful_request": exporter_observation["allocation_count_per_request"],
                "allocated_bytes_per_successful_request": exporter_observation["allocated_bytes_per_request"],
                "ttft_p50": exporter_observation["ttft_p50"],
                "ttft_p90": exporter_observation["ttft_p90"],
                "ttft_p99": exporter_observation["ttft_p99"],
                "itl_p50": exporter_observation["itl_p50"],
                "itl_p90": exporter_observation["itl_p90"],
                "itl_p99": exporter_observation["itl_p99"],
            }
        }),
    );

    let allocation_log = required_receipt(&evidence_root.join("probes"), "allocation-probes.log")?;
    let prefix = "AIPERF_ALLOCATION_SAMPLE ";
    let mut allocation_samples = serde_json::Map::new();
    for line in fs::read_to_string(&allocation_log)?.lines() {
        if let Some(json) = line.strip_prefix(prefix) {
            let sample: serde_json::Value = serde_json::from_str(json)?;
            let name = sample["path"]
                .as_str()
                .ok_or("allocation sample lacks path")?;
            if allocation_samples
                .insert(
                    name.to_owned(),
                    normalized_allocation_sample(&sample, false)?,
                )
                .is_some()
            {
                return Err(format!("duplicate allocation sample `{name}`").into());
            }
        }
    }
    allocation_samples.insert(
        "exporter_capture".to_owned(),
        normalized_allocation_sample(&exporter_observation, true)?,
    );
    let expected_allocations = [
        "endpoint_preparation",
        "endpoint_formatting",
        "transport_dispatch",
        "response_reduction",
        "full_successful_request",
        "exporter_capture",
    ]
    .into_iter()
    .collect::<BTreeSet<_>>();
    if allocation_samples
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>()
        != expected_allocations
    {
        return Err("allocation sample set mismatch".into());
    }

    let repetitions = raw_repetitions.clone();
    let mut repetition_bytes = serde_json::to_vec(&repetitions)?;
    repetition_bytes.push(b'\n');
    let source_fact =
        |path: &Path, relative: &str| -> Result<serde_json::Value, Box<dyn std::error::Error>> {
            let (bytes, _) = digest_file(path)?;
            Ok(serde_json::json!({
                "path": relative,
                "bytes": bytes,
                "blake3": digest_string(path)?,
            }))
        };
    let results = MeasurementResults {
        schema_version: 1,
        generation: generation.to_owned(),
        build_commands: serde_json::Value::Object(builds),
        runtime_measurements: serde_json::Value::Object(runtime),
        allocation_probe: serde_json::Value::Object(allocation_samples),
        allocation_sources: serde_json::json!({
            "allocation_log": source_fact(&allocation_log, "probes/allocation-probes.log")?,
            "exporter_log": source_fact(&exporter_log, "runtime/exporter-100k/process.log")?,
            "exporter_observation": source_fact(&exporter_observation_path, "runtime/exporter-100k/observation.json")?,
        }),
        exporter_observable_policy_blake3: digest_string(&required_receipt(
            &evidence_root.join("identity"),
            "exporter-observable-policy.json",
        )?)?,
        exporter_corpus_blake3: digest_string(&required_receipt(
            &evidence_root.join("identity"),
            "exporter-static-calibration-corpus.json",
        )?)?,
        exporter_build_receipt_blake3: digest_string(&required_receipt(
            &evidence_root.join("builds/engine"),
            "summary.txt",
        )?)?,
        exporter_repetition_receipts_blake3: format!("blake3:{}", blake3::hash(&repetition_bytes)),
        exporter_repetition_receipts: serde_json::Value::Array(repetitions),
    };
    let experiment_identity_blake3 = digest_string(&required_receipt(
        &evidence_root.join("identity"),
        "experiment-identity.json",
    )?)?;
    validate_measurement_results(&results, generation, &experiment_identity_blake3)?;
    let mut file = File::create(output)?;
    serde_json::to_writer_pretty(&mut file, &results)?;
    writeln!(file)?;
    Ok(())
}

fn read_measurement_results(
    identity_root: &Path,
    generation: &str,
) -> Result<MeasurementResults, Box<dyn std::error::Error>> {
    let path = required_receipt(identity_root, "measurement-results.json")?;
    let results: MeasurementResults = serde_json::from_reader(File::open(path)?)
        .map_err(|error| format!("invalid strict measurement receipt: {error}"))?;
    let experiment_identity_blake3 = digest_string(&required_receipt(
        identity_root,
        "experiment-identity.json",
    )?)?;
    validate_measurement_results(&results, generation, &experiment_identity_blake3)?;
    let policy_blake3 = digest_string(&required_receipt(
        identity_root,
        "exporter-observable-policy.json",
    )?)?;
    let corpus_blake3 = digest_string(&required_receipt(
        identity_root,
        "exporter-static-calibration-corpus.json",
    )?)?;
    if results.exporter_observable_policy_blake3 != policy_blake3
        || results.exporter_corpus_blake3 != corpus_blake3
    {
        return Err("measurement exporter pre-run policy/corpus binding mismatch".into());
    }
    Ok(results)
}

fn refresh_contract(
    mode: InventoryRefreshMode,
    generation: &str,
    inventory_path: &Path,
    receipts: &Path,
    topology_output: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    validate_generation(generation)?;
    let capture_root = (mode != InventoryRefreshMode::PreCapture).then_some(receipts);
    let identity_root = capture_root
        .map(|root| root.join("evidence/identity"))
        .unwrap_or_else(|| receipts.to_path_buf());
    let source_archive = required_receipt(&identity_root, "source-tree.tar")?;
    let baseline_lock = required_receipt(&identity_root, "baseline-Cargo.lock")?;
    let effective_lock = required_receipt(&identity_root, "Cargo.lock")?;
    let projection_list = required_receipt(&identity_root, "measurement-source-projection.txt")?;
    let projection_archive = required_receipt(&identity_root, "measurement-source-projection.tar")?;
    let effective_archive = required_receipt(&identity_root, "effective-source-tree.tar")?;
    let harness = required_receipt(&identity_root, "capture-plugin-baseline.sh")?;
    let owned_command_helper =
        required_receipt(&identity_root, "plugin-baseline-owned-command.sh")?;
    let invalidations_path = required_receipt(&identity_root, "invalidations.tsv")?;
    validate_projection_list(&projection_list)?;
    let invalidations = read_invalidations(&invalidations_path, generation)?;
    let mut contents = fs::read_to_string(inventory_path)?;
    contents = replace_top_level_scalar(
        &contents,
        "source_projection_rule",
        &source_projection_rule(&projection_list)?,
    )?;
    let mut identity = if mode == InventoryRefreshMode::PreCapture {
        identity_from_inventory(&contents)?
    } else {
        let captured = required_receipt(&identity_root, "experiment-identity.json")?;
        let identity: BTreeMap<String, serde_json::Value> =
            serde_json::from_reader(File::open(captured)?)?;
        for required in [
            "baseline_revision",
            "rustc",
            "cargo",
            "target",
            "admitted_environment_blake3",
        ] {
            if !identity
                .get(required)
                .is_some_and(serde_json::Value::is_string)
            {
                return Err(format!("captured experiment identity lacks `{required}`").into());
            }
        }
        identity
    };
    let measurements = if mode == InventoryRefreshMode::PreCapture {
        None
    } else {
        Some(read_measurement_results(&identity_root, generation)?)
    };
    if mode != InventoryRefreshMode::PreCapture {
        let captured_topology = required_receipt(&identity_root, "package-topology.json")?;
        let topology: serde_json::Value = serde_json::from_reader(File::open(captured_topology)?)?;
        if topology
            .get("schema_version")
            .and_then(serde_json::Value::as_u64)
            != Some(1)
            || topology
                .get("generation")
                .and_then(serde_json::Value::as_str)
                != Some(generation)
            || topology.get("host_commit") != identity.get("baseline_revision")
            || topology.get("rustc") != identity.get("rustc")
            || topology.get("target") != identity.get("target")
            || topology.get("cargo_profile") != identity.get("cargo_profile")
        {
            return Err("captured topology generation/profile identity mismatch".into());
        }
    }
    let compact_allocation = measurements
        .as_ref()
        .map(|results| render_compact_allocation(generation, results))
        .transpose()?;
    if let Some(results) = measurements.as_ref() {
        let builds = results
            .build_commands
            .as_object()
            .ok_or("measurement build_commands must be an object")?;
        let compared = builds
            .iter()
            .map(|(name, build)| {
                let digest = build
                    .get("artifact_digest")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| format!("measurement build `{name}` lacks artifact_digest"))?;
                Ok((name.clone(), serde_json::Value::String(digest.to_owned())))
            })
            .collect::<Result<serde_json::Map<_, _>, Box<dyn std::error::Error>>>()?;
        identity.insert(
            "compared_artifact_digests".to_owned(),
            serde_json::Value::Object(compared),
        );
        contents = replace_section(
            &contents,
            "build_commands:\n",
            "runtime_scenarios:",
            &render_yaml_section("build_commands", &results.build_commands)?,
        )?;
        let inventory_document: serde_yaml::Value = serde_yaml::from_str(&contents)?;
        let mut scenarios = serde_json::to_value(
            inventory_document
                .get("runtime_scenarios")
                .ok_or("inventory lacks runtime_scenarios")?,
        )?;
        for scenario in scenarios
            .as_array_mut()
            .ok_or("measurement runtime_scenarios must be an array")?
        {
            let scenario = scenario
                .as_object_mut()
                .ok_or("measurement runtime scenario must be an object")?;
            let name = scenario
                .get("name")
                .and_then(serde_json::Value::as_str)
                .ok_or("inventory runtime scenario lacks a name")?
                .to_owned();
            let measurement = results
                .runtime_measurements
                .get(&name)
                .and_then(serde_json::Value::as_object)
                .ok_or_else(|| format!("measurement lacks runtime scenario `{name}`"))?;
            for field in [
                "artifact_digest",
                "process_log_digest",
                "baseline_observation",
            ] {
                scenario.insert(
                    field.to_owned(),
                    measurement
                        .get(field)
                        .cloned()
                        .ok_or_else(|| format!("measurement scenario `{name}` lacks `{field}`"))?,
                );
            }
            for (field, identity_field) in [
                ("harness_blake3", "harness_blake3"),
                ("mock_server_blake3", "mock_server_blake3"),
                ("firmware", "firmware"),
                ("memory_topology", "memory_topology"),
            ] {
                scenario.insert(
                    field.to_owned(),
                    identity
                        .get(identity_field)
                        .cloned()
                        .ok_or_else(|| format!("experiment identity lacks `{identity_field}`"))?,
                );
            }
            if name == "exporter_100k" {
                for (field, value) in exporter_measurement_binding(results)? {
                    scenario.insert(field, value);
                }
            }
        }
        contents = replace_section(
            &contents,
            "runtime_scenarios:\n",
            "allocation_probe:",
            &render_yaml_section("runtime_scenarios", &scenarios)?,
        )?;
        contents = replace_section(
            &contents,
            "allocation_probe:\n",
            "artifacts:",
            &render_yaml_section("allocation_probe", &results.allocation_probe)?,
        )?;
    }
    let inventory_document: serde_yaml::Value = serde_yaml::from_str(&contents)?;
    let mut scenarios = serde_json::to_value(
        inventory_document
            .get("runtime_scenarios")
            .ok_or("inventory lacks runtime_scenarios")?,
    )?;
    let mut exporter_count = 0_usize;
    let authored_exporter = exporter_authored_contract(&identity_root)?;
    for scenario in scenarios
        .as_array_mut()
        .ok_or("inventory runtime_scenarios must be an array")?
    {
        let scenario = scenario
            .as_object_mut()
            .ok_or("inventory runtime scenario must be an object")?;
        if scenario.get("name").and_then(serde_json::Value::as_str) == Some("exporter_100k") {
            exporter_count += 1;
            for (field, value) in &authored_exporter {
                scenario.insert(field.clone(), value.clone());
            }
        }
    }
    if exporter_count != 1 {
        return Err("inventory must contain exactly one exporter_100k scenario".into());
    }
    contents = replace_section(
        &contents,
        "runtime_scenarios:\n",
        "allocation_probe:",
        &render_yaml_section("runtime_scenarios", &scenarios)?,
    )?;
    let inventory_document: serde_yaml::Value = serde_yaml::from_str(&contents)?;
    let mut allocation_probe = serde_json::to_value(
        inventory_document
            .get("allocation_probe")
            .ok_or("inventory lacks allocation_probe")?,
    )?;
    let exporter_allocation = allocation_probe
        .get_mut("exporter_capture")
        .and_then(serde_json::Value::as_object_mut)
        .ok_or("inventory allocation_probe lacks exporter_capture")?;
    for field in [
        "corpus_records",
        "sample_repetitions",
        "processed_records",
        "retained_artifact_records",
    ] {
        exporter_allocation.insert(
            field.to_owned(),
            authored_exporter
                .get(field)
                .cloned()
                .ok_or_else(|| format!("authored exporter contract lacks `{field}`"))?,
        );
    }
    contents = replace_section(
        &contents,
        "allocation_probe:\n",
        "artifacts:",
        &render_yaml_section("allocation_probe", &allocation_probe)?,
    )?;
    for (name, path) in [
        ("baseline_source_tree_blake3", &source_archive),
        ("baseline_cargo_lock_blake3", &baseline_lock),
        ("measurement_source_projection_blake3", &projection_archive),
        (
            "measurement_source_projection_list_blake3",
            &projection_list,
        ),
        ("effective_cargo_lock_blake3", &effective_lock),
        ("effective_source_tree_blake3", &effective_archive),
        ("harness_blake3", &harness),
        ("owned_command_helper_blake3", &owned_command_helper),
    ] {
        identity.insert(
            name.to_owned(),
            serde_json::Value::String(format!("blake3:{}", digest_file(path)?.1)),
        );
    }
    for (name, value) in [
        (
            "exporter_observable_policy_blake3",
            digest_string(&required_receipt(
                &identity_root,
                "exporter-observable-policy.json",
            )?)?,
        ),
        (
            "exporter_corpus_blake3",
            digest_string(&required_receipt(
                &identity_root,
                "exporter-static-calibration-corpus.json",
            )?)?,
        ),
        ("exporter_receipt_schema_version", "1".to_owned()),
        ("exporter_observable_policy_schema_version", "1".to_owned()),
        ("exporter_corpus_records", "100000".to_owned()),
        ("exporter_sample_repetitions", "16".to_owned()),
        ("exporter_processed_records", "1600000".to_owned()),
        ("exporter_retained_artifact_records", "100000".to_owned()),
        ("exporter_pair_id", "task1-static-calibration".to_owned()),
        ("exporter_member", "static".to_owned()),
        ("exporter_attempt_ordinal", "0".to_owned()),
        (
            "exporter_planned_schedule",
            "one original-static calibration member;16 sequential repetitions;no dynamic member;no sleep or padding".to_owned(),
        ),
    ] {
        identity.insert(name.to_owned(), serde_json::Value::String(value));
    }
    identity.insert(
        "canonical_inventory_digest".to_owned(),
        serde_json::Value::String(ZERO_DIGEST.to_owned()),
    );
    for key in ["rustc", "target", "cargo_profile"] {
        let value = identity
            .get(key)
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| format!("experiment identity lacks string `{key}`"))?;
        contents = replace_top_level_scalar(&contents, key, value)?;
    }
    contents = replace_section(
        &contents,
        "experiment_identity_json: |\n",
        "experiment_identity_digest:",
        &render_identity(&identity)?,
    )?;
    let harness_digest = format!("blake3:{}", digest_file(&harness)?.1);
    contents = replace_inventory_digest_field(&contents, "harness_blake3", &harness_digest)?;
    let mut ledger = String::from("invalid_capture_attempts:\n");
    for invalidation in invalidations {
        ledger.push_str(&format!(
            "  - generation: {}\n    status: {}\n    reason: {}\n",
            invalidation.generation,
            invalidation.status,
            serde_json::to_string(&invalidation.reason)?
        ));
    }
    contents = replace_section(
        &contents,
        "invalid_capture_attempts:\n",
        "raw_samples:",
        &ledger,
    )?;
    let evidence_facts = match mode {
        InventoryRefreshMode::PreCapture => None,
        InventoryRefreshMode::PostCapture => Some(staged_evidence_facts(
            capture_root.ok_or("post-capture root is absent")?,
            generation,
        )?),
    };
    let admission = match mode {
        InventoryRefreshMode::PreCapture => "pre_capture".to_owned(),
        InventoryRefreshMode::PostCapture => format!("locally_authenticated_{generation}"),
    };
    let raw_samples = render_raw_samples(generation, &admission, evidence_facts.as_ref())?;
    contents = replace_section(
        &contents,
        "raw_samples:\n",
        "canonical_inventory_digest:",
        &raw_samples,
    )?;
    let (harness_bytes, _) = digest_file(&harness)?;
    contents = upsert_artifact_line(
        &contents,
        "capture_harness",
        &format!(
            "capture_harness: {{path: rust/scripts/capture-plugin-baseline.sh, bytes: {harness_bytes}, blake3: {harness_digest}}}"
        ),
    )?;
    let mut pending_outputs = Vec::<(PathBuf, Vec<u8>)>::new();
    if let Some(repository) = inventory_path.ancestors().nth(3) {
        for (key, relative) in [
            ("readme", "artifacts/native-plugin-baseline/README.md"),
            (
                "allocation_probe",
                "artifacts/native-plugin-baseline/allocation-probe.json",
            ),
            (
                "package_topology",
                "artifacts/native-plugin-baseline/package-topology.json",
            ),
        ] {
            if key == "allocation_probe" && compact_allocation.is_some() {
                continue;
            }
            let path = repository.join(relative);
            if path.is_file() {
                let (bytes, digest) = digest_file(&path)?;
                contents = upsert_artifact_line(
                    &contents,
                    key,
                    &format!(
                        "{key}: {{path: {relative}, bytes: {bytes}, blake3: blake3:{digest}}}"
                    ),
                )?;
            }
        }
    }
    if let Some(bytes) = compact_allocation.as_ref() {
        let compact_root = topology_output
            .and_then(Path::parent)
            .ok_or("measured allocation output requires a compact artifact root")?;
        let digest = blake3::hash(bytes);
        contents = upsert_artifact_line(
            &contents,
            "allocation_probe",
            &format!(
                "allocation_probe: {{path: artifacts/native-plugin-baseline/allocation-probe.json, bytes: {}, blake3: blake3:{digest}}}",
                bytes.len()
            ),
        )?;
        pending_outputs.push((compact_root.join("allocation-probe.json"), bytes.clone()));
    }
    if mode == InventoryRefreshMode::PreCapture {
        if let Some(topology_output) = topology_output {
            let captured_topology = required_receipt(&identity_root, "package-topology.json")?;
            let (topology_bytes, topology_digest) = digest_file(&captured_topology)?;
            contents = upsert_artifact_line(
                &contents,
                "package_topology",
                &format!(
                    "package_topology: {{path: artifacts/native-plugin-baseline/package-topology.json, bytes: {topology_bytes}, blake3: blake3:{topology_digest}}}"
                ),
            )?;
            pending_outputs.push((topology_output.to_path_buf(), fs::read(captured_topology)?));
        }
    } else {
        let captured_topology = required_receipt(&identity_root, "package-topology.json")?;
        let metadata = required_receipt(&identity_root, "cargo-metadata.json")?;
        let workspace_tree = required_receipt(&identity_root, "cargo-tree-workspace.txt")?;
        let cli_tree = required_receipt(&identity_root, "cargo-tree-cli.txt")?;
        let topology_json: serde_json::Value =
            serde_json::from_reader(File::open(&captured_topology)?)?;
        let expected_commands = serde_json::json!([
            "cargo metadata --locked --format-version 1",
            "cargo tree --locked --workspace --edges normal,build --prefix depth",
            "cargo tree --locked -p aiperf-cli --edges normal,build --prefix depth",
        ]);
        if topology_json
            .get("schema_version")
            .and_then(serde_json::Value::as_u64)
            != Some(1)
            || topology_json
                .get("generation")
                .and_then(serde_json::Value::as_str)
                != Some(generation)
            || topology_json.get("host_commit") != identity.get("baseline_revision")
            || topology_json.get("rustc") != identity.get("rustc")
            || topology_json.get("target") != identity.get("target")
            || topology_json.get("cargo_profile") != identity.get("cargo_profile")
            || topology_json.pointer("/measurement/commands") != Some(&expected_commands)
            || topology_json.pointer("/measurement/cargo_lock_blake3")
                != identity.get("effective_cargo_lock_blake3")
            || topology_json.pointer("/measurement/cargo_metadata_blake3")
                != Some(&serde_json::Value::String(format!(
                    "blake3:{}",
                    digest_file(&metadata)?.1
                )))
            || topology_json.pointer("/measurement/cargo_tree_blake3")
                != Some(&serde_json::Value::String(format!(
                    "blake3:{}",
                    digest_file(&workspace_tree)?.1
                )))
            || topology_json.pointer("/measurement/cargo_cli_tree_blake3")
                != Some(&serde_json::Value::String(format!(
                    "blake3:{}",
                    digest_file(&cli_tree)?.1
                )))
            || topology_json.pointer("/measurement/raw_metadata")
                != Some(&serde_json::Value::String(
                    "identity/cargo-metadata.json".to_owned(),
                ))
            || topology_json.pointer("/measurement/raw_tree")
                != Some(&serde_json::Value::String(
                    "identity/cargo-tree-workspace.txt".to_owned(),
                ))
            || topology_json.pointer("/measurement/raw_cli_tree")
                != Some(&serde_json::Value::String(
                    "identity/cargo-tree-cli.txt".to_owned(),
                ))
            || !topology_json
                .get("workspace_packages")
                .is_some_and(serde_json::Value::is_array)
            || !topology_json
                .get("cargo_projection")
                .is_some_and(serde_json::Value::is_array)
        {
            return Err("captured topology does not bind the captured experiment identity".into());
        }
        let topology_output =
            topology_output.ok_or("post-capture refresh requires topology output")?;
        let (topology_bytes, topology_digest) = digest_file(&captured_topology)?;
        contents = upsert_artifact_line(
            &contents,
            "package_topology",
            &format!(
                "package_topology: {{path: artifacts/native-plugin-baseline/package-topology.json, bytes: {topology_bytes}, blake3: blake3:{topology_digest}}}"
            ),
        )?;
        pending_outputs.push((topology_output.to_path_buf(), fs::read(captured_topology)?));
        let compact_root = topology_output
            .parent()
            .ok_or("topology output has no parent")?;
        let facts = evidence_facts
            .as_ref()
            .ok_or("captured evidence facts are absent")?;
        let compact_manifest = compact_root.join("evidence-manifest.json");
        pending_outputs.push((compact_manifest, facts.compact_manifest.clone()));
    }
    contents = refreshed_inventory(&contents)?;
    pending_outputs.push((inventory_path.to_path_buf(), contents.into_bytes()));
    transactional_write_all(&pending_outputs)?;
    Ok(())
}

fn command_text(program: &str, arguments: &[&str]) -> Result<String, Box<dyn std::error::Error>> {
    let output = Command::new(program).args(arguments).output()?;
    if !output.status.success() {
        return Err(format!("{program} {:?} failed with {}", arguments, output.status).into());
    }
    let bytes = if output.stdout.is_empty() {
        output.stderr
    } else {
        output.stdout
    };
    let value = String::from_utf8(bytes)?.trim().to_owned();
    if value.is_empty() {
        return Err(format!("{program} {:?} produced no identity output", arguments).into());
    }
    Ok(value)
}

fn rustc_identity() -> Result<String, Box<dyn std::error::Error>> {
    let version = command_text("rustc", &["--version"])?;
    let verbose = command_text("rustc", &["-vV"])?;
    let llvm = verbose
        .lines()
        .find_map(|line| line.strip_prefix("LLVM version: "))
        .filter(|value| !value.is_empty())
        .ok_or("rustc verbose identity lacks LLVM version")?;
    Ok(format!("{version};LLVM {llvm}"))
}

fn read_trimmed(path: impl AsRef<Path>, fallback: &str) -> String {
    fs::read_to_string(path)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| fallback.to_owned())
}

fn cpu_field(cpuinfo: &str, field: &str) -> String {
    cpuinfo
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            (name.trim() == field).then(|| value.trim().to_owned())
        })
        .unwrap_or_else(|| format!("unavailable:{field}"))
}

fn write_experiment_identity(
    output: &Path,
    identity_root: &Path,
    artifacts: &[(&str, &Path)],
    mock_server: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut identity = BTreeMap::<String, serde_json::Value>::new();
    let string = |value: String| serde_json::Value::String(value);
    let digest = |path: &Path| -> Result<String, Box<dyn std::error::Error>> {
        Ok(format!("blake3:{}", digest_file(path)?.1))
    };
    for (name, value) in [
        (
            "baseline_revision",
            "caa3ff6fcf20ffe36a7704abe16274bedadbb9fb".to_owned(),
        ),
        (
            "baseline_source_tree_blake3",
            digest(&identity_root.join("source-tree.tar"))?,
        ),
        (
            "baseline_cargo_lock_blake3",
            digest(&identity_root.join("baseline-Cargo.lock"))?,
        ),
        (
            "measurement_source_projection_blake3",
            digest(&identity_root.join("measurement-source-projection.tar"))?,
        ),
        (
            "measurement_source_projection_list_blake3",
            digest(&identity_root.join("measurement-source-projection.txt"))?,
        ),
        (
            "effective_cargo_lock_blake3",
            digest(&identity_root.join("Cargo.lock"))?,
        ),
        (
            "effective_source_tree_blake3",
            digest(&identity_root.join("effective-source-tree.tar"))?,
        ),
        (
            "harness_blake3",
            digest(&identity_root.join("capture-plugin-baseline.sh"))?,
        ),
        (
            "owned_command_helper_blake3",
            digest(&identity_root.join("plugin-baseline-owned-command.sh"))?,
        ),
        (
            "exporter_observable_policy_blake3",
            digest(&identity_root.join("exporter-observable-policy.json"))?,
        ),
        (
            "exporter_corpus_blake3",
            digest(&identity_root.join("exporter-static-calibration-corpus.json"))?,
        ),
        ("exporter_receipt_schema_version", "1".to_owned()),
        ("exporter_observable_policy_schema_version", "1".to_owned()),
        ("exporter_corpus_records", "100000".to_owned()),
        ("exporter_sample_repetitions", "16".to_owned()),
        ("exporter_processed_records", "1600000".to_owned()),
        ("exporter_retained_artifact_records", "100000".to_owned()),
        ("exporter_pair_id", "task1-static-calibration".to_owned()),
        ("exporter_member", "static".to_owned()),
        ("exporter_attempt_ordinal", "0".to_owned()),
        (
            "exporter_planned_schedule",
            "one original-static calibration member;16 sequential repetitions;no dynamic member;no sleep or padding"
                .to_owned(),
        ),
        ("mock_server_blake3", digest(mock_server)?),
        ("rustc", rustc_identity()?),
        ("cargo", command_text("cargo", &["--version"])?),
        (
            "rustc_sysroot",
            command_text("rustc", &["--print", "sysroot"])?,
        ),
        (
            "target",
            command_text("rustc", &["-vV"])?
                .lines()
                .find_map(|line| line.strip_prefix("host: "))
                .unwrap_or("unknown-target")
                .to_owned(),
        ),
        ("python", command_text("python3", &["--version"])?),
        ("tini", command_text("tini", &["--version"])?),
        ("kernel", command_text("uname", &["-srvmo"])?),
        ("cargo_profile", "release".to_owned()),
        ("cargo_build_jobs", std::env::var("CARGO_BUILD_JOBS")?),
        ("cargo_incremental", std::env::var("CARGO_INCREMENTAL")?),
        ("cargo_target_dir", std::env::var("CARGO_TARGET_DIR")?),
        ("cargo_home", std::env::var("CARGO_HOME")?),
        ("rustup_home", std::env::var("RUSTUP_HOME")?),
        (
            "feature_sets",
            "default;engine;grpc;parquet;dynosim;full".to_owned(),
        ),
        (
            "allocator_provider",
            "mimalloc 0.1.52;libmimalloc-sys 0.1.49;aiperf global allocator".to_owned(),
        ),
        (
            "affinity_isolation",
            format!(
                "cpuset={};isolated={};mock=0-3;client=4-7|4-71",
                read_trimmed("/sys/fs/cgroup/cpuset.cpus.effective", "unavailable"),
                read_trimmed("/sys/devices/system/cpu/isolated", "none")
            ),
        ),
        (
            "frequency_governor",
            read_trimmed(
                "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
                "unavailable:no-cpufreq-interface",
            ),
        ),
        (
            "firmware",
            format!(
                "{} {} {}",
                read_trimmed("/sys/class/dmi/id/bios_vendor", "unavailable"),
                read_trimmed("/sys/class/dmi/id/bios_version", "unavailable"),
                read_trimmed("/sys/class/dmi/id/bios_date", "unavailable")
            ),
        ),
    ] {
        identity.insert(name.to_owned(), string(value));
    }
    let cpuinfo = fs::read_to_string("/proc/cpuinfo")?;
    for (name, field) in [
        ("cpu_model", "model name"),
        ("cpu_family", "cpu family"),
        ("cpu_model_number", "model"),
        ("cpu_stepping", "stepping"),
        ("microcode", "microcode"),
    ] {
        identity.insert(name.to_owned(), string(cpu_field(&cpuinfo, field)));
    }
    let mut nodes = Vec::new();
    if let Ok(entries) = fs::read_dir("/sys/devices/system/node") {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name
                .strip_prefix("node")
                .is_some_and(|suffix| suffix.bytes().all(|byte| byte.is_ascii_digit()))
            {
                nodes.push(format!(
                    "{name}={}",
                    read_trimmed(entry.path().join("cpulist"), "unavailable")
                ));
            }
        }
    }
    nodes.sort();
    identity.insert(
        "memory_topology".to_owned(),
        string(format!("{} NUMA nodes;{}", nodes.len(), nodes.join(";"))),
    );
    let mut compared = serde_json::Map::new();
    for (name, path) in artifacts {
        compared.insert((*name).to_owned(), string(digest(path)?));
    }
    identity.insert(
        "compared_artifact_digests".to_owned(),
        serde_json::Value::Object(compared),
    );
    let admitted_environment = fs::read(identity_root.join("admitted-environment.txt"))?;
    identity.insert(
        "admitted_environment_blake3".to_owned(),
        string(format!("blake3:{}", blake3::hash(&admitted_environment))),
    );
    let mut file = File::create(output)?;
    serde_json::to_writer_pretty(&mut file, &identity)?;
    writeln!(file)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let Some(first) = arguments.next() else {
        return Err(
            "usage: evidence_digest [manifest ROOT | verify MANIFEST ROOT | topology HOST_COMMIT RUSTC TARGET LOCK METADATA WORKSPACE_TREE CLI_TREE | refresh-inventory INVENTORY | FILE ...]".into(),
        );
    };
    #[cfg(debug_assertions)]
    if first == "verify-staged-bundle-test-limits" {
        let generation = arguments.next().ok_or(
            "verify-staged-bundle-test-limits requires GENERATION BUNDLE MANIFEST EXTRACTION_PARENT RECEIPT",
        )?;
        let bundle = arguments.next().ok_or(
            "verify-staged-bundle-test-limits requires GENERATION BUNDLE MANIFEST EXTRACTION_PARENT RECEIPT",
        )?;
        let manifest = arguments.next().ok_or(
            "verify-staged-bundle-test-limits requires GENERATION BUNDLE MANIFEST EXTRACTION_PARENT RECEIPT",
        )?;
        let extraction_parent = arguments.next().ok_or(
            "verify-staged-bundle-test-limits requires GENERATION BUNDLE MANIFEST EXTRACTION_PARENT RECEIPT",
        )?;
        let receipt = arguments.next().ok_or(
            "verify-staged-bundle-test-limits requires GENERATION BUNDLE MANIFEST EXTRACTION_PARENT RECEIPT",
        )?;
        if arguments.next().is_some() {
            return Err("verify-staged-bundle-test-limits accepts exactly GENERATION BUNDLE MANIFEST EXTRACTION_PARENT RECEIPT".into());
        }
        return verify_staged_bundle(
            generation.to_str().ok_or("generation must be UTF-8")?,
            Path::new(&bundle),
            Path::new(&manifest),
            Path::new(&extraction_parent),
            Path::new(&receipt),
            archive_limits_from_test_environment("STAGED", STAGED_ARCHIVE_LIMITS)?,
        );
    }
    #[cfg(debug_assertions)]
    if first == "extract-captured-source-test-limits" {
        let reviewed_inventory = arguments.next().ok_or(
            "extract-captured-source-test-limits requires REVIEWED_INVENTORY CAPTURE_ROOT DESTINATION",
        )?;
        let capture_root = arguments.next().ok_or(
            "extract-captured-source-test-limits requires REVIEWED_INVENTORY CAPTURE_ROOT DESTINATION",
        )?;
        let destination = arguments.next().ok_or(
            "extract-captured-source-test-limits requires REVIEWED_INVENTORY CAPTURE_ROOT DESTINATION",
        )?;
        if arguments.next().is_some() {
            return Err("extract-captured-source-test-limits accepts exactly REVIEWED_INVENTORY CAPTURE_ROOT DESTINATION".into());
        }
        return extract_authenticated_captured_source(
            Path::new(&reviewed_inventory),
            Path::new(&capture_root),
            Path::new(&destination),
            archive_limits_from_test_environment(
                "CAPTURED_SOURCE",
                CAPTURED_SOURCE_ARCHIVE_LIMITS,
            )?,
        );
    }
    if first == "verify-staged-bundle" {
        let generation = arguments.next().ok_or(
            "verify-staged-bundle requires GENERATION BUNDLE MANIFEST EXTRACTION_PARENT RECEIPT",
        )?;
        let bundle = arguments.next().ok_or(
            "verify-staged-bundle requires GENERATION BUNDLE MANIFEST EXTRACTION_PARENT RECEIPT",
        )?;
        let manifest = arguments.next().ok_or(
            "verify-staged-bundle requires GENERATION BUNDLE MANIFEST EXTRACTION_PARENT RECEIPT",
        )?;
        let extraction_parent = arguments.next().ok_or(
            "verify-staged-bundle requires GENERATION BUNDLE MANIFEST EXTRACTION_PARENT RECEIPT",
        )?;
        let receipt = arguments.next().ok_or(
            "verify-staged-bundle requires GENERATION BUNDLE MANIFEST EXTRACTION_PARENT RECEIPT",
        )?;
        if arguments.next().is_some() {
            return Err(
                "verify-staged-bundle accepts exactly GENERATION BUNDLE MANIFEST EXTRACTION_PARENT RECEIPT".into(),
            );
        }
        return verify_staged_bundle(
            generation.to_str().ok_or("generation must be UTF-8")?,
            Path::new(&bundle),
            Path::new(&manifest),
            Path::new(&extraction_parent),
            Path::new(&receipt),
            STAGED_ARCHIVE_LIMITS,
        );
    }
    if first == "extract-captured-source" {
        let reviewed_inventory = arguments.next().ok_or(
            "extract-captured-source requires REVIEWED_INVENTORY CAPTURE_ROOT DESTINATION",
        )?;
        let capture_root = arguments.next().ok_or(
            "extract-captured-source requires REVIEWED_INVENTORY CAPTURE_ROOT DESTINATION",
        )?;
        let destination = arguments.next().ok_or(
            "extract-captured-source requires REVIEWED_INVENTORY CAPTURE_ROOT DESTINATION",
        )?;
        if arguments.next().is_some() {
            return Err("extract-captured-source accepts exactly REVIEWED_INVENTORY CAPTURE_ROOT DESTINATION".into());
        }
        return extract_authenticated_captured_source(
            Path::new(&reviewed_inventory),
            Path::new(&capture_root),
            Path::new(&destination),
            CAPTURED_SOURCE_ARCHIVE_LIMITS,
        );
    }
    if first == "manifest" {
        let root = arguments.next().ok_or("manifest requires ROOT")?;
        if arguments.next().is_some() {
            return Err("manifest accepts exactly one ROOT".into());
        }
        return manifest(Path::new(&root));
    }
    if first == "verify" {
        let manifest = arguments
            .next()
            .ok_or("verify requires MANIFEST and ROOT")?;
        let root = arguments
            .next()
            .ok_or("verify requires MANIFEST and ROOT")?;
        if arguments.next().is_some() {
            return Err("verify accepts exactly MANIFEST and ROOT".into());
        }
        return verify_manifest(Path::new(&manifest), Path::new(&root));
    }
    if first == "topology" {
        let generation = arguments.next().ok_or("topology requires GENERATION")?;
        let host_commit = arguments.next().ok_or("topology requires HOST_COMMIT")?;
        let rustc = arguments.next().ok_or("topology requires RUSTC")?;
        let target = arguments.next().ok_or("topology requires TARGET")?;
        let cargo_profile = arguments.next().ok_or("topology requires CARGO_PROFILE")?;
        let cargo_lock = arguments.next().ok_or("topology requires LOCK")?;
        let metadata = arguments.next().ok_or("topology requires METADATA")?;
        let workspace_tree = arguments.next().ok_or("topology requires WORKSPACE_TREE")?;
        let cli_tree = arguments.next().ok_or("topology requires CLI_TREE")?;
        if arguments.next().is_some() {
            return Err("topology accepts exactly nine arguments".into());
        }
        return write_topology(
            generation.to_str().ok_or("generation must be UTF-8")?,
            host_commit.to_str().ok_or("host commit must be UTF-8")?,
            rustc.to_str().ok_or("rustc must be UTF-8")?,
            target.to_str().ok_or("target must be UTF-8")?,
            cargo_profile
                .to_str()
                .ok_or("cargo profile must be UTF-8")?,
            Path::new(&cargo_lock),
            Path::new(&metadata),
            Path::new(&workspace_tree),
            Path::new(&cli_tree),
        );
    }
    if first == "normalize-cargo-receipt" {
        let kind = arguments
            .next()
            .ok_or("normalize-cargo-receipt requires KIND")?;
        let source_root = arguments
            .next()
            .ok_or("normalize-cargo-receipt requires SOURCE_ROOT")?;
        let input = arguments
            .next()
            .ok_or("normalize-cargo-receipt requires INPUT")?;
        let output = arguments
            .next()
            .ok_or("normalize-cargo-receipt requires OUTPUT")?;
        if arguments.next().is_some() {
            return Err(
                "normalize-cargo-receipt accepts exactly KIND SOURCE_ROOT INPUT OUTPUT".into(),
            );
        }
        return normalize_cargo_receipt(
            kind.to_str().ok_or("receipt kind must be UTF-8")?,
            Path::new(&source_root),
            Path::new(&input),
            Path::new(&output),
        );
    }
    if first == "refresh-inventory" {
        let inventory = arguments
            .next()
            .ok_or("refresh-inventory requires INVENTORY")?;
        if arguments.next().is_some() {
            return Err("refresh-inventory accepts exactly one INVENTORY".into());
        }
        return refresh_inventory(Path::new(&inventory));
    }
    if first == "refresh-contract" {
        let mode = arguments.next().ok_or("refresh-contract requires MODE")?;
        let generation = arguments
            .next()
            .ok_or("refresh-contract requires GENERATION")?;
        let inventory = arguments
            .next()
            .ok_or("refresh-contract requires INVENTORY")?;
        let receipts = arguments
            .next()
            .ok_or("refresh-contract requires RECEIPTS")?;
        let topology = arguments.next();
        if arguments.next().is_some() {
            return Err(
                "refresh-contract accepts MODE GENERATION INVENTORY RECEIPTS [TOPOLOGY]".into(),
            );
        }
        let mode = InventoryRefreshMode::parse(mode.to_str().ok_or("mode must be UTF-8")?)?;
        if mode != InventoryRefreshMode::PreCapture && topology.is_none() {
            return Err("post-capture requires topology output".into());
        }
        return refresh_contract(
            mode,
            generation.to_str().ok_or("generation must be UTF-8")?,
            Path::new(&inventory),
            Path::new(&receipts),
            topology.as_deref().map(Path::new),
        );
    }
    if first == "publish-baseline" {
        let mode = arguments.next().ok_or("publish-baseline requires MODE")?;
        let candidate_root = arguments
            .next()
            .ok_or("publish-baseline requires CANDIDATE_ROOT")?;
        let repository_root = arguments
            .next()
            .ok_or("publish-baseline requires REPOSITORY_ROOT")?;
        if arguments.next().is_some() {
            return Err(
                "publish-baseline accepts exactly MODE CANDIDATE_ROOT REPOSITORY_ROOT".into(),
            );
        }
        return publish_baseline_candidates(
            InventoryRefreshMode::parse(mode.to_str().ok_or("mode must be UTF-8")?)?,
            Path::new(&candidate_root),
            Path::new(&repository_root),
        );
    }
    if first == "verify-baseline-publication" {
        let mode = arguments
            .next()
            .ok_or("verify-baseline-publication requires MODE REPOSITORY_ROOT")?;
        let repository_root = arguments
            .next()
            .ok_or("verify-baseline-publication requires MODE REPOSITORY_ROOT")?;
        if arguments.next().is_some() {
            return Err("verify-baseline-publication accepts exactly MODE REPOSITORY_ROOT".into());
        }
        return verify_baseline_publication(
            InventoryRefreshMode::parse(mode.to_str().ok_or("mode must be UTF-8")?)?,
            Path::new(&repository_root),
        );
    }
    if first == "recover-baseline-publication" {
        let mode = arguments
            .next()
            .ok_or("recover-baseline-publication requires MODE REPOSITORY_ROOT")?;
        let repository_root = arguments
            .next()
            .ok_or("recover-baseline-publication requires MODE REPOSITORY_ROOT")?;
        if arguments.next().is_some() {
            return Err("recover-baseline-publication accepts exactly MODE REPOSITORY_ROOT".into());
        }
        return recover_baseline_publication(
            InventoryRefreshMode::parse(mode.to_str().ok_or("mode must be UTF-8")?)?,
            Path::new(&repository_root),
        );
    }
    if first == "measurement-results" {
        let generation = arguments
            .next()
            .ok_or("measurement-results requires GENERATION EVIDENCE_ROOT OUTPUT")?;
        let evidence_root = arguments
            .next()
            .ok_or("measurement-results requires GENERATION EVIDENCE_ROOT OUTPUT")?;
        let output = arguments
            .next()
            .ok_or("measurement-results requires GENERATION EVIDENCE_ROOT OUTPUT")?;
        if arguments.next().is_some() {
            return Err(
                "measurement-results accepts exactly GENERATION EVIDENCE_ROOT OUTPUT".into(),
            );
        }
        return write_measurement_results(
            generation.to_str().ok_or("generation must be UTF-8")?,
            Path::new(&evidence_root),
            Path::new(&output),
        );
    }
    if first == "experiment-identity" {
        let output = arguments
            .next()
            .ok_or("experiment-identity requires OUTPUT")?;
        let identity_root = arguments
            .next()
            .ok_or("experiment-identity requires IDENTITY_ROOT")?;
        let mock_server = arguments
            .next()
            .ok_or("experiment-identity requires MOCK_SERVER")?;
        let names = ["default", "engine", "grpc", "parquet", "dynosim", "full"];
        let paths = names
            .iter()
            .map(|_| {
                arguments
                    .next()
                    .ok_or("experiment-identity requires six build artifacts")
            })
            .collect::<Result<Vec<_>, _>>()?;
        if arguments.next().is_some() {
            return Err("experiment-identity accepts OUTPUT IDENTITY_ROOT MOCK_SERVER and six build artifacts".into());
        }
        let artifacts = names
            .iter()
            .zip(paths.iter())
            .map(|(name, path)| (*name, Path::new(path)))
            .collect::<Vec<_>>();
        return write_experiment_identity(
            Path::new(&output),
            Path::new(&identity_root),
            &artifacts,
            Path::new(&mock_server),
        );
    }
    for path in std::iter::once(first).chain(arguments) {
        let path = PathBuf::from(path);
        let (bytes, digest) = digest_file(&path)?;
        println!("blake3:{digest}\t{bytes}\t{}", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod archive_tests {
    use std::fs::File;
    use std::io;
    use std::path::Path;

    use flate2::Compression;
    use flate2::write::GzEncoder;
    use tar::{Builder, EntryType, Header};

    use super::{
        Manifest, ManifestFile, STAGED_ARCHIVE_LIMITS, digest_open_file,
        extract_and_verify_staged_archive, open_regular_nofollow,
    };

    fn extract(
        archive_file: &mut File,
        manifest_bytes: &[u8],
        extraction_parent: &Path,
    ) -> Result<(Manifest, u64, blake3::Hash), Box<dyn std::error::Error>> {
        let (bytes, digest) = digest_open_file(archive_file)?;
        extract_and_verify_staged_archive(
            archive_file,
            manifest_bytes,
            bytes,
            digest,
            extraction_parent,
            STAGED_ARCHIVE_LIMITS,
        )
    }

    fn append_regular(builder: &mut Builder<GzEncoder<File>>, path: &str, contents: &[u8]) {
        let mut header = Header::new_gnu();
        header.set_entry_type(EntryType::Regular);
        header.set_mode(0o644);
        header.set_size(contents.len() as u64);
        header.set_cksum();
        builder
            .append_data(&mut header, path, contents)
            .expect("regular archive member is appended");
    }

    fn archive_builder(path: &Path) -> Builder<GzEncoder<File>> {
        Builder::new(GzEncoder::new(
            File::create(path).expect("archive output is created"),
            Compression::default(),
        ))
    }

    fn finish_archive(builder: Builder<GzEncoder<File>>) {
        builder
            .into_inner()
            .expect("tar stream is finished")
            .finish()
            .expect("gzip stream is finished");
    }

    fn write_manifest(path: &Path, files: Vec<ManifestFile>) {
        serde_json::to_writer_pretty(
            File::create(path).expect("manifest output is created"),
            &Manifest {
                schema_version: 1,
                files,
            },
        )
        .expect("manifest is serialized");
    }

    #[test]
    fn staged_archive_rejects_symlink_and_hardlink_members() {
        for entry_type in [EntryType::Symlink, EntryType::Link] {
            let directory = tempfile::tempdir().expect("archive test directory is created");
            let archive = directory.path().join("archive.tar.gz");
            let manifest = directory.path().join("evidence-manifest.json");
            write_manifest(&manifest, Vec::new());
            let mut builder = archive_builder(&archive);
            append_regular(
                &mut builder,
                "evidence-manifest.json",
                &std::fs::read(&manifest).expect("manifest bytes are readable"),
            );
            let mut header = Header::new_gnu();
            header.set_entry_type(entry_type);
            header.set_mode(0o777);
            header.set_size(0);
            header
                .set_link_name("../outside")
                .expect("link target is accepted");
            header.set_cksum();
            builder
                .append_data(&mut header, "evidence/link", io::empty())
                .expect("link archive member is appended");
            finish_archive(builder);

            let mut archive_file = open_regular_nofollow(&archive).expect("archive is acquired");
            let manifest_bytes = std::fs::read(&manifest).expect("manifest is snapshotted");
            let error = extract(&mut archive_file, &manifest_bytes, directory.path())
                .err()
                .expect("non-regular archive member must be rejected");
            assert!(
                error.to_string().contains("regular file or directory"),
                "unexpected link refusal: {error}"
            );
        }
    }

    #[test]
    fn staged_archive_rejects_newline_member_names() {
        let directory = tempfile::tempdir().expect("archive test directory is created");
        let archive = directory.path().join("archive.tar.gz");
        let manifest = directory.path().join("evidence-manifest.json");
        let contents = b"newline member";
        let member = "evidence/one\nevidence/two";
        write_manifest(
            &manifest,
            vec![ManifestFile {
                path: member.to_owned(),
                bytes: contents.len() as u64,
                blake3: format!("blake3:{}", blake3::hash(contents)),
            }],
        );
        let mut builder = archive_builder(&archive);
        append_regular(&mut builder, member, contents);
        append_regular(
            &mut builder,
            "evidence-manifest.json",
            &std::fs::read(&manifest).expect("manifest bytes are readable"),
        );
        finish_archive(builder);

        let mut archive_file = open_regular_nofollow(&archive).expect("archive is acquired");
        let manifest_bytes = std::fs::read(&manifest).expect("manifest is snapshotted");
        let error = extract(&mut archive_file, &manifest_bytes, directory.path())
            .err()
            .expect("newline archive member must be rejected");
        assert!(
            error.to_string().contains("unsafe member"),
            "unexpected newline refusal: {error}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn staged_evidence_acquisition_rejects_symlinks() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().expect("acquisition test directory is created");
        let target = directory.path().join("target.tar.gz");
        let alias = directory.path().join("download.tar.gz");
        std::fs::write(&target, b"archive bytes").expect("target bytes are written");
        symlink(&target, &alias).expect("download symlink is created");
        let error =
            open_regular_nofollow(&alias).expect_err("symlinked staged evidence must be refused");
        assert!(error.to_string().contains("no-follow regular file"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TASK1_EXPORTER_POLICY_JCS: &[u8] = b"{\"mode\":\"static_calibration\",\"receiver_transport_fields_removed\":[],\"scenarios\":[{\"allows_empty\":false,\"observable_kind\":\"artifact_tree\",\"provenance_slots\":[],\"scenario_id\":\"exporter_100k\"}],\"schema_version\":1}\n";

    fn write_manifest(path: &Path, files: Vec<ManifestFile>) {
        serde_json::to_writer_pretty(
            File::create(path).expect("create manifest fixture"),
            &Manifest {
                schema_version: 1,
                files,
            },
        )
        .expect("write manifest fixture");
    }

    #[test]
    fn exporter_policy_accepts_the_exact_normative_task1_jcs_vector() {
        let directory = tempfile::tempdir().expect("policy fixture directory");
        fs::write(
            directory.path().join("exporter-observable-policy.json"),
            TASK1_EXPORTER_POLICY_JCS,
        )
        .expect("policy fixture is written");
        fs::write(
            directory
                .path()
                .join("exporter-static-calibration-corpus.json"),
            include_bytes!("../../../benchmarks/exporter-static-calibration-corpus.json"),
        )
        .expect("corpus fixture is written");

        exporter_authored_contract(directory.path())
            .expect("the exact normative Task-1 policy must validate");
    }

    #[test]
    fn checked_in_exporter_policy_is_the_exact_normative_task1_jcs_vector() {
        assert_eq!(
            include_bytes!("../../../benchmarks/exporter-observable-policy.json"),
            TASK1_EXPORTER_POLICY_JCS
        );
    }

    #[test]
    fn exporter_policy_rejects_structural_order_and_canonical_mutations() {
        let mutations = [
            (
                "duplicate field",
                "{\"mode\":\"static_calibration\",\"mode\":\"static_calibration\",\"receiver_transport_fields_removed\":[],\"scenarios\":[],\"schema_version\":1}\n",
            ),
            (
                "unknown field",
                "{\"extra\":0,\"mode\":\"static_calibration\",\"receiver_transport_fields_removed\":[],\"scenarios\":[],\"schema_version\":1}\n",
            ),
            (
                "missing field",
                "{\"receiver_transport_fields_removed\":[],\"scenarios\":[],\"schema_version\":1}\n",
            ),
            (
                "noncanonical key order",
                "{\"schema_version\":1,\"mode\":\"static_calibration\",\"receiver_transport_fields_removed\":[],\"scenarios\":[]}\n",
            ),
            (
                "noncanonical whitespace",
                "{\"mode\": \"static_calibration\",\"receiver_transport_fields_removed\":[],\"scenarios\":[],\"schema_version\":1}\n",
            ),
            (
                "unordered scenario array",
                "{\"mode\":\"static_calibration\",\"receiver_transport_fields_removed\":[],\"scenarios\":[{\"allows_empty\":false,\"observable_kind\":\"captured_stream\",\"provenance_slots\":[],\"scenario_id\":\"z\"},{\"allows_empty\":false,\"observable_kind\":\"captured_stream\",\"provenance_slots\":[],\"scenario_id\":\"a\"}],\"schema_version\":1}\n",
            ),
            (
                "unordered metadata-key array",
                "{\"mode\":\"static_calibration\",\"receiver_transport_fields_removed\":[{\"keys\":[\"z\",\"a\"],\"protocol\":\"otel_http_v1\"}],\"scenarios\":[],\"schema_version\":1}\n",
            ),
            (
                "static slot with dynamic expected value",
                "{\"mode\":\"static_calibration\",\"receiver_transport_fields_removed\":[],\"scenarios\":[{\"allows_empty\":false,\"observable_kind\":\"artifact_tree\",\"provenance_slots\":[{\"dynamic_expected\":{\"encoding\":\"canonical_json\",\"value\":\"dynamic\"},\"locator\":{\"kind\":\"json_pointer\",\"pointer\":\"/digest\"},\"output_selector\":{\"kind\":\"artifact_content\",\"path\":\"out.json\"},\"replacement\":{\"encoding\":\"canonical_json\",\"value\":\"replacement\"},\"slot_id\":\"lock\",\"static_expected\":{\"encoding\":\"canonical_json\",\"value\":\"static\"}}],\"scenario_id\":\"exporter\"}],\"schema_version\":1}\n",
            ),
        ];

        for (name, bytes) in mutations {
            let authenticated_receiver_protocols = if name == "unordered metadata-key array" {
                BTreeSet::from(["otel_http_v1"])
            } else {
                BTreeSet::new()
            };
            let error = match parse_exporter_observable_policy(
                bytes.as_bytes(),
                &authenticated_receiver_protocols,
            ) {
                Ok(_) => panic!("accepted policy mutation: {name}"),
                Err(error) => error,
            };
            if name == "unordered metadata-key array" {
                assert_eq!(
                    error.to_string(),
                    "exporter policy metadata keys must be lower-case, sorted, and unique"
                );
            }
        }
    }

    #[test]
    fn exporter_policy_rejects_a_selector_from_the_wrong_observable_class() {
        let bytes = b"{\"mode\":\"static_calibration\",\"receiver_transport_fields_removed\":[],\"scenarios\":[{\"allows_empty\":false,\"observable_kind\":\"artifact_tree\",\"provenance_slots\":[{\"locator\":{\"kind\":\"json_pointer\",\"pointer\":\"/digest\"},\"output_selector\":{\"kind\":\"captured_stream\"},\"replacement\":{\"encoding\":\"canonical_json\",\"value\":\"replacement\"},\"slot_id\":\"lock\",\"static_expected\":{\"encoding\":\"canonical_json\",\"value\":\"static\"}}],\"scenario_id\":\"exporter\"}],\"schema_version\":1}\n";

        assert!(parse_exporter_observable_policy(bytes, &BTreeSet::new()).is_err());
    }

    #[test]
    fn exporter_policy_accepts_normative_rfc8785_numbers_and_utf16_key_order() {
        let bytes = "{\"mode\":\"static_calibration\",\"receiver_transport_fields_removed\":[],\"scenarios\":[{\"allows_empty\":false,\"observable_kind\":\"artifact_tree\",\"provenance_slots\":[{\"locator\":{\"kind\":\"json_pointer\",\"pointer\":\"/identity\"},\"output_selector\":{\"kind\":\"artifact_content\",\"path\":\"out.json\"},\"replacement\":{\"encoding\":\"canonical_json\",\"value\":{\"𐀀\":0,\"\":1e+30}},\"slot_id\":\"identity\",\"static_expected\":{\"encoding\":\"canonical_json\",\"value\":{\"𐀀\":0,\"\":1e+30}}}],\"scenario_id\":\"exporter\"}],\"schema_version\":1}\n";

        parse_exporter_observable_policy(bytes.as_bytes(), &BTreeSet::new())
            .expect("the literal RFC 8785 counterexample must validate");

        for noncanonical in [
            bytes.replace("\"𐀀\":0", "\"𐀀\":-0.0"),
            bytes.replace("\"𐀀\":0,\"\":1e+30", "\"\":1e+30,\"𐀀\":0"),
            bytes.replace("\"𐀀\":0", "\"𐀀\":9007199254740993"),
            bytes.replace("\"𐀀\":0", "\"𐀀\":1e-07"),
            bytes.replace("\"𐀀\":0", "\"𐀀\":0,\"𐀀\":1"),
        ] {
            assert!(
                parse_exporter_observable_policy(noncanonical.as_bytes(), &BTreeSet::new())
                    .is_err(),
                "noncanonical RFC 8785 mutation was accepted"
            );
        }
    }

    #[test]
    fn exporter_policy_rejects_removal_protocol_absent_from_authenticated_receiver_set() {
        const USED_ONLY: &[u8] = b"{\"mode\":\"static_calibration\",\"receiver_transport_fields_removed\":[{\"keys\":[\"date\"],\"protocol\":\"otel_http_v1\"}],\"scenarios\":[{\"allows_empty\":false,\"observable_kind\":\"receiver_transcript\",\"provenance_slots\":[],\"scenario_id\":\"receiver\"}],\"schema_version\":1}\n";
        const USED_PLUS_EXTRA: &[u8] = b"{\"mode\":\"static_calibration\",\"receiver_transport_fields_removed\":[{\"keys\":[\"date\"],\"protocol\":\"otel_http_v1\"},{\"keys\":[\"x-amzn-requestid\"],\"protocol\":\"sagemaker_eventstream\"}],\"scenarios\":[{\"allows_empty\":false,\"observable_kind\":\"receiver_transcript\",\"provenance_slots\":[],\"scenario_id\":\"receiver\"}],\"schema_version\":1}\n";
        let authenticated_receiver_protocols = BTreeSet::from(["otel_http_v1"]);

        parse_exporter_observable_policy(USED_ONLY, &authenticated_receiver_protocols)
            .expect("removal for the authenticated receiver protocol must validate");
        assert!(
            parse_exporter_observable_policy(USED_PLUS_EXTRA, &authenticated_receiver_protocols)
                .is_err(),
            "removal for a protocol absent from the authenticated receiver set was accepted"
        );
    }

    #[test]
    fn exporter_policy_rejects_unused_transport_and_overlapping_output_slots() {
        let cases = [
            serde_json::json!({
                "mode": "static_calibration",
                "receiver_transport_fields_removed": [{"keys": ["date"], "protocol": "otel_http_v1"}],
                "scenarios": [{
                    "allows_empty": false,
                    "observable_kind": "artifact_tree",
                    "provenance_slots": [],
                    "scenario_id": "artifact",
                }],
                "schema_version": 1,
            }),
            serde_json::json!({
                "mode": "static_calibration",
                "receiver_transport_fields_removed": [],
                "scenarios": [{
                    "allows_empty": false,
                    "observable_kind": "captured_stream",
                    "provenance_slots": [
                        {
                            "locator": {"kind": "byte_range", "length": 4, "offset": 0},
                            "output_selector": {"kind": "captured_stream"},
                            "replacement": {"encoding": "hex_bytes", "value": "00"},
                            "slot_id": "first",
                            "static_expected": {"encoding": "hex_bytes", "value": "00"},
                        },
                        {
                            "locator": {"kind": "byte_range", "length": 2, "offset": 3},
                            "output_selector": {"kind": "captured_stream"},
                            "replacement": {"encoding": "hex_bytes", "value": "00"},
                            "slot_id": "second",
                            "static_expected": {"encoding": "hex_bytes", "value": "00"},
                        },
                    ],
                    "scenario_id": "stream",
                }],
                "schema_version": 1,
            }),
            serde_json::json!({
                "mode": "static_calibration",
                "receiver_transport_fields_removed": [],
                "scenarios": [{
                    "allows_empty": false,
                    "observable_kind": "artifact_tree",
                    "provenance_slots": [
                        {
                            "locator": {"kind": "json_pointer", "pointer": "/a"},
                            "output_selector": {"kind": "artifact_content", "path": "out.json"},
                            "replacement": {"encoding": "canonical_json", "value": 0},
                            "slot_id": "ancestor",
                            "static_expected": {"encoding": "canonical_json", "value": 0},
                        },
                        {
                            "locator": {"kind": "json_pointer", "pointer": "/a/b"},
                            "output_selector": {"kind": "artifact_content", "path": "out.json"},
                            "replacement": {"encoding": "canonical_json", "value": 0},
                            "slot_id": "descendant",
                            "static_expected": {"encoding": "canonical_json", "value": 0},
                        },
                    ],
                    "scenario_id": "json",
                }],
                "schema_version": 1,
            }),
            serde_json::json!({
                "mode": "static_calibration",
                "receiver_transport_fields_removed": [],
                "scenarios": [{
                    "allows_empty": false,
                    "observable_kind": "captured_stream",
                    "provenance_slots": [
                        {
                            "locator": {"kind": "whole_output"},
                            "output_selector": {"kind": "captured_stream"},
                            "replacement": {"encoding": "hex_bytes", "value": "00"},
                            "slot_id": "all",
                            "static_expected": {"encoding": "hex_bytes", "value": "00"},
                        },
                        {
                            "locator": {"kind": "byte_range", "length": 1, "offset": 0},
                            "output_selector": {"kind": "captured_stream"},
                            "replacement": {"encoding": "hex_bytes", "value": "00"},
                            "slot_id": "part",
                            "static_expected": {"encoding": "hex_bytes", "value": "00"},
                        },
                    ],
                    "scenario_id": "whole",
                }],
                "schema_version": 1,
            }),
        ];

        for (index, case) in cases.into_iter().enumerate() {
            let policy: ExporterObservablePolicyV1 =
                serde_json::from_value(case).expect("counterexample policy parses");
            let authenticated_receiver_protocols = if index == 0 {
                BTreeSet::from(["otel_http_v1"])
            } else {
                BTreeSet::new()
            };
            let error = match validate_exporter_policy(&policy, &authenticated_receiver_protocols) {
                Ok(()) => panic!("structurally invalid policy was accepted"),
                Err(error) => error,
            };
            if index == 0 {
                assert_eq!(
                    error.to_string(),
                    "exporter policy contains a transport removal unused by every receiver scenario"
                );
            }
        }
    }

    #[test]
    fn artifact_tree_requires_the_exact_jcs_vector_and_lower_case_digest() {
        const CANONICAL: &[u8] = b"[{\"blake3\":\"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\",\"kind\":\"regular_file\",\"length\":3,\"path\":\"out.txt\"}]\n";
        parse_artifact_tree_observable(CANONICAL).expect("literal JCS artifact tree validates");

        for mutation in [
            b"[{\"path\":\"out.txt\",\"kind\":\"regular_file\",\"length\":3,\"blake3\":\"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"}]\n".as_slice(),
            b"[{\"blake3\":\"blake3:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\",\"kind\":\"regular_file\",\"length\":3,\"path\":\"out.txt\"}]\n".as_slice(),
            b"[{\"blake3\":\"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\",\"kind\":\"regular_file\",\"length\":3,\"path\":\"out.txt\",\"path\":\"other.txt\"}]\n".as_slice(),
            b"[{\"blake3\":\"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\",\"kind\":\"regular_file\",\"length\":3,\"path\":\"out.txt\",\"unknown\":0}]\n".as_slice(),
        ] {
            assert!(parse_artifact_tree_observable(mutation).is_err());
        }
    }

    #[test]
    fn transactional_write_rejects_duplicate_destinations_before_staging() {
        let directory = tempfile::tempdir().expect("duplicate transaction fixture root");
        let destination = directory.path().join("output.txt");
        let error = transactional_write_all(&[
            (destination.clone(), b"first".to_vec()),
            (destination.clone(), b"second".to_vec()),
        ])
        .expect_err("duplicate destination must fail before staging");
        assert!(error.to_string().contains("duplicate transactional output"));
        assert!(!destination.exists());
        assert_eq!(
            fs::read_dir(directory.path())
                .expect("fixture root remains readable")
                .count(),
            0
        );
    }

    #[test]
    fn verification_rejects_changed_and_unlisted_files() {
        let directory = tempfile::tempdir().expect("temporary evidence directory");
        let root = directory.path().join("evidence");
        fs::create_dir(&root).expect("create evidence root");
        fs::write(root.join("sample.txt"), b"frozen").expect("write evidence sample");
        let manifest_path = directory.path().join("manifest.json");
        write_manifest(
            &manifest_path,
            vec![ManifestFile {
                path: "sample.txt".to_owned(),
                bytes: 6,
                blake3: format!("blake3:{}", blake3::hash(b"frozen")),
            }],
        );

        verify_manifest(&manifest_path, &root).expect("unaltered evidence verifies");
        fs::write(root.join("sample.txt"), b"changed").expect("mutate evidence sample");
        assert!(verify_manifest(&manifest_path, &root).is_err());
        fs::write(root.join("sample.txt"), b"frozen").expect("restore evidence sample");
        fs::write(root.join("extra.txt"), b"extra").expect("add unlisted evidence");
        assert!(verify_manifest(&manifest_path, &root).is_err());
    }

    #[test]
    fn verification_rejects_parent_paths() {
        let directory = tempfile::tempdir().expect("temporary evidence directory");
        let root = directory.path().join("evidence");
        fs::create_dir(&root).expect("create evidence root");
        let manifest_path = directory.path().join("manifest.json");
        write_manifest(
            &manifest_path,
            vec![ManifestFile {
                path: "../outside".to_owned(),
                bytes: 0,
                blake3: format!("blake3:{}", blake3::hash(b"")),
            }],
        );

        assert!(verify_manifest(&manifest_path, &root).is_err());
    }

    #[test]
    fn command_identity_uses_stderr_when_successful_tool_writes_no_stdout() {
        assert_eq!(
            command_text("sh", &["-c", "printf 'stderr-version' >&2"])
                .expect("successful stderr-only identity command is captured"),
            "stderr-version"
        );
    }

    #[test]
    fn rustc_identity_includes_the_llvm_toolchain_component() {
        let identity = rustc_identity().expect("rustc identity is captured");
        assert!(identity.starts_with("rustc "));
        assert!(identity.contains(";LLVM "));
    }

    #[test]
    fn cargo_receipt_normalization_removes_materialization_root_identity() {
        let directory = tempfile::tempdir().expect("normalization test directory is created");
        let first = directory.path().join("first.json");
        let second = directory.path().join("second.json");
        let first_output = directory.path().join("first-normalized.json");
        let second_output = directory.path().join("second-normalized.json");
        fs::write(&first, r#"{"manifest_path":"/tmp/first/rust/Cargo.toml"}"#)
            .expect("first receipt is written");
        fs::write(
            &second,
            r#"{"manifest_path":"/tmp/second/rust/Cargo.toml"}"#,
        )
        .expect("second receipt is written");
        normalize_cargo_receipt("metadata", Path::new("/tmp/first"), &first, &first_output)
            .expect("first receipt is normalized");
        normalize_cargo_receipt(
            "metadata",
            Path::new("/tmp/second"),
            &second,
            &second_output,
        )
        .expect("second receipt is normalized");
        assert_eq!(
            fs::read(first_output).expect("first normalized bytes"),
            fs::read(second_output).expect("second normalized bytes")
        );
    }
}
