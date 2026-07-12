// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Factory-owned worker launch attestation and process-tree isolation.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fmt::{self, Display};
use std::fs::File;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::canonical::{is_sha256, sha256_hex};
use crate::provider::{
    EvaluationDistributionDescriptor, EvaluationProviderError, EvaluatorIsolationRequirements,
    ProviderLaunchContext,
};
use crate::provider_protocol::{EvaluationDistributionId, ScopedProxyBinding};

/// One file in the immutable worker launch closure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchClosureFile {
    /// Absolute factory-owned path.
    pub path: PathBuf,
    /// Expected raw file SHA-256.
    pub artifact_content_sha256: String,
}

/// Immutable factory-owned worker launch recipe.
#[derive(Debug, Clone)]
pub struct AttestedWorkerLaunch {
    /// Registered distribution selected by this recipe.
    pub distribution_id: EvaluationDistributionId,
    /// Absolute executable path; never authored run configuration.
    pub program: PathBuf,
    /// Literal argument vector; no shell expansion.
    pub args: Vec<OsString>,
    /// Clean environment closure.
    pub environment: BTreeMap<OsString, OsString>,
    /// Absolute contained working directory.
    pub current_dir: PathBuf,
    /// Root exposed read-only to the isolated worker.
    pub worker_root: PathBuf,
    /// Complete immutable executable/source/lock closure.
    pub closure: Vec<LaunchClosureFile>,
}

impl AttestedWorkerLaunch {
    /// Validate path containment, unique closure entries, and no secret environment keys.
    pub fn validate(&self) -> Result<(), EvaluationProviderError> {
        if !self.program.is_absolute()
            || !self.current_dir.is_absolute()
            || !self.worker_root.is_absolute()
            || self.closure.is_empty()
        {
            return Err(EvaluationProviderError::Launch(
                "worker launch paths were not absolute or the closure was empty".to_string(),
            ));
        }
        for key in self.environment.keys() {
            let normalized = key.to_string_lossy().to_ascii_lowercase();
            if [
                "api_key",
                "authorization",
                "aws_secret_access_key",
                "credential",
                "password",
                "token",
            ]
            .iter()
            .any(|forbidden| normalized.contains(forbidden))
            {
                return Err(EvaluationProviderError::Launch(
                    "worker environment allowlist contained a credential-shaped key".to_string(),
                ));
            }
        }
        let root = normalize_absolute(&self.worker_root)?;
        if root == Path::new("/") {
            return Err(EvaluationProviderError::Launch(
                "worker rootfs cannot expose the host filesystem root".to_string(),
            ));
        }
        if !normalize_absolute(&self.program)?.starts_with(&root)
            || !normalize_absolute(&self.current_dir)?.starts_with(&root)
        {
            return Err(EvaluationProviderError::Launch(
                "worker executable/current directory escaped the immutable worker root".to_string(),
            ));
        }
        let mut paths = BTreeSet::new();
        let mut includes_program = false;
        let normalized_program = normalize_absolute(&self.program)?;
        for entry in &self.closure {
            let normalized_entry = normalize_absolute(&entry.path)?;
            if !is_sha256(&entry.artifact_content_sha256)
                || !entry.path.is_absolute()
                || !normalized_entry.starts_with(&root)
                || !paths.insert(normalized_entry.clone())
            {
                return Err(EvaluationProviderError::Launch(
                    "worker closure contained an invalid digest, escape, or duplicate path"
                        .to_string(),
                ));
            }
            includes_program |= normalized_entry == normalized_program;
        }
        if !includes_program {
            return Err(EvaluationProviderError::Launch(
                "worker launch closure did not attest the executable".to_string(),
            ));
        }
        Ok(())
    }
}

/// Independently measured launch closure evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaunchAttestation {
    /// Selected immutable distribution.
    pub distribution_id: EvaluationDistributionId,
    /// Measured executable digest.
    pub executable_sha256: String,
    /// Deterministic digest over sorted worker-root-relative path/content pairs.
    pub launch_closure_sha256: String,
    /// Number of independently verified files.
    pub verified_files: usize,
}

/// Replaceable launch-closure attestor.
pub trait EvaluatorLaunchAttestor: Send + Sync {
    /// Hash/verify the complete selected closure before process creation.
    fn attest(
        &self,
        launch: &AttestedWorkerLaunch,
        distribution: &EvaluationDistributionDescriptor,
    ) -> Result<LaunchAttestation, EvaluationProviderError>;
}

/// SHA-256 file-closure attestor using no worker-reported evidence.
#[derive(Debug, Clone, Copy, Default)]
pub struct Sha256LaunchAttestor;

impl EvaluatorLaunchAttestor for Sha256LaunchAttestor {
    fn attest(
        &self,
        launch: &AttestedWorkerLaunch,
        distribution: &EvaluationDistributionDescriptor,
    ) -> Result<LaunchAttestation, EvaluationProviderError> {
        launch.validate()?;
        if launch.distribution_id != distribution.distribution_id {
            return Err(EvaluationProviderError::FactoryMismatch(
                "worker launch recipe selected a different registered distribution".to_string(),
            ));
        }
        let worker_root = normalize_absolute(&launch.worker_root)?;
        let declared_paths = launch
            .closure
            .iter()
            .map(|entry| normalize_absolute(&entry.path))
            .collect::<Result<BTreeSet<_>, _>>()?;
        verify_complete_closure(&worker_root, &declared_paths)?;
        let mut measured = Vec::with_capacity(launch.closure.len());
        let mut executable_sha256 = None;
        let normalized_program = normalize_absolute(&launch.program)?;
        for expected in &launch.closure {
            let metadata = std::fs::symlink_metadata(&expected.path).map_err(|error| {
                EvaluationProviderError::Launch(format!(
                    "failed to inspect launch-closure file {:?}: {error}",
                    expected.path
                ))
            })?;
            if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
                return Err(EvaluationProviderError::Launch(format!(
                    "launch-closure path {:?} was not a regular non-symlink file",
                    expected.path
                )));
            }
            let digest = hash_file(&expected.path)?;
            if digest != expected.artifact_content_sha256 {
                return Err(EvaluationProviderError::Launch(format!(
                    "launch-closure digest mismatch for {:?}",
                    expected.path
                )));
            }
            let absolute = normalize_absolute(&expected.path)?;
            if absolute == normalized_program {
                executable_sha256 = Some(digest.clone());
            }
            let relative = absolute.strip_prefix(&worker_root).map_err(|_| {
                EvaluationProviderError::Launch(
                    "launch-closure path escaped its normalized worker root".to_string(),
                )
            })?;
            if relative.as_os_str().is_empty() {
                return Err(EvaluationProviderError::Launch(
                    "launch-closure path cannot name the worker root itself".to_string(),
                ));
            }
            let relative = relative.to_str().ok_or_else(|| {
                EvaluationProviderError::Launch(
                    "launch-closure logical path was not valid UTF-8".to_string(),
                )
            })?;
            measured.push((relative.to_string(), digest));
        }
        measured.sort_by(|left, right| left.0.as_bytes().cmp(right.0.as_bytes()));
        let mut closure_bytes = Vec::new();
        for (path, digest) in &measured {
            closure_bytes.extend_from_slice(path.as_bytes());
            closure_bytes.push(0);
            closure_bytes.extend_from_slice(digest.as_bytes());
            closure_bytes.push(b'\n');
        }
        let launch_closure_sha256 = sha256_hex(&closure_bytes);
        if launch_closure_sha256 != distribution.launch_closure_sha256 {
            return Err(EvaluationProviderError::Launch(
                "measured launch closure did not match the registered distribution".to_string(),
            ));
        }
        Ok(LaunchAttestation {
            distribution_id: launch.distribution_id.clone(),
            executable_sha256: executable_sha256.expect("launch.validate requires program entry"),
            launch_closure_sha256,
            verified_files: measured.len(),
        })
    }
}

fn verify_complete_closure(
    worker_root: &Path,
    declared: &BTreeSet<PathBuf>,
) -> Result<(), EvaluationProviderError> {
    let root_metadata = std::fs::symlink_metadata(worker_root).map_err(|error| {
        EvaluationProviderError::Launch(format!("failed to inspect worker rootfs: {error}"))
    })?;
    if !root_metadata.file_type().is_dir() || root_metadata.file_type().is_symlink() {
        return Err(EvaluationProviderError::Launch(
            "worker rootfs was not a regular non-symlink directory".to_string(),
        ));
    }
    let mut pending = vec![worker_root.to_path_buf()];
    let mut discovered = BTreeSet::new();
    while let Some(directory) = pending.pop() {
        let entries = std::fs::read_dir(&directory).map_err(|error| {
            EvaluationProviderError::Launch(format!(
                "failed to enumerate worker rootfs directory {:?}: {error}",
                directory
            ))
        })?;
        for entry in entries {
            let entry = entry.map_err(|error| {
                EvaluationProviderError::Launch(format!(
                    "failed to enumerate worker rootfs entry: {error}"
                ))
            })?;
            let path = normalize_absolute(&entry.path())?;
            let metadata = std::fs::symlink_metadata(&path).map_err(|error| {
                EvaluationProviderError::Launch(format!(
                    "failed to inspect worker rootfs entry {:?}: {error}",
                    path
                ))
            })?;
            if metadata.file_type().is_dir() && !metadata.file_type().is_symlink() {
                pending.push(path);
            } else if metadata.file_type().is_file() && !metadata.file_type().is_symlink() {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::MetadataExt as _;
                    if metadata.nlink() != 1 {
                        return Err(EvaluationProviderError::Launch(format!(
                            "worker rootfs file {:?} was a hard link",
                            path
                        )));
                    }
                }
                discovered.insert(path);
            } else {
                return Err(EvaluationProviderError::Launch(format!(
                    "worker rootfs entry {:?} was a symlink or special file",
                    path
                )));
            }
        }
    }
    if &discovered != declared {
        return Err(EvaluationProviderError::Launch(
            "worker rootfs files did not exactly match the declared launch closure".to_string(),
        ));
    }
    Ok(())
}

/// Hard resource ceilings applied to the complete evaluator process tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluatorResourceLimits {
    /// Address-space bytes.
    pub address_space_bytes: u64,
    /// Maximum file bytes.
    pub file_size_bytes: u64,
    /// Maximum open descriptors.
    pub open_files: u64,
    /// Maximum processes/threads.
    pub processes: u64,
    /// CPU seconds.
    pub cpu_seconds: u64,
}

impl Default for EvaluatorResourceLimits {
    fn default() -> Self {
        Self {
            address_space_bytes: 16 * 1024 * 1024 * 1024,
            file_size_bytes: 8 * 1024 * 1024 * 1024,
            open_files: 4_096,
            processes: 1_024,
            cpu_seconds: 86_400,
        }
    }
}

/// Independently inspectable evidence that a prepared launch satisfies isolation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluatorIsolationEvidence {
    /// Stable isolation implementation/profile identity.
    pub profile_id: String,
    /// Digest over implementation, immutable launcher, mounts, limits, and grant scope.
    pub proof_sha256: String,
    /// Observable properties enforced for the full process subtree.
    pub enforced: EvaluatorIsolationRequirements,
}

impl EvaluatorIsolationEvidence {
    /// Fail closed unless every mandatory outcome is proven.
    pub fn validate_strict(&self) -> Result<(), EvaluationProviderError> {
        if self.profile_id.trim().is_empty() || !is_sha256(&self.proof_sha256) {
            return Err(EvaluationProviderError::Launch(
                "evaluator isolation evidence was incomplete".to_string(),
            ));
        }
        self.enforced
            .clone()
            .validate()
            .map_err(EvaluationProviderError::registry)
    }
}

/// Fully prepared program/argv/environment after isolation lowering.
#[derive(Debug, Clone)]
pub struct PreparedEvaluatorLaunch {
    /// Isolation launcher program.
    pub program: PathBuf,
    /// Literal isolation and worker arguments.
    pub args: Vec<OsString>,
    /// Clean allowlisted environment supplied to the launcher.
    pub environment: BTreeMap<OsString, OsString>,
    /// Launcher working directory.
    pub current_dir: PathBuf,
    /// Applied resource limits.
    pub resource_limits: EvaluatorResourceLimits,
    /// Complete isolation proof.
    pub evidence: EvaluatorIsolationEvidence,
}

/// Replaceable platform implementation for the full evaluator isolation outcome.
pub trait EvaluatorIsolation: Send + Sync {
    /// Prove the platform isolation mechanism is present and immutable without
    /// starting an evaluator worker.
    fn check_available(&self) -> Result<(), EvaluationProviderError>;

    /// Prepare an isolated process-tree launch from a factory-owned recipe.
    fn prepare(
        &self,
        launch: &AttestedWorkerLaunch,
        attestation: &LaunchAttestation,
        context: &ProviderLaunchContext,
    ) -> Result<PreparedEvaluatorLaunch, EvaluationProviderError>;

    /// Verify no process in the isolated subtree remains after worker exit.
    fn verify_quiescent(
        &self,
        root_pid: u32,
    ) -> Result<IsolationQuiescenceProof, EvaluationProviderError>;
}

/// Unforgeable-at-API-level evidence returned only after isolation quiescence checks.
#[derive(Debug, Clone)]
pub struct IsolationQuiescenceProof {
    root_pid: u32,
    proof_sha256: String,
}

impl IsolationQuiescenceProof {
    /// Root process whose complete subtree was checked.
    pub fn root_pid(&self) -> u32 {
        self.root_pid
    }

    /// Digest naming the quiescence check result.
    pub fn proof_sha256(&self) -> &str {
        &self.proof_sha256
    }

    pub(crate) fn verified(root_pid: u32, proof_sha256: String) -> Self {
        Self {
            root_pid,
            proof_sha256,
        }
    }
}

/// Production Linux isolation using an independently attested Bubblewrap binary.
///
/// `--unshare-all` creates private user, PID, mount, IPC, UTS, cgroup, and
/// network namespaces. The independently materialized immutable worker rootfs
/// is mounted at `/` so its attested dynamic loader and libraries retain their
/// logical absolute paths. Only the contained staging root and an optional
/// Unix-domain proxy socket are added to that rootfs.
pub struct BubblewrapEvaluatorIsolation {
    bubblewrap: PathBuf,
    bubblewrap_sha256: String,
    limits: EvaluatorResourceLimits,
}

impl BubblewrapEvaluatorIsolation {
    /// Construct a fail-closed Bubblewrap profile with an expected binary digest.
    pub fn new(
        bubblewrap: impl Into<PathBuf>,
        bubblewrap_sha256: impl Into<String>,
        limits: EvaluatorResourceLimits,
    ) -> Result<Self, EvaluationProviderError> {
        let bubblewrap = bubblewrap.into();
        let bubblewrap_sha256 = bubblewrap_sha256.into();
        if !bubblewrap.is_absolute() || !is_sha256(&bubblewrap_sha256) {
            return Err(EvaluationProviderError::Launch(
                "Bubblewrap path/digest was not absolute and immutable".to_string(),
            ));
        }
        Ok(Self {
            bubblewrap,
            bubblewrap_sha256,
            limits,
        })
    }

    #[cfg(target_os = "linux")]
    fn verify_binary(&self) -> Result<(), EvaluationProviderError> {
        use std::os::unix::fs::PermissionsExt as _;

        let metadata = std::fs::symlink_metadata(&self.bubblewrap).map_err(|error| {
            EvaluationProviderError::Launch(format!(
                "failed to inspect Bubblewrap isolation binary: {error}"
            ))
        })?;
        if !metadata.file_type().is_file()
            || metadata.file_type().is_symlink()
            || metadata.permissions().mode() & 0o111 == 0
        {
            return Err(EvaluationProviderError::Launch(
                "Bubblewrap isolation binary was not a regular executable file".to_string(),
            ));
        }
        if hash_file(&self.bubblewrap)? != self.bubblewrap_sha256 {
            return Err(EvaluationProviderError::Launch(
                "Bubblewrap binary digest did not match registered isolation profile".to_string(),
            ));
        }
        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    fn verify_binary(&self) -> Result<(), EvaluationProviderError> {
        Err(EvaluationProviderError::Launch(
            "Bubblewrap evaluator isolation is available only on Linux".to_string(),
        ))
    }
}

impl EvaluatorIsolation for BubblewrapEvaluatorIsolation {
    fn check_available(&self) -> Result<(), EvaluationProviderError> {
        self.verify_binary()
    }

    fn prepare(
        &self,
        launch: &AttestedWorkerLaunch,
        attestation: &LaunchAttestation,
        context: &ProviderLaunchContext,
    ) -> Result<PreparedEvaluatorLaunch, EvaluationProviderError> {
        self.verify_binary()?;
        context
            .validate()
            .map_err(EvaluationProviderError::registry)?;
        let worker_root = std::fs::canonicalize(&launch.worker_root).map_err(|error| {
            EvaluationProviderError::Launch(format!("failed to resolve worker root: {error}"))
        })?;
        if worker_root == Path::new("/") {
            return Err(EvaluationProviderError::Launch(
                "worker rootfs cannot expose the host filesystem root".to_string(),
            ));
        }
        let mountpoint_identities =
            validate_rootfs_mountpoints(&worker_root, context.proxy.is_some())?;
        let staging = std::fs::canonicalize(&context.staging_dir).map_err(|error| {
            EvaluationProviderError::Launch(format!("failed to resolve staging root: {error}"))
        })?;
        let program = std::fs::canonicalize(&launch.program).map_err(|error| {
            EvaluationProviderError::Launch(format!("failed to resolve worker program: {error}"))
        })?;
        let current_dir = std::fs::canonicalize(&launch.current_dir).map_err(|error| {
            EvaluationProviderError::Launch(format!(
                "failed to resolve worker current directory: {error}"
            ))
        })?;
        let relative_program = program.strip_prefix(&worker_root).map_err(|_| {
            EvaluationProviderError::Launch("worker executable escaped worker root".to_string())
        })?;
        let relative_current = current_dir.strip_prefix(&worker_root).map_err(|_| {
            EvaluationProviderError::Launch(
                "worker current directory escaped worker root".to_string(),
            )
        })?;
        if relative_program.as_os_str().is_empty() {
            return Err(EvaluationProviderError::Launch(
                "worker program could not name the rootfs itself".to_string(),
            ));
        }
        let inside_program = Path::new("/").join(relative_program);
        let inside_current = Path::new("/").join(relative_current);

        let mut args = vec![
            OsString::from("--die-with-parent"),
            OsString::from("--new-session"),
            OsString::from("--unshare-all"),
            OsString::from("--clearenv"),
            OsString::from("--ro-bind"),
            worker_root.as_os_str().to_owned(),
            OsString::from("/"),
            OsString::from("--dir"),
            OsString::from("/staging"),
            OsString::from("--bind"),
            staging.as_os_str().to_owned(),
            OsString::from("/staging"),
            OsString::from("--dir"),
            OsString::from("/proc"),
            OsString::from("--proc"),
            OsString::from("/proc"),
            OsString::from("--dir"),
            OsString::from("/dev"),
            OsString::from("--dev"),
            OsString::from("/dev"),
            OsString::from("--chdir"),
            inside_current.as_os_str().to_owned(),
        ];
        for (key, value) in &launch.environment {
            args.push(OsString::from("--setenv"));
            args.push(key.clone());
            args.push(value.clone());
        }
        if let Some(proxy) = &context.proxy {
            bind_proxy_socket(proxy, &mut args)?;
        }
        args.push(OsString::from("--"));
        args.push(inside_program.into_os_string());
        args.extend(launch.args.iter().cloned());

        let proof_input = format!(
            "aiperf-bwrap-rootfs-v2\n{}\n{}\n{}\n{}\n{}\n{:?}\n{:?}\n{}",
            self.bubblewrap_sha256,
            attestation.launch_closure_sha256,
            context
                .binding_sha256()
                .map_err(EvaluationProviderError::registry)?,
            worker_root.display(),
            staging.display(),
            self.limits,
            context
                .proxy
                .as_ref()
                .map(|binding| &binding.grant.process_scope_sha256),
            mountpoint_identities.join(","),
        );
        let evidence = EvaluatorIsolationEvidence {
            profile_id: "linux-bubblewrap-rootfs-process-tree-v2".to_string(),
            proof_sha256: sha256_hex(proof_input.as_bytes()),
            enforced: EvaluatorIsolationRequirements::strict_process_tree(),
        };
        evidence.validate_strict()?;
        Ok(PreparedEvaluatorLaunch {
            program: self.bubblewrap.clone(),
            args,
            environment: BTreeMap::new(),
            current_dir: PathBuf::from("/"),
            resource_limits: self.limits,
            evidence,
        })
    }

    fn verify_quiescent(
        &self,
        root_pid: u32,
    ) -> Result<IsolationQuiescenceProof, EvaluationProviderError> {
        #[cfg(target_os = "linux")]
        {
            let proc_path = PathBuf::from(format!("/proc/{root_pid}"));
            if proc_path.exists() {
                return Err(EvaluationProviderError::Quiescence(format!(
                    "isolated root process {root_pid} remained live"
                )));
            }
            Ok(IsolationQuiescenceProof::verified(
                root_pid,
                sha256_hex(format!("linux-bwrap-quiescent-v1:{root_pid}").as_bytes()),
            ))
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = root_pid;
            Err(EvaluationProviderError::Quiescence(
                "Bubblewrap process-tree proof is available only on Linux".to_string(),
            ))
        }
    }
}

fn bind_proxy_socket(
    proxy: &ScopedProxyBinding,
    args: &mut Vec<OsString>,
) -> Result<(), EvaluationProviderError> {
    proxy.validate()?;
    let Some(destination) = proxy.local_locator.strip_prefix("unix://") else {
        return Err(EvaluationProviderError::Launch(
            "network-denied Bubblewrap workers require a scoped Unix-domain proxy locator"
                .to_string(),
        ));
    };
    let host_path = &proxy.host_socket_path;
    if !host_path.is_absolute() || destination != "/run/aiperf/evaluator-proxy.sock" {
        return Err(EvaluationProviderError::Launch(
            "scoped proxy host/source socket mapping was invalid".to_string(),
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileTypeExt as _;
        let metadata = std::fs::symlink_metadata(host_path).map_err(|error| {
            EvaluationProviderError::Launch(format!(
                "failed to inspect scoped proxy host socket: {error}"
            ))
        })?;
        if !metadata.file_type().is_socket() || metadata.file_type().is_symlink() {
            return Err(EvaluationProviderError::Launch(
                "scoped proxy bind source was not a real Unix socket".to_string(),
            ));
        }
    }
    // The attested rootfs is mounted read-only. A private empty tmpfs supplies
    // the destination inode for the one read-only socket bind without exposing
    // the host socket directory or any sibling host paths.
    args.push(OsString::from("--tmpfs"));
    args.push(OsString::from("/run/aiperf"));
    args.push(OsString::from("--ro-bind"));
    args.push(host_path.as_os_str().to_owned());
    args.push(OsString::from(destination));
    Ok(())
}

fn validate_rootfs_mountpoints(
    worker_root: &Path,
    proxy_enabled: bool,
) -> Result<Vec<String>, EvaluationProviderError> {
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt as _;

    let required = if proxy_enabled {
        &["staging", "proc", "dev", "run/aiperf"][..]
    } else {
        &["staging", "proc", "dev"][..]
    };
    let mut identities = Vec::with_capacity(required.len());
    for relative in required {
        let path = worker_root.join(relative);
        let metadata = std::fs::symlink_metadata(&path).map_err(|error| {
            EvaluationProviderError::Launch(format!(
                "required rootfs mountpoint {relative:?} was unavailable: {error}"
            ))
        })?;
        if !metadata.file_type().is_dir() || metadata.file_type().is_symlink() {
            return Err(EvaluationProviderError::Launch(format!(
                "required rootfs mountpoint {relative:?} was not a real directory"
            )));
        }
        let canonical = std::fs::canonicalize(&path).map_err(|error| {
            EvaluationProviderError::Launch(format!(
                "failed to resolve rootfs mountpoint {relative:?}: {error}"
            ))
        })?;
        if canonical != path
            || std::fs::read_dir(&path)
                .map_err(|error| {
                    EvaluationProviderError::Launch(format!(
                        "failed to enumerate rootfs mountpoint {relative:?}: {error}"
                    ))
                })?
                .next()
                .is_some()
        {
            return Err(EvaluationProviderError::Launch(format!(
                "required rootfs mountpoint {relative:?} was not normalized and empty"
            )));
        }
        #[cfg(unix)]
        identities.push(format!(
            "{relative}:{:o}",
            metadata.permissions().mode() & 0o7777
        ));
        #[cfg(not(unix))]
        identities.push(relative.to_string());
    }
    Ok(identities)
}

fn normalize_absolute(path: &Path) -> Result<PathBuf, EvaluationProviderError> {
    if !path.is_absolute() {
        return Err(EvaluationProviderError::Launch(
            "launch identity path was not absolute".to_string(),
        ));
    }
    let mut output = PathBuf::new();
    for component in path.components() {
        match component {
            Component::RootDir | Component::Prefix(_) | Component::Normal(_) => {
                output.push(component.as_os_str());
            }
            Component::CurDir => {}
            Component::ParentDir => {
                if !output.pop() {
                    return Err(EvaluationProviderError::Launch(
                        "launch identity path escaped its root".to_string(),
                    ));
                }
            }
        }
    }
    Ok(output)
}

fn hash_file(path: &Path) -> Result<String, EvaluationProviderError> {
    let mut file = File::open(path).map_err(|error| {
        EvaluationProviderError::Launch(format!("failed to open {:?}: {error}", path))
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|error| {
            EvaluationProviderError::Launch(format!("failed to hash {:?}: {error}", path))
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let digest = hasher.finalize();
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    Ok(output)
}

/// Launch attestation/isolation setup failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IsolationError(String);

impl Display for IsolationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for IsolationError {}

#[cfg(test)]
mod tests {
    use std::io::{Read as _, Write};
    use std::os::fd::AsRawFd as _;
    use std::os::unix::fs::PermissionsExt;
    use std::os::unix::net::{UnixListener, UnixStream};
    use std::os::unix::process::CommandExt as _;
    use std::process::{Command, Stdio};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use super::*;
    use crate::provider::{EvaluatorProcessRootBinder, ProviderRegistryError};
    use crate::provider_protocol::{
        EvaluationDistributionId, EvaluationSessionId, LogicalServiceId, OperationPurpose,
        ScopedProxyGrant, ScopedProxySecret, SemanticOperationId,
    };

    #[test]
    fn launch_attestation_hashes_files_not_worker_claims() {
        let root = std::env::temp_dir().join(format!(
            "aiperf-attest-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let program = root.join("worker");
        let mut file = File::create(&program).unwrap();
        file.write_all(b"worker-v1").unwrap();
        drop(file);
        std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o755)).unwrap();
        let digest = hash_file(&program).unwrap();
        let closure_line = format!("worker\0{}\n", digest);
        let closure_digest = sha256_hex(closure_line.as_bytes());
        let launch = AttestedWorkerLaunch {
            distribution_id: EvaluationDistributionId::new("fixture").unwrap(),
            program: program.clone(),
            args: Vec::new(),
            environment: BTreeMap::new(),
            current_dir: root.clone(),
            worker_root: root.clone(),
            closure: vec![LaunchClosureFile {
                path: program.clone(),
                artifact_content_sha256: digest,
            }],
        };
        let distribution = EvaluationDistributionDescriptor {
            distribution_id: EvaluationDistributionId::new("fixture").unwrap(),
            package: "fixture".to_string(),
            package_version: "1".to_string(),
            provider_source_sha256: "a".repeat(64),
            worker_source_sha256: "b".repeat(64),
            dependency_lock_sha256: "c".repeat(64),
            oci_digest: None,
            launch_closure_sha256: closure_digest,
        };
        Sha256LaunchAttestor.attest(&launch, &distribution).unwrap();

        let relocated_root = root.with_extension("relocated");
        let _ = std::fs::remove_dir_all(&relocated_root);
        std::fs::create_dir_all(&relocated_root).unwrap();
        let relocated_program = relocated_root.join("worker");
        std::fs::copy(&program, &relocated_program).unwrap();
        let relocated_launch = AttestedWorkerLaunch {
            program: relocated_program.clone(),
            current_dir: relocated_root.clone(),
            worker_root: relocated_root.clone(),
            closure: vec![LaunchClosureFile {
                path: relocated_program,
                artifact_content_sha256: launch.closure[0].artifact_content_sha256.clone(),
            }],
            ..launch.clone()
        };
        Sha256LaunchAttestor
            .attest(&relocated_launch, &distribution)
            .unwrap();
        std::fs::write(relocated_root.join("undeclared"), b"escape").unwrap();
        assert!(
            Sha256LaunchAttestor
                .attest(&relocated_launch, &distribution)
                .is_err()
        );

        std::fs::write(&program, b"tampered").unwrap();
        assert!(Sha256LaunchAttestor.attest(&launch, &distribution).is_err());
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(relocated_root);
    }

    #[test]
    fn launch_environment_rejects_credential_shaped_keys() {
        let root = PathBuf::from("/tmp/worker-root");
        let launch = AttestedWorkerLaunch {
            distribution_id: EvaluationDistributionId::new("fixture").unwrap(),
            program: root.join("worker"),
            args: Vec::new(),
            environment: BTreeMap::from([(OsString::from("API_KEY"), OsString::from("secret"))]),
            current_dir: root.clone(),
            worker_root: root.clone(),
            closure: vec![LaunchClosureFile {
                path: root.join("worker"),
                artifact_content_sha256: "a".repeat(64),
            }],
        };
        assert!(launch.validate().is_err());
    }

    #[derive(Debug)]
    struct NoopProcessRootBinder;

    impl EvaluatorProcessRootBinder for NoopProcessRootBinder {
        fn bind_attested_root(&self, _root_pid: u32) -> Result<(), ProviderRegistryError> {
            Ok(())
        }
    }

    #[cfg(target_os = "linux")]
    fn registered_bubblewrap() -> (PathBuf, String, EvaluatorResourceLimits) {
        let manifest_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../src/aiperf/accuracy/evaluation/manifests/stock_distributions.json");
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(manifest_path).unwrap()).unwrap();
        let distributions = manifest["distributions"].as_array().unwrap();
        let first = &distributions[0]["isolation"];
        let path = PathBuf::from(first["bubblewrap"].as_str().unwrap());
        let digest = first["bubblewrap_sha256"].as_str().unwrap().to_string();
        let limits: EvaluatorResourceLimits =
            serde_json::from_value(first["resource_limits"].clone()).unwrap();
        for distribution in distributions {
            let isolation = &distribution["isolation"];
            assert_eq!(isolation["bubblewrap"].as_str(), path.to_str());
            assert_eq!(
                isolation["bubblewrap_sha256"].as_str(),
                Some(digest.as_str())
            );
            assert_eq!(
                isolation["profile_id"].as_str(),
                Some("linux-bubblewrap-rootfs-process-tree-v2")
            );
        }
        (path, digest, limits)
    }

    #[cfg(target_os = "linux")]
    fn find_executable(name: &str) -> PathBuf {
        std::env::var_os("PATH")
            .and_then(|path| {
                std::env::split_paths(&path)
                    .map(|directory| directory.join(name))
                    .find(|candidate| candidate.is_file())
            })
            .and_then(|path| std::fs::canonicalize(path).ok())
            .unwrap_or_else(|| panic!("{name} must be available for the Bubblewrap e2e test"))
    }

    #[cfg(target_os = "linux")]
    fn copy_regular_file(source: &Path, worker_root: &Path, destination: &Path) {
        assert!(destination.is_absolute());
        let target = worker_root.join(destination.strip_prefix("/").unwrap());
        std::fs::create_dir_all(target.parent().unwrap()).unwrap();
        std::fs::copy(source, target).unwrap();
    }

    #[cfg(target_os = "linux")]
    fn copy_regular_tree(source: &Path, worker_root: &Path, destination: &Path) {
        let target = worker_root.join(destination.strip_prefix("/").unwrap());
        std::fs::create_dir_all(&target).unwrap();
        for entry in std::fs::read_dir(source).unwrap() {
            let entry = entry.unwrap();
            if matches!(
                entry.file_name().to_str(),
                Some("site-packages" | "__pycache__")
            ) {
                continue;
            }
            let file_type = entry.file_type().unwrap();
            let child_destination = destination.join(entry.file_name());
            if file_type.is_dir() {
                copy_regular_tree(&entry.path(), worker_root, &child_destination);
            } else if file_type.is_file() {
                copy_regular_file(&entry.path(), worker_root, &child_destination);
            } else {
                panic!("Python runtime fixture contained a non-regular entry");
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn copy_dynamic_dependencies(binary: &Path, worker_root: &Path) {
        let output = Command::new("ldd").arg(binary).output().unwrap();
        assert!(output.status.success());
        let stdout = String::from_utf8(output.stdout).unwrap();
        let mut copied = BTreeSet::new();
        for token in stdout.split_ascii_whitespace() {
            let dependency = Path::new(token.trim_end_matches(':'));
            if dependency.is_absolute()
                && dependency.is_file()
                && copied.insert(dependency.to_path_buf())
            {
                copy_regular_file(dependency, worker_root, dependency);
                if dependency
                    .file_name()
                    .is_some_and(|name| name.to_string_lossy().starts_with("ld-linux"))
                {
                    let loader = Path::new("/lib64").join(dependency.file_name().unwrap());
                    copy_regular_file(dependency, worker_root, &loader);
                }
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn materialize_minimal_python(worker_root: &Path) -> PathBuf {
        let python = if Path::new("/usr/bin/python3").is_file() {
            std::fs::canonicalize("/usr/bin/python3").unwrap()
        } else {
            find_executable("python3")
        };
        let query = Command::new(&python)
            .args([
                "-S",
                "-c",
                "import encodings,_socket; print(encodings.__path__[0]); print(_socket.__file__)",
            ])
            .output()
            .unwrap();
        assert!(
            query.status.success(),
            "querying the Python runtime failed: {}",
            String::from_utf8_lossy(&query.stderr)
        );
        let paths = String::from_utf8(query.stdout).unwrap();
        let mut paths = paths.lines();
        let encodings = PathBuf::from(paths.next().unwrap());
        let socket_extension = PathBuf::from(paths.next().unwrap());
        assert!(paths.next().is_none());

        let contained_python = PathBuf::from("/usr/bin/python3");
        copy_regular_file(&python, worker_root, &contained_python);
        copy_dynamic_dependencies(&python, worker_root);
        let stdlib = encodings.parent().unwrap();
        copy_regular_tree(
            stdlib,
            worker_root,
            &Path::new("/usr/lib")
                .join(stdlib.file_name().expect("Python stdlib version directory")),
        );
        copy_dynamic_dependencies(&socket_extension, worker_root);
        worker_root.join(contained_python.strip_prefix("/").unwrap())
    }

    #[cfg(target_os = "linux")]
    fn peer_credentials(stream: &UnixStream) -> (u32, u32) {
        let mut credentials = std::mem::MaybeUninit::<libc::ucred>::uninit();
        let mut length = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
        // SAFETY: the output buffer and its exact byte length remain valid for
        // the duration of the `SO_PEERCRED` syscall.
        let result = unsafe {
            libc::getsockopt(
                stream.as_raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_PEERCRED,
                credentials.as_mut_ptr().cast(),
                &mut length,
            )
        };
        assert_eq!(result, 0);
        assert_eq!(length as usize, std::mem::size_of::<libc::ucred>());
        // SAFETY: successful `getsockopt` initialized the complete structure.
        let credentials = unsafe { credentials.assume_init() };
        (credentials.pid as u32, credentials.uid)
    }

    #[cfg(target_os = "linux")]
    fn is_descendant(mut pid: u32, root_pid: u32) -> bool {
        for _ in 0..256 {
            if pid == root_pid {
                return true;
            }
            if pid <= 1 {
                return false;
            }
            let Ok(status) = std::fs::read_to_string(format!("/proc/{pid}/status")) else {
                return false;
            };
            let Some(parent) = status.lines().find_map(|line| {
                line.strip_prefix("PPid:")
                    .and_then(|value| value.trim().parse::<u32>().ok())
            }) else {
                return false;
            };
            pid = parent;
        }
        false
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn registered_bubblewrap_ro_bound_proxy_preserves_host_peer_ancestry() {
        let (bubblewrap, digest, limits) = registered_bubblewrap();
        let isolation = BubblewrapEvaluatorIsolation::new(&bubblewrap, digest, limits).unwrap();
        if let Err(error) = isolation.check_available() {
            eprintln!("skipping unavailable immutable Bubblewrap profile: {error}");
            return;
        }

        let base = std::env::temp_dir().join(format!(
            "aiperf-real-bwrap-proxy-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        let worker_root = base.join("rootfs");
        let staging = base.join("staging");
        for mountpoint in ["staging", "proc", "dev", "run/aiperf"] {
            std::fs::create_dir_all(worker_root.join(mountpoint)).unwrap();
        }
        std::fs::create_dir_all(&staging).unwrap();
        let contained_python = materialize_minimal_python(&worker_root);
        let socket_path = base.join("evaluator-proxy.sock");
        let listener = UnixListener::bind(&socket_path).unwrap();
        listener.set_nonblocking(true).unwrap();

        let session_id = EvaluationSessionId::new("real-bwrap-e2e").unwrap();
        let proxy = ScopedProxyBinding {
            local_locator: "unix:///run/aiperf/evaluator-proxy.sock".to_string(),
            host_socket_path: socket_path.clone(),
            grant: ScopedProxyGrant {
                grant_id: "real-bwrap-grant".to_string(),
                session_id: session_id.clone(),
                secret: ScopedProxySecret::new("s".repeat(48)).unwrap(),
                service_ids: vec![LogicalServiceId::new("candidate").unwrap()],
                semantic_operation_ids: vec![SemanticOperationId::new("model.generate").unwrap()],
                purposes: vec![OperationPurpose::new("primary").unwrap()],
                process_scope_sha256: "a".repeat(64),
                max_operations: 2,
                max_concurrent_operations: 1,
                max_request_bytes: 1024,
                max_response_bytes: 1024,
                max_stream_events: 8,
                expires_after_ms: 10_000,
            },
        };
        let script = r#"
import posix as os
os.fstat(3)
os.fstat(4)
try:
    import _socket
except BaseException as error:
    os.write(2, (type(error).__name__ + ': ' + str(error)).encode())
    os._exit(34)
try:
    stream = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    stream.connect('/run/aiperf/evaluator-proxy.sock')
    stream.sendall(b'I')
    reply = stream.recv(1)
    os._exit(0 if reply == b'A' else 31)
except BaseException as error:
    os.write(2, (type(error).__name__ + ': ' + str(error)).encode())
    os._exit(33)
"#;
        let launch = AttestedWorkerLaunch {
            distribution_id: EvaluationDistributionId::new("real-bwrap-fixture").unwrap(),
            program: contained_python.clone(),
            args: vec![
                OsString::from("-S"),
                OsString::from("-c"),
                OsString::from(script),
            ],
            environment: BTreeMap::from([(OsString::from("PYTHONHOME"), OsString::from("/usr"))]),
            current_dir: worker_root.clone(),
            worker_root: worker_root.clone(),
            closure: vec![LaunchClosureFile {
                path: contained_python.clone(),
                artifact_content_sha256: hash_file(&contained_python).unwrap(),
            }],
        };
        let context = ProviderLaunchContext {
            session_id,
            staging_dir: staging,
            proxy: Some(proxy),
            process_root_binder: Some(Arc::new(NoopProcessRootBinder)),
            protocol_limits: crate::provider::EvaluatorProtocolLimits::default(),
            launch_nonce: "real-bwrap-e2e-nonce-0123456789abcdef".to_string(),
        };
        let prepared = isolation
            .prepare(
                &launch,
                &LaunchAttestation {
                    distribution_id: launch.distribution_id.clone(),
                    executable_sha256: hash_file(&contained_python).unwrap(),
                    launch_closure_sha256: "b".repeat(64),
                    verified_files: 1,
                },
                &context,
            )
            .unwrap();

        let control_read = File::open("/dev/null").unwrap();
        let control_write = File::open("/dev/null").unwrap();
        let control_read_fd = control_read.as_raw_fd();
        let control_write_fd = control_write.as_raw_fd();
        let mut command = Command::new(&prepared.program);
        command
            .args(&prepared.args)
            .env_clear()
            .envs(&prepared.environment)
            .current_dir(&prepared.current_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        // SAFETY: only descriptor duplication/close syscalls run between fork
        // and exec, and every captured value is a plain descriptor integer.
        unsafe {
            command.pre_exec(move || {
                if libc::dup2(control_read_fd, 3) < 0 || libc::dup2(control_write_fd, 4) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                for descriptor in [control_read_fd, control_write_fd] {
                    if descriptor > 4 {
                        libc::close(descriptor);
                    }
                }
                Ok(())
            });
        }
        let mut isolated = command.spawn().unwrap();
        let root_pid = isolated.id();
        drop(control_read);
        drop(control_write);

        let sibling_script = r#"
import os, socket
stream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
stream.connect(os.environ['AIPERF_TEST_SOCKET'])
stream.sendall(b'S')
raise SystemExit(0 if stream.recv(1) == b'D' else 41)
"#;
        let mut sibling = Command::new(find_executable("python3"))
            .args(["-S", "-c", sibling_script])
            .env("AIPERF_TEST_SOCKET", &socket_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();

        let expected_uid = unsafe { libc::geteuid() };
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut isolated_authorized = None;
        let mut sibling_authorized = None;
        while isolated_authorized.is_none() || sibling_authorized.is_none() {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    let mut marker = [0_u8; 1];
                    stream.read_exact(&mut marker).unwrap();
                    let (peer_pid, peer_uid) = peer_credentials(&stream);
                    let authorized = peer_uid == expected_uid && is_descendant(peer_pid, root_pid);
                    match marker[0] {
                        b'I' => {
                            let duplicate = isolated_authorized.replace(authorized).is_some();
                            stream
                                .write_all(if authorized && !duplicate { b"A" } else { b"D" })
                                .unwrap();
                        }
                        b'S' => {
                            let duplicate = sibling_authorized.replace(authorized).is_some();
                            stream
                                .write_all(if authorized || duplicate { b"A" } else { b"D" })
                                .unwrap();
                        }
                        other => panic!("unexpected proxy fixture marker {other}"),
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    if Instant::now() >= deadline {
                        let _ = isolated.kill();
                        let _ = sibling.kill();
                        let isolated_output = isolated.wait_with_output().unwrap();
                        let sibling_output = sibling.wait_with_output().unwrap();
                        panic!(
                            "Bubblewrap proxy fixture timed out; isolated stderr: {}; sibling stderr: {}",
                            String::from_utf8_lossy(&isolated_output.stderr),
                            String::from_utf8_lossy(&sibling_output.stderr)
                        );
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(error) => panic!("accepting proxy fixture connection failed: {error}"),
            }
        }
        let isolated_output = isolated.wait_with_output().unwrap();
        let sibling_output = sibling.wait_with_output().unwrap();
        assert_eq!(isolated_authorized, Some(true));
        assert_eq!(sibling_authorized, Some(false));
        assert!(
            isolated_output.status.success(),
            "isolated Python failed: {}",
            String::from_utf8_lossy(&isolated_output.stderr)
        );
        assert!(
            sibling_output.status.success(),
            "host sibling failed: {}",
            String::from_utf8_lossy(&sibling_output.stderr)
        );
        isolation.verify_quiescent(root_pid).unwrap();
        let _ = std::fs::remove_dir_all(base);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn bubblewrap_availability_requires_exact_regular_executable() {
        let root =
            std::env::temp_dir().join(format!("aiperf-bwrap-availability-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let binary = root.join("bwrap");
        std::fs::write(&binary, b"fixture-bwrap").unwrap();
        std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o755)).unwrap();
        let digest = hash_file(&binary).unwrap();
        let isolation =
            BubblewrapEvaluatorIsolation::new(&binary, digest, EvaluatorResourceLimits::default())
                .unwrap();
        isolation.check_available().unwrap();

        std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(isolation.check_available().is_err());
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn bubblewrap_mounts_attested_rootfs_at_root_for_dynamic_runtime_paths() {
        let base = std::env::temp_dir().join(format!("aiperf-bwrap-rootfs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let worker_root = base.join("rootfs");
        let staging = base.join("staging");
        let program = worker_root.join("bin/worker");
        let bubblewrap = base.join("bwrap");
        std::fs::create_dir_all(program.parent().unwrap()).unwrap();
        for mountpoint in ["staging", "proc", "dev", "run/aiperf"] {
            std::fs::create_dir_all(worker_root.join(mountpoint)).unwrap();
        }
        std::fs::create_dir_all(&staging).unwrap();
        std::fs::write(&program, b"worker").unwrap();
        std::fs::write(&bubblewrap, b"bwrap").unwrap();
        std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o755)).unwrap();
        std::fs::set_permissions(&bubblewrap, std::fs::Permissions::from_mode(0o755)).unwrap();
        let isolation = BubblewrapEvaluatorIsolation::new(
            &bubblewrap,
            hash_file(&bubblewrap).unwrap(),
            EvaluatorResourceLimits::default(),
        )
        .unwrap();
        let launch = AttestedWorkerLaunch {
            distribution_id: EvaluationDistributionId::new("fixture").unwrap(),
            program,
            args: vec![OsString::from("--fixture")],
            environment: BTreeMap::new(),
            current_dir: worker_root.clone(),
            worker_root: worker_root.clone(),
            closure: vec![LaunchClosureFile {
                path: worker_root.join("bin/worker"),
                artifact_content_sha256: hash_file(&worker_root.join("bin/worker")).unwrap(),
            }],
        };
        let context = ProviderLaunchContext {
            session_id: EvaluationSessionId::new("fixture-session").unwrap(),
            staging_dir: staging,
            proxy: None,
            process_root_binder: None,
            protocol_limits: crate::provider::EvaluatorProtocolLimits::default(),
            launch_nonce: "fixture-bwrap-nonce-0123456789abcdef".to_string(),
        };
        let prepared = isolation
            .prepare(
                &launch,
                &LaunchAttestation {
                    distribution_id: launch.distribution_id.clone(),
                    executable_sha256: "a".repeat(64),
                    launch_closure_sha256: "b".repeat(64),
                    verified_files: 1,
                },
                &context,
            )
            .unwrap();
        let args = prepared
            .args
            .iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        let root_bind = args
            .windows(3)
            .find(|window| window[0] == "--ro-bind" && window[2] == "/")
            .expect("worker rootfs must be mounted at /");
        assert_eq!(root_bind[1], worker_root.to_string_lossy());
        assert!(
            args.windows(2)
                .any(|window| window == ["--", "/bin/worker"])
        );
        assert_eq!(
            prepared.evidence.profile_id,
            "linux-bubblewrap-rootfs-process-tree-v2"
        );
        std::fs::write(worker_root.join("dev/undeclared"), b"not-empty").unwrap();
        assert!(
            isolation
                .prepare(
                    &launch,
                    &LaunchAttestation {
                        distribution_id: launch.distribution_id.clone(),
                        executable_sha256: "a".repeat(64),
                        launch_closure_sha256: "b".repeat(64),
                        verified_files: 1,
                    },
                    &context,
                )
                .is_err()
        );
        let _ = std::fs::remove_dir_all(base);
    }
}
