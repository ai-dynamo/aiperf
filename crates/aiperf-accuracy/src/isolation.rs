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

const SANDBOX_UID: u32 = 65_534;
const SANDBOX_GID: u32 = 65_534;
const PRIVATE_STAGING_MODE: u32 = 0o700;
pub(crate) const BOOTSTRAP_PROCESS_LIMIT_ENV: &str = "AIPERF_EVALUATOR_BOOTSTRAP_PROCESS_LIMIT";
// Runtime state (HOME/TMP/XDG) is routed to a private, ephemeral `/work` tmpfs
// rather than the host-bound `/staging` tree. `/staging` is walked and sealed as
// the evaluator artifact tree, so provider runtime files written under HOME/XDG
// (e.g. Inspect's `.xdg-data/inspect_ai/traces/*.log.gz`) would otherwise fail
// the seal as undeclared artifacts. Keeping them on a tmpfs that is never sealed
// is structural — it does not require enumerating each provider's cache files.
const REQUIRED_RUNTIME_ENVIRONMENT: [(&str, &str); 5] = [
    ("HOME", "/work"),
    ("TMPDIR", "/work"),
    ("XDG_CONFIG_HOME", "/work"),
    ("XDG_DATA_HOME", "/work"),
    ("XDG_CACHE_HOME", "/work"),
];

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct LinuxProcessIdentity {
    pid: u32,
    start_time_ticks: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct LinuxPidNamespaceIdentity {
    device: u64,
    inode: u64,
}

/// Host-observed identity of one isolated evaluator process tree.
///
/// The identity captures concrete `(pid, start-time)` pairs plus each private
/// PID namespace and its PID-1 process while the attested root is live. Linux
/// terminates every namespace member when that init exits, so pinning and
/// checking the init also covers descendants forked after observation. A
/// future platform isolation implementation can populate an equivalent
/// subtree identity behind [`EvaluatorIsolation`] without changing the
/// supervisor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IsolationProcessTreeIdentity {
    root: LinuxProcessIdentity,
    outer_uid: u32,
    processes: BTreeSet<LinuxProcessIdentity>,
    isolated_pid_namespaces: BTreeSet<LinuxPidNamespaceIdentity>,
    isolated_pid_namespace_inits: BTreeMap<LinuxPidNamespaceIdentity, LinuxProcessIdentity>,
    require_isolated_pid_namespace: bool,
}

impl IsolationProcessTreeIdentity {
    /// Host PID of the attested isolation root.
    pub fn root_pid(&self) -> u32 {
        self.root.pid
    }

    #[cfg(test)]
    pub(crate) fn fixture(
        root_pid: u32,
        previous: Option<&Self>,
    ) -> Result<Self, EvaluationProviderError> {
        capture_linux_process_tree(root_pid, previous, false)
    }
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

    /// Observe the live isolated subtree before allowing worker execution.
    ///
    /// Re-observation receives the prior identity so implementations can
    /// monotonically retain namespaces and short-lived process identities.
    fn observe_process_tree(
        &self,
        root_pid: u32,
        previous: Option<&IsolationProcessTreeIdentity>,
    ) -> Result<IsolationProcessTreeIdentity, EvaluationProviderError>;

    /// Verify no captured process or process in a captured namespace remains.
    fn verify_quiescent(
        &self,
        identity: &IsolationProcessTreeIdentity,
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
        let staging_identities = prepare_private_staging_layout(&staging)?;
        let mut environment = exact_staging_environment(&launch.environment)?;
        if environment
            .insert(
                OsString::from(BOOTSTRAP_PROCESS_LIMIT_ENV),
                OsString::from(self.limits.processes.to_string()),
            )
            .is_some()
        {
            return Err(EvaluationProviderError::Launch(
                "worker environment reserved the evaluator bootstrap process-limit key".to_string(),
            ));
        }
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
            OsString::from("--uid"),
            OsString::from(SANDBOX_UID.to_string()),
            OsString::from("--gid"),
            OsString::from(SANDBOX_GID.to_string()),
            OsString::from("--cap-drop"),
            OsString::from("ALL"),
            OsString::from("--clearenv"),
            OsString::from("--ro-bind"),
            worker_root.as_os_str().to_owned(),
            OsString::from("/"),
            OsString::from("--tmpfs"),
            OsString::from("/work"),
            OsString::from("--chmod"),
            OsString::from("1777"),
            OsString::from("/work"),
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
        for (key, value) in &environment {
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
            "aiperf-bwrap-rootfs-v3\n{}\n{}\n{}\n{}\n{}\nuid={}\ngid={}\ncap-drop=ALL\n{:?}\n{:?}\n{}\n{}\n{:?}",
            self.bubblewrap_sha256,
            attestation.launch_closure_sha256,
            context
                .binding_sha256()
                .map_err(EvaluationProviderError::registry)?,
            worker_root.display(),
            staging.display(),
            SANDBOX_UID,
            SANDBOX_GID,
            self.limits,
            context
                .proxy
                .as_ref()
                .map(|binding| &binding.grant.process_scope_sha256),
            mountpoint_identities.join(","),
            staging_identities.join(","),
            environment,
        );
        let evidence = EvaluatorIsolationEvidence {
            profile_id: "linux-bubblewrap-rootfs-process-tree-v3".to_string(),
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

    fn observe_process_tree(
        &self,
        root_pid: u32,
        previous: Option<&IsolationProcessTreeIdentity>,
    ) -> Result<IsolationProcessTreeIdentity, EvaluationProviderError> {
        #[cfg(target_os = "linux")]
        {
            capture_linux_process_tree(root_pid, previous, true)
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (root_pid, previous);
            Err(EvaluationProviderError::Quiescence(
                "Bubblewrap process-tree observation is available only on Linux".to_string(),
            ))
        }
    }

    fn verify_quiescent(
        &self,
        identity: &IsolationProcessTreeIdentity,
    ) -> Result<IsolationQuiescenceProof, EvaluationProviderError> {
        #[cfg(target_os = "linux")]
        {
            verify_linux_process_tree_quiescent(identity)?;
            let root_pid = identity.root.pid;
            let identity_sha256 = process_tree_identity_sha256(identity);
            Ok(IsolationQuiescenceProof::verified(
                root_pid,
                sha256_hex(
                    format!("linux-bwrap-quiescent-v2:{root_pid}:{identity_sha256}").as_bytes(),
                ),
            ))
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = identity;
            Err(EvaluationProviderError::Quiescence(
                "Bubblewrap process-tree proof is available only on Linux".to_string(),
            ))
        }
    }
}

fn exact_staging_environment(
    authored: &BTreeMap<OsString, OsString>,
) -> Result<BTreeMap<OsString, OsString>, EvaluationProviderError> {
    let mut environment = authored.clone();
    for (key, expected) in REQUIRED_RUNTIME_ENVIRONMENT {
        let key = OsString::from(key);
        let expected = OsString::from(expected);
        if let Some(actual) = environment.get(&key)
            && actual != &expected
        {
            return Err(EvaluationProviderError::Launch(format!(
                "worker environment supplied an invalid private runtime path for {}",
                key.to_string_lossy()
            )));
        }
        environment.insert(key, expected);
    }
    Ok(environment)
}

fn prepare_private_staging_layout(staging: &Path) -> Result<Vec<String>, EvaluationProviderError> {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};

        let expected_uid = unsafe { libc::geteuid() };
        let expected_gid = unsafe { libc::getegid() };
        // Only `/staging` is host-bound and sealed. Runtime roots (HOME/TMP/XDG)
        // now live on the ephemeral `/work` tmpfs and need no host directory.
        let mut identities = Vec::with_capacity(1);
        for (name, path) in [("staging", staging.to_path_buf())] {
            let metadata = std::fs::symlink_metadata(&path).map_err(|error| {
                EvaluationProviderError::Launch(format!(
                    "failed to inspect private staging directory {name:?}: {error}"
                ))
            })?;
            if !metadata.file_type().is_dir()
                || metadata.file_type().is_symlink()
                || metadata.uid() != expected_uid
                || metadata.gid() != expected_gid
            {
                return Err(EvaluationProviderError::Launch(format!(
                    "private staging directory {name:?} was not a host-owned real directory"
                )));
            }
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(PRIVATE_STAGING_MODE))
                .map_err(|error| {
                    EvaluationProviderError::Launch(format!(
                        "failed to secure private staging directory {name:?}: {error}"
                    ))
                })?;
            let secured = std::fs::symlink_metadata(&path).map_err(|error| {
                EvaluationProviderError::Launch(format!(
                    "failed to re-inspect private staging directory {name:?}: {error}"
                ))
            })?;
            if secured.uid() != expected_uid
                || secured.gid() != expected_gid
                || secured.permissions().mode() & 0o7777 != PRIVATE_STAGING_MODE
            {
                return Err(EvaluationProviderError::Launch(format!(
                    "private staging directory {name:?} ownership or permissions drifted"
                )));
            }
            identities.push(format!(
                "{name}:{}:{}:{:o}",
                secured.uid(),
                secured.gid(),
                secured.permissions().mode() & 0o7777
            ));
        }
        Ok(identities)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = staging;
        Err(EvaluationProviderError::Launch(
            "private evaluator staging layout is available only on Linux".to_string(),
        ))
    }
}

#[cfg(target_os = "linux")]
fn capture_linux_process_tree(
    root_pid: u32,
    previous: Option<&IsolationProcessTreeIdentity>,
    require_isolated_pid_namespace: bool,
) -> Result<IsolationProcessTreeIdentity, EvaluationProviderError> {
    use std::os::unix::fs::MetadataExt as _;

    if let Some(previous) = previous
        && previous.root.pid != root_pid
    {
        return Err(EvaluationProviderError::Quiescence(
            "process-tree observation changed isolation root PID".to_string(),
        ));
    }
    let Some(root_stat) = read_linux_process_stat(root_pid)? else {
        return previous.cloned().ok_or_else(|| {
            EvaluationProviderError::Quiescence(format!(
                "isolated root process {root_pid} exited before its subtree identity was captured"
            ))
        });
    };
    let root = LinuxProcessIdentity {
        pid: root_pid,
        start_time_ticks: root_stat.start_time_ticks,
    };
    let root_metadata = std::fs::metadata(format!("/proc/{root_pid}")).map_err(|error| {
        EvaluationProviderError::Quiescence(format!(
            "failed to inspect isolation root ownership: {error}"
        ))
    })?;
    let outer_uid = root_metadata.uid();
    if outer_uid != unsafe { libc::geteuid() } {
        return Err(EvaluationProviderError::Quiescence(
            "isolation root did not retain the supervising host identity".to_string(),
        ));
    }
    if let Some(previous) = previous
        && (previous.root != root || previous.outer_uid != outer_uid)
    {
        return Err(EvaluationProviderError::Quiescence(
            "isolation root PID was reused while the worker was active".to_string(),
        ));
    }

    let mut stats = BTreeMap::new();
    let proc_entries = std::fs::read_dir("/proc").map_err(|error| {
        EvaluationProviderError::Quiescence(format!("failed to enumerate /proc: {error}"))
    })?;
    for entry in proc_entries {
        let entry = entry.map_err(|error| {
            EvaluationProviderError::Quiescence(format!(
                "failed to enumerate a /proc entry: {error}"
            ))
        })?;
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|value| value.parse::<u32>().ok())
        else {
            continue;
        };
        let metadata = match entry.metadata() {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(EvaluationProviderError::Quiescence(format!(
                    "failed to inspect process {pid} ownership: {error}"
                )));
            }
        };
        if metadata.uid() != outer_uid {
            continue;
        }
        if let Some(stat) = read_linux_process_stat(pid)? {
            stats.insert(pid, stat);
        }
    }

    let host_namespace = namespace_identity(Path::new("/proc/self/ns/pid"))?;
    let mut processes = previous
        .map(|identity| identity.processes.clone())
        .unwrap_or_default();
    let mut isolated_pid_namespaces = previous
        .map(|identity| identity.isolated_pid_namespaces.clone())
        .unwrap_or_default();
    let mut isolated_pid_namespace_inits = previous
        .map(|identity| identity.isolated_pid_namespace_inits.clone())
        .unwrap_or_default();
    for (&pid, stat) in &stats {
        if !linux_process_is_descendant(pid, root_pid, &stats) {
            continue;
        }
        let process = LinuxProcessIdentity {
            pid,
            start_time_ticks: stat.start_time_ticks,
        };
        processes.insert(process);
        let namespace_path = PathBuf::from(format!("/proc/{pid}/ns/pid"));
        if let Some(namespace) = namespace_identity_if_present(&namespace_path)?
            && namespace != host_namespace
        {
            isolated_pid_namespaces.insert(namespace);
            if linux_process_namespace_pid(pid)? == Some(1)
                && isolated_pid_namespace_inits
                    .insert(namespace, process)
                    .is_some_and(|existing| existing != process)
            {
                return Err(EvaluationProviderError::Quiescence(
                    "one PID namespace exposed multiple init identities".to_string(),
                ));
            }
        }
    }
    processes.insert(root);

    // Reading the namespace metadata above follows the procfs magic symlink;
    // pin its kernel object identity rather than trusting its display text.
    let _ = std::fs::metadata("/proc/self/ns/pid").map(|metadata| metadata.ino());
    Ok(IsolationProcessTreeIdentity {
        root,
        outer_uid,
        processes,
        isolated_pid_namespaces,
        isolated_pid_namespace_inits,
        require_isolated_pid_namespace: previous
            .is_some_and(|identity| identity.require_isolated_pid_namespace)
            || require_isolated_pid_namespace,
    })
}

#[cfg(target_os = "linux")]
#[derive(Debug, Clone, Copy)]
struct LinuxProcessStat {
    parent_pid: u32,
    start_time_ticks: u64,
}

#[cfg(target_os = "linux")]
fn read_linux_process_stat(pid: u32) -> Result<Option<LinuxProcessStat>, EvaluationProviderError> {
    let path = PathBuf::from(format!("/proc/{pid}/stat"));
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(EvaluationProviderError::Quiescence(format!(
                "failed to inspect process {pid}: {error}"
            )));
        }
    };
    let close = text.rfind(')').ok_or_else(|| {
        EvaluationProviderError::Quiescence(format!("process {pid} had malformed procfs stat"))
    })?;
    let fields = text[close + 1..]
        .split_ascii_whitespace()
        .collect::<Vec<_>>();
    if fields.len() <= 19 {
        return Err(EvaluationProviderError::Quiescence(format!(
            "process {pid} had incomplete procfs stat"
        )));
    }
    let parent_pid = fields[1].parse::<u32>().map_err(|_| {
        EvaluationProviderError::Quiescence(format!("process {pid} had invalid parent identity"))
    })?;
    let start_time_ticks = fields[19].parse::<u64>().map_err(|_| {
        EvaluationProviderError::Quiescence(format!(
            "process {pid} had invalid start-time identity"
        ))
    })?;
    Ok(Some(LinuxProcessStat {
        parent_pid,
        start_time_ticks,
    }))
}

#[cfg(target_os = "linux")]
fn linux_process_namespace_pid(pid: u32) -> Result<Option<u32>, EvaluationProviderError> {
    let path = PathBuf::from(format!("/proc/{pid}/status"));
    let status = match std::fs::read_to_string(&path) {
        Ok(status) => status,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(EvaluationProviderError::Quiescence(format!(
                "failed to inspect process {pid} namespace PID: {error}"
            )));
        }
    };
    status
        .lines()
        .find_map(|line| line.strip_prefix("NSpid:"))
        .and_then(|value| value.split_ascii_whitespace().last())
        .map(|value| {
            value.parse::<u32>().map_err(|_| {
                EvaluationProviderError::Quiescence(format!(
                    "process {pid} had an invalid namespace PID identity"
                ))
            })
        })
        .transpose()
}

#[cfg(target_os = "linux")]
fn linux_process_is_descendant(
    mut pid: u32,
    root_pid: u32,
    stats: &BTreeMap<u32, LinuxProcessStat>,
) -> bool {
    for _ in 0..256 {
        if pid == root_pid {
            return true;
        }
        let Some(stat) = stats.get(&pid) else {
            return false;
        };
        if stat.parent_pid <= 1 || stat.parent_pid == pid {
            return false;
        }
        pid = stat.parent_pid;
    }
    false
}

#[cfg(target_os = "linux")]
fn namespace_identity(path: &Path) -> Result<LinuxPidNamespaceIdentity, EvaluationProviderError> {
    namespace_identity_if_present(path)?.ok_or_else(|| {
        EvaluationProviderError::Quiescence(format!(
            "PID namespace identity {} disappeared",
            path.display()
        ))
    })
}

#[cfg(target_os = "linux")]
fn namespace_identity_if_present(
    path: &Path,
) -> Result<Option<LinuxPidNamespaceIdentity>, EvaluationProviderError> {
    use std::os::unix::fs::MetadataExt as _;

    match std::fs::metadata(path) {
        Ok(metadata) => Ok(Some(LinuxPidNamespaceIdentity {
            device: metadata.dev(),
            inode: metadata.ino(),
        })),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(EvaluationProviderError::Quiescence(format!(
            "failed to inspect PID namespace {}: {error}",
            path.display()
        ))),
    }
}

#[cfg(target_os = "linux")]
fn verify_linux_process_tree_quiescent(
    identity: &IsolationProcessTreeIdentity,
) -> Result<(), EvaluationProviderError> {
    if identity.require_isolated_pid_namespace && identity.isolated_pid_namespaces.is_empty() {
        return Err(EvaluationProviderError::Quiescence(
            "isolated worker exited without a captured private PID namespace".to_string(),
        ));
    }
    if identity.isolated_pid_namespaces.len() != identity.isolated_pid_namespace_inits.len()
        || !identity.isolated_pid_namespaces.iter().all(|namespace| {
            identity
                .isolated_pid_namespace_inits
                .contains_key(namespace)
        })
    {
        return Err(EvaluationProviderError::Quiescence(
            "isolated worker exited without pinning every PID namespace init".to_string(),
        ));
    }
    for process in &identity.processes {
        if read_linux_process_stat(process.pid)?
            .is_some_and(|stat| stat.start_time_ticks == process.start_time_ticks)
        {
            return Err(EvaluationProviderError::Quiescence(format!(
                "captured evaluator process {} remained live",
                process.pid
            )));
        }
    }
    Ok(())
}

fn process_tree_identity_sha256(identity: &IsolationProcessTreeIdentity) -> String {
    let mut proof = format!(
        "{}:{}:{}:{}\n",
        identity.root.pid,
        identity.root.start_time_ticks,
        identity.outer_uid,
        identity.require_isolated_pid_namespace
    );
    for process in &identity.processes {
        proof.push_str(&format!(
            "process:{}:{}\n",
            process.pid, process.start_time_ticks
        ));
    }
    for namespace in &identity.isolated_pid_namespaces {
        proof.push_str(&format!("pidns:{}:{}\n", namespace.device, namespace.inode));
    }
    for (namespace, init) in &identity.isolated_pid_namespace_inits {
        proof.push_str(&format!(
            "pidns-init:{}:{}:{}:{}\n",
            namespace.device, namespace.inode, init.pid, init.start_time_ticks
        ));
    }
    sha256_hex(proof.as_bytes())
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
    use std::net::TcpListener;
    use std::os::fd::{AsRawFd as _, FromRawFd as _};
    use std::os::unix::fs::PermissionsExt;
    use std::os::unix::net::{UnixListener, UnixStream};
    use std::os::unix::process::CommandExt as _;
    use std::process::{Command, Stdio};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use super::*;
    use crate::provider::{EvaluatorProcessRootBinder, ProviderRegistryError};
    use crate::provider_protocol::{
        EvaluationDistributionId, EvaluationIdentityComponent, EvaluationSessionId,
        LogicalServiceId, OperationPurpose, ScopedProxyGrant, ScopedProxySecret,
        SemanticOperationId,
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
            identity_components: vec![EvaluationIdentityComponent {
                name: "fixture-worker".to_string(),
                version: "1".to_string(),
                source_sha256: "b".repeat(64),
                source_commit: None,
                base_source_sha256: None,
                overlay_policy: None,
                overlays: Vec::new(),
            }],
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
                Some("linux-bubblewrap-rootfs-process-tree-v3")
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
                "import encodings,_socket,resource; print(encodings.__path__[0]); print(_socket.__file__); print(resource.__file__)",
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
        let resource_extension = PathBuf::from(paths.next().unwrap());
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
        copy_dynamic_dependencies(&resource_extension, worker_root);
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
    fn effective_resource_limit(resource: libc::__rlimit_resource_t, requested: u64) -> u64 {
        let mut current = std::mem::MaybeUninit::<libc::rlimit>::uninit();
        // SAFETY: the fixed resource identifier is valid and the successful
        // syscall initializes the complete output structure.
        assert_eq!(
            unsafe { libc::getrlimit(resource, current.as_mut_ptr()) },
            0
        );
        // SAFETY: `getrlimit` succeeded above.
        requested.min(unsafe { current.assume_init() }.rlim_max)
    }

    #[cfg(target_os = "linux")]
    fn anonymous_pipe() -> (File, File) {
        let mut descriptors = [0_i32; 2];
        assert_eq!(
            unsafe { libc::pipe2(descriptors.as_mut_ptr(), libc::O_CLOEXEC) },
            0
        );
        // SAFETY: successful `pipe2` returned two independently owned files.
        let read = unsafe { File::from_raw_fd(descriptors[0]) };
        // SAFETY: successful `pipe2` returned two independently owned files.
        let write = unsafe { File::from_raw_fd(descriptors[1]) };
        (read, write)
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn registered_bubblewrap_enforces_literal_isolation_acceptance_contract() {
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
        let control_bootstrap = worker_root.join(
            "runtime/lib/python3.12/site-packages/aiperf/accuracy/evaluation/control_bootstrap.py",
        );
        let acceptance_worker = worker_root
            .join("runtime/lib/python3.12/site-packages/aiperf/accuracy/evaluation/worker.py");
        std::fs::create_dir_all(control_bootstrap.parent().unwrap()).unwrap();
        std::fs::write(
            &control_bootstrap,
            include_bytes!("../../../src/aiperf/accuracy/evaluation/control_bootstrap.py"),
        )
        .unwrap();
        let socket_path = base.join("evaluator-proxy.sock");
        let listener = UnixListener::bind(&socket_path).unwrap();
        listener.set_nonblocking(true).unwrap();
        let sibling_socket_path = base.join("sibling-provider.sock");
        let _sibling_listener = UnixListener::bind(&sibling_socket_path).unwrap();
        let host_secret_path = base.join("undeclared-host-secret");
        std::fs::write(&host_secret_path, b"not mounted").unwrap();
        let loopback_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let loopback_port = loopback_listener.local_addr().unwrap().port();

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
import os
import resource
os.fstat(3)
os.fstat(4)
if os.read(3, 1) != b'R':
    os._exit(35)
try:
    import _socket
except BaseException as error:
    os.write(2, (type(error).__name__ + ': ' + str(error)).encode())
    os._exit(34)

def require(condition, code):
    if not condition:
        os._exit(code)

required_environment = {
    'HOME': '/staging/home',
    'TMPDIR': '/staging/tmp',
    'XDG_CONFIG_HOME': '/staging/.xdg-config',
    'XDG_DATA_HOME': '/staging/.xdg-data',
    'XDG_CACHE_HOME': '/staging/.xdg-cache',
}
for key, expected in required_environment.items():
    require(os.environ.get(key) == expected, 40)
    metadata = os.stat(expected)
    require(metadata.st_uid == 65534 and metadata.st_gid == 65534, 41)
    require(metadata.st_mode & 0o7777 == 0o700, 42)
    with open(expected + '/write-proof', 'wb') as output:
        output.write(b'private')

require(os.geteuid() == 65534 and os.getegid() == 65534, 43)
status = {}
with open('/proc/self/status', 'rt', encoding='utf-8') as source:
    for line in source:
        key, _, value = line.partition(':')
        status[key] = value.strip()
require(int(status['CapEff'], 16) == 0, 44)
require(status['NoNewPrivs'] == '1', 45)

expected_limits = [
    (resource.RLIMIT_AS, int(os.environ['AIPERF_EXPECT_RLIMIT_AS'])),
    (resource.RLIMIT_FSIZE, int(os.environ['AIPERF_EXPECT_RLIMIT_FSIZE'])),
    (resource.RLIMIT_NOFILE, int(os.environ['AIPERF_EXPECT_RLIMIT_NOFILE'])),
    (resource.RLIMIT_NPROC, int(os.environ['AIPERF_EXPECT_RLIMIT_NPROC'])),
    (resource.RLIMIT_CPU, int(os.environ['AIPERF_EXPECT_RLIMIT_CPU'])),
]
for resource_id, expected in expected_limits:
    require(resource.getrlimit(resource_id) == (expected, expected), 46)
    try:
        resource.setrlimit(resource_id, (expected + 1, expected))
    except (OSError, ValueError):
        pass
    else:
        os._exit(54)
    require(resource.getrlimit(resource_id) == (expected, expected), 55)
require('AIPERF_EVALUATOR_BOOTSTRAP_PROCESS_LIMIT' not in os.environ, 56)

open_descriptors = []
for descriptor in range(64):
    try:
        os.fstat(descriptor)
        open_descriptors.append(descriptor)
    except OSError:
        pass
require(open_descriptors == [0, 1, 2, 3, 4], 47)
unexpected_descriptor = int(os.environ['AIPERF_UNEXPECTED_FD'])
require(unexpected_descriptor > 4, 52)
try:
    os.fstat(unexpected_descriptor)
except OSError:
    pass
else:
    os._exit(53)
os.set_inheritable(3, False)
os.set_inheritable(4, False)
descendant = """
import os
for descriptor in (3, 4):
    try:
        os.fstat(descriptor)
        raise SystemExit(1)
    except OSError:
        pass
raise SystemExit(0)
"""
descendant_pid = os.posix_spawn(
    '/usr/bin/python3',
    ['python3', '-S', '-c', descendant],
    os.environ,
)
_, descendant_status = os.waitpid(descendant_pid, 0)
require(os.waitstatus_to_exitcode(descendant_status) == 0, 48)

def denied_socket(family, address):
    try:
        stream = _socket.socket(family, _socket.SOCK_STREAM)
        stream.settimeout(0.5)
        stream.connect(address)
    except BaseException:
        try:
            stream.close()
        except BaseException:
            pass
        return
    stream.close()
    os._exit(49)

try:
    _socket.getaddrinfo('example.com', 443)
except BaseException:
    pass
else:
    os._exit(50)
denied_socket(_socket.AF_INET, ('1.1.1.1', 443))
denied_socket(_socket.AF_INET6, ('::1', 443))
denied_socket(_socket.AF_INET, ('127.0.0.1', int(os.environ['AIPERF_HOST_PORT'])))
denied_socket(_socket.AF_UNIX, '/run/aiperf/sibling-provider.sock')
denied_socket(_socket.AF_UNIX, '/var/run/docker.sock')
denied_socket(_socket.AF_UNIX, '/run/provider.sock')
require(not os.path.exists(os.environ['AIPERF_UNDECLARED_HOST_PATH']), 51)

try:
    stream = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    stream.connect('/run/aiperf/evaluator-proxy.sock')
    stream.sendall(b'I')
    reply = stream.recv(1)
    if reply == b'A':
        os.write(4, b'C')
        os._exit(0)
    os._exit(31)
except BaseException as error:
    os.write(2, (type(error).__name__ + ': ' + str(error)).encode())
    os._exit(33)
"#;
        std::fs::write(&acceptance_worker, script).unwrap();
        let secret = File::open(&host_secret_path).unwrap();
        let unexpected_fd = unsafe { libc::fcntl(secret.as_raw_fd(), libc::F_DUPFD_CLOEXEC, 5) };
        assert!(unexpected_fd >= 5);
        // SAFETY: successful `F_DUPFD_CLOEXEC` returned one new owned descriptor.
        let unexpected = unsafe { File::from_raw_fd(unexpected_fd) };
        drop(secret);
        let unexpected_fd = unexpected.as_raw_fd();
        let unexpected_flags = unsafe { libc::fcntl(unexpected_fd, libc::F_GETFD) };
        assert!(unexpected_flags >= 0);
        assert_eq!(
            unsafe {
                libc::fcntl(
                    unexpected_fd,
                    libc::F_SETFD,
                    unexpected_flags & !libc::FD_CLOEXEC,
                )
            },
            0
        );
        let launch = AttestedWorkerLaunch {
            distribution_id: EvaluationDistributionId::new("real-bwrap-fixture").unwrap(),
            program: contained_python.clone(),
            args: vec![
                OsString::from("-I"),
                OsString::from("-S"),
                OsString::from(
                    "/runtime/lib/python3.12/site-packages/aiperf/accuracy/evaluation/control_bootstrap.py",
                ),
            ],
            environment: BTreeMap::from([
                (OsString::from("PYTHONHOME"), OsString::from("/usr")),
                (
                    OsString::from("AIPERF_HOST_PORT"),
                    OsString::from(loopback_port.to_string()),
                ),
                (
                    OsString::from("AIPERF_UNDECLARED_HOST_PATH"),
                    host_secret_path.as_os_str().to_owned(),
                ),
                (
                    OsString::from("AIPERF_UNEXPECTED_FD"),
                    OsString::from(unexpected_fd.to_string()),
                ),
                (
                    OsString::from("AIPERF_EXPECT_RLIMIT_AS"),
                    OsString::from(
                        effective_resource_limit(libc::RLIMIT_AS, limits.address_space_bytes)
                            .to_string(),
                    ),
                ),
                (
                    OsString::from("AIPERF_EXPECT_RLIMIT_FSIZE"),
                    OsString::from(
                        effective_resource_limit(libc::RLIMIT_FSIZE, limits.file_size_bytes)
                            .to_string(),
                    ),
                ),
                (
                    OsString::from("AIPERF_EXPECT_RLIMIT_NOFILE"),
                    OsString::from(
                        effective_resource_limit(libc::RLIMIT_NOFILE, limits.open_files)
                            .to_string(),
                    ),
                ),
                (
                    OsString::from("AIPERF_EXPECT_RLIMIT_NPROC"),
                    OsString::from(
                        effective_resource_limit(libc::RLIMIT_NPROC, limits.processes).to_string(),
                    ),
                ),
                (
                    OsString::from("AIPERF_EXPECT_RLIMIT_CPU"),
                    OsString::from(
                        effective_resource_limit(libc::RLIMIT_CPU, limits.cpu_seconds).to_string(),
                    ),
                ),
            ]),
            current_dir: worker_root.clone(),
            worker_root: worker_root.clone(),
            closure: vec![
                LaunchClosureFile {
                    path: contained_python.clone(),
                    artifact_content_sha256: hash_file(&contained_python).unwrap(),
                },
                LaunchClosureFile {
                    path: control_bootstrap.clone(),
                    artifact_content_sha256: hash_file(&control_bootstrap).unwrap(),
                },
                LaunchClosureFile {
                    path: acceptance_worker.clone(),
                    artifact_content_sha256: hash_file(&acceptance_worker).unwrap(),
                },
            ],
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
                    verified_files: 3,
                },
                &context,
            )
            .unwrap();

        let (request_child, mut request_parent) = anonymous_pipe();
        let (mut response_parent, response_child) = anonymous_pipe();
        let prepared_limits = prepared.resource_limits;
        let mut command = Command::new(&prepared.program);
        command
            .args(&prepared.args)
            .env_clear()
            .envs(&prepared.environment)
            .current_dir(&prepared.current_dir)
            .stdin(Stdio::from(request_child))
            .stdout(Stdio::from(response_child))
            .stderr(Stdio::piped());
        // SAFETY: only async-signal-safe syscalls run between fork and exec.
        unsafe {
            command.pre_exec(move || {
                crate::supervisor::mark_unexpected_fds_close_on_exec()?;
                if libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                crate::supervisor::set_resource_limit(
                    libc::RLIMIT_AS,
                    prepared_limits.address_space_bytes,
                )?;
                crate::supervisor::set_resource_limit(
                    libc::RLIMIT_FSIZE,
                    prepared_limits.file_size_bytes,
                )?;
                crate::supervisor::set_resource_limit(
                    libc::RLIMIT_NOFILE,
                    prepared_limits.open_files,
                )?;
                crate::supervisor::set_resource_limit(
                    libc::RLIMIT_CPU,
                    prepared_limits.cpu_seconds,
                )?;
                Ok(())
            });
        }
        let mut isolated = command.spawn().unwrap();
        // `Command` retains its configured child pipe ends after `spawn`.
        // Releasing them makes an exact response EOF part of this acceptance
        // contract instead of an artifact masked by the test harness.
        drop(command);
        let root_pid = isolated.id();
        let mut process_tree = isolation.observe_process_tree(root_pid, None).unwrap();
        request_parent.write_all(b"R").unwrap();
        drop(request_parent);
        drop(unexpected);

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
                            process_tree = isolation
                                .observe_process_tree(root_pid, Some(&process_tree))
                                .unwrap();
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
        let mut control_output = Vec::new();
        response_parent.read_to_end(&mut control_output).unwrap();
        assert_eq!(isolated_authorized, Some(true));
        assert_eq!(sibling_authorized, Some(false));
        assert_eq!(control_output, b"C", "Bubblewrap wrote to protocol stdout");
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
        let quiescence_deadline = Instant::now() + Duration::from_secs(1);
        loop {
            match isolation.verify_quiescent(&process_tree) {
                Ok(_) => break,
                Err(_error) if Instant::now() < quiescence_deadline => {
                    std::thread::sleep(Duration::from_millis(1));
                }
                Err(error) => panic!("Bubblewrap subtree did not quiesce: {error}"),
            }
        }
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
            "linux-bubblewrap-rootfs-process-tree-v3"
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
