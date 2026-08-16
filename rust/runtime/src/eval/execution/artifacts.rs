// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Safe collection and transfer of explicitly declared benchmark artifacts.

use std::{
    collections::BTreeSet,
    fs,
    io::{self, Read, Write},
    os::unix::fs::PermissionsExt,
    path::{Component, Path, PathBuf},
    rc::Rc,
    time::Duration,
};

use tar::{Archive, EntryType};
use tempfile::NamedTempFile;

use crate::{clock::Clock, eval::ArtifactDigest};

use super::{
    ArtifactSpec, DockerRuntime, EvalExecutionError, EvalExecutionPhase, VerifierCollectHook,
    task_environment::{ServiceArchiveRequest, ServiceExecRequest, TaskEnvironmentLease},
};

/// A monotonic deadline shared by collection hooks and archive snapshots.
#[derive(Clone)]
pub(crate) struct Deadline {
    clock: Rc<dyn Clock>,
    started_ns: i64,
    timeout: Duration,
}

impl Deadline {
    pub(crate) fn from_timeout(clock: Rc<dyn Clock>, timeout: Duration) -> Self {
        Self {
            started_ns: clock.now_ns(),
            clock,
            timeout,
        }
    }

    fn remaining(&self) -> Result<Duration, EvalExecutionError> {
        let elapsed_ns = self.clock.now_ns().saturating_sub(self.started_ns).max(0) as u64;
        self.timeout
            .checked_sub(Duration::from_nanos(elapsed_ns))
            .filter(|remaining| !remaining.is_zero())
            .ok_or(EvalExecutionError::Timeout {
                phase: EvalExecutionPhase::CollectionHook,
                timeout: self.timeout,
            })
    }
}

/// Collects exactly the declared container files into a private host directory.
pub fn collect_artifacts(
    runtime: &dyn DockerRuntime,
    container: &str,
    artifacts: &[ArtifactSpec],
    destination: &Path,
) -> Result<Vec<(String, ArtifactDigest)>, EvalExecutionError> {
    fs::create_dir_all(destination).map_err(artifact_error)?;
    fs::set_permissions(destination, fs::Permissions::from_mode(0o700)).map_err(artifact_error)?;
    let mut collected = Vec::new();
    let mut destinations = BTreeSet::new();
    for artifact in artifacts {
        let archive = runtime.copy_archive(container, artifact.source())?;
        collect_archive(
            artifact,
            archive,
            destination,
            &mut destinations,
            &mut collected,
        )?;
    }
    collected.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(collected)
}

/// Collects declared files from their owning task services into one frozen snapshot.
pub(crate) fn collect_service_artifacts(
    lease: &mut dyn TaskEnvironmentLease,
    artifacts: &[ArtifactSpec],
    destination: &Path,
    deadline: Deadline,
) -> Result<Vec<(String, ArtifactDigest)>, EvalExecutionError> {
    fs::create_dir_all(destination).map_err(artifact_error)?;
    fs::set_permissions(destination, fs::Permissions::from_mode(0o700)).map_err(artifact_error)?;
    let mut collected = Vec::new();
    let mut destinations = BTreeSet::new();
    for artifact in artifacts {
        collect_one_service_artifact(
            lease,
            artifact,
            destination,
            deadline.clone(),
            &mut destinations,
            &mut collected,
        )?;
    }
    collected.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(collected)
}

/// Collects a terminal step's service evidence in its required shutdown order.
pub(crate) fn collect_service_evidence(
    lease: &mut dyn TaskEnvironmentLease,
    hooks: &[VerifierCollectHook],
    artifacts: &[ArtifactSpec],
    destination: &Path,
    deadline: Deadline,
) -> Result<Vec<(String, ArtifactDigest)>, EvalExecutionError> {
    let outcome = (|| {
        fs::create_dir_all(destination).map_err(artifact_error)?;
        fs::set_permissions(destination, fs::Permissions::from_mode(0o700))
            .map_err(artifact_error)?;
        let mut collected = Vec::new();
        let mut destinations = BTreeSet::new();
        let main = lease.main_service().clone();
        for hook in hooks.iter().filter(|hook| hook.service() == &main) {
            run_collection_hook(lease, hook, deadline.clone())?;
        }
        for artifact in artifacts
            .iter()
            .filter(|artifact| artifact.service_name() == &main)
        {
            collect_one_service_artifact(
                lease,
                artifact,
                destination,
                deadline.clone(),
                &mut destinations,
                &mut collected,
            )?;
        }
        let has_non_main = hooks.iter().any(|hook| hook.service() != &main)
            || artifacts
                .iter()
                .any(|artifact| artifact.service_name() != &main);
        if has_non_main {
            deadline.remaining()?;
            let remaining = deadline.remaining()?;
            match lease.stop_main(remaining) {
                Ok(()) => {}
                Err(EvalExecutionError::Timeout { timeout, .. }) => {
                    return Err(EvalExecutionError::Timeout {
                        phase: EvalExecutionPhase::CollectionHook,
                        timeout,
                    });
                }
                Err(error) => {
                    return Err(EvalExecutionError::CollectionHook {
                        service: main.as_str().to_owned(),
                        reason: error.to_string(),
                    });
                }
            }
        }
        for hook in hooks.iter().filter(|hook| hook.service() != &main) {
            run_collection_hook(lease, hook, deadline.clone())?;
        }
        for artifact in artifacts
            .iter()
            .filter(|artifact| artifact.service_name() != &main)
        {
            collect_one_service_artifact(
                lease,
                artifact,
                destination,
                deadline.clone(),
                &mut destinations,
                &mut collected,
            )?;
        }
        collected.sort_by(|left, right| left.0.cmp(&right.0));
        Ok(collected)
    })();
    if outcome.is_err() {
        let _ = lease.teardown_after_terminal_failure(Duration::from_secs(60));
    }
    outcome
}

fn collect_one_service_artifact(
    lease: &mut dyn TaskEnvironmentLease,
    artifact: &ArtifactSpec,
    destination: &Path,
    deadline: Deadline,
    destinations: &mut BTreeSet<String>,
    collected: &mut Vec<(String, ArtifactDigest)>,
) -> Result<(), EvalExecutionError> {
    deadline.remaining()?;
    let archive = lease.archive(ServiceArchiveRequest {
        service: artifact.service_name(),
        source: artifact.source(),
        deadline: deadline.remaining()?,
    })?;
    collect_archive(artifact, archive, destination, destinations, collected)?;
    deadline.remaining()?;
    Ok(())
}

/// Runs one declared collection hook without exposing phase secrets to the service.
pub(crate) fn run_collection_hook(
    lease: &mut dyn TaskEnvironmentLease,
    hook: &VerifierCollectHook,
    deadline: Deadline,
) -> Result<(), EvalExecutionError> {
    let timeout = hook.timeout().min(deadline.remaining()?);
    let request = ServiceExecRequest {
        service: hook.service(),
        arguments: hook.command(),
        public_environment: Default::default(),
        secret_environment: Default::default(),
        phase: EvalExecutionPhase::CollectionHook,
        user: hook.user(),
        workdir: None,
        network_lease: "default",
        deadline: Some(timeout),
    };
    match lease.exec(request) {
        Ok(()) => deadline.remaining().map(|_| ()),
        Err(EvalExecutionError::Timeout { timeout, .. }) => Err(EvalExecutionError::Timeout {
            phase: EvalExecutionPhase::CollectionHook,
            timeout,
        }),
        Err(error) => Err(EvalExecutionError::CollectionHook {
            service: hook.service().as_str().to_owned(),
            reason: error.to_string(),
        }),
    }
}

/// Copies a verified collection directory into an isolated verifier directory.
pub fn transfer_artifacts(
    source: &Path,
    destination: &Path,
    collected: &[(String, ArtifactDigest)],
) -> Result<(), EvalExecutionError> {
    fs::create_dir_all(destination).map_err(artifact_error)?;
    for (relative, digest) in collected {
        let relative = relative_path(relative)?;
        let source_path = safe_child(source, &relative)?;
        let metadata = fs::symlink_metadata(&source_path).map_err(artifact_error)?;
        if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
            return Err(EvalExecutionError::ArtifactCollection(format!(
                "collected artifact is not a regular file: {}",
                relative.display()
            )));
        }
        let mut source_file = fs::File::open(source_path).map_err(artifact_error)?;
        write_artifact_stream(
            destination,
            relative.to_string_lossy().as_ref(),
            &mut source_file,
            Some(digest),
        )?;
    }
    Ok(())
}

fn collect_archive(
    artifact: &ArtifactSpec,
    archive: Box<dyn Read>,
    destination: &Path,
    destinations: &mut BTreeSet<String>,
    collected: &mut Vec<(String, ArtifactDigest)>,
) -> Result<(), EvalExecutionError> {
    let source_name = Path::new(artifact.source())
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            EvalExecutionError::ArtifactCollection("invalid artifact source".to_owned())
        })?;
    let destination_root = artifact.destination().unwrap_or(source_name);
    let destination_root = relative_path(destination_root)?;
    let mut archive = Archive::new(archive);
    for entry in archive.entries().map_err(artifact_error)? {
        let mut entry = entry.map_err(artifact_error)?;
        let entry_type = entry.header().entry_type();
        if is_rejected_entry_type(entry_type) {
            return Err(EvalExecutionError::ArtifactCollection(
                "archive contains a link or special file".to_owned(),
            ));
        }
        if !entry_type.is_file() && !entry_type.is_dir() {
            return Err(EvalExecutionError::ArtifactCollection(
                "archive contains an unsupported entry".to_owned(),
            ));
        }
        let path = entry.path().map_err(artifact_error)?;
        let path = archive_relative(&path, source_name, artifact.is_exact_file())?;
        if path.as_os_str().is_empty() {
            if entry_type.is_dir() {
                continue;
            }
            return Err(EvalExecutionError::ArtifactCollection(
                "artifact archive has an empty file path".to_owned(),
            ));
        }
        if entry_type.is_dir() {
            continue;
        }
        let relative_source = path.to_string_lossy();
        if artifact
            .exclude()
            .iter()
            .any(|pattern| glob_matches(pattern, &relative_source))
        {
            continue;
        }
        let relative = if artifact.is_exact_file() {
            destination_root.clone()
        } else {
            destination_root.join(path)
        };
        let relative = relative_path(relative.to_string_lossy().as_ref())?;
        let key = relative.to_string_lossy().into_owned();
        if !destinations.insert(key.clone()) {
            return Err(EvalExecutionError::ArtifactCollection(format!(
                "duplicate artifact destination: {key}"
            )));
        }
        let digest = write_artifact_stream(destination, &key, &mut entry, None)?;
        collected.push((key, digest));
    }
    let mut archive = archive.into_inner();
    io::copy(&mut archive, &mut io::sink()).map_err(artifact_error)?;
    Ok(())
}

fn is_rejected_entry_type(entry_type: EntryType) -> bool {
    entry_type.is_symlink()
        || entry_type.is_hard_link()
        || entry_type.is_block_special()
        || entry_type.is_character_special()
        || entry_type.is_fifo()
}

fn archive_relative(
    path: &Path,
    source_name: &str,
    is_exact_file: bool,
) -> Result<PathBuf, EvalExecutionError> {
    let path = relative_path(path.to_string_lossy().as_ref())?;
    let mut components = path.components();
    let first = components.next();
    let remaining = components.as_path();
    let has_source_root =
        first.and_then(|component| component.as_os_str().to_str()) == Some(source_name);
    if is_exact_file {
        if !has_source_root || !remaining.as_os_str().is_empty() {
            return Err(EvalExecutionError::ArtifactCollection(
                "exact artifact archive must contain its requested regular file".to_owned(),
            ));
        }
        return Ok(PathBuf::from(source_name));
    }
    if !has_source_root {
        return Err(EvalExecutionError::ArtifactCollection(
            "directory artifact archive contains a member outside its declared source".to_owned(),
        ));
    }
    Ok(remaining.to_path_buf())
}

fn relative_path(path: &str) -> Result<PathBuf, EvalExecutionError> {
    let path = Path::new(path);
    if path.as_os_str().is_empty() || path.is_absolute() {
        return Err(EvalExecutionError::ArtifactCollection(
            "artifact destination must be a nonempty relative path".to_owned(),
        ));
    }
    if path
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(EvalExecutionError::ArtifactCollection(
            "artifact archive path escapes its collection root".to_owned(),
        ));
    }
    Ok(path.to_path_buf())
}

fn safe_child(root: &Path, relative: &Path) -> Result<PathBuf, EvalExecutionError> {
    let mut current = root.to_path_buf();
    for component in relative.components() {
        current.push(component);
        if let Ok(metadata) = fs::symlink_metadata(&current) {
            if metadata.file_type().is_symlink() {
                return Err(EvalExecutionError::ArtifactCollection(format!(
                    "artifact path contains a symlink: {}",
                    current.display()
                )));
            }
        }
    }
    Ok(current)
}

fn write_artifact_stream(
    root: &Path,
    relative: &str,
    source: &mut dyn Read,
    expected_digest: Option<&ArtifactDigest>,
) -> Result<ArtifactDigest, EvalExecutionError> {
    let relative = relative_path(relative)?;
    let parent = ensure_parent_directories(root, &relative)?;
    let path = safe_child(root, &relative)?;
    let mut temporary = NamedTempFile::new_in(parent).map_err(artifact_error)?;
    let digest = {
        let mut writer = HashingWriter::new(temporary.as_file_mut());
        io::copy(source, &mut writer).map_err(artifact_error)?;
        writer.flush().map_err(artifact_error)?;
        writer.finish()?
    };
    if expected_digest.is_some_and(|expected| expected != &digest) {
        return Err(EvalExecutionError::ArtifactCollection(format!(
            "collected artifact digest changed: {}",
            relative.display()
        )));
    }
    let file = temporary.persist_noclobber(path).map_err(|error| {
        EvalExecutionError::ArtifactCollection(format!(
            "artifact destination already exists: {} ({})",
            relative.display(),
            error.error
        ))
    })?;
    file.set_permissions(fs::Permissions::from_mode(0o644))
        .map_err(artifact_error)?;
    Ok(digest)
}

fn ensure_parent_directories(root: &Path, relative: &Path) -> Result<PathBuf, EvalExecutionError> {
    let mut parent = root.to_path_buf();
    let components = relative.components().collect::<Vec<_>>();
    for component in &components[..components.len().saturating_sub(1)] {
        parent.push(component);
        match fs::create_dir(&parent) {
            Ok(()) => fs::set_permissions(&parent, fs::Permissions::from_mode(0o755))
                .map_err(artifact_error)?,
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(artifact_error(error)),
        }
        let metadata = fs::symlink_metadata(&parent).map_err(artifact_error)?;
        if metadata.file_type().is_symlink() || !metadata.file_type().is_dir() {
            return Err(EvalExecutionError::ArtifactCollection(format!(
                "artifact parent is not a directory: {}",
                parent.display()
            )));
        }
    }
    Ok(parent)
}

struct HashingWriter<'a> {
    file: &'a mut fs::File,
    hasher: blake3::Hasher,
}

impl<'a> HashingWriter<'a> {
    fn new(file: &'a mut fs::File) -> Self {
        Self {
            file,
            hasher: blake3::Hasher::new(),
        }
    }

    fn finish(self) -> Result<ArtifactDigest, EvalExecutionError> {
        ArtifactDigest::parse(format!("blake3:{}", self.hasher.finalize().to_hex()))
            .map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))
    }
}

impl Write for HashingWriter<'_> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let count = self.file.write(buffer)?;
        self.hasher.update(&buffer[..count]);
        Ok(count)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

fn glob_matches(pattern: &str, path: &str) -> bool {
    glob_matches_bytes(pattern.as_bytes(), path.as_bytes())
}

fn glob_matches_bytes(pattern: &[u8], path: &[u8]) -> bool {
    match pattern.split_first() {
        None => path.is_empty(),
        Some((&b'*', remainder)) => {
            let recursive = remainder.first() == Some(&b'*');
            let remainder = if recursive {
                &remainder[1..]
            } else {
                remainder
            };
            (0..=path.len()).any(|index| {
                (recursive || !path[..index].contains(&b'/'))
                    && glob_matches_bytes(remainder, &path[index..])
            })
        }
        Some((&byte, remainder)) => {
            path.first() == Some(&byte) && glob_matches_bytes(remainder, &path[1..])
        }
    }
}

fn artifact_error(error: std::io::Error) -> EvalExecutionError {
    EvalExecutionError::ArtifactCollection(error.to_string())
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        io::{self, Read},
        rc::Rc,
        time::Duration,
    };

    use tempfile::tempdir;

    use crate::{
        clock::{RealClock, SimClock},
        eval::ComposeServiceName,
    };

    use super::*;
    use crate::eval::execution::task_environment::{
        ServiceArchiveRequest, ServiceExecRequest, ServiceHandle, TaskEnvironmentLease,
    };

    #[test]
    fn service_collection_uses_each_declared_service_archive() {
        let mut lease = RecordingLease::with_archives([
            ("main", tar_archive("main.txt", b"main")),
            ("api", tar_archive("api.txt", b"api")),
        ]);
        let artifacts = [
            ArtifactSpec::exact_file_for_service(
                "/workspace/main.txt".to_owned(),
                ComposeServiceName::main(),
            ),
            ArtifactSpec::exact_file_for_service(
                "/evidence/api.txt".to_owned(),
                ComposeServiceName::parse("api").unwrap(),
            ),
        ];
        let destination = tempdir().unwrap();

        let collected = collect_service_artifacts(
            &mut lease,
            &artifacts,
            destination.path(),
            Deadline::from_timeout(RealClock::new(), Duration::from_secs(5)),
        )
        .unwrap();

        assert_eq!(
            lease.archives,
            ["main:/workspace/main.txt", "api:/evidence/api.txt"]
        );
        assert_eq!(collected.len(), 2);
    }

    #[test]
    fn collection_hook_forwards_exact_argv_without_secrets() {
        let mut lease = RecordingLease::with_archives([]);
        let hook = VerifierCollectHook {
            service: ComposeServiceName::parse("api").unwrap(),
            command: vec!["collector".to_owned(), "--format=json".to_owned()],
            timeout: Duration::from_secs(3),
            user: Some("1000".to_owned()),
        };

        run_collection_hook(
            &mut lease,
            &hook,
            Deadline::from_timeout(RealClock::new(), Duration::from_secs(5)),
        )
        .unwrap();

        assert_eq!(
            lease.executions,
            [(
                "api".to_owned(),
                vec!["collector".to_owned(), "--format=json".to_owned()],
                Some("1000".to_owned()),
                0,
            )]
        );
    }

    #[test]
    fn expired_collection_window_runs_no_hook() {
        let mut lease = RecordingLease::with_archives([]);
        let hook = VerifierCollectHook {
            service: ComposeServiceName::main(),
            command: vec!["collector".to_owned()],
            timeout: Duration::from_secs(3),
            user: None,
        };

        let error = run_collection_hook(
            &mut lease,
            &hook,
            Deadline::from_timeout(RealClock::new(), Duration::ZERO),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            EvalExecutionError::Timeout {
                phase: EvalExecutionPhase::CollectionHook,
                ..
            }
        ));
        assert!(lease.executions.is_empty());
    }

    #[test]
    fn collection_timeout_uses_the_injected_virtual_clock() {
        let clock = Rc::new(SimClock::new());
        let mut lease = RecordingLease::with_archives([]);
        lease.advance_after_hook_to = Some((clock.clone(), 5));
        let hook = hook("main", ["collector"]);
        let deadline = Deadline::from_timeout(clock.clone(), Duration::from_nanos(5));

        let error = run_collection_hook(&mut lease, &hook, deadline).unwrap_err();

        assert!(matches!(
            error,
            EvalExecutionError::Timeout {
                phase: EvalExecutionPhase::CollectionHook,
                timeout,
            } if timeout == Duration::from_nanos(5)
        ));
        assert_eq!(lease.executions.len(), 1);
    }

    #[test]
    fn provider_hook_timeout_is_classified_as_collection_work() {
        let mut lease = RecordingLease::with_archives([]);
        lease.timeout_hook_service = Some("main".to_owned());
        let hook = hook("main", ["collector"]);

        let error = run_collection_hook(
            &mut lease,
            &hook,
            Deadline::from_timeout(RealClock::new(), Duration::from_secs(5)),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            EvalExecutionError::Timeout {
                phase: EvalExecutionPhase::CollectionHook,
                ..
            }
        ));
    }

    #[test]
    fn terminal_evidence_stops_main_before_sidecar_and_tears_down_on_hook_failure() {
        let mut lease = RecordingLease::with_archives([
            ("main", tar_archive("main.txt", b"main")),
            ("api", tar_archive("api.txt", b"api")),
        ]);
        lease.fail_hook_service = Some("main".to_owned());
        let hooks = [hook("main", ["main-hook"]), hook("api", ["api-hook"])];
        let artifacts = [
            ArtifactSpec::exact_file_for_service(
                "/workspace/main.txt".to_owned(),
                ComposeServiceName::main(),
            ),
            ArtifactSpec::exact_file_for_service(
                "/evidence/api.txt".to_owned(),
                ComposeServiceName::parse("api").unwrap(),
            ),
        ];
        let destination = tempdir().unwrap();

        let error = collect_service_evidence(
            &mut lease,
            &hooks,
            &artifacts,
            destination.path(),
            Deadline::from_timeout(RealClock::new(), Duration::from_secs(5)),
        )
        .unwrap_err();

        assert!(matches!(error, EvalExecutionError::CollectionHook { .. }));
        assert_eq!(
            lease.events,
            ["hook:main", "terminal-teardown:60000000000"]
        );
    }

    #[test]
    fn terminal_evidence_orders_main_then_sidecar_services() {
        let mut lease = RecordingLease::with_archives([
            ("main", tar_archive("main.txt", b"main")),
            ("api", tar_archive("api.txt", b"api")),
        ]);
        let hooks = [hook("main", ["main-hook"]), hook("api", ["api-hook"])];
        let artifacts = [
            ArtifactSpec::exact_file_for_service(
                "/workspace/main.txt".to_owned(),
                ComposeServiceName::main(),
            ),
            ArtifactSpec::exact_file_for_service(
                "/evidence/api.txt".to_owned(),
                ComposeServiceName::parse("api").unwrap(),
            ),
        ];
        let destination = tempdir().unwrap();

        collect_service_evidence(
            &mut lease,
            &hooks,
            &artifacts,
            destination.path(),
            Deadline::from_timeout(RealClock::new(), Duration::from_secs(5)),
        )
        .unwrap();

        assert_eq!(
            lease.events,
            [
                "hook:main",
                "archive:main:/workspace/main.txt",
                "stop:main",
                "hook:api",
                "archive:api:/evidence/api.txt",
            ]
        );
    }

    #[test]
    fn stopping_main_uses_the_remaining_collection_deadline() {
        let mut lease = RecordingLease::with_archives([]);
        let hooks = [hook("api", ["api-hook"])];
        let destination = tempdir().unwrap();

        collect_service_evidence(
            &mut lease,
            &hooks,
            &[],
            destination.path(),
            Deadline::from_timeout(RealClock::new(), Duration::from_secs(5)),
        )
        .unwrap();

        assert_eq!(lease.stop_deadlines.len(), 1);
        assert!(lease.stop_deadlines[0] <= Duration::from_secs(5));
        assert!(!lease.stop_deadlines[0].is_zero());
    }

    #[test]
    fn stop_timeout_prevents_sidecar_evidence_and_tears_down() {
        let mut lease = RecordingLease::with_archives([]);
        lease.timeout_stop = true;
        let hooks = [hook("api", ["api-hook"])];
        let destination = tempdir().unwrap();

        let error = collect_service_evidence(
            &mut lease,
            &hooks,
            &[],
            destination.path(),
            Deadline::from_timeout(RealClock::new(), Duration::from_secs(5)),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            EvalExecutionError::Timeout {
                phase: EvalExecutionPhase::CollectionHook,
                ..
            }
        ));
        assert_eq!(
            lease.events,
            ["stop:main", "terminal-teardown:60000000000"]
        );
    }

    #[test]
    fn archive_timeout_prevents_later_service_work_and_tears_down() {
        let mut lease = RecordingLease::with_archives([
            ("main", tar_archive("main.txt", b"main")),
            ("api", tar_archive("api.txt", b"api")),
        ]);
        lease.timeout_archive_service = Some("main".to_owned());
        let hooks = [hook("main", ["main-hook"]), hook("api", ["api-hook"])];
        let artifacts = [
            ArtifactSpec::exact_file_for_service(
                "/workspace/main.txt".to_owned(),
                ComposeServiceName::main(),
            ),
            ArtifactSpec::exact_file_for_service(
                "/evidence/api.txt".to_owned(),
                ComposeServiceName::parse("api").unwrap(),
            ),
        ];
        let destination = tempdir().unwrap();

        let error = collect_service_evidence(
            &mut lease,
            &hooks,
            &artifacts,
            destination.path(),
            Deadline::from_timeout(RealClock::new(), Duration::from_secs(5)),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            EvalExecutionError::Timeout {
                phase: EvalExecutionPhase::CollectionHook,
                ..
            }
        ));
        assert_eq!(
            lease.events,
            [
                "hook:main",
                "archive:main:/workspace/main.txt",
                "terminal-teardown:60000000000",
            ]
        );
    }

    fn hook<const N: usize>(service: &str, command: [&str; N]) -> VerifierCollectHook {
        VerifierCollectHook {
            service: ComposeServiceName::parse(service).unwrap(),
            command: command.into_iter().map(str::to_owned).collect(),
            timeout: Duration::from_secs(3),
            user: None,
        }
    }

    fn tar_archive(path: &str, contents: &[u8]) -> Vec<u8> {
        let mut archive = Vec::new();
        let mut builder = tar::Builder::new(&mut archive);
        let mut header = tar::Header::new_gnu();
        header.set_size(contents.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder.append_data(&mut header, path, contents).unwrap();
        builder.finish().unwrap();
        drop(builder);
        archive
    }

    struct RecordingLease {
        main: ComposeServiceName,
        archives: Vec<String>,
        executions: Vec<(String, Vec<String>, Option<String>, usize)>,
        events: Vec<String>,
        fail_hook_service: Option<String>,
        timeout_hook_service: Option<String>,
        timeout_archive_service: Option<String>,
        advance_after_hook_to: Option<(Rc<SimClock>, i64)>,
        stop_deadlines: Vec<Duration>,
        timeout_stop: bool,
        bytes: BTreeMap<String, Vec<u8>>,
    }

    impl RecordingLease {
        fn with_archives(entries: impl IntoIterator<Item = (&'static str, Vec<u8>)>) -> Self {
            Self {
                main: ComposeServiceName::main(),
                archives: Vec::new(),
                executions: Vec::new(),
                events: Vec::new(),
                fail_hook_service: None,
                timeout_hook_service: None,
                timeout_archive_service: None,
                advance_after_hook_to: None,
                stop_deadlines: Vec::new(),
                timeout_stop: false,
                bytes: entries
                    .into_iter()
                    .map(|(service, archive)| (service.to_owned(), archive))
                    .collect(),
            }
        }
    }

    impl TaskEnvironmentLease for RecordingLease {
        fn main_service(&self) -> &ComposeServiceName {
            &self.main
        }
        fn service(&self, name: &ComposeServiceName) -> Result<ServiceHandle, EvalExecutionError> {
            Ok(ServiceHandle::new(name.clone()))
        }
        fn exec(&mut self, request: ServiceExecRequest<'_>) -> Result<(), EvalExecutionError> {
            self.events
                .push(format!("hook:{}", request.service.as_str()));
            self.executions.push((
                request.service.as_str().to_owned(),
                request.arguments.to_vec(),
                request.user.map(str::to_owned),
                request.secret_environment.len(),
            ));
            if let Some((clock, time_ns)) = &self.advance_after_hook_to {
                clock.advance_to(*time_ns);
            }
            if self.timeout_hook_service.as_deref() == Some(request.service.as_str()) {
                return Err(EvalExecutionError::Timeout {
                    phase: EvalExecutionPhase::Agent,
                    timeout: request.deadline.unwrap_or_default(),
                });
            }
            if self.fail_hook_service.as_deref() == Some(request.service.as_str()) {
                return Err(EvalExecutionError::ProcessFailure("hook failed".to_owned()));
            }
            Ok(())
        }
        fn archive(
            &mut self,
            request: ServiceArchiveRequest<'_>,
        ) -> Result<Box<dyn Read>, EvalExecutionError> {
            self.archives
                .push(format!("{}:{}", request.service.as_str(), request.source));
            self.events.push(format!(
                "archive:{}:{}",
                request.service.as_str(),
                request.source
            ));
            if self.timeout_archive_service.as_deref() == Some(request.service.as_str()) {
                return Err(EvalExecutionError::Timeout {
                    phase: EvalExecutionPhase::CollectionHook,
                    timeout: request.deadline,
                });
            }
            let bytes = self
                .bytes
                .get(request.service.as_str())
                .cloned()
                .ok_or_else(|| {
                    EvalExecutionError::ArtifactCollection("missing archive".to_owned())
                })?;
            Ok(Box::new(io::Cursor::new(bytes)))
        }
        fn copy_into(
            &mut self,
            service: &ComposeServiceName,
            source: &str,
            destination: &str,
        ) -> Result<(), EvalExecutionError> {
            self.events
                .push(format!("copy:{}:{source}:{destination}", service.as_str()));
            Ok(())
        }
        fn stop_main(&mut self, deadline: Duration) -> Result<(), EvalExecutionError> {
            self.stop_deadlines.push(deadline);
            self.events.push("stop:main".to_owned());
            if self.timeout_stop {
                return Err(EvalExecutionError::Timeout {
                    phase: EvalExecutionPhase::Agent,
                    timeout: deadline,
                });
            }
            Ok(())
        }
        fn main_image_id(&self) -> Result<&str, EvalExecutionError> {
            Ok("image")
        }
        fn teardown(&mut self) -> Result<(), EvalExecutionError> {
            self.events.push("teardown".to_owned());
            Ok(())
        }
        fn teardown_after_terminal_failure(
            &mut self,
            deadline: Duration,
        ) -> Result<(), EvalExecutionError> {
            self.events
                .push(format!("terminal-teardown:{}", deadline.as_nanos()));
            Ok(())
        }
    }
}
