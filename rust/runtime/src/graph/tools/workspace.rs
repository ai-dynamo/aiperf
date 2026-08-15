// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transportable workspace specifications and deterministic Pinch fixture staging.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path, PathBuf};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::dataset::{Handle, Payload, SegmentPool, SegmentStore};

use super::environment::TraceEnvironmentError;

/// Serializable filesystem image a worker materializes before opening a sandbox.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct WorkspaceSpec {
    /// Files to materialize in lexical destination order.
    pub files: Vec<WorkspaceFile>,
    /// Container-visible current working directory.
    pub workdir: String,
    /// Interpreter argv prefix used for each authored command.
    pub interpreter: Vec<String>,
    /// Whether the provisioned workspace is mounted into the sandbox.
    pub mount_workspace: bool,
    /// Per-command default deadline in nanoseconds.
    pub command_timeout_ns: u64,
}

impl WorkspaceSpec {
    /// Build an image-native workspace which intentionally has no host mount.
    pub fn image_native(
        workdir: impl Into<String>,
        interpreter: Vec<String>,
        timeout_ns: u64,
    ) -> Self {
        Self {
            files: Vec::new(),
            workdir: workdir.into(),
            interpreter,
            mount_workspace: false,
            command_timeout_ns: timeout_ns,
        }
    }
}

/// One digest-addressed file in a [`WorkspaceSpec`].
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct WorkspaceFile {
    /// Safe relative destination below the workspace root.
    pub destination: String,
    /// Raw-byte segment stored independently of the controller filesystem.
    pub content: Handle,
    /// Whether the materialized file must have an executable mode.
    pub is_executable: bool,
}

/// Worker-local result of materializing a workspace specification.
#[derive(Debug)]
pub struct ProvisionedWorkspace {
    /// Worker-owned workspace root, never serialized in a graph program.
    pub root: PathBuf,
    _owner: tempfile::TempDir,
}

/// Materializes an already-resolved workspace on the worker which owns it.
#[async_trait(?Send)]
pub trait WorkspaceProvisioner {
    /// Provision the exact staged content before trace measurement begins.
    async fn provision(
        &self,
        spec: &WorkspaceSpec,
    ) -> Result<ProvisionedWorkspace, TraceEnvironmentError>;
}

/// Worker-local materializer backed by the graph program's frozen segment store.
pub struct SegmentWorkspaceProvisioner<'a> {
    segments: &'a dyn SegmentStore,
}

impl<'a> SegmentWorkspaceProvisioner<'a> {
    /// Borrow the worker-visible content-addressed segment store.
    pub fn new(segments: &'a dyn SegmentStore) -> Self {
        Self { segments }
    }
}

#[async_trait(?Send)]
impl WorkspaceProvisioner for SegmentWorkspaceProvisioner<'_> {
    async fn provision(
        &self,
        spec: &WorkspaceSpec,
    ) -> Result<ProvisionedWorkspace, TraceEnvironmentError> {
        if !spec.mount_workspace {
            return Err(TraceEnvironmentError::new(
                "cannot provision a workspace whose mount policy is disabled",
            ));
        }
        let owner = tempfile::tempdir().map_err(|error| {
            TraceEnvironmentError::new(format!("cannot create worker workspace: {error}"))
        })?;
        for file in &spec.files {
            validate_relative(&file.destination, "workspace destination")?;
            let Payload::Raw { wire } = self.segments.get(file.content).map_err(|error| {
                TraceEnvironmentError::new(format!(
                    "cannot resolve workspace file {:?}: {error}",
                    file.destination
                ))
            })?
            else {
                return Err(TraceEnvironmentError::new(format!(
                    "workspace file {:?} does not reference a raw-byte segment",
                    file.destination
                )));
            };
            let destination = owner.path().join(&file.destination);
            if let Some(parent) = destination.parent() {
                fs::create_dir_all(parent).map_err(|error| {
                    TraceEnvironmentError::new(format!(
                        "cannot create workspace directory {:?}: {error}",
                        parent
                    ))
                })?;
            }
            fs::write(&destination, wire).map_err(|error| {
                TraceEnvironmentError::new(format!(
                    "cannot materialize workspace file {:?}: {error}",
                    file.destination
                ))
            })?;
            set_executable(&destination, file.is_executable)?;
        }
        Ok(ProvisionedWorkspace {
            root: owner.path().to_path_buf(),
            _owner: owner,
        })
    }
}

#[cfg(unix)]
fn set_executable(path: &Path, is_executable: bool) -> Result<(), TraceEnvironmentError> {
    use std::os::unix::fs::PermissionsExt as _;
    if !is_executable {
        return Ok(());
    }
    let mut permissions = fs::metadata(path)
        .map_err(|error| TraceEnvironmentError::new(error.to_string()))?
        .permissions();
    permissions.set_mode(permissions.mode() | 0o111);
    fs::set_permissions(path, permissions)
        .map_err(|error| TraceEnvironmentError::new(error.to_string()))
}

#[cfg(not(unix))]
fn set_executable(_path: &Path, _is_executable: bool) -> Result<(), TraceEnvironmentError> {
    Ok(())
}

/// One supported task-pack source entry for a Pinch workspace.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkspaceEntrySource {
    /// UTF-8 literal authored directly by a task definition.
    Literal {
        /// Destination relative to the workspace root.
        destination: String,
        /// Exact literal bytes.
        content: String,
    },
    /// A task-pack asset copied below a safe workspace destination.
    Asset {
        /// Asset-relative source path.
        source: String,
        /// Destination relative to the workspace root.
        destination: String,
    },
}

impl WorkspaceEntrySource {
    /// Construct one literal task fixture.
    pub fn literal(destination: impl Into<String>, content: impl Into<String>) -> Self {
        Self::Literal {
            destination: destination.into(),
            content: content.into(),
        }
    }

    /// Construct one rooted asset copy request.
    pub fn asset(source: impl Into<String>, destination: impl Into<String>) -> Self {
        Self::Asset {
            source: source.into(),
            destination: destination.into(),
        }
    }
}

/// Stages Pinch task-pack fixtures into the shared content-addressed segment pool.
pub struct PinchWorkspaceStager<'a> {
    root: &'a Path,
    segments: &'a mut SegmentPool,
}

impl<'a> PinchWorkspaceStager<'a> {
    /// Construct a stager rooted at one already-selected task pack.
    pub fn new(root: &'a Path, segments: &'a mut SegmentPool) -> Self {
        Self { root, segments }
    }

    /// Validate, read, and intern source entries without retaining host paths.
    pub fn stage(
        self,
        entries: impl IntoIterator<Item = WorkspaceEntrySource>,
    ) -> Result<WorkspaceSpec, TraceEnvironmentError> {
        let root = self.root.canonicalize().map_err(|error| {
            TraceEnvironmentError::new(format!("cannot canonicalize Pinch task-pack root: {error}"))
        })?;
        let mut files = BTreeMap::<String, (Vec<u8>, bool)>::new();
        for entry in entries {
            match entry {
                WorkspaceEntrySource::Literal {
                    destination,
                    content,
                } => insert_file(&mut files, destination, content.into_bytes(), false)?,
                WorkspaceEntrySource::Asset {
                    source,
                    destination,
                } => {
                    validate_relative(&source, "Pinch asset source")?;
                    validate_relative(&destination, "Pinch asset destination")?;
                    if source != "assets" && !source.starts_with("assets/") {
                        return Err(TraceEnvironmentError::new(format!(
                            "Pinch asset source {source:?} is outside the assets root"
                        )));
                    }
                    let source_path = root.join(&source);
                    let source_metadata = fs::symlink_metadata(&source_path).map_err(|error| {
                        TraceEnvironmentError::new(format!(
                            "cannot inspect Pinch asset {source:?}: {error}"
                        ))
                    })?;
                    if source_metadata.file_type().is_symlink() {
                        return Err(TraceEnvironmentError::new(format!(
                            "Pinch asset {source:?} is a symlink"
                        )));
                    }
                    let canonical_source = source_path.canonicalize().map_err(|error| {
                        TraceEnvironmentError::new(format!(
                            "cannot canonicalize Pinch asset {source:?}: {error}"
                        ))
                    })?;
                    if !canonical_source.starts_with(&root) {
                        return Err(TraceEnvironmentError::new(format!(
                            "Pinch asset {source:?} escapes its task-pack root"
                        )));
                    }
                    stage_asset_tree(&canonical_source, Path::new(&destination), &mut files)?;
                }
            }
        }
        let files = files
            .into_iter()
            .map(|(destination, (bytes, is_executable))| {
                let content = self
                    .segments
                    .intern_raw(None, bytes)
                    .map_err(|error| TraceEnvironmentError::new(error.to_string()))?;
                Ok(WorkspaceFile {
                    destination,
                    content,
                    is_executable,
                })
            })
            .collect::<Result<Vec<_>, TraceEnvironmentError>>()?;
        Ok(WorkspaceSpec {
            files,
            workdir: "/workspace".into(),
            interpreter: vec!["bash".into(), "-lc".into()],
            mount_workspace: true,
            command_timeout_ns: 30_000_000_000,
        })
    }
}

fn stage_asset_tree(
    source: &Path,
    destination: &Path,
    files: &mut BTreeMap<String, (Vec<u8>, bool)>,
) -> Result<(), TraceEnvironmentError> {
    let metadata = fs::symlink_metadata(source).map_err(|error| {
        TraceEnvironmentError::new(format!("cannot inspect Pinch asset {:?}: {error}", source))
    })?;
    if metadata.file_type().is_symlink() {
        return Err(TraceEnvironmentError::new(format!(
            "Pinch asset {:?} is a symlink",
            source
        )));
    }
    if metadata.is_file() {
        let destination = destination.to_string_lossy().into_owned();
        let bytes = fs::read(source).map_err(|error| {
            TraceEnvironmentError::new(format!("cannot read Pinch asset {:?}: {error}", source))
        })?;
        return insert_file(&mut *files, destination, bytes, is_executable(&metadata));
    }
    if !metadata.is_dir() {
        return Err(TraceEnvironmentError::new(format!(
            "Pinch asset {:?} is neither a file nor directory",
            source
        )));
    }
    let mut children = fs::read_dir(source)
        .map_err(|error| {
            TraceEnvironmentError::new(format!("cannot list Pinch asset {:?}: {error}", source))
        })?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| {
            TraceEnvironmentError::new(format!("cannot list Pinch asset {:?}: {error}", source))
        })?;
    children.sort_by_key(|entry| entry.file_name());
    for child in children {
        let child_name = child.file_name();
        stage_asset_tree(&child.path(), &destination.join(child_name), files)?;
    }
    Ok(())
}

fn insert_file(
    files: &mut BTreeMap<String, (Vec<u8>, bool)>,
    destination: String,
    bytes: Vec<u8>,
    is_executable: bool,
) -> Result<(), TraceEnvironmentError> {
    validate_relative(&destination, "Pinch workspace destination")?;
    if files
        .insert(destination.clone(), (bytes, is_executable))
        .is_some()
    {
        return Err(TraceEnvironmentError::new(format!(
            "duplicate Pinch workspace destination {destination:?}"
        )));
    }
    Ok(())
}

fn validate_relative(value: &str, field: &str) -> Result<(), TraceEnvironmentError> {
    let path = Path::new(value);
    if value.is_empty()
        || path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(TraceEnvironmentError::new(format!(
            "{field} {value:?} is not a safe relative path"
        )));
    }
    Ok(())
}

#[cfg(unix)]
fn is_executable(metadata: &fs::Metadata) -> bool {
    use std::os::unix::fs::PermissionsExt as _;
    metadata.permissions().mode() & 0o111 != 0
}

#[cfg(not(unix))]
fn is_executable(_metadata: &fs::Metadata) -> bool {
    false
}
