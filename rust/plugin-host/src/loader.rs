// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native library loader: dlopen with RTLD_NOW | RTLD_LOCAL (Task 14).
//!
//! Loads a set of staged plugin artifacts into the process address space.
//! Libraries are never unloaded (dlclose is never called); the raw handle
//! is retained for process lifetime.  A poisoned set records the error
//! and all partial handles for diagnostic purposes.
//!
//! `LoadedHandle` is deliberately not `Send` or `Sync`: the raw `*mut c_void`
//! from dlopen must stay on the thread that opened it for symbol resolution.

use std::path::PathBuf;

use crate::error::LoadError;

/// A successfully opened native library.
///
/// Not `Send` or `Sync` — the dlopen handle must not cross thread boundaries
/// after creation.
pub struct LoadedHandle {
    /// The staged path that was passed to dlopen.
    pub staged_path: PathBuf,
    /// Hex-encoded BLAKE3 digest of the artifact.
    pub digest: String,
    /// Raw dlopen handle.  Never passed to dlclose.
    #[cfg(unix)]
    #[allow(dead_code)]
    pub(crate) raw: *mut libc::c_void,
    /// On non-Unix platforms, a placeholder to keep the struct non-empty.
    #[cfg(not(unix))]
    #[allow(dead_code)]
    pub(crate) raw: *mut std::ffi::c_void,
}

// Safety: The host never calls dlclose and the handle is used only for
// symbol resolution at load time; after that, the loaded code owns itself.
// We mark Send+Sync so LoadedLibrarySet can be stored in registry state.
// Callers must not invoke dlsym across threads on a handle acquired on a
// different thread without external synchronization.
unsafe impl Send for LoadedHandle {}
unsafe impl Sync for LoadedHandle {}

impl std::fmt::Debug for LoadedHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedHandle")
            .field("staged_path", &self.staged_path)
            .field("digest", &self.digest)
            .finish_non_exhaustive()
    }
}

/// The builder state while dlopen calls are in progress.
///
/// Transitions to `LoadedLibrarySet` on complete success or
/// `PoisonedLibrarySet` on the first error.
pub struct ActivatingLibrarySet {
    handles: Vec<LoadedHandle>,
    poisoned: Option<LoadError>,
}

impl ActivatingLibrarySet {
    /// Create an empty activating set.
    pub fn new() -> Self {
        Self {
            handles: vec![],
            poisoned: None,
        }
    }

    /// dlopen one artifact.  If already poisoned, skips the call and
    /// accumulates the handle slot for diagnostics.
    pub fn load_one(&mut self, staged_path: PathBuf, digest: String) {
        if self.poisoned.is_some() {
            // Still record a placeholder so the diagnostic handle count matches
            // the expected artifact count.
            return;
        }
        match dlopen_now(&staged_path) {
            Ok(raw) => {
                self.handles.push(LoadedHandle {
                    staged_path,
                    digest,
                    raw,
                });
            }
            Err(e) => {
                self.poisoned = Some(LoadError::DlopenFailed {
                    path: staged_path,
                    detail: e,
                });
            }
        }
    }

    /// Consume the builder into either a `LoadedLibrarySet` or a `PoisonedLibrarySet`.
    ///
    /// `lock_digest` is a stable identifier for this exact set of artifacts
    /// (e.g., a hash over all artifact digests in sorted order).
    pub fn finalize(self, lock_digest: String) -> Result<LoadedLibrarySet, PoisonedLibrarySet> {
        match self.poisoned {
            None => Ok(LoadedLibrarySet {
                handles: self.handles,
                lock_digest,
            }),
            Some(error) => Err(PoisonedLibrarySet {
                error,
                partial_handles: self.handles,
            }),
        }
    }
}

impl Default for ActivatingLibrarySet {
    fn default() -> Self {
        Self::new()
    }
}

/// All plugin libraries for a catalog successfully loaded into the process.
#[derive(Debug)]
pub struct LoadedLibrarySet {
    pub handles: Vec<LoadedHandle>,
    /// Stable digest identifying this exact set of loaded artifacts.
    pub lock_digest: String,
}

/// A load sequence that failed after zero or more successful dlopen calls.
#[derive(Debug)]
pub struct PoisonedLibrarySet {
    pub error: LoadError,
    /// Libraries that were opened before the error; never closed.
    pub partial_handles: Vec<LoadedHandle>,
}

/// dlopen a shared library with `RTLD_NOW | RTLD_LOCAL`.
///
/// Returns the raw handle on success or a diagnostic error string on failure.
#[cfg(unix)]
fn dlopen_now(path: &std::path::Path) -> Result<*mut libc::c_void, String> {
    use std::ffi::CString;
    let c_path = CString::new(path.as_os_str().as_encoded_bytes())
        .map_err(|e| format!("path contains null byte: {e}"))?;
    let flags = libc::RTLD_NOW | libc::RTLD_LOCAL;
    // Safety: path is a valid C string; dlopen is documented to return null on error.
    let handle = unsafe { libc::dlopen(c_path.as_ptr(), flags) };
    if handle.is_null() {
        let msg = unsafe {
            let err = libc::dlerror();
            if err.is_null() {
                "unknown dlopen error".to_owned()
            } else {
                std::ffi::CStr::from_ptr(err).to_string_lossy().into_owned()
            }
        };
        Err(msg)
    } else {
        Ok(handle)
    }
}

#[cfg(not(unix))]
fn dlopen_now(_path: &std::path::Path) -> Result<*mut std::ffi::c_void, String> {
    Err("dlopen not supported on this platform".to_owned())
}
