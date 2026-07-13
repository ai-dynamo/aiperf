// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Capability-gated archive object-store CAS seam and exact memory adapter.

use std::collections::BTreeMap;
use std::fmt::{self, Debug, Display, Formatter};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, RwLock};

use async_trait::async_trait;
use bytes::Bytes;

use crate::{Digest, ObjectVersionKind, StableObjectVersion, domain_digest};

/// Computes the exact-byte digest required by the archive object-store seam.
#[must_use]
pub fn archive_object_digest(bytes: &[u8]) -> Digest {
    domain_digest("aiperf.archive.object.v1", &[bytes])
}

/// Immutable named-object visibility semantics advertised by an adapter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NamedObjectVisibility {
    /// A successful write is immediately visible to named reads.
    Immediate,
    /// Missing reads may retry only within this explicit horizon.
    BoundedLag {
        /// Maximum authored consistency horizon.
        horizon_ns: u64,
    },
}

/// Capabilities that must be proved before authoritative remote resume.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArchiveStoreCapabilities {
    /// Immutable objects have atomic create-if-absent semantics.
    pub immutable_create_if_absent: bool,
    /// Named reads verify cryptographic exact-byte digests.
    pub exact_byte_verification: bool,
    /// Head updates are linearizable compare-and-swap operations.
    pub linearizable_head_cas: bool,
    /// Named-object read visibility contract.
    pub named_object_visibility: NamedObjectVisibility,
}

impl ArchiveStoreCapabilities {
    /// Validates all capabilities required for authoritative archive publication.
    pub fn require_authoritative(self) -> Result<(), ArchiveStoreError> {
        if !self.immutable_create_if_absent {
            return Err(ArchiveStoreError::MissingCapability(
                "immutable_create_if_absent",
            ));
        }
        if !self.exact_byte_verification {
            return Err(ArchiveStoreError::MissingCapability(
                "exact_byte_verification",
            ));
        }
        if !self.linearizable_head_cas {
            return Err(ArchiveStoreError::MissingCapability(
                "linearizable_head_cas",
            ));
        }
        if matches!(
            self.named_object_visibility,
            NamedObjectVisibility::BoundedLag { horizon_ns: 0 }
        ) {
            return Err(ArchiveStoreError::MissingCapability(
                "nonzero_named_object_visibility_horizon",
            ));
        }
        Ok(())
    }
}

/// Result of a create-if-absent operation, including a stable CAS version.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CreateReceipt {
    /// True when this call created the object; false for exact idempotent reuse.
    pub created: bool,
    /// Stable provider-neutral object version.
    pub version: StableObjectVersion,
}

/// Verified head bytes, digest, and stable CAS version.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VersionedHead {
    /// Exact head bytes.
    pub body: Bytes,
    /// Exact-byte object digest.
    pub digest: Digest,
    /// Stable CAS version.
    pub version: StableObjectVersion,
}

/// Narrow archive-store contract; provider SDK types never cross this boundary.
#[async_trait]
pub trait ArchiveObjectStore: Debug + Send + Sync {
    /// Returns currently proved adapter capabilities.
    fn capabilities(&self) -> ArchiveStoreCapabilities;

    /// Creates an immutable object or verifies byte-identical idempotent reuse.
    async fn put_if_absent(
        &self,
        key: &str,
        body: Bytes,
        digest: Digest,
    ) -> Result<CreateReceipt, ArchiveStoreError>;

    /// Reads one named object and verifies its exact-byte digest.
    async fn get_verified(&self, key: &str, expected: Digest) -> Result<Bytes, ArchiveStoreError>;

    /// Reads a head with the stable version required for CAS.
    async fn read_head(&self, key: &str) -> Result<Option<VersionedHead>, ArchiveStoreError>;

    /// Creates the first head only when absent.
    async fn create_head_if_absent(
        &self,
        key: &str,
        replacement: Bytes,
        digest: Digest,
    ) -> Result<CreateReceipt, HeadUpdateError>;

    /// Linearizably replaces an existing head at one exact version.
    async fn compare_and_swap_head(
        &self,
        key: &str,
        expected_version: &StableObjectVersion,
        replacement: Bytes,
        digest: Digest,
    ) -> Result<StableObjectVersion, HeadUpdateError>;
}

/// Named-object read/create failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArchiveStoreError {
    /// Object key is empty, absolute, or contains traversal/empty components.
    InvalidKey(String),
    /// Named object does not exist.
    NotFound(String),
    /// Supplied or returned bytes fail exact digest verification.
    DigestMismatch {
        /// Object key.
        key: String,
        /// Expected exact-byte digest.
        expected: Digest,
        /// Observed exact-byte digest.
        actual: Digest,
    },
    /// Create-if-absent found unequal bytes at the same key.
    AlreadyExistsDifferent(String),
    /// Adapter cannot prove one required capability.
    MissingCapability(&'static str),
    /// Adapter-local state lock was poisoned.
    AdapterState,
    /// Explicit transport/provider failure with no claim about mutation.
    Transport(String),
}

impl Display for ArchiveStoreError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidKey(key) => write!(formatter, "invalid archive object key {key:?}"),
            Self::NotFound(key) => write!(formatter, "archive object not found: {key}"),
            Self::DigestMismatch {
                key,
                expected,
                actual,
            } => write!(
                formatter,
                "archive object {key} digest mismatch: expected {expected}, found {actual}"
            ),
            Self::AlreadyExistsDifferent(key) => {
                write!(
                    formatter,
                    "archive object {key} already exists with unequal bytes"
                )
            }
            Self::MissingCapability(capability) => {
                write!(
                    formatter,
                    "archive store lacks required capability {capability}"
                )
            }
            Self::AdapterState => formatter.write_str("archive store adapter state is unavailable"),
            Self::Transport(message) => {
                write!(formatter, "archive store transport failure: {message}")
            }
        }
    }
}

impl std::error::Error for ArchiveStoreError {}

/// Conditional head-update result that distinguishes conflict from uncertainty.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HeadUpdateError {
    /// A verified different current head/version won the CAS.
    Conflict {
        /// Current head when readable.
        current: Option<VersionedHead>,
    },
    /// Transport outcome is uncertain; caller must reread and reconcile.
    Uncertain(String),
    /// Definite store failure.
    Store(ArchiveStoreError),
}

impl Display for HeadUpdateError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Conflict { .. } => formatter.write_str("archive head compare-and-swap conflict"),
            Self::Uncertain(message) => {
                write!(formatter, "archive head outcome uncertain: {message}")
            }
            Self::Store(error) => write!(formatter, "archive head update failed: {error}"),
        }
    }
}

impl std::error::Error for HeadUpdateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Store(error) => Some(error),
            _ => None,
        }
    }
}

/// Deterministic fault for testing uncertain conditional operations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryStoreFault {
    /// No fault.
    None,
    /// Next CAS fails uncertain before applying bytes.
    CasUncertainBeforeApply,
    /// Next CAS applies bytes, then returns uncertain.
    CasUncertainAfterApply,
    /// Next first-head create applies bytes, then returns uncertain.
    CreateHeadUncertainAfterApply,
}

#[derive(Clone, Debug)]
struct StoredObject {
    body: Bytes,
    digest: Digest,
    version: StableObjectVersion,
}

/// Exact in-memory adapter implementing the full create/verify/CAS contract.
#[derive(Debug)]
pub struct MemoryArchiveObjectStore {
    objects: RwLock<BTreeMap<String, StoredObject>>,
    next_version: AtomicU64,
    fault: Mutex<MemoryStoreFault>,
}

impl Default for MemoryArchiveObjectStore {
    fn default() -> Self {
        Self {
            objects: RwLock::new(BTreeMap::new()),
            next_version: AtomicU64::new(1),
            fault: Mutex::new(MemoryStoreFault::None),
        }
    }
}

impl MemoryArchiveObjectStore {
    /// Selects one deterministic fault consumed by the next matching operation.
    pub fn set_fault(&self, fault: MemoryStoreFault) -> Result<(), ArchiveStoreError> {
        *self
            .fault
            .lock()
            .map_err(|_| ArchiveStoreError::AdapterState)? = fault;
        Ok(())
    }

    /// Replaces bytes without updating digest/version for corruption-path tests.
    pub fn corrupt_for_test(&self, key: &str, replacement: Bytes) -> Result<(), ArchiveStoreError> {
        let mut objects = self
            .objects
            .write()
            .map_err(|_| ArchiveStoreError::AdapterState)?;
        let object = objects
            .get_mut(key)
            .ok_or_else(|| ArchiveStoreError::NotFound(key.to_owned()))?;
        object.body = replacement;
        Ok(())
    }

    fn next_stable_version(&self) -> StableObjectVersion {
        let sequence = self.next_version.fetch_add(1, Ordering::SeqCst);
        StableObjectVersion::new(
            "memory-archive-store-v1",
            ObjectVersionKind::Generation,
            sequence.to_be_bytes().to_vec(),
        )
        .expect("memory adapter version fields are nonempty")
    }

    fn consume_fault(&self, selected: MemoryStoreFault) -> Result<bool, ArchiveStoreError> {
        let mut fault = self
            .fault
            .lock()
            .map_err(|_| ArchiveStoreError::AdapterState)?;
        if *fault == selected {
            *fault = MemoryStoreFault::None;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn verified_head(key: &str, object: &StoredObject) -> Result<VersionedHead, ArchiveStoreError> {
        verify_digest(key, &object.body, object.digest)?;
        Ok(VersionedHead {
            body: object.body.clone(),
            digest: object.digest,
            version: object.version.clone(),
        })
    }
}

#[async_trait]
impl ArchiveObjectStore for MemoryArchiveObjectStore {
    fn capabilities(&self) -> ArchiveStoreCapabilities {
        ArchiveStoreCapabilities {
            immutable_create_if_absent: true,
            exact_byte_verification: true,
            linearizable_head_cas: true,
            named_object_visibility: NamedObjectVisibility::Immediate,
        }
    }

    async fn put_if_absent(
        &self,
        key: &str,
        body: Bytes,
        digest: Digest,
    ) -> Result<CreateReceipt, ArchiveStoreError> {
        validate_key(key)?;
        verify_digest(key, &body, digest)?;
        let mut objects = self
            .objects
            .write()
            .map_err(|_| ArchiveStoreError::AdapterState)?;
        if let Some(existing) = objects.get(key) {
            if existing.body != body || existing.digest != digest {
                return Err(ArchiveStoreError::AlreadyExistsDifferent(key.to_owned()));
            }
            return Ok(CreateReceipt {
                created: false,
                version: existing.version.clone(),
            });
        }
        let version = self.next_stable_version();
        objects.insert(
            key.to_owned(),
            StoredObject {
                body,
                digest,
                version: version.clone(),
            },
        );
        Ok(CreateReceipt {
            created: true,
            version,
        })
    }

    async fn get_verified(&self, key: &str, expected: Digest) -> Result<Bytes, ArchiveStoreError> {
        validate_key(key)?;
        let objects = self
            .objects
            .read()
            .map_err(|_| ArchiveStoreError::AdapterState)?;
        let object = objects
            .get(key)
            .ok_or_else(|| ArchiveStoreError::NotFound(key.to_owned()))?;
        verify_digest(key, &object.body, expected)?;
        Ok(object.body.clone())
    }

    async fn read_head(&self, key: &str) -> Result<Option<VersionedHead>, ArchiveStoreError> {
        validate_key(key)?;
        let objects = self
            .objects
            .read()
            .map_err(|_| ArchiveStoreError::AdapterState)?;
        objects
            .get(key)
            .map(|object| Self::verified_head(key, object))
            .transpose()
    }

    async fn create_head_if_absent(
        &self,
        key: &str,
        replacement: Bytes,
        digest: Digest,
    ) -> Result<CreateReceipt, HeadUpdateError> {
        validate_key(key).map_err(HeadUpdateError::Store)?;
        verify_digest(key, &replacement, digest).map_err(HeadUpdateError::Store)?;
        let mut objects = self
            .objects
            .write()
            .map_err(|_| HeadUpdateError::Store(ArchiveStoreError::AdapterState))?;
        if let Some(current) = objects.get(key) {
            return Err(HeadUpdateError::Conflict {
                current: Some(Self::verified_head(key, current).map_err(HeadUpdateError::Store)?),
            });
        }
        let version = self.next_stable_version();
        objects.insert(
            key.to_owned(),
            StoredObject {
                body: replacement,
                digest,
                version: version.clone(),
            },
        );
        if self
            .consume_fault(MemoryStoreFault::CreateHeadUncertainAfterApply)
            .map_err(HeadUpdateError::Store)?
        {
            return Err(HeadUpdateError::Uncertain(
                "injected after first-head creation".to_owned(),
            ));
        }
        Ok(CreateReceipt {
            created: true,
            version,
        })
    }

    async fn compare_and_swap_head(
        &self,
        key: &str,
        expected_version: &StableObjectVersion,
        replacement: Bytes,
        digest: Digest,
    ) -> Result<StableObjectVersion, HeadUpdateError> {
        validate_key(key).map_err(HeadUpdateError::Store)?;
        verify_digest(key, &replacement, digest).map_err(HeadUpdateError::Store)?;
        if self
            .consume_fault(MemoryStoreFault::CasUncertainBeforeApply)
            .map_err(HeadUpdateError::Store)?
        {
            return Err(HeadUpdateError::Uncertain(
                "injected before CAS application".to_owned(),
            ));
        }
        let mut objects = self
            .objects
            .write()
            .map_err(|_| HeadUpdateError::Store(ArchiveStoreError::AdapterState))?;
        let Some(current) = objects.get(key) else {
            return Err(HeadUpdateError::Conflict { current: None });
        };
        if &current.version != expected_version {
            return Err(HeadUpdateError::Conflict {
                current: Some(Self::verified_head(key, current).map_err(HeadUpdateError::Store)?),
            });
        }
        Self::verified_head(key, current).map_err(HeadUpdateError::Store)?;
        let version = self.next_stable_version();
        objects.insert(
            key.to_owned(),
            StoredObject {
                body: replacement,
                digest,
                version: version.clone(),
            },
        );
        if self
            .consume_fault(MemoryStoreFault::CasUncertainAfterApply)
            .map_err(HeadUpdateError::Store)?
        {
            return Err(HeadUpdateError::Uncertain(
                "injected after CAS application".to_owned(),
            ));
        }
        Ok(version)
    }
}

fn validate_key(key: &str) -> Result<(), ArchiveStoreError> {
    if key.is_empty()
        || key.starts_with('/')
        || key.ends_with('/')
        || key
            .split('/')
            .any(|component| component.is_empty() || component == "." || component == "..")
    {
        return Err(ArchiveStoreError::InvalidKey(key.to_owned()));
    }
    Ok(())
}

fn verify_digest(key: &str, body: &[u8], expected: Digest) -> Result<(), ArchiveStoreError> {
    let actual = archive_object_digest(body);
    if actual != expected {
        return Err(ArchiveStoreError::DigestMismatch {
            key: key.to_owned(),
            expected,
            actual,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn immutable_create_is_exactly_idempotent_and_never_overwrites() {
        let store = MemoryArchiveObjectStore::default();
        let body = Bytes::from_static(b"immutable");
        let digest = archive_object_digest(&body);
        let first = store
            .put_if_absent("parts/a", body.clone(), digest)
            .await
            .unwrap();
        assert!(first.created);
        let second = store
            .put_if_absent("parts/a", body.clone(), digest)
            .await
            .unwrap();
        assert!(!second.created);
        assert_eq!(first.version, second.version);
        let other = Bytes::from_static(b"other");
        assert!(matches!(
            store
                .put_if_absent("parts/a", other.clone(), archive_object_digest(&other))
                .await,
            Err(ArchiveStoreError::AlreadyExistsDifferent(_))
        ));
    }

    #[tokio::test]
    async fn first_head_creation_and_cas_are_linearizable() {
        let store = MemoryArchiveObjectStore::default();
        let first = Bytes::from_static(b"head-1");
        let receipt = store
            .create_head_if_absent("LATEST", first.clone(), archive_object_digest(&first))
            .await
            .unwrap();
        assert!(matches!(
            store
                .create_head_if_absent("LATEST", first.clone(), archive_object_digest(&first))
                .await,
            Err(HeadUpdateError::Conflict { .. })
        ));
        let second = Bytes::from_static(b"head-2");
        let second_version = store
            .compare_and_swap_head(
                "LATEST",
                &receipt.version,
                second.clone(),
                archive_object_digest(&second),
            )
            .await
            .unwrap();
        assert_ne!(receipt.version, second_version);
        assert!(matches!(
            store
                .compare_and_swap_head(
                    "LATEST",
                    &receipt.version,
                    first.clone(),
                    archive_object_digest(&first),
                )
                .await,
            Err(HeadUpdateError::Conflict { .. })
        ));
    }

    #[tokio::test]
    async fn uncertain_after_apply_is_resolved_only_by_verified_reread() {
        let store = MemoryArchiveObjectStore::default();
        let first = Bytes::from_static(b"head-1");
        let receipt = store
            .create_head_if_absent("LATEST", first.clone(), archive_object_digest(&first))
            .await
            .unwrap();
        let second = Bytes::from_static(b"head-2");
        store
            .set_fault(MemoryStoreFault::CasUncertainAfterApply)
            .unwrap();
        assert!(matches!(
            store
                .compare_and_swap_head(
                    "LATEST",
                    &receipt.version,
                    second.clone(),
                    archive_object_digest(&second),
                )
                .await,
            Err(HeadUpdateError::Uncertain(_))
        ));
        let reread = store.read_head("LATEST").await.unwrap().unwrap();
        assert_eq!(reread.body, second);
        assert_eq!(reread.digest, archive_object_digest(&reread.body));
    }

    #[tokio::test]
    async fn corruption_fails_exact_byte_verification() {
        let store = MemoryArchiveObjectStore::default();
        let body = Bytes::from_static(b"immutable");
        let digest = archive_object_digest(&body);
        store.put_if_absent("parts/a", body, digest).await.unwrap();
        store
            .corrupt_for_test("parts/a", Bytes::from_static(b"corrupt"))
            .unwrap();
        assert!(matches!(
            store.get_verified("parts/a", digest).await,
            Err(ArchiveStoreError::DigestMismatch { .. })
        ));
    }

    #[test]
    fn authoritative_capability_gate_fails_each_missing_proof() {
        let full = MemoryArchiveObjectStore::default().capabilities();
        full.require_authoritative().unwrap();
        for capabilities in [
            ArchiveStoreCapabilities {
                immutable_create_if_absent: false,
                ..full
            },
            ArchiveStoreCapabilities {
                exact_byte_verification: false,
                ..full
            },
            ArchiveStoreCapabilities {
                linearizable_head_cas: false,
                ..full
            },
            ArchiveStoreCapabilities {
                named_object_visibility: NamedObjectVisibility::BoundedLag { horizon_ns: 0 },
                ..full
            },
        ] {
            assert!(matches!(
                capabilities.require_authoritative(),
                Err(ArchiveStoreError::MissingCapability(_))
            ));
        }
    }

    #[tokio::test]
    async fn corrupted_current_head_never_masquerades_as_a_conflict() {
        let store = MemoryArchiveObjectStore::default();
        let first = Bytes::from_static(b"head-1");
        let receipt = store
            .create_head_if_absent("LATEST", first.clone(), archive_object_digest(&first))
            .await
            .unwrap();
        store
            .corrupt_for_test("LATEST", Bytes::from_static(b"corrupt"))
            .unwrap();
        let second = Bytes::from_static(b"head-2");
        assert!(matches!(
            store
                .compare_and_swap_head(
                    "LATEST",
                    &receipt.version,
                    second.clone(),
                    archive_object_digest(&second),
                )
                .await,
            Err(HeadUpdateError::Store(
                ArchiveStoreError::DigestMismatch { .. }
            ))
        ));
        assert!(matches!(
            store
                .create_head_if_absent("LATEST", second.clone(), archive_object_digest(&second))
                .await,
            Err(HeadUpdateError::Store(
                ArchiveStoreError::DigestMismatch { .. }
            ))
        ));
    }
}
