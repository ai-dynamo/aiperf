// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Content-neutral checkpoint result descriptors and budgeted read values.

use std::{
    mem::size_of,
    num::{NonZeroU64, NonZeroUsize},
};

use serde::{Deserialize, Deserializer, Serialize};

use super::{
    budget::BudgetLease,
    checkpoint::{BudgetedCheckpointBytes, CheckpointEpoch, CheckpointError, StreamRunIdentity},
    identity::{ContentDigest, GlobalSequence},
};

/// Stable cell coordinate attached to one result segment.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CellId(u32);

impl CellId {
    /// Construct a cell coordinate.
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Return the cell coordinate.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Stable worker coordinate attached to one result segment.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct WorkerId(u32);

impl WorkerId {
    /// Construct a worker coordinate.
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Return the worker coordinate.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Stable nonempty result projection identity.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct ResultProjectionId(Box<str>);

impl ResultProjectionId {
    /// Construct a compact nonempty projection identity.
    pub fn new(value: impl Into<String>) -> Result<Self, CheckpointError> {
        let value = value.into();
        if value.is_empty() {
            return Err(CheckpointError::ObjectVerification);
        }
        Ok(Self(value.into_boxed_str()))
    }

    /// Borrow the projection text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Return the compact nested allocation retained by the projection.
    #[must_use]
    pub fn retained_allocation_bytes(&self) -> usize {
        self.0.len()
    }
}

impl<'de> Deserialize<'de> for ResultProjectionId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

/// Version of the immutable result payload schema.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ResultSchemaVersion(u32);

impl ResultSchemaVersion {
    /// Construct a result schema version.
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Return the result schema version.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Immutable metadata for one committed result payload.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResultSegmentDescriptor {
    /// Logical run owning the segment.
    pub run: StreamRunIdentity,
    /// Result epoch owning the segment.
    pub epoch: CheckpointEpoch,
    /// Producing cell.
    pub cell_id: CellId,
    /// Producing worker.
    pub worker_id: WorkerId,
    /// Result projection identity.
    pub projection: ResultProjectionId,
    /// Result payload schema version.
    pub schema: ResultSchemaVersion,
    /// First global sequence represented by the payload.
    pub first_sequence: GlobalSequence,
    /// Last global sequence represented by the payload.
    pub last_sequence: GlobalSequence,
    /// Logical result item count.
    pub item_count: u64,
    /// Exact payload byte length.
    pub byte_length: u64,
    /// Digest of canonical logical membership.
    pub membership_root: ContentDigest,
    /// Raw BLAKE3 digest of the payload bytes.
    pub payload_digest: ContentDigest,
}

/// Singular result descriptor with inseparable retained-allocation authority.
///
/// ```compile_fail
/// # use aiperf_runtime::streaming::results::BudgetedResultDescriptor;
/// # fn cannot_separate(value: BudgetedResultDescriptor) {
/// let _descriptor = value.descriptor;
/// let _lease = value.lease;
/// # }
/// ```
///
/// ```compile_fail
/// # use aiperf_runtime::streaming::results::BudgetedResultDescriptor;
/// # fn cannot_use_backend_transfer(value: BudgetedResultDescriptor) {
/// let _ = value.into_backend_parts();
/// # }
/// ```
#[derive(Debug)]
pub struct BudgetedResultDescriptor {
    descriptor: ResultSegmentDescriptor,
    lease: BudgetLease,
}

impl BudgetedResultDescriptor {
    /// Bind one descriptor to its exact compact retained allocation charge.
    pub fn new(
        descriptor: ResultSegmentDescriptor,
        lease: BudgetLease,
    ) -> Result<Self, CheckpointError> {
        let bytes = descriptor_retained_bytes(&descriptor)?;
        if lease.charged_items() != 1 || lease.charged_bytes() != bytes {
            return Err(CheckpointError::ObjectVerification);
        }
        Ok(Self { descriptor, lease })
    }

    /// Borrow the checked descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &ResultSegmentDescriptor {
        &self.descriptor
    }

    /// Return the exact descriptor allocation charge.
    #[must_use]
    pub fn charged_bytes(&self) -> usize {
        self.lease.charged_bytes()
    }

    pub(crate) fn into_backend_parts(self) -> (ResultSegmentDescriptor, BudgetLease) {
        (self.descriptor, self.lease)
    }
}

/// Verified result payload and its inseparable budgeted descriptor.
///
/// ```compile_fail
/// # use aiperf_runtime::streaming::results::ResultPartition;
/// # fn cannot_separate(value: ResultPartition) {
/// let _descriptor = value.descriptor;
/// let _payload = value.payload;
/// # }
/// ```
#[derive(Debug)]
pub struct ResultPartition {
    descriptor: BudgetedResultDescriptor,
    payload: BudgetedCheckpointBytes,
}

impl ResultPartition {
    /// Verify a descriptor against its separately budgeted payload.
    pub fn new(
        descriptor: BudgetedResultDescriptor,
        payload: BudgetedCheckpointBytes,
    ) -> Result<Self, CheckpointError> {
        verify_payload(descriptor.descriptor(), &payload)?;
        Ok(Self {
            descriptor,
            payload,
        })
    }

    /// Borrow the verified descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &ResultSegmentDescriptor {
        self.descriptor.descriptor()
    }

    /// Return the descriptor's compact allocation charge.
    #[must_use]
    pub fn descriptor_charged_bytes(&self) -> usize {
        self.descriptor.charged_bytes()
    }

    /// Borrow the verified payload bytes.
    #[must_use]
    pub fn payload_bytes(&self) -> &[u8] {
        self.payload.as_bytes()
    }

    /// Move both separately budgeted values without dismantling either authority.
    #[must_use]
    pub fn into_parts(self) -> (BudgetedResultDescriptor, BudgetedCheckpointBytes) {
        (self.descriptor, self.payload)
    }
}

/// Descriptor collection with inseparable aggregate allocation authority.
///
/// ```compile_fail
/// # use aiperf_runtime::streaming::results::BudgetedResultDescriptors;
/// # fn cannot_separate(value: BudgetedResultDescriptors) {
/// let _descriptors = value.descriptors;
/// let _lease = value.lease;
/// # }
/// ```
#[derive(Debug)]
pub struct BudgetedResultDescriptors {
    descriptors: Box<[ResultSegmentDescriptor]>,
    lease: BudgetLease,
}

impl BudgetedResultDescriptors {
    pub(crate) fn new(
        descriptors: Box<[ResultSegmentDescriptor]>,
        lease: BudgetLease,
    ) -> Result<Self, CheckpointError> {
        let bytes = descriptors_retained_bytes(&descriptors)?;
        if lease.charged_items() != descriptors.len() || lease.charged_bytes() != bytes {
            return Err(CheckpointError::ObjectVerification);
        }
        Ok(Self { descriptors, lease })
    }

    /// Borrow all checked descriptors.
    #[must_use]
    pub fn descriptors(&self) -> &[ResultSegmentDescriptor] {
        &self.descriptors
    }

    /// Return the exact aggregate retained allocation charge.
    #[must_use]
    pub fn charged_bytes(&self) -> usize {
        self.lease.charged_bytes()
    }
}

/// Prepared result epoch returned by transaction staging.
#[derive(Debug)]
pub struct PreparedResultEpoch {
    index_root: ContentDigest,
    descriptors: BudgetedResultDescriptors,
    item_count: u64,
    byte_length: u64,
}

impl PreparedResultEpoch {
    pub(crate) fn new(
        index_root: ContentDigest,
        descriptors: BudgetedResultDescriptors,
        item_count: u64,
        byte_length: u64,
    ) -> Result<Self, CheckpointError> {
        let (computed_items, computed_bytes) = result_totals(descriptors.descriptors())?;
        if computed_items != item_count || computed_bytes != byte_length {
            return Err(CheckpointError::ObjectVerification);
        }
        Ok(Self {
            index_root,
            descriptors,
            item_count,
            byte_length,
        })
    }

    /// Borrow the canonical immutable index root.
    #[must_use]
    pub const fn index_root(&self) -> &ContentDigest {
        &self.index_root
    }

    /// Borrow the prepared descriptor inventory.
    #[must_use]
    pub fn descriptors(&self) -> &[ResultSegmentDescriptor] {
        self.descriptors.descriptors()
    }

    /// Return the aggregate logical item count.
    #[must_use]
    pub const fn item_count(&self) -> u64 {
        self.item_count
    }

    /// Return the aggregate payload byte length.
    #[must_use]
    pub const fn byte_length(&self) -> u64 {
        self.byte_length
    }

    /// Move the summary while preserving its descriptor allocation authority.
    #[must_use]
    pub fn into_parts(self) -> (ContentDigest, BudgetedResultDescriptors, u64, u64) {
        (
            self.index_root,
            self.descriptors,
            self.item_count,
            self.byte_length,
        )
    }
}

/// Verified budgeted reader for one immutable result payload.
#[derive(Debug)]
pub struct ResultSegmentReader {
    payload: BudgetedCheckpointBytes,
}

impl ResultSegmentReader {
    pub(crate) fn new(
        descriptor: &ResultSegmentDescriptor,
        payload: BudgetedCheckpointBytes,
    ) -> Result<Self, CheckpointError> {
        verify_payload(descriptor, &payload)?;
        Ok(Self { payload })
    }

    /// Borrow the verified result payload.
    #[must_use]
    pub fn payload_bytes(&self) -> &[u8] {
        self.payload.as_bytes()
    }

    /// Move the inseparable payload and read-budget authority.
    #[must_use]
    pub fn into_payload(self) -> BudgetedCheckpointBytes {
        self.payload
    }
}

/// Stable position within one immutable result-index block.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResultIndexCursor {
    /// Root whose reachability grants scan authority.
    pub root: ContentDigest,
    /// Reachable immutable block containing the offset.
    pub block: ContentDigest,
    /// Next item offset within the block.
    pub item_offset: u32,
}

/// Caller-owned upper bounds for one result-index page.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResultIndexReadBudget {
    /// Maximum descriptors returned in one page.
    pub max_items: NonZeroUsize,
    /// Maximum retained descriptor bytes returned in one page.
    pub max_bytes: NonZeroU64,
}

/// One budgeted page of reachable result descriptors.
///
/// ```compile_fail
/// # use aiperf_runtime::streaming::results::ResultIndexPage;
/// # fn cannot_separate(value: ResultIndexPage) {
/// let _descriptors = value.descriptors;
/// let _next = value.next;
/// # }
/// ```
#[derive(Debug)]
pub struct ResultIndexPage {
    descriptors: BudgetedResultDescriptors,
    next: Option<ResultIndexCursor>,
}

impl ResultIndexPage {
    pub(crate) fn new(
        descriptors: BudgetedResultDescriptors,
        next: Option<ResultIndexCursor>,
    ) -> Result<Self, CheckpointError> {
        u64::try_from(descriptors.charged_bytes())
            .map_err(|_| CheckpointError::ObjectVerification)?;
        Ok(Self { descriptors, next })
    }

    /// Borrow the reachable descriptors in this page.
    #[must_use]
    pub fn descriptors(&self) -> &[ResultSegmentDescriptor] {
        self.descriptors.descriptors()
    }

    /// Borrow the next cursor, when more descriptors are reachable.
    #[must_use]
    pub const fn next(&self) -> Option<&ResultIndexCursor> {
        self.next.as_ref()
    }

    /// Return the page's exact retained allocation charge.
    #[must_use]
    pub fn charged_bytes(&self) -> u64 {
        self.descriptors.charged_bytes() as u64
    }

    /// Move the page while preserving aggregate descriptor authority.
    #[must_use]
    pub fn into_parts(self) -> (BudgetedResultDescriptors, Option<ResultIndexCursor>) {
        (self.descriptors, self.next)
    }
}

pub(crate) fn descriptor_retained_bytes(
    descriptor: &ResultSegmentDescriptor,
) -> Result<usize, CheckpointError> {
    size_of::<ResultSegmentDescriptor>()
        .checked_add(descriptor.projection.retained_allocation_bytes())
        .ok_or(CheckpointError::ObjectVerification)
}

pub(crate) fn descriptors_retained_bytes(
    descriptors: &[ResultSegmentDescriptor],
) -> Result<usize, CheckpointError> {
    descriptors.iter().try_fold(0usize, |total, descriptor| {
        total
            .checked_add(descriptor_retained_bytes(descriptor)?)
            .ok_or(CheckpointError::ObjectVerification)
    })
}

pub(crate) fn result_totals(
    descriptors: &[ResultSegmentDescriptor],
) -> Result<(u64, u64), CheckpointError> {
    descriptors
        .iter()
        .try_fold((0u64, 0u64), |(items, bytes), descriptor| {
            Ok((
                items
                    .checked_add(descriptor.item_count)
                    .ok_or(CheckpointError::ObjectVerification)?,
                bytes
                    .checked_add(descriptor.byte_length)
                    .ok_or(CheckpointError::ObjectVerification)?,
            ))
        })
}

pub(crate) fn canonical_result_index_root(
    descriptors: &[ResultSegmentDescriptor],
) -> Result<ContentDigest, CheckpointError> {
    let encoded = serde_json::to_vec(descriptors).map_err(|error| CheckpointError::Storage {
        message: format!("could not encode result index: {error}"),
    })?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"aiperf.streaming.result-index.v1");
    hasher.update(&(encoded.len() as u64).to_le_bytes());
    hasher.update(&encoded);
    Ok(ContentDigest::from_bytes(*hasher.finalize().as_bytes()))
}

fn verify_payload(
    descriptor: &ResultSegmentDescriptor,
    payload: &BudgetedCheckpointBytes,
) -> Result<(), CheckpointError> {
    let byte_length =
        u64::try_from(payload.as_bytes().len()).map_err(|_| CheckpointError::ObjectVerification)?;
    let digest = ContentDigest::from_bytes(*blake3::hash(payload.as_bytes()).as_bytes());
    if payload.charged_bytes() != payload.as_bytes().len()
        || descriptor.byte_length != byte_length
        || descriptor.payload_digest != digest
    {
        return Err(CheckpointError::ObjectVerification);
    }
    Ok(())
}
