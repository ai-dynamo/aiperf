// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! AWS S3 implementation of the conditional checkpoint object store.
//!
//! Object versions are ETags rather than bucket version ids, so the store works
//! against an unversioned bucket and the pointer compare-and-swap maps directly
//! onto S3's own conditional-write headers: `If-None-Match: *` creates the first
//! pointer and `If-Match: <etag>` replaces exactly the revision the writer
//! observed. A provider that answers `501 Not Implemented` to those headers has
//! no exact conditional update and fails capability agreement before any effect.
//!
//! Uploads stream. An object at or below the single-put threshold is sent in one
//! request; anything larger goes through multipart upload with each part bounded
//! by the configured part size, so a multi-gibibyte object is never assembled in
//! one buffer. Restores are ranged reads bounded by the caller's chunk budget.
//!
//! This module consumes only the neutral AWS client construction seam in
//! [`crate::streaming::aws`]. It never imports the S3 streaming *source*: the
//! checkpoint plane must not become a second, weaker path to source discovery.

use std::{num::NonZeroUsize, rc::Rc};

use async_trait::async_trait;
use aws_sdk_s3::{
    Client,
    error::SdkError,
    primitives::ByteStream,
    types::{CompletedMultipartUpload, CompletedPart},
};
use bytes::Bytes;

use crate::{
    clock::Clock,
    streaming::{
        aws::{AwsClientSettings, AwsClockProjection, AwsS3ClientFactory},
        checkpoint::CheckpointError,
        budget::{BudgetLimits, StreamingResourceBudget},
        checkpoints::object_store::{
            BudgetOwnedObjectChunk, BudgetOwnedObjectPage, BudgetOwnedObjectReader,
            ConditionalObjectStore, ObjectKey, ObjectListBudget, ObjectListCursor, ObjectMetadata,
            ObjectReadBudget, ObjectReadRange, ObjectVersion, PointerObject,
            conditional_write_unsupported_error, immutable_object_key,
            object_limit_exceeded_error, provider_error, stale_writer_error,
        },
    },
};

/// Smallest part size S3 accepts for any multipart part except the last.
const MINIMUM_MULTIPART_PART_BYTES: usize = 5 * 1024 * 1024;

/// HTTP status S3-compatible providers use to refuse conditional-write headers.
const NOT_IMPLEMENTED_STATUS: u16 = 501;

/// HTTP status returned when a conditional pointer write loses its race.
const PRECONDITION_FAILED_STATUS: u16 = 412;

/// Status returned when a concurrent conditional write is already in flight.
const CONFLICT_STATUS: u16 = 409;

/// Validated bucket, threshold, and part sizing for one S3 checkpoint store.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AwsObjectStoreSettings {
    /// Bucket owning every checkpoint object.
    pub bucket: String,
    /// Checkpoint prefix every object address derives from.
    ///
    /// This must equal the prefix the backend was constructed with: immutable
    /// object addresses are a pure function of it and the content digest.
    pub prefix: ObjectKey,
    /// Bytes this store may retain across concurrently leased chunks and pages.
    pub max_retained_bytes: NonZeroUsize,
    /// Chunks and pages this store may retain concurrently.
    pub max_retained_items: NonZeroUsize,
    /// Largest object sent in one `PutObject` request.
    pub single_put_threshold_bytes: NonZeroUsize,
    /// Bytes buffered per multipart part above that threshold.
    pub multipart_part_bytes: NonZeroUsize,
}

impl AwsObjectStoreSettings {
    /// Reject every authored value S3 cannot honor, before any client is built.
    pub fn validate(&self) -> Result<(), CheckpointError> {
        if self.bucket.is_empty() {
            return Err(provider_error("bucket must not be empty"));
        }
        if self.multipart_part_bytes.get() < MINIMUM_MULTIPART_PART_BYTES {
            return Err(provider_error(
                "multipart part size must be at least 5 MiB",
            ));
        }
        Ok(())
    }
}

/// Conditional checkpoint object store backed by AWS S3.
pub struct AwsConditionalObjectStore {
    client: Client,
    settings: AwsObjectStoreSettings,
    // Chunks and pages this store hands back own permits until dropped, so the
    // store owns the budget those permits are drawn from.
    retention: StreamingResourceBudget,
    // The projection binds the run clock to the credential authority's shared
    // time cell and must outlive the client it was built with.
    projection: AwsClockProjection,
}

impl std::fmt::Debug for AwsConditionalObjectStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AwsConditionalObjectStore")
            .field("settings", &self.settings)
            .finish()
    }
}

impl AwsConditionalObjectStore {
    /// Build one worker-local store from the neutral AWS client factory.
    pub fn new(
        factory: &AwsS3ClientFactory,
        clock: Rc<dyn Clock>,
        settings: AwsObjectStoreSettings,
    ) -> Result<Self, CheckpointError> {
        settings.validate()?;
        let retention = StreamingResourceBudget::new(BudgetLimits {
            max_items: settings.max_retained_items.get(),
            max_bytes: settings.max_retained_bytes.get(),
        })
        .map_err(|_| provider_error("invalid object store retention budget"))?;
        let (client, projection) = factory.build_client(clock);
        Ok(Self {
            client,
            settings,
            retention,
            projection,
        })
    }

    fn begin_operation(&self) {
        self.projection.publish();
    }

    async fn put_single(
        &self,
        key: &ObjectKey,
        bytes: Bytes,
    ) -> Result<ObjectVersion, CheckpointError> {
        self.begin_operation();
        let response = self
            .client
            .put_object()
            .bucket(&self.settings.bucket)
            .key(key.as_str())
            .body(ByteStream::from(bytes))
            .send()
            .await
            .map_err(|error| map_sdk_error("put object", &error))?;
        etag_version(response.e_tag())
    }

    async fn put_multipart(
        &self,
        key: &ObjectKey,
        mut object: Box<dyn BudgetOwnedObjectReader>,
        first: BudgetOwnedObjectChunk,
    ) -> Result<ObjectVersion, CheckpointError> {
        self.begin_operation();
        let created = self
            .client
            .create_multipart_upload()
            .bucket(&self.settings.bucket)
            .key(key.as_str())
            .send()
            .await
            .map_err(|error| map_sdk_error("create multipart upload", &error))?;
        let upload_id = created
            .upload_id()
            .ok_or_else(|| provider_error("multipart upload id absent"))?
            .to_string();
        let result = self
            .upload_all_parts(key, &upload_id, &mut object, first)
            .await;
        match result {
            Ok(parts) => self.complete_multipart(key, &upload_id, parts).await,
            Err(error) => {
                // Abort eagerly: an abandoned upload retains billable parts that
                // no committed pointer will ever reference.
                self.begin_operation();
                let _ = self
                    .client
                    .abort_multipart_upload()
                    .bucket(&self.settings.bucket)
                    .key(key.as_str())
                    .upload_id(&upload_id)
                    .send()
                    .await;
                Err(error)
            }
        }
    }

    async fn upload_all_parts(
        &self,
        key: &ObjectKey,
        upload_id: &str,
        object: &mut Box<dyn BudgetOwnedObjectReader>,
        first: BudgetOwnedObjectChunk,
    ) -> Result<Vec<CompletedPart>, CheckpointError> {
        let part_bytes = self.settings.multipart_part_bytes.get();
        let mut parts = Vec::new();
        let mut buffer: Vec<u8> = Vec::with_capacity(part_bytes);
        let mut pending = Some(first);
        let mut part_number = 1i32;
        loop {
            let chunk = match pending.take() {
                Some(chunk) => Some(chunk),
                None => object.next_chunk(part_bytes).await?,
            };
            match chunk {
                Some(chunk) => {
                    buffer.extend_from_slice(&chunk.bytes);
                    // Dropping the chunk releases its permit as soon as its bytes
                    // have been copied into the bounded part buffer.
                    drop(chunk);
                    if buffer.len() >= part_bytes {
                        parts.push(self.upload_part(key, upload_id, part_number, &mut buffer).await?);
                        part_number += 1;
                    }
                }
                None => {
                    if !buffer.is_empty() || parts.is_empty() {
                        parts.push(self.upload_part(key, upload_id, part_number, &mut buffer).await?);
                    }
                    return Ok(parts);
                }
            }
        }
    }

    async fn upload_part(
        &self,
        key: &ObjectKey,
        upload_id: &str,
        part_number: i32,
        buffer: &mut Vec<u8>,
    ) -> Result<CompletedPart, CheckpointError> {
        let body = Bytes::from(std::mem::take(buffer).into_boxed_slice());
        self.begin_operation();
        let uploaded = self
            .client
            .upload_part()
            .bucket(&self.settings.bucket)
            .key(key.as_str())
            .upload_id(upload_id)
            .part_number(part_number)
            .body(ByteStream::from(body))
            .send()
            .await
            .map_err(|error| map_sdk_error("upload part", &error))?;
        let e_tag = uploaded
            .e_tag()
            .ok_or_else(|| provider_error("uploaded part has no entity tag"))?;
        Ok(CompletedPart::builder()
            .part_number(part_number)
            .e_tag(e_tag)
            .build())
    }

    async fn complete_multipart(
        &self,
        key: &ObjectKey,
        upload_id: &str,
        parts: Vec<CompletedPart>,
    ) -> Result<ObjectVersion, CheckpointError> {
        self.begin_operation();
        let completed = self
            .client
            .complete_multipart_upload()
            .bucket(&self.settings.bucket)
            .key(key.as_str())
            .upload_id(upload_id)
            .multipart_upload(
                CompletedMultipartUpload::builder()
                    .set_parts(Some(parts))
                    .build(),
            )
            .send()
            .await
            .map_err(|error| map_sdk_error("complete multipart upload", &error))?;
        etag_version(completed.e_tag())
    }
}

#[async_trait(?Send)]
impl ConditionalObjectStore for AwsConditionalObjectStore {
    async fn put_immutable(
        &self,
        mut object: Box<dyn BudgetOwnedObjectReader>,
    ) -> Result<ObjectVersion, CheckpointError> {
        // The address of an immutable object is a pure function of its digest,
        // so the caller supplies no key and the store derives one.
        let key = immutable_object_key(&self.settings.prefix, &object.content_digest());
        let threshold = self.settings.single_put_threshold_bytes.get();
        let declared = usize::try_from(object.content_length())
            .map_err(|_| object_limit_exceeded_error())?;
        if declared <= threshold {
            let mut assembled = Vec::with_capacity(declared);
            while let Some(chunk) = object.next_chunk(threshold).await? {
                assembled.extend_from_slice(&chunk.bytes);
                drop(chunk);
                if assembled.len() > threshold {
                    return Err(object_limit_exceeded_error());
                }
            }
            if assembled.len() != declared {
                return Err(CheckpointError::ObjectVerification);
            }
            return self
                .put_single(&key, Bytes::from(assembled.into_boxed_slice()))
                .await;
        }
        let part_bytes = self.settings.multipart_part_bytes.get();
        let Some(first) = object.next_chunk(part_bytes).await? else {
            return Err(CheckpointError::ObjectVerification);
        };
        self.put_multipart(&key, object, first).await
    }

    async fn compare_and_swap_pointer(
        &self,
        key: &ObjectKey,
        expected: Option<&ObjectVersion>,
        next: PointerObject,
    ) -> Result<ObjectVersion, CheckpointError> {
        self.begin_operation();
        let mut request = self
            .client
            .put_object()
            .bucket(&self.settings.bucket)
            .key(key.as_str())
            .body(ByteStream::from(next.bytes));
        request = match expected {
            // Absent expectation means "create only": any existing pointer means
            // another writer already published a generation this one did not see.
            None => request.if_none_match("*"),
            Some(version) => request.if_match(version.as_str()),
        };
        let response = request
            .send()
            .await
            .map_err(|error| map_conditional_error(&error))?;
        // The lease is released only once the pointer write has landed.
        drop(next.lease);
        etag_version(response.e_tag())
    }

    async fn get_version_range(
        &self,
        key: &ObjectKey,
        version: &ObjectVersion,
        range: ObjectReadRange,
        budget: ObjectReadBudget,
    ) -> Result<BudgetOwnedObjectChunk, CheckpointError> {
        let length = usize::try_from(range.length).map_err(|_| object_limit_exceeded_error())?;
        if length == 0 || length > budget.max_chunk_bytes {
            return Err(object_limit_exceeded_error());
        }
        let last = range
            .offset
            .checked_add(range.length)
            .and_then(|end| end.checked_sub(1))
            .ok_or_else(object_limit_exceeded_error)?;
        self.begin_operation();
        let response = self
            .client
            .get_object()
            .bucket(&self.settings.bucket)
            .key(key.as_str())
            // `If-Match` binds the range to the exact revision the caller
            // resolved, so a concurrent overwrite cannot splice foreign bytes
            // into a restore.
            .if_match(version.as_str())
            .range(format!("bytes={}-{last}", range.offset))
            .send()
            .await
            .map_err(|error| map_sdk_error("get object range", &error))?;
        let bytes = response
            .body
            .collect()
            .await
            .map_err(|_| provider_error("read object range body"))?
            .into_bytes();
        if bytes.len() > length {
            return Err(object_limit_exceeded_error());
        }
        let lease = self
            .retention
            .acquire(1, bytes.len())
            .await
            .map_err(|_| object_limit_exceeded_error())?;
        Ok(BudgetOwnedObjectChunk { bytes, lease })
    }

    async fn list_versions(
        &self,
        prefix: &ObjectKey,
        cursor: Option<&ObjectListCursor>,
        budget: ObjectListBudget,
    ) -> Result<BudgetOwnedObjectPage, CheckpointError> {
        let max_keys = i32::try_from(budget.max_items.get()).unwrap_or(i32::MAX);
        self.begin_operation();
        let mut request = self
            .client
            .list_objects_v2()
            .bucket(&self.settings.bucket)
            .prefix(prefix.as_str())
            .max_keys(max_keys);
        if let Some(cursor) = cursor {
            request = request.continuation_token(cursor.as_str());
        }
        let response = request
            .send()
            .await
            .map_err(|error| map_sdk_error("list objects", &error))?;
        let mut entries = Vec::new();
        let mut retained = 0usize;
        for object in response.contents() {
            let (Some(key), Some(e_tag), Some(size)) =
                (object.key(), object.e_tag(), object.size())
            else {
                return Err(provider_error("listed object is missing key, etag, or size"));
            };
            retained = retained
                .checked_add(std::mem::size_of::<ObjectMetadata>() + key.len() + e_tag.len())
                .ok_or_else(object_limit_exceeded_error)?;
            if retained > budget.max_metadata_bytes.get() {
                return Err(object_limit_exceeded_error());
            }
            entries.push(ObjectMetadata {
                key: ObjectKey::new(key),
                version: ObjectVersion::new(normalize_etag(e_tag)),
                byte_length: u64::try_from(size).map_err(|_| object_limit_exceeded_error())?,
            });
        }
        let next = response
            .next_continuation_token()
            .map(ObjectListCursor::new);
        let lease = self
            .retention
            .acquire(entries.len(), retained)
            .await
            .map_err(|_| object_limit_exceeded_error())?;
        Ok(BudgetOwnedObjectPage {
            objects: entries.into_boxed_slice(),
            next,
            lease,
        })
    }

    async fn delete_version(
        &self,
        key: &ObjectKey,
        version: &ObjectVersion,
    ) -> Result<(), CheckpointError> {
        self.begin_operation();
        self.client
            .delete_object()
            .bucket(&self.settings.bucket)
            .key(key.as_str())
            .if_match(version.as_str())
            .send()
            .await
            .map_err(|error| map_sdk_error("delete object", &error))?;
        Ok(())
    }
}

fn etag_version(e_tag: Option<&str>) -> Result<ObjectVersion, CheckpointError> {
    e_tag
        .map(|value| ObjectVersion::new(normalize_etag(value)))
        .ok_or_else(|| provider_error("response carries no entity tag"))
}

/// Strip the quoting S3 applies to entity tags so versions compare exactly.
fn normalize_etag(value: &str) -> &str {
    value.trim_matches('"')
}

fn status_of<E>(error: &SdkError<E>) -> Option<u16> {
    match error {
        SdkError::ServiceError(service) => Some(service.raw().status().as_u16()),
        _ => None,
    }
}

fn map_sdk_error<E>(context: &str, error: &SdkError<E>) -> CheckpointError {
    match status_of(error) {
        Some(NOT_IMPLEMENTED_STATUS) => conditional_write_unsupported_error(),
        Some(PRECONDITION_FAILED_STATUS) | Some(CONFLICT_STATUS) => stale_writer_error(),
        _ => provider_error(context),
    }
}

fn map_conditional_error<E>(error: &SdkError<E>) -> CheckpointError {
    match status_of(error) {
        // A provider that does not implement conditional writes cannot host an
        // atomic generation pointer, and that is a configuration error rather
        // than a runtime fault.
        Some(NOT_IMPLEMENTED_STATUS) => conditional_write_unsupported_error(),
        Some(PRECONDITION_FAILED_STATUS) | Some(CONFLICT_STATUS) => stale_writer_error(),
        _ => provider_error("conditional pointer write"),
    }
}

/// Deferred S3 store that resolves its SDK configuration on first provider use.
///
/// Backend preparation is synchronous while `SdkConfig` resolution is not, so
/// the client is built inside the first provider call rather than blocking a
/// current-thread runtime at startup. Nothing observable is deferred: the first
/// call that would touch the provider is also the first call that can report a
/// configuration failure.
pub struct LazyAwsConditionalObjectStore {
    client_settings: AwsClientSettings,
    profile: Option<String>,
    store_settings: AwsObjectStoreSettings,
    clock: Rc<dyn Clock>,
    prepared: tokio::sync::OnceCell<AwsConditionalObjectStore>,
}

impl std::fmt::Debug for LazyAwsConditionalObjectStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LazyAwsConditionalObjectStore")
            .field("client_settings", &self.client_settings)
            .field("store_settings", &self.store_settings)
            .finish()
    }
}

impl LazyAwsConditionalObjectStore {
    /// Retain validated settings without contacting any provider.
    pub fn new(
        client_settings: AwsClientSettings,
        profile: Option<String>,
        store_settings: AwsObjectStoreSettings,
        clock: Rc<dyn Clock>,
    ) -> Result<Self, CheckpointError> {
        store_settings.validate()?;
        Ok(Self {
            client_settings,
            profile,
            store_settings,
            clock,
            prepared: tokio::sync::OnceCell::new(),
        })
    }

    async fn store(&self) -> Result<&AwsConditionalObjectStore, CheckpointError> {
        self.prepared
            .get_or_try_init(|| async {
                let settings = AwsClientSettings {
                    region: self.client_settings.region.clone(),
                    endpoint_url: self.client_settings.endpoint_url.clone(),
                    force_path_style: self.client_settings.force_path_style,
                    proxy: self.client_settings.proxy.clone(),
                    operation_timeout_ns: self.client_settings.operation_timeout_ns,
                    connect_timeout_ns: self.client_settings.connect_timeout_ns,
                };
                let factory =
                    AwsS3ClientFactory::prepare_default_chain(settings, self.profile.as_deref())
                        .await
                        .map_err(|error| provider_error(&format!("prepare s3 client: {error}")))?;
                AwsConditionalObjectStore::new(
                    &factory,
                    Rc::clone(&self.clock),
                    self.store_settings.clone(),
                )
            })
            .await
    }
}

#[async_trait(?Send)]
impl ConditionalObjectStore for LazyAwsConditionalObjectStore {
    async fn put_immutable(
        &self,
        object: Box<dyn BudgetOwnedObjectReader>,
    ) -> Result<ObjectVersion, CheckpointError> {
        self.store().await?.put_immutable(object).await
    }

    async fn compare_and_swap_pointer(
        &self,
        key: &ObjectKey,
        expected: Option<&ObjectVersion>,
        next: PointerObject,
    ) -> Result<ObjectVersion, CheckpointError> {
        self.store()
            .await?
            .compare_and_swap_pointer(key, expected, next)
            .await
    }

    async fn get_version_range(
        &self,
        key: &ObjectKey,
        version: &ObjectVersion,
        range: ObjectReadRange,
        budget: ObjectReadBudget,
    ) -> Result<BudgetOwnedObjectChunk, CheckpointError> {
        self.store()
            .await?
            .get_version_range(key, version, range, budget)
            .await
    }

    async fn list_versions(
        &self,
        prefix: &ObjectKey,
        cursor: Option<&ObjectListCursor>,
        budget: ObjectListBudget,
    ) -> Result<BudgetOwnedObjectPage, CheckpointError> {
        self.store()
            .await?
            .list_versions(prefix, cursor, budget)
            .await
    }

    async fn delete_version(
        &self,
        key: &ObjectKey,
        version: &ObjectVersion,
    ) -> Result<(), CheckpointError> {
        self.store().await?.delete_version(key, version).await
    }
}
