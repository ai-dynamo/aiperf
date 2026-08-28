// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Narrow provider-neutral S3 seam and its AWS transport.
//!
//! The trait is deliberately two operations wide: list one bounded page, and
//! read one immutable object generation under a conditional bind. Every
//! reconciliation, frontier, identity, and retry decision lives in
//! [`super::s3`], so the whole policy surface is testable against [`S3Client`]
//! with no AWS dependency at all.
//!
//! The trait is `?Send`: the streaming source seam is worker-local
//! (`PreparedStreamingDatasetSource`, `SourcePartitionContent`, and every reader
//! trait are `#[async_trait(?Send)]`) and the AWS transport holds the
//! `Rc<dyn Clock>`-backed `AwsClockProjection` that `streaming::aws` requires be
//! published before each operation.
//!
//! No AWS type crosses this module's public surface, and no `SdkError` text is
//! retained: [`S3ClientError`] is a closed `Copy` classification, which is what
//! structurally prevents an endpoint, a presigned query string, or a
//! `Proxy-Authorization` value from reaching a log or an artifact.

use std::fmt;
use std::num::NonZeroUsize;
use std::sync::Arc;

use async_trait::async_trait;
use aws_sdk_s3::config::http::HttpResponse;
use aws_sdk_s3::error::SdkError;
use aws_sdk_s3::primitives::ByteStream;
use bytes::Bytes;

use crate::streaming::aws::{AwsClockProjection, AwsCredentialProviderAuthority};

/// Largest listing page any policy may request.
pub const MAX_LIST_PAGE_KEYS: u16 = 1_000;

/// One bounded listing request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct S3ListRequest {
    /// Bucket being listed.
    pub bucket: String,
    /// Key prefix scoping the listing.
    pub prefix: Option<String>,
    /// Exclusive lower key bound derived from a sealed interval.
    pub start_after: Option<String>,
    /// Opaque provider continuation token; never durable, never a frontier.
    pub continuation_token: Option<String>,
    /// Hard page bound; `1..=MAX_LIST_PAGE_KEYS`.
    pub max_keys: u16,
}

/// One listed object generation, before identity is derived.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct S3ListedObject {
    /// Object key.
    pub key: String,
    /// Provider-reported byte length.
    pub size_bytes: u64,
    /// Provider ETag, opaque and never treated as a content digest.
    pub etag: Option<String>,
    /// Provider version id when bucket versioning is enabled.
    pub version_id: Option<String>,
}

/// One bounded listing page.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct S3ListPage {
    /// Objects in provider key order.
    pub objects: Vec<S3ListedObject>,
    /// Token for the next page, when the listing is truncated.
    pub next_continuation_token: Option<String>,
}

/// Half-open byte range within one immutable object generation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct S3ByteRange {
    /// Inclusive first byte.
    pub offset: u64,
    /// Exclusive last byte.
    pub end: u64,
}

/// One conditional read of an exact immutable object generation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct S3GetRequest {
    /// Bucket owning the object.
    pub bucket: String,
    /// Object key.
    pub key: String,
    /// Exact version id, when the generation is version-qualified.
    pub version_id: Option<String>,
    /// `If-Match` ETag used when no version id is available.
    pub if_match_etag: Option<String>,
    /// Optional bounded range; `None` reads from byte zero.
    pub range: Option<S3ByteRange>,
}

/// Bounded forward reader over one immutable object generation.
#[async_trait(?Send)]
pub trait S3ObjectReader {
    /// Return at most `max_bytes`, or `None` at end of body.
    async fn next_chunk(&mut self, max_bytes: NonZeroUsize)
    -> Result<Option<Bytes>, S3ClientError>;
}

/// Response metadata plus the bounded body reader.
pub struct S3ObjectBody {
    /// ETag reported by the read.
    pub etag: Option<String>,
    /// Version id reported by the read.
    pub version_id: Option<String>,
    /// Length reported by the read.
    pub content_length: Option<u64>,
    /// Bounded forward reader.
    pub reader: Box<dyn S3ObjectReader>,
}

impl fmt::Debug for S3ObjectBody {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S3ObjectBody")
            .field("has_version_id", &self.version_id.is_some())
            .field("content_length", &self.content_length)
            .finish_non_exhaustive()
    }
}

/// Closed, text-free provider failure classification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum S3ClientError {
    /// The provider throttled or is temporarily overloaded.
    Throttled,
    /// The request timed out before a complete response.
    Timeout,
    /// The connection failed before or during dispatch.
    Transport,
    /// Authorization was refused; credentials may be refreshable.
    Unauthorized,
    /// The listed key or version no longer exists.
    NotFound,
    /// A conditional read failed: the key now names different bytes.
    PreconditionFailed,
    /// The provider response could not be interpreted under the S3 contract.
    Malformed,
}

impl S3ClientError {
    /// Whether a bounded retry may succeed without changing object identity.
    #[must_use]
    pub const fn is_retryable(self) -> bool {
        matches!(self, Self::Throttled | Self::Timeout | Self::Transport)
    }

    /// Whether the failure is an authorization state a credential refresh may clear.
    #[must_use]
    pub const fn is_authorization(self) -> bool {
        matches!(self, Self::Unauthorized)
    }

    /// Whether the failure proves the frozen generation no longer holds.
    #[must_use]
    pub const fn is_identity_violation(self) -> bool {
        matches!(self, Self::PreconditionFailed)
    }

    /// Stable machine-readable code, safe for a `tracing` field.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::Throttled => "throttled",
            Self::Timeout => "timeout",
            Self::Transport => "transport",
            Self::Unauthorized => "unauthorized",
            Self::NotFound => "not_found",
            Self::PreconditionFailed => "precondition_failed",
            Self::Malformed => "malformed",
        }
    }
}

impl fmt::Display for S3ClientError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.code())
    }
}

impl std::error::Error for S3ClientError {}

/// Narrow provider-neutral S3 seam.
#[async_trait(?Send)]
pub trait S3Client: fmt::Debug {
    /// List one bounded page. Pagination never advances a frontier.
    async fn list_page(&self, request: S3ListRequest) -> Result<S3ListPage, S3ClientError>;

    /// Read one exact immutable object generation under a conditional bind.
    async fn get_version(&self, request: S3GetRequest) -> Result<S3ObjectBody, S3ClientError>;

    /// Drop cached credentials so the next operation refreshes.
    ///
    /// The frozen object identity is untouched, so a later successful attempt
    /// acquires the same generation. The default is a no-op for transports that
    /// carry no refreshable credential authority.
    fn invalidate_credentials(&self) {}
}

// ---------------------------------------------------------------------------
// AWS transport
// ---------------------------------------------------------------------------

/// Thin adapter over the client `streaming::aws` constructed.
///
/// It owns no policy. Its only responsibilities are publishing the run clock
/// before each operation, translating the neutral request/response vocabulary,
/// and classifying an `SdkError` to a closed code while dropping its text.
pub struct AwsS3Transport {
    client: aws_sdk_s3::Client,
    projection: AwsClockProjection,
    authority: Arc<AwsCredentialProviderAuthority>,
}

impl fmt::Debug for AwsS3Transport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Neither `aws_sdk_s3::Client` nor the authority may be rendered: the
        // former's `Debug` can reach a credential provider.
        formatter
            .debug_struct("AwsS3Transport")
            .field("credential_source_id", &self.authority.source_id())
            .finish()
    }
}

impl AwsS3Transport {
    /// Bind a prepared client to its clock projection and credential authority.
    #[must_use]
    pub fn new(
        client: aws_sdk_s3::Client,
        projection: AwsClockProjection,
        authority: Arc<AwsCredentialProviderAuthority>,
    ) -> Self {
        Self {
            client,
            projection,
            authority,
        }
    }
}

#[async_trait(?Send)]
impl S3Client for AwsS3Transport {
    async fn list_page(&self, request: S3ListRequest) -> Result<S3ListPage, S3ClientError> {
        self.projection.publish();
        let max_keys = i32::from(request.max_keys.min(MAX_LIST_PAGE_KEYS));
        let mut call = self
            .client
            .list_objects_v2()
            .bucket(request.bucket)
            .max_keys(max_keys);
        if let Some(prefix) = request.prefix {
            call = call.prefix(prefix);
        }
        if let Some(start_after) = request.start_after {
            call = call.start_after(start_after);
        }
        if let Some(token) = request.continuation_token {
            call = call.continuation_token(token);
        }
        let output = call.send().await.map_err(classify_sdk_error)?;

        let mut objects = Vec::with_capacity(output.contents().len());
        for object in output.contents() {
            let (Some(key), Some(size)) = (object.key(), object.size()) else {
                return Err(S3ClientError::Malformed);
            };
            let size_bytes = u64::try_from(size).map_err(|_| S3ClientError::Malformed)?;
            objects.push(S3ListedObject {
                key: key.to_owned(),
                size_bytes,
                etag: object.e_tag().map(str::to_owned),
                // `ListObjectsV2` never reports a version id; a versioned
                // inventory reaches this seam through a manifest or a
                // provider-side snapshot rather than a silent substitution.
                version_id: None,
            });
        }
        Ok(S3ListPage {
            objects,
            next_continuation_token: output.next_continuation_token().map(str::to_owned),
        })
    }

    async fn get_version(&self, request: S3GetRequest) -> Result<S3ObjectBody, S3ClientError> {
        self.projection.publish();
        let mut call = self
            .client
            .get_object()
            .bucket(request.bucket)
            .key(request.key);
        if let Some(version_id) = request.version_id {
            call = call.version_id(version_id);
        }
        if let Some(etag) = request.if_match_etag {
            call = call.if_match(etag);
        }
        if let Some(range) = request.range {
            // S3 byte ranges are inclusive on both ends.
            let last = range.end.checked_sub(1).ok_or(S3ClientError::Malformed)?;
            call = call.range(format!("bytes={}-{}", range.offset, last));
        }
        let output = call.send().await.map_err(classify_sdk_error)?;

        let content_length = match output.content_length() {
            Some(length) => Some(u64::try_from(length).map_err(|_| S3ClientError::Malformed)?),
            None => None,
        };
        Ok(S3ObjectBody {
            etag: output.e_tag().map(str::to_owned),
            version_id: output.version_id().map(str::to_owned),
            content_length,
            reader: Box::new(AwsByteStreamReader {
                stream: output.body,
                pending: Bytes::new(),
                is_done: false,
            }),
        })
    }

    fn invalidate_credentials(&self) {
        self.authority.invalidate();
    }
}

/// Splits transport-sized chunks down to the caller's bound.
///
/// `ByteStream::next` returns whatever the connector produced, but the streaming
/// seam validates that a sequential chunk is non-empty and no larger than the
/// requested bound, so the remainder is carried here rather than handed back.
struct AwsByteStreamReader {
    stream: ByteStream,
    pending: Bytes,
    is_done: bool,
}

#[async_trait(?Send)]
impl S3ObjectReader for AwsByteStreamReader {
    async fn next_chunk(
        &mut self,
        max_bytes: NonZeroUsize,
    ) -> Result<Option<Bytes>, S3ClientError> {
        while self.pending.is_empty() {
            if self.is_done {
                return Ok(None);
            }
            match self.stream.next().await {
                Some(Ok(chunk)) => self.pending = chunk,
                Some(Err(_)) => return Err(S3ClientError::Transport),
                None => {
                    self.is_done = true;
                    return Ok(None);
                }
            }
        }
        let take = max_bytes.get().min(self.pending.len());
        Ok(Some(self.pending.split_to(take)))
    }
}

/// Classify an SDK error to a closed code, dropping every string it carries.
///
/// An `SdkError`'s `Display` can echo the endpoint, a presigned query string, or
/// a proxy authorization value, so the error value is inspected for its status
/// and then discarded.
fn classify_sdk_error<E>(error: SdkError<E, HttpResponse>) -> S3ClientError {
    if let Some(response) = error.raw_response() {
        return match response.status().as_u16() {
            401 | 403 => S3ClientError::Unauthorized,
            404 => S3ClientError::NotFound,
            412 => S3ClientError::PreconditionFailed,
            429 | 500..=599 => S3ClientError::Throttled,
            _ => S3ClientError::Malformed,
        };
    }
    match error {
        SdkError::TimeoutError(_) => S3ClientError::Timeout,
        SdkError::ConstructionFailure(_) => S3ClientError::Malformed,
        _ => S3ClientError::Transport,
    }
}
