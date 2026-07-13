// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact raw-body encryption and its crash-recoverable equality registry.
//!
//! Exact entities are deliberately separate from structured telemetry rows.
//! This module freezes the v1 AES-256-GCM-SIV envelope, keeps plaintext facts
//! inside the encrypted private prefix, and serializes every nonce reservation
//! needed to make create-if-absent retries byte-identical after recovery.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Debug, Display, Formatter};

use aes_gcm_siv::aead::{Aead, KeyInit, Payload};
use aes_gcm_siv::{Aes256GcmSiv, Nonce};
use rand::TryRngCore;
use rand::rngs::OsRng;

use crate::canonical_json::{CanonicalJsonError, CanonicalJsonValue};
use crate::descriptor::CanonicalDescriptor;
pub use crate::descriptor::RAW_ENVELOPE_V1;
use crate::digest::{Digest, domain_digest};
use crate::identity::ArchiveId;
use crate::key::keyed_domain_digest;

/// Exact AES-256 key width required by the v1 profile.
pub const RAW_ENVELOPE_KEY_BYTES: usize = 32;
/// Exact random nonce width required by the v1 profile.
pub const RAW_ENVELOPE_NONCE_BYTES: usize = 12;
/// Exact authentication-tag width produced by the v1 profile.
pub const RAW_ENVELOPE_TAG_BYTES: usize = 16;
/// Exact digest width used as v1 additional authenticated data.
pub const RAW_ENVELOPE_AAD_BYTES: usize = 32;
/// Hard maximum exact encoded entity size under the v1 profile.
pub const RAW_ENVELOPE_MAX_PLAINTEXT_BYTES: u64 = 1 << 30;
/// Hard maximum successful envelope creations for one key ID.
pub const RAW_ENVELOPE_MAX_OBJECTS_PER_KEY: u64 = 1 << 29;
/// Maximum random draws used to find one unreserved nonce.
pub const RAW_ENVELOPE_MAX_NONCE_DRAWS: usize = 16;
/// Stable algorithm spelling embedded in every public v1 header.
pub const RAW_ENVELOPE_ALGORITHM_V1: &str = "AEAD_AES_256_GCM_SIV";

const RAW_ENVELOPE_VERSION: u64 = 1;
const RAW_ENVELOPE_MAGIC: &[u8; 16] = b"AIPERFRAWENV1\0\0\0";
const RAW_PRIVATE_MAGIC: &[u8; 16] = b"AIPERFRAWPRIV1\0\0";
const RAW_REGISTRY_MAGIC: &[u8; 16] = b"AIPERFRAWREG1\0\0\0";
const RAW_PRIVATE_PREFIX_BYTES: u64 = 16 + 8 + 32;
const RAW_MAX_CIPHERTEXT_BYTES: u64 =
    RAW_ENVELOPE_MAX_PLAINTEXT_BYTES + RAW_PRIVATE_PREFIX_BYTES + RAW_ENVELOPE_TAG_BYTES as u64;
const MAX_PUBLIC_HEADER_BYTES: usize = 64 * 1024;
const MAX_KEY_ID_BYTES: usize = u8::MAX as usize;

/// Typed capabilities of one raw-envelope profile.
#[derive(Clone, Copy, Debug)]
pub struct RawEnvelopeDescriptor {
    /// Stable profile selection ID.
    pub profile_id: &'static str,
    /// Public algorithm spelling.
    pub algorithm: &'static str,
    /// Envelope version written into the public header.
    pub envelope_version: u64,
    /// Required key width.
    pub key_bytes: usize,
    /// Required random nonce width.
    pub nonce_bytes: usize,
    /// Authentication-tag width.
    pub tag_bytes: usize,
    /// Hard plaintext byte limit.
    pub max_plaintext_bytes: u64,
    /// Hard successful-object limit for one key ID.
    pub max_objects_per_key: u64,
    /// Checked-in canonical byte authority.
    pub canonical: CanonicalDescriptor,
}

/// Exact descriptor selected by `aead_aes_256_gcm_siv_random96_v1`.
pub static AES_256_GCM_SIV_RANDOM96_V1_DESCRIPTOR: RawEnvelopeDescriptor = RawEnvelopeDescriptor {
    profile_id: "aead_aes_256_gcm_siv_random96_v1",
    algorithm: RAW_ENVELOPE_ALGORITHM_V1,
    envelope_version: RAW_ENVELOPE_VERSION,
    key_bytes: RAW_ENVELOPE_KEY_BYTES,
    nonce_bytes: RAW_ENVELOPE_NONCE_BYTES,
    tag_bytes: RAW_ENVELOPE_TAG_BYTES,
    max_plaintext_bytes: RAW_ENVELOPE_MAX_PLAINTEXT_BYTES,
    max_objects_per_key: RAW_ENVELOPE_MAX_OBJECTS_PER_KEY,
    canonical: RAW_ENVELOPE_V1,
};

/// Injected source of cryptographic nonces.
pub trait RawNonceSource: Debug + Send {
    /// Fills the entire caller-provided nonce buffer or fails closed.
    fn fill_nonce(&mut self, bytes: &mut [u8]) -> Result<(), RawNonceError>;
}

/// Product nonce source backed directly by the operating-system CSPRNG.
#[derive(Clone, Copy, Debug, Default)]
pub struct OsRawNonceSource;

impl RawNonceSource for OsRawNonceSource {
    fn fill_nonce(&mut self, bytes: &mut [u8]) -> Result<(), RawNonceError> {
        OsRng
            .try_fill_bytes(bytes)
            .map_err(|_| RawNonceError::EntropyUnavailable)
    }
}

/// Failure to obtain cryptographic randomness.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RawNonceError {
    /// The operating-system or injected entropy source failed.
    EntropyUnavailable,
}

impl Display for RawNonceError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str("raw-envelope nonce entropy is unavailable")
    }
}

impl std::error::Error for RawNonceError {}

/// Resolved 256-bit AEAD key bound to its public rotation ID.
pub struct RawEnvelopeKey {
    key_id: String,
    bytes: [u8; RAW_ENVELOPE_KEY_BYTES],
}

impl RawEnvelopeKey {
    /// Binds resolved secret bytes to one validated public key ID.
    pub fn new(
        key_id: impl Into<String>,
        bytes: [u8; RAW_ENVELOPE_KEY_BYTES],
    ) -> Result<Self, RawKeyError> {
        let key_id = key_id.into();
        validate_key_id(&key_id)?;
        Ok(Self { key_id, bytes })
    }

    /// Returns the public rotation ID; secret key bytes are never exposed.
    #[must_use]
    pub fn key_id(&self) -> &str {
        &self.key_id
    }

    fn secret_bytes(&self) -> &[u8; RAW_ENVELOPE_KEY_BYTES] {
        &self.bytes
    }
}

impl Debug for RawEnvelopeKey {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RawEnvelopeKey")
            .field("key_id", &self.key_id)
            .field("key_bytes", &"<redacted>")
            .finish()
    }
}

impl Drop for RawEnvelopeKey {
    fn drop(&mut self) {
        self.bytes.fill(0);
    }
}

/// Prepared provider for separately classified raw-envelope keys.
pub trait ArchiveRawKeyProvider: Debug + Send + Sync {
    /// Resolves one key by public rotation ID without logging secret material.
    fn resolve_key(&self, key_id: &str) -> Result<RawEnvelopeKey, RawKeyError>;
}

/// In-memory raw-key provider for prepared secret resolvers and tests.
pub struct MemoryRawKeyProvider {
    keys: BTreeMap<String, [u8; RAW_ENVELOPE_KEY_BYTES]>,
}

impl MemoryRawKeyProvider {
    /// Builds a provider while rejecting invalid or repeated key IDs.
    pub fn new<I>(keys: I) -> Result<Self, RawKeyError>
    where
        I: IntoIterator<Item = (String, [u8; RAW_ENVELOPE_KEY_BYTES])>,
    {
        let mut output = BTreeMap::new();
        for (key_id, bytes) in keys {
            validate_key_id(&key_id)?;
            if output.insert(key_id, bytes).is_some() {
                return Err(RawKeyError::DuplicateKeyId);
            }
        }
        Ok(Self { keys: output })
    }
}

impl Debug for MemoryRawKeyProvider {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MemoryRawKeyProvider")
            .field("key_count", &self.keys.len())
            .field("key_material", &"<redacted>")
            .finish()
    }
}

impl ArchiveRawKeyProvider for MemoryRawKeyProvider {
    fn resolve_key(&self, key_id: &str) -> Result<RawEnvelopeKey, RawKeyError> {
        let bytes = self.keys.get(key_id).ok_or(RawKeyError::Unavailable)?;
        RawEnvelopeKey::new(key_id, *bytes)
    }
}

/// Failure to validate or resolve a raw-envelope key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RawKeyError {
    /// A requested key ID is unavailable.
    Unavailable,
    /// A public key ID is empty, too long, or contains a control scalar.
    InvalidKeyId,
    /// A provider configuration repeats one public key ID.
    DuplicateKeyId,
}

impl Display for RawKeyError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Unavailable => "raw-envelope key is unavailable",
            Self::InvalidKeyId => "raw-envelope key ID is invalid",
            Self::DuplicateKeyId => "raw-envelope key provider contains a duplicate key ID",
        })
    }
}

impl std::error::Error for RawKeyError {}

/// Pluggable raw-body sealing profile.
pub trait RawEnvelopeProfile: Debug + Send + Sync {
    /// Returns the immutable descriptor selected by configuration.
    fn descriptor(&self) -> &'static RawEnvelopeDescriptor;

    /// Seals exact private bytes under the supplied nonce and 32-byte AAD.
    fn seal(
        &self,
        key: &RawEnvelopeKey,
        nonce: &[u8],
        aad: &[u8],
        plaintext: &[u8],
    ) -> Result<RawEnvelope, RawEnvelopeError>;
}

/// Exact RFC 8452 AES-256-GCM-SIV/random-96-bit profile.
#[derive(Clone, Copy, Debug, Default)]
pub struct Aes256GcmSivRandom96V1;

impl RawEnvelopeProfile for Aes256GcmSivRandom96V1 {
    fn descriptor(&self) -> &'static RawEnvelopeDescriptor {
        &AES_256_GCM_SIV_RANDOM96_V1_DESCRIPTOR
    }

    fn seal(
        &self,
        key: &RawEnvelopeKey,
        nonce: &[u8],
        aad: &[u8],
        plaintext: &[u8],
    ) -> Result<RawEnvelope, RawEnvelopeError> {
        validate_aead_inputs(nonce, aad)?;
        let cipher = Aes256GcmSiv::new_from_slice(key.secret_bytes())
            .map_err(|_| RawEnvelopeError::InvalidKeyLength)?;
        let ciphertext_and_tag = cipher
            .encrypt(
                Nonce::from_slice(nonce),
                Payload {
                    msg: plaintext,
                    aad,
                },
            )
            .map_err(|_| RawEnvelopeError::SealFailed)?;
        if ciphertext_and_tag.len() != plaintext.len() + RAW_ENVELOPE_TAG_BYTES {
            return Err(RawEnvelopeError::UnexpectedCiphertextLength);
        }
        Ok(RawEnvelope { ciphertext_and_tag })
    }
}

impl Aes256GcmSivRandom96V1 {
    /// Authenticates, opens, and validates one complete v1 physical object.
    pub fn open(
        &self,
        key: &RawEnvelopeKey,
        raw_object_subkey: &[u8; 32],
        object: &RawEnvelopeObjectV1,
    ) -> Result<Vec<u8>, RawEnvelopeError> {
        if key.key_id() != object.header.key_id()
            || object.header.algorithm() != RAW_ENVELOPE_ALGORITHM_V1
            || object.header.envelope_version() != RAW_ENVELOPE_VERSION
        {
            return Err(RawEnvelopeError::HeaderKeyOrProfileMismatch);
        }
        let header_bytes = object.header.canonical_bytes();
        let aad = raw_envelope_aad_v1(&header_bytes);
        let cipher = Aes256GcmSiv::new_from_slice(key.secret_bytes())
            .map_err(|_| RawEnvelopeError::InvalidKeyLength)?;
        let private = cipher
            .decrypt(
                Nonce::from_slice(object.header.nonce()),
                Payload {
                    msg: object.envelope.ciphertext_and_tag(),
                    aad: aad.as_bytes(),
                },
            )
            .map_err(|_| RawEnvelopeError::OpenFailed)?;
        let plaintext = decode_private_plaintext(&private)?;
        if raw_object_id_v1(raw_object_subkey, &plaintext) != object.header.raw_object_id() {
            return Err(RawEnvelopeError::PrivateIntegrityMismatch);
        }
        Ok(plaintext)
    }
}

/// AEAD ciphertext with its trailing authentication tag.
#[derive(Clone, Eq, PartialEq)]
pub struct RawEnvelope {
    ciphertext_and_tag: Vec<u8>,
}

impl RawEnvelope {
    /// Returns the exact ciphertext followed by its 16-byte tag.
    #[must_use]
    pub fn ciphertext_and_tag(&self) -> &[u8] {
        &self.ciphertext_and_tag
    }
}

impl Debug for RawEnvelope {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RawEnvelope")
            .field("ciphertext_and_tag_bytes", &self.ciphertext_and_tag.len())
            .field("ciphertext_and_tag", &"<redacted>")
            .finish()
    }
}

/// Canonical public header authenticated by the v1 32-byte AAD digest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RawEnvelopePublicHeaderV1 {
    algorithm: String,
    archive_id: ArchiveId,
    ciphertext_length: u64,
    envelope_version: u64,
    key_id: String,
    nonce: [u8; RAW_ENVELOPE_NONCE_BYTES],
    raw_object_id: Digest,
}

impl RawEnvelopePublicHeaderV1 {
    /// Constructs the exact v1 public header.
    pub fn new(
        archive_id: ArchiveId,
        raw_object_id: Digest,
        key_id: impl Into<String>,
        nonce: [u8; RAW_ENVELOPE_NONCE_BYTES],
        ciphertext_length: u64,
    ) -> Result<Self, RawEnvelopeError> {
        let key_id = key_id.into();
        validate_key_id(&key_id).map_err(RawEnvelopeError::Key)?;
        if !(RAW_ENVELOPE_TAG_BYTES as u64..=RAW_MAX_CIPHERTEXT_BYTES).contains(&ciphertext_length)
        {
            return Err(RawEnvelopeError::UnexpectedCiphertextLength);
        }
        Ok(Self {
            algorithm: RAW_ENVELOPE_ALGORITHM_V1.to_owned(),
            archive_id,
            ciphertext_length,
            envelope_version: RAW_ENVELOPE_VERSION,
            key_id,
            nonce,
            raw_object_id,
        })
    }

    /// Returns exact canonical header bytes.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        object([
            ("algorithm", string(self.algorithm.clone())),
            ("archive_id", string(uuid(self.archive_id.as_bytes()))),
            (
                "ciphertext_length",
                CanonicalJsonValue::Integer(i128::from(self.ciphertext_length)),
            ),
            (
                "envelope_version",
                CanonicalJsonValue::Integer(i128::from(self.envelope_version)),
            ),
            ("key_id", string(self.key_id.clone())),
            ("nonce", string(hex(&self.nonce))),
            ("raw_object_id", string(self.raw_object_id.to_hex())),
        ])
        .to_bytes()
    }

    /// Decodes bytes that must already be the exact canonical v1 header.
    pub fn decode(bytes: &[u8]) -> Result<Self, RawEnvelopeError> {
        let value = CanonicalJsonValue::parse_canonical(bytes)
            .map_err(RawEnvelopeError::CanonicalHeader)?;
        let values = value
            .as_object()
            .ok_or(RawEnvelopeError::InvalidPublicHeader)?;
        if values.len() != 7 {
            return Err(RawEnvelopeError::InvalidPublicHeader);
        }
        let algorithm = text(values, "algorithm")?.to_owned();
        let archive_id = ArchiveId::new(parse_uuid(text(values, "archive_id")?)?)
            .map_err(|_| RawEnvelopeError::InvalidPublicHeader)?;
        let ciphertext_length = unsigned(values, "ciphertext_length")?;
        let envelope_version = unsigned(values, "envelope_version")?;
        let key_id = text(values, "key_id")?.to_owned();
        validate_key_id(&key_id).map_err(RawEnvelopeError::Key)?;
        let nonce = parse_hex_array(text(values, "nonce")?)?;
        let raw_object_id = Digest::parse(text(values, "raw_object_id")?)
            .map_err(|_| RawEnvelopeError::InvalidPublicHeader)?;
        if algorithm != RAW_ENVELOPE_ALGORITHM_V1
            || envelope_version != RAW_ENVELOPE_VERSION
            || !(RAW_ENVELOPE_TAG_BYTES as u64..=RAW_MAX_CIPHERTEXT_BYTES)
                .contains(&ciphertext_length)
        {
            return Err(RawEnvelopeError::InvalidPublicHeader);
        }
        Ok(Self {
            algorithm,
            archive_id,
            ciphertext_length,
            envelope_version,
            key_id,
            nonce,
            raw_object_id,
        })
    }

    /// Returns the public algorithm spelling.
    #[must_use]
    pub fn algorithm(&self) -> &str {
        &self.algorithm
    }

    /// Returns the archive UUID bound into AAD.
    #[must_use]
    pub const fn archive_id(&self) -> ArchiveId {
        self.archive_id
    }

    /// Returns ciphertext-plus-tag byte length.
    #[must_use]
    pub const fn ciphertext_length(&self) -> u64 {
        self.ciphertext_length
    }

    /// Returns the public envelope version.
    #[must_use]
    pub const fn envelope_version(&self) -> u64 {
        self.envelope_version
    }

    /// Returns the public key rotation ID.
    #[must_use]
    pub fn key_id(&self) -> &str {
        &self.key_id
    }

    /// Returns the exact random 96-bit nonce.
    #[must_use]
    pub const fn nonce(&self) -> &[u8; RAW_ENVELOPE_NONCE_BYTES] {
        &self.nonce
    }

    /// Returns the keyed equality identity.
    #[must_use]
    pub const fn raw_object_id(&self) -> Digest {
        self.raw_object_id
    }
}

/// Complete immutable physical raw object.
#[derive(Clone, Eq, PartialEq)]
pub struct RawEnvelopeObjectV1 {
    header: RawEnvelopePublicHeaderV1,
    envelope: RawEnvelope,
    ciphertext_digest: Digest,
}

impl RawEnvelopeObjectV1 {
    /// Returns the authenticated public header.
    #[must_use]
    pub const fn header(&self) -> &RawEnvelopePublicHeaderV1 {
        &self.header
    }

    /// Returns the domain-separated ciphertext-and-tag digest.
    #[must_use]
    pub const fn ciphertext_digest(&self) -> Digest {
        self.ciphertext_digest
    }

    /// Returns the exact ciphertext followed by its tag.
    #[must_use]
    pub fn ciphertext_and_tag(&self) -> &[u8] {
        self.envelope.ciphertext_and_tag()
    }

    /// Encodes the exact create-if-absent object bytes.
    #[must_use]
    pub fn exact_bytes(&self) -> Vec<u8> {
        let header = self.header.canonical_bytes();
        let mut output = Vec::with_capacity(
            RAW_ENVELOPE_MAGIC.len()
                + 4
                + header.len()
                + 8
                + self.envelope.ciphertext_and_tag.len()
                + Digest::BYTE_LEN,
        );
        output.extend_from_slice(RAW_ENVELOPE_MAGIC);
        output.extend_from_slice(
            &u32::try_from(header.len())
                .expect("validated public header length fits u32")
                .to_be_bytes(),
        );
        output.extend_from_slice(&header);
        output.extend_from_slice(&self.header.ciphertext_length.to_be_bytes());
        output.extend_from_slice(&self.envelope.ciphertext_and_tag);
        output.extend_from_slice(self.ciphertext_digest.as_bytes());
        output
    }

    /// Decodes and validates exact v1 physical bytes without decrypting them.
    pub fn decode(bytes: &[u8]) -> Result<Self, RawEnvelopeError> {
        let mut cursor = ByteCursor::new(bytes);
        if cursor.array::<16>()? != *RAW_ENVELOPE_MAGIC {
            return Err(RawEnvelopeError::InvalidEnvelopeBytes);
        }
        let header_length =
            usize::try_from(cursor.u32()?).map_err(|_| RawEnvelopeError::LengthOverflow)?;
        if header_length > MAX_PUBLIC_HEADER_BYTES {
            return Err(RawEnvelopeError::InvalidEnvelopeBytes);
        }
        let header = RawEnvelopePublicHeaderV1::decode(cursor.bytes(header_length)?)?;
        let ciphertext_length = cursor.u64()?;
        if ciphertext_length != header.ciphertext_length() {
            return Err(RawEnvelopeError::UnexpectedCiphertextLength);
        }
        let ciphertext_length =
            usize::try_from(ciphertext_length).map_err(|_| RawEnvelopeError::LengthOverflow)?;
        let ciphertext_and_tag = cursor.bytes(ciphertext_length)?.to_vec();
        let ciphertext_digest = Digest::from_bytes(cursor.array::<32>()?);
        cursor.finish()?;
        let expected = ciphertext_digest_v1(&ciphertext_and_tag);
        if ciphertext_digest != expected {
            return Err(RawEnvelopeError::CiphertextDigestMismatch);
        }
        Ok(Self {
            header,
            envelope: RawEnvelope { ciphertext_and_tag },
            ciphertext_digest,
        })
    }
}

impl Debug for RawEnvelopeObjectV1 {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RawEnvelopeObjectV1")
            .field("header", &self.header)
            .field("ciphertext_digest", &self.ciphertext_digest)
            .field(
                "ciphertext_and_tag_bytes",
                &self.envelope.ciphertext_and_tag.len(),
            )
            .field("ciphertext_and_tag", &"<redacted>")
            .finish()
    }
}

/// Computes the keyed equality ID over exact encoded entity bytes.
#[must_use]
pub fn raw_object_id_v1(raw_object_subkey: &[u8; 32], exact_encoded_entity: &[u8]) -> Digest {
    keyed_domain_digest(
        raw_object_subkey,
        "aiperf.archive.raw-object.v1",
        &[exact_encoded_entity],
    )
}

/// Computes the exact 32-byte AAD from canonical public-header bytes.
#[must_use]
pub fn raw_envelope_aad_v1(canonical_public_header: &[u8]) -> Digest {
    domain_digest("aiperf.archive.raw-aad.v1", &[canonical_public_header])
}

/// Borrowed equality candidate produced before the archive owner selects a nonce.
#[derive(Clone, Copy)]
pub struct RawObjectCandidate<'a> {
    /// Keyed equality identity calculated by the projection worker.
    pub raw_object_id: Digest,
    /// Exact encoded entity lease consumed or dropped by the archive owner.
    pub exact_encoded_entity: &'a [u8],
}

impl Debug for RawObjectCandidate<'_> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RawObjectCandidate")
            .field("raw_object_id", &self.raw_object_id)
            .field(
                "exact_encoded_entity_bytes",
                &self.exact_encoded_entity.len(),
            )
            .field("exact_encoded_entity", &"<redacted>")
            .finish()
    }
}

/// Required physical coverage for one shared encrypted raw object.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RawCoverageRequirementV1 {
    required_local: bool,
    required_remote: bool,
}

impl RawCoverageRequirementV1 {
    /// Constructs a coverage requirement; remote coverage always implies local.
    pub fn new(required_local: bool, required_remote: bool) -> Result<Self, RawRegistryError> {
        if !required_local {
            return Err(RawRegistryError::InvalidCoverage);
        }
        Ok(Self {
            required_local,
            required_remote,
        })
    }

    /// Constructs the local-only coverage requirement.
    #[must_use]
    pub const fn local_only() -> Self {
        Self {
            required_local: true,
            required_remote: false,
        }
    }

    /// Constructs the local-plus-remote coverage requirement.
    #[must_use]
    pub const fn local_and_remote() -> Self {
        Self {
            required_local: true,
            required_remote: true,
        }
    }

    /// Whether verified local coverage is required.
    #[must_use]
    pub const fn required_local(self) -> bool {
        self.required_local
    }

    /// Whether verified remote coverage is required.
    #[must_use]
    pub const fn required_remote(self) -> bool {
        self.required_remote
    }
}

/// Public physical-object descriptor indexed by raw equality ID.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RawObjectDescriptorV1 {
    /// Keyed raw equality identity.
    pub raw_object_id: Digest,
    /// Deterministic create-if-absent object key.
    pub object_key: String,
    /// Digest of ciphertext plus its tag.
    pub ciphertext_digest: Digest,
    /// Number of ciphertext-plus-tag bytes.
    pub ciphertext_bytes: u64,
    /// Public envelope algorithm.
    pub envelope_algorithm: String,
    /// Public key rotation ID.
    pub key_id: String,
    /// Exact random nonce.
    pub nonce: [u8; RAW_ENVELOPE_NONCE_BYTES],
    /// Required local/remote coverage.
    pub coverage: RawCoverageRequirementV1,
}

impl RawObjectDescriptorV1 {
    /// Derives a secret-free descriptor from one verified physical object.
    pub fn from_object(
        object: &RawEnvelopeObjectV1,
        coverage: RawCoverageRequirementV1,
    ) -> Result<Self, RawRegistryError> {
        let ciphertext_bytes = u64::try_from(object.ciphertext_and_tag().len())
            .map_err(|_| RawRegistryError::LengthOverflow)?;
        Ok(Self {
            raw_object_id: object.header.raw_object_id,
            object_key: raw_object_key_v1(object.header.raw_object_id),
            ciphertext_digest: object.ciphertext_digest,
            ciphertext_bytes,
            envelope_algorithm: object.header.algorithm.clone(),
            key_id: object.header.key_id.clone(),
            nonce: object.header.nonce,
            coverage,
        })
    }

    /// Encodes exact canonical descriptor bytes.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        object([
            (
                "ciphertext_bytes",
                CanonicalJsonValue::Integer(i128::from(self.ciphertext_bytes)),
            ),
            ("ciphertext_digest", string(self.ciphertext_digest.to_hex())),
            (
                "envelope_algorithm",
                string(self.envelope_algorithm.clone()),
            ),
            ("key_id", string(self.key_id.clone())),
            ("nonce", string(hex(&self.nonce))),
            ("object_key", string(self.object_key.clone())),
            ("raw_object_id", string(self.raw_object_id.to_hex())),
            (
                "required_local_coverage",
                CanonicalJsonValue::Bool(self.coverage.required_local),
            ),
            (
                "required_remote_coverage",
                CanonicalJsonValue::Bool(self.coverage.required_remote),
            ),
        ])
        .to_bytes()
    }

    /// Decodes exact canonical descriptor bytes.
    pub fn decode(bytes: &[u8]) -> Result<Self, RawRegistryError> {
        let value = CanonicalJsonValue::parse_canonical(bytes)
            .map_err(RawRegistryError::CanonicalDescriptor)?;
        let values = value
            .as_object()
            .ok_or(RawRegistryError::InvalidObjectDescriptor)?;
        if values.len() != 9 {
            return Err(RawRegistryError::InvalidObjectDescriptor);
        }
        let raw_object_id = parse_digest(values, "raw_object_id")?;
        let object_key = text_registry(values, "object_key")?.to_owned();
        let ciphertext_digest = parse_digest(values, "ciphertext_digest")?;
        let ciphertext_bytes = unsigned_registry(values, "ciphertext_bytes")?;
        let envelope_algorithm = text_registry(values, "envelope_algorithm")?.to_owned();
        let key_id = text_registry(values, "key_id")?.to_owned();
        validate_key_id(&key_id).map_err(RawRegistryError::Key)?;
        let nonce = parse_hex_array_registry(text_registry(values, "nonce")?)?;
        let coverage = RawCoverageRequirementV1::new(
            bool_registry(values, "required_local_coverage")?,
            bool_registry(values, "required_remote_coverage")?,
        )?;
        if object_key != raw_object_key_v1(raw_object_id)
            || envelope_algorithm != RAW_ENVELOPE_ALGORITHM_V1
            || !(RAW_ENVELOPE_TAG_BYTES as u64..=RAW_MAX_CIPHERTEXT_BYTES)
                .contains(&ciphertext_bytes)
        {
            return Err(RawRegistryError::InvalidObjectDescriptor);
        }
        Ok(Self {
            raw_object_id,
            object_key,
            ciphertext_digest,
            ciphertext_bytes,
            envelope_algorithm,
            key_id,
            nonce,
            coverage,
        })
    }
}

/// Append-only `(key_id, nonce)` reservation descriptor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RawNonceReservationV1 {
    /// Public key rotation ID.
    pub key_id: String,
    /// Exact random nonce reserved for the archive lifetime.
    pub nonce: [u8; RAW_ENVELOPE_NONCE_BYTES],
    /// Raw equality identity for which the seal succeeded.
    pub raw_object_id: Digest,
    /// Monotonic successful-object sequence local to this key ID.
    pub key_local_successful_object_sequence: u64,
}

impl RawNonceReservationV1 {
    /// Encodes exact canonical reservation bytes.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        object([
            ("key_id", string(self.key_id.clone())),
            (
                "key_local_successful_object_sequence",
                CanonicalJsonValue::Integer(i128::from(self.key_local_successful_object_sequence)),
            ),
            ("nonce", string(hex(&self.nonce))),
            ("raw_object_id", string(self.raw_object_id.to_hex())),
        ])
        .to_bytes()
    }

    /// Decodes exact canonical reservation bytes.
    pub fn decode(bytes: &[u8]) -> Result<Self, RawRegistryError> {
        let value = CanonicalJsonValue::parse_canonical(bytes)
            .map_err(RawRegistryError::CanonicalDescriptor)?;
        let values = value
            .as_object()
            .ok_or(RawRegistryError::InvalidNonceDescriptor)?;
        if values.len() != 4 {
            return Err(RawRegistryError::InvalidNonceDescriptor);
        }
        let key_id = text_registry(values, "key_id")?.to_owned();
        validate_key_id(&key_id).map_err(RawRegistryError::Key)?;
        let nonce = parse_hex_array_registry(text_registry(values, "nonce")?)?;
        let raw_object_id = parse_digest(values, "raw_object_id")?;
        let key_local_successful_object_sequence =
            unsigned_registry(values, "key_local_successful_object_sequence")?;
        if key_local_successful_object_sequence == 0 {
            return Err(RawRegistryError::InvalidNonceDescriptor);
        }
        Ok(Self {
            key_id,
            nonce,
            raw_object_id,
            key_local_successful_object_sequence,
        })
    }

    /// Returns the domain-separated immutable reservation identity.
    #[must_use]
    pub fn reservation_id(&self) -> Digest {
        domain_digest(
            "aiperf.archive.raw-nonce-reservation.v1",
            &[self.key_id.as_bytes(), &self.nonce],
        )
    }
}

/// Transactional state of one physical object entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum RawObjectStateV1 {
    /// Envelope bytes and reservation are durable in the accepting transaction.
    Pending = 1,
    /// The accepting generation/index transaction has committed.
    Committed = 2,
}

impl RawObjectStateV1 {
    fn from_u8(value: u8) -> Result<Self, RawRegistryError> {
        match value {
            1 => Ok(Self::Pending),
            2 => Ok(Self::Committed),
            _ => Err(RawRegistryError::InvalidObjectState),
        }
    }
}

/// One persistent committed or pending physical-object entry.
#[derive(Clone, Eq, PartialEq)]
pub struct RawRegisteredObjectV1 {
    state: RawObjectStateV1,
    descriptor: RawObjectDescriptorV1,
    exact_envelope_bytes: Vec<u8>,
}

impl RawRegisteredObjectV1 {
    fn new(
        state: RawObjectStateV1,
        descriptor: RawObjectDescriptorV1,
        exact_envelope_bytes: Vec<u8>,
    ) -> Result<Self, RawRegistryError> {
        let object = RawEnvelopeObjectV1::decode(&exact_envelope_bytes)
            .map_err(RawRegistryError::Envelope)?;
        validate_object_descriptor(&descriptor, &object)?;
        Ok(Self {
            state,
            descriptor,
            exact_envelope_bytes,
        })
    }

    /// Returns pending or committed transactional state.
    #[must_use]
    pub const fn state(&self) -> RawObjectStateV1 {
        self.state
    }

    /// Returns the public physical-object descriptor.
    #[must_use]
    pub const fn descriptor(&self) -> &RawObjectDescriptorV1 {
        &self.descriptor
    }

    /// Returns exact byte-identical create-if-absent retry bytes.
    #[must_use]
    pub fn exact_envelope_bytes(&self) -> &[u8] {
        &self.exact_envelope_bytes
    }

    /// Decodes the complete public/encrypted object without opening plaintext.
    pub fn envelope_object(&self) -> Result<RawEnvelopeObjectV1, RawEnvelopeError> {
        RawEnvelopeObjectV1::decode(&self.exact_envelope_bytes)
    }
}

impl Debug for RawRegisteredObjectV1 {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RawRegisteredObjectV1")
            .field("state", &self.state)
            .field("descriptor", &self.descriptor)
            .field("exact_envelope_bytes", &"<redacted>")
            .field("envelope_byte_count", &self.exact_envelope_bytes.len())
            .finish()
    }
}

/// Configurable limits that may only tighten the v1 profile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RawRegistryLimitsV1 {
    max_plaintext_bytes: u64,
    max_successful_objects_per_key: u64,
}

impl RawRegistryLimitsV1 {
    /// Constructs stricter-or-equal positive limits.
    pub fn new(
        max_plaintext_bytes: u64,
        max_successful_objects_per_key: u64,
    ) -> Result<Self, RawRegistryError> {
        if max_plaintext_bytes == 0
            || max_plaintext_bytes > RAW_ENVELOPE_MAX_PLAINTEXT_BYTES
            || max_successful_objects_per_key == 0
            || max_successful_objects_per_key > RAW_ENVELOPE_MAX_OBJECTS_PER_KEY
        {
            return Err(RawRegistryError::InvalidConfiguredLimit);
        }
        Ok(Self {
            max_plaintext_bytes,
            max_successful_objects_per_key,
        })
    }

    /// Returns the admitted exact entity size.
    #[must_use]
    pub const fn max_plaintext_bytes(self) -> u64 {
        self.max_plaintext_bytes
    }

    /// Returns the successful-object cap for one key ID.
    #[must_use]
    pub const fn max_successful_objects_per_key(self) -> u64 {
        self.max_successful_objects_per_key
    }
}

impl Default for RawRegistryLimitsV1 {
    fn default() -> Self {
        Self {
            max_plaintext_bytes: RAW_ENVELOPE_MAX_PLAINTEXT_BYTES,
            max_successful_objects_per_key: RAW_ENVELOPE_MAX_OBJECTS_PER_KEY,
        }
    }
}

/// Inputs injected into one serialized archive-owner prepare operation.
pub struct RawPrepareContext<'a> {
    /// Secret keyed-digest subkey used to verify the candidate equality ID.
    pub raw_object_subkey: &'a [u8; 32],
    /// Public key rotation ID selected by policy.
    pub key_id: &'a str,
    /// Prepared raw-key provider.
    pub key_provider: &'a dyn ArchiveRawKeyProvider,
    /// Injected OS or deterministic-test nonce source.
    pub nonce_source: &'a mut dyn RawNonceSource,
    /// Descriptor-selected envelope profile.
    pub profile: &'a dyn RawEnvelopeProfile,
    /// Physical durability coverage required by the writer.
    pub coverage: RawCoverageRequirementV1,
}

impl Debug for RawPrepareContext<'_> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RawPrepareContext")
            .field("raw_object_subkey", &"<redacted>")
            .field("key_id", &self.key_id)
            .field("key_provider", &self.key_provider)
            .field("nonce_source", &self.nonce_source)
            .field("profile", &self.profile)
            .field("coverage", &self.coverage)
            .finish()
    }
}

/// Whether prepare created or reused an exact registered object.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RawPrepareDispositionV1 {
    /// A new unique nonce was sealed and registered as pending.
    CreatedPending,
    /// An already-pending equality ID supplied the existing exact bytes.
    ReusedPending,
    /// An already-committed equality ID supplied the existing exact bytes.
    ReusedCommitted,
}

/// Result of preparing one equality candidate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RawPrepareOutcomeV1 {
    /// Keyed equality identity used to look up exact physical bytes.
    pub raw_object_id: Digest,
    /// Whether a nonce/seal occurred or an existing object was reused.
    pub disposition: RawPrepareDispositionV1,
}

/// Persistent owner of raw-object and append-only nonce registries.
#[derive(Debug)]
pub struct RawObjectRegistry {
    archive_id: ArchiveId,
    limits: RawRegistryLimitsV1,
    objects: BTreeMap<Digest, RawRegisteredObjectV1>,
    nonce_reservations: BTreeMap<(String, [u8; RAW_ENVELOPE_NONCE_BYTES]), RawNonceReservationV1>,
    key_usage: BTreeMap<String, u64>,
    retired_key_ids: BTreeSet<String>,
}

impl RawObjectRegistry {
    /// Constructs an empty registry for one immutable archive identity.
    #[must_use]
    pub fn new(archive_id: ArchiveId, limits: RawRegistryLimitsV1) -> Self {
        Self {
            archive_id,
            limits,
            objects: BTreeMap::new(),
            nonce_reservations: BTreeMap::new(),
            key_usage: BTreeMap::new(),
            retired_key_ids: BTreeSet::new(),
        }
    }

    /// Returns the archive identity bound into every public header.
    #[must_use]
    pub const fn archive_id(&self) -> ArchiveId {
        self.archive_id
    }

    /// Looks up one pending or committed physical object.
    #[must_use]
    pub fn object(&self, raw_object_id: Digest) -> Option<&RawRegisteredObjectV1> {
        self.objects.get(&raw_object_id)
    }

    /// Looks up one immutable nonce reservation.
    #[must_use]
    pub fn nonce_reservation(
        &self,
        key_id: &str,
        nonce: &[u8; RAW_ENVELOPE_NONCE_BYTES],
    ) -> Option<&RawNonceReservationV1> {
        self.nonce_reservations.get(&(key_id.to_owned(), *nonce))
    }

    /// Returns successful envelope creations charged to one key ID.
    #[must_use]
    pub fn key_usage(&self, key_id: &str) -> u64 {
        self.key_usage.get(key_id).copied().unwrap_or(0)
    }

    /// Permanently retires a key ID for new equality IDs.
    pub fn retire_key(&mut self, key_id: &str) -> Result<(), RawRegistryError> {
        validate_key_id(key_id).map_err(RawRegistryError::Key)?;
        self.retired_key_ids.insert(key_id.to_owned());
        Ok(())
    }

    /// Returns whether new objects are forbidden for one retired key ID.
    #[must_use]
    pub fn is_key_retired(&self, key_id: &str) -> bool {
        self.retired_key_ids.contains(key_id)
    }

    /// Reuses an equality ID or performs exactly one successful randomized seal.
    pub fn prepare_candidate(
        &mut self,
        candidate: RawObjectCandidate<'_>,
        context: RawPrepareContext<'_>,
    ) -> Result<RawPrepareOutcomeV1, RawRegistryError> {
        if raw_object_id_v1(context.raw_object_subkey, candidate.exact_encoded_entity)
            != candidate.raw_object_id
        {
            return Err(RawRegistryError::CandidateIdentityMismatch);
        }

        if let Some(existing) = self.objects.get(&candidate.raw_object_id) {
            if existing.descriptor.coverage != context.coverage {
                return Err(RawRegistryError::CoverageMismatch);
            }
            let disposition = match existing.state {
                RawObjectStateV1::Pending => RawPrepareDispositionV1::ReusedPending,
                RawObjectStateV1::Committed => RawPrepareDispositionV1::ReusedCommitted,
            };
            return Ok(RawPrepareOutcomeV1 {
                raw_object_id: candidate.raw_object_id,
                disposition,
            });
        }

        validate_profile(context.profile.descriptor())?;
        validate_key_id(context.key_id).map_err(RawRegistryError::Key)?;
        let plaintext_bytes = u64::try_from(candidate.exact_encoded_entity.len())
            .map_err(|_| RawRegistryError::LengthOverflow)?;
        if plaintext_bytes > self.limits.max_plaintext_bytes {
            return Err(RawRegistryError::PlaintextLimitExceeded);
        }
        if self.retired_key_ids.contains(context.key_id) {
            return Err(RawRegistryError::KeyRetired);
        }
        let usage = self.key_usage(context.key_id);
        if usage >= self.limits.max_successful_objects_per_key {
            return Err(RawRegistryError::KeyObjectLimitReached);
        }

        let key = context
            .key_provider
            .resolve_key(context.key_id)
            .map_err(RawRegistryError::Key)?;
        if key.key_id() != context.key_id {
            return Err(RawRegistryError::ResolvedKeyIdMismatch);
        }

        for _ in 0..RAW_ENVELOPE_MAX_NONCE_DRAWS {
            let mut nonce = [0_u8; RAW_ENVELOPE_NONCE_BYTES];
            context
                .nonce_source
                .fill_nonce(&mut nonce)
                .map_err(RawRegistryError::Nonce)?;
            let nonce_key = (context.key_id.to_owned(), nonce);
            if self.nonce_reservations.contains_key(&nonce_key) {
                continue;
            }

            let object = build_envelope_object(
                self.archive_id,
                candidate.raw_object_id,
                &key,
                nonce,
                candidate.exact_encoded_entity,
                context.profile,
            )?;
            let sequence = usage
                .checked_add(1)
                .ok_or(RawRegistryError::LengthOverflow)?;
            let reservation = RawNonceReservationV1 {
                key_id: context.key_id.to_owned(),
                nonce,
                raw_object_id: candidate.raw_object_id,
                key_local_successful_object_sequence: sequence,
            };
            let descriptor = RawObjectDescriptorV1::from_object(&object, context.coverage)?;
            let record = RawRegisteredObjectV1::new(
                RawObjectStateV1::Pending,
                descriptor,
                object.exact_bytes(),
            )?;

            // The single archive owner makes these infallible map updates one
            // state transition. Persistence writes both in one accepting WAL
            // frame; recovery rejects either object-without-reservation shape.
            self.nonce_reservations.insert(nonce_key, reservation);
            self.objects.insert(candidate.raw_object_id, record);
            self.key_usage.insert(context.key_id.to_owned(), sequence);
            return Ok(RawPrepareOutcomeV1 {
                raw_object_id: candidate.raw_object_id,
                disposition: RawPrepareDispositionV1::CreatedPending,
            });
        }
        Err(RawRegistryError::NonceCollisionExhausted)
    }

    /// Commits one pending object after its accepting generation commits.
    pub fn commit_object(&mut self, raw_object_id: Digest) -> Result<(), RawRegistryError> {
        let record = self
            .objects
            .get_mut(&raw_object_id)
            .ok_or(RawRegistryError::PendingObjectNotFound)?;
        if record.state != RawObjectStateV1::Pending {
            return Err(RawRegistryError::ObjectAlreadyCommitted);
        }
        record.state = RawObjectStateV1::Committed;
        Ok(())
    }

    /// Aborts one pending object while retaining its nonce reservation forever.
    pub fn abort_object(&mut self, raw_object_id: Digest) -> Result<(), RawRegistryError> {
        let state = self
            .objects
            .get(&raw_object_id)
            .map(|record| record.state)
            .ok_or(RawRegistryError::PendingObjectNotFound)?;
        if state != RawObjectStateV1::Pending {
            return Err(RawRegistryError::ObjectAlreadyCommitted);
        }
        self.objects.remove(&raw_object_id);
        Ok(())
    }

    /// Serializes exact deterministic committed, pending, reservation, and retirement state.
    pub fn durable_bytes(&self) -> Result<Vec<u8>, RawRegistryError> {
        validate_registry(self)?;
        let mut output = Vec::new();
        output.extend_from_slice(RAW_REGISTRY_MAGIC);
        output.extend_from_slice(self.archive_id.as_bytes());
        output.extend_from_slice(&self.limits.max_plaintext_bytes.to_be_bytes());
        output.extend_from_slice(&self.limits.max_successful_objects_per_key.to_be_bytes());
        output.extend_from_slice(&usize_u64(self.objects.len())?.to_be_bytes());
        for record in self.objects.values() {
            output.push(record.state as u8);
            let descriptor = record.descriptor.canonical_bytes();
            write_u32_bytes(&mut output, &descriptor)?;
            write_u64_bytes(&mut output, &record.exact_envelope_bytes)?;
        }
        output.extend_from_slice(&usize_u64(self.nonce_reservations.len())?.to_be_bytes());
        for reservation in self.nonce_reservations.values() {
            write_u32_bytes(&mut output, &reservation.canonical_bytes())?;
        }
        output.extend_from_slice(&usize_u64(self.retired_key_ids.len())?.to_be_bytes());
        for key_id in &self.retired_key_ids {
            let bytes = key_id.as_bytes();
            output.extend_from_slice(
                &u16::try_from(bytes.len())
                    .map_err(|_| RawRegistryError::LengthOverflow)?
                    .to_be_bytes(),
            );
            output.extend_from_slice(bytes);
        }
        let checksum = domain_digest("aiperf.archive.raw-registry.v1", &[&output]);
        output.extend_from_slice(checksum.as_bytes());
        Ok(output)
    }

    /// Recovers exact state from one checksummed deterministic snapshot.
    pub fn recover(bytes: &[u8]) -> Result<Self, RawRegistryError> {
        if bytes.len() < Digest::BYTE_LEN {
            return Err(RawRegistryError::InvalidRegistrySnapshot);
        }
        let payload_length = bytes.len() - Digest::BYTE_LEN;
        let (payload, checksum_bytes) = bytes.split_at(payload_length);
        let expected = domain_digest("aiperf.archive.raw-registry.v1", &[payload]);
        if checksum_bytes != expected.as_bytes() {
            return Err(RawRegistryError::RegistryChecksumMismatch);
        }

        let mut cursor = ByteCursor::new(payload);
        if cursor.array::<16>()? != *RAW_REGISTRY_MAGIC {
            return Err(RawRegistryError::InvalidRegistrySnapshot);
        }
        let archive_id = ArchiveId::new(cursor.array::<16>()?)
            .map_err(|_| RawRegistryError::InvalidRegistrySnapshot)?;
        let limits = RawRegistryLimitsV1::new(cursor.u64()?, cursor.u64()?)?;
        let object_count = bounded_count(cursor.u64()?, cursor.remaining())?;
        let mut objects = Vec::with_capacity(object_count);
        for _ in 0..object_count {
            let state = RawObjectStateV1::from_u8(cursor.u8()?)?;
            let descriptor_length =
                usize::try_from(cursor.u32()?).map_err(|_| RawRegistryError::LengthOverflow)?;
            let descriptor = RawObjectDescriptorV1::decode(cursor.bytes(descriptor_length)?)?;
            let envelope_length =
                usize::try_from(cursor.u64()?).map_err(|_| RawRegistryError::LengthOverflow)?;
            let envelope = cursor.bytes(envelope_length)?.to_vec();
            objects.push(RawRegisteredObjectV1::new(state, descriptor, envelope)?);
        }
        let reservation_count = bounded_count(cursor.u64()?, cursor.remaining())?;
        let mut reservations = Vec::with_capacity(reservation_count);
        for _ in 0..reservation_count {
            let length =
                usize::try_from(cursor.u32()?).map_err(|_| RawRegistryError::LengthOverflow)?;
            reservations.push(RawNonceReservationV1::decode(cursor.bytes(length)?)?);
        }
        let retired_count = bounded_count(cursor.u64()?, cursor.remaining())?;
        let mut retired = Vec::with_capacity(retired_count);
        for _ in 0..retired_count {
            let length = usize::from(cursor.u16()?);
            let value = std::str::from_utf8(cursor.bytes(length)?)
                .map_err(|_| RawRegistryError::InvalidRegistrySnapshot)?
                .to_owned();
            retired.push(value);
        }
        cursor.finish()?;
        Self::recover_from_parts(archive_id, limits, objects, reservations, retired)
    }

    /// Rebuilds state from verified index plus WAL records.
    pub fn recover_from_parts<I, N, R>(
        archive_id: ArchiveId,
        limits: RawRegistryLimitsV1,
        objects: I,
        reservations: N,
        retired_key_ids: R,
    ) -> Result<Self, RawRegistryError>
    where
        I: IntoIterator<Item = RawRegisteredObjectV1>,
        N: IntoIterator<Item = RawNonceReservationV1>,
        R: IntoIterator<Item = String>,
    {
        let mut registry = Self::new(archive_id, limits);
        for record in objects {
            let raw_object_id = record.descriptor.raw_object_id;
            if registry.objects.insert(raw_object_id, record).is_some() {
                return Err(RawRegistryError::DuplicateObjectId);
            }
        }
        for reservation in reservations {
            let key = (reservation.key_id.clone(), reservation.nonce);
            if registry
                .nonce_reservations
                .insert(key, reservation)
                .is_some()
            {
                return Err(RawRegistryError::DuplicateNonceReservation);
            }
        }
        for key_id in retired_key_ids {
            validate_key_id(&key_id).map_err(RawRegistryError::Key)?;
            if !registry.retired_key_ids.insert(key_id) {
                return Err(RawRegistryError::DuplicateRetiredKeyId);
            }
        }
        validate_registry(&registry)?;
        registry.rebuild_key_usage();
        Ok(registry)
    }

    fn rebuild_key_usage(&mut self) {
        self.key_usage.clear();
        for reservation in self.nonce_reservations.values() {
            *self
                .key_usage
                .entry(reservation.key_id.clone())
                .or_default() += 1;
        }
    }
}

/// Raw-envelope, provider, or registry terminalization failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RawEnvelopeError {
    /// A key width disagreed with AES-256.
    InvalidKeyLength,
    /// The profile received a nonce other than 96 bits.
    InvalidNonceLength,
    /// The profile received additional data other than 32 bytes.
    InvalidAadLength,
    /// Exact encoded entity bytes exceed the v1 hard bound.
    PlaintextTooLarge,
    /// A frozen integer width overflowed.
    LengthOverflow,
    /// AEAD sealing failed without exposing inputs.
    SealFailed,
    /// AEAD opening or authentication failed without exposing inputs.
    OpenFailed,
    /// The selected profile returned an impossible ciphertext length.
    UnexpectedCiphertextLength,
    /// Canonical public header decoding failed.
    CanonicalHeader(CanonicalJsonError),
    /// A public header field is absent or invalid.
    InvalidPublicHeader,
    /// Wire framing is malformed or trailing.
    InvalidEnvelopeBytes,
    /// Stored ciphertext bytes disagree with their public digest.
    CiphertextDigestMismatch,
    /// The private decrypted prefix is malformed.
    InvalidPrivatePlaintext,
    /// Decrypted digest, length, or keyed equality ID disagrees.
    PrivateIntegrityMismatch,
    /// The opening key/profile disagrees with the public header.
    HeaderKeyOrProfileMismatch,
    /// Public key-ID validation failed.
    Key(RawKeyError),
}

impl Display for RawEnvelopeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidKeyLength => "raw-envelope key length is invalid",
            Self::InvalidNonceLength => "raw-envelope nonce length is invalid",
            Self::InvalidAadLength => "raw-envelope additional-data length is invalid",
            Self::PlaintextTooLarge => "raw-envelope plaintext exceeds the profile limit",
            Self::LengthOverflow => "raw-envelope length overflowed a frozen width",
            Self::SealFailed => "raw-envelope sealing failed",
            Self::OpenFailed => "raw-envelope authentication or opening failed",
            Self::UnexpectedCiphertextLength => "raw-envelope ciphertext length is invalid",
            Self::CanonicalHeader(_) => "raw-envelope public header is not canonical",
            Self::InvalidPublicHeader => "raw-envelope public header is invalid",
            Self::InvalidEnvelopeBytes => "raw-envelope physical bytes are invalid",
            Self::CiphertextDigestMismatch => "raw-envelope ciphertext digest mismatch",
            Self::InvalidPrivatePlaintext => "raw-envelope private metadata is invalid",
            Self::PrivateIntegrityMismatch => "raw-envelope private integrity check failed",
            Self::HeaderKeyOrProfileMismatch => "raw-envelope key or profile mismatch",
            Self::Key(_) => "raw-envelope key metadata is invalid",
        })
    }
}

impl std::error::Error for RawEnvelopeError {}

/// Invalid limits, transitions, persistence bytes, or injected dependencies.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RawRegistryError {
    /// A configured cap is zero or weaker than the hard v1 cap.
    InvalidConfiguredLimit,
    /// The candidate ID does not match its exact bytes under the supplied subkey.
    CandidateIdentityMismatch,
    /// A profile descriptor is not the exact v1 profile.
    UnsupportedProfile,
    /// Exact encoded entity bytes exceed the configured cap.
    PlaintextLimitExceeded,
    /// A retired key ID cannot encrypt a new equality ID.
    KeyRetired,
    /// A key ID reached its configured successful-object cap.
    KeyObjectLimitReached,
    /// Sixteen random draws all collided with durable reservations.
    NonceCollisionExhausted,
    /// A provider returned a different public key ID than requested.
    ResolvedKeyIdMismatch,
    /// Duplicate references disagree about required physical coverage.
    CoverageMismatch,
    /// Remote coverage or no coverage was requested without required local coverage.
    InvalidCoverage,
    /// The requested pending object does not exist.
    PendingObjectNotFound,
    /// A committed object cannot be committed again or aborted.
    ObjectAlreadyCommitted,
    /// Persistent bytes use an unknown object state.
    InvalidObjectState,
    /// A public raw-object descriptor is invalid.
    InvalidObjectDescriptor,
    /// A public nonce descriptor is invalid.
    InvalidNonceDescriptor,
    /// A snapshot has malformed framing or trailing bytes.
    InvalidRegistrySnapshot,
    /// A snapshot checksum disagrees with its exact payload.
    RegistryChecksumMismatch,
    /// Recovery saw the same raw equality ID twice.
    DuplicateObjectId,
    /// Recovery saw the same `(key ID, nonce)` twice.
    DuplicateNonceReservation,
    /// Recovery saw the same key-local successful sequence twice.
    DuplicateKeySequence,
    /// Recovery saw a gap or non-one-based key-local sequence.
    NonCanonicalKeySequence,
    /// An object has no matching immutable nonce reservation.
    ObjectReservationMismatch,
    /// Recovery repeats one retired key ID.
    DuplicateRetiredKeyId,
    /// A length or count overflowed a frozen width.
    LengthOverflow,
    /// Canonical public descriptor decoding failed.
    CanonicalDescriptor(CanonicalJsonError),
    /// CSPRNG failure.
    Nonce(RawNonceError),
    /// Key lookup or metadata failure.
    Key(RawKeyError),
    /// AEAD or physical-envelope failure.
    Envelope(RawEnvelopeError),
}

impl Display for RawRegistryError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidConfiguredLimit => "raw registry limits are invalid",
            Self::CandidateIdentityMismatch => "raw candidate equality identity mismatch",
            Self::UnsupportedProfile => "raw envelope profile is unsupported",
            Self::PlaintextLimitExceeded => "raw candidate exceeds the configured byte limit",
            Self::KeyRetired => "raw envelope key ID is retired",
            Self::KeyObjectLimitReached => "raw envelope key reached its object limit",
            Self::NonceCollisionExhausted => "raw envelope nonce collision retries exhausted",
            Self::ResolvedKeyIdMismatch => "raw key provider returned the wrong key ID",
            Self::CoverageMismatch => "raw object coverage requirement mismatch",
            Self::InvalidCoverage => "raw object coverage requirement is invalid",
            Self::PendingObjectNotFound => "raw pending object was not found",
            Self::ObjectAlreadyCommitted => "raw object is already committed",
            Self::InvalidObjectState => "raw object state is invalid",
            Self::InvalidObjectDescriptor => "raw object descriptor is invalid",
            Self::InvalidNonceDescriptor => "raw nonce descriptor is invalid",
            Self::InvalidRegistrySnapshot => "raw registry snapshot is invalid",
            Self::RegistryChecksumMismatch => "raw registry snapshot checksum mismatch",
            Self::DuplicateObjectId => "raw registry contains a duplicate object ID",
            Self::DuplicateNonceReservation => {
                "raw registry contains a duplicate nonce reservation"
            }
            Self::DuplicateKeySequence => "raw registry contains a duplicate key-local sequence",
            Self::NonCanonicalKeySequence => "raw registry key-local sequences are non-canonical",
            Self::ObjectReservationMismatch => "raw object and nonce reservation disagree",
            Self::DuplicateRetiredKeyId => "raw registry repeats a retired key ID",
            Self::LengthOverflow => "raw registry length overflowed a frozen width",
            Self::CanonicalDescriptor(_) => "raw registry descriptor is not canonical",
            Self::Nonce(_) => "raw registry nonce generation failed",
            Self::Key(_) => "raw registry key resolution failed",
            Self::Envelope(_) => "raw registry envelope operation failed",
        })
    }
}

impl std::error::Error for RawRegistryError {}

fn build_envelope_object(
    archive_id: ArchiveId,
    raw_object_id: Digest,
    key: &RawEnvelopeKey,
    nonce: [u8; RAW_ENVELOPE_NONCE_BYTES],
    plaintext: &[u8],
    profile: &dyn RawEnvelopeProfile,
) -> Result<RawEnvelopeObjectV1, RawRegistryError> {
    let plaintext_length =
        u64::try_from(plaintext.len()).map_err(|_| RawRegistryError::LengthOverflow)?;
    if plaintext_length > RAW_ENVELOPE_MAX_PLAINTEXT_BYTES {
        return Err(RawRegistryError::Envelope(
            RawEnvelopeError::PlaintextTooLarge,
        ));
    }
    let private = encode_private_plaintext(plaintext)?;
    let ciphertext_length = u64::try_from(private.len())
        .map_err(|_| RawRegistryError::LengthOverflow)?
        .checked_add(RAW_ENVELOPE_TAG_BYTES as u64)
        .ok_or(RawRegistryError::LengthOverflow)?;
    let header = RawEnvelopePublicHeaderV1::new(
        archive_id,
        raw_object_id,
        key.key_id(),
        nonce,
        ciphertext_length,
    )
    .map_err(RawRegistryError::Envelope)?;
    let header_bytes = header.canonical_bytes();
    let aad = raw_envelope_aad_v1(&header_bytes);
    let envelope = profile
        .seal(key, &nonce, aad.as_bytes(), &private)
        .map_err(RawRegistryError::Envelope)?;
    if u64::try_from(envelope.ciphertext_and_tag.len()) != Ok(ciphertext_length) {
        return Err(RawRegistryError::Envelope(
            RawEnvelopeError::UnexpectedCiphertextLength,
        ));
    }
    let ciphertext_digest = ciphertext_digest_v1(&envelope.ciphertext_and_tag);
    Ok(RawEnvelopeObjectV1 {
        header,
        envelope,
        ciphertext_digest,
    })
}

fn encode_private_plaintext(plaintext: &[u8]) -> Result<Vec<u8>, RawRegistryError> {
    let plaintext_length =
        u64::try_from(plaintext.len()).map_err(|_| RawRegistryError::LengthOverflow)?;
    if plaintext_length > RAW_ENVELOPE_MAX_PLAINTEXT_BYTES {
        return Err(RawRegistryError::Envelope(
            RawEnvelopeError::PlaintextTooLarge,
        ));
    }
    let digest = domain_digest("aiperf.archive.raw-plaintext.v1", &[plaintext]);
    let mut private = Vec::with_capacity(
        usize::try_from(RAW_PRIVATE_PREFIX_BYTES)
            .expect("private prefix fits usize")
            .saturating_add(plaintext.len()),
    );
    private.extend_from_slice(RAW_PRIVATE_MAGIC);
    private.extend_from_slice(&plaintext_length.to_be_bytes());
    private.extend_from_slice(digest.as_bytes());
    private.extend_from_slice(plaintext);
    Ok(private)
}

fn decode_private_plaintext(private: &[u8]) -> Result<Vec<u8>, RawEnvelopeError> {
    let mut cursor = ByteCursor::new(private);
    if cursor.array::<16>()? != *RAW_PRIVATE_MAGIC {
        return Err(RawEnvelopeError::InvalidPrivatePlaintext);
    }
    let plaintext_length = cursor.u64()?;
    if plaintext_length > RAW_ENVELOPE_MAX_PLAINTEXT_BYTES {
        return Err(RawEnvelopeError::InvalidPrivatePlaintext);
    }
    let digest = Digest::from_bytes(cursor.array::<32>()?);
    let plaintext_length =
        usize::try_from(plaintext_length).map_err(|_| RawEnvelopeError::LengthOverflow)?;
    let plaintext = cursor.bytes(plaintext_length)?;
    cursor.finish()?;
    if domain_digest("aiperf.archive.raw-plaintext.v1", &[plaintext]) != digest {
        return Err(RawEnvelopeError::PrivateIntegrityMismatch);
    }
    Ok(plaintext.to_vec())
}

fn ciphertext_digest_v1(ciphertext_and_tag: &[u8]) -> Digest {
    domain_digest("aiperf.archive.raw-ciphertext.v1", &[ciphertext_and_tag])
}

fn raw_object_key_v1(raw_object_id: Digest) -> String {
    format!("raw/objects/{}.raw", raw_object_id.to_hex())
}

fn validate_profile(descriptor: &RawEnvelopeDescriptor) -> Result<(), RawRegistryError> {
    descriptor
        .canonical
        .validate()
        .map_err(|_| RawRegistryError::UnsupportedProfile)?;
    if descriptor.profile_id != AES_256_GCM_SIV_RANDOM96_V1_DESCRIPTOR.profile_id
        || descriptor.algorithm != RAW_ENVELOPE_ALGORITHM_V1
        || descriptor.envelope_version != RAW_ENVELOPE_VERSION
        || descriptor.key_bytes != RAW_ENVELOPE_KEY_BYTES
        || descriptor.nonce_bytes != RAW_ENVELOPE_NONCE_BYTES
        || descriptor.tag_bytes != RAW_ENVELOPE_TAG_BYTES
        || descriptor.max_plaintext_bytes != RAW_ENVELOPE_MAX_PLAINTEXT_BYTES
        || descriptor.max_objects_per_key != RAW_ENVELOPE_MAX_OBJECTS_PER_KEY
        || descriptor.canonical.fingerprint() != RAW_ENVELOPE_V1.fingerprint()
    {
        return Err(RawRegistryError::UnsupportedProfile);
    }
    Ok(())
}

fn validate_aead_inputs(nonce: &[u8], aad: &[u8]) -> Result<(), RawEnvelopeError> {
    if nonce.len() != RAW_ENVELOPE_NONCE_BYTES {
        return Err(RawEnvelopeError::InvalidNonceLength);
    }
    if aad.len() != RAW_ENVELOPE_AAD_BYTES {
        return Err(RawEnvelopeError::InvalidAadLength);
    }
    Ok(())
}

fn validate_object_descriptor(
    descriptor: &RawObjectDescriptorV1,
    object: &RawEnvelopeObjectV1,
) -> Result<(), RawRegistryError> {
    let expected = RawObjectDescriptorV1::from_object(object, descriptor.coverage)?;
    if descriptor != &expected {
        return Err(RawRegistryError::InvalidObjectDescriptor);
    }
    Ok(())
}

fn validate_registry(registry: &RawObjectRegistry) -> Result<(), RawRegistryError> {
    let mut sequences: BTreeMap<&str, BTreeSet<u64>> = BTreeMap::new();
    for ((key_id, nonce), reservation) in &registry.nonce_reservations {
        if key_id != &reservation.key_id || nonce != &reservation.nonce {
            return Err(RawRegistryError::InvalidNonceDescriptor);
        }
        let values = sequences.entry(key_id).or_default();
        if !values.insert(reservation.key_local_successful_object_sequence) {
            return Err(RawRegistryError::DuplicateKeySequence);
        }
    }
    for values in sequences.values() {
        let expected_length =
            u64::try_from(values.len()).map_err(|_| RawRegistryError::LengthOverflow)?;
        if values.iter().copied().ne(1..=expected_length) {
            return Err(RawRegistryError::NonCanonicalKeySequence);
        }
        if expected_length > registry.limits.max_successful_objects_per_key {
            return Err(RawRegistryError::KeyObjectLimitReached);
        }
    }
    for (raw_object_id, record) in &registry.objects {
        if raw_object_id != &record.descriptor.raw_object_id {
            return Err(RawRegistryError::InvalidObjectDescriptor);
        }
        let object = record
            .envelope_object()
            .map_err(RawRegistryError::Envelope)?;
        if object.header.archive_id != registry.archive_id {
            return Err(RawRegistryError::InvalidObjectDescriptor);
        }
        validate_object_descriptor(&record.descriptor, &object)?;
        let reservation = registry
            .nonce_reservations
            .get(&(record.descriptor.key_id.clone(), record.descriptor.nonce))
            .ok_or(RawRegistryError::ObjectReservationMismatch)?;
        if reservation.raw_object_id != *raw_object_id {
            return Err(RawRegistryError::ObjectReservationMismatch);
        }
    }
    for key_id in &registry.retired_key_ids {
        validate_key_id(key_id).map_err(RawRegistryError::Key)?;
    }
    Ok(())
}

fn validate_key_id(key_id: &str) -> Result<(), RawKeyError> {
    if key_id.is_empty() || key_id.len() > MAX_KEY_ID_BYTES || key_id.chars().any(char::is_control)
    {
        return Err(RawKeyError::InvalidKeyId);
    }
    Ok(())
}

fn write_u32_bytes(output: &mut Vec<u8>, bytes: &[u8]) -> Result<(), RawRegistryError> {
    output.extend_from_slice(
        &u32::try_from(bytes.len())
            .map_err(|_| RawRegistryError::LengthOverflow)?
            .to_be_bytes(),
    );
    output.extend_from_slice(bytes);
    Ok(())
}

fn write_u64_bytes(output: &mut Vec<u8>, bytes: &[u8]) -> Result<(), RawRegistryError> {
    output.extend_from_slice(&usize_u64(bytes.len())?.to_be_bytes());
    output.extend_from_slice(bytes);
    Ok(())
}

fn usize_u64(value: usize) -> Result<u64, RawRegistryError> {
    u64::try_from(value).map_err(|_| RawRegistryError::LengthOverflow)
}

fn bounded_count(value: u64, remaining: usize) -> Result<usize, RawRegistryError> {
    let value = usize::try_from(value).map_err(|_| RawRegistryError::LengthOverflow)?;
    if value > remaining {
        return Err(RawRegistryError::InvalidRegistrySnapshot);
    }
    Ok(value)
}

fn object<const N: usize>(entries: [(&str, CanonicalJsonValue); N]) -> CanonicalJsonValue {
    CanonicalJsonValue::object(
        entries
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value)),
    )
    .expect("static raw-envelope keys are unique")
}

fn string(value: impl Into<String>) -> CanonicalJsonValue {
    CanonicalJsonValue::String(value.into())
}

fn text<'a>(
    values: &'a BTreeMap<String, CanonicalJsonValue>,
    field: &'static str,
) -> Result<&'a str, RawEnvelopeError> {
    values
        .get(field)
        .and_then(CanonicalJsonValue::as_str)
        .ok_or(RawEnvelopeError::InvalidPublicHeader)
}

fn unsigned(
    values: &BTreeMap<String, CanonicalJsonValue>,
    field: &'static str,
) -> Result<u64, RawEnvelopeError> {
    values
        .get(field)
        .and_then(CanonicalJsonValue::as_i128)
        .and_then(|value| u64::try_from(value).ok())
        .ok_or(RawEnvelopeError::InvalidPublicHeader)
}

fn text_registry<'a>(
    values: &'a BTreeMap<String, CanonicalJsonValue>,
    field: &'static str,
) -> Result<&'a str, RawRegistryError> {
    values
        .get(field)
        .and_then(CanonicalJsonValue::as_str)
        .ok_or(RawRegistryError::InvalidObjectDescriptor)
}

fn unsigned_registry(
    values: &BTreeMap<String, CanonicalJsonValue>,
    field: &'static str,
) -> Result<u64, RawRegistryError> {
    values
        .get(field)
        .and_then(CanonicalJsonValue::as_i128)
        .and_then(|value| u64::try_from(value).ok())
        .ok_or(RawRegistryError::InvalidObjectDescriptor)
}

fn bool_registry(
    values: &BTreeMap<String, CanonicalJsonValue>,
    field: &'static str,
) -> Result<bool, RawRegistryError> {
    match values.get(field) {
        Some(CanonicalJsonValue::Bool(value)) => Ok(*value),
        _ => Err(RawRegistryError::InvalidObjectDescriptor),
    }
}

fn parse_digest(
    values: &BTreeMap<String, CanonicalJsonValue>,
    field: &'static str,
) -> Result<Digest, RawRegistryError> {
    Digest::parse(text_registry(values, field)?)
        .map_err(|_| RawRegistryError::InvalidObjectDescriptor)
}

fn uuid(bytes: &[u8; 16]) -> String {
    let encoded = hex(bytes);
    format!(
        "{}-{}-{}-{}-{}",
        &encoded[0..8],
        &encoded[8..12],
        &encoded[12..16],
        &encoded[16..20],
        &encoded[20..32]
    )
}

fn parse_uuid(value: &str) -> Result<[u8; 16], RawEnvelopeError> {
    if value.len() != 36
        || value.as_bytes().get(8) != Some(&b'-')
        || value.as_bytes().get(13) != Some(&b'-')
        || value.as_bytes().get(18) != Some(&b'-')
        || value.as_bytes().get(23) != Some(&b'-')
    {
        return Err(RawEnvelopeError::InvalidPublicHeader);
    }
    let compact: String = value
        .chars()
        .filter(|character| *character != '-')
        .collect();
    parse_hex_array(&compact)
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

fn parse_hex_array<const N: usize>(value: &str) -> Result<[u8; N], RawEnvelopeError> {
    if value.len() != N * 2 {
        return Err(RawEnvelopeError::InvalidPublicHeader);
    }
    let mut output = [0_u8; N];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let high = decode_hex(pair[0]).ok_or(RawEnvelopeError::InvalidPublicHeader)?;
        let low = decode_hex(pair[1]).ok_or(RawEnvelopeError::InvalidPublicHeader)?;
        output[index] = (high << 4) | low;
    }
    Ok(output)
}

fn parse_hex_array_registry<const N: usize>(value: &str) -> Result<[u8; N], RawRegistryError> {
    parse_hex_array(value).map_err(|_| RawRegistryError::InvalidObjectDescriptor)
}

const fn decode_hex(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

struct ByteCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> ByteCursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    const fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.offset)
    }

    fn bytes(&mut self, length: usize) -> Result<&'a [u8], RawEnvelopeError> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or(RawEnvelopeError::LengthOverflow)?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(RawEnvelopeError::InvalidEnvelopeBytes)?;
        self.offset = end;
        Ok(value)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], RawEnvelopeError> {
        self.bytes(N)?
            .try_into()
            .map_err(|_| RawEnvelopeError::InvalidEnvelopeBytes)
    }

    fn u8(&mut self) -> Result<u8, RawRegistryError> {
        self.bytes(1)
            .map(|bytes| bytes[0])
            .map_err(|_| RawRegistryError::InvalidRegistrySnapshot)
    }

    fn u16(&mut self) -> Result<u16, RawRegistryError> {
        self.array::<2>()
            .map(u16::from_be_bytes)
            .map_err(|_| RawRegistryError::InvalidRegistrySnapshot)
    }

    fn u32(&mut self) -> Result<u32, RawEnvelopeError> {
        self.array::<4>().map(u32::from_be_bytes)
    }

    fn u64(&mut self) -> Result<u64, RawEnvelopeError> {
        self.array::<8>().map(u64::from_be_bytes)
    }

    fn finish(self) -> Result<(), RawEnvelopeError> {
        if self.offset == self.bytes.len() {
            Ok(())
        } else {
            Err(RawEnvelopeError::InvalidEnvelopeBytes)
        }
    }
}

impl From<RawEnvelopeError> for RawRegistryError {
    fn from(error: RawEnvelopeError) -> Self {
        match error {
            RawEnvelopeError::LengthOverflow => Self::LengthOverflow,
            other => Self::Envelope(other),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use super::*;

    const SUBKEY: [u8; 32] = [0x42; 32];

    #[derive(Debug)]
    struct ScriptedNonceSource {
        draws: VecDeque<Result<[u8; RAW_ENVELOPE_NONCE_BYTES], RawNonceError>>,
        calls: usize,
    }

    impl ScriptedNonceSource {
        fn new<I>(draws: I) -> Self
        where
            I: IntoIterator<Item = Result<[u8; RAW_ENVELOPE_NONCE_BYTES], RawNonceError>>,
        {
            Self {
                draws: draws.into_iter().collect(),
                calls: 0,
            }
        }
    }

    impl RawNonceSource for ScriptedNonceSource {
        fn fill_nonce(&mut self, bytes: &mut [u8]) -> Result<(), RawNonceError> {
            self.calls += 1;
            let draw = self
                .draws
                .pop_front()
                .unwrap_or(Err(RawNonceError::EntropyUnavailable))?;
            if bytes.len() != RAW_ENVELOPE_NONCE_BYTES {
                return Err(RawNonceError::EntropyUnavailable);
            }
            bytes.copy_from_slice(&draw);
            Ok(())
        }
    }

    struct CountingProvider {
        key_id: String,
        key: [u8; RAW_ENVELOPE_KEY_BYTES],
        available: AtomicBool,
        calls: AtomicUsize,
    }

    impl CountingProvider {
        fn new(key_id: &str, key: [u8; RAW_ENVELOPE_KEY_BYTES]) -> Self {
            Self {
                key_id: key_id.to_owned(),
                key,
                available: AtomicBool::new(true),
                calls: AtomicUsize::new(0),
            }
        }

        fn unavailable(key_id: &str) -> Self {
            let provider = Self::new(key_id, [0; RAW_ENVELOPE_KEY_BYTES]);
            provider.available.store(false, Ordering::Relaxed);
            provider
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::Relaxed)
        }
    }

    impl Debug for CountingProvider {
        fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
            formatter
                .debug_struct("CountingProvider")
                .field("key_id", &self.key_id)
                .field("key", &"<redacted>")
                .finish()
        }
    }

    impl ArchiveRawKeyProvider for CountingProvider {
        fn resolve_key(&self, key_id: &str) -> Result<RawEnvelopeKey, RawKeyError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            if !self.available.load(Ordering::Relaxed) || key_id != self.key_id {
                return Err(RawKeyError::Unavailable);
            }
            RawEnvelopeKey::new(key_id, self.key)
        }
    }

    #[derive(Debug)]
    struct FailingProfile;

    impl RawEnvelopeProfile for FailingProfile {
        fn descriptor(&self) -> &'static RawEnvelopeDescriptor {
            &AES_256_GCM_SIV_RANDOM96_V1_DESCRIPTOR
        }

        fn seal(
            &self,
            _key: &RawEnvelopeKey,
            _nonce: &[u8],
            _aad: &[u8],
            _plaintext: &[u8],
        ) -> Result<RawEnvelope, RawEnvelopeError> {
            Err(RawEnvelopeError::SealFailed)
        }
    }

    fn archive() -> ArchiveId {
        ArchiveId::new([0x11; 16]).unwrap()
    }

    fn candidate(bytes: &[u8]) -> RawObjectCandidate<'_> {
        RawObjectCandidate {
            raw_object_id: raw_object_id_v1(&SUBKEY, bytes),
            exact_encoded_entity: bytes,
        }
    }

    fn prepare_bytes(
        registry: &mut RawObjectRegistry,
        bytes: &[u8],
        key_id: &str,
        provider: &dyn ArchiveRawKeyProvider,
        nonce_source: &mut dyn RawNonceSource,
        profile: &dyn RawEnvelopeProfile,
    ) -> Result<RawPrepareOutcomeV1, RawRegistryError> {
        registry.prepare_candidate(
            candidate(bytes),
            RawPrepareContext {
                raw_object_subkey: &SUBKEY,
                key_id,
                key_provider: provider,
                nonce_source,
                profile,
                coverage: RawCoverageRequirementV1::local_only(),
            },
        )
    }

    #[test]
    fn checked_in_descriptor_is_canonical_and_pins_exact_profile() {
        RAW_ENVELOPE_V1.validate().unwrap();
        let descriptor = &AES_256_GCM_SIV_RANDOM96_V1_DESCRIPTOR;
        assert_eq!(descriptor.key_bytes, 32);
        assert_eq!(descriptor.nonce_bytes, 12);
        assert_eq!(descriptor.tag_bytes, 16);
        assert_eq!(descriptor.max_plaintext_bytes, 1 << 30);
        assert_eq!(descriptor.max_objects_per_key, 1 << 29);
        assert_eq!(RAW_ENVELOPE_MAX_NONCE_DRAWS, 16);
    }

    #[test]
    fn canonical_public_header_is_the_only_aad_preimage() {
        let raw_object_id = Digest::from_bytes([0x33; 32]);
        let header =
            RawEnvelopePublicHeaderV1::new(archive(), raw_object_id, "key-1", [1; 12], 79).unwrap();
        let expected = format!(
            "{{\"algorithm\":\"AEAD_AES_256_GCM_SIV\",\"archive_id\":\"11111111-1111-1111-1111-111111111111\",\"ciphertext_length\":79,\"envelope_version\":1,\"key_id\":\"key-1\",\"nonce\":\"010101010101010101010101\",\"raw_object_id\":\"{}\"}}",
            raw_object_id.to_hex()
        );
        assert_eq!(header.canonical_bytes(), expected.as_bytes());
        assert_eq!(
            raw_envelope_aad_v1(expected.as_bytes()),
            domain_digest("aiperf.archive.raw-aad.v1", &[expected.as_bytes()])
        );
        assert_eq!(
            RawEnvelopePublicHeaderV1::decode(expected.as_bytes()).unwrap(),
            header
        );
    }

    #[test]
    fn aes_profile_round_trips_private_digest_and_length_metadata() {
        let mut registry = RawObjectRegistry::new(archive(), RawRegistryLimitsV1::default());
        let provider = CountingProvider::new("key-1", [0x55; 32]);
        let profile = Aes256GcmSivRandom96V1;
        let mut nonce = ScriptedNonceSource::new([Ok([0x77; 12])]);
        let outcome = prepare_bytes(
            &mut registry,
            b"exact encoded payload",
            "key-1",
            &provider,
            &mut nonce,
            &profile,
        )
        .unwrap();
        let record = registry.object(outcome.raw_object_id).unwrap();
        let object = record.envelope_object().unwrap();
        let key = provider.resolve_key("key-1").unwrap();
        assert_eq!(
            profile.open(&key, &SUBKEY, &object).unwrap(),
            b"exact encoded payload"
        );
        assert_eq!(
            object.header().ciphertext_length(),
            RAW_PRIVATE_PREFIX_BYTES + 21 + RAW_ENVELOPE_TAG_BYTES as u64
        );
        assert_eq!(
            RawEnvelopeObjectV1::decode(record.exact_envelope_bytes()).unwrap(),
            object
        );
    }

    #[test]
    fn duplicate_pending_and_committed_candidates_reuse_exact_bytes_without_dependencies() {
        let mut registry = RawObjectRegistry::new(archive(), RawRegistryLimitsV1::default());
        let provider = CountingProvider::new("key-1", [9; 32]);
        let profile = Aes256GcmSivRandom96V1;
        let mut nonce = ScriptedNonceSource::new([Ok([1; 12])]);
        let created = prepare_bytes(
            &mut registry,
            b"same bytes",
            "key-1",
            &provider,
            &mut nonce,
            &profile,
        )
        .unwrap();
        let exact = registry
            .object(created.raw_object_id)
            .unwrap()
            .exact_envelope_bytes()
            .to_vec();
        assert_eq!(provider.calls(), 1);

        let unavailable = CountingProvider::unavailable("key-1");
        let mut no_entropy = ScriptedNonceSource::new([]);
        let reused = prepare_bytes(
            &mut registry,
            b"same bytes",
            "key-1",
            &unavailable,
            &mut no_entropy,
            &profile,
        )
        .unwrap();
        assert_eq!(reused.disposition, RawPrepareDispositionV1::ReusedPending);
        assert_eq!(unavailable.calls(), 0);
        assert_eq!(no_entropy.calls, 0);
        assert_eq!(
            registry
                .object(reused.raw_object_id)
                .unwrap()
                .exact_envelope_bytes(),
            exact
        );

        registry.commit_object(reused.raw_object_id).unwrap();
        let reused = prepare_bytes(
            &mut registry,
            b"same bytes",
            "key-1",
            &unavailable,
            &mut no_entropy,
            &profile,
        )
        .unwrap();
        assert_eq!(reused.disposition, RawPrepareDispositionV1::ReusedCommitted);
        assert_eq!(registry.key_usage("key-1"), 1);
    }

    #[test]
    fn forced_nonce_collision_retries_and_charges_only_success() {
        let mut registry = RawObjectRegistry::new(archive(), RawRegistryLimitsV1::default());
        let provider = CountingProvider::new("key-1", [9; 32]);
        let profile = Aes256GcmSivRandom96V1;
        let mut first = ScriptedNonceSource::new([Ok([7; 12])]);
        prepare_bytes(
            &mut registry,
            b"one",
            "key-1",
            &provider,
            &mut first,
            &profile,
        )
        .unwrap();

        let mut second = ScriptedNonceSource::new([Ok([7; 12]), Ok([8; 12])]);
        prepare_bytes(
            &mut registry,
            b"two",
            "key-1",
            &provider,
            &mut second,
            &profile,
        )
        .unwrap();
        assert_eq!(second.calls, 2);
        assert_eq!(registry.key_usage("key-1"), 2);
        assert!(registry.nonce_reservation("key-1", &[7; 12]).is_some());
        assert!(registry.nonce_reservation("key-1", &[8; 12]).is_some());
    }

    #[test]
    fn sixteen_collisions_fail_closed_without_advancing_usage() {
        let mut registry = RawObjectRegistry::new(archive(), RawRegistryLimitsV1::default());
        let provider = CountingProvider::new("key-1", [9; 32]);
        let profile = Aes256GcmSivRandom96V1;
        let mut first = ScriptedNonceSource::new([Ok([7; 12])]);
        prepare_bytes(
            &mut registry,
            b"one",
            "key-1",
            &provider,
            &mut first,
            &profile,
        )
        .unwrap();
        let mut collisions =
            ScriptedNonceSource::new((0..16).map(|_| Ok([7; 12])).collect::<Vec<_>>());
        assert_eq!(
            prepare_bytes(
                &mut registry,
                b"two",
                "key-1",
                &provider,
                &mut collisions,
                &profile,
            ),
            Err(RawRegistryError::NonceCollisionExhausted)
        );
        assert_eq!(collisions.calls, 16);
        assert_eq!(registry.key_usage("key-1"), 1);
        assert!(registry.object(candidate(b"two").raw_object_id).is_none());
    }

    #[test]
    fn rng_key_and_aead_failures_leave_no_partial_registry_state() {
        let provider = CountingProvider::new("key-1", [9; 32]);
        let profile = Aes256GcmSivRandom96V1;

        let mut registry = RawObjectRegistry::new(archive(), RawRegistryLimitsV1::default());
        let mut failed_rng = ScriptedNonceSource::new([Err(RawNonceError::EntropyUnavailable)]);
        assert_eq!(
            prepare_bytes(
                &mut registry,
                b"one",
                "key-1",
                &provider,
                &mut failed_rng,
                &profile,
            ),
            Err(RawRegistryError::Nonce(RawNonceError::EntropyUnavailable))
        );
        assert_eq!(registry.key_usage("key-1"), 0);

        let unavailable = CountingProvider::unavailable("key-1");
        let mut unused_nonce = ScriptedNonceSource::new([Ok([1; 12])]);
        assert_eq!(
            prepare_bytes(
                &mut registry,
                b"one",
                "key-1",
                &unavailable,
                &mut unused_nonce,
                &profile,
            ),
            Err(RawRegistryError::Key(RawKeyError::Unavailable))
        );
        assert_eq!(unused_nonce.calls, 0);
        assert_eq!(registry.key_usage("key-1"), 0);

        let mut failed_aead_nonce = ScriptedNonceSource::new([Ok([2; 12])]);
        assert_eq!(
            prepare_bytes(
                &mut registry,
                b"one",
                "key-1",
                &provider,
                &mut failed_aead_nonce,
                &FailingProfile,
            ),
            Err(RawRegistryError::Envelope(RawEnvelopeError::SealFailed))
        );
        assert_eq!(registry.key_usage("key-1"), 0);
        assert!(registry.nonce_reservation("key-1", &[2; 12]).is_none());
    }

    #[test]
    fn strict_limits_force_rekey_and_retirement_never_reopens_key_id() {
        assert_eq!(
            RawRegistryLimitsV1::new(RAW_ENVELOPE_MAX_PLAINTEXT_BYTES + 1, 1),
            Err(RawRegistryError::InvalidConfiguredLimit)
        );
        assert_eq!(
            RawRegistryLimitsV1::new(1, RAW_ENVELOPE_MAX_OBJECTS_PER_KEY + 1),
            Err(RawRegistryError::InvalidConfiguredLimit)
        );

        let limits = RawRegistryLimitsV1::new(3, 2).unwrap();
        let mut registry = RawObjectRegistry::new(archive(), limits);
        let providers = MemoryRawKeyProvider::new([
            ("key-1".to_owned(), [1; 32]),
            ("key-2".to_owned(), [2; 32]),
        ])
        .unwrap();
        let profile = Aes256GcmSivRandom96V1;
        let mut nonces =
            ScriptedNonceSource::new([Ok([1; 12]), Ok([2; 12]), Ok([3; 12]), Ok([4; 12])]);
        prepare_bytes(
            &mut registry,
            b"a",
            "key-1",
            &providers,
            &mut nonces,
            &profile,
        )
        .unwrap();
        prepare_bytes(
            &mut registry,
            b"b",
            "key-1",
            &providers,
            &mut nonces,
            &profile,
        )
        .unwrap();
        assert_eq!(
            prepare_bytes(
                &mut registry,
                b"c",
                "key-1",
                &providers,
                &mut nonces,
                &profile,
            ),
            Err(RawRegistryError::KeyObjectLimitReached)
        );
        prepare_bytes(
            &mut registry,
            b"c",
            "key-2",
            &providers,
            &mut nonces,
            &profile,
        )
        .unwrap();
        registry.retire_key("key-2").unwrap();
        assert_eq!(
            prepare_bytes(
                &mut registry,
                b"d",
                "key-2",
                &providers,
                &mut nonces,
                &profile,
            ),
            Err(RawRegistryError::KeyRetired)
        );
        assert_eq!(
            prepare_bytes(
                &mut registry,
                b"toolong",
                "key-2",
                &providers,
                &mut nonces,
                &profile,
            ),
            Err(RawRegistryError::PlaintextLimitExceeded)
        );
    }

    #[test]
    fn abort_keeps_nonce_and_usage_durable_across_recovery() {
        let limits = RawRegistryLimitsV1::new(1024, 4).unwrap();
        let mut registry = RawObjectRegistry::new(archive(), limits);
        let provider = CountingProvider::new("key-1", [9; 32]);
        let profile = Aes256GcmSivRandom96V1;
        let mut nonce = ScriptedNonceSource::new([Ok([5; 12])]);
        let first = prepare_bytes(
            &mut registry,
            b"aborted",
            "key-1",
            &provider,
            &mut nonce,
            &profile,
        )
        .unwrap();
        registry.abort_object(first.raw_object_id).unwrap();
        assert!(registry.object(first.raw_object_id).is_none());
        assert_eq!(registry.key_usage("key-1"), 1);

        let snapshot = registry.durable_bytes().unwrap();
        let mut recovered = RawObjectRegistry::recover(&snapshot).unwrap();
        assert_eq!(recovered.key_usage("key-1"), 1);
        assert!(recovered.nonce_reservation("key-1", &[5; 12]).is_some());

        let mut retry = ScriptedNonceSource::new([Ok([5; 12]), Ok([6; 12])]);
        prepare_bytes(
            &mut recovered,
            b"aborted",
            "key-1",
            &provider,
            &mut retry,
            &profile,
        )
        .unwrap();
        assert_eq!(retry.calls, 2);
        assert_eq!(recovered.key_usage("key-1"), 2);
    }

    #[test]
    fn pending_and_committed_crash_snapshots_preserve_exact_retry_bytes() {
        let mut registry = RawObjectRegistry::new(archive(), RawRegistryLimitsV1::default());
        let provider = CountingProvider::new("key-1", [9; 32]);
        let profile = Aes256GcmSivRandom96V1;
        let mut nonce = ScriptedNonceSource::new([Ok([1; 12])]);
        let outcome = prepare_bytes(
            &mut registry,
            b"crash boundary",
            "key-1",
            &provider,
            &mut nonce,
            &profile,
        )
        .unwrap();
        let pending_bytes = registry
            .object(outcome.raw_object_id)
            .unwrap()
            .exact_envelope_bytes()
            .to_vec();

        let mut recovered = RawObjectRegistry::recover(&registry.durable_bytes().unwrap()).unwrap();
        assert_eq!(
            recovered.object(outcome.raw_object_id).unwrap().state(),
            RawObjectStateV1::Pending
        );
        assert_eq!(
            recovered
                .object(outcome.raw_object_id)
                .unwrap()
                .exact_envelope_bytes(),
            pending_bytes
        );
        recovered.commit_object(outcome.raw_object_id).unwrap();

        let recovered = RawObjectRegistry::recover(&recovered.durable_bytes().unwrap()).unwrap();
        assert_eq!(
            recovered.object(outcome.raw_object_id).unwrap().state(),
            RawObjectStateV1::Committed
        );
        assert_eq!(
            recovered
                .object(outcome.raw_object_id)
                .unwrap()
                .exact_envelope_bytes(),
            pending_bytes
        );
    }

    #[test]
    fn recovery_rejects_corrupt_and_impossible_crash_states() {
        let mut registry = RawObjectRegistry::new(archive(), RawRegistryLimitsV1::default());
        let provider = CountingProvider::new("key-1", [9; 32]);
        let profile = Aes256GcmSivRandom96V1;
        let mut nonce = ScriptedNonceSource::new([Ok([1; 12])]);
        let outcome = prepare_bytes(
            &mut registry,
            b"state",
            "key-1",
            &provider,
            &mut nonce,
            &profile,
        )
        .unwrap();

        let mut corrupt = registry.durable_bytes().unwrap();
        corrupt[20] ^= 1;
        assert_eq!(
            RawObjectRegistry::recover(&corrupt).unwrap_err(),
            RawRegistryError::RegistryChecksumMismatch
        );

        let record = registry.object(outcome.raw_object_id).unwrap().clone();
        assert_eq!(
            RawObjectRegistry::recover_from_parts(
                archive(),
                RawRegistryLimitsV1::default(),
                [record],
                [],
                [],
            )
            .unwrap_err(),
            RawRegistryError::ObjectReservationMismatch
        );

        let reservation = registry
            .nonce_reservation("key-1", &[1; 12])
            .unwrap()
            .clone();
        let mut duplicate_sequence = reservation.clone();
        duplicate_sequence.nonce = [2; 12];
        assert_eq!(
            RawObjectRegistry::recover_from_parts(
                archive(),
                RawRegistryLimitsV1::default(),
                [],
                [reservation, duplicate_sequence],
                [],
            )
            .unwrap_err(),
            RawRegistryError::DuplicateKeySequence
        );
    }

    #[test]
    fn public_descriptors_exclude_plaintext_source_frame_and_key_material() {
        let mut registry = RawObjectRegistry::new(archive(), RawRegistryLimitsV1::default());
        let provider = CountingProvider::new("key-1", [0xde; 32]);
        let profile = Aes256GcmSivRandom96V1;
        let mut nonce = ScriptedNonceSource::new([Ok([1; 12])]);
        let outcome = prepare_bytes(
            &mut registry,
            b"known secret body marker",
            "key-1",
            &provider,
            &mut nonce,
            &profile,
        )
        .unwrap();
        let object = registry.object(outcome.raw_object_id).unwrap();
        let descriptor = String::from_utf8(object.descriptor().canonical_bytes()).unwrap();
        assert!(!descriptor.contains("plaintext"));
        assert!(!descriptor.contains("source"));
        assert!(!descriptor.contains("frame"));
        assert!(!descriptor.contains("known secret body marker"));
        let reservation = registry.nonce_reservation("key-1", &[1; 12]).unwrap();
        let reservation = String::from_utf8(reservation.canonical_bytes()).unwrap();
        assert!(!reservation.contains("plaintext"));
        assert!(!reservation.contains("source"));
        assert!(!reservation.contains("frame"));
        assert!(!reservation.contains("known secret body marker"));
    }

    #[test]
    fn key_envelope_and_errors_have_redacted_diagnostics() {
        let key = RawEnvelopeKey::new("rotation-1", [222; 32]).unwrap();
        let key_debug = format!("{key:?}");
        assert!(key_debug.contains("<redacted>"));
        assert!(!key_debug.contains("222"));

        let candidate = RawObjectCandidate {
            raw_object_id: Digest::from_bytes([3; 32]),
            exact_encoded_entity: b"never log this exact marker",
        };
        let candidate_debug = format!("{candidate:?}");
        assert!(candidate_debug.contains("<redacted>"));
        assert!(!candidate_debug.contains("never log this exact marker"));
        assert!(
            !RawEnvelopeError::SealFailed
                .to_string()
                .contains("never log this exact marker")
        );
    }

    #[test]
    fn candidate_mismatch_and_ciphertext_tamper_fail_closed() {
        let mut registry = RawObjectRegistry::new(archive(), RawRegistryLimitsV1::default());
        let provider = CountingProvider::new("key-1", [9; 32]);
        let profile = Aes256GcmSivRandom96V1;
        let mut nonce = ScriptedNonceSource::new([Ok([1; 12])]);
        let mismatched = RawObjectCandidate {
            raw_object_id: raw_object_id_v1(&SUBKEY, b"other"),
            exact_encoded_entity: b"body",
        };
        assert_eq!(
            registry.prepare_candidate(
                mismatched,
                RawPrepareContext {
                    raw_object_subkey: &SUBKEY,
                    key_id: "key-1",
                    key_provider: &provider,
                    nonce_source: &mut nonce,
                    profile: &profile,
                    coverage: RawCoverageRequirementV1::local_only(),
                },
            ),
            Err(RawRegistryError::CandidateIdentityMismatch)
        );
        assert_eq!(provider.calls(), 0);
        assert_eq!(nonce.calls, 0);

        let outcome = prepare_bytes(
            &mut registry,
            b"body",
            "key-1",
            &provider,
            &mut nonce,
            &profile,
        )
        .unwrap();
        let mut exact = registry
            .object(outcome.raw_object_id)
            .unwrap()
            .exact_envelope_bytes()
            .to_vec();
        let header_length = u32::from_be_bytes(exact[16..20].try_into().unwrap()) as usize;
        exact[20 + header_length + 8] ^= 1;
        assert_eq!(
            RawEnvelopeObjectV1::decode(&exact),
            Err(RawEnvelopeError::CiphertextDigestMismatch)
        );
    }

    #[test]
    fn os_csprng_concrete_fills_independent_96_bit_nonces() {
        let mut source = OsRawNonceSource;
        let mut first = [0_u8; RAW_ENVELOPE_NONCE_BYTES];
        let mut second = [0_u8; RAW_ENVELOPE_NONCE_BYTES];
        source.fill_nonce(&mut first).unwrap();
        source.fill_nonce(&mut second).unwrap();
        assert_ne!(first, second);
    }
}
