// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Versioned authenticated envelope for sensitive streaming session state.
//!
//! The envelope protects participant state that contains live target output.
//! Its associated data binds every fact that must not be substitutable — run,
//! generation, participant, schema, policy digest, key id, and the exact
//! plaintext length — while the plaintext BLAKE3 is sealed under the same tag
//! as an encrypted prefix, so a ciphertext lifted from one generation cannot be
//! replayed into another and an equal-length substitution cannot open. Keys and
//! plaintext are zeroized on drop, and neither the key nor the plaintext is
//! reachable through `Debug`.
//!
//! Key material never appears in authored configuration: configuration names a
//! [`SensitiveStateKeyId`], and the bytes behind it are resolved once per
//! process from a launcher-owned inherited descriptor or an exact-`0600`
//! no-follow file.

use std::collections::BTreeMap;
use std::fmt;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use chacha20poly1305::aead::{Aead, KeyInit, Payload};
use chacha20poly1305::{XChaCha20Poly1305, XNonce};
use rand::TryRngCore;
use serde::{Deserialize, Serialize};
use zeroize::{Zeroize, Zeroizing};

use super::{
    checkpoint::{CheckpointGeneration, CheckpointParticipantId, StreamRunIdentity},
    failure::{SensitiveStateError, SensitiveStateFailureCode},
    identity::ContentDigest,
};

pub use super::policy::SensitiveStateKeyId;

/// Envelope format version written by this generation of the runtime.
pub const SENSITIVE_STATE_ENVELOPE_VERSION: u8 = 1;

/// Domain separator for the envelope's associated data.
const AAD_DOMAIN: &[u8] = b"aiperf.stream.sensitive-state.v1";

/// Fixed descriptor a launcher may pass sensitive-state material on.
pub const SENSITIVE_STATE_FD: i32 = 4;

/// Environment variable carrying only the descriptor number, never key bytes.
pub const SENSITIVE_STATE_FD_ENV: &str = "AIPERF_STREAMING_SENSITIVE_STATE_FD";

/// Environment variable naming a private key file. Contents never appear in env.
pub const SENSITIVE_STATE_FILE_ENV: &str = "AIPERF_STREAMING_SENSITIVE_STATE_FILE";

/// Upper bound on one process's sensitive-state material.
///
/// One entry is a selector plus 32 hex-encoded key bytes, so this admits
/// thousands of selectors while bounding a hostile or truncated source.
pub const MAX_SENSITIVE_MATERIAL_BYTES: usize = 64 * 1024;

fn refuse(code: SensitiveStateFailureCode) -> SensitiveStateError {
    SensitiveStateError::new(code)
}

/// Resolved symmetric key material.
///
/// `key` is `Zeroizing`, so the bytes are wiped when the value drops. There is
/// no `Debug` derive: the manual implementation below prints the id only.
pub struct SensitiveStateKey {
    /// Selector this material was resolved for.
    pub key_id: SensitiveStateKeyId,
    /// Raw 256-bit key, zeroized on drop.
    pub key: Zeroizing<[u8; 32]>,
}

impl fmt::Debug for SensitiveStateKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SensitiveStateKey")
            .field("key_id", &self.key_id)
            .finish_non_exhaustive()
    }
}

/// Narrow host authority that turns a key id into key material.
///
/// `Send + Sync` because the resolver is installed once per process and shared
/// like every other `Arc<dyn …>` host authority.
pub trait StreamingSensitiveStateKeyResolver: fmt::Debug + Send + Sync {
    /// Resolve one selector, or refuse.
    fn resolve(&self, key_id: &SensitiveStateKeyId)
    -> Result<SensitiveStateKey, SensitiveStateError>;
}

/// Complete associated-data context for one envelope.
///
/// Every field is bound with length-delimited framing, so no two distinct
/// contexts can produce the same associated-data byte string by concatenation
/// ambiguity. The context is a value rather than a builder precisely so a
/// caller cannot omit a binding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SensitiveStateContext {
    /// Logical run owning the protected state.
    pub run: StreamRunIdentity,
    /// Checkpoint generation the state is staged into.
    pub generation: CheckpointGeneration,
    /// Participant that owns the state.
    pub participant: CheckpointParticipantId,
    /// Participant state schema identity.
    pub schema_id: String,
    /// Participant state schema version.
    pub schema_version: u32,
    /// Digest of the resolved sensitive-state policy.
    pub policy_digest: ContentDigest,
}

/// Versioned authenticated envelope.
///
/// The serialized form carries only the version, key id, nonce, and ciphertext.
/// The manual `Debug` prints no ciphertext byte, so an envelope in a log line or
/// a panic message cannot leak protected content.
#[derive(Clone, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SensitiveStateEnvelope {
    /// Envelope format version.
    pub version: u8,
    /// Selector for the key that sealed this envelope.
    pub key_id: SensitiveStateKeyId,
    /// Fresh 24-byte XChaCha20-Poly1305 nonce.
    pub nonce: [u8; 24],
    /// Ciphertext with appended Poly1305 tag.
    pub ciphertext: Vec<u8>,
}

impl fmt::Debug for SensitiveStateEnvelope {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SensitiveStateEnvelope")
            .field("version", &self.version)
            .field("key_id", &self.key_id)
            .field("ciphertext_len", &self.ciphertext.len())
            .finish_non_exhaustive()
    }
}

fn update_field(buffer: &mut Vec<u8>, field: &[u8]) {
    buffer.extend_from_slice(&(field.len() as u64).to_le_bytes());
    buffer.extend_from_slice(field);
}

/// Length of the BLAKE3 plaintext digest carried inside the sealed message.
const PLAINTEXT_DIGEST_LEN: usize = 32;

/// Poly1305 tag length appended by XChaCha20-Poly1305.
const TAG_LEN: usize = 16;

/// Build the length-delimited associated data for one envelope.
///
/// Order and framing are frozen: changing either invalidates every previously
/// written envelope, so this function *is* the format.
///
/// The plaintext length is bound here, where it is recomputable at open time
/// from the ciphertext length. The plaintext BLAKE3 is bound as an authenticated
/// *encrypted* prefix instead (see [`encrypt_sensitive`]): binding it in the
/// clear would publish a confirmation oracle for guessed state, while sealing it
/// under the same tag still makes a truncation or an equal-length substitution
/// fail authentication rather than open to a different value.
fn associated_data(
    context: &SensitiveStateContext,
    key_id: &SensitiveStateKeyId,
    plaintext_len: u64,
) -> Vec<u8> {
    let mut buffer = Vec::new();
    update_field(&mut buffer, AAD_DOMAIN);
    update_field(&mut buffer, context.run.logical_replay_run().as_bytes());
    update_field(&mut buffer, &context.generation.epoch.get().to_le_bytes());
    update_field(&mut buffer, context.generation.digest.as_bytes());
    update_field(&mut buffer, context.participant.as_str().as_bytes());
    update_field(&mut buffer, context.schema_id.as_bytes());
    update_field(&mut buffer, &context.schema_version.to_le_bytes());
    update_field(&mut buffer, context.policy_digest.as_bytes());
    update_field(&mut buffer, key_id.as_str().as_bytes());
    update_field(&mut buffer, &plaintext_len.to_le_bytes());
    buffer
}

fn mint_nonce() -> Result<[u8; 24], SensitiveStateError> {
    let mut nonce = [0_u8; 24];
    // Random rather than counter-based: a 192-bit XChaCha nonce is designed for
    // random generation, and a counter would have to survive checkpoint restore
    // where a restored-and-diverged value is catastrophic nonce reuse.
    rand::rngs::OsRng
        .try_fill_bytes(&mut nonce)
        .map_err(|_| refuse(SensitiveStateFailureCode::NonceUnavailable))?;
    Ok(nonce)
}

/// Seal `plaintext` under the key named by `key_id`, bound to `context`.
///
/// The sealed message is `blake3(plaintext) || plaintext`, so the digest is
/// authenticated by the same tag but never exposed in the clear.
pub fn encrypt_sensitive(
    resolver: &dyn StreamingSensitiveStateKeyResolver,
    key_id: &SensitiveStateKeyId,
    context: &SensitiveStateContext,
    plaintext: &[u8],
) -> Result<SensitiveStateEnvelope, SensitiveStateError> {
    let material = resolver.resolve(key_id)?;
    let cipher = XChaCha20Poly1305::new_from_slice(material.key.as_slice())
        .map_err(|_| refuse(SensitiveStateFailureCode::KeyMalformed))?;
    let aad = associated_data(context, key_id, plaintext.len() as u64);
    let mut message = Zeroizing::new(Vec::with_capacity(PLAINTEXT_DIGEST_LEN + plaintext.len()));
    message.extend_from_slice(blake3::hash(plaintext).as_bytes());
    message.extend_from_slice(plaintext);
    let nonce = mint_nonce()?;
    let ciphertext = cipher
        .encrypt(
            XNonce::from_slice(&nonce),
            Payload {
                msg: &message,
                aad: &aad,
            },
        )
        .map_err(|_| refuse(SensitiveStateFailureCode::Authentication))?;
    Ok(SensitiveStateEnvelope {
        version: SENSITIVE_STATE_ENVELOPE_VERSION,
        key_id: key_id.clone(),
        nonce,
        ciphertext,
    })
}

/// Open `envelope`, refusing any context mismatch.
///
/// A wrong key, a flipped ciphertext or tag bit, and any associated-data
/// mismatch all surface as [`SensitiveStateFailureCode::Authentication`], and
/// none of them returns partial plaintext: the AEAD open fails before any byte
/// is handed back, and the recovered digest is re-verified afterwards.
pub fn decrypt_sensitive(
    resolver: &dyn StreamingSensitiveStateKeyResolver,
    context: &SensitiveStateContext,
    envelope: &SensitiveStateEnvelope,
) -> Result<Zeroizing<Vec<u8>>, SensitiveStateError> {
    if envelope.version != SENSITIVE_STATE_ENVELOPE_VERSION {
        return Err(refuse(SensitiveStateFailureCode::UnsupportedVersion));
    }
    let material = resolver.resolve(&envelope.key_id)?;
    let cipher = XChaCha20Poly1305::new_from_slice(material.key.as_slice())
        .map_err(|_| refuse(SensitiveStateFailureCode::KeyMalformed))?;
    // The plaintext length is recomputed from the ciphertext rather than read
    // from the envelope, so a caller cannot steer the associated data.
    let plaintext_len = envelope
        .ciphertext
        .len()
        .checked_sub(TAG_LEN + PLAINTEXT_DIGEST_LEN)
        .ok_or_else(|| refuse(SensitiveStateFailureCode::Authentication))?;
    let aad = associated_data(context, &envelope.key_id, plaintext_len as u64);
    let opened = Zeroizing::new(
        cipher
            .decrypt(
                XNonce::from_slice(&envelope.nonce),
                Payload {
                    msg: &envelope.ciphertext,
                    aad: &aad,
                },
            )
            .map_err(|_| refuse(SensitiveStateFailureCode::Authentication))?,
    );
    if opened.len() != PLAINTEXT_DIGEST_LEN + plaintext_len {
        return Err(refuse(SensitiveStateFailureCode::Authentication));
    }
    let (digest, plaintext) = opened.split_at(PLAINTEXT_DIGEST_LEN);
    if blake3::hash(plaintext).as_bytes() != digest {
        return Err(refuse(SensitiveStateFailureCode::Authentication));
    }
    Ok(Zeroizing::new(plaintext.to_vec()))
}

/// Process-wide resolver holding zeroizing material keyed by selector.
pub struct NativeSensitiveStateKeyResolver {
    keys: BTreeMap<SensitiveStateKeyId, Zeroizing<[u8; 32]>>,
}

impl fmt::Debug for NativeSensitiveStateKeyResolver {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeSensitiveStateKeyResolver")
            .field("key_ids", &self.keys.keys().collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

impl NativeSensitiveStateKeyResolver {
    /// Build a resolver directly from decoded material.
    #[must_use]
    pub fn new(keys: BTreeMap<SensitiveStateKeyId, Zeroizing<[u8; 32]>>) -> Self {
        Self { keys }
    }

    /// Parse `selector <64-hex>` lines, wiping the parse buffer on the way out.
    pub fn parse(material: &mut Zeroizing<Vec<u8>>) -> Result<Self, SensitiveStateError> {
        let text = std::str::from_utf8(material)
            .map_err(|_| refuse(SensitiveStateFailureCode::KeyMalformed))?;
        let mut keys = BTreeMap::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let (selector, hex) = line
                .split_once(char::is_whitespace)
                .ok_or_else(|| refuse(SensitiveStateFailureCode::KeyMalformed))?;
            let hex = hex.trim();
            if selector.is_empty() || hex.len() != 64 {
                return Err(refuse(SensitiveStateFailureCode::KeyMalformed));
            }
            let mut key = [0_u8; 32];
            for (index, byte) in key.iter_mut().enumerate() {
                *byte = u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16)
                    .map_err(|_| refuse(SensitiveStateFailureCode::KeyMalformed))?;
            }
            if keys
                .insert(SensitiveStateKeyId::new(selector), Zeroizing::new(key))
                .is_some()
            {
                return Err(refuse(SensitiveStateFailureCode::KeyMalformed));
            }
        }
        if keys.is_empty() {
            return Err(refuse(SensitiveStateFailureCode::KeyMalformed));
        }
        material.zeroize();
        Ok(Self { keys })
    }
}

impl StreamingSensitiveStateKeyResolver for NativeSensitiveStateKeyResolver {
    fn resolve(
        &self,
        key_id: &SensitiveStateKeyId,
    ) -> Result<SensitiveStateKey, SensitiveStateError> {
        let key = self
            .keys
            .get(key_id)
            .ok_or_else(|| refuse(SensitiveStateFailureCode::KeyUnavailable))?;
        Ok(SensitiveStateKey {
            key_id: key_id.clone(),
            key: Zeroizing::new(**key),
        })
    }
}

/// Resolver installed when no sensitive-state source is configured.
///
/// Every `resolve` refuses with `KeyUnavailable`, so a `target_closed_loop`
/// policy fails closed on a process that was never given material rather than
/// silently running unprotected.
#[derive(Debug, Default)]
pub struct RefusingSensitiveStateKeyResolver;

impl StreamingSensitiveStateKeyResolver for RefusingSensitiveStateKeyResolver {
    fn resolve(
        &self,
        _key_id: &SensitiveStateKeyId,
    ) -> Result<SensitiveStateKey, SensitiveStateError> {
        Err(refuse(SensitiveStateFailureCode::KeyUnavailable))
    }
}

/// Where one process's sensitive-state material comes from.
#[derive(Debug)]
pub enum SensitiveMaterialSource {
    /// Read once from a launcher-owned inherited descriptor.
    InheritedFd(i32),
    /// Read one exact-`0600`, no-follow, regular file.
    DeploymentFile(PathBuf),
}

impl SensitiveMaterialSource {
    /// Select this process's source from the environment, if any is named.
    ///
    /// The descriptor variant wins: a launcher that passes a private pipe has
    /// already avoided putting material on the filesystem.
    #[must_use]
    pub fn from_environment() -> Option<Self> {
        if let Some(value) = std::env::var_os(SENSITIVE_STATE_FD_ENV) {
            let descriptor = value
                .to_str()
                .and_then(|text| text.trim().parse::<i32>().ok())
                .unwrap_or(SENSITIVE_STATE_FD);
            return Some(Self::InheritedFd(descriptor));
        }
        std::env::var_os(SENSITIVE_STATE_FILE_ENV)
            .filter(|value| !value.is_empty())
            .map(|value| Self::DeploymentFile(PathBuf::from(value)))
    }
}

/// Read one exact-`0600`, no-follow, regular key file.
///
/// The four checks run in order and none is optional: `O_NOFOLLOW` refuses a
/// symlink, `O_NONBLOCK` defeats a FIFO planted at the path, `is_file` refuses
/// a directory or device, the *exact* `0600` mask refuses any group or other
/// bit, and the size bound refuses an oversized source before it is read.
pub fn read_private_key_file(path: &Path) -> Result<Zeroizing<Vec<u8>>, SensitiveStateError> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options
        .open(path)
        .map_err(|_| refuse(SensitiveStateFailureCode::KeyNotPrivate))?;
    let metadata = file
        .metadata()
        .map_err(|_| refuse(SensitiveStateFailureCode::KeyNotPrivate))?;
    if !metadata.is_file() {
        return Err(refuse(SensitiveStateFailureCode::KeyNotPrivate));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o7777 != 0o600 {
            return Err(refuse(SensitiveStateFailureCode::KeyNotPrivate));
        }
    }
    if metadata.len() > MAX_SENSITIVE_MATERIAL_BYTES as u64 {
        return Err(refuse(SensitiveStateFailureCode::KeyMalformed));
    }
    read_bounded(file)
}

/// Read at most `MAX_SENSITIVE_MATERIAL_BYTES`, refusing anything longer.
///
/// Reads one byte past the bound so an oversized source is refused without the
/// full content ever entering the process.
pub fn read_bounded(reader: impl Read) -> Result<Zeroizing<Vec<u8>>, SensitiveStateError> {
    let mut bytes = Zeroizing::new(Vec::new());
    let mut limited = reader.take((MAX_SENSITIVE_MATERIAL_BYTES + 1) as u64);
    limited
        .read_to_end(&mut bytes)
        .map_err(|_| refuse(SensitiveStateFailureCode::KeyMalformed))?;
    if bytes.len() > MAX_SENSITIVE_MATERIAL_BYTES {
        return Err(refuse(SensitiveStateFailureCode::KeyMalformed));
    }
    Ok(bytes)
}

/// Read one source's material without consuming the process-wide claim.
pub fn read_material(
    source: &SensitiveMaterialSource,
) -> Result<Zeroizing<Vec<u8>>, SensitiveStateError> {
    match source {
        SensitiveMaterialSource::DeploymentFile(path) => read_private_key_file(path),
        #[cfg(unix)]
        SensitiveMaterialSource::InheritedFd(descriptor) => {
            use std::os::fd::FromRawFd;
            // The descriptor is a one-shot private pipe owned by the launcher;
            // taking ownership here closes it as soon as the read completes so
            // no later code can re-read or leak it.
            let file = unsafe { std::fs::File::from_raw_fd(*descriptor) };
            read_bounded(file)
        }
        #[cfg(not(unix))]
        SensitiveMaterialSource::InheritedFd(_) => {
            Err(refuse(SensitiveStateFailureCode::KeyNotPrivate))
        }
    }
}

/// One-shot claim on this process's sensitive-state material.
static MATERIAL_CLAIMED: AtomicBool = AtomicBool::new(false);

/// Acquire this process's material exactly once.
///
/// A second acquisition is a typed refusal rather than a second read of a
/// one-shot pipe, which would otherwise return nothing and look like missing
/// material. When no source is named, the refusing resolver is installed so
/// `target_closed_loop` fails closed.
pub fn acquire_process_sensitive_state()
-> Result<Arc<dyn StreamingSensitiveStateKeyResolver>, SensitiveStateError> {
    if MATERIAL_CLAIMED.swap(true, Ordering::SeqCst) {
        return Err(refuse(SensitiveStateFailureCode::KeyUnavailable));
    }
    let Some(source) = SensitiveMaterialSource::from_environment() else {
        return Ok(Arc::new(RefusingSensitiveStateKeyResolver));
    };
    let mut material = read_material(&source)?;
    let resolver = NativeSensitiveStateKeyResolver::parse(&mut material)?;
    Ok(Arc::new(resolver))
}
