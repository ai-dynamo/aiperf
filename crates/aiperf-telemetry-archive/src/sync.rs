// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Immutable object publication and versioned remote-head synchronization.
//!
//! Publication walks only the verified local object graph, uploads every
//! immutable dependency with create-if-absent plus exact-byte verification,
//! and changes discovery authority with one provider-version CAS. An uncertain
//! CAS is never guessed: the synchronizer rereads and accepts only the exact
//! desired remote head. Terminal success is reported only after its remote
//! publication receipt is durable in the local receipt journal.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Debug, Display, Formatter};
use std::path::Path;
use std::sync::Arc;

use aiperf_clock::Clock;

use crate::manifest::{generation_key, index_root_key};
use crate::{
    AppendReceipt, ArchiveObjectStore, ArchiveState, ArchiveStoreError, CanonicalJsonError,
    CanonicalJsonValue, Digest, HeadDescriptorV1, HeadUpdateError, IndexObjectKind,
    LocalArchiveRepository, ManifestError, NamedObjectVisibility, NoDurabilityFaults,
    ObservationKind, PartitionDescriptorV1, RawObjectDescriptorV1, RawRegistryError, ReceiptError,
    ReceiptEventV1, ReceiptJournal, ReceiptObserverEpochId, ReceiptTargetV1,
    RemotePublicationTargetV1, SessionId, StableObjectVersion, WriterClaimState,
    archive_object_digest, domain_digest,
};

const REMOTE_LATEST_KEY: &str = "LATEST";
const REMOTE_LATEST_VERSION: u64 = 1;

/// Exact pre-activation writer fence carried by every nonterminal remote head.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WriterClaimV1 {
    /// Local generation sequence at which this session became authoritative.
    pub claim_epoch: u64,
    /// Unique collection session owning the claim.
    pub writer_session_id: SessionId,
    /// Canonical qualified spool identity from immutable genesis.
    pub canonical_spool_id: Digest,
    /// Generation that first made this collection session authoritative.
    pub session_started_generation_hash: Digest,
}

impl WriterClaimV1 {
    fn value(self) -> CanonicalJsonValue {
        object(vec![
            (
                "canonical_spool_id",
                string(self.canonical_spool_id.to_hex()),
            ),
            (
                "claim_epoch",
                CanonicalJsonValue::Integer(i128::from(self.claim_epoch)),
            ),
            (
                "session_started_generation_hash",
                string(self.session_started_generation_hash.to_hex()),
            ),
            (
                "writer_session_id",
                string(uuid(self.writer_session_id.as_bytes())),
            ),
        ])
    }

    fn from_value(value: &CanonicalJsonValue) -> Result<Self, ArchiveSyncError> {
        let fields = value
            .as_object()
            .ok_or(ArchiveSyncError::InvalidRemoteHead("writer claim"))?;
        if fields.len() != 4 {
            return Err(ArchiveSyncError::InvalidRemoteHead("writer claim fields"));
        }
        Ok(Self {
            claim_epoch: parse_u64(fields, "claim_epoch")?,
            writer_session_id: SessionId::new(parse_uuid(parse_text(
                fields,
                "writer_session_id",
            )?)?)?,
            canonical_spool_id: parse_digest(parse_text(fields, "canonical_spool_id")?)?,
            session_started_generation_hash: parse_digest(parse_text(
                fields,
                "session_started_generation_hash",
            )?)?,
        })
    }
}

/// Authenticated remote discovery state and writer claim.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RemoteLatestV1 {
    /// Immutable local generation/root descriptor selected remotely.
    pub head: HeadDescriptorV1,
    /// Remote publication lifecycle, which may be ahead of local collection state.
    pub publication_state: ArchiveState,
    /// Active writer fence; absent only after terminal publication.
    pub writer_claim: Option<WriterClaimV1>,
}

impl RemoteLatestV1 {
    /// Validates claim/state shape and content-addressed head keys.
    pub fn validate(&self) -> Result<(), ArchiveSyncError> {
        HeadDescriptorV1::decode_canonical(&self.head.canonical_bytes())?;
        if (self.publication_state == ArchiveState::RemotelyFinalized)
            != self.writer_claim.is_none()
        {
            return Err(ArchiveSyncError::InvalidRemoteHead(
                "terminal state and writer claim disagree",
            ));
        }
        if let Some(claim) = self.writer_claim
            && (claim.claim_epoch > self.head.local_commit_seq
                || claim.canonical_spool_id == Digest::from_bytes([0; 32])
                || claim.session_started_generation_hash == Digest::from_bytes([0; 32])
                || (claim.claim_epoch == 0
                    && claim.session_started_generation_hash != self.head.genesis_hash))
        {
            return Err(ArchiveSyncError::InvalidRemoteHead("writer claim identity"));
        }
        Ok(())
    }

    /// Encodes exact canonical remote discovery bytes.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ArchiveSyncError> {
        self.validate()?;
        let head = CanonicalJsonValue::parse_canonical(&self.head.canonical_bytes())?;
        Ok(object(vec![
            ("head", head),
            (
                "publication_state",
                string(archive_state_name(self.publication_state)),
            ),
            (
                "version",
                CanonicalJsonValue::Integer(i128::from(REMOTE_LATEST_VERSION)),
            ),
            (
                "writer_claim",
                self.writer_claim
                    .map_or(CanonicalJsonValue::Null, WriterClaimV1::value),
            ),
        ])
        .to_bytes())
    }

    /// Decodes and validates exact canonical remote discovery bytes.
    pub fn decode(bytes: &[u8]) -> Result<Self, ArchiveSyncError> {
        let value = CanonicalJsonValue::parse_canonical(bytes)?;
        let fields = value
            .as_object()
            .ok_or(ArchiveSyncError::InvalidRemoteHead("object"))?;
        if fields.len() != 4
            || fields.get("version").and_then(CanonicalJsonValue::as_i128)
                != Some(i128::from(REMOTE_LATEST_VERSION))
        {
            return Err(ArchiveSyncError::InvalidRemoteHead("version/fields"));
        }
        let head = HeadDescriptorV1::decode_canonical(
            &fields
                .get("head")
                .ok_or(ArchiveSyncError::InvalidRemoteHead("head"))?
                .to_bytes(),
        )?;
        let publication_state = parse_archive_state(
            fields
                .get("publication_state")
                .and_then(CanonicalJsonValue::as_str)
                .ok_or(ArchiveSyncError::InvalidRemoteHead("publication_state"))?,
        )?;
        let writer_claim = match fields.get("writer_claim") {
            Some(CanonicalJsonValue::Null) => None,
            Some(value) => Some(WriterClaimV1::from_value(value)?),
            _ => return Err(ArchiveSyncError::InvalidRemoteHead("writer claim")),
        };
        let remote = Self {
            head,
            publication_state,
            writer_claim,
        };
        remote.validate()?;
        if remote.canonical_bytes()? != bytes {
            return Err(ArchiveSyncError::InvalidRemoteHead("canonical fields"));
        }
        Ok(remote)
    }

    /// Hashes exact installed `LATEST` bytes for publication receipts.
    pub fn hash(&self) -> Result<Digest, ArchiveSyncError> {
        Ok(domain_digest(
            "aiperf.archive.remote-latest.v1",
            &[&self.canonical_bytes()?],
        ))
    }
}

/// Durable observer epoch used to stamp a verified remote head update.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RemotePublicationObservationV1 {
    /// Previously durable receipt observer epoch.
    pub observer_epoch_id: ReceiptObserverEpochId,
}

/// Verified remote head result before or after terminal receipt persistence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RemotePublicationCompletionV1 {
    /// Installed remote discovery value.
    pub remote: RemoteLatestV1,
    /// Stable provider version returned or reread after CAS.
    pub object_version: StableObjectVersion,
    /// Whether this call reconciled an already/uncertainly installed value.
    pub observation_kind: ObservationKind,
    /// Durable local publication receipt for terminal updates.
    pub receipt: Option<AppendReceipt>,
}

/// Capability-gated immutable uploader and remote head CAS owner.
#[derive(Clone)]
pub struct ArchiveRemoteSynchronizer {
    store: Arc<dyn ArchiveObjectStore>,
}

impl Debug for ArchiveRemoteSynchronizer {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ArchiveRemoteSynchronizer")
            .field("capabilities", &self.store.capabilities())
            .finish_non_exhaustive()
    }
}

impl ArchiveRemoteSynchronizer {
    /// Requires authoritative store capabilities and immediate named reads.
    pub fn new(store: Arc<dyn ArchiveObjectStore>) -> Result<Self, ArchiveSyncError> {
        let capabilities = store.capabilities();
        capabilities.require_authoritative()?;
        if capabilities.named_object_visibility != NamedObjectVisibility::Immediate {
            return Err(ArchiveSyncError::VisibilityPolicyRequired);
        }
        Ok(Self { store })
    }

    /// Uploads the verified local graph and installs/advances an active claim.
    pub async fn publish_active(
        &self,
        repository: &LocalArchiveRepository,
        writer_session_id: SessionId,
    ) -> Result<RemotePublicationCompletionV1, ArchiveSyncError> {
        self.upload_immutable_graph(repository).await?;
        let writer_claim = writer_claim(repository, writer_session_id)?;
        let desired = RemoteLatestV1 {
            head: repository.head().clone(),
            publication_state: repository.head().archive_state,
            writer_claim: Some(writer_claim),
        };
        self.install_head(repository, desired, writer_claim, false)
            .await
    }

    /// Clears the active claim and journals the exact terminal remote version.
    pub async fn finalize_remote(
        &self,
        repository: &LocalArchiveRepository,
        writer_session_id: SessionId,
        observation: RemotePublicationObservationV1,
        observation_clock: &dyn Clock,
    ) -> Result<RemotePublicationCompletionV1, ArchiveSyncError> {
        if repository.head().archive_state != ArchiveState::LocallyFinalized {
            return Err(ArchiveSyncError::LocalNotFinalized);
        }
        self.upload_immutable_graph(repository).await?;
        let writer_claim = writer_claim(repository, writer_session_id)?;

        // A source-free terminal sync may be the first remote interaction.
        if self.store.read_head(REMOTE_LATEST_KEY).await?.is_none() {
            let bootstrap = RemoteLatestV1 {
                head: repository.head().clone(),
                publication_state: ArchiveState::LocallyFinalized,
                writer_claim: Some(writer_claim),
            };
            let bytes = bootstrap.canonical_bytes()?;
            match self
                .store
                .create_head_if_absent(
                    REMOTE_LATEST_KEY,
                    bytes.clone().into(),
                    archive_object_digest(&bytes),
                )
                .await
            {
                Ok(_) | Err(HeadUpdateError::Uncertain(_)) => {}
                Err(HeadUpdateError::Conflict { .. }) => {}
                Err(HeadUpdateError::Store(error)) => return Err(error.into()),
            }
        }

        let desired = RemoteLatestV1 {
            head: repository.head().clone(),
            publication_state: ArchiveState::RemotelyFinalized,
            writer_claim: None,
        };
        let mut completion = self
            .install_head(repository, desired.clone(), writer_claim, true)
            .await?;
        // The owning LocalSet stamps only after the response or exact recovery
        // reread is observed; a pre-CAS timestamp would fabricate causality.
        let observation_clock_ns = observation_clock.now_ns();
        let mut journal = ReceiptJournal::recover(
            repository.spool(),
            repository.head().archive_id,
            &NoDurabilityFaults,
        )?;
        let receipt_seq = journal.last_receipt_seq().map_or(Ok(0), |value| {
            value
                .checked_add(1)
                .ok_or(ArchiveSyncError::SequenceOverflow)
        })?;
        let target = ReceiptTargetV1::remote_publication(RemotePublicationTargetV1 {
            archive_id: repository.head().archive_id,
            generation_hash: repository.head().generation_hash,
            index_root_hash: repository.head().index_root_hash,
            installed_head_hash: desired.hash()?,
            object_version: completion.object_version.clone(),
            archive_state: ArchiveState::RemotelyFinalized,
            writer_claim_state: WriterClaimState::Absent,
        })?;
        let event = ReceiptEventV1::new(
            repository.head().archive_id,
            receipt_seq,
            target.receipt_target_id,
            observation.observer_epoch_id,
            completion.observation_kind,
            observation_clock_ns,
        );
        let receipt = AppendReceipt {
            receipt_target_id: target.receipt_target_id,
            event_id: event.event_id,
            receipt_seq,
        };
        journal.record_event(target, event, &NoDurabilityFaults)?;
        completion.receipt = Some(receipt);
        Ok(completion)
    }

    async fn upload_immutable_graph(
        &self,
        repository: &LocalArchiveRepository,
    ) -> Result<(), ArchiveSyncError> {
        let objects = publication_objects(repository)?;
        for (key, bytes) in objects {
            let digest = archive_object_digest(&bytes);
            self.store
                .put_if_absent(&key, bytes.clone().into(), digest)
                .await?;
            self.store.get_verified(&key, digest).await?;
        }
        Ok(())
    }

    async fn install_head(
        &self,
        repository: &LocalArchiveRepository,
        desired: RemoteLatestV1,
        writer_claim: WriterClaimV1,
        terminal: bool,
    ) -> Result<RemotePublicationCompletionV1, ArchiveSyncError> {
        let desired_bytes = desired.canonical_bytes()?;
        let desired_digest = archive_object_digest(&desired_bytes);
        let ancestry = generation_ancestry(repository)?;
        let current = self.store.read_head(REMOTE_LATEST_KEY).await?;
        let (version, observation_kind) = match current {
            None if !terminal => {
                match self
                    .store
                    .create_head_if_absent(
                        REMOTE_LATEST_KEY,
                        desired_bytes.clone().into(),
                        desired_digest,
                    )
                    .await
                {
                    Ok(receipt) => (receipt.version, ObservationKind::ResponseObserved),
                    Err(error) => {
                        return self
                            .reconcile_head_error(error, &desired, desired_digest)
                            .await;
                    }
                }
            }
            None => return Err(ArchiveSyncError::MissingRemoteBootstrap),
            Some(current) => {
                if current.digest == desired_digest && current.body.as_ref() == desired_bytes {
                    (current.version, ObservationKind::RecoveryVerified)
                } else {
                    let decoded = RemoteLatestV1::decode(&current.body)?;
                    if decoded.head.archive_id != repository.head().archive_id
                        || decoded.head.genesis_hash != repository.head().genesis_hash
                        || decoded.writer_claim != Some(writer_claim)
                        || !ancestry.contains(&decoded.head.generation_hash)
                    {
                        return Err(ArchiveSyncError::RemoteConflict(Box::new(decoded)));
                    }
                    match self
                        .store
                        .compare_and_swap_head(
                            REMOTE_LATEST_KEY,
                            &current.version,
                            desired_bytes.clone().into(),
                            desired_digest,
                        )
                        .await
                    {
                        Ok(version) => (version, ObservationKind::ResponseObserved),
                        Err(error) => {
                            return self
                                .reconcile_head_error(error, &desired, desired_digest)
                                .await;
                        }
                    }
                }
            }
        };
        Ok(RemotePublicationCompletionV1 {
            remote: desired,
            object_version: version,
            observation_kind,
            receipt: None,
        })
    }

    async fn reconcile_head_error(
        &self,
        error: HeadUpdateError,
        desired: &RemoteLatestV1,
        desired_digest: Digest,
    ) -> Result<RemotePublicationCompletionV1, ArchiveSyncError> {
        match error {
            HeadUpdateError::Store(error) => return Err(error.into()),
            HeadUpdateError::Conflict { .. } | HeadUpdateError::Uncertain(_) => {}
        }
        let current = self
            .store
            .read_head(REMOTE_LATEST_KEY)
            .await?
            .ok_or(ArchiveSyncError::UncertainHeadOutcome)?;
        if current.digest != desired_digest || current.body.as_ref() != desired.canonical_bytes()? {
            return Err(match RemoteLatestV1::decode(&current.body) {
                Ok(remote) => ArchiveSyncError::RemoteConflict(Box::new(remote)),
                Err(_) => ArchiveSyncError::UncertainHeadOutcome,
            });
        }
        Ok(RemotePublicationCompletionV1 {
            remote: desired.clone(),
            object_version: current.version,
            observation_kind: ObservationKind::RecoveryVerified,
            receipt: None,
        })
    }
}

fn publication_objects(
    repository: &LocalArchiveRepository,
) -> Result<BTreeMap<String, Vec<u8>>, ArchiveSyncError> {
    let mut objects = BTreeMap::new();
    for (hash, bytes) in repository.index().page_objects() {
        insert_object(&mut objects, index_root_key(hash), bytes.to_vec())?;
    }
    for entry in repository.index().entries() {
        match entry.key().as_bytes().first().copied() {
            Some(kind) if kind == IndexObjectKind::TablePartition as u8 => {
                let descriptor =
                    PartitionDescriptorV1::from_canonical_bytes(entry.descriptor_bytes())?;
                let bytes = repository
                    .spool()
                    .read_relative(Path::new(&descriptor.physical_object_key))?;
                insert_object(&mut objects, descriptor.physical_object_key, bytes)?;
            }
            Some(kind) if kind == IndexObjectKind::SharedRawObject as u8 => {
                let descriptor = RawObjectDescriptorV1::decode(entry.descriptor_bytes())?;
                let bytes = repository
                    .spool()
                    .read_relative(Path::new(&descriptor.object_key))?;
                insert_object(&mut objects, descriptor.object_key, bytes)?;
            }
            Some(kind)
                if kind == IndexObjectKind::ProjectionCoverage as u8
                    || kind == IndexObjectKind::RawNonceReservation as u8 => {}
            _ => return Err(ArchiveSyncError::UnknownIndexObject),
        }
    }
    for (key, bytes, _) in generation_chain(repository)? {
        insert_object(&mut objects, key, bytes)?;
    }
    Ok(objects)
}

fn generation_chain(
    repository: &LocalArchiveRepository,
) -> Result<Vec<(String, Vec<u8>, Digest)>, ArchiveSyncError> {
    let mut sequence = repository.head().local_commit_seq;
    let mut hash = repository.head().generation_hash;
    let mut reverse = Vec::new();
    loop {
        let key = generation_key(sequence, hash);
        let bytes = repository.spool().read_relative(Path::new(&key))?;
        let generation = crate::GenerationObjectV1::decode(&bytes)?;
        if generation.hash != hash || generation.key != key {
            return Err(ArchiveSyncError::GenerationChain);
        }
        let parent = generation.generation.parent_generation_hash;
        reverse.push((key, bytes, hash));
        if sequence == 0 {
            if parent.is_some() {
                return Err(ArchiveSyncError::GenerationChain);
            }
            break;
        }
        sequence -= 1;
        hash = parent.ok_or(ArchiveSyncError::GenerationChain)?;
    }
    reverse.reverse();
    Ok(reverse)
}

fn generation_ancestry(
    repository: &LocalArchiveRepository,
) -> Result<BTreeSet<Digest>, ArchiveSyncError> {
    Ok(generation_chain(repository)?
        .into_iter()
        .map(|(_, _, hash)| hash)
        .collect())
}

fn writer_claim(
    repository: &LocalArchiveRepository,
    writer_session_id: SessionId,
) -> Result<WriterClaimV1, ArchiveSyncError> {
    for (_, bytes, generation_hash) in generation_chain(repository)? {
        let generation = crate::GenerationObjectV1::decode(&bytes)?;
        if generation.generation.session_id == Some(writer_session_id) {
            return Ok(WriterClaimV1 {
                claim_epoch: generation.generation.local_commit_seq,
                writer_session_id,
                canonical_spool_id: repository.genesis().canonical_spool_id,
                session_started_generation_hash: generation_hash,
            });
        }
    }
    Err(ArchiveSyncError::SessionGenerationMissing)
}

fn insert_object(
    objects: &mut BTreeMap<String, Vec<u8>>,
    key: String,
    bytes: Vec<u8>,
) -> Result<(), ArchiveSyncError> {
    match objects.insert(key.clone(), bytes.clone()) {
        Some(existing) if existing != bytes => Err(ArchiveSyncError::ObjectCollision(key)),
        _ => Ok(()),
    }
}

fn object(entries: Vec<(&str, CanonicalJsonValue)>) -> CanonicalJsonValue {
    CanonicalJsonValue::object(
        entries
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value)),
    )
    .expect("static remote-head keys are unique")
}

fn string(value: impl Into<String>) -> CanonicalJsonValue {
    CanonicalJsonValue::String(value.into())
}

const fn archive_state_name(state: ArchiveState) -> &'static str {
    match state {
        ArchiveState::Open => "open",
        ArchiveState::StopRequested => "stop_requested",
        ArchiveState::LocallyFinalized => "locally_finalized",
        ArchiveState::RemotelyFinalized => "remotely_finalized",
        ArchiveState::Failed => "failed",
    }
}

fn parse_archive_state(value: &str) -> Result<ArchiveState, ArchiveSyncError> {
    match value {
        "open" => Ok(ArchiveState::Open),
        "stop_requested" => Ok(ArchiveState::StopRequested),
        "locally_finalized" => Ok(ArchiveState::LocallyFinalized),
        "remotely_finalized" => Ok(ArchiveState::RemotelyFinalized),
        "failed" => Ok(ArchiveState::Failed),
        _ => Err(ArchiveSyncError::InvalidRemoteHead("archive state")),
    }
}

fn parse_text<'a>(
    fields: &'a BTreeMap<String, CanonicalJsonValue>,
    field: &'static str,
) -> Result<&'a str, ArchiveSyncError> {
    fields
        .get(field)
        .and_then(CanonicalJsonValue::as_str)
        .ok_or(ArchiveSyncError::InvalidRemoteHead(field))
}

fn parse_u64(
    fields: &BTreeMap<String, CanonicalJsonValue>,
    field: &'static str,
) -> Result<u64, ArchiveSyncError> {
    let value = fields
        .get(field)
        .and_then(CanonicalJsonValue::as_i128)
        .ok_or(ArchiveSyncError::InvalidRemoteHead(field))?;
    u64::try_from(value).map_err(|_| ArchiveSyncError::InvalidRemoteHead(field))
}

fn parse_digest(value: &str) -> Result<Digest, ArchiveSyncError> {
    Digest::parse(value).map_err(|_| ArchiveSyncError::InvalidRemoteHead("digest"))
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(char::from(DIGITS[usize::from(byte >> 4)]));
        output.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    output
}

fn uuid(bytes: &[u8; 16]) -> String {
    let hex = hex(bytes);
    format!(
        "{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..]
    )
}

fn parse_uuid(value: &str) -> Result<[u8; 16], ArchiveSyncError> {
    if value.len() != 36
        || !value.is_ascii()
        || value.bytes().enumerate().any(|(index, byte)| match index {
            8 | 13 | 18 | 23 => byte != b'-',
            _ => !byte.is_ascii_digit() && !(b'a'..=b'f').contains(&byte),
        })
    {
        return Err(ArchiveSyncError::InvalidRemoteHead("uuid"));
    }
    let compact = value
        .bytes()
        .filter(|byte| *byte != b'-')
        .collect::<Vec<_>>();
    let mut output = [0_u8; 16];
    for (index, byte) in output.iter_mut().enumerate() {
        let offset = index * 2;
        let pair = std::str::from_utf8(&compact[offset..offset + 2])
            .map_err(|_| ArchiveSyncError::InvalidRemoteHead("uuid"))?;
        *byte = u8::from_str_radix(pair, 16)
            .map_err(|_| ArchiveSyncError::InvalidRemoteHead("uuid"))?;
    }
    Ok(output)
}

/// Immutable upload, remote discovery, CAS, or receipt failure.
#[derive(Debug)]
pub enum ArchiveSyncError {
    /// Store capability or operation failed.
    Store(ArchiveStoreError),
    /// Local spool read/verification failed.
    Spool(crate::SpoolError),
    /// Manifest bytes or links failed.
    Manifest(ManifestError),
    /// Partition descriptor failed.
    Partition(crate::ParquetProjectionError),
    /// Raw descriptor failed.
    Raw(RawRegistryError),
    /// Receipt target/event transaction failed.
    Receipt(ReceiptError),
    /// Canonical remote discovery JSON failed.
    Canonical(CanonicalJsonError),
    /// Typed ID construction failed.
    Identity(crate::FrameIdentityError),
    /// Bounded-lag visibility requires an injected Clock retry policy.
    VisibilityPolicyRequired,
    /// Remote discovery bytes violate the closed v1 shape.
    InvalidRemoteHead(&'static str),
    /// Local archive is not yet sealed.
    LocalNotFinalized,
    /// Terminal CAS had no active bootstrap to clear.
    MissingRemoteBootstrap,
    /// Current remote head/claim is a verified competing value.
    RemoteConflict(Box<RemoteLatestV1>),
    /// CAS failed uncertain and verified reread did not resolve it.
    UncertainHeadOutcome,
    /// Local generation ancestry is incomplete or inconsistent.
    GenerationChain,
    /// Requested writer session has no authoritative generation in local history.
    SessionGenerationMissing,
    /// Two graph objects selected the same key with unequal bytes.
    ObjectCollision(String),
    /// Primary index contains an unsupported object-kind key.
    UnknownIndexObject,
    /// Receipt sequence overflowed.
    SequenceOverflow,
}

macro_rules! from_error {
    ($source:ty, $variant:ident) => {
        impl From<$source> for ArchiveSyncError {
            fn from(value: $source) -> Self {
                Self::$variant(value)
            }
        }
    };
}

from_error!(ArchiveStoreError, Store);
from_error!(crate::SpoolError, Spool);
from_error!(ManifestError, Manifest);
from_error!(crate::ParquetProjectionError, Partition);
from_error!(RawRegistryError, Raw);
from_error!(ReceiptError, Receipt);
from_error!(CanonicalJsonError, Canonical);
from_error!(crate::FrameIdentityError, Identity);

impl Display for ArchiveSyncError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store(error) => write!(formatter, "archive remote store failed: {error}"),
            Self::Spool(error) => write!(formatter, "archive local spool failed: {error}"),
            Self::Manifest(error) => write!(formatter, "archive manifest failed: {error}"),
            Self::Partition(error) => write!(formatter, "archive partition failed: {error}"),
            Self::Raw(error) => write!(formatter, "archive raw descriptor failed: {error}"),
            Self::Receipt(error) => write!(formatter, "archive receipt failed: {error}"),
            Self::Canonical(error) => write!(formatter, "remote head JSON failed: {error}"),
            Self::Identity(error) => write!(formatter, "remote head identity failed: {error}"),
            Self::VisibilityPolicyRequired => {
                formatter.write_str("bounded-lag store requires an injected visibility policy")
            }
            Self::InvalidRemoteHead(field) => write!(formatter, "invalid remote head {field}"),
            Self::LocalNotFinalized => formatter.write_str("local archive is not finalized"),
            Self::MissingRemoteBootstrap => formatter.write_str("remote bootstrap is missing"),
            Self::RemoteConflict(_) => formatter.write_str("remote archive head/claim conflict"),
            Self::UncertainHeadOutcome => formatter.write_str("remote head outcome is uncertain"),
            Self::GenerationChain => formatter.write_str("local generation chain is invalid"),
            Self::SessionGenerationMissing => {
                formatter.write_str("writer session has no local authoritative generation")
            }
            Self::ObjectCollision(key) => {
                write!(formatter, "publication object collision at {key}")
            }
            Self::UnknownIndexObject => formatter.write_str("unknown publication index object"),
            Self::SequenceOverflow => formatter.write_str("remote receipt sequence overflow"),
        }
    }
}

impl std::error::Error for ArchiveSyncError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Store(error) => Some(error),
            Self::Spool(error) => Some(error),
            Self::Manifest(error) => Some(error),
            Self::Partition(error) => Some(error),
            Self::Raw(error) => Some(error),
            Self::Receipt(error) => Some(error),
            Self::Canonical(error) => Some(error),
            Self::Identity(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tempfile::TempDir;

    use super::*;
    use crate::{
        ArchiveId, CanonicalJsonValue, EpochAnchor, ExecutionId, GenerationTransactionKind,
        GenesisV1, IndexMutationSetV1, MemoryArchiveObjectStore, MemoryStoreFault, QualifiedSpool,
        ReceiptObserverEpochV1, TimeDomain,
    };
    use aiperf_clock::SimClock;

    fn archive_id() -> ArchiveId {
        ArchiveId::new([0x11; 16]).unwrap()
    }

    fn session_id(byte: u8) -> SessionId {
        SessionId::new([byte; 16]).unwrap()
    }

    fn genesis(session_id: SessionId) -> GenesisV1 {
        GenesisV1 {
            archive_id: archive_id(),
            canonical_spool_id: Digest::from_bytes([1; 32]),
            archive_identity_digest: Digest::from_bytes([2; 32]),
            archive_key_digest: Digest::from_bytes([3; 32]),
            writer_compatibility_id: Digest::from_bytes([4; 32]),
            runner_distribution_id: Digest::from_bytes([5; 32]),
            source_descriptors: CanonicalJsonValue::Array(vec![]),
            persistent_writer_identity: CanonicalJsonValue::object([(
                "writer".to_owned(),
                CanonicalJsonValue::String("parquet-v1".to_owned()),
            )])
            .unwrap(),
            initial_session_id: Some(session_id),
            time_domain: TimeDomain::Real,
            epoch_anchor: Some(EpochAnchor {
                clock_ns: 0,
                unix_epoch_ns: 1_700_000_000_000_000_000,
                capture_uncertainty_ns: 1,
            }),
        }
    }

    fn finalized_repository() -> (
        TempDir,
        LocalArchiveRepository,
        SessionId,
        ReceiptObserverEpochId,
    ) {
        let directory = tempfile::tempdir().unwrap();
        let spool = QualifiedSpool::open(directory.path().join("archive")).unwrap();
        let session_id = session_id(0x22);
        let mut repository =
            LocalArchiveRepository::create_new(spool, genesis(session_id), &NoDurabilityFaults)
                .unwrap();
        let epoch = ReceiptObserverEpochV1::new(
            ExecutionId::new([0x33; 16]).unwrap(),
            None,
            TimeDomain::Virtual,
            100,
            None,
            0,
            Digest::from_bytes([5; 32]),
        )
        .unwrap();
        let epoch_id = epoch.observer_epoch_id;
        ReceiptJournal::bootstrap(
            repository.spool(),
            repository.head().archive_id,
            epoch,
            &NoDurabilityFaults,
        )
        .unwrap();
        repository
            .commit(
                &IndexMutationSetV1::new(vec![], vec![]).unwrap(),
                GenerationTransactionKind::LocalFinalization,
                ArchiveState::LocallyFinalized,
                Some(session_id),
                Some("requested".to_owned()),
                &NoDurabilityFaults,
            )
            .unwrap();
        (directory, repository, session_id, epoch_id)
    }

    #[test]
    fn remote_latest_round_trips_claim_and_terminal_shapes() {
        let (_directory, repository, session_id, _) = finalized_repository();
        let active = RemoteLatestV1 {
            head: repository.head().clone(),
            publication_state: ArchiveState::LocallyFinalized,
            writer_claim: Some(writer_claim(&repository, session_id).unwrap()),
        };
        assert_eq!(
            RemoteLatestV1::decode(&active.canonical_bytes().unwrap()).unwrap(),
            active
        );
        let terminal = RemoteLatestV1 {
            head: repository.head().clone(),
            publication_state: ArchiveState::RemotelyFinalized,
            writer_claim: None,
        };
        assert_eq!(
            RemoteLatestV1::decode(&terminal.canonical_bytes().unwrap()).unwrap(),
            terminal
        );
    }

    #[tokio::test]
    async fn terminal_cas_clears_claim_and_persists_local_receipt() {
        let (_directory, repository, owner_session_id, epoch_id) = finalized_repository();
        let store = Arc::new(MemoryArchiveObjectStore::default());
        let synchronizer = ArchiveRemoteSynchronizer::new(store.clone()).unwrap();
        let clock = SimClock::new();
        let active = synchronizer
            .publish_active(&repository, owner_session_id)
            .await
            .unwrap();
        assert_eq!(
            active.remote.writer_claim.unwrap().writer_session_id,
            owner_session_id
        );

        let finalized = synchronizer
            .finalize_remote(
                &repository,
                owner_session_id,
                RemotePublicationObservationV1 {
                    observer_epoch_id: epoch_id,
                },
                &clock,
            )
            .await
            .unwrap();
        assert_eq!(
            finalized.remote.publication_state,
            ArchiveState::RemotelyFinalized
        );
        assert_eq!(finalized.remote.writer_claim, None);
        assert_eq!(
            finalized.observation_kind,
            ObservationKind::ResponseObserved
        );
        assert!(finalized.receipt.is_some());

        let installed = store.read_head(REMOTE_LATEST_KEY).await.unwrap().unwrap();
        assert_eq!(
            RemoteLatestV1::decode(&installed.body).unwrap(),
            finalized.remote
        );
        let journal = ReceiptJournal::recover(
            repository.spool(),
            repository.head().archive_id,
            &NoDurabilityFaults,
        )
        .unwrap();
        assert_eq!(journal.target_count(), 1);
        assert_eq!(journal.event_count(), 1);
    }

    #[tokio::test]
    async fn uncertain_terminal_cas_is_accepted_only_after_exact_reread() {
        let (_directory, repository, session_id, epoch_id) = finalized_repository();
        let store = Arc::new(MemoryArchiveObjectStore::default());
        let synchronizer = ArchiveRemoteSynchronizer::new(store.clone()).unwrap();
        let clock = SimClock::new();
        synchronizer
            .publish_active(&repository, session_id)
            .await
            .unwrap();
        store
            .set_fault(MemoryStoreFault::CasUncertainAfterApply)
            .unwrap();

        let finalized = synchronizer
            .finalize_remote(
                &repository,
                session_id,
                RemotePublicationObservationV1 {
                    observer_epoch_id: epoch_id,
                },
                &clock,
            )
            .await
            .unwrap();
        assert_eq!(
            finalized.observation_kind,
            ObservationKind::RecoveryVerified
        );
        assert!(finalized.receipt.is_some());
    }

    #[tokio::test]
    async fn receipt_failure_never_reports_remote_finalization_and_retry_recovers() {
        let (_directory, repository, session_id, _) = finalized_repository();
        let store = Arc::new(MemoryArchiveObjectStore::default());
        let synchronizer = ArchiveRemoteSynchronizer::new(store.clone()).unwrap();
        let clock = SimClock::new();
        synchronizer
            .publish_active(&repository, session_id)
            .await
            .unwrap();
        let retry_epoch = ReceiptObserverEpochV1::new(
            ExecutionId::new([0x55; 16]).unwrap(),
            None,
            TimeDomain::Virtual,
            500,
            None,
            0,
            Digest::from_bytes([5; 32]),
        )
        .unwrap();

        let error = synchronizer
            .finalize_remote(
                &repository,
                session_id,
                RemotePublicationObservationV1 {
                    observer_epoch_id: retry_epoch.observer_epoch_id,
                },
                &clock,
            )
            .await
            .unwrap_err();
        assert!(matches!(error, ArchiveSyncError::Receipt(_)));
        let installed = store.read_head(REMOTE_LATEST_KEY).await.unwrap().unwrap();
        assert_eq!(
            RemoteLatestV1::decode(&installed.body)
                .unwrap()
                .publication_state,
            ArchiveState::RemotelyFinalized
        );

        let mut journal = ReceiptJournal::recover(
            repository.spool(),
            repository.head().archive_id,
            &NoDurabilityFaults,
        )
        .unwrap();
        journal
            .append_observer_epoch(retry_epoch.clone(), &NoDurabilityFaults)
            .unwrap();
        drop(journal);
        let recovered = synchronizer
            .finalize_remote(
                &repository,
                session_id,
                RemotePublicationObservationV1 {
                    observer_epoch_id: retry_epoch.observer_epoch_id,
                },
                &clock,
            )
            .await
            .unwrap();
        assert_eq!(
            recovered.observation_kind,
            ObservationKind::RecoveryVerified
        );
        assert!(recovered.receipt.is_some());
    }

    #[tokio::test]
    async fn competing_writer_claim_fails_without_changing_remote_head() {
        let (_directory, repository, owner_session_id, epoch_id) = finalized_repository();
        let store = Arc::new(MemoryArchiveObjectStore::default());
        let synchronizer = ArchiveRemoteSynchronizer::new(store.clone()).unwrap();
        let clock = SimClock::new();
        let active = synchronizer
            .publish_active(&repository, owner_session_id)
            .await
            .unwrap();
        let current = store.read_head(REMOTE_LATEST_KEY).await.unwrap().unwrap();
        let mut competing = active.remote.clone();
        competing.writer_claim.as_mut().unwrap().canonical_spool_id = Digest::from_bytes([9; 32]);
        let competing_bytes = competing.canonical_bytes().unwrap();
        store
            .compare_and_swap_head(
                REMOTE_LATEST_KEY,
                &current.version,
                competing_bytes.clone().into(),
                archive_object_digest(&competing_bytes),
            )
            .await
            .unwrap();

        let error = synchronizer
            .finalize_remote(
                &repository,
                owner_session_id,
                RemotePublicationObservationV1 {
                    observer_epoch_id: epoch_id,
                },
                &clock,
            )
            .await
            .unwrap_err();
        assert!(matches!(error, ArchiveSyncError::RemoteConflict(_)));
        let installed = store.read_head(REMOTE_LATEST_KEY).await.unwrap().unwrap();
        assert_eq!(RemoteLatestV1::decode(&installed.body).unwrap(), competing);
    }
}
