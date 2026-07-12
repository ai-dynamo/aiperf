// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical immutable generation objects and local discovery heads.

use std::collections::BTreeMap;
use std::fmt::{self, Display, Formatter};

use crate::{
    ArchiveId, CanonicalJsonError, CanonicalJsonValue, Digest, EpochAnchor, IndexKey,
    IndexMutationSetV1, IndexRootV1, SessionId, domain_digest,
};

const ENVELOPE_MAGIC: &str = "aiperf.archive.envelope.v1";
const GENERATION_TYPE: &str = "manifest-generation";
const LOCAL_LATEST_TYPE: &str = "local-latest";

/// Real or virtual Clock domain recorded by one collection session.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TimeDomain {
    /// A real monotonic Clock with one Unix epoch anchor.
    Real,
    /// A virtual Clock with no Unix epoch interpretation.
    Virtual,
}

impl TimeDomain {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Real => "real",
            Self::Virtual => "virtual",
        }
    }

    fn parse(value: &str) -> Result<Self, ManifestError> {
        match value {
            "real" => Ok(Self::Real),
            "virtual" => Ok(Self::Virtual),
            _ => Err(ManifestError::InvalidField("time_domain")),
        }
    }
}

/// Immutable archive lifecycle state stored in generations and heads.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArchiveState {
    /// Frame admission remains open.
    Open,
    /// Stop has been requested and admission is closing.
    StopRequested,
    /// The local generation is sealed and cannot admit frames.
    LocallyFinalized,
    /// A verified remote head references the sealed generation.
    RemotelyFinalized,
    /// The archive failed before authoritative finalization.
    Failed,
}

impl ArchiveState {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::StopRequested => "stop_requested",
            Self::LocallyFinalized => "locally_finalized",
            Self::RemotelyFinalized => "remotely_finalized",
            Self::Failed => "failed",
        }
    }

    fn parse(value: &str) -> Result<Self, ManifestError> {
        match value {
            "open" => Ok(Self::Open),
            "stop_requested" => Ok(Self::StopRequested),
            "locally_finalized" => Ok(Self::LocallyFinalized),
            "remotely_finalized" => Ok(Self::RemotelyFinalized),
            "failed" => Ok(Self::Failed),
            _ => Err(ManifestError::InvalidField("archive_state")),
        }
    }
}

/// The reason one immutable generation transaction exists.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GenerationTransactionKind {
    /// Full create-only generation zero.
    Genesis,
    /// A new collection session became authoritative.
    SessionStarted,
    /// Index/object coverage advanced.
    Checkpoint,
    /// Local admission closed and all covered frames were sealed.
    LocalFinalization,
    /// A retention-only generation protects preceding-head rollback.
    RetentionCheckpoint,
    /// Remote publication or compaction recorded a state transition.
    StateTransition,
}

impl GenerationTransactionKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Genesis => "genesis",
            Self::SessionStarted => "session_started",
            Self::Checkpoint => "checkpoint",
            Self::LocalFinalization => "local_finalization",
            Self::RetentionCheckpoint => "retention_checkpoint",
            Self::StateTransition => "state_transition",
        }
    }

    fn parse(value: &str) -> Result<Self, ManifestError> {
        match value {
            "genesis" => Ok(Self::Genesis),
            "session_started" => Ok(Self::SessionStarted),
            "checkpoint" => Ok(Self::Checkpoint),
            "local_finalization" => Ok(Self::LocalFinalization),
            "retention_checkpoint" => Ok(Self::RetentionCheckpoint),
            "state_transition" => Ok(Self::StateTransition),
            _ => Err(ManifestError::InvalidField("transaction_kind")),
        }
    }
}

/// Full secret-free persistent archive identity recorded only in generation zero.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenesisV1 {
    /// Non-zero archive identity.
    pub archive_id: ArchiveId,
    /// Random identity of the one canonical qualified spool.
    pub canonical_spool_id: Digest,
    /// Digest of every persistent collection/writer policy input.
    pub archive_identity_digest: Digest,
    /// Digest identifying the archive-key provider/key derivation authority.
    pub archive_key_digest: Digest,
    /// Frozen writer compatibility ID.
    pub writer_compatibility_id: Digest,
    /// Exact runner distribution provenance.
    pub runner_distribution_id: Digest,
    /// Canonical secret-free source/policy descriptors.
    pub source_descriptors: CanonicalJsonValue,
    /// Factory-produced canonical persistent writer identity.
    pub persistent_writer_identity: CanonicalJsonValue,
    /// Initial collection session, absent only for a source-free archive bootstrap.
    pub initial_session_id: Option<SessionId>,
    /// Clock domain for the initial session.
    pub time_domain: TimeDomain,
    /// Bracketed epoch anchor for real time, absent for virtual time.
    pub epoch_anchor: Option<EpochAnchor>,
}

impl GenesisV1 {
    /// Validates cross-field time/session invariants.
    pub fn validate(&self) -> Result<(), ManifestError> {
        match (self.time_domain, self.epoch_anchor) {
            (TimeDomain::Real, Some(_)) | (TimeDomain::Virtual, None) => {}
            _ => return Err(ManifestError::InvalidField("epoch_anchor")),
        }
        if !matches!(self.source_descriptors, CanonicalJsonValue::Array(_)) {
            return Err(ManifestError::InvalidField("source_descriptors"));
        }
        if !matches!(
            self.persistent_writer_identity,
            CanonicalJsonValue::Object(_)
        ) {
            return Err(ManifestError::InvalidField("persistent_writer_identity"));
        }
        Ok(())
    }

    fn to_value(&self) -> Result<CanonicalJsonValue, ManifestError> {
        self.validate()?;
        Ok(object(vec![
            ("archive_id", string(uuid(self.archive_id.as_bytes()))),
            (
                "archive_identity_digest",
                string(self.archive_identity_digest.to_hex()),
            ),
            (
                "archive_key_digest",
                string(self.archive_key_digest.to_hex()),
            ),
            (
                "canonical_spool_id",
                string(self.canonical_spool_id.to_hex()),
            ),
            ("epoch_anchor", epoch_anchor_value(self.epoch_anchor)),
            (
                "initial_session_id",
                optional_session(self.initial_session_id),
            ),
            (
                "persistent_writer_identity",
                self.persistent_writer_identity.clone(),
            ),
            (
                "runner_distribution_id",
                string(self.runner_distribution_id.to_hex()),
            ),
            ("source_descriptors", self.source_descriptors.clone()),
            ("time_domain", string(self.time_domain.as_str())),
            (
                "writer_compatibility_id",
                string(self.writer_compatibility_id.to_hex()),
            ),
        ]))
    }

    fn from_value(value: &CanonicalJsonValue) -> Result<Self, ManifestError> {
        let object = as_object(value, "genesis")?;
        let archive_id = parse_archive_id(text(object, "archive_id")?)?;
        let initial_session_id = parse_optional_session(object.get("initial_session_id"))?;
        let time_domain = TimeDomain::parse(text(object, "time_domain")?)?;
        let genesis = Self {
            archive_id,
            canonical_spool_id: digest(object, "canonical_spool_id")?,
            archive_identity_digest: digest(object, "archive_identity_digest")?,
            archive_key_digest: digest(object, "archive_key_digest")?,
            writer_compatibility_id: digest(object, "writer_compatibility_id")?,
            runner_distribution_id: digest(object, "runner_distribution_id")?,
            source_descriptors: object
                .get("source_descriptors")
                .cloned()
                .ok_or(ManifestError::InvalidField("source_descriptors"))?,
            persistent_writer_identity: object
                .get("persistent_writer_identity")
                .cloned()
                .ok_or(ManifestError::InvalidField("persistent_writer_identity"))?,
            initial_session_id,
            time_domain,
            epoch_anchor: parse_epoch_anchor(object.get("epoch_anchor"))?,
        };
        genesis.validate()?;
        Ok(genesis)
    }
}

/// One canonical removal/addition record embedded in generation history.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GenerationMutationV1 {
    /// Exact parent descriptor removal.
    Remove {
        /// Exact composite/tagged key.
        key: IndexKey,
        /// Exact parent descriptor hash.
        descriptor_hash: Digest,
    },
    /// Exact child descriptor addition.
    Add {
        /// Exact composite/tagged key.
        key: IndexKey,
        /// Exact child descriptor hash.
        descriptor_hash: Digest,
    },
}

impl GenerationMutationV1 {
    /// Freezes removals first, then additions, from the validated index transaction.
    #[must_use]
    pub fn from_set(set: &IndexMutationSetV1) -> Vec<Self> {
        set.removals()
            .iter()
            .map(|removal| Self::Remove {
                key: removal.key.clone(),
                descriptor_hash: removal.expected_descriptor_hash,
            })
            .chain(set.additions().iter().map(|addition| Self::Add {
                key: addition.key().clone(),
                descriptor_hash: addition.descriptor_hash(),
            }))
            .collect()
    }

    fn to_value(&self) -> CanonicalJsonValue {
        let (operation, key, descriptor_hash) = match self {
            Self::Remove {
                key,
                descriptor_hash,
            } => ("remove", key, descriptor_hash),
            Self::Add {
                key,
                descriptor_hash,
            } => ("add", key, descriptor_hash),
        };
        object(vec![
            ("descriptor_hash", string(descriptor_hash.to_hex())),
            ("key", string(hex(key.as_bytes()))),
            ("operation", string(operation)),
        ])
    }

    fn from_value(value: &CanonicalJsonValue) -> Result<Self, ManifestError> {
        let object = as_object(value, "mutation")?;
        let key = IndexKey::new(decode_hex(text(object, "key")?)?)
            .map_err(|_| ManifestError::InvalidField("mutation key"))?;
        let descriptor_hash = digest(object, "descriptor_hash")?;
        match text(object, "operation")? {
            "remove" => Ok(Self::Remove {
                key,
                descriptor_hash,
            }),
            "add" => Ok(Self::Add {
                key,
                descriptor_hash,
            }),
            _ => Err(ManifestError::InvalidField("mutation operation")),
        }
    }
}

/// One immutable hash-linked generation payload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenerationV1 {
    /// Archive identity.
    pub archive_id: ArchiveId,
    /// Monotonic local commit sequence.
    pub local_commit_seq: u64,
    /// Parent generation hash, absent only for generation zero.
    pub parent_generation_hash: Option<Digest>,
    /// Generation-zero hash, absent within generation zero to avoid self-reference.
    pub genesis_hash: Option<Digest>,
    /// Complete resulting index root.
    pub index_root: IndexRootV1,
    /// Resulting lifecycle state.
    pub archive_state: ArchiveState,
    /// Transaction kind.
    pub transaction_kind: GenerationTransactionKind,
    /// Session made current by this transaction when applicable.
    pub session_id: Option<SessionId>,
    /// Canonical index mutations in removals-then-additions order.
    pub mutations: Vec<GenerationMutationV1>,
    /// Full generation-zero identity, present only for genesis.
    pub genesis: Option<GenesisV1>,
    /// Frozen termination reason, when this generation closes/fails state.
    pub termination_reason: Option<String>,
}

impl GenerationV1 {
    /// Validates generation-zero and descendant shape invariants.
    pub fn validate(&self) -> Result<(), ManifestError> {
        if self.local_commit_seq == 0 {
            if self.transaction_kind != GenerationTransactionKind::Genesis
                || self.parent_generation_hash.is_some()
                || self.genesis_hash.is_some()
                || self.genesis.as_ref().map(|value| value.archive_id) != Some(self.archive_id)
                || !self.mutations.is_empty()
            {
                return Err(ManifestError::InvalidGenesisShape);
            }
            self.genesis
                .as_ref()
                .ok_or(ManifestError::InvalidGenesisShape)?
                .validate()?;
        } else if self.transaction_kind == GenerationTransactionKind::Genesis
            || self.parent_generation_hash.is_none()
            || self.genesis_hash.is_none()
            || self.genesis.is_some()
        {
            return Err(ManifestError::InvalidDescendantShape);
        }
        let mut additions_started = false;
        let mut preceding: Option<&IndexKey> = None;
        for mutation in &self.mutations {
            let (is_addition, key) = match mutation {
                GenerationMutationV1::Remove { key, .. } => (false, key),
                GenerationMutationV1::Add { key, .. } => (true, key),
            };
            if !is_addition && additions_started {
                return Err(ManifestError::NonCanonicalMutationOrder);
            }
            if is_addition && !additions_started {
                additions_started = true;
                preceding = None;
            }
            if preceding.is_some_and(|previous| previous >= key) {
                return Err(ManifestError::NonCanonicalMutationOrder);
            }
            preceding = Some(key);
        }
        Ok(())
    }

    fn to_value(&self) -> Result<CanonicalJsonValue, ManifestError> {
        self.validate()?;
        Ok(object(vec![
            ("archive_id", string(uuid(self.archive_id.as_bytes()))),
            ("archive_state", string(self.archive_state.as_str())),
            (
                "genesis",
                self.genesis
                    .as_ref()
                    .map(GenesisV1::to_value)
                    .transpose()?
                    .unwrap_or(CanonicalJsonValue::Null),
            ),
            ("genesis_hash", optional_digest(self.genesis_hash)),
            ("index_root", index_root_value(&self.index_root)),
            (
                "local_commit_seq",
                integer(i128::from(self.local_commit_seq)),
            ),
            (
                "mutations",
                CanonicalJsonValue::Array(
                    self.mutations
                        .iter()
                        .map(GenerationMutationV1::to_value)
                        .collect(),
                ),
            ),
            (
                "parent_generation_hash",
                optional_digest(self.parent_generation_hash),
            ),
            ("session_id", optional_session(self.session_id)),
            (
                "termination_reason",
                self.termination_reason
                    .as_ref()
                    .map_or(CanonicalJsonValue::Null, |value| string(value.clone())),
            ),
            ("transaction_kind", string(self.transaction_kind.as_str())),
        ]))
    }

    fn from_value(value: &CanonicalJsonValue) -> Result<Self, ManifestError> {
        let object = as_object(value, "generation")?;
        let mutations = as_array(object.get("mutations"), "mutations")?
            .iter()
            .map(GenerationMutationV1::from_value)
            .collect::<Result<Vec<_>, _>>()?;
        let generation = Self {
            archive_id: parse_archive_id(text(object, "archive_id")?)?,
            local_commit_seq: unsigned(object, "local_commit_seq")?,
            parent_generation_hash: parse_optional_digest(object.get("parent_generation_hash"))?,
            genesis_hash: parse_optional_digest(object.get("genesis_hash"))?,
            index_root: parse_index_root(
                object
                    .get("index_root")
                    .ok_or(ManifestError::InvalidField("index_root"))?,
            )?,
            archive_state: ArchiveState::parse(text(object, "archive_state")?)?,
            transaction_kind: GenerationTransactionKind::parse(text(object, "transaction_kind")?)?,
            session_id: parse_optional_session(object.get("session_id"))?,
            mutations,
            genesis: match object.get("genesis") {
                Some(CanonicalJsonValue::Null) => None,
                Some(value) => Some(GenesisV1::from_value(value)?),
                None => return Err(ManifestError::InvalidField("genesis")),
            },
            termination_reason: optional_text(object.get("termination_reason"))?,
        };
        generation.validate()?;
        Ok(generation)
    }
}

/// Exact immutable generation envelope, content hash, and storage key.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenerationObjectV1 {
    /// Decoded generation payload.
    pub generation: GenerationV1,
    /// Exact canonical envelope bytes.
    pub bytes: Vec<u8>,
    /// Content hash over exact envelope bytes.
    pub hash: Digest,
    /// Deterministic create-only object key.
    pub key: String,
}

impl GenerationObjectV1 {
    /// Constructs a generation envelope and content-addressed key.
    pub fn new(generation: GenerationV1) -> Result<Self, ManifestError> {
        let payload = generation.to_value()?;
        let bytes = encode_envelope(GENERATION_TYPE, payload);
        let hash = manifest_object_hash(&bytes);
        let key = generation_key(generation.local_commit_seq, hash);
        Ok(Self {
            generation,
            bytes,
            hash,
            key,
        })
    }

    /// Decodes and verifies an exact canonical generation envelope.
    pub fn decode(bytes: &[u8]) -> Result<Self, ManifestError> {
        let payload = decode_envelope(GENERATION_TYPE, bytes)?;
        let generation = GenerationV1::from_value(&payload)?;
        let hash = manifest_object_hash(bytes);
        let key = generation_key(generation.local_commit_seq, hash);
        Ok(Self {
            generation,
            bytes: bytes.to_vec(),
            hash,
            key,
        })
    }
}

/// One immutable head descriptor embedded in local/remote discovery pointers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HeadDescriptorV1 {
    /// Archive identity.
    pub archive_id: ArchiveId,
    /// Current local commit sequence.
    pub local_commit_seq: u64,
    /// Exact generation object key.
    pub generation_key: String,
    /// Exact generation object hash.
    pub generation_hash: Digest,
    /// Exact root page key.
    pub index_root_key: String,
    /// Exact root page hash.
    pub index_root_hash: Digest,
    /// Parent generation hash, absent only for genesis.
    pub parent_generation_hash: Option<Digest>,
    /// Generation-zero hash.
    pub genesis_hash: Digest,
    /// Resulting archive state.
    pub archive_state: ArchiveState,
}

impl HeadDescriptorV1 {
    /// Builds a head from a verified generation object.
    pub fn from_generation(object: &GenerationObjectV1) -> Result<Self, ManifestError> {
        let generation = &object.generation;
        let genesis_hash = if generation.local_commit_seq == 0 {
            object.hash
        } else {
            generation
                .genesis_hash
                .ok_or(ManifestError::InvalidDescendantShape)?
        };
        Ok(Self {
            archive_id: generation.archive_id,
            local_commit_seq: generation.local_commit_seq,
            generation_key: object.key.clone(),
            generation_hash: object.hash,
            index_root_key: index_root_key(generation.index_root.root_hash),
            index_root_hash: generation.index_root.root_hash,
            parent_generation_hash: generation.parent_generation_hash,
            genesis_hash,
            archive_state: generation.archive_state,
        })
    }

    /// Encodes this descriptor without an outer pointer envelope.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        self.to_value().to_bytes()
    }

    /// Hashes exact canonical head bytes for receipt/publication targets.
    #[must_use]
    pub fn hash(&self) -> Digest {
        domain_digest("aiperf.archive.manifest.v1", &[&self.canonical_bytes()])
    }

    fn to_value(&self) -> CanonicalJsonValue {
        object(vec![
            ("archive_id", string(uuid(self.archive_id.as_bytes()))),
            ("archive_state", string(self.archive_state.as_str())),
            ("generation_hash", string(self.generation_hash.to_hex())),
            ("generation_key", string(self.generation_key.clone())),
            ("genesis_hash", string(self.genesis_hash.to_hex())),
            ("index_root_hash", string(self.index_root_hash.to_hex())),
            ("index_root_key", string(self.index_root_key.clone())),
            (
                "local_commit_seq",
                integer(i128::from(self.local_commit_seq)),
            ),
            (
                "parent_generation_hash",
                optional_digest(self.parent_generation_hash),
            ),
        ])
    }

    fn from_value(value: &CanonicalJsonValue) -> Result<Self, ManifestError> {
        let object = as_object(value, "head")?;
        let head = Self {
            archive_id: parse_archive_id(text(object, "archive_id")?)?,
            local_commit_seq: unsigned(object, "local_commit_seq")?,
            generation_key: text(object, "generation_key")?.to_owned(),
            generation_hash: digest(object, "generation_hash")?,
            index_root_key: text(object, "index_root_key")?.to_owned(),
            index_root_hash: digest(object, "index_root_hash")?,
            parent_generation_hash: parse_optional_digest(object.get("parent_generation_hash"))?,
            genesis_hash: digest(object, "genesis_hash")?,
            archive_state: ArchiveState::parse(text(object, "archive_state")?)?,
        };
        if head.generation_key != generation_key(head.local_commit_seq, head.generation_hash)
            || head.index_root_key != index_root_key(head.index_root_hash)
        {
            return Err(ManifestError::InvalidField("content-addressed key"));
        }
        Ok(head)
    }
}

/// Checksummed fixed discovery pointer containing current and preceding heads.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LocalLatestV1 {
    /// Current authoritative head candidate.
    pub current: HeadDescriptorV1,
    /// One preceding fallback head.
    pub preceding: Option<HeadDescriptorV1>,
}

impl LocalLatestV1 {
    /// Encodes the exact canonical checksummed pointer bytes.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let payload = object(vec![
            ("current", self.current.to_value()),
            (
                "preceding",
                self.preceding
                    .as_ref()
                    .map_or(CanonicalJsonValue::Null, HeadDescriptorV1::to_value),
            ),
        ]);
        encode_envelope(LOCAL_LATEST_TYPE, payload)
    }

    /// Decodes and verifies the exact canonical pointer envelope.
    pub fn decode(bytes: &[u8]) -> Result<Self, ManifestError> {
        let payload = decode_envelope(LOCAL_LATEST_TYPE, bytes)?;
        let object = as_object(&payload, "local latest")?;
        let current = HeadDescriptorV1::from_value(
            object
                .get("current")
                .ok_or(ManifestError::InvalidField("current"))?,
        )?;
        let preceding = match object.get("preceding") {
            Some(CanonicalJsonValue::Null) => None,
            Some(value) => Some(HeadDescriptorV1::from_value(value)?),
            None => return Err(ManifestError::InvalidField("preceding")),
        };
        if let Some(preceding) = &preceding {
            if current.archive_id != preceding.archive_id
                || current.local_commit_seq != preceding.local_commit_seq + 1
                || current.parent_generation_hash != Some(preceding.generation_hash)
                || current.genesis_hash != preceding.genesis_hash
            {
                return Err(ManifestError::InvalidHeadLink);
            }
        } else if current.local_commit_seq != 0 {
            // A repaired rollback pointer may intentionally retain only its current head.
        }
        Ok(Self { current, preceding })
    }
}

/// Invalid immutable manifest bytes, links, or identity fields.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ManifestError {
    /// Canonical JSON failed or was not exact canonical bytes.
    Canonical(CanonicalJsonError),
    /// A required field has the wrong shape/value.
    InvalidField(&'static str),
    /// Generation zero violates its closed shape.
    InvalidGenesisShape,
    /// A nonzero generation violates its closed shape.
    InvalidDescendantShape,
    /// Generation mutation ordering is not removals then additions, each ascending.
    NonCanonicalMutationOrder,
    /// Envelope type/magic/version is wrong.
    EnvelopeType,
    /// Envelope payload byte length is wrong.
    PayloadLength,
    /// Envelope checksum is wrong.
    Checksum,
    /// Current and preceding head descriptors do not form one link.
    InvalidHeadLink,
    /// Integer/collection length overflowed a frozen width.
    LengthOverflow,
}

impl Display for ManifestError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Canonical(error) => write!(formatter, "invalid canonical manifest JSON: {error}"),
            Self::InvalidField(field) => write!(formatter, "invalid manifest field {field}"),
            Self::InvalidGenesisShape => formatter.write_str("invalid generation-zero shape"),
            Self::InvalidDescendantShape => {
                formatter.write_str("invalid descendant generation shape")
            }
            Self::NonCanonicalMutationOrder => {
                formatter.write_str("non-canonical generation mutation order")
            }
            Self::EnvelopeType => {
                formatter.write_str("manifest envelope magic/type/version mismatch")
            }
            Self::PayloadLength => formatter.write_str("manifest envelope payload length mismatch"),
            Self::Checksum => formatter.write_str("manifest envelope checksum mismatch"),
            Self::InvalidHeadLink => formatter.write_str("current/preceding head link mismatch"),
            Self::LengthOverflow => formatter.write_str("manifest length overflow"),
        }
    }
}

impl std::error::Error for ManifestError {}

/// Returns the deterministic generation object key.
#[must_use]
pub fn generation_key(commit_seq: u64, hash: Digest) -> String {
    format!("manifests/generation-{commit_seq}-{}.json", hash.to_hex())
}

/// Returns the deterministic index page key.
#[must_use]
pub fn index_root_key(hash: Digest) -> String {
    format!("manifest-index/{}.json", hash.to_hex())
}

fn encode_envelope(kind: &str, payload: CanonicalJsonValue) -> Vec<u8> {
    let payload_bytes = payload.to_bytes();
    let version = 1_u64.to_be_bytes();
    let checksum = domain_digest(
        "aiperf.archive.manifest.v1",
        &[kind.as_bytes(), &version, &payload_bytes],
    );
    object(vec![
        ("checksum", string(checksum.to_hex())),
        ("magic", string(ENVELOPE_MAGIC)),
        ("payload", payload),
        (
            "payload_byte_length",
            integer(i128::try_from(payload_bytes.len()).expect("usize fits i128")),
        ),
        ("type", string(kind)),
        ("version", integer(1)),
    ])
    .to_bytes()
}

fn decode_envelope(kind: &str, bytes: &[u8]) -> Result<CanonicalJsonValue, ManifestError> {
    let value = CanonicalJsonValue::parse_canonical(bytes).map_err(ManifestError::Canonical)?;
    let object = as_object(&value, "envelope")?;
    if text(object, "magic")? != ENVELOPE_MAGIC
        || text(object, "type")? != kind
        || integer_field(object, "version")? != 1
    {
        return Err(ManifestError::EnvelopeType);
    }
    let payload = object
        .get("payload")
        .cloned()
        .ok_or(ManifestError::InvalidField("payload"))?;
    let payload_bytes = payload.to_bytes();
    if integer_field(object, "payload_byte_length")?
        != i128::try_from(payload_bytes.len()).map_err(|_| ManifestError::LengthOverflow)?
    {
        return Err(ManifestError::PayloadLength);
    }
    let version = 1_u64.to_be_bytes();
    let expected = domain_digest(
        "aiperf.archive.manifest.v1",
        &[kind.as_bytes(), &version, &payload_bytes],
    );
    if digest(object, "checksum")? != expected {
        return Err(ManifestError::Checksum);
    }
    Ok(payload)
}

fn manifest_object_hash(bytes: &[u8]) -> Digest {
    domain_digest("aiperf.archive.manifest.v1", &[bytes])
}

fn index_root_value(root: &IndexRootV1) -> CanonicalJsonValue {
    object(vec![
        ("height", integer(i128::from(root.height))),
        (
            "logical_entry_count",
            integer(i128::from(root.logical_entry_count)),
        ),
        ("maximum_key", optional_key(root.maximum_key.as_ref())),
        ("minimum_key", optional_key(root.minimum_key.as_ref())),
        (
            "root_byte_length",
            integer(i128::from(root.root_byte_length)),
        ),
        ("root_hash", string(root.root_hash.to_hex())),
    ])
}

fn parse_index_root(value: &CanonicalJsonValue) -> Result<IndexRootV1, ManifestError> {
    let object = as_object(value, "index_root")?;
    Ok(IndexRootV1 {
        root_hash: digest(object, "root_hash")?,
        root_byte_length: unsigned(object, "root_byte_length")?,
        height: u16::try_from(unsigned(object, "height")?)
            .map_err(|_| ManifestError::InvalidField("height"))?,
        logical_entry_count: unsigned(object, "logical_entry_count")?,
        minimum_key: parse_optional_key(object.get("minimum_key"))?,
        maximum_key: parse_optional_key(object.get("maximum_key"))?,
    })
}

fn epoch_anchor_value(anchor: Option<EpochAnchor>) -> CanonicalJsonValue {
    anchor.map_or(CanonicalJsonValue::Null, |anchor| {
        object(vec![
            (
                "capture_uncertainty_ns",
                integer(i128::from(anchor.capture_uncertainty_ns)),
            ),
            ("clock_ns", integer(i128::from(anchor.clock_ns))),
            ("unix_epoch_ns", string(anchor.unix_epoch_ns.to_string())),
        ])
    })
}

fn parse_epoch_anchor(
    value: Option<&CanonicalJsonValue>,
) -> Result<Option<EpochAnchor>, ManifestError> {
    match value {
        Some(CanonicalJsonValue::Null) => Ok(None),
        Some(value) => {
            let object = as_object(value, "epoch_anchor")?;
            let clock_ns = i64::try_from(integer_field(object, "clock_ns")?)
                .map_err(|_| ManifestError::InvalidField("clock_ns"))?;
            let capture_uncertainty_ns = unsigned(object, "capture_uncertainty_ns")?;
            let unix_epoch_ns = text(object, "unix_epoch_ns")?
                .parse::<i128>()
                .map_err(|_| ManifestError::InvalidField("unix_epoch_ns"))?;
            Ok(Some(EpochAnchor {
                clock_ns,
                unix_epoch_ns,
                capture_uncertainty_ns,
            }))
        }
        None => Err(ManifestError::InvalidField("epoch_anchor")),
    }
}

fn object(entries: Vec<(&str, CanonicalJsonValue)>) -> CanonicalJsonValue {
    CanonicalJsonValue::object(
        entries
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value)),
    )
    .expect("static manifest keys are unique")
}

fn string(value: impl Into<String>) -> CanonicalJsonValue {
    CanonicalJsonValue::String(value.into())
}

const fn integer(value: i128) -> CanonicalJsonValue {
    CanonicalJsonValue::Integer(value)
}

fn optional_digest(value: Option<Digest>) -> CanonicalJsonValue {
    value.map_or(CanonicalJsonValue::Null, |digest| string(digest.to_hex()))
}

fn optional_session(value: Option<SessionId>) -> CanonicalJsonValue {
    value.map_or(CanonicalJsonValue::Null, |session| {
        string(uuid(session.as_bytes()))
    })
}

fn optional_key(value: Option<&IndexKey>) -> CanonicalJsonValue {
    value.map_or(CanonicalJsonValue::Null, |key| string(hex(key.as_bytes())))
}

fn as_object<'a>(
    value: &'a CanonicalJsonValue,
    field: &'static str,
) -> Result<&'a BTreeMap<String, CanonicalJsonValue>, ManifestError> {
    value.as_object().ok_or(ManifestError::InvalidField(field))
}

fn as_array<'a>(
    value: Option<&'a CanonicalJsonValue>,
    field: &'static str,
) -> Result<&'a [CanonicalJsonValue], ManifestError> {
    match value {
        Some(CanonicalJsonValue::Array(values)) => Ok(values),
        _ => Err(ManifestError::InvalidField(field)),
    }
}

fn text<'a>(
    object: &'a BTreeMap<String, CanonicalJsonValue>,
    field: &'static str,
) -> Result<&'a str, ManifestError> {
    object
        .get(field)
        .and_then(CanonicalJsonValue::as_str)
        .ok_or(ManifestError::InvalidField(field))
}

fn integer_field(
    object: &BTreeMap<String, CanonicalJsonValue>,
    field: &'static str,
) -> Result<i128, ManifestError> {
    object
        .get(field)
        .and_then(CanonicalJsonValue::as_i128)
        .ok_or(ManifestError::InvalidField(field))
}

fn unsigned(
    object: &BTreeMap<String, CanonicalJsonValue>,
    field: &'static str,
) -> Result<u64, ManifestError> {
    u64::try_from(integer_field(object, field)?).map_err(|_| ManifestError::InvalidField(field))
}

fn digest(
    object: &BTreeMap<String, CanonicalJsonValue>,
    field: &'static str,
) -> Result<Digest, ManifestError> {
    Digest::parse(text(object, field)?).map_err(|_| ManifestError::InvalidField(field))
}

fn parse_optional_digest(
    value: Option<&CanonicalJsonValue>,
) -> Result<Option<Digest>, ManifestError> {
    match value {
        Some(CanonicalJsonValue::Null) => Ok(None),
        Some(CanonicalJsonValue::String(value)) => Digest::parse(value)
            .map(Some)
            .map_err(|_| ManifestError::InvalidField("optional digest")),
        _ => Err(ManifestError::InvalidField("optional digest")),
    }
}

fn optional_text(value: Option<&CanonicalJsonValue>) -> Result<Option<String>, ManifestError> {
    match value {
        Some(CanonicalJsonValue::Null) => Ok(None),
        Some(CanonicalJsonValue::String(value)) => Ok(Some(value.clone())),
        _ => Err(ManifestError::InvalidField("optional string")),
    }
}

fn parse_optional_session(
    value: Option<&CanonicalJsonValue>,
) -> Result<Option<SessionId>, ManifestError> {
    match value {
        Some(CanonicalJsonValue::Null) => Ok(None),
        Some(CanonicalJsonValue::String(value)) => parse_session_id(value).map(Some),
        _ => Err(ManifestError::InvalidField("session_id")),
    }
}

fn parse_optional_key(
    value: Option<&CanonicalJsonValue>,
) -> Result<Option<IndexKey>, ManifestError> {
    match value {
        Some(CanonicalJsonValue::Null) => Ok(None),
        Some(CanonicalJsonValue::String(value)) => IndexKey::new(decode_hex(value)?)
            .map(Some)
            .map_err(|_| ManifestError::InvalidField("index key")),
        _ => Err(ManifestError::InvalidField("index key")),
    }
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

fn parse_archive_id(value: &str) -> Result<ArchiveId, ManifestError> {
    ArchiveId::new(parse_uuid(value)?).map_err(|_| ManifestError::InvalidField("archive_id"))
}

fn parse_session_id(value: &str) -> Result<SessionId, ManifestError> {
    SessionId::new(parse_uuid(value)?).map_err(|_| ManifestError::InvalidField("session_id"))
}

fn parse_uuid(value: &str) -> Result<[u8; 16], ManifestError> {
    if value.len() != 36
        || value.as_bytes()[8] != b'-'
        || value.as_bytes()[13] != b'-'
        || value.as_bytes()[18] != b'-'
        || value.as_bytes()[23] != b'-'
    {
        return Err(ManifestError::InvalidField("uuid"));
    }
    let compact: String = value
        .chars()
        .filter(|character| *character != '-')
        .collect();
    decode_hex(&compact)?
        .try_into()
        .map_err(|_| ManifestError::InvalidField("uuid"))
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

fn decode_hex(value: &str) -> Result<Vec<u8>, ManifestError> {
    if value.len() % 2 != 0 {
        return Err(ManifestError::InvalidField("hex"));
    }
    let mut bytes = Vec::with_capacity(value.len() / 2);
    for pair in value.as_bytes().chunks_exact(2) {
        let high = nibble(pair[0]).ok_or(ManifestError::InvalidField("hex"))?;
        let low = nibble(pair[1]).ok_or(ManifestError::InvalidField("hex"))?;
        bytes.push((high << 4) | low);
    }
    Ok(bytes)
}

fn nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IndexSnapshot;

    fn archive() -> ArchiveId {
        ArchiveId::new([0x11; 16]).unwrap()
    }

    fn session() -> SessionId {
        SessionId::new([0x22; 16]).unwrap()
    }

    fn genesis(root: IndexRootV1) -> GenerationV1 {
        GenerationV1 {
            archive_id: archive(),
            local_commit_seq: 0,
            parent_generation_hash: None,
            genesis_hash: None,
            index_root: root,
            archive_state: ArchiveState::Open,
            transaction_kind: GenerationTransactionKind::Genesis,
            session_id: Some(session()),
            mutations: vec![],
            genesis: Some(GenesisV1 {
                archive_id: archive(),
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
                initial_session_id: Some(session()),
                time_domain: TimeDomain::Real,
                epoch_anchor: Some(EpochAnchor {
                    clock_ns: 10,
                    unix_epoch_ns: 1_700_000_000_000_000_000,
                    capture_uncertainty_ns: 2,
                }),
            }),
            termination_reason: None,
        }
    }

    #[test]
    fn generation_and_head_round_trip_exact_canonical_bytes() {
        let root = IndexSnapshot::empty().unwrap().root().clone();
        let object = GenerationObjectV1::new(genesis(root)).unwrap();
        assert_eq!(GenerationObjectV1::decode(&object.bytes).unwrap(), object);
        let head = HeadDescriptorV1::from_generation(&object).unwrap();
        let pointer = LocalLatestV1 {
            current: head,
            preceding: None,
        };
        assert_eq!(
            LocalLatestV1::decode(&pointer.canonical_bytes()).unwrap(),
            pointer
        );
    }

    #[test]
    fn envelope_corruption_and_noncanonical_json_fail_closed() {
        let root = IndexSnapshot::empty().unwrap().root().clone();
        let object = GenerationObjectV1::new(genesis(root)).unwrap();
        let mut corrupt = object.bytes.clone();
        let offset = corrupt.iter().position(|byte| *byte == b'1').unwrap();
        corrupt[offset] = b'2';
        assert!(GenerationObjectV1::decode(&corrupt).is_err());

        let mut spaced = object.bytes.clone();
        spaced.insert(1, b' ');
        assert!(matches!(
            GenerationObjectV1::decode(&spaced),
            Err(ManifestError::Canonical(CanonicalJsonError::NonCanonical))
        ));
    }

    #[test]
    fn current_and_preceding_must_be_one_exact_link() {
        let root = IndexSnapshot::empty().unwrap().root().clone();
        let genesis_object = GenerationObjectV1::new(genesis(root.clone())).unwrap();
        let genesis_head = HeadDescriptorV1::from_generation(&genesis_object).unwrap();
        let descendant = GenerationObjectV1::new(GenerationV1 {
            archive_id: archive(),
            local_commit_seq: 1,
            parent_generation_hash: Some(genesis_object.hash),
            genesis_hash: Some(genesis_object.hash),
            index_root: root,
            archive_state: ArchiveState::Open,
            transaction_kind: GenerationTransactionKind::Checkpoint,
            session_id: Some(session()),
            mutations: vec![],
            genesis: None,
            termination_reason: None,
        })
        .unwrap();
        let descendant_head = HeadDescriptorV1::from_generation(&descendant).unwrap();
        let pointer = LocalLatestV1 {
            current: descendant_head,
            preceding: Some(genesis_head.clone()),
        };
        assert!(LocalLatestV1::decode(&pointer.canonical_bytes()).is_ok());

        let invalid = LocalLatestV1 {
            current: genesis_head.clone(),
            preceding: Some(genesis_head),
        };
        assert!(matches!(
            LocalLatestV1::decode(&invalid.canonical_bytes()),
            Err(ManifestError::InvalidHeadLink)
        ));
    }
}
