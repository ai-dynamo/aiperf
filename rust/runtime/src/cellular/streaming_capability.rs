// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Controller-to-cell propagation of one accepted streaming capability
//! agreement.
//!
//! The controller selects a streaming capability combination once, admits it
//! through [`StreamingCapabilityAgreement`], and seals the selected identifiers
//! plus two digests into [`StreamingCapabilityPropagation`]. Each cell resolves
//! those identifiers against its **own** frozen registry, re-runs the same
//! agreement over its own linked descriptors, and proves the result is
//! identical.
//!
//! No descriptor field is ever accepted from the wire. The propagated
//! identifiers are lookup keys; every capability rule runs over `'static`
//! metadata compiled into the deciding process. A cell can therefore only ever
//! execute with a factory it linked itself: an unknown identifier is refused,
//! never fetched, never synthesized, and never stubbed.

use serde::{Deserialize, Serialize};

use crate::engine::registry::{
    StreamingCapabilityAgreement, StreamingCapabilityError, StreamingCapabilityPlan,
    StreamingSelectedDescriptors,
};
use crate::extensions::AIPerfRegistry;
use crate::streaming::action::{ActionPlacement, ActionResultRetention};
use crate::streaming::format::{FormatProjection, FormatStateRetention};
use crate::streaming::session::{SessionPlacement, SessionStateRetention};
use crate::streaming::source::StreamingSourceRetention;

/// Fixed protocol version for the propagated streaming capability agreement.
pub const STREAMING_CAPABILITY_PROPAGATION_VERSION: u16 = 1;

const DESCRIPTOR_DIGEST_DOMAIN: &[u8] = b"aiperf-streaming-descriptor-v1\0";
const PLAN_DIGEST_DOMAIN: &[u8] = b"aiperf-streaming-plan-v1\0";

/// One selected streaming capability axis, used for diagnostics and digest
/// domain separation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamingCapabilityCategory {
    /// Streaming dataset source.
    Source,
    /// Streaming dataset format/decoder.
    Format,
    /// Streaming session program.
    SessionProgram,
    /// Streaming action-sink binding.
    ActionSink,
    /// Protocol-v2 transport.
    Transport,
    /// Endpoint dialect.
    Endpoint,
    /// Streaming checkpoint backend.
    CheckpointBackend,
}

impl StreamingCapabilityCategory {
    /// Stable lowercase tag used in diagnostics and digest domain separation.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Source => "stream_source",
            Self::Format => "stream_format",
            Self::SessionProgram => "stream_session_program",
            Self::ActionSink => "stream_action_sink",
            Self::Transport => "transport",
            Self::Endpoint => "endpoint",
            Self::CheckpointBackend => "stream_checkpoint_backend",
        }
    }
}

impl std::fmt::Display for StreamingCapabilityCategory {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.tag())
    }
}

/// The controller's accepted streaming capability agreement, in wire form.
///
/// Owned data only: the runtime descriptors this mirrors are built from
/// `&'static` metadata and are `Serialize`-only by design, so they cannot be
/// decoded from untrusted bytes. Encode and decode this DTO through
/// [`StreamingCapabilityPropagation::encode`] and
/// [`StreamingCapabilityPropagation::decode`], which use
/// `rmp_serde::to_vec_named` / `rmp_serde::from_slice`; the named encoding is
/// what makes `deny_unknown_fields` load-bearing, because the positional
/// `to_vec` encoding carries no field names for the decoder to reject.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamingCapabilityPropagation {
    /// Propagation protocol version.
    pub version: u16,
    /// Selected streaming source identifier.
    pub source: String,
    /// Selected streaming format identifier.
    pub format: String,
    /// Selected streaming session-program identifier.
    pub session: String,
    /// Selected streaming action-sink identifier.
    pub action_sink: String,
    /// Selected protocol-v2 transport identifier.
    pub transport: String,
    /// Selected endpoint dialect identifier, when one was selected.
    pub endpoint: Option<String>,
    /// Selected streaming checkpoint backend identifier, when one was selected.
    pub checkpoint_backend: Option<String>,
    /// Conformance digest of the controller's source descriptor.
    pub source_digest: [u8; 32],
    /// Conformance digest of the controller's format descriptor.
    pub format_digest: [u8; 32],
    /// Conformance digest of the controller's session-program descriptor.
    pub session_digest: [u8; 32],
    /// Conformance digest of the controller's action-sink descriptor.
    pub action_sink_digest: [u8; 32],
    /// Conformance digest of the controller's transport descriptor.
    pub transport_digest: [u8; 32],
    /// Conformance digest of the controller's endpoint descriptor, when selected.
    pub endpoint_digest: Option<[u8; 32]>,
    /// Conformance digest of the controller's checkpoint backend descriptor.
    pub checkpoint_backend_digest: Option<[u8; 32]>,
    /// Canonical action schema the controller's agreement settled on.
    pub agreed_action_schema: String,
    /// Canonical fragment schema the controller's agreement settled on.
    pub agreed_fragment_schema: String,
    /// Digest over the complete accepted plan, including every derived fact.
    pub plan_digest: [u8; 32],
}

/// Why a controller could not seal its own accepted selection.
///
/// Sealing runs the agreement itself, so a selection that does not admit is
/// reported here rather than surfacing later as a cell-side refusal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamingCapabilitySealError {
    /// The agreement refused the selection.
    Refused(StreamingCapabilityError),
    /// One frozen descriptor could not be encoded for its conformance digest.
    DescriptorEncode {
        /// Axis whose descriptor could not be encoded.
        category: StreamingCapabilityCategory,
    },
    /// The accepted plan could not be encoded for its digest.
    PlanEncode,
}

impl std::fmt::Display for StreamingCapabilitySealError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Refused(error) => write!(formatter, "{error}"),
            Self::DescriptorEncode { category } => write!(
                formatter,
                "the linked {category} descriptor could not be encoded for its conformance digest"
            ),
            Self::PlanEncode => formatter
                .write_str("the accepted streaming plan could not be encoded for its digest"),
        }
    }
}

impl std::error::Error for StreamingCapabilitySealError {}

/// Why a cell could not reproduce the controller's streaming agreement.
///
/// Every variant is terminal for the run. There is no renegotiation and no
/// degraded mode: a cell that cannot reproduce the agreement never issues.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StreamingCapabilityNegotiationError {
    /// The controller sent a propagation version this cell does not implement.
    UnsupportedVersion {
        /// Version this cell implements.
        expected: u16,
        /// Version the controller sent.
        received: u16,
    },
    /// The propagated bytes are not a strict, complete propagation document.
    Malformed,
    /// This cell has no factory registered under a selected identifier.
    MissingFactory {
        /// Axis whose identifier could not be resolved.
        category: StreamingCapabilityCategory,
        /// Identifier the controller selected.
        id: String,
    },
    /// This cell has a factory under the selected identifier, but its declared
    /// capability metadata differs from the controller's.
    DescriptorMismatch {
        /// Axis whose descriptor diverged.
        category: StreamingCapabilityCategory,
        /// Identifier whose descriptor diverged.
        id: String,
    },
    /// The agreement refused the same selection when run over this cell's
    /// descriptors.
    AgreementRefused(StreamingCapabilityError),
    /// A schema the controller's agreement settled on is not the schema this
    /// cell's agreement settles on.
    AgreedSchemaMismatch {
        /// `"action"` or `"fragment"`.
        role: &'static str,
        /// Schema the controller agreed.
        expected: String,
        /// Schema this cell agreed.
        local: String,
    },
    /// Every named check passed but a derived plan fact still differs.
    PlanMismatch,
}

impl std::fmt::Display for StreamingCapabilityNegotiationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedVersion { expected, received } => write!(
                formatter,
                "streaming capability propagation version {received} is not supported \
                 (this cell implements {expected})"
            ),
            Self::Malformed => formatter.write_str("streaming capability propagation is malformed"),
            Self::MissingFactory { category, id } => write!(
                formatter,
                "this cell has no registered {category} factory {id:?} selected by the controller"
            ),
            Self::DescriptorMismatch { category, id } => write!(
                formatter,
                "this cell's {category} factory {id:?} declares different capability metadata \
                 than the controller's"
            ),
            Self::AgreementRefused(error) => write!(
                formatter,
                "this cell's streaming capability agreement refused the controller's selection: \
                 {error}"
            ),
            Self::AgreedSchemaMismatch {
                role,
                expected,
                local,
            } => write!(
                formatter,
                "controller agreed {role} schema {expected:?} but this cell agrees {local:?}"
            ),
            Self::PlanMismatch => formatter
                .write_str("this cell's accepted streaming plan differs from the controller's"),
        }
    }
}

impl std::error::Error for StreamingCapabilityNegotiationError {}

/// The controller could not encode its own accepted agreement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamingCapabilityPropagationEncodeError;

impl std::fmt::Display for StreamingCapabilityPropagationEncodeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("streaming capability propagation could not be encoded")
    }
}

impl std::error::Error for StreamingCapabilityPropagationEncodeError {}

/// Canonical plan-digest input, assembled from public plan accessors only.
///
/// Serialize-only and private: it exists to give the plan one stable byte
/// image, never to be decoded. Building it from accessors rather than plan
/// internals keeps `engine/registry.rs` untouched by this module.
#[derive(Serialize)]
struct PlanDigestInput<'a> {
    version: u16,
    source: &'a str,
    format: &'a str,
    session: &'a str,
    action_sink: &'a str,
    transport: &'a str,
    endpoint: Option<&'a str>,
    checkpoint_backend: Option<&'a str>,
    agreed_action_schema: &'a str,
    agreed_fragment_schema: &'a str,
    source_retention: StreamingSourceRetention,
    format_retention: FormatStateRetention,
    session_retention: SessionStateRetention,
    action_retention: ActionResultRetention,
    needs_spill_authority: bool,
    needs_durable_resume: bool,
    session_placement: SessionPlacement,
    action_placement: ActionPlacement,
    format_projection: FormatProjection,
}

fn update_field(hasher: &mut blake3::Hasher, field: &[u8]) {
    hasher.update(&(field.len() as u64).to_le_bytes());
    hasher.update(field);
}

/// Domain-separated, length-prefixed BLAKE3 over an ordered field list, so a
/// field boundary can never be forged by concatenating adjacent values.
fn domain_hash(domain: &'static [u8], fields: &[&[u8]]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    update_field(&mut hasher, domain);
    for field in fields {
        update_field(&mut hasher, field);
    }
    *hasher.finalize().as_bytes()
}

/// Conformance digest of one frozen descriptor.
///
/// Named MessagePack means a field *rename* changes the digest — correct, it is
/// a contract change — while reordering the fields in the source does not.
///
/// Returns `None` only when the descriptor cannot be encoded, which is a
/// build-time bug in the registering extension rather than runtime data; the
/// caller turns that into a fail-closed refusal instead of panicking.
fn descriptor_digest<D: Serialize + ?Sized>(
    category: StreamingCapabilityCategory,
    descriptor: &D,
) -> Option<[u8; 32]> {
    let encoded = rmp_serde::to_vec_named(descriptor).ok()?;
    Some(domain_hash(
        DESCRIPTOR_DIGEST_DOMAIN,
        &[category.tag().as_bytes(), &encoded],
    ))
}

/// Digest over one accepted plan, covering every derived fact the two named
/// agreed schemas do not: retention, spill/durability requirements, both
/// placements, and the format projection.
fn plan_digest(plan: &StreamingCapabilityPlan) -> Option<[u8; 32]> {
    let ids = plan.selected_ids();
    let retention = plan.retention();
    let input = PlanDigestInput {
        version: STREAMING_CAPABILITY_PROPAGATION_VERSION,
        source: ids.source,
        format: ids.format,
        session: ids.session,
        action_sink: ids.action_sink,
        transport: ids.transport,
        endpoint: ids.endpoint,
        checkpoint_backend: ids.checkpoint_backend,
        agreed_action_schema: plan.agreed_action_schema(),
        agreed_fragment_schema: plan.agreed_fragment_schema(),
        source_retention: retention.source,
        format_retention: retention.format,
        session_retention: retention.session,
        action_retention: retention.action_sink,
        needs_spill_authority: retention.needs_spill_authority,
        needs_durable_resume: retention.needs_durable_resume,
        session_placement: plan.session_placement(),
        action_placement: plan.action_placement(),
        format_projection: plan.format_projection(),
    };
    let encoded = rmp_serde::to_vec_named(&input).ok()?;
    Some(domain_hash(PLAN_DIGEST_DOMAIN, &[&encoded]))
}

impl StreamingCapabilityPropagation {
    /// Admit one selection and seal it for propagation to every cell.
    ///
    /// Running the agreement here rather than accepting a caller-supplied plan
    /// makes a selection/plan mismatch unrepresentable: there is one admission
    /// and one sealed result.
    ///
    /// # Errors
    ///
    /// Returns [`StreamingCapabilitySealError::Refused`] when the agreement
    /// rejects the selection, and an encode variant when a linked descriptor or
    /// the accepted plan cannot be encoded for its digest.
    pub fn seal(
        selected: StreamingSelectedDescriptors,
    ) -> Result<Self, StreamingCapabilitySealError> {
        let plan = StreamingCapabilityAgreement::validate(selected)
            .map_err(StreamingCapabilitySealError::Refused)?;
        let ids = plan.selected_ids();

        // A descriptor that cannot encode is a linking bug in this process;
        // refuse the selection rather than ship a zero digest.
        fn seal_digest<D: Serialize + ?Sized>(
            category: StreamingCapabilityCategory,
            descriptor: &D,
        ) -> Result<[u8; 32], StreamingCapabilitySealError> {
            descriptor_digest(category, descriptor)
                .ok_or(StreamingCapabilitySealError::DescriptorEncode { category })
        }

        let source_digest = seal_digest(StreamingCapabilityCategory::Source, selected.source)?;
        let format_digest = seal_digest(StreamingCapabilityCategory::Format, selected.format)?;
        let session_digest = seal_digest(
            StreamingCapabilityCategory::SessionProgram,
            selected.session,
        )?;
        let action_sink_digest = seal_digest(
            StreamingCapabilityCategory::ActionSink,
            selected.action_sink,
        )?;
        let transport_digest =
            seal_digest(StreamingCapabilityCategory::Transport, selected.transport)?;
        let endpoint_digest = match selected.endpoint {
            Some(endpoint) => Some(seal_digest(
                StreamingCapabilityCategory::Endpoint,
                endpoint,
            )?),
            None => None,
        };
        let checkpoint_backend_digest = match selected.checkpoint_backend {
            Some(backend) => Some(seal_digest(
                StreamingCapabilityCategory::CheckpointBackend,
                backend,
            )?),
            None => None,
        };

        Ok(Self {
            version: STREAMING_CAPABILITY_PROPAGATION_VERSION,
            source: ids.source.to_owned(),
            format: ids.format.to_owned(),
            session: ids.session.to_owned(),
            action_sink: ids.action_sink.to_owned(),
            transport: ids.transport.to_owned(),
            endpoint: ids.endpoint.map(str::to_owned),
            checkpoint_backend: ids.checkpoint_backend.map(str::to_owned),
            source_digest,
            format_digest,
            session_digest,
            action_sink_digest,
            transport_digest,
            endpoint_digest,
            checkpoint_backend_digest,
            agreed_action_schema: plan.agreed_action_schema().to_owned(),
            agreed_fragment_schema: plan.agreed_fragment_schema().to_owned(),
            plan_digest: plan_digest(&plan).ok_or(StreamingCapabilitySealError::PlanEncode)?,
        })
    }

    /// Encode canonical propagation bytes for the register reply.
    ///
    /// `to_vec_named` is required: the positional `to_vec` encoding would make
    /// `deny_unknown_fields` inert on the decode side.
    ///
    /// # Errors
    ///
    /// Returns an error only when this owned DTO cannot be MessagePack-encoded.
    pub fn encode(&self) -> Result<Vec<u8>, StreamingCapabilityPropagationEncodeError> {
        rmp_serde::to_vec_named(self).map_err(|_| StreamingCapabilityPropagationEncodeError)
    }

    /// Decode strict propagation bytes received from the controller.
    ///
    /// # Errors
    ///
    /// Returns [`StreamingCapabilityNegotiationError::Malformed`] for bytes
    /// that are not a strict, complete propagation document, and
    /// [`StreamingCapabilityNegotiationError::UnsupportedVersion`] for a
    /// version this build does not implement. No registry lookup happens on
    /// either path.
    pub fn decode(bytes: &[u8]) -> Result<Self, StreamingCapabilityNegotiationError> {
        let propagation: Self = rmp_serde::from_slice(bytes)
            .map_err(|_| StreamingCapabilityNegotiationError::Malformed)?;
        if propagation.version != STREAMING_CAPABILITY_PROPAGATION_VERSION {
            return Err(StreamingCapabilityNegotiationError::UnsupportedVersion {
                expected: STREAMING_CAPABILITY_PROPAGATION_VERSION,
                received: propagation.version,
            });
        }
        Ok(propagation)
    }

    /// Rebuild this cell's own agreement from its locally registered factories
    /// and prove it is the controller's.
    ///
    /// Pure: registry lookups, `'static` descriptor reads, one agreement run,
    /// and two digests. No I/O, no clock, no factory `prepare`, no lease.
    ///
    /// # Errors
    ///
    /// Returns the first failed check: a selected identifier this cell does not
    /// have, a same-identifier descriptor whose declared capability differs, a
    /// local agreement refusal, an agreed-schema divergence, or a derived-plan
    /// divergence.
    pub fn reconstruct(
        &self,
        registry: &AIPerfRegistry,
    ) -> Result<StreamingCapabilityPlan, StreamingCapabilityNegotiationError> {
        let missing = |category: StreamingCapabilityCategory, id: &str| {
            StreamingCapabilityNegotiationError::MissingFactory {
                category,
                id: id.to_owned(),
            }
        };

        let source = registry
            .stream_source_factory(&self.source)
            .ok_or_else(|| missing(StreamingCapabilityCategory::Source, &self.source))?
            .descriptor();
        let format = registry
            .stream_format_factory(&self.format)
            .ok_or_else(|| missing(StreamingCapabilityCategory::Format, &self.format))?
            .descriptor();
        let session = registry
            .stream_session_program_factory(&self.session)
            .ok_or_else(|| missing(StreamingCapabilityCategory::SessionProgram, &self.session))?
            .descriptor();
        let action_sink = registry
            .stream_action_sink_factory(&self.action_sink)
            .ok_or_else(|| missing(StreamingCapabilityCategory::ActionSink, &self.action_sink))?
            .descriptor();
        let transport = registry
            .transport_descriptors()
            .into_iter()
            .find(|descriptor| descriptor.id == self.transport)
            .ok_or_else(|| missing(StreamingCapabilityCategory::Transport, &self.transport))?;
        let endpoint = match self.endpoint.as_deref() {
            Some(id) => Some(
                registry
                    .endpoints()
                    .descriptors()
                    .find(|descriptor| descriptor.id == id)
                    .ok_or_else(|| missing(StreamingCapabilityCategory::Endpoint, id))?,
            ),
            None => None,
        };
        let checkpoint_backend = match self.checkpoint_backend.as_deref() {
            Some(id) => Some(
                registry
                    .stream_checkpoint_backend_factory(id)
                    .ok_or_else(|| missing(StreamingCapabilityCategory::CheckpointBackend, id))?
                    .descriptor(),
            ),
            None => None,
        };

        self.check_digest(
            StreamingCapabilityCategory::Source,
            &self.source,
            self.source_digest,
            source,
        )?;
        self.check_digest(
            StreamingCapabilityCategory::Format,
            &self.format,
            self.format_digest,
            format,
        )?;
        self.check_digest(
            StreamingCapabilityCategory::SessionProgram,
            &self.session,
            self.session_digest,
            session,
        )?;
        self.check_digest(
            StreamingCapabilityCategory::ActionSink,
            &self.action_sink,
            self.action_sink_digest,
            action_sink,
        )?;
        self.check_digest(
            StreamingCapabilityCategory::Transport,
            &self.transport,
            self.transport_digest,
            transport,
        )?;
        // An identifier present without its digest (or the reverse) is not a
        // representable seal, so it is refused rather than skipped: a skipped
        // optional axis would be an unchecked descriptor.
        match (self.endpoint.as_deref(), endpoint, self.endpoint_digest) {
            (Some(id), Some(descriptor), Some(expected)) => self.check_digest(
                StreamingCapabilityCategory::Endpoint,
                id,
                expected,
                descriptor,
            )?,
            (None, None, None) => {}
            _ => return Err(StreamingCapabilityNegotiationError::Malformed),
        }
        match (
            self.checkpoint_backend.as_deref(),
            checkpoint_backend,
            self.checkpoint_backend_digest,
        ) {
            (Some(id), Some(descriptor), Some(expected)) => self.check_digest(
                StreamingCapabilityCategory::CheckpointBackend,
                id,
                expected,
                descriptor,
            )?,
            (None, None, None) => {}
            _ => return Err(StreamingCapabilityNegotiationError::Malformed),
        }

        let selected = StreamingSelectedDescriptors {
            source,
            format,
            session,
            action_sink,
            transport,
            endpoint,
            checkpoint_backend,
        };
        let plan = StreamingCapabilityAgreement::validate(selected)
            .map_err(StreamingCapabilityNegotiationError::AgreementRefused)?;

        if plan.agreed_action_schema() != self.agreed_action_schema {
            return Err(StreamingCapabilityNegotiationError::AgreedSchemaMismatch {
                role: "action",
                expected: self.agreed_action_schema.clone(),
                local: plan.agreed_action_schema().to_owned(),
            });
        }
        if plan.agreed_fragment_schema() != self.agreed_fragment_schema {
            return Err(StreamingCapabilityNegotiationError::AgreedSchemaMismatch {
                role: "fragment",
                expected: self.agreed_fragment_schema.clone(),
                local: plan.agreed_fragment_schema().to_owned(),
            });
        }
        // Catches every derived fact the two named schemas do not cover:
        // retention, spill/durability requirements, both placements, and the
        // format projection.
        if plan_digest(&plan) != Some(self.plan_digest) {
            return Err(StreamingCapabilityNegotiationError::PlanMismatch);
        }
        Ok(plan)
    }

    fn check_digest<D: Serialize + ?Sized>(
        &self,
        category: StreamingCapabilityCategory,
        id: &str,
        expected: [u8; 32],
        descriptor: &D,
    ) -> Result<(), StreamingCapabilityNegotiationError> {
        let mismatch = || StreamingCapabilityNegotiationError::DescriptorMismatch {
            category,
            id: id.to_owned(),
        };
        if descriptor_digest(category, descriptor).ok_or_else(mismatch)? == expected {
            Ok(())
        } else {
            Err(mismatch())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::engine::registry::{ClockKind, TransportDescriptor};
    use crate::streaming::action::{EndpointRetrySafety, StreamingActionSinkDescriptor};
    use crate::streaming::format::StreamingFormatDescriptor;
    use crate::streaming::identity::ContentDigest;
    use crate::streaming::session::{SessionClosureCapability, StreamingSessionProgramDescriptor};
    use crate::streaming::source::{
        PartitionAccessKind, StreamingResumeGranularity, StreamingSourceDescriptor,
        StreamingSourceMode, StreamingSourceOrdering, StreamingSourcePlacement,
    };

    static SOURCE: StreamingSourceDescriptor = StreamingSourceDescriptor {
        id: "source_a",
        description: "test-only streaming source",
        modes: &[StreamingSourceMode::Finite, StreamingSourceMode::Follow],
        access: &[PartitionAccessKind::Sequential],
        ordering: StreamingSourceOrdering::EventTime,
        resume: &[StreamingResumeGranularity::Record],
        has_event_time: true,
        has_stable_record_ids: true,
        retention: StreamingSourceRetention::BoundedMemory,
        placement: StreamingSourcePlacement::ControllerOnly,
        supports_virtual_clock: true,
    };

    static FORMAT: StreamingFormatDescriptor = StreamingFormatDescriptor {
        id: "format_a",
        description: "test-only streaming format",
        semantic_digest: ContentDigest::from_bytes([0u8; 32]),
        media_types: &["application/jsonl"],
        input_schemas: &["test.source.v1"],
        required_access: PartitionAccessKind::Sequential,
        projection: FormatProjection::FullRecord,
        output_schema: "test.fragment.v1",
        has_event_time: true,
        has_stable_record_ids: true,
        retention: FormatStateRetention::BoundedMemory,
        supports_virtual_clock: true,
    };

    static SESSION_CONTROLLER: StreamingSessionProgramDescriptor =
        StreamingSessionProgramDescriptor {
            id: "session_a",
            description: "test-only session program",
            fragment_input_schemas: &["test.fragment.v1"],
            action_schemas: &["test.action.v1"],
            closure: &[SessionClosureCapability::ExplicitClose],
            retention: SessionStateRetention::BoundedMemory,
            placement: SessionPlacement::ControllerCanonical,
            supports_virtual_clock: true,
        };

    // Identical to `SESSION_CONTROLLER` except for `placement`, which the two
    // agreed schemas do not carry — the exact divergence `plan_digest` exists
    // to catch. `RoutedByStableSession` needs a shared checkpoint backend, so
    // this fixture is only agreed without one when paired with `SINK_LOCAL`.
    static SESSION_ROUTED: StreamingSessionProgramDescriptor = StreamingSessionProgramDescriptor {
        id: "session_a",
        description: "test-only session program",
        fragment_input_schemas: &["test.fragment.v1"],
        action_schemas: &["test.action.v1"],
        closure: &[SessionClosureCapability::ExplicitClose],
        retention: SessionStateRetention::BoundedMemory,
        placement: SessionPlacement::RoutedByStableSession,
        supports_virtual_clock: true,
    };

    static SINK: StreamingActionSinkDescriptor = StreamingActionSinkDescriptor {
        id: "sink_a",
        description: "test-only action sink",
        accepted_schemas: &["test.action.v1"],
        transport_ids: &["dry_run"],
        endpoint_kinds: &["chat"],
        retention: ActionResultRetention::StreamingTerminal,
        placement: ActionPlacement::WorkerLocal,
        supports_virtual_clock: true,
    };

    static TRANSPORT: TransportDescriptor = TransportDescriptor {
        id: "dry_run",
        description: "test-only transport",
        clock: ClockKind::Real,
        features: &[],
        url_schemes: &[],
    };

    fn selection(
        session: &'static StreamingSessionProgramDescriptor,
    ) -> StreamingSelectedDescriptors {
        StreamingSelectedDescriptors {
            source: &SOURCE,
            format: &FORMAT,
            session,
            action_sink: &SINK,
            transport: &TRANSPORT,
            endpoint: None,
            checkpoint_backend: None,
        }
    }

    #[test]
    fn descriptor_digest_is_domain_separated_per_category() {
        let as_source = descriptor_digest(StreamingCapabilityCategory::Source, &SOURCE)
            .expect("a static descriptor encodes");
        let as_format = descriptor_digest(StreamingCapabilityCategory::Format, &SOURCE)
            .expect("a static descriptor encodes");
        assert_ne!(
            as_source, as_format,
            "the same descriptor bytes under two categories must not share a digest"
        );
    }

    #[test]
    fn plan_digest_covers_retention_and_placement() {
        let controller_plan =
            StreamingCapabilityAgreement::validate(selection(&SESSION_CONTROLLER))
                .expect("the fixture selection is agreed");
        let routed_plan = StreamingCapabilityAgreement::validate(selection(&SESSION_ROUTED))
            .expect("the fixture selection is agreed");

        assert_eq!(
            controller_plan.selected_ids(),
            routed_plan.selected_ids(),
            "the two plans must agree on every selected ID"
        );
        assert_eq!(
            controller_plan.agreed_action_schema(),
            routed_plan.agreed_action_schema()
        );
        assert_eq!(
            controller_plan.agreed_fragment_schema(),
            routed_plan.agreed_fragment_schema()
        );
        assert_ne!(
            plan_digest(&controller_plan),
            plan_digest(&routed_plan),
            "session placement must be covered by the plan digest"
        );
    }

    /// Independent named-MessagePack authoring of one propagation document,
    /// optionally carrying one field the strict DTO does not declare.
    ///
    /// Authoring the wire document here rather than mutating encoded bytes is
    /// what lets the same fixture prove both halves of the claim: without
    /// `credential` it must decode and equal the sealed value, and with it the
    /// decode must fail. If [`StreamingCapabilityPropagation`] ever gains a
    /// field, the first half fails loudly instead of the second half passing
    /// for the wrong reason.
    #[derive(Serialize)]
    struct AuthoredDocument<'a> {
        version: u16,
        source: &'a str,
        format: &'a str,
        session: &'a str,
        action_sink: &'a str,
        transport: &'a str,
        endpoint: Option<&'a str>,
        checkpoint_backend: Option<&'a str>,
        source_digest: [u8; 32],
        format_digest: [u8; 32],
        session_digest: [u8; 32],
        action_sink_digest: [u8; 32],
        transport_digest: [u8; 32],
        endpoint_digest: Option<[u8; 32]>,
        checkpoint_backend_digest: Option<[u8; 32]>,
        agreed_action_schema: &'a str,
        agreed_fragment_schema: &'a str,
        plan_digest: [u8; 32],
        #[serde(skip_serializing_if = "Option::is_none")]
        credential: Option<&'a str>,
    }

    impl<'a> AuthoredDocument<'a> {
        fn mirroring(sealed: &'a StreamingCapabilityPropagation) -> Self {
            Self {
                version: sealed.version,
                source: &sealed.source,
                format: &sealed.format,
                session: &sealed.session,
                action_sink: &sealed.action_sink,
                transport: &sealed.transport,
                endpoint: sealed.endpoint.as_deref(),
                checkpoint_backend: sealed.checkpoint_backend.as_deref(),
                source_digest: sealed.source_digest,
                format_digest: sealed.format_digest,
                session_digest: sealed.session_digest,
                action_sink_digest: sealed.action_sink_digest,
                transport_digest: sealed.transport_digest,
                endpoint_digest: sealed.endpoint_digest,
                checkpoint_backend_digest: sealed.checkpoint_backend_digest,
                agreed_action_schema: &sealed.agreed_action_schema,
                agreed_fragment_schema: &sealed.agreed_fragment_schema,
                plan_digest: sealed.plan_digest,
                credential: None,
            }
        }

        fn encoded(&self) -> Vec<u8> {
            rmp_serde::to_vec_named(self).expect("an owned document encodes")
        }
    }

    #[test]
    fn a_sealed_propagation_round_trips_and_rejects_unknown_fields() {
        let sealed = StreamingCapabilityPropagation::seal(selection(&SESSION_CONTROLLER))
            .expect("the fixture selection seals");
        let bytes = sealed.encode().expect("an owned DTO encodes");
        assert_eq!(
            StreamingCapabilityPropagation::decode(&bytes).expect("canonical bytes decode"),
            sealed
        );

        let mut authored = AuthoredDocument::mirroring(&sealed);
        assert_eq!(
            StreamingCapabilityPropagation::decode(&authored.encoded())
                .expect("the authored mirror is the complete declared field set"),
            sealed,
            "the mirror must cover exactly the strict DTO's fields"
        );

        authored.credential = Some("must-not-be-accepted");
        assert_eq!(
            StreamingCapabilityPropagation::decode(&authored.encoded()),
            Err(StreamingCapabilityNegotiationError::Malformed),
            "an unknown field must be refused, not ignored"
        );
    }

    #[test]
    fn an_unsupported_version_is_refused_before_any_registry_lookup() {
        let sealed = StreamingCapabilityPropagation::seal(selection(&SESSION_CONTROLLER))
            .expect("the fixture selection seals");
        let bumped = StreamingCapabilityPropagation {
            version: STREAMING_CAPABILITY_PROPAGATION_VERSION + 1,
            ..sealed
        };
        let bytes = bumped.encode().expect("an owned DTO encodes");
        assert_eq!(
            StreamingCapabilityPropagation::decode(&bytes),
            Err(StreamingCapabilityNegotiationError::UnsupportedVersion {
                expected: STREAMING_CAPABILITY_PROPAGATION_VERSION,
                received: STREAMING_CAPABILITY_PROPAGATION_VERSION + 1,
            })
        );
    }
}
