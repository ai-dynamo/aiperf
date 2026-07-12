// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frozen telemetry-archive component registries and preparation seams.
//!
//! Every authored archive component is selected exactly once through its own
//! immutable registry. Validation returns factory-owned erased values, so
//! execution never recovers an implementation through a string branch or
//! `Any`. Collection and source-free synchronization deliberately produce
//! different bundles: synchronization cannot accidentally prepare a writer,
//! rotation policy, admission policy, enricher, sanitizer, or raw-body policy.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fmt::{self, Debug, Display, Formatter};
use std::sync::Arc;

use aiperf_telemetry_archive::sync::WriterClaimId;
use aiperf_telemetry_archive::{
    ArchiveAdmissionMode, ArchiveAdmissionPolicy, ArchiveId, ArchiveKeyProvider,
    ArchiveObjectStore, ArchiveRecoveryPolicy, ArchiveSampleView, ArchiveSanitizer,
    ArchiveSchemasV1, ArchiveSink, ArchiveStoreError, ArchiveWalFrameDecoder,
    AttachedBestEffortAdmissionPolicy, Blake3ArchiveKeyProvider, BoundedSegmentRotationPolicy,
    CanonicalJsonValue, CreateNewRecoveryPolicy, Digest, DurabilityFaultInjector,
    ExactResumeRecoveryPolicy, FileArchiveObjectStore, LocalArchiveRepository, NoopEnricher,
    NoopSanitizer, OwnedLocalArchiveSinkFactory, OwnedReceiptJournalMode, ParquetRotationConfigV1,
    PrimaryWatchAdmissionPolicy, SanitizationError, SanitizedSample, SegmentRotationPolicy,
    StaticLabelEnricher, TelemetryEnricher, WalSegmentHeaderV1, domain_digest,
};
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use url::Url;
use uuid::Uuid;

use crate::telemetry_watch::{
    NormalizedArchiveUri, TelemetryArchiveSpecV2, TelemetryArchiveSyncSpecV2,
};

const WRITER_FAMILY: &str = "writer";
const STORE_ACCESS_FAMILY: &str = "store_access";
const ROTATION_FAMILY: &str = "rotation";
const ADMISSION_FAMILY: &str = "admission";
const RECOVERY_FAMILY: &str = "recovery";
const ARCHIVE_KEY_FAMILY: &str = "archive_key";
const ENRICHER_FAMILY: &str = "enricher";
const SANITIZER_FAMILY: &str = "sanitizer";
const RAW_BODY_FAMILY: &str = "raw_body";

/// Stable capability facts shared by every archive-component factory.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArchiveComponentDescriptor {
    /// Frozen wire ID.
    pub id: &'static str,
    /// Human-readable implementation summary.
    pub description: &'static str,
}

/// Canonical, secret-free identity emitted by one strict component factory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidatedArchiveComponentIdentity {
    /// Component family whose registry selected this value.
    pub family: &'static str,
    /// Frozen factory wire ID.
    pub factory_id: &'static str,
    /// Factory-produced effective configuration with defaults made explicit.
    pub canonical_config: CanonicalJsonValue,
    /// Domain-separated digest of the complete canonical descriptor.
    pub digest: Digest,
}

impl ValidatedArchiveComponentIdentity {
    fn new(
        family: &'static str,
        factory_id: &'static str,
        canonical_config: CanonicalJsonValue,
    ) -> Result<Self, ArchiveComponentError> {
        let descriptor = CanonicalJsonValue::object([
            ("config".to_owned(), canonical_config.clone()),
            (
                "family".to_owned(),
                CanonicalJsonValue::String(family.to_owned()),
            ),
            (
                "type".to_owned(),
                CanonicalJsonValue::String(factory_id.to_owned()),
            ),
        ])
        .map_err(|error| ArchiveComponentError::Canonical(error.to_string()))?;
        let digest = domain_digest(
            "aiperf.archive.component-identity.v1",
            &[descriptor.to_bytes().as_slice()],
        );
        Ok(Self {
            family,
            factory_id,
            canonical_config,
            digest,
        })
    }

    /// Returns the complete canonical family/type/config descriptor.
    pub fn canonical_descriptor(&self) -> CanonicalJsonValue {
        CanonicalJsonValue::object([
            ("config".to_owned(), self.canonical_config.clone()),
            (
                "family".to_owned(),
                CanonicalJsonValue::String(self.family.to_owned()),
            ),
            (
                "type".to_owned(),
                CanonicalJsonValue::String(self.factory_id.to_owned()),
            ),
        ])
        .expect("validated component descriptor fields are unique")
    }
}

/// Product placement in which a persistent archive is being collected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArchiveCollectionPlacement {
    /// The archive is the standalone watch product.
    StandalonePrimary,
    /// The archive is a best-effort benchmark attachment.
    AttachedBestEffort,
}

impl ArchiveCollectionPlacement {
    const fn expected_admission(self) -> ArchiveAdmissionMode {
        match self {
            Self::StandalonePrimary => ArchiveAdmissionMode::PrimaryWatch,
            Self::AttachedBestEffort => ArchiveAdmissionMode::AttachedBestEffort,
        }
    }
}

/// Protocol-level recovery operation implemented by a selected policy factory.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArchiveRecoveryOperation {
    /// Create generation zero and reject existing authority.
    CreateNew,
    /// Verify persistent identity and resume the exact existing authority.
    ExactResume,
    /// Reconcile and terminally publish without activating sources.
    FinalizeRemote,
}

/// Identity values bound only after exact local authority is discovered.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArchiveRecoveryExpectation {
    /// Existing or newly generated archive identity.
    pub archive_id: ArchiveId,
    /// Fully assembled persistent collection identity digest.
    pub persistent_identity_digest: Digest,
    /// Credential-free normalized archive target digest discovered from genesis.
    pub archive_target_digest: Digest,
}

/// Provider-owned, already validated object-store preparation request.
#[derive(Clone, Copy, Debug)]
pub struct ArchiveObjectStorePrepareRequest<'a> {
    /// Store-access factory that validated this request.
    pub access_factory_id: &'static str,
    /// Credential-free normalized target.
    pub target: &'a NormalizedArchiveUri,
    /// Factory-produced canonical effective access configuration.
    pub canonical_config: &'a CanonicalJsonValue,
    /// Optional provider-held credential reference.
    pub credential_provider: Option<&'a str>,
}

/// Injected bridge from validated access selectors to a provider SDK adapter.
pub trait ArchiveObjectStoreProvider: Debug + Send + Sync {
    /// Prepares one narrow archive-store handle without exposing SDK types.
    fn prepare(
        &self,
        request: ArchiveObjectStorePrepareRequest<'_>,
    ) -> Result<Arc<dyn ArchiveObjectStore>, ArchiveComponentError>;
}

/// Injected bridge from a secret reference to process-local archive key material.
pub trait ArchiveKeyProviderResolver: Debug + Send + Sync {
    /// Resolves one provider-held reference into the archive key seam.
    fn resolve(
        &self,
        secret_reference: &str,
    ) -> Result<Arc<dyn ArchiveKeyProvider>, ArchiveComponentError>;
}

/// Stock local provider for `file://` archive targets.
///
/// Object-store schemes intentionally fail with an injection diagnostic: the
/// base runner has no cloud SDK or ambient-credential fallback. Deployments
/// install a provider implementation through the runner execution factories.
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeArchiveObjectStoreProvider;

impl ArchiveObjectStoreProvider for NativeArchiveObjectStoreProvider {
    fn prepare(
        &self,
        request: ArchiveObjectStorePrepareRequest<'_>,
    ) -> Result<Arc<dyn ArchiveObjectStore>, ArchiveComponentError> {
        if request.target.scheme() != "file" {
            return Err(ArchiveComponentError::Prepare(format!(
                "archive target scheme {:?} requires an injected object-store provider; the base runner never resolves ambient cloud credentials",
                request.target.scheme()
            )));
        }
        let parsed = Url::parse(request.target.as_str()).map_err(|error| {
            ArchiveComponentError::Prepare(format!(
                "validated file archive target could not be parsed: {error}"
            ))
        })?;
        let path = parsed.to_file_path().map_err(|()| {
            ArchiveComponentError::Prepare(
                "validated file archive target could not be converted to a local path".to_owned(),
            )
        })?;
        FileArchiveObjectStore::open(path)
            .map(|store| Arc::new(store) as Arc<dyn ArchiveObjectStore>)
            .map_err(|error| ArchiveComponentError::Prepare(error.to_string()))
    }
}

/// Environment-backed provider-held archive key resolver for the base runner.
///
/// A reference such as `archive-identity` maps to
/// `AIPERF_ARCHIVE_KEY_ARCHIVE_IDENTITY`. Values are exactly 32 bytes encoded
/// as 64 hexadecimal digits, `hex:<digits>`, or `base64:<RFC4648>`. Errors name
/// only the public variable/reference and never include secret material.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EnvironmentArchiveKeyProviderResolver {
    prefix: String,
}

impl Default for EnvironmentArchiveKeyProviderResolver {
    fn default() -> Self {
        Self {
            prefix: "AIPERF_ARCHIVE_KEY_".to_owned(),
        }
    }
}

impl EnvironmentArchiveKeyProviderResolver {
    /// Uses an explicit public environment-variable prefix.
    pub fn new(prefix: impl Into<String>) -> Result<Self, ArchiveComponentError> {
        let prefix = prefix.into();
        if prefix.is_empty()
            || prefix.chars().any(|character| {
                !(character.is_ascii_uppercase() || character.is_ascii_digit() || character == '_')
            })
        {
            return Err(ArchiveComponentError::InvalidArchive(
                "archive-key environment prefix must contain only uppercase ASCII letters, digits, and underscores"
                    .to_owned(),
            ));
        }
        Ok(Self { prefix })
    }

    /// Returns the public environment variable derived from a provider reference.
    pub fn variable_name(&self, reference: &str) -> Result<String, ArchiveComponentError> {
        validate_secret_reference(reference)?;
        let suffix = reference
            .bytes()
            .map(|byte| {
                if byte.is_ascii_alphanumeric() {
                    char::from(byte.to_ascii_uppercase())
                } else {
                    '_'
                }
            })
            .collect::<String>();
        Ok(format!("{}{suffix}", self.prefix))
    }
}

impl ArchiveKeyProviderResolver for EnvironmentArchiveKeyProviderResolver {
    fn resolve(
        &self,
        secret_reference: &str,
    ) -> Result<Arc<dyn ArchiveKeyProvider>, ArchiveComponentError> {
        let variable = self.variable_name(secret_reference)?;
        let value = env::var(&variable).map_err(|_| {
            ArchiveComponentError::Prepare(format!(
                "archive key environment variable {variable} is missing or is not UTF-8"
            ))
        })?;
        let key = decode_archive_master_key(&value).ok_or_else(|| {
            ArchiveComponentError::Prepare(format!(
                "archive key environment variable {variable} must encode exactly 32 bytes as hex or base64"
            ))
        })?;
        Blake3ArchiveKeyProvider::new(SECRET_PROVIDER_KEY_DESCRIPTOR.id, key)
            .map(|provider| Arc::new(provider) as Arc<dyn ArchiveKeyProvider>)
            .map_err(|error| {
                ArchiveComponentError::Prepare(format!(
                    "archive key provider for reference {secret_reference:?} could not be prepared: {error}"
                ))
            })
    }
}

/// Already prepared invocation-only store-access handle.
#[derive(Clone)]
pub struct PreparedArchiveStoreAccess {
    /// Invocation identity excluded from persistent writer identity.
    pub identity: ValidatedArchiveComponentIdentity,
    /// Capability-proved store handle.
    pub store: Arc<dyn ArchiveObjectStore>,
}

impl Debug for PreparedArchiveStoreAccess {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedArchiveStoreAccess")
            .field("identity", &self.identity)
            .field("capabilities", &self.store.capabilities())
            .finish_non_exhaustive()
    }
}

/// Prepared whole-frame and Parquet partition rotation policies.
pub struct PreparedArchiveRotation {
    /// Physical Parquet partition bounds.
    pub parquet: ParquetRotationConfigV1,
    /// Clock-aware WAL/segment rotation policy.
    pub segment: Box<dyn SegmentRotationPolicy>,
}

impl Debug for PreparedArchiveRotation {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedArchiveRotation")
            .field("parquet", &self.parquet)
            .field("segment", &self.segment)
            .finish()
    }
}

/// Prepared writer seam retained until repository/WAL authority is available.
pub trait PreparedArchiveWriter: Debug + Send + Sync {
    /// Exact checked-in schemas owned by this writer compatibility version.
    fn schemas(&self) -> &ArchiveSchemasV1;

    /// Canonical persistent writer identity stored in genesis.
    fn persistent_writer_identity(&self) -> &CanonicalJsonValue;

    /// Frozen compatibility digest stored in genesis and every WAL header.
    fn writer_compatibility_id(&self) -> Digest;

    /// Creates a new owned sink after generation zero and the session header exist.
    fn prepare_new_sink(
        &self,
        repository: LocalArchiveRepository,
        header: WalSegmentHeaderV1,
        receipts: OwnedReceiptJournalMode,
    ) -> Result<Box<dyn ArchiveSink>, ArchiveComponentError>;

    /// Recovers the exact open WAL before allowing a resumed source session.
    fn prepare_resumed_sink(
        &self,
        repository: LocalArchiveRepository,
        new_session_id: aiperf_telemetry_archive::SessionId,
        maximum_frame_bytes: u64,
        receipts: OwnedReceiptJournalMode,
        decoder: &dyn ArchiveWalFrameDecoder,
    ) -> Result<Box<dyn ArchiveSink>, ArchiveComponentError>;
}

/// Prepared raw-body policy selected independently from structured sanitization.
pub trait PreparedRawBodyPolicy: Debug + Send + Sync {
    /// Whether exact encoded response bodies may enter archive projection.
    fn retains_exact_body(&self) -> bool;
}

/// Writer-owned strict validation result.
pub trait ValidatedArchiveWriterComponent: Debug + Send + Sync {
    /// Canonical component identity.
    fn identity(&self) -> &ValidatedArchiveComponentIdentity;

    /// Canonical persistent writer layout identity.
    fn persistent_writer_identity(&self) -> &CanonicalJsonValue;

    /// Frozen writer compatibility digest.
    fn writer_compatibility_id(&self) -> Digest;

    /// Prepares a writer against the separately selected rotation policy.
    fn prepare(
        self: Box<Self>,
        rotation: ParquetRotationConfigV1,
        faults: Arc<dyn DurabilityFaultInjector>,
    ) -> Result<Box<dyn PreparedArchiveWriter>, ArchiveComponentError>;
}

/// Invocation-store-owned strict validation result.
pub trait ValidatedArchiveStoreAccessComponent: Debug + Send + Sync {
    /// Canonical invocation identity.
    fn identity(&self) -> &ValidatedArchiveComponentIdentity;

    /// Rejects a target scheme this access adapter cannot implement.
    fn validate_target(&self, target: &NormalizedArchiveUri) -> Result<(), ArchiveComponentError>;

    /// Resolves provider credentials and prepares the narrow store handle.
    fn prepare(
        self: Box<Self>,
        target: &NormalizedArchiveUri,
        provider: &dyn ArchiveObjectStoreProvider,
    ) -> Result<PreparedArchiveStoreAccess, ArchiveComponentError>;
}

/// Rotation-owned strict validation result.
pub trait ValidatedArchiveRotationComponent: Debug + Send + Sync {
    /// Canonical persistent policy identity.
    fn identity(&self) -> &ValidatedArchiveComponentIdentity;

    /// Prepares both physical and Clock-age rotation decisions.
    fn prepare(self: Box<Self>) -> Result<PreparedArchiveRotation, ArchiveComponentError>;
}

/// Admission-owned strict validation result.
pub trait ValidatedArchiveAdmissionComponent: Debug + Send + Sync {
    /// Canonical persistent policy identity.
    fn identity(&self) -> &ValidatedArchiveComponentIdentity;

    /// Product semantics implemented by this policy.
    fn mode(&self) -> ArchiveAdmissionMode;

    /// Prepares the nonblocking ingress policy.
    fn prepare(self: Box<Self>) -> Arc<dyn ArchiveAdmissionPolicy>;
}

/// Recovery-owned strict validation result.
pub trait ValidatedArchiveRecoveryComponent: Debug + Send + Sync {
    /// Canonical invocation policy identity.
    fn identity(&self) -> &ValidatedArchiveComponentIdentity;

    /// Protocol operation implemented by this policy.
    fn operation(&self) -> ArchiveRecoveryOperation;

    /// Authored archive identity required by exact collect-resume.
    fn expected_archive_id(&self) -> Option<ArchiveId>;

    /// Explicit crashed writer claim required by exact remote takeover.
    fn expected_prior_claim_id(&self) -> Option<WriterClaimId>;

    /// Binds collect recovery after authoritative identity discovery.
    fn bind_collect(
        self: Box<Self>,
        expectation: ArchiveRecoveryExpectation,
    ) -> Result<Box<dyn ArchiveRecoveryPolicy>, ArchiveComponentError>;

    /// Verifies that this selection is source-free remote finalization.
    fn bind_finalize_remote(self: Box<Self>) -> Result<(), ArchiveComponentError>;
}

/// Archive-key-factory-owned strict validation result.
pub trait ValidatedArchiveKeyComponent: Debug + Send + Sync {
    /// Canonical persistent provider identity.
    fn identity(&self) -> &ValidatedArchiveComponentIdentity;

    /// Resolves process-local key material through the injected provider.
    fn prepare(
        self: Box<Self>,
        resolver: &dyn ArchiveKeyProviderResolver,
    ) -> Result<Arc<dyn ArchiveKeyProvider>, ArchiveComponentError>;
}

/// Enricher-factory-owned strict validation result.
pub trait ValidatedTelemetryEnricherComponent: Debug + Send + Sync {
    /// Canonical persistent policy identity.
    fn identity(&self) -> &ValidatedArchiveComponentIdentity;

    /// Prepares one immutable additive enricher.
    fn prepare(self: Box<Self>) -> Result<Arc<dyn TelemetryEnricher>, ArchiveComponentError>;
}

/// Sanitizer-factory-owned strict validation result.
pub trait ValidatedArchiveSanitizerComponent: Debug + Send + Sync {
    /// Canonical persistent policy identity.
    fn identity(&self) -> &ValidatedArchiveComponentIdentity;

    /// Prepares one optional structured policy after the mandatory baseline.
    fn prepare(self: Box<Self>) -> Result<Arc<dyn ArchiveSanitizer>, ArchiveComponentError>;
}

/// Raw-body-factory-owned strict validation result.
pub trait ValidatedRawBodyComponent: Debug + Send + Sync {
    /// Canonical persistent policy identity.
    fn identity(&self) -> &ValidatedArchiveComponentIdentity;

    /// Prepares one exact-body retention policy.
    fn prepare(self: Box<Self>) -> Result<Box<dyn PreparedRawBodyPolicy>, ArchiveComponentError>;
}

/// Strict writer component factory.
pub trait ArchiveWriterComponentFactory: Debug + Send + Sync {
    /// Frozen capability descriptor.
    fn descriptor(&self) -> &'static ArchiveComponentDescriptor;

    /// Strictly validates this writer's authored object.
    fn validate(
        &self,
        config: &RawValue,
    ) -> Result<Box<dyn ValidatedArchiveWriterComponent>, ArchiveComponentError>;
}

/// Strict invocation store-access component factory.
pub trait ArchiveStoreAccessComponentFactory: Debug + Send + Sync {
    /// Frozen capability descriptor.
    fn descriptor(&self) -> &'static ArchiveComponentDescriptor;

    /// Strictly validates this access adapter's authored object.
    fn validate(
        &self,
        config: &RawValue,
    ) -> Result<Box<dyn ValidatedArchiveStoreAccessComponent>, ArchiveComponentError>;
}

/// Strict segment/partition rotation component factory.
pub trait ArchiveRotationComponentFactory: Debug + Send + Sync {
    /// Frozen capability descriptor.
    fn descriptor(&self) -> &'static ArchiveComponentDescriptor;

    /// Strictly validates this rotation policy's authored object.
    fn validate(
        &self,
        config: &RawValue,
    ) -> Result<Box<dyn ValidatedArchiveRotationComponent>, ArchiveComponentError>;
}

/// Strict ingress admission component factory.
pub trait ArchiveAdmissionComponentFactory: Debug + Send + Sync {
    /// Frozen capability descriptor.
    fn descriptor(&self) -> &'static ArchiveComponentDescriptor;

    /// Strictly validates this admission policy's authored object.
    fn validate(
        &self,
        config: &RawValue,
    ) -> Result<Box<dyn ValidatedArchiveAdmissionComponent>, ArchiveComponentError>;
}

/// Strict recovery component factory.
pub trait ArchiveRecoveryComponentFactory: Debug + Send + Sync {
    /// Frozen capability descriptor.
    fn descriptor(&self) -> &'static ArchiveComponentDescriptor;

    /// Strictly validates this recovery policy's authored object.
    fn validate(
        &self,
        config: &RawValue,
    ) -> Result<Box<dyn ValidatedArchiveRecoveryComponent>, ArchiveComponentError>;
}

/// Strict archive-key provider component factory.
pub trait ArchiveKeyComponentFactory: Debug + Send + Sync {
    /// Frozen capability descriptor.
    fn descriptor(&self) -> &'static ArchiveComponentDescriptor;

    /// Strictly validates this secret-provider selector.
    fn validate(
        &self,
        config: &RawValue,
    ) -> Result<Box<dyn ValidatedArchiveKeyComponent>, ArchiveComponentError>;
}

/// Strict additive enrichment component factory.
pub trait TelemetryEnricherComponentFactory: Debug + Send + Sync {
    /// Frozen capability descriptor.
    fn descriptor(&self) -> &'static ArchiveComponentDescriptor;

    /// Strictly validates this enricher's authored object.
    fn validate(
        &self,
        config: &RawValue,
    ) -> Result<Box<dyn ValidatedTelemetryEnricherComponent>, ArchiveComponentError>;
}

/// Strict structured sanitizer component factory.
pub trait ArchiveSanitizerComponentFactory: Debug + Send + Sync {
    /// Frozen capability descriptor.
    fn descriptor(&self) -> &'static ArchiveComponentDescriptor;

    /// Strictly validates this sanitizer's authored object.
    fn validate(
        &self,
        config: &RawValue,
    ) -> Result<Box<dyn ValidatedArchiveSanitizerComponent>, ArchiveComponentError>;
}

/// Strict exact raw-body retention component factory.
pub trait RawBodyComponentFactory: Debug + Send + Sync {
    /// Frozen capability descriptor.
    fn descriptor(&self) -> &'static ArchiveComponentDescriptor;

    /// Strictly validates this retention policy's authored object.
    fn validate(
        &self,
        config: &RawValue,
    ) -> Result<Box<dyn ValidatedRawBodyComponent>, ArchiveComponentError>;
}

macro_rules! define_component_registry {
    ($name:ident, $factory:ident, $validated:ident, $family:expr, $doc:literal) => {
        #[doc = $doc]
        #[derive(Clone)]
        pub struct $name {
            factories: Arc<BTreeMap<String, Arc<dyn $factory>>>,
        }

        impl $name {
            /// Freezes unique, syntactically valid factory IDs.
            pub fn new(
                factories: impl IntoIterator<Item = Arc<dyn $factory>>,
            ) -> Result<Self, ArchiveComponentError> {
                let mut by_id = BTreeMap::new();
                for factory in factories {
                    let id = factory.descriptor().id;
                    validate_factory_id(id, $family)?;
                    if by_id.insert(id.to_owned(), factory).is_some() {
                        return Err(ArchiveComponentError::DuplicateFactory {
                            family: $family,
                            id: id.to_owned(),
                        });
                    }
                }
                Ok(Self {
                    factories: Arc::new(by_id),
                })
            }

            /// Resolves and strictly validates one authored component.
            pub fn validate(
                &self,
                id: &str,
                config: &RawValue,
            ) -> Result<Box<dyn $validated>, ArchiveComponentError> {
                let factory = self.factories.get(id).ok_or_else(|| {
                    ArchiveComponentError::UnknownFactory {
                        family: $family,
                        requested: id.to_owned(),
                        available: self.factories.keys().cloned().collect(),
                    }
                })?;
                factory.validate(config)
            }

            /// Returns deterministic descriptors for distribution capabilities.
            pub fn descriptors(
                &self,
            ) -> impl ExactSizeIterator<Item = &'static ArchiveComponentDescriptor> {
                self.factories.values().map(|factory| factory.descriptor())
            }
        }

        impl Debug for $name {
            fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
                formatter
                    .debug_struct(stringify!($name))
                    .field("ids", &self.factories.keys().collect::<Vec<_>>())
                    .finish()
            }
        }
    };
}

define_component_registry!(
    ArchiveWriterFactoryRegistry,
    ArchiveWriterComponentFactory,
    ValidatedArchiveWriterComponent,
    WRITER_FAMILY,
    "Immutable archive-writer factory registry."
);
define_component_registry!(
    ArchiveStoreAccessFactoryRegistry,
    ArchiveStoreAccessComponentFactory,
    ValidatedArchiveStoreAccessComponent,
    STORE_ACCESS_FAMILY,
    "Immutable invocation store-access factory registry."
);
define_component_registry!(
    ArchiveRotationFactoryRegistry,
    ArchiveRotationComponentFactory,
    ValidatedArchiveRotationComponent,
    ROTATION_FAMILY,
    "Immutable rotation-policy factory registry."
);
define_component_registry!(
    ArchiveAdmissionFactoryRegistry,
    ArchiveAdmissionComponentFactory,
    ValidatedArchiveAdmissionComponent,
    ADMISSION_FAMILY,
    "Immutable admission-policy factory registry."
);
define_component_registry!(
    ArchiveRecoveryFactoryRegistry,
    ArchiveRecoveryComponentFactory,
    ValidatedArchiveRecoveryComponent,
    RECOVERY_FAMILY,
    "Immutable recovery-policy factory registry."
);
define_component_registry!(
    ArchiveKeyFactoryRegistry,
    ArchiveKeyComponentFactory,
    ValidatedArchiveKeyComponent,
    ARCHIVE_KEY_FAMILY,
    "Immutable archive-key provider factory registry."
);
define_component_registry!(
    TelemetryEnricherFactoryRegistry,
    TelemetryEnricherComponentFactory,
    ValidatedTelemetryEnricherComponent,
    ENRICHER_FAMILY,
    "Immutable telemetry-enricher factory registry."
);
define_component_registry!(
    ArchiveSanitizerFactoryRegistry,
    ArchiveSanitizerComponentFactory,
    ValidatedArchiveSanitizerComponent,
    SANITIZER_FAMILY,
    "Immutable structured-sanitizer factory registry."
);
define_component_registry!(
    RawBodyFactoryRegistry,
    RawBodyComponentFactory,
    ValidatedRawBodyComponent,
    RAW_BODY_FAMILY,
    "Immutable exact raw-body retention factory registry."
);

/// Extensible factory inputs frozen into one runner distribution.
pub struct TelemetryArchiveComponentFactories {
    /// Writer implementations.
    pub writers: Vec<Arc<dyn ArchiveWriterComponentFactory>>,
    /// Store-access implementations.
    pub store_access: Vec<Arc<dyn ArchiveStoreAccessComponentFactory>>,
    /// Rotation implementations.
    pub rotations: Vec<Arc<dyn ArchiveRotationComponentFactory>>,
    /// Admission implementations.
    pub admissions: Vec<Arc<dyn ArchiveAdmissionComponentFactory>>,
    /// Recovery implementations.
    pub recoveries: Vec<Arc<dyn ArchiveRecoveryComponentFactory>>,
    /// Archive-key implementations.
    pub archive_keys: Vec<Arc<dyn ArchiveKeyComponentFactory>>,
    /// Enrichment implementations.
    pub enrichers: Vec<Arc<dyn TelemetryEnricherComponentFactory>>,
    /// Optional sanitization implementations.
    pub sanitizers: Vec<Arc<dyn ArchiveSanitizerComponentFactory>>,
    /// Raw-body implementations.
    pub raw_bodies: Vec<Arc<dyn RawBodyComponentFactory>>,
}

impl Debug for TelemetryArchiveComponentFactories {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TelemetryArchiveComponentFactories")
            .field("writer_count", &self.writers.len())
            .field("store_access_count", &self.store_access.len())
            .field("rotation_count", &self.rotations.len())
            .field("admission_count", &self.admissions.len())
            .field("recovery_count", &self.recoveries.len())
            .field("archive_key_count", &self.archive_keys.len())
            .field("enricher_count", &self.enrichers.len())
            .field("sanitizer_count", &self.sanitizers.len())
            .field("raw_body_count", &self.raw_bodies.len())
            .finish()
    }
}

/// One exact immutable archive-component universe.
#[derive(Clone, Debug)]
pub struct TelemetryArchiveComponentRegistries {
    /// Writer registry.
    pub writers: ArchiveWriterFactoryRegistry,
    /// Invocation store-access registry.
    pub store_access: ArchiveStoreAccessFactoryRegistry,
    /// Rotation registry.
    pub rotations: ArchiveRotationFactoryRegistry,
    /// Admission registry.
    pub admissions: ArchiveAdmissionFactoryRegistry,
    /// Recovery registry.
    pub recoveries: ArchiveRecoveryFactoryRegistry,
    /// Archive-key registry.
    pub archive_keys: ArchiveKeyFactoryRegistry,
    /// Enrichment registry.
    pub enrichers: TelemetryEnricherFactoryRegistry,
    /// Optional sanitizer registry.
    pub sanitizers: ArchiveSanitizerFactoryRegistry,
    /// Raw-body registry.
    pub raw_bodies: RawBodyFactoryRegistry,
}

impl TelemetryArchiveComponentRegistries {
    /// Freezes one explicitly composed implementation universe.
    pub fn new(
        factories: TelemetryArchiveComponentFactories,
    ) -> Result<Self, ArchiveComponentError> {
        Ok(Self {
            writers: ArchiveWriterFactoryRegistry::new(factories.writers)?,
            store_access: ArchiveStoreAccessFactoryRegistry::new(factories.store_access)?,
            rotations: ArchiveRotationFactoryRegistry::new(factories.rotations)?,
            admissions: ArchiveAdmissionFactoryRegistry::new(factories.admissions)?,
            recoveries: ArchiveRecoveryFactoryRegistry::new(factories.recoveries)?,
            archive_keys: ArchiveKeyFactoryRegistry::new(factories.archive_keys)?,
            enrichers: TelemetryEnricherFactoryRegistry::new(factories.enrichers)?,
            sanitizers: ArchiveSanitizerFactoryRegistry::new(factories.sanitizers)?,
            raw_bodies: RawBodyFactoryRegistry::new(factories.raw_bodies)?,
        })
    }

    /// Stock component universe compiled into the base runner.
    pub fn stock() -> Self {
        Self::new(TelemetryArchiveComponentFactories {
            writers: vec![Arc::new(ParquetArchiveWriterFactory)],
            store_access: vec![
                Arc::new(LocalFilesystemStoreAccessFactory),
                Arc::new(ObjectStoreAccessFactory),
            ],
            rotations: vec![Arc::new(RowsBytesAgeRotationFactory)],
            admissions: vec![
                Arc::new(PrimaryDurableAdmissionFactory),
                Arc::new(AttachedBestEffortAdmissionFactory),
            ],
            recoveries: vec![
                Arc::new(CreateNewRecoveryFactory),
                Arc::new(ExactResumeRecoveryFactory),
                Arc::new(FinalizeRemoteRecoveryFactory),
            ],
            archive_keys: vec![Arc::new(SecretProviderArchiveKeyFactory)],
            enrichers: vec![
                Arc::new(NoopEnricherFactory),
                Arc::new(StaticLabelsEnricherFactory),
            ],
            sanitizers: vec![
                Arc::new(NoopSanitizerFactory),
                Arc::new(AllowDenySanitizerFactory),
            ],
            raw_bodies: vec![Arc::new(NoRawBodyFactory)],
        })
        .expect("stock telemetry archive component IDs are valid and unique")
    }

    /// Strictly validates every persistent collect/attachment component.
    pub fn validate_collect(
        &self,
        archive: TelemetryArchiveSpecV2,
        placement: ArchiveCollectionPlacement,
    ) -> Result<ValidatedTelemetryArchiveCollectComponents, ArchiveComponentError> {
        archive
            .validate_static()
            .map_err(|error| ArchiveComponentError::InvalidArchive(error.to_string()))?;
        let TelemetryArchiveSpecV2 {
            target,
            local_spool,
            spool_quota_bytes,
            spool_quota_files,
            required,
            writer,
            store_access,
            rotation,
            admission,
            recovery,
            archive_key,
            enrichers,
            sanitizers,
            raw_body,
        } = archive;

        validate_store_spool_separation(&target, &local_spool)?;

        let writer = self.writers.validate(writer.id.as_str(), &writer.config)?;
        let store_access = self
            .store_access
            .validate(store_access.id.as_str(), &store_access.config)?;
        store_access.validate_target(&target)?;
        let rotation = self
            .rotations
            .validate(rotation.id.as_str(), &rotation.config)?;
        let admission = self
            .admissions
            .validate(admission.id.as_str(), &admission.config)?;
        if admission.mode() != placement.expected_admission() {
            return Err(ArchiveComponentError::IncompatibleSelection(format!(
                "archive admission {:?} is incompatible with {placement:?}",
                admission.mode()
            )));
        }
        let recovery = self
            .recoveries
            .validate(recovery.id.as_str(), &recovery.config)?;
        if recovery.operation() == ArchiveRecoveryOperation::FinalizeRemote {
            return Err(ArchiveComponentError::IncompatibleSelection(
                "collect archive cannot select finalize_remote recovery".to_owned(),
            ));
        }
        let archive_key = self
            .archive_keys
            .validate(archive_key.id.as_str(), &archive_key.config)?;
        let enrichers = enrichers
            .into_iter()
            .map(|component| {
                self.enrichers
                    .validate(component.id.as_str(), &component.config)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let sanitizers = sanitizers
            .into_iter()
            .map(|component| {
                self.sanitizers
                    .validate(component.id.as_str(), &component.config)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let raw_body = self
            .raw_bodies
            .validate(raw_body.id.as_str(), &raw_body.config)?;
        let baseline_sanitizer_identity = baseline_sanitizer_identity()?;

        Ok(ValidatedTelemetryArchiveCollectComponents {
            target,
            local_spool,
            spool_quota_bytes,
            spool_quota_files,
            required,
            writer,
            store_access,
            rotation,
            admission,
            recovery,
            archive_key,
            enrichers,
            sanitizers,
            raw_body,
            baseline_sanitizer_identity,
        })
    }

    /// Strictly validates the source-free remote-finalization component subset.
    pub fn validate_sync(
        &self,
        archive: TelemetryArchiveSyncSpecV2,
    ) -> Result<ValidatedTelemetryArchiveSyncComponents, ArchiveComponentError> {
        let TelemetryArchiveSyncSpecV2 {
            archive_id,
            target,
            local_spool,
            store_access,
            recovery,
            archive_key,
        } = archive;
        validate_store_spool_separation(&target, &local_spool)?;
        let store_access = self
            .store_access
            .validate(store_access.id.as_str(), &store_access.config)?;
        store_access.validate_target(&target)?;
        let recovery = self
            .recoveries
            .validate(recovery.id.as_str(), &recovery.config)?;
        if recovery.operation() != ArchiveRecoveryOperation::FinalizeRemote {
            return Err(ArchiveComponentError::IncompatibleSelection(
                "source-free synchronization requires finalize_remote recovery".to_owned(),
            ));
        }
        let archive_key = self
            .archive_keys
            .validate(archive_key.id.as_str(), &archive_key.config)?;
        Ok(ValidatedTelemetryArchiveSyncComponents {
            archive_id,
            target,
            local_spool,
            store_access,
            recovery,
            archive_key,
        })
    }
}

/// Fully validated persistent collect components with no prepared credentials or IO.
#[derive(Debug)]
pub struct ValidatedTelemetryArchiveCollectComponents {
    /// Normalized archive target.
    pub target: NormalizedArchiveUri,
    /// Qualified-spool candidate path.
    pub local_spool: std::path::PathBuf,
    /// Total spool byte quota.
    pub spool_quota_bytes: u64,
    /// Total spool file quota.
    pub spool_quota_files: u64,
    /// Whether archive degradation fails the run outcome.
    pub required: bool,
    writer: Box<dyn ValidatedArchiveWriterComponent>,
    store_access: Box<dyn ValidatedArchiveStoreAccessComponent>,
    rotation: Box<dyn ValidatedArchiveRotationComponent>,
    admission: Box<dyn ValidatedArchiveAdmissionComponent>,
    recovery: Box<dyn ValidatedArchiveRecoveryComponent>,
    archive_key: Box<dyn ValidatedArchiveKeyComponent>,
    enrichers: Vec<Box<dyn ValidatedTelemetryEnricherComponent>>,
    sanitizers: Vec<Box<dyn ValidatedArchiveSanitizerComponent>>,
    raw_body: Box<dyn ValidatedRawBodyComponent>,
    baseline_sanitizer_identity: ValidatedArchiveComponentIdentity,
}

impl ValidatedTelemetryArchiveCollectComponents {
    /// Returns persistent component identities in deterministic pipeline order.
    pub fn persistent_component_identities(&self) -> Vec<ValidatedArchiveComponentIdentity> {
        let mut identities = Vec::with_capacity(7 + self.enrichers.len() + self.sanitizers.len());
        identities.push(self.writer.identity().clone());
        identities.push(self.rotation.identity().clone());
        identities.push(self.admission.identity().clone());
        identities.push(self.archive_key.identity().clone());
        identities.extend(self.enrichers.iter().map(|value| value.identity().clone()));
        identities.push(self.baseline_sanitizer_identity.clone());
        identities.extend(self.sanitizers.iter().map(|value| value.identity().clone()));
        identities.push(self.raw_body.identity().clone());
        identities
    }

    /// Returns invocation-only identities excluded from genesis collection identity.
    pub fn invocation_component_identities(&self) -> [ValidatedArchiveComponentIdentity; 2] {
        [
            self.store_access.identity().clone(),
            self.recovery.identity().clone(),
        ]
    }

    /// Returns the factory-produced writer identity before side effects.
    pub fn persistent_writer_identity(&self) -> &CanonicalJsonValue {
        self.writer.persistent_writer_identity()
    }

    /// Returns the writer compatibility digest before side effects.
    pub fn writer_compatibility_id(&self) -> Digest {
        self.writer.writer_compatibility_id()
    }

    /// Resolves providers and prepares the complete collect policy graph.
    pub fn prepare(
        self,
        context: ArchiveCollectComponentPrepareContext<'_>,
    ) -> Result<PreparedTelemetryArchiveCollectComponents, ArchiveComponentError> {
        let persistent_component_identities = self.persistent_component_identities();
        let invocation_component_identities = self.invocation_component_identities();
        let rotation = self.rotation.prepare()?;
        let writer = self
            .writer
            .prepare(rotation.parquet, context.durability_faults)?;
        let store_access = self
            .store_access
            .prepare(&self.target, context.store_provider)?;
        let archive_key = self.archive_key.prepare(context.key_resolver)?;
        let enrichers = self
            .enrichers
            .into_iter()
            .map(|value| value.prepare())
            .collect::<Result<Vec<_>, _>>()?;
        let optional_sanitizers = self
            .sanitizers
            .into_iter()
            .map(|value| value.prepare())
            .collect::<Result<Vec<_>, _>>()?;
        let sanitizer: Arc<dyn ArchiveSanitizer> =
            Arc::new(ArchiveSanitizerChain::new(optional_sanitizers));

        Ok(PreparedTelemetryArchiveCollectComponents {
            target: self.target,
            local_spool: self.local_spool,
            spool_quota_bytes: self.spool_quota_bytes,
            spool_quota_files: self.spool_quota_files,
            required: self.required,
            writer,
            store_access,
            rotation,
            admission: self.admission.prepare(),
            recovery: PreparedArchiveRecoverySelection {
                component: self.recovery,
            },
            archive_key,
            enrichers,
            sanitizer,
            raw_body: self.raw_body.prepare()?,
            persistent_component_identities,
            invocation_component_identities,
        })
    }
}

/// Provider/fault context used only after branch-complete collect validation.
pub struct ArchiveCollectComponentPrepareContext<'a> {
    /// Injected invocation store adapter provider.
    pub store_provider: &'a dyn ArchiveObjectStoreProvider,
    /// Injected provider-held archive key resolver.
    pub key_resolver: &'a dyn ArchiveKeyProviderResolver,
    /// Durability fault seam, normally the no-fault implementation.
    pub durability_faults: Arc<dyn DurabilityFaultInjector>,
}

impl Debug for ArchiveCollectComponentPrepareContext<'_> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ArchiveCollectComponentPrepareContext")
            .field("store_provider", &self.store_provider)
            .field("key_resolver", &self.key_resolver)
            .field("durability_faults", &self.durability_faults)
            .finish()
    }
}

/// Prepared collect components ready for spool qualification and source startup.
pub struct PreparedTelemetryArchiveCollectComponents {
    /// Normalized archive target.
    pub target: NormalizedArchiveUri,
    /// Qualified-spool candidate path.
    pub local_spool: std::path::PathBuf,
    /// Total spool byte quota.
    pub spool_quota_bytes: u64,
    /// Total spool file quota.
    pub spool_quota_files: u64,
    /// Whether archive degradation fails the run outcome.
    pub required: bool,
    /// Prepared writer compatibility implementation.
    pub writer: Box<dyn PreparedArchiveWriter>,
    /// Prepared invocation store handle.
    pub store_access: PreparedArchiveStoreAccess,
    /// Prepared physical and Clock rotation policies.
    pub rotation: PreparedArchiveRotation,
    /// Prepared ingress policy.
    pub admission: Arc<dyn ArchiveAdmissionPolicy>,
    /// Recovery selection awaiting authoritative identity binding.
    pub recovery: PreparedArchiveRecoverySelection,
    /// Prepared process-local archive key provider.
    pub archive_key: Arc<dyn ArchiveKeyProvider>,
    /// Ordered additive enrichers.
    pub enrichers: Vec<Arc<dyn TelemetryEnricher>>,
    /// Mandatory-baseline plus authored sanitizer chain.
    pub sanitizer: Arc<dyn ArchiveSanitizer>,
    /// Prepared exact-body policy.
    pub raw_body: Box<dyn PreparedRawBodyPolicy>,
    /// Persistent component identities in deterministic pipeline order.
    pub persistent_component_identities: Vec<ValidatedArchiveComponentIdentity>,
    /// Store/recovery invocation identities.
    pub invocation_component_identities: [ValidatedArchiveComponentIdentity; 2],
}

impl Debug for PreparedTelemetryArchiveCollectComponents {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedTelemetryArchiveCollectComponents")
            .field("target", &self.target)
            .field("local_spool", &self.local_spool)
            .field("spool_quota_bytes", &self.spool_quota_bytes)
            .field("spool_quota_files", &self.spool_quota_files)
            .field("required", &self.required)
            .field("writer", &self.writer)
            .field("store_access", &self.store_access)
            .field("rotation", &self.rotation)
            .field("admission", &self.admission)
            .field("recovery", &self.recovery)
            .field("archive_key", &self.archive_key.provider_id())
            .field("enricher_count", &self.enrichers.len())
            .field("sanitizer", &self.sanitizer)
            .field("raw_body", &self.raw_body)
            .finish_non_exhaustive()
    }
}

impl PreparedTelemetryArchiveCollectComponents {
    /// Returns the genesis identity of the credential-free normalized target.
    pub fn archive_target_digest(&self) -> Digest {
        aiperf_telemetry_archive::manifest::archive_target_digest(self.target.as_str())
    }
}

/// Validated source-free synchronization selectors.
#[derive(Debug)]
pub struct ValidatedTelemetryArchiveSyncComponents {
    /// Exact stored archive UUID.
    pub archive_id: Uuid,
    /// Normalized target verified against genesis during preparation.
    pub target: NormalizedArchiveUri,
    /// Existing qualified spool path.
    pub local_spool: std::path::PathBuf,
    store_access: Box<dyn ValidatedArchiveStoreAccessComponent>,
    recovery: Box<dyn ValidatedArchiveRecoveryComponent>,
    archive_key: Box<dyn ValidatedArchiveKeyComponent>,
}

impl ValidatedTelemetryArchiveSyncComponents {
    /// Returns the three invocation identities without any persistent writer fields.
    pub fn invocation_component_identities(&self) -> [ValidatedArchiveComponentIdentity; 3] {
        [
            self.store_access.identity().clone(),
            self.recovery.identity().clone(),
            self.archive_key.identity().clone(),
        ]
    }

    /// Resolves only store/key providers and finalization recovery policy.
    pub fn prepare(
        self,
        context: ArchiveSyncComponentPrepareContext<'_>,
    ) -> Result<PreparedTelemetryArchiveSyncComponents, ArchiveComponentError> {
        let invocation_component_identities = self.invocation_component_identities();
        let store_access = self
            .store_access
            .prepare(&self.target, context.store_provider)?;
        let archive_key = self.archive_key.prepare(context.key_resolver)?;
        let recovery = PreparedArchiveRecoverySelection {
            component: self.recovery,
        };
        recovery.operation_checked(ArchiveRecoveryOperation::FinalizeRemote)?;
        Ok(PreparedTelemetryArchiveSyncComponents {
            archive_id: self.archive_id,
            target: self.target,
            local_spool: self.local_spool,
            store_access,
            recovery,
            archive_key,
            invocation_component_identities,
        })
    }
}

/// Provider context used only by source-free archive synchronization.
#[derive(Clone, Copy)]
pub struct ArchiveSyncComponentPrepareContext<'a> {
    /// Injected invocation store adapter provider.
    pub store_provider: &'a dyn ArchiveObjectStoreProvider,
    /// Injected provider-held archive key resolver.
    pub key_resolver: &'a dyn ArchiveKeyProviderResolver,
}

impl Debug for ArchiveSyncComponentPrepareContext<'_> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ArchiveSyncComponentPrepareContext")
            .field("store_provider", &self.store_provider)
            .field("key_resolver", &self.key_resolver)
            .finish()
    }
}

/// Prepared source-free synchronization subset.
pub struct PreparedTelemetryArchiveSyncComponents {
    /// Exact stored archive UUID.
    pub archive_id: Uuid,
    /// Normalized target.
    pub target: NormalizedArchiveUri,
    /// Existing qualified spool path.
    pub local_spool: std::path::PathBuf,
    /// Prepared invocation store handle.
    pub store_access: PreparedArchiveStoreAccess,
    /// Verified finalization-only recovery selection.
    pub recovery: PreparedArchiveRecoverySelection,
    /// Prepared process-local archive key provider.
    pub archive_key: Arc<dyn ArchiveKeyProvider>,
    /// Store/recovery/key invocation identities.
    pub invocation_component_identities: [ValidatedArchiveComponentIdentity; 3],
}

impl Debug for PreparedTelemetryArchiveSyncComponents {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedTelemetryArchiveSyncComponents")
            .field("archive_id", &self.archive_id)
            .field("target", &self.target)
            .field("local_spool", &self.local_spool)
            .field("store_access", &self.store_access)
            .field("recovery", &self.recovery)
            .field("archive_key", &self.archive_key.provider_id())
            .finish_non_exhaustive()
    }
}

impl PreparedTelemetryArchiveSyncComponents {
    /// Returns the authored target identity required by source-free recovery.
    pub fn archive_target_digest(&self) -> Digest {
        aiperf_telemetry_archive::manifest::archive_target_digest(self.target.as_str())
    }
}

/// Prepared recovery factory retained across authoritative spool discovery.
pub struct PreparedArchiveRecoverySelection {
    component: Box<dyn ValidatedArchiveRecoveryComponent>,
}

impl PreparedArchiveRecoverySelection {
    /// Returns the protocol operation without consulting its wire ID.
    pub fn operation(&self) -> ArchiveRecoveryOperation {
        self.component.operation()
    }

    /// Returns the authored archive identity required by exact collect-resume.
    pub fn expected_archive_id(&self) -> Option<ArchiveId> {
        self.component.expected_archive_id()
    }

    /// Returns the explicitly authored crashed claim required by exact resume.
    pub fn expected_prior_claim_id(&self) -> Option<WriterClaimId> {
        self.component.expected_prior_claim_id()
    }

    fn operation_checked(
        &self,
        expected: ArchiveRecoveryOperation,
    ) -> Result<(), ArchiveComponentError> {
        if self.operation() != expected {
            return Err(ArchiveComponentError::IncompatibleSelection(format!(
                "recovery operation {:?} does not satisfy required {expected:?}",
                self.operation()
            )));
        }
        Ok(())
    }

    /// Binds create-new or exact-resume after persistent identity is known.
    pub fn bind_collect(
        self,
        expectation: ArchiveRecoveryExpectation,
    ) -> Result<Box<dyn ArchiveRecoveryPolicy>, ArchiveComponentError> {
        if self.operation() == ArchiveRecoveryOperation::FinalizeRemote {
            return Err(ArchiveComponentError::IncompatibleSelection(
                "finalize_remote recovery cannot activate collection".to_owned(),
            ));
        }
        self.component.bind_collect(expectation)
    }

    /// Consumes and verifies the source-free finalization selection.
    pub fn bind_finalize_remote(self) -> Result<(), ArchiveComponentError> {
        self.operation_checked(ArchiveRecoveryOperation::FinalizeRemote)?;
        self.component.bind_finalize_remote()
    }
}

impl Debug for PreparedArchiveRecoverySelection {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedArchiveRecoverySelection")
            .field("operation", &self.operation())
            .field("component", &self.component)
            .finish()
    }
}

static PARQUET_WRITER_DESCRIPTOR: ArchiveComponentDescriptor = ArchiveComponentDescriptor {
    id: "parquet_archive_v1",
    description: "schema-v1 WAL-backed immutable Parquet archive writer",
};
static LOCAL_FILESYSTEM_STORE_DESCRIPTOR: ArchiveComponentDescriptor = ArchiveComponentDescriptor {
    id: "local_filesystem",
    description: "provider-adapted local file target with authoritative object semantics",
};
static OBJECT_STORE_DESCRIPTOR: ArchiveComponentDescriptor = ArchiveComponentDescriptor {
    id: "object_store",
    description: "provider-adapted object target with create/verify/head-CAS semantics",
};
static ROWS_BYTES_AGE_ROTATION_DESCRIPTOR: ArchiveComponentDescriptor =
    ArchiveComponentDescriptor {
        id: "rows_bytes_age",
        description: "whole-frame row, byte, and Clock-age rotation",
    };
static PRIMARY_DURABLE_ADMISSION_DESCRIPTOR: ArchiveComponentDescriptor =
    ArchiveComponentDescriptor {
        id: "primary_durable",
        description: "primary watch durable admission",
    };
static ATTACHED_BEST_EFFORT_ADMISSION_DESCRIPTOR: ArchiveComponentDescriptor =
    ArchiveComponentDescriptor {
        id: "attached_best_effort",
        description: "attached archive nonblocking visible-loss admission",
    };
static CREATE_NEW_RECOVERY_DESCRIPTOR: ArchiveComponentDescriptor = ArchiveComponentDescriptor {
    id: "create_new",
    description: "create generation zero only when no authority exists",
};
static EXACT_RESUME_RECOVERY_DESCRIPTOR: ArchiveComponentDescriptor = ArchiveComponentDescriptor {
    id: "exact_resume",
    description: "resume only exact persistent identity and compatible ancestry",
};
static FINALIZE_REMOTE_RECOVERY_DESCRIPTOR: ArchiveComponentDescriptor =
    ArchiveComponentDescriptor {
        id: "finalize_remote",
        description: "source-free terminal remote publication",
    };
static SECRET_PROVIDER_KEY_DESCRIPTOR: ArchiveComponentDescriptor = ArchiveComponentDescriptor {
    id: "secret_provider",
    description: "provider-held archive identity key",
};
static NOOP_ENRICHER_DESCRIPTOR: ArchiveComponentDescriptor = ArchiveComponentDescriptor {
    id: "noop",
    description: "no additional telemetry attributes",
};
static STATIC_LABELS_ENRICHER_DESCRIPTOR: ArchiveComponentDescriptor = ArchiveComponentDescriptor {
    id: "static_labels",
    description: "validated static attributes added to every archived sample",
};
static NOOP_SANITIZER_DESCRIPTOR: ArchiveComponentDescriptor = ArchiveComponentDescriptor {
    id: "noop",
    description: "no additional policy after mandatory credential sanitization",
};
static ALLOW_DENY_SANITIZER_DESCRIPTOR: ArchiveComponentDescriptor = ArchiveComponentDescriptor {
    id: "allow_deny_keys",
    description: "structured label/attribute allow and deny key policy",
};
static NO_RAW_BODY_DESCRIPTOR: ArchiveComponentDescriptor = ArchiveComponentDescriptor {
    id: "none",
    description: "retain no exact response body",
};

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct EmptyConfig {}

/// Built-in Parquet v1 writer factory.
#[derive(Clone, Copy, Debug, Default)]
pub struct ParquetArchiveWriterFactory;

impl ArchiveWriterComponentFactory for ParquetArchiveWriterFactory {
    fn descriptor(&self) -> &'static ArchiveComponentDescriptor {
        &PARQUET_WRITER_DESCRIPTOR
    }

    fn validate(
        &self,
        config: &RawValue,
    ) -> Result<Box<dyn ValidatedArchiveWriterComponent>, ArchiveComponentError> {
        let config: EmptyConfig = decode_config(WRITER_FAMILY, self.descriptor().id, config)?;
        let identity = component_identity(WRITER_FAMILY, self.descriptor().id, &config)?;
        let schemas =
            ArchiveSchemasV1::load().map_err(|error| ArchiveComponentError::InvalidConfig {
                family: WRITER_FAMILY,
                id: self.descriptor().id.to_owned(),
                message: error.to_string(),
            })?;
        let persistent_writer_identity = parquet_writer_identity(&schemas)?;
        let writer_compatibility_id = domain_digest(
            "aiperf.archive.writer-compatibility.v1",
            &[persistent_writer_identity.to_bytes().as_slice()],
        );
        Ok(Box::new(ValidatedParquetArchiveWriter {
            identity,
            schemas,
            persistent_writer_identity,
            writer_compatibility_id,
        }))
    }
}

#[derive(Debug)]
struct ValidatedParquetArchiveWriter {
    identity: ValidatedArchiveComponentIdentity,
    schemas: ArchiveSchemasV1,
    persistent_writer_identity: CanonicalJsonValue,
    writer_compatibility_id: Digest,
}

impl ValidatedArchiveWriterComponent for ValidatedParquetArchiveWriter {
    fn identity(&self) -> &ValidatedArchiveComponentIdentity {
        &self.identity
    }

    fn persistent_writer_identity(&self) -> &CanonicalJsonValue {
        &self.persistent_writer_identity
    }

    fn writer_compatibility_id(&self) -> Digest {
        self.writer_compatibility_id
    }

    fn prepare(
        self: Box<Self>,
        rotation: ParquetRotationConfigV1,
        faults: Arc<dyn DurabilityFaultInjector>,
    ) -> Result<Box<dyn PreparedArchiveWriter>, ArchiveComponentError> {
        let sink_factory =
            OwnedLocalArchiveSinkFactory::new(self.schemas.clone(), rotation, faults)
                .map_err(|error| ArchiveComponentError::Prepare(error.to_string()))?;
        Ok(Box::new(PreparedParquetArchiveWriter {
            schemas: self.schemas,
            persistent_writer_identity: self.persistent_writer_identity,
            writer_compatibility_id: self.writer_compatibility_id,
            sink_factory,
        }))
    }
}

struct PreparedParquetArchiveWriter {
    schemas: ArchiveSchemasV1,
    persistent_writer_identity: CanonicalJsonValue,
    writer_compatibility_id: Digest,
    sink_factory: OwnedLocalArchiveSinkFactory,
}

impl Debug for PreparedParquetArchiveWriter {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedParquetArchiveWriter")
            .field("writer_compatibility_id", &self.writer_compatibility_id)
            .field("sink_factory", &self.sink_factory)
            .finish_non_exhaustive()
    }
}

impl PreparedArchiveWriter for PreparedParquetArchiveWriter {
    fn schemas(&self) -> &ArchiveSchemasV1 {
        &self.schemas
    }

    fn persistent_writer_identity(&self) -> &CanonicalJsonValue {
        &self.persistent_writer_identity
    }

    fn writer_compatibility_id(&self) -> Digest {
        self.writer_compatibility_id
    }

    fn prepare_new_sink(
        &self,
        repository: LocalArchiveRepository,
        header: WalSegmentHeaderV1,
        receipts: OwnedReceiptJournalMode,
    ) -> Result<Box<dyn ArchiveSink>, ArchiveComponentError> {
        self.sink_factory
            .prepare(repository, header, receipts)
            .map(|sink| Box::new(sink) as Box<dyn ArchiveSink>)
            .map_err(|error| ArchiveComponentError::Prepare(error.to_string()))
    }

    fn prepare_resumed_sink(
        &self,
        repository: LocalArchiveRepository,
        new_session_id: aiperf_telemetry_archive::SessionId,
        maximum_frame_bytes: u64,
        receipts: OwnedReceiptJournalMode,
        decoder: &dyn ArchiveWalFrameDecoder,
    ) -> Result<Box<dyn ArchiveSink>, ArchiveComponentError> {
        self.sink_factory
            .resume(
                repository,
                new_session_id,
                maximum_frame_bytes,
                receipts,
                decoder,
            )
            .map(|sink| Box::new(sink) as Box<dyn ArchiveSink>)
            .map_err(|error| ArchiveComponentError::Prepare(error.to_string()))
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct LocalFilesystemStoreAccessFactory;

impl ArchiveStoreAccessComponentFactory for LocalFilesystemStoreAccessFactory {
    fn descriptor(&self) -> &'static ArchiveComponentDescriptor {
        &LOCAL_FILESYSTEM_STORE_DESCRIPTOR
    }

    fn validate(
        &self,
        config: &RawValue,
    ) -> Result<Box<dyn ValidatedArchiveStoreAccessComponent>, ArchiveComponentError> {
        let config: EmptyConfig = decode_config(STORE_ACCESS_FAMILY, self.descriptor().id, config)?;
        Ok(Box::new(ValidatedStoreAccess {
            identity: component_identity(STORE_ACCESS_FAMILY, self.descriptor().id, &config)?,
            supported_schemes: &["file"],
            credential_provider: None,
        }))
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ObjectStoreAccessConfig {
    #[serde(default)]
    credential_provider: Option<String>,
}

#[derive(Clone, Copy, Debug, Default)]
struct ObjectStoreAccessFactory;

impl ArchiveStoreAccessComponentFactory for ObjectStoreAccessFactory {
    fn descriptor(&self) -> &'static ArchiveComponentDescriptor {
        &OBJECT_STORE_DESCRIPTOR
    }

    fn validate(
        &self,
        config: &RawValue,
    ) -> Result<Box<dyn ValidatedArchiveStoreAccessComponent>, ArchiveComponentError> {
        let config: ObjectStoreAccessConfig =
            decode_config(STORE_ACCESS_FAMILY, self.descriptor().id, config)?;
        if let Some(reference) = &config.credential_provider {
            validate_external_reference(reference, "object_store credential_provider")?;
        }
        Ok(Box::new(ValidatedStoreAccess {
            identity: component_identity(STORE_ACCESS_FAMILY, self.descriptor().id, &config)?,
            supported_schemes: &["s3", "gs", "az"],
            credential_provider: config.credential_provider,
        }))
    }
}

#[derive(Debug)]
struct ValidatedStoreAccess {
    identity: ValidatedArchiveComponentIdentity,
    supported_schemes: &'static [&'static str],
    credential_provider: Option<String>,
}

impl ValidatedArchiveStoreAccessComponent for ValidatedStoreAccess {
    fn identity(&self) -> &ValidatedArchiveComponentIdentity {
        &self.identity
    }

    fn validate_target(&self, target: &NormalizedArchiveUri) -> Result<(), ArchiveComponentError> {
        if !self.supported_schemes.contains(&target.scheme()) {
            return Err(ArchiveComponentError::IncompatibleSelection(format!(
                "archive store access {:?} does not support target scheme {:?}",
                self.identity.factory_id,
                target.scheme()
            )));
        }
        Ok(())
    }

    fn prepare(
        self: Box<Self>,
        target: &NormalizedArchiveUri,
        provider: &dyn ArchiveObjectStoreProvider,
    ) -> Result<PreparedArchiveStoreAccess, ArchiveComponentError> {
        self.validate_target(target)?;
        let store = provider.prepare(ArchiveObjectStorePrepareRequest {
            access_factory_id: self.identity.factory_id,
            target,
            canonical_config: &self.identity.canonical_config,
            credential_provider: self.credential_provider.as_deref(),
        })?;
        store
            .capabilities()
            .require_authoritative()
            .map_err(|error| ArchiveComponentError::Prepare(error.to_string()))?;
        Ok(PreparedArchiveStoreAccess {
            identity: self.identity,
            store,
        })
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RowsBytesAgeRotationConfig {
    #[serde(default = "default_target_rows")]
    target_rows: u64,
    #[serde(default = "default_target_uncompressed_bytes")]
    target_uncompressed_bytes: u64,
    #[serde(default = "default_hard_rows")]
    hard_rows: u64,
    #[serde(default = "default_hard_bytes")]
    hard_bytes: u64,
    #[serde(default = "default_time_bucket_ns")]
    time_bucket_ns: i64,
    #[serde(default = "default_maximum_age_ns")]
    maximum_age_ns: u64,
}

const fn default_target_rows() -> u64 {
    100_000
}

const fn default_target_uncompressed_bytes() -> u64 {
    64 * 1024 * 1024
}

const fn default_hard_rows() -> u64 {
    1_000_000
}

const fn default_hard_bytes() -> u64 {
    1024 * 1024 * 1024
}

const fn default_time_bucket_ns() -> i64 {
    60_000_000_000
}

const fn default_maximum_age_ns() -> u64 {
    60_000_000_000
}

#[derive(Clone, Copy, Debug, Default)]
struct RowsBytesAgeRotationFactory;

impl ArchiveRotationComponentFactory for RowsBytesAgeRotationFactory {
    fn descriptor(&self) -> &'static ArchiveComponentDescriptor {
        &ROWS_BYTES_AGE_ROTATION_DESCRIPTOR
    }

    fn validate(
        &self,
        config: &RawValue,
    ) -> Result<Box<dyn ValidatedArchiveRotationComponent>, ArchiveComponentError> {
        let config: RowsBytesAgeRotationConfig =
            decode_config(ROTATION_FAMILY, self.descriptor().id, config)?;
        let parquet = ParquetRotationConfigV1 {
            target_rows: config.target_rows,
            target_uncompressed_bytes: config.target_uncompressed_bytes,
            hard_rows: config.hard_rows,
            hard_bytes: config.hard_bytes,
            time_bucket_ns: config.time_bucket_ns,
        }
        .validate()
        .map_err(|error| ArchiveComponentError::InvalidConfig {
            family: ROTATION_FAMILY,
            id: self.descriptor().id.to_owned(),
            message: error.to_string(),
        })?;
        BoundedSegmentRotationPolicy::new(
            None,
            Some(config.target_rows),
            Some(config.target_uncompressed_bytes),
            Some(config.maximum_age_ns),
        )
        .map_err(|error| ArchiveComponentError::InvalidConfig {
            family: ROTATION_FAMILY,
            id: self.descriptor().id.to_owned(),
            message: error.to_string(),
        })?;
        Ok(Box::new(ValidatedRowsBytesAgeRotation {
            identity: component_identity(ROTATION_FAMILY, self.descriptor().id, &config)?,
            parquet,
            maximum_age_ns: config.maximum_age_ns,
        }))
    }
}

#[derive(Debug)]
struct ValidatedRowsBytesAgeRotation {
    identity: ValidatedArchiveComponentIdentity,
    parquet: ParquetRotationConfigV1,
    maximum_age_ns: u64,
}

impl ValidatedArchiveRotationComponent for ValidatedRowsBytesAgeRotation {
    fn identity(&self) -> &ValidatedArchiveComponentIdentity {
        &self.identity
    }

    fn prepare(self: Box<Self>) -> Result<PreparedArchiveRotation, ArchiveComponentError> {
        let segment = BoundedSegmentRotationPolicy::new(
            None,
            Some(self.parquet.target_rows),
            Some(self.parquet.target_uncompressed_bytes),
            Some(self.maximum_age_ns),
        )
        .map_err(|error| ArchiveComponentError::Prepare(error.to_string()))?;
        Ok(PreparedArchiveRotation {
            parquet: self.parquet,
            segment: Box::new(segment),
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct PrimaryDurableAdmissionFactory;

#[derive(Clone, Copy, Debug, Default)]
struct AttachedBestEffortAdmissionFactory;

impl ArchiveAdmissionComponentFactory for PrimaryDurableAdmissionFactory {
    fn descriptor(&self) -> &'static ArchiveComponentDescriptor {
        &PRIMARY_DURABLE_ADMISSION_DESCRIPTOR
    }

    fn validate(
        &self,
        config: &RawValue,
    ) -> Result<Box<dyn ValidatedArchiveAdmissionComponent>, ArchiveComponentError> {
        validate_admission(
            self.descriptor(),
            config,
            ArchiveAdmissionMode::PrimaryWatch,
        )
    }
}

impl ArchiveAdmissionComponentFactory for AttachedBestEffortAdmissionFactory {
    fn descriptor(&self) -> &'static ArchiveComponentDescriptor {
        &ATTACHED_BEST_EFFORT_ADMISSION_DESCRIPTOR
    }

    fn validate(
        &self,
        config: &RawValue,
    ) -> Result<Box<dyn ValidatedArchiveAdmissionComponent>, ArchiveComponentError> {
        validate_admission(
            self.descriptor(),
            config,
            ArchiveAdmissionMode::AttachedBestEffort,
        )
    }
}

fn validate_admission(
    descriptor: &'static ArchiveComponentDescriptor,
    raw: &RawValue,
    mode: ArchiveAdmissionMode,
) -> Result<Box<dyn ValidatedArchiveAdmissionComponent>, ArchiveComponentError> {
    let config: EmptyConfig = decode_config(ADMISSION_FAMILY, descriptor.id, raw)?;
    Ok(Box::new(ValidatedAdmission {
        identity: component_identity(ADMISSION_FAMILY, descriptor.id, &config)?,
        mode,
    }))
}

#[derive(Debug)]
struct ValidatedAdmission {
    identity: ValidatedArchiveComponentIdentity,
    mode: ArchiveAdmissionMode,
}

impl ValidatedArchiveAdmissionComponent for ValidatedAdmission {
    fn identity(&self) -> &ValidatedArchiveComponentIdentity {
        &self.identity
    }

    fn mode(&self) -> ArchiveAdmissionMode {
        self.mode
    }

    fn prepare(self: Box<Self>) -> Arc<dyn ArchiveAdmissionPolicy> {
        match self.mode {
            ArchiveAdmissionMode::PrimaryWatch => Arc::new(PrimaryWatchAdmissionPolicy),
            ArchiveAdmissionMode::AttachedBestEffort => Arc::new(AttachedBestEffortAdmissionPolicy),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct CreateNewRecoveryFactory;

#[derive(Clone, Copy, Debug, Default)]
struct ExactResumeRecoveryFactory;

#[derive(Clone, Copy, Debug, Default)]
struct FinalizeRemoteRecoveryFactory;

macro_rules! impl_recovery_factory {
    ($factory:ident, $descriptor:ident, $operation:expr) => {
        impl ArchiveRecoveryComponentFactory for $factory {
            fn descriptor(&self) -> &'static ArchiveComponentDescriptor {
                &$descriptor
            }

            fn validate(
                &self,
                config: &RawValue,
            ) -> Result<Box<dyn ValidatedArchiveRecoveryComponent>, ArchiveComponentError> {
                let config: EmptyConfig =
                    decode_config(RECOVERY_FAMILY, self.descriptor().id, config)?;
                Ok(Box::new(ValidatedRecovery {
                    identity: component_identity(RECOVERY_FAMILY, self.descriptor().id, &config)?,
                    operation: $operation,
                    expected_archive_id: None,
                    expected_prior_claim_id: None,
                }))
            }
        }
    };
}

impl_recovery_factory!(
    CreateNewRecoveryFactory,
    CREATE_NEW_RECOVERY_DESCRIPTOR,
    ArchiveRecoveryOperation::CreateNew
);
impl_recovery_factory!(
    FinalizeRemoteRecoveryFactory,
    FINALIZE_REMOTE_RECOVERY_DESCRIPTOR,
    ArchiveRecoveryOperation::FinalizeRemote
);

#[derive(Debug)]
struct ValidatedRecovery {
    identity: ValidatedArchiveComponentIdentity,
    operation: ArchiveRecoveryOperation,
    expected_archive_id: Option<ArchiveId>,
    expected_prior_claim_id: Option<WriterClaimId>,
}

impl ValidatedArchiveRecoveryComponent for ValidatedRecovery {
    fn identity(&self) -> &ValidatedArchiveComponentIdentity {
        &self.identity
    }

    fn operation(&self) -> ArchiveRecoveryOperation {
        self.operation
    }

    fn expected_archive_id(&self) -> Option<ArchiveId> {
        self.expected_archive_id
    }

    fn expected_prior_claim_id(&self) -> Option<WriterClaimId> {
        self.expected_prior_claim_id
    }

    fn bind_collect(
        self: Box<Self>,
        expectation: ArchiveRecoveryExpectation,
    ) -> Result<Box<dyn ArchiveRecoveryPolicy>, ArchiveComponentError> {
        match self.operation {
            ArchiveRecoveryOperation::CreateNew => Ok(Box::new(CreateNewRecoveryPolicy)),
            ArchiveRecoveryOperation::ExactResume => {
                let expected_archive_id = self.expected_archive_id.ok_or_else(|| {
                    ArchiveComponentError::Prepare(
                        "exact_resume is missing its validated authored archive ID".to_owned(),
                    )
                })?;
                if expectation.archive_id != expected_archive_id {
                    return Err(ArchiveComponentError::IncompatibleSelection(format!(
                        "exact_resume discovered archive ID {} but the authored archive ID is {}",
                        uuid::Uuid::from_bytes(*expectation.archive_id.as_bytes()),
                        uuid::Uuid::from_bytes(*expected_archive_id.as_bytes())
                    )));
                }
                Ok(Box::new(ExactResumeRecoveryPolicy::new(
                    expected_archive_id,
                    expectation.persistent_identity_digest,
                    self.expected_prior_claim_id.ok_or_else(|| {
                        ArchiveComponentError::Prepare(
                            "exact_resume is missing its validated prior claim ID".to_owned(),
                        )
                    })?,
                )))
            }
            ArchiveRecoveryOperation::FinalizeRemote => {
                Err(ArchiveComponentError::IncompatibleSelection(
                    "finalize_remote cannot bind a collect recovery policy".to_owned(),
                ))
            }
        }
    }

    fn bind_finalize_remote(self: Box<Self>) -> Result<(), ArchiveComponentError> {
        if self.operation != ArchiveRecoveryOperation::FinalizeRemote {
            return Err(ArchiveComponentError::IncompatibleSelection(format!(
                "recovery operation {:?} cannot perform source-free finalization",
                self.operation
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ExactResumeRecoveryConfig {
    archive_id: Uuid,
    prior_claim_id: String,
}

impl ArchiveRecoveryComponentFactory for ExactResumeRecoveryFactory {
    fn descriptor(&self) -> &'static ArchiveComponentDescriptor {
        &EXACT_RESUME_RECOVERY_DESCRIPTOR
    }

    fn validate(
        &self,
        config: &RawValue,
    ) -> Result<Box<dyn ValidatedArchiveRecoveryComponent>, ArchiveComponentError> {
        let config: ExactResumeRecoveryConfig =
            decode_config(RECOVERY_FAMILY, self.descriptor().id, config)?;
        if config.archive_id.is_nil() {
            return Err(ArchiveComponentError::InvalidConfig {
                family: RECOVERY_FAMILY,
                id: self.descriptor().id.to_owned(),
                message: "archive_id cannot be nil".to_owned(),
            });
        }
        let expected_archive_id =
            ArchiveId::new(*config.archive_id.as_bytes()).map_err(|error| {
                ArchiveComponentError::InvalidConfig {
                    family: RECOVERY_FAMILY,
                    id: self.descriptor().id.to_owned(),
                    message: error.to_string(),
                }
            })?;
        let expected_prior_claim_id =
            WriterClaimId::parse(&config.prior_claim_id).map_err(|error| {
                ArchiveComponentError::InvalidConfig {
                    family: RECOVERY_FAMILY,
                    id: self.descriptor().id.to_owned(),
                    message: format!("prior_claim_id: {error}"),
                }
            })?;
        if config.prior_claim_id != expected_prior_claim_id.to_hex() {
            return Err(ArchiveComponentError::InvalidConfig {
                family: RECOVERY_FAMILY,
                id: self.descriptor().id.to_owned(),
                message: "prior_claim_id must be 64 lowercase hexadecimal characters".to_owned(),
            });
        }
        Ok(Box::new(ValidatedRecovery {
            identity: component_identity(RECOVERY_FAMILY, self.descriptor().id, &config)?,
            operation: ArchiveRecoveryOperation::ExactResume,
            expected_archive_id: Some(expected_archive_id),
            expected_prior_claim_id: Some(expected_prior_claim_id),
        }))
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SecretProviderArchiveKeyConfig {
    id: String,
}

#[derive(Clone, Copy, Debug, Default)]
struct SecretProviderArchiveKeyFactory;

impl ArchiveKeyComponentFactory for SecretProviderArchiveKeyFactory {
    fn descriptor(&self) -> &'static ArchiveComponentDescriptor {
        &SECRET_PROVIDER_KEY_DESCRIPTOR
    }

    fn validate(
        &self,
        config: &RawValue,
    ) -> Result<Box<dyn ValidatedArchiveKeyComponent>, ArchiveComponentError> {
        let config: SecretProviderArchiveKeyConfig =
            decode_config(ARCHIVE_KEY_FAMILY, self.descriptor().id, config)?;
        validate_secret_reference(&config.id)?;
        Ok(Box::new(ValidatedSecretProviderArchiveKey {
            identity: component_identity(ARCHIVE_KEY_FAMILY, self.descriptor().id, &config)?,
            secret_reference: config.id,
        }))
    }
}

#[derive(Debug)]
struct ValidatedSecretProviderArchiveKey {
    identity: ValidatedArchiveComponentIdentity,
    secret_reference: String,
}

impl ValidatedArchiveKeyComponent for ValidatedSecretProviderArchiveKey {
    fn identity(&self) -> &ValidatedArchiveComponentIdentity {
        &self.identity
    }

    fn prepare(
        self: Box<Self>,
        resolver: &dyn ArchiveKeyProviderResolver,
    ) -> Result<Arc<dyn ArchiveKeyProvider>, ArchiveComponentError> {
        let provider = resolver.resolve(&self.secret_reference)?;
        if provider.provider_id() != self.identity.factory_id {
            return Err(ArchiveComponentError::Prepare(format!(
                "archive key resolver returned provider ID {:?}; expected factory ID {:?}",
                provider.provider_id(),
                self.identity.factory_id
            )));
        }
        Ok(provider)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct NoopEnricherFactory;

impl TelemetryEnricherComponentFactory for NoopEnricherFactory {
    fn descriptor(&self) -> &'static ArchiveComponentDescriptor {
        &NOOP_ENRICHER_DESCRIPTOR
    }

    fn validate(
        &self,
        config: &RawValue,
    ) -> Result<Box<dyn ValidatedTelemetryEnricherComponent>, ArchiveComponentError> {
        let config: EmptyConfig = decode_config(ENRICHER_FAMILY, self.descriptor().id, config)?;
        Ok(Box::new(ValidatedNoopEnricher {
            identity: component_identity(ENRICHER_FAMILY, self.descriptor().id, &config)?,
        }))
    }
}

#[derive(Debug)]
struct ValidatedNoopEnricher {
    identity: ValidatedArchiveComponentIdentity,
}

impl ValidatedTelemetryEnricherComponent for ValidatedNoopEnricher {
    fn identity(&self) -> &ValidatedArchiveComponentIdentity {
        &self.identity
    }

    fn prepare(self: Box<Self>) -> Result<Arc<dyn TelemetryEnricher>, ArchiveComponentError> {
        Ok(Arc::new(NoopEnricher))
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct StaticLabelsEnricherConfig {
    attributes: BTreeMap<String, String>,
}

#[derive(Clone, Copy, Debug, Default)]
struct StaticLabelsEnricherFactory;

impl TelemetryEnricherComponentFactory for StaticLabelsEnricherFactory {
    fn descriptor(&self) -> &'static ArchiveComponentDescriptor {
        &STATIC_LABELS_ENRICHER_DESCRIPTOR
    }

    fn validate(
        &self,
        config: &RawValue,
    ) -> Result<Box<dyn ValidatedTelemetryEnricherComponent>, ArchiveComponentError> {
        let config: StaticLabelsEnricherConfig =
            decode_config(ENRICHER_FAMILY, self.descriptor().id, config)?;
        let effective = config.clone();
        let enricher = StaticLabelEnricher::new(config.attributes).map_err(|error| {
            ArchiveComponentError::InvalidConfig {
                family: ENRICHER_FAMILY,
                id: self.descriptor().id.to_owned(),
                message: error.to_string(),
            }
        })?;
        Ok(Box::new(ValidatedStaticLabelsEnricher {
            identity: component_identity(ENRICHER_FAMILY, self.descriptor().id, &effective)?,
            enricher,
        }))
    }
}

#[derive(Debug)]
struct ValidatedStaticLabelsEnricher {
    identity: ValidatedArchiveComponentIdentity,
    enricher: StaticLabelEnricher,
}

impl ValidatedTelemetryEnricherComponent for ValidatedStaticLabelsEnricher {
    fn identity(&self) -> &ValidatedArchiveComponentIdentity {
        &self.identity
    }

    fn prepare(self: Box<Self>) -> Result<Arc<dyn TelemetryEnricher>, ArchiveComponentError> {
        Ok(Arc::new(self.enricher))
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct NoopSanitizerFactory;

impl ArchiveSanitizerComponentFactory for NoopSanitizerFactory {
    fn descriptor(&self) -> &'static ArchiveComponentDescriptor {
        &NOOP_SANITIZER_DESCRIPTOR
    }

    fn validate(
        &self,
        config: &RawValue,
    ) -> Result<Box<dyn ValidatedArchiveSanitizerComponent>, ArchiveComponentError> {
        let config: EmptyConfig = decode_config(SANITIZER_FAMILY, self.descriptor().id, config)?;
        Ok(Box::new(ValidatedNoopSanitizer {
            identity: component_identity(SANITIZER_FAMILY, self.descriptor().id, &config)?,
        }))
    }
}

#[derive(Debug)]
struct ValidatedNoopSanitizer {
    identity: ValidatedArchiveComponentIdentity,
}

impl ValidatedArchiveSanitizerComponent for ValidatedNoopSanitizer {
    fn identity(&self) -> &ValidatedArchiveComponentIdentity {
        &self.identity
    }

    fn prepare(self: Box<Self>) -> Result<Arc<dyn ArchiveSanitizer>, ArchiveComponentError> {
        Ok(Arc::new(NoopSanitizer))
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct AllowDenySanitizerConfig {
    #[serde(default)]
    allow_labels: Option<BTreeSet<String>>,
    #[serde(default)]
    deny_labels: BTreeSet<String>,
    #[serde(default)]
    allow_attributes: Option<BTreeSet<String>>,
    #[serde(default)]
    deny_attributes: BTreeSet<String>,
}

#[derive(Clone, Copy, Debug, Default)]
struct AllowDenySanitizerFactory;

impl ArchiveSanitizerComponentFactory for AllowDenySanitizerFactory {
    fn descriptor(&self) -> &'static ArchiveComponentDescriptor {
        &ALLOW_DENY_SANITIZER_DESCRIPTOR
    }

    fn validate(
        &self,
        config: &RawValue,
    ) -> Result<Box<dyn ValidatedArchiveSanitizerComponent>, ArchiveComponentError> {
        let config: AllowDenySanitizerConfig =
            decode_config(SANITIZER_FAMILY, self.descriptor().id, config)?;
        validate_allow_deny_config(&config)?;
        Ok(Box::new(ValidatedAllowDenySanitizer {
            identity: component_identity(SANITIZER_FAMILY, self.descriptor().id, &config)?,
            config,
        }))
    }
}

#[derive(Debug)]
struct ValidatedAllowDenySanitizer {
    identity: ValidatedArchiveComponentIdentity,
    config: AllowDenySanitizerConfig,
}

impl ValidatedArchiveSanitizerComponent for ValidatedAllowDenySanitizer {
    fn identity(&self) -> &ValidatedArchiveComponentIdentity {
        &self.identity
    }

    fn prepare(self: Box<Self>) -> Result<Arc<dyn ArchiveSanitizer>, ArchiveComponentError> {
        Ok(Arc::new(AllowDenyKeySanitizer {
            allow_labels: self.config.allow_labels,
            deny_labels: self.config.deny_labels,
            allow_attributes: self.config.allow_attributes,
            deny_attributes: self.config.deny_attributes,
        }))
    }
}

/// Mandatory non-disableable removal of known credential-shaped structured keys.
#[derive(Clone, Copy, Debug, Default)]
pub struct BaselineCredentialSanitizer;

impl ArchiveSanitizer for BaselineCredentialSanitizer {
    fn sanitize_sample(
        &self,
        sample: ArchiveSampleView<'_>,
    ) -> Result<SanitizedSample, SanitizationError> {
        Ok(SanitizedSample {
            labels: sample
                .labels
                .iter()
                .filter(|(key, _)| !is_known_credential_key(key))
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
            attributes: sample
                .attributes
                .iter()
                .filter(|(key, _)| !is_known_credential_key(key))
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        })
    }
}

#[derive(Debug)]
struct AllowDenyKeySanitizer {
    allow_labels: Option<BTreeSet<String>>,
    deny_labels: BTreeSet<String>,
    allow_attributes: Option<BTreeSet<String>>,
    deny_attributes: BTreeSet<String>,
}

impl ArchiveSanitizer for AllowDenyKeySanitizer {
    fn sanitize_sample(
        &self,
        sample: ArchiveSampleView<'_>,
    ) -> Result<SanitizedSample, SanitizationError> {
        Ok(SanitizedSample {
            labels: filter_map(sample.labels, self.allow_labels.as_ref(), &self.deny_labels),
            attributes: filter_map(
                sample.attributes,
                self.allow_attributes.as_ref(),
                &self.deny_attributes,
            ),
        })
    }
}

#[derive(Debug)]
struct ArchiveSanitizerChain {
    optional: Vec<Arc<dyn ArchiveSanitizer>>,
}

impl ArchiveSanitizerChain {
    fn new(optional: Vec<Arc<dyn ArchiveSanitizer>>) -> Self {
        Self { optional }
    }
}

impl ArchiveSanitizer for ArchiveSanitizerChain {
    fn sanitize_sample(
        &self,
        sample: ArchiveSampleView<'_>,
    ) -> Result<SanitizedSample, SanitizationError> {
        let mut sanitized = BaselineCredentialSanitizer.sanitize_sample(sample)?;
        for policy in &self.optional {
            sanitized = policy.sanitize_sample(ArchiveSampleView {
                source_id: sample.source_id,
                metric_family: sample.metric_family,
                semantic_type: sample.semantic_type,
                labels: &sanitized.labels,
                attributes: &sanitized.attributes,
            })?;
        }
        Ok(sanitized)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct NoRawBodyFactory;

impl RawBodyComponentFactory for NoRawBodyFactory {
    fn descriptor(&self) -> &'static ArchiveComponentDescriptor {
        &NO_RAW_BODY_DESCRIPTOR
    }

    fn validate(
        &self,
        config: &RawValue,
    ) -> Result<Box<dyn ValidatedRawBodyComponent>, ArchiveComponentError> {
        let config: EmptyConfig = decode_config(RAW_BODY_FAMILY, self.descriptor().id, config)?;
        Ok(Box::new(ValidatedNoRawBody {
            identity: component_identity(RAW_BODY_FAMILY, self.descriptor().id, &config)?,
        }))
    }
}

#[derive(Debug)]
struct ValidatedNoRawBody {
    identity: ValidatedArchiveComponentIdentity,
}

impl ValidatedRawBodyComponent for ValidatedNoRawBody {
    fn identity(&self) -> &ValidatedArchiveComponentIdentity {
        &self.identity
    }

    fn prepare(self: Box<Self>) -> Result<Box<dyn PreparedRawBodyPolicy>, ArchiveComponentError> {
        Ok(Box::new(NoRawBodyPolicy))
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct NoRawBodyPolicy;

impl PreparedRawBodyPolicy for NoRawBodyPolicy {
    fn retains_exact_body(&self) -> bool {
        false
    }
}

fn decode_config<T: for<'de> Deserialize<'de>>(
    family: &'static str,
    id: &str,
    config: &RawValue,
) -> Result<T, ArchiveComponentError> {
    serde_json::from_str(config.get()).map_err(|error| ArchiveComponentError::InvalidConfig {
        family,
        id: id.to_owned(),
        message: error.to_string(),
    })
}

fn component_identity<T: Serialize>(
    family: &'static str,
    factory_id: &'static str,
    config: &T,
) -> Result<ValidatedArchiveComponentIdentity, ArchiveComponentError> {
    let bytes = serde_json::to_vec(config)
        .map_err(|error| ArchiveComponentError::Canonical(error.to_string()))?;
    let canonical = CanonicalJsonValue::parse(&bytes)
        .map_err(|error| ArchiveComponentError::Canonical(error.to_string()))?;
    ValidatedArchiveComponentIdentity::new(family, factory_id, canonical)
}

fn baseline_sanitizer_identity() -> Result<ValidatedArchiveComponentIdentity, ArchiveComponentError>
{
    component_identity(SANITIZER_FAMILY, "baseline_credentials", &EmptyConfig {})
}

fn parquet_writer_identity(
    schemas: &ArchiveSchemasV1,
) -> Result<CanonicalJsonValue, ArchiveComponentError> {
    let schema_fingerprints = schemas
        .iter()
        .map(|schema| {
            CanonicalJsonValue::object([
                (
                    "fingerprint".to_owned(),
                    CanonicalJsonValue::String(schema.fingerprint().to_hex()),
                ),
                (
                    "table".to_owned(),
                    CanonicalJsonValue::String(schema.table_name().to_owned()),
                ),
            ])
            .map_err(|error| ArchiveComponentError::Canonical(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    CanonicalJsonValue::object([
        (
            "compression".to_owned(),
            CanonicalJsonValue::String("uncompressed".to_owned()),
        ),
        (
            "created_by".to_owned(),
            CanonicalJsonValue::String("aiperf-telemetry-archive-v1".to_owned()),
        ),
        (
            "dictionary_enabled".to_owned(),
            CanonicalJsonValue::Bool(false),
        ),
        (
            "factory".to_owned(),
            CanonicalJsonValue::String(PARQUET_WRITER_DESCRIPTOR.id.to_owned()),
        ),
        (
            "parquet_writer_version".to_owned(),
            CanonicalJsonValue::String("2.0".to_owned()),
        ),
        (
            "schema_fingerprints".to_owned(),
            CanonicalJsonValue::Array(schema_fingerprints),
        ),
    ])
    .map_err(|error| ArchiveComponentError::Canonical(error.to_string()))
}

fn validate_factory_id(value: &str, family: &'static str) -> Result<(), ArchiveComponentError> {
    let mut bytes = value.bytes();
    let Some(first) = bytes.next() else {
        return Err(ArchiveComponentError::InvalidFactoryId {
            family,
            id: value.to_owned(),
        });
    };
    if !first.is_ascii_lowercase()
        || !bytes.all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
    {
        return Err(ArchiveComponentError::InvalidFactoryId {
            family,
            id: value.to_owned(),
        });
    }
    Ok(())
}

fn validate_external_reference(value: &str, field: &str) -> Result<(), ArchiveComponentError> {
    if value.is_empty()
        || value.len() > 256
        || value.trim() != value
        || value.chars().any(char::is_control)
    {
        return Err(ArchiveComponentError::InvalidArchive(format!(
            "{field} must be 1..=256 bytes without surrounding whitespace or control characters"
        )));
    }
    Ok(())
}

fn validate_secret_reference(value: &str) -> Result<(), ArchiveComponentError> {
    validate_external_reference(value, "archive_key secret-provider id")?;
    let mut bytes = value.bytes();
    let Some(first) = bytes.next() else {
        return Err(ArchiveComponentError::InvalidArchive(
            "archive_key secret-provider id cannot be empty".to_owned(),
        ));
    };
    if !first.is_ascii_lowercase()
        || !bytes.all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
    {
        return Err(ArchiveComponentError::InvalidArchive(
            "archive_key secret-provider id must start with a lowercase ASCII letter and contain only lowercase letters, digits, and hyphens"
                .to_owned(),
        ));
    }
    Ok(())
}

fn validate_store_spool_separation(
    target: &NormalizedArchiveUri,
    local_spool: &std::path::Path,
) -> Result<(), ArchiveComponentError> {
    if target.scheme() != "file" {
        return Ok(());
    }
    let parsed = Url::parse(target.as_str()).map_err(|error| {
        ArchiveComponentError::InvalidArchive(format!(
            "validated file archive target could not be parsed: {error}"
        ))
    })?;
    let target_path = parsed.to_file_path().map_err(|()| {
        ArchiveComponentError::InvalidArchive(
            "validated file archive target could not be converted to a local path".to_owned(),
        )
    })?;
    if target_path.starts_with(local_spool) || local_spool.starts_with(&target_path) {
        return Err(ArchiveComponentError::InvalidArchive(
            "file archive target and local spool must be disjoint paths".to_owned(),
        ));
    }
    Ok(())
}

fn validate_allow_deny_config(
    config: &AllowDenySanitizerConfig,
) -> Result<(), ArchiveComponentError> {
    if config.allow_labels.is_none()
        && config.deny_labels.is_empty()
        && config.allow_attributes.is_none()
        && config.deny_attributes.is_empty()
    {
        return Err(ArchiveComponentError::InvalidConfig {
            family: SANITIZER_FAMILY,
            id: ALLOW_DENY_SANITIZER_DESCRIPTOR.id.to_owned(),
            message: "allow_deny_keys requires at least one allow or deny rule".to_owned(),
        });
    }
    for (field, values) in [
        ("allow_labels", config.allow_labels.as_ref()),
        ("allow_attributes", config.allow_attributes.as_ref()),
    ] {
        if let Some(values) = values {
            validate_structured_keys(field, values)?;
        }
    }
    validate_structured_keys("deny_labels", &config.deny_labels)?;
    validate_structured_keys("deny_attributes", &config.deny_attributes)?;
    if config
        .allow_labels
        .as_ref()
        .is_some_and(|allow| !allow.is_disjoint(&config.deny_labels))
        || config
            .allow_attributes
            .as_ref()
            .is_some_and(|allow| !allow.is_disjoint(&config.deny_attributes))
    {
        return Err(ArchiveComponentError::InvalidConfig {
            family: SANITIZER_FAMILY,
            id: ALLOW_DENY_SANITIZER_DESCRIPTOR.id.to_owned(),
            message: "allow and deny key sets must be disjoint".to_owned(),
        });
    }
    Ok(())
}

fn validate_structured_keys(
    field: &str,
    values: &BTreeSet<String>,
) -> Result<(), ArchiveComponentError> {
    if let Some(value) = values
        .iter()
        .find(|value| value.is_empty() || value.trim() != value.as_str() || value.contains('\0'))
    {
        return Err(ArchiveComponentError::InvalidConfig {
            family: SANITIZER_FAMILY,
            id: ALLOW_DENY_SANITIZER_DESCRIPTOR.id.to_owned(),
            message: format!("{field} contains invalid key {value:?}"),
        });
    }
    Ok(())
}

fn is_known_credential_key(value: &str) -> bool {
    matches!(
        value.to_ascii_lowercase().as_str(),
        "authorization"
            | "proxy_authorization"
            | "proxy-authorization"
            | "x_api_key"
            | "x-api-key"
            | "api_key"
            | "api-key"
            | "access_token"
            | "access-token"
            | "client_secret"
            | "client-secret"
            | "password"
    )
}

fn filter_map(
    values: &BTreeMap<String, String>,
    allow: Option<&BTreeSet<String>>,
    deny: &BTreeSet<String>,
) -> BTreeMap<String, String> {
    values
        .iter()
        .filter(|(key, _)| allow.is_none_or(|allowed| allowed.contains(*key)))
        .filter(|(key, _)| !deny.contains(*key))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

fn decode_archive_master_key(value: &str) -> Option<[u8; 32]> {
    let decoded = if let Some(hex) = value.strip_prefix("hex:") {
        decode_hex(hex)?
    } else if let Some(base64) = value.strip_prefix("base64:") {
        base64::engine::general_purpose::STANDARD
            .decode(base64)
            .ok()?
    } else {
        decode_hex(value)?
    };
    if decoded.len() != 32 {
        return None;
    }
    let mut key = [0_u8; 32];
    key.copy_from_slice(&decoded);
    Some(key)
}

fn decode_hex(value: &str) -> Option<Vec<u8>> {
    if !value.len().is_multiple_of(2) {
        return None;
    }
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let high = hex_nibble(pair[0])?;
            let low = hex_nibble(pair[1])?;
            Some((high << 4) | low)
        })
        .collect()
}

const fn hex_nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

/// Strict archive-component registry, validation, or preparation failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArchiveComponentError {
    /// A compiled factory descriptor has an invalid stable ID.
    InvalidFactoryId {
        /// Component family.
        family: &'static str,
        /// Invalid descriptor ID.
        id: String,
    },
    /// Two compiled factories in one family use the same ID.
    DuplicateFactory {
        /// Component family.
        family: &'static str,
        /// Duplicated ID.
        id: String,
    },
    /// The authored ID is absent from this exact runner distribution.
    UnknownFactory {
        /// Component family.
        family: &'static str,
        /// Requested wire ID.
        requested: String,
        /// Deterministic compiled IDs.
        available: Vec<String>,
    },
    /// A selected factory rejected its strict object.
    InvalidConfig {
        /// Component family.
        family: &'static str,
        /// Selected factory ID.
        id: String,
        /// Secret-free diagnostic.
        message: String,
    },
    /// Common archive structure or provider reference is invalid.
    InvalidArchive(String),
    /// Individually valid policies cannot be composed in this product branch.
    IncompatibleSelection(String),
    /// Canonical persistent/invocation identity construction failed.
    Canonical(String),
    /// Side-effectful provider or writer preparation failed.
    Prepare(String),
}

impl Display for ArchiveComponentError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFactoryId { family, id } => {
                write!(formatter, "invalid {family} factory ID {id:?}")
            }
            Self::DuplicateFactory { family, id } => {
                write!(formatter, "duplicate {family} factory ID {id:?}")
            }
            Self::UnknownFactory {
                family,
                requested,
                available,
            } => write!(
                formatter,
                "{family} factory {requested:?} is unavailable; compiled factories: {}",
                available.join(", ")
            ),
            Self::InvalidConfig {
                family,
                id,
                message,
            } => write!(formatter, "invalid {family} {id:?} config: {message}"),
            Self::InvalidArchive(message)
            | Self::IncompatibleSelection(message)
            | Self::Canonical(message)
            | Self::Prepare(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for ArchiveComponentError {}

impl From<ArchiveStoreError> for ArchiveComponentError {
    fn from(error: ArchiveStoreError) -> Self {
        Self::Prepare(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use aiperf_telemetry_archive::{
        Blake3ArchiveKeyProvider, MemoryArchiveObjectStore, NoDurabilityFaults,
    };
    use serde_json::json;

    use super::*;

    #[derive(Debug)]
    struct MemoryStoreProvider;

    impl ArchiveObjectStoreProvider for MemoryStoreProvider {
        fn prepare(
            &self,
            _request: ArchiveObjectStorePrepareRequest<'_>,
        ) -> Result<Arc<dyn ArchiveObjectStore>, ArchiveComponentError> {
            Ok(Arc::new(MemoryArchiveObjectStore::default()))
        }
    }

    #[derive(Debug)]
    struct MemoryKeyResolver;

    impl ArchiveKeyProviderResolver for MemoryKeyResolver {
        fn resolve(
            &self,
            _secret_reference: &str,
        ) -> Result<Arc<dyn ArchiveKeyProvider>, ArchiveComponentError> {
            Blake3ArchiveKeyProvider::new("secret_provider", [7; 32])
                .map(|provider| Arc::new(provider) as Arc<dyn ArchiveKeyProvider>)
                .map_err(|error| ArchiveComponentError::Prepare(error.to_string()))
        }
    }

    fn collect_archive() -> TelemetryArchiveSpecV2 {
        serde_json::from_value(json!({
            "target": "file:///tmp/aiperf-archive-components",
            "local_spool": "/tmp/aiperf-archive-components-spool",
            "spool_quota_bytes": 1000000,
            "spool_quota_files": 1000,
            "required": true,
            "writer": {"type": "parquet_archive_v1", "config": {}},
            "store_access": {"type": "local_filesystem", "config": {}},
            "rotation": {"type": "rows_bytes_age", "config": {}},
            "admission": {"type": "primary_durable", "config": {}},
            "recovery": {"type": "create_new", "config": {}},
            "archive_key": {"type": "secret_provider", "config": {"id": "archive-identity"}},
            "enrichers": [{"type": "static_labels", "config": {"attributes": {"cluster": "lab-a"}}}],
            "sanitizers": [{"type": "allow_deny_keys", "config": {"deny_labels": ["tenant"]}}],
            "raw_body": {"type": "none", "config": {}}
        }))
        .unwrap()
    }

    #[test]
    fn stock_registry_validates_and_prepares_every_collect_family() {
        let validated = TelemetryArchiveComponentRegistries::stock()
            .validate_collect(
                collect_archive(),
                ArchiveCollectionPlacement::StandalonePrimary,
            )
            .unwrap();
        assert_eq!(
            validated
                .persistent_component_identities()
                .iter()
                .map(|identity| identity.factory_id)
                .collect::<Vec<_>>(),
            vec![
                "parquet_archive_v1",
                "rows_bytes_age",
                "primary_durable",
                "secret_provider",
                "static_labels",
                "baseline_credentials",
                "allow_deny_keys",
                "none",
            ]
        );
        let prepared = validated
            .prepare(ArchiveCollectComponentPrepareContext {
                store_provider: &MemoryStoreProvider,
                key_resolver: &MemoryKeyResolver,
                durability_faults: Arc::new(NoDurabilityFaults),
            })
            .unwrap();
        assert_eq!(
            prepared.recovery.operation(),
            ArchiveRecoveryOperation::CreateNew
        );
        assert!(!prepared.raw_body.retains_exact_body());
        assert_eq!(prepared.enrichers.len(), 1);
    }

    #[test]
    fn unknown_raw_policy_unknown_fields_and_target_mismatch_fail_closed() {
        let registries = TelemetryArchiveComponentRegistries::stock();
        let mut value = serde_json::to_value(collect_archive_for_value()).unwrap();
        value["raw_body"]["type"] = json!("plaintext");
        let archive: TelemetryArchiveSpecV2 = serde_json::from_value(value).unwrap();
        let error = registries
            .validate_collect(archive, ArchiveCollectionPlacement::StandalonePrimary)
            .unwrap_err()
            .to_string();
        assert!(error.contains("raw_body"), "{error}");
        assert!(error.contains("none"), "{error}");

        let mut value = serde_json::to_value(collect_archive_for_value()).unwrap();
        value["writer"]["config"] = json!({"compression": "zstd"});
        let archive: TelemetryArchiveSpecV2 = serde_json::from_value(value).unwrap();
        assert!(
            registries
                .validate_collect(archive, ArchiveCollectionPlacement::StandalonePrimary)
                .is_err()
        );

        let mut value = serde_json::to_value(collect_archive_for_value()).unwrap();
        value["store_access"]["type"] = json!("object_store");
        let archive: TelemetryArchiveSpecV2 = serde_json::from_value(value).unwrap();
        assert!(
            registries
                .validate_collect(archive, ArchiveCollectionPlacement::StandalonePrimary)
                .is_err()
        );
    }

    #[test]
    fn sync_preparation_contains_no_persistent_writer_family() {
        let sync: TelemetryArchiveSyncSpecV2 = serde_json::from_value(json!({
            "archive_id": "018f84a7-1f3c-7c21-8be2-7e8dbf9536b1",
            "target": "s3://benchmarks/archive-id/",
            "local_spool": "/tmp/aiperf-archive-components-spool",
            "store_access": {"type": "object_store", "config": {"credential_provider": "archive-store"}},
            "recovery": {"type": "finalize_remote", "config": {}},
            "archive_key": {"type": "secret_provider", "config": {"id": "archive-identity"}}
        }))
        .unwrap();
        let prepared = TelemetryArchiveComponentRegistries::stock()
            .validate_sync(sync)
            .unwrap()
            .prepare(ArchiveSyncComponentPrepareContext {
                store_provider: &MemoryStoreProvider,
                key_resolver: &MemoryKeyResolver,
            })
            .unwrap();
        assert_eq!(
            prepared.recovery.operation(),
            ArchiveRecoveryOperation::FinalizeRemote
        );
        assert!(
            prepared
                .invocation_component_identities
                .iter()
                .all(|identity| identity.family != WRITER_FAMILY)
        );
    }

    #[test]
    fn exact_resume_requires_and_exposes_authored_archive_and_prior_claim_identity() {
        let registries = TelemetryArchiveComponentRegistries::stock();
        let mut missing = collect_archive_for_value();
        missing["recovery"] = json!({"type": "exact_resume", "config": {}});
        let missing: TelemetryArchiveSpecV2 = serde_json::from_value(missing).unwrap();
        let error = registries
            .validate_collect(missing, ArchiveCollectionPlacement::StandalonePrimary)
            .unwrap_err()
            .to_string();
        assert!(error.contains("archive_id"), "{error}");

        let authored_uuid = Uuid::parse_str("018f84a7-1f3c-7c21-8be2-7e8dbf9536b1").unwrap();
        let expected = ArchiveId::new(*authored_uuid.as_bytes()).unwrap();
        let prior_claim = WriterClaimId::from_digest(Digest::from_bytes([0x77; 32]));
        let mut exact = collect_archive_for_value();
        exact["recovery"] = json!({
            "type": "exact_resume",
            "config": {
                "archive_id": authored_uuid,
                "prior_claim_id": prior_claim.to_hex()
            }
        });
        let exact: TelemetryArchiveSpecV2 = serde_json::from_value(exact).unwrap();
        let validated = registries
            .validate_collect(exact, ArchiveCollectionPlacement::StandalonePrimary)
            .unwrap();
        assert_eq!(validated.recovery.expected_archive_id(), Some(expected));
        assert_eq!(
            validated.recovery.expected_prior_claim_id(),
            Some(prior_claim)
        );
    }

    #[test]
    fn mandatory_baseline_precedes_authored_allow_deny_policy() {
        use aiperf_prometheus::SemanticType;

        let labels = BTreeMap::from([
            ("authorization".to_owned(), "secret".to_owned()),
            ("instance".to_owned(), "node-a".to_owned()),
            ("tenant".to_owned(), "private".to_owned()),
        ]);
        let attributes = BTreeMap::from([
            ("cluster".to_owned(), "lab-a".to_owned()),
            ("password".to_owned(), "secret".to_owned()),
        ]);
        let chain = ArchiveSanitizerChain::new(vec![Arc::new(AllowDenyKeySanitizer {
            allow_labels: None,
            deny_labels: BTreeSet::from(["tenant".to_owned()]),
            allow_attributes: None,
            deny_attributes: BTreeSet::new(),
        })]);
        let sanitized = chain
            .sanitize_sample(ArchiveSampleView {
                source_id: "node-a",
                metric_family: "requests_total",
                semantic_type: SemanticType::Counter,
                labels: &labels,
                attributes: &attributes,
            })
            .unwrap();
        assert_eq!(
            sanitized.labels,
            BTreeMap::from([("instance".to_owned(), "node-a".to_owned())])
        );
        assert_eq!(
            sanitized.attributes,
            BTreeMap::from([("cluster".to_owned(), "lab-a".to_owned())])
        );
    }

    #[test]
    fn stock_providers_are_local_only_and_key_inputs_are_exact() {
        let temporary = tempfile::tempdir().unwrap();
        let target: NormalizedArchiveUri = Url::from_directory_path(temporary.path())
            .unwrap()
            .to_string()
            .parse()
            .unwrap();
        let config = CanonicalJsonValue::Object(BTreeMap::new());
        let store = NativeArchiveObjectStoreProvider
            .prepare(ArchiveObjectStorePrepareRequest {
                access_factory_id: "local_filesystem",
                target: &target,
                canonical_config: &config,
                credential_provider: None,
            })
            .unwrap();
        store.capabilities().require_authoritative().unwrap();

        let object_target: NormalizedArchiveUri = "s3://benchmarks/archive-id/".parse().unwrap();
        let error = NativeArchiveObjectStoreProvider
            .prepare(ArchiveObjectStorePrepareRequest {
                access_factory_id: "object_store",
                target: &object_target,
                canonical_config: &config,
                credential_provider: Some("archive-store"),
            })
            .unwrap_err()
            .to_string();
        assert!(error.contains("injected object-store provider"), "{error}");

        let resolver = EnvironmentArchiveKeyProviderResolver::default();
        assert_eq!(
            resolver.variable_name("archive-identity").unwrap(),
            "AIPERF_ARCHIVE_KEY_ARCHIVE_IDENTITY"
        );
        assert_eq!(decode_archive_master_key(&"07".repeat(32)), Some([7; 32]));
        assert_eq!(
            decode_archive_master_key(&format!(
                "base64:{}",
                base64::engine::general_purpose::STANDARD.encode([9; 32])
            )),
            Some([9; 32])
        );
        assert_eq!(decode_archive_master_key("redacted-invalid"), None);
    }

    fn collect_archive_for_value() -> serde_json::Value {
        json!({
            "target": "file:///tmp/aiperf-archive-components",
            "local_spool": "/tmp/aiperf-archive-components-spool",
            "spool_quota_bytes": 1000000,
            "spool_quota_files": 1000,
            "required": true,
            "writer": {"type": "parquet_archive_v1", "config": {}},
            "store_access": {"type": "local_filesystem", "config": {}},
            "rotation": {"type": "rows_bytes_age", "config": {}},
            "admission": {"type": "primary_durable", "config": {}},
            "recovery": {"type": "create_new", "config": {}},
            "archive_key": {"type": "secret_provider", "config": {"id": "archive-identity"}},
            "enrichers": [],
            "sanitizers": [],
            "raw_body": {"type": "none", "config": {}}
        })
    }
}
