// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The receipt a factory returns beside its opaque validated value.
//!
//! Validated configuration is owned by the exact factory that produced it and is
//! opaque to the host, so the host cannot inspect it to decide whether a later
//! run is the same run. The receipt is what it inspects instead: it binds the
//! selected category, the canonical factory identifier, the descriptor bytes,
//! both configuration digests, the host capabilities granted, and the captures
//! the factory committed to.
//!
//! Two digests are carried rather than one because they answer different
//! questions. The **authored** digest covers the configuration exactly as the
//! user wrote it, so a reformatted-but-equivalent document is a different
//! authored digest. The **semantic** digest covers the factory's normalized
//! interpretation, so that same reformatting is the same semantic digest. A
//! sweep that varies formatting alone is detectable as identical work; a sweep
//! that varies meaning is not.
//!
//! [`FactoryValidationReceiptV1::FIELD_KEYS`] is the complete key set, in the
//! order a canonical rendering emits it. A field added without extending that
//! constant is a field no golden test covers.

use crate::capture::ExporterCaptureRequirementsV1;
use crate::id::RegistryId;
use crate::validation::{ContentDigest, HostResourceSetV1, PluginCategory, ValidationError};

/// The complete result of validating one factory's authored configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FactoryValidationReceiptV1 {
    category: PluginCategory,
    factory_id: RegistryId,
    descriptor_digest: ContentDigest,
    authored_config_digest: ContentDigest,
    semantic_config_digest: ContentDigest,
    host_resources: HostResourceSetV1,
    capture_requirements: ExporterCaptureRequirementsV1,
}

impl FactoryValidationReceiptV1 {
    /// The receipt's complete field key set, in canonical rendering order.
    pub const FIELD_KEYS: &'static [&'static str] = &[
        "category",
        "factory_id",
        "descriptor_digest",
        "authored_config_digest",
        "semantic_config_digest",
        "host_resources",
        "capture_requirements",
    ];

    /// Build a receipt, normalizing the host-resource request.
    ///
    /// `capture_requirements` is meaningful only for
    /// [`PluginCategory::Exporter`]; other categories pass the default, which is
    /// the finalized report alone.
    pub fn new(
        category: PluginCategory,
        factory_id: RegistryId,
        descriptor_digest: ContentDigest,
        authored_config_digest: ContentDigest,
        semantic_config_digest: ContentDigest,
        host_resources: HostResourceSetV1,
        capture_requirements: ExporterCaptureRequirementsV1,
    ) -> Self {
        Self {
            category,
            factory_id,
            descriptor_digest,
            authored_config_digest,
            semantic_config_digest,
            host_resources,
            capture_requirements,
        }
    }

    /// The category position the factory occupies.
    pub const fn category(&self) -> PluginCategory {
        self.category
    }

    /// The canonical registered identifier of the factory.
    pub const fn factory_id(&self) -> &RegistryId {
        &self.factory_id
    }

    /// Digest of the descriptor bytes the host validated.
    pub const fn descriptor_digest(&self) -> ContentDigest {
        self.descriptor_digest
    }

    /// Digest of the configuration exactly as authored.
    pub const fn authored_config_digest(&self) -> ContentDigest {
        self.authored_config_digest
    }

    /// Digest of the factory's normalized interpretation of that configuration.
    pub const fn semantic_config_digest(&self) -> ContentDigest {
        self.semantic_config_digest
    }

    /// The sorted host capabilities the factory was granted.
    pub const fn host_resources(&self) -> &HostResourceSetV1 {
        &self.host_resources
    }

    /// The sorted captures the factory committed to.
    pub const fn capture_requirements(&self) -> &ExporterCaptureRequirementsV1 {
        &self.capture_requirements
    }

    /// Refuse a receipt produced under a different category than the caller's.
    pub fn expect_category(&self, expected: PluginCategory) -> Result<(), ValidationError> {
        if self.category == expected {
            Ok(())
        } else {
            Err(ValidationError::CategoryMismatch {
                expected,
                found: self.category,
            })
        }
    }
}
