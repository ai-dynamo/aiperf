// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opaque validated values and the native endpoint compatibility trait.
//!
//! A factory's validated configuration stays inside the factory. What crosses
//! the boundary is a prepared handle: the receipt the host reasons about, plus a
//! plugin-owned trait object the host only ever calls through. The host never
//! downcasts one of these and never inspects the plugin's configuration type —
//! `Any`/`TypeId` are not stable across separately compiled artifacts, so a
//! downcast that appeared to work would be undefined behavior waiting on a
//! rebuild.
//!
//! Every handle is `Rc`-shaped rather than `Box`-shaped. Plugin libraries stay
//! resident for the whole process and a prepared value is shared by every worker
//! placement, so cloning a handle must be a refcount bump rather than a
//! reconstruction.

use std::rc::Rc;

use aiperf_core::endpoint::{Handle, Overrides, SegmentReader};

use crate::capture::PreparedExporterV1;
use crate::category::CategoryError;
use crate::factory::FactoryValidationReceiptV1;
use crate::id::RegistryId;
use crate::transport::{RequestTransportExecution, TransportExecutionShapeV1};

/// The native endpoint compatibility trait an endpoint plugin implements.
///
/// The formatter is given the three boundary facts and nothing else: the frozen
/// segments named by dense handles, the read view that resolves them to exact
/// pre-serialized wire bytes, and the authored per-dispatch top-level fields.
/// It does not see the segment arena, its payload kinds, or the dataset error
/// taxonomy.
pub trait Endpoint {
    /// The registered identifier of this endpoint dialect.
    fn id(&self) -> &RegistryId;

    /// Compose one request body from frozen segments and per-dispatch overrides.
    fn format_payload(
        &self,
        segments: &dyn SegmentReader,
        handles: &[Handle],
        overrides: &Overrides,
    ) -> Result<Vec<u8>, CategoryError>;
}

/// An endpoint validated by the exact factory that authored its configuration.
#[derive(Clone)]
pub struct PreparedEndpoint {
    receipt: FactoryValidationReceiptV1,
    endpoint: Rc<dyn Endpoint>,
}

impl PreparedEndpoint {
    /// Bind one validated endpoint to its receipt.
    pub fn new(receipt: FactoryValidationReceiptV1, endpoint: Rc<dyn Endpoint>) -> Self {
        Self { receipt, endpoint }
    }

    /// The receipt describing what was validated.
    pub const fn receipt(&self) -> &FactoryValidationReceiptV1 {
        &self.receipt
    }

    /// The endpoint the host calls through.
    pub fn endpoint(&self) -> Rc<dyn Endpoint> {
        self.endpoint.clone()
    }
}

/// A transport validated by the exact factory that authored its configuration.
///
/// The declared execution shape is stored beside the handle so the host places
/// the transport without calling into it: a placement decision that required a
/// plugin call could not be made before the plugin is loadable.
#[derive(Clone)]
pub struct PreparedTransport {
    receipt: FactoryValidationReceiptV1,
    shape: TransportExecutionShapeV1,
    execution: Option<Rc<dyn RequestTransportExecution>>,
}

impl PreparedTransport {
    /// Bind one validated request transport to its receipt.
    pub fn request(
        receipt: FactoryValidationReceiptV1,
        execution: Rc<dyn RequestTransportExecution>,
    ) -> Self {
        Self {
            receipt,
            shape: TransportExecutionShapeV1::Request,
            execution: Some(execution),
        }
    }

    /// Bind one validated direct transport to its receipt.
    ///
    /// A direct transport contributes no [`RequestTransportExecution`]: it drives
    /// its own execution through the narrow `aiperf_core::services` traits.
    pub fn direct(receipt: FactoryValidationReceiptV1) -> Self {
        Self {
            receipt,
            shape: TransportExecutionShapeV1::Direct,
            execution: None,
        }
    }

    /// The receipt describing what was validated.
    pub const fn receipt(&self) -> &FactoryValidationReceiptV1 {
        &self.receipt
    }

    /// The single execution shape this transport declared.
    pub const fn shape(&self) -> TransportExecutionShapeV1 {
        self.shape
    }

    /// The worker-kernel execution, present only for a request transport.
    pub fn execution(&self) -> Option<Rc<dyn RequestTransportExecution>> {
        self.execution.clone()
    }
}

/// An exporter validated by the exact factory that authored its configuration.
#[derive(Clone)]
pub struct PreparedExporter {
    receipt: FactoryValidationReceiptV1,
    exporter: Rc<dyn PreparedExporterV1>,
}

impl PreparedExporter {
    /// Bind one validated exporter to its receipt.
    pub fn new(receipt: FactoryValidationReceiptV1, exporter: Rc<dyn PreparedExporterV1>) -> Self {
        Self { receipt, exporter }
    }

    /// The receipt describing what was validated.
    pub const fn receipt(&self) -> &FactoryValidationReceiptV1 {
        &self.receipt
    }

    /// The exporter the host invokes after the report is finalized.
    pub fn exporter(&self) -> Rc<dyn PreparedExporterV1> {
        self.exporter.clone()
    }
}
