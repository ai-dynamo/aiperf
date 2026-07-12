// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Trait-backed Rust host executors and logical route registry.
//!
//! Provider operations identify a registered semantic operation and logical
//! service; they never contain a transport method, caller-selected upstream
//! URL, or credential. Factories advertise executable schemas, construct
//! worker-local executors, and enter a deterministic frozen registry. Future
//! operation families can implement the same traits without adding a closed
//! mode enum to the runtime.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::rc::Rc;

use aiperf_accuracy::HostOperationUsage;
use anyhow::{Context, Result, anyhow, ensure};
use async_trait::async_trait;
use serde_json::Value;

use super::ledger::HostTerminalClass;
use super::retry::{InferenceTransportAttempt, OperationCancellation};
use crate::scheduled::ScheduledRuntime;

/// Per-run host executor context supplied after the scheduler is constructed.
///
/// Non-inference executors may ignore the scheduler. Inference factories call
/// [`require_scheduled`](Self::require_scheduled) and therefore cannot be
/// prepared as a descriptor-only capability.
#[derive(Clone, Default)]
pub struct HostExecutorRuntime {
    scheduled: Option<Rc<ScheduledRuntime>>,
}

impl HostExecutorRuntime {
    /// Build a live runtime context for an executing evaluation workload.
    pub fn scheduled(runtime: Rc<ScheduledRuntime>) -> Self {
        Self {
            scheduled: Some(runtime),
        }
    }

    /// Require the ordinary scheduled runtime for an inference executor.
    pub fn require_scheduled(&self) -> Result<Rc<ScheduledRuntime>> {
        self.scheduled
            .clone()
            .ok_or_else(|| anyhow!("host executor requires a live scheduled runtime"))
    }
}

impl fmt::Debug for HostExecutorRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HostExecutorRuntime")
            .field("scheduled", &self.scheduled.is_some())
            .finish()
    }
}

/// Stable family label used for capability inventory and report grouping.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HostOperationFamily(String);

impl HostOperationFamily {
    /// Build a validated open family identifier.
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        validate_open_id(&value, "host operation family")?;
        Ok(Self(value))
    }

    /// Borrow the validated registry key.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for HostOperationFamily {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Open semantic operation identifier, for example `model.generate`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegisteredOperationId(String);

impl RegisteredOperationId {
    /// Build a validated dotted operation identifier.
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        validate_operation_id(&value)?;
        Ok(Self(value))
    }

    /// Borrow the validated registry key.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for RegisteredOperationId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Immutable schema and capability inventory for one executable host operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostOperationDescriptor {
    /// Open semantic operation ID.
    pub operation_id: RegisteredOperationId,
    /// Executor family.
    pub family: HostOperationFamily,
    /// Versioned request schema fingerprint.
    pub request_schema_fingerprint: String,
    /// Versioned response/event schema fingerprint.
    pub response_schema_fingerprint: String,
    /// Versioned incremental-event schema fingerprint.
    ///
    /// This is present exactly when [`Self::true_streaming`] is true. A
    /// terminal-only adapter must not advertise an unused stream schema.
    pub stream_schema_fingerprint: Option<String>,
    /// Whether the executor emits real incremental typed deltas.
    pub true_streaming: bool,
    /// Maximum request payload bytes after canonical serialization.
    pub max_request_bytes: usize,
    /// Maximum terminal payload bytes after canonical serialization.
    pub max_response_bytes: usize,
    /// Endpoint dialect capabilities required by this operation.
    pub endpoint_capabilities: BTreeSet<String>,
}

impl HostOperationDescriptor {
    /// Validate bounded sizes, fingerprints, and endpoint capability IDs.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.max_request_bytes > 0,
            "operation {} request bound must be positive",
            self.operation_id
        );
        ensure!(
            self.max_response_bytes > 0,
            "operation {} response bound must be positive",
            self.operation_id
        );
        validate_sha256(
            &self.request_schema_fingerprint,
            "request schema fingerprint",
        )?;
        validate_sha256(
            &self.response_schema_fingerprint,
            "response schema fingerprint",
        )?;
        ensure!(
            self.true_streaming == self.stream_schema_fingerprint.is_some(),
            "operation {} streaming flag and stream schema must agree",
            self.operation_id
        );
        if let Some(fingerprint) = &self.stream_schema_fingerprint {
            validate_sha256(fingerprint, "stream schema fingerprint")?;
        }
        for capability in &self.endpoint_capabilities {
            validate_open_id(capability, "endpoint capability")?;
        }
        Ok(())
    }
}

/// Secret-free provider operation ready for a Rust host executor.
#[derive(Debug, Clone, PartialEq)]
pub struct HostOperationEnvelope {
    /// Globally unique logical operation ID.
    pub operation_id: String,
    /// Parent evaluation unit.
    pub unit_id: String,
    /// Parent case occurrence.
    pub case_id: String,
    /// Parent semantic attempt.
    pub semantic_attempt_id: String,
    /// Logical call identity.
    pub logical_call_id: String,
    /// Logical service resolved through [`EvaluationRouteTable`].
    pub service_id: String,
    /// Registered semantic operation.
    pub semantic_operation_id: RegisteredOperationId,
    /// Provider purpose label used for safe reporting.
    pub purpose: String,
    /// Schema-validated typed operation payload.
    pub payload: Value,
    /// Whether the payload is restricted judge/verifier material.
    pub restricted: bool,
    /// Whether the provider requested typed incremental events.
    pub stream: bool,
}

/// One typed delta emitted by a host executor.
#[derive(Debug, Clone, PartialEq)]
pub struct HostExecutionDelta {
    /// Monotonic zero-based delta ordinal for this operation.
    pub ordinal: usize,
    /// Schema-validated normalized event payload.
    pub payload: Value,
}

/// Exactly one terminal returned by a host executor.
#[derive(Debug, Clone, PartialEq)]
pub struct HostExecutionTerminal {
    /// Rust-owned terminal class.
    pub class: HostTerminalClass,
    /// Schema-validated normalized terminal payload.
    pub payload: Value,
    /// Rust-authoritative operation usage.
    pub usage: HostOperationUsage,
    /// Whether registered route policy may retry this terminal.
    pub retryable: bool,
    /// Exact Rust transport-attempt lineage. Queued cancellation is the only
    /// logical terminal that legitimately contains no transport attempt.
    pub transport_attempts: Vec<InferenceTransportAttempt>,
}

/// Typed streaming event sink supplied by the evaluation workload.
#[async_trait(?Send)]
pub trait HostExecutionEventSink {
    /// Publish one ordered normalized delta.
    async fn publish(&self, delta: HostExecutionDelta) -> Result<()>;
}

/// Worker-local executor for one registered host operation family.
#[async_trait(?Send)]
pub trait HostOperationExecutor {
    /// Execute one admitted operation to exactly one terminal.
    async fn execute(
        &self,
        operation: &HostOperationEnvelope,
        events: &dyn HostExecutionEventSink,
        cancellation: OperationCancellation,
    ) -> Result<HostExecutionTerminal>;
}

/// Pure request/response schema validator owned by an executor factory.
pub trait HostOperationSchemaValidator {
    /// Validate the provider request without performing a host effect.
    fn validate_request(&self, payload: &Value) -> Result<()>;
    /// Validate one normalized incremental event.
    fn validate_stream(&self, payload: &Value) -> Result<()>;
    /// Validate a normalized terminal/delta payload before it crosses the pipe.
    fn validate_response(&self, payload: &Value) -> Result<()>;
}

/// Factory for worker-local executable host operation implementations.
pub trait HostOperationExecutorFactory {
    /// Immutable capability/schema descriptor.
    fn descriptor(&self) -> &HostOperationDescriptor;
    /// Side-effect-free schema validator.
    fn validator(&self) -> &dyn HostOperationSchemaValidator;
    /// Prepare one executor for one logical route.
    fn prepare(
        &self,
        runtime: &HostExecutorRuntime,
        route: &EvaluationRoute,
    ) -> Result<Rc<dyn HostOperationExecutor>>;
}

/// Marker seam for inference operation executors.
pub trait InferenceHostExecutor: HostOperationExecutor {}

/// Marker seam for immutable asset executors.
pub trait AssetHostExecutor: HostOperationExecutor {}

/// Marker seam for sandbox lifecycle executors.
pub trait SandboxHostExecutor: HostOperationExecutor {}

/// Marker seam for mediated process executors.
pub trait ProcessHostExecutor: HostOperationExecutor {}

/// Marker seam for audited stdio-only MCP executors.
pub trait McpHostExecutor: HostOperationExecutor {}

/// Authenticated local compatibility ingress backed by the same executor registry.
#[async_trait(?Send)]
pub trait CompatibilityProxyIngress {
    /// Scoped local locator. It must never be an upstream route.
    fn local_locator(&self) -> &str;
    /// Stop accepting new local operations and drain/revoke every grant.
    async fn shutdown(&self) -> Result<()>;
}

/// Secret-free logical route selected by authored runner configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluationRoute {
    /// Provider-visible logical service ID.
    pub service_id: String,
    /// Safe semantic purpose.
    pub purpose: String,
    /// Authored model alias.
    pub model: String,
    /// Endpoint profile selected and prepared by Rust.
    pub endpoint_profile: String,
    /// Credential-free prepared endpoint identity digest.
    pub prepared_identity_sha256: String,
    /// Executable endpoint capability inventory.
    pub endpoint_capabilities: BTreeSet<String>,
}

impl EvaluationRoute {
    /// Validate route identity without resolving a URL or credential.
    pub fn validate(&self) -> Result<()> {
        validate_open_id(&self.service_id, "logical service ID")?;
        validate_open_id(&self.purpose, "logical service purpose")?;
        ensure!(
            !self.model.trim().is_empty(),
            "route model must not be empty"
        );
        validate_open_id(&self.endpoint_profile, "endpoint profile")?;
        validate_sha256(&self.prepared_identity_sha256, "prepared endpoint identity")?;
        for capability in &self.endpoint_capabilities {
            validate_open_id(capability, "endpoint capability")?;
        }
        Ok(())
    }
}

/// Frozen logical service table. No entry stores an endpoint URL or credential.
#[derive(Debug, Clone)]
pub struct EvaluationRouteTable {
    routes: BTreeMap<String, EvaluationRoute>,
}

impl EvaluationRouteTable {
    /// Freeze validated routes in deterministic service-ID order.
    pub fn new(routes: impl IntoIterator<Item = EvaluationRoute>) -> Result<Self> {
        let mut table = BTreeMap::new();
        for route in routes {
            route.validate()?;
            let service_id = route.service_id.clone();
            ensure!(
                table.insert(service_id.clone(), route).is_none(),
                "duplicate evaluation logical service {service_id:?}"
            );
        }
        ensure!(
            !table.is_empty(),
            "evaluation route table must not be empty"
        );
        Ok(Self { routes: table })
    }

    /// Resolve one provider logical service without accepting a caller URL.
    pub fn resolve(&self, service_id: &str) -> Result<&EvaluationRoute> {
        self.routes
            .get(service_id)
            .ok_or_else(|| anyhow!("unknown evaluation logical service {service_id:?}"))
    }

    /// Routes in deterministic service-ID order.
    pub fn routes(&self) -> impl ExactSizeIterator<Item = &EvaluationRoute> {
        self.routes.values()
    }
}

/// Mutable deterministic host executor registry builder.
#[derive(Default)]
pub struct HostExecutorRegistryBuilder {
    factories: BTreeMap<RegisteredOperationId, Rc<dyn HostOperationExecutorFactory>>,
}

impl HostExecutorRegistryBuilder {
    /// Register one executable operation; duplicate IDs fail transactionally.
    pub fn register(&mut self, factory: Rc<dyn HostOperationExecutorFactory>) -> Result<()> {
        factory.descriptor().validate()?;
        let operation_id = factory.descriptor().operation_id.clone();
        ensure!(
            !self.factories.contains_key(&operation_id),
            "duplicate host operation executor {operation_id}"
        );
        self.factories.insert(operation_id, factory);
        Ok(())
    }

    /// Freeze the registry after validating each descriptor/factory identity.
    pub fn freeze(self) -> Result<HostExecutorRegistry> {
        ensure!(
            !self.factories.is_empty(),
            "host executor registry must contain at least one executable operation"
        );
        Ok(HostExecutorRegistry {
            factories: self.factories,
        })
    }
}

/// Frozen executable host executor registry.
#[derive(Clone)]
pub struct HostExecutorRegistry {
    factories: BTreeMap<RegisteredOperationId, Rc<dyn HostOperationExecutorFactory>>,
}

impl HostExecutorRegistry {
    /// Resolve one registered operation or fail before provider execution.
    pub fn factory(
        &self,
        operation_id: &RegisteredOperationId,
    ) -> Result<&Rc<dyn HostOperationExecutorFactory>> {
        self.factories
            .get(operation_id)
            .ok_or_else(|| anyhow!("no executable host operation adapter for {operation_id}"))
    }

    /// Validate and prepare one provider operation against its logical route.
    pub fn prepare(
        &self,
        operation: &HostOperationEnvelope,
        routes: &EvaluationRouteTable,
        runtime: &HostExecutorRuntime,
    ) -> Result<Rc<dyn HostOperationExecutor>> {
        let factory = self.factory(&operation.semantic_operation_id)?;
        let descriptor = factory.descriptor();
        ensure!(
            !operation.stream || descriptor.true_streaming,
            "operation {} requested true streaming from a terminal-only adapter",
            operation.semantic_operation_id
        );
        let request_bytes = serde_json::to_vec(&operation.payload)
            .context("serializing evaluator host operation for its size bound")?;
        ensure!(
            request_bytes.len() <= descriptor.max_request_bytes,
            "operation {} request exceeds its {} byte bound",
            operation.semantic_operation_id,
            descriptor.max_request_bytes
        );
        factory.validator().validate_request(&operation.payload)?;
        let route = routes.resolve(&operation.service_id)?;
        let missing = descriptor
            .endpoint_capabilities
            .difference(&route.endpoint_capabilities)
            .cloned()
            .collect::<Vec<_>>();
        ensure!(
            missing.is_empty(),
            "route {:?} lacks operation {} endpoint capabilities: {}",
            route.service_id,
            operation.semantic_operation_id,
            missing.join(", ")
        );
        factory.prepare(runtime, route)
    }

    /// Deterministic executable capability inventory.
    pub fn descriptors(&self) -> impl ExactSizeIterator<Item = &HostOperationDescriptor> {
        self.factories.values().map(|factory| factory.descriptor())
    }

    /// Validate and bound one normalized incremental event.
    pub fn validate_stream(
        &self,
        operation_id: &RegisteredOperationId,
        payload: &Value,
    ) -> Result<()> {
        let factory = self.factory(operation_id)?;
        ensure!(
            factory.descriptor().true_streaming,
            "operation {operation_id} emitted a stream event from a terminal-only adapter"
        );
        validate_payload_bound(
            operation_id,
            payload,
            factory.descriptor().max_response_bytes,
            "stream event",
        )?;
        factory.validator().validate_stream(payload)
    }

    /// Validate and bound one completed normalized terminal result.
    pub fn validate_response(
        &self,
        operation_id: &RegisteredOperationId,
        payload: &Value,
    ) -> Result<()> {
        let factory = self.factory(operation_id)?;
        validate_payload_bound(
            operation_id,
            payload,
            factory.descriptor().max_response_bytes,
            "terminal result",
        )?;
        factory.validator().validate_response(payload)
    }
}

fn validate_payload_bound(
    operation_id: &RegisteredOperationId,
    payload: &Value,
    maximum: usize,
    kind: &str,
) -> Result<()> {
    let encoded = serde_json::to_vec(payload)
        .with_context(|| format!("serializing evaluator {kind} for its size bound"))?;
    ensure!(
        encoded.len() <= maximum,
        "operation {operation_id} {kind} exceeds its {maximum} byte bound"
    );
    Ok(())
}

fn validate_open_id(value: &str, field: &str) -> Result<()> {
    ensure!(
        !value.is_empty() && value.len() <= 128,
        "{field} must contain 1..=128 bytes"
    );
    let mut bytes = value.bytes();
    ensure!(
        bytes.next().is_some_and(|byte| byte.is_ascii_lowercase()),
        "{field} must begin with a lowercase ASCII letter"
    );
    ensure!(
        bytes.all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_'),
        "{field} must match [a-z][a-z0-9_]*"
    );
    Ok(())
}

fn validate_operation_id(value: &str) -> Result<()> {
    ensure!(
        !value.is_empty() && value.len() <= 192,
        "semantic operation ID must contain 1..=192 bytes"
    );
    let segments = value.split('.').collect::<Vec<_>>();
    ensure!(
        segments.len() >= 2,
        "semantic operation ID must have at least two dotted segments"
    );
    for segment in segments {
        validate_open_id(segment, "semantic operation ID segment")?;
    }
    Ok(())
}

fn validate_sha256(value: &str, field: &str) -> Result<()> {
    ensure!(
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{field} must be 64 lowercase hexadecimal digits"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    struct ObjectValidator;

    impl HostOperationSchemaValidator for ObjectValidator {
        fn validate_request(&self, payload: &Value) -> Result<()> {
            ensure!(payload.is_object(), "fixture request must be an object");
            Ok(())
        }

        fn validate_stream(&self, payload: &Value) -> Result<()> {
            ensure!(
                payload.is_object(),
                "fixture stream event must be an object"
            );
            Ok(())
        }

        fn validate_response(&self, payload: &Value) -> Result<()> {
            ensure!(payload.is_object(), "fixture response must be an object");
            Ok(())
        }
    }

    struct FixtureExecutor;

    #[async_trait(?Send)]
    impl HostOperationExecutor for FixtureExecutor {
        async fn execute(
            &self,
            _operation: &HostOperationEnvelope,
            _events: &dyn HostExecutionEventSink,
            _cancellation: OperationCancellation,
        ) -> Result<HostExecutionTerminal> {
            Ok(HostExecutionTerminal {
                class: HostTerminalClass::Completed,
                payload: serde_json::json!({"ok": true}),
                usage: HostOperationUsage::default(),
                retryable: false,
                transport_attempts: Vec::new(),
            })
        }
    }

    struct FixtureFactory {
        descriptor: HostOperationDescriptor,
        prepared: Cell<usize>,
    }

    impl HostOperationExecutorFactory for FixtureFactory {
        fn descriptor(&self) -> &HostOperationDescriptor {
            &self.descriptor
        }

        fn validator(&self) -> &dyn HostOperationSchemaValidator {
            &ObjectValidator
        }

        fn prepare(
            &self,
            _runtime: &HostExecutorRuntime,
            _route: &EvaluationRoute,
        ) -> Result<Rc<dyn HostOperationExecutor>> {
            self.prepared.set(self.prepared.get() + 1);
            Ok(Rc::new(FixtureExecutor))
        }
    }

    fn factory(operation: &str, streaming: bool) -> Rc<FixtureFactory> {
        Rc::new(FixtureFactory {
            descriptor: HostOperationDescriptor {
                operation_id: RegisteredOperationId::new(operation).unwrap(),
                family: HostOperationFamily::new("inference").unwrap(),
                request_schema_fingerprint: "a".repeat(64),
                response_schema_fingerprint: "b".repeat(64),
                stream_schema_fingerprint: streaming.then(|| "d".repeat(64)),
                true_streaming: streaming,
                max_request_bytes: 1_024,
                max_response_bytes: 1_024,
                endpoint_capabilities: BTreeSet::from(["chat".into()]),
            },
            prepared: Cell::new(0),
        })
    }

    fn routes() -> EvaluationRouteTable {
        EvaluationRouteTable::new([EvaluationRoute {
            service_id: "primary".into(),
            purpose: "primary".into(),
            model: "candidate".into(),
            endpoint_profile: "candidate_openai".into(),
            prepared_identity_sha256: "c".repeat(64),
            endpoint_capabilities: BTreeSet::from(["chat".into()]),
        }])
        .unwrap()
    }

    fn operation(id: &str) -> HostOperationEnvelope {
        HostOperationEnvelope {
            operation_id: "operation".into(),
            unit_id: "unit".into(),
            case_id: "case".into(),
            semantic_attempt_id: "attempt".into(),
            logical_call_id: "call".into(),
            service_id: "primary".into(),
            semantic_operation_id: RegisteredOperationId::new(id).unwrap(),
            purpose: "primary".into(),
            payload: serde_json::json!({"messages": []}),
            restricted: false,
            stream: false,
        }
    }

    #[test]
    fn registry_rejects_duplicate_open_operation_ids() {
        let mut builder = HostExecutorRegistryBuilder::default();
        builder.register(factory("model.generate", false)).unwrap();
        assert!(builder.register(factory("model.generate", true)).is_err());
    }

    #[test]
    fn prepare_resolves_logical_service_and_capabilities_without_url() {
        let fixture = factory("model.generate", false);
        let mut builder = HostExecutorRegistryBuilder::default();
        builder.register(fixture.clone()).unwrap();
        let registry = builder.freeze().unwrap();
        registry
            .prepare(
                &operation("model.generate"),
                &routes(),
                &HostExecutorRuntime::default(),
            )
            .unwrap();
        assert_eq!(fixture.prepared.get(), 1);
        let route = routes();
        let resolved = route.resolve("primary").unwrap();
        assert_eq!(resolved.endpoint_profile, "candidate_openai");
        assert!(!resolved.model.contains("://"));
    }

    #[test]
    fn missing_operation_route_capability_and_streaming_fail_before_effect() {
        let terminal = factory("model.generate", false);
        let mut builder = HostExecutorRegistryBuilder::default();
        builder.register(terminal.clone()).unwrap();
        let registry = builder.freeze().unwrap();
        assert!(
            registry
                .prepare(
                    &operation("model.embed"),
                    &routes(),
                    &HostExecutorRuntime::default(),
                )
                .is_err()
        );
        let mut streaming = operation("model.generate");
        streaming.stream = true;
        assert!(
            registry
                .prepare(&streaming, &routes(), &HostExecutorRuntime::default(),)
                .is_err()
        );

        let incompatible = EvaluationRouteTable::new([EvaluationRoute {
            service_id: "primary".into(),
            purpose: "primary".into(),
            model: "candidate".into(),
            endpoint_profile: "candidate_embedding".into(),
            prepared_identity_sha256: "d".repeat(64),
            endpoint_capabilities: BTreeSet::from(["embedding".into()]),
        }])
        .unwrap();
        assert!(
            registry
                .prepare(
                    &operation("model.generate"),
                    &incompatible,
                    &HostExecutorRuntime::default(),
                )
                .is_err()
        );
        assert_eq!(terminal.prepared.get(), 0);
    }

    #[test]
    fn route_table_rejects_duplicate_or_url_shaped_service_ids() {
        let route = EvaluationRoute {
            service_id: "primary".into(),
            purpose: "primary".into(),
            model: "candidate".into(),
            endpoint_profile: "candidate_openai".into(),
            prepared_identity_sha256: "e".repeat(64),
            endpoint_capabilities: BTreeSet::new(),
        };
        assert!(EvaluationRouteTable::new([route.clone(), route]).is_err());
        let invalid = EvaluationRoute {
            service_id: "https://upstream.invalid".into(),
            purpose: "primary".into(),
            model: "candidate".into(),
            endpoint_profile: "candidate_openai".into(),
            prepared_identity_sha256: "e".repeat(64),
            endpoint_capabilities: BTreeSet::new(),
        };
        assert!(EvaluationRouteTable::new([invalid]).is_err());
    }
}
