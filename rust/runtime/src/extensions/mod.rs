// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Statically linked extension composition for native AIPerf.
//!
//! Rust does not discover trait implementations automatically. A distribution
//! links ordinary Cargo dependencies, constructs one [`AIPerfRegistry`], and
//! explicitly applies their [`AIPerfExtension`] implementations before starting
//! any runtime or worker. The resulting implementation universe is fixed by the
//! build; there is no manifest scanning, dynamic import, or stable-library ABI.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{self, Display};
use std::sync::Arc;

mod transactional;

pub(crate) use transactional::commit_on_clone;
pub use transactional::{DuplicateName, TransactionalRegistry};

use crate::adaptive::ActuatorRegistry;
use crate::dataset::{
    DatasetError, EndpointResolver as DatasetEndpointResolver, LoaderRegistry, SamplerRegistry,
};
use crate::endpoints::{
    Endpoint, EndpointFactory, EndpointId, EndpointRegistry, EndpointRegistryBuilder,
    EndpointRegistryError,
};
#[cfg(feature = "engine")]
use crate::engine::registry::{TransportFactory, WorkloadFactory};
use crate::export::ExporterRegistry;

/// Error returned while constructing or extending an [`AIPerfRegistry`].
#[derive(Debug)]
pub enum ExtensionError {
    /// A dataset-format, sampler, or endpoint registry rejected an entry.
    Dataset(DatasetError),
    /// The frozen endpoint registry rejected a factory descriptor.
    Endpoint(EndpointRegistryError),
    /// An extension supplied an empty stable name.
    EmptyExtensionName,
    /// An extension name was already applied to this aggregate.
    DuplicateExtension(String),
    /// An extension rejected its own configuration or prerequisites.
    Rejected(String),
    /// A named extension failed while populating a staged aggregate.
    ExtensionRegistration {
        /// Stable extension name.
        name: String,
        /// Underlying typed registry error.
        source: Box<ExtensionError>,
    },
}

impl ExtensionError {
    /// Construct an extension-defined validation failure.
    pub fn rejected(message: impl Into<String>) -> Self {
        Self::Rejected(message.into())
    }
}

impl Display for ExtensionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dataset(error) => Display::fmt(error, f),
            Self::Endpoint(error) => Display::fmt(error, f),
            Self::EmptyExtensionName => f.write_str("extension name cannot be empty"),
            Self::DuplicateExtension(name) => {
                write!(f, "duplicate AIPerf extension {name:?}")
            }
            Self::Rejected(message) => write!(f, "extension registration rejected: {message}"),
            Self::ExtensionRegistration { name, source } => {
                write!(f, "AIPerf extension {name:?} failed to register: {source}")
            }
        }
    }
}

impl Error for ExtensionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Dataset(error) => Some(error),
            Self::Endpoint(error) => Some(error),
            Self::ExtensionRegistration { source, .. } => Some(source.as_ref()),
            Self::EmptyExtensionName | Self::DuplicateExtension(_) | Self::Rejected(_) => None,
        }
    }
}

impl From<DatasetError> for ExtensionError {
    fn from(error: DatasetError) -> Self {
        Self::Dataset(error)
    }
}

impl From<EndpointRegistryError> for ExtensionError {
    fn from(error: EndpointRegistryError) -> Self {
        Self::Endpoint(error)
    }
}

/// One statically linked package that contributes named trait implementations.
///
/// Implementations should contain no run state. Registration executes once in
/// the application composition root, before clocks, transports, or workers are
/// constructed.
pub trait AIPerfExtension {
    /// Stable package-level name used for duplicate detection and diagnostics.
    fn name(&self) -> &str;

    /// Add this package's implementations through the typed category registries.
    fn register(&self, registry: &mut AIPerfRegistry) -> Result<(), ExtensionError>;
}

/// Composition-root seam that constructs one frozen registry universe.
///
/// Stock runners use [`BuiltinAIPerfRegistryFactory`]. A custom statically
/// linked runner can inject its own factory and apply an explicit ordered set
/// of [`AIPerfExtension`] values without changing benchmark execution code.
pub trait AIPerfRegistryFactory {
    /// Build the registry used by capabilities, validation, and execution.
    fn build(&self) -> Result<AIPerfRegistry, ExtensionError>;
}

/// Factory for the stock in-tree registry set.
#[derive(Debug, Clone, Copy, Default)]
pub struct BuiltinAIPerfRegistryFactory;

impl AIPerfRegistryFactory for BuiltinAIPerfRegistryFactory {
    fn build(&self) -> Result<AIPerfRegistry, ExtensionError> {
        // The whole stock distribution is ONE ordered list of per-component
        // built-in [`AIPerfExtension`] values applied over an empty base. The
        // only conditional compilation here is the feature-gate line on each
        // optional extension: gating a component out of this list removes it
        // from the registry and therefore from the auto-derived `--capabilities`
        // catalog with no separate bookkeeping.
        //
        // Built-in extensions are applied through [`with_builtin_extensions`],
        // which composes their implementations without recording provenance
        // names: the native report's `run.extensions` array is reserved for
        // third-party add-ons layered on top of the stock build.
        //
        // [`with_builtin_extensions`]: AIPerfRegistry::with_builtin_extensions
        AIPerfRegistry::empty_or_base().with_builtin_extensions([
            &BuiltinLoadersExtension as &dyn AIPerfExtension,
            &BuiltinSamplersExtension,
            &BuiltinEndpointsExtension,
            &BuiltinExportersExtension,
            &BuiltinActuatorsExtension,
            #[cfg(feature = "engine")]
            &crate::engine::registry::HttpExtension,
            #[cfg(all(feature = "engine", feature = "grpc"))]
            &crate::engine::registry::GrpcExtension,
            #[cfg(all(feature = "engine", feature = "dynosim"))]
            &crate::engine::registry::DynosimExtension,
        ])
    }
}

/// Built-in dataset-format loaders (`synthetic`, `sharegpt`, recorded traces, …).
#[derive(Debug, Clone, Copy, Default)]
pub struct BuiltinLoadersExtension;

impl AIPerfExtension for BuiltinLoadersExtension {
    fn name(&self) -> &str {
        "aiperf.builtin.loaders"
    }

    fn register(&self, registry: &mut AIPerfRegistry) -> Result<(), ExtensionError> {
        registry.dataset_formats.register_builtin_formats()?;
        Ok(())
    }
}

/// Built-in conversation samplers (`random`, `sequential`, `shuffle`).
#[derive(Debug, Clone, Copy, Default)]
pub struct BuiltinSamplersExtension;

impl AIPerfExtension for BuiltinSamplersExtension {
    fn name(&self) -> &str {
        "aiperf.builtin.samplers"
    }

    fn register(&self, registry: &mut AIPerfRegistry) -> Result<(), ExtensionError> {
        registry.samplers.register_builtin_strategies()?;
        Ok(())
    }
}

/// Built-in endpoint dialects (OpenAI/Anthropic/KServe/Riva and the raw/template
/// factories).
#[derive(Debug, Clone, Copy, Default)]
pub struct BuiltinEndpointsExtension;

impl AIPerfExtension for BuiltinEndpointsExtension {
    fn name(&self) -> &str {
        "aiperf.builtin.endpoints"
    }

    fn register(&self, registry: &mut AIPerfRegistry) -> Result<(), ExtensionError> {
        registry.extend_endpoints(EndpointRegistryBuilder::register_builtins)
    }
}

/// Built-in post-report exporter sinks (genai-perf v1 JSON/CSV, server-metrics,
/// timeslice, accuracy, parquet, console, OTLP, MLflow, W&B) in canonical emit
/// order.
#[derive(Debug, Clone, Copy, Default)]
pub struct BuiltinExportersExtension;

impl AIPerfExtension for BuiltinExportersExtension {
    fn name(&self) -> &str {
        "aiperf.builtin.exporters"
    }

    fn register(&self, registry: &mut AIPerfRegistry) -> Result<(), ExtensionError> {
        registry
            .exporters
            .register_builtins()
            .map_err(|error| ExtensionError::rejected(error.to_string()))
    }
}

/// Built-in adaptive control-variable actuator factories (`concurrency`,
/// `prefill_concurrency`, `request_rate`, `users`).
#[derive(Debug, Clone, Copy, Default)]
pub struct BuiltinActuatorsExtension;

impl AIPerfExtension for BuiltinActuatorsExtension {
    fn name(&self) -> &str {
        "aiperf.builtin.actuators"
    }

    fn register(&self, registry: &mut AIPerfRegistry) -> Result<(), ExtensionError> {
        registry
            .actuators
            .register_builtins()
            .map_err(|error| ExtensionError::rejected(error.to_string()))
    }
}

/// Aggregate of every runtime-name registry shared by the native CLI paths.
///
/// Directly injected seams such as clocks, transports, observers, segment
/// stores, and materializers intentionally remain constructor arguments rather
/// than entries in this name-based catalog.
#[derive(Clone)]
pub struct AIPerfRegistry {
    dataset_formats: LoaderRegistry,
    samplers: SamplerRegistry,
    endpoints: EndpointRegistry,
    /// Name-keyed, explicitly ordered post-report exporter sinks. Populated by
    /// [`BuiltinExportersExtension`]; consumed by the report-persistence call
    /// sites (`persist_prepared_report`, the cellular controller).
    exporters: ExporterRegistry,
    /// Id-keyed adaptive control-variable actuator factories. Populated by
    /// [`BuiltinActuatorsExtension`]; the online adaptive construction path reads
    /// the same built-in set through [`ActuatorRegistry::with_builtin_actuators`].
    actuators: ActuatorRegistry,
    /// Name-keyed protocol-v2 transport factories. Selection resolves a transport
    /// by id from this one catalog; there is no separate runner registry.
    #[cfg(feature = "engine")]
    pub(crate) transports: TransactionalRegistry<Arc<dyn TransportFactory>>,
    /// Name-keyed protocol-v2 workload factories. Any registered workload runs
    /// over any registered transport; the workload owns transport-specific
    /// preparation resolved by id.
    #[cfg(feature = "engine")]
    pub(crate) workloads: TransactionalRegistry<Arc<dyn WorkloadFactory>>,
    extension_names: BTreeSet<String>,
}

impl AIPerfRegistry {
    /// Construct the empty base with every category registry unpopulated.
    ///
    /// This is the composition-root starting point: built-in and third-party
    /// [`AIPerfExtension`] values populate the category registries from here.
    /// It contributes nothing to the registered set and no provenance names.
    pub fn empty_or_base() -> Self {
        Self {
            dataset_formats: LoaderRegistry::new(),
            samplers: SamplerRegistry::new(),
            endpoints: EndpointRegistryBuilder::new().freeze(),
            exporters: ExporterRegistry::new(),
            actuators: ActuatorRegistry::new(),
            #[cfg(feature = "engine")]
            transports: TransactionalRegistry::new(),
            #[cfg(feature = "engine")]
            workloads: TransactionalRegistry::new(),
            extension_names: BTreeSet::new(),
        }
    }

    /// Construct the complete native in-tree category registry set.
    ///
    /// This is the stock loaders/samplers/endpoints universe with empty
    /// transport/workload sub-registries; it is the base a custom distribution
    /// layers additional [`AIPerfExtension`] values on top of. The built-in
    /// components are applied through [`Self::with_builtin_extensions`] so this
    /// base contributes zero provenance names.
    pub fn builtin() -> Result<Self, ExtensionError> {
        Self::empty_or_base().with_builtin_extensions([
            &BuiltinLoadersExtension as &dyn AIPerfExtension,
            &BuiltinSamplersExtension,
            &BuiltinEndpointsExtension,
            &BuiltinExportersExtension,
            &BuiltinActuatorsExtension,
        ])
    }

    /// Apply one linked extension transactionally.
    ///
    /// The startup-only maps are staged on a clone so a later duplicate cannot
    /// leave earlier entries from the same extension visible.
    pub fn register_extension(
        &mut self,
        extension: &dyn AIPerfExtension,
    ) -> Result<(), ExtensionError> {
        self.register_extension_inner(extension, true)
    }

    /// Apply one built-in extension transactionally without recording a
    /// provenance name.
    ///
    /// Built-in components are baked into the stock distribution rather than
    /// layered on top of it, so the native report's `run.extensions` array (fed
    /// by [`Self::extension_names`]) stays empty for the built-in build. The
    /// staging and duplicate-component rejection are otherwise identical to
    /// [`Self::register_extension`].
    fn register_builtin_extension(
        &mut self,
        extension: &dyn AIPerfExtension,
    ) -> Result<(), ExtensionError> {
        self.register_extension_inner(extension, false)
    }

    fn register_extension_inner(
        &mut self,
        extension: &dyn AIPerfExtension,
        record_name: bool,
    ) -> Result<(), ExtensionError> {
        let name = validate_extension_name(extension.name())?;
        if self.extension_names.contains(&name) {
            return Err(ExtensionError::DuplicateExtension(name));
        }

        commit_on_clone(self, |staged| {
            extension
                .register(staged)
                .map_err(|source| ExtensionError::ExtensionRegistration {
                    name: name.clone(),
                    source: Box::new(source),
                })?;
            if record_name {
                staged.extension_names.insert(name.clone());
            }
            Ok(())
        })
    }

    /// Apply an ordered collection of linked extensions transactionally one at a time.
    pub fn with_extensions<'a>(
        mut self,
        extensions: impl IntoIterator<Item = &'a dyn AIPerfExtension>,
    ) -> Result<Self, ExtensionError> {
        for extension in extensions {
            self.register_extension(extension)?;
        }
        Ok(self)
    }

    /// Apply an ordered collection of built-in extensions transactionally.
    ///
    /// Like [`Self::with_extensions`] but the applied names are not recorded in
    /// [`Self::extension_names`]; use this for the stock in-tree component set
    /// so the native report's `run.extensions` array stays reserved for
    /// third-party add-ons.
    pub fn with_builtin_extensions<'a>(
        mut self,
        extensions: impl IntoIterator<Item = &'a dyn AIPerfExtension>,
    ) -> Result<Self, ExtensionError> {
        for extension in extensions {
            self.register_builtin_extension(extension)?;
        }
        Ok(self)
    }

    /// Registered dataset formats used for explicit lookup and auto-detection.
    pub fn dataset_formats(&self) -> &LoaderRegistry {
        &self.dataset_formats
    }

    /// Mutable dataset-format registry for extension setup.
    pub fn dataset_formats_mut(&mut self) -> &mut LoaderRegistry {
        &mut self.dataset_formats
    }

    /// Registered conversation sampler factories.
    pub fn samplers(&self) -> &SamplerRegistry {
        &self.samplers
    }

    /// Mutable sampler registry for extension setup.
    pub fn samplers_mut(&mut self) -> &mut SamplerRegistry {
        &mut self.samplers
    }

    /// Registered endpoint dialect adapters.
    pub fn endpoints(&self) -> &EndpointRegistry {
        &self.endpoints
    }

    /// Registered post-report exporter sinks in canonical emit order.
    pub fn exporters(&self) -> &ExporterRegistry {
        &self.exporters
    }

    /// Registered adaptive control-variable actuator factories.
    pub fn actuators(&self) -> &ActuatorRegistry {
        &self.actuators
    }

    /// Register one statically linked endpoint factory during startup.
    ///
    /// A new frozen value replaces the old catalog only after the descriptor
    /// and every alias pass atomic collision validation.
    pub fn register_endpoint_factory<F>(&mut self, factory: F) -> Result<(), ExtensionError>
    where
        F: EndpointFactory + 'static,
    {
        self.extend_endpoints(|builder| builder.register_factory(factory))
    }

    /// Register several endpoint factories over one clone/freeze cycle.
    ///
    /// An extension mutates a fresh builder cloned from the current frozen
    /// catalog and installs the atomically re-frozen result only after every
    /// descriptor and alias passes collision validation. This is the batched
    /// form of [`Self::register_endpoint_factory`], avoiding one freeze per
    /// factory when a built-in extension contributes the whole dialect set.
    pub fn extend_endpoints(
        &mut self,
        register: impl FnOnce(&mut EndpointRegistryBuilder) -> Result<(), EndpointRegistryError>,
    ) -> Result<(), ExtensionError> {
        let mut builder = self.endpoints.to_builder();
        register(&mut builder)?;
        self.endpoints = builder.freeze();
        Ok(())
    }

    /// Clone the authoritative catalog behind the protocol-v1 dataset lookup
    /// trait. The adapter holds no names or implementations of its own.
    pub fn endpoint_resolver(&self) -> Arc<dyn DatasetEndpointResolver> {
        Arc::new(LegacyDatasetEndpointResolver {
            endpoints: self.endpoints.clone(),
            default: EndpointId::new("chat").expect("built-in default ID is valid"),
        })
    }

    /// Names of successfully applied extensions in deterministic order.
    pub fn extension_names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.extension_names.iter().map(String::as_str)
    }
}

#[derive(Clone, Debug)]
struct LegacyDatasetEndpointResolver {
    endpoints: EndpointRegistry,
    default: EndpointId,
}

impl DatasetEndpointResolver for LegacyDatasetEndpointResolver {
    fn resolve(&self, name: Option<&str>) -> crate::dataset::Result<Arc<dyn Endpoint>> {
        let id = match name {
            Some(name) => EndpointId::new(name),
            None => Ok(self.default.clone()),
        }
        .map_err(|error| DatasetError::Validation(error.to_string()))?;
        self.endpoints
            .legacy_endpoint(&id)
            .map_err(|error| DatasetError::Validation(error.to_string()))
    }

    fn resolve_type(
        &self,
        endpoint_type: crate::endpoints::EndpointType,
    ) -> crate::dataset::Result<Arc<dyn Endpoint>> {
        let id = EndpointId::new(endpoint_type.canonical_id())
            .expect("legacy endpoint canonical IDs are valid");
        self.endpoints
            .legacy_endpoint(&id)
            .map_err(|error| DatasetError::Validation(error.to_string()))
    }
}

// Preserve the declared name verbatim for duplicate detection and diagnostics:
// case/separator normalization would false-collide distinct names (e.g.
// `foo_bar` vs `foo-bar`, `Foo` vs `foo`) and report a name the extension never
// declared. Only reject a name that is empty (or whitespace-only).
fn validate_extension_name(name: &str) -> Result<String, ExtensionError> {
    if name.trim().is_empty() {
        Err(ExtensionError::EmptyExtensionName)
    } else {
        Ok(name.to_string())
    }
}
