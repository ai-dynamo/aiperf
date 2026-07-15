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

use crate::dataset::{
    DatasetError, EndpointResolver as DatasetEndpointResolver, LoaderRegistry, SamplerRegistry,
};
use crate::endpoints::{
    Endpoint, EndpointFactory, EndpointId, EndpointRegistry, EndpointRegistryError,
};
#[cfg(feature = "runner-protocol")]
use crate::runner_protocol::registry::{RunnerTransportFactory, RunnerWorkloadFactory};

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
        #[allow(unused_mut)]
        let mut registry = AIPerfRegistry::builtin()?;
        // The runner transport/workload universe is the same catalog of record;
        // fold the built-in protocol-v2 components into the one product registry
        // whenever this build links the runner-protocol layer.
        #[cfg(feature = "runner-protocol")]
        crate::runner_protocol::registry::install_builtin_runner_components(&mut registry)
            .map_err(|error| ExtensionError::rejected(format!("{error:#}")))?;
        Ok(registry)
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
    /// Name-keyed protocol-v2 transport factories. Selection resolves a transport
    /// by id from this one catalog; there is no separate runner registry.
    #[cfg(feature = "runner-protocol")]
    pub(crate) transports: TransactionalRegistry<Arc<dyn RunnerTransportFactory>>,
    /// Name-keyed protocol-v2 workload factories. Any registered workload runs
    /// over any registered transport; the workload owns transport-specific
    /// preparation resolved by id.
    #[cfg(feature = "runner-protocol")]
    pub(crate) workloads: TransactionalRegistry<Arc<dyn RunnerWorkloadFactory>>,
    extension_names: BTreeSet<String>,
}

impl AIPerfRegistry {
    /// Construct the complete native in-tree registry set.
    pub fn builtin() -> Result<Self, ExtensionError> {
        Ok(Self {
            dataset_formats: LoaderRegistry::with_builtin_formats()?,
            samplers: SamplerRegistry::with_builtin_strategies()?,
            endpoints: EndpointRegistry::builtin()?,
            #[cfg(feature = "runner-protocol")]
            transports: TransactionalRegistry::new(),
            #[cfg(feature = "runner-protocol")]
            workloads: TransactionalRegistry::new(),
            extension_names: BTreeSet::new(),
        })
    }

    /// Apply one linked extension transactionally.
    ///
    /// The startup-only maps are staged on a clone so a later duplicate cannot
    /// leave earlier entries from the same extension visible.
    pub fn register_extension(
        &mut self,
        extension: &dyn AIPerfExtension,
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
            staged.extension_names.insert(name.clone());
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

    /// Register one statically linked endpoint factory during startup.
    ///
    /// A new frozen value replaces the old catalog only after the descriptor
    /// and every alias pass atomic collision validation.
    pub fn register_endpoint_factory<F>(&mut self, factory: F) -> Result<(), ExtensionError>
    where
        F: EndpointFactory + 'static,
    {
        let mut builder = self.endpoints.to_builder();
        builder.register_factory(factory)?;
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
