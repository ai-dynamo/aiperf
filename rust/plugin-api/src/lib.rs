// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The source API native AIPerf plugins are compiled against.
//!
//! This crate fixes the shape of the host/plugin boundary: how a plugin library
//! is entered, what identity it declares, how it registers capabilities, and
//! which errors it may return. It holds no host implementation and no runtime
//! dependency — a plugin links it, and so does the host loader, so both sides
//! agree on the boundary by construction.
//!
//! - [`id`] — [`RegistryId`], the normalized identifier every boundary name is.
//! - [`descriptor`] — [`PluginSourceApiVersion`] and the static package and
//!   category descriptors a plugin declares.
//! - [`extension`] — [`PluginEntryV1`], [`PluginDeclarationV1`],
//!   [`AIPerfExtension`], and [`PluginRegistrar`].
//! - [`error`] — the typed errors that cross the boundary instead of unwinding.
//! - [`ownership`] — the generation-1 ownership table the documentation guard
//!   checks `docs/specs/plugin-api-ownership.md` against.
//!
//! # Boundary rules
//!
//! Every artifact on either side of the boundary is built `panic = abort`, so a
//! panic is process-fatal rather than an unwind across a library edge. Every
//! recoverable failure is a typed value. Plugin-allocated storage is borrowed
//! for `'static` at the entry call and is never freed by the host: plugin
//! library handles are retained until process teardown, so plugin code outlives
//! every host value derived from it.

#![deny(missing_docs)]

pub mod descriptor;
pub mod error;
pub mod extension;
pub mod id;
pub mod ownership;

pub use descriptor::{PluginCategoryDescriptor, PluginPackageDescriptor, PluginSourceApiVersion};
pub use error::{ExtensionError, RegistryIdError, SourceApiVersionError};
pub use extension::{
    AIPerfExtension, PLUGIN_ENTRY_SYMBOL_V1, PluginDeclarationV1, PluginEntryV1, PluginRegistrar,
};
pub use id::{REGISTRY_ID_MAX_LEN, REGISTRY_ID_NORMALIZATION_VERSION, RegistryId};
pub use ownership::{CallPhase, GENERATION_1_SURFACE, OwnershipRow, StorageOwner};

/// The source API version this crate implements, as a canonical string.
///
/// [`PluginSourceApiVersion::CURRENT`] is the parsed form. The two must agree;
/// the `source_api_version_string_matches_parsed_current` test in
/// `tests/ownership_table.rs` is what holds them together.
pub const PLUGIN_SOURCE_API_VERSION: &str = "1.0.0";

/// The source API version this crate implements.
///
/// Returns the parsed [`PluginSourceApiVersion::CURRENT`]. Callers compare a
/// package's declared version against this to decide loadability.
pub const fn plugin_source_api_version() -> PluginSourceApiVersion {
    PluginSourceApiVersion::CURRENT
}
