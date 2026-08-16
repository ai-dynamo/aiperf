// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Native acquisition and normalization of Harbor-compatible task packages.

mod acquire;
mod harbor;
mod normalize;
mod source_snapshot;

pub use acquire::{HarborSource, NativeSourceAcquirer, SourceAcquirer};
pub use harbor::{HarborImportError, HarborImporter, ImportedTask};
pub use normalize::HarborTaskPackage;
pub(crate) use normalize::MaterializedSource;
pub use source_snapshot::AcquiredSource;
