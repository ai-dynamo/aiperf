// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Strict schema-1.1 NativeGraph package contracts.

mod package;

pub use package::{
    AdapterId, AdapterRole, AdapterSpec, GenerationDefaults, HeaderSecretRef, ModelBindingId,
    ModelBindingSpec, ModelCapturePolicy, ModelSecretId, NativeGraphPackagePlan,
    NativeGraphProfile, NativeGraphProgramSource, TokenizerBindingSpec,
};
pub(crate) use package::{
    NativeGraphPackageDraft, NativeGraphSectionDto, resolve_native_graph_package,
};
