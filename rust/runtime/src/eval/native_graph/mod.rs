// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Strict schema-1.1 NativeGraph package contracts.

mod matrix;
mod package;
mod result;
mod suite;

pub use matrix::{
    EpisodeAssignment, EpisodeRunner, LocalNativeGraphSuiteScheduler,
    LocalNativeGraphSuiteSchedulerFactory, MatrixError, NativeGraphSuiteScheduler, ResourceLimits,
    SuiteSchedulerFactory, run_resolved_suite,
};
pub use result::{
    EpisodeAggregate, EpisodeComparability, EpisodeExecution, EpisodeIntegrity, EpisodeResult,
    EpisodeResultError, EpisodeScoreState, aggregate_episode_results,
};
pub use suite::{
    AuthoredNativeGraphSuite, EpisodeAssignmentId, ModelCapacityKey, NativeGraphSuiteDefinition,
    NativeGraphSuiteManifest, ResolvedEpisodeTrial, ResolvedNativeGraphSuite, ResourceLeaseRequest,
    SelectedModelBinding, SuiteError, SuiteRunId, SuiteTrialSpec, parse_native_graph_suite_toml,
};

pub use package::{
    AdapterId, AdapterRole, AdapterSpec, GenerationDefaults, HeaderSecretRef, ModelBindingId,
    ModelBindingSpec, ModelCapturePolicy, ModelSecretId, NativeGraphPackagePlan,
    NativeGraphProfile, NativeGraphProgramSource, TokenizerBindingSpec,
};
pub(crate) use package::{
    NativeGraphPackageDraft, NativeGraphSectionDto, resolve_native_graph_package,
};
