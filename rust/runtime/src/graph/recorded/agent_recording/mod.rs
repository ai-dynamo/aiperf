// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Strict Mini-SWE-Agent recording input and canonical fixture discovery.

mod discovery;
mod fixture;
mod lowering;
mod schema;

#[cfg(test)]
mod tests;

pub use discovery::{
    RecordedAgentInputError, RecordedAgentInputSource, ValidatedRecordedAgentCorpus,
    ValidatedRecordedAgentTrace, discover_recorded_agent_input,
};
pub use fixture::{CanonicalReplayFixture, CanonicalReplayFixtureDigestIndex};
pub use lowering::{
    BuiltinReplayRequestProfileResolver, RecordedAgentLoweringError, ReplayRequestProfile,
    ReplayRequestProfileResolver, lower_recorded_agent_corpus,
};
pub use schema::{
    ExpectedCorpusShape, RecordedAgentEvent, RecordedAgentRecording, RecordedAgentReplayManifest,
    RecordedProviderRequest, ReplayDefaults, ReplayTaskIdentity,
};
