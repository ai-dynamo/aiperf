// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Strict Mini-SWE-Agent replay and non-executable imported session discovery.

mod discovery;
mod fixture;
pub mod import;
mod lowering;
mod recipes;
mod schema;

#[cfg(test)]
mod tests;

pub use discovery::{
    RecordedAgentInputError, RecordedAgentInputSource, ValidatedRecordedAgentCorpus,
    ValidatedRecordedAgentTrace, discover_recorded_agent_input,
};
pub use fixture::{CanonicalReplayFixture, CanonicalReplayFixtureDigestIndex};
pub use import::{
    ImportedAgentError, ImportedAgentReadSet, ImportedAgentSession, ImportedAgentSource,
    ImportedAgentSourceFile, ImportedModelCall, ImportedSessionFamily, ImportedSubagentParent,
    RawJsonMessage, detect_imported_agent_source, discover_imported_agent_read_set,
    parse_claude_session, parse_codex_session, parse_imported_agent_sessions,
};
pub use lowering::{
    BuiltinReplayRequestProfileResolver, RecordedAgentLoweringError, ReplayRequestProfile,
    ReplayRequestProfileResolver, lower_recorded_agent_corpus,
};
pub use recipes::resolve_recorded_environment;
pub use schema::{
    ExpectedCorpusShape, RecordedAgentEvent, RecordedAgentMetadata, RecordedAgentRecording,
    RecordedAgentReplayManifest, RecordedProviderRequest, ReplayDefaults, ReplayTaskIdentity,
};
