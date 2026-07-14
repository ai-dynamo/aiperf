// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SLA-driven adaptive load control.
//!
//! This crate implements the `ramp_until_fail` controller described by
//! `specs/2026-07-11-aiperf-rust-adaptive-scale-design.md`. It is transport
//! neutral: returned-request observations enter through [`WindowSampler`], time
//! comes from [`crate::clock::Clock`], and every mutable load knob sits behind
//! [`ControlActuator`]. The same controller can therefore drive live HTTP, a
//! socket-backed mock, or an in-process simulated sink without branching on the
//! execution mode.

pub mod actuator;
pub mod artifacts;
pub mod controller;
pub mod error;
pub mod observer;
pub mod runtime;
pub mod sla;
pub mod step;
pub mod window;

pub use actuator::{
    ControlActuator, ControlSnapshot, PrefillConcurrencyActuator, RequestRateActuator,
    SessionConcurrencyActuator, UserTarget, UsersActuator,
};
pub use artifacts::{
    ADAPTIVE_SCHEMA_VERSION, AdaptiveArtifactSink, AdaptiveCandidate, AdaptiveEvent,
    AdaptiveResult, AdaptiveSummary, AdaptiveTotals, ArtifactValue, CorrelationContext,
    FileArtifactSink,
};
pub use controller::{
    AdaptiveStatus, AssessmentOutcome, CandidateObservation, Controller, ControllerEvent,
    ControllerEventKind, ControllerPhase, ControllerSnapshot, RampUntilFailController,
    RampUntilFailOptions,
};
pub use error::AdaptiveError;
pub use observer::AdaptiveObserver;
pub use runtime::{AdaptiveScale, AdaptiveScaleOptions, SharedArtifactSink, SharedWindowSampler};
pub use sla::{DefaultSlaEvaluator, SlaEvaluator, SlaFilter, SlaOp, SlaStat, SlaValues};
pub use step::{FixedPercentStep, SlaMarginStep, StepPolicy, StepPolicySnapshot};
pub use window::{RequestSample, TumblingWindowSampler, WindowSampler, WindowStats};
