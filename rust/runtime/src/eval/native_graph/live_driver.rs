// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded staged trace driver for source-lowered NativeGraph programs.

use std::collections::BTreeMap;
use std::num::NonZeroU32;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;

use crate::dataset::Handle;
use crate::graph::driver::{
    TraceDriverCapabilities, TraceDriverContext, TraceDriverError, TraceDriverProvenance,
    TraceDriverSpec, TraceIdentity, TraceProgramDriver, TraceProgramDriverFactory,
    TraceStageDirective, TraceStageResult, WorkerIdentity,
};
use crate::graph::model::{GraphTracePlan, GraphTraceProgram};
use crate::graph::sink::GraphReplyStatus;
use crate::graph::supplement::TraceTerminalSupplement;

use super::lowering::{
    NativeGraphControlContract, validate_control_flow_contract, validate_native_graph_stage,
};

const LIVE_DRIVER_KIND: &str = "native_graph_live";

/// Factory for the Rust-owned staged NativeGraph driver family.
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeGraphLiveTraceProgramDriverFactory;

impl TraceProgramDriverFactory for NativeGraphLiveTraceProgramDriverFactory {
    fn capabilities(
        &self,
        spec: &TraceDriverSpec,
    ) -> Result<TraceDriverCapabilities, TraceDriverError> {
        let _ = validate_live_driver_spec(spec)?;
        let capabilities = TraceDriverCapabilities {
            has_live_turns: true,
            ..TraceDriverCapabilities::default()
        };
        capabilities.validate(spec, &spec.kind)?;
        Ok(capabilities)
    }

    fn create(
        &self,
        worker: WorkerIdentity,
        trace: &TraceIdentity,
        spec: &TraceDriverSpec,
    ) -> Result<Box<dyn TraceProgramDriver>, TraceDriverError> {
        let (data, provenance) = validate_live_driver_spec(spec)?;
        Ok(Box::new(NativeGraphLiveTraceProgramDriver {
            worker,
            trace: trace.clone(),
            stage_bound: data.control_flow.stage_bound,
            terminal_outputs: data.control_flow.terminal_outputs.clone(),
            control_flow: data.control_flow,
            provenance,
            state: LiveDriverState::Unopened,
        }))
    }
}

/// Worker-local cursor over one source-lowered NativeGraph program.
struct NativeGraphLiveTraceProgramDriver {
    worker: WorkerIdentity,
    trace: TraceIdentity,
    stage_bound: NonZeroU32,
    terminal_outputs: Vec<String>,
    control_flow: NativeGraphControlContract,
    provenance: TraceDriverProvenance,
    state: LiveDriverState,
}

enum LiveDriverState {
    Unopened,
    Ready(GraphTracePlan),
    AwaitingObservation { plan_identity: String },
    ReadyToComplete(BTreeMap<String, Handle>),
    Finished,
}

#[async_trait(?Send)]
impl TraceProgramDriver for NativeGraphLiveTraceProgramDriver {
    async fn open(
        &mut self,
        program: &GraphTraceProgram,
        context: &TraceDriverContext<'_>,
    ) -> Result<(), TraceDriverError> {
        if !matches!(self.state, LiveDriverState::Unopened) {
            return Err(TraceDriverError::new(
                "native graph live driver is already open",
            ));
        }
        if program.driver.kind != LIVE_DRIVER_KIND || self.trace != *context.trace {
            return Err(TraceDriverError::new(
                "native graph live driver received another program or trace",
            ));
        }
        let (program_data, program_provenance) = validate_live_driver_spec(&program.driver)?;
        if program_data.control_flow.stage_bound != self.stage_bound {
            return Err(TraceDriverError::new(
                "native graph live driver stage bound does not match its selected program",
            ));
        }
        if program_data.control_flow != self.control_flow {
            return Err(TraceDriverError::new(
                "native graph live driver control-flow contract does not match its selected program",
            ));
        }
        if program_provenance != self.provenance {
            return Err(TraceDriverError::new(
                "native graph live driver source provenance does not match its selected program",
            ));
        }
        if !self.terminal_outputs.is_empty() {
            return Err(TraceDriverError::new(
                "native graph live driver requires frozen terminal handles before stage execution",
            ));
        }
        validate_native_graph_stage(&program.profiling, &self.control_flow)
            .map_err(|error| TraceDriverError::new(error.to_string()))?;
        self.state = LiveDriverState::Ready(program.profiling.clone());
        Ok(())
    }

    fn stage_bound(&self) -> Option<NonZeroU32> {
        Some(self.stage_bound)
    }

    async fn next_stage(
        &mut self,
        _context: &TraceDriverContext<'_>,
    ) -> Result<Option<TraceStageDirective>, TraceDriverError> {
        match std::mem::replace(&mut self.state, LiveDriverState::Finished) {
            LiveDriverState::Ready(plan) => {
                validate_native_graph_stage(&plan, &self.control_flow)
                    .map_err(|error| TraceDriverError::new(error.to_string()))?;
                let plan_identity = format!("{}::stage-0", self.trace.trace_id);
                self.state = LiveDriverState::AwaitingObservation { plan_identity };
                Ok(Some(TraceStageDirective::Execute(plan)))
            }
            LiveDriverState::ReadyToComplete(outputs) => {
                self.state = LiveDriverState::Finished;
                Ok(Some(TraceStageDirective::Complete(
                    TraceTerminalSupplement::new(
                        self.trace.run_id.clone(),
                        self.trace.trajectory_id.clone(),
                        self.trace.trace_id.clone(),
                        self.worker.worker_id,
                        LIVE_DRIVER_KIND,
                    )
                    .with_terminal_outputs(outputs),
                )))
            }
            LiveDriverState::AwaitingObservation { plan_identity } => {
                self.state = LiveDriverState::AwaitingObservation { plan_identity };
                Err(TraceDriverError::new(
                    "native graph live driver requested a stage before observing the prior stage",
                ))
            }
            LiveDriverState::Unopened => {
                self.state = LiveDriverState::Unopened;
                Err(TraceDriverError::new(
                    "native graph live driver was not opened before stage selection",
                ))
            }
            LiveDriverState::Finished => Ok(None),
        }
    }

    async fn observe_stage(&mut self, result: TraceStageResult) -> Result<(), TraceDriverError> {
        let LiveDriverState::AwaitingObservation { plan_identity } = &self.state else {
            return Err(TraceDriverError::new(
                "native graph live driver received an unexpected stage observation",
            ));
        };
        if result.plan_identity != *plan_identity {
            return Err(TraceDriverError::new(format!(
                "native graph live driver observed {:?}, expected {:?}",
                result.plan_identity, plan_identity
            )));
        }
        if result.terminal_status != GraphReplyStatus::Completed {
            return Err(TraceDriverError::new(
                "native graph live driver received a non-completed graph stage",
            ));
        }
        let outputs = self
            .terminal_outputs
            .iter()
            .map(|channel| {
                result.output_handles.get(channel).cloned().map_or_else(
                    || Err(TraceDriverError::new(format!("native graph live driver did not receive declared terminal output {channel:?}"))),
                    |handle| Ok((channel.clone(), handle)),
                )
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        self.state = LiveDriverState::ReadyToComplete(outputs);
        Ok(())
    }

    async fn close(&mut self) -> Result<(), TraceDriverError> {
        self.state = LiveDriverState::Finished;
        Ok(())
    }

    async fn run(
        &mut self,
        _program: &GraphTraceProgram,
        _context: &TraceDriverContext<'_>,
    ) -> Result<TraceTerminalSupplement, TraceDriverError> {
        Err(TraceDriverError::new(
            "native graph live driver requires bounded staged graph execution",
        ))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LiveDriverData {
    control_flow: NativeGraphControlContract,
}

fn validate_live_driver_spec(
    spec: &TraceDriverSpec,
) -> Result<(LiveDriverData, TraceDriverProvenance), TraceDriverError> {
    let data = parse_live_driver_data(spec)?;
    validate_control_flow_contract(&data.control_flow)
        .map_err(|error| TraceDriverError::new(error.to_string()))?;
    let provenance = spec.source_provenance().cloned().ok_or_else(|| {
        TraceDriverError::new("native graph live driver is missing immutable source provenance")
    })?;
    if !provenance.matches_source_digest(&data.control_flow.source_snapshot_digest) {
        return Err(TraceDriverError::new(
            "native graph live driver source digest does not match immutable source provenance",
        ));
    }
    if !provenance.matches_static_projection_digest(&data.control_flow.static_projection_digest) {
        return Err(TraceDriverError::new(
            "native graph live driver static projection does not match immutable static projection provenance",
        ));
    }
    Ok((data, provenance))
}

fn parse_live_driver_data(spec: &TraceDriverSpec) -> Result<LiveDriverData, TraceDriverError> {
    if spec.kind != LIVE_DRIVER_KIND {
        return Err(TraceDriverError::new(format!(
            "native graph live factory cannot create trace driver {:?}",
            spec.kind
        )));
    }
    let fields = serde_json::Map::from_iter(spec.data.clone());
    serde_json::from_value(Value::Object(fields)).map_err(|error| {
        TraceDriverError::new(format!("invalid native graph live driver data: {error}"))
    })
}
