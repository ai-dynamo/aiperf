// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Profiling-phase server-profiler sidecar composition.

use std::cell::Cell;
use std::rc::Rc;

use anyhow::Result;

use crate::cellular::ModuloCellPartition;
use crate::cellular::transport::CellPhaseSignal;
use crate::engine::cellular_cell::{await_controller_phase_advance, send_controller_phase_signal};
use crate::engine::control_hooks::ServerProfilerCoordinator;
use crate::phase_runtime::ScheduledPhaseSidecar;
use crate::timing::LocalPhaseFuture;

/// Build the profiling-phase sidecar for one phase name.
pub(crate) fn sidecar(
    coordinator: Rc<ServerProfilerCoordinator>,
    phase_name: impl Into<String>,
) -> Rc<dyn ScheduledPhaseSidecar> {
    let phase_name = phase_name.into();
    if ModuloCellPartition::from_env().is_some() {
        Rc::new(ServerProfilerSidecar {
            mode: ServerProfilerMode::Cellular { phase_name },
        })
    } else {
        Rc::new(ServerProfilerSidecar {
            mode: ServerProfilerMode::Local {
                coordinator,
                has_ownership: Rc::new(Cell::new(false)),
            },
        })
    }
}

struct ServerProfilerSidecar {
    mode: ServerProfilerMode,
}

enum ServerProfilerMode {
    Local {
        coordinator: Rc<ServerProfilerCoordinator>,
        has_ownership: Rc<Cell<bool>>,
    },
    Cellular {
        phase_name: String,
    },
}

impl ScheduledPhaseSidecar for ServerProfilerSidecar {
    fn start(&self) -> LocalPhaseFuture<Result<()>> {
        match &self.mode {
            ServerProfilerMode::Local {
                coordinator,
                has_ownership,
            } => {
                let coordinator = coordinator.clone();
                let has_ownership = has_ownership.clone();
                Box::pin(async move {
                    coordinator.acquire().await?;
                    has_ownership.set(true);
                    Ok(())
                })
            }
            ServerProfilerMode::Cellular { phase_name } => {
                let phase_name = phase_name.clone();
                Box::pin(async move {
                    send_controller_phase_signal(&phase_name, CellPhaseSignal::Ready).await?;
                    await_controller_phase_advance(&phase_name).await?;
                    Ok(())
                })
            }
        }
    }

    fn finish(&self) -> LocalPhaseFuture<Result<()>> {
        match &self.mode {
            ServerProfilerMode::Local {
                coordinator,
                has_ownership,
            } => {
                let coordinator = coordinator.clone();
                let has_ownership = has_ownership.clone();
                Box::pin(async move {
                    if !has_ownership.replace(false) {
                        return Ok(());
                    }
                    if let Err(error) = coordinator.release().await {
                        tracing::warn!(
                            error = %error,
                            "server profiler stop failed after profiling drain"
                        );
                    }
                    Ok(())
                })
            }
            ServerProfilerMode::Cellular { phase_name } => {
                let phase_name = phase_name.clone();
                Box::pin(async move {
                    send_controller_phase_signal(&phase_name, CellPhaseSignal::Complete).await?;
                    Ok(())
                })
            }
        }
    }
}
