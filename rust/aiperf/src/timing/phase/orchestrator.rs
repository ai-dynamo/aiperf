// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ordered warmup → profiling orchestration with optional seamless overlap.
//!
//! A fresh [`PhaseRunner`]
//! owns each phase's counters and lifecycle. The shared [`PhaseRunnerFactory`]
//! owns long-lived execution state such as the conversation source, slot pools,
//! cancellation policy, and endpoint selector, so debt-draining capacity and
//! deterministic sampler state survive phase boundaries without IPC.

use std::cell::{Cell, RefCell};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::rc::Rc;

use crate::clock::Clock;
use tokio::sync::Notify;

use super::{
    ClockPhaseRunner, LocalPhaseFuture, PhaseConfig, PhaseExecutionFactory, PhaseKind,
    PhaseObserver, PhaseRunError, PhaseRunner, PhaseStats,
};

/// Factory seam creating one fresh runner per phase.
pub trait PhaseRunnerFactory {
    /// Build a runner with isolated lifecycle/progress state.
    fn create(&self, config: PhaseConfig) -> Result<Rc<dyn PhaseRunner>, PhaseRunError>;

    /// Cancel backend work shared across all active phase runners.
    fn cancel_all(&self) -> LocalPhaseFuture<Result<(), PhaseRunError>> {
        Box::pin(async { Ok(()) })
    }
}

/// Default runner factory over one shared execution factory and clock.
pub struct ClockPhaseRunnerFactory {
    clock: Rc<dyn Clock>,
    observer: Rc<dyn PhaseObserver>,
    execution_factory: Rc<dyn PhaseExecutionFactory>,
}

impl ClockPhaseRunnerFactory {
    /// Create a factory whose execution state is shared across every phase.
    pub fn new(
        clock: Rc<dyn Clock>,
        observer: Rc<dyn PhaseObserver>,
        execution_factory: Rc<dyn PhaseExecutionFactory>,
    ) -> Self {
        Self {
            clock,
            observer,
            execution_factory,
        }
    }
}

impl PhaseRunnerFactory for ClockPhaseRunnerFactory {
    fn create(&self, config: PhaseConfig) -> Result<Rc<dyn PhaseRunner>, PhaseRunError> {
        Ok(Rc::new(ClockPhaseRunner::new(
            config,
            self.clock.clone(),
            self.observer.clone(),
            self.execution_factory.clone(),
        )?))
    }

    fn cancel_all(&self) -> LocalPhaseFuture<Result<(), PhaseRunError>> {
        let factory = self.execution_factory.clone();
        Box::pin(async move { factory.cancel_all().await.map_err(PhaseRunError::Execution) })
    }
}

/// Object-safe multi-phase orchestration seam.
pub trait PhaseOrchestrator {
    /// Run every configured phase and return final snapshots in config order.
    fn run_all(&self) -> LocalPhaseFuture<Result<Vec<PhaseStats>, PhaseOrchestratorError>>;

    /// Cancel shared in-flight work, then signal every active runner.
    fn cancel(&self) -> LocalPhaseFuture<Result<(), PhaseOrchestratorError>>;

    /// Number of runners that are issuing or waiting for returns.
    fn active_phase_count(&self) -> usize;
}

struct OrchestratorInner {
    configs: Vec<PhaseConfig>,
    runner_factory: Rc<dyn PhaseRunnerFactory>,
    observer: Rc<dyn PhaseObserver>,
    active: RefCell<Vec<Rc<dyn PhaseRunner>>>,
    seamless_failures: SeamlessFailureSignal,
    run_started: Cell<bool>,
    cancelled: Cell<bool>,
}

#[derive(Clone)]
struct SeamlessFailure {
    phase_id: String,
    source: PhaseRunError,
}

#[derive(Default)]
struct SeamlessFailureSignal {
    first: RefCell<Option<SeamlessFailure>>,
    notify: Notify,
}

impl SeamlessFailureSignal {
    fn record(&self, failure: SeamlessFailure) {
        let mut first = self.first.borrow_mut();
        if first.is_none() {
            *first = Some(failure);
            self.notify.notify_waiters();
        }
    }

    fn first(&self) -> Option<SeamlessFailure> {
        self.first.borrow().clone()
    }

    async fn wait(&self) -> SeamlessFailure {
        loop {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if let Some(failure) = self.first() {
                return failure;
            }
            notified.await;
        }
    }
}

/// Default ordered phase orchestrator.
#[derive(Clone)]
pub struct ClockPhaseOrchestrator {
    inner: Rc<OrchestratorInner>,
}

impl ClockPhaseOrchestrator {
    /// Validate phase order and construct an idle orchestrator.
    pub fn new(
        configs: Vec<PhaseConfig>,
        runner_factory: Rc<dyn PhaseRunnerFactory>,
        observer: Rc<dyn PhaseObserver>,
    ) -> Result<Self, PhaseOrchestratorError> {
        validate_phase_order(&configs)?;
        Ok(Self {
            inner: Rc::new(OrchestratorInner {
                configs,
                runner_factory,
                observer,
                active: RefCell::new(Vec::new()),
                seamless_failures: SeamlessFailureSignal::default(),
                run_started: Cell::new(false),
                cancelled: Cell::new(false),
            }),
        })
    }

    async fn run_all_entry(self) -> Result<Vec<PhaseStats>, PhaseOrchestratorError> {
        if self.inner.run_started.replace(true) {
            return Err(PhaseOrchestratorError::AlreadyRun);
        }

        // Mirrors Python's `phase_orchestrator.py` "Initialized N phase(s)" line.
        let phase_names: Vec<&str> = self.inner.configs.iter().map(|c| c.id.as_str()).collect();
        tracing::info!(
            "Initialized {} phase(s): {:?}",
            self.inner.configs.len(),
            phase_names,
        );

        let mut ordered_runners = Vec::with_capacity(self.inner.configs.len());
        for (index, config) in self.inner.configs.iter().cloned().enumerate() {
            self.fail_if_seamless_predecessor_failed().await?;
            if self.inner.cancelled.get() {
                break;
            }
            self.prune_completed();
            let phase_id = config.id.clone();
            let is_final = index + 1 == self.inner.configs.len();
            let seamless_non_final = config.seamless && !is_final;
            let runner = match self.inner.runner_factory.create(config) {
                Ok(runner) => runner,
                Err(source) => {
                    let _ = self.cancel_active().await;
                    return Err(PhaseOrchestratorError::Runner { phase_id, source });
                }
            };
            self.inner.active.borrow_mut().push(runner.clone());
            ordered_runners.push(runner.clone());

            self.await_runner_operation(&phase_id, runner.run(is_final))
                .await?;

            if seamless_non_final {
                self.spawn_active_cleanup(runner);
            } else {
                self.await_runner_operation(&phase_id, runner.wait_complete())
                    .await?;
                self.remove_active(&runner);
            }
            if self.inner.cancelled.get() {
                break;
            }
        }

        let mut final_stats = Vec::with_capacity(ordered_runners.len());
        for runner in ordered_runners {
            let phase_id = runner.config().id;
            final_stats.push(
                self.await_runner_operation(&phase_id, runner.wait_complete())
                    .await?,
            );
            self.remove_active(&runner);
        }
        self.inner.observer.on_phases_complete(final_stats.clone());
        Ok(final_stats)
    }

    fn spawn_active_cleanup(&self, runner: Rc<dyn PhaseRunner>) {
        let inner = Rc::downgrade(&self.inner);
        let runner_for_wait = runner.clone();
        let phase_id = runner.config().id;
        tokio::task::spawn_local(async move {
            let result = runner_for_wait.wait_complete().await;
            if let Some(inner) = inner.upgrade() {
                if let Err(source) = result {
                    inner
                        .seamless_failures
                        .record(SeamlessFailure { phase_id, source });
                }
                inner
                    .active
                    .borrow_mut()
                    .retain(|active| !Rc::ptr_eq(active, &runner));
            }
        });
    }

    async fn await_runner_operation(
        &self,
        phase_id: &str,
        operation: LocalPhaseFuture<Result<PhaseStats, PhaseRunError>>,
    ) -> Result<PhaseStats, PhaseOrchestratorError> {
        if let Some(failure) = self.inner.seamless_failures.first() {
            let _ = self.cancel_active().await;
            return Err(background_failure(failure));
        }
        let background = self.inner.seamless_failures.wait();
        tokio::pin!(background);
        tokio::pin!(operation);
        tokio::select! {
            biased;
            failure = &mut background => {
                let _ = self.cancel_active().await;
                let _ = operation.await;
                Err(background_failure(failure))
            }
            result = &mut operation => match result {
                Ok(stats) => Ok(stats),
                Err(source) => {
                    let _ = self.cancel_active().await;
                    Err(PhaseOrchestratorError::Runner {
                        phase_id: phase_id.to_string(),
                        source,
                    })
                }
            }
        }
    }

    async fn fail_if_seamless_predecessor_failed(&self) -> Result<(), PhaseOrchestratorError> {
        let Some(failure) = self.inner.seamless_failures.first() else {
            return Ok(());
        };
        let _ = self.cancel_active().await;
        Err(background_failure(failure))
    }

    fn remove_active(&self, runner: &Rc<dyn PhaseRunner>) {
        self.inner
            .active
            .borrow_mut()
            .retain(|active| !Rc::ptr_eq(active, runner));
    }

    fn prune_completed(&self) {
        self.inner
            .active
            .borrow_mut()
            .retain(|runner| !runner.is_complete());
    }

    async fn cancel_active(&self) -> Result<(), PhaseOrchestratorError> {
        self.inner.cancelled.set(true);
        // Flip each active runner's lifecycle to cancelled BEFORE cancelling the
        // shared backend. Cancelling shared issuance first can let a runner
        // observe "sending complete" and finish through the normal completion
        // path, dropping the `was_cancelled` flag even though the run was
        // interrupted; signalling the runner first guarantees the Cancelled
        // completion path and a `was_cancelled = true` snapshot.
        let active = self.inner.active.borrow().clone();
        for runner in &active {
            if !runner.is_complete() {
                runner.cancel();
            }
        }
        let shared_result = self.inner.runner_factory.cancel_all().await;
        shared_result.map_err(PhaseOrchestratorError::Cancel)
    }
}

fn background_failure(failure: SeamlessFailure) -> PhaseOrchestratorError {
    PhaseOrchestratorError::Runner {
        phase_id: failure.phase_id,
        source: failure.source,
    }
}

impl PhaseOrchestrator for ClockPhaseOrchestrator {
    fn run_all(&self) -> LocalPhaseFuture<Result<Vec<PhaseStats>, PhaseOrchestratorError>> {
        let orchestrator = self.clone();
        Box::pin(async move { orchestrator.run_all_entry().await })
    }

    fn cancel(&self) -> LocalPhaseFuture<Result<(), PhaseOrchestratorError>> {
        let orchestrator = self.clone();
        Box::pin(async move { orchestrator.cancel_active().await })
    }

    fn active_phase_count(&self) -> usize {
        self.prune_completed();
        self.inner.active.borrow().len()
    }
}

fn validate_phase_order(configs: &[PhaseConfig]) -> Result<(), PhaseOrchestratorError> {
    if configs.is_empty() {
        return Err(PhaseOrchestratorError::NoPhases);
    }
    let mut profiling_seen = false;
    let mut ids = std::collections::BTreeSet::new();
    for config in configs {
        config
            .validate()
            .map_err(|source| PhaseOrchestratorError::InvalidConfig {
                phase_id: config.id.clone(),
                source,
            })?;
        if !ids.insert(config.id.clone()) {
            return Err(PhaseOrchestratorError::DuplicatePhaseId(config.id.clone()));
        }
        match config.kind {
            PhaseKind::Warmup if profiling_seen => {
                return Err(PhaseOrchestratorError::WarmupAfterProfiling(
                    config.id.clone(),
                ));
            }
            PhaseKind::Profiling => profiling_seen = true,
            PhaseKind::Warmup => {}
        }
    }
    if !profiling_seen {
        return Err(PhaseOrchestratorError::ProfilingPhaseRequired);
    }
    Ok(())
}

/// Multi-phase configuration or execution failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PhaseOrchestratorError {
    /// No phases were configured.
    NoPhases,
    /// A benchmark cannot consist exclusively of warmup traffic.
    ProfilingPhaseRequired,
    /// One phase failed local validation.
    InvalidConfig {
        /// Stable phase identifier.
        phase_id: String,
        /// Validation failure.
        source: super::PhaseConfigError,
    },
    /// Two phases used the same stable identifier.
    DuplicatePhaseId(String),
    /// A warmup phase appeared after profiling began.
    WarmupAfterProfiling(String),
    /// The same orchestrator was started more than once.
    AlreadyRun,
    /// One phase runner failed.
    Runner {
        /// Stable phase identifier.
        phase_id: String,
        /// Runner failure.
        source: PhaseRunError,
    },
    /// Shared cancellation failed.
    Cancel(PhaseRunError),
}

impl Display for PhaseOrchestratorError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoPhases => write!(f, "at least one phase must be configured"),
            Self::ProfilingPhaseRequired => {
                write!(f, "at least one profiling phase must be configured")
            }
            Self::InvalidConfig { phase_id, source } => {
                write!(f, "invalid configuration for phase {phase_id:?}: {source}")
            }
            Self::DuplicatePhaseId(id) => write!(f, "duplicate phase id {id:?}"),
            Self::WarmupAfterProfiling(id) => {
                write!(f, "warmup phase {id:?} cannot follow a profiling phase")
            }
            Self::AlreadyRun => write!(f, "phase orchestrator was already run"),
            Self::Runner { phase_id, source } => {
                write!(f, "phase {phase_id:?} failed: {source}")
            }
            Self::Cancel(source) => write!(f, "shared phase cancellation failed: {source}"),
        }
    }
}

impl Error for PhaseOrchestratorError {}
