// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    fs,
    num::NonZeroUsize,
    rc::Rc,
    time::Duration,
};

use aiperf_runtime::eval::{
    AgentVariantRef, ArtifactDigest, AttemptId, EpisodeComparability, EpisodeExecution,
    EpisodeIntegrity, EpisodeResult, EpisodeRunner, EpisodeScoreState, HarborImporter,
    HarborSource, LocalNativeGraphSuiteScheduler, MatrixError, ModelCapacityKey, ModelIdentity,
    NativeGraphSuiteManifest, NativeSourceAcquirer, PolicyIdentity, ResourceLeaseRequest,
    ResourceLimits, RuntimeIdentity, SuiteRunId, SuiteTrialSpec, TrialBudget, TrialSpec,
    aggregate_episode_results, run_resolved_suite,
};
use async_trait::async_trait;
use tokio::{sync::Notify, task::LocalSet};

#[tokio::test(flavor = "current_thread")]
async fn completion_order_never_changes_manifest_order() {
    let suite = four_trial_suite();
    let manifest_order = suite
        .trials()
        .iter()
        .map(|trial| trial.attempt_id().as_str().to_owned())
        .collect::<Vec<_>>();
    let scheduler = LocalNativeGraphSuiteScheduler::new(
        ResourceLimits::new(2, 2, 128, BTreeMap::new()).unwrap(),
    )
    .unwrap();
    let stats = Rc::new(ConcurrencyStats::default());
    let runner = Rc::new(DelayedScoredRunner {
        delays_ms: [40, 10, 30, 20],
        stats: stats.clone(),
    });

    let results = run_resolved_suite(&scheduler, suite, runner).await.unwrap();

    assert_eq!(stats.completions.borrow().first(), Some(&1));
    assert_eq!(
        results
            .iter()
            .map(|result| result.attempt_id().as_str())
            .collect::<Vec<_>>(),
        manifest_order
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
    );
    assert_eq!(stats.peak.get(), 2, "the scheduler did not fill both slots");
}

#[tokio::test(flavor = "current_thread")]
async fn episode_slots_are_shared_across_concurrent_suite_runs() {
    let scheduler = LocalNativeGraphSuiteScheduler::new(
        ResourceLimits::new(1, 2, 128, BTreeMap::new()).unwrap(),
    )
    .unwrap();
    let stats = Rc::new(ConcurrencyStats::default());
    let runner = Rc::new(UniformDelayedRunner {
        delay: Duration::from_millis(30),
        stats: stats.clone(),
    });

    let (first, second) = tokio::join!(
        run_resolved_suite(
            &scheduler,
            single_trial_suite(
                ResourceLeaseRequest::new(1, 64, BTreeMap::new()).unwrap(),
                "slot-a"
            ),
            runner.clone(),
        ),
        run_resolved_suite(
            &scheduler,
            single_trial_suite(
                ResourceLeaseRequest::new(1, 64, BTreeMap::new()).unwrap(),
                "slot-b"
            ),
            runner,
        ),
    );

    assert!(first.is_ok());
    assert!(second.is_ok());
    assert_eq!(stats.peak.get(), 1, "episode slots must span suite runs");
}

#[tokio::test(flavor = "current_thread")]
async fn resource_pools_are_shared_across_concurrent_suite_runs() {
    let scheduler = LocalNativeGraphSuiteScheduler::new(
        ResourceLimits::new(2, 1, 128, BTreeMap::new()).unwrap(),
    )
    .unwrap();
    let stats = Rc::new(ConcurrencyStats::default());
    let runner = Rc::new(UniformDelayedRunner {
        delay: Duration::from_millis(30),
        stats: stats.clone(),
    });

    let (first, second) = tokio::join!(
        run_resolved_suite(
            &scheduler,
            single_trial_suite(
                ResourceLeaseRequest::new(1, 64, BTreeMap::new()).unwrap(),
                "cpu-a"
            ),
            runner.clone(),
        ),
        run_resolved_suite(
            &scheduler,
            single_trial_suite(
                ResourceLeaseRequest::new(1, 64, BTreeMap::new()).unwrap(),
                "cpu-b"
            ),
            runner,
        ),
    );

    assert!(first.is_ok());
    assert!(second.is_ok());
    assert_eq!(stats.peak.get(), 1, "CPU capacity must span suite runs");
}

#[tokio::test(flavor = "current_thread")]
async fn memory_pools_are_shared_across_concurrent_suite_runs() {
    let scheduler = LocalNativeGraphSuiteScheduler::new(
        ResourceLimits::new(2, 2, 64, BTreeMap::new()).unwrap(),
    )
    .unwrap();
    let stats = Rc::new(ConcurrencyStats::default());
    let runner = Rc::new(UniformDelayedRunner {
        delay: Duration::from_millis(30),
        stats: stats.clone(),
    });

    let (first, second) = tokio::join!(
        run_resolved_suite(
            &scheduler,
            single_trial_suite(
                ResourceLeaseRequest::new(1, 64, BTreeMap::new()).unwrap(),
                "memory-a"
            ),
            runner.clone(),
        ),
        run_resolved_suite(
            &scheduler,
            single_trial_suite(
                ResourceLeaseRequest::new(1, 64, BTreeMap::new()).unwrap(),
                "memory-b"
            ),
            runner,
        ),
    );

    assert!(first.is_ok());
    assert!(second.is_ok());
    assert_eq!(stats.peak.get(), 1, "memory capacity must span suite runs");
}

#[tokio::test(flavor = "current_thread")]
async fn model_binding_weights_are_shared_across_concurrent_suite_runs() {
    let model_binding = primary_model_binding();
    let mut model_capacity = BTreeMap::new();
    model_capacity.insert(model_binding.clone(), 1);
    let scheduler = LocalNativeGraphSuiteScheduler::new(
        ResourceLimits::new(2, 2, 128, model_capacity).unwrap(),
    )
    .unwrap();
    let stats = Rc::new(ConcurrencyStats::default());
    let runner = Rc::new(UniformDelayedRunner {
        delay: Duration::from_millis(30),
        stats: stats.clone(),
    });
    let mut request_weights = BTreeMap::new();
    request_weights.insert(model_binding, 1);

    let (first, second) = tokio::join!(
        run_resolved_suite(
            &scheduler,
            single_trial_suite(
                ResourceLeaseRequest::new(1, 64, request_weights.clone()).unwrap(),
                "model-a",
            ),
            runner.clone(),
        ),
        run_resolved_suite(
            &scheduler,
            single_trial_suite(
                ResourceLeaseRequest::new(1, 64, request_weights).unwrap(),
                "model-b",
            ),
            runner,
        ),
    );

    assert!(first.is_ok());
    assert!(second.is_ok());
    assert_eq!(stats.peak.get(), 1, "model capacity must span suite runs");
}

#[tokio::test(flavor = "current_thread")]
async fn pending_suite_retries_when_another_suite_releases_capacity() {
    let local = LocalSet::new();
    local
        .run_until(async {
            let scheduler = Rc::new(
                LocalNativeGraphSuiteScheduler::new(
                    ResourceLimits::new(2, 2, 128, BTreeMap::new()).unwrap(),
                )
                .unwrap(),
            );
            let blocker_started = Rc::new(Notify::new());
            let release_blocker = Rc::new(Notify::new());
            let blocker = tokio::task::spawn_local({
                let scheduler = scheduler.clone();
                let blocker_started = blocker_started.clone();
                let release_blocker = release_blocker.clone();
                async move {
                    run_resolved_suite(
                        scheduler.as_ref(),
                        single_trial_suite(
                            ResourceLeaseRequest::new(1, 64, BTreeMap::new()).unwrap(),
                            "wake-blocker",
                        ),
                        Rc::new(ShortScoredRunner {
                            blocker_started,
                            release_blocker,
                        }),
                    )
                    .await
                }
            });
            blocker_started.notified().await;

            let long_started = Rc::new(Notify::new());
            let second_started = Rc::new(Notify::new());
            let allow_long_finish = Rc::new(Notify::new());
            let pending = tokio::task::spawn_local({
                let scheduler = scheduler.clone();
                let long_started = long_started.clone();
                let second_started = second_started.clone();
                let allow_long_finish = allow_long_finish.clone();
                async move {
                    run_resolved_suite(
                        scheduler.as_ref(),
                        two_trial_suite(
                            ResourceLeaseRequest::new(1, 64, BTreeMap::new()).unwrap(),
                            "wake-pending",
                        ),
                        Rc::new(LongThenScoredRunner {
                            long_started,
                            second_started,
                            allow_long_finish,
                        }),
                    )
                    .await
                }
            });
            long_started.notified().await;
            release_blocker.notify_one();

            assert!(
                tokio::time::timeout(Duration::from_millis(250), second_started.notified())
                    .await
                    .is_ok(),
                "the pending suite must retry after the other suite releases capacity"
            );
            allow_long_finish.notify_one();
            assert!(blocker.await.unwrap().is_ok());
            assert!(pending.await.unwrap().is_ok());
        })
        .await;
}

#[tokio::test(flavor = "current_thread")]
async fn runner_error_releases_a_global_resource_lease() {
    let scheduler = LocalNativeGraphSuiteScheduler::new(
        ResourceLimits::new(2, 1, 128, BTreeMap::new()).unwrap(),
    )
    .unwrap();
    let events = Rc::new(RefCell::new(Vec::new()));
    let runner = Rc::new(ErrorThenScoredRunner {
        calls: Cell::new(0),
        events: events.clone(),
    });

    let (first, second) = tokio::join!(
        run_resolved_suite(
            &scheduler,
            single_trial_suite(
                ResourceLeaseRequest::new(1, 64, BTreeMap::new()).unwrap(),
                "error-a"
            ),
            runner.clone(),
        ),
        run_resolved_suite(
            &scheduler,
            single_trial_suite(
                ResourceLeaseRequest::new(1, 64, BTreeMap::new()).unwrap(),
                "error-b"
            ),
            runner,
        ),
    );

    assert_eq!(
        first,
        Err(MatrixError::RunnerExecutionFailed(
            "injected runner failure".to_owned()
        ))
    );
    assert!(second.is_ok());
    assert_eq!(
        events.borrow().as_slice(),
        ["first-started", "first-failed", "second-started"],
        "the second run may start only after the first error released its lease"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn cancelled_run_releases_its_global_resource_lease() {
    let local = LocalSet::new();
    local
        .run_until(async {
            let scheduler = Rc::new(
                LocalNativeGraphSuiteScheduler::new(
                    ResourceLimits::new(1, 1, 64, BTreeMap::new()).unwrap(),
                )
                .unwrap(),
            );
            let started = Rc::new(Notify::new());
            let blocking = Rc::new(BlockingRunner {
                started: started.clone(),
            });
            let cancelled = tokio::task::spawn_local({
                let scheduler = scheduler.clone();
                async move {
                    run_resolved_suite(
                        scheduler.as_ref(),
                        single_trial_suite(
                            ResourceLeaseRequest::new(1, 64, BTreeMap::new()).unwrap(),
                            "cancelled",
                        ),
                        blocking,
                    )
                    .await
                }
            });
            started.notified().await;
            cancelled.abort();
            assert!(cancelled.await.unwrap_err().is_cancelled());

            let results = run_resolved_suite(
                scheduler.as_ref(),
                single_trial_suite(
                    ResourceLeaseRequest::new(1, 64, BTreeMap::new()).unwrap(),
                    "after-cancel",
                ),
                Rc::new(ImmediateScoredRunner),
            )
            .await
            .unwrap();
            assert_eq!(results.len(), 1);
        })
        .await;
}

#[tokio::test(flavor = "current_thread")]
async fn runner_cannot_substitute_a_result_for_another_attempt() {
    let scheduler = LocalNativeGraphSuiteScheduler::new(
        ResourceLimits::new(1, 1, 64, BTreeMap::new()).unwrap(),
    )
    .unwrap();

    let result =
        run_resolved_suite(&scheduler, four_trial_suite(), Rc::new(MismatchedRunner)).await;

    assert_eq!(
        result,
        Err(MatrixError::RunnerResultIdentityMismatch { output_index: 0 })
    );
}

#[test]
fn valid_failed_zero_score_remains_in_the_quality_denominator() {
    let summary = aggregate_episode_results([
        result(
            "failed",
            EpisodeIntegrity::Valid,
            EpisodeExecution::Failed,
            0.0,
        ),
        result(
            "completed",
            EpisodeIntegrity::Valid,
            EpisodeExecution::Completed,
            1.0,
        ),
        result(
            "invalid",
            EpisodeIntegrity::InvalidProvider,
            EpisodeExecution::Failed,
            0.0,
        ),
    ])
    .unwrap();

    assert_eq!(
        (summary.valid_attempts(), summary.invalid_attempts()),
        (2, 1)
    );
    assert_eq!(summary.mean_reward(), Some(0.5));
}

#[test]
fn truncated_or_unscored_results_are_retained_but_excluded_from_reward_aggregation() {
    let summary = aggregate_episode_results([
        result_with_axes(
            "scored-failure",
            EpisodeIntegrity::Valid,
            EpisodeExecution::Failed,
            EpisodeScoreState::Verified { reward: 0.0 },
            EpisodeComparability::Scored,
        ),
        result_with_axes(
            "verified-but-unscored",
            EpisodeIntegrity::Valid,
            EpisodeExecution::Completed,
            EpisodeScoreState::Verified { reward: 1.0 },
            EpisodeComparability::Unscored,
        ),
        result_with_axes(
            "truncated-unavailable",
            EpisodeIntegrity::Valid,
            EpisodeExecution::Truncated,
            EpisodeScoreState::Unavailable,
            EpisodeComparability::Unscored,
        ),
        result_with_axes(
            "invalid-scored",
            EpisodeIntegrity::InvalidEvidence,
            EpisodeExecution::Completed,
            EpisodeScoreState::Verified { reward: 1.0 },
            EpisodeComparability::Scored,
        ),
    ])
    .unwrap();

    assert_eq!(summary.valid_attempts(), 3);
    assert_eq!(summary.invalid_attempts(), 1);
    assert_eq!(summary.scored_valid_attempts(), 1);
    assert_eq!(summary.unscored_valid_attempts(), 2);
    assert_eq!(summary.mean_reward(), Some(0.0));
}

#[derive(Default)]
struct ConcurrencyStats {
    current: Cell<usize>,
    peak: Cell<usize>,
    completions: RefCell<Vec<usize>>,
}

struct DelayedScoredRunner {
    delays_ms: [u64; 4],
    stats: Rc<ConcurrencyStats>,
}

struct UniformDelayedRunner {
    delay: Duration,
    stats: Rc<ConcurrencyStats>,
}

struct ErrorThenScoredRunner {
    calls: Cell<usize>,
    events: Rc<RefCell<Vec<&'static str>>>,
}

struct ShortScoredRunner {
    blocker_started: Rc<Notify>,
    release_blocker: Rc<Notify>,
}

struct LongThenScoredRunner {
    long_started: Rc<Notify>,
    second_started: Rc<Notify>,
    allow_long_finish: Rc<Notify>,
}

struct BlockingRunner {
    started: Rc<Notify>,
}

struct ImmediateScoredRunner;

struct MismatchedRunner;

#[async_trait(?Send)]
impl EpisodeRunner for MismatchedRunner {
    async fn run(
        &self,
        assignment: aiperf_runtime::eval::EpisodeAssignment,
    ) -> Result<EpisodeResult, aiperf_runtime::eval::MatrixError> {
        Ok(EpisodeResult::new(
            assignment.trial_digest().clone(),
            AttemptId::new("wrong-attempt").unwrap(),
            EpisodeIntegrity::Valid,
            EpisodeExecution::Completed,
            EpisodeScoreState::Verified { reward: 1.0 },
            EpisodeComparability::Scored,
            Vec::new(),
        )
        .unwrap())
    }
}

#[async_trait(?Send)]
impl EpisodeRunner for DelayedScoredRunner {
    async fn run(
        &self,
        assignment: aiperf_runtime::eval::EpisodeAssignment,
    ) -> Result<EpisodeResult, aiperf_runtime::eval::MatrixError> {
        let current = self.stats.current.get() + 1;
        self.stats.current.set(current);
        self.stats.peak.set(self.stats.peak.get().max(current));
        tokio::time::sleep(Duration::from_millis(
            self.delays_ms[assignment.manifest_index()],
        ))
        .await;
        self.stats
            .completions
            .borrow_mut()
            .push(assignment.manifest_index());
        self.stats.current.set(self.stats.current.get() - 1);
        Ok(EpisodeResult::new(
            assignment.trial_digest().clone(),
            assignment.attempt_id().clone(),
            EpisodeIntegrity::Valid,
            EpisodeExecution::Completed,
            EpisodeScoreState::Verified { reward: 1.0 },
            EpisodeComparability::Scored,
            Vec::new(),
        )
        .unwrap())
    }
}

#[async_trait(?Send)]
impl EpisodeRunner for UniformDelayedRunner {
    async fn run(
        &self,
        assignment: aiperf_runtime::eval::EpisodeAssignment,
    ) -> Result<EpisodeResult, aiperf_runtime::eval::MatrixError> {
        let current = self.stats.current.get() + 1;
        self.stats.current.set(current);
        self.stats.peak.set(self.stats.peak.get().max(current));
        tokio::time::sleep(self.delay).await;
        self.stats.current.set(self.stats.current.get() - 1);
        scored_result(assignment)
    }
}

#[async_trait(?Send)]
impl EpisodeRunner for ErrorThenScoredRunner {
    async fn run(
        &self,
        assignment: aiperf_runtime::eval::EpisodeAssignment,
    ) -> Result<EpisodeResult, aiperf_runtime::eval::MatrixError> {
        let call = self.calls.get();
        self.calls.set(call + 1);
        if call == 0 {
            self.events.borrow_mut().push("first-started");
            tokio::time::sleep(Duration::from_millis(30)).await;
            self.events.borrow_mut().push("first-failed");
            return Err(aiperf_runtime::eval::MatrixError::RunnerExecutionFailed(
                "injected runner failure".to_owned(),
            ));
        }
        self.events.borrow_mut().push("second-started");
        scored_result(assignment)
    }
}

#[async_trait(?Send)]
impl EpisodeRunner for ShortScoredRunner {
    async fn run(
        &self,
        assignment: aiperf_runtime::eval::EpisodeAssignment,
    ) -> Result<EpisodeResult, aiperf_runtime::eval::MatrixError> {
        self.blocker_started.notify_one();
        self.release_blocker.notified().await;
        scored_result(assignment)
    }
}

#[async_trait(?Send)]
impl EpisodeRunner for LongThenScoredRunner {
    async fn run(
        &self,
        assignment: aiperf_runtime::eval::EpisodeAssignment,
    ) -> Result<EpisodeResult, aiperf_runtime::eval::MatrixError> {
        if assignment.manifest_index() == 0 {
            self.long_started.notify_one();
            self.allow_long_finish.notified().await;
        } else {
            self.second_started.notify_one();
        }
        scored_result(assignment)
    }
}

#[async_trait(?Send)]
impl EpisodeRunner for BlockingRunner {
    async fn run(
        &self,
        _assignment: aiperf_runtime::eval::EpisodeAssignment,
    ) -> Result<EpisodeResult, aiperf_runtime::eval::MatrixError> {
        self.started.notify_one();
        std::future::pending().await
    }
}

#[async_trait(?Send)]
impl EpisodeRunner for ImmediateScoredRunner {
    async fn run(
        &self,
        assignment: aiperf_runtime::eval::EpisodeAssignment,
    ) -> Result<EpisodeResult, aiperf_runtime::eval::MatrixError> {
        scored_result(assignment)
    }
}

fn scored_result(
    assignment: aiperf_runtime::eval::EpisodeAssignment,
) -> Result<EpisodeResult, aiperf_runtime::eval::MatrixError> {
    EpisodeResult::new(
        assignment.trial_digest().clone(),
        assignment.attempt_id().clone(),
        EpisodeIntegrity::Valid,
        EpisodeExecution::Completed,
        EpisodeScoreState::Verified { reward: 1.0 },
        EpisodeComparability::Scored,
        Vec::new(),
    )
    .map_err(|_| {
        aiperf_runtime::eval::MatrixError::RunnerExecutionFailed(
            "test result construction failed".to_owned(),
        )
    })
}

fn result(
    id: &str,
    integrity: EpisodeIntegrity,
    execution: EpisodeExecution,
    reward: f64,
) -> EpisodeResult {
    EpisodeResult::new(
        ArtifactDigest::from_bytes(id.as_bytes()),
        AttemptId::new(format!("attempt-{id}")).unwrap(),
        integrity,
        execution,
        EpisodeScoreState::Verified { reward },
        EpisodeComparability::Scored,
        Vec::new(),
    )
    .unwrap()
}

fn result_with_axes(
    id: &str,
    integrity: EpisodeIntegrity,
    execution: EpisodeExecution,
    score: EpisodeScoreState,
    comparability: EpisodeComparability,
) -> EpisodeResult {
    EpisodeResult::new(
        ArtifactDigest::from_bytes(id.as_bytes()),
        AttemptId::new(format!("attempt-{id}")).unwrap(),
        integrity,
        execution,
        score,
        comparability,
        Vec::new(),
    )
    .unwrap()
}

fn four_trial_suite() -> aiperf_runtime::eval::ResolvedNativeGraphSuite {
    let task = native_task_fixture();
    let source = HarborSource::local(task.path().to_string_lossy()).unwrap();
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&source)
        .unwrap();
    let lease = ResourceLeaseRequest::new(1, 64, BTreeMap::new()).unwrap();
    NativeGraphSuiteManifest::new(
        [3, 5, 7, 11]
            .into_iter()
            .map(|seed| {
                SuiteTrialSpec::from_imported(
                    imported.clone(),
                    trial(imported.task.clone(), seed),
                    NonZeroUsize::new(1).unwrap(),
                    lease.clone(),
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap()
    .resolve(SuiteRunId::new(ArtifactDigest::from_bytes(b"matrix-run")))
    .unwrap()
}

fn single_trial_suite(
    resources: ResourceLeaseRequest,
    run_id: &str,
) -> aiperf_runtime::eval::ResolvedNativeGraphSuite {
    let task = native_task_fixture();
    let source = HarborSource::local(task.path().to_string_lossy()).unwrap();
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&source)
        .unwrap();
    NativeGraphSuiteManifest::new(vec![
        SuiteTrialSpec::from_imported(
            imported.clone(),
            trial(imported.task.clone(), 17),
            NonZeroUsize::new(1).unwrap(),
            resources,
        )
        .unwrap(),
    ])
    .unwrap()
    .resolve(SuiteRunId::new(ArtifactDigest::from_bytes(
        run_id.as_bytes(),
    )))
    .unwrap()
}

fn two_trial_suite(
    resources: ResourceLeaseRequest,
    run_id: &str,
) -> aiperf_runtime::eval::ResolvedNativeGraphSuite {
    let task = native_task_fixture();
    let source = HarborSource::local(task.path().to_string_lossy()).unwrap();
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&source)
        .unwrap();
    NativeGraphSuiteManifest::new(
        [17, 19]
            .into_iter()
            .map(|seed| {
                SuiteTrialSpec::from_imported(
                    imported.clone(),
                    trial(imported.task.clone(), seed),
                    NonZeroUsize::new(1).unwrap(),
                    resources.clone(),
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap()
    .resolve(SuiteRunId::new(ArtifactDigest::from_bytes(
        run_id.as_bytes(),
    )))
    .unwrap()
}

fn primary_model_binding() -> ModelCapacityKey {
    let task = native_task_fixture();
    let source = HarborSource::local(task.path().to_string_lossy()).unwrap();
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&source)
        .unwrap();
    ModelCapacityKey::from_task_binding(
        &imported.task,
        &imported.package.native_graph().unwrap().model_bindings()[0],
    )
}

fn trial(task: aiperf_runtime::eval::EvalTaskRef, seed: u64) -> TrialSpec {
    TrialSpec::new(
        task,
        AgentVariantRef::new("native-graph").unwrap(),
        ModelIdentity::new("provider-default", "example-model").unwrap(),
        seed,
        PolicyIdentity::new(ArtifactDigest::from_bytes(b"policy")),
        TrialBudget::new(30.0, 30.0).unwrap(),
        ArtifactDigest::from_bytes(b"environment"),
        ArtifactDigest::from_bytes(b"verifier"),
        RuntimeIdentity::new("native").unwrap(),
    )
    .unwrap()
}

fn native_task_fixture() -> tempfile::TempDir {
    let task = tempfile::tempdir().unwrap();
    fs::create_dir_all(task.path().join("environment")).unwrap();
    fs::create_dir_all(task.path().join("tests")).unwrap();
    fs::create_dir_all(task.path().join("tools")).unwrap();
    fs::write(
        task.path().join("environment/Dockerfile"),
        b"FROM scratch\n",
    )
    .unwrap();
    fs::write(task.path().join("instruction.md"), b"Do work.\n").unwrap();
    fs::write(task.path().join("tests/test.sh"), b"exit 0\n").unwrap();
    fs::write(
        task.path().join("task.toml"),
        r#"schema_version = "1.1"

[task]
name = "example/native-graph-matrix"

[native_graph]
profile = "native_graph"
program = "agent_graph.json"
model_bindings = "models.toml"
adapter_manifest = "adapters.toml"
"#,
    )
    .unwrap();
    fs::write(task.path().join("agent_graph.json"), b"{}\n").unwrap();
    fs::write(
        task.path().join("models.toml"),
        r#"[[model_bindings]]
id = "primary"
endpoint_profile_id = "provider-default"
endpoint_factory_id = "chat"
transport_factory_id = "http"
model = "example-model"
urls = ["https://provider.example/v1"]
streaming = true
request_timeout_ms = 30000
capture = "metadata"

[model_bindings.tokenizer]
type = "local"
name = "builtin"
revision = "main"
apply_chat_template = true

[model_bindings.generation]
"#,
    )
    .unwrap();
    fs::write(
        task.path().join("adapters.toml"),
        r#"[[adapters]]
id = "tool-adapter"
role = "tool"
argv = ["tools/adapter.py"]
executable = "tools/adapter.py"
"#,
    )
    .unwrap();
    fs::write(task.path().join("tools/adapter.py"), b"print('adapter')\n").unwrap();
    task
}
