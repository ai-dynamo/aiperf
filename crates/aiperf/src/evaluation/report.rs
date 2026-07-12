// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native-v2 report join for provider-neutral evaluation runs.
//!
//! This module does not aggregate provider scores. It joins provider-authored
//! semantic outcomes with factory-approved public projections, Rust-owned
//! route/accounting facts, and Rust-sealed artifacts.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use aiperf_accuracy::{
    ArtifactVisibility, CanonicalJson, CaseOutcomeKind, EvaluationCaseId, EvaluationCaseTemplateId,
    EvaluationFinishCandidate, EvaluationIdentityComponent, PublicAggregateProjectionPolicy,
    PublicEvaluationMetadataProjector, PublicScoreProjectionPolicy, SealedEvaluationArtifacts,
    ValidatedPublicCaseProjections, is_sha256, validate_no_secret_control_value,
};
use aiperf_metrics::{
    EvaluationAggregateMetricReport, EvaluationArtifactReport, EvaluationCaseErrorReport,
    EvaluationCaseOutcomeKind, EvaluationCaseReport, EvaluationIdentityReport,
    EvaluationPublicScoreReport, EvaluationReport, EvaluationRouteReport,
    EvaluationRouteSummaryReport,
};
use anyhow::{Result, anyhow, ensure};
use serde_json::{Value, json};

use super::host::EvaluationRouteTable;

/// Report-safe occurrence identity retained by the Rust evaluation workload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluationCaseReportFacts {
    /// Frozen provider case template.
    pub template_id: EvaluationCaseTemplateId,
    /// Model-safe task label copied from the frozen template.
    pub task: String,
    /// Model-safe immutable source label copied from the frozen template.
    pub source: String,
}

/// Rust/provider facts required to build the generic native-v2 evaluation block.
#[derive(Clone)]
pub struct EvaluationReportFacts {
    /// Factory-schema-projected, secret-free resolved configuration.
    pub safe_config: Value,
    /// Exact occurrence-to-template/reporting identity map.
    pub cases: BTreeMap<EvaluationCaseId, EvaluationCaseReportFacts>,
    /// Factory-owned executable public score projection validators.
    pub public_score_projection_policy: PublicScoreProjectionPolicy,
    /// Factory-owned executable public aggregate projection validators.
    pub public_aggregate_projection_policy: PublicAggregateProjectionPolicy,
    /// Factory-owned executable case/numeric metadata projector.
    pub public_metadata_projector: Arc<dyn PublicEvaluationMetadataProjector>,
    /// Rust-authoritative per-route accounting summaries.
    pub route_summaries: BTreeMap<String, EvaluationRouteSummaryReport>,
}

impl fmt::Debug for EvaluationReportFacts {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EvaluationReportFacts")
            .field("case_count", &self.cases.len())
            .field(
                "public_score_projection_policy",
                &self.public_score_projection_policy,
            )
            .field(
                "public_aggregate_projection_policy",
                &self.public_aggregate_projection_policy,
            )
            .field(
                "public_metadata_schema_sha256",
                &self.public_metadata_projector.schema_sha256(),
            )
            .field("route_summary_count", &self.route_summaries.len())
            .finish()
    }
}

/// Convert one validated provider candidate and sealed tree into native-v2 output.
pub fn build_evaluation_report(
    candidate: &EvaluationFinishCandidate,
    sealed: &SealedEvaluationArtifacts,
    routes: &EvaluationRouteTable,
    facts: &EvaluationReportFacts,
) -> Result<EvaluationReport> {
    let mut candidate = candidate.clone();
    candidate
        .validate()
        .map_err(|error| anyhow!(error.to_string()))?;
    let safe_config = CanonicalJson::new(facts.safe_config.clone())
        .map_err(|error| anyhow!(error.to_string()))?;
    validate_no_secret_control_value(&safe_config).map_err(|error| anyhow!(error.to_string()))?;
    let public_cases = ValidatedPublicCaseProjections::new(
        &candidate.outcomes,
        &facts.public_score_projection_policy,
    )
    .map_err(|error| anyhow!(error.to_string()))?;

    let route_reports = build_routes(routes, &facts.route_summaries)?;
    validate_case_fact_manifest(&candidate, &facts.cases)?;
    let sealed_by_id = validate_sealed_manifest(&candidate, sealed)?;
    let artifact_refs_by_id = sealed
        .entries
        .iter()
        .enumerate()
        .map(|(ordinal, artifact)| (&artifact.artifact_id, opaque_artifact_ref(ordinal)))
        .collect::<BTreeMap<_, _>>();

    let templates = candidate
        .identity
        .case_templates
        .iter()
        .map(|template| (&template.template_id, template))
        .collect::<BTreeMap<_, _>>();
    let template_ordinals = candidate
        .identity
        .case_templates
        .iter()
        .enumerate()
        .map(|(ordinal, template)| (&template.template_id, ordinal))
        .collect::<BTreeMap<_, _>>();
    let mut completed_count = 0_usize;
    let mut infrastructure_error_count = 0_usize;
    let mut cancelled_count = 0_usize;
    let mut case_reports = Vec::with_capacity(candidate.outcomes.len());
    for (case_ordinal, outcome) in candidate.outcomes.iter().enumerate() {
        let case_facts = facts.cases.get(&outcome.case_id).ok_or_else(|| {
            anyhow!(
                "missing report identity for evaluation case {}",
                outcome.case_id
            )
        })?;
        let template = templates.get(&case_facts.template_id).ok_or_else(|| {
            anyhow!(
                "case {} referenced unknown template {}",
                outcome.case_id,
                case_facts.template_id
            )
        })?;
        ensure!(
            case_facts.task == template.task && case_facts.source == template.source,
            "case {} reporting labels drifted from frozen template {}",
            outcome.case_id,
            case_facts.template_id
        );
        let template_ordinal = template_ordinals
            .get(&case_facts.template_id)
            .copied()
            .ok_or_else(|| anyhow!("case report template omitted its host ordinal"))?;
        let artifact_refs = outcome
            .artifact_refs
            .iter()
            .map(|artifact| {
                let sealed = sealed_by_id.get(&artifact.artifact_id).ok_or_else(|| {
                    anyhow!(
                        "case {} referenced unsealed artifact {}",
                        outcome.case_id,
                        artifact.artifact_id
                    )
                })?;
                ensure!(
                    sealed.path == artifact.path && sealed.visibility == artifact.visibility,
                    "case {} artifact {} identity drifted after sealing",
                    outcome.case_id,
                    artifact.artifact_id
                );
                artifact_refs_by_id
                    .get(&artifact.artifact_id)
                    .cloned()
                    .ok_or_else(|| anyhow!("sealed artifact omitted its opaque report reference"))
            })
            .collect::<Result<Vec<_>>>()?;
        let projected_case = facts
            .public_metadata_projector
            .project_case(&case_facts.task, &case_facts.source)
            .map_err(|error| anyhow!(error.to_string()))?;
        let (task, source) = projected_case
            .map(|projection| (projection.task, projection.source))
            .unwrap_or_else(|| {
                (
                    format!("task-{template_ordinal:08}"),
                    format!("source-{template_ordinal:08}"),
                )
            });
        let (kind, scores, numeric_metrics, primary_score, error) = match &outcome.outcome {
            CaseOutcomeKind::Completed { completed } => {
                completed_count += 1;
                let validated_scores = public_cases.scores(&outcome.case_id).ok_or_else(|| {
                    anyhow!("completed case omitted its validated public projection set")
                })?;
                let mut scores = BTreeMap::new();
                let mut public_score_names = BTreeMap::new();
                let mut numeric_metrics = BTreeMap::new();
                for (provider_name, projection) in validated_scores {
                    let public_name = facts
                        .public_metadata_projector
                        .project_numeric_metric(provider_name)
                        .map_err(|error| anyhow!(error.to_string()))?
                        .ok_or_else(|| {
                            anyhow!("validated public score omitted its factory-owned report label")
                        })?;
                    ensure!(
                        public_score_names
                            .insert(provider_name.clone(), public_name.clone())
                            .is_none()
                            && scores
                                .insert(
                                    public_name.clone(),
                                    EvaluationPublicScoreReport {
                                        value: projection.value().value().clone(),
                                        projection_schema: projection.schema_sha256().to_string(),
                                    },
                                )
                                .is_none(),
                        "public score projection produced a duplicate stable label"
                    );
                    let numeric_value = projection.value().value().as_f64().ok_or_else(|| {
                        anyhow!("validated public score was not a direct JSON number")
                    })?;
                    numeric_metrics.insert(public_name, numeric_value);
                }
                for (name, value) in &completed.numeric_metrics {
                    let Some(public_name) = facts
                        .public_metadata_projector
                        .project_numeric_metric(name)
                        .map_err(|error| anyhow!(error.to_string()))?
                    else {
                        continue;
                    };
                    if let Some(projected_score) = numeric_metrics.get(&public_name) {
                        ensure!(
                            *projected_score == value.get(),
                            "public numeric metric disagreed with its validated public score"
                        );
                    } else {
                        ensure!(
                            numeric_metrics.insert(public_name, value.get()).is_none(),
                            "public numeric metric projection produced a duplicate name"
                        );
                    }
                }
                let primary_score = completed
                    .primary_score
                    .as_ref()
                    .and_then(|name| public_score_names.get(name))
                    .cloned();
                (
                    EvaluationCaseOutcomeKind::Completed,
                    scores,
                    numeric_metrics,
                    primary_score,
                    None,
                )
            }
            CaseOutcomeKind::InfrastructureError { error } => {
                infrastructure_error_count += 1;
                (
                    EvaluationCaseOutcomeKind::InfrastructureError,
                    BTreeMap::new(),
                    BTreeMap::new(),
                    None,
                    Some(EvaluationCaseErrorReport {
                        stage: error.stage.to_string(),
                        kind: error.error_kind.clone(),
                        retryable: error.retryable,
                        message: "Evaluator provider reported an infrastructure error".to_string(),
                    }),
                )
            }
            CaseOutcomeKind::Cancelled { stage, reason: _ } => {
                cancelled_count += 1;
                (
                    EvaluationCaseOutcomeKind::Cancelled,
                    BTreeMap::new(),
                    BTreeMap::new(),
                    None,
                    Some(EvaluationCaseErrorReport {
                        stage: stage.to_string(),
                        kind: "cancelled".to_string(),
                        retryable: false,
                        message: "Evaluation was cancelled".to_string(),
                    }),
                )
            }
        };
        case_reports.push(EvaluationCaseReport {
            case_id: format!("case-{case_ordinal:08}"),
            template_id: format!("template-{template_ordinal:08}"),
            task,
            source,
            outcome: kind,
            scores,
            numeric_metrics,
            primary_score,
            error,
            artifact_refs,
        });
    }
    ensure!(
        case_reports.len() == public_cases.case_count()
            && completed_count == public_cases.completed_count()
            && infrastructure_error_count == public_cases.infrastructure_error_count()
            && cancelled_count == public_cases.cancelled_count(),
        "public case projection counts drifted during report construction"
    );

    let aggregates = facts
        .public_aggregate_projection_policy
        .project(&candidate.aggregates, &public_cases, &safe_config)
        .map_err(|error| anyhow!(error.to_string()))?
        .into_iter()
        .map(|projection| {
            Ok(EvaluationAggregateMetricReport {
                scorer: projection.scorer().to_string(),
                reducer: projection.reducer().to_string(),
                metric: projection.metric().to_string(),
                value: projection.value(),
                scored_count: usize::try_from(projection.scored_count())
                    .map_err(|_| anyhow!("aggregate scored count exceeded host size"))?,
                unscored_count: usize::try_from(projection.unscored_count())
                    .map_err(|_| anyhow!("aggregate unscored count exceeded host size"))?,
                projection_schema: projection.schema_sha256().to_string(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let artifact_reports = sealed
        .entries
        .iter()
        .enumerate()
        .map(|(ordinal, artifact)| EvaluationArtifactReport {
            artifact_ref: opaque_artifact_ref(ordinal),
            path: (artifact.visibility == ArtifactVisibility::PublicProjection)
                .then(|| artifact.path.clone()),
            media_type: (artifact.visibility == ArtifactVisibility::PublicProjection)
                .then(|| artifact.media_type.clone()),
            visibility: match artifact.visibility {
                ArtifactVisibility::Restricted => "restricted",
                ArtifactVisibility::PublicProjection => "public",
            }
            .to_string(),
            size_bytes: (artifact.visibility == ArtifactVisibility::PublicProjection)
                .then_some(artifact.size_bytes),
            artifact_content_sha256: (artifact.visibility == ArtifactVisibility::PublicProjection)
                .then(|| artifact.artifact_content_sha256.clone()),
            projection_schema: (artifact.visibility == ArtifactVisibility::PublicProjection)
                .then(|| artifact.public_projection_schema_sha256.clone())
                .flatten(),
        })
        .collect();

    let identity = build_identity(&candidate)?;
    Ok(EvaluationReport {
        identity,
        config: safe_config.into_value(),
        routes: route_reports,
        case_count: case_reports.len(),
        completed_count,
        infrastructure_error_count,
        cancelled_count,
        cases: case_reports,
        aggregates,
        route_summaries: facts.route_summaries.clone(),
        artifacts: artifact_reports,
    })
}

fn opaque_artifact_ref(ordinal: usize) -> String {
    format!("artifact-{ordinal:08}")
}

fn build_routes(
    routes: &EvaluationRouteTable,
    summaries: &BTreeMap<String, EvaluationRouteSummaryReport>,
) -> Result<Vec<EvaluationRouteReport>> {
    let route_ids = routes
        .routes()
        .map(|route| route.service_id.as_str())
        .collect::<BTreeSet<_>>();
    let summary_ids = summaries
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    ensure!(
        route_ids == summary_ids,
        "evaluation route summaries did not exactly match the frozen route table"
    );
    routes
        .routes()
        .map(|route| {
            route.validate()?;
            Ok(EvaluationRouteReport {
                service_id: route.service_id.clone(),
                purpose: route.purpose.clone(),
                model: route.model.clone(),
                endpoint_profile: route.endpoint_profile.clone(),
                prepared_identity_sha256: route.prepared_identity_sha256.clone(),
            })
        })
        .collect()
}

fn validate_case_fact_manifest(
    candidate: &EvaluationFinishCandidate,
    facts: &BTreeMap<EvaluationCaseId, EvaluationCaseReportFacts>,
) -> Result<()> {
    let outcomes = candidate
        .outcomes
        .iter()
        .map(|outcome| &outcome.case_id)
        .collect::<BTreeSet<_>>();
    let fact_ids = facts.keys().collect::<BTreeSet<_>>();
    ensure!(
        outcomes == fact_ids,
        "evaluation case report facts did not exactly match provider outcomes"
    );
    Ok(())
}

fn validate_sealed_manifest<'a>(
    candidate: &EvaluationFinishCandidate,
    sealed: &'a SealedEvaluationArtifacts,
) -> Result<
    BTreeMap<
        &'a aiperf_accuracy::EvaluationArtifactId,
        &'a aiperf_accuracy::SealedEvaluationArtifact,
    >,
> {
    ensure!(
        is_sha256(&sealed.provider_bundle_sha256) && is_sha256(&sealed.quiescence_proof_sha256),
        "sealed evaluation artifact proof contained an invalid digest"
    );
    ensure!(
        candidate.artifacts.len() == sealed.entries.len(),
        "sealed evaluation artifact count differed from candidate manifest"
    );
    let sealed_by_id = sealed
        .entries
        .iter()
        .map(|artifact| (&artifact.artifact_id, artifact))
        .collect::<BTreeMap<_, _>>();
    ensure!(
        sealed_by_id.len() == sealed.entries.len(),
        "sealed evaluation artifacts contained duplicate IDs"
    );
    for (candidate_entry, sealed_entry) in candidate.artifacts.iter().zip(&sealed.entries) {
        ensure!(
            candidate_entry.artifact_id == sealed_entry.artifact_id
                && candidate_entry.path == sealed_entry.path
                && candidate_entry.media_type == sealed_entry.media_type
                && candidate_entry.visibility == sealed_entry.visibility,
            "sealed evaluation artifact order or identity drifted from candidate manifest"
        );
        ensure!(
            is_sha256(&sealed_entry.artifact_content_sha256),
            "sealed evaluation artifact contained an invalid content digest"
        );
        match sealed_entry.visibility {
            ArtifactVisibility::Restricted => {
                ensure!(
                    sealed_entry.public_projection_schema_sha256.is_none()
                        && sealed_entry.size_bytes == candidate_entry.size_bytes
                        && sealed_entry.artifact_content_sha256
                            == candidate_entry.artifact_content_sha256,
                    "restricted evaluation artifact drifted after sealing"
                );
            }
            ArtifactVisibility::PublicProjection => {
                ensure!(
                    sealed_entry
                        .public_projection_schema_sha256
                        .as_deref()
                        .is_some_and(is_sha256),
                    "public evaluation artifact omitted its factory schema digest"
                );
            }
        }
    }
    let bundle = sealed_by_id
        .get(&candidate.provider_bundle.artifact_id)
        .ok_or_else(|| anyhow!("sealed artifacts omitted canonical provider bundle"))?;
    ensure!(
        candidate.provider_bundle.visibility == ArtifactVisibility::Restricted
            && bundle.visibility == ArtifactVisibility::Restricted,
        "canonical provider bundle must remain restricted"
    );
    ensure!(
        bundle.path == candidate.provider_bundle.path
            && bundle.artifact_content_sha256 == sealed.provider_bundle_sha256,
        "canonical provider bundle path/digest drifted after sealing"
    );
    Ok(sealed_by_id)
}

fn build_identity(candidate: &EvaluationFinishCandidate) -> Result<EvaluationIdentityReport> {
    let source = &candidate.identity;
    let host_identity = CanonicalJson::new(json!({
        "host": {
            "capability_inventory_sha256": source.host.capability_inventory_sha256,
            "isolation_proof_sha256": source.host.isolation_proof_sha256,
            "runner_sha256": source.host.runner_sha256,
            "schema_inventory_sha256": source.host.schema_inventory_sha256,
        },
        "prepared_endpoints_sha256": source.prepared_endpoints_sha256,
        "route_map_sha256": source.route_map_sha256,
        "sandbox_sha256": source.sandbox_sha256,
    }))
    .map_err(|error| anyhow!(error.to_string()))?
    .normalized_result_sha256();
    let mut components = BTreeMap::new();
    project_identity_component("dataset", &source.dataset, &mut components)?;
    for (ordinal, component) in source.components.iter().enumerate() {
        project_identity_component(
            &format!("component-{ordinal:08}"),
            component,
            &mut components,
        )?;
    }
    Ok(EvaluationIdentityReport {
        evaluator_protocol: source.worker.evaluator_protocol,
        provider: source.worker.provider_id.to_string(),
        distribution: source.worker.distribution_id.to_string(),
        provider_source_sha256: source.worker.provider_source_sha256.clone(),
        worker_source_sha256: source.worker.worker_source_sha256.clone(),
        dependency_lock_sha256: source.worker.dependency_lock_sha256.clone(),
        authored_schema_fingerprint: source.config_schema_sha256.clone(),
        resolved_config_sha256: source.resolved_config_sha256.clone(),
        ordered_manifest_sha256: source.ordered_manifest_sha256.clone(),
        host_identity_sha256: host_identity,
        isolation_proof_sha256: source.host.isolation_proof_sha256.clone(),
        container_digest: source.worker.oci_digest.clone(),
        components,
    })
}

fn project_identity_component(
    prefix: &str,
    component: &EvaluationIdentityComponent,
    output: &mut BTreeMap<String, String>,
) -> Result<()> {
    let mut insert = |suffix: &str, value: String| {
        ensure!(
            output.insert(format!("{prefix}.{suffix}"), value).is_none(),
            "duplicate host-projected evaluation identity fact"
        );
        Ok(())
    };
    insert("effective_source_sha256", component.source_sha256.clone())?;
    if let Some(source_commit) = &component.source_commit {
        insert("source_commit", source_commit.clone())?;
    }
    if let Some(base_source_sha256) = &component.base_source_sha256 {
        insert("base_source_sha256", base_source_sha256.clone())?;
    }
    if let Some(overlay_policy) = &component.overlay_policy {
        insert("overlay_policy", overlay_policy.clone())?;
    }
    for (ordinal, overlay) in component.overlays.iter().enumerate() {
        insert(
            &format!("overlay-{ordinal:08}.artifact_content_sha256"),
            overlay.artifact_content_sha256.clone(),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use aiperf_accuracy::{
        AggregateMetric, ArtifactRef, CaseOutcome, CompletedCaseOutcome, EvaluationArtifactId,
        EvaluationArtifactManifestEntry, EvaluationCaseTemplateDescriptor,
        EvaluationDistributionId, EvaluationError, EvaluationExecutionGranularity,
        EvaluationHostIdentity, EvaluationIdentity, EvaluationIdentityComponent,
        EvaluationProviderId, EvaluationSourceOverlayIdentity, EvaluationStage,
        EvaluationUnitTemplateDescriptor, EvaluationUnitTemplateId, EvaluationWorkerIdentity,
        ExactBinaryMeanAggregateValidator, FiniteBinaryNumberProjectionValidator, FiniteF64,
        FrozenPublicEvaluationMetadataPolicy, ProviderScore, PublicAggregateMetadataRule,
        PublicAggregateProjectionPolicy, PublicCaseMetadataRule, PublicNumericMetricRule,
        SOURCE_OVERLAY_POLICY_V1, SealedEvaluationArtifact,
    };

    use super::*;
    use crate::evaluation::host::{EvaluationRoute, EvaluationRouteTable};

    fn worker() -> EvaluationWorkerIdentity {
        EvaluationWorkerIdentity {
            evaluator_protocol: 2,
            provider_id: EvaluationProviderId::new("fixture").unwrap(),
            distribution_id: EvaluationDistributionId::new("fixture-dist").unwrap(),
            package: "fixture".to_string(),
            package_version: "1".to_string(),
            provider_source_sha256: "a".repeat(64),
            worker_source_sha256: "b".repeat(64),
            dependency_lock_sha256: "c".repeat(64),
            python_version: "3.12".to_string(),
            launch_nonce: "n".repeat(32),
            oci_digest: None,
            operations: [
                "plan_session",
                "bind_assets",
                "next_units",
                "instantiate_units",
                "start_units",
                "poll_events",
                "submit_host_events",
                "cancel_units",
                "finalize_session",
                "shutdown",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
        }
    }

    fn candidate() -> EvaluationFinishCandidate {
        let case_templates = ["completed", "infra", "cancelled"]
            .into_iter()
            .map(|name| EvaluationCaseTemplateDescriptor {
                template_id: aiperf_accuracy::EvaluationCaseTemplateId::new(format!(
                    "hidden-template-{name}-sentinel"
                ))
                .unwrap(),
                task: format!("hidden-task-{name}-sentinel"),
                source: "hidden-source-sentinel".to_string(),
            })
            .collect::<Vec<_>>();
        let unit_templates = case_templates
            .iter()
            .map(|case| EvaluationUnitTemplateDescriptor {
                unit_template_id: EvaluationUnitTemplateId::new(format!(
                    "unit-{}",
                    case.template_id
                ))
                .unwrap(),
                case_template_ids: vec![case.template_id.clone()],
                granularity: EvaluationExecutionGranularity::Case,
                scheduling_class: "fixture".to_string(),
            })
            .collect();
        let bundle_id = EvaluationArtifactId::new("hidden-artifact-id-sentinel").unwrap();
        EvaluationFinishCandidate {
            identity: EvaluationIdentity {
                canonical_json_codec: aiperf_accuracy::CANONICAL_JSON_CODEC.to_string(),
                worker: worker(),
                config_schema_sha256: "d".repeat(64),
                resolved_config_sha256: "e".repeat(64),
                dataset: EvaluationIdentityComponent {
                    name: "hidden-dataset-name-sentinel".to_string(),
                    version: "hidden-dataset-version-sentinel".to_string(),
                    source_sha256: "f".repeat(64),
                    source_commit: None,
                    base_source_sha256: None,
                    overlay_policy: None,
                    overlays: Vec::new(),
                },
                components: vec![EvaluationIdentityComponent {
                    name: "hidden-component-name-sentinel".to_string(),
                    version: "hidden-component-version-sentinel".to_string(),
                    source_sha256: "1".repeat(64),
                    source_commit: Some("a".repeat(40)),
                    base_source_sha256: Some("b".repeat(64)),
                    overlay_policy: Some(SOURCE_OVERLAY_POLICY_V1.to_string()),
                    overlays: vec![EvaluationSourceOverlayIdentity {
                        overlay_id: "hidden-overlay-id-sentinel".to_string(),
                        artifact_content_sha256: "c".repeat(64),
                    }],
                }],
                ordered_manifest_sha256: "2".repeat(64),
                case_templates,
                unit_templates,
                policies: CanonicalJson::new(json!({"exclude_infrastructure": true})).unwrap(),
                host: EvaluationHostIdentity {
                    runner_sha256: "3".repeat(64),
                    capability_inventory_sha256: "4".repeat(64),
                    schema_inventory_sha256: "5".repeat(64),
                    isolation_proof_sha256: "6".repeat(64),
                },
                route_map_sha256: "7".repeat(64),
                prepared_endpoints_sha256: "8".repeat(64),
                sandbox_sha256: None,
            },
            outcomes: vec![
                CaseOutcome {
                    case_id: EvaluationCaseId::new("hidden-case-completed-sentinel").unwrap(),
                    outcome: CaseOutcomeKind::Completed {
                        completed: CompletedCaseOutcome {
                            scores: BTreeMap::from([(
                                "hidden-score-name-sentinel".to_string(),
                                ProviderScore {
                                    value: CanonicalJson::new(json!(0)).unwrap(),
                                    public_projection: Some(CanonicalJson::new(json!(0)).unwrap()),
                                },
                            )]),
                            numeric_metrics: BTreeMap::from([(
                                "hidden-score-name-sentinel".to_string(),
                                FiniteF64::new(0.0).unwrap(),
                            )]),
                            primary_score: Some("hidden-score-name-sentinel".to_string()),
                            annotations: None,
                        },
                    },
                    artifact_refs: vec![ArtifactRef {
                        artifact_id: bundle_id.clone(),
                        path: "hidden-path-sentinel.eval".to_string(),
                        visibility: ArtifactVisibility::Restricted,
                    }],
                },
                CaseOutcome {
                    case_id: EvaluationCaseId::new("hidden-case-infra-sentinel").unwrap(),
                    outcome: CaseOutcomeKind::InfrastructureError {
                        error: EvaluationError::new(
                            EvaluationStage::new("verifier").unwrap(),
                            "verifier_failure",
                            false,
                            "HIDDEN_EXPECTED_ANSWER_INFRA_SENTINEL",
                        )
                        .unwrap(),
                    },
                    artifact_refs: Vec::new(),
                },
                CaseOutcome {
                    case_id: EvaluationCaseId::new("hidden-case-cancelled-sentinel").unwrap(),
                    outcome: CaseOutcomeKind::Cancelled {
                        stage: EvaluationStage::new("solving").unwrap(),
                        reason: "HIDDEN_PRIVATE_TEST_CANCEL_SENTINEL".to_string(),
                    },
                    artifact_refs: Vec::new(),
                },
            ],
            aggregates: vec![AggregateMetric {
                scorer: "hidden-aggregate-scorer-sentinel".to_string(),
                reducer: "hidden-aggregate-reducer-sentinel".to_string(),
                metric: "hidden-aggregate-metric-sentinel".to_string(),
                value: FiniteF64::new(0.0).unwrap(),
                scored_count: 1,
                unscored_count: 2,
                definition: CanonicalJson::new(json!({"reducer": "mean"})).unwrap(),
            }],
            artifacts: vec![EvaluationArtifactManifestEntry {
                artifact_id: bundle_id.clone(),
                path: "hidden-path-sentinel.eval".to_string(),
                media_type: "application/x-hidden-media-sentinel".to_string(),
                visibility: ArtifactVisibility::Restricted,
                size_bytes: 7,
                artifact_content_sha256: "9".repeat(64),
            }],
            provider_bundle: ArtifactRef {
                artifact_id: bundle_id,
                path: "hidden-path-sentinel.eval".to_string(),
                visibility: ArtifactVisibility::Restricted,
            },
            normalized_result_sha256: "0".repeat(64),
        }
    }

    fn sealed() -> SealedEvaluationArtifacts {
        SealedEvaluationArtifacts {
            root: "/tmp/sealed-fixture".into(),
            entries: vec![SealedEvaluationArtifact {
                artifact_id: EvaluationArtifactId::new("hidden-artifact-id-sentinel").unwrap(),
                path: "hidden-path-sentinel.eval".to_string(),
                media_type: "application/x-hidden-media-sentinel".to_string(),
                visibility: ArtifactVisibility::Restricted,
                size_bytes: 7,
                artifact_content_sha256: "9".repeat(64),
                public_projection_schema_sha256: None,
            }],
            provider_bundle_sha256: "9".repeat(64),
            quiescence_proof_sha256: "a".repeat(64),
        }
    }

    fn routes() -> EvaluationRouteTable {
        EvaluationRouteTable::new([EvaluationRoute {
            service_id: "primary".to_string(),
            purpose: "primary".to_string(),
            model: "candidate".to_string(),
            endpoint_profile: "candidate_profile".to_string(),
            prepared_identity_sha256: "b".repeat(64),
            endpoint_capabilities: BTreeSet::from(["chat".to_string()]),
        }])
        .unwrap()
    }

    fn facts(candidate: &EvaluationFinishCandidate) -> EvaluationReportFacts {
        let names = ["completed", "infra", "cancelled"];
        let cases = candidate
            .outcomes
            .iter()
            .zip(names)
            .map(|(outcome, name)| {
                (
                    outcome.case_id.clone(),
                    EvaluationCaseReportFacts {
                        template_id: aiperf_accuracy::EvaluationCaseTemplateId::new(format!(
                            "hidden-template-{name}-sentinel"
                        ))
                        .unwrap(),
                        task: format!("hidden-task-{name}-sentinel"),
                        source: "hidden-source-sentinel".to_string(),
                    },
                )
            })
            .collect();
        let mut public_score_projection_policy = PublicScoreProjectionPolicy::restricted_only();
        public_score_projection_policy
            .register(
                "hidden-score-name-sentinel",
                Arc::new(FiniteBinaryNumberProjectionValidator),
            )
            .unwrap();
        let mut public_aggregate_projection_policy =
            PublicAggregateProjectionPolicy::restricted_only();
        public_aggregate_projection_policy
            .register(
                "fixture_accuracy_mean",
                Arc::new(
                    ExactBinaryMeanAggregateValidator::accuracy(
                        "hidden-aggregate-scorer-sentinel",
                        "hidden-aggregate-reducer-sentinel",
                        "hidden-aggregate-metric-sentinel",
                        CanonicalJson::new(json!({"reducer": "mean"})).unwrap(),
                        "hidden-score-name-sentinel",
                    )
                    .unwrap(),
                ),
            )
            .unwrap();
        EvaluationReportFacts {
            safe_config: json!({"benchmark": "fixture"}),
            cases,
            public_score_projection_policy,
            public_aggregate_projection_policy,
            public_metadata_projector: Arc::new(
                FrozenPublicEvaluationMetadataPolicy::new(
                    Vec::new(),
                    vec![PublicNumericMetricRule {
                        provider_name: "hidden-score-name-sentinel".to_string(),
                        public_name: "accuracy".to_string(),
                    }],
                    Vec::new(),
                )
                .unwrap(),
            ),
            route_summaries: BTreeMap::from([(
                "primary".to_string(),
                EvaluationRouteSummaryReport {
                    logical_operations: 1,
                    transport_attempts: 1,
                    completed: 1,
                    prompt_tokens: Some(4),
                    completion_tokens: Some(1),
                    ..Default::default()
                },
            )]),
        }
    }

    #[test]
    fn exact_factory_rules_project_case_score_and_validated_aggregate_facts() {
        let candidate = candidate();
        let mut facts = facts(&candidate);
        facts.public_metadata_projector = Arc::new(
            FrozenPublicEvaluationMetadataPolicy::new(
                vec![PublicCaseMetadataRule {
                    provider_task: "hidden-task-completed-sentinel".to_string(),
                    provider_source: "hidden-source-sentinel".to_string(),
                    public_task: "gsm8k".to_string(),
                    public_source: "openai/gsm8k@test".to_string(),
                }],
                vec![PublicNumericMetricRule {
                    provider_name: "hidden-score-name-sentinel".to_string(),
                    public_name: "accuracy".to_string(),
                }],
                vec![PublicAggregateMetadataRule {
                    provider_scorer: "hidden-aggregate-scorer-sentinel".to_string(),
                    provider_reducer: "hidden-aggregate-reducer-sentinel".to_string(),
                    provider_metric: "hidden-aggregate-metric-sentinel".to_string(),
                    public_scorer: "accuracy".to_string(),
                    public_reducer: "mean".to_string(),
                    public_metric: "accuracy".to_string(),
                    definition: CanonicalJson::new(json!({"reducer": "mean"})).unwrap(),
                }],
            )
            .unwrap(),
        );
        let report = build_evaluation_report(&candidate, &sealed(), &routes(), &facts).unwrap();
        assert_eq!(report.cases[0].task, "gsm8k");
        assert_eq!(report.cases[0].source, "openai/gsm8k@test");
        assert_eq!(report.cases[0].numeric_metrics["accuracy"], 0.0);
        assert_eq!(report.aggregates.len(), 1);
        assert_eq!(report.aggregates[0].scorer, "accuracy");
        assert_eq!(report.aggregates[0].reducer, "mean");
        assert_eq!(report.aggregates[0].metric, "accuracy");
        assert_eq!(report.aggregates[0].value, 0.0);
        assert_eq!(report.aggregates[0].scored_count, 1);
        assert_eq!(report.aggregates[0].unscored_count, 2);
        assert!(is_sha256(&report.aggregates[0].projection_schema));

        let mut drifted = candidate;
        drifted.aggregates[0].definition =
            CanonicalJson::new(json!({"reducer": "private_weighted_mean"})).unwrap();
        let error = build_evaluation_report(&drifted, &sealed(), &routes(), &facts).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("tuple or definition did not match")
        );
    }

    #[test]
    fn report_preserves_zero_infrastructure_cancel_and_closes_private_domains() {
        let candidate = candidate();
        let report =
            build_evaluation_report(&candidate, &sealed(), &routes(), &facts(&candidate)).unwrap();
        assert_eq!(report.completed_count, 1);
        assert_eq!(report.infrastructure_error_count, 1);
        assert_eq!(report.cancelled_count, 1);
        assert_eq!(
            report.cases[0].outcome,
            EvaluationCaseOutcomeKind::Completed
        );
        assert_eq!(report.cases[0].numeric_metrics["accuracy"], 0.0);
        assert!(
            !report.cases[0]
                .numeric_metrics
                .contains_key("hidden-score-name-sentinel")
        );
        assert_eq!(report.aggregates.len(), 1);
        assert_eq!(report.aggregates[0].value, 0.0);
        assert_eq!(report.aggregates[0].scored_count, 1);
        assert_eq!(report.aggregates[0].unscored_count, 2);
        assert_eq!(report.cases[0].case_id, "case-00000000");
        assert_eq!(report.cases[0].template_id, "template-00000000");
        assert_eq!(report.cases[0].task, "task-00000000");
        assert_eq!(report.cases[0].source, "source-00000000");
        assert_eq!(report.cases[0].scores["accuracy"].value, json!(0));
        assert_eq!(report.cases[1].scores.len(), 0);
        assert_eq!(
            report.cases[1].outcome,
            EvaluationCaseOutcomeKind::InfrastructureError
        );
        assert_eq!(
            report.cases[2].outcome,
            EvaluationCaseOutcomeKind::Cancelled
        );
        assert_eq!(report.artifacts[0].artifact_ref, "artifact-00000000");
        assert_eq!(report.artifacts[0].path, None);
        assert_eq!(report.artifacts[0].media_type, None);
        assert_eq!(report.artifacts[0].size_bytes, None);
        assert_eq!(report.artifacts[0].artifact_content_sha256, None);
        assert_eq!(report.artifacts[0].projection_schema, None);
        assert_eq!(report.cases[0].artifact_refs, ["artifact-00000000"]);
        let encoded = serde_json::to_string(&report).unwrap();
        assert!(!encoded.contains("HIDDEN_EXPECTED_ANSWER_INFRA_SENTINEL"));
        assert!(!encoded.contains("HIDDEN_PRIVATE_TEST_CANCEL_SENTINEL"));
        assert!(!encoded.contains("hidden-artifact-id-sentinel"));
        assert!(!encoded.contains("hidden-path-sentinel"));
        assert!(!encoded.contains("hidden-media-sentinel"));
        assert!(!encoded.contains("hidden-case"));
        assert!(!encoded.contains("hidden-template"));
        assert!(!encoded.contains("hidden-task"));
        assert!(!encoded.contains("hidden-source"));
        assert!(!encoded.contains("hidden-dataset"));
        assert!(!encoded.contains("hidden-component"));
        assert!(!encoded.contains("hidden-score"));
        assert!(!encoded.contains("hidden-aggregate"));
        assert!(!encoded.contains("\"definition\""));
        assert!(!encoded.contains("hidden-overlay-id-sentinel"));
        assert_eq!(
            report.identity.components["component-00000000.overlay-00000000.artifact_content_sha256"],
            "c".repeat(64)
        );
        assert_eq!(
            report.identity.components["component-00000000.overlay_policy"],
            SOURCE_OVERLAY_POLICY_V1
        );
    }

    #[test]
    fn restricted_artifact_digests_are_not_public_dictionary_oracles() {
        let candidate = candidate();
        let original =
            build_evaluation_report(&candidate, &sealed(), &routes(), &facts(&candidate)).unwrap();

        let mut changed_candidate = candidate.clone();
        changed_candidate.artifacts[0].size_bytes = 123_456;
        changed_candidate.artifacts[0].artifact_content_sha256 = "d".repeat(64);
        changed_candidate.normalized_result_sha256 = "e".repeat(64);
        let mut changed_sealed = sealed();
        changed_sealed.entries[0].size_bytes = 123_456;
        changed_sealed.entries[0].artifact_content_sha256 = "d".repeat(64);
        changed_sealed.provider_bundle_sha256 = "d".repeat(64);
        let changed = build_evaluation_report(
            &changed_candidate,
            &changed_sealed,
            &routes(),
            &facts(&changed_candidate),
        )
        .unwrap();

        assert_eq!(
            serde_json::to_value(original).unwrap(),
            serde_json::to_value(changed).unwrap()
        );
    }

    #[test]
    fn public_projection_artifact_exposes_only_factory_reviewed_fields() {
        let mut candidate = candidate();
        let public_id = EvaluationArtifactId::new("public-projection-result").unwrap();
        candidate.artifacts.push(EvaluationArtifactManifestEntry {
            artifact_id: public_id.clone(),
            path: "public/result.json".to_string(),
            media_type: "application/json".to_string(),
            visibility: ArtifactVisibility::PublicProjection,
            size_bytes: 11,
            artifact_content_sha256: "d".repeat(64),
        });
        candidate.outcomes[0].artifact_refs.push(ArtifactRef {
            artifact_id: public_id.clone(),
            path: "public/result.json".to_string(),
            visibility: ArtifactVisibility::PublicProjection,
        });
        let mut sealed = sealed();
        sealed.entries.push(SealedEvaluationArtifact {
            artifact_id: public_id,
            path: "public/result.json".to_string(),
            media_type: "application/json".to_string(),
            visibility: ArtifactVisibility::PublicProjection,
            size_bytes: 11,
            artifact_content_sha256: "d".repeat(64),
            public_projection_schema_sha256: Some("e".repeat(64)),
        });

        let report =
            build_evaluation_report(&candidate, &sealed, &routes(), &facts(&candidate)).unwrap();
        let artifact = &report.artifacts[1];
        assert_eq!(artifact.artifact_ref, "artifact-00000001");
        assert_eq!(artifact.path.as_deref(), Some("public/result.json"));
        assert_eq!(artifact.media_type.as_deref(), Some("application/json"));
        assert_eq!(artifact.visibility, "public");
        assert_eq!(artifact.size_bytes, Some(11));
        assert_eq!(artifact.artifact_content_sha256, Some("d".repeat(64)));
        assert_eq!(artifact.projection_schema, Some("e".repeat(64)));
    }

    #[test]
    fn metadata_allowlist_cannot_publish_an_unvalidated_aggregate_value() {
        let candidate = candidate();
        let mut facts = facts(&candidate);
        facts.public_aggregate_projection_policy =
            PublicAggregateProjectionPolicy::restricted_only();
        facts.public_metadata_projector = Arc::new(
            FrozenPublicEvaluationMetadataPolicy::new(
                Vec::new(),
                vec![PublicNumericMetricRule {
                    provider_name: "hidden-score-name-sentinel".to_string(),
                    public_name: "accuracy".to_string(),
                }],
                vec![PublicAggregateMetadataRule {
                    provider_scorer: "hidden-aggregate-scorer-sentinel".to_string(),
                    provider_reducer: "hidden-aggregate-reducer-sentinel".to_string(),
                    provider_metric: "hidden-aggregate-metric-sentinel".to_string(),
                    public_scorer: "accuracy".to_string(),
                    public_reducer: "mean".to_string(),
                    public_metric: "accuracy".to_string(),
                    definition: CanonicalJson::new(json!({"reducer": "mean"})).unwrap(),
                }],
            )
            .unwrap(),
        );

        let report = build_evaluation_report(&candidate, &sealed(), &routes(), &facts).unwrap();
        assert!(report.aggregates.is_empty());
    }

    #[test]
    fn report_rejects_unregistered_public_score_and_artifact_drift() {
        let candidate = candidate();
        let mut missing_schema_facts = facts(&candidate);
        missing_schema_facts.public_score_projection_policy =
            PublicScoreProjectionPolicy::restricted_only();
        assert!(
            build_evaluation_report(&candidate, &sealed(), &routes(), &missing_schema_facts)
                .is_err()
        );

        let mut sealed = sealed();
        sealed.entries[0].artifact_content_sha256 = "d".repeat(64);
        assert!(
            build_evaluation_report(&candidate, &sealed, &routes(), &facts(&candidate)).is_err()
        );
    }
}
