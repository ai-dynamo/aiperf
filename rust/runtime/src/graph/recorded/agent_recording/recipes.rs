// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure recorded-agent environment recipe selection.

use crate::graph::driver::ReplayTaskIdentity;
use crate::graph::tools::{
    EnvironmentRecipe, ResolvedTraceEnvironment, TraceEnvironmentError, WorkspaceSpec,
};

use super::schema::RecordedAgentMetadata;

/// Resolve the stock PinchBench or SWE-Bench environment before placement.
pub fn resolve_recorded_environment(
    task: &ReplayTaskIdentity,
    metadata: &RecordedAgentMetadata,
    pinch_image: &str,
    tool_image: Option<&str>,
    is_scenario: bool,
) -> Result<ResolvedTraceEnvironment, TraceEnvironmentError> {
    match task.adapter.as_str() {
        "pinchbench" => {
            if pinch_image.trim().is_empty() {
                return Err(TraceEnvironmentError::new(
                    "PinchBench environment requires a configured image",
                ));
            }
            Ok(ResolvedTraceEnvironment {
                kind: EnvironmentRecipe::PinchBench,
                image: pinch_image.into(),
                workspace: WorkspaceSpec {
                    files: Vec::new(),
                    workdir: "/workspace".into(),
                    interpreter: vec!["bash".into(), "-lc".into()],
                    mount_workspace: true,
                    command_timeout_ns: 30_000_000_000,
                },
            })
        }
        "swebench" => {
            let image = swe_image(metadata).or_else(|| {
                (!is_scenario)
                    .then(|| {
                        tool_image
                            .filter(|image| !image.trim().is_empty())
                            .map(str::to_owned)
                    })
                    .flatten()
            });
            let image = image.ok_or_else(|| {
                TraceEnvironmentError::new(format!(
                    "SWE-Bench task {:?} has no resolved image",
                    task.task_id
                ))
            })?;
            Ok(ResolvedTraceEnvironment {
                kind: EnvironmentRecipe::SweBench,
                image,
                workspace: WorkspaceSpec::image_native(
                    "/testbed",
                    vec!["bash".into(), "-c".into()],
                    60_000_000_000,
                ),
            })
        }
        adapter => Err(TraceEnvironmentError::new(format!(
            "recorded environment has no recipe for adapter {adapter:?}"
        ))),
    }
}

fn swe_image(metadata: &RecordedAgentMetadata) -> Option<String> {
    non_empty(metadata.docker_image.as_deref())
        .or_else(|| nested_string(metadata, "image_name"))
        .or_else(|| nested_string(metadata, "docker_image"))
        .or_else(|| {
            nested_string(metadata, "instance_id")
                .map(|instance_id| format!("swebench/sweb.eval.x86_64.{instance_id}:latest"))
        })
}

fn nested_string(metadata: &RecordedAgentMetadata, key: &str) -> Option<String> {
    metadata
        .instance
        .as_ref()?
        .as_object()?
        .get(key)?
        .as_str()
        .and_then(|value| non_empty(Some(value)))
}

fn non_empty(value: Option<&str>) -> Option<String> {
    value
        .filter(|value| !value.trim().is_empty())
        .map(str::to_owned)
}
