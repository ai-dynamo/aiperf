// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Contract coverage for recorded-agent environment recipes and workspace staging.

use std::fs;

use aiperf_runtime::dataset::{Payload, SegmentPool, SegmentStore};
use aiperf_runtime::graph::driver::ReplayTaskIdentity;
use aiperf_runtime::graph::recorded::agent_recording::RecordedAgentMetadata;
use aiperf_runtime::graph::tools::{
    CommandDisposition, EnvironmentRecipe, GuardedToolCommandPolicy, PinchWorkspaceStager,
    ToolCommandPolicy, ToolSandboxCapabilities, WorkspaceEntrySource, resolve_recorded_environment,
};

#[test]
fn swe_recipe_prefers_nested_image_then_uses_testbed_without_mount() {
    // This catches a resolver that chooses the low-level image fallback before a
    // recording's nested task image, or that accidentally hides image-native files.
    let task = ReplayTaskIdentity {
        adapter: "swebench".into(),
        family: "swe-sample".into(),
        task_id: "django__django-11099".into(),
        primary_role: None,
    };
    let metadata = RecordedAgentMetadata {
        docker_image: None,
        instance: Some(serde_json::json!({
            "image_name": "swebench/sweb.eval.x86_64.django_11099:latest",
            "docker_image": "ignored/nested-fallback:latest",
            "instance_id": "django__django-11099",
        })),
        ..RecordedAgentMetadata::default()
    };

    let recipe =
        resolve_recorded_environment(&task, &metadata, "pinch:fixed", Some("low:level"), false)
            .expect("nested SWE image resolves");

    assert_eq!(
        recipe.image,
        "swebench/sweb.eval.x86_64.django_11099:latest"
    );
    assert_eq!(recipe.workspace.workdir, "/testbed");
    assert_eq!(recipe.workspace.interpreter, ["bash", "-c"]);
    assert_eq!(recipe.workspace.command_timeout_ns, 60_000_000_000);
    assert!(!recipe.workspace.mount_workspace);
    assert_eq!(recipe.kind, EnvironmentRecipe::SweBench);
}

#[test]
fn guarded_policy_blocks_an_installer_in_a_quoted_separator_aware_segment() {
    // This catches a policy that splits quoted separators or checks only the
    // command's first segment, allowing a later installer to mutate the image.
    let policy = GuardedToolCommandPolicy;
    let disposition = policy
        .evaluate("echo 'keep && quoted'; BUILD=1 sudo env python3 -m pip install pkg")
        .expect("policy parses the recorded shell command");
    let CommandDisposition::Synthetic(result) = disposition else {
        panic!("installer must become an agent-visible synthetic result");
    };
    assert_eq!(result.exit_code, 127);
    assert!(result.output.starts_with(b"recorded-agent replay blocked"));
}

#[test]
fn guarded_policy_blocks_an_installer_after_an_unquoted_newline() {
    // This catches a policy that treats only punctuation as a shell separator,
    // allowing a later physical line to install packages in the task image.
    let disposition = GuardedToolCommandPolicy
        .evaluate("echo safe\napt-get install package")
        .expect("policy parses the recorded multiline shell command");
    let CommandDisposition::Synthetic(result) = disposition else {
        panic!("installer after a newline must become a synthetic result");
    };
    assert_eq!(result.exit_code, 127);
}

#[test]
fn guarded_policy_blocks_an_installer_nested_in_shell_control() {
    // This catches command substitution or shell control syntax that hides an
    // installer behind a non-installer first token in its top-level segment.
    let disposition = GuardedToolCommandPolicy
        .evaluate("echo $(apt-get install package)")
        .expect("policy inspects a controlled shell construct conservatively");
    let CommandDisposition::Synthetic(result) = disposition else {
        panic!("nested installer must become a synthetic result");
    };
    assert_eq!(result.exit_code, 127);
}

#[test]
fn guarded_policy_blocks_detachment_nested_in_shell_control() {
    // This catches checking only the top-level executable while a nested shell
    // payload or command substitution starts a process outside containment.
    for command in [
        "bash -c 'setsid true &'",
        "bash --norc -c 'setsid true'",
        "bash -c -- 'setsid true'",
        "bash -c -x 'setsid true'",
        "bash -c -o nounset 'setsid true'",
        "bash -c -O extglob 'setsid true'",
        "echo $(setsid true)",
        "true & setsid true",
        "printf ok | (setsid true)",
        "printf ok | { setsid true; }",
        "f(){ setsid true; }; f",
    ] {
        let disposition = GuardedToolCommandPolicy
            .evaluate(command)
            .expect("policy inspects nested detaching commands");
        let CommandDisposition::Synthetic(result) = disposition else {
            panic!("nested detaching command must become a synthetic result: {command}");
        };
        assert_eq!(result.exit_code, 127, "{command}");
        assert!(
            result
                .output
                .windows(b"detaching".len())
                .any(|window| window == b"detaching"),
            "{command}"
        );
    }
}

#[test]
fn guarded_policy_blocks_an_installer_spliced_by_command_substitution() {
    // This catches a policy that recognizes only the source spelling of a
    // command word, although Bash removes the substitution before dispatch.
    let disposition = GuardedToolCommandPolicy
        .evaluate("a$(true)pt-get install package")
        .expect("policy inspects an expanded command word conservatively");
    let CommandDisposition::Synthetic(result) = disposition else {
        panic!("substitution-spliced installer must become a synthetic result");
    };
    assert_eq!(result.exit_code, 127);
}

#[test]
fn guarded_policy_blocks_an_installer_with_whitespace_in_command_substitution() {
    // This catches splitting a command substitution into arguments before its
    // expanded bytes are joined into the executable command word by Bash.
    let disposition = GuardedToolCommandPolicy
        .evaluate("$(printf a)pt-get install package")
        .expect("policy keeps command substitutions in their command word");
    let CommandDisposition::Synthetic(result) = disposition else {
        panic!("substitution-built installer must become a synthetic result: {disposition:?}");
    };
    assert_eq!(result.exit_code, 127);
}

#[test]
fn guarded_policy_allows_an_installer_like_argument_to_a_safe_command() {
    // This catches scanning every argument instead of just the executable
    // command position; the substitution runs `true`, not an installer.
    let disposition = GuardedToolCommandPolicy
        .evaluate("echo a$(true)pt-get")
        .expect("policy parses a safe command with an expanded argument");
    assert_eq!(disposition, CommandDisposition::Execute);
}

#[test]
fn guarded_policy_blocks_an_installer_after_the_command_prefix() {
    // This catches treating Bash's `command` builtin as an executable instead
    // of continuing to the command word it explicitly dispatches.
    let disposition = GuardedToolCommandPolicy
        .evaluate("command apt-get install package")
        .expect("policy parses the command builtin prefix");
    let CommandDisposition::Synthetic(result) = disposition else {
        panic!("command-prefixed installer must become a synthetic result");
    };
    assert_eq!(result.exit_code, 127);
}

#[test]
fn guarded_policy_blocks_an_expansion_only_command_word() {
    // This catches allowing an opaque executable whose expansion can resolve
    // to an installer; static policy cannot establish that it is `echo`.
    let disposition = GuardedToolCommandPolicy
        .evaluate("$(printf echo) safe")
        .expect("policy handles an expansion-only executable word");
    let CommandDisposition::Synthetic(result) = disposition else {
        panic!("expansion-only executable must become a synthetic result");
    };
    assert_eq!(result.exit_code, 127);
}

#[test]
fn guarded_policy_blocks_an_expansion_only_installer_command_word() {
    // This catches an empty static skeleton bypassing installer matching when
    // the command substitution itself produces the full executable name.
    let disposition = GuardedToolCommandPolicy
        .evaluate("$(printf apt-get) install package")
        .expect("policy fails closed for an opaque executable expansion");
    let CommandDisposition::Synthetic(result) = disposition else {
        panic!("expansion-only installer must become a synthetic result");
    };
    assert_eq!(result.exit_code, 127);
}

#[test]
fn guarded_policy_allows_command_query_options() {
    // This catches treating the `command` builtin's query-only options as an
    // installer invocation when they merely inspect command resolution.
    let disposition = GuardedToolCommandPolicy
        .evaluate("command -v apt-get")
        .expect("policy parses a command query");
    assert_eq!(disposition, CommandDisposition::Execute);
}

#[test]
fn guarded_policy_allows_command_query_option_clusters() {
    // This catches examining only the first option after `command`; `-v` and
    // `-V` remain query-only when preceded or combined with the `-p` option.
    for command in [
        "command -p -v apt-get",
        "command -vp apt-get",
        "command -pV apt-get",
    ] {
        let disposition = GuardedToolCommandPolicy
            .evaluate(command)
            .expect("policy parses command option clusters");
        assert_eq!(disposition, CommandDisposition::Execute, "{command}");
    }
}

#[test]
fn pinch_recipe_refuses_a_sandbox_without_workspace_materialization() {
    // This catches preflight that provisions an empty Pinch workspace on a
    // backend which cannot stage the task's digest-addressed fixture files.
    let task = ReplayTaskIdentity {
        adapter: "pinchbench".into(),
        family: "pinchbench-openclaw".into(),
        task_id: "task-1".into(),
        primary_role: None,
    };
    let recipe = resolve_recorded_environment(
        &task,
        &RecordedAgentMetadata::default(),
        "pinch:fixed",
        None,
        true,
    )
    .expect("Pinch recipe resolves before preflight");
    let error = ToolSandboxCapabilities {
        has_persistent_workspace: true,
        has_workspace_materialization: false,
        has_network_disabled: true,
        has_timeout_recycle: true,
    }
    .validate(&recipe)
    .expect_err("Pinch staging requires file materialization");
    assert!(error.to_string().contains("materialize"));
}

#[test]
fn recipe_refuses_an_unknown_adapter_even_when_the_family_looks_known() {
    // This catches recipe selection by the descriptive family, which would
    // accidentally give an unregistered source the SWE-Bench environment.
    let task = ReplayTaskIdentity {
        adapter: "unknown".into(),
        family: "swe-corpus".into(),
        task_id: "task-1".into(),
        primary_role: None,
    };

    let error = resolve_recorded_environment(
        &task,
        &RecordedAgentMetadata::default(),
        "pinch:fixed",
        Some("low:level"),
        false,
    )
    .expect_err("only registered source adapters select environment recipes");
    assert!(error.to_string().contains("adapter"));
}

#[test]
fn pinch_staging_keeps_root_contained_sorted_assets_as_raw_segments() {
    // This catches staging that serializes host paths, accepts a directory escape,
    // or loses an executable fixture's byte/mode contract.
    let root = tempfile::tempdir().expect("temporary task pack root");
    fs::create_dir_all(root.path().join("assets/bin")).expect("asset tree exists");
    fs::write(
        root.path().join("assets/bin/run.sh"),
        b"#!/bin/bash\necho ok\n",
    )
    .expect("executable fixture is written");
    fs::write(root.path().join("assets/README"), b"asset text").expect("plain fixture is written");
    let mut permissions = fs::metadata(root.path().join("assets/bin/run.sh"))
        .expect("fixture metadata")
        .permissions();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        permissions.set_mode(0o755);
        fs::set_permissions(root.path().join("assets/bin/run.sh"), permissions)
            .expect("fixture mode is set");
    }

    let mut segments = SegmentPool::new();
    let workspace = PinchWorkspaceStager::new(root.path(), &mut segments)
        .stage([
            WorkspaceEntrySource::literal("z.txt", "literal"),
            WorkspaceEntrySource::asset("assets", "copied"),
        ])
        .expect("root-contained task files stage");

    let destinations: Vec<_> = workspace
        .files
        .iter()
        .map(|file| file.destination.as_str())
        .collect();
    assert_eq!(
        destinations,
        ["copied/README", "copied/bin/run.sh", "z.txt"]
    );
    assert!(workspace.files[1].is_executable);
    let Payload::Raw { wire } = segments
        .get(workspace.files[1].content)
        .expect("staged file is an interned raw segment")
    else {
        panic!("workspace fixture must use a raw segment");
    };
    assert_eq!(wire.as_ref(), b"#!/bin/bash\necho ok\n");
}
