// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public discovery and source-detection contracts for imported agent sessions.

use std::fs;
use std::path::{Path, PathBuf};

use aiperf_runtime::config::model::dataset::RecordedAgentSourceFormat::{Auto, ClaudeCode, Codex};
use aiperf_runtime::graph::recorded::agent_recording::{
    ImportedAgentSource, ImportedSessionFamily, detect_imported_agent_source,
    discover_imported_agent_read_set,
};
use tempfile::tempdir;

fn fixture(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/recorded_agent_session_import")
        .join(path)
}

fn write(path: &Path, contents: &str) {
    fs::create_dir_all(path.parent().expect("test path has parent")).expect("create test parent");
    fs::write(path, contents).expect("write test source");
}

#[test]
fn codex_directory_is_recursive_and_sorted_with_root_relative_names() {
    let root = tempdir().expect("temporary root");
    write(
        &root.path().join("2026/04/b.jsonl"),
        "{\"type\":\"session_meta\",\"payload\":{}}\n",
    );
    write(
        &root.path().join("2026/03/a.jsonl"),
        "{\"type\":\"session_meta\",\"payload\":{}}\n",
    );
    write(&root.path().join("ignored.txt"), "not a session");

    let read_set = discover_imported_agent_read_set(root.path(), None, Codex, None)
        .expect("Codex tree discovers");
    assert_eq!(read_set.source, ImportedAgentSource::Codex);
    assert_eq!(
        read_set
            .files
            .iter()
            .map(|file| file.relative_path.as_path())
            .collect::<Vec<_>>(),
        vec![Path::new("2026/03/a.jsonl"), Path::new("2026/04/b.jsonl")],
    );
    assert!(
        read_set
            .files
            .iter()
            .all(|file| file.family == ImportedSessionFamily::Session)
    );
}

#[test]
fn claude_detection_uses_a_non_discriminating_lead_in() {
    let file = fixture("claude_code/linear.jsonl");
    assert_eq!(
        detect_imported_agent_source(&file).expect("Claude source detected"),
        ImportedAgentSource::ClaudeCode
    );
}

#[test]
fn auto_rejects_directories_without_reading_them() {
    let directory = fixture("codex");
    let error = discover_imported_agent_read_set(&directory, None, Auto, None)
        .expect_err("directory auto detection must fail");
    assert!(error.to_string().contains("explicit source_format"));
}

#[test]
fn claude_read_set_is_shallow_and_subagents_are_opt_in_by_effective_default() {
    let directory = fixture("claude_code/with_subagent");
    let with_subagents = discover_imported_agent_read_set(&directory, None, ClaudeCode, None)
        .expect("Claude tree discovers");
    assert_eq!(
        with_subagents
            .files
            .iter()
            .map(|file| (file.relative_path.clone(), file.family))
            .collect::<Vec<_>>(),
        vec![
            (PathBuf::from("main.jsonl"), ImportedSessionFamily::Session),
            (
                PathBuf::from("main/subagents/agent-aaa.jsonl"),
                ImportedSessionFamily::Subagent,
            ),
        ],
    );
    let without_subagents =
        discover_imported_agent_read_set(&directory, None, ClaudeCode, Some(false))
            .expect("Claude mains discover");
    assert_eq!(without_subagents.files.len(), 1);
    assert_eq!(
        without_subagents.files[0].relative_path,
        Path::new("main.jsonl")
    );
}

#[test]
fn auto_detection_is_bounded_and_rejects_ambiguous_or_malformed_records() {
    let root = tempdir().expect("temporary root");
    let beyond_bound = root.path().join("beyond-bound.jsonl");
    let mut records = (0..20)
        .map(|_| "{\"type\":\"unrelated\"}\n")
        .collect::<String>();
    records.push_str("{\"type\":\"session_meta\",\"payload\":{}}\n");
    write(&beyond_bound, &records);
    assert!(detect_imported_agent_source(&beyond_bound).is_err());

    let ambiguous = root.path().join("ambiguous.jsonl");
    write(
        &ambiguous,
        "{\"type\":\"session_meta\",\"payload\":{},\"sessionId\":\"x\",\"parentUuid\":null}\n",
    );
    assert!(
        detect_imported_agent_source(&ambiguous)
            .expect_err("mixed marker record fails")
            .to_string()
            .contains("ambiguous")
    );

    let malformed = root.path().join("malformed.jsonl");
    write(&malformed, "not json\n");
    assert!(
        detect_imported_agent_source(&malformed)
            .expect_err("invalid JSON fails")
            .to_string()
            .contains("invalid JSON")
    );
}

#[test]
fn source_detection_uses_only_supported_provider_markers() {
    let root = tempdir().expect("temporary root");
    for (name, record) in [
        ("meta", "{\"type\":\"session_meta\",\"payload\":{}}\n"),
        ("event", "{\"type\":\"event_msg\",\"payload\":{}}\n"),
        ("response", "{\"type\":\"response_item\",\"payload\":{}}\n"),
        ("turn", "{\"type\":\"turn_context\",\"payload\":{}}\n"),
    ] {
        let path = root.path().join(format!("codex-{name}.jsonl"));
        write(&path, record);
        assert_eq!(
            detect_imported_agent_source(&path).expect("supported Codex marker"),
            ImportedAgentSource::Codex
        );
    }
    for (name, record) in [
        ("parent", "{\"sessionId\":\"x\",\"parentUuid\":null}\n"),
        (
            "permission",
            "{\"type\":\"permission-mode\",\"sessionId\":\"x\"}\n",
        ),
        (
            "history",
            "{\"type\":\"file-history-snapshot\",\"sessionId\":\"x\"}\n",
        ),
        ("summary", "{\"type\":\"summary\",\"sessionId\":\"x\"}\n"),
    ] {
        let path = root.path().join(format!("claude-{name}.jsonl"));
        write(&path, record);
        assert_eq!(
            detect_imported_agent_source(&path).expect("supported Claude marker"),
            ImportedAgentSource::ClaudeCode
        );
    }
    for (name, record) in [
        ("codex-without-payload", "{\"type\":\"response_item\"}\n"),
        (
            "codex-scalar-payload",
            "{\"type\":\"session_meta\",\"payload\":false}\n",
        ),
        ("claude-session-only", "{\"sessionId\":\"not-a-marker\"}\n"),
        (
            "claude-unknown-type",
            "{\"type\":\"unknown\",\"sessionId\":\"x\"}\n",
        ),
        (
            "claude-permission-without-session",
            "{\"type\":\"permission-mode\"}\n",
        ),
        (
            "claude-history-without-session",
            "{\"type\":\"file-history-snapshot\"}\n",
        ),
        ("claude-summary-without-session", "{\"type\":\"summary\"}\n"),
    ] {
        let path = root.path().join(format!("near-{name}.jsonl"));
        write(&path, record);
        assert!(detect_imported_agent_source(&path).is_err());
    }
}

#[test]
fn bounded_detection_does_not_read_the_twenty_first_nonempty_record() {
    let root = tempdir().expect("temporary root");
    let path = root.path().join("bounded.jsonl");
    let mut records = (0..20)
        .map(|_| "{\"type\":\"unrelated\"}\n")
        .collect::<String>();
    records.push_str("[]\n");
    write(&path, &records);
    let error = detect_imported_agent_source(&path)
        .expect_err("the first twenty records have no marker")
        .to_string();
    assert!(error.contains("no recognized source marker"));
    assert!(!error.contains("record must be a JSON object"));
}

#[test]
fn discovery_rejects_non_object_records_without_leaking_private_source_values() {
    let root = tempdir().expect("temporary root");
    let non_object = root.path().join("non-object.jsonl");
    write(&non_object, "[]\n");
    assert!(
        detect_imported_agent_source(&non_object)
            .expect_err("array source record fails")
            .to_string()
            .contains("record must be a JSON object")
    );

    let private_value = "PRIVATE_PROMPT_REASONING_CWD_TOOL_RESULT";
    let private_source = root.path().join("private.jsonl");
    write(&private_source, &format!("not-json-{private_value}\n"));
    let error = detect_imported_agent_source(&private_source)
        .expect_err("malformed private source fails")
        .to_string();
    assert!(!error.contains(private_value));
}

#[test]
fn explicit_source_validates_all_files_and_refuses_selected_symlinks() {
    let root = tempdir().expect("temporary root");
    write(
        &root.path().join("codex.jsonl"),
        "{\"type\":\"session_meta\",\"payload\":{}}\n",
    );
    write(
        &root.path().join("claude.jsonl"),
        "{\"type\":\"user\",\"sessionId\":\"x\",\"parentUuid\":null}\n",
    );
    assert!(discover_imported_agent_read_set(root.path(), None, Codex, None).is_err());

    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;

        let selected = root.path().join("selected.jsonl");
        symlink(root.path().join("codex.jsonl"), &selected).expect("test symlink");
        assert!(discover_imported_agent_read_set(&selected, None, Codex, None).is_err());
    }
}

#[test]
fn explicit_directories_can_be_empty_and_claude_excludes_nested_impostors() {
    let root = tempdir().expect("temporary root");
    let empty_codex = root.path().join("empty-codex");
    let empty_claude = root.path().join("empty-claude");
    fs::create_dir_all(&empty_codex).expect("create Codex directory");
    fs::create_dir_all(&empty_claude).expect("create Claude directory");
    assert!(
        discover_imported_agent_read_set(&empty_codex, None, Codex, None)
            .expect("empty Codex tree is a valid exact set")
            .files
            .is_empty()
    );
    assert!(
        discover_imported_agent_read_set(&empty_claude, None, ClaudeCode, None)
            .expect("empty Claude tree is a valid exact set")
            .files
            .is_empty()
    );

    let claude = root.path().join("claude");
    write(
        &claude.join("main.jsonl"),
        "{\"type\":\"user\",\"sessionId\":\"main\",\"parentUuid\":null}\n",
    );
    write(
        &claude.join("nested/impostor.jsonl"),
        "{\"type\":\"user\",\"sessionId\":\"impostor\",\"parentUuid\":null}\n",
    );
    write(
        &claude.join("main/subagents/not-agent.jsonl"),
        "{\"type\":\"user\",\"sessionId\":\"impostor\",\"parentUuid\":null}\n",
    );
    write(
        &claude.join("main/subagents/agent-.jsonl"),
        "{\"type\":\"user\",\"sessionId\":\"impostor\",\"parentUuid\":null}\n",
    );
    write(
        &claude.join("main/subagents/nested/agent-hidden.jsonl"),
        "{\"type\":\"user\",\"sessionId\":\"impostor\",\"parentUuid\":null}\n",
    );
    let read_set = discover_imported_agent_read_set(&claude, None, ClaudeCode, None)
        .expect("only documented Claude layout is selected");
    assert_eq!(read_set.files.len(), 1);
    assert_eq!(read_set.files[0].relative_path, Path::new("main.jsonl"));
}

#[cfg(unix)]
#[test]
fn discovery_rejects_intermediate_symlinks_and_duplicate_canonical_aliases() {
    use std::os::unix::fs::symlink;

    let root = tempdir().expect("temporary root");
    let target = root.path().join("target");
    write(
        &target.join("history.jsonl"),
        "{\"type\":\"session_meta\",\"payload\":{}}\n",
    );
    symlink(&target, root.path().join("linked")).expect("test intermediate symlink");
    assert!(discover_imported_agent_read_set(root.path(), None, Codex, None).is_err());

    let claude = root.path().join("claude");
    write(
        &claude.join("main.jsonl"),
        "{\"sessionId\":\"main\",\"parentUuid\":null}\n",
    );
    let alias = claude.join("main/subagents/agent-alias.jsonl");
    fs::create_dir_all(alias.parent().expect("subagent alias has a parent"))
        .expect("create duplicate alias parent");
    symlink(claude.join("main.jsonl"), &alias).expect("test duplicate canonical alias");
    let error = discover_imported_agent_read_set(&claude, None, ClaudeCode, None)
        .expect_err("symlink duplicate aliases are forbidden before selection");
    assert!(error.to_string().contains("symlink"));
}

#[cfg(unix)]
#[test]
fn discovery_rejects_unreadable_selected_source_when_permissions_enforce_it() {
    use std::os::unix::fs::PermissionsExt;

    let root = tempdir().expect("temporary root");
    let path = root.path().join("unreadable.jsonl");
    write(&path, "{\"type\":\"session_meta\",\"payload\":{}}\n");
    fs::set_permissions(&path, fs::Permissions::from_mode(0o000))
        .expect("remove test source permissions");
    if fs::File::open(&path).is_err() {
        assert!(discover_imported_agent_read_set(&path, None, Codex, None).is_err());
    }
    // Privileged CI users can read mode-000 files; permission enforcement is
    // exercised above whenever the platform exposes an unreadable file.
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600))
        .expect("restore test source permissions");
}

#[test]
fn replay_root_containment_rejects_selected_paths_outside_the_authored_root() {
    let root = tempdir().expect("temporary root");
    let replay_root = root.path().join("replay-root");
    fs::create_dir_all(&replay_root).expect("create replay root");
    let outside = root.path().join("outside.jsonl");
    write(&outside, "{\"type\":\"session_meta\",\"payload\":{}}\n");
    assert!(discover_imported_agent_read_set(&outside, Some(&replay_root), Codex, None).is_err());
}
