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
        "{\"type\":\"session_meta\"}\n",
    );
    write(
        &root.path().join("2026/03/a.jsonl"),
        "{\"type\":\"session_meta\"}\n",
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
        .map(|_| "{\"type\":\"event_msg\"}\n")
        .collect::<String>();
    records.push_str("{\"type\":\"session_meta\"}\n");
    write(&beyond_bound, &records);
    assert!(detect_imported_agent_source(&beyond_bound).is_err());

    let ambiguous = root.path().join("ambiguous.jsonl");
    write(
        &ambiguous,
        "{\"type\":\"session_meta\",\"sessionId\":\"x\"}\n",
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
fn explicit_source_validates_all_files_and_refuses_selected_symlinks() {
    let root = tempdir().expect("temporary root");
    write(
        &root.path().join("codex.jsonl"),
        "{\"type\":\"session_meta\"}\n",
    );
    write(
        &root.path().join("claude.jsonl"),
        "{\"type\":\"user\",\"sessionId\":\"x\"}\n",
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
        "{\"type\":\"user\",\"sessionId\":\"main\"}\n",
    );
    write(
        &claude.join("nested/impostor.jsonl"),
        "{\"type\":\"user\",\"sessionId\":\"impostor\"}\n",
    );
    write(
        &claude.join("main/subagents/not-agent.jsonl"),
        "{\"type\":\"user\",\"sessionId\":\"impostor\"}\n",
    );
    let read_set = discover_imported_agent_read_set(&claude, None, ClaudeCode, None)
        .expect("only documented Claude layout is selected");
    assert_eq!(read_set.files.len(), 1);
    assert_eq!(read_set.files[0].relative_path, Path::new("main.jsonl"));
}

#[test]
fn replay_root_containment_rejects_selected_paths_outside_the_authored_root() {
    let root = tempdir().expect("temporary root");
    let replay_root = root.path().join("replay-root");
    fs::create_dir_all(&replay_root).expect("create replay root");
    let outside = root.path().join("outside.jsonl");
    write(&outside, "{\"type\":\"session_meta\"}\n");
    assert!(discover_imported_agent_read_set(&outside, Some(&replay_root), Codex, None).is_err());
}
