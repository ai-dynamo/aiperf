// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Contract tests for the exhaustive plugin task-gate dispatcher.

use std::{collections::BTreeMap, fs, path::PathBuf};

fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn markdown_code(cell: &str) -> &str {
    cell.strip_prefix('`')
        .and_then(|cell| cell.strip_suffix('`'))
        .unwrap_or_else(|| panic!("matrix cell must be one Markdown code span: {cell}"))
}

fn plan_section<'a>(plan: &'a str, start: &str, end: &str) -> &'a str {
    let section = plan
        .split_once(start)
        .unwrap_or_else(|| panic!("missing plan section {start}"))
        .1;
    section
        .split_once(end)
        .unwrap_or_else(|| panic!("missing plan section terminator {end}"))
        .0
}

fn planned_task_commands(plan: &str) -> BTreeMap<String, String> {
    plan_section(
        plan,
        "## Task Gate Matrix",
        "### Implementation-unit Gate Matrix",
    )
    .lines()
    .filter_map(|line| {
        let cells: Vec<_> = line.split('|').map(str::trim).collect();
        if cells.len() != 5 || cells[1].parse::<u8>().ok().is_none() {
            return None;
        }
        let task = cells[1].parse::<u8>().expect("numeric task was validated");
        Some((
            task.to_string(),
            format!("{} && {}", markdown_code(cells[2]), markdown_code(cells[3])),
        ))
    })
    .collect()
}

fn planned_unit_commands(plan: &str) -> BTreeMap<String, String> {
    let expected_units = [
        "12-core",
        "12-elf",
        "12-macho",
        "12-pe",
        "33-websocket",
        "33-dry-run",
        "34-dynosim-offline",
        "34-dynosim-online",
        "37a-tooling",
        "37b-package",
        "38a-harness",
        "38b-benchmark",
        "39a-basic",
        "39a-parquet",
        "39a-mlflow",
        "39a-wandb",
        "39a-otel",
        "39a-endpoints-grpc-bindings",
        "39a-http",
        "39a-grpc",
        "39a-websocket",
        "39a-dry-run",
        "39a-dynosim-offline",
        "39a-dynosim-online",
        "39b",
    ];
    let mut commands = BTreeMap::new();
    for line in plan_section(
        plan,
        "### Implementation-unit Gate Matrix",
        "## Dependency and parallelism waves",
    )
    .lines()
    {
        let cells: Vec<_> = line.split('|').map(str::trim).collect();
        if cells.len() != 4 || !cells[1].starts_with('`') {
            continue;
        }
        let unit = markdown_code(cells[1]);
        if !expected_units.contains(&unit) {
            continue;
        }
        let prior = commands.insert(unit.to_owned(), markdown_code(cells[2]).to_owned());
        assert!(
            prior.is_none(),
            "duplicate implementation-unit matrix row {unit}"
        );
    }
    assert_eq!(
        commands.keys().map(String::as_str).collect::<Vec<_>>(),
        expected_units
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>(),
        "implementation-unit matrix is incomplete"
    );
    commands
}

fn script_commands(script: &str) -> BTreeMap<String, String> {
    let mut commands = BTreeMap::new();
    let case_body = script
        .split_once("case \"$task\" in\n")
        .expect("dispatcher must case-match the task")
        .1
        .split_once("\nesac")
        .expect("dispatcher case must terminate")
        .0;
    for line in case_body.lines() {
        let Some((identifier, command)) = line.trim().split_once(") ") else {
            continue;
        };
        if identifier == "*" {
            continue;
        }
        let command = command
            .strip_suffix(" ;;")
            .unwrap_or_else(|| panic!("case {identifier} must end in ` ;;`"));
        let prior = commands.insert(identifier.to_owned(), command.to_owned());
        assert!(prior.is_none(), "duplicate script case {identifier}");
    }
    commands
}

#[test]
fn task_gate_dispatcher_exactly_matches_both_planned_matrices() {
    let root = repository_root();
    let script_path = root.join("rust/scripts/run-plugin-task-gates.sh");
    let script = fs::read_to_string(&script_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", script_path.display()));
    let plan_path = root
        .join("docs/superpowers/plans/2026-08-26-native-rust-runtime-plugins-implementation.md");
    let plan = fs::read_to_string(&plan_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", plan_path.display()));

    let tasks = planned_task_commands(&plan);
    assert_eq!(
        tasks.keys().map(String::as_str).collect::<Vec<_>>(),
        (1..=40)
            .map(|task| task.to_string())
            .collect::<std::collections::BTreeSet<_>>()
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        "Task Gate Matrix must contain exactly Tasks 1 through 40"
    );
    let units = planned_unit_commands(&plan);
    let expected: BTreeMap<_, _> = tasks.into_iter().chain(units).collect();
    assert_eq!(script_commands(&script), expected);

    assert!(script.starts_with(
        "#!/bin/sh\n# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n# SPDX-License-Identifier: Apache-2.0\n\nset -eu\n\ntask=${1-}\n"
    ));
    assert!(script.contains("  *) echo \"unknown plugin task gate: $task\" >&2; exit 64 ;;\n"));
    assert!(
        script.ends_with("esac || exit $?\n\ncargo fmt --check\n"),
        "formatting must follow every successful task or unit gate"
    );
    assert_eq!(script.matches("cargo fmt --check").count(), 1);
}
