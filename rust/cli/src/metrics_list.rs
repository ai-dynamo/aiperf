// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! `aiperf metrics` command: introspection of the metric definition registry.
//!
//! Render-time only. `metrics list` prints one row per registered definition
//! (sorted by namespaced id); `metrics describe <id>` prints a single
//! definition's fields; `--markdown` on `list` emits a docs-ready table.

use aiperf_runtime::definitions::{self, Definition};

/// Namespaced id under which `def` is registered.
///
/// Metric ids are keyed as `aiperf.<bare id>` while analyzer definitions carry
/// an already-namespaced `def.id`. The registry exposes the keys through
/// [`definitions::ids_sorted`], so we resolve the display id by matching a
/// definition back to a sorted key rather than re-deriving the namespace.
fn namespaced_id_for(def: &'static Definition, keys: &[String]) -> String {
    for key in keys {
        if let Some(found) = definitions::definition(key) {
            if std::ptr::eq(found, def) {
                return key.clone();
            }
        }
    }
    // Fallback: the bare id is used as a last resort if no key matched (should
    // not happen for registered definitions).
    def.id.to_string()
}

/// Entry point for `aiperf metrics ...`.
pub fn run(args: &[String]) -> anyhow::Result<i32> {
    match args.first().map(String::as_str) {
        Some("list") => run_list(&args[1..]),
        Some("describe") => run_describe(&args[1..]),
        Some(other) => {
            eprintln!(
                "aiperf metrics: unknown subcommand {other:?}; expected `list` or `describe`"
            );
            Ok(2)
        }
        None => {
            eprintln!("aiperf metrics: missing subcommand; expected `list` or `describe`");
            Ok(2)
        }
    }
}

/// Returns every definition paired with its display id, sorted by that id.
fn sorted_rows() -> Vec<(String, &'static Definition)> {
    let keys = definitions::ids_sorted();
    let mut rows: Vec<(String, &'static Definition)> = definitions::all_definitions()
        .into_iter()
        .map(|def| (namespaced_id_for(def, &keys), def))
        .collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    rows
}

fn run_list(args: &[String]) -> anyhow::Result<i32> {
    let markdown = args.iter().any(|a| a == "--markdown");
    let rows = sorted_rows();

    if markdown {
        println!("| id | header | unit | larger_is_better |");
        println!("| --- | --- | --- | --- |");
        for (id, def) in &rows {
            println!(
                "| `{}` | {} | {} | {} |",
                id, def.header, def.unit, def.larger_is_better
            );
        }
        return Ok(0);
    }

    // Fixed-width plain table.
    let id_w = rows
        .iter()
        .map(|(id, _)| id.len())
        .chain(std::iter::once("id".len()))
        .max()
        .unwrap_or(2);
    let hdr_w = rows
        .iter()
        .map(|(_, d)| d.header.len())
        .chain(std::iter::once("header".len()))
        .max()
        .unwrap_or(6);
    let unit_w = rows
        .iter()
        .map(|(_, d)| d.unit.to_string().len())
        .chain(std::iter::once("unit".len()))
        .max()
        .unwrap_or(4);

    println!(
        "{:<id_w$}  {:<hdr_w$}  {:<unit_w$}  larger_is_better",
        "id", "header", "unit"
    );
    for (id, def) in &rows {
        println!(
            "{:<id_w$}  {:<hdr_w$}  {:<unit_w$}  {}",
            id,
            def.header,
            def.unit.to_string(),
            def.larger_is_better
        );
    }
    Ok(0)
}

fn run_describe(args: &[String]) -> anyhow::Result<i32> {
    let Some(id) = args.iter().find(|a| !a.starts_with('-')) else {
        eprintln!("aiperf metrics describe: missing <id>");
        return Ok(2);
    };

    let Some(def) = definitions::definition(id) else {
        eprintln!("aiperf metrics describe: unknown metric {id:?}");
        return Ok(1);
    };

    let keys = definitions::ids_sorted();
    println!("id:               {}", namespaced_id_for(def, &keys));
    println!("header:           {}", def.header);
    if let Some(sh) = def.short_header {
        println!("short_header:     {sh}");
    }
    println!("unit:             {}", def.unit);
    println!("display_unit:     {}", def.effective_display_unit());
    println!("larger_is_better: {}", def.larger_is_better);
    println!("value_type:       {:?}", def.value_type);
    println!("group:            {:?}", def.group);
    if !def.aliases.is_empty() {
        println!("aliases:          {}", def.aliases.join(", "));
    }
    if let Some(dep) = def.deprecated_since {
        println!("deprecated_since: {dep}");
    }
    Ok(0)
}
