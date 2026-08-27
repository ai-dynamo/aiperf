// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replays first-parent merge history against the measured ABI universe.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde::Serialize;
use syn::spanned::Spanned;

use crate::abi_closure::{Baseline, Entry};

/// Inclusive source line span for one ABI-facing definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceSpan {
    /// First definition line, 1-based.
    pub start: usize,
    /// Last definition line, 1-based.
    pub end: usize,
}

/// Classification counts under one granularity.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct Classification {
    /// Units that rebuild the ABI universe.
    pub universe: usize,
    /// Units isolated to one initial-generation plugin category.
    pub one_plugin: usize,
    /// Remaining host-only code units.
    pub host_only: usize,
}

/// Rebuild classification for one merge-history window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ChurnReport {
    /// First-parent merge commits in the window that changed Rust source.
    pub code_units: usize,
    /// Classification when any ABI-contributing file rebuilds the universe.
    pub file_granular: Classification,
    /// Classification when only a hunk overlapping a type definition rebuilds it.
    pub type_granular: Classification,
}

/// Locate one nominal type definition using Rust syntax rather than text matching.
pub fn definition_span(source: &str, name: &str) -> Option<(usize, usize)> {
    let file = syn::parse_file(source).ok()?;
    definition_in_items(&file.items, name)
}

/// Return whether any unified-diff hunk overlaps a type on its old or new side.
pub fn patch_overlaps_type(
    patch: &str,
    old_spans: &BTreeMap<String, Vec<SourceSpan>>,
    new_spans: &BTreeMap<String, Vec<SourceSpan>>,
) -> bool {
    let mut old_path = None;
    let mut new_path = None;
    for line in patch.lines() {
        if let Some(path) = line.strip_prefix("--- a/") {
            old_path = Some(path.to_owned());
        } else if let Some(path) = line.strip_prefix("+++ b/") {
            new_path = Some(path.to_owned());
        } else if line.starts_with("@@ ") {
            let mut parts = line.split_whitespace();
            let _marker = parts.next();
            let old_range = parts.next().and_then(parse_hunk_range);
            let new_range = parts.next().and_then(parse_hunk_range);
            if old_range.is_some_and(|range| {
                old_path.as_ref().is_some_and(|path| {
                    old_spans
                        .get(path)
                        .is_some_and(|spans| spans.iter().any(|span| range.overlaps(*span)))
                })
            }) || new_range.is_some_and(|range| {
                new_path.as_ref().is_some_and(|path| {
                    new_spans
                        .get(path)
                        .is_some_and(|spans| spans.iter().any(|span| range.overlaps(*span)))
                })
            }) {
                return true;
            }
        }
    }
    false
}

/// Measure ABI rebuild churn over first-parent merge commits.
pub fn measure(
    repository: &Path,
    baseline: &Baseline,
    since: &str,
    merges: usize,
) -> Result<ChurnReport> {
    if merges == 0 {
        bail!("--merges must be greater than zero");
    }
    let revisions = git_lines(
        repository,
        [
            "rev-list",
            "--first-parent",
            "--merges",
            &format!("--max-count={merges}"),
            since,
        ],
    )?;
    let abi_files: BTreeSet<String> = baseline
        .entries
        .iter()
        .map(|entry| format!("rust/{}", entry.file))
        .collect();
    let entries_by_file = entries_by_file(&baseline.entries);
    let mut file_granular = Classification::default();
    let mut type_granular = Classification::default();
    let mut code_units = 0;

    for revision in revisions {
        let parent = format!("{revision}^1");
        let paths = git_lines(
            repository,
            [
                "diff",
                "--name-only",
                "--no-renames",
                &parent,
                &revision,
                "--",
                "rust",
            ],
        )?
        .into_iter()
        .filter(|path| path.ends_with(".rs"))
        .collect::<Vec<_>>();
        if paths.is_empty() {
            continue;
        }
        code_units += 1;
        let is_one_plugin = one_plugin_root(&paths).is_some();
        let is_file_universe = paths.iter().any(|path| abi_files.contains(path));
        classify(&mut file_granular, is_file_universe, is_one_plugin);

        let changed_abi_files = paths
            .iter()
            .filter(|path| abi_files.contains(*path))
            .cloned()
            .collect::<Vec<_>>();
        let is_type_universe = if changed_abi_files.is_empty() {
            false
        } else {
            let old_spans = spans_at(repository, &parent, &changed_abi_files, &entries_by_file)?;
            let new_spans = spans_at(repository, &revision, &changed_abi_files, &entries_by_file)?;
            let patch = git_output(
                repository,
                [
                    "diff",
                    "--unified=0",
                    "--no-ext-diff",
                    "--no-renames",
                    &parent,
                    &revision,
                    "--",
                    "rust",
                ],
            )?;
            patch_overlaps_type(&patch, &old_spans, &new_spans)
        };
        classify(&mut type_granular, is_type_universe, is_one_plugin);
    }

    Ok(ChurnReport {
        code_units,
        file_granular,
        type_granular,
    })
}

fn definition_in_items(items: &[syn::Item], name: &str) -> Option<(usize, usize)> {
    for item in items {
        let matched = match item {
            syn::Item::Enum(item) if item.ident == name => Some(item.span()),
            syn::Item::Struct(item) if item.ident == name => Some(item.span()),
            syn::Item::Trait(item) if item.ident == name => Some(item.span()),
            syn::Item::Type(item) if item.ident == name => Some(item.span()),
            syn::Item::Union(item) if item.ident == name => Some(item.span()),
            _ => None,
        };
        if let Some(span) = matched {
            return Some((span.start().line, span.end().line));
        }
        if let syn::Item::Mod(module) = item {
            if let Some((_, nested)) = &module.content {
                if let Some(span) = definition_in_items(nested, name) {
                    return Some(span);
                }
            }
        }
    }
    None
}

#[derive(Debug, Clone, Copy)]
struct HunkRange {
    start: usize,
    count: usize,
}

impl HunkRange {
    fn overlaps(self, span: SourceSpan) -> bool {
        if self.count == 0 {
            return self.start >= span.start.saturating_sub(1) && self.start <= span.end;
        }
        let end = self.start.saturating_add(self.count - 1);
        self.start <= span.end && end >= span.start
    }
}

fn parse_hunk_range(raw: &str) -> Option<HunkRange> {
    let coordinates = raw.strip_prefix(['-', '+'])?;
    let mut pieces = coordinates.split(',');
    let start = pieces.next()?.parse().ok()?;
    let count = pieces.next().map(str::parse).transpose().ok()?.unwrap_or(1);
    Some(HunkRange { start, count })
}

fn entries_by_file(entries: &[Entry]) -> BTreeMap<String, Vec<String>> {
    let mut by_file: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for entry in entries {
        by_file
            .entry(format!("rust/{}", entry.file))
            .or_default()
            .push(entry.name.clone());
    }
    by_file
}

fn spans_at(
    repository: &Path,
    revision: &str,
    paths: &[String],
    entries_by_file: &BTreeMap<String, Vec<String>>,
) -> Result<BTreeMap<String, Vec<SourceSpan>>> {
    let mut spans = BTreeMap::new();
    for path in paths {
        let Some(names) = entries_by_file.get(path) else {
            continue;
        };
        let Some(source) = git_show_optional(repository, revision, path)? else {
            continue;
        };
        let definitions = names
            .iter()
            .filter_map(|name| definition_span(&source, name))
            .map(|(start, end)| SourceSpan { start, end })
            .collect::<Vec<_>>();
        spans.insert(path.clone(), definitions);
    }
    Ok(spans)
}

fn one_plugin_root(paths: &[String]) -> Option<&str> {
    let mut roots = paths.iter().filter_map(|path| {
        let relative = path.strip_prefix("rust/runtime/src/")?;
        let root = relative.split('/').next()?;
        ["endpoints", "transport", "export"]
            .contains(&root)
            .then_some(root)
    });
    let first = roots.next()?;
    if roots.all(|root| root == first)
        && paths.iter().all(|path| {
            path.strip_prefix("rust/runtime/src/")
                .and_then(|relative| relative.split('/').next())
                == Some(first)
        })
    {
        Some(first)
    } else {
        None
    }
}

fn classify(counts: &mut Classification, is_universe: bool, is_one_plugin: bool) {
    if is_universe {
        counts.universe += 1;
    } else if is_one_plugin {
        counts.one_plugin += 1;
    } else {
        counts.host_only += 1;
    }
}

fn git_show_optional(repository: &Path, revision: &str, path: &str) -> Result<Option<String>> {
    let output = Command::new("git")
        .current_dir(repository)
        .args(["show", &format!("{revision}:{path}")])
        .output()
        .with_context(|| format!("reading {path} at {revision}"))?;
    if !output.status.success() {
        return Ok(None);
    }
    String::from_utf8(output.stdout)
        .context("git show emitted non-UTF-8 Rust source")
        .map(Some)
}

fn git_lines<const N: usize>(repository: &Path, arguments: [&str; N]) -> Result<Vec<String>> {
    Ok(git_output(repository, arguments)?
        .lines()
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
        .collect())
}

fn git_output<const N: usize>(repository: &Path, arguments: [&str; N]) -> Result<String> {
    let output = Command::new("git")
        .current_dir(repository)
        .args(arguments)
        .output()
        .context("running git for ABI churn replay")?;
    if !output.status.success() {
        bail!("git ABI churn command failed with {}", output.status);
    }
    String::from_utf8(output.stdout).context("git emitted non-UTF-8 output")
}
