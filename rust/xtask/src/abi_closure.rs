// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Computes the host/plugin ABI-facing type closure from rustdoc JSON.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

const RUSTDOC_TOOLCHAIN: &str = "nightly-2026-08-01";
const RUSTDOC_FORMAT_VERSION: u64 = 61;
static RUSTDOC_GENERATION: OnceLock<Mutex<()>> = OnceLock::new();

/// One ABI-facing type and where it is defined.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    /// Type name as written in source.
    pub name: String,
    /// Repository-relative path of the defining file.
    pub file: String,
    /// First line of the definition, 1-based.
    pub start: usize,
    /// Last line of the definition, 1-based inclusive.
    pub end: usize,
}

/// The seed set and blocked edges that define the boundary.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Seeds {
    /// Boundary entry points.
    pub seeds: Vec<String>,
    /// Edges at which the normative design stops traversal.
    pub blocked: Vec<String>,
}

impl Seeds {
    /// Read and validate one seed file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("reading seed file {}", path.display()))?;
        let seeds: Self = toml::from_str(&raw)
            .with_context(|| format!("parsing seed file {}", path.display()))?;
        if seeds.seeds.is_empty() {
            bail!("seed file {} contains no boundary seeds", path.display());
        }
        Ok(seeds)
    }
}

/// The measured closure.
#[derive(Debug)]
pub struct Closure {
    /// Reachable nominal types, keyed by an unambiguous display name.
    pub types: BTreeMap<String, Entry>,
    /// Distinct files contributing at least one reachable type.
    pub files: BTreeSet<String>,
    /// Total lines occupied by reachable type definitions.
    pub type_lines: usize,
    /// Total lines in the contributing files.
    pub file_lines: usize,
}

/// Stable JSON representation committed as the closure baseline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Baseline {
    /// Number of reachable nominal types.
    pub types: usize,
    /// Number of files defining reachable nominal types.
    pub files: usize,
    /// Lines occupied by reachable nominal type definitions.
    pub type_lines: usize,
    /// Total lines in files that define reachable nominal types.
    pub file_lines: usize,
    /// Reachable definitions, ordered by their map key.
    pub entries: Vec<Entry>,
}

impl Baseline {
    /// Project an in-memory closure into its stable serialized representation.
    pub fn from_closure(closure: &Closure) -> Self {
        Self {
            types: closure.types.len(),
            files: closure.files.len(),
            type_lines: closure.type_lines,
            file_lines: closure.file_lines,
            entries: closure.types.values().cloned().collect(),
        }
    }
}

/// Return the Cargo workspace root containing the xtask package.
pub fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Compute the closure by breadth-first traversal from the configured seeds.
pub fn compute(seeds: &Seeds) -> Result<Closure> {
    let workspace = workspace_root();
    let rustdoc = RustdocIndex::build(&workspace)?;
    let blocked: BTreeSet<&str> = seeds.blocked.iter().map(String::as_str).collect();
    let mut queue = VecDeque::new();
    for seed in &seeds.seeds {
        let matches = rustdoc.nominal_definitions_named(seed);
        match matches.as_slice() {
            [id] => queue.push_back((*id).to_owned()),
            [] => bail!("ABI seed {seed:?} is absent from aiperf-runtime rustdoc JSON"),
            _ => bail!("ABI seed {seed:?} resolves to multiple nominal type definitions"),
        }
    }

    let mut visited = BTreeSet::new();
    let mut entries = Vec::new();
    while let Some(id) = queue.pop_front() {
        if !visited.insert(id.clone()) {
            continue;
        }
        let Some(item) = rustdoc.item(&id) else {
            continue;
        };
        let name = item.get("name").and_then(Value::as_str);
        if name.is_some_and(|name| blocked.contains(name)) {
            continue;
        }
        if item.get("crate_id").and_then(Value::as_u64) != Some(0) {
            continue;
        }
        if is_nominal_definition(item) {
            entries.push(entry_from_item(item)?);
        }
        for referenced in structural_references(item) {
            if !visited.contains(&referenced) {
                queue.push_back(referenced);
            }
        }
    }

    let mut types = BTreeMap::new();
    for entry in entries {
        let key = if types.contains_key(&entry.name) {
            format!("{}@{}:{}", entry.name, entry.file, entry.start)
        } else {
            entry.name.clone()
        };
        types.insert(key, entry);
    }
    let files: BTreeSet<String> = types.values().map(|entry| entry.file.clone()).collect();
    let type_lines = types
        .values()
        .map(|entry| entry.end.saturating_sub(entry.start) + 1)
        .sum();
    let file_lines = files
        .iter()
        .map(|file| {
            std::fs::read_to_string(workspace.join(file))
                .with_context(|| format!("reading ABI source file {file}"))
                .map(|source| source.lines().count())
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .sum();

    Ok(Closure {
        types,
        files,
        type_lines,
        file_lines,
    })
}

struct RustdocIndex {
    index: Map<String, Value>,
}

impl RustdocIndex {
    fn build(workspace: &Path) -> Result<Self> {
        let generation = RUSTDOC_GENERATION.get_or_init(|| Mutex::new(()));
        let _guard = generation
            .lock()
            .map_err(|_| anyhow::anyhow!("rustdoc generation lock was poisoned"))?;
        let cargo = rustup_which("cargo")?;
        let rustc = rustup_which("rustc")?;
        let rustdoc = rustup_which("rustdoc")?;
        let rustc_config = cargo_path_config("rustc", &rustc);
        let rustdoc_config = cargo_path_config("rustdoc", &rustdoc);
        let status = Command::new(&cargo)
            .current_dir(workspace)
            .args([
                "rustdoc",
                "--config",
                &rustc_config,
                "--config",
                &rustdoc_config,
                "-p",
                "aiperf-runtime",
                "--lib",
                "--features",
                "engine",
                "--",
                "--document-private-items",
                "-Z",
                "unstable-options",
                "--output-format",
                "json",
                "--cap-lints",
                "allow",
            ])
            .status()
            .with_context(|| format!("running pinned {RUSTDOC_TOOLCHAIN} cargo rustdoc"))?;
        if !status.success() {
            bail!(
                "pinned rustdoc JSON generation failed with {status}; install {RUSTDOC_TOOLCHAIN} with rustup and retry"
            );
        }

        let json_path = workspace.join("target/doc/aiperf_runtime.json");
        let root: Value = serde_json::from_reader(
            std::fs::File::open(&json_path)
                .with_context(|| format!("opening {}", json_path.display()))?,
        )
        .with_context(|| format!("parsing {}", json_path.display()))?;
        let format_version = root.get("format_version").and_then(Value::as_u64);
        if format_version != Some(RUSTDOC_FORMAT_VERSION) {
            bail!(
                "unsupported rustdoc JSON format {format_version:?}; {RUSTDOC_TOOLCHAIN} requires {RUSTDOC_FORMAT_VERSION}"
            );
        }
        if root.get("includes_private").and_then(Value::as_bool) != Some(true) {
            bail!("rustdoc JSON omitted private fields required by the ABI closure")
        }
        let index = root
            .get("index")
            .and_then(Value::as_object)
            .context("rustdoc JSON has no object-valued index")?
            .clone();
        Ok(Self { index })
    }

    fn item(&self, id: &str) -> Option<&Value> {
        self.index.get(id)
    }

    fn nominal_definitions_named(&self, name: &str) -> Vec<&str> {
        self.index
            .iter()
            .filter_map(|(id, item)| {
                (item.get("crate_id").and_then(Value::as_u64) == Some(0)
                    && item.get("name").and_then(Value::as_str) == Some(name)
                    && is_nominal_definition(item))
                .then_some(id.as_str())
            })
            .collect()
    }
}

fn rustup_which(component: &str) -> Result<PathBuf> {
    let output = Command::new("rustup")
        .args(["which", component, "--toolchain", RUSTDOC_TOOLCHAIN])
        .output()
        .with_context(|| format!("locating {component} for {RUSTDOC_TOOLCHAIN}"))?;
    if !output.status.success() {
        bail!(
            "{RUSTDOC_TOOLCHAIN} is unavailable; install it with `rustup toolchain install {RUSTDOC_TOOLCHAIN} --profile minimal`"
        );
    }
    let path = String::from_utf8(output.stdout).context("rustup emitted a non-UTF-8 path")?;
    Ok(PathBuf::from(path.trim()))
}

fn cargo_path_config(component: &str, path: &Path) -> String {
    format!("build.{component}={:?}", path.display().to_string())
}

fn is_nominal_definition(item: &Value) -> bool {
    item.get("inner")
        .and_then(Value::as_object)
        .is_some_and(|inner| {
            ["struct", "enum", "union", "trait", "type_alias"]
                .iter()
                .any(|kind| inner.contains_key(*kind))
        })
}

fn entry_from_item(item: &Value) -> Result<Entry> {
    let name = item
        .get("name")
        .and_then(Value::as_str)
        .context("nominal rustdoc item has no name")?;
    let span = item
        .get("span")
        .and_then(Value::as_object)
        .with_context(|| format!("rustdoc item {name} has no source span"))?;
    let file = span
        .get("filename")
        .and_then(Value::as_str)
        .with_context(|| format!("rustdoc item {name} has no source filename"))?;
    let start = span_line(span, "begin", name)?;
    let end = span_line(span, "end", name)?;
    Ok(Entry {
        name: name.to_owned(),
        file: file.to_owned(),
        start,
        end,
    })
}

fn span_line(span: &Map<String, Value>, endpoint: &str, name: &str) -> Result<usize> {
    span.get(endpoint)
        .and_then(Value::as_array)
        .and_then(|position| position.first())
        .and_then(Value::as_u64)
        .map(|line| line as usize)
        .with_context(|| format!("rustdoc item {name} has no {endpoint} line"))
}

fn structural_references(item: &Value) -> BTreeSet<String> {
    let mut references = BTreeSet::new();
    let Some(inner) = item.get("inner").and_then(Value::as_object) else {
        return references;
    };
    for (kind, value) in inner {
        match kind.as_str() {
            "struct" | "enum" | "union" => {
                collect_member_ids(value, &mut references);
                let mut structural = value.clone();
                if let Some(object) = structural.as_object_mut() {
                    object.remove("impls");
                    object.remove("variants");
                    object.remove("fields");
                    object.remove("kind");
                }
                collect_named_ids(&structural, &mut references);
            }
            "trait" => {
                collect_array_ids(value.get("items"), &mut references);
                let mut structural = value.clone();
                if let Some(object) = structural.as_object_mut() {
                    object.remove("items");
                    object.remove("implementations");
                }
                collect_named_ids(&structural, &mut references);
            }
            "struct_field" | "variant" | "function" | "type_alias" | "assoc_type"
            | "assoc_const" | "constant" | "static" | "opaque_ty" => {
                collect_named_ids(value, &mut references);
                if kind == "variant" {
                    collect_member_ids(value, &mut references);
                }
            }
            _ => {}
        }
    }
    references
}

fn collect_member_ids(value: &Value, output: &mut BTreeSet<String>) {
    if let Some(kind) = value.get("kind") {
        if let Some(plain) = kind.get("plain") {
            collect_array_ids(plain.get("fields"), output);
        }
        if let Some(tuple) = kind.get("tuple") {
            collect_array_ids(Some(tuple), output);
        }
        if let Some(struct_kind) = kind.get("struct") {
            collect_array_ids(struct_kind.get("fields"), output);
        }
    }
    collect_array_ids(value.get("variants"), output);
    collect_array_ids(value.get("fields"), output);
}

fn collect_array_ids(value: Option<&Value>, output: &mut BTreeSet<String>) {
    let Some(values) = value.and_then(Value::as_array) else {
        return;
    };
    for value in values {
        if let Some(id) = value.as_u64() {
            output.insert(id.to_string());
        }
    }
}

fn collect_named_ids(value: &Value, output: &mut BTreeSet<String>) {
    match value {
        Value::Object(object) => {
            for (key, value) in object {
                if key == "id" {
                    if let Some(id) = value.as_u64() {
                        output.insert(id.to_string());
                    }
                } else {
                    collect_named_ids(value, output);
                }
            }
        }
        Value::Array(values) => {
            for value in values {
                collect_named_ids(value, output);
            }
        }
        _ => {}
    }
}
