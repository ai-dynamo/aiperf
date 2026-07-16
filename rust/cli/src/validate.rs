// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Native `aiperf validate mooncake-trace` — pure-Rust port of
//! `aiperf.dataset.agentic_code_gen.cli.validate` +
//! `reporting.trace.validate_mooncake_trace` + the `MooncakeTrace` model
//! validators (`dataset/loader/models.py`). No Python: checks each JSONL row
//! against the Mooncake schema (field types + the exactly-one-input-mode /
//! hash_ids / payload / tools / messages rules) and prints the same
//! pass/fail summary.
//!
//! The pass-path message (`Validation passed: N rows are Mooncake-compatible.`)
//! is byte-exact vs Python; per-error wording is Rust-native (pydantic's
//! multi-line error text is not reproduced verbatim) but reports the same line
//! and rule.

use std::path::PathBuf;

use serde_json::Value;

/// Validate one decoded JSON row against the `MooncakeTrace` model rules.
/// Returns `Err(reason)` on the first violated rule.
fn validate_row(v: &Value) -> Result<(), String> {
    let obj = v
        .as_object()
        .ok_or_else(|| "row is not a JSON object".to_string())?;

    // Field-type checks (mirror the pydantic field annotations).
    let is_int = |k: &str| {
        obj.get(k)
            .is_none_or(|x| x.is_null() || x.is_i64() || x.is_u64())
    };
    let is_num = |k: &str| obj.get(k).is_none_or(|x| x.is_null() || x.is_number());
    let is_str = |k: &str| obj.get(k).is_none_or(|x| x.is_null() || x.is_string());
    let is_arr = |k: &str| obj.get(k).is_none_or(|x| x.is_null() || x.is_array());
    let is_obj = |k: &str| obj.get(k).is_none_or(|x| x.is_null() || x.is_object());
    for k in ["input_length", "output_length"] {
        if !is_int(k) {
            return Err(format!("'{k}' must be an integer"));
        }
    }
    for k in ["timestamp", "delay"] {
        if !is_num(k) {
            return Err(format!("'{k}' must be a number"));
        }
    }
    for k in ["text_input", "session_id"] {
        if !is_str(k) {
            return Err(format!("'{k}' must be a string"));
        }
    }
    for k in ["messages", "tools", "hash_ids"] {
        if !is_arr(k) {
            return Err(format!("'{k}' must be a list"));
        }
    }
    for k in ["payload", "extra"] {
        if !is_obj(k) {
            return Err(format!("'{k}' must be an object"));
        }
    }
    // hash_ids must be a list of ints.
    if let Some(h) = obj.get("hash_ids").filter(|x| !x.is_null()) {
        if !h
            .as_array()
            .unwrap()
            .iter()
            .all(|e| e.is_i64() || e.is_u64())
        {
            return Err("'hash_ids' must be a list of integers".to_string());
        }
    }

    let present = |k: &str| obj.get(k).is_some_and(|x| !x.is_null());

    // Exactly one input mode.
    let modes = [
        present("input_length"),
        present("text_input"),
        present("messages"),
        present("payload"),
    ];
    let mode_count = modes.iter().filter(|&&m| m).count();
    if mode_count == 0 {
        return Err(
            "Exactly one of 'input_length', 'text_input', 'messages', or 'payload' must be provided"
                .to_string(),
        );
    }
    if mode_count > 1 {
        return Err(
            "'input_length', 'text_input', 'messages', and 'payload' are mutually exclusive. Use only one of them."
                .to_string(),
        );
    }

    // hash_ids only with input_length.
    if present("hash_ids") && !present("input_length") {
        return Err(
            "'hash_ids' is only allowed when 'input_length' is provided, not when 'text_input', 'messages', or 'payload' are provided"
                .to_string(),
        );
    }

    // payload non-empty.
    if let Some(p) = obj.get("payload").filter(|x| !x.is_null()) {
        if p.as_object().unwrap().is_empty() {
            return Err("'payload' must be a non-empty dict".to_string());
        }
    }

    // tools only with messages, non-empty.
    if present("tools") {
        if !present("messages") {
            return Err("'tools' is only allowed when 'messages' is provided".to_string());
        }
        if obj["tools"].as_array().unwrap().is_empty() {
            return Err("'tools' must be a non-empty list".to_string());
        }
    }

    // messages: non-empty list of dicts each with a 'role' key.
    if let Some(m) = obj.get("messages").filter(|x| !x.is_null()) {
        let arr = m.as_array().unwrap();
        if arr.is_empty() {
            return Err("'messages' must be a non-empty list".to_string());
        }
        for (i, msg) in arr.iter().enumerate() {
            if !msg.as_object().is_some_and(|o| o.contains_key("role")) {
                return Err(format!(
                    "Each message must have a 'role' key, but message at index {i} does not"
                ));
            }
        }
    }

    Ok(())
}

/// Validate a Mooncake JSONL file: returns `(line_count, errors)` capped at
/// `max_errors` (mirrors `validate_mooncake_trace`).
fn validate_mooncake(path: &std::path::Path) -> anyhow::Result<(usize, Vec<String>)> {
    const MAX_ERRORS: usize = 10;
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("failed to read {}: {e}", path.display()))?;
    let mut line_count = 0;
    let mut errors = Vec::new();
    for (line_num, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        line_count += 1;
        let result = serde_json::from_str::<Value>(line)
            .map_err(|e| format!("{e}"))
            .and_then(|v| validate_row(&v));
        if let Err(reason) = result {
            errors.push(format!("Line {}: {reason}", line_num + 1));
            if errors.len() >= MAX_ERRORS {
                break;
            }
        }
    }
    Ok((line_count, errors))
}

/// Validate a Mooncake JSONL file, exposed for the `synthesize` command's
/// post-generation validation step (mirrors `cli.py::_validate_mooncake_or_exit`).
pub fn validate_mooncake_public(path: &std::path::Path) -> anyhow::Result<(usize, Vec<String>)> {
    validate_mooncake(path)
}

/// Run `aiperf validate mooncake-trace --input <path>`.
pub fn run(args: &[String]) -> anyhow::Result<i32> {
    let mut target: Option<String> = None;
    let mut input: Option<PathBuf> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--input" => {
                input = Some(PathBuf::from(
                    it.next()
                        .ok_or_else(|| anyhow::anyhow!("--input needs a value"))?,
                ));
            }
            other if other.starts_with('-') => anyhow::bail!("unknown validate flag {other:?}"),
            other => target = Some(other.to_string()),
        }
    }
    match target.as_deref() {
        Some("mooncake-trace") => {}
        Some(t) => anyhow::bail!("unknown validate target {t:?} (expected mooncake-trace)"),
        None => anyhow::bail!("validate requires a target (mooncake-trace)"),
    }
    let input = input.ok_or_else(|| anyhow::anyhow!("validate requires --input"))?;
    if !input.is_file() {
        println!("Validation failed: {} is not a file.", input.display());
        return Ok(1);
    }

    let (line_count, errors) = validate_mooncake(&input)?;
    if !errors.is_empty() {
        println!("Validation failed with {} error(s):", errors.len());
        for err in &errors {
            println!("  {err}");
        }
        return Ok(1);
    }
    println!("Validation passed: {line_count} rows are Mooncake-compatible.");
    Ok(0)
}
