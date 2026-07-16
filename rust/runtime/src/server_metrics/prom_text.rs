// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared Prometheus text-exposition label grammar.
//!
//! The server-metrics and GPU-telemetry decoders both parse the classic
//! Prometheus `metric{label="value",...} value` sample syntax. The value-split
//! scanner, `{...}` label-set splitter, and quoted-value unescaper are identical
//! between them; only label-name validation differs (the server-metrics parser
//! enforces the full Prometheus name grammar, while the DCGM decoder only
//! rejects empty names). [`parse_labels`] therefore takes the name validator as
//! a closure so each caller keeps its exact behavior and error text.

use std::collections::BTreeMap;

/// Finds the byte index of the first unquoted, top-level whitespace separating
/// the `metric{...}` head from the sample value, or `None` when the line has no
/// value token.
pub(crate) fn sample_value_split(line: &str) -> Option<usize> {
    let mut in_quotes = false;
    let mut escaped = false;
    let mut brace_depth = 0_u32;
    for (index, byte) in line.bytes().enumerate() {
        if escaped {
            escaped = false;
            continue;
        }
        match byte {
            b'\\' if in_quotes => escaped = true,
            b'"' => in_quotes = !in_quotes,
            b'{' if !in_quotes => brace_depth += 1,
            b'}' if !in_quotes => brace_depth = brace_depth.saturating_sub(1),
            b' ' | b'\t' if !in_quotes && brace_depth == 0 => return Some(index),
            _ => {}
        }
    }
    None
}

/// Splits a `metric` head into its name and label map, validating each label
/// name with `validate_name`.
pub(crate) fn parse_metric_and_labels(
    metric: &str,
    validate_name: impl Fn(&str) -> Result<(), String>,
) -> Result<(String, BTreeMap<String, String>), String> {
    let Some(open) = metric.find('{') else {
        return Ok((metric.to_string(), BTreeMap::new()));
    };
    if !metric.ends_with('}') {
        return Err("unterminated label set".to_string());
    }
    let name = metric[..open].to_string();
    let labels = parse_labels(&metric[open + 1..metric.len() - 1], validate_name)?;
    Ok((name, labels))
}

/// Parses the comma-separated `name="value"` body inside `{...}`, validating
/// each label name via `validate_name`.
pub(crate) fn parse_labels(
    mut input: &str,
    validate_name: impl Fn(&str) -> Result<(), String>,
) -> Result<BTreeMap<String, String>, String> {
    let mut labels = BTreeMap::new();
    while !input.trim_start().is_empty() {
        input = input.trim_start();
        let equals = input
            .find('=')
            .ok_or_else(|| "label has no '='".to_string())?;
        let name = input[..equals].trim();
        validate_name(name)?;
        input = input[equals + 1..].trim_start();
        let Some(rest) = input.strip_prefix('"') else {
            return Err(format!("label {name:?} has an unquoted value"));
        };
        let (value, consumed) = parse_quoted_label(rest)?;
        labels.insert(name.to_string(), value);
        input = rest[consumed..].trim_start();
        if input.is_empty() {
            break;
        }
        let Some(rest) = input.strip_prefix(',') else {
            return Err("labels must be comma-separated".to_string());
        };
        input = rest;
    }
    Ok(labels)
}

/// Reads one double-quoted label value starting after the opening quote,
/// unescaping `\n`, `\\`, and `\"`, and returns the value plus the number of
/// bytes consumed including the closing quote.
pub(crate) fn parse_quoted_label(input: &str) -> Result<(String, usize), String> {
    let mut output = String::new();
    let mut escaped = false;
    for (index, character) in input.char_indices() {
        if escaped {
            output.push(match character {
                'n' => '\n',
                '\\' => '\\',
                '"' => '"',
                other => other,
            });
            escaped = false;
            continue;
        }
        match character {
            '\\' => escaped = true,
            '"' => return Ok((output, index + character.len_utf8())),
            other => output.push(other),
        }
    }
    Err("unterminated quoted label".to_string())
}
