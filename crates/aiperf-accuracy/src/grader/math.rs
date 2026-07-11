// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native math answer extraction, normalization, and equivalence.
//!
//! Source-grounded in `src/aiperf/accuracy/graders/math.py:1-623` and
//! `_math_strip.py:1-385`. The native evaluator keeps the reference strategy
//! order and implements constant arithmetic/LaTeX equivalence without a Python
//! sympy process.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::LazyLock;

use aiperf_metrics::GradingResult;
use async_trait::async_trait;
use regex::Regex;

use super::Grader;
use crate::AccuracyError;

const BOXED: &str = "\\boxed{";
const ABS_TOLERANCE: f64 = 1e-4;
const SYMBOLIC_TOLERANCE: f64 = 1e-7;

// Keep this list in the source order used by the inherited recipe. Some entries
// deliberately contain regex metacharacters because the Python implementation
// interpolates them into the pattern without escaping.
const UNIT_TEXTS_BASE: &[&str] = &[
    "east",
    "degree",
    "mph",
    "kmph",
    "ft",
    "m square",
    " m east",
    "sq m",
    "deg",
    "mile",
    "q .",
    "monkey",
    "prime",
    "ratio",
    "profit of rs",
    "rd",
    "o",
    "gm",
    "p . m",
    "lb",
    "tile",
    "per",
    "dm",
    "lt",
    "gain",
    "ab",
    "way",
    "west",
    "a .",
    "b .",
    "c .",
    "d .",
    "e .",
    "f .",
    "g .",
    "h .",
    "t",
    "a",
    "h",
    "no change",
    "men",
    "soldier",
    "pie",
    "bc",
    "excess",
    "st",
    "inches",
    "noon",
    "percent",
    "by",
    "gal",
    "kmh",
    "c",
    "acre",
    "rise",
    "a . m",
    "th",
    "π r 2",
    "sq",
    "mark",
    "l",
    "toy",
    "coin",
    "sq . m",
    "gallon",
    "° f",
    "profit",
    "minw",
    "yr",
    "women",
    "feet",
    "am",
    "pm",
    "hr",
    "cu cm",
    "square",
    "v â € ™",
    "are",
    "rupee",
    "rounds",
    "cubic",
    "cc",
    "mtr",
    "s",
    "ohm",
    "number",
    "kmph",
    "day",
    "hour",
    "minute",
    "min",
    "second",
    "man",
    "woman",
    "sec",
    "cube",
    "mt",
    "sq inch",
    "mp",
    "∏ cm ³",
    "hectare",
    "more",
    "sec",
    "unit",
    "cu . m",
    "cm 2",
    "rs .",
    "rs",
    "kg",
    "g",
    "month",
    "km",
    "m",
    "cm",
    "mm",
    "apple",
    "liter",
    "loss",
    "yard",
    "pure",
    "year",
    "increase",
    "decrease",
    "d",
    "less",
    "Surface",
    "litre",
    "pi sq m",
    "s .",
    "metre",
    "meter",
    "inch",
];

static MATRIX_BEGIN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\\begin\{array\}\{.*?\}").expect("static matrix regex"));
static MATRIX_END: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\\end\{array\}").expect("static matrix regex"));
static TRAILING_TEXT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\\text\{.*?\}$").expect("static text regex"));
static TEXT_COMMAND: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\\text\{(.*?)\}").expect("static text regex"));
static MBOX_COMMAND: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\\mbox\{.*?\}").expect("static mbox regex"));
static MONTH: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b",
    )
    .expect("static month regex")
});
static NUMERIC_ZERO_WITH_SUFFIX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(\d+)\.0*([^\d])").expect("static numeric-zero regex"));
static NUMERIC_ZERO_AT_END: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(\d+)\.0*$").expect("static numeric-zero regex"));
static UNIT_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    UNIT_TEXTS_BASE
        .iter()
        .map(|unit| (*unit).to_string())
        .chain(UNIT_TEXTS_BASE.iter().map(|unit| format!("{unit}s")))
        .map(|unit| {
            Regex::new(&format!(r"(^|\W){unit}($|\W)"))
                .unwrap_or_else(|error| panic!("invalid inherited unit regex {unit:?}: {error}"))
        })
        .collect()
});

/// Native math/AIME grader.
#[derive(Debug)]
pub struct MathGrader {
    answer_phrase: Regex,
    number: Regex,
    choice: Regex,
}

impl MathGrader {
    /// Builds the precompiled grader.
    pub fn new() -> Self {
        Self {
            answer_phrase: Regex::new(
                r"(?i)(?:final\s+answer|the\s+answer\s+is|answer\s*[:=]|answer\s+is)\s*[:=]?\s*",
            )
            .expect("static regex"),
            number: Regex::new(r"-?\d+(?:\.\d+)?(?:/\d+)?").expect("static regex"),
            choice: Regex::new(r"(?i)\b([A-E])\b").expect("static regex"),
        }
    }

    /// Extracts `(answer, fallback_used)` with boxed > final-answer phrase >
    /// last-number priority.
    pub fn extract_answer(&self, response_text: &str) -> (String, bool) {
        if response_text.is_empty() {
            return (String::new(), true);
        }
        if let Some(boxed) = extract_last_boxed(response_text) {
            return (format_response_tail(&boxed), false);
        }
        if let Some(tail) = self.last_answer_phrase_tail(response_text) {
            if let Some(boxed) = extract_last_boxed(&tail) {
                return (format_response_tail(&boxed), true);
            }
            if tail.contains(['\\', '{', '}']) {
                return (tail, true);
            }
            if let Some(number) = self.number.find_iter(&tail).last() {
                return (number.as_str().to_string(), true);
            }
            return (tail, true);
        }
        if let Some(number) = self.number.find_iter(response_text).last() {
            return (number.as_str().to_string(), true);
        }
        (response_text.trim().to_string(), true)
    }

    fn last_answer_phrase_tail(&self, response: &str) -> Option<String> {
        let marker = self.answer_phrase.find_iter(response).last()?;
        let remainder = &response[marker.end()..];
        let bytes = remainder.as_bytes();
        let mut end = remainder.len();
        for index in 0..bytes.len() {
            if bytes[index] == b'\n' {
                end = index;
                break;
            }
            if bytes[index] == b'.' {
                let left_digit = index > 0 && bytes[index - 1].is_ascii_digit();
                let right_digit = index + 1 < bytes.len() && bytes[index + 1].is_ascii_digit();
                if !(left_digit && right_digit) {
                    end = index;
                    break;
                }
            }
        }
        Some(remainder[..end].trim().to_string())
    }

    /// Compares already-extracted answers under the native math policy.
    pub fn equivalent(&self, prediction: &str, reference: &str) -> bool {
        math_equal(
            &strip_math_string(prediction),
            &strip_math_string(reference),
            &self.choice,
            0,
        )
    }
}

impl Default for MathGrader {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait(?Send)]
impl Grader for MathGrader {
    fn name(&self) -> &'static str {
        "math"
    }

    async fn grade(
        &self,
        response_text: &str,
        ground_truth: &str,
    ) -> Result<GradingResult, AccuracyError> {
        let (prediction, fallback) = self.extract_answer(response_text);
        let prediction_normalized = strip_math_string(&prediction);
        let gold_normalized = strip_math_string(ground_truth);
        let correct = math_equal(&prediction_normalized, &gold_normalized, &self.choice, 0);
        Ok(GradingResult {
            correct,
            unparsed: fallback,
            confidence: Some(if correct { 1.0 } else { 0.0 }),
            extracted: Some(prediction.clone()),
            ground_truth: ground_truth.trim().to_string(),
            reasoning: Some(format!(
                "extracted {prediction:?} (normalized {prediction_normalized:?}); ground truth {:?} (normalized {gold_normalized:?}); native math equivalence={correct}{}",
                ground_truth.trim(),
                if fallback { " (regex fallback)" } else { "" }
            )),
        })
    }
}

pub(super) fn extract_last_boxed(text: &str) -> Option<String> {
    let start = text.rfind(BOXED)? + BOXED.len();
    let mut depth = 0usize;
    for (offset, character) in text[start..].char_indices() {
        match character {
            '{' => depth += 1,
            '}' if depth == 0 => return Some(text[start..start + offset].to_string()),
            '}' => depth -= 1,
            _ => {}
        }
    }
    None
}

fn format_response_tail(content: &str) -> String {
    let mut output = content
        .lines()
        .map(str::trim_start)
        .collect::<Vec<_>>()
        .join("");
    if output.starts_with(':') {
        output.remove(0);
    }
    if output.ends_with(['.', '/']) {
        output.pop();
    }
    output
}

/// Hendrycks/ToRA-style normalization used before comparison.
pub(super) fn strip_math_string(input: &str) -> String {
    // This order follows `_math_strip.py:212-385`; seemingly redundant
    // replacements and the non-escaped unit regexes are observable behavior.
    let mut value = input.trim().replace('\n', "");
    while value.ends_with('.') {
        value.pop();
    }
    value = value.replace("\\!", "");

    value = MATRIX_BEGIN
        .replace_all(&value, "\\begin{pmatrix}")
        .into_owned();
    value = MATRIX_END
        .replace_all(&value, "\\end{pmatrix}")
        .into_owned();
    value = value.replace("bmatrix", "pmatrix");

    value = value
        .replace("tfrac", "frac")
        .replace("dfrac", "frac")
        .replace("\\neq", "\\ne")
        .replace("\\leq", "\\le")
        .replace("\\geq", "\\ge")
        .replace("\\left", "")
        .replace("\\right", "")
        .replace("\\{", "{")
        .replace("\\}", "}");

    let without_trailing_text = TRAILING_TEXT.replace(&value, "").trim().to_string();
    if !without_trailing_text.is_empty() && without_trailing_text != value {
        value = without_trailing_text;
    }
    for pattern in UNIT_PATTERNS.iter() {
        let candidate = pattern.replace_all(&value, "$1$2").into_owned();
        if !candidate.is_empty() {
            value = candidate;
        }
    }

    value = value
        .replace("^{\\circ}", "")
        .replace("^\\circ", "")
        .replace("\\$", "")
        .replace('$', "")
        .replace("\\(", "")
        .replace("\\)", "");

    value = TEXT_COMMAND.replace_all(&value, "$1").into_owned();
    for assignment in [
        "x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to",
    ] {
        value = value.replace(assignment, "");
    }
    value = value
        .replace("\\emptyset", "{}")
        .replace("(-\\infty,\\infty)", "\\mathbb{R}")
        .replace("\\%", "")
        .replace('%', "");

    value = MONTH.replace_all(&value, "").into_owned();
    value = value.replace(" .", " 0.").replace("{.", "{0.");

    value = value.replace("infinity", "\\infty");
    if !value.contains("\\infty") {
        value = value.replace("inf", "\\infty");
    }
    value = value
        .replace("+\\inity", "\\infty")
        .replace("and", "")
        .replace("\\mathbf", "");
    value = MBOX_COMMAND.replace_all(&value, "").into_owned();
    if value.contains('j') && !value.contains('i') {
        value = value.replace('j', "i");
    }

    value = NUMERIC_ZERO_WITH_SUFFIX
        .replace_all(&value, "$1$2")
        .into_owned();
    value = NUMERIC_ZERO_AT_END.replace(&value, "$1").into_owned();
    if value.is_empty() {
        return value;
    }
    if value.starts_with('.') {
        value.insert(0, '0');
    }
    if let Some((lhs, rhs)) = value.split_once('=')
        && !rhs.contains('=')
        && lhs.len() <= 2
    {
        value = rhs.to_string();
    }

    value = fix_sqrt(&value);
    value = value.replace(' ', "");
    value = fix_fracs(&value);
    value = fix_simple_slash(&value);
    value
}

fn fix_sqrt(input: &str) -> String {
    static SQRT_WORD: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\\sqrt(\w+)").expect("static sqrt regex"));
    SQRT_WORD.replace_all(input, "\\sqrt{$1}").into_owned()
}

fn fix_fracs(input: &str) -> String {
    let mut parts = input.split("\\frac");
    let mut output = parts.next().unwrap_or_default().to_string();
    for part in parts {
        output.push_str("\\frac");
        if part.starts_with('{') {
            output.push_str(part);
            continue;
        }
        let mut characters = part.char_indices();
        let Some((_, first)) = characters.next() else {
            return input.to_string();
        };
        let Some((second_offset, second)) = characters.next() else {
            return input.to_string();
        };
        let remainder_offset = second_offset + second.len_utf8();
        output.push('{');
        output.push(first);
        output.push('}');
        if second == '{' {
            output.push('{');
            output.push_str(&part[remainder_offset..]);
        } else {
            output.push('{');
            output.push(second);
            output.push('}');
            output.push_str(&part[remainder_offset..]);
        }
    }
    output
}

fn fix_simple_slash(input: &str) -> String {
    let Some((numerator, denominator)) = input.split_once('/') else {
        return input.to_string();
    };
    if denominator.contains('/') {
        return input.to_string();
    }
    if (numerator.parse::<i128>().is_ok() || numerator.contains("sqrt"))
        && (denominator.parse::<i128>().is_ok() || denominator.contains("sqrt"))
    {
        format!("\\frac{{{numerator}}}{{{denominator}}}")
    } else {
        input.to_string()
    }
}

fn math_equal(prediction: &str, reference: &str, choice: &Regex, depth: usize) -> bool {
    if depth > 5 {
        return false;
    }
    if prediction.trim().eq_ignore_ascii_case(reference.trim()) {
        return true;
    }
    if matches!(reference, "A" | "B" | "C" | "D" | "E")
        && choice
            .captures_iter(prediction)
            .last()
            .and_then(|captures| captures.get(1))
            .is_some_and(|letter| letter.as_str().eq_ignore_ascii_case(reference))
    {
        return true;
    }
    for prefix in [
        "(A)", "(B)", "(C)", "(D)", "(E)", "A.", "B.", "C.", "D.", "E.", "A)", "B)", "C)", "D)",
        "E)", "**A**", "**B**", "**C**", "**D**", "**E**", "A:", "B:", "C:", "D:", "E:",
    ] {
        if let Some(stripped) = prediction.strip_prefix(prefix)
            && math_equal(stripped.trim(), reference, choice, depth + 1)
        {
            return true;
        }
    }
    if prediction.contains(',') && reference.contains(',') {
        let mut left = prediction.split(',').map(str::trim).collect::<Vec<_>>();
        let mut right = reference.split(',').map(str::trim).collect::<Vec<_>>();
        left.sort_unstable();
        right.sort_unstable();
        if left.len() == right.len()
            && left
                .iter()
                .zip(right)
                .all(|(left, right)| math_equal(left, right, choice, depth + 1))
        {
            return true;
        }
    }
    if let (Some(left), Some(right)) = (parse_number(prediction), parse_number(reference)) {
        return [right / 100.0, right, right * 100.0]
            .into_iter()
            .any(|right| close(left, right));
    }
    if prediction.is_empty() {
        return false;
    }
    if strip_brackets(prediction).eq_ignore_ascii_case(&strip_brackets(reference)) {
        return true;
    }
    if let (Some((left_lhs, left_rhs)), Some((right_lhs, right_rhs))) =
        (single_equation(prediction), single_equation(reference))
    {
        let left = format!("({left_lhs})-({left_rhs})");
        let right = format!("({right_lhs})-({right_rhs})");
        if expression_equivalent(&left, &right) {
            return true;
        }
    }
    if let Some((lhs, rhs)) = single_equation(prediction)
        && lhs.trim().len() <= 2
        && !reference.contains('=')
    {
        return math_equal(rhs, reference, choice, depth + 1);
    }
    if let Some((lhs, rhs)) = single_equation(reference)
        && lhs.trim().len() <= 2
        && !prediction.contains('=')
    {
        return math_equal(prediction, rhs, choice, depth + 1);
    }
    expression_equivalent(prediction, reference)
}

fn parse_number(input: &str) -> Option<f64> {
    let input = input.replace(',', "");
    if let Ok(value) = input.parse::<f64>() {
        return Some(value);
    }
    if let Some(percent) = input
        .strip_suffix('%')
        .or_else(|| input.strip_suffix("\\%"))
    {
        return percent.parse::<f64>().ok().map(|value| value / 100.0);
    }
    if let Some((numerator, denominator)) = parse_latex_fraction(&input) {
        let numerator = parse_number(numerator)?;
        let denominator = parse_number(denominator)?;
        return (denominator != 0.0).then_some(numerator / denominator);
    }
    None
}

fn parse_latex_fraction(input: &str) -> Option<(&str, &str)> {
    let rest = input.strip_prefix("\\frac{")?;
    let numerator_end = matching_brace_end(rest, 0)?;
    let numerator = &rest[..numerator_end];
    let denominator_start = numerator_end + 1;
    let rest = rest.get(denominator_start..)?.strip_prefix('{')?;
    let denominator_end = matching_brace_end(rest, 0)?;
    (denominator_end + 1 == rest.len()).then_some((numerator, &rest[..denominator_end]))
}

fn strip_brackets(input: &str) -> String {
    input
        .chars()
        .filter(|character| !matches!(character, '{' | '}' | '(' | ')' | '[' | ']'))
        .collect()
}

fn single_equation(input: &str) -> Option<(&str, &str)> {
    let (left, right) = input.split_once('=')?;
    (!right.contains('=')).then_some((left, right))
}

fn close(left: f64, right: f64) -> bool {
    left.is_finite() && right.is_finite() && (left - right).abs() <= ABS_TOLERANCE
}

fn expression_equivalent(left: &str, right: &str) -> bool {
    let left_expression = latex_to_expression(left);
    let right_expression = latex_to_expression(right);
    let variables = expression_variables(&left_expression)
        .union(&expression_variables(&right_expression))
        .cloned()
        .collect::<Vec<_>>();

    if variables.is_empty() {
        if let (Some(left), Some(right)) = (
            evaluate_expression(&left_expression, &BTreeMap::new()),
            evaluate_expression(&right_expression, &BTreeMap::new()),
        ) {
            return symbolic_close(left, right);
        }
    } else {
        // A native deterministic identity check replaces the Python/SymPy
        // subprocess. Multiple independent points make algebraic identities
        // such as `(x+1)^2 == x^2+2x+1` work while avoiding a single-point
        // false positive. Undefined points are skipped symmetrically.
        const SAMPLES: &[f64] = &[
            0.37, 1.13, -0.71, 2.07, -1.89, 3.31, 0.83, -2.47, 4.19, -3.73, 1.61, 5.03,
        ];
        let mut compared = 0usize;
        for (sample_index, base) in SAMPLES.iter().copied().enumerate() {
            let environment = variables
                .iter()
                .enumerate()
                .map(|(variable_index, variable)| {
                    let hash_offset = stable_name_offset(variable);
                    let value = base
                        + (variable_index as f64 + 1.0) * 0.173
                        + hash_offset
                        + sample_index as f64 * 0.011;
                    (variable.clone(), value)
                })
                .collect::<BTreeMap<_, _>>();
            match (
                evaluate_expression(&left_expression, &environment),
                evaluate_expression(&right_expression, &environment),
            ) {
                (Some(left), Some(right)) if symbolic_close(left, right) => compared += 1,
                (Some(_), Some(_)) => return false,
                _ => {}
            }
        }
        if compared >= 6 {
            return true;
        }
    }

    canonical_expression(&left_expression) == canonical_expression(&right_expression)
}

fn canonical_expression(input: &str) -> String {
    input
        .chars()
        .filter(|character| !character.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect()
}

fn symbolic_close(left: f64, right: f64) -> bool {
    left.is_finite()
        && right.is_finite()
        && (left - right).abs() <= SYMBOLIC_TOLERANCE * left.abs().max(right.abs()).max(1.0)
}

fn stable_name_offset(name: &str) -> f64 {
    let hash = name.bytes().fold(0xcbf29ce484222325_u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    });
    (hash % 997) as f64 / 9_970.0
}

fn expression_variables(input: &str) -> BTreeSet<String> {
    let mut variables = BTreeSet::new();
    let bytes = input.as_bytes();
    let mut cursor = 0usize;
    while cursor < bytes.len() {
        if bytes[cursor].is_ascii_alphabetic() || bytes[cursor] == b'_' {
            let start = cursor;
            cursor += 1;
            while cursor < bytes.len()
                && (bytes[cursor].is_ascii_alphanumeric() || bytes[cursor] == b'_')
            {
                cursor += 1;
            }
            let identifier = &input[start..cursor];
            if !is_function(identifier) && !matches!(identifier, "pi" | "e") {
                variables.insert(identifier.to_string());
            }
        } else {
            cursor += 1;
        }
    }
    variables
}

fn evaluate_expression(input: &str, variables: &BTreeMap<String, f64>) -> Option<f64> {
    let mut parser = ExpressionParser::new(input, variables);
    let value = parser.expression()?;
    parser.skip_whitespace();
    (parser.cursor == parser.input.len() && value.is_finite()).then_some(value)
}

fn latex_to_expression(input: &str) -> String {
    let mut value = input.to_string();
    for _ in 0..16 {
        let expanded_fraction = expand_command(&value, "\\frac", |parts| {
            format!("(({})/({}))", parts[0], parts[1])
        });
        let expanded_sqrt = expand_command(&expanded_fraction, "\\sqrt", |parts| {
            format!("sqrt({})", parts[0])
        });
        if expanded_sqrt == value {
            break;
        }
        value = expanded_sqrt;
    }
    for command in ["\\operatorname", "\\mathrm", "\\mathit", "\\text"] {
        value = expand_command(&value, command, |parts| parts[0].clone());
    }
    value = value
        .replace("\\cdot", "*")
        .replace("\\times", "*")
        .replace("\\pi", "pi")
        .replace("\\sin", "sin")
        .replace("\\cos", "cos")
        .replace("\\tan", "tan")
        .replace("\\arcsin", "asin")
        .replace("\\arccos", "acos")
        .replace("\\arctan", "atan")
        .replace("\\ln", "ln")
        .replace("\\log", "log")
        .replace("\\exp", "exp")
        .replace("^{", "^(")
        .replace('{', "(")
        .replace('}', ")");
    // Remaining alphabetic LaTeX commands are symbols (`\theta` ->
    // `theta`) for the deterministic native evaluator.
    let mut output = String::with_capacity(value.len());
    let mut characters = value.chars().peekable();
    while let Some(character) = characters.next() {
        if character == '\\'
            && characters
                .peek()
                .is_some_and(|next| next.is_ascii_alphabetic())
        {
            continue;
        }
        output.push(character);
    }
    output
}

fn expand_command(input: &str, command: &str, render: impl Fn(&[String]) -> String) -> String {
    let arity = if command == "\\frac" { 2 } else { 1 };
    let mut output = String::new();
    let mut cursor = 0;
    while let Some(relative) = input[cursor..].find(command) {
        let start = cursor + relative;
        output.push_str(&input[cursor..start]);
        let mut part_start = start + command.len();
        let mut parts = Vec::with_capacity(arity);
        let mut valid = true;
        for _ in 0..arity {
            if input.as_bytes().get(part_start) != Some(&b'{') {
                valid = false;
                break;
            }
            let content_start = part_start + 1;
            let Some(end) = matching_brace_end(input, content_start) else {
                valid = false;
                break;
            };
            parts.push(input[content_start..end].to_string());
            part_start = end + 1;
        }
        if !valid {
            output.push_str(command);
            cursor = start + command.len();
            continue;
        }
        output.push_str(&render(&parts));
        cursor = part_start;
    }
    output.push_str(&input[cursor..]);
    output
}

fn matching_brace_end(input: &str, content_start: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (relative, character) in input[content_start..].char_indices() {
        match character {
            '{' => depth += 1,
            '}' if depth == 0 => return Some(content_start + relative),
            '}' => depth -= 1,
            _ => {}
        }
    }
    None
}

struct ExpressionParser<'a> {
    input: &'a [u8],
    cursor: usize,
    variables: &'a BTreeMap<String, f64>,
}

impl<'a> ExpressionParser<'a> {
    fn new(input: &'a str, variables: &'a BTreeMap<String, f64>) -> Self {
        Self {
            input: input.as_bytes(),
            cursor: 0,
            variables,
        }
    }

    fn expression(&mut self) -> Option<f64> {
        let mut value = self.term()?;
        loop {
            self.skip_whitespace();
            match self.peek() {
                Some(b'+') => {
                    self.cursor += 1;
                    value += self.term()?;
                }
                Some(b'-') => {
                    self.cursor += 1;
                    value -= self.term()?;
                }
                _ => return Some(value),
            }
        }
    }

    fn term(&mut self) -> Option<f64> {
        let mut value = self.unary()?;
        loop {
            self.skip_whitespace();
            match self.peek() {
                Some(b'*') => {
                    self.cursor += 1;
                    value *= self.unary()?;
                }
                Some(b'/') => {
                    self.cursor += 1;
                    let denominator = self.unary()?;
                    if denominator == 0.0 {
                        return None;
                    }
                    value /= denominator;
                }
                Some(byte) if starts_implicit_factor(byte) => value *= self.unary()?,
                _ => return Some(value),
            }
        }
    }

    fn power(&mut self) -> Option<f64> {
        let mut value = self.atom()?;
        while self.peek() == Some(b'!') {
            self.cursor += 1;
            value = factorial(value)?;
            self.skip_whitespace();
        }
        self.skip_whitespace();
        if self.peek() == Some(b'^') {
            self.cursor += 1;
            Some(value.powf(self.unary()?))
        } else {
            Some(value)
        }
    }

    fn unary(&mut self) -> Option<f64> {
        self.skip_whitespace();
        match self.peek() {
            Some(b'+') => {
                self.cursor += 1;
                self.unary()
            }
            Some(b'-') => {
                self.cursor += 1;
                self.unary().map(|value| -value)
            }
            _ => self.power(),
        }
    }

    fn atom(&mut self) -> Option<f64> {
        self.skip_whitespace();
        if self.peek() == Some(b'(') {
            self.cursor += 1;
            let value = self.expression()?;
            self.skip_whitespace();
            (self.peek() == Some(b')')).then(|| self.cursor += 1)?;
            return Some(value);
        }
        let start = self.cursor;
        if self
            .peek()
            .is_some_and(|byte| byte.is_ascii_digit() || byte == b'.')
        {
            while self.peek().is_some_and(|byte| {
                byte.is_ascii_digit() || matches!(byte, b'.' | b'e' | b'E' | b'+' | b'-')
            }) {
                if matches!(self.peek(), Some(b'+' | b'-'))
                    && !matches!(
                        self.input.get(self.cursor.wrapping_sub(1)),
                        Some(b'e' | b'E')
                    )
                {
                    break;
                }
                self.cursor += 1;
            }
            return std::str::from_utf8(&self.input[start..self.cursor])
                .ok()?
                .parse()
                .ok();
        }

        if !self
            .peek()
            .is_some_and(|byte| byte.is_ascii_alphabetic() || byte == b'_')
        {
            return None;
        }
        self.cursor += 1;
        while self
            .peek()
            .is_some_and(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            self.cursor += 1;
        }
        let identifier = std::str::from_utf8(&self.input[start..self.cursor]).ok()?;
        if identifier == "pi" {
            return Some(std::f64::consts::PI);
        }
        if identifier == "e" {
            return Some(std::f64::consts::E);
        }
        if is_function(identifier) {
            self.skip_whitespace();
            let argument = if self.peek() == Some(b'(') {
                self.cursor += 1;
                let value = self.expression()?;
                self.skip_whitespace();
                if self.peek() != Some(b')') {
                    return None;
                }
                self.cursor += 1;
                value
            } else {
                self.unary()?
            };
            return apply_function(identifier, argument);
        }
        self.variables.get(identifier).copied()
    }

    fn skip_whitespace(&mut self) {
        while self.peek().is_some_and(|byte| byte.is_ascii_whitespace()) {
            self.cursor += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.cursor).copied()
    }
}

fn starts_implicit_factor(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || byte.is_ascii_digit() || matches!(byte, b'_' | b'.' | b'(')
}

fn is_function(identifier: &str) -> bool {
    matches!(
        identifier,
        "sqrt"
            | "sin"
            | "cos"
            | "tan"
            | "asin"
            | "acos"
            | "atan"
            | "ln"
            | "log"
            | "exp"
            | "abs"
            | "floor"
            | "ceil"
    )
}

fn apply_function(identifier: &str, argument: f64) -> Option<f64> {
    let value = match identifier {
        "sqrt" if argument >= 0.0 => argument.sqrt(),
        "sin" => argument.sin(),
        "cos" => argument.cos(),
        "tan" => argument.tan(),
        "asin" => argument.asin(),
        "acos" => argument.acos(),
        "atan" => argument.atan(),
        "ln" => argument.ln(),
        "log" => argument.log10(),
        "exp" => argument.exp(),
        "abs" => argument.abs(),
        "floor" => argument.floor(),
        "ceil" => argument.ceil(),
        _ => return None,
    };
    value.is_finite().then_some(value)
}

fn factorial(value: f64) -> Option<f64> {
    let rounded = value.round();
    if value < 0.0 || value > 170.0 || (value - rounded).abs() > f64::EPSILON {
        return None;
    }
    Some((1..=rounded as u64).fold(1.0, |product, factor| product * factor as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balanced_box_and_latex_expression_equivalence() {
        assert_eq!(
            extract_last_boxed("x \\boxed{1} then \\boxed{\\frac{1}{2}}"),
            Some("\\frac{1}{2}".to_string())
        );
        let grader = MathGrader::new();
        assert!(grader.equivalent("\\frac{1}{2}", "0.5"));
        assert!(grader.equivalent("\\sqrt{2}", "2^{1/2}"));
        assert!(grader.equivalent("(x+1)^2", "x^2+2x+1"));
        assert!(grader.equivalent("x+x", "2x"));
        assert!(grader.equivalent("2(x+1)", "2x+2"));
        assert!(!grader.equivalent("x^2", "x^3"));
        assert!(!grader.equivalent("24", "25"));
    }

    #[test]
    fn strip_string_matches_inherited_recipe_edge_cases() {
        // Expected strings were generated by `_math_strip.py:234-385`.
        for (input, expected) in [
            ("5 mph", "5"),
            (
                "\\begin{array}{cc}1&2\\\\3&4\\end{array}",
                "\\begin{pmatrix}1&2\\\\3&4\\end{pmatrix}",
            ),
            ("42\\text{ meters}", "42"),
            ("January 5", "5"),
            ("\\mbox{units}42", "42"),
            ("\\sqrt12", "\\sqrt{12}"),
            ("\\frac1{2}", "\\frac{1}{2}"),
            ("3.000x", "3x"),
            ("{.5}", "{0.5}"),
            ("x=42", "42"),
            ("7 and 8", "78"),
            ("\\mathbf{x}", "{x}"),
            ("5 inches", "5"),
        ] {
            assert_eq!(strip_math_string(input), expected, "input {input:?}");
        }
    }

    #[tokio::test]
    async fn extraction_priority_and_fallback_are_retained() {
        let grader = MathGrader::new();
        let boxed = grader
            .grade("first 99, then \\boxed{42}", "42")
            .await
            .unwrap();
        assert!(boxed.correct);
        assert!(!boxed.unparsed);
        let phrase = grader
            .grade("the answer is 5. Wait, the answer is 12", "12")
            .await
            .unwrap();
        assert!(phrase.correct);
        assert!(phrase.unparsed);
        let fraction = grader
            .grade("The answer is \\frac{1}{2}", "1/2")
            .await
            .unwrap();
        assert!(fraction.correct);
    }
}
