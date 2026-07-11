// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native math answer extraction, normalization, and equivalence.
//!
//! Source-grounded in `src/aiperf/accuracy/graders/math.py:1-623` and
//! `_math_strip.py:1-385`. The native evaluator keeps the reference strategy
//! order and implements constant arithmetic/LaTeX equivalence without a Python
//! sympy process.

use aiperf_metrics::GradingResult;
use async_trait::async_trait;
use std::sync::LazyLock;

use regex::Regex;

use super::Grader;
use crate::AccuracyError;

const BOXED: &str = "\\boxed{";
const ABS_TOLERANCE: f64 = 1e-4;

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
        .flat_map(|unit| [unit.to_string(), format!("{unit}s")])
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
    match (evaluate_expression(left), evaluate_expression(right)) {
        (Some(left), Some(right)) => close(left, right),
        _ => canonical_expression(left) == canonical_expression(right),
    }
}

fn canonical_expression(input: &str) -> String {
    latex_to_expression(input)
        .chars()
        .filter(|character| !character.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect()
}

fn evaluate_expression(input: &str) -> Option<f64> {
    let expression = latex_to_expression(input);
    let mut parser = ExpressionParser::new(&expression);
    let value = parser.expression()?;
    parser.skip_whitespace();
    (parser.cursor == parser.input.len() && value.is_finite()).then_some(value)
}

fn latex_to_expression(input: &str) -> String {
    let mut value = expand_command(input, "\\frac", |parts| {
        format!("(({})/({}))", parts[0], parts[1])
    });
    value = expand_command(&value, "\\sqrt", |parts| format!("sqrt({})", parts[0]));
    value = value
        .replace("\\cdot", "*")
        .replace("\\times", "*")
        .replace("\\pi", "pi")
        .replace("^{", "^(")
        .replace('{', "(")
        .replace('}', ")");
    value
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
}

impl<'a> ExpressionParser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input: input.as_bytes(),
            cursor: 0,
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
        let mut value = self.power()?;
        loop {
            self.skip_whitespace();
            match self.peek() {
                Some(b'*') => {
                    self.cursor += 1;
                    value *= self.power()?;
                }
                Some(b'/') => {
                    self.cursor += 1;
                    let denominator = self.power()?;
                    if denominator == 0.0 {
                        return None;
                    }
                    value /= denominator;
                }
                _ => return Some(value),
            }
        }
    }

    fn power(&mut self) -> Option<f64> {
        let value = self.unary()?;
        self.skip_whitespace();
        if self.peek() == Some(b'^') {
            self.cursor += 1;
            Some(value.powf(self.power()?))
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
            _ => self.atom(),
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
        if self.remaining().starts_with(b"sqrt") {
            self.cursor += 4;
            self.skip_whitespace();
            if self.peek() != Some(b'(') {
                return None;
            }
            self.cursor += 1;
            let value = self.expression()?;
            self.skip_whitespace();
            if self.peek() != Some(b')') || value < 0.0 {
                return None;
            }
            self.cursor += 1;
            return Some(value.sqrt());
        }
        if self.remaining().starts_with(b"pi") {
            self.cursor += 2;
            return Some(std::f64::consts::PI);
        }
        let start = self.cursor;
        while self
            .peek()
            .is_some_and(|byte| byte.is_ascii_digit() || byte == b'.')
        {
            self.cursor += 1;
        }
        if self.cursor == start {
            return None;
        }
        std::str::from_utf8(&self.input[start..self.cursor])
            .ok()?
            .parse()
            .ok()
    }

    fn skip_whitespace(&mut self) {
        while self.peek().is_some_and(|byte| byte.is_ascii_whitespace()) {
            self.cursor += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.cursor).copied()
    }

    fn remaining(&self) -> &[u8] {
        &self.input[self.cursor..]
    }
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
        assert!(!grader.equivalent("24", "25"));
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
