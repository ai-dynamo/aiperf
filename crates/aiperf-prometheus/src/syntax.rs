// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded format-specific lexical decoding before semantic role assembly.

use std::collections::BTreeMap;

use crate::error::{LimitKind, ParseError, ParseErrorKind};
use crate::format::ExpositionFormat;
use crate::limits::ParseLimits;
use crate::model::{Exemplar, LabelSet};
use crate::number::{ExactNumber, SourceTimestamp, parse_number_lexeme};

#[derive(Debug)]
pub(crate) struct ParsedDocument {
    pub(crate) descriptors: Vec<ParsedDescriptor>,
    pub(crate) samples: Vec<ParsedSample>,
}

#[derive(Debug)]
pub(crate) struct ParsedDescriptor {
    pub(crate) line: usize,
    pub(crate) family: String,
    pub(crate) kind: DescriptorKind,
}

#[derive(Debug)]
pub(crate) enum DescriptorKind {
    Help(String),
    Type(String),
    Unit(String),
}

#[derive(Debug)]
pub(crate) struct ParsedSample {
    pub(crate) line: usize,
    pub(crate) emitted_name: String,
    pub(crate) labels: LabelSet,
    pub(crate) value: ExactNumber,
    pub(crate) source_timestamp: SourceTimestamp,
    pub(crate) exemplar: Option<Exemplar>,
}

pub(crate) fn parse_document(
    format: ExpositionFormat,
    exact_body: &[u8],
    limits: &ParseLimits,
) -> Result<ParsedDocument, ParseError> {
    if exact_body.len() > limits.max_decoded_bytes {
        return Err(ParseError::body(
            ParseErrorKind::LimitExceeded(LimitKind::DecodedBytes),
            "decoded exposition exceeds max_decoded_bytes",
        ));
    }
    let body = std::str::from_utf8(exact_body).map_err(|error| {
        ParseError::line(
            0,
            error.valid_up_to().saturating_add(1),
            ParseErrorKind::InvalidUtf8,
            "decoded exposition is not valid UTF-8",
        )
    })?;
    if body.starts_with('\u{feff}') {
        return Err(ParseError::line(
            1,
            1,
            ParseErrorKind::Syntax,
            "UTF-8 byte-order marks are not permitted",
        ));
    }
    if let Some(index) = exact_body.iter().position(|byte| *byte == b'\r') {
        let (line, column) = byte_location(exact_body, index);
        return Err(ParseError::line(
            line,
            column,
            ParseErrorKind::Syntax,
            "carriage returns are not valid exposition line endings",
        ));
    }
    if format == ExpositionFormat::PrometheusText004 && !body.is_empty() && !body.ends_with('\n') {
        return Err(ParseError::body(
            ParseErrorKind::Syntax,
            "Prometheus text 0.0.4 must end with a line feed",
        ));
    }

    let physical_line_count = if body.is_empty() {
        0
    } else {
        body.bytes().filter(|byte| *byte == b'\n').count() + usize::from(!body.ends_with('\n'))
    };
    if physical_line_count > limits.max_lines {
        return Err(ParseError::body(
            ParseErrorKind::LimitExceeded(LimitKind::Lines),
            "exposition exceeds max_lines",
        ));
    }
    let without_optional_final_lf = body.strip_suffix('\n').unwrap_or(body);
    let lines = if without_optional_final_lf.is_empty() {
        Vec::new()
    } else {
        without_optional_final_lf.split('\n').collect::<Vec<_>>()
    };
    for (index, line) in lines.iter().enumerate() {
        if line.len() > limits.max_line_bytes {
            return Err(ParseError::line(
                index + 1,
                0,
                ParseErrorKind::LimitExceeded(LimitKind::LineBytes),
                "physical line exceeds max_line_bytes",
            ));
        }
    }

    let content_lines = match format {
        ExpositionFormat::PrometheusText004 => lines.as_slice(),
        ExpositionFormat::OpenMetricsText100 => {
            let Some((last, preceding)) = lines.split_last() else {
                return Err(ParseError::body(
                    ParseErrorKind::EndOfFile,
                    "OpenMetrics exposition is missing terminal # EOF",
                ));
            };
            if *last != "# EOF" {
                return Err(ParseError::line(
                    lines.len(),
                    1,
                    ParseErrorKind::EndOfFile,
                    "OpenMetrics terminal line must be exactly # EOF",
                ));
            }
            preceding
        }
    };

    let mut descriptors = Vec::new();
    let mut samples = Vec::new();
    let mut exemplar_count = 0_usize;
    for (index, line) in content_lines.iter().enumerate() {
        let line_number = index + 1;
        if line.is_empty() {
            if format == ExpositionFormat::PrometheusText004 {
                continue;
            }
            return Err(ParseError::line(
                line_number,
                1,
                ParseErrorKind::Syntax,
                "OpenMetrics does not permit blank lines",
            ));
        }
        if line.starts_with('#') {
            match parse_descriptor(format, line_number, line, limits)? {
                Some(descriptor) => descriptors.push(descriptor),
                None if format == ExpositionFormat::PrometheusText004 => {}
                None => {
                    return Err(ParseError::line(
                        line_number,
                        1,
                        ParseErrorKind::Metadata,
                        "OpenMetrics permits only HELP, TYPE, UNIT, and terminal EOF directives",
                    ));
                }
            }
            continue;
        }
        let sample = parse_sample(format, line_number, line, limits)?;
        if sample.exemplar.is_some() {
            exemplar_count += 1;
            if exemplar_count > limits.max_exemplars {
                return Err(ParseError::line(
                    line_number,
                    1,
                    ParseErrorKind::LimitExceeded(LimitKind::Exemplars),
                    "exposition exceeds max_exemplars",
                ));
            }
        }
        samples.push(sample);
        if samples.len() > limits.max_wire_samples {
            return Err(ParseError::line(
                line_number,
                1,
                ParseErrorKind::LimitExceeded(LimitKind::WireSamples),
                "exposition exceeds max_wire_samples",
            ));
        }
    }
    Ok(ParsedDocument {
        descriptors,
        samples,
    })
}

fn parse_descriptor(
    format: ExpositionFormat,
    line_number: usize,
    line: &str,
    limits: &ParseLimits,
) -> Result<Option<ParsedDescriptor>, ParseError> {
    let (kind, rest) = if let Some(rest) = line.strip_prefix("# HELP ") {
        ("HELP", rest)
    } else if let Some(rest) = line.strip_prefix("# TYPE ") {
        ("TYPE", rest)
    } else if format == ExpositionFormat::OpenMetricsText100 {
        if let Some(rest) = line.strip_prefix("# UNIT ") {
            ("UNIT", rest)
        } else {
            if line.starts_with("# HELP")
                || line.starts_with("# TYPE")
                || line.starts_with("# UNIT")
                || line.starts_with("# EOF")
            {
                return Err(ParseError::line(
                    line_number,
                    1,
                    ParseErrorKind::Metadata,
                    "malformed OpenMetrics metadata directive",
                ));
            }
            return Ok(None);
        }
    } else {
        if line.starts_with("# HELP") || line.starts_with("# TYPE") {
            return Err(ParseError::line(
                line_number,
                1,
                ParseErrorKind::Metadata,
                "malformed Prometheus metadata directive",
            ));
        }
        return Ok(None);
    };

    let Some(separator) = rest.find(' ') else {
        return Err(ParseError::line(
            line_number,
            1,
            ParseErrorKind::Metadata,
            format!("{kind} directive is missing its value separator"),
        ));
    };
    let family = &rest[..separator];
    let raw_value = &rest[separator + 1..];
    validate_metric_name(line_number, family, limits)?;
    if raw_value.len() > limits.max_metadata_value_bytes {
        return Err(ParseError::line(
            line_number,
            separator + 2,
            ParseErrorKind::LimitExceeded(LimitKind::MetadataValueBytes),
            "metadata value exceeds max_metadata_value_bytes",
        ));
    }
    let descriptor_kind = match kind {
        "HELP" => DescriptorKind::Help(decode_escaped(
            format,
            line_number,
            separator + 2,
            raw_value,
        )?),
        "TYPE" => {
            if raw_value.is_empty() || raw_value.bytes().any(is_horizontal_space) {
                return Err(ParseError::line(
                    line_number,
                    separator + 2,
                    ParseErrorKind::Metadata,
                    "TYPE must contain exactly one type token",
                ));
            }
            let valid = match format {
                ExpositionFormat::PrometheusText004 => {
                    matches!(
                        raw_value,
                        "counter" | "gauge" | "histogram" | "summary" | "untyped"
                    )
                }
                ExpositionFormat::OpenMetricsText100 => matches!(
                    raw_value,
                    "unknown"
                        | "gauge"
                        | "counter"
                        | "stateset"
                        | "info"
                        | "histogram"
                        | "gaugehistogram"
                        | "summary"
                ),
            };
            if !valid {
                return Err(ParseError::line(
                    line_number,
                    separator + 2,
                    ParseErrorKind::Metadata,
                    format!("unsupported {format} TYPE token {raw_value:?}"),
                ));
            }
            DescriptorKind::Type(raw_value.to_string())
        }
        "UNIT" => {
            if !raw_value.bytes().all(is_metric_name_char) {
                return Err(ParseError::line(
                    line_number,
                    separator + 2,
                    ParseErrorKind::Metadata,
                    "UNIT contains a character outside the metric-name alphabet",
                ));
            }
            DescriptorKind::Unit(raw_value.to_string())
        }
        _ => unreachable!(),
    };
    Ok(Some(ParsedDescriptor {
        line: line_number,
        family: family.to_string(),
        kind: descriptor_kind,
    }))
}

fn parse_sample(
    format: ExpositionFormat,
    line_number: usize,
    line: &str,
    limits: &ParseLimits,
) -> Result<ParsedSample, ParseError> {
    let bytes = line.as_bytes();
    let mut cursor = 0_usize;
    if bytes
        .first()
        .is_none_or(|byte| !is_metric_name_initial(*byte))
    {
        return Err(ParseError::line(
            line_number,
            1,
            ParseErrorKind::Syntax,
            "sample must begin with a valid metric name",
        ));
    }
    cursor += 1;
    while cursor < bytes.len() && is_metric_name_char(bytes[cursor]) {
        cursor += 1;
    }
    let emitted_name = &line[..cursor];
    validate_metric_name(line_number, emitted_name, limits)?;

    let labels = if bytes.get(cursor) == Some(&b'{') {
        parse_label_set(format, line_number, line, &mut cursor, limits, false)?
    } else {
        LabelSet::new()
    };
    if cursor == bytes.len() || !is_horizontal_space(bytes[cursor]) {
        return Err(ParseError::line(
            line_number,
            cursor + 1,
            ParseErrorKind::Syntax,
            "sample name or labels must be followed by horizontal whitespace",
        ));
    }
    skip_horizontal_space(bytes, &mut cursor);
    let value_lexeme = take_token(line, &mut cursor);
    if value_lexeme.is_empty() {
        return Err(ParseError::line(
            line_number,
            cursor + 1,
            ParseErrorKind::Number,
            "sample is missing its value",
        ));
    }
    check_numeric_limit(line_number, value_lexeme, limits)?;
    let value = parse_number_lexeme(format, value_lexeme).map_err(|error| {
        ParseError::line(
            line_number,
            1,
            ParseErrorKind::Number,
            format!("invalid sample value {value_lexeme:?}: {error}"),
        )
    })?;

    let mut source_timestamp = SourceTimestamp::absent();
    let mut exemplar = None;
    if cursor < bytes.len() {
        skip_horizontal_space(bytes, &mut cursor);
        if cursor < bytes.len() && bytes[cursor] != b'#' {
            let timestamp_lexeme = take_token(line, &mut cursor);
            check_numeric_limit(line_number, timestamp_lexeme, limits)?;
            source_timestamp =
                SourceTimestamp::parse(format, timestamp_lexeme).map_err(|error| {
                    ParseError::line(
                        line_number,
                        1,
                        ParseErrorKind::Number,
                        format!("invalid source timestamp {timestamp_lexeme:?}: {error}"),
                    )
                })?;
            if cursor < bytes.len() {
                skip_horizontal_space(bytes, &mut cursor);
            }
        }
        if cursor < bytes.len() {
            if bytes[cursor] != b'#' {
                return Err(ParseError::line(
                    line_number,
                    cursor + 1,
                    ParseErrorKind::Syntax,
                    "unexpected data after sample timestamp",
                ));
            }
            if format != ExpositionFormat::OpenMetricsText100 {
                return Err(ParseError::line(
                    line_number,
                    cursor + 1,
                    ParseErrorKind::UnsupportedFeature,
                    "Prometheus text 0.0.4 does not support exemplars",
                ));
            }
            exemplar = Some(parse_exemplar(line_number, line, &mut cursor, limits)?);
        }
    }
    if cursor != bytes.len() {
        return Err(ParseError::line(
            line_number,
            cursor + 1,
            ParseErrorKind::Syntax,
            "unexpected trailing sample data",
        ));
    }
    Ok(ParsedSample {
        line: line_number,
        emitted_name: emitted_name.to_string(),
        labels,
        value,
        source_timestamp,
        exemplar,
    })
}

fn parse_exemplar(
    line_number: usize,
    line: &str,
    cursor: &mut usize,
    limits: &ParseLimits,
) -> Result<Exemplar, ParseError> {
    let bytes = line.as_bytes();
    *cursor += 1;
    if *cursor == bytes.len() || !is_horizontal_space(bytes[*cursor]) {
        return Err(ParseError::line(
            line_number,
            *cursor + 1,
            ParseErrorKind::Exemplar,
            "exemplar marker must be followed by horizontal whitespace",
        ));
    }
    skip_horizontal_space(bytes, cursor);
    if bytes.get(*cursor) != Some(&b'{') {
        return Err(ParseError::line(
            line_number,
            *cursor + 1,
            ParseErrorKind::Exemplar,
            "exemplar must contain a label set",
        ));
    }
    let labels = parse_label_set(
        ExpositionFormat::OpenMetricsText100,
        line_number,
        line,
        cursor,
        limits,
        true,
    )?;
    let codepoints = labels
        .iter()
        .map(|(name, value)| name.chars().count() + value.chars().count())
        .sum::<usize>();
    if codepoints > limits.max_exemplar_label_codepoints {
        return Err(ParseError::line(
            line_number,
            1,
            ParseErrorKind::LimitExceeded(LimitKind::ExemplarLabelCodepoints),
            "exemplar label set exceeds max_exemplar_label_codepoints",
        ));
    }
    if *cursor == bytes.len() || !is_horizontal_space(bytes[*cursor]) {
        return Err(ParseError::line(
            line_number,
            *cursor + 1,
            ParseErrorKind::Exemplar,
            "exemplar label set must be followed by its value",
        ));
    }
    skip_horizontal_space(bytes, cursor);
    let value_lexeme = take_token(line, cursor);
    check_numeric_limit(line_number, value_lexeme, limits)?;
    let value = parse_number_lexeme(ExpositionFormat::OpenMetricsText100, value_lexeme).map_err(
        |error| {
            ParseError::line(
                line_number,
                1,
                ParseErrorKind::Exemplar,
                format!("invalid exemplar value {value_lexeme:?}: {error}"),
            )
        },
    )?;
    let mut timestamp = SourceTimestamp::absent();
    if *cursor < bytes.len() {
        skip_horizontal_space(bytes, cursor);
        let timestamp_lexeme = take_token(line, cursor);
        check_numeric_limit(line_number, timestamp_lexeme, limits)?;
        timestamp = SourceTimestamp::parse(ExpositionFormat::OpenMetricsText100, timestamp_lexeme)
            .map_err(|error| {
                ParseError::line(
                    line_number,
                    1,
                    ParseErrorKind::Exemplar,
                    format!("invalid exemplar timestamp {timestamp_lexeme:?}: {error}"),
                )
            })?;
    }
    if *cursor != bytes.len() {
        return Err(ParseError::line(
            line_number,
            *cursor + 1,
            ParseErrorKind::Exemplar,
            "unexpected trailing exemplar data",
        ));
    }
    Ok(Exemplar {
        labels,
        value,
        timestamp,
    })
}

fn parse_label_set(
    format: ExpositionFormat,
    line_number: usize,
    line: &str,
    cursor: &mut usize,
    limits: &ParseLimits,
    exemplar: bool,
) -> Result<LabelSet, ParseError> {
    let bytes = line.as_bytes();
    *cursor += 1;
    let mut labels = BTreeMap::new();
    if bytes.get(*cursor) == Some(&b'}') {
        *cursor += 1;
        return Ok(labels);
    }
    loop {
        let start = *cursor;
        if bytes
            .get(*cursor)
            .is_none_or(|byte| !is_label_name_initial(*byte))
        {
            return Err(ParseError::line(
                line_number,
                *cursor + 1,
                ParseErrorKind::Label,
                "label set contains an invalid label name",
            ));
        }
        *cursor += 1;
        while *cursor < bytes.len() && is_label_name_char(bytes[*cursor]) {
            *cursor += 1;
        }
        let name = &line[start..*cursor];
        if name.len() > limits.max_label_name_bytes {
            return Err(ParseError::line(
                line_number,
                start + 1,
                ParseErrorKind::LimitExceeded(LimitKind::LabelNameBytes),
                "label name exceeds max_label_name_bytes",
            ));
        }
        if bytes.get(*cursor) != Some(&b'=') || bytes.get(*cursor + 1) != Some(&b'\"') {
            return Err(ParseError::line(
                line_number,
                *cursor + 1,
                ParseErrorKind::Label,
                "label name must be followed by =\"",
            ));
        }
        *cursor += 2;
        let value = parse_quoted_value(format, line_number, line, cursor)?;
        if value.len() > limits.max_label_value_bytes {
            return Err(ParseError::line(
                line_number,
                start + 1,
                ParseErrorKind::LimitExceeded(LimitKind::LabelValueBytes),
                "decoded label value exceeds max_label_value_bytes",
            ));
        }
        if labels.insert(name.to_string(), value).is_some() {
            return Err(ParseError::line(
                line_number,
                start + 1,
                ParseErrorKind::Label,
                format!("duplicate label name {name:?}"),
            ));
        }
        let label_limit = if exemplar {
            limits.max_exemplar_labels
        } else {
            limits.max_labels_per_sample
        };
        if labels.len() > label_limit {
            return Err(ParseError::line(
                line_number,
                start + 1,
                ParseErrorKind::LimitExceeded(if exemplar {
                    LimitKind::ExemplarLabels
                } else {
                    LimitKind::LabelsPerSample
                }),
                "label set exceeds its configured label-count limit",
            ));
        }
        match bytes.get(*cursor) {
            Some(b',') => *cursor += 1,
            Some(b'}') => {
                *cursor += 1;
                break;
            }
            _ => {
                return Err(ParseError::line(
                    line_number,
                    *cursor + 1,
                    ParseErrorKind::Label,
                    "quoted label must be followed by comma or closing brace",
                ));
            }
        }
    }
    Ok(labels)
}

fn parse_quoted_value(
    format: ExpositionFormat,
    line_number: usize,
    line: &str,
    cursor: &mut usize,
) -> Result<String, ParseError> {
    let bytes = line.as_bytes();
    let mut decoded = String::new();
    while *cursor < bytes.len() {
        match bytes[*cursor] {
            b'\"' => {
                *cursor += 1;
                return Ok(decoded);
            }
            b'\\' => {
                *cursor += 1;
                if *cursor == bytes.len() {
                    return Err(ParseError::line(
                        line_number,
                        *cursor + 1,
                        ParseErrorKind::Label,
                        "label value ends in an incomplete escape",
                    ));
                }
                match bytes[*cursor] {
                    b'n' => {
                        decoded.push('\n');
                        *cursor += 1;
                    }
                    b'\"' => {
                        decoded.push('\"');
                        *cursor += 1;
                    }
                    b'\\' => {
                        decoded.push('\\');
                        *cursor += 1;
                    }
                    _ if format == ExpositionFormat::OpenMetricsText100 => {
                        let character = line[*cursor..].chars().next().expect("valid UTF-8 tail");
                        decoded.push(character);
                        *cursor += character.len_utf8();
                    }
                    _ => {
                        return Err(ParseError::line(
                            line_number,
                            *cursor,
                            ParseErrorKind::Label,
                            "Prometheus label value contains an undefined escape",
                        ));
                    }
                }
            }
            _ => {
                let character = line[*cursor..].chars().next().expect("valid UTF-8 tail");
                decoded.push(character);
                *cursor += character.len_utf8();
            }
        }
    }
    Err(ParseError::line(
        line_number,
        line.len() + 1,
        ParseErrorKind::Label,
        "unterminated quoted label value",
    ))
}

fn decode_escaped(
    format: ExpositionFormat,
    line_number: usize,
    column: usize,
    raw: &str,
) -> Result<String, ParseError> {
    let bytes = raw.as_bytes();
    let mut cursor = 0_usize;
    let mut decoded = String::new();
    while cursor < bytes.len() {
        if bytes[cursor] != b'\\' {
            let character = raw[cursor..].chars().next().expect("valid UTF-8 tail");
            decoded.push(character);
            cursor += character.len_utf8();
            continue;
        }
        cursor += 1;
        if cursor == bytes.len() {
            return Err(ParseError::line(
                line_number,
                column + cursor,
                ParseErrorKind::Metadata,
                "metadata value ends in an incomplete escape",
            ));
        }
        match bytes[cursor] {
            b'n' => decoded.push('\n'),
            b'\\' => decoded.push('\\'),
            b'\"' if format == ExpositionFormat::OpenMetricsText100 => decoded.push('\"'),
            _ if format == ExpositionFormat::OpenMetricsText100 => {
                let character = raw[cursor..].chars().next().expect("valid UTF-8 tail");
                decoded.push(character);
                cursor += character.len_utf8();
                continue;
            }
            _ => {
                return Err(ParseError::line(
                    line_number,
                    column + cursor,
                    ParseErrorKind::Metadata,
                    "metadata value contains an undefined escape",
                ));
            }
        }
        cursor += 1;
    }
    Ok(decoded)
}

fn validate_metric_name(
    line_number: usize,
    name: &str,
    limits: &ParseLimits,
) -> Result<(), ParseError> {
    if name.len() > limits.max_metric_name_bytes {
        return Err(ParseError::line(
            line_number,
            1,
            ParseErrorKind::LimitExceeded(LimitKind::MetricNameBytes),
            "metric name exceeds max_metric_name_bytes",
        ));
    }
    let bytes = name.as_bytes();
    if bytes
        .first()
        .is_none_or(|byte| !is_metric_name_initial(*byte))
        || !bytes[1..].iter().all(|byte| is_metric_name_char(*byte))
    {
        return Err(ParseError::line(
            line_number,
            1,
            ParseErrorKind::Syntax,
            format!("invalid metric name {name:?}"),
        ));
    }
    Ok(())
}

fn check_numeric_limit(
    line_number: usize,
    lexeme: &str,
    limits: &ParseLimits,
) -> Result<(), ParseError> {
    if lexeme.len() > limits.max_numeric_lexeme_bytes {
        return Err(ParseError::line(
            line_number,
            1,
            ParseErrorKind::LimitExceeded(LimitKind::NumericLexemeBytes),
            "numeric lexeme exceeds max_numeric_lexeme_bytes",
        ));
    }
    Ok(())
}

fn take_token<'a>(line: &'a str, cursor: &mut usize) -> &'a str {
    let start = *cursor;
    let bytes = line.as_bytes();
    while *cursor < bytes.len() && !is_horizontal_space(bytes[*cursor]) {
        *cursor += 1;
    }
    &line[start..*cursor]
}

fn skip_horizontal_space(bytes: &[u8], cursor: &mut usize) {
    while *cursor < bytes.len() && is_horizontal_space(bytes[*cursor]) {
        *cursor += 1;
    }
}

fn is_horizontal_space(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t')
}

fn is_metric_name_initial(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || matches!(byte, b'_' | b':')
}

fn is_metric_name_char(byte: u8) -> bool {
    is_metric_name_initial(byte) || byte.is_ascii_digit()
}

fn is_label_name_initial(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || byte == b'_'
}

fn is_label_name_char(byte: u8) -> bool {
    is_label_name_initial(byte) || byte.is_ascii_digit()
}

fn byte_location(body: &[u8], index: usize) -> (usize, usize) {
    let preceding = &body[..index];
    let line = preceding.iter().filter(|byte| **byte == b'\n').count() + 1;
    let column = preceding
        .iter()
        .rposition(|byte| *byte == b'\n')
        .map_or(index + 1, |line_feed| index - line_feed);
    (line, column)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quoted_commas_quotes_backslashes_and_utf8_are_lossless() {
        let document = parse_document(
            ExpositionFormat::OpenMetricsText100,
            "metric{model=\"A, \\\"quoted\\\" \\\\ Ω\",path=\"line\\nnext\"} 1 # {trace=\"α\"} 0.5 1.25\n# EOF\n"
                .as_bytes(),
            &ParseLimits::default(),
        )
        .unwrap();
        let sample = &document.samples[0];
        assert_eq!(sample.labels["model"], "A, \"quoted\" \\ Ω");
        assert_eq!(sample.labels["path"], "line\nnext");
        assert_eq!(sample.exemplar.as_ref().unwrap().labels["trace"], "α");
        assert_eq!(
            sample
                .exemplar
                .as_ref()
                .unwrap()
                .timestamp
                .lexeme
                .as_deref(),
            Some("1.25")
        );
    }

    #[test]
    fn openmetrics_requires_exact_terminal_eof() {
        let error = parse_document(
            ExpositionFormat::OpenMetricsText100,
            b"metric 1\n",
            &ParseLimits::default(),
        )
        .unwrap_err();
        assert_eq!(error.kind, ParseErrorKind::EndOfFile);
    }

    #[test]
    fn malformed_label_is_atomic() {
        let error = parse_document(
            ExpositionFormat::PrometheusText004,
            b"ok 1\nbad{label=\"unterminated} 2\n",
            &ParseLimits::default(),
        )
        .unwrap_err();
        assert_eq!(error.kind, ParseErrorKind::Label);
        assert_eq!(error.line, 2);
    }
}
