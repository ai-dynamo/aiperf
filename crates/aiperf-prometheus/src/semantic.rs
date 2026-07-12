// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Format-specific family roles and lossless structured MetricPoint assembly.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use crate::error::{LimitKind, ParseError, ParseErrorKind};
use crate::format::ExpositionFormat;
use crate::limits::ParseLimits;
use crate::model::{
    CountOrigin, CounterValue, Exposition, HistogramBucket, HistogramValue,
    InfoLabelPartitionStatus, InfoValue, LabelSet, MetadataLine, Metric, MetricFamily, MetricPoint,
    MetricValue, PointTimeStatus, QuantileValue, SemanticType, StateValue, SummaryValue,
    WireSample, WireSampleRole,
};
use crate::number::{
    CreatedTimestamp, ExactNumber, NumberKind, SourceTimestamp, parse_number_lexeme,
};
use crate::syntax::{DescriptorKind, ParsedDescriptor, ParsedDocument, ParsedSample};

#[derive(Debug)]
struct FamilyDraft {
    name: String,
    source_type_token: String,
    semantic_type: SemanticType,
    help: Option<MetadataLine>,
    type_line: Option<usize>,
    unit: Option<MetadataLine>,
    help_seen: bool,
    type_seen: bool,
    unit_seen: bool,
    family_seq: u64,
    samples: Vec<WireSample>,
}

impl FamilyDraft {
    fn implicit(format: ExpositionFormat, name: String, line: usize) -> Self {
        let source_type_token = match format {
            ExpositionFormat::PrometheusText004 => "untyped",
            ExpositionFormat::OpenMetricsText100 => "unknown",
        };
        Self {
            name,
            source_type_token: source_type_token.to_string(),
            semantic_type: SemanticType::Unknown,
            help: None,
            type_line: None,
            unit: None,
            help_seen: false,
            type_seen: false,
            unit_seen: false,
            family_seq: line as u64,
            samples: Vec::new(),
        }
    }

    fn first_sample_line(&self) -> Option<usize> {
        self.samples.first().map(|sample| sample.line)
    }
}

pub(crate) fn assemble_exposition(
    format: ExpositionFormat,
    document: ParsedDocument,
    limits: &ParseLimits,
) -> Result<Exposition, ParseError> {
    let ParsedDocument {
        descriptors,
        samples,
    } = document;
    let wire_sample_count = samples.len();
    let mut families = Vec::<FamilyDraft>::new();
    let mut family_by_name = BTreeMap::<String, usize>::new();
    for descriptor in descriptors {
        apply_descriptor(
            format,
            descriptor,
            limits,
            &mut families,
            &mut family_by_name,
        )?;
    }

    let mut previous_openmetrics_family = None::<usize>;
    let mut closed_openmetrics_families = BTreeSet::<usize>::new();
    for sample in samples {
        let (family_index, role) =
            resolve_sample_role(format, &sample, limits, &mut families, &mut family_by_name)?;
        if format == ExpositionFormat::OpenMetricsText100
            && previous_openmetrics_family != Some(family_index)
        {
            if closed_openmetrics_families.contains(&family_index) {
                return Err(semantic_error(
                    sample.line,
                    "OpenMetrics metric families must not be interleaved",
                ));
            }
            if let Some(previous) = previous_openmetrics_family.replace(family_index) {
                closed_openmetrics_families.insert(previous);
            }
        }
        validate_exemplar_owner(&sample, role)?;
        families[family_index].samples.push(WireSample {
            line: sample.line,
            emitted_name: sample.emitted_name,
            role,
            labels: sample.labels,
            value: sample.value,
            source_timestamp: sample.source_timestamp,
            exemplar: sample.exemplar,
        });
    }

    if format == ExpositionFormat::OpenMetricsText100 {
        validate_openmetrics_family_name_collisions(&families)?;
    }
    for family in &families {
        validate_family_metadata(format, family)?;
    }

    families.sort_by_key(|family| family.family_seq);
    let mut metric_count = 0_usize;
    let mut point_count = 0_usize;
    let mut output = Vec::with_capacity(families.len());
    for family in families {
        let metrics = build_metrics(format, &family, limits)?;
        metric_count = metric_count.checked_add(metrics.len()).ok_or_else(|| {
            ParseError::body(
                ParseErrorKind::LimitExceeded(LimitKind::Metrics),
                "metric count overflowed",
            )
        })?;
        if metric_count > limits.max_metrics {
            return Err(ParseError::body(
                ParseErrorKind::LimitExceeded(LimitKind::Metrics),
                "exposition exceeds max_metrics",
            ));
        }
        point_count = point_count
            .checked_add(
                metrics
                    .iter()
                    .map(|metric| metric.points.len())
                    .sum::<usize>(),
            )
            .ok_or_else(|| {
                ParseError::body(
                    ParseErrorKind::LimitExceeded(LimitKind::MetricPoints),
                    "metric-point count overflowed",
                )
            })?;
        if point_count > limits.max_metric_points {
            return Err(ParseError::body(
                ParseErrorKind::LimitExceeded(LimitKind::MetricPoints),
                "exposition exceeds max_metric_points",
            ));
        }
        output.push(MetricFamily {
            name: family.name,
            source_type_token: family.source_type_token,
            semantic_type: family.semantic_type,
            help: family.help,
            type_line: family.type_line,
            unit: family.unit,
            metrics,
            family_seq: family.family_seq,
        });
    }
    Ok(Exposition {
        format,
        families: output,
        wire_sample_count,
    })
}

fn apply_descriptor(
    format: ExpositionFormat,
    descriptor: ParsedDescriptor,
    limits: &ParseLimits,
    families: &mut Vec<FamilyDraft>,
    family_by_name: &mut BTreeMap<String, usize>,
) -> Result<(), ParseError> {
    let family_index = if let Some(index) = family_by_name.get(&descriptor.family).copied() {
        index
    } else {
        if families.len() >= limits.max_families {
            return Err(ParseError::line(
                descriptor.line,
                1,
                ParseErrorKind::LimitExceeded(LimitKind::Families),
                "exposition exceeds max_families",
            ));
        }
        let index = families.len();
        family_by_name.insert(descriptor.family.clone(), index);
        families.push(FamilyDraft::implicit(
            format,
            descriptor.family.clone(),
            descriptor.line,
        ));
        index
    };
    let family = &mut families[family_index];
    family.family_seq = family.family_seq.min(descriptor.line as u64);
    match descriptor.kind {
        DescriptorKind::Help(value) => {
            if family.help_seen {
                return Err(metadata_error(
                    descriptor.line,
                    format!("duplicate HELP directive for family {:?}", family.name),
                ));
            }
            family.help_seen = true;
            if !value.is_empty() {
                family.help = Some(MetadataLine {
                    value,
                    line: descriptor.line,
                });
            }
        }
        DescriptorKind::Type(value) => {
            if family.type_seen {
                return Err(metadata_error(
                    descriptor.line,
                    format!("duplicate TYPE directive for family {:?}", family.name),
                ));
            }
            family.type_seen = true;
            family.type_line = Some(descriptor.line);
            family.semantic_type = semantic_type_for_token(&value);
            family.source_type_token = value;
        }
        DescriptorKind::Unit(value) => {
            if family.unit_seen {
                return Err(metadata_error(
                    descriptor.line,
                    format!("duplicate UNIT directive for family {:?}", family.name),
                ));
            }
            family.unit_seen = true;
            if !value.is_empty() {
                family.unit = Some(MetadataLine {
                    value,
                    line: descriptor.line,
                });
            }
        }
    }
    Ok(())
}

fn semantic_type_for_token(token: &str) -> SemanticType {
    match token {
        "unknown" | "untyped" => SemanticType::Unknown,
        "gauge" => SemanticType::Gauge,
        "counter" => SemanticType::Counter,
        "stateset" => SemanticType::StateSet,
        "info" => SemanticType::Info,
        "histogram" => SemanticType::Histogram,
        "gaugehistogram" => SemanticType::GaugeHistogram,
        "summary" => SemanticType::Summary,
        _ => unreachable!("syntax parser admits only known type tokens"),
    }
}

fn resolve_sample_role(
    format: ExpositionFormat,
    sample: &ParsedSample,
    limits: &ParseLimits,
    families: &mut Vec<FamilyDraft>,
    family_by_name: &mut BTreeMap<String, usize>,
) -> Result<(usize, WireSampleRole), ParseError> {
    let candidates = families
        .iter()
        .enumerate()
        .filter_map(|(index, family)| {
            role_for(format, family, &sample.emitted_name).map(|role| (index, role))
        })
        .collect::<Vec<_>>();
    match candidates.as_slice() {
        [(index, role)] => Ok((*index, *role)),
        [] => {
            if family_by_name.contains_key(&sample.emitted_name) {
                return Err(semantic_error(
                    sample.line,
                    format!(
                        "sample {:?} is not a valid role for its declared family type",
                        sample.emitted_name
                    ),
                ));
            }
            if families.len() >= limits.max_families {
                return Err(ParseError::line(
                    sample.line,
                    1,
                    ParseErrorKind::LimitExceeded(LimitKind::Families),
                    "exposition exceeds max_families",
                ));
            }
            let index = families.len();
            family_by_name.insert(sample.emitted_name.clone(), index);
            families.push(FamilyDraft::implicit(
                format,
                sample.emitted_name.clone(),
                sample.line,
            ));
            Ok((index, WireSampleRole::Scalar))
        }
        _ => Err(semantic_error(
            sample.line,
            format!(
                "sample name {:?} is ambiguous across declared metric families",
                sample.emitted_name
            ),
        )),
    }
}

fn role_for(
    format: ExpositionFormat,
    family: &FamilyDraft,
    emitted_name: &str,
) -> Option<WireSampleRole> {
    let name = family.name.as_str();
    match (format, family.semantic_type) {
        (_, SemanticType::Unknown | SemanticType::Gauge) if emitted_name == name => {
            Some(WireSampleRole::Scalar)
        }
        (ExpositionFormat::PrometheusText004, SemanticType::Counter) if emitted_name == name => {
            Some(WireSampleRole::CounterTotal)
        }
        (ExpositionFormat::OpenMetricsText100, SemanticType::Counter) => {
            match emitted_name.strip_prefix(name) {
                Some("_total") => Some(WireSampleRole::CounterTotal),
                Some("_created") => Some(WireSampleRole::CounterCreated),
                _ => None,
            }
        }
        (ExpositionFormat::OpenMetricsText100, SemanticType::StateSet) if emitted_name == name => {
            Some(WireSampleRole::State)
        }
        (ExpositionFormat::OpenMetricsText100, SemanticType::Info)
            if emitted_name == format!("{name}_info") =>
        {
            Some(WireSampleRole::Info)
        }
        (_, SemanticType::Histogram) => match emitted_name.strip_prefix(name) {
            Some("_bucket") => Some(WireSampleRole::HistogramBucket),
            Some("_sum") => Some(WireSampleRole::HistogramSum),
            Some("_count") => Some(WireSampleRole::HistogramCount),
            Some("_created") if format == ExpositionFormat::OpenMetricsText100 => {
                Some(WireSampleRole::HistogramCreated)
            }
            _ => None,
        },
        (ExpositionFormat::OpenMetricsText100, SemanticType::GaugeHistogram) => {
            match emitted_name.strip_prefix(name) {
                Some("_bucket") => Some(WireSampleRole::GaugeHistogramBucket),
                Some("_gsum") => Some(WireSampleRole::GaugeHistogramSum),
                Some("_gcount") => Some(WireSampleRole::GaugeHistogramCount),
                _ => None,
            }
        }
        (_, SemanticType::Summary) => match emitted_name.strip_prefix(name) {
            Some("") => Some(WireSampleRole::SummaryQuantile),
            Some("_sum") => Some(WireSampleRole::SummarySum),
            Some("_count") => Some(WireSampleRole::SummaryCount),
            Some("_created") if format == ExpositionFormat::OpenMetricsText100 => {
                Some(WireSampleRole::SummaryCreated)
            }
            _ => None,
        },
        _ => None,
    }
}

fn validate_exemplar_owner(sample: &ParsedSample, role: WireSampleRole) -> Result<(), ParseError> {
    if sample.exemplar.is_some()
        && !matches!(
            role,
            WireSampleRole::CounterTotal
                | WireSampleRole::HistogramBucket
                | WireSampleRole::GaugeHistogramBucket
        )
    {
        return Err(ParseError::line(
            sample.line,
            1,
            ParseErrorKind::Exemplar,
            format!("role {role:?} cannot own an exemplar"),
        ));
    }
    if let Some(exemplar) = &sample.exemplar
        && (exemplar.value.kind != NumberKind::Finite || exemplar.value.finite_value.is_none())
    {
        return Err(ParseError::line(
            sample.line,
            1,
            ParseErrorKind::Exemplar,
            "exemplar value must have a finite binary64 projection",
        ));
    }
    Ok(())
}

fn validate_openmetrics_family_name_collisions(families: &[FamilyDraft]) -> Result<(), ParseError> {
    let mut owner = BTreeMap::<String, &str>::new();
    for family in families {
        for emitted_name in possible_emitted_names(family) {
            if let Some(previous) = owner.insert(emitted_name.clone(), &family.name)
                && previous != family.name
            {
                return Err(semantic_error(
                    family.family_seq as usize,
                    format!(
                        "OpenMetrics family names {:?} and {:?} collide at emitted sample name {emitted_name:?}",
                        previous, family.name
                    ),
                ));
            }
        }
    }
    Ok(())
}

fn possible_emitted_names(family: &FamilyDraft) -> Vec<String> {
    let name = &family.name;
    match family.semantic_type {
        SemanticType::Unknown | SemanticType::Gauge | SemanticType::StateSet => vec![name.clone()],
        SemanticType::Counter => vec![format!("{name}_total"), format!("{name}_created")],
        SemanticType::Info => vec![format!("{name}_info")],
        SemanticType::Histogram => vec![
            format!("{name}_bucket"),
            format!("{name}_sum"),
            format!("{name}_count"),
            format!("{name}_created"),
        ],
        SemanticType::GaugeHistogram => vec![
            format!("{name}_bucket"),
            format!("{name}_gsum"),
            format!("{name}_gcount"),
        ],
        SemanticType::Summary => vec![
            name.clone(),
            format!("{name}_sum"),
            format!("{name}_count"),
            format!("{name}_created"),
        ],
    }
}

fn validate_family_metadata(
    format: ExpositionFormat,
    family: &FamilyDraft,
) -> Result<(), ParseError> {
    if let Some(first_sample_line) = family.first_sample_line() {
        for metadata_line in [
            family.help.as_ref().map(|line| line.line),
            family.type_line,
            family.unit.as_ref().map(|line| line.line),
        ]
        .into_iter()
        .flatten()
        {
            if metadata_line > first_sample_line {
                return Err(metadata_error(
                    metadata_line,
                    format!(
                        "metadata for family {:?} appears after its first sample",
                        family.name
                    ),
                ));
            }
        }
    }
    if format == ExpositionFormat::OpenMetricsText100
        && let Some(unit) = &family.unit
    {
        let suffix = format!("_{}", unit.value);
        if !family.name.ends_with(&suffix) {
            return Err(metadata_error(
                unit.line,
                format!(
                    "UNIT {:?} is not an underscore-delimited suffix of family {:?}",
                    unit.value, family.name
                ),
            ));
        }
        if matches!(
            family.semantic_type,
            SemanticType::Info | SemanticType::StateSet
        ) {
            return Err(metadata_error(
                unit.line,
                "Info and StateSet families must have an empty unit",
            ));
        }
    }
    Ok(())
}

#[derive(Debug)]
struct MetricSamples {
    labels: LabelSet,
    wires: Vec<WireSample>,
}

fn build_metrics(
    format: ExpositionFormat,
    family: &FamilyDraft,
    limits: &ParseLimits,
) -> Result<Vec<Metric>, ParseError> {
    let mut metrics = Vec::<MetricSamples>::new();
    let mut metric_by_labels = BTreeMap::<Vec<(String, String)>, usize>::new();
    let mut previous_openmetrics_metric = None::<Vec<(String, String)>>;
    let mut closed_openmetrics_metrics = BTreeSet::<Vec<(String, String)>>::new();
    for wire in family.samples.iter().cloned() {
        let labels = metric_labels(family, &wire)?;
        let key = labels
            .iter()
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect::<Vec<_>>();
        if format == ExpositionFormat::OpenMetricsText100
            && previous_openmetrics_metric.as_ref() != Some(&key)
        {
            if closed_openmetrics_metrics.contains(&key) {
                return Err(semantic_error(
                    wire.line,
                    format!(
                        "OpenMetrics metrics in family {:?} must not be interleaved",
                        family.name
                    ),
                ));
            }
            if let Some(previous) = previous_openmetrics_metric.replace(key.clone()) {
                closed_openmetrics_metrics.insert(previous);
            }
        }
        let index = if let Some(index) = metric_by_labels.get(&key).copied() {
            index
        } else {
            let index = metrics.len();
            metric_by_labels.insert(key, index);
            metrics.push(MetricSamples {
                labels,
                wires: Vec::new(),
            });
            index
        };
        metrics[index].wires.push(wire);
    }

    let mut output = Vec::with_capacity(metrics.len());
    for metric in metrics {
        let point_wires = split_metric_points(family, metric.wires)?;
        let mut points = Vec::with_capacity(point_wires.len());
        for wires in point_wires {
            points.push(assemble_metric_point(
                format,
                family,
                &metric.labels,
                wires,
                limits,
            )?);
        }
        validate_metric_point_order(format, family, &points)?;
        validate_counter_progression(family, &points)?;
        output.push(Metric {
            labels: metric.labels,
            points,
        });
    }
    Ok(output)
}

fn metric_labels(family: &FamilyDraft, wire: &WireSample) -> Result<LabelSet, ParseError> {
    let mut labels = wire.labels.clone();
    match wire.role {
        WireSampleRole::State if labels.remove(&family.name).is_none() => {
            return Err(semantic_error(
                wire.line,
                format!("StateSet sample must contain role label {:?}", family.name),
            ));
        }
        WireSampleRole::HistogramBucket | WireSampleRole::GaugeHistogramBucket
            if labels.remove("le").is_none() =>
        {
            return Err(semantic_error(
                wire.line,
                "histogram bucket sample is missing its le role label",
            ));
        }
        WireSampleRole::SummaryQuantile if labels.remove("quantile").is_none() => {
            return Err(semantic_error(
                wire.line,
                "summary base sample is missing its quantile role label",
            ));
        }
        _ => {}
    }
    if matches!(
        family.semantic_type,
        SemanticType::Histogram | SemanticType::GaugeHistogram
    ) && wire.role != WireSampleRole::HistogramBucket
        && wire.role != WireSampleRole::GaugeHistogramBucket
        && labels.contains_key("le")
    {
        return Err(semantic_error(
            wire.line,
            "histogram metric labels must not contain le",
        ));
    }
    if family.semantic_type == SemanticType::Summary
        && wire.role != WireSampleRole::SummaryQuantile
        && labels.contains_key("quantile")
    {
        return Err(semantic_error(
            wire.line,
            "summary metric labels must not contain quantile",
        ));
    }
    Ok(labels)
}

fn split_metric_points(
    family: &FamilyDraft,
    wires: Vec<WireSample>,
) -> Result<Vec<Vec<WireSample>>, ParseError> {
    if matches!(
        family.semantic_type,
        SemanticType::Unknown | SemanticType::Gauge | SemanticType::Info
    ) {
        return Ok(wires.into_iter().map(|wire| vec![wire]).collect());
    }
    let mut output = Vec::new();
    let mut current = Vec::new();
    let mut signatures = BTreeSet::<String>::new();
    for wire in wires {
        let signature = role_signature(family, &wire)?;
        if signatures.contains(&signature) && !current.is_empty() {
            output.push(std::mem::take(&mut current));
            signatures.clear();
        }
        signatures.insert(signature);
        current.push(wire);
    }
    if !current.is_empty() {
        output.push(current);
    }
    Ok(output)
}

fn role_signature(family: &FamilyDraft, wire: &WireSample) -> Result<String, ParseError> {
    let signature = match wire.role {
        WireSampleRole::State => format!(
            "state\0{}",
            wire.labels
                .get(&family.name)
                .ok_or_else(|| semantic_error(wire.line, "StateSet role label is missing"))?
        ),
        WireSampleRole::HistogramBucket | WireSampleRole::GaugeHistogramBucket => format!(
            "bucket\0{}",
            wire.labels
                .get("le")
                .ok_or_else(|| semantic_error(wire.line, "histogram le role label is missing"))?
        ),
        WireSampleRole::SummaryQuantile => format!(
            "quantile\0{}",
            wire.labels.get("quantile").ok_or_else(|| semantic_error(
                wire.line,
                "summary quantile role label is missing"
            ))?
        ),
        role => format!("role\0{role:?}"),
    };
    Ok(signature)
}

fn assemble_metric_point(
    format: ExpositionFormat,
    family: &FamilyDraft,
    labels: &LabelSet,
    wires: Vec<WireSample>,
    limits: &ParseLimits,
) -> Result<MetricPoint, ParseError> {
    let line = wires.first().map_or(0, |wire| wire.line);
    let (point_time_status, source_timestamp) = point_timestamp(&wires);
    if format == ExpositionFormat::OpenMetricsText100
        && matches!(
            point_time_status,
            PointTimeStatus::MixedComponents | PointTimeStatus::PartialComponents
        )
    {
        return Err(semantic_error(
            line,
            "OpenMetrics components of one MetricPoint must share one explicit timestamp state",
        ));
    }
    let value = match family.semantic_type {
        SemanticType::Unknown | SemanticType::Gauge => {
            let wire = exactly_one_wire(&wires, line, "scalar point")?;
            MetricValue::Scalar {
                value: wire.value.clone(),
                exemplar: wire.exemplar.clone(),
            }
        }
        SemanticType::Counter => assemble_counter(&wires)?,
        SemanticType::StateSet => assemble_stateset(family, &wires, limits)?,
        SemanticType::Info => assemble_info(labels, &wires)?,
        SemanticType::Histogram | SemanticType::GaugeHistogram => {
            assemble_histogram(format, family.semantic_type, &wires, limits)?
        }
        SemanticType::Summary => assemble_summary(format, &wires, limits)?,
    };
    Ok(MetricPoint {
        metric_point_seq: line as u64,
        labels: labels.clone(),
        point_time_status,
        source_timestamp,
        value,
        wire_samples: wires,
    })
}

fn exactly_one_wire<'a>(
    wires: &'a [WireSample],
    line: usize,
    description: &str,
) -> Result<&'a WireSample, ParseError> {
    if let [wire] = wires {
        Ok(wire)
    } else {
        Err(semantic_error(
            line,
            format!("{description} must contain exactly one wire sample"),
        ))
    }
}

fn assemble_counter(wires: &[WireSample]) -> Result<MetricValue, ParseError> {
    let line = wires.first().map_or(0, |wire| wire.line);
    let total = one_role(wires, WireSampleRole::CounterTotal)?
        .ok_or_else(|| semantic_error(line, "counter point is missing its total"))?;
    ensure_nonnegative_non_nan(&total.value, total.line, "counter total")?;
    let created_wire = one_role(wires, WireSampleRole::CounterCreated)?;
    let created = created_from_wire(created_wire)?;
    Ok(MetricValue::Counter(CounterValue {
        total: total.value.clone(),
        created,
        exemplar: total.exemplar.clone(),
    }))
}

fn assemble_stateset(
    family: &FamilyDraft,
    wires: &[WireSample],
    limits: &ParseLimits,
) -> Result<MetricValue, ParseError> {
    if wires.len() > limits.max_states_per_point {
        return Err(ParseError::line(
            wires.first().map_or(0, |wire| wire.line),
            1,
            ParseErrorKind::LimitExceeded(LimitKind::StatesPerPoint),
            "StateSet point exceeds max_states_per_point",
        ));
    }
    let mut states = Vec::with_capacity(wires.len());
    let mut names = BTreeSet::new();
    for wire in wires {
        if wire.role != WireSampleRole::State {
            return Err(semantic_error(
                wire.line,
                "non-state role in StateSet point",
            ));
        }
        if !wire.value.is_zero() && !wire.value.is_one() {
            return Err(semantic_error(
                wire.line,
                "StateSet values must be exactly zero or one",
            ));
        }
        let state = wire.labels.get(&family.name).ok_or_else(|| {
            semantic_error(
                wire.line,
                "StateSet sample is missing its family-named label",
            )
        })?;
        if !names.insert(state.clone()) {
            return Err(semantic_error(
                wire.line,
                format!("duplicate StateSet state {state:?}"),
            ));
        }
        states.push(StateValue {
            state: state.clone(),
            enabled: wire.value.clone(),
        });
    }
    if states.is_empty() {
        return Err(semantic_error(
            0,
            "StateSet point must contain at least one state",
        ));
    }
    Ok(MetricValue::StateSet(states))
}

fn assemble_info(labels: &LabelSet, wires: &[WireSample]) -> Result<MetricValue, ParseError> {
    let wire = exactly_one_wire(
        wires,
        wires.first().map_or(0, |wire| wire.line),
        "Info point",
    )?;
    if !wire.value.is_one() {
        return Err(semantic_error(
            wire.line,
            "Info sample value must be exactly one",
        ));
    }
    Ok(MetricValue::Info(InfoValue {
        wire_merged_labels: labels.clone(),
        partitioned_metric_labels: None,
        partitioned_value_labels: None,
        partition_policy_id: None,
        partition_status: InfoLabelPartitionStatus::UnavailableFromText,
    }))
}

fn assemble_histogram(
    format: ExpositionFormat,
    semantic_type: SemanticType,
    wires: &[WireSample],
    limits: &ParseLimits,
) -> Result<MetricValue, ParseError> {
    let gauge_histogram = semantic_type == SemanticType::GaugeHistogram;
    let bucket_role = if gauge_histogram {
        WireSampleRole::GaugeHistogramBucket
    } else {
        WireSampleRole::HistogramBucket
    };
    let sum_role = if gauge_histogram {
        WireSampleRole::GaugeHistogramSum
    } else {
        WireSampleRole::HistogramSum
    };
    let count_role = if gauge_histogram {
        WireSampleRole::GaugeHistogramCount
    } else {
        WireSampleRole::HistogramCount
    };
    let bucket_wires = wires
        .iter()
        .filter(|wire| wire.role == bucket_role)
        .collect::<Vec<_>>();
    if bucket_wires.is_empty() {
        return Err(semantic_error(
            wires.first().map_or(0, |wire| wire.line),
            "histogram point must contain at least one bucket",
        ));
    }
    if bucket_wires.len() > limits.max_buckets_per_point {
        return Err(ParseError::line(
            bucket_wires[0].line,
            1,
            ParseErrorKind::LimitExceeded(LimitKind::BucketsPerPoint),
            "histogram point exceeds max_buckets_per_point",
        ));
    }
    let mut buckets = Vec::with_capacity(bucket_wires.len());
    for wire in bucket_wires {
        ensure_nonnegative_integer(&wire.value, wire.line, "histogram bucket count")?;
        let lexeme = wire
            .labels
            .get("le")
            .ok_or_else(|| semantic_error(wire.line, "histogram bucket is missing its le label"))?;
        if lexeme.len() > limits.max_numeric_lexeme_bytes {
            return Err(ParseError::line(
                wire.line,
                1,
                ParseErrorKind::LimitExceeded(LimitKind::NumericLexemeBytes),
                "histogram bound exceeds max_numeric_lexeme_bytes",
            ));
        }
        let upper_bound = parse_number_lexeme(format, lexeme).map_err(|error| {
            ParseError::line(
                wire.line,
                1,
                ParseErrorKind::Number,
                format!("invalid histogram bound {lexeme:?}: {error}"),
            )
        })?;
        if upper_bound.kind == NumberKind::NaN {
            return Err(semantic_error(wire.line, "histogram bound must not be NaN"));
        }
        if let Some(exemplar) = &wire.exemplar
            && number_cmp(&exemplar.value, &upper_bound) == Some(Ordering::Greater)
        {
            return Err(ParseError::line(
                wire.line,
                1,
                ParseErrorKind::Exemplar,
                "histogram exemplar exceeds its bucket upper bound",
            ));
        }
        buckets.push(HistogramBucket {
            upper_bound_lexeme: lexeme.clone(),
            upper_bound,
            cumulative_count: wire.value.clone(),
            exemplar: wire.exemplar.clone(),
        });
    }
    for pair in buckets.windows(2) {
        let order = number_cmp(&pair[0].upper_bound, &pair[1].upper_bound).ok_or_else(|| {
            semantic_error(
                wires[0].line,
                "histogram bounds must be ordered real numbers",
            )
        })?;
        if order != Ordering::Less {
            return Err(semantic_error(
                wires[0].line,
                if order == Ordering::Equal {
                    "histogram bounds must be numerically unique"
                } else {
                    "histogram buckets must be emitted in increasing bound order"
                },
            ));
        }
        if pair[0]
            .cumulative_count
            .finite_cmp(&pair[1].cumulative_count)
            == Some(Ordering::Greater)
        {
            return Err(semantic_error(
                wires[0].line,
                "histogram bucket counts must be cumulative",
            ));
        }
    }
    let positive_infinity = buckets
        .last()
        .filter(|bucket| bucket.upper_bound.kind == NumberKind::PositiveInfinity)
        .ok_or_else(|| {
            semantic_error(wires[0].line, "histogram point must end with a +Inf bucket")
        })?;
    let emitted_count = one_role(wires, count_role)?;
    if format == ExpositionFormat::PrometheusText004 && emitted_count.is_none() {
        return Err(semantic_error(
            wires[0].line,
            "Prometheus histogram point must emit _count",
        ));
    }
    if let Some(count) = emitted_count {
        ensure_nonnegative_integer(&count.value, count.line, "histogram count")?;
        if count.value.finite_cmp(&positive_infinity.cumulative_count) != Some(Ordering::Equal) {
            return Err(semantic_error(
                count.line,
                "histogram count must equal the +Inf bucket",
            ));
        }
    }
    let (count, count_origin) = if let Some(count) = emitted_count {
        (count.value.clone(), CountOrigin::EmittedAndValidated)
    } else {
        (
            positive_infinity.cumulative_count.clone(),
            CountOrigin::DerivedFromPositiveInfinity,
        )
    };
    let sum_wire = one_role(wires, sum_role)?;
    let has_negative_bound = buckets.iter().any(|bucket| {
        matches!(bucket.upper_bound.kind, NumberKind::NegativeInfinity)
            || bucket.upper_bound.is_negative()
    });
    let sum = if let Some(sum) = sum_wire {
        if gauge_histogram {
            ensure_non_nan(&sum.value, sum.line, "gauge-histogram sum")?;
            if !has_negative_bound && sum.value.is_negative() {
                return Err(semantic_error(
                    sum.line,
                    "gauge-histogram sum cannot be negative without a negative bound",
                ));
            }
        } else {
            ensure_nonnegative_non_nan(&sum.value, sum.line, "histogram sum")?;
            if has_negative_bound {
                return Err(semantic_error(
                    sum.line,
                    "counter histogram with a negative bound must omit its sum",
                ));
            }
        }
        sum.value.clone()
    } else {
        ExactNumber::absent()
    };
    let created = if gauge_histogram {
        CreatedTimestamp::absent()
    } else {
        created_from_wire(one_role(wires, WireSampleRole::HistogramCreated)?)?
    };
    Ok(MetricValue::Histogram(HistogramValue {
        sum,
        count,
        count_origin,
        created,
        buckets,
    }))
}

fn assemble_summary(
    format: ExpositionFormat,
    wires: &[WireSample],
    limits: &ParseLimits,
) -> Result<MetricValue, ParseError> {
    let count = if let Some(count) = one_role(wires, WireSampleRole::SummaryCount)? {
        ensure_nonnegative_integer(&count.value, count.line, "summary count")?;
        count.value.clone()
    } else {
        ExactNumber::absent()
    };
    let sum = if let Some(sum) = one_role(wires, WireSampleRole::SummarySum)? {
        ensure_nonnegative_non_nan(&sum.value, sum.line, "summary sum")?;
        sum.value.clone()
    } else {
        ExactNumber::absent()
    };
    let created = created_from_wire(one_role(wires, WireSampleRole::SummaryCreated)?)?;
    let quantile_wires = wires
        .iter()
        .filter(|wire| wire.role == WireSampleRole::SummaryQuantile)
        .collect::<Vec<_>>();
    if quantile_wires.len() > limits.max_quantiles_per_point {
        return Err(ParseError::line(
            quantile_wires.first().map_or(0, |wire| wire.line),
            1,
            ParseErrorKind::LimitExceeded(LimitKind::QuantilesPerPoint),
            "summary point exceeds max_quantiles_per_point",
        ));
    }
    let zero = parse_number_lexeme(format, "0").expect("zero is valid in both grammars");
    let one = parse_number_lexeme(format, "1").expect("one is valid in both grammars");
    let mut quantiles = Vec::with_capacity(quantile_wires.len());
    for wire in quantile_wires {
        let lexeme = wire.labels.get("quantile").ok_or_else(|| {
            semantic_error(wire.line, "summary quantile is missing its role label")
        })?;
        if lexeme.len() > limits.max_numeric_lexeme_bytes {
            return Err(ParseError::line(
                wire.line,
                1,
                ParseErrorKind::LimitExceeded(LimitKind::NumericLexemeBytes),
                "summary quantile exceeds max_numeric_lexeme_bytes",
            ));
        }
        let quantile = parse_number_lexeme(format, lexeme).map_err(|error| {
            ParseError::line(
                wire.line,
                1,
                ParseErrorKind::Number,
                format!("invalid summary quantile {lexeme:?}: {error}"),
            )
        })?;
        if number_cmp(&quantile, &zero).is_none_or(|order| order == Ordering::Less)
            || number_cmp(&quantile, &one).is_none_or(|order| order == Ordering::Greater)
        {
            return Err(semantic_error(
                wire.line,
                "summary quantile must be a finite number in [0, 1]",
            ));
        }
        if wire.value.kind != NumberKind::NaN {
            ensure_nonnegative_non_nan(&wire.value, wire.line, "summary quantile value")?;
        }
        quantiles.push(QuantileValue {
            quantile_lexeme: lexeme.clone(),
            quantile,
            value: wire.value.clone(),
        });
    }
    quantiles.sort_by(|left, right| {
        number_cmp(&left.quantile, &right.quantile).unwrap_or(Ordering::Equal)
    });
    for pair in quantiles.windows(2) {
        if number_cmp(&pair[0].quantile, &pair[1].quantile) == Some(Ordering::Equal) {
            return Err(semantic_error(
                wires.first().map_or(0, |wire| wire.line),
                "summary quantiles must be numerically unique",
            ));
        }
    }
    Ok(MetricValue::Summary(SummaryValue {
        sum,
        count,
        created,
        quantiles,
    }))
}

fn one_role(wires: &[WireSample], role: WireSampleRole) -> Result<Option<&WireSample>, ParseError> {
    let mut matching = wires.iter().filter(|wire| wire.role == role);
    let first = matching.next();
    if let Some(duplicate) = matching.next() {
        return Err(semantic_error(
            duplicate.line,
            format!("duplicate semantic role {role:?} in one MetricPoint"),
        ));
    }
    Ok(first)
}

fn created_from_wire(wire: Option<&WireSample>) -> Result<CreatedTimestamp, ParseError> {
    let Some(wire) = wire else {
        return Ok(CreatedTimestamp::absent());
    };
    if wire.value.kind != NumberKind::Finite {
        return Err(semantic_error(
            wire.line,
            "Created value must be a finite Unix-seconds timestamp",
        ));
    }
    let lexeme = wire
        .value
        .source_lexeme
        .as_deref()
        .expect("finite wire values retain their source lexeme");
    CreatedTimestamp::parse_openmetrics(lexeme).map_err(|error| {
        semantic_error(
            wire.line,
            format!("invalid semantic Created timestamp {lexeme:?}: {error}"),
        )
    })
}

fn ensure_nonnegative_integer(
    value: &ExactNumber,
    line: usize,
    description: &str,
) -> Result<(), ParseError> {
    if value.kind != NumberKind::Finite || !value.is_integer() || value.is_negative() {
        return Err(semantic_error(
            line,
            format!("{description} must be a non-negative finite integer"),
        ));
    }
    Ok(())
}

fn ensure_nonnegative_non_nan(
    value: &ExactNumber,
    line: usize,
    description: &str,
) -> Result<(), ParseError> {
    if matches!(value.kind, NumberKind::NaN | NumberKind::NegativeInfinity)
        || value.is_negative()
        || value.kind == NumberKind::Absent
    {
        return Err(semantic_error(
            line,
            format!("{description} must be non-negative and non-NaN"),
        ));
    }
    Ok(())
}

fn ensure_non_nan(value: &ExactNumber, line: usize, description: &str) -> Result<(), ParseError> {
    if matches!(value.kind, NumberKind::NaN | NumberKind::Absent) {
        return Err(semantic_error(
            line,
            format!("{description} must not be NaN"),
        ));
    }
    Ok(())
}

fn point_timestamp(wires: &[WireSample]) -> (PointTimeStatus, SourceTimestamp) {
    let explicit = wires
        .iter()
        .filter(|wire| wire.source_timestamp.lexeme.is_some())
        .collect::<Vec<_>>();
    if explicit.is_empty() {
        return (PointTimeStatus::AllAbsent, SourceTimestamp::absent());
    }
    if explicit.len() != wires.len() {
        return (
            PointTimeStatus::PartialComponents,
            SourceTimestamp::absent(),
        );
    }
    let first = &explicit[0].source_timestamp;
    if explicit
        .iter()
        .all(|wire| first.exact_eq(&wire.source_timestamp))
    {
        (PointTimeStatus::UniformExplicit, first.clone())
    } else {
        (PointTimeStatus::MixedComponents, SourceTimestamp::absent())
    }
}

fn validate_metric_point_order(
    format: ExpositionFormat,
    family: &FamilyDraft,
    points: &[MetricPoint],
) -> Result<(), ParseError> {
    if format != ExpositionFormat::OpenMetricsText100 || points.len() < 2 {
        return Ok(());
    }
    for point in points {
        if point.point_time_status != PointTimeStatus::UniformExplicit {
            return Err(semantic_error(
                point.metric_point_seq as usize,
                format!(
                    "repeated MetricPoints in OpenMetrics family {:?} require explicit timestamps",
                    family.name
                ),
            ));
        }
    }
    for pair in points.windows(2) {
        let left = pair[0]
            .source_timestamp
            .exact_decimal
            .as_ref()
            .expect("uniform timestamp has exact decimal");
        let right = pair[1]
            .source_timestamp
            .exact_decimal
            .as_ref()
            .expect("uniform timestamp has exact decimal");
        if left.numeric_cmp(right) != Ordering::Less {
            return Err(semantic_error(
                pair[1].metric_point_seq as usize,
                "repeated OpenMetrics MetricPoint timestamps must increase monotonically",
            ));
        }
    }
    Ok(())
}

fn validate_counter_progression(
    family: &FamilyDraft,
    points: &[MetricPoint],
) -> Result<(), ParseError> {
    if family.semantic_type != SemanticType::Counter || points.len() < 2 {
        return Ok(());
    }
    for pair in points.windows(2) {
        let (MetricValue::Counter(previous), MetricValue::Counter(current)) =
            (&pair[0].value, &pair[1].value)
        else {
            unreachable!("counter family points carry counter values")
        };
        if number_cmp(&current.total, &previous.total) == Some(Ordering::Less) {
            let previous_created = &previous.created.value;
            let current_created = &current.created.value;
            if current_created.lexeme.is_none() || current_created.exact_eq(previous_created) {
                return Err(semantic_error(
                    pair[1].metric_point_seq as usize,
                    "counter total decreased without a distinct Created reset timestamp",
                ));
            }
        }
    }
    Ok(())
}

fn number_cmp(left: &ExactNumber, right: &ExactNumber) -> Option<Ordering> {
    match (left.kind, right.kind) {
        (NumberKind::NaN | NumberKind::Absent, _) | (_, NumberKind::NaN | NumberKind::Absent) => {
            None
        }
        (NumberKind::NegativeInfinity, NumberKind::NegativeInfinity)
        | (NumberKind::PositiveInfinity, NumberKind::PositiveInfinity) => Some(Ordering::Equal),
        (NumberKind::NegativeInfinity, _) | (_, NumberKind::PositiveInfinity) => {
            Some(Ordering::Less)
        }
        (NumberKind::PositiveInfinity, _) | (_, NumberKind::NegativeInfinity) => {
            Some(Ordering::Greater)
        }
        (NumberKind::Finite, NumberKind::Finite) => left.finite_cmp(right),
    }
}

fn metadata_error(line: usize, message: impl Into<String>) -> ParseError {
    ParseError::line(line, 1, ParseErrorKind::Metadata, message)
}

fn semantic_error(line: usize, message: impl Into<String>) -> ParseError {
    ParseError::line(line, 1, ParseErrorKind::Semantic, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::syntax::parse_document;

    fn parse(format: ExpositionFormat, body: &str) -> Result<Exposition, ParseError> {
        let limits = ParseLimits::default();
        assemble_exposition(
            format,
            parse_document(format, body.as_bytes(), &limits)?,
            &limits,
        )
    }

    #[test]
    fn classic_histograms_are_isolated_by_complete_base_labels() {
        let exposition = parse(
            ExpositionFormat::PrometheusText004,
            "# TYPE latency histogram\nlatency_bucket{route=\"a\",le=\"1\"} 2\nlatency_bucket{route=\"b\",le=\"1\"} 5\nlatency_bucket{route=\"a\",le=\"+Inf\"} 3\nlatency_count{route=\"a\"} 3\nlatency_bucket{route=\"b\",le=\"+Inf\"} 8\nlatency_count{route=\"b\"} 8\n",
        )
        .unwrap();
        let family = exposition.family("latency").unwrap();
        assert_eq!(family.metrics.len(), 2);
        for metric in &family.metrics {
            let MetricValue::Histogram(value) = &metric.points[0].value else {
                panic!("expected histogram")
            };
            let expected = if metric.labels["route"] == "a" {
                "3"
            } else {
                "8"
            };
            assert_eq!(value.count.source_lexeme.as_deref(), Some(expected));
        }
    }

    #[test]
    fn openmetrics_count_origin_and_wire_roles_are_distinct() {
        let exposition = parse(
            ExpositionFormat::OpenMetricsText100,
            "# TYPE latency histogram\nlatency_bucket{le=\"1.0\"} 2\nlatency_bucket{le=\"+Inf\"} 3 # {trace_id=\"abc\"} 2.5\n# EOF\n",
        )
        .unwrap();
        let point = &exposition.family("latency").unwrap().metrics[0].points[0];
        let MetricValue::Histogram(value) = &point.value else {
            panic!("expected histogram")
        };
        assert_eq!(value.count_origin, CountOrigin::DerivedFromPositiveInfinity);
        assert_eq!(point.wire_samples.len(), 2);
        assert!(value.buckets[1].exemplar.is_some());
    }

    #[test]
    fn info_preserves_complete_merged_identity() {
        let exposition = parse(
            ExpositionFormat::OpenMetricsText100,
            "# TYPE build info\nbuild_info{entity=\"runner\",revision=\"abc\",version=\"1\"} 1\n# EOF\n",
        )
        .unwrap();
        let metric = &exposition.family("build").unwrap().metrics[0];
        assert_eq!(metric.labels.len(), 3);
        let MetricValue::Info(value) = &metric.points[0].value else {
            panic!("expected info")
        };
        assert_eq!(value.wire_merged_labels, metric.labels);
        assert_eq!(
            value.partition_status,
            InfoLabelPartitionStatus::UnavailableFromText
        );
        assert!(value.partitioned_metric_labels.is_none());
    }

    #[test]
    fn classic_component_timestamp_statuses_keep_wire_lexemes() {
        let mixed = parse(
            ExpositionFormat::PrometheusText004,
            "# TYPE latency histogram\nlatency_bucket{le=\"+Inf\"} 1 1000\nlatency_count 1 1001\n",
        )
        .unwrap();
        let point = &mixed.family("latency").unwrap().metrics[0].points[0];
        assert_eq!(point.point_time_status, PointTimeStatus::MixedComponents);
        assert_eq!(point.source_timestamp, SourceTimestamp::absent());
        assert_eq!(
            point.wire_samples[0].source_timestamp.lexeme.as_deref(),
            Some("1000")
        );

        let partial = parse(
            ExpositionFormat::PrometheusText004,
            "# TYPE latency histogram\nlatency_bucket{le=\"+Inf\"} 1 1000\nlatency_count 1\n",
        )
        .unwrap();
        assert_eq!(
            partial.family("latency").unwrap().metrics[0].points[0].point_time_status,
            PointTimeStatus::PartialComponents
        );
    }

    #[test]
    fn arbitrary_classic_created_name_remains_unknown_scalar() {
        let exposition =
            parse(ExpositionFormat::PrometheusText004, "process_created 123\n").unwrap();
        let family = exposition.family("process_created").unwrap();
        assert_eq!(family.semantic_type, SemanticType::Unknown);
        assert_eq!(
            family.metrics[0].points[0].wire_samples[0].role,
            WireSampleRole::Scalar
        );
    }
}
