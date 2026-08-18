// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenTelemetry JSON GenAI graph-input adapter.
//!
//! The adapter consumes OTLP/HTTP JSON exports directly and projects their
//! causal GenAI span topology into the executable Graph IR.  The current IR has
//! no recorded-output replay node, so non-inference spans are folded into
//! trace-local state and their observed duration is carried by rerouted edges.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::io::Read;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, ensure};
use async_trait::async_trait;
use bytes::Bytes;
use flate2::read::GzDecoder;
use serde::Deserialize;
use serde_json::{Map, Value, value::RawValue};

use super::{
    CacheBustTarget, GraphInputAdapter, GraphInputContext, PreparedRunnerGraphInput, TStarWindow,
};
use crate::dataset::SegmentPool;
use crate::graph::input::{GraphInputBundle, GraphInputMetadata, GraphInputWarning};
use crate::graph::model::{
    ChannelSpec, ExecutableGraphNode, GraphRecord, GraphTracePlan, GraphTraceProgram, LlmNode,
    LlmRequestSpec, PromptItem, START_NODE_ID, StaticEdge, TraceRecord,
};
use crate::graph::validate::validate;

/// Native OTLP/HTTP JSON adapter for OpenInference and OTel GenAI spans.
#[derive(Debug)]
pub(super) struct OtlpGenaiRunnerGraphInputAdapter;

#[async_trait(?Send)]
impl GraphInputAdapter for OtlpGenaiRunnerGraphInputAdapter {
    fn format(&self) -> &'static str {
        "otlp_genai"
    }

    async fn load(
        &self,
        raw: &RawValue,
        _context: &GraphInputContext<'_>,
    ) -> Result<PreparedRunnerGraphInput> {
        let input: OtlpGenaiInput =
            serde_json::from_str(raw.get()).context("decoding direct otlp_genai graph input")?;
        input.validate(self.format())?;
        let root = input.read()?;
        let bundle = lower_otlp(root, input.entries)?;
        ensure!(
            !bundle.programs.is_empty(),
            "OTLP GenAI input contains no executable GenAI client spans"
        );
        Ok(PreparedRunnerGraphInput {
            bundle,
            random_seed: input.random_seed,
            default_output_tokens: input.default_output_tokens()?,
            allow_dataset_wrap: true,
            t_star_window: TStarWindow::default(),
            cache_bust_target: CacheBustTarget::None,
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OtlpGenaiInput {
    #[serde(rename = "type")]
    input_type: String,
    format: String,
    #[serde(default)]
    path: Option<PathBuf>,
    #[serde(default)]
    records: Option<Value>,
    #[serde(default = "default_sequential")]
    sampling: String,
    #[serde(default)]
    entries: Option<usize>,
    #[serde(default)]
    random_seed: Option<u64>,
    #[serde(default)]
    osl: Option<OslSpec>,
    #[serde(default)]
    options: Map<String, Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OslSpec {
    value: f64,
}

impl OtlpGenaiInput {
    fn validate(&self, expected_format: &str) -> Result<()> {
        ensure!(
            self.input_type == "file",
            "otlp_genai input type must be \"file\""
        );
        ensure!(
            self.format == expected_format,
            "direct graph adapter {expected_format:?} received dataset.format={:?}",
            self.format
        );
        ensure!(
            self.path.is_some() ^ self.records.is_some(),
            "otlp_genai requires exactly one of path or records"
        );
        ensure!(
            self.sampling.eq_ignore_ascii_case("sequential"),
            "otlp_genai requires sequential root selection"
        );
        ensure!(
            self.entries != Some(0),
            "otlp_genai entries must be positive when configured"
        );
        ensure!(
            self.options.is_empty(),
            "otlp_genai does not support dataset options"
        );
        Ok(())
    }

    fn default_output_tokens(&self) -> Result<usize> {
        let value = self.osl.as_ref().map_or(1.0, |osl| osl.value);
        ensure!(
            value.is_finite() && value > 0.0 && value < usize::MAX as f64,
            "otlp_genai osl.value must be a finite positive usize"
        );
        Ok(value.ceil() as usize)
    }

    fn read(&self) -> Result<Vec<Value>> {
        match (&self.path, &self.records) {
            (Some(path), None) => read_path(path),
            (None, Some(records)) => Ok(vec![records.clone()]),
            _ => unreachable!("source exclusivity validated"),
        }
    }
}

fn default_sequential() -> String {
    "sequential".into()
}

fn read_path(path: &PathBuf) -> Result<Vec<Value>> {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase);
    let inner_extension = if extension.as_deref() == Some("gz") {
        path.file_stem()
            .and_then(|stem| std::path::Path::new(stem).extension())
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase)
    } else {
        extension.clone()
    };
    ensure!(
        matches!(inner_extension.as_deref(), Some("json" | "jsonl")),
        "OTLP input {path:?} must end in .json, .jsonl, .json.gz, or .jsonl.gz"
    );
    let mut bytes = Vec::new();
    if extension.as_deref() == Some("gz") {
        GzDecoder::new(
            fs::File::open(path).with_context(|| format!("opening OTLP input {path:?}"))?,
        )
        .read_to_end(&mut bytes)
        .with_context(|| format!("decompressing OTLP input {path:?}"))?;
    } else {
        bytes = fs::read(path).with_context(|| format!("reading OTLP input {path:?}"))?;
    }
    let is_jsonl = inner_extension.as_deref() == Some("jsonl");
    if is_jsonl {
        bytes
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.iter().all(u8::is_ascii_whitespace))
            .map(|line| serde_json::from_slice(line).context("decoding OTLP JSONL record"))
            .collect()
    } else {
        Ok(vec![
            serde_json::from_slice(&bytes).context("decoding OTLP JSON")?,
        ])
    }
}

#[derive(Debug, Clone)]
struct Span {
    trace_id: String,
    span_id: String,
    parent_span_id: Option<String>,
    name: String,
    kind: i64,
    start_time_unix_nano: Option<u64>,
    end_time_unix_nano: Option<u64>,
    attributes: BTreeMap<String, Value>,
    is_streaming: bool,
}

fn lower_otlp(roots: Vec<Value>, limit: Option<usize>) -> Result<GraphInputBundle> {
    let spans = roots
        .into_iter()
        .map(flatten_resource_spans)
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    ensure!(!spans.is_empty(), "OTLP input contains no spans");
    let mut traces = Vec::<(String, Vec<Span>)>::new();
    for span in spans {
        if let Some((_, grouped)) = traces
            .iter_mut()
            .find(|(trace_id, _)| *trace_id == span.trace_id)
        {
            grouped.push(span);
        } else {
            traces.push((span.trace_id.clone(), vec![span]));
        }
    }
    let mut pool = SegmentPool::new();
    let mut programs = Vec::new();
    let mut warnings = BTreeSet::new();
    for (trace_id, spans) in traces {
        if let Some(program) = lower_trace(&trace_id, &spans, &mut pool, &mut warnings)? {
            programs.push(program);
            if limit.is_some_and(|limit| programs.len() >= limit) {
                break;
            }
        }
    }
    let root_count = programs.len();
    let node_count = programs
        .iter()
        .map(|program: &GraphTraceProgram| program.profiling.graph.llm_node_count())
        .sum();
    Ok(GraphInputBundle {
        programs,
        segments: Arc::new(pool.freeze()),
        metadata: GraphInputMetadata {
            format: "otlp_genai".into(),
            root_count,
            node_count,
            warning_facts: warnings.into_iter().collect(),
        },
    })
}

fn flatten_resource_spans(root: Value) -> Result<Vec<Span>> {
    let resource_spans = root
        .as_object()
        .and_then(|object| object.get("resourceSpans"))
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("OTLP root must contain an array resourceSpans"))?;
    let mut spans = Vec::new();
    for resource in resource_spans {
        let scope_spans = resource
            .as_object()
            .and_then(|object| object.get("scopeSpans"))
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("OTLP resourceSpans entry must contain an array scopeSpans"))?;
        for scope in scope_spans {
            let raw_spans = scope
                .as_object()
                .and_then(|object| object.get("spans"))
                .and_then(Value::as_array)
                .ok_or_else(|| anyhow!("OTLP scopeSpans entry must contain an array spans"))?;
            for raw in raw_spans {
                spans.push(decode_span(raw)?);
            }
        }
    }
    Ok(spans)
}

fn decode_span(raw: &Value) -> Result<Span> {
    let object = raw
        .as_object()
        .ok_or_else(|| anyhow!("OTLP span must be an object"))?;
    let required = |name| {
        object
            .get(name)
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| anyhow!("OTLP span {name} must be a string"))
    };
    let attributes = object
        .get("attributes")
        .map_or(Ok(BTreeMap::new()), decode_attributes)?;
    let is_streaming = object
        .get("events")
        .and_then(Value::as_array)
        .is_some_and(|events| {
            events
                .iter()
                .any(|event| event.get("name").and_then(Value::as_str) == Some("gen_ai.choice"))
        });
    Ok(Span {
        trace_id: required("traceId")?,
        span_id: required("spanId")?,
        parent_span_id: object
            .get("parentSpanId")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .map(str::to_owned),
        name: required("name")?,
        kind: object
            .get("kind")
            .and_then(Value::as_i64)
            .unwrap_or_default(),
        start_time_unix_nano: timestamp(object, "startTimeUnixNano")?,
        end_time_unix_nano: timestamp(object, "endTimeUnixNano")?,
        attributes,
        is_streaming,
    })
}

fn timestamp(object: &Map<String, Value>, name: &str) -> Result<Option<u64>> {
    let Some(value) = object.get(name) else {
        return Ok(None);
    };
    match value {
        Value::String(value) => value
            .parse()
            .with_context(|| format!("decoding OTLP span {name}"))
            .map(Some),
        Value::Number(value) => value
            .as_u64()
            .ok_or_else(|| anyhow!("OTLP span {name} must be an unsigned integer"))
            .map(Some),
        _ => Err(anyhow!("OTLP span {name} must be an unsigned integer")),
    }
}

fn decode_attributes(raw: &Value) -> Result<BTreeMap<String, Value>> {
    let entries = raw
        .as_array()
        .ok_or_else(|| anyhow!("OTLP span attributes must be an array"))?;
    entries
        .iter()
        .map(|entry| {
            let object = entry
                .as_object()
                .ok_or_else(|| anyhow!("OTLP attribute must be an object"))?;
            let key = object
                .get("key")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("OTLP attribute key must be a string"))?;
            let value = object
                .get("value")
                .ok_or_else(|| anyhow!("OTLP attribute {key:?} has no value"))?;
            Ok((key.to_owned(), decode_any_value(value)?))
        })
        .collect()
}

fn decode_any_value(raw: &Value) -> Result<Value> {
    let object = raw
        .as_object()
        .ok_or_else(|| anyhow!("OTLP AnyValue must be an object"))?;
    let mut variants = object.iter().filter(|(name, _)| {
        matches!(
            name.as_str(),
            "stringValue"
                | "boolValue"
                | "intValue"
                | "doubleValue"
                | "bytesValue"
                | "arrayValue"
                | "kvlistValue"
        )
    });
    let Some((name, value)) = variants.next() else {
        return Err(anyhow!("OTLP AnyValue has no supported variant"));
    };
    ensure!(
        variants.next().is_none(),
        "OTLP AnyValue contains multiple variants"
    );
    match name.as_str() {
        "stringValue" | "boolValue" | "doubleValue" | "bytesValue" => Ok(value.clone()),
        "intValue" => match value {
            Value::String(value) => value
                .parse::<i64>()
                .map(Value::from)
                .context("decoding OTLP intValue"),
            Value::Number(_) => Ok(value.clone()),
            _ => Err(anyhow!("OTLP intValue must be a number or decimal string")),
        },
        "arrayValue" => value
            .get("values")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("OTLP arrayValue must contain an array values"))?
            .iter()
            .map(decode_any_value)
            .collect::<Result<Vec<_>>>()
            .map(Value::Array),
        "kvlistValue" => {
            let entries = value
                .get("values")
                .and_then(Value::as_array)
                .ok_or_else(|| anyhow!("OTLP kvlistValue must contain an array values"))?;
            let mut decoded = Map::new();
            for entry in entries {
                let object = entry
                    .as_object()
                    .ok_or_else(|| anyhow!("OTLP kvlist entry must be an object"))?;
                let key = object
                    .get("key")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("OTLP kvlist key must be a string"))?;
                decoded.insert(
                    key.into(),
                    decode_any_value(
                        object
                            .get("value")
                            .ok_or_else(|| anyhow!("OTLP kvlist entry has no value"))?,
                    )?,
                );
            }
            Ok(Value::Object(decoded))
        }
        _ => unreachable!("variant filtered above"),
    }
}

fn lower_trace(
    trace_id: &str,
    spans: &[Span],
    pool: &mut SegmentPool,
    warnings: &mut BTreeSet<GraphInputWarning>,
) -> Result<Option<GraphTraceProgram>> {
    let mut by_id = HashMap::new();
    for span in spans {
        ensure!(
            by_id.insert(span.span_id.as_str(), span).is_none(),
            "OTLP trace {trace_id:?} contains duplicate span id {:?}",
            span.span_id
        );
    }
    let mut kept = BTreeSet::new();
    for span in spans.iter().filter(|span| is_genai(span)) {
        let mut cursor = Some(span);
        while let Some(current) = cursor {
            if !kept.insert(current.span_id.clone()) {
                break;
            }
            cursor = current
                .parent_span_id
                .as_deref()
                .and_then(|id| by_id.get(id).copied());
        }
    }
    if kept.is_empty() {
        return Ok(None);
    }
    let llm_ids = spans
        .iter()
        .filter(|span| kept.contains(&span.span_id) && is_llm(span))
        .map(|span| span.span_id.clone())
        .collect::<BTreeSet<_>>();
    if llm_ids.is_empty() {
        return Ok(None);
    }
    let collapsed = kept.len() - llm_ids.len();
    if collapsed > 0 {
        warnings.insert(GraphInputWarning::new(
            "otlp_genai_replay_spans_collapsed",
            BTreeMap::from([("count".into(), collapsed.to_string())]),
        ));
    }
    let mut graph = GraphRecord::default();
    let mut initial_state = BTreeMap::new();
    let mut names = BTreeSet::new();
    let mut node_by_span = HashMap::new();
    for span in spans
        .iter()
        .filter(|span| kept.contains(&span.span_id) && !llm_ids.contains(&span.span_id))
    {
        let channel = format!("otlp_replay_{}_output", span.span_id);
        graph.state.insert(channel.clone(), ChannelSpec::default());
        if let Some(value) = replay_output(span) {
            initial_state.insert(channel, value);
        }
    }
    for span in spans.iter().filter(|span| llm_ids.contains(&span.span_id)) {
        let node_id = unique_node_id(&span.name, &span.span_id, &mut names);
        let messages = input_messages(span);
        let wire = serde_json::to_vec(&messages).context("serializing OTLP input messages")?;
        let raw_messages = pool.intern_raw(None, Bytes::from(wire))?;
        let output = format!("{node_id}_response");
        graph.state.insert(output.clone(), ChannelSpec::default());
        graph.nodes.insert(
            node_id.clone(),
            ExecutableGraphNode::Llm(LlmNode {
                output,
                streaming: span.is_streaming,
                inputs: Vec::new(),
                min_start_delay_us: None,
                max_tokens: span_output_tokens(span),
                items: vec![PromptItem::RawMessages { raw_messages }],
                request: span
                    .attributes
                    .get("gen_ai.request.model")
                    .and_then(Value::as_str)
                    .map(|model| LlmRequestSpec {
                        tools: None,
                        model: Some(model.into()),
                        additional_body: None,
                    }),
                metadata: otlp_metadata(span, trace_id),
            }),
        );
        node_by_span.insert(span.span_id.as_str(), node_id);
    }
    graph.edges = fold_span_edges(spans, &kept, &llm_ids, &node_by_span);
    let validation = validate(&graph);
    ensure!(
        validation.is_empty(),
        "OTLP trace {trace_id:?} lowered to an invalid graph: {}",
        validation
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("; ")
    );
    Ok(Some(GraphTraceProgram::static_graph(GraphTracePlan {
        graph,
        trace: TraceRecord {
            id: trace_id.to_ascii_lowercase(),
            graph_ref: None,
            initial_state,
        },
        arrival_offset_ns: None,
    })))
}

/// Emit only executable LLM edges while preserving the elapsed time spent in
/// each skipped ancestor span. OTLP parentage is a forest, so following each
/// retained root through replay children cannot introduce an execution cycle.
fn fold_span_edges(
    spans: &[Span],
    kept: &BTreeSet<String>,
    llm_ids: &BTreeSet<String>,
    node_by_span: &HashMap<&str, String>,
) -> Vec<StaticEdge> {
    let mut children = HashMap::<&str, Vec<&Span>>::new();
    let mut roots = Vec::new();
    for span in spans.iter().filter(|span| kept.contains(&span.span_id)) {
        match span
            .parent_span_id
            .as_deref()
            .filter(|id| kept.contains(*id))
        {
            Some(parent) => children.entry(parent).or_default().push(span),
            None => roots.push(span),
        }
    }
    let mut edges = Vec::new();
    for root in roots {
        emit_folded_descendants(
            root,
            START_NODE_ID,
            &[],
            &children,
            llm_ids,
            node_by_span,
            &mut edges,
        );
    }
    edges
}

fn emit_folded_descendants<'a>(
    span: &'a Span,
    source: &str,
    skipped_intervals: &[(u64, u64)],
    children: &HashMap<&'a str, Vec<&'a Span>>,
    llm_ids: &BTreeSet<String>,
    node_by_span: &HashMap<&str, String>,
    edges: &mut Vec<StaticEdge>,
) {
    let is_executable = llm_ids.contains(&span.span_id);
    if is_executable {
        let target = node_by_span[span.span_id.as_str()].as_str();
        if source != target {
            let delay = interval_duration_us(skipped_intervals);
            edges.push(StaticEdge {
                source: source.to_owned(),
                target: target.to_owned(),
                // START has no completion timestamp in the executor. A leading
                // collapsed span therefore uses the node's start-floor anchor.
                delay_after_predecessor_us: (source != START_NODE_ID && delay > 0.0)
                    .then_some(delay),
                min_start_delay_us: (source == START_NODE_ID && delay > 0.0).then_some(delay),
                delay_after_predecessor_start_us: None,
                delay_after_predecessor_first_token_us: None,
            });
        }
        for child in children.get(span.span_id.as_str()).into_iter().flatten() {
            emit_folded_descendants(child, target, &[], children, llm_ids, node_by_span, edges);
        }
    } else {
        let mut accumulated = skipped_intervals.to_vec();
        if let (Some(start), Some(end)) = (span.start_time_unix_nano, span.end_time_unix_nano)
            && end >= start
        {
            accumulated.push((start, end));
        }
        for child in children.get(span.span_id.as_str()).into_iter().flatten() {
            emit_folded_descendants(
                child,
                source,
                &accumulated,
                children,
                llm_ids,
                node_by_span,
                edges,
            );
        }
    }
}

fn interval_duration_us(intervals: &[(u64, u64)]) -> f64 {
    let mut intervals = intervals.to_vec();
    intervals.sort_unstable();
    let mut merged = 0_u64;
    let mut current: Option<(u64, u64)> = None;
    for (start, end) in intervals {
        match current {
            Some((current_start, current_end)) if start <= current_end => {
                current = Some((current_start, current_end.max(end)));
            }
            Some((current_start, current_end)) => {
                merged = merged.saturating_add(current_end - current_start);
                current = Some((start, end));
            }
            None => current = Some((start, end)),
        }
    }
    if let Some((start, end)) = current {
        merged = merged.saturating_add(end - start);
    }
    merged as f64 / 1_000.0
}

fn span_output_tokens(span: &Span) -> Option<usize> {
    ["gen_ai.request.max_tokens", "gen_ai.usage.output_tokens"]
        .into_iter()
        .filter_map(|key| span.attributes.get(key))
        .find_map(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value > 0)
}

fn replay_output(span: &Span) -> Option<Value> {
    if let Some(output) = span.attributes.get("output.value") {
        return match output {
            Value::String(value) => serde_json::from_str(value)
                .ok()
                .or_else(|| Some(output.clone())),
            value => Some(value.clone()),
        };
    }
    match span.attributes.get("gen_ai.output.messages") {
        Some(Value::String(value)) => match serde_json::from_str(value) {
            Ok(Value::Array(messages)) => Some(message_array(messages)),
            Ok(_) | Err(_) => None,
        },
        Some(Value::Array(messages)) => Some(message_array(messages.clone())),
        _ => None,
    }
}

fn is_genai(span: &Span) -> bool {
    span.attributes.contains_key("openinference.span.kind")
        || span.attributes.keys().any(|key| key.starts_with("gen_ai."))
}
fn is_llm(span: &Span) -> bool {
    match span
        .attributes
        .get("openinference.span.kind")
        .and_then(Value::as_str)
    {
        Some("LLM" | "AGENT") => true,
        Some(_) => false,
        None => match span
            .attributes
            .get("gen_ai.operation.name")
            .and_then(Value::as_str)
        {
            Some(
                "chat" | "text_completion" | "generate_content" | "invoke_agent" | "create_message",
            ) => true,
            Some(_) => false,
            None => span.kind == 3,
        },
    }
}
fn input_messages(span: &Span) -> Value {
    if let Some(input) = span.attributes.get("input.value") {
        return match input {
            Value::String(value) => match serde_json::from_str(value) {
                Ok(Value::Array(messages)) => message_array(messages),
                Ok(Value::Object(message)) => Value::Array(vec![Value::Object(message)]),
                Ok(_) => Value::Array(Vec::new()),
                Err(_) => Value::Array(vec![serde_json::json!({"role":"user","content":value})]),
            },
            Value::Array(messages) => message_array(messages.clone()),
            Value::Object(message) => Value::Array(vec![Value::Object(message.clone())]),
            _ => Value::Array(Vec::new()),
        };
    }
    match span.attributes.get("gen_ai.input.messages") {
        None => Value::Array(Vec::new()),
        Some(Value::String(value)) => match serde_json::from_str(value) {
            Ok(Value::Array(messages)) => message_array(messages),
            Ok(_) | Err(_) => Value::Array(Vec::new()),
        },
        Some(Value::Array(messages)) => message_array(messages.clone()),
        Some(_) => Value::Array(Vec::new()),
    }
}

fn message_array(messages: Vec<Value>) -> Value {
    Value::Array(
        messages
            .into_iter()
            .filter(Value::is_object)
            .collect::<Vec<_>>(),
    )
}

fn otlp_metadata(span: &Span, trace_id: &str) -> BTreeMap<String, Value> {
    let mut metadata = BTreeMap::from([
        ("otlp.span_id".into(), Value::String(span.span_id.clone())),
        ("otlp.trace_id".into(), Value::String(trace_id.to_owned())),
    ]);
    for key in ["gen_ai.system", "server.address", "server.port"] {
        if let Some(value) = span.attributes.get(key) {
            metadata.insert(format!("otlp.{key}"), value.clone());
        }
    }
    metadata
}
fn unique_node_id(name: &str, span_id: &str, used: &mut BTreeSet<String>) -> String {
    let base = format!(
        "{}_{}",
        sanitize(name),
        span_id
            .chars()
            .rev()
            .take(8)
            .collect::<String>()
            .chars()
            .rev()
            .collect::<String>()
    );
    if used.insert(base.clone()) {
        return base;
    }
    let mut index = 2;
    loop {
        let candidate = format!("{base}-{index}");
        if used.insert(candidate.clone()) {
            return candidate;
        }
        index += 1;
    }
}
fn sanitize(name: &str) -> String {
    let value = name
        .trim()
        .chars()
        .map(|character| {
            if character.is_alphanumeric() || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('_')
        .to_owned();
    (!value.is_empty())
        .then_some(value)
        .unwrap_or_else(|| "span".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_output_parses_a_string_encoded_genai_message_array() {
        let span = Span {
            trace_id: "trace".to_owned(),
            span_id: "span".to_owned(),
            parent_span_id: None,
            name: "replay".to_owned(),
            kind: 0,
            start_time_unix_nano: None,
            end_time_unix_nano: None,
            attributes: BTreeMap::from([(
                "gen_ai.output.messages".to_owned(),
                Value::String(
                    r#"[{\"role\":\"assistant\",\"content\":\"completed\"},42]"#.to_owned(),
                ),
            )]),
            is_streaming: false,
        };

        assert_eq!(
            replay_output(&span),
            Some(serde_json::json!([{"role":"assistant","content":"completed"}]))
        );
    }
}
