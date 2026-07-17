// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Higher-level conversation renderers: migration, deploy, security,
//! distributed, observability, db optimize, architecture review, and incident
//! response.
//!
//! Like the base conversations, each renderer joins `[User]`/`[Assistant]` turns
//! with `\n\n` and draws bridge phrases, embedded tool/config/error blocks, and
//! identifiers in their draw order.

use super::prompts_conv::{
    BRIDGE_ANALYZE, BRIDGE_ARCHITECTURE_TRADEOFF, BRIDGE_DATA_ARCHITECTURE, BRIDGE_DEPLOY,
    BRIDGE_DISTRIBUTED, BRIDGE_EXPLAIN, BRIDGE_FIX, BRIDGE_OBSERVABILITY, BRIDGE_PERF,
    BRIDGE_REFACTOR, BRIDGE_SECURITY, BRIDGE_SUMMARY, BRIDGE_TEST, BRIDGE_WRITE_TEST,
    FOLLOWUP_QUESTIONS, LANGUAGES, conv_bridge, conv_ids,
};
use super::templates::TemplateRenderer;
use super::vocab::{TABLES, lang_index};
use super::{cicd_docs, errors_diff, json_blocks, sql, tool, tool_long};
use crate::graph::recorded::RecordedTraceError;

/// Lowercase hex-digit choice alphabet.
const HEX_LOWER: &[&str] = &["a", "b", "c", "d", "e", "f"];
/// Decimal-digit choice alphabet.
const DIGITS: &[&str] = &["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

/// Multi-file migration conversation.
pub(super) fn conv_migration(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    let lang = Some(lang_index(r.pick(LANGUAGES)?));
    let ids = conv_ids(r)?;
    let cls = ids.cls;
    let method = ids.method;
    let module = ids.module;

    let search_verbose = tool_long::tool_search_verbose(r, lang)?;
    let b1 = conv_bridge(r, BRIDGE_ANALYZE, &ids)?;
    let read_long = tool_long::tool_read_long(r, lang)?;
    let b2 = conv_bridge(r, BRIDGE_ANALYZE, &ids)?;
    let read1 = tool::tool_read(r, lang)?;
    let b3 = conv_bridge(r, BRIDGE_ANALYZE, &ids)?;
    let read2 = tool::tool_read(r, lang)?;
    let edit1 = tool::tool_edit(r, lang)?;
    let edit2 = tool::tool_edit(r, lang)?;
    let edit3 = tool::tool_edit(r, lang)?;
    let edit4 = tool::tool_edit(r, lang)?;
    let b4 = conv_bridge(r, BRIDGE_TEST, &ids)?;
    let bash_verbose = tool_long::tool_bash_verbose(r, lang)?;
    let b5 = conv_bridge(r, FOLLOWUP_QUESTIONS, &ids)?;
    let b6 = conv_bridge(r, BRIDGE_EXPLAIN, &ids)?;
    let b7 = conv_bridge(r, BRIDGE_SUMMARY, &ids)?;

    let turns = vec![
        format!(
            "[User]\nMigrate {cls}.{method}() from sync to async. It's called across \
             multiple files in {module}. Update all callers and add backward compat."
        ),
        format!("[Assistant]\nLet me find all the callers first.\n\n{search_verbose}"),
        format!("[Assistant]\n{b1}\n\n{read_long}"),
        format!("[Assistant]\n{b2}\n\n{read1}"),
        format!("[Assistant]\n{b3}\n\n{read2}"),
        format!(
            "[Assistant]\nI'll start with the core change to {cls}, then update each \
             caller.\n\n{edit1}"
        ),
        format!("[Assistant]\nNow updating the first caller.\n\n{edit2}"),
        format!("[Assistant]\nUpdating the second caller.\n\n{edit3}"),
        format!(
            "[Assistant]\nUpdating the third caller and adding the backward-compat \
             wrapper.\n\n{edit4}"
        ),
        format!("[Assistant]\n{b4}\n\n{bash_verbose}"),
        format!("[User]\n{b5}"),
        format!("[Assistant]\n{b6}\n\n{b7}"),
    ];
    Ok(turns.join("\n\n"))
}

/// Deployment troubleshooting — inspect pod, config, logs,
/// fix, verify.
pub(super) fn conv_deploy(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    let lang = Some(lang_index(r.pick(LANGUAGES)?));
    let ids = conv_ids(r)?;
    let module = ids.module;

    let config_block = cicd_docs::config_file(r, lang)?;
    let json_resp = json_blocks::render(r)?;

    let b_deploy1 = conv_bridge(r, BRIDGE_DEPLOY, &ids)?;
    let n1 = r.number(1000, 9999)?;
    let c1 = r.pick(HEX_LOWER)?;
    let c2 = r.pick(HEX_LOWER)?;
    let d1 = r.pick(DIGITS)?;
    let c3 = r.pick(HEX_LOWER)?;
    let d2 = r.pick(DIGITS)?;
    let n2 = r.number(1000, 9999)?;
    let b_deploy2 = conv_bridge(r, BRIDGE_DEPLOY, &ids)?;
    let et = errors_diff::error_traceback(r, lang)?;
    let b_fix = conv_bridge(r, BRIDGE_FIX, &ids)?;
    let edit1 = tool::tool_edit(r, lang)?;
    let edit2 = tool::tool_edit(r, lang)?;
    let b_test = conv_bridge(r, BRIDGE_TEST, &ids)?;
    let b_summary = conv_bridge(r, BRIDGE_SUMMARY, &ids)?;

    let turns = vec![
        format!(
            "[User]\nThe {module} service keeps crashing after deploy. The health \
             check is failing and pods are in CrashLoopBackOff."
        ),
        format!(
            r#"[Assistant]
{b_deploy1}

<tool_name>bash</tool_name>
<parameter name="command">kubectl describe pod {module}-{n1}-{c1}{c2}{d1}{c3}{d2}</parameter>
<result>
Name:         {module}-deployment-{n2}
Namespace:    default
Status:       Running
Containers:
  {module}:
    Image:          registry.internal/{module}:latest
    State:          Waiting (CrashLoopBackOff)
    Last State:     Terminated (Error, exit code 1)
    Ready:          False
    Restart Count:  7
    Limits:
      cpu:     2
      memory:  512Mi
    Requests:
      cpu:     500m
      memory:  256Mi
    Liveness:   http-get http://:8080/health delay=10s timeout=3s period=5s
    Readiness:  http-get http://:8080/ready delay=5s timeout=3s period=5s
Events:
  Warning  BackOff  2m (x7 over 10m)  kubelet  Back-off restarting failed container
</result>"#
        ),
        format!(
            r#"[Assistant]
The memory limit looks too low. Let me check the config.

<tool_name>read</tool_name>
<parameter name="file_path">kubernetes/deployment.yaml</parameter>
<result>
{config_block}
</result>"#
        ),
        format!(
            r#"[Assistant]
{b_deploy2}

<tool_name>bash</tool_name>
<parameter name="command">kubectl logs deploy/{module} --tail=30</parameter>
<result>
{et}
</result>"#
        ),
        format!("[Assistant]\n{b_fix}\n\n{edit1}"),
        format!("[Assistant]\nLet me also increase the memory limits.\n\n{edit2}"),
        format!(
            r#"[Assistant]
{b_test}

<tool_name>bash</tool_name>
<parameter name="command">kubectl apply -f kubernetes/deployment.yaml && kubectl rollout status deploy/{module} --timeout=120s</parameter>
<result>
deployment.apps/{module} configured
Waiting for deployment "{module}" rollout to finish: 1 old replicas are pending termination...
deployment "{module}" successfully rolled out
</result>"#
        ),
        format!(
            r#"[Assistant]
Let me verify the health check is passing now.

<tool_name>bash</tool_name>
<parameter name="command">curl -s http://localhost:8080/health | python -m json.tool</parameter>
<result>
{json_resp}
</result>"#
        ),
        format!("[Assistant]\n{b_summary}"),
    ];
    Ok(turns.join("\n\n"))
}

/// Vulnerability investigation — find, analyze, fix, test.
pub(super) fn conv_security(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    let lang = Some(lang_index(r.pick(LANGUAGES)?));
    let ids = conv_ids(r)?;
    let module = ids.module;
    let method = ids.method;
    let var = ids.var;

    let b1 = conv_bridge(r, BRIDGE_SECURITY, &ids)?;
    let read_long = tool_long::tool_read_long(r, lang)?;
    let b2 = conv_bridge(r, BRIDGE_ANALYZE, &ids)?;
    let search_verbose = tool_long::tool_search_verbose(r, lang)?;
    let b3 = conv_bridge(r, BRIDGE_ARCHITECTURE_TRADEOFF, &ids)?;
    let b4 = conv_bridge(r, BRIDGE_SECURITY, &ids)?;
    let edit1 = tool::tool_edit(r, lang)?;
    let b5 = conv_bridge(r, BRIDGE_WRITE_TEST, &ids)?;
    let edit2 = tool::tool_edit(r, lang)?;
    let b6 = conv_bridge(r, BRIDGE_TEST, &ids)?;
    let bash_verbose = tool_long::tool_bash_verbose(r, lang)?;
    let b7 = conv_bridge(r, BRIDGE_SUMMARY, &ids)?;

    let turns = [
        format!(
            "[User]\nI think there's a security vulnerability in the {module} service. \
             The {method}() endpoint accepts user input for {var} without proper validation."
        ),
        format!("[Assistant]\n{b1}\n\n{read_long}"),
        format!("[Assistant]\n{b2}\n\n{search_verbose}"),
        format!("[Assistant]\n{b3}"),
        format!("[Assistant]\n{b4}\n\n{edit1}"),
        format!("[Assistant]\n{b5}\n\n{edit2}"),
        format!("[Assistant]\n{b6}\n\n{bash_verbose}"),
        format!("[Assistant]\n{b7}"),
    ];
    Ok(turns.join("\n\n"))
}

/// Distributed-systems debugging — inconsistency,
/// analyze replication, fix consensus.
pub(super) fn conv_distributed(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    let lang = Some(lang_index(r.pick(LANGUAGES)?));
    let ids = conv_ids(r)?;
    let module = ids.module;
    let var = ids.var;
    let cls = ids.cls;
    let method = ids.method;

    let config_block = cicd_docs::config_file(r, lang)?;

    let b1 = conv_bridge(r, BRIDGE_DISTRIBUTED, &ids)?;
    let b2 = conv_bridge(r, BRIDGE_ANALYZE, &ids)?;
    let search_verbose = tool_long::tool_search_verbose(r, lang)?;
    let b3 = conv_bridge(r, BRIDGE_ARCHITECTURE_TRADEOFF, &ids)?;
    let b4 = conv_bridge(r, BRIDGE_DISTRIBUTED, &ids)?;
    let edit = tool::tool_edit(r, lang)?;
    let b5 = conv_bridge(r, BRIDGE_TEST, &ids)?;
    let bash_verbose = tool_long::tool_bash_verbose(r, lang)?;
    let b6 = conv_bridge(r, FOLLOWUP_QUESTIONS, &ids)?;
    let b7 = conv_bridge(r, BRIDGE_DISTRIBUTED, &ids)?;
    let b8 = conv_bridge(r, BRIDGE_SUMMARY, &ids)?;

    let turns = [
        format!(
            "[User]\nThere are inconsistent reads across replicas in the {module} \
             service. After writing to {var} via {cls}.{method}(), some replicas \
             return stale data."
        ),
        format!(
            r#"[Assistant]
{b1}

<tool_name>read</tool_name>
<parameter name="file_path">config/replication.yaml</parameter>
<result>
{config_block}
</result>"#
        ),
        format!("[Assistant]\n{b2}\n\n{search_verbose}"),
        format!("[Assistant]\n{b3}"),
        format!("[Assistant]\n{b4}\n\n{edit}"),
        format!("[Assistant]\n{b5}\n\n{bash_verbose}"),
        format!("[User]\n{b6}"),
        format!("[Assistant]\n{b7}\n\n{b8}"),
    ];
    Ok(turns.join("\n\n"))
}

/// Observability gap — add tracing, metrics,
/// structured logging.
pub(super) fn conv_observability(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    let lang = Some(lang_index(r.pick(LANGUAGES)?));
    let ids = conv_ids(r)?;
    let module = ids.module;
    let cls = ids.cls;
    let method = ids.method;

    let config_block = cicd_docs::config_file(r, lang)?;

    let b1 = conv_bridge(r, BRIDGE_ANALYZE, &ids)?;
    let read_long = tool_long::tool_read_long(r, lang)?;
    let b2 = conv_bridge(r, BRIDGE_OBSERVABILITY, &ids)?;
    let search_verbose = tool_long::tool_search_verbose(r, lang)?;
    let b3 = conv_bridge(r, BRIDGE_OBSERVABILITY, &ids)?;
    let edit1 = tool::tool_edit(r, lang)?;
    let b4 = conv_bridge(r, BRIDGE_OBSERVABILITY, &ids)?;
    let edit2 = tool::tool_edit(r, lang)?;
    let b5 = conv_bridge(r, BRIDGE_TEST, &ids)?;
    let json_resp = json_blocks::render(r)?;
    let b6 = conv_bridge(r, BRIDGE_SUMMARY, &ids)?;

    let turns = [
        format!(
            "[User]\nCan't debug a production latency spike in {module}. There's no \
             tracing or metrics on {cls}.{method}()."
        ),
        format!("[Assistant]\n{b1}\n\n{read_long}"),
        format!("[Assistant]\n{b2}\n\n{search_verbose}"),
        format!("[Assistant]\n{b3}\n\n{edit1}"),
        format!("[Assistant]\n{b4}\n\n{edit2}"),
        format!(
            r#"[Assistant]
Let me also add the telemetry configuration.

<tool_name>read</tool_name>
<parameter name="file_path">config/telemetry.yaml</parameter>
<result>
{config_block}
</result>"#
        ),
        format!(
            r#"[Assistant]
{b5}

<tool_name>bash</tool_name>
<parameter name="command">curl -s http://localhost:8080/metrics | head -20</parameter>
<result>
{json_resp}
</result>"#
        ),
        format!("[Assistant]\n{b6}"),
    ];
    Ok(turns.join("\n\n"))
}

/// Database optimization — EXPLAIN, read ORM code, add
/// index, benchmark.
pub(super) fn conv_db_optimize(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    let lang = Some(lang_index(r.pick(LANGUAGES)?));
    let ids = conv_ids(r)?;
    let table = r.pick(TABLES)?;
    let sql_block = sql::query(r)?;
    let method = ids.method;
    let var = ids.var;

    let b1 = conv_bridge(r, BRIDGE_DATA_ARCHITECTURE, &ids)?;
    let read_long = tool_long::tool_read_long(r, lang)?;
    let b2 = conv_bridge(r, BRIDGE_ARCHITECTURE_TRADEOFF, &ids)?;
    let b3 = conv_bridge(r, BRIDGE_DATA_ARCHITECTURE, &ids)?;
    let edit = tool::tool_edit(r, lang)?;
    let b4 = conv_bridge(r, BRIDGE_TEST, &ids)?;
    let bash_verbose = tool_long::tool_bash_verbose(r, lang)?;
    let b5 = conv_bridge(r, BRIDGE_ARCHITECTURE_TRADEOFF, &ids)?;
    let b6 = conv_bridge(r, BRIDGE_SUMMARY, &ids)?;

    let turns = [
        format!(
            "[User]\nThe {method}() query on the {table} table is taking over 5 seconds \
             in production. Can you optimize it?"
        ),
        format!(
            r#"[Assistant]
Let me run EXPLAIN ANALYZE to see the query plan.

<tool_name>bash</tool_name>
<parameter name="command">psql -d mydb -c "EXPLAIN ANALYZE SELECT * FROM {table} WHERE {var} = 'test'"</parameter>
<result>
{sql_block}
</result>"#
        ),
        format!("[Assistant]\n{b1}\n\n{read_long}"),
        format!("[Assistant]\n{b2}"),
        format!("[Assistant]\n{b3}\n\n{edit}"),
        format!("[Assistant]\n{b4}\n\n{bash_verbose}"),
        format!("[User]\nShould we also partition the {table} table?"),
        format!("[Assistant]\n{b5}\n\n{b6}"),
    ];
    Ok(turns.join("\n\n"))
}

/// Architecture-review conversation.
pub(super) fn conv_architecture_review(
    r: &mut TemplateRenderer,
) -> Result<String, RecordedTraceError> {
    let lang = Some(lang_index(r.pick(LANGUAGES)?));
    let ids = conv_ids(r)?;
    let module = ids.module;
    let cls = ids.cls;

    let b1 = conv_bridge(r, BRIDGE_ANALYZE, &ids)?;
    let read_long1 = tool_long::tool_read_long(r, lang)?;
    let b2 = conv_bridge(r, BRIDGE_ANALYZE, &ids)?;
    let read_long2 = tool_long::tool_read_long(r, lang)?;
    let b3 = conv_bridge(r, BRIDGE_ANALYZE, &ids)?;
    let search_verbose = tool_long::tool_search_verbose(r, lang)?;
    let b4 = conv_bridge(r, BRIDGE_ARCHITECTURE_TRADEOFF, &ids)?;
    let b5 = conv_bridge(r, BRIDGE_ARCHITECTURE_TRADEOFF, &ids)?;
    let b6 = conv_bridge(r, BRIDGE_REFACTOR, &ids)?;
    let edit1 = tool::tool_edit(r, lang)?;
    let b7 = conv_bridge(r, BRIDGE_ARCHITECTURE_TRADEOFF, &ids)?;
    let b8 = conv_bridge(r, BRIDGE_PERF, &ids)?;
    let b9 = conv_bridge(r, BRIDGE_FIX, &ids)?;
    let edit2 = tool::tool_edit(r, lang)?;
    let b10 = conv_bridge(r, BRIDGE_TEST, &ids)?;
    let bash_verbose = tool_long::tool_bash_verbose(r, lang)?;
    let b11 = conv_bridge(r, BRIDGE_SUMMARY, &ids)?;

    let turns = vec![
        format!(
            "[User]\nCan you do an architecture review of the {module} service? I'm \
             concerned about coupling and scalability."
        ),
        format!("[Assistant]\n{b1}\n\n{read_long1}"),
        format!("[Assistant]\n{b2}\n\n{read_long2}"),
        format!("[Assistant]\n{b3}\n\n{search_verbose}"),
        format!("[Assistant]\n{b4}\n\n{b5}"),
        format!("[Assistant]\n{b6}\n\n{edit1}"),
        format!(
            "[User]\nWhat about the scalability of {cls}? Will this approach hold up \
             under 10x traffic?"
        ),
        format!("[Assistant]\n{b7}\n\n{b8}"),
        format!("[Assistant]\n{b9}\n\n{edit2}"),
        format!("[Assistant]\n{b10}\n\n{bash_verbose}"),
        format!("[Assistant]\n{b11}"),
    ];
    Ok(turns.join("\n\n"))
}

/// Production incident — diagnose, fix, add
/// monitoring, post-mortem.
pub(super) fn conv_incident_response(
    r: &mut TemplateRenderer,
) -> Result<String, RecordedTraceError> {
    let lang = Some(lang_index(r.pick(LANGUAGES)?));
    let ids = conv_ids(r)?;
    let module = ids.module;
    let error = ids.error;
    let cls = ids.cls;
    let method = ids.method;

    let config_block = cicd_docs::config_file(r, lang)?;
    let error_block = errors_diff::error_traceback(r, lang)?;

    let b1 = conv_bridge(r, BRIDGE_DEPLOY, &ids)?;
    let b2 = conv_bridge(r, BRIDGE_ARCHITECTURE_TRADEOFF, &ids)?;
    let b3 = conv_bridge(r, BRIDGE_FIX, &ids)?;
    let edit1 = tool::tool_edit(r, lang)?;
    let edit2 = tool::tool_edit(r, lang)?;
    let b4 = conv_bridge(r, BRIDGE_OBSERVABILITY, &ids)?;
    let edit3 = tool::tool_edit(r, lang)?;
    let b5 = conv_bridge(r, BRIDGE_TEST, &ids)?;
    let bash_verbose = tool_long::tool_bash_verbose(r, lang)?;

    let turns = vec![
        format!(
            "[User]\nProduction incident: the {module} service is down and it's causing \
             cascading failures in downstream services."
        ),
        r#"[Assistant]
Let me check the service health immediately.

<tool_name>bash</tool_name>
<parameter name="command">curl -s http://localhost:8080/health || echo 'Connection refused'</parameter>
<result>
Connection refused
</result>"#
            .to_string(),
        format!(
            r#"[Assistant]
{b1}

<tool_name>read</tool_name>
<parameter name="file_path">kubernetes/deployment.yaml</parameter>
<result>
{config_block}
</result>"#
        ),
        format!(
            r#"[Assistant]
Let me check the logs for the root cause.

<tool_name>bash</tool_name>
<parameter name="command">kubectl logs deploy/{module} --tail=50</parameter>
<result>
{error_block}
</result>"#
        ),
        format!("[Assistant]\n{b2}"),
        format!("[Assistant]\n{b3}\n\n{edit1}"),
        format!(
            "[Assistant]\nNow let me add a circuit breaker to prevent cascading \
             failures.\n\n{edit2}"
        ),
        format!("[Assistant]\n{b4}\n\n{edit3}"),
        format!("[Assistant]\n{b5}\n\n{bash_verbose}"),
        format!(
            "[Assistant]\nPost-mortem summary: The {module} service experienced a \
             cascading failure triggered by {error}. The root cause was \
             {cls}.{method}() not handling the error gracefully, which caused the \
             health check to fail and pods to restart in a loop. Fixes applied: error \
             handling in {method}(), circuit breaker pattern for downstream calls, and \
             Prometheus alerts for early detection."
        ),
    ];
    Ok(turns.join("\n\n"))
}
