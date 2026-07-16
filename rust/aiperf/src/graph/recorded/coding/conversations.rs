// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Base multi-turn conversation renderers.
//!
//! Each renderer stitches `[User]`/`[Assistant]` turns together with `\n\n`
//! (Python `"\n\n".join(turns)`), drawing bridge phrases and per-language tool
//! blocks in the exact order the Python f-string list evaluates them so the
//! seeded RNG stream stays byte-for-byte aligned. The conversation's chosen
//! language (`r.choice(_LANGUAGES)`) is threaded into every per-language
//! sub-generator.

use super::prompts_conv::{
    BRIDGE_ANALYZE, BRIDGE_EXPLAIN, BRIDGE_FIX, BRIDGE_PERF, BRIDGE_REFACTOR, BRIDGE_SUMMARY,
    BRIDGE_TEST, BRIDGE_WRITE_TEST, FOLLOWUP_QUESTIONS, LANGUAGES, conv_bridge, conv_ids,
    conv_user_msg,
};
use super::templates::TemplateRenderer;
use super::vocab::lang_index;
use super::{cicd_docs, errors_diff, ml, tool, tool_long};
use crate::graph::recorded::RecordedTraceError;

/// `_gen_conv_bugfix`: read → fix → test → summarize.
pub(super) fn conv_bugfix(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    let lang = Some(lang_index(r.pick(LANGUAGES)?));
    let ids = conv_ids(r)?;

    let user = conv_user_msg(r, &ids)?;
    let b_analyze = conv_bridge(r, BRIDGE_ANALYZE, &ids)?;
    let read_long = tool_long::tool_read_long(r, lang)?;
    let b_fix = conv_bridge(r, BRIDGE_FIX, &ids)?;
    let edit = tool::tool_edit(r, lang)?;
    let b_test = conv_bridge(r, BRIDGE_TEST, &ids)?;
    let bash = tool::tool_bash(r, lang)?;
    let b_summary = conv_bridge(r, BRIDGE_SUMMARY, &ids)?;

    let turns = [
        format!("[User]\n{user}"),
        format!("[Assistant]\n{b_analyze}\n\n{read_long}"),
        format!("[Assistant]\n{b_fix}\n\n{edit}"),
        format!("[Assistant]\n{b_test}\n\n{bash}"),
        format!("[Assistant]\n{b_summary}"),
    ];
    Ok(turns.join("\n\n"))
}

/// `_gen_conv_review`: diff → read → fix → follow-up question.
pub(super) fn conv_review(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    let lang = Some(lang_index(r.pick(LANGUAGES)?));
    let ids = conv_ids(r)?;

    let user = conv_user_msg(r, &ids)?;
    let git_diff = errors_diff::git_diff(r, lang)?;
    let b_analyze = conv_bridge(r, BRIDGE_ANALYZE, &ids)?;
    let read_long = tool_long::tool_read_long(r, lang)?;
    let b_fix = conv_bridge(r, BRIDGE_FIX, &ids)?;
    let edit = tool::tool_edit(r, lang)?;
    let b_followup = conv_bridge(r, FOLLOWUP_QUESTIONS, &ids)?;

    let turns = [
        format!("[User]\n{user}\n\n{git_diff}"),
        format!("[Assistant]\n{b_analyze}\n\n{read_long}"),
        format!("[Assistant]\n{b_fix}\n\n{edit}"),
        format!("[User]\n{b_followup}"),
    ];
    Ok(turns.join("\n\n"))
}

/// `_gen_conv_feature`: search → read → implement → write test → run test.
pub(super) fn conv_feature(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    let lang = Some(lang_index(r.pick(LANGUAGES)?));
    let ids = conv_ids(r)?;

    let user = conv_user_msg(r, &ids)?;
    let b1 = conv_bridge(r, BRIDGE_ANALYZE, &ids)?;
    let search_verbose = tool_long::tool_search_verbose(r, lang)?;
    let b2 = conv_bridge(r, BRIDGE_ANALYZE, &ids)?;
    let read_long = tool_long::tool_read_long(r, lang)?;
    let b3 = conv_bridge(r, BRIDGE_FIX, &ids)?;
    let edit1 = tool::tool_edit(r, lang)?;
    let b4 = conv_bridge(r, BRIDGE_WRITE_TEST, &ids)?;
    let edit2 = tool::tool_edit(r, lang)?;
    let b5 = conv_bridge(r, BRIDGE_TEST, &ids)?;
    let bash_verbose = tool_long::tool_bash_verbose(r, lang)?;

    let turns = [
        format!("[User]\n{user}"),
        format!("[Assistant]\n{b1}\n\n{search_verbose}"),
        format!("[Assistant]\n{b2}\n\n{read_long}"),
        format!("[Assistant]\n{b3}\n\n{edit1}"),
        format!("[Assistant]\n{b4}\n\n{edit2}"),
        format!("[Assistant]\n{b5}\n\n{bash_verbose}"),
    ];
    Ok(turns.join("\n\n"))
}

/// `_gen_conv_debug`: error report → read → search → fix → summarize.
pub(super) fn conv_debug(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    let lang = Some(lang_index(r.pick(LANGUAGES)?));
    let ids = conv_ids(r)?;
    // choice([lambda error_traceback(lang), cuda_error]): index 0 = traceback.
    let error_block = match r.index(2)? {
        0 => errors_diff::error_traceback(r, lang)?,
        _ => ml::cuda_error(r)?,
    };

    let user = conv_user_msg(r, &ids)?;
    let b1 = conv_bridge(r, BRIDGE_ANALYZE, &ids)?;
    let read_long = tool_long::tool_read_long(r, lang)?;
    let b2 = conv_bridge(r, BRIDGE_ANALYZE, &ids)?;
    let search_verbose = tool_long::tool_search_verbose(r, lang)?;
    let b3 = conv_bridge(r, BRIDGE_FIX, &ids)?;
    let edit = tool::tool_edit(r, lang)?;
    let b4 = conv_bridge(r, BRIDGE_SUMMARY, &ids)?;

    let turns = [
        format!("[User]\n{user}\n\n{error_block}"),
        format!("[Assistant]\n{b1}\n\n{read_long}"),
        format!("[Assistant]\n{b2}\n\n{search_verbose}"),
        format!("[Assistant]\n{b3}\n\n{edit}"),
        format!("[Assistant]\n{b4}"),
    ];
    Ok(turns.join("\n\n"))
}

/// `_gen_conv_qa`: read → explain → follow-up → apply.
pub(super) fn conv_qa(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    let lang = Some(lang_index(r.pick(LANGUAGES)?));
    let ids = conv_ids(r)?;

    let user = conv_user_msg(r, &ids)?;
    let b1 = conv_bridge(r, BRIDGE_ANALYZE, &ids)?;
    let read_long = tool_long::tool_read_long(r, lang)?;
    let b2 = conv_bridge(r, BRIDGE_EXPLAIN, &ids)?;
    let b3 = conv_bridge(r, FOLLOWUP_QUESTIONS, &ids)?;
    let b4 = conv_bridge(r, BRIDGE_FIX, &ids)?;
    let edit = tool::tool_edit(r, lang)?;

    let turns = [
        format!("[User]\n{user}"),
        format!("[Assistant]\n{b1}\n\n{read_long}"),
        format!("[Assistant]\n{b2}"),
        format!("[User]\n{b3}"),
        format!("[Assistant]\n{b4}\n\n{edit}"),
    ];
    Ok(turns.join("\n\n"))
}

/// `_gen_conv_refactor`: multi-file refactoring — search callers, read multiple
/// files, edit each.
pub(super) fn conv_refactor(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    let lang = Some(lang_index(r.pick(LANGUAGES)?));
    let ids = conv_ids(r)?;

    let user = conv_user_msg(r, &ids)?;
    let b1 = conv_bridge(r, BRIDGE_ANALYZE, &ids)?;
    let search_verbose = tool_long::tool_search_verbose(r, lang)?;
    let b2 = conv_bridge(r, BRIDGE_ANALYZE, &ids)?;
    let read_long = tool_long::tool_read_long(r, lang)?;
    let b3 = conv_bridge(r, BRIDGE_ANALYZE, &ids)?;
    let read = tool::tool_read(r, lang)?;
    let b4 = conv_bridge(r, BRIDGE_REFACTOR, &ids)?;
    let edit1 = tool::tool_edit(r, lang)?;
    let edit2 = tool::tool_edit(r, lang)?;
    let b5 = conv_bridge(r, BRIDGE_REFACTOR, &ids)?;
    let edit3 = tool::tool_edit(r, lang)?;
    let b6 = conv_bridge(r, BRIDGE_TEST, &ids)?;
    let bash_verbose = tool_long::tool_bash_verbose(r, lang)?;
    let b7 = conv_bridge(r, BRIDGE_SUMMARY, &ids)?;

    let turns = vec![
        format!("[User]\n{user}"),
        format!("[Assistant]\n{b1}\n\n{search_verbose}"),
        format!("[Assistant]\n{b2}\n\n{read_long}"),
        format!("[Assistant]\n{b3}\n\n{read}"),
        format!("[Assistant]\n{b4}\n\n{edit1}"),
        format!("[Assistant]\nNow let me update the callers.\n\n{edit2}"),
        format!("[Assistant]\n{b5}\n\n{edit3}"),
        format!("[Assistant]\n{b6}\n\n{bash_verbose}"),
        format!("[Assistant]\n{b7}"),
    ];
    Ok(turns.join("\n\n"))
}

/// `_gen_conv_perf`: performance investigation — profile, read hot path,
/// optimize, benchmark.
pub(super) fn conv_perf(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    let lang = Some(lang_index(r.pick(LANGUAGES)?));
    let ids = conv_ids(r)?;

    let user = conv_user_msg(r, &ids)?;
    let b1 = conv_bridge(r, BRIDGE_PERF, &ids)?;
    let bash = tool::tool_bash(r, lang)?;
    let b2 = conv_bridge(r, BRIDGE_ANALYZE, &ids)?;
    let read_long = tool_long::tool_read_long(r, lang)?;
    let b3 = conv_bridge(r, BRIDGE_PERF, &ids)?;
    let search = tool::tool_search(r, lang)?;
    let b4 = conv_bridge(r, BRIDGE_FIX, &ids)?;
    let edit = tool::tool_edit(r, lang)?;
    let b5 = conv_bridge(r, BRIDGE_TEST, &ids)?;
    let bash_verbose = tool_long::tool_bash_verbose(r, lang)?;
    let b6 = conv_bridge(r, FOLLOWUP_QUESTIONS, &ids)?;
    let b7 = conv_bridge(r, BRIDGE_EXPLAIN, &ids)?;
    let b8 = conv_bridge(r, BRIDGE_SUMMARY, &ids)?;

    let turns = [
        format!("[User]\n{user}"),
        format!("[Assistant]\n{b1}\n\n{bash}"),
        format!("[Assistant]\n{b2}\n\n{read_long}"),
        format!("[Assistant]\n{b3}\n\n{search}"),
        format!("[Assistant]\n{b4}\n\n{edit}"),
        format!("[Assistant]\n{b5}\n\n{bash_verbose}"),
        format!("[User]\n{b6}"),
        format!("[Assistant]\n{b7}\n\n{b8}"),
    ];
    Ok(turns.join("\n\n"))
}

/// `_gen_conv_cicd`: CI/CD debugging — failing pipeline, read logs, fix config,
/// re-run.
pub(super) fn conv_cicd(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    let lang = Some(lang_index(r.pick(LANGUAGES)?));
    let ids = conv_ids(r)?;
    let ci_output = cicd_docs::cicd_output(r, lang)?;
    let module = ids.module;

    let b1 = conv_bridge(r, BRIDGE_ANALYZE, &ids)?;
    let read1 = tool::tool_read(r, lang)?;
    let b2 = conv_bridge(r, BRIDGE_ANALYZE, &ids)?;
    let read2 = tool::tool_read(r, lang)?;
    let b3 = conv_bridge(r, BRIDGE_FIX, &ids)?;
    let edit = tool::tool_edit(r, lang)?;
    let b4 = conv_bridge(r, BRIDGE_TEST, &ids)?;
    let bash_verbose = tool_long::tool_bash_verbose(r, lang)?;
    let b5 = conv_bridge(r, BRIDGE_SUMMARY, &ids)?;
    let b6 = conv_bridge(r, FOLLOWUP_QUESTIONS, &ids)?;
    let b7 = conv_bridge(r, BRIDGE_EXPLAIN, &ids)?;

    let turns = [
        format!(
            "[User]\nThe CI pipeline is failing on the {module} service. \
             Can you take a look?\n\n{ci_output}"
        ),
        format!("[Assistant]\n{b1}\n\n{read1}"),
        format!("[Assistant]\n{b2}\n\n{read2}"),
        format!("[Assistant]\n{b3}\n\n{edit}"),
        format!("[Assistant]\n{b4}\n\n{bash_verbose}"),
        format!("[Assistant]\n{b5}"),
        format!("[User]\n{b6}"),
        format!("[Assistant]\n{b7}"),
    ];
    Ok(turns.join("\n\n"))
}

/// `_gen_conv_ml_debug`: ML/GPU debugging — CUDA error, read training code, fix,
/// re-run. This renderer never draws a language; the edit is always Python.
pub(super) fn conv_ml_debug(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    let ids = conv_ids(r)?;

    let cuda_err = ml::cuda_error(r)?;
    let training_code = ml::training_code(r)?;
    let training_log = ml::training_log(r)?;
    let inference_code = ml::inference_code(r)?;

    let b1 = conv_bridge(r, BRIDGE_ANALYZE, &ids)?;
    let b2 = conv_bridge(r, BRIDGE_ANALYZE, &ids)?;
    let b3 = conv_bridge(r, BRIDGE_FIX, &ids)?;
    let edit = tool::tool_edit(r, Some(0))?;
    let b4 = conv_bridge(r, BRIDGE_TEST, &ids)?;
    let b5 = conv_bridge(r, BRIDGE_SUMMARY, &ids)?;
    let b6 = conv_bridge(r, BRIDGE_ANALYZE, &ids)?;
    let b7 = conv_bridge(r, BRIDGE_EXPLAIN, &ids)?;
    let b8 = conv_bridge(r, BRIDGE_SUMMARY, &ids)?;

    let turns = [
        format!(
            "[User]\nI'm getting a CUDA error during training. Here's the error:\n\n{cuda_err}"
        ),
        format!(
            "[Assistant]\n{b1}\n\n\
             <tool_name>read</tool_name>\n\
             <parameter name=\"file_path\">train.py</parameter>\n\
             <result>\n{training_code}\n</result>"
        ),
        format!(
            "[Assistant]\n{b2}\n\n\
             <tool_name>read</tool_name>\n\
             <parameter name=\"file_path\">inference.py</parameter>\n\
             <result>\n{inference_code}\n</result>"
        ),
        format!("[Assistant]\n{b3}\n\n{edit}"),
        format!(
            "[Assistant]\n{b4}\n\n\
             <tool_name>bash</tool_name>\n\
             <parameter name=\"command\">python train.py --max-steps 10</parameter>\n\
             <result>\n{training_log}\n</result>"
        ),
        format!("[Assistant]\n{b5}"),
        "[User]\nCan you also check if the inference script has the same issue?".to_string(),
        format!("[Assistant]\n{b6}\n\n{b7}\n\n{b8}"),
    ];
    Ok(turns.join("\n\n"))
}

/// `_gen_conv_test_write`: test-writing session — read code, write tests,
/// iterate on failures.
pub(super) fn conv_test_write(r: &mut TemplateRenderer) -> Result<String, RecordedTraceError> {
    let lang = Some(lang_index(r.pick(LANGUAGES)?));
    let ids = conv_ids(r)?;
    let cls = ids.cls;
    let method = ids.method;

    let b1 = conv_bridge(r, BRIDGE_ANALYZE, &ids)?;
    let read_long = tool_long::tool_read_long(r, lang)?;
    let b2 = conv_bridge(r, BRIDGE_ANALYZE, &ids)?;
    let search = tool::tool_search(r, lang)?;
    let b3 = conv_bridge(r, BRIDGE_WRITE_TEST, &ids)?;
    let edit1 = tool::tool_edit(r, lang)?;
    let b4 = conv_bridge(r, BRIDGE_TEST, &ids)?;
    let bash_verbose = tool_long::tool_bash_verbose(r, lang)?;
    let b5 = conv_bridge(r, FOLLOWUP_QUESTIONS, &ids)?;
    let b6 = conv_bridge(r, BRIDGE_WRITE_TEST, &ids)?;
    let edit2 = tool::tool_edit(r, lang)?;
    let b7 = conv_bridge(r, BRIDGE_TEST, &ids)?;
    let bash = tool::tool_bash(r, lang)?;
    let b8 = conv_bridge(r, BRIDGE_SUMMARY, &ids)?;

    let turns = vec![
        format!(
            "[User]\nWrite comprehensive tests for {cls}.{method}(). \
             Cover the happy path, edge cases, and error handling."
        ),
        format!("[Assistant]\n{b1}\n\n{read_long}"),
        format!("[Assistant]\n{b2}\n\n{search}"),
        format!("[Assistant]\n{b3}\n\n{edit1}"),
        format!("[Assistant]\n{b4}\n\n{bash_verbose}"),
        format!("[User]\n{b5}"),
        format!("[Assistant]\n{b6}\n\n{edit2}"),
        format!("[Assistant]\n{b7}\n\n{bash}"),
        format!("[Assistant]\n{b8}"),
    ];
    Ok(turns.join("\n\n"))
}
