// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Isolated native orchestration for LiveCodeBench Python execution.
//!
//! The payload/test semantics are ported from
//! `src/aiperf/accuracy/graders/code_execution.py:1-362`. Execution is an
//! injectable trait; the built Linux implementation uses bubblewrap with no
//! network or workspace mount plus `prlimit` CPU/address-space/process limits.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Duration;

use aiperf_metrics::GradingResult;
use async_trait::async_trait;
use base64::Engine;
use flate2::read::ZlibDecoder;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

use super::Grader;
use crate::AccuracyError;

const PYTHON_RUNNER: &str = r#"
import ast
import builtins
import io
import json
import os
import shutil
import signal
import subprocess
import sys
from decimal import Decimal

IMPORTS = '''from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json
sys.setrecursionlimit(50000)
'''

class TimeoutException(Exception):
    pass

def timeout_handler(_signum, _frame):
    raise TimeoutException("alarm went off")

def reliability_guard():
    os.environ["OMP_NUM_THREADS"] = "1"
    for name in ("kill", "system", "putenv", "remove", "removedirs", "rmdir",
                 "fchdir", "setuid", "fork", "forkpty", "killpg", "rename",
                 "renames", "truncate", "replace", "unlink", "fchmod", "fchown",
                 "chmod", "chown", "chroot", "lchflags", "lchmod", "lchown"):
        if hasattr(os, name):
            setattr(os, name, None)
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None
    subprocess.Popen = None
    builtins.quit = None
    builtins.help = None

def clean_if_name(code):
    try:
        tree = ast.parse(code)
        last = tree.body[-1]
        if isinstance(last, ast.If) and ast.unparse(last.test).strip() == "__name__ == '__main__'":
            code = ast.unparse(tree.body[:-1]) + "\n" + ast.unparse(last.body)
    except Exception:
        pass
    return code

def make_function(code):
    try:
        imports = []
        body = []
        tree = ast.parse(code)
        for statement in tree.body:
            (imports if isinstance(statement, (ast.Import, ast.ImportFrom)) else body).append(statement)
        function = ast.FunctionDef(
            name="wrapped_function",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=body,
            decorator_list=[],
            lineno=1,
            col_offset=0,
        )
        ast.fix_missing_locations(function)
        return IMPORTS + "\n" + ast.unparse(imports) + "\n" + ast.unparse(function)
    except Exception:
        return code

def compile_callable(code, fn_name, timeout):
    signal.alarm(timeout)
    try:
        namespace = {"__name__": "candidate"}
        if fn_name:
            exec(compile(IMPORTS + "\n\n" + code, "<candidate>", "exec"), namespace, namespace)
            owner = namespace["Solution"]() if "class Solution" in code else namespace
            method = getattr(owner, fn_name) if not isinstance(owner, dict) else owner.get(fn_name)
        else:
            exec(compile(make_function(clean_if_name(code)), "<candidate>", "exec"), namespace, namespace)
            method = namespace.get("wrapped_function")
        if method is None:
            raise AttributeError("candidate has no function " + (fn_name or "wrapped_function"))
        return method
    finally:
        signal.alarm(0)

def call_inputs(raw):
    if not isinstance(raw, str):
        raise TypeError("call-based input must be a newline-delimited JSON string")
    return [json.loads(line) for line in raw.split("\n")]

def expected_call_output(raw):
    return json.loads(raw) if isinstance(raw, str) else raw

def run_call_case(method, case):
    prediction = method(*call_inputs(case.get("input")))
    if isinstance(prediction, tuple):
        prediction = list(prediction)
    return prediction == expected_call_output(case.get("output"))

def stripped_lines(value):
    return [line.strip() for line in str(value).strip().split("\n")]

def decimal_line(value):
    try:
        return [Decimal(item) for item in value.split()]
    except Exception:
        return None

def stdout_equal(actual, expected):
    predicted = stripped_lines(actual)
    expected = stripped_lines(expected)
    if len(predicted) != len(expected):
        return False
    for predicted_line, expected_line in zip(predicted, expected):
        if predicted_line == expected_line:
            continue
        predicted_decimal = decimal_line(predicted_line)
        expected_decimal = decimal_line(expected_line)
        if predicted_decimal is None or expected_decimal is None or predicted_decimal != expected_decimal:
            return False
    return True

def run_stdio_case(method, case):
    raw = case.get("input", "")
    if isinstance(raw, list):
        raw = "\n".join(raw)
    raw = str(raw)
    old_stdin, old_stdout, old_open = sys.stdin, sys.stdout, builtins.open
    capture = io.StringIO()
    try:
        sys.stdin = io.StringIO(raw)
        sys.stdout = capture
        builtins.open = lambda *_args, **_kwargs: io.StringIO(raw)
        try:
            method()
        except SystemExit:
            pass
        return stdout_equal(capture.getvalue(), case.get("output", ""))
    finally:
        sys.stdin, sys.stdout, builtins.open = old_stdin, old_stdout, old_open

try:
    request = json.load(sys.stdin)
    code = request["code"]
    cases = request["cases"]
    fn_name = request.get("fn_name")
    timeout = max(1, int(request.get("timeout_seconds", 6)))
    signal.signal(signal.SIGALRM, timeout_handler)
    reliability_guard()
    method = compile_callable(code, fn_name, timeout)
    for index, case in enumerate(cases):
        signal.alarm(timeout)
        try:
            passed = run_call_case(method, case) if fn_name else run_stdio_case(method, case)
            if not passed:
                print(json.dumps({"passed": False, "failed_case": index, "error": "output mismatch"}))
                break
        except BaseException as error:
            print(json.dumps({"passed": False, "failed_case": index, "error": type(error).__name__ + ": " + str(error)}))
            break
        finally:
            signal.alarm(0)
    else:
        print(json.dumps({"passed": True, "failed_case": None, "error": None}))
except BaseException as error:
    signal.alarm(0)
    print(json.dumps({"passed": False, "failed_case": None, "error": "runner: " + type(error).__name__ + ": " + str(error)}))
"#;

/// One public or private code-execution test case.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodeTestCase {
    /// Serialized stdin or function arguments.
    pub input: Value,
    /// Expected stdout or return value.
    pub output: Value,
}

/// Fully decoded execution request.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CodeExecutionRequest {
    /// Extracted Python candidate.
    pub code: String,
    /// Public plus private test cases.
    pub cases: Vec<CodeTestCase>,
    /// Function name for call-based problems; absent for stdin/stdout problems.
    pub fn_name: Option<String>,
}

/// Result returned by a code-execution backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeExecutionOutcome {
    /// Whether every test passed.
    pub passed: bool,
    /// First failing case, when reported.
    pub failed_case: Option<usize>,
    /// Backend/candidate diagnostic.
    pub diagnostic: Option<String>,
}

/// Isolation backend seam for code-generation graders.
#[async_trait(?Send)]
pub trait CodeExecutor {
    /// Validate backend prerequisites before a run begins.
    fn check_available(&self) -> Result<(), AccuracyError>;
    /// Execute one candidate against all decoded cases.
    async fn execute(
        &self,
        request: &CodeExecutionRequest,
    ) -> Result<CodeExecutionOutcome, AccuracyError>;
}

/// Linux bubblewrap + prlimit Python execution backend.
#[derive(Debug, Clone)]
pub struct BubblewrapPythonExecutor {
    bubblewrap: PathBuf,
    prlimit: PathBuf,
    python: PathBuf,
    timeout: Duration,
    address_space_bytes: u64,
}

impl BubblewrapPythonExecutor {
    /// Builds the standard no-network, no-workspace sandbox.
    pub fn new() -> Self {
        Self {
            bubblewrap: PathBuf::from("/usr/bin/bwrap"),
            prlimit: PathBuf::from("/usr/bin/prlimit"),
            python: PathBuf::from("/usr/bin/python3"),
            timeout: Duration::from_secs(6),
            address_space_bytes: 512 * 1024 * 1024,
        }
    }

    /// Overrides the per-test timeout; the enclosing wall limit scales with case count.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    fn command(&self, case_count: usize) -> Command {
        let per_case_seconds = self.timeout.as_secs().max(1);
        let cpu_seconds = per_case_seconds
            .saturating_mul(case_count.max(1) as u64)
            .saturating_add(5);
        let mut command = Command::new(&self.prlimit);
        command
            .arg(format!("--as={}", self.address_space_bytes))
            .arg(format!("--cpu={cpu_seconds}"))
            // RLIMIT_NPROC is charged against the host UID, including processes
            // outside a container's PID namespace. A low-looking value can make
            // bubblewrap itself fail before isolation on shared CI hosts.
            .arg("--nproc=65536")
            .arg("--nofile=128")
            .arg("--")
            .arg(&self.bubblewrap)
            .args([
                "--unshare-all",
                "--die-with-parent",
                "--new-session",
                "--ro-bind",
                "/usr",
                "/usr",
            ]);
        for path in ["/lib", "/lib64", "/etc"] {
            if Path::new(path).exists() {
                command.args(["--ro-bind", path, path]);
            }
        }
        command
            .args([
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--tmpfs",
                "/tmp",
                "--chdir",
                "/tmp",
                "--clearenv",
                "--setenv",
                "PATH",
                "/usr/bin",
            ])
            .arg(&self.python)
            .args(["-I", "-S", "-c", PYTHON_RUNNER]);
        command
    }
}

impl Default for BubblewrapPythonExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize)]
struct RunnerResponse {
    passed: bool,
    failed_case: Option<usize>,
    error: Option<String>,
}

#[async_trait(?Send)]
impl CodeExecutor for BubblewrapPythonExecutor {
    fn check_available(&self) -> Result<(), AccuracyError> {
        for (name, path) in [
            ("bubblewrap", &self.bubblewrap),
            ("prlimit", &self.prlimit),
            ("python", &self.python),
        ] {
            if !path.is_file() {
                return Err(AccuracyError::GraderExecution(format!(
                    "code-execution backend requires {name} at {}",
                    path.display()
                )));
            }
        }
        Ok(())
    }

    async fn execute(
        &self,
        request: &CodeExecutionRequest,
    ) -> Result<CodeExecutionOutcome, AccuracyError> {
        self.check_available()?;
        let input = serde_json::to_vec(&serde_json::json!({
            "code": request.code,
            "cases": request.cases,
            "fn_name": request.fn_name,
            "timeout_seconds": self.timeout.as_secs().max(1),
        }))
        .map_err(|error| {
            AccuracyError::GraderExecution(format!("serializing code execution request: {error}"))
        })?;
        let mut command = self.command(request.cases.len());
        command
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);
        let mut child = command.spawn().map_err(|error| {
            AccuracyError::GraderExecution(format!("spawning isolated Python: {error}"))
        })?;
        let mut stdin = child.stdin.take().ok_or_else(|| {
            AccuracyError::GraderExecution("isolated Python had no stdin pipe".to_string())
        })?;
        stdin.write_all(&input).await.map_err(|error| {
            AccuracyError::GraderExecution(format!("writing isolated Python input: {error}"))
        })?;
        drop(stdin);
        let wall_timeout = self
            .timeout
            .saturating_mul(u32::try_from(request.cases.len()).unwrap_or(u32::MAX))
            .saturating_add(Duration::from_secs(5));
        let output = tokio::time::timeout(wall_timeout, child.wait_with_output())
            .await
            .map_err(|_| {
                AccuracyError::GraderExecution(format!(
                    "isolated candidate exceeded {:?} wall timeout",
                    wall_timeout
                ))
            })?
            .map_err(|error| {
                AccuracyError::GraderExecution(format!("waiting for isolated Python: {error}"))
            })?;
        if !output.status.success() {
            return Err(AccuracyError::GraderExecution(format!(
                "isolated Python exited {}: {}",
                output.status,
                truncate(&String::from_utf8_lossy(&output.stderr), 1_000)
            )));
        }
        let response: RunnerResponse = serde_json::from_slice(&output.stdout).map_err(|error| {
            AccuracyError::GraderExecution(format!(
                "decoding isolated Python response: {error}; stdout={}",
                truncate(&String::from_utf8_lossy(&output.stdout), 1_000)
            ))
        })?;
        Ok(CodeExecutionOutcome {
            passed: response.passed,
            failed_case: response.failed_case,
            diagnostic: response.error,
        })
    }
}

/// LiveCodeBench pass@1 grader over an injected isolation backend.
pub struct CodeExecutionGrader {
    executor: Rc<dyn CodeExecutor>,
}

impl CodeExecutionGrader {
    /// Builds a grader with an explicit execution backend.
    pub fn new(executor: Rc<dyn CodeExecutor>) -> Self {
        Self { executor }
    }

    /// Builds the standard Linux bubblewrap-backed grader.
    pub fn bubblewrap() -> Self {
        Self::new(Rc::new(BubblewrapPythonExecutor::new()))
    }

    /// Checks isolation prerequisites for CLI preflight.
    pub fn check_available(&self) -> Result<(), AccuracyError> {
        self.executor.check_available()
    }

    /// Extracts the contents between the final pair of fence-bearing lines.
    ///
    /// This is byte-equivalent to Lighteval's `extract_code` at
    /// `src/lighteval/tasks/tasks/lcb/codegen_metrics.py:655-664` for
    /// newline-delimited model output; the fence language is deliberately not
    /// restricted because the reference does not restrict it.
    pub fn extract_code(&self, response_text: &str) -> String {
        let lines = response_text.split('\n').collect::<Vec<_>>();
        let fences = lines
            .iter()
            .enumerate()
            .filter_map(|(index, line)| line.contains("```").then_some(index))
            .collect::<Vec<_>>();
        if fences.len() < 2 {
            return String::new();
        }
        lines[fences[fences.len() - 2] + 1..fences[fences.len() - 1]].join("\n")
    }
}

impl Default for CodeExecutionGrader {
    fn default() -> Self {
        Self::bubblewrap()
    }
}

#[async_trait(?Send)]
impl Grader for CodeExecutionGrader {
    fn name(&self) -> &'static str {
        "code-execution"
    }

    fn check_available(&self) -> Result<(), AccuracyError> {
        self.executor.check_available()
    }

    async fn grade(
        &self,
        response_text: &str,
        ground_truth: &str,
    ) -> Result<GradingResult, AccuracyError> {
        let code = self.extract_code(response_text);
        if code.is_empty() {
            return Ok(grading_failure("no fenced Python code block extracted", ""));
        }
        let payload: Value = match serde_json::from_str(ground_truth) {
            Ok(payload) => payload,
            Err(error) => {
                return Ok(grading_failure(
                    &format!("ground truth is not JSON: {error}"),
                    &code,
                ));
            }
        };
        let (cases, function) = match decode_payload(&payload) {
            Ok(decoded) => decoded,
            Err(error) => return Ok(grading_failure(&error.to_string(), &code)),
        };
        let request = CodeExecutionRequest {
            code: code.clone(),
            cases,
            fn_name: function,
        };
        let outcome = match self.executor.execute(&request).await {
            Ok(outcome) => outcome,
            Err(error) => return Ok(grading_failure(&error.to_string(), &code)),
        };
        Ok(GradingResult {
            correct: outcome.passed,
            unparsed: false,
            confidence: Some(if outcome.passed { 1.0 } else { 0.0 }),
            extracted: Some(code.clone()),
            ground_truth: "<lcb test cases>".to_string(),
            reasoning: Some(format!(
                "isolated code execution pass@1={}; snippet length={}{}",
                if outcome.passed { 1 } else { 0 },
                code.len(),
                outcome
                    .diagnostic
                    .as_deref()
                    .map(|diagnostic| format!("; {diagnostic}"))
                    .unwrap_or_default()
            )),
        })
    }
}

fn decode_payload(payload: &Value) -> Result<(Vec<CodeTestCase>, Option<String>), AccuracyError> {
    let object = payload.as_object().ok_or_else(|| {
        AccuracyError::GraderExecution("LCB ground truth must be a JSON object".to_string())
    })?;
    let mut cases = parse_cases(object.get("public_test_cases"), false)?;
    cases.extend(parse_cases(object.get("private_test_cases"), true)?);
    if cases.is_empty() {
        return Err(AccuracyError::GraderExecution(
            "LCB payload has no public or private test cases".to_string(),
        ));
    }
    let metadata = object.get("metadata").cloned().unwrap_or(Value::Null);
    let metadata = match metadata {
        Value::String(text) if !text.is_empty() => {
            serde_json::from_str(&text).unwrap_or(Value::Null)
        }
        value => value,
    };
    let function = metadata
        .as_object()
        .and_then(|metadata| metadata.get("func_name"))
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .map(str::to_string);
    Ok((cases, function))
}

fn parse_cases(
    value: Option<&Value>,
    encoded_private: bool,
) -> Result<Vec<CodeTestCase>, AccuracyError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let decoded = match value {
        Value::Null => return Ok(Vec::new()),
        Value::Array(_) => value.clone(),
        Value::String(text) if text.is_empty() => return Ok(Vec::new()),
        Value::String(text) => {
            if encoded_private {
                decode_private_blob(text)
                    .or_else(|_| serde_json::from_str(text))
                    .map_err(|error| {
                        AccuracyError::GraderExecution(format!(
                            "decoding private test cases: {error}"
                        ))
                    })?
            } else {
                serde_json::from_str(text).map_err(|error| {
                    AccuracyError::GraderExecution(format!("decoding public test cases: {error}"))
                })?
            }
        }
        _ => {
            return Err(AccuracyError::GraderExecution(
                "LCB test cases must be an array or JSON/encoded string".to_string(),
            ));
        }
    };
    serde_json::from_value(decoded).map_err(|error| {
        AccuracyError::GraderExecution(format!("validating LCB test cases: {error}"))
    })
}

fn decode_private_blob(text: &str) -> Result<Value, serde_json::Error> {
    let compressed = base64::engine::general_purpose::STANDARD
        .decode(text)
        .map_err(|error| json_error(error.to_string()))?;
    let mut pickle = Vec::new();
    ZlibDecoder::new(compressed.as_slice())
        .read_to_end(&mut pickle)
        .map_err(|error| json_error(error.to_string()))?;
    let json =
        pickled_unicode(&pickle).ok_or_else(|| json_error("pickle contains no Unicode string"))?;
    serde_json::from_str(json)
}

fn pickled_unicode(pickle: &[u8]) -> Option<&str> {
    let mut cursor = 0usize;
    while cursor < pickle.len() {
        match pickle[cursor] {
            0x8c => {
                let length = *pickle.get(cursor + 1)? as usize;
                let start = cursor + 2;
                return std::str::from_utf8(pickle.get(start..start + length)?).ok();
            }
            b'X' => {
                let bytes: [u8; 4] = pickle.get(cursor + 1..cursor + 5)?.try_into().ok()?;
                let length = u32::from_le_bytes(bytes) as usize;
                let start = cursor + 5;
                return std::str::from_utf8(pickle.get(start..start + length)?).ok();
            }
            0x8d => {
                let bytes: [u8; 8] = pickle.get(cursor + 1..cursor + 9)?.try_into().ok()?;
                let length = usize::try_from(u64::from_le_bytes(bytes)).ok()?;
                let start = cursor + 9;
                return std::str::from_utf8(pickle.get(start..start + length)?).ok();
            }
            _ => cursor += 1,
        }
    }
    None
}

fn json_error(message: impl Into<String>) -> serde_json::Error {
    <serde_json::Error as serde::de::Error>::custom(message.into())
}

fn grading_failure(reason: &str, code: &str) -> GradingResult {
    GradingResult {
        correct: false,
        unparsed: true,
        confidence: Some(0.0),
        extracted: (!code.is_empty()).then(|| truncate(code, 200)),
        ground_truth: "<lcb test cases>".to_string(),
        reasoning: Some(format!("LCB grader failed: {reason}")),
    }
}

fn truncate(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_plain_and_upstream_encoded_private_cases() {
        let encoded = "eJxrYJmqxAABPXLR1UqZeQWlJUpWSoYxeUo6SvmlJVCuUm3sFD0A934MFQ==";
        let decoded = decode_private_blob(encoded).unwrap();
        assert_eq!(decoded[0]["input"], "1\n");
        assert_eq!(decoded[0]["output"], "1");
        let plain = parse_cases(
            Some(&serde_json::json!([{"input":"x","output":"y"}])),
            false,
        )
        .unwrap();
        assert_eq!(plain.len(), 1);
    }

    #[test]
    fn extracts_last_python_block() {
        let grader = CodeExecutionGrader::default();
        assert_eq!(
            grader.extract_code("```python\nprint(0)\n```\n```python\nprint(1)\n```"),
            "print(1)"
        );
    }

    #[tokio::test]
    #[ignore = "requires Linux user namespaces and bubblewrap"]
    async fn bubblewrap_executes_without_network_or_workspace_mount() {
        let executor = BubblewrapPythonExecutor::new().with_timeout(Duration::from_secs(5));
        let outcome = executor
            .execute(&CodeExecutionRequest {
                code: r#"import os
import socket
print(os.path.exists('/home'))
try:
    socket.create_connection(('1.1.1.1', 53), timeout=0.1)
    print('network-visible')
except OSError:
    print('network-isolated')"#
                    .to_string(),
                cases: vec![CodeTestCase {
                    input: Value::String(String::new()),
                    output: Value::String("False\nnetwork-isolated".to_string()),
                }],
                fn_name: None,
            })
            .await
            .unwrap();
        assert!(outcome.passed, "{:?}", outcome.diagnostic);
    }

    #[tokio::test]
    #[ignore = "requires Linux user namespaces and bubblewrap"]
    async fn bubblewrap_matches_lighteval_call_and_stdio_comparisons() {
        let executor = BubblewrapPythonExecutor::new().with_timeout(Duration::from_secs(3));
        let call = executor
            .execute(&CodeExecutionRequest {
                code: "class Solution:\n    def add(self, left, right):\n        return (left + right,)"
                    .to_string(),
                cases: vec![CodeTestCase {
                    input: Value::String("1\n3".to_string()),
                    output: Value::String("[4]".to_string()),
                }],
                fn_name: Some("add".to_string()),
            })
            .await
            .unwrap();
        assert!(call.passed, "{:?}", call.diagnostic);

        let stdio = executor
            .execute(&CodeExecutionRequest {
                code: "value = float(input())\nprint(f'  {value:.3f}  ')".to_string(),
                cases: vec![CodeTestCase {
                    input: Value::String("1".to_string()),
                    output: Value::String("1.0000".to_string()),
                }],
                fn_name: None,
            })
            .await
            .unwrap();
        assert!(stdio.passed, "{:?}", stdio.diagnostic);
    }
}
