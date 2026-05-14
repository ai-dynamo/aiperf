<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->

# Accuracy Benchmarking: Stub Implementation Guide

This document catalogs every stubbed method in the accuracy benchmarking scaffolding. The scaffolding is fully integrated into the plugin system, CLI, and config pipeline — the performance benchmarking path is unaffected.

**Status summary:** As of the AIME loader landing on top of PR #815, `MultipleChoiceGrader`, `MathGrader`, `CodeExecutionGrader`, `LightevalExprGrader`, `LightevalLatexGrader`, `LightevalGPQAGrader`, `MMLUBenchmark`, and `AIMEBenchmark` are fully implemented; the remaining grader (`exact_match`) and benchmarks (`hellaswag`, `bigbench`, `aime24`, `aime25`, `math_500`, `gpqa_diamond`, `lcb_codegeneration`) are still stubs and ship behind `NotImplementedError` until each follow-up branch lands. Use the implemented classes as canonical references when filling in the remaining stubs.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Data Models](#data-models)
- [Protocols](#protocols)
- [CLI Configuration](#cli-configuration)
- [Graders](#graders)
- [Benchmarks](#benchmarks)
- [Processors](#processors)
- [Exporters](#exporters)
- [Plugin Registration](#plugin-registration)
- [Implementation Notes](#implementation-notes)

---

## Architecture Overview

```mermaid
graph TD
    A[AccuracyConfig<br/>7 CLI flags --accuracy-*<br/>enabled property] --> B[AccuracyBenchmark<br/>9 benchmarks<br/>load_problems]
    A --> C[AccuracyGrader<br/>4 graders<br/>grade + extract]
    B --> D[AccuracyRecordProcessor<br/>process_record]
    C --> D
    D --> E[AccuracyResultsProcessor<br/>process_result<br/>summarize]
    E --> F[AccuracyConsoleExporter<br/>AccuracyDataExporter]
```

All processors and exporters **self-disable** when `user_config.accuracy.enabled is False` by raising their respective `Disabled` exceptions in `__init__`. This is the same pattern used by `RawRecordWriterProcessor`, `ServerMetricsCsvExporter`, etc.

---

## Data Models

**File:** `src/aiperf/accuracy/models.py`

### GradingResult

Return type for all grader `grade()` methods.

```python
class GradingResult(AIPerfBaseModel):
    correct: bool           # Whether the response was graded as correct
    confidence: float       # Confidence score (0.0 to 1.0)
    reasoning: str          # Explanation of the grading decision
    extracted_answer: str   # Answer extracted from the model response
    ground_truth: str       # Expected correct answer
```

### BenchmarkProblem

Return type for all benchmark `load_problems()` methods.

```python
class BenchmarkProblem(AIPerfBaseModel):
    prompt: str                       # The prompt to send to the LLM
    ground_truth: str                 # The expected correct answer
    task: str                         # Task/subtask name within the benchmark
    metadata: dict = {}               # Additional problem metadata
    few_shot_examples: list[dict] = [] # Few-shot examples to prepend
```

---

## Protocols

**File:** `src/aiperf/accuracy/protocols.py`

### AccuracyGraderProtocol

```python
@runtime_checkable
class AccuracyGraderProtocol(Protocol):
    def __init__(self, user_config: UserConfig, **kwargs) -> None: ...
    async def grade(self, response_text: str, ground_truth: str, **kwargs) -> GradingResult: ...
    def extract_answer(self, response_text: str, **kwargs) -> str: ...
```

### AccuracyBenchmarkProtocol

```python
@runtime_checkable
class AccuracyBenchmarkProtocol(Protocol):
    def __init__(self, user_config: UserConfig, **kwargs) -> None: ...
    async def load_problems(self, tasks: list[str] | None, n_shots: int, enable_cot: bool) -> list[BenchmarkProblem]: ...
```

---

## CLI Configuration

**File:** `src/aiperf/common/config/accuracy_config.py`

All 7 flags appear under the `Accuracy` group in `aiperf profile --help`.

| CLI Flag | Field | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--accuracy-benchmark` | `benchmark` | `str \| None` | `None` | Benchmark to run (e.g., `mmlu`, `aime`). **Enables accuracy mode when set.** |
| `--accuracy-tasks` | `tasks` | `list[str] \| None` | `None` | Subtasks to evaluate (e.g., MMLU subjects) |
| `--accuracy-n-shots` | `n_shots` | `int` (0-8) | `0` | Number of few-shot examples |
| `--accuracy-enable-cot` | `enable_cot` | `bool` | `False` | Enable chain-of-thought prompting |
| `--accuracy-grader` | `grader` | `str \| None` | `None` | Override benchmark's default grader |
| `--accuracy-system-prompt` | `system_prompt` | `str \| None` | `None` | Custom system prompt override |
| `--accuracy-verbose` | `verbose` | `bool` | `False` | Show per-problem grading details |

**Key property:** `AccuracyConfig.enabled -> bool` returns `self.benchmark is not None`.

**Stub validator** in `UserConfig.validate_accuracy_config()` is a no-op `pass` — add validation logic here (e.g., verify benchmark name is a valid `AccuracyBenchmarkType`).

---

## Graders

All graders inherit from `BaseGrader(AIPerfLoggerMixin)` and must implement 2 methods.

### Base Class

**File:** `src/aiperf/accuracy/graders/base.py`

```python
class BaseGrader(AIPerfLoggerMixin):
    def __init__(self, user_config: UserConfig, **kwargs) -> None
    async def grade(self, response_text: str, ground_truth: str, **kwargs) -> GradingResult     # raises NotImplementedError
    def extract_answer(self, response_text: str, **kwargs) -> str                               # raises NotImplementedError
```

### Implemented

| # | Class | File | Plugin Key | Description |
|---|-------|------|------------|-------------|
| 1 | `MultipleChoiceGrader` | `graders/multiple_choice.py` | `multiple_choice` | **IMPLEMENTED in PR #815** — canonical reference for new graders. Matches choice labels (A/B/C/D) by regex extraction then exact comparison. |
| 2 | `MathGrader` | `graders/math.py` | `math` | **IMPLEMENTED with the AIME loader.** Extracts the last `\boxed{...}` (balanced braces), falls back to "the answer is X" / last-number heuristics. Comparison uses a sympy + latex2sympy2-extended symbolic parsing path when the `[accuracy]` extras are installed (ported from the trt-llm benchmark recipe's `math_equal`/`strip_string`); when those packages are missing, the grader transparently falls back to a stdlib `Fraction` parsing + normalized string equality comparison and emits a one-time warning. |
| 3 | `CodeExecutionGrader` | `graders/code_execution.py` | `code_execution` | **IMPLEMENTED with the AIME loader.** Wraps lighteval's `codegen_metrics` to grade LCB-style code-generation responses by sandboxed execution: extracts the response's code block via lighteval's `extract_code`, runs it against the bundled public + private test cases in a `ProcessPoolExecutor` with `num_process_evaluate=8`, and reports pass@1. Requires the `[accuracy]` extras (lighteval); raises `RuntimeError` at construction if missing. Used by AIP-881 (LCB CodeGen). |
| 4 | `LightevalExprGrader` | `graders/lighteval_grader.py` | `lighteval_expr` | **IMPLEMENTED with the AIME loader.** Wraps lighteval's `MultilingualExtractiveMatchMetric` configured with `ExprExtractionConfig` for gold and `(ExprExtractionConfig, LatexExtractionConfig(boxed_match_priority=0))` for predictions — matches the trt-llm recipe's `expr_gold_metric`. Used by AIP-875/876 (AIME24/25). Requires the `[accuracy]` extras. |
| 5 | `LightevalLatexGrader` | `graders/lighteval_grader.py` | `lighteval_latex` | **IMPLEMENTED with the AIME loader.** Same shape as `LightevalExprGrader` but the gold extractor uses `LatexExtractionConfig` — matches the trt-llm recipe's `latex_gold_metric`. Used by AIP-879 (MATH-500). Requires the `[accuracy]` extras. |
| 6 | `LightevalGPQAGrader` | `graders/lighteval_grader.py` | `lighteval_gpqa` | **IMPLEMENTED with the AIME loader.** Wraps `MultilingualExtractiveMatchMetric` with `IndicesExtractionConfig(prefix_for_extraction="NativeLetters")` to extract A/B/C/D in both gold and prediction — matches the trt-llm recipe's `gpqa_metric`. Used by AIP-880 (GPQA-Diamond). Requires the `[accuracy]` extras. |

### Still Stubbed

| # | Class | File | Plugin Key | Description |
|---|-------|------|------------|-------------|
| 1 | `ExactMatchGrader` | `graders/exact_match.py` | `exact_match` | Exact string matching against ground truth |

**Each grader has 2 methods to implement:**

```python
async def grade(self, response_text: str, ground_truth: str, **kwargs) -> GradingResult
def extract_answer(self, response_text: str, **kwargs) -> str
```

---

## Benchmarks

All benchmarks use `AIPerfLoggerMixin` and must implement 1 method.

### Implemented

| # | Class | File | Plugin Key | Default Grader | Default N-Shots | Notes |
|---|-------|------|------------|----------------|-----------------|-------|
| 1 | `MMLUBenchmark` | `benchmarks/mmlu.py` | `mmlu` | `multiple_choice` | 5 | **IMPLEMENTED in PR #815** — canonical reference for new benchmarks. Downloads via HuggingFace datasets, handles few-shot formatting and CoT. |
| 2 | `AIMEBenchmark` | `benchmarks/aime.py` | `aime` | `math` | 0 | **IMPLEMENTED.** Loads `Maxwell-Jia/AIME_2024`, instructs the model to wrap its final integer in `\boxed{}`, supports few-shot priming and chain-of-thought. |

### Still Stubbed

| # | Class | File | Plugin Key | Default Grader | Default N-Shots |
|---|-------|------|------------|----------------|-----------------|
| 1 | `HellaSwagBenchmark` | `benchmarks/hellaswag.py` | `hellaswag` | `multiple_choice` | 0 |
| 2 | `BigBenchBenchmark` | `benchmarks/bigbench.py` | `bigbench` | `exact_match` | 3 |
| 3 | `AIME24Benchmark` | `benchmarks/aime24.py` | `aime24` | `math` | 0 |
| 4 | `AIME25Benchmark` | `benchmarks/aime25.py` | `aime25` | `math` | 0 |
| 5 | `Math500Benchmark` | `benchmarks/math_500.py` | `math_500` | `math` | 0 |
| 6 | `GPQADiamondBenchmark` | `benchmarks/gpqa_diamond.py` | `gpqa_diamond` | `multiple_choice` | 0 |
| 7 | `LCBCodeGenerationBenchmark` | `benchmarks/lcb_codegeneration.py` | `lcb_codegeneration` | `code_execution` | 0 |

**Each benchmark has 1 method to implement:**

```python
async def load_problems(
    self, tasks: list[str] | None, n_shots: int, enable_cot: bool
) -> list[BenchmarkProblem]
```

Default grader and n-shots are stored in `plugins.yaml` metadata and can be read at runtime via:
```python
plugins.get_metadata(PluginType.ACCURACY_BENCHMARK, "mmlu")  # -> {"default_grader": "multiple_choice", "default_n_shots": 5}
```

---

## Processors

### AccuracyRecordProcessor — IMPLEMENTED in PR #815

**File:** `src/aiperf/accuracy/accuracy_record_processor.py`
**Parent:** `AIPerfLifecycleMixin`
**Implements:** `RecordProcessorProtocol`
**Plugin key:** `accuracy_record` (under `record_processor`)
**Disables via:** `PostProcessorDisabled` when `not user_config.accuracy.enabled`

This class is fully implemented and serves as the canonical reference for wiring grading into the record processing pipeline.

```python
async def process_record(
    self, record: ParsedResponseRecord, metadata: MetricRecordMetadata
) -> MetricRecordDict                                                          # IMPLEMENTED in PR #815
```

**Reference implementation:** `MetricRecordProcessor` in `src/aiperf/post_processors/metric_record_processor.py`

### AccuracyResultsProcessor — IMPLEMENTED in PR #815

**File:** `src/aiperf/accuracy/accuracy_results_processor.py`
**Parent:** `AIPerfLifecycleMixin`
**Implements:** `ResultsProcessorProtocol`
**Plugin key:** `accuracy_results` (under `results_processor`)
**Disables via:** `PostProcessorDisabled` when `not user_config.accuracy.enabled`

This class is fully implemented and serves as the canonical reference for aggregating per-task accuracy metrics.

```python
async def process_result(self, record_data: MetricRecordsData) -> None         # IMPLEMENTED in PR #815
async def summarize(self) -> list[MetricResult]                                # IMPLEMENTED in PR #815
```

**Reference implementation:** `MetricResultsProcessor` in `src/aiperf/post_processors/metric_results_processor.py`

---

## Exporters

### AccuracyConsoleExporter — IMPLEMENTED in PR #815

**File:** `src/aiperf/accuracy/accuracy_console_exporter.py`
**Parent:** `AIPerfLoggerMixin`
**Implements:** `ConsoleExporterProtocol`
**Plugin key:** `accuracy` (under `console_exporter`)
**Disables via:** `ConsoleExporterDisabled` when `not user_config.accuracy.enabled`

This class is fully implemented and serves as the canonical reference for displaying accuracy results in the terminal.

```python
async def export(self, console: Console) -> None                               # IMPLEMENTED in PR #815
```

**Reference implementation:** `ConsoleMetricsExporter` in `src/aiperf/exporters/console_metrics_exporter.py`

### AccuracyDataExporter — IMPLEMENTED in PR #815

**File:** `src/aiperf/accuracy/accuracy_data_exporter.py`
**Parent:** `AIPerfLoggerMixin`
**Implements:** `DataExporterProtocol`
**Plugin key:** `accuracy_csv` (under `data_exporter`)
**Disables via:** `DataExporterDisabled` when `not user_config.accuracy.enabled`

This class is fully implemented and serves as the canonical reference for writing accuracy results to CSV.

```python
def get_export_info(self) -> FileExportInfo                                    # IMPLEMENTED in PR #815
async def export(self) -> None                                                 # IMPLEMENTED in PR #815
```

**Reference implementation:** `MetricsCsvExporter` in `src/aiperf/exporters/metrics_csv_exporter.py`

---

## Plugin Registration

All stubs are registered in `src/aiperf/plugin/plugins.yaml` and `src/aiperf/plugin/categories.yaml`.

### New Plugin Categories

| Category | Protocol | Generated Enum |
|----------|----------|----------------|
| `accuracy_grader` | `AccuracyGraderProtocol` | `AccuracyGraderType` |
| `accuracy_benchmark` | `AccuracyBenchmarkProtocol` | `AccuracyBenchmarkType` |

### New PluginType Members

- `PluginType.ACCURACY_GRADER`
- `PluginType.ACCURACY_BENCHMARK`

### Registrations in Existing Categories

| Category | Plugin Key | Class |
|----------|-----------|-------|
| `record_processor` | `accuracy_record` | `AccuracyRecordProcessor` |
| `results_processor` | `accuracy_results` | `AccuracyResultsProcessor` |
| `console_exporter` | `accuracy` | `AccuracyConsoleExporter` |
| `data_exporter` | `accuracy_csv` | `AccuracyDataExporter` |

---

## Implementation Notes

### Method Count Summary

| Component | Implemented | Still Stubbed | Methods per Stub | Remaining Methods |
|-----------|-------------|---------------|------------------|-------------------|
| Graders | 1 (`MultipleChoiceGrader`) | 3 | 2 (`grade`, `extract_answer`) | 6 |
| Benchmarks | 1 (`MMLUBenchmark`) | 8 | 1 (`load_problems`) | 8 |
| Record Processor | 1 (`AccuracyRecordProcessor`) | 0 | — | 0 |
| Results Processor | 1 (`AccuracyResultsProcessor`) | 0 | — | 0 |
| Console Exporter | 1 (`AccuracyConsoleExporter`) | 0 | — | 0 |
| Data Exporter | 1 (`AccuracyDataExporter`) | 0 | — | 0 |
| Config Validator | 0 | 1 | 1 (`validate_accuracy_config`) | 1 |
| **Total** | **6** | **13** | | **15** |

### Self-Disabling Pattern

Processors and exporters raise their `Disabled` exception **in `__init__`** when accuracy is off. The existing framework catches these and silently skips the plugin. No code changes needed to support this — it uses the same pattern as `RawRecordWriterProcessor` and `ServerMetricsCsvExporter`.

### Suggested Implementation Order

The processors, exporters, and one grader/benchmark pair are already wired end-to-end. Start from the already-working pipeline:

1. **Graders** — use `MultipleChoiceGrader` as reference; implement `ExactMatchGrader` next (simplest), then `MathGrader`
2. **Benchmarks** — use `MMLUBenchmark` as reference; implement dataset loading for each remaining benchmark
3. **Config validator** — validate benchmark name against `AccuracyBenchmarkType` enum in `UserConfig.validate_accuracy_config()`

### Key Files for Reference

| What | Where |
|------|-------|
| **Canonical grader** | `src/aiperf/accuracy/graders/multiple_choice.py` |
| **Canonical benchmark** | `src/aiperf/accuracy/benchmarks/mmlu.py` |
| **Canonical record processor** | `src/aiperf/accuracy/accuracy_record_processor.py` |
| **Canonical results processor** | `src/aiperf/accuracy/accuracy_results_processor.py` |
| **Canonical console exporter** | `src/aiperf/accuracy/accuracy_console_exporter.py` |
| **Canonical data exporter** | `src/aiperf/accuracy/accuracy_data_exporter.py` |
| Disabled exception pattern | `src/aiperf/post_processors/raw_record_writer_processor.py:47` |
| Record processor protocol | `src/aiperf/post_processors/protocols.py` |
| Exporter protocols | `src/aiperf/exporters/protocols.py` |
| Plugin lookup API | `plugins.get_class(PluginType.ACCURACY_GRADER, "exact_match")` |
| Metadata lookup API | `plugins.get_metadata(PluginType.ACCURACY_BENCHMARK, "mmlu")` |
