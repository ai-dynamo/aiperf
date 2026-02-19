<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->

# Accuracy Benchmarking: Stub Implementation Guide

This document catalogs every stubbed method in the accuracy benchmarking scaffolding. All stubs currently raise `NotImplementedError` and are ready for implementation. The scaffolding is fully integrated into the plugin system, CLI, and config pipeline -- the performance benchmarking path is unaffected.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Data Models](#data-models)
- [Protocols](#protocols)
- [CLI Configuration](#cli-configuration)
- [Graders (4 stubs)](#graders)
- [Benchmarks (9 stubs)](#benchmarks)
- [Processors (2 stubs)](#processors)
- [Exporters (2 stubs)](#exporters)
- [Plugin Registration](#plugin-registration)
- [Implementation Notes](#implementation-notes)

---

## Architecture Overview

```
                  +-------------------+
                  | AccuracyConfig    |  <-- 7 CLI flags (--accuracy-*)
                  | enabled property  |
                  +--------+----------+
                           |
           +---------------+----------------+
           |                                |
  +--------v---------+          +-----------v-----------+
  | AccuracyBenchmark|          | AccuracyGrader        |
  | (9 benchmarks)   |          | (4 graders)           |
  | load_problems()  |          | grade() + extract()   |
  +--------+---------+          +-----------+-----------+
           |                                |
           v                                v
  +------------------+          +------------------------+
  | AccuracyRecord   |          | AccuracyResults        |
  | Processor        |          | Processor              |
  | process_record() |          | process_result()       |
  +--------+---------+          | summarize()            |
           |                    +-----------+------------+
           |                                |
           +---------------+----------------+
                           |
              +------------v-----------+
              | AccuracyConsoleExporter |
              | AccuracyDataExporter    |
              +------------------------+
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

**Stub validator** in `UserConfig.validate_accuracy_config()` is a no-op `pass` -- add validation logic here (e.g., verify benchmark name is a valid `AccuracyBenchmarkType`).

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

### Stub Implementations

| # | Class | File | Plugin Key | Description |
|---|-------|------|------------|-------------|
| 1 | `ExactMatchGrader` | `graders/exact_match.py` | `exact_match` | Exact string matching against ground truth |
| 2 | `MathGrader` | `graders/math.py` | `math` | Mathematical expression equivalence |
| 3 | `MultipleChoiceGrader` | `graders/multiple_choice.py` | `multiple_choice` | Match choice labels (A/B/C/D) |
| 4 | `CodeExecutionGrader` | `graders/code_execution.py` | `code_execution` | Execute code and compare output |

**Each grader has 2 methods to implement:**

```python
async def grade(self, response_text: str, ground_truth: str, **kwargs) -> GradingResult
def extract_answer(self, response_text: str, **kwargs) -> str
```

---

## Benchmarks

All benchmarks use `AIPerfLoggerMixin` and must implement 1 method.

### Stub Implementations

| # | Class | File | Plugin Key | Default Grader | Default N-Shots |
|---|-------|------|------------|----------------|-----------------|
| 1 | `MMLUBenchmark` | `benchmarks/mmlu.py` | `mmlu` | `multiple_choice` | 5 |
| 2 | `AIMEBenchmark` | `benchmarks/aime.py` | `aime` | `math` | 0 |
| 3 | `HellaSwagBenchmark` | `benchmarks/hellaswag.py` | `hellaswag` | `multiple_choice` | 0 |
| 4 | `BigBenchBenchmark` | `benchmarks/bigbench.py` | `bigbench` | `exact_match` | 3 |
| 5 | `AIME24Benchmark` | `benchmarks/aime24.py` | `aime24` | `math` | 0 |
| 6 | `AIME25Benchmark` | `benchmarks/aime25.py` | `aime25` | `math` | 0 |
| 7 | `Math500Benchmark` | `benchmarks/math_500.py` | `math_500` | `math` | 0 |
| 8 | `GPQADiamondBenchmark` | `benchmarks/gpqa_diamond.py` | `gpqa_diamond` | `multiple_choice` | 0 |
| 9 | `LCBCodeGenerationBenchmark` | `benchmarks/lcb_codegeneration.py` | `lcb_codegeneration` | `code_execution` | 0 |

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

### AccuracyRecordProcessor

**File:** `src/aiperf/accuracy/accuracy_record_processor.py`
**Parent:** `AIPerfLifecycleMixin`
**Implements:** `RecordProcessorProtocol`
**Plugin key:** `accuracy_record` (under `record_processor`)
**Disables via:** `PostProcessorDisabled` when `not user_config.accuracy.enabled`

```python
async def process_record(
    self, record: ParsedResponseRecord, metadata: MetricRecordMetadata
) -> MetricRecordDict                                                          # raises NotImplementedError
```

**Reference implementation:** `MetricRecordProcessor` in `src/aiperf/post_processors/metric_record_processor.py`

### AccuracyResultsProcessor

**File:** `src/aiperf/accuracy/accuracy_results_processor.py`
**Parent:** `AIPerfLifecycleMixin`
**Implements:** `ResultsProcessorProtocol`
**Plugin key:** `accuracy_results` (under `results_processor`)
**Disables via:** `PostProcessorDisabled` when `not user_config.accuracy.enabled`

```python
async def process_result(self, record_data: MetricRecordsData) -> None         # raises NotImplementedError
async def summarize(self) -> list[MetricResult]                                # raises NotImplementedError
```

**Reference implementation:** `MetricResultsProcessor` in `src/aiperf/post_processors/metric_results_processor.py`

---

## Exporters

### AccuracyConsoleExporter

**File:** `src/aiperf/accuracy/accuracy_console_exporter.py`
**Parent:** `AIPerfLoggerMixin`
**Implements:** `ConsoleExporterProtocol`
**Plugin key:** `accuracy` (under `console_exporter`)
**Disables via:** `ConsoleExporterDisabled` when `not user_config.accuracy.enabled`

```python
async def export(self, console: Console) -> None                               # raises NotImplementedError
```

**Reference implementation:** `ConsoleMetricsExporter` in `src/aiperf/exporters/console_metrics_exporter.py`

### AccuracyDataExporter

**File:** `src/aiperf/accuracy/accuracy_data_exporter.py`
**Parent:** `AIPerfLoggerMixin`
**Implements:** `DataExporterProtocol`
**Plugin key:** `accuracy_csv` (under `data_exporter`)
**Disables via:** `DataExporterDisabled` when `not user_config.accuracy.enabled`

```python
def get_export_info(self) -> FileExportInfo                                    # raises NotImplementedError
async def export(self) -> None                                                 # raises NotImplementedError
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

| Component | Stubs | Methods per Stub | Total Methods |
|-----------|-------|------------------|---------------|
| Graders | 4 | 2 (`grade`, `extract_answer`) | 8 |
| Benchmarks | 9 | 1 (`load_problems`) | 9 |
| Record Processor | 1 | 1 (`process_record`) | 1 |
| Results Processor | 1 | 2 (`process_result`, `summarize`) | 2 |
| Console Exporter | 1 | 1 (`export`) | 1 |
| Data Exporter | 1 | 2 (`get_export_info`, `export`) | 2 |
| Config Validator | 1 | 1 (`validate_accuracy_config`) | 1 |
| **Total** | **18** | | **24** |

### Self-Disabling Pattern

Processors and exporters raise their `Disabled` exception **in `__init__`** when accuracy is off. The existing framework catches these and silently skips the plugin. No code changes needed to support this -- it uses the same pattern as `RawRecordWriterProcessor` and `ServerMetricsCsvExporter`.

### Suggested Implementation Order

1. **Models** -- finalize `GradingResult` and `BenchmarkProblem` fields if needed
2. **Graders** -- start with `ExactMatchGrader` (simplest), then `MultipleChoiceGrader`
3. **Benchmarks** -- start with `MMLUBenchmark` (most common), wire up dataset loading
4. **Config validator** -- validate benchmark name against `AccuracyBenchmarkType` enum
5. **Processors** -- wire grading into the record processing pipeline
6. **Exporters** -- display results in console and write CSV output
7. **Integration** -- decide how `BenchmarkProblem` flows into the `Conversation`-based pipeline

### Key Files for Reference

| What | Where |
|------|-------|
| Disabled exception pattern | `src/aiperf/post_processors/raw_record_writer_processor.py:47` |
| Record processor protocol | `src/aiperf/post_processors/protocols.py` |
| Exporter protocols | `src/aiperf/exporters/protocols.py` |
| Console exporter example | `src/aiperf/exporters/console_metrics_exporter.py` |
| Data exporter example | `src/aiperf/exporters/metrics_csv_exporter.py` |
| Plugin lookup API | `plugins.get_class(PluginType.ACCURACY_GRADER, "exact_match")` |
| Metadata lookup API | `plugins.get_metadata(PluginType.ACCURACY_BENCHMARK, "mmlu")` |
