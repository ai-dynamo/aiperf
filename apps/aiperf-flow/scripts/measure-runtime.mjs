/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Deterministic runtime frame measurement summarizer for AIPerf Flow.
 *
 * Consumes real per-frame evaluation / draw / total samples (or Performance
 * Timeline measures produced by E2E) and emits a machine-readable quality
 * report with p50, p95, worst, memory, and environment metadata. Rejects
 * synthetic and malformed telemetry with explicit errors. Importable for
 * tests; runnable as a CLI.
 */

import { readFileSync, writeFileSync } from "node:fs";
import os from "node:os";
import process from "node:process";
import { pathToFileURL } from "node:url";

import { Command } from "commander";
import { z } from "zod";

/** Report schema version for machine consumers. */
export const MEASURE_RUNTIME_SCHEMA_VERSION = 1;

/**
 * User Timing measure names emitted by the runtime and collected by Playwright.
 * Kept in lockstep with `e2e/helpers/runtime-metrics.ts` and the canvas/renderer
 * instrumentation seam.
 */
export const RUNTIME_PERFORMANCE_ENTRY_NAMES = Object.freeze({
  evaluation: "aiperf-flow:evaluation",
  draw: "aiperf-flow:draw",
  total: "aiperf-flow:total",
});

/** Frame-budget thresholds (ms) for reference (60 fps) and degraded (30 fps). */
export const FRAME_BUDGET_MS = Object.freeze({
  reference: 1000 / 60,
  degraded: 1000 / 30,
});

const SYNTHETIC_SOURCE_MARKERS = new Set([
  "synthetic",
  "fixture",
  "mock",
  "placeholder",
  "fake",
  "stub",
]);

const finiteNonNegative = z.number().finite().nonnegative();

/** One timed frame with separately measured phases. */
export const FrameSampleSchema = z
  .object({
    evaluationMs: finiteNonNegative,
    drawMs: finiteNonNegative,
    totalMs: finiteNonNegative,
  })
  .strict();

/** Optional process memory snapshot attached to a report. */
export const MemorySnapshotSchema = z
  .object({
    heapUsedBytes: finiteNonNegative,
    heapTotalBytes: finiteNonNegative,
    rssBytes: finiteNonNegative,
    externalBytes: finiteNonNegative.optional(),
  })
  .strict();

/** Host / runtime environment recorded with the report. */
export const EnvironmentSnapshotSchema = z
  .object({
    node: z.string().min(1),
    platform: z.string().min(1),
    arch: z.string().min(1),
    cpus: z.number().int().positive(),
    totalMemoryBytes: finiteNonNegative.optional(),
  })
  .strict();

const LatencySummarySchema = z
  .object({
    p50: finiteNonNegative,
    p95: finiteNonNegative,
    worst: finiteNonNegative,
  })
  .strict();

export const RuntimeMeasurementReportSchema = z
  .object({
    schemaVersion: z.literal(MEASURE_RUNTIME_SCHEMA_VERSION),
    sampleCount: z.number().int().nonnegative(),
    frames: z
      .object({
        evaluationMs: LatencySummarySchema,
        drawMs: LatencySummarySchema,
        totalMs: LatencySummarySchema,
      })
      .strict(),
    memory: MemorySnapshotSchema,
    environment: EnvironmentSnapshotSchema,
  })
  .strict();

const MeasurementInputSchema = z
  .object({
    samples: z.array(FrameSampleSchema).min(1),
    memory: MemorySnapshotSchema.optional(),
    environment: EnvironmentSnapshotSchema.optional(),
  })
  .strict();

const PerformanceEntrySchema = z
  .object({
    name: z.string().min(1),
    duration: finiteNonNegative,
    entryType: z.string().min(1).optional(),
  })
  .strict();

/**
 * Format a Zod failure into a stable, single-line diagnostic.
 *
 * @param {z.ZodError} error
 * @returns {string}
 */
export function formatZodIssues(error) {
  return error.issues
    .map((issue) => {
      const path = issue.path.length === 0 ? "<root>" : issue.path.join(".");
      return `${path}: ${issue.message}`;
    })
    .join("; ");
}

/**
 * @param {unknown} error
 * @param {string} prefix
 * @returns {Error}
 */
function asMeasurementError(error, prefix) {
  if (error instanceof z.ZodError) {
    return new Error(`${prefix}: ${formatZodIssues(error)}`);
  }
  if (error instanceof Error) {
    return error;
  }
  return new Error(`${prefix}: ${String(error)}`);
}

/**
 * Reject payloads that declare synthetic / fixture provenance.
 *
 * @param {unknown} decoded
 */
export function assertNotSyntheticMeasurementInput(decoded) {
  if (decoded === null || typeof decoded !== "object") {
    return;
  }

  if (Array.isArray(decoded)) {
    for (let index = 0; index < decoded.length; index += 1) {
      const sample = decoded[index];
      if (
        sample !== null &&
        typeof sample === "object" &&
        !Array.isArray(sample) &&
        "synthetic" in sample &&
        sample.synthetic === true
      ) {
        throw new Error(
          `rejected synthetic measurement data: samples[${index}].synthetic=true (E2E must supply real evaluation/draw/total timings)`,
        );
      }
    }
    return;
  }

  const record = /** @type {Record<string, unknown>} */ (decoded);

  if (record.synthetic === true) {
    throw new Error(
      "rejected synthetic measurement data: synthetic=true is not allowed (pass real Playwright evaluation/draw/total samples)",
    );
  }

  if (typeof record.source === "string") {
    const normalized = record.source.trim().toLowerCase();
    if (SYNTHETIC_SOURCE_MARKERS.has(normalized)) {
      throw new Error(
        `rejected synthetic measurement data: source=${JSON.stringify(record.source)} (expected live Playwright or Performance Timeline telemetry)`,
      );
    }
  }

  if (typeof record.provenance === "string") {
    const normalized = record.provenance.trim().toLowerCase();
    if (SYNTHETIC_SOURCE_MARKERS.has(normalized)) {
      throw new Error(
        `rejected synthetic measurement data: provenance=${JSON.stringify(record.provenance)} (expected live Playwright or Performance Timeline telemetry)`,
      );
    }
  }

  if (Array.isArray(record.samples)) {
    for (let index = 0; index < record.samples.length; index += 1) {
      const sample = record.samples[index];
      if (
        sample !== null &&
        typeof sample === "object" &&
        !Array.isArray(sample) &&
        "synthetic" in sample &&
        /** @type {{ synthetic?: unknown }} */ (sample).synthetic === true
      ) {
        throw new Error(
          `rejected synthetic measurement data: samples[${index}].synthetic=true (E2E must supply real evaluation/draw/total timings)`,
        );
      }
    }
  }
}

/**
 * Ensure nested total/evaluation/draw phases are physically consistent.
 *
 * @param {z.infer<typeof FrameSampleSchema>} sample
 * @param {number} index
 */
export function assertFrameSampleConsistency(sample, index) {
  if (sample.totalMs + Number.EPSILON < sample.evaluationMs) {
    throw new Error(
      `malformed frame sample at index ${index}: totalMs (${sample.totalMs}) is less than evaluationMs (${sample.evaluationMs})`,
    );
  }
  if (sample.totalMs + Number.EPSILON < sample.drawMs) {
    throw new Error(
      `malformed frame sample at index ${index}: totalMs (${sample.totalMs}) is less than drawMs (${sample.drawMs})`,
    );
  }
}

/**
 * Pair Performance Timeline measures into per-frame samples.
 *
 * @param {readonly unknown[]} entries
 * @returns {z.infer<typeof FrameSampleSchema>[]}
 */
export function samplesFromPerformanceEntries(entries) {
  if (!Array.isArray(entries) || entries.length === 0) {
    throw new Error(
      "malformed telemetry: entries must be a non-empty array of Performance measures",
    );
  }

  /** @type {number[]} */
  const evaluation = [];
  /** @type {number[]} */
  const draw = [];
  /** @type {number[]} */
  const total = [];

  for (let index = 0; index < entries.length; index += 1) {
    let entry;
    try {
      entry = PerformanceEntrySchema.parse(entries[index]);
    } catch (error) {
      throw asMeasurementError(
        error,
        `malformed telemetry: entries[${index}]`,
      );
    }

    if (entry.entryType !== undefined && entry.entryType !== "measure") {
      throw new Error(
        `malformed telemetry: entries[${index}] entryType must be "measure" (got ${JSON.stringify(entry.entryType)})`,
      );
    }

    switch (entry.name) {
      case RUNTIME_PERFORMANCE_ENTRY_NAMES.evaluation:
        evaluation.push(entry.duration);
        break;
      case RUNTIME_PERFORMANCE_ENTRY_NAMES.draw:
        draw.push(entry.duration);
        break;
      case RUNTIME_PERFORMANCE_ENTRY_NAMES.total:
        total.push(entry.duration);
        break;
      default:
        throw new Error(
          `malformed telemetry: entries[${index}] has unrecognized measure name ${JSON.stringify(entry.name)} (expected ${RUNTIME_PERFORMANCE_ENTRY_NAMES.evaluation}, ${RUNTIME_PERFORMANCE_ENTRY_NAMES.draw}, or ${RUNTIME_PERFORMANCE_ENTRY_NAMES.total})`,
        );
    }
  }

  const counts = [evaluation.length, draw.length, total.length];
  if (counts.some((count) => count === 0)) {
    throw new Error(
      `malformed telemetry: missing phase measures evaluation=${evaluation.length}, draw=${draw.length}, total=${total.length}`,
    );
  }
  if (new Set(counts).size !== 1) {
    throw new Error(
      `malformed telemetry: phase measure counts do not match evaluation=${evaluation.length}, draw=${draw.length}, total=${total.length}`,
    );
  }

  return evaluation.map((evaluationMs, index) => {
    const sample = {
      evaluationMs,
      drawMs: draw[index],
      totalMs: total[index],
    };
    assertFrameSampleConsistency(sample, index);
    return sample;
  });
}

/**
 * Nearest-rank percentile over a finite numeric series.
 *
 * @param {readonly number[]} values
 * @param {number} percentile Inclusive percentile in (0, 100].
 * @returns {number}
 */
export function percentile(values, percentile) {
  if (!Array.isArray(values) || values.length === 0) {
    throw new Error("percentile requires a non-empty finite sample series");
  }
  if (
    typeof percentile !== "number" ||
    !Number.isFinite(percentile) ||
    percentile <= 0 ||
    percentile > 100
  ) {
    throw new Error(`percentile must be in (0, 100], got ${percentile}`);
  }
  for (const value of values) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new Error(`percentile rejects non-finite sample: ${value}`);
    }
  }

  const sorted = [...values].sort((a, b) => a - b);
  const rank = Math.ceil((percentile / 100) * sorted.length);
  return sorted[Math.max(0, rank - 1)];
}

/**
 * Summarize one numeric series into p50 / p95 / worst.
 *
 * @param {readonly number[]} values
 * @returns {{ p50: number, p95: number, worst: number }}
 */
export function summarizeSeries(values) {
  if (!Array.isArray(values) || values.length === 0) {
    throw new Error("summarizeSeries requires a non-empty sample series");
  }
  return {
    p50: percentile(values, 50),
    p95: percentile(values, 95),
    worst: Math.max(...values),
  };
}

/**
 * Capture process memory using Node's process.memoryUsage().
 *
 * @param {NodeJS.MemoryUsage} [usage]
 * @returns {z.infer<typeof MemorySnapshotSchema>}
 */
export function captureMemorySnapshot(usage = process.memoryUsage()) {
  return MemorySnapshotSchema.parse({
    heapUsedBytes: usage.heapUsed,
    heapTotalBytes: usage.heapTotal,
    rssBytes: usage.rss,
    externalBytes: usage.external,
  });
}

/**
 * Capture host / runtime environment metadata.
 *
 * @param {Partial<z.infer<typeof EnvironmentSnapshotSchema>>} [overrides]
 * @returns {z.infer<typeof EnvironmentSnapshotSchema>}
 */
export function captureEnvironmentSnapshot(overrides = {}) {
  return EnvironmentSnapshotSchema.parse({
    node: process.version,
    platform: os.platform(),
    arch: os.arch(),
    cpus: os.cpus().length,
    totalMemoryBytes: os.totalmem(),
    ...overrides,
  });
}

/**
 * Normalize decoded JSON into validated measurement input.
 *
 * Accepts:
 * - `{ samples: [...] }` from `e2e/helpers/runtime-metrics.ts`
 * - a bare sample array
 * - `{ entries: [...] }` Performance Timeline measures from Playwright
 *
 * Rejects synthetic provenance and inconsistent / malformed payloads.
 *
 * @param {unknown} decoded
 * @returns {z.infer<typeof MeasurementInputSchema>}
 */
export function ingestRuntimeTelemetry(decoded) {
  assertNotSyntheticMeasurementInput(decoded);

  if (Array.isArray(decoded)) {
    try {
      const parsed = MeasurementInputSchema.parse({ samples: decoded });
      parsed.samples.forEach(assertFrameSampleConsistency);
      return parsed;
    } catch (error) {
      throw asMeasurementError(error, "malformed measurement samples");
    }
  }

  if (decoded === null || typeof decoded !== "object") {
    throw new Error(
      `malformed measurement JSON: expected object or sample array, got ${typeof decoded}`,
    );
  }

  const record = /** @type {Record<string, unknown>} */ (decoded);

  if (Array.isArray(record.entries) && record.samples === undefined) {
    const samples = samplesFromPerformanceEntries(record.entries);
    const rest = { ...record };
    delete rest.entries;
    delete rest.source;
    delete rest.provenance;
    delete rest.synthetic;
    try {
      const parsed = MeasurementInputSchema.parse({ ...rest, samples });
      parsed.samples.forEach(assertFrameSampleConsistency);
      return parsed;
    } catch (error) {
      throw asMeasurementError(error, "malformed measurement input");
    }
  }

  const rest = { ...record };
  delete rest.source;
  delete rest.provenance;
  delete rest.synthetic;
  delete rest.entries;

  try {
    const parsed = MeasurementInputSchema.parse(rest);
    parsed.samples.forEach(assertFrameSampleConsistency);
    return parsed;
  } catch (error) {
    throw asMeasurementError(error, "malformed measurement input");
  }
}

/**
 * Build a deterministic machine-readable runtime measurement report.
 *
 * @param {unknown} input Samples plus optional memory / environment snapshots.
 * @returns {z.infer<typeof RuntimeMeasurementReportSchema>}
 */
export function summarizeRuntimeMeasurements(input) {
  const parsed = ingestRuntimeTelemetry(input);
  const evaluationMs = parsed.samples.map((sample) => sample.evaluationMs);
  const drawMs = parsed.samples.map((sample) => sample.drawMs);
  const totalMs = parsed.samples.map((sample) => sample.totalMs);

  const report = {
    schemaVersion: MEASURE_RUNTIME_SCHEMA_VERSION,
    sampleCount: parsed.samples.length,
    frames: {
      evaluationMs: summarizeSeries(evaluationMs),
      drawMs: summarizeSeries(drawMs),
      totalMs: summarizeSeries(totalMs),
    },
    memory: parsed.memory ?? captureMemorySnapshot(),
    environment: parsed.environment ?? captureEnvironmentSnapshot(),
  };

  return RuntimeMeasurementReportSchema.parse(report);
}

/**
 * Compare a report's total-frame p95 against a named frame-budget profile.
 *
 * Report generation and JSON output are unchanged; callers decide whether a
 * budget miss is fatal. Existing default CLI behavior remains report-only.
 *
 * @param {z.infer<typeof RuntimeMeasurementReportSchema>} report
 * @param {"reference" | "degraded"} profile
 * @returns {{ profile: "reference" | "degraded", budgetMs: number, totalP95Ms: number, withinBudget: boolean }}
 */
export function evaluateFrameBudget(report, profile) {
  if (profile !== "reference" && profile !== "degraded") {
    throw new Error(
      `unknown frame-budget profile ${JSON.stringify(profile)} (expected "reference" or "degraded")`,
    );
  }
  const budgetMs = FRAME_BUDGET_MS[profile];
  const totalP95Ms = report.frames.totalMs.p95;
  return {
    profile,
    budgetMs,
    totalP95Ms,
    withinBudget: totalP95Ms <= budgetMs,
  };
}

/**
 * Serialize a report as stable, machine-readable JSON.
 *
 * @param {z.infer<typeof RuntimeMeasurementReportSchema>} report
 * @returns {string}
 */
export function formatRuntimeMeasurementReport(report) {
  return `${JSON.stringify(RuntimeMeasurementReportSchema.parse(report), null, 2)}\n`;
}

/**
 * Parse CLI / file JSON into measurement input.
 *
 * Accepts either `{ samples: [...] }`, `{ entries: [...] }`, or a bare sample
 * array. Rejects synthetic and malformed telemetry with explicit errors.
 *
 * @param {string} raw
 * @returns {z.infer<typeof MeasurementInputSchema>}
 */
export function parseMeasurementInputJson(raw) {
  let decoded;
  try {
    decoded = JSON.parse(raw);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`invalid measurement JSON: ${message}`);
  }

  return ingestRuntimeTelemetry(decoded);
}

/**
 * CLI entry: read samples, emit a quality report.
 *
 * @param {string[]} argv
 * @param {{ stdin?: string, stdoutWrite?: (text: string) => void }} [io]
 * @returns {number} Process exit code.
 */
export function runMeasureRuntimeCli(argv, io = {}) {
  const program = new Command();
  program
    .name("measure-runtime")
    .description(
      "Summarize AIPerf Flow evaluation/draw/total frame samples into a machine-readable quality report",
    )
    .option("-i, --input <path>", "JSON file of frame samples (default: stdin)")
    .option("-o, --output <path>", "Write report JSON to path (default: stdout)")
    .option(
      "--profile <name>",
      `Optional frame-budget profile for stderr diagnostics ("reference"=${FRAME_BUDGET_MS.reference}ms, "degraded"=${FRAME_BUDGET_MS.degraded}ms)`,
    )
    .option(
      "--fail-on-budget",
      "Exit 2 when --profile is set and total p95 exceeds the profile budget",
    )
    .allowExcessArguments(false)
    .exitOverride();

  try {
    program.parse(argv, { from: "user" });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    process.stderr.write(`${message}\n`);
    return 1;
  }

  const options = program.opts();
  let raw;
  try {
    raw =
      options.input === undefined
        ? (io.stdin ?? readFileSync(0, "utf8"))
        : readFileSync(options.input, "utf8");
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    process.stderr.write(`failed to read input: ${message}\n`);
    return 1;
  }

  if (typeof raw === "string" && raw.trim() === "") {
    process.stderr.write(
      "measurement failed: empty input (pass real E2E evaluation/draw/total JSON via --input or stdin)\n",
    );
    return 1;
  }

  let report;
  try {
    report = summarizeRuntimeMeasurements(parseMeasurementInputJson(raw));
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    process.stderr.write(`measurement failed: ${message}\n`);
    return 1;
  }

  const text = formatRuntimeMeasurementReport(report);
  const write = io.stdoutWrite ?? ((chunk) => process.stdout.write(chunk));
  try {
    if (options.output === undefined) {
      write(text);
    } else {
      writeFileSync(options.output, text, "utf8");
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    process.stderr.write(`failed to write output: ${message}\n`);
    return 1;
  }

  if (options.profile !== undefined) {
    let budget;
    try {
      budget = evaluateFrameBudget(report, options.profile);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      process.stderr.write(`measurement failed: ${message}\n`);
      return 1;
    }
    process.stderr.write(
      `frame budget ${budget.profile}: total p95=${budget.totalP95Ms}ms budget=${budget.budgetMs}ms ${budget.withinBudget ? "ok" : "exceeded"}\n`,
    );
    if (options.failOnBudget === true && !budget.withinBudget) {
      return 2;
    }
  }

  return 0;
}

function isCliEntry() {
  const entry = process.argv[1];
  if (!entry) {
    return false;
  }
  return import.meta.url === pathToFileURL(entry).href;
}

if (isCliEntry()) {
  process.exitCode = runMeasureRuntimeCli(process.argv.slice(2));
}
