/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Deterministic runtime frame measurement summarizer for AIPerf Flow.
 *
 * Consumes per-frame evaluation / draw / total samples and emits a
 * machine-readable quality report with p50, p95, worst, memory, and
 * environment metadata. Importable for tests; runnable as a CLI.
 */

import { readFileSync, writeFileSync } from "node:fs";
import os from "node:os";
import process from "node:process";
import { pathToFileURL } from "node:url";

import { Command } from "commander";
import { z } from "zod";

/** Report schema version for machine consumers. */
export const MEASURE_RUNTIME_SCHEMA_VERSION = 1;

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
 * Build a deterministic machine-readable runtime measurement report.
 *
 * @param {unknown} input Samples plus optional memory / environment snapshots.
 * @returns {z.infer<typeof RuntimeMeasurementReportSchema>}
 */
export function summarizeRuntimeMeasurements(input) {
  const parsed = MeasurementInputSchema.parse(input);
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
 * Accepts either `{ samples: [...] }` or a bare sample array.
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

  if (Array.isArray(decoded)) {
    return MeasurementInputSchema.parse({ samples: decoded });
  }
  return MeasurementInputSchema.parse(decoded);
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
