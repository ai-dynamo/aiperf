/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import {
  MEASURE_RUNTIME_SCHEMA_VERSION,
  captureEnvironmentSnapshot,
  captureMemorySnapshot,
  formatRuntimeMeasurementReport,
  parseMeasurementInputJson,
  percentile,
  runMeasureRuntimeCli,
  summarizeRuntimeMeasurements,
  summarizeSeries,
} from "./measure-runtime.mjs";

const FIXED_MEMORY = {
  heapUsedBytes: 1_048_576,
  heapTotalBytes: 2_097_152,
  rssBytes: 4_194_304,
  externalBytes: 8_192,
};

const FIXED_ENVIRONMENT = {
  node: "v22.15.0",
  platform: "linux",
  arch: "x64",
  cpus: 8,
  totalMemoryBytes: 16_000_000_000,
};

const SAMPLES = [
  { evaluationMs: 1.0, drawMs: 2.0, totalMs: 3.5 },
  { evaluationMs: 2.0, drawMs: 2.5, totalMs: 4.8 },
  { evaluationMs: 1.5, drawMs: 3.0, totalMs: 4.9 },
  { evaluationMs: 4.0, drawMs: 5.0, totalMs: 9.5 },
  { evaluationMs: 1.2, drawMs: 2.2, totalMs: 3.6 },
  { evaluationMs: 1.8, drawMs: 2.8, totalMs: 4.7 },
  { evaluationMs: 2.2, drawMs: 3.2, totalMs: 5.5 },
  { evaluationMs: 1.1, drawMs: 2.1, totalMs: 3.4 },
  { evaluationMs: 3.0, drawMs: 4.0, totalMs: 7.2 },
  { evaluationMs: 1.4, drawMs: 2.4, totalMs: 4.0 },
  { evaluationMs: 1.6, drawMs: 2.6, totalMs: 4.3 },
  { evaluationMs: 2.5, drawMs: 3.5, totalMs: 6.1 },
  { evaluationMs: 1.3, drawMs: 2.3, totalMs: 3.8 },
  { evaluationMs: 1.7, drawMs: 2.7, totalMs: 4.5 },
  { evaluationMs: 1.9, drawMs: 2.9, totalMs: 5.0 },
  { evaluationMs: 2.1, drawMs: 3.1, totalMs: 5.3 },
  { evaluationMs: 1.05, drawMs: 2.05, totalMs: 3.2 },
  { evaluationMs: 2.4, drawMs: 3.4, totalMs: 5.9 },
  { evaluationMs: 1.25, drawMs: 2.15, totalMs: 3.55 },
  { evaluationMs: 5.0, drawMs: 6.0, totalMs: 11.5 },
];

test("percentile uses nearest-rank and rejects non-finite values", () => {
  assert.equal(percentile([1, 2, 3, 4], 50), 2);
  assert.equal(percentile([1, 2, 3, 4], 95), 4);
  assert.equal(percentile([10], 50), 10);
  assert.throws(() => percentile([], 50), /non-empty/);
  assert.throws(() => percentile([1, Number.NaN], 50), /non-finite/);
  assert.throws(() => percentile([1], 0), /\(0, 100]/);
});

test("summarizeSeries reports p50, p95, and worst deterministically", () => {
  const summary = summarizeSeries([1, 2, 3, 4, 100]);
  assert.deepEqual(summary, { p50: 3, p95: 100, worst: 100 });
});

test("summarizeRuntimeMeasurements separates evaluation, draw, and total", () => {
  const report = summarizeRuntimeMeasurements({
    samples: SAMPLES,
    memory: FIXED_MEMORY,
    environment: FIXED_ENVIRONMENT,
  });

  assert.equal(report.schemaVersion, MEASURE_RUNTIME_SCHEMA_VERSION);
  assert.equal(report.sampleCount, SAMPLES.length);
  assert.deepEqual(report.memory, FIXED_MEMORY);
  assert.deepEqual(report.environment, FIXED_ENVIRONMENT);

  assert.deepEqual(report.frames.evaluationMs, {
    p50: 1.7,
    p95: 4.0,
    worst: 5.0,
  });
  assert.deepEqual(report.frames.drawMs, {
    p50: 2.7,
    p95: 5.0,
    worst: 6.0,
  });
  assert.deepEqual(report.frames.totalMs, {
    p50: 4.7,
    p95: 9.5,
    worst: 11.5,
  });
});

test("summarizeRuntimeMeasurements rejects non-finite and empty input", () => {
  assert.throws(
    () =>
      summarizeRuntimeMeasurements({
        samples: [{ evaluationMs: Number.POSITIVE_INFINITY, drawMs: 1, totalMs: 1 }],
      }),
    /Infinity|finite|Invalid/,
  );
  assert.throws(
    () =>
      summarizeRuntimeMeasurements({
        samples: [{ evaluationMs: 1, drawMs: Number.NaN, totalMs: 1 }],
      }),
    /NaN|finite|Invalid/,
  );
  assert.throws(() => summarizeRuntimeMeasurements({ samples: [] }), /at least|min|Invalid/i);
  assert.throws(
    () =>
      summarizeRuntimeMeasurements({
        samples: [{ evaluationMs: 1, drawMs: 1, totalMs: 1, extra: true }],
      }),
    /unrecognized|strict|Invalid/i,
  );
});

test("formatRuntimeMeasurementReport emits stable machine-readable JSON", () => {
  const report = summarizeRuntimeMeasurements({
    samples: [
      { evaluationMs: 1, drawMs: 2, totalMs: 3 },
      { evaluationMs: 2, drawMs: 3, totalMs: 5 },
    ],
    memory: FIXED_MEMORY,
    environment: FIXED_ENVIRONMENT,
  });
  const text = formatRuntimeMeasurementReport(report);
  const again = formatRuntimeMeasurementReport(report);
  assert.equal(text, again);
  assert.equal(text.endsWith("\n"), true);
  assert.deepEqual(JSON.parse(text), report);
});

test("parseMeasurementInputJson accepts bare arrays and wrapped objects", () => {
  const bare = parseMeasurementInputJson(
    JSON.stringify([{ evaluationMs: 1, drawMs: 2, totalMs: 3 }]),
  );
  assert.equal(bare.samples.length, 1);

  const wrapped = parseMeasurementInputJson(
    JSON.stringify({
      samples: [{ evaluationMs: 1, drawMs: 2, totalMs: 3 }],
      memory: FIXED_MEMORY,
    }),
  );
  assert.deepEqual(wrapped.memory, FIXED_MEMORY);
});

test("capture helpers produce finite validated snapshots", () => {
  const memory = captureMemorySnapshot({
    rss: 100,
    heapTotal: 50,
    heapUsed: 25,
    external: 5,
    arrayBuffers: 0,
  });
  assert.deepEqual(memory, {
    heapUsedBytes: 25,
    heapTotalBytes: 50,
    rssBytes: 100,
    externalBytes: 5,
  });

  const environment = captureEnvironmentSnapshot({
    node: "v22.0.0",
    platform: "linux",
    arch: "x64",
    cpus: 4,
    totalMemoryBytes: 1024,
  });
  assert.equal(environment.cpus, 4);
  assert.equal(Number.isFinite(environment.totalMemoryBytes), true);
});

test("CLI writes a machine-readable report from an input file", () => {
  const dir = mkdtempSync(join(tmpdir(), "measure-runtime-"));
  const inputPath = join(dir, "samples.json");
  const outputPath = join(dir, "report.json");
  try {
    writeFileSync(
      inputPath,
      JSON.stringify({
        samples: SAMPLES.slice(0, 4),
        memory: FIXED_MEMORY,
        environment: FIXED_ENVIRONMENT,
      }),
      "utf8",
    );

    const chunks = [];
    const code = runMeasureRuntimeCli(["--input", inputPath, "--output", outputPath], {
      stdoutWrite: (text) => chunks.push(text),
    });
    assert.equal(code, 0);
    assert.equal(chunks.length, 0);

    const report = JSON.parse(readFileSync(outputPath, "utf8"));
    assert.equal(report.schemaVersion, MEASURE_RUNTIME_SCHEMA_VERSION);
    assert.equal(report.sampleCount, 4);
    assert.equal(report.frames.totalMs.worst, 9.5);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
