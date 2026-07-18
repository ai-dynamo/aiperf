// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Testable command services for the `aiperf-flow` CLI.
//!
//! Commander wiring lives in `main.ts`; these functions own file I/O and
//! diagnostics so unit tests can exercise format/check without spawning a
//! process.

import { compileSource, packFlow, type PackedFile } from "@aiperf/flow-compiler";
import { formatDocument, parseDocument } from "@aiperf/flow-language";
import {
  FOUNDATION_CAPABILITIES,
  hasErrors,
  type Diagnostic,
} from "@aiperf/flow-schema";
import {
  lstat,
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  realpath,
  rename,
  rm,
  writeFile,
} from "node:fs/promises";
import {
  basename,
  dirname,
  isAbsolute,
  join,
  relative,
  resolve,
  sep,
} from "node:path";

/** Exit code, stdout, and stderr for one CLI command invocation. */
export type CommandResult = Readonly<{
  exitCode: number;
  stdout: string;
  stderr: string;
}>;

export type FormatRequest = Readonly<{
  paths: readonly string[];
  check: boolean;
}>;

export type CheckRequest = Readonly<{
  paths: readonly string[];
  strict: boolean;
  json: boolean;
}>;

export type InspectRequest = Readonly<{
  path: string;
  mode: "ast" | "ir" | "manifest";
}>;

export type CapabilitiesRequest = Readonly<{
  json: boolean;
}>;

export type BuildRequest = Readonly<{
  path: string;
  outDir: string;
  strict: boolean;
  clean: boolean;
}>;

function emptyResult(exitCode: number, stdout = "", stderr = ""): CommandResult {
  return { exitCode, stdout, stderr };
}

function formatHumanDiagnostic(diagnostic: Diagnostic): string {
  const { range } = diagnostic;
  const location = `${range.source}:${range.start.line}:${range.start.column}`;
  const repair = diagnostic.repair === undefined ? "" : ` (${diagnostic.repair})`;
  return `${location}: ${diagnostic.severity} ${diagnostic.code}: ${diagnostic.message}${repair}`;
}

function serializeDiagnosticsJson(diagnostics: readonly Diagnostic[]): string {
  return `${JSON.stringify(
    diagnostics.map((diagnostic) => ({
      code: diagnostic.code,
      severity: diagnostic.severity,
      message: diagnostic.message,
      source: diagnostic.range.source,
      start: diagnostic.range.start,
      end: diagnostic.range.end,
      ...(diagnostic.repair === undefined ? {} : { repair: diagnostic.repair }),
    })),
    null,
    2,
  )}\n`;
}

function renderDiagnostics(
  diagnostics: readonly Diagnostic[],
  json: boolean,
): string {
  if (diagnostics.length === 0) {
    return json ? "[]\n" : "";
  }
  if (json) {
    return serializeDiagnosticsJson(diagnostics);
  }
  return `${diagnostics.map(formatHumanDiagnostic).join("\n")}\n`;
}

/** Formats one or more `.flow` sources, optionally checking without writing. */
export async function formatCommand(request: FormatRequest): Promise<CommandResult> {
  if (request.paths.length === 0) {
    return emptyResult(1, "", "format: at least one source path is required\n");
  }

  const stderrParts: string[] = [];
  let exitCode = 0;
  let rewritten = 0;
  let unchanged = 0;

  for (const path of request.paths) {
    let source: string;
    try {
      source = await readFile(path, "utf8");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      stderrParts.push(`format: failed to read ${path}: ${message}`);
      exitCode = 1;
      continue;
    }

    const parsed = parseDocument(source, path);
    if (!parsed.ok) {
      stderrParts.push(renderDiagnostics(parsed.diagnostics, false).trimEnd());
      exitCode = 1;
      continue;
    }

    const formatted = formatDocument(parsed.value);
    if (formatted === source) {
      unchanged += 1;
      continue;
    }

    if (request.check) {
      stderrParts.push(`format: ${path} is not canonically formatted`);
      exitCode = 1;
      continue;
    }

    try {
      await writeFile(path, formatted, "utf8");
      rewritten += 1;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      stderrParts.push(`format: failed to write ${path}: ${message}`);
      exitCode = 1;
    }
  }

  const summary =
    request.check || exitCode !== 0
      ? ""
      : `Formatted ${rewritten} file(s); ${unchanged} already canonical.\n`;

  const stderr = stderrParts.length === 0 ? "" : `${stderrParts.join("\n")}\n`;
  return { exitCode, stdout: summary, stderr };
}

/** Compiles sources against foundation capabilities and reports diagnostics. */
export async function checkCommand(request: CheckRequest): Promise<CommandResult> {
  if (request.paths.length === 0) {
    return emptyResult(1, "", "check: at least one source path is required\n");
  }

  const allDiagnostics: Diagnostic[] = [];
  let exitCode = 0;

  for (const path of request.paths) {
    let source: string;
    try {
      source = await readFile(path, "utf8");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      allDiagnostics.push({
        code: "cli.read",
        severity: "error",
        message: `failed to read ${path}: ${message}`,
        range: {
          source: path,
          start: { offset: 0, line: 1, column: 1 },
          end: { offset: 0, line: 1, column: 1 },
        },
      });
      exitCode = 1;
      continue;
    }

    const result = compileSource({
      source,
      sourceName: path,
      capabilities: FOUNDATION_CAPABILITIES,
      strict: request.strict,
    });

    allDiagnostics.push(...result.diagnostics);
    if (!result.ok || hasErrors(result.diagnostics)) {
      exitCode = 1;
    }
  }

  const rendered = renderDiagnostics(allDiagnostics, request.json);
  if (request.json) {
    return { exitCode, stdout: rendered, stderr: "" };
  }

  return {
    exitCode,
    stdout: exitCode === 0 ? "ok\n" : "",
    stderr: rendered,
  };
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function isMissingPath(error: unknown): boolean {
  return (
    error instanceof Error &&
    "code" in error &&
    (error as NodeJS.ErrnoException).code === "ENOENT"
  );
}

function validatePackedFiles(files: readonly PackedFile[]): string | undefined {
  const paths = new Set<string>();
  for (const file of files) {
    const segments = file.path.split("/");
    if (
      file.path === "" ||
      isAbsolute(file.path) ||
      file.path.includes("\\") ||
      segments.some(
        (segment) => segment === "" || segment === "." || segment === "..",
      )
    ) {
      return `unsafe packed file path "${file.path}"`;
    }
    if (paths.has(file.path)) {
      return `duplicate packed file path "${file.path}"`;
    }
    paths.add(file.path);
  }
  return undefined;
}

function pathContains(parent: string, child: string): boolean {
  const pathFromParent = relative(resolve(parent), resolve(child));
  return (
    pathFromParent === "" ||
    (pathFromParent !== ".." &&
      !pathFromParent.startsWith(`..${sep}`) &&
      !isAbsolute(pathFromParent))
  );
}

async function inspectOutputDirectory(
  outDir: string,
  clean: boolean,
): Promise<{ exists: boolean } | string> {
  try {
    const stats = await lstat(outDir);
    if (stats.isSymbolicLink()) {
      return `output path ${outDir} is a symbolic link`;
    }
    if (!stats.isDirectory()) {
      return `output path ${outDir} exists and is not a directory`;
    }
    if (!clean && (await readdir(outDir)).length > 0) {
      return `output directory ${outDir} is not empty; pass --clean to replace it`;
    }
    return { exists: true };
  } catch (error) {
    if (isMissingPath(error)) {
      return { exists: false };
    }
    return `failed to inspect output directory ${outDir}: ${errorMessage(error)}`;
  }
}

async function writePackedFiles(
  stagingDir: string,
  files: readonly PackedFile[],
): Promise<void> {
  for (const file of files) {
    const destination = join(stagingDir, ...file.path.split("/"));
    if (!pathContains(stagingDir, destination)) {
      throw new Error(
        `packed file path escapes the output directory: ${file.path}`,
      );
    }
    await mkdir(dirname(destination), { recursive: true });
    await writeFile(destination, file.content, { flag: "wx" });
  }
}

async function publishPackedFiles(
  outDir: string,
  files: readonly PackedFile[],
  outputExists: boolean,
): Promise<void> {
  const targetDir = resolve(outDir);
  const parent = dirname(targetDir);
  await mkdir(parent, { recursive: true });
  const stagingDir = await mkdtemp(join(parent, `.${basename(outDir)}.tmp-`));
  const backupDir = `${stagingDir}.previous`;

  try {
    await writePackedFiles(stagingDir, files);
    if (!outputExists) {
      await rename(stagingDir, targetDir);
      return;
    }

    await rename(targetDir, backupDir);
    try {
      await rename(stagingDir, targetDir);
    } catch (error) {
      await rename(backupDir, targetDir);
      throw error;
    }
    try {
      await rm(backupDir, { recursive: true, force: true });
    } catch (error) {
      await rename(targetDir, stagingDir);
      await rename(backupDir, targetDir);
      await rm(stagingDir, { recursive: true, force: true });
      throw error;
    }
  } catch (error) {
    await rm(stagingDir, { recursive: true, force: true });
    throw error;
  }
}

/** Compiles one source and safely publishes its deterministic packed files. */
export async function buildCommand(request: BuildRequest): Promise<CommandResult> {
  let source: string;
  try {
    source = await readFile(request.path, "utf8");
  } catch (error) {
    return emptyResult(
      2,
      "",
      `build: failed to read ${request.path}: ${errorMessage(error)}\n`,
    );
  }

  const compiled = compileSource({
    source,
    sourceName: request.path,
    capabilities: FOUNDATION_CAPABILITIES,
    strict: request.strict,
  });
  if (!compiled.ok || hasErrors(compiled.diagnostics)) {
    return emptyResult(1, "", renderDiagnostics(compiled.diagnostics, false));
  }

  const packed = packFlow(compiled.value, request.path);
  const unsafePath = validatePackedFiles(packed.files);
  if (unsafePath !== undefined) {
    return emptyResult(2, "", `build: ${unsafePath}\n`);
  }

  if (request.clean && pathContains(request.outDir, request.path)) {
    return emptyResult(
      2,
      "",
      `build: refusing --clean because source ${request.path} is inside output directory ${request.outDir}\n`,
    );
  }

  const output = await inspectOutputDirectory(request.outDir, request.clean);
  if (typeof output === "string") {
    return emptyResult(2, "", `build: ${output}\n`);
  }
  if (request.clean && output.exists) {
    try {
      if (
        pathContains(await realpath(request.outDir), await realpath(request.path))
      ) {
        return emptyResult(
          2,
          "",
          `build: refusing --clean because source ${request.path} resolves inside output directory ${request.outDir}\n`,
        );
      }
    } catch (error) {
      return emptyResult(
        2,
        "",
        `build: failed to resolve source and output paths: ${errorMessage(
          error,
        )}\n`,
      );
    }
  }

  try {
    await publishPackedFiles(request.outDir, packed.files, output.exists);
  } catch (error) {
    return emptyResult(
      2,
      "",
      `build: failed to write ${request.outDir}: ${errorMessage(error)}\n`,
    );
  }

  return {
    exitCode: 0,
    stdout: `Built ${packed.files.length} file(s) in ${request.outDir} (pack ${packed.manifest.id}, content hash ${packed.manifest.contentHash}).\n`,
    stderr: renderDiagnostics(compiled.diagnostics, false),
  };
}

/** Stub: AST/IR/manifest inspect is not wired yet. */
export async function inspectCommand(request: InspectRequest): Promise<CommandResult> {
  return emptyResult(
    1,
    "",
    `inspect: not implemented yet (source=${request.path}, mode=${request.mode})\n`,
  );
}

/** Stub: capability listing is not wired yet. */
export async function capabilitiesCommand(
  request: CapabilitiesRequest,
): Promise<CommandResult> {
  void request;
  return emptyResult(1, "", "capabilities: not implemented yet\n");
}
