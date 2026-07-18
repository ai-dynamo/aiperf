#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `aiperf-flow` process entry: Commander surface over command services.

import { Command, Option } from "commander";

import {
  buildCommand,
  capabilitiesCommand,
  checkCommand,
  formatCommand,
  inspectCommand,
  type CommandResult,
} from "./commands.js";
import { FLOW_CLI_VERSION } from "./index.js";

function applyResult(result: CommandResult): void {
  if (result.stdout.length > 0) {
    process.stdout.write(result.stdout);
  }
  if (result.stderr.length > 0) {
    process.stderr.write(result.stderr);
  }
  process.exitCode = result.exitCode;
}

async function main(argv: readonly string[]): Promise<void> {
  const program = new Command()
    .name("aiperf-flow")
    .description("AIPerf Flow language toolchain")
    .version(String(FLOW_CLI_VERSION));

  program
    .command("format")
    .description("Canonicalize .flow source formatting")
    .argument("<sources...>", "One or more .flow source paths")
    .option("--check", "Exit non-zero if sources are not already canonical", false)
    .action(async (sources: string[], options: { check: boolean }) => {
      applyResult(await formatCommand({ paths: sources, check: options.check }));
    });

  program
    .command("check")
    .description("Compile .flow sources against foundation capabilities")
    .argument("<sources...>", "One or more .flow source paths")
    .option("--strict", "Treat validation warnings as errors", false)
    .option("--json", "Emit diagnostics as JSON on stdout", false)
    .action(
      async (sources: string[], options: { strict: boolean; json: boolean }) => {
        applyResult(
          await checkCommand({
            paths: sources,
            strict: options.strict,
            json: options.json,
          }),
        );
      },
    );

  program
    .command("build")
    .description("Build a static Flow site (not yet implemented)")
    .argument("<source>", ".flow source path")
    .requiredOption("--out <directory>", "Output directory")
    .option("--strict", "Treat validation warnings as errors", false)
    .option("--clean", "Allow replacing a nonempty output directory", false)
    .action(
      async (
        source: string,
        options: { out: string; strict: boolean; clean: boolean },
      ) => {
        applyResult(
          await buildCommand({
            path: source,
            outDir: options.out,
            strict: options.strict,
            clean: options.clean,
          }),
        );
      },
    );

  program
    .command("inspect")
    .description("Inspect AST, IR, or pack manifest (not yet implemented)")
    .argument("<source>", ".flow source path")
    .addOption(new Option("--ast", "Emit document AST").conflicts(["ir", "manifest"]))
    .addOption(new Option("--ir", "Emit Flow IR").conflicts(["ast", "manifest"]))
    .addOption(
      new Option("--manifest", "Emit pack manifest").conflicts(["ast", "ir"]),
    )
    .action(
      async (
        source: string,
        options: { ast?: boolean; ir?: boolean; manifest?: boolean },
      ) => {
        const mode = options.ast
          ? "ast"
          : options.manifest
            ? "manifest"
            : "ir";
        if (!options.ast && !options.ir && !options.manifest) {
          applyResult({
            exitCode: 1,
            stdout: "",
            stderr: "inspect: one of --ast, --ir, or --manifest is required\n",
          });
          return;
        }
        applyResult(await inspectCommand({ path: source, mode }));
      },
    );

  program
    .command("capabilities")
    .description("List known capability descriptors (not yet implemented)")
    .option("--json", "Emit descriptors as JSON", false)
    .action(async (options: { json: boolean }) => {
      applyResult(await capabilitiesCommand({ json: options.json }));
    });

  await program.parseAsync([...argv]);
}

await main(process.argv);
