// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { execFile } from "node:child_process";
import { dirname, resolve } from "node:path";
import { promisify } from "node:util";
import { fileURLToPath, pathToFileURL } from "node:url";

import { z } from "zod";

import { architectureCatalog } from "../src/content/index";
import {
  validateArchitectureCatalog,
  validateWorkspaceCrates,
} from "../src/domain/integrity";

const execFileAsync = promisify(execFile);
const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const repositoryPath = resolve(scriptDirectory, "../../..");
const repositoryRoot = pathToFileURL(`${repositoryPath}/`);

const cargoMetadataSchema = z.object({
  packages: z.array(
    z.object({
      name: z.string().min(1),
      manifest_path: z.string().min(1),
      dependencies: z.array(
        z.object({
          kind: z.enum(["normal", "build", "dev"]).nullable().optional(),
          name: z.string().min(1),
          path: z.string().nullable().optional(),
        }).passthrough(),
      ),
    }),
  ),
});

async function cargoMetadata() {
  const environment = { ...process.env };
  delete environment.RUSTC_WRAPPER;
  delete environment.RUSTC_WORKSPACE_WRAPPER;
  const { stdout: cargoPath } = await execFileAsync("rustup", ["which", "cargo"], {
    cwd: repositoryPath,
    env: environment,
  });
  const { stdout } = await execFileAsync(
    cargoPath.trim(),
    ["metadata", "--no-deps", "--format-version", "1"],
    {
      cwd: repositoryPath,
      env: environment,
      maxBuffer: 16 * 1024 * 1024,
    },
  );
  return cargoMetadataSchema.parse(JSON.parse(stdout));
}

await validateArchitectureCatalog(architectureCatalog, repositoryRoot);
const metadata = await cargoMetadata();
validateWorkspaceCrates(
  architectureCatalog,
  metadata.packages,
  repositoryRoot,
);

console.log(
  `Architecture Atlas content is valid: ${architectureCatalog.components.length} components, ${architectureCatalog.edges.length} edges, ${architectureCatalog.crates.length} crates.`,
);
