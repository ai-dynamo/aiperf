// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import { z } from "zod";

const contentSchema = z
  .object({
    schemaVersion: z.literal(1),
    pages: z
      .array(
        z
          .object({
            id: z.string().min(1),
            title: z.string().min(1),
            route: z.string().startsWith("/"),
          })
          .strict(),
      )
      .min(1),
  })
  .strict();

const contentUrl = new URL("../content/foundation.json", import.meta.url);
const content = JSON.parse(await readFile(fileURLToPath(contentUrl), "utf8"));
contentSchema.parse(content);

console.log("Architecture Atlas foundation content is valid.");
