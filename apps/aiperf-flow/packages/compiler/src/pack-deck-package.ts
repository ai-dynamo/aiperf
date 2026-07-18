/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Serialize DeckPackage artifacts as schemaVersion:1 JSON.

import { writeFile } from "node:fs/promises";

import type { DeckPackage } from "@aiperf/flow-schema";

import { canonicalJson } from "./pack.js";

/** Serializes a DeckPackage to deterministic schemaVersion:1 JSON text. */
export function packDeckPackageToJson(pkg: DeckPackage): string {
  const payload: DeckPackage = {
    ...pkg,
    schemaVersion: 1,
  };
  return `${new TextDecoder().decode(canonicalJson(payload))}\n`;
}

/** Writes a DeckPackage to `path` as schemaVersion:1 JSON. */
export async function writeDeckPackage(
  path: string,
  pkg: DeckPackage,
): Promise<void> {
  await writeFile(path, packDeckPackageToJson(pkg), "utf8");
}
