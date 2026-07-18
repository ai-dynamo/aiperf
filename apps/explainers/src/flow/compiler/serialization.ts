/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Browser-safe deterministic DeckPackage serialization.

import type { DeckPackage } from "../schema/index.js";

import { canonicalJson } from "./pack.js";

/** Serializes a DeckPackage to deterministic schemaVersion:1 JSON text. */
export function packDeckPackageToJson(pkg: DeckPackage): string {
  const payload: DeckPackage = { ...pkg, schemaVersion: 1 };
  return `${new TextDecoder().decode(canonicalJson(payload))}\n`;
}
