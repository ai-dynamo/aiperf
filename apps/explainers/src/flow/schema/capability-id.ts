/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { RenderNodeIr } from "./ir.js";

/** Resolves the runtime capability id for a render node, defaulting foundation kinds. */
export function resolveCapabilityId(node: RenderNodeIr): string {
  if ("capabilityId" in node && node.capabilityId !== undefined) {
    return node.capabilityId;
  }
  if ("capability" in node && node.capability !== undefined) {
    return node.capability;
  }
  return `core.${node.kind}`;
}
