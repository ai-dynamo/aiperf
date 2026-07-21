/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { EdgeTypes } from "@xyflow/react";
import { FlowEdge } from "./FlowEdge.js";

/** Registered once, passed to every `<ReactFlow edgeTypes={edgeTypes}>` in the app. */
export const edgeTypes: EdgeTypes = {
  flow: FlowEdge,
};
