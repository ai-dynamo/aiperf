// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public exports for `@aiperf/flow-cli`.

export const FLOW_CLI_VERSION = 1 as const;

export {
  buildCommand,
  capabilitiesCommand,
  checkCommand,
  formatCommand,
  inspectCommand,
  type BuildRequest,
  type CapabilitiesRequest,
  type CheckRequest,
  type CommandResult,
  type FormatRequest,
  type InspectRequest,
} from "./commands.js";
