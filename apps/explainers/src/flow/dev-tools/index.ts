/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Lazy browser developer diagnostics for compiled Flow deck packages.

export {
  createDevEvaluatorRegistry,
  registeredDevCapabilityIds,
} from "./evaluator-registry.js";
export {
  runDevDiagnostics,
  verifyPackageIr,
  type VerificationFinding,
  type VerifyPackageOptions,
} from "./verify-deck.js";
