// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const PHASE_DONE = new Set(['Succeeded', 'Completed', 'Archived']);

export function trialContributesMetrics(phase) {
  return PHASE_DONE.has(phase);
}
