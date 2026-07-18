// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Evaluated action and progress for one authored timeline target. */
export type TimelineTargetState = Readonly<{
  action: string;
  progress: number;
}>;

/** Immutable timeline state at one canonical scene time. */
export type TimelineSnapshot = Readonly<{
  timeMs: number;
  complete: boolean;
  targets: Readonly<Record<string, TimelineTargetState>>;
}>;
