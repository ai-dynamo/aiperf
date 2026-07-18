/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/** Source location contracts shared by Flow packages. */

export type SourcePosition = Readonly<{
  offset: number;
  line: number;
  column: number;
}>;

export type SourceRange = Readonly<{
  source: string;
  start: SourcePosition;
  end: SourcePosition;
}>;
