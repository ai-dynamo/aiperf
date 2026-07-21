/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Node, Edge } from "@xyflow/react";

export type SlideDefinition = {
  id: string;
  eyebrow: string;
  title: string;
  lede: string;
  narration: string;
  caption: string;
  nodes: Node[];
  edges: Edge[];
  /** Reveal order for `nodes` (by id); defaults to `nodes`' own order if omitted. */
  revealOrder?: readonly string[];
};

export type DeckDefinition = {
  id: string;
  title: string;
  slides: readonly SlideDefinition[];
};
