// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { DisplayList, SourceReference } from "../display-list.js";

/** One semantic entity exposed consistently by every render backend. */
export type SemanticEntityProjection = Readonly<{
  id: string;
  label: string;
  role?: string;
  description?: string;
  kind?: string;
  evidenceIds?: readonly string[];
  source?: SourceReference;
}>;

/** One directed semantic relationship between projected entities. */
export type SemanticRelationProjection = Readonly<{
  id: string;
  fromId: string;
  toId: string;
  label?: string;
  role?: string;
  source?: SourceReference;
}>;

/** Backend-neutral accessibility and interaction meaning for a scene. */
export type SemanticProjection = Readonly<{
  sceneId: string;
  entities: readonly SemanticEntityProjection[];
  relations: readonly SemanticRelationProjection[];
  readingOrder: readonly string[];
  transcriptCueId?: string;
  captions?: readonly string[];
}>;

/** One immutable scene snapshot at an integer virtual time. */
export type EvaluatedScene = Readonly<{
  sceneId: string;
  atMs: number;
  displayList: DisplayList;
  semantic: SemanticProjection;
}>;
