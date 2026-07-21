/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Small presentation helpers shared across the Velo mechanism pages. Scoped to this deck
//! folder (not a shared `src/prose` primitive): every mechanism page opens with the same
//! eyebrow/title/sentence header, and every page highlights "active" React Flow nodes with
//! the same cyan accent, so the class strings live here as literals for Tailwind's scanner.

import clsx from "clsx";
import { Eyebrow } from "../../prose/Eyebrow.js";
import { inkClassName } from "../../theme/tokens.js";

/**
 * Node highlight class — a cyan border + cyan ink, expressed as complete literal strings so
 * Tailwind's JIT compiler emits them (a `border-category-${role}` interpolation would be purged).
 * `undefined` leaves a node at its component-default border.
 */
export const NODE_ACTIVE = "border-category-cyan text-category-cyan";

/** Standard eyebrow / title / sentence header shared by every mechanism instrument page. */
export function MechHeader({
  eyebrow,
  title,
  sentence,
}: {
  eyebrow: string;
  title: string;
  sentence: string;
}): React.JSX.Element {
  return (
    <div>
      <Eyebrow tone="cyan">{eyebrow}</Eyebrow>
      <h2 className={clsx("mt-1 text-lg font-semibold", inkClassName("primary"))}>{title}</h2>
      <p className={clsx("mt-1 max-w-3xl text-sm", inkClassName("secondary"))}>{sentence}</p>
    </div>
  );
}
