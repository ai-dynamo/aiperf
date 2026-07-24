/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! "Systems Chalk" card — a numbered badge + title, an optional in-card mini-diagram, and a body
//! line, on a neutral softly-shadowed dark card whose accent is set per card. Ported from the
//! approved mockup `systems-chalk-hub-spoke.html` (`.card` / `.card-head` / `.number` / `.card h3`
//! / `.card p`). The card's `--accent` drives the number badge, the diagram accents, and the hover
//! border. Presentational and layout-agnostic — used as a spoke in `HubSpoke` or standalone.

import clsx from "clsx";
import type { CategoryRole } from "../theme/tokens.js";

const ACCENT = "var(--accent, var(--color-accent-primary))";

export interface ChalkCardProps {
  /** Accent category color for this card (sets `--accent`). Defaults to the primary cyan. */
  accent?: CategoryRole;
  /** Small number/label in the badge (e.g. a step ordinal). Omit for no badge. */
  badge?: number | string;
  /** Card heading. */
  title: string;
  /** Optional in-card mini-diagram (compose `MiniDiagram` atoms inside a `Diagram`). */
  diagram?: React.ReactNode;
  /** Body line under the diagram. */
  children?: React.ReactNode;
  className?: string;
}

/** A Systems Chalk card. Set `accent` to color its badge/diagram/hover; add a `diagram`. */
export function ChalkCard({
  accent,
  badge,
  title,
  diagram,
  children,
  className,
}: ChalkCardProps): React.JSX.Element {
  const accentVar = accent ? `var(--color-category-${accent})` : undefined;
  return (
    <article
      className={clsx(
        "rounded-[13px] border border-white/10 bg-surface-elevated px-4 pt-[15px] pb-3.5",
        "shadow-[0_12px_28px_rgba(0,0,0,0.2)] transition-[transform,border-color,background-color]",
        "hover:-translate-y-0.5 hover:border-[color:var(--accent,var(--color-accent-primary))] hover:bg-surface-panel",
        className,
      )}
      style={accentVar ? ({ "--accent": accentVar } as React.CSSProperties) : undefined}
    >
      <div className="flex items-center gap-[9px]">
        {badge !== undefined && (
          <span
            className="grid h-[22px] w-[22px] shrink-0 place-items-center rounded-full border font-mono text-[10px] font-bold"
            style={{ color: ACCENT, borderColor: ACCENT }}
          >
            {badge}
          </span>
        )}
        <h3 className="m-0 text-[13px] font-[650] tracking-[-0.01em] text-ink-primary">{title}</h3>
      </div>
      {diagram}
      {children !== undefined && (
        <p className="m-0 text-[10px] leading-[1.4] text-ink-secondary">{children}</p>
      )}
    </article>
  );
}
