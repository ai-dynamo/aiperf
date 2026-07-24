/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! "Systems Chalk" in-card mini-diagram atoms — the little labeled node chips, dashed arrows, mini
//! bar-charts, cylinder (db) and round nodes that sit inside a `ChalkCard`. Ported faithfully from
//! the approved brainstorm mockup `systems-chalk-hub-spoke.html` (the `.node`/`.arrow`/`.mini-bars`
//! rules). Color comes from `currentColor`: wrap accented atoms in `<span className="text-[accent]">`
//! or pass `accent` to color one against the card's `--accent`.

import clsx from "clsx";

const ACCENT = "var(--accent, var(--color-accent-primary))";

/** The flex row a card's mini-diagram lives in (`.diagram`). */
export function Diagram({ children, className }: { children: React.ReactNode; className?: string }): React.JSX.Element {
  return (
    <div
      className={clsx(
        "my-2.5 flex h-14 items-center justify-center gap-[11px] text-ink-tertiary",
        className,
      )}
    >
      {children}
    </div>
  );
}

/** A labeled bordered chip (`.node`). `accent` colors its border+text with the card accent. */
export function NodeChip({
  children,
  accent = false,
}: {
  children: React.ReactNode;
  accent?: boolean;
}): React.JSX.Element {
  return (
    <span
      className="grid h-[31px] min-w-[36px] place-items-center rounded-[5px] border px-[7px] font-mono text-[8px] font-bold whitespace-nowrap"
      style={accent ? { color: ACCENT, borderColor: ACCENT } : undefined}
    >
      {children}
    </span>
  );
}

/** A round node (`.node.round`) — a small circle, e.g. a queue slot number. */
export function RoundNode({
  children,
  accent = false,
}: {
  children: React.ReactNode;
  accent?: boolean;
}): React.JSX.Element {
  return (
    <span
      className="grid h-[31px] w-[33px] place-items-center rounded-full border font-mono text-[8px] font-bold"
      style={accent ? { color: ACCENT, borderColor: ACCENT } : undefined}
    >
      {children}
    </span>
  );
}

/** A cylinder/database node (`.node.db`) — a KV store, cache, etc. */
export function DbNode({
  children,
  accent = false,
}: {
  children: React.ReactNode;
  accent?: boolean;
}): React.JSX.Element {
  return (
    <span
      className="relative grid h-[31px] min-w-[36px] place-items-center rounded-[50%/17%] border px-[7px] font-mono text-[8px] font-bold before:absolute before:top-[5px] before:right-[-1px] before:left-[-1px] before:h-px before:bg-current before:content-['']"
      style={accent ? { color: ACCENT, borderColor: ACCENT } : undefined}
    >
      {children}
    </span>
  );
}

/** A dashed connector arrow between two diagram atoms (`.arrow`). */
export function MiniArrow(): React.JSX.Element {
  return (
    <span
      className="relative block h-px w-8 opacity-80 after:absolute after:top-[-3px] after:right-[-1px] after:h-[6px] after:w-[6px] after:rotate-45 after:border-t after:border-r after:border-current after:content-['']"
      style={{
        backgroundImage: "repeating-linear-gradient(90deg, currentColor 0 3px, transparent 3px 7px)",
      }}
      aria-hidden="true"
    />
  );
}

/** A bidirectional dashed connector (⇄) for handshake / request-response pairs. */
export function BiArrow(): React.JSX.Element {
  return (
    <span
      className="relative block h-px w-8 opacity-80 before:absolute before:top-[-3px] before:left-[-1px] before:h-[6px] before:w-[6px] before:-rotate-[135deg] before:border-t before:border-r before:border-current before:content-[''] after:absolute after:top-[-3px] after:right-[-1px] after:h-[6px] after:w-[6px] after:rotate-45 after:border-t after:border-r after:border-current after:content-['']"
      style={{
        backgroundImage: "repeating-linear-gradient(90deg, currentColor 0 3px, transparent 3px 7px)",
      }}
      aria-hidden="true"
    />
  );
}

/** A small decision diamond (a rotated square) — a branch/predicate node. */
export function DiamondNode({
  children,
  accent = false,
}: {
  children: React.ReactNode;
  accent?: boolean;
}): React.JSX.Element {
  return (
    <span className="relative inline-grid h-[30px] w-[30px] place-items-center">
      <span
        className="absolute inset-0 rotate-45 rounded-[3px] border"
        style={accent ? { borderColor: "var(--accent, var(--color-accent-primary))" } : undefined}
        aria-hidden="true"
      />
      <span
        className="relative font-mono text-[8px] font-bold"
        style={accent ? { color: "var(--accent, var(--color-accent-primary))" } : undefined}
      >
        {children}
      </span>
    </span>
  );
}

/** A little bar chart (`.mini-bars`), bars colored with the card accent. Heights are 0–100 (%). */
export function MiniBars({ heights }: { heights: readonly number[] }): React.JSX.Element {
  return (
    <span className="flex h-8 items-end gap-[3px]">
      {heights.map((h, i) => (
        <i
          key={i}
          className="block w-[6px] rounded-t-[2px] opacity-85"
          style={{ height: `${h}%`, backgroundColor: ACCENT }}
        />
      ))}
    </span>
  );
}
