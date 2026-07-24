/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! "Systems Chalk" hub-and-spoke scene — a central hub card ringed by spoke `ChalkCard`s on an
//! ellipse, each joined to the hub by a dashed guide wire (the first optionally a live cyan one).
//! Ported/generalized from the approved mockup `systems-chalk-hub-spoke.html` (`.field`/`.hub`/
//! `.wires`/`.card` positions): the mockup fixes seven positions; this places any number of spokes
//! evenly around the hub and draws each wire in the same viewBox space the cards are laid out in.

import clsx from "clsx";
import { ChalkCard, type ChalkCardProps } from "./ChalkCard.js";

// The wire/placement viewBox. Cards are positioned as a %-of-container derived from these coords, and
// the wires are drawn in the same space (preserveAspectRatio="none"), so the two always align.
const VB_W = 1000;
const VB_H = 760;
const CX = VB_W / 2;
const CY = VB_H / 2;
const RX = 400; // ellipse radii — the ring the spoke centers sit on (big enough that the
const RY = 312; // top/bottom cards clear the central hub)

export interface HubSpokeProps {
  /** The center hub. */
  hub: { kicker?: string; title: string; body?: string };
  /** Spoke cards, placed evenly around the hub starting at the top and going clockwise. */
  spokes: ReadonlyArray<ChalkCardProps>;
  /** Index of the spoke whose wire is drawn "live" (animated cyan). Default 0; -1 for none. */
  liveWire?: number;
  className?: string;
}

interface Placed {
  x: number;
  y: number;
}

/** Evenly place `n` points on the ellipse, starting at top (-90°), clockwise. */
function ringPoints(n: number): Placed[] {
  return Array.from({ length: n }, (_, i) => {
    const ang = -Math.PI / 2 + (2 * Math.PI * i) / Math.max(n, 1);
    return { x: CX + RX * Math.cos(ang), y: CY + RY * Math.sin(ang) };
  });
}

/** A curved guide wire from the hub center to a spoke point. */
function wirePath(p: Placed): string {
  const c1x = CX + (p.x - CX) * 0.36;
  const c1y = CY + (p.y - CY) * 0.12;
  const c2x = CX + (p.x - CX) * 0.78;
  const c2y = p.y;
  return `M${CX} ${CY} C${c1x} ${c1y} ${c2x} ${c2y} ${p.x} ${p.y}`;
}

/**
 * A hub-and-spoke scene. Give it a `hub` and the `spokes`; it rings the spokes around the hub and
 * wires each to it. Responsive: collapses to a single stacked column on narrow screens.
 */
export function HubSpoke({ hub, spokes, liveWire = 0, className }: HubSpokeProps): React.JSX.Element {
  const points = ringPoints(spokes.length);
  return (
    <div className={clsx("@container", className)}>
      {/* Ring layout on wide viewports; stacked column when narrow. */}
      <div className="relative mx-auto hidden h-[820px] w-full @2xl:block">
        <svg
          className="pointer-events-none absolute inset-0 z-0 overflow-visible"
          viewBox={`0 0 ${VB_W} ${VB_H}`}
          preserveAspectRatio="none"
          aria-hidden="true"
          width="100%"
          height="100%"
        >
          {points.map((p, i) => (
            <path
              key={i}
              d={wirePath(p)}
              fill="none"
              strokeLinecap="round"
              stroke={i === liveWire ? "var(--color-accent-primary)" : "var(--color-ink-quaternary)"}
              strokeWidth={i === liveWire ? 2 : 1.5}
              strokeDasharray={i === liveWire ? undefined : "3 7"}
              opacity={i === liveWire ? 0.9 : 0.6}
              style={
                i === liveWire
                  ? { filter: "drop-shadow(0 0 4px color-mix(in srgb, var(--color-accent-primary) 30%, transparent))" }
                  : undefined
              }
            />
          ))}
        </svg>

        <HubCard hub={hub} className="absolute top-1/2 left-1/2 z-[3] -translate-x-1/2 -translate-y-1/2" />

        {spokes.map((spoke, i) => {
          const p = points[i]!;
          return (
            <div
              key={i}
              className="absolute z-[2] w-[252px]"
              style={{ left: `${(p.x / VB_W) * 100}%`, top: `${(p.y / VB_H) * 100}%`, transform: "translate(-50%, -50%)" }}
            >
              <ChalkCard {...spoke} />
            </div>
          );
        })}
      </div>

      {/* Narrow fallback: hub then spokes in a single readable column. */}
      <div className="grid grid-cols-1 gap-3 @2xl:hidden">
        <HubCard hub={hub} />
        {spokes.map((spoke, i) => (
          <ChalkCard key={i} {...spoke} />
        ))}
      </div>
    </div>
  );
}

/** The center hub — a violet-tinted card with cyan connection ports (`.hub`). */
function HubCard({
  hub,
  className,
}: {
  hub: HubSpokeProps["hub"];
  className?: string;
}): React.JSX.Element {
  return (
    <div
      className={clsx(
        "grid w-[250px] place-items-center rounded-[18px] border px-5 py-4 text-center",
        "border-[color:color-mix(in_srgb,var(--color-category-purple)_50%,transparent)]",
        "shadow-[0_18px_60px_rgba(0,0,0,0.34)]",
        // subtle violet→cyan wash over the elevated surface
        "bg-[linear-gradient(145deg,color-mix(in_srgb,var(--color-category-purple)_20%,transparent),color-mix(in_srgb,var(--color-category-cyan)_6%,transparent)),var(--color-surface-elevated)]",
        // cyan connection ports on the left/right edges
        "relative before:absolute before:top-1/2 before:left-[-6px] before:h-2 before:w-2 before:-translate-y-1/2 before:rounded-full before:border-2 before:border-accent-primary before:bg-surface-page before:content-['']",
        "after:absolute after:top-1/2 after:right-[-6px] after:h-2 after:w-2 after:-translate-y-1/2 after:rounded-full after:border-2 after:border-accent-primary after:bg-surface-page after:content-['']",
        className,
      )}
    >
      {hub.kicker !== undefined && (
        <div className="font-mono text-[10px] font-bold tracking-[0.18em] text-category-purple">
          {hub.kicker}
        </div>
      )}
      <h2 className="mt-2 mb-1 text-[25px] leading-[1.05] font-[610] tracking-[-0.03em] text-ink-primary">
        {hub.title}
      </h2>
      {hub.body !== undefined && (
        <p className="m-0 max-w-[190px] text-[12px] leading-[1.5] text-ink-secondary">{hub.body}</p>
      )}
    </div>
  );
}
