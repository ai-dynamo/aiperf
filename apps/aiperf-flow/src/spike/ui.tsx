/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared chrome for the spike pages.
//!
//! Each spike grew its own header, panel, and label markup, which drifted into six different type
//! scales — most of them too small to read comfortably. These are the sizes settled on for the
//! two-clocks page, in one place so a page can be about its subject rather than about its padding.

import type React from "react";

/** Title block. The lede is the one-paragraph statement of what the page is arguing. */
export function SpikeHeader({
  title,
  children,
}: {
  title: string;
  children?: React.ReactNode;
}): React.JSX.Element {
  return (
    <>
      <div className="mb-1 flex items-baseline gap-3">
        <span className="text-sm font-bold uppercase tracking-[0.2em] text-ink-link">Spike</span>
        <h1 className="text-3xl font-extrabold">{title}</h1>
      </div>
      {children !== undefined && (
        <div className="mb-4 max-w-5xl space-y-3 text-base leading-relaxed text-ink-secondary">
          {children}
        </div>
      )}
    </>
  );
}

/** The strip of controls under the header. */
export function ControlBar({ children }: { children: React.ReactNode }): React.JSX.Element {
  return (
    <div className="mb-4 rounded-lg border border-white/10 bg-surface-elevated px-4 py-3">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3">{children}</div>
    </div>
  );
}

/** A button that reads as pressed when `active`, and as a plain control otherwise. */
export function Toggle({
  active,
  onClick,
  children,
  disabled,
  title,
}: {
  active?: boolean;
  onClick: () => void;
  children: React.ReactNode;
  disabled?: boolean;
  title?: string;
}): React.JSX.Element {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`rounded border px-3.5 py-1.5 text-base font-semibold disabled:opacity-35 ${
        active === true
          ? "border-transparent bg-accent-primary text-black"
          : "border-white/15 bg-surface-panel text-ink-secondary"
      }`}
    >
      {children}
    </button>
  );
}

/** A bordered region with a small all-caps label. */
export function Panel({
  label,
  hint,
  children,
  className,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
  className?: string;
}): React.JSX.Element {
  return (
    <section className={`rounded-lg border border-white/10 bg-surface-elevated p-4 ${className ?? ""}`}>
      <div className="mb-2 flex flex-wrap items-baseline gap-x-2">
        <h2 className="text-[12px] font-bold tracking-widest text-ink-secondary">{label}</h2>
        {hint !== undefined && <span className="text-[13px] text-ink-quaternary">{hint}</span>}
      </div>
      {children}
    </section>
  );
}

/** Explanatory text inside a panel. Small, but not smaller than a footnote elsewhere. */
export function Note({ children }: { children: React.ReactNode }): React.JSX.Element {
  return <p className="mt-3 text-[13px] leading-relaxed text-ink-quaternary">{children}</p>;
}

/** The source citation that closes every spike. */
export function SourceNote({ children }: { children: React.ReactNode }): React.JSX.Element {
  return <p className="mt-4 max-w-6xl text-[13px] leading-relaxed text-ink-quaternary">{children}</p>;
}

/** One key for a diagram. `mark` carries the colour; the label never should. */
export function LegendItem({
  mark,
  children,
  color,
}: {
  mark: string;
  children: React.ReactNode;
  color?: string;
}): React.JSX.Element {
  return (
    <span className="flex items-center gap-1.5">
      <span style={{ color: color ?? "var(--color-ink-secondary)" }}>{mark}</span>
      {children}
    </span>
  );
}

/** A row of legend items, sized to sit under a diagram without competing with it. */
export function Legend({ children }: { children: React.ReactNode }): React.JSX.Element {
  return (
    <div className="mb-1 flex flex-wrap items-center gap-x-4 gap-y-1 text-[13px] text-ink-quaternary">
      {children}
    </div>
  );
}

/** A labelled number, for the run-state readouts that sit at the right of a control bar. */
export function Readout({
  label,
  value,
  color,
}: {
  label: string;
  value: React.ReactNode;
  color?: string;
}): React.JSX.Element {
  return (
    <span className="text-base tabular-nums">
      <span className="text-ink-tertiary">{label}</span>{" "}
      <strong style={{ color }}>{value}</strong>
    </span>
  );
}

/**
 * A count against a target, drawn.
 *
 * Several spikes state a "3 of 5 arrived" relationship in prose. Drawing it means the reader sees
 * the distance to the target rather than doing the subtraction.
 */
export function Meter({
  value,
  target,
  color,
  width = 120,
}: {
  value: number;
  target: number;
  color: string;
  width?: number;
}): React.JSX.Element {
  const cells = Math.max(1, target);
  return (
    <span className="inline-flex items-center gap-[3px]" style={{ width }}>
      {Array.from({ length: cells }, (_, i) => (
        <span
          key={i}
          className="h-3 flex-1 rounded-[2px]"
          style={{
            background: i < value ? color : "rgba(255,255,255,0.09)",
            outline: i < value ? "none" : "1px dashed rgba(255,255,255,0.14)",
            outlineOffset: -1,
          }}
        />
      ))}
    </span>
  );
}
