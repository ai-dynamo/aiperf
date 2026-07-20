/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useLayoutEffect, useRef, useState, type SVGProps } from "react";
import { useHostTheme } from "../ui";
import { usePrefersReducedMotion } from "./usePrefersReducedMotion";

export type MotionSignalProps = Omit<SVGProps<SVGCircleElement>, "color"> & {
  path: string;
  color?: string;
  duration?: string;
  delay?: string;
  /**
   * Timeline-driven progress along `path` in [0, 1].
   * When set, positions the dot via path length instead of looping SMIL.
   */
  progress?: number;
  /** When false, suppress SMIL / progress rendering (paused or inactive cue). */
  active?: boolean;
  /** When true, hide the traveling dot (final-frame / a11y reduced motion). */
  reducedMotion?: boolean;
  /**
   * When true, freeze the indefinite SMIL loop in place without unmounting
   * it (playback paused). Ignored in timeline `progress` mode, where the
   * dot position is already driven by (frozen) `playbackTimeMs` upstream.
   */
  paused?: boolean;
};

function clamp01(value: number): number {
  if (value <= 0) {
    return 0;
  }
  if (value >= 1) {
    return 1;
  }
  return value;
}

/**
 * Traveling accent dot along an authored SVG path.
 * Supports indefinite SMIL (MentalModel parity) or timeline `progress`.
 */
export function MotionSignal({
  path,
  color,
  duration = "2.2s",
  delay = "0s",
  progress,
  active = true,
  reducedMotion: reducedMotionProp,
  paused = false,
  className,
  r = 5,
  children,
  "aria-hidden": ariaHidden = true,
  ...circleProps
}: MotionSignalProps) {
  const theme = useHostTheme();
  const prefersReducedMotion = usePrefersReducedMotion();
  const reducedMotion = reducedMotionProp === true || prefersReducedMotion;
  const motionClassName = ["motion-signal", className].filter(Boolean).join(" ");
  const measureRef = useRef<SVGPathElement | null>(null);
  const circleRef = useRef<SVGCircleElement | null>(null);
  const [point, setPoint] = useState<Readonly<{ x: number; y: number }> | null>(
    null,
  );

  const timelineMode = typeof progress === "number" && Number.isFinite(progress);

  // The indefinite SMIL loop below is un-keyed to `playbackTimeMs` (see the
  // `active` doc comment on why it stays mounted across pause/resume), so
  // freezing it takes the SVG document's own pause/unpause rather than a
  // React re-render. Multiple mounted SMIL dots calling this redundantly is
  // harmless: `pauseAnimations`/`unpauseAnimations` are idempotent flags on
  // the owning document's animation timeline, not per-element toggles.
  useLayoutEffect(() => {
    if (timelineMode) {
      return;
    }
    const svg = circleRef.current?.ownerSVGElement;
    if (svg === null || svg === undefined) {
      return;
    }
    if (typeof svg.pauseAnimations !== "function") {
      return;
    }
    if (paused) {
      svg.pauseAnimations();
    } else if (typeof svg.unpauseAnimations === "function") {
      svg.unpauseAnimations();
    }
  }, [timelineMode, paused]);

  useLayoutEffect(() => {
    if (!timelineMode || reducedMotion || !active) {
      setPoint(null);
      return;
    }
    const element = measureRef.current;
    if (element === null) {
      return;
    }
    const length = element.getTotalLength();
    if (!(length > 0)) {
      setPoint(null);
      return;
    }
    const along = element.getPointAtLength(clamp01(progress) * length);
    setPoint({ x: along.x, y: along.y });
  }, [timelineMode, progress, path, reducedMotion, active]);

  if (reducedMotion || !active) {
    return null;
  }

  if (timelineMode) {
    const p = clamp01(progress as number);
    // Fade in early, hold, fade out near the end — mirrors SMIL keyTimes.
    let opacity = 1;
    if (p <= 0) {
      opacity = 0;
    } else if (p < 0.08) {
      opacity = p / 0.08;
    } else if (p > 0.9) {
      opacity = clamp01((1 - p) / 0.1);
    }
    if (opacity <= 0 || point === null) {
      return (
        <path
          ref={measureRef}
          d={path}
          fill="none"
          stroke="none"
          aria-hidden="true"
          focusable={false}
        />
      );
    }
    return (
      <>
        <path
          ref={measureRef}
          d={path}
          fill="none"
          stroke="none"
          aria-hidden="true"
          focusable={false}
        />
        <circle
          {...circleProps}
          cx={point.x}
          cy={point.y}
          r={r}
          fill={color ?? theme.category.green}
          opacity={opacity}
          className={motionClassName}
          aria-hidden={ariaHidden}
          focusable={false}
        >
          {children}
        </circle>
      </>
    );
  }

  return (
    <circle
      {...circleProps}
      ref={circleRef}
      r={r}
      fill={color ?? theme.category.green}
      // SMIL opacity animation begins at `delay`; stay invisible until then
      // so the dot does not flash at the SVG origin (0,0).
      opacity={0}
      className={motionClassName}
      aria-hidden={ariaHidden}
      data-motion-paused={paused ? "true" : undefined}
    >
      <animate
        attributeName="opacity"
        values="0;1;1;0"
        keyTimes="0;0.08;0.9;1"
        calcMode="spline"
        keySplines="0.4 0 0.2 1;0.4 0 0.2 1;0.4 0 0.2 1"
        begin={delay}
        dur={duration}
        repeatCount="indefinite"
        fill="freeze"
      />
      <animateMotion
        path={path}
        begin={delay}
        dur={duration}
        repeatCount="indefinite"
        fill="freeze"
        calcMode="spline"
        keyTimes="0;1"
        keySplines="0.45 0.05 0.55 0.95"
      />
      {children}
    </circle>
  );
}
