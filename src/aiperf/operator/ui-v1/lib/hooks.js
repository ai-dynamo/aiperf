// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Small collection of shared Preact hooks for the operator UI.
 */

import { useEffect, useRef, useState } from 'preact/hooks';

/** Animate a number toward ``targetValue`` using requestAnimationFrame.
 *
 *  Re-starts the ramp whenever ``targetValue`` changes, easing from the
 *  previously displayed value to the new target over ``duration`` ms with an
 *  easeOutCubic curve (mirrors ``var(--ease-out)`` in CSS). Non-numeric
 *  targets are passed through unchanged so placeholder strings like ``"---"``
 *  keep rendering.
 *
 *  @param {number|string} targetValue
 *  @param {{duration?: number, formatter?: (v:number)=>any}} [opts]
 *  @returns {any} the current animated value (or raw target when not numeric)
 */
export function useCountUp(targetValue, opts = {}) {
  const { duration = 400, formatter = (v) => v } = opts;
  const [display, setDisplay] = useState(targetValue);
  const fromRef = useRef(typeof targetValue === 'number' ? targetValue : 0);
  const rafRef = useRef(0);

  useEffect(() => {
    if (typeof targetValue !== 'number' || !isFinite(targetValue)) {
      setDisplay(targetValue);
      return undefined;
    }
    const from = typeof fromRef.current === 'number' ? fromRef.current : 0;
    const to = targetValue;
    if (from === to) {
      setDisplay(formatter(to));
      return undefined;
    }
    const start = performance.now();
    const step = (now) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      const v = from + (to - from) * eased;
      setDisplay(formatter(v));
      if (t < 1) {
        rafRef.current = requestAnimationFrame(step);
      } else {
        fromRef.current = to;
      }
    };
    rafRef.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(rafRef.current);
  }, [targetValue, duration]);

  return display;
}
