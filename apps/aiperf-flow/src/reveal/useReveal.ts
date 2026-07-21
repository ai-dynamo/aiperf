/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useState } from "react";

const DEFAULT_STEP_MS = 220;

/**
 * Reveals `order`'s ids one at a time on a fixed-interval timer, starting
 * with the first id visible immediately. Mirrors `.flow`'s
 * `after X reveal Y duration N` chained-timeline convention without a
 * parsed timeline language: deck authors just pass the reveal order.
 */
export function useReveal(
  order: readonly string[],
  opts?: { stepMs?: number },
): ReadonlySet<string> {
  const stepMs = opts?.stepMs ?? DEFAULT_STEP_MS;
  const [revealedCount, setRevealedCount] = useState(order.length > 0 ? 1 : 0);

  useEffect(() => {
    if (order.length === 0) {
      return;
    }

    const timers: NodeJS.Timeout[] = [];
    for (let i = 1; i < order.length; i++) {
      const nextCount = i + 1;
      const timeDelay = i * stepMs;
      timers.push(
        setTimeout(() => {
          setRevealedCount(nextCount);
        }, timeDelay),
      );
    }

    return () => timers.forEach((timer) => clearTimeout(timer));
  }, [order.length, stepMs]);

  return new Set(order.slice(0, revealedCount));
}
