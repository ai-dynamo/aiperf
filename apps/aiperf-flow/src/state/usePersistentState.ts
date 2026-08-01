/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useCallback, useState } from "react";

/**
 * `useState` mirrored into `localStorage` under `${prefix}:${key}`.
 *
 * For settings a viewer should not have to re-pick every visit (narration on/off,
 * chosen voice, playback speed). Transient playback state — whether a deck is
 * currently playing, and which step it is on — is deliberately *not* stored, so a
 * reload always reopens at the start gate.
 *
 * Storage access is guarded: Safari private mode throws on both read and write.
 */
export function usePersistentState<T>(
  prefix: string,
  key: string,
  defaultValue: T,
): [T, (action: T | ((prev: T) => T)) => void] {
  const storageKey = `${prefix}:${key}`;
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = window.localStorage.getItem(storageKey);
      if (raw == null) return defaultValue;
      return JSON.parse(raw) as T;
    } catch {
      return defaultValue;
    }
  });

  const setPersisted = useCallback(
    (action: T | ((prev: T) => T)) => {
      setValue((prev) => {
        const next = typeof action === "function" ? (action as (prev: T) => T)(prev) : action;
        try {
          window.localStorage.setItem(storageKey, JSON.stringify(next));
        } catch {
          /* Storage unavailable; keep the in-memory value. */
        }
        return next;
      });
    },
    [storageKey],
  );

  return [value, setPersisted];
}
