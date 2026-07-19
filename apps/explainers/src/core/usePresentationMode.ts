/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type RefObject,
} from "react";

const CHROME_IDLE_MS = 3_000;

/** Show presentation controls during activity, then clear them from the stage. */
export function useIdleChrome(
  enabled: boolean,
  pinned = false,
): { chromeVisible: boolean; revealChrome: () => void } {
  const [chromeVisible, setChromeVisible] = useState(true);
  const timerRef = useRef<number | undefined>(undefined);

  const revealChrome = useCallback(() => {
    setChromeVisible(true);
    window.clearTimeout(timerRef.current);
    if (enabled && !pinned) {
      timerRef.current = window.setTimeout(
        () => setChromeVisible(false),
        CHROME_IDLE_MS,
      );
    }
  }, [enabled, pinned]);

  useEffect(() => {
    if (!enabled || pinned) {
      window.clearTimeout(timerRef.current);
      setChromeVisible(true);
      return;
    }

    const reveal = () => revealChrome();
    window.addEventListener("pointermove", reveal, { passive: true });
    window.addEventListener("keydown", reveal);
    window.addEventListener("focusin", reveal);
    revealChrome();

    return () => {
      window.clearTimeout(timerRef.current);
      window.removeEventListener("pointermove", reveal);
      window.removeEventListener("keydown", reveal);
      window.removeEventListener("focusin", reveal);
    };
  }, [enabled, pinned, revealChrome]);

  return { chromeVisible, revealChrome };
}

/**
 * Enter browser fullscreen when available and retain a CSS theater fallback.
 * The fallback also keeps presentation mode active if the browser rejects the
 * fullscreen request (notably embedded and mobile WebKit contexts).
 */
export function usePresentMode(
  shellRef: RefObject<HTMLElement | null>,
): {
  presenting: boolean;
  togglePresent: () => Promise<void>;
} {
  const [pseudoFullscreen, setPseudoFullscreen] = useState(false);
  const [browserFullscreen, setBrowserFullscreen] = useState(false);

  useEffect(() => {
    const syncFullscreen = () => {
      setBrowserFullscreen(document.fullscreenElement === shellRef.current);
    };
    document.addEventListener("fullscreenchange", syncFullscreen);
    syncFullscreen();
    return () => document.removeEventListener("fullscreenchange", syncFullscreen);
  }, [shellRef]);

  const togglePresent = useCallback(async () => {
    if (browserFullscreen) {
      await document.exitFullscreen();
      setPseudoFullscreen(false);
      return;
    }

    if (pseudoFullscreen) {
      setPseudoFullscreen(false);
      return;
    }

    const shell = shellRef.current;
    if (shell?.requestFullscreen) {
      try {
        await shell.requestFullscreen();
        return;
      } catch {
        // Continue into the CSS fallback when the host blocks fullscreen.
      }
    }
    setPseudoFullscreen(true);
  }, [browserFullscreen, pseudoFullscreen, shellRef]);

  return {
    presenting: browserFullscreen || pseudoFullscreen,
    togglePresent,
  };
}
