/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useState } from "react";

/**
 * Available English narration voices, best-default first.
 *
 * Voices load asynchronously in most browsers: the first `getVoices()` call
 * often returns an empty list, and `voiceschanged` fires once the engine has
 * enumerated them.
 */
export function useSpeechVoices(): readonly SpeechSynthesisVoice[] {
  const [voices, setVoices] = useState<readonly SpeechSynthesisVoice[]>([]);

  useEffect(() => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return;

    const update = () => {
      const available = window.speechSynthesis
        .getVoices()
        .filter((voice) => voice.lang.toLowerCase().startsWith("en"))
        .sort((left, right) => {
          if (left.default !== right.default) return left.default ? -1 : 1;
          if (left.localService !== right.localService) return left.localService ? -1 : 1;
          return left.name.localeCompare(right.name);
        });
      setVoices(available);
    };

    update();
    window.speechSynthesis.addEventListener("voiceschanged", update);
    return () => window.speechSynthesis.removeEventListener("voiceschanged", update);
  }, []);

  return voices;
}
