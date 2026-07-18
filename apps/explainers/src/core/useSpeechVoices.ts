import { useEffect, useState } from "react";

export function useSpeechVoices(): readonly SpeechSynthesisVoice[] {
  const [voices, setVoices] = useState<readonly SpeechSynthesisVoice[]>([]);

  useEffect(() => {
    if (!("speechSynthesis" in window)) return;

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
