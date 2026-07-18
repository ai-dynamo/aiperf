export const NARRATION_RATE = 1.08;
const WORDS_PER_MINUTE = 150 * NARRATION_RATE;
const POST_NARRATION_PAUSE_MS = 600;

export function narrationSupported(): boolean {
  return (
    typeof window !== "undefined" &&
    "speechSynthesis" in window &&
    typeof SpeechSynthesisUtterance !== "undefined"
  );
}

export function splitWords(text: string): readonly string[] {
  return text.trim().split(/\s+/).filter(Boolean);
}

export function estimateNarrationMs(text: string): number {
  const words = splitWords(text).length;
  return Math.max(2500, Math.round((words / WORDS_PER_MINUTE) * 60_000) + POST_NARRATION_PAUSE_MS);
}

export function stopNarration(): void {
  if (typeof window !== "undefined" && "speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
}

/**
 * Must run inside a user-gesture handler. Browsers refuse to speak until then;
 * after a successful unlock, later slides can speak without another click.
 */
export function unlockSpeech(): boolean {
  if (!narrationSupported()) return false;
  try {
    window.speechSynthesis.getVoices();
    const prime = new SpeechSynthesisUtterance(" ");
    prime.volume = 0;
    prime.rate = NARRATION_RATE;
    window.speechSynthesis.speak(prime);
    window.speechSynthesis.cancel();
    return true;
  } catch {
    return false;
  }
}

function wordIndexFromChar(text: string, charIndex: number): number {
  if (charIndex <= 0) return 0;
  const prefix = text.slice(0, Math.min(charIndex, text.length));
  const count = splitWords(prefix).length;
  return Math.max(0, count - 1);
}

export function speakNarration(
  text: string,
  options: {
    useSpeech: boolean;
    voiceURI?: string;
    onWord?: (wordIndex: number) => void;
    onComplete: () => void;
  },
): () => void {
  stopNarration();
  const words = splitWords(text);
  const fallbackMs = estimateNarrationMs(text);
  const timers: number[] = [];

  const clearTimers = () => {
    for (const timer of timers) window.clearTimeout(timer);
    timers.length = 0;
  };

  const driveEstimatedWords = () => {
    if (words.length === 0) return;
    const speakMs = Math.max(1200, fallbackMs - POST_NARRATION_PAUSE_MS);
    const stepMs = speakMs / words.length;
    words.forEach((_, index) => {
      timers.push(
        window.setTimeout(() => {
          options.onWord?.(index);
        }, Math.round(index * stepMs)),
      );
    });
  };

  if (!options.useSpeech || !narrationSupported()) {
    options.onWord?.(0);
    driveEstimatedWords();
    timers.push(window.setTimeout(options.onComplete, fallbackMs));
    return () => clearTimers();
  }

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = NARRATION_RATE;
  utterance.pitch = 1;
  if (options.voiceURI) {
    const voice = window.speechSynthesis
      .getVoices()
      .find((candidate) => candidate.voiceURI === options.voiceURI);
    if (voice) utterance.voice = voice;
  }

  let done = false;
  const finish = () => {
    if (done) return;
    done = true;
    options.onWord?.(Math.max(0, words.length - 1));
    timers.push(window.setTimeout(options.onComplete, POST_NARRATION_PAUSE_MS));
  };

  options.onWord?.(0);
  utterance.onboundary = (event) => {
    if (event.name && event.name !== "word") return;
    options.onWord?.(wordIndexFromChar(text, event.charIndex));
  };
  utterance.onend = finish;
  utterance.onerror = (event) => {
    // Intentional stopNarration()/cancel must not schedule fallback advancement.
    if (done || event.error === "interrupted" || event.error === "canceled") {
      return;
    }
    // Some engines lack reliable boundary events; keep timing alive with estimates.
    driveEstimatedWords();
    timers.push(window.setTimeout(finish, fallbackMs));
  };

  window.speechSynthesis.speak(utterance);
  timers.push(window.setTimeout(finish, fallbackMs + 4000));

  return () => {
    done = true;
    clearTimers();
    stopNarration();
  };
}
