/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Shared UI primitives for AIPerf explainer sites.
 * Chrome styles live in `index.css` (CSS vars); SVG/diagram code uses `theme`.
 */
import {
  createContext,
  useCallback,
  useContext,
  useState,
  type CSSProperties,
  type ReactNode,
} from "react";
import { tokens } from "./tokens";

export const theme = {
  kind: "dark",
  text: { ...tokens.text },
  bg: {
    editor: tokens.bg.page,
    chrome: tokens.bg.chrome,
    elevated: tokens.bg.elevated,
  },
  fill: { ...tokens.fill },
  stroke: { ...tokens.stroke },
  accent: { ...tokens.accent },
  category: { ...tokens.category },
} as const;

export type Theme = typeof theme;

const ThemeContext = createContext<Theme>(theme);

export function useHostTheme(): Theme {
  return useContext(ThemeContext);
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  return <ThemeContext.Provider value={theme}>{children}</ThemeContext.Provider>;
}

export function useCanvasState<T>(
  storagePrefix: string,
  key: string,
  defaultValue: T,
): [T, (action: T | ((prev: T) => T)) => void] {
  const storageKey = `${storagePrefix}:${key}`;
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
          /* ignore */
        }
        return next;
      });
    },
    [storageKey],
  );

  return [value, setPersisted];
}

export function Stack({
  gap = 0,
  children,
  style,
}: {
  gap?: number;
  children?: ReactNode;
  style?: CSSProperties;
}) {
  return <div style={{ display: "flex", flexDirection: "column", gap, ...style }}>{children}</div>;
}

export function Row({
  gap = 0,
  align = "stretch",
  wrap = false,
  children,
  style,
}: {
  gap?: number;
  align?: "start" | "center" | "end" | "stretch";
  wrap?: boolean;
  children?: ReactNode;
  style?: CSSProperties;
}) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "row",
        gap,
        alignItems: align === "start" ? "flex-start" : align === "end" ? "flex-end" : align,
        flexWrap: wrap ? "wrap" : "nowrap",
        ...style,
      }}
    >
      {children}
    </div>
  );
}

export function Spacer() {
  return <div style={{ flex: 1 }} />;
}

export function Divider() {
  return <div className="ex-divider" />;
}

export function Text({
  tone = "primary",
  weight,
  children,
}: {
  tone?: "primary" | "secondary" | "tertiary";
  weight?: "medium" | "bold";
  children?: ReactNode;
}) {
  const t = useHostTheme();
  const color =
    tone === "secondary" ? t.text.secondary : tone === "tertiary" ? t.text.tertiary : t.text.primary;
  return (
    <span style={{ color, fontWeight: weight === "bold" ? 700 : weight === "medium" ? 600 : 400 }}>
      {children}
    </span>
  );
}

export function Button({
  children,
  onClick,
  disabled,
  variant = "primary",
  style,
  className,
}: {
  children?: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: "primary" | "secondary";
  style?: CSSProperties;
  className?: string;
}) {
  const variantClass = variant === "secondary" ? "ex-btn--secondary" : "ex-btn--primary";
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className={["ex-btn", variantClass, className].filter(Boolean).join(" ")}
      style={style}
    >
      {children}
    </button>
  );
}

export function Pill({
  children,
  active,
  onClick,
  title,
  size = "md",
}: {
  children?: ReactNode;
  active?: boolean;
  onClick?: () => void;
  title?: string;
  size?: "sm" | "md";
}) {
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      className={[
        "ex-pill",
        size === "sm" ? "ex-pill--sm" : "ex-pill--md",
        active ? "ex-pill--active" : "",
      ]
        .filter(Boolean)
        .join(" ")}
      style={{ cursor: onClick ? "pointer" : "default" }}
    >
      {children}
    </button>
  );
}

export function Code({ children, style }: { children?: ReactNode; style?: CSSProperties }) {
  return (
    <code className="ex-code" style={style}>
      {children}
    </code>
  );
}

export function Card({ children }: { children?: ReactNode }) {
  return <div className="ex-panel">{children}</div>;
}

export function CardHeader({ children, trailing }: { children?: ReactNode; trailing?: ReactNode }) {
  return (
    <div className="ex-panel__header">
      <span>{children}</span>
      {trailing}
    </div>
  );
}

export function CardBody({ children }: { children?: ReactNode }) {
  return <div className="ex-panel__body">{children}</div>;
}

export function VoicePicker({
  voices,
  selectedVoiceURI,
  onVoiceSelect,
  speechAvailable,
}: {
  voices: readonly SpeechSynthesisVoice[];
  selectedVoiceURI: string;
  onVoiceSelect: (voiceURI: string) => void;
  speechAvailable: boolean;
}) {
  if (!speechAvailable) return null;

  return (
    <div>
      <div className="ex-eyebrow" style={{ marginBottom: 9 }}>
        Voice
      </div>
      <div role="radiogroup" aria-label="Narration voice" style={{ display: "flex", flexWrap: "wrap", gap: 7 }}>
        <Pill active={selectedVoiceURI === ""} onClick={() => onVoiceSelect("")}>
          Default
        </Pill>
        {voices.map((voice) => (
          <Pill
            key={voice.voiceURI}
            active={selectedVoiceURI === voice.voiceURI}
            onClick={() => onVoiceSelect(voice.voiceURI)}
            title={`${voice.name} · ${voice.lang}`}
          >
            {voice.name}
          </Pill>
        ))}
      </div>
    </div>
  );
}

export function Subtitles({
  text,
  activeWordIndex,
  visible,
}: {
  text: string;
  activeWordIndex: number;
  visible: boolean;
}) {
  if (!visible) return null;
  const words = text.trim().split(/\s+/).filter(Boolean);
  if (words.length === 0) return null;

  return (
    <div aria-live="polite" className="ex-subtitles">
      <div className="ex-subtitles__label">SUBTITLES</div>
      <div style={{ fontSize: 18, lineHeight: 1.55, fontWeight: 500 }}>
        {words.map((word, index) => {
          const active = index === activeWordIndex;
          const spoken = activeWordIndex >= 0 && index < activeWordIndex;
          const stateClass = active
            ? "ex-subtitles__word--active"
            : spoken
              ? "ex-subtitles__word--spoken"
              : "ex-subtitles__word--pending";
          return (
            <span key={`${index}-${word}`} className={`ex-subtitles__word ${stateClass}`}>
              {word}
            </span>
          );
        })}
      </div>
    </div>
  );
}

export function StartGate({
  title,
  onStartWithNarration,
  onStartSilent,
  speechAvailable,
  voices,
  selectedVoiceURI,
  onVoiceSelect,
}: {
  title: string;
  onStartWithNarration: () => void;
  onStartSilent: () => void;
  speechAvailable: boolean;
  voices: readonly SpeechSynthesisVoice[];
  selectedVoiceURI: string;
  onVoiceSelect: (voiceURI: string) => void;
}) {
  return (
    <div className="ex-gate">
      <div className="ex-gate__card">
        <div className="ex-eyebrow ex-eyebrow--accent" style={{ marginBottom: 8 }}>
          AIPerf Explainers
        </div>
        <div style={{ color: "var(--ex-text-primary)", fontSize: 22, fontWeight: 700, lineHeight: 1.25, marginBottom: 10 }}>
          {title}
        </div>
        <div style={{ color: "var(--ex-text-secondary)", fontSize: 15, lineHeight: 1.55, marginBottom: 22 }}>
          Browsers block spoken audio until you click once. After that, slides advance and narrate automatically.
        </div>
        {speechAvailable ? (
          <div style={{ marginBottom: 20 }}>
            <VoicePicker
              voices={voices}
              selectedVoiceURI={selectedVoiceURI}
              onVoiceSelect={onVoiceSelect}
              speechAvailable={speechAvailable}
            />
          </div>
        ) : null}
        <div style={{ display: "flex", flexDirection: "row", flexWrap: "wrap", gap: 10 }}>
          <Button style={{ flex: "1 1 180px" }} onClick={onStartWithNarration} disabled={!speechAvailable}>
            {speechAvailable ? "Play with audio" : "Audio unavailable"}
          </Button>
          <Button variant="secondary" style={{ flex: "1 1 180px" }} onClick={onStartSilent}>
            Play without audio
          </Button>
        </div>
      </div>
    </div>
  );
}
