/**
 * Shared UI primitives for AIPerf explainer sites.
 */
import {
  createContext,
  useCallback,
  useContext,
  useState,
  type CSSProperties,
  type ReactNode,
} from "react";

export const theme = {
  kind: "dark",
  text: {
    primary: "#E4E4E4EB",
    secondary: "#E4E4E48D",
    tertiary: "#E4E4E45E",
    quaternary: "#E4E4E442",
    link: "#87c3ff",
    onAccent: "#191c22",
  },
  bg: {
    editor: "#181818",
    chrome: "#141414",
    elevated: "#1f1f1f",
  },
  fill: {
    primary: "#E4E4E430",
    secondary: "#E4E4E41E",
    tertiary: "#E4E4E411",
    quaternary: "#E4E4E40A",
  },
  stroke: {
    primary: "#E4E4E433",
    secondary: "#E4E4E41F",
    tertiary: "#E4E4E414",
  },
  accent: {
    primary: "#3FA266",
    control: "#3FA266",
  },
  category: {
    green: "#3FA266",
    yellow: "#F1B467",
    purple: "#9386F2",
    blue: "#599CE7",
    red: "#FC6B83",
    orange: "#F0A060",
    cyan: "#5BC0DE",
    gray: "#8A8A8A",
  },
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
  const t = useHostTheme();
  return <div style={{ height: 1, background: t.stroke.tertiary, width: "100%" }} />;
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
}: {
  children?: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: "primary" | "secondary";
  style?: CSSProperties;
}) {
  const t = useHostTheme();
  const base: CSSProperties =
    variant === "secondary"
      ? { background: t.fill.tertiary, color: t.text.primary, border: `1px solid ${t.stroke.secondary}` }
      : { background: t.accent.primary, color: t.text.onAccent, border: "1px solid transparent" };

  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      style={{
        ...base,
        opacity: disabled ? 0.45 : 1,
        cursor: disabled ? "not-allowed" : "pointer",
        borderRadius: 6,
        padding: "8px 14px",
        fontSize: 14,
        fontWeight: 600,
        fontFamily: "inherit",
        ...style,
      }}
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
  const t = useHostTheme();
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      style={{
        borderRadius: 999,
        border: `1px solid ${active ? t.category.green : t.stroke.secondary}`,
        background: active ? t.category.green : t.fill.quaternary,
        color: active ? t.text.onAccent : t.text.secondary,
        padding: size === "sm" ? "2px 8px" : "4px 10px",
        fontSize: size === "sm" ? 12 : 13,
        fontWeight: 650,
        cursor: onClick ? "pointer" : "default",
        fontFamily: "inherit",
      }}
    >
      {children}
    </button>
  );
}

export function Code({ children, style }: { children?: ReactNode; style?: CSSProperties }) {
  const t = useHostTheme();
  return (
    <code
      style={{
        display: "inline-block",
        fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
        fontSize: 13,
        background: t.fill.tertiary,
        border: `1px solid ${t.stroke.secondary}`,
        borderRadius: 6,
        padding: "8px 10px",
        color: t.text.primary,
        ...style,
      }}
    >
      {children}
    </code>
  );
}

export function Card({ children }: { children?: ReactNode }) {
  const t = useHostTheme();
  return (
    <div
      style={{
        border: `1px solid ${t.stroke.secondary}`,
        borderRadius: 8,
        background: t.bg.elevated,
        overflow: "hidden",
      }}
    >
      {children}
    </div>
  );
}

export function CardHeader({ children, trailing }: { children?: ReactNode; trailing?: ReactNode }) {
  const t = useHostTheme();
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 12,
        padding: "12px 16px",
        borderBottom: `1px solid ${t.stroke.tertiary}`,
        color: t.text.primary,
        fontWeight: 700,
        fontSize: 15,
      }}
    >
      <span>{children}</span>
      {trailing}
    </div>
  );
}

export function CardBody({ children }: { children?: ReactNode }) {
  return <div style={{ padding: 16 }}>{children}</div>;
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
  const t = useHostTheme();
  if (!speechAvailable) return null;

  return (
    <div>
      <div style={{ color: t.text.secondary, fontSize: 12, fontWeight: 700, letterSpacing: "0.05em", marginBottom: 9 }}>
        VOICE
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
  const t = useHostTheme();
  if (!visible) return null;
  const words = text.trim().split(/\s+/).filter(Boolean);
  if (words.length === 0) return null;

  return (
    <div
      aria-live="polite"
      style={{
        border: `1px solid ${t.stroke.secondary}`,
        borderRadius: 8,
        background: t.fill.quaternary,
        padding: "14px 16px",
        minHeight: 64,
      }}
    >
      <div
        style={{
          color: t.text.secondary,
          fontSize: 11,
          fontWeight: 700,
          letterSpacing: "0.05em",
          marginBottom: 8,
        }}
      >
        SUBTITLES
      </div>
      <div style={{ fontSize: 18, lineHeight: 1.55, fontWeight: 500 }}>
        {words.map((word, index) => {
          const active = index === activeWordIndex;
          const spoken = activeWordIndex >= 0 && index < activeWordIndex;
          return (
            <span
              key={`${index}-${word}`}
              style={{
                color: active ? t.text.onAccent : spoken ? t.text.primary : t.text.secondary,
                background: active ? t.category.green : "transparent",
                borderRadius: 4,
                padding: active ? "1px 4px" : "1px 0",
                marginRight: 5,
                transition: "background 80ms ease, color 80ms ease",
              }}
            >
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
  const t = useHostTheme();
  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 1000,
        display: "grid",
        placeItems: "center",
        padding: 24,
        background: "rgba(10, 10, 10, 0.92)",
      }}
    >
      <div
        style={{
          width: "min(480px, 100%)",
          border: `1px solid ${t.stroke.secondary}`,
          borderRadius: 12,
          background: t.bg.elevated,
          padding: 28,
        }}
      >
        <div style={{ color: t.category.green, fontSize: 13, fontWeight: 700, letterSpacing: "0.05em", marginBottom: 8 }}>
          AIPERF EXPLAINERS
        </div>
        <div style={{ color: t.text.primary, fontSize: 22, fontWeight: 700, lineHeight: 1.25, marginBottom: 10 }}>
          {title}
        </div>
        <div style={{ color: t.text.secondary, fontSize: 15, lineHeight: 1.55, marginBottom: 22 }}>
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
          <Button
            style={{ background: t.category.green, color: t.text.onAccent, flex: "1 1 180px" }}
            onClick={onStartWithNarration}
            disabled={!speechAvailable}
          >
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
