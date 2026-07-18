// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mental model diagram primitives for explainer scenes.
//! Composable Flow symbols for architecture visualization, data flow, timelines, and theme display.

import type { CSSProperties, ReactNode } from "react";
import { useMemo } from "react";
import type { ResolvedTheme } from "../theme/types.js";

/** Props shared by all mental model primitives. */
interface MentalModelProps {
  /** Theme used for all colors and styling. */
  theme: ResolvedTheme;
  /** Optional CSS class name. */
  className?: string;
  /** Optional inline styles. */
  style?: CSSProperties;
}

/**
 * Resolves a theme role to its actual color value.
 * Handles both ThemeValueIr discriminant and string color values.
 */
function resolveThemeColor(
  theme: ResolvedTheme,
  role: string,
  fallback: string = "#f2eee3"
): string {
  const value = theme.values[role as keyof typeof theme.values];
  if (!value) {
    return fallback;
  }

  // Handle ThemeValueIr discriminant
  if (typeof value === "object" && "kind" in value && "value" in value) {
    const themeValue = value as { kind: string; value: string };
    if (themeValue.kind === "color" && typeof themeValue.value === "string") {
      return themeValue.value;
    }
  }

  // Fallback if value is already a string
  if (typeof value === "string" && value.startsWith("#")) {
    return value;
  }

  return fallback;
}

// ============================================================================
// ArchitectureBox: A composable box for displaying architecture elements
// ============================================================================

interface ArchitectureBoxProps extends MentalModelProps {
  /** Box title/label. */
  label: string;
  /** Box description or content. */
  description?: string;
  /** Optional box icon or identifier. */
  icon?: ReactNode;
  /** Width in pixels or CSS value. */
  width?: string | number;
  /** Height in pixels or CSS value. */
  height?: string | number;
  /** Box type for styling (e.g., "service", "process", "boundary"). */
  boxType?: "service" | "process" | "boundary" | "default";
  /** Child content. */
  children?: ReactNode;
}

export function ArchitectureBox({
  label,
  description,
  icon,
  width = 160,
  height = 80,
  boxType = "default",
  theme,
  className,
  style,
  children,
}: ArchitectureBoxProps): JSX.Element {
  const backgroundColor = useMemo(() => {
    switch (boxType) {
      case "service":
        return resolveThemeColor(theme, "accent.execute", "#7dce82");
      case "process":
        return resolveThemeColor(theme, "surface.primary", "#303334");
      case "boundary":
        return resolveThemeColor(theme, "accent.alert", "#f07972");
      default:
        return resolveThemeColor(theme, "surface.secondary", "#383c3e");
    }
  }, [boxType, theme]);

  const borderColor = resolveThemeColor(theme, "structure.divider", "#d7dada");
  const textColor = resolveThemeColor(theme, "ink.primary", "#f1f3f2");

  const boxStyle: CSSProperties = {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    width,
    height,
    backgroundColor,
    border: `2px solid ${borderColor}`,
    borderRadius: "8px",
    padding: "12px",
    color: textColor,
    fontFamily: "var(--flow-font-body, Inter, sans-serif)",
    fontSize: "14px",
    fontWeight: 500,
    gap: "8px",
    ...style,
  };

  return (
    <div className={className} style={boxStyle}>
      {icon && <div style={{ fontSize: "24px" }}>{icon}</div>}
      <div style={{ textAlign: "center", fontWeight: 600 }}>{label}</div>
      {description && (
        <div
          style={{
            fontSize: "12px",
            opacity: 0.8,
            textAlign: "center",
            marginTop: "4px",
          }}
        >
          {description}
        </div>
      )}
      {children}
    </div>
  );
}

// ============================================================================
// DataFlowArrow: A composable arrow showing data flow between components
// ============================================================================

interface DataFlowArrowProps extends MentalModelProps {
  /** Arrow direction. */
  direction?: "up" | "down" | "left" | "right" | "diagonal-down-right";
  /** Arrow label (e.g., "Request", "Response"). */
  label?: string;
  /** Arrow style variant. */
  variant?: "solid" | "dashed" | "dotted";
  /** Width/length of arrow in pixels. */
  length?: number;
}

export function DataFlowArrow({
  direction = "right",
  label,
  variant = "solid",
  length = 60,
  theme,
  className,
  style,
}: DataFlowArrowProps): JSX.Element {
  const arrowColor = resolveThemeColor(theme, "structure.divider", "#7aa2f7");

  const getStrokeDasharray = () => {
    switch (variant) {
      case "dashed":
        return "8 4";
      case "dotted":
        return "2 3";
      default:
        return "none";
    }
  };

  const getRotation = () => {
    switch (direction) {
      case "down":
        return "90deg";
      case "left":
        return "180deg";
      case "up":
        return "270deg";
      case "diagonal-down-right":
        return "45deg";
      default:
        return "0deg";
    }
  };

  const arrowSize = 12;
  const svgWidth = length;
  const svgHeight = arrowSize + 8;

  return (
    <div
      className={className}
      style={{
        display: "inline-flex",
        flexDirection: "column",
        alignItems: "center",
        gap: "4px",
        transform: `rotate(${getRotation()})`,
        ...style,
      }}
    >
      <svg
        width={svgWidth}
        height={svgHeight}
        viewBox={`0 0 ${svgWidth} ${svgHeight}`}
        preserveAspectRatio="none"
      >
        <line
          x1="0"
          y1={svgHeight / 2}
          x2={length - arrowSize}
          y2={svgHeight / 2}
          stroke={arrowColor}
          strokeWidth="2"
          strokeDasharray={getStrokeDasharray()}
        />
        <polygon
          points={`${length - arrowSize},${svgHeight / 2 - 4} ${length},${svgHeight / 2} ${length - arrowSize},${svgHeight / 2 + 4}`}
          fill={arrowColor}
        />
      </svg>
      {label && (
        <span
          style={{
            fontSize: "12px",
            color: resolveThemeColor(theme, "ink.primary", "#f1f3f2"),
            fontWeight: 500,
            transform: `rotate(-${getRotation()})`,
            whiteSpace: "nowrap",
            background: resolveThemeColor(theme, "surface.primary", "#303334"),
            padding: "2px 6px",
            borderRadius: "4px",
          }}
        >
          {label}
        </span>
      )}
    </div>
  );
}

// ============================================================================
// GridLayout: A grid-based layout system for composing primitives
// ============================================================================

interface GridLayoutProps extends MentalModelProps {
  /** Number of columns. */
  columns?: number;
  /** Gap between grid items in pixels. */
  gap?: number;
  /** Grid child elements. */
  children: ReactNode;
  /** Grid alignment ("start" | "center" | "end"). */
  align?: "start" | "center" | "end";
}

export function GridLayout({
  columns = 3,
  gap = 16,
  children,
  align = "center",
  theme,
  className,
  style,
}: GridLayoutProps): JSX.Element {
  const gridStyle: CSSProperties = {
    display: "grid",
    gridTemplateColumns: `repeat(${columns}, 1fr)`,
    gap,
    alignItems: align,
    justifyContent: align,
    padding: "16px",
    backgroundColor: resolveThemeColor(
      theme,
      "surface.primary",
      "#232526"
    ),
    borderRadius: "12px",
    border: `1px solid ${resolveThemeColor(theme, "structure.divider", "#d7dada")}`,
    ...style,
  };

  return (
    <div className={className} style={gridStyle}>
      {children}
    </div>
  );
}

// ============================================================================
// SequenceTimeline: A timeline component for showing sequences
// ============================================================================

interface TimelineStep {
  id: string;
  label: string;
  description?: string;
  completed?: boolean;
}

interface SequenceTimelineProps extends MentalModelProps {
  /** Timeline steps. */
  steps: TimelineStep[];
  /** Current active step index. */
  activeStep?: number;
  /** Timeline orientation ("horizontal" | "vertical"). */
  orientation?: "horizontal" | "vertical";
  /** Step size in pixels. */
  stepSize?: number;
}

export function SequenceTimeline({
  steps,
  activeStep = 0,
  orientation = "horizontal",
  stepSize = 60,
  theme,
  className,
  style,
}: SequenceTimelineProps): JSX.Element {
  const completedColor = resolveThemeColor(theme, "accent.execute", "#7dce82");
  const activeColor = resolveThemeColor(theme, "accent.attention", "#f0cf58");
  const inactiveColor = resolveThemeColor(theme, "structure.divider", "#777d80");
  const textColor = resolveThemeColor(theme, "ink.primary", "#f1f3f2");

  const isVertical = orientation === "vertical";

  const containerStyle: CSSProperties = {
    display: "flex",
    flexDirection: isVertical ? "column" : "row",
    alignItems: isVertical ? "flex-start" : "center",
    gap: "8px",
    padding: "16px",
    backgroundColor: resolveThemeColor(theme, "surface.secondary", "#292c2d"),
    borderRadius: "12px",
    ...style,
  };

  return (
    <div className={className} style={containerStyle}>
      {steps.map((step, index) => (
        <div
          key={step.id}
          style={{
            display: "flex",
            flexDirection: isVertical ? "row" : "column",
            alignItems: "center",
            gap: isVertical ? "12px" : "8px",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              width: stepSize,
              height: stepSize,
              borderRadius: "50%",
              backgroundColor:
                index < activeStep
                  ? completedColor
                  : index === activeStep
                    ? activeColor
                    : inactiveColor,
              color: "#000",
              fontWeight: 600,
              fontSize: "14px",
              flexShrink: 0,
            }}
          >
            {index < activeStep ? "✓" : index + 1}
          </div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ color: textColor, fontWeight: 600, fontSize: "14px" }}>
              {step.label}
            </div>
            {step.description && (
              <div
                style={{
                  color: resolveThemeColor(
                    theme,
                    "ink.muted",
                    "#aeb4b5"
                  ),
                  fontSize: "12px",
                  marginTop: "4px",
                }}
              >
                {step.description}
              </div>
            )}
          </div>
          {index < steps.length - 1 && (
            <div
              style={{
                width: isVertical ? "2px" : "24px",
                height: isVertical ? "20px" : "2px",
                backgroundColor: inactiveColor,
                margin: isVertical ? "0 0 0 28px" : "0",
              }}
            />
          )}
        </div>
      ))}
    </div>
  );
}

// ============================================================================
// ThemePalette: A component to display theme colors
// ============================================================================

interface PaletteColor {
  role: string;
  label: string;
}

interface ThemePaletteProps extends MentalModelProps {
  /** Colors to display. */
  colors: PaletteColor[];
  /** Palette size ("small" | "medium" | "large"). */
  size?: "small" | "medium" | "large";
  /** Display format ("grid" | "row"). */
  format?: "grid" | "row";
}

export function ThemePalette({
  colors,
  size = "medium",
  format = "grid",
  theme,
  className,
  style,
}: ThemePaletteProps): JSX.Element {
  const sizeMap = {
    small: 40,
    medium: 60,
    large: 80,
  };
  const colorSize = sizeMap[size];

  const containerStyle: CSSProperties = {
    display: format === "grid" ? "grid" : "flex",
    gridTemplateColumns: format === "grid" ? "repeat(3, 1fr)" : undefined,
    flexDirection: format === "row" ? "row" : undefined,
    gap: "16px",
    padding: "16px",
    backgroundColor: resolveThemeColor(theme, "surface.secondary", "#292c2d"),
    borderRadius: "12px",
    flexWrap: "wrap",
    ...style,
  };

  return (
    <div className={className} style={containerStyle}>
      {colors.map((color) => (
        <div
          key={color.role}
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: "8px",
          }}
        >
          <div
            style={{
              width: colorSize,
              height: colorSize,
              backgroundColor: resolveThemeColor(theme, color.role, "#999"),
              borderRadius: "8px",
              border: `1px solid ${resolveThemeColor(theme, "structure.divider", "#d7dada")}`,
              cursor: "pointer",
              transition: "transform 0.2s",
              flexShrink: 0,
            }}
            title={`Theme role: ${color.role}`}
          />
          <div
            style={{
              fontSize: "12px",
              color: resolveThemeColor(theme, "ink.primary", "#f1f3f2"),
              fontWeight: 500,
              textAlign: "center",
              maxWidth: `${colorSize + 20}px`,
              wordBreak: "break-word",
            }}
          >
            {color.label}
          </div>
          <div
            style={{
              fontSize: "11px",
              color: resolveThemeColor(theme, "ink.muted", "#aeb4b5"),
              fontFamily: "var(--flow-font-data, monospace)",
            }}
          >
            {resolveThemeColor(theme, color.role, "#999")}
          </div>
        </div>
      ))}
    </div>
  );
}

// ============================================================================
// ContrastMatrix: A component showing contrast relationships
// ============================================================================

interface ContrastPair {
  foreground: string;
  background: string;
  label: string;
  ratio?: number;
}

interface ContrastMatrixProps extends MentalModelProps {
  /** Contrast pairs to display. */
  pairs: ContrastPair[];
  /** Minimum acceptable contrast ratio (WCAG standard). */
  minRatio?: number;
}

export function ContrastMatrix({
  pairs,
  minRatio = 4.5,
  theme,
  className,
  style,
}: ContrastMatrixProps): JSX.Element {
  const borderColor = resolveThemeColor(theme, "structure.divider", "#d7dada");
  const passColor = resolveThemeColor(theme, "accent.execute", "#7dce82");
  const failColor = resolveThemeColor(theme, "accent.alert", "#f07972");

  const containerStyle: CSSProperties = {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
    gap: "16px",
    padding: "16px",
    backgroundColor: resolveThemeColor(theme, "surface.secondary", "#292c2d"),
    borderRadius: "12px",
    border: `1px solid ${borderColor}`,
    ...style,
  };

  return (
    <div className={className} style={containerStyle}>
      {pairs.map((pair, index) => {
        const passes = !pair.ratio || pair.ratio >= minRatio;
        const statusColor = passes ? passColor : failColor;

        return (
          <div
            key={`contrast-${index}`}
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "12px",
              padding: "12px",
              backgroundColor: resolveThemeColor(theme, "surface.primary", "#303334"),
              borderRadius: "8px",
              border: `2px solid ${statusColor}`,
            }}
          >
            <div style={{ fontSize: "14px", fontWeight: 600, color: resolveThemeColor(theme, "ink.primary", "#f1f3f2") }}>
              {pair.label}
            </div>
            <div style={{ display: "flex", gap: "8px", height: "40px" }}>
              <div
                style={{
                  flex: 1,
                  backgroundColor: pair.foreground,
                  borderRadius: "4px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: "12px",
                  color: pair.background,
                  fontWeight: 600,
                }}
              >
                Fg
              </div>
              <div
                style={{
                  flex: 1,
                  backgroundColor: pair.background,
                  borderRadius: "4px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: "12px",
                  color: pair.foreground,
                  fontWeight: 600,
                }}
              >
                Bg
              </div>
            </div>
            {pair.ratio !== undefined && (
              <div
                style={{
                  fontSize: "12px",
                  color: statusColor,
                  fontWeight: 600,
                  textAlign: "center",
                }}
              >
                Ratio: {pair.ratio.toFixed(2)}:1 {passes ? "✓" : "✗"}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ============================================================================
// LegendBlock: A component for displaying legends
// ============================================================================

interface LegendEntry {
  id: string;
  label: string;
  color?: string;
  icon?: ReactNode;
  description?: string;
}

interface LegendBlockProps extends MentalModelProps {
  /** Legend entries. */
  entries: LegendEntry[];
  /** Legend title. */
  title?: string;
  /** Legend orientation ("vertical" | "horizontal"). */
  orientation?: "vertical" | "horizontal";
}

export function LegendBlock({
  entries,
  title,
  orientation = "vertical",
  theme,
  className,
  style,
}: LegendBlockProps): JSX.Element {
  const containerStyle: CSSProperties = {
    display: "flex",
    flexDirection: orientation === "vertical" ? "column" : "row",
    gap: orientation === "vertical" ? "12px" : "24px",
    padding: "16px",
    backgroundColor: resolveThemeColor(theme, "surface.secondary", "#292c2d"),
    borderRadius: "12px",
    border: `1px solid ${resolveThemeColor(theme, "structure.divider", "#d7dada")}`,
    flexWrap: "wrap",
    ...style,
  };

  return (
    <div className={className} style={containerStyle}>
      {title && (
        <div
          style={{
            fontSize: "16px",
            fontWeight: 600,
            color: resolveThemeColor(theme, "ink.primary", "#f1f3f2"),
            marginBottom: "8px",
            width: "100%",
          }}
        >
          {title}
        </div>
      )}
      <div
        style={{
          display: "flex",
          flexDirection: orientation === "vertical" ? "column" : "row",
          gap: orientation === "vertical" ? "12px" : "24px",
          flexWrap: orientation === "horizontal" ? "wrap" : "nowrap",
          width: "100%",
        }}
      >
        {entries.map((entry) => (
          <div
            key={entry.id}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
            }}
          >
            {entry.color ? (
              <div
                style={{
                  width: "16px",
                  height: "16px",
                  backgroundColor: entry.color,
                  borderRadius: "4px",
                  flexShrink: 0,
                  border: `1px solid ${resolveThemeColor(theme, "structure.divider", "#d7dada")}`,
                }}
              />
            ) : entry.icon ? (
              <div style={{ fontSize: "16px", flexShrink: 0 }}>{entry.icon}</div>
            ) : null}
            <div style={{ display: "flex", flexDirection: "column", gap: "2px" }}>
              <div
                style={{
                  fontSize: "14px",
                  fontWeight: 500,
                  color: resolveThemeColor(theme, "ink.primary", "#f1f3f2"),
                }}
              >
                {entry.label}
              </div>
              {entry.description && (
                <div
                  style={{
                    fontSize: "12px",
                    color: resolveThemeColor(theme, "ink.muted", "#aeb4b5"),
                  }}
                >
                  {entry.description}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
