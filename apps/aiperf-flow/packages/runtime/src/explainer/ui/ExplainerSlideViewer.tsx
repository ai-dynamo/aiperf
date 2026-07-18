// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from 'react';
import {
  DEFAULT_MARKER_TIP,
  isMarkerEndNone,
  markerDomId,
  markerGeometry,
  resolveMarkerTip,
  tipInsetUserUnits,
  type ResolvedMarkerTip,
} from '../arrow-tips.js';

interface ExplainerSlideViewerProps {
  deck: any;
  slideIndex?: number;
  onSlideChange?: (newIndex: number) => void;
}

interface NodeGeometry {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface NodeStyle {
  fill?: string;
  stroke?: string;
  strokeWidth?: number;
  fontSize?: number;
  fontWeight?: string;
  markerEnd?: unknown;
  dashed?: unknown;
  strokeStyle?: unknown;
  strokeDasharray?: unknown;
}

const INK_FILLS = new Set(['#1f2937', '#4b5563']);

// Compiled decks originally targeted a white canvas, so most box fills are
// saturated "brand" tones (blue/purple/green/etc). We keep those as accent
// left-borders on a dark card instead of flat full-bleed fills, which reads
// much closer to the legacy explainers app's categorized box style.
function accentFor(fill?: string): string {
  if (!fill) return '#777d80';
  return fill;
}

function getNodeGeometry(node: any): NodeGeometry {
  const layout = node.layout || {};
  return {
    x: layout.x ?? 0,
    y: layout.y ?? 0,
    width: layout.width ?? 0,
    height: layout.height ?? 0,
  };
}

function getNodeStyle(node: any): NodeStyle {
  const style = node.style || {};
  return {
    fill: style.fill,
    stroke: style.stroke,
    strokeWidth: style.strokeWidth,
    fontSize: style.fontSize,
    fontWeight: style.fontWeight,
    markerEnd: style.markerEnd,
    dashed: style.dashed,
    strokeStyle: style.strokeStyle,
    strokeDasharray: style.strokeDasharray ?? style.dashArray,
  };
}

function textColorFor(depth: number): string {
  return depth === 0 ? '#aeb4b5' : '#f8fafc';
}

function lineIsDashed(style: NodeStyle): boolean {
  if (style.dashed === true || style.dashed === 1 || style.dashed === 'true') {
    return true;
  }
  if (style.strokeStyle === 'dashed' || style.strokeStyle === 'dotted') {
    return true;
  }
  return typeof style.strokeDasharray === 'string' && style.strokeDasharray.length > 0;
}

function tipForLine(style: NodeStyle): ResolvedMarkerTip | null {
  if (isMarkerEndNone(style.markerEnd as any)) {
    return null;
  }
  return resolveMarkerTip(style.markerEnd as any, DEFAULT_MARKER_TIP);
}

function collectTips(roots: any[]): ResolvedMarkerTip[] {
  const byKey = new Map<string, ResolvedMarkerTip>();
  const visit = (node: any) => {
    const capability = node.capability || '';
    if (capability === 'core.line' || capability === 'core.path' || capability === 'core.arrow' || capability === 'core.connector') {
      const tip = tipForLine(getNodeStyle(node));
      if (tip !== null) {
        byKey.set(tip.key, tip);
      }
    }
    if (Array.isArray(node.children)) {
      for (const child of node.children) visit(child);
    }
  };
  for (const root of roots) visit(root);
  if (byKey.size === 0) {
    byKey.set(DEFAULT_MARKER_TIP.key, DEFAULT_MARKER_TIP);
  }
  return [...byKey.values()];
}

let markerSeq = 0;

function renderNode(
  node: any,
  depth: number,
  keyPrefix: string,
  markerPrefix: string,
): React.ReactNode[] {
  const geom = getNodeGeometry(node);
  const style = getNodeStyle(node);
  const capability = node.capability || '';
  const key = `${keyPrefix}-${node.id ?? capability}`;
  const out: React.ReactNode[] = [];

  if (capability === 'core.rect') {
    const isContainer = depth === 0;
    if (isContainer) {
      out.push(
        <rect
          key={key}
          x={geom.x}
          y={geom.y}
          width={geom.width}
          height={geom.height}
          rx={10}
          fill="none"
          stroke="#596266"
          strokeWidth={style.strokeWidth ?? 1.5}
        />,
      );
    } else {
      const accent = accentFor(style.fill);
      out.push(
        <g key={key}>
          <rect
            x={geom.x}
            y={geom.y}
            width={geom.width}
            height={geom.height}
            rx={10}
            fill="#2f3335"
            stroke="rgba(255,255,255,0.14)"
            strokeWidth={1}
            filter="url(#explainer-card-shadow)"
          />
          <rect
            x={geom.x}
            y={geom.y}
            width={4}
            height={geom.height}
            rx={2}
            fill={accent}
          />
        </g>,
      );
    }
  } else if (capability === 'core.text') {
    const size = style.fontSize ?? 14;
    const fill =
      style.fill && !INK_FILLS.has(style.fill.toLowerCase())
        ? style.fill
        : textColorFor(depth);
    out.push(
      <text
        key={key}
        x={geom.x + 12}
        y={geom.y + 10 + size * 0.8}
        fontSize={size}
        fontWeight={(style.fontWeight as any) || 500}
        fontFamily='Inter, "IBM Plex Sans", "Segoe UI", sans-serif'
        fill={fill}
      >
        {node.text || ''}
      </text>,
    );
  } else if (capability === 'core.line') {
    const from = node.from || {};
    const to = node.to || {};
    const x1 = Number(from.x ?? 0);
    const y1 = Number(from.y ?? 0);
    let x2 = Number(to.x ?? 0);
    let y2 = Number(to.y ?? 0);
    const stroke =
      style.stroke && !INK_FILLS.has((style.stroke || '').toLowerCase())
        ? style.stroke
        : '#8b9296';
    const strokeWidth = Number(style.strokeWidth ?? 1.5);
    const tip = tipForLine(style);
    if (tip !== null) {
      const tipInset = tipInsetUserUnits(
        tip,
        Number.isFinite(strokeWidth) ? strokeWidth : 1.5,
      );
      const dx = x2 - x1;
      const dy = y2 - y1;
      const length = Math.hypot(dx, dy);
      if (length > tipInset + 0.5) {
        const scale = (length - tipInset) / length;
        x2 = x1 + dx * scale;
        y2 = y1 + dy * scale;
      }
    }
    const dashed = lineIsDashed(style);
    out.push(
      <line
        key={key}
        x1={x1}
        y1={y1}
        x2={x2}
        y2={y2}
        stroke={stroke}
        strokeWidth={strokeWidth}
        strokeLinecap="butt"
        strokeDasharray={dashed ? '8 4' : undefined}
        markerEnd={
          tip !== null ? `url(#${markerDomId(markerPrefix, tip)})` : undefined
        }
        data-flow-tip={tip?.key}
      />,
    );
  }

  if (Array.isArray(node.children)) {
    node.children.forEach((child: any, i: number) => {
      out.push(...renderNode(child, depth + 1, `${key}-${i}`, markerPrefix));
    });
  }

  return out;
}

/**
 * Native Flow IR renderer for explainer slides.
 * Renders compiled .flow Scene IR objects as SVG.
 * Scene IR is compiled from .flow source through native Flow compiler pipeline.
 */
export function ExplainerSlideViewer({
  deck,
  slideIndex = 0,
}: ExplainerSlideViewerProps): React.ReactNode {
  const slide = deck.slides[slideIndex];

  if (!slide) {
    return <div>No slide at index {slideIndex}</div>;
  }

  const sceneIr = slide.sceneBlock;
  if (!sceneIr) {
    return <div>No scene visualization available for slide {slideIndex}</div>;
  }

  const roots = sceneIr.roots || [];
  let maxX = 500;
  let maxY = 500;
  roots.forEach((root: any) => {
    const geom = getNodeGeometry(root);
    maxX = Math.max(maxX, geom.x + geom.width);
    maxY = Math.max(maxY, geom.y + geom.height);
  });

  const margin = 40;
  const width = maxX + margin;
  const height = maxY + margin;
  const markerPrefix = React.useMemo(
    () => `explainer-arrow-${markerSeq++}`,
    [deck?.id, slideIndex],
  );
  const tips = collectTips(roots);

  return (
    <div
      style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'auto',
        backgroundColor: 'var(--flow-board, #232526)',
      }}
    >
      <div
        style={{
          background: 'var(--flow-panel, #292c2d)',
          borderRadius: '12px',
          border: '1px solid var(--flow-guide, #777d80)',
          boxShadow: '0 8px 24px rgba(0, 0, 0, 0.35)',
          margin: '20px auto',
          overflow: 'hidden',
          maxWidth: '100%',
        }}
      >
        {slide.eyebrow ? (
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '10px 18px',
              background: 'var(--flow-raised, #303334)',
              borderBottom: '1px solid var(--flow-guide, #777d80)',
              fontFamily: 'Inter, "IBM Plex Sans", "Segoe UI", sans-serif',
            }}
          >
            <span
              style={{
                fontSize: '0.75rem',
                fontWeight: 700,
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                color: 'var(--flow-signal, #71d8d0)',
              }}
            >
              {slide.eyebrow}
            </span>
            {slide.title && slide.title !== slide.eyebrow ? (
              <span
                style={{
                  fontSize: '0.8rem',
                  color: 'var(--flow-chalk-muted, #aeb4b5)',
                }}
              >
                {slide.title}
              </span>
            ) : null}
          </div>
        ) : null}

        <svg
          width={width}
          height={height}
          viewBox={`0 0 ${width} ${height}`}
          style={{ display: 'block', maxWidth: '100%', height: 'auto' }}
        >
          <defs>
            <filter id="explainer-card-shadow" x="-20%" y="-20%" width="140%" height="140%">
              <feDropShadow dx="0" dy="3" stdDeviation="4" floodColor="#000000" floodOpacity="0.35" />
            </filter>
            {tips.map((tip) => {
              const geom = markerGeometry(tip);
              return (
                <marker
                  key={tip.key}
                  id={markerDomId(markerPrefix, tip)}
                  markerWidth={geom.markerWidth}
                  markerHeight={geom.markerHeight}
                  refX={geom.refX}
                  refY={geom.refY}
                  orient="auto"
                  markerUnits="strokeWidth"
                >
                  {geom.children}
                </marker>
              );
            })}
          </defs>
          <rect x={0} y={0} width={width} height={height} fill="var(--flow-panel, #292c2d)" />
          {roots.map((root: any, i: number) =>
            renderNode(root, 0, `root-${i}`, markerPrefix),
          )}
        </svg>

        {slide.caption ? (
          <div
            style={{
              padding: '12px 18px',
              borderTop: '1px solid var(--flow-guide, #777d80)',
              fontFamily: 'Inter, "IBM Plex Sans", "Segoe UI", sans-serif',
              fontSize: '0.85rem',
              color: 'var(--flow-chalk-muted, #aeb4b5)',
            }}
          >
            {slide.caption}
          </div>
        ) : null}
      </div>
    </div>
  );
}
