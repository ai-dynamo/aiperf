// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useEffect, useRef } from 'react';

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
}

/**
 * Native Flow IR renderer for explainer slides.
 * Renders compiled .flow Scene IR objects directly to canvas.
 * Scene IR is compiled from .flow source through native Flow compiler pipeline.
 */
export function ExplainerSlideViewer({
  deck,
  slideIndex = 0,
  onSlideChange,
}: ExplainerSlideViewerProps): React.ReactNode {
  const slide = deck.slides[slideIndex];
  const canvasRef = useRef<HTMLCanvasElement>(null);

  if (!slide) {
    return <div>No slide at index {slideIndex}</div>;
  }

  // Use precompiled Scene IR from sceneBlock
  const sceneIr = slide.sceneBlock;

  // Extract geometry from node
  function getNodeGeometry(node: any): NodeGeometry {
    const layout = node.layout || {};
    return {
      x: layout.x ?? 0,
      y: layout.y ?? 0,
      width: layout.width ?? 0,
      height: layout.height ?? 0,
    };
  }

  // Extract style from node
  function getNodeStyle(node: any): NodeStyle {
    const style = node.style || {};
    return {
      fill: style.fill,
      stroke: style.stroke,
      strokeWidth: style.strokeWidth,
      fontSize: style.fontSize,
      fontWeight: style.fontWeight,
    };
  }

  // Root containers (the outermost boxes, e.g. "aiperf binary") get a plain
  // outline instead of a filled card, so nested colored nodes stay legible.
  function renderNode(
    ctx: CanvasRenderingContext2D,
    node: any,
    depth: number,
  ): void {
    const geom = getNodeGeometry(node);
    const style = getNodeStyle(node);
    const capability = node.capability || '';
    const radius = 8;

    if (capability === 'core.rect') {
      const isContainer = depth === 0;
      ctx.save();
      ctx.beginPath();
      roundedRectPath(ctx, geom.x, geom.y, geom.width, geom.height, radius);

      if (!isContainer && style.fill) {
        ctx.shadowColor = 'rgba(0, 0, 0, 0.35)';
        ctx.shadowBlur = 10;
        ctx.shadowOffsetY = 3;
        ctx.fillStyle = style.fill;
        ctx.fill();
        ctx.shadowColor = 'transparent';
      }

      ctx.lineWidth = style.strokeWidth ?? (isContainer ? 1.5 : 1);
      ctx.strokeStyle = isContainer
        ? '#596266'
        : (style.stroke ?? 'rgba(255, 255, 255, 0.18)');
      ctx.stroke();
      ctx.restore();
    } else if (capability === 'core.text') {
      const size = style.fontSize ?? 14;
      ctx.font = `${style.fontWeight || 500} ${size}px Inter, "IBM Plex Sans", "Segoe UI", sans-serif`;
      // Compiled decks originally targeted a white canvas, so most text
      // fills are dark "ink" tones that go near-invisible on our dark
      // panel. Only trust an explicit fill when it isn't one of those.
      const INK_FILLS = new Set(['#1f2937', '#4b5563']);
      ctx.fillStyle =
        style.fill && !INK_FILLS.has(style.fill.toLowerCase())
          ? style.fill
          : textColorFor(node, depth);
      ctx.textBaseline = 'top';
      ctx.fillText(node.text || '', geom.x + 12, geom.y + 10);
    } else if (capability === 'core.line') {
      const from = node.from || {};
      const to = node.to || {};
      ctx.save();
      ctx.strokeStyle = style.stroke || 'rgba(241, 243, 242, 0.45)';
      ctx.lineWidth = style.strokeWidth ?? 1.5;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(from.x ?? 0, from.y ?? 0);
      ctx.lineTo(to.x ?? 0, to.y ?? 0);
      ctx.stroke();
      ctx.restore();
    }

    // Recursively render children
    if (Array.isArray(node.children)) {
      for (const child of node.children) {
        renderNode(ctx, child, depth + 1);
      }
    }
  }

  function roundedRectPath(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    width: number,
    height: number,
    radius: number,
  ): void {
    const r = Math.min(radius, width / 2, height / 2);
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + width, y, x + width, y + height, r);
    ctx.arcTo(x + width, y + height, x, y + height, r);
    ctx.arcTo(x, y + height, x, y, r);
    ctx.arcTo(x, y, x + width, y, r);
    ctx.closePath();
  }

  // Text on a filled colored node reads better light; text on the bare
  // container/background reads better as the app's muted chalk color.
  function textColorFor(node: any, depth: number): string {
    if (depth === 0) return '#aeb4b5';
    return '#f8fafc';
  }

  // Render scene to canvas
  useEffect(() => {
    if (!sceneIr || !canvasRef.current) {
      return;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size based on scene bounds
    const roots = sceneIr.roots || [];
    let maxX = 500, maxY = 500;

    roots.forEach((root: any) => {
      const geom = getNodeGeometry(root);
      maxX = Math.max(maxX, geom.x + geom.width);
      maxY = Math.max(maxY, geom.y + geom.height);
    });

    const margin = 40;
    const cssWidth = maxX + margin;
    const cssHeight = maxY + margin;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = cssWidth * dpr;
    canvas.height = cssHeight * dpr;
    canvas.style.width = `${cssWidth}px`;
    canvas.style.height = `${cssHeight}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Clear canvas with the app's dark panel surface, not a white card
    ctx.fillStyle = '#292c2d';
    ctx.fillRect(0, 0, cssWidth, cssHeight);

    // Render all root nodes
    roots.forEach((root: any) => {
      renderNode(ctx, root, 0);
    });
  }, [sceneIr]);

  if (!sceneIr) {
    return <div>No scene visualization available for slide {slideIndex}</div>;
  }

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
      <canvas
        ref={canvasRef}
        style={{
          display: 'block',
          maxWidth: '100%',
          height: 'auto',
          backgroundColor: 'var(--flow-panel, #292c2d)',
          borderRadius: '12px',
          border: '1px solid var(--flow-guide, #777d80)',
          boxShadow: '0 8px 24px rgba(0, 0, 0, 0.35)',
          margin: '20px auto',
        }}
      />
    </div>
  );
}
