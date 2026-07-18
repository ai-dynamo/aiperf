// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useEffect, useRef } from 'react';

interface ExplainerSlideViewerProps {
  deck: any;
  slideIndex?: number;
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

  // Render a single node to canvas
  function renderNode(ctx: CanvasRenderingContext2D, node: any): void {
    const geom = getNodeGeometry(node);
    const style = getNodeStyle(node);
    const capability = node.capability || '';

    if (capability === 'core.rect') {
      // Draw rectangle
      if (style.fill) {
        ctx.fillStyle = style.fill;
        ctx.fillRect(geom.x, geom.y, geom.width, geom.height);
      }
      if (style.stroke) {
        ctx.strokeStyle = style.stroke;
        ctx.lineWidth = style.strokeWidth ?? 1;
        ctx.strokeRect(geom.x, geom.y, geom.width, geom.height);
      }
    } else if (capability === 'core.text') {
      // Draw text
      if (style.fontSize) {
        ctx.font = `${style.fontWeight || 'normal'} ${style.fontSize}px sans-serif`;
      }
      ctx.fillStyle = style.fill || '#000';
      ctx.textBaseline = 'top';
      ctx.fillText(node.text || '', geom.x, geom.y);
    } else if (capability === 'core.line') {
      // Draw line
      const from = node.from || {};
      const to = node.to || {};
      ctx.strokeStyle = style.stroke || '#000';
      ctx.lineWidth = style.strokeWidth ?? 1;
      ctx.beginPath();
      ctx.moveTo(from.x ?? 0, from.y ?? 0);
      ctx.lineTo(to.x ?? 0, to.y ?? 0);
      ctx.stroke();
    }

    // Recursively render children
    if (Array.isArray(node.children)) {
      for (const child of node.children) {
        renderNode(ctx, child);
      }
    }
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

    canvas.width = maxX + 50;
    canvas.height = maxY + 50;

    // Clear canvas
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Render all root nodes
    roots.forEach((root: any) => {
      renderNode(ctx, root);
    });
  }, [sceneIr]);

  if (!sceneIr) {
    return <div>No scene visualization available for slide {slideIndex}</div>;
  }

  return (
    <div style={{ width: '100%', height: '100vh', overflow: 'auto', backgroundColor: '#f5f5f5' }}>
      <canvas
        ref={canvasRef}
        style={{
          display: 'block',
          maxWidth: '100%',
          height: 'auto',
          backgroundColor: '#fff',
          margin: '20px auto',
        }}
      />
    </div>
  );
}
