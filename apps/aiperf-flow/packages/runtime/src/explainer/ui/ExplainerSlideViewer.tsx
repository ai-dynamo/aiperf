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

  const isPrevDisabled = slideIndex === 0;
  const isNextDisabled = slideIndex >= deck.slides.length - 1;

  const handlePrevious = (): void => {
    if (!isPrevDisabled && onSlideChange) {
      onSlideChange(slideIndex - 1);
    }
  };

  const handleNext = (): void => {
    if (!isNextDisabled && onSlideChange) {
      onSlideChange(slideIndex + 1);
    }
  };

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
    <div
      style={{
        width: '100%',
        height: '100vh',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'auto',
        backgroundColor: 'var(--flow-board, #232526)',
      }}
    >
      <div
        style={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          overflow: 'auto',
          minHeight: 0,
        }}
      >
        <canvas
          ref={canvasRef}
          style={{
            display: 'block',
            maxWidth: '100%',
            height: 'auto',
            backgroundColor: 'var(--flow-raised, #303334)',
            margin: '20px auto',
          }}
        />
      </div>

      <nav
        aria-label="Slide navigation"
        style={{
          display: 'flex',
          flexDirection: 'row',
          justifyContent: 'space-between',
          alignItems: 'center',
          gap: '0.75rem',
          padding: '1rem',
          borderTop: '2px solid var(--flow-guide, #777d80)',
          backgroundColor: 'var(--flow-panel, #292c2d)',
        }}
      >
        <button
          disabled={isPrevDisabled}
          onClick={handlePrevious}
          type="button"
          style={{
            minHeight: '2.5rem',
            padding: '0.48rem 0.75rem',
            border: '2px solid var(--flow-guide, #777d80)',
            borderRadius: 'calc(12px - 4px)',
            color: 'var(--flow-chalk, #f1f3f2)',
            backgroundColor: 'var(--flow-control-surface, #383c3e)',
            cursor: isPrevDisabled ? 'not-allowed' : 'pointer',
            opacity: isPrevDisabled ? 0.5 : 1,
            fontFamily: 'inherit',
            fontSize: 'inherit',
            transition: 'all 160ms ease-out',
          }}
          onMouseEnter={(e) => {
            if (!isPrevDisabled) {
              const target = e.target as HTMLButtonElement;
              target.style.borderColor = 'var(--flow-signal, #71d8d0)';
              target.style.color = 'var(--flow-signal, #71d8d0)';
            }
          }}
          onMouseLeave={(e) => {
            const target = e.target as HTMLButtonElement;
            target.style.borderColor = 'var(--flow-guide, #777d80)';
            target.style.color = 'var(--flow-chalk, #f1f3f2)';
          }}
        >
          ← Previous
        </button>

        <div
          style={{
            color: 'var(--flow-chalk-muted, #aeb4b5)',
            fontFamily: '"IBM Plex Mono", "Cascadia Code", monospace',
            fontSize: '0.875rem',
            fontWeight: '500',
            whiteSpace: 'nowrap',
          }}
        >
          Slide {slideIndex + 1} of {deck.slides.length}
        </div>

        <button
          disabled={isNextDisabled}
          onClick={handleNext}
          type="button"
          style={{
            minHeight: '2.5rem',
            padding: '0.48rem 0.75rem',
            border: '2px solid var(--flow-guide, #777d80)',
            borderRadius: 'calc(12px - 4px)',
            color: 'var(--flow-chalk, #f1f3f2)',
            backgroundColor: 'var(--flow-control-surface, #383c3e)',
            cursor: isNextDisabled ? 'not-allowed' : 'pointer',
            opacity: isNextDisabled ? 0.5 : 1,
            fontFamily: 'inherit',
            fontSize: 'inherit',
            transition: 'all 160ms ease-out',
          }}
          onMouseEnter={(e) => {
            if (!isNextDisabled) {
              const target = e.target as HTMLButtonElement;
              target.style.borderColor = 'var(--flow-signal, #71d8d0)';
              target.style.color = 'var(--flow-signal, #71d8d0)';
            }
          }}
          onMouseLeave={(e) => {
            const target = e.target as HTMLButtonElement;
            target.style.borderColor = 'var(--flow-guide, #777d80)';
            target.style.color = 'var(--flow-chalk, #f1f3f2)';
          }}
        >
          Next →
        </button>
      </nav>
    </div>
  );
}
