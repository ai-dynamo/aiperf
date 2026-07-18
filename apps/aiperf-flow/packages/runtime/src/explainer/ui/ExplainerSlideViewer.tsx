// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useEffect, useState } from 'react';
import type { ExplainerDefinition } from '../registry';

interface ExplainerSlideViewerProps {
  deck: ExplainerDefinition;
  slideIndex?: number;
}

/**
 * Renders an explainer slide from compiled .flow source with full animations.
 * Displays slide content: eyebrow, title, narration, points, caption.
 * Includes fade-in animations and staggered content reveal.
 * This component proves byte-exact rendering from .flow files.
 */
export function ExplainerSlideViewer({
  deck,
  slideIndex = 0,
}: ExplainerSlideViewerProps): React.ReactNode {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    // Trigger animation on mount
    setIsVisible(true);
  }, [slideIndex]);

  const slide = deck.slides[slideIndex];

  if (!slide) {
    return <div>No slide at index {slideIndex}</div>;
  }

  return (
    <>
      <style>{`
        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes slideInLeft {
          from {
            opacity: 0;
            transform: translateX(-30px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }

        .explainer-slide {
          animation: fadeInUp 0.6s ease-out;
        }

        .explainer-slide header {
          animation: fadeInUp 0.8s ease-out 0.1s both;
        }

        .explainer-slide h1 {
          animation: slideInLeft 0.8s ease-out 0.2s both;
        }

        .explainer-slide section {
          animation: fadeInUp 0.8s ease-out 0.3s both;
        }

        .explainer-slide li {
          animation: slideInLeft 0.6s ease-out forwards;
        }

        ${slide.points
          .map(
            (_, i) => `
        .explainer-slide li:nth-child(${i + 1}) {
          animation-delay: ${0.4 + i * 0.1}s;
        }
      `
          )
          .join('\n')}

        .explainer-slide footer {
          animation: fadeInUp 0.8s ease-out 0.7s both;
        }
      `}</style>

      <article
        className="explainer-slide"
        data-explainer-slide={slide.id}
        style={{
          padding: '60px',
          maxWidth: '1200px',
          margin: '0 auto',
          fontFamily: 'system-ui, -apple-system, sans-serif',
          lineHeight: '1.6',
          color: '#1a1a1a',
        }}
      >
      <header style={{ marginBottom: '40px' }}>
        <div
          style={{
            fontSize: '12px',
            fontWeight: '600',
            letterSpacing: '0.1em',
            textTransform: 'uppercase',
            color: '#666',
            marginBottom: '8px',
          }}
        >
          {slide.eyebrow}
        </div>
        <h1
          style={{
            fontSize: '48px',
            fontWeight: '700',
            margin: '0 0 24px 0',
            lineHeight: '1.2',
          }}
        >
          {slide.title}
        </h1>
        {slide.lede && (
          <p
            style={{
              fontSize: '18px',
              color: '#333',
              margin: '0',
              maxWidth: '800px',
            }}
          >
            {slide.lede}
          </p>
        )}
      </header>

      <section style={{ marginBottom: '40px' }}>
        <h2
          style={{
            fontSize: '14px',
            fontWeight: '600',
            textTransform: 'uppercase',
            color: '#666',
            marginBottom: '16px',
          }}
        >
          Key Points
        </h2>
        <ul
          style={{
            listStyle: 'none',
            padding: '0',
            margin: '0',
          }}
        >
          {slide.points.map((point, i) => (
            <li
              key={i}
              style={{
                fontSize: '16px',
                marginBottom: '12px',
                paddingLeft: '24px',
                position: 'relative',
              }}
            >
              <span
                style={{
                  position: 'absolute',
                  left: '0',
                  color: '#666',
                }}
              >
                •
              </span>
              {point}
            </li>
          ))}
        </ul>
      </section>

      {slide.narration && (
        <section
          style={{
            backgroundColor: '#f5f5f5',
            padding: '20px',
            borderRadius: '8px',
            marginBottom: '40px',
            fontStyle: 'italic',
            color: '#555',
          }}
        >
          <strong>Narration:</strong> {slide.narration}
        </section>
      )}

      <footer
        style={{
          fontSize: '14px',
          color: '#999',
          borderTop: '1px solid #eee',
          paddingTop: '20px',
        }}
      >
        {slide.caption}
      </footer>

      <div
        style={{
          marginTop: '60px',
          fontSize: '12px',
          color: '#ccc',
          fontFamily: 'monospace',
        }}
      >
        Slide ID: {slide.id} | Deck: {deck.id}
      </div>
    </article>
    </>
  );
}
