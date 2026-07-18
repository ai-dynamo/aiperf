/**
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useCallback } from 'react';
import type { ExplainerDefinition, SlideDefinition } from '@aiperf/flow-compiler';
import type { ResolvedTheme } from '../../theme/registry-runtime.js';

export interface ResponsiveLayoutProps {
  deck: ExplainerDefinition;
  currentSlide: SlideDefinition;
  slideIndex: number;
  totalSlides: number;
  onNavigate: (index: number) => void;
  theme: ResolvedTheme;
  isImmersive?: boolean;
  onImmersiveToggle?: (immersive: boolean) => void;
}

export interface ViewportSize {
  width: number;
  height: number;
  isMobile: boolean;
  isTablet: boolean;
  isDesktop: boolean;
}

export type LayoutMode = 'mobile' | 'tablet' | 'desktop' | 'immersive';

/**
 * Responsive layout component that adapts to viewport size.
 * Breakpoints:
 * - Mobile: < 640px
 * - Tablet: 640px - 1024px
 * - Desktop: > 1024px
 * - Immersive: full-screen overlay
 */
export function ResponsiveLayout(props: ResponsiveLayoutProps): JSX.Element {
  const {
    deck,
    currentSlide,
    slideIndex,
    totalSlides,
    onNavigate,
    theme,
    isImmersive = false,
    onImmersiveToggle,
  } = props;

  const [viewport, setViewport] = useState<ViewportSize>(getViewportSize());
  const [sidebarOpen, setSidebarOpen] = useState(!viewport.isMobile);

  useEffect(() => {
    const handleResize = () => {
      const newViewport = getViewportSize();
      setViewport(newViewport);
      // Auto-close sidebar on mobile
      if (newViewport.isMobile) {
        setSidebarOpen(false);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const layoutMode: LayoutMode = isImmersive
    ? 'immersive'
    : viewport.isDesktop
      ? 'desktop'
      : viewport.isTablet
        ? 'tablet'
        : 'mobile';

  const handlePrevSlide = useCallback(() => {
    if (slideIndex > 0) {
      onNavigate(slideIndex - 1);
    }
  }, [slideIndex, onNavigate]);

  const handleNextSlide = useCallback(() => {
    if (slideIndex < totalSlides - 1) {
      onNavigate(slideIndex + 1);
    }
  }, [slideIndex, totalSlides, onNavigate]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      switch (e.key) {
        case 'ArrowLeft':
          e.preventDefault();
          handlePrevSlide();
          break;
        case 'ArrowRight':
          e.preventDefault();
          handleNextSlide();
          break;
        case 'Escape':
          if (isImmersive && onImmersiveToggle) {
            e.preventDefault();
            onImmersiveToggle(false);
          }
          break;
        default:
          break;
      }
    },
    [handlePrevSlide, handleNextSlide, isImmersive, onImmersiveToggle]
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  if (layoutMode === 'immersive') {
    return (
      <ImmersiveLayout
        deck={deck}
        currentSlide={currentSlide}
        slideIndex={slideIndex}
        totalSlides={totalSlides}
        onNavigate={onNavigate}
        theme={theme}
        onClose={() => onImmersiveToggle?.(false)}
        onPrev={handlePrevSlide}
        onNext={handleNextSlide}
      />
    );
  }

  return (
    <div
      className={`explainer-responsive-layout explainer-${layoutMode}`}
      style={getLayoutStyles(theme, layoutMode)}
    >
      {/* Topbar */}
      <div className="explainer-topbar" style={getTopbarStyles(theme, layoutMode)}>
        {layoutMode !== 'desktop' && (
          <button
            className="sidebar-toggle"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            aria-label="Toggle navigation"
            style={getButtonStyles(theme)}
          >
            ☰
          </button>
        )}
        <div className="topbar-title">{deck.eyebrowLabel}</div>
        <div className="progress-indicator">
          {slideIndex + 1} / {totalSlides}
        </div>
      </div>

      <div className="explainer-body" style={getBodyStyles(layoutMode)}>
        {/* Sidebar */}
        {(layoutMode === 'desktop' || sidebarOpen) && (
          <aside
            className={`explainer-sidebar ${sidebarOpen ? 'open' : 'closed'}`}
            style={getSidebarStyles(theme, layoutMode)}
          >
            <nav>
              {deck.slides.map((slide, idx) => (
                <button
                  key={idx}
                  className={`slide-nav-item ${idx === slideIndex ? 'active' : ''}`}
                  onClick={() => {
                    onNavigate(idx);
                    if (layoutMode !== 'desktop') setSidebarOpen(false);
                  }}
                  style={getNavItemStyles(theme, idx === slideIndex)}
                >
                  <span className="nav-number">{idx + 1}</span>
                  <span className="nav-title">{slide.title}</span>
                </button>
              ))}
            </nav>
          </aside>
        )}

        {/* Main slide content */}
        <main className="explainer-main" style={getMainStyles(layoutMode)}>
          <article className="slide-content" style={getSlideContentStyles(theme)}>
            <header>
              <div className="slide-eyebrow">{currentSlide.eyebrow}</div>
              <h1 className="slide-title">{currentSlide.title}</h1>
              <p className="slide-lede">{currentSlide.lede}</p>
            </header>

            {currentSlide.term && (
              <div className="slide-term" style={getTermStyles(theme)}>
                <strong>{currentSlide.term.word}:</strong>
                <span>{currentSlide.term.meaning}</span>
              </div>
            )}

            {currentSlide.points.length > 0 && (
              <ul className="slide-points">
                {currentSlide.points.map((point, idx) => (
                  <li key={idx}>{point}</li>
                ))}
              </ul>
            )}

            {currentSlide.caption && (
              <p className="slide-caption">{currentSlide.caption}</p>
            )}

            {layoutMode === 'mobile' && (
              <button
                className="immersive-button"
                onClick={() => onImmersiveToggle?.(true)}
                style={getImmersiveButtonStyles(theme)}
              >
                Full Screen
              </button>
            )}
          </article>
        </main>
      </div>

      {/* Bottom controls */}
      <footer className="explainer-controls" style={getControlsStyles(theme, layoutMode)}>
        <button
          className="nav-button prev"
          onClick={handlePrevSlide}
          disabled={slideIndex === 0}
          aria-label="Previous slide"
          style={getButtonStyles(theme, slideIndex === 0)}
        >
          ← Prev
        </button>

        <div className="slide-progress">
          <div
            className="progress-bar"
            style={{
              ...getProgressBarStyles(theme),
              width: `${((slideIndex + 1) / totalSlides) * 100}%`,
            }}
          />
        </div>

        <button
          className="nav-button next"
          onClick={handleNextSlide}
          disabled={slideIndex === totalSlides - 1}
          aria-label="Next slide"
          style={getButtonStyles(theme, slideIndex === totalSlides - 1)}
        >
          Next →
        </button>

        {layoutMode === 'desktop' && (
          <button
            className="immersive-button"
            onClick={() => onImmersiveToggle?.(true)}
            style={getImmersiveButtonStyles(theme)}
          >
            Full Screen
          </button>
        )}
      </footer>
    </div>
  );
}

/**
 * Immersive full-screen layout
 */
function ImmersiveLayout(props: {
  deck: ExplainerDefinition;
  currentSlide: SlideDefinition;
  slideIndex: number;
  totalSlides: number;
  onNavigate: (index: number) => void;
  theme: ResolvedTheme;
  onClose: () => void;
  onPrev: () => void;
  onNext: () => void;
}): JSX.Element {
  const { currentSlide, slideIndex, totalSlides, onNavigate, theme, onClose, onPrev, onNext } =
    props;

  return (
    <div className="explainer-immersive" style={getImmersiveLayoutStyles(theme)}>
      {/* Full-screen content */}
      <div className="immersive-content">
        <h1 className="immersive-title">{currentSlide.title}</h1>
        <p className="immersive-lede">{currentSlide.lede}</p>
        {currentSlide.points.length > 0 && (
          <ul className="immersive-points">
            {currentSlide.points.map((point, idx) => (
              <li key={idx}>{point}</li>
            ))}
          </ul>
        )}
      </div>

      {/* Immersive controls overlay */}
      <div className="immersive-controls-overlay" style={getImmersiveControlsOverlayStyles()}>
        {/* Top-left: close button */}
        <button
          className="immersive-close"
          onClick={onClose}
          aria-label="Exit full screen"
          style={getCloseButtonStyles(theme)}
        >
          ✕
        </button>

        {/* Center: navigation */}
        <div className="immersive-nav-center">
          <button
            className="immersive-nav-prev"
            onClick={onPrev}
            disabled={slideIndex === 0}
            aria-label="Previous"
            style={getImmersiveTouchButtonStyles(theme)}
          >
            ←
          </button>
          <span className="immersive-counter">
            {slideIndex + 1} / {totalSlides}
          </span>
          <button
            className="immersive-nav-next"
            onClick={onNext}
            disabled={slideIndex === totalSlides - 1}
            aria-label="Next"
            style={getImmersiveTouchButtonStyles(theme)}
          >
            →
          </button>
        </div>

        {/* Bottom: progress bar */}
        <div className="immersive-progress" style={getImmersiveProgressStyles()}>
          <div
            className="immersive-progress-bar"
            style={{
              ...getProgressBarStyles(theme),
              width: `${((slideIndex + 1) / totalSlides) * 100}%`,
            }}
          />
        </div>

        {/* Right sidebar: slide thumbnail navigation (touch-friendly) */}
        <div className="immersive-thumbnails" style={getImmersiveThumbnailsStyles()}>
          {props.deck.slides.map((_, idx) => (
            <button
              key={idx}
              className={`immersive-thumbnail ${idx === slideIndex ? 'active' : ''}`}
              onClick={() => onNavigate(idx)}
              style={getImmersiveThumbnailStyles(theme, idx === slideIndex)}
              aria-label={`Go to slide ${idx + 1}`}
            >
              {idx + 1}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Style Generators
// ============================================================================

function getViewportSize(): ViewportSize {
  const width = typeof window !== 'undefined' ? window.innerWidth : 1024;
  const height = typeof window !== 'undefined' ? window.innerHeight : 768;

  return {
    width,
    height,
    isMobile: width < 640,
    isTablet: width >= 640 && width < 1024,
    isDesktop: width >= 1024,
  };
}

function getThemeColor(theme: ResolvedTheme, role: string): string {
  const val = theme.values[role as keyof typeof theme.values];
  return typeof val === 'string' ? val : '#f2eee3';
}

function getLayoutStyles(theme: ResolvedTheme, mode: LayoutMode): React.CSSProperties {
  return {
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    backgroundColor: getThemeColor(theme, 'surface.primary'),
    color: getThemeColor(theme, 'ink.primary'),
    fontFamily: 'system-ui, -apple-system, sans-serif',
    overflow: 'hidden',
  };
}

function getTopbarStyles(theme: ResolvedTheme, mode: LayoutMode): React.CSSProperties {
  const mobilePadding = mode === 'mobile' ? '0.75rem' : '1rem';
  return {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: mobilePadding,
    borderBottom: `1px solid ${getThemeColor(theme, 'structure.divider')}`,
    backgroundColor: getThemeColor(theme, 'surface.secondary'),
    minHeight: '3rem',
    gap: mode === 'mobile' ? '0.5rem' : '1rem',
  };
}

function getBodyStyles(mode: LayoutMode): React.CSSProperties {
  return {
    display: 'flex',
    flex: 1,
    overflow: mode === 'immersive' ? 'hidden' : 'auto',
    gap: mode === 'desktop' ? '1rem' : 0,
  };
}

function getSidebarStyles(theme: ResolvedTheme, mode: LayoutMode): React.CSSProperties {
  if (mode === 'desktop') {
    return {
      width: '250px',
      borderRight: `1px solid ${getThemeColor(theme, 'structure.divider')}`,
      overflowY: 'auto',
      padding: '1rem',
      display: 'flex',
      flexDirection: 'column',
    };
  }

  // Mobile/tablet overlay
  return {
    position: 'fixed',
    left: 0,
    top: '3rem',
    bottom: 'auto',
    width: '100%',
    maxWidth: '300px',
    backgroundColor: getThemeColor(theme, 'surface.primary'),
    borderRight: `1px solid ${getThemeColor(theme, 'structure.divider')}`,
    overflowY: 'auto',
    padding: '1rem',
    zIndex: 1000,
    boxShadow: '2px 0 8px rgba(0,0,0,0.2)',
  };
}

function getNavItemStyles(theme: ResolvedTheme, isActive: boolean): React.CSSProperties {
  return {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    padding: '0.5rem',
    marginBottom: '0.5rem',
    border: 'none',
    borderRadius: '4px',
    backgroundColor: isActive ? getThemeColor(theme, 'accent.execute') : 'transparent',
    color: isActive ? getThemeColor(theme, 'surface.primary') : getThemeColor(theme, 'ink.primary'),
    cursor: 'pointer',
    fontSize: '0.875rem',
    transition: 'background-color 200ms ease',
  };
}

function getMainStyles(mode: LayoutMode): React.CSSProperties {
  return {
    flex: 1,
    overflowY: 'auto',
    padding: mode === 'mobile' ? '1rem' : mode === 'tablet' ? '1.5rem' : '2rem',
    maxWidth: mode === 'desktop' ? '900px' : undefined,
    margin: mode === 'desktop' ? '0 auto' : 0,
    width: '100%',
  };
}

function getSlideContentStyles(theme: ResolvedTheme): React.CSSProperties {
  return {
    display: 'flex',
    flexDirection: 'column',
    gap: '1rem',
    animation: 'fadeIn 300ms ease',
  };
}

function getTermStyles(theme: ResolvedTheme): React.CSSProperties {
  return {
    padding: '1rem',
    borderLeft: `4px solid ${getThemeColor(theme, 'accent.execute')}`,
    backgroundColor: getThemeColor(theme, 'surface.secondary'),
    borderRadius: '4px',
    fontSize: '0.95rem',
  };
}

function getControlsStyles(theme: ResolvedTheme, mode: LayoutMode): React.CSSProperties {
  return {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: mode === 'mobile' ? '0.5rem' : '1rem',
    padding: mode === 'mobile' ? '0.75rem' : '1rem',
    borderTop: `1px solid ${getThemeColor(theme, 'structure.divider')}`,
    backgroundColor: getThemeColor(theme, 'surface.secondary'),
    minHeight: '3rem',
  };
}

function getProgressBarStyles(theme: ResolvedTheme): React.CSSProperties {
  return {
    height: '3px',
    backgroundColor: getThemeColor(theme, 'accent.execute'),
    transition: 'width 300ms ease',
  };
}

function getButtonStyles(theme: ResolvedTheme, disabled = false): React.CSSProperties {
  return {
    padding: '0.5rem 1rem',
    border: `1px solid ${getThemeColor(theme, 'structure.divider')}`,
    borderRadius: '4px',
    backgroundColor: getThemeColor(theme, 'surface.secondary'),
    color: getThemeColor(theme, 'ink.primary'),
    cursor: disabled ? 'not-allowed' : 'pointer',
    opacity: disabled ? 0.5 : 1,
    fontSize: '0.875rem',
    transition: 'background-color 200ms ease',
  };
}

function getImmersiveButtonStyles(theme: ResolvedTheme): React.CSSProperties {
  return {
    padding: '0.5rem 1rem',
    border: 'none',
    borderRadius: '4px',
    backgroundColor: getThemeColor(theme, 'accent.execute'),
    color: getThemeColor(theme, 'surface.primary'),
    cursor: 'pointer',
    fontSize: '0.875rem',
    fontWeight: 'bold',
    transition: 'background-color 200ms ease',
  };
}

function getImmersiveLayoutStyles(theme: ResolvedTheme): React.CSSProperties {
  return {
    position: 'fixed',
    top: 0,
    left: 0,
    width: '100vw',
    height: '100vh',
    backgroundColor: getThemeColor(theme, 'surface.primary'),
    color: getThemeColor(theme, 'ink.primary'),
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 2000,
    overflow: 'hidden',
  };
}

function getImmersiveControlsOverlayStyles(): React.CSSProperties {
  return {
    position: 'fixed',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    pointerEvents: 'none',
    zIndex: 2001,
  };
}

function getCloseButtonStyles(theme: ResolvedTheme): React.CSSProperties {
  return {
    position: 'fixed',
    top: '1rem',
    left: '1rem',
    padding: '0.75rem 1rem',
    backgroundColor: getThemeColor(theme, 'surface.secondary'),
    color: getThemeColor(theme, 'ink.primary'),
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '1.5rem',
    pointerEvents: 'auto',
    zIndex: 2002,
  };
}

function getImmersiveTouchButtonStyles(theme: ResolvedTheme): React.CSSProperties {
  return {
    padding: '1rem 1.5rem',
    backgroundColor: getThemeColor(theme, 'accent.execute'),
    color: getThemeColor(theme, 'surface.primary'),
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    fontSize: '1.25rem',
    fontWeight: 'bold',
    pointerEvents: 'auto',
    transition: 'transform 200ms ease',
  };
}

function getImmersiveProgressStyles(): React.CSSProperties {
  return {
    position: 'fixed',
    bottom: 0,
    left: 0,
    width: '100%',
    height: '4px',
    pointerEvents: 'none',
  };
}

function getImmersiveThumbnailsStyles(): React.CSSProperties {
  return {
    position: 'fixed',
    right: '1rem',
    top: '50%',
    transform: 'translateY(-50%)',
    display: 'flex',
    flexDirection: 'column',
    gap: '0.5rem',
    pointerEvents: 'auto',
    maxHeight: '80vh',
    overflowY: 'auto',
  };
}

function getImmersiveThumbnailStyles(
  theme: ResolvedTheme,
  isActive: boolean
): React.CSSProperties {
  return {
    width: '2.5rem',
    height: '2.5rem',
    padding: '0',
    borderRadius: '4px',
    border: `2px solid ${isActive ? getThemeColor(theme, 'accent.execute') : getThemeColor(theme, 'structure.divider')}`,
    backgroundColor: isActive
      ? getThemeColor(theme, 'accent.execute')
      : getThemeColor(theme, 'surface.secondary'),
    color: isActive ? getThemeColor(theme, 'surface.primary') : getThemeColor(theme, 'ink.primary'),
    cursor: 'pointer',
    fontSize: '0.75rem',
    fontWeight: 'bold',
    transition: 'all 200ms ease',
  };
}

// Add CSS animations
export const responsiveLayoutStyles = `
@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.slide-progress {
  flex: 1;
  height: 4px;
  background-color: currentColor;
  opacity: 0.2;
  border-radius: 2px;
  overflow: hidden;
}

.slide-points {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.slide-points li {
  padding: 0.5rem 0 0.5rem 1.5rem;
  position: relative;
}

.slide-points li:before {
  content: '•';
  position: absolute;
  left: 0;
  font-weight: bold;
  color: var(--accent-execute, #72d6a2);
}

.immersive-content {
  text-align: center;
  z-index: 2000;
  pointer-events: none;
}

.immersive-title {
  font-size: 3rem;
  font-weight: bold;
  margin-bottom: 1rem;
}

.immersive-lede {
  font-size: 1.5rem;
  margin-bottom: 2rem;
  opacity: 0.9;
}

.immersive-points {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 1rem;
  max-width: 600px;
  margin: 0 auto;
}

.immersive-points li {
  font-size: 1.25rem;
  padding: 0.5rem;
}

.immersive-nav-center {
  position: fixed;
  bottom: 3rem;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  gap: 2rem;
  align-items: center;
  pointer-events: auto;
  z-index: 2002;
}

.immersive-counter {
  font-size: 1rem;
  opacity: 0.8;
  min-width: 60px;
  text-align: center;
}

@media (max-width: 640px) {
  .immersive-title {
    font-size: 2rem;
  }

  .immersive-lede {
    font-size: 1.125rem;
  }

  .immersive-points li {
    font-size: 1rem;
  }

  .immersive-nav-center {
    gap: 1rem;
  }

  .immersive-thumbnails {
    max-height: none !important;
    max-width: 3rem !important;
    overflow-x: hidden;
  }
}
`;

export default ResponsiveLayout;
