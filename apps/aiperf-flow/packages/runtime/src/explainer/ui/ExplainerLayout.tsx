// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ExplainerLayout: Top-level React component for explainer slide presentations.
//! Renders topbar (title, progress), sidebar (outline), main (slide renderer), and bottom (controls).
//! Integrates theme, narrator, and immersive context.

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ExplainerDefinition, SlideDefinition } from '@aiperf/flow-compiler';
import type { NarratorBackend } from '../narrative/narrator.js';
import type { ResolvedTheme } from '../theme/types.js';
import { ExplainerThemeContext, type SlideCss } from '../theme-context.js';
import { SlideshowController } from '../controller.js';
import { NarratorBinding } from '../narrator-binding.js';
import { SubtitleRenderer } from './SubtitleRenderer.js';
import { ImmersiveExplainerContext } from '../immersive-integration.js';

/**
 * Props for ExplainerLayout component.
 */
export interface ExplainerLayoutProps {
  /** Explainer deck definition */
  deck: ExplainerDefinition;
  /** Current slide index (0-based) */
  slideIndex: number;
  /** Callback when slide index changes */
  onNavigate: (index: number) => void;
  /** Narrator backend for speech synthesis */
  narrator: NarratorBackend;
  /** Resolved theme for styling */
  theme: ResolvedTheme;
  /** Optional immersive mode flag */
  immersive?: boolean;
  /** Optional callback when narration completes */
  onNarrationComplete?: () => void;
}

/**
 * ExplainerLayout component: Main shell for slide-based explainer presentations.
 *
 * Renders a multi-panel layout:
 * - Topbar: Deck title, current slide number / total, progress bar
 * - Sidebar: Outline of all slides with navigation
 * - Main: Current slide content (title, lede, points, narration, scene)
 * - Bottom: Navigation buttons (prev, play/pause, next) and controls
 *
 * Integrates theme colors via CSS variables, narrator sync for speech,
 * and immersive preview support for cinematic mode.
 */
export function ExplainerLayout({
  deck,
  slideIndex,
  onNavigate,
  narrator,
  theme,
  immersive = false,
  onNarrationComplete,
}: ExplainerLayoutProps): React.ReactElement {
  // Local state for layout and narrator
  const [isPlaying, setIsPlaying] = useState(false);
  const [narrationPaused, setNarrationPaused] = useState(false);

  // Theme and scene context
  const themeContext = useMemo(() => new ExplainerThemeContext(), []);
  const immersiveContext = useMemo(() => new ImmersiveExplainerContext(), []);

  // Slideshow and narrator binding
  const [controller, setController] = useState<SlideshowController | null>(null);
  const bindingRef = useRef<NarratorBinding | null>(null);
  // Refs let the once-created controller's completion hook read the latest
  // navigation callback without being recreated on every render.
  const navigateRef = useRef(onNavigate);
  navigateRef.current = onNavigate;
  const onNarrationCompleteRef = useRef(onNarrationComplete);
  onNarrationCompleteRef.current = onNarrationComplete;

  // Initialize controller and binding on mount or when deck changes
  useEffect(() => {
    const created = new SlideshowController(deck, narrator, {
      onNarrationComplete: () => {
        onNarrationCompleteRef.current?.();
        const next = created.currentSlideIndex + 1;
        if (next < deck.slides.length) {
          navigateRef.current(next);
        }
      },
    });
    bindingRef.current = new NarratorBinding(created, narrator);
    setController(created);
    return () => {
      created.stopNarration();
    };
  }, [deck, narrator]);

  // Get current slide
  const currentSlide = useMemo(() => {
    return deck.slides[slideIndex];
  }, [deck.slides, slideIndex]);

  // Apply theme to current slide
  const slideCss = useMemo(() => {
    if (!currentSlide) return null;
    return themeContext.applyThemeToSlide(currentSlide, theme);
  }, [currentSlide, themeContext, theme]);

  // Handle navigation. Cancel any in-flight narration before moving so audio
  // and captions never bleed across the slide boundary.
  const handlePrevSlide = useCallback(async () => {
    if (slideIndex > 0) {
      controller?.stopNarration();
      setIsPlaying(false);
      setNarrationPaused(false);
      onNavigate(slideIndex - 1);
    }
  }, [slideIndex, onNavigate, controller]);

  const handleNextSlide = useCallback(async () => {
    if (slideIndex < deck.slides.length - 1) {
      controller?.stopNarration();
      setIsPlaying(false);
      setNarrationPaused(false);
      onNavigate(slideIndex + 1);
    }
  }, [slideIndex, deck.slides.length, onNavigate, controller]);

  const handleJumpToSlide = useCallback(
    (index: number) => {
      controller?.stopNarration();
      setIsPlaying(false);
      setNarrationPaused(false);
      onNavigate(index);
    },
    [onNavigate, controller]
  );

  // Handle play/pause
  const handlePlayPause = useCallback(() => {
    if (isPlaying) {
      if (narrationPaused) {
        bindingRef.current?.resumeNarration();
        setNarrationPaused(false);
      } else {
        bindingRef.current?.pauseNarration();
        setNarrationPaused(true);
      }
    } else {
      setIsPlaying(true);
      setNarrationPaused(false);
      // Start narration (and word-synced subtitles) for the current slide.
      void controller?.jumpToSlide(slideIndex);
    }
  }, [isPlaying, narrationPaused, slideIndex, controller]);

  const handleSkipNarration = useCallback(() => {
    bindingRef.current?.skipNarration();
    setIsPlaying(false);
  }, []);

  // Determine immersive viewport settings
  const sceneId = currentSlide?.sceneId;
  const immersiveScene = useMemo(() => {
    if (!immersive || !sceneId) return null;
    return immersiveContext.expandSlideToViewport(sceneId);
  }, [immersive, sceneId, immersiveContext]);

  const immersiveControls = useMemo(() => {
    if (!immersive) return null;
    return immersiveContext.applyImmersiveControls();
  }, [immersive, immersiveContext]);

  // Container styles
  const containerStyle: React.CSSProperties = {
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    width: '100%',
    backgroundColor: slideCss?.backgroundColor || '#24282b',
    color: slideCss?.color || '#f2eee3',
    fontFamily: 'system-ui, sans-serif',
    overflow: 'hidden',
  };

  const mainContentStyle: React.CSSProperties = {
    display: 'flex',
    flex: 1,
    overflow: 'hidden',
    gap: '16px',
    padding: '16px',
  };

  const sidebarStyle: React.CSSProperties = {
    width: immersive ? '0px' : '200px',
    overflowY: 'auto',
    borderRight: `1px solid ${slideCss?.borderColor || '#444'}`,
    paddingRight: '8px',
    display: immersive ? 'none' : 'block',
  };

  const mainSlideStyle: React.CSSProperties = {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    overflowY: 'auto',
    gap: '24px',
  };

  return (
    <div style={containerStyle}>
      {/* Topbar */}
      <TopBar
        deck={deck}
        currentIndex={slideIndex}
        totalSlides={deck.slides.length}
        theme={theme}
        slideCss={slideCss}
      />

      {/* Main content area */}
      <div style={mainContentStyle}>
        {/* Sidebar: Slide outline */}
        {!immersive && (
          <aside style={sidebarStyle}>
            <SlideOutline
              slides={deck.slides}
              currentIndex={slideIndex}
              onJumpToSlide={handleJumpToSlide}
              theme={theme}
              slideCss={slideCss}
            />
          </aside>
        )}

        {/* Main slide rendering area */}
        <main style={mainSlideStyle}>
          {currentSlide && (
            <SlideRenderer
              slide={currentSlide}
              slideIndex={slideIndex}
              totalSlides={deck.slides.length}
              theme={theme}
              slideCss={slideCss}
              isImmersive={immersive}
              immersiveScene={immersiveScene}
              immersiveControls={immersiveControls}
            />
          )}
          {/* Word-synchronized captions for the active narration. */}
          <SubtitleRenderer controller={controller} />
        </main>
      </div>

      {/* Bottom controls */}
      <BottomControls
        currentIndex={slideIndex}
        totalSlides={deck.slides.length}
        isPlaying={isPlaying}
        isPaused={narrationPaused}
        onPrev={handlePrevSlide}
        onPlayPause={handlePlayPause}
        onNext={handleNextSlide}
        onSkip={handleSkipNarration}
        canGoPrev={slideIndex > 0}
        canGoNext={slideIndex < deck.slides.length - 1}
        theme={theme}
        slideCss={slideCss}
      />
    </div>
  );
}

/**
 * Topbar component: Displays deck title, progress indicator, and slide counter.
 */
interface TopBarProps {
  deck: ExplainerDefinition;
  currentIndex: number;
  totalSlides: number;
  theme: ResolvedTheme;
  slideCss: SlideCss | null;
}

function TopBar({
  deck,
  currentIndex,
  totalSlides,
  theme,
  slideCss,
}: TopBarProps): React.ReactElement {
  const progress = totalSlides > 0 ? (currentIndex + 1) / totalSlides : 0;

  const topBarStyle: React.CSSProperties = {
    padding: '16px 24px',
    borderBottom: `1px solid ${slideCss?.borderColor || '#444'}`,
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  };

  const progressBarStyle: React.CSSProperties = {
    flex: 1,
    height: '4px',
    backgroundColor: slideCss?.borderColor || '#444',
    borderRadius: '2px',
    overflow: 'hidden',
    margin: '0 16px',
  };

  const progressFillStyle: React.CSSProperties = {
    height: '100%',
    width: `${progress * 100}%`,
    backgroundColor: slideCss?.accentColor || '#72d6a2',
    transition: 'width 0.3s ease',
  };

  const counterStyle: React.CSSProperties = {
    fontSize: '14px',
    fontWeight: '500',
    minWidth: '80px',
    textAlign: 'right',
  };

  return (
    <div style={topBarStyle}>
      <h1 style={{ margin: 0, fontSize: '18px', fontWeight: '600' }}>
        {deck.eyebrowLabel}: {deck.topic}
      </h1>
      <div style={progressBarStyle}>
        <div style={progressFillStyle} />
      </div>
      <div style={counterStyle}>
        {currentIndex + 1} / {totalSlides}
      </div>
    </div>
  );
}

/**
 * SlideOutline component: Sidebar showing all slides with clickable navigation.
 */
interface SlideOutlineProps {
  slides: SlideDefinition[];
  currentIndex: number;
  onJumpToSlide: (index: number) => void;
  theme: ResolvedTheme;
  slideCss: SlideCss | null;
}

function SlideOutline({
  slides,
  currentIndex,
  onJumpToSlide,
  theme,
  slideCss,
}: SlideOutlineProps): React.ReactElement {
  return (
    <nav style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
      <h2 style={{ fontSize: '14px', fontWeight: '600', margin: '0 0 8px 0' }}>
        Outline
      </h2>
      {slides.map((slide, index) => (
        <button
          key={index}
          onClick={() => onJumpToSlide(index)}
          style={{
            padding: '8px 12px',
            backgroundColor:
              index === currentIndex ? slideCss?.accentColor : 'transparent',
            color: index === currentIndex ? '#000' : slideCss?.color,
            border: `1px solid ${slideCss?.borderColor || '#666'}`,
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '12px',
            textAlign: 'left',
            transition: 'all 0.2s ease',
          }}
          onMouseEnter={(e) => {
            if (index !== currentIndex) {
              e.currentTarget.style.backgroundColor =
                slideCss?.borderColor || '#666';
            }
          }}
          onMouseLeave={(e) => {
            if (index !== currentIndex) {
              e.currentTarget.style.backgroundColor = 'transparent';
            }
          }}
          aria-label={`Go to slide ${index + 1}: ${slide.title}`}
          aria-current={index === currentIndex ? 'page' : undefined}
        >
          {slide.eyebrow}: {slide.title}
        </button>
      ))}
    </nav>
  );
}

/**
 * SlideRenderer component: Main slide content display.
 */
interface SlideRendererProps {
  slide: SlideDefinition;
  slideIndex: number;
  totalSlides: number;
  theme: ResolvedTheme;
  slideCss: SlideCss | null;
  isImmersive: boolean;
  immersiveScene: any | null;
  immersiveControls: any | null;
}

function SlideRenderer({
  slide,
  slideIndex,
  totalSlides,
  theme,
  slideCss,
  isImmersive,
  immersiveScene,
  immersiveControls,
}: SlideRendererProps): React.ReactElement {
  const slideContentStyle: React.CSSProperties = {
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
  };

  const eyebrowStyle: React.CSSProperties = {
    fontSize: '12px',
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    color: slideCss?.accentColor || '#72d6a2',
    margin: 0,
  };

  const titleStyle: React.CSSProperties = {
    fontSize: '32px',
    fontWeight: '700',
    margin: 0,
    lineHeight: 1.2,
  };

  const ledeStyle: React.CSSProperties = {
    fontSize: '16px',
    fontWeight: '400',
    color: slideCss?.color || '#f2eee3',
    margin: 0,
    opacity: 0.9,
  };

  const pointsStyle: React.CSSProperties = {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
    margin: '16px 0 0 0',
  };

  const pointItemStyle: React.CSSProperties = {
    fontSize: '14px',
    lineHeight: 1.6,
    paddingLeft: '16px',
    position: 'relative',
  };

  const pointBulletStyle: React.CSSProperties = {
    position: 'absolute',
    left: 0,
    color: slideCss?.accentColor || '#72d6a2',
    fontWeight: '700',
  };

  const termStyle: React.CSSProperties = {
    backgroundColor: slideCss?.borderColor || '#444',
    padding: '12px',
    borderRadius: '4px',
    marginTop: '16px',
  };

  const termWordStyle: React.CSSProperties = {
    fontSize: '14px',
    fontWeight: '700',
    color: slideCss?.accentColor || '#72d6a2',
    margin: '0 0 4px 0',
  };

  const termMeaningStyle: React.CSSProperties = {
    fontSize: '13px',
    color: slideCss?.color || '#f2eee3',
    margin: 0,
    opacity: 0.85,
  };

  const captionStyle: React.CSSProperties = {
    fontSize: '12px',
    fontStyle: 'italic',
    color: slideCss?.color || '#f2eee3',
    opacity: 0.7,
    marginTop: '8px',
  };

  return (
    <div style={slideContentStyle}>
      <div>
        <p style={eyebrowStyle}>{slide.eyebrow}</p>
        <h2 style={titleStyle}>{slide.title}</h2>
        <p style={ledeStyle}>{slide.lede}</p>
      </div>

      {slide.points && slide.points.length > 0 && (
        <div style={pointsStyle}>
          {slide.points.map((point, idx) => (
            <div key={idx} style={pointItemStyle}>
              <span style={pointBulletStyle}>•</span>
              {point}
            </div>
          ))}
        </div>
      )}

      {slide.term && (
        <div style={termStyle}>
          <p style={termWordStyle}>{slide.term.word}</p>
          <p style={termMeaningStyle}>{slide.term.meaning}</p>
        </div>
      )}

      {slide.caption && <p style={captionStyle}>{slide.caption}</p>}

      {isImmersive && immersiveScene && (
        <div
          style={{
            marginTop: '24px',
            padding: '16px',
            backgroundColor: slideCss?.borderColor || '#444',
            borderRadius: '4px',
            fontSize: '13px',
          }}
        >
          <strong>Immersive Mode:</strong> {immersiveScene.overlayContent.title}
        </div>
      )}
    </div>
  );
}

/**
 * BottomControls component: Navigation and playback buttons.
 */
interface BottomControlsProps {
  currentIndex: number;
  totalSlides: number;
  isPlaying: boolean;
  isPaused: boolean;
  onPrev: () => void;
  onPlayPause: () => void;
  onNext: () => void;
  onSkip: () => void;
  canGoPrev: boolean;
  canGoNext: boolean;
  theme: ResolvedTheme;
  slideCss: SlideCss | null;
}

function BottomControls({
  currentIndex,
  totalSlides,
  isPlaying,
  isPaused,
  onPrev,
  onPlayPause,
  onNext,
  onSkip,
  canGoPrev,
  canGoNext,
  theme,
  slideCss,
}: BottomControlsProps): React.ReactElement {
  const bottomStyle: React.CSSProperties = {
    padding: '16px 24px',
    borderTop: `1px solid ${slideCss?.borderColor || '#444'}`,
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    gap: '12px',
  };

  const buttonStyle = (disabled: boolean): React.CSSProperties => ({
    padding: '8px 16px',
    backgroundColor: disabled ? '#666' : slideCss?.accentColor || '#72d6a2',
    color: disabled ? '#999' : '#000',
    border: 'none',
    borderRadius: '4px',
    cursor: disabled ? 'not-allowed' : 'pointer',
    fontSize: '14px',
    fontWeight: '600',
    transition: 'all 0.2s ease',
    opacity: disabled ? 0.5 : 1,
  });

  return (
    <div style={bottomStyle}>
      <button
        onClick={onPrev}
        disabled={!canGoPrev}
        style={buttonStyle(!canGoPrev)}
        aria-label="Previous slide"
      >
        ← Prev
      </button>

      <button
        onClick={onPlayPause}
        style={buttonStyle(false)}
        aria-label={isPlaying ? 'Pause' : 'Play'}
      >
        {isPlaying ? (isPaused ? '▶ Play' : '⏸ Pause') : '▶ Play'}
      </button>

      <button
        onClick={onSkip}
        style={buttonStyle(!isPlaying)}
        disabled={!isPlaying}
        aria-label="Skip narration"
      >
        ⏭ Skip
      </button>

      <button
        onClick={onNext}
        disabled={!canGoNext}
        style={buttonStyle(!canGoNext)}
        aria-label="Next slide"
      >
        Next →
      </button>
    </div>
  );
}

export default ExplainerLayout;
