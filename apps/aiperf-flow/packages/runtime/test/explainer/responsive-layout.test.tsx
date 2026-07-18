/**
 * SPDX-License-Identifier: Apache-2.0
 */

// @vitest-environment jsdom

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';
import React from 'react';
import { ResponsiveLayout } from '../../src/explainer/ui/responsive-layout.js';
import type { ExplainerDefinition } from '@aiperf/flow-compiler';
import type { ResolvedTheme } from '../../src/theme/registry-runtime.js';

describe('ResponsiveLayout', () => {
  let deck: ExplainerDefinition;
  let theme: ResolvedTheme;
  let onNavigate: ReturnType<typeof vi.fn>;
  let onImmersiveToggle: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 1024,
    });
    Object.defineProperty(window, 'innerHeight', {
      writable: true,
      configurable: true,
      value: 768,
    });

    deck = {
      id: 'test',
      route: '/test',
      topic: 'intro',
      eyebrowLabel: 'Test Deck',
      startGateTitle: 'Begin?',
      slides: [
        {
          eyebrow: 'Intro',
          title: 'Welcome',
          lede: 'Getting started with our explainer',
          narration: 'Welcome to the explainer.',
          points: ['Point 1', 'Point 2', 'Point 3'],
          caption: 'Slide 1 caption',
        },
        {
          eyebrow: 'Details',
          title: 'Key Concepts',
          lede: 'Understanding the fundamentals',
          narration: 'Here are the key concepts.',
          term: { word: 'Concept', meaning: 'An abstract idea' },
          points: ['Detail A', 'Detail B'],
          caption: 'Slide 2 caption',
        },
        {
          eyebrow: 'Summary',
          title: 'Conclusion',
          lede: 'What we learned',
          narration: 'Thanks for learning with us.',
          points: [],
          caption: 'Final slide',
        },
      ],
      scenesById: new Map(),
    };

    theme = {
      id: 'systems-chalk',
      values: {
        'surface.primary': '#24282b',
        'surface.secondary': '#2e3236',
        'ink.primary': '#f2eee3',
        'accent.execute': '#72d6a2',
        'structure.divider': '#3d4147',
      } as any,
    };

    onNavigate = vi.fn();
    onImmersiveToggle = vi.fn();
  });

  afterEach(() => {
    vi.clearAllMocks();
    cleanup();
  });

  describe('Desktop Layout (>= 1024px)', () => {
    beforeEach(() => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 1400,
      });
    });

    it('renders desktop layout with sidebar visible', () => {
      render(
        <ResponsiveLayout
          deck={deck}
          currentSlide={deck.slides[0]!}
          slideIndex={0}
          totalSlides={deck.slides.length}
          onNavigate={onNavigate}
          theme={theme}
          onImmersiveToggle={onImmersiveToggle}
        />
      );

      expect(screen.getByText('Test Deck')).toBeTruthy();
      expect(screen.getByRole('navigation')).toBeTruthy();
      expect(screen.getByRole('heading', { name: 'Welcome' })).toBeTruthy();
    });

    it('displays all slides in sidebar navigation', () => {
      render(
        <ResponsiveLayout
          deck={deck}
          currentSlide={deck.slides[0]!}
          slideIndex={0}
          totalSlides={deck.slides.length}
          onNavigate={onNavigate}
          theme={theme}
          onImmersiveToggle={onImmersiveToggle}
        />
      );

      const nav = screen.getByRole('navigation');
      expect(nav.textContent).toContain('Welcome');
      expect(nav.textContent).toContain('Key Concepts');
      expect(nav.textContent).toContain('Conclusion');
    });

    it('highlights active slide in sidebar', () => {
      const { rerender } = render(
        <ResponsiveLayout
          deck={deck}
          currentSlide={deck.slides[0]!}
          slideIndex={0}
          totalSlides={deck.slides.length}
          onNavigate={onNavigate}
          theme={theme}
          onImmersiveToggle={onImmersiveToggle}
        />
      );

      let nav = screen.getByRole('navigation');
      let activeItem = nav.querySelector('.slide-nav-item.active');
      expect(activeItem).toBeTruthy();

      rerender(
        <ResponsiveLayout
          deck={deck}
          currentSlide={deck.slides[1]!}
          slideIndex={1}
          totalSlides={deck.slides.length}
          onNavigate={onNavigate}
          theme={theme}
          onImmersiveToggle={onImmersiveToggle}
        />
      );

      nav = screen.getByRole('navigation');
      activeItem = nav.querySelector('.slide-nav-item.active');
      expect(activeItem?.textContent).toContain('Key Concepts');
    });

    it('shows full screen button in desktop layout', () => {
      render(
        <ResponsiveLayout
          deck={deck}
          currentSlide={deck.slides[0]!}
          slideIndex={0}
          totalSlides={deck.slides.length}
          onNavigate={onNavigate}
          theme={theme}
          onImmersiveToggle={onImmersiveToggle}
        />
      );

      const fullscreenButtons = screen.getAllByText('Full Screen');
      expect(fullscreenButtons.length).toBeGreaterThan(0);
    });
  });

  describe('Tablet Layout (640px - 1024px)', () => {
    beforeEach(() => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 768,
      });
    });

    it('renders tablet layout with hidden sidebar', () => {
      render(
        <ResponsiveLayout
          deck={deck}
          currentSlide={deck.slides[0]!}
          slideIndex={0}
          totalSlides={deck.slides.length}
          onNavigate={onNavigate}
          theme={theme}
          onImmersiveToggle={onImmersiveToggle}
        />
      );

      expect(screen.getByText('Test Deck')).toBeTruthy();
      expect(screen.getByRole('button', { name: /toggle navigation/i })).toBeTruthy();
    });

    it('shows sidebar toggle button on tablet', () => {
      render(
        <ResponsiveLayout
          deck={deck}
          currentSlide={deck.slides[0]!}
          slideIndex={0}
          totalSlides={deck.slides.length}
          onNavigate={onNavigate}
          theme={theme}
          onImmersiveToggle={onImmersiveToggle}
        />
      );

      const toggle = screen.getByRole('button', { name: /toggle navigation/i });
      expect(toggle).toBeTruthy();
    });

    it('shows toggle button for sidebar on tablet', () => {
      render(
        <ResponsiveLayout
          deck={deck}
          currentSlide={deck.slides[0]!}
          slideIndex={0}
          totalSlides={deck.slides.length}
          onNavigate={onNavigate}
          theme={theme}
          onImmersiveToggle={onImmersiveToggle}
        />
      );

      const toggle = screen.getByRole('button', { name: /toggle navigation/i });
      expect(toggle).toBeTruthy();
    });
  });

  describe('Mobile Layout (< 640px)', () => {
    beforeEach(() => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      });
    });

    it('renders mobile layout with stacked content', () => {
      render(
        <ResponsiveLayout
          deck={deck}
          currentSlide={deck.slides[0]!}
          slideIndex={0}
          totalSlides={deck.slides.length}
          onNavigate={onNavigate}
          theme={theme}
          onImmersiveToggle={onImmersiveToggle}
        />
      );

      expect(screen.getByText('Test Deck')).toBeTruthy();
      expect(screen.getByText('1 / 3')).toBeTruthy();
    });

    it('shows sidebar toggle on mobile', () => {
      render(
        <ResponsiveLayout
          deck={deck}
          currentSlide={deck.slides[0]!}
          slideIndex={0}
          totalSlides={deck.slides.length}
          onNavigate={onNavigate}
          theme={theme}
          onImmersiveToggle={onImmersiveToggle}
        />
      );

      expect(screen.getByRole('button', { name: /toggle navigation/i })).toBeTruthy();
    });

    it('shows full screen button in mobile layout', () => {
      render(
        <ResponsiveLayout
          deck={deck}
          currentSlide={deck.slides[0]!}
          slideIndex={0}
          totalSlides={deck.slides.length}
          onNavigate={onNavigate}
          theme={theme}
          onImmersiveToggle={onImmersiveToggle}
        />
      );

      const fullscreenButton = screen.getByText('Full Screen');
      expect(fullscreenButton).toBeTruthy();
    });

    it('displays slide content with appropriate layout for mobile', () => {
      render(
        <ResponsiveLayout
          deck={deck}
          currentSlide={deck.slides[0]!}
          slideIndex={0}
          totalSlides={deck.slides.length}
          onNavigate={onNavigate}
          theme={theme}
          onImmersiveToggle={onImmersiveToggle}
        />
      );

      expect(screen.getByRole('heading', { name: 'Welcome' })).toBeTruthy();
      expect(screen.getByText('Getting started with our explainer')).toBeTruthy();
    });
  });

  describe('Responsive Breakpoints', () => {
    it('transitions layout on window resize', async () => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 1400,
      });

      render(
        <ResponsiveLayout
          deck={deck}
          currentSlide={deck.slides[0]!}
          slideIndex={0}
          totalSlides={deck.slides.length}
          onNavigate={onNavigate}
          theme={theme}
          onImmersiveToggle={onImmersiveToggle}
        />
      );

      // Initially desktop
      expect(screen.getByRole('navigation')).toBeTruthy();

      // Resize to mobile
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      });

      fireEvent.resize(window);

      await waitFor(() => {
        const nav = screen.queryByRole('navigation');
        expect(nav).toBeNull();
      });
    });
  });

  describe('Navigation Controls', () => {
    beforeEach(() => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 1024,
      });
    });

    it('navigates to previous slide', () => {
      render(
        <ResponsiveLayout
          deck={deck}
          currentSlide={deck.slides[1]!}
          slideIndex={1}
          totalSlides={deck.slides.length}
          onNavigate={onNavigate}
          theme={theme}
          onImmersiveToggle={onImmersiveToggle}
        />
      );

      const prevButton = screen.getByRole('button', { name: /previous slide/i });
      fireEvent.click(prevButton);

      expect(onNavigate).toHaveBeenCalledWith(0);
    });

    it('navigates to next slide', () => {
      render(
        <ResponsiveLayout
          deck={deck}
          currentSlide={deck.slides[0]!}
          slideIndex={0}
          totalSlides={deck.slides.length}
          onNavigate={onNavigate}
          theme={theme}
          onImmersiveToggle={onImmersiveToggle}
        />
      );

      const nextButton = screen.getByRole('button', { name: /next slide/i });
      fireEvent.click(nextButton);

      expect(onNavigate).toHaveBeenCalledWith(1);
    });

    it('disables prev button on first slide', () => {
      render(
        <ResponsiveLayout
          deck={deck}
          currentSlide={deck.slides[0]!}
          slideIndex={0}
          totalSlides={deck.slides.length}
          onNavigate={onNavigate}
          theme={theme}
          onImmersiveToggle={onImmersiveToggle}
        />
      );

      const prevButton = screen.getByRole('button', { name: /previous slide/i });
      expect((prevButton as HTMLButtonElement).disabled).toBe(true);
    });

    it('disables next button on last slide', () => {
      render(
        <ResponsiveLayout
          deck={deck}
          currentSlide={deck.slides[2]!}
          slideIndex={2}
          totalSlides={deck.slides.length}
          onNavigate={onNavigate}
          theme={theme}
          onImmersiveToggle={onImmersiveToggle}
        />
      );

      const nextButton = screen.getByRole('button', { name: /next slide/i });
      expect((nextButton as HTMLButtonElement).disabled).toBe(true);
    });

    it('navigates via arrow keys', () => {
      render(
        <ResponsiveLayout
          deck={deck}
          currentSlide={deck.slides[1]!}
          slideIndex={1}
          totalSlides={deck.slides.length}
          onNavigate={onNavigate}
          theme={theme}
          onImmersiveToggle={onImmersiveToggle}
        />
      );

      fireEvent.keyDown(window, { key: 'ArrowRight' });
      expect(onNavigate).toHaveBeenCalledWith(2);

      onNavigate.mockClear();

      fireEvent.keyDown(window, { key: 'ArrowLeft' });
      expect(onNavigate).toHaveBeenCalledWith(0);
    });
  });

  describe('Slide Content Display', () => {
    beforeEach(() => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 1024,
      });
    });

    it('displays slide title and lede', () => {
      render(
        <ResponsiveLayout
          deck={deck}
          currentSlide={deck.slides[0]!}
          slideIndex={0}
          totalSlides={deck.slides.length}
          onNavigate={onNavigate}
          theme={theme}
          onImmersiveToggle={onImmersiveToggle}
        />
      );

      expect(screen.getByRole('heading', { name: 'Welcome' })).toBeTruthy();
      expect(screen.getByText('Getting started with our explainer')).toBeTruthy();
    });

    it('displays bullet points', () => {
      render(
        <ResponsiveLayout
          deck={deck}
          currentSlide={deck.slides[0]!}
          slideIndex={0}
          totalSlides={deck.slides.length}
          onNavigate={onNavigate}
          theme={theme}
          onImmersiveToggle={onImmersiveToggle}
        />
      );

      expect(screen.getByText('Point 1')).toBeTruthy();
      expect(screen.getByText('Point 2')).toBeTruthy();
      expect(screen.getByText('Point 3')).toBeTruthy();
    });

    it('displays term definition when present', () => {
      render(
        <ResponsiveLayout
          deck={deck}
          currentSlide={deck.slides[1]!}
          slideIndex={1}
          totalSlides={deck.slides.length}
          onNavigate={onNavigate}
          theme={theme}
          onImmersiveToggle={onImmersiveToggle}
        />
      );

      expect(screen.getByText('Concept:')).toBeTruthy();
      expect(screen.getByText('An abstract idea')).toBeTruthy();
    });

    it('displays caption', () => {
      render(
        <ResponsiveLayout
          deck={deck}
          currentSlide={deck.slides[0]!}
          slideIndex={0}
          totalSlides={deck.slides.length}
          onNavigate={onNavigate}
          theme={theme}
          onImmersiveToggle={onImmersiveToggle}
        />
      );

      expect(screen.getByText('Slide 1 caption')).toBeTruthy();
    });

    it('updates content when slide index changes', () => {
      const { rerender } = render(
        <ResponsiveLayout
          deck={deck}
          currentSlide={deck.slides[0]!}
          slideIndex={0}
          totalSlides={deck.slides.length}
          onNavigate={onNavigate}
          theme={theme}
          onImmersiveToggle={onImmersiveToggle}
        />
      );

      expect(screen.getByRole('heading', { name: 'Welcome' })).toBeTruthy();

      rerender(
        <ResponsiveLayout
          deck={deck}
          currentSlide={deck.slides[1]!}
          slideIndex={1}
          totalSlides={deck.slides.length}
          onNavigate={onNavigate}
          theme={theme}
          onImmersiveToggle={onImmersiveToggle}
        />
      );

      expect(screen.getByRole('heading', { name: 'Key Concepts' })).toBeTruthy();
    });
  });

  describe('Immersive Mode', () => {
    beforeEach(() => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 1024,
      });
    });

    it('renders immersive layout when isImmersive is true', () => {
      render(
        <ResponsiveLayout
          deck={deck}
          currentSlide={deck.slides[0]!}
          slideIndex={0}
          totalSlides={deck.slides.length}
          onNavigate={onNavigate}
          theme={theme}
          isImmersive={true}
          onImmersiveToggle={onImmersiveToggle}
        />
      );

      expect(screen.getByRole('heading', { name: 'Welcome' })).toBeTruthy();
      const closeButton = screen.getByRole('button', { name: /exit full screen/i });
      expect(closeButton).toBeTruthy();
    });

    it('shows close button in immersive mode', () => {
      render(
        <ResponsiveLayout
          deck={deck}
          currentSlide={deck.slides[0]!}
          slideIndex={0}
          totalSlides={deck.slides.length}
          onNavigate={onNavigate}
          theme={theme}
          isImmersive={true}
          onImmersiveToggle={onImmersiveToggle}
        />
      );

      const closeButton = screen.getByRole('button', { name: /exit full screen/i });
      fireEvent.click(closeButton);
      expect(onImmersiveToggle).toHaveBeenCalledWith(false);
    });

    it('shows progress indicator in immersive mode', () => {
      render(
        <ResponsiveLayout
          deck={deck}
          currentSlide={deck.slides[1]!}
          slideIndex={1}
          totalSlides={deck.slides.length}
          onNavigate={onNavigate}
          theme={theme}
          isImmersive={true}
          onImmersiveToggle={onImmersiveToggle}
        />
      );

      expect(screen.getByText('2 / 3')).toBeTruthy();
    });

    it('closes immersive mode with Escape key', () => {
      render(
        <ResponsiveLayout
          deck={deck}
          currentSlide={deck.slides[0]!}
          slideIndex={0}
          totalSlides={deck.slides.length}
          onNavigate={onNavigate}
          theme={theme}
          isImmersive={true}
          onImmersiveToggle={onImmersiveToggle}
        />
      );

      fireEvent.keyDown(window, { key: 'Escape' });
      expect(onImmersiveToggle).toHaveBeenCalledWith(false);
    });
  });

  describe('Progress Indicator', () => {
    beforeEach(() => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 1024,
      });
    });

    it('displays current slide position', () => {
      render(
        <ResponsiveLayout
          deck={deck}
          currentSlide={deck.slides[1]!}
          slideIndex={1}
          totalSlides={deck.slides.length}
          onNavigate={onNavigate}
          theme={theme}
          onImmersiveToggle={onImmersiveToggle}
        />
      );

      expect(screen.getByText('2 / 3')).toBeTruthy();
    });

    it('updates progress indicator on slide change', () => {
      const { rerender } = render(
        <ResponsiveLayout
          deck={deck}
          currentSlide={deck.slides[0]!}
          slideIndex={0}
          totalSlides={deck.slides.length}
          onNavigate={onNavigate}
          theme={theme}
          onImmersiveToggle={onImmersiveToggle}
        />
      );

      expect(screen.getByText('1 / 3')).toBeTruthy();

      rerender(
        <ResponsiveLayout
          deck={deck}
          currentSlide={deck.slides[2]!}
          slideIndex={2}
          totalSlides={deck.slides.length}
          onNavigate={onNavigate}
          theme={theme}
          onImmersiveToggle={onImmersiveToggle}
        />
      );

      expect(screen.getByText('3 / 3')).toBeTruthy();
    });
  });

  describe('Theme Application', () => {
    beforeEach(() => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 1024,
      });
    });

    it('applies theme colors to layout', () => {
      const { container } = render(
        <ResponsiveLayout
          deck={deck}
          currentSlide={deck.slides[0]!}
          slideIndex={0}
          totalSlides={deck.slides.length}
          onNavigate={onNavigate}
          theme={theme}
          onImmersiveToggle={onImmersiveToggle}
        />
      );

      const layout = container.querySelector('.explainer-responsive-layout');
      const styles = window.getComputedStyle(layout!);
      expect(styles.backgroundColor).toBeTruthy();
    });

    it('uses theme colors in immersive mode', () => {
      const { container } = render(
        <ResponsiveLayout
          deck={deck}
          currentSlide={deck.slides[0]!}
          slideIndex={0}
          totalSlides={deck.slides.length}
          onNavigate={onNavigate}
          theme={theme}
          isImmersive={true}
          onImmersiveToggle={onImmersiveToggle}
        />
      );

      const immersive = container.querySelector('.explainer-immersive');
      expect(immersive).toBeTruthy();
    });
  });
});
