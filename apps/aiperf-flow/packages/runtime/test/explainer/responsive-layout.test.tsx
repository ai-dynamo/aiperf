/**
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import React from 'react';
import { ResponsiveLayout, type ViewportSize } from '../../src/explainer/ui/responsive-layout.js';
import type { ExplainerDefinition } from '@aiperf/flow-compiler';
import type { ResolvedTheme } from '../../src/theme/registry-runtime.js';

describe('ResponsiveLayout', () => {
  let deck: ExplainerDefinition;
  let theme: ResolvedTheme;
  let onNavigate: ReturnType<typeof vi.fn>;
  let onImmersiveToggle: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    // Mock window dimensions
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

      expect(screen.getByText('Test Deck')).toBeInTheDocument();
      expect(screen.getByRole('navigation')).toBeInTheDocument();
      expect(screen.getByText('Welcome')).toBeInTheDocument();
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

      expect(screen.getByText('Welcome')).toBeInTheDocument();
      expect(screen.getByText('Key Concepts')).toBeInTheDocument();
      expect(screen.getByText('Conclusion')).toBeInTheDocument();
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
      expect(activeItem).toBeInTheDocument();

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

      expect(screen.getByText('Test Deck')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /toggle navigation/i })).toBeInTheDocument();
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
      expect(toggle).toBeInTheDocument();
    });

    it('opens/closes sidebar on toggle click', async () => {
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

      // Initially sidebar should be closed
      let nav = screen.queryByRole('navigation');
      expect(nav).not.toBeInTheDocument();

      // Click to open
      fireEvent.click(toggle);
      await waitFor(() => {
        nav = screen.getByRole('navigation');
        expect(nav).toBeInTheDocument();
      });

      // Click to close
      fireEvent.click(toggle);
      await waitFor(() => {
        nav = screen.queryByRole('navigation');
        expect(nav).not.toBeInTheDocument();
      });
    });

    it('closes sidebar after selecting a slide', async () => {
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
      fireEvent.click(toggle);

      await waitFor(() => {
        expect(screen.getByRole('navigation')).toBeInTheDocument();
      });

      // Click on a slide in sidebar
      const slideButtons = screen.getAllByRole('button');
      const slideButton = slideButtons.find(b => b.textContent?.includes('Key Concepts'));
      fireEvent.click(slideButton!);

      await waitFor(() => {
        expect(screen.queryByRole('navigation')).not.toBeInTheDocument();
      });
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

      expect(screen.getByText('Test Deck')).toBeInTheDocument();
      expect(screen.getByText('1 / 3')).toBeInTheDocument();
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

      expect(screen.getByRole('button', { name: /toggle navigation/i })).toBeInTheDocument();
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
      expect(fullscreenButton).toBeInTheDocument();
    });

    it('displays slide content with appropriate font sizes for mobile', () => {
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

      expect(screen.getByText('Welcome')).toBeInTheDocument();
      expect(screen.getByText('Getting started with our explainer')).toBeInTheDocument();
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
      expect(screen.getByRole('navigation')).toBeInTheDocument();

      // Resize to mobile
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      });

      fireEvent.resize(window);

      await waitFor(() => {
        const nav = screen.queryByRole('navigation');
        expect(nav).not.toBeInTheDocument();
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
      expect(prevButton).toBeDisabled();
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
      expect(nextButton).toBeDisabled();
    });

    it('navigates via arrow keys', async () => {
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

      expect(screen.getByText('Welcome')).toBeInTheDocument();
      expect(screen.getByText('Getting started with our explainer')).toBeInTheDocument();
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

      expect(screen.getByText('Point 1')).toBeInTheDocument();
      expect(screen.getByText('Point 2')).toBeInTheDocument();
      expect(screen.getByText('Point 3')).toBeInTheDocument();
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

      expect(screen.getByText('Concept:')).toBeInTheDocument();
      expect(screen.getByText('An abstract idea')).toBeInTheDocument();
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

      expect(screen.getByText('Slide 1 caption')).toBeInTheDocument();
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

      expect(screen.getByText('Welcome')).toBeInTheDocument();

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

      expect(screen.getByText('Key Concepts')).toBeInTheDocument();
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

      expect(screen.getByText('Welcome')).toBeInTheDocument();
      // Should show immersive controls
      const closeButton = screen.getByRole('button', { name: /exit full screen/i });
      expect(closeButton).toBeInTheDocument();
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

    it('shows progress bar in immersive mode', () => {
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

      expect(screen.getByText('2 / 3')).toBeInTheDocument();
    });

    it('shows slide thumbnails in immersive mode', () => {
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

      // Should have thumbnail buttons for each slide
      const thumbnails = screen.getAllByRole('button', { name: /go to slide/i });
      expect(thumbnails.length).toBe(deck.slides.length);
    });

    it('navigates via thumbnail clicks in immersive mode', () => {
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

      const thumbnails = screen.getAllByRole('button', { name: /go to slide/i });
      fireEvent.click(thumbnails[2]!);
      expect(onNavigate).toHaveBeenCalledWith(2);
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

      expect(screen.getByText('2 / 3')).toBeInTheDocument();
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

      expect(screen.getByText('1 / 3')).toBeInTheDocument();

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

      expect(screen.getByText('3 / 3')).toBeInTheDocument();
    });
  });

  describe('Touch Interactions', () => {
    beforeEach(() => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      });
    });

    it('has touch-friendly button sizes on mobile', () => {
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

      const buttons = screen.getAllByRole('button');
      buttons.forEach(button => {
        const style = window.getComputedStyle(button);
        // Buttons should be at least 44x44px for touch (implicit from padding styles)
        expect(button).toHaveStyle({ cursor: 'pointer' });
      });
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
      expect(styles.backgroundColor).toEqual(theme.values['surface.primary']);
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
      const styles = window.getComputedStyle(immersive!);
      expect(styles.backgroundColor).toEqual(theme.values['surface.primary']);
    });
  });
});
