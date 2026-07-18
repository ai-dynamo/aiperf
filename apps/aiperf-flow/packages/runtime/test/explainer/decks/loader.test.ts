/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { ExplainerRegistry } from '../../../src/explainer/registry.js';
import { loadBuiltinDecks, AIPERF_FLOW_SYSTEM_DECK } from '../../../src/explainer/decks/loader.js';

describe('Explainer Decks Loader', () => {
  afterEach(() => {
    ExplainerRegistry.clear();
  });

  describe('AIPerf Flow System Deck', () => {
    it('has valid deck metadata', () => {
      const deck = AIPERF_FLOW_SYSTEM_DECK;
      expect(deck.id).toBe('aiperf-flow-system');
      expect(deck.route).toBe('/explainers/aiperf-flow-system');
      expect(deck.topic).toBe('aiperf-architecture');
      expect(deck.eyebrowLabel).toBe('AIPerf Flow System');
      expect(deck.startGateTitle).toBe('Explore Request Lifecycle');
    });

    it('has 9 slides', () => {
      const deck = AIPERF_FLOW_SYSTEM_DECK;
      expect(deck.slides.length).toBe(9);
    });

    it('all slides have required fields', () => {
      const deck = AIPERF_FLOW_SYSTEM_DECK;
      deck.slides.forEach((slide, index) => {
        expect(slide.eyebrow).toBeTruthy(`Slide ${index} missing eyebrow`);
        expect(slide.title).toBeTruthy(`Slide ${index} missing title`);
        expect(slide.lede).toBeTruthy(`Slide ${index} missing lede`);
        expect(slide.narration).toBeTruthy(`Slide ${index} missing narration`);
        expect(Array.isArray(slide.points)).toBe(true);
        expect(slide.caption).toBeTruthy(`Slide ${index} missing caption`);
      });
    });

    it('slides progress through AIPerf concepts', () => {
      const deck = AIPERF_FLOW_SYSTEM_DECK;
      const expectedSequence = [
        'Architecture',
        'Lifecycle',
        'Admission',
        'Transport',
        'Worker',
        'Stream',
        'Measurement',
        'Visualization',
        'Integration',
      ];

      deck.slides.forEach((slide, index) => {
        const titleLower = slide.title.toLowerCase();
        const expectedKeyword = expectedSequence[index];
        expect(titleLower).toContain(expectedKeyword.toLowerCase());
      });
    });

    it('includes terminology definitions', () => {
      const deck = AIPERF_FLOW_SYSTEM_DECK;
      const slidesWithTerms = deck.slides.filter(s => s.term);
      expect(slidesWithTerms.length).toBeGreaterThan(0);

      slidesWithTerms.forEach(slide => {
        expect(slide.term?.word).toBeTruthy();
        expect(slide.term?.meaning).toBeTruthy();
      });
    });
  });

  describe('Deck Loader', () => {
    it('loads built-in decks into registry', () => {
      loadBuiltinDecks(ExplainerRegistry);

      const deck = ExplainerRegistry.getDeck('aiperf-flow-system');
      expect(deck).toBeDefined();
      expect(deck?.id).toBe('aiperf-flow-system');
    });

    it('registers deck by route', () => {
      loadBuiltinDecks(ExplainerRegistry);

      const deck = ExplainerRegistry.getDeckByRoute('/explainers/aiperf-flow-system');
      expect(deck).toBeDefined();
      expect(deck?.id).toBe('aiperf-flow-system');
    });

    it('deck is queryable via getRouteMap', () => {
      loadBuiltinDecks(ExplainerRegistry);

      const routeMap = ExplainerRegistry.getRouteMap();
      expect(routeMap.get('/explainers/aiperf-flow-system')).toBe('aiperf-flow-system');
    });

    it('deck appears in getAllDecks', () => {
      loadBuiltinDecks(ExplainerRegistry);

      const allDecks = ExplainerRegistry.getAllDecks();
      expect(allDecks.length).toBeGreaterThan(0);
      expect(allDecks[0]?.id).toBe('aiperf-flow-system');
    });

    it('loader prevents duplicate registration', () => {
      loadBuiltinDecks(ExplainerRegistry);
      expect(() => loadBuiltinDecks(ExplainerRegistry)).toThrow(/duplicate|already/i);
    });
  });

  describe('Deck Content Fidelity', () => {
    beforeEach(() => {
      loadBuiltinDecks(ExplainerRegistry);
    });

    it('registered deck matches loader definition', () => {
      const registered = ExplainerRegistry.getDeck('aiperf-flow-system');
      const loader = AIPERF_FLOW_SYSTEM_DECK;

      expect(registered?.id).toBe(loader.id);
      expect(registered?.route).toBe(loader.route);
      expect(registered?.slides.length).toBe(loader.slides.length);
    });

    it('slide content is accurate after registration', () => {
      const deck = ExplainerRegistry.getDeck('aiperf-flow-system');
      if (!deck) {
        throw new Error('Deck not registered');
      }

      const expectedNarrationContent = [
        'load generator',
        'request passes through',
        'admission queue',
        'transport layer',
        'worker',
        'observer',
        'measurement',
        'visualization',
        'measurement and visualization',
      ];

      deck.slides.forEach((slide, index) => {
        expect(slide.narration.toLowerCase()).toContain(
          expectedNarrationContent[index]!.toLowerCase()
        );
      });
    });

    it('narration covers key AIPerf concepts', () => {
      const deck = ExplainerRegistry.getDeck('aiperf-flow-system');
      if (!deck) {
        throw new Error('Deck not registered');
      }

      const allNarration = deck.slides.map(s => s.narration).join(' ');
      const keyTerms = [
        'clock',
        'request',
        'evidence',
        'admission',
        'transport',
        'worker',
        'observer',
        'measurement',
      ];

      keyTerms.forEach(term => {
        expect(allNarration.toLowerCase()).toContain(term.toLowerCase());
      });
    });

    it('slides provide educational value', () => {
      const deck = ExplainerRegistry.getDeck('aiperf-flow-system');
      if (!deck) {
        throw new Error('Deck not registered');
      }

      deck.slides.forEach((slide, index) => {
        // Each slide should have substantive narration
        expect(slide.narration.trim().length).toBeGreaterThan(100);

        // Each slide should have bullet points
        expect(slide.points.length).toBeGreaterThan(0);

        // Captions should be concise summaries
        expect(slide.caption.trim().length).toBeGreaterThan(10);
      });
    });
  });

  describe('Accessibility', () => {
    beforeEach(() => {
      loadBuiltinDecks(ExplainerRegistry);
    });

    it('all slides have narration for audio accessibility', () => {
      const deck = ExplainerRegistry.getDeck('aiperf-flow-system');
      if (!deck) {
        throw new Error('Deck not registered');
      }

      deck.slides.forEach((slide, index) => {
        expect(slide.narration).toBeTruthy();
        expect(slide.narration.trim().length).toBeGreaterThan(0);
      });
    });

    it('some slides include terminology definitions', () => {
      const deck = ExplainerRegistry.getDeck('aiperf-flow-system');
      if (!deck) {
        throw new Error('Deck not registered');
      }

      const slidesWithTerms = deck.slides.filter(s => s.term);
      expect(slidesWithTerms.length).toBeGreaterThan(0);
    });

    it('deck is keyboard navigable (has slide structure)', () => {
      const deck = ExplainerRegistry.getDeck('aiperf-flow-system');
      if (!deck) {
        throw new Error('Deck not registered');
      }

      // Each slide should have the structure to support keyboard navigation
      expect(deck.slides.length).toBeGreaterThan(0);
      deck.slides.forEach((slide, index) => {
        expect(slide.title).toBeTruthy(`Slide ${index} missing title for navigation`);
      });
    });
  });
});
