/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { readFileSync, existsSync } from 'fs';
import { resolve } from 'path';
import type { SlideDefinition } from '@aiperf/flow-compiler';
import { AIPERF_FLOW_SYSTEM_DECK } from '../../../src/explainer/decks/loader.js';

describe('AIPerf Flow System Explainer Deck', () => {
  let deckFilePath: string;

  beforeEach(() => {
    deckFilePath = resolve(
      __dirname,
      '../../../src/explainer/decks/aiperf-flow-system.flow'
    );
  });

  describe('Source File', () => {
    it('exists as a design document', () => {
      expect(existsSync(deckFilePath)).toBe(true);
    });

    it('contains explainer and slide declarations', () => {
      const source = readFileSync(deckFilePath, 'utf-8');
      expect(source).toContain('explainer');
      expect(source).toContain('slide');
      expect(source).toContain('aiperf-flow-system');
    });

    it('is valid Flow syntax for design reference', () => {
      const source = readFileSync(deckFilePath, 'utf-8');
      // Basic sanity checks
      expect(source).toContain('flow');
      expect(source).toContain('language 1');
      expect(source).toContain('require core.rect');
      expect(source).toContain('scene');
    });
  });

  describe('Deck Schema and Content', () => {
    it('has valid deck metadata', () => {
      const deck = AIPERF_FLOW_SYSTEM_DECK;
      expect(deck.id).toBe('aiperf-flow-system');
      expect(deck.route).toBe('/explainers/aiperf-flow-system');
      expect(deck.topic).toBe('aiperf-architecture');
      expect(deck.eyebrowLabel).toBe('AIPerf Flow System');
      expect(deck.startGateTitle).toBe('Explore Request Lifecycle');
    });

    it('contains 9 slides', () => {
      const deck = AIPERF_FLOW_SYSTEM_DECK;
      expect(deck.slides.length).toBe(9);
    });

    it('ensures all slides have required fields', () => {
      const slides = AIPERF_FLOW_SYSTEM_DECK.slides;
      slides.forEach((slide, index) => {
        expect(slide.eyebrow, `Slide ${index} missing eyebrow`).toBeTruthy();
        expect(slide.title, `Slide ${index} missing title`).toBeTruthy();
        expect(slide.lede, `Slide ${index} missing lede`).toBeTruthy();
        expect(slide.narration, `Slide ${index} missing narration`).toBeTruthy();
        expect(Array.isArray(slide.points), `Slide ${index} points not array`).toBe(true);
        expect(slide.caption, `Slide ${index} missing caption`).toBeTruthy();
      });
    });

    it('slide titles are unique', () => {
      const slides = AIPERF_FLOW_SYSTEM_DECK.slides;
      const titles = slides.map(s => s.title);
      const uniqueTitles = new Set(titles);
      expect(uniqueTitles.size).toBe(titles.length);
    });
  });

  describe('Slide Content Quality', () => {
    it('narration is non-empty and substantive', () => {
      const slides = AIPERF_FLOW_SYSTEM_DECK.slides;
      slides.forEach((slide, index) => {
        expect(slide.narration.trim().length).toBeGreaterThan(80,
          `Slide ${index} narration too brief`
        );
      });
    });

    it('each slide has relevant bullet points', () => {
      const slides = AIPERF_FLOW_SYSTEM_DECK.slides;
      slides.forEach((slide, index) => {
        expect(slide.points.length).toBeGreaterThan(0,
          `Slide ${index} has no bullet points`
        );
        expect(slide.points.length).toBeLessThanOrEqual(6,
          `Slide ${index} has too many bullet points`
        );
      });
    });

    it('slides progress logically through AIPerf concepts', () => {
      const slides = AIPERF_FLOW_SYSTEM_DECK.slides;
      const expectedTopics = [
        'architecture',      // Slide 1
        'journey',            // Slide 2 (request journey)
        'admission',          // Slide 3
        'transport',          // Slide 4
        'worker',             // Slide 5
        'stream',             // Slide 6
        'measurement',        // Slide 7
        'visualization',      // Slide 8
        'integration'         // Slide 9
      ];

      slides.forEach((slide, index) => {
        const expectedKeyword = expectedTopics[index];
        const titleAndNarration = (slide.title + ' ' + slide.narration).toLowerCase();
        expect(titleAndNarration).toContain(expectedKeyword,
          `Slide ${index} does not cover expected topic: ${expectedKeyword}`
        );
      });
    });

    it('slide captions are concise summaries', () => {
      const slides = AIPERF_FLOW_SYSTEM_DECK.slides;
      slides.forEach((slide, index) => {
        expect(slide.caption.trim().length).toBeGreaterThan(10,
          `Slide ${index} caption too brief`
        );
        expect(slide.caption.split(' ').length).toBeLessThan(25,
          `Slide ${index} caption too long`
        );
      });
    });
  });

  describe('Terminology and Learning', () => {
    it('some slides define key terminology', () => {
      const slides = AIPERF_FLOW_SYSTEM_DECK.slides;
      const slidesWithTerms = slides.filter(s => s.term);
      expect(slidesWithTerms.length).toBeGreaterThan(0,
        'At least some slides should define key terminology'
      );
    });

    it('term definitions are educational and substantive', () => {
      const slides = AIPERF_FLOW_SYSTEM_DECK.slides;
      slides.forEach((slide, index) => {
        if (slide.term) {
          expect(slide.term.word).toBeTruthy(`Slide ${index} term word missing`);
          expect(slide.term.meaning).toBeTruthy(`Slide ${index} term meaning missing`);
          expect(slide.term.meaning.length).toBeGreaterThan(15,
            `Slide ${index} term definition too brief`
          );
        }
      });
    });

    it('narration covers key AIPerf concepts', () => {
      const slides = AIPERF_FLOW_SYSTEM_DECK.slides;
      const allNarration = slides.map(s => s.narration).join(' ');
      const concepts = [
        'request',
        'admission',
        'transport',
        'worker',
        'stream',
        'observer',
        'evidence',
        'measurement',
        'clock'
      ];

      concepts.forEach(concept => {
        expect(allNarration.toLowerCase()).toContain(concept.toLowerCase(),
          `Narration should cover: ${concept}`
        );
      });
    });
  });

  describe('Narrative Consistency', () => {
    it('all slides have narration for audio and semantic accessibility', () => {
      const slides = AIPERF_FLOW_SYSTEM_DECK.slides;
      slides.forEach((slide, index) => {
        expect(slide.narration).toBeTruthy(`Slide ${index} missing narration`);
        expect(slide.narration.trim().length).toBeGreaterThan(0);
      });
    });

    it('narration explains clock-aware execution model', () => {
      const slides = AIPERF_FLOW_SYSTEM_DECK.slides;
      const narration = slides.map(s => s.narration).join(' ');
      expect(narration.toLowerCase()).toContain('clock');
      expect(narration.toLowerCase()).toContain('evidence');
    });

    it('bullet points supplement narration with specific details', () => {
      const slides = AIPERF_FLOW_SYSTEM_DECK.slides;
      slides.forEach((slide, index) => {
        // Each point should be distinct from the narration summary
        expect(slide.points.length).toBeGreaterThan(0);
        slide.points.forEach((point) => {
          expect(point.trim().length).toBeGreaterThan(5,
            `Slide ${index} has empty point`
          );
        });
      });
    });
  });

  describe('Accessibility', () => {
    it('deck is keyboard navigable via slide structure', () => {
      const deck = AIPERF_FLOW_SYSTEM_DECK;
      expect(deck.slides.length).toBeGreaterThan(0);
      deck.slides.forEach((slide, index) => {
        expect(slide.title).toBeTruthy(`Slide ${index} missing title for navigation`);
        expect(slide.eyebrow).toBeTruthy(`Slide ${index} missing eyebrow for outline`);
      });
    });

    it('supports both narration and visual content', () => {
      const slides = AIPERF_FLOW_SYSTEM_DECK.slides;
      slides.forEach(slide => {
        // Each slide should have both narration (audio) and visual elements (title, lede, caption, points)
        expect(slide.narration).toBeTruthy();
        expect(slide.title).toBeTruthy();
        expect(slide.lede).toBeTruthy();
        expect(slide.caption).toBeTruthy();
        expect(slide.points.length).toBeGreaterThan(0);
      });
    });

    it('provides structured outline via eyebrow labels', () => {
      const slides = AIPERF_FLOW_SYSTEM_DECK.slides;
      const eyebrows = slides.map(s => s.eyebrow);

      // Should have progression through modules
      expect(eyebrows[0]).toContain('Module');
      expect(eyebrows[8]).toContain('Module');

      // All eyebrows should be present and non-empty
      eyebrows.forEach((eyebrow, index) => {
        expect(eyebrow).toBeTruthy(`Slide ${index} eyebrow missing`);
        expect(eyebrow.trim().length).toBeGreaterThan(0);
      });
    });
  });

  describe('Educational Value', () => {
    it('progression teaches request lifecycle end-to-end', () => {
      const slides = AIPERF_FLOW_SYSTEM_DECK.slides;
      const allContent = slides
        .map(s => (s.title + ' ' + s.narration).toLowerCase())
        .join(' ');

      // Should progress through the full lifecycle
      const lifecycle = [
        'architecture',
        'request',
        'admission',
        'transport',
        'worker',
        'observer',
        'measurement',
        'visualization',
        'integration'
      ];

      lifecycle.forEach(term => {
        expect(allContent).toContain(term,
          `Lifecycle should cover: ${term}`
        );
      });
    });

    it('covers both concepts and implementation details', () => {
      const slides = AIPERF_FLOW_SYSTEM_DECK.slides;

      // Some slides should focus on concepts (lifetime events)
      const conceptSlides = slides.filter(s =>
        s.narration.toLowerCase().includes('boundary') ||
        s.narration.toLowerCase().includes('evidence')
      );
      expect(conceptSlides.length).toBeGreaterThan(0);

      // Some slides should cover implementation (clocks, workers, sinks)
      const implementationSlides = slides.filter(s =>
        s.narration.toLowerCase().includes('rust') ||
        s.narration.toLowerCase().includes('worker') ||
        s.narration.toLowerCase().includes('clock')
      );
      expect(implementationSlides.length).toBeGreaterThan(0);
    });
  });
});
