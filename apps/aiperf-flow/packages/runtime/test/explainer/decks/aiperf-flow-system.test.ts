/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { readFileSync } from 'fs';
import { resolve } from 'path';
import { parseDocument } from '@aiperf/flow-language';
import { compileSource } from '@aiperf/flow-compiler';
import type { ExplainerDefinition, SlideDefinition } from '@aiperf/flow-compiler';

describe('AIPerf Flow System Explainer Deck', () => {
  let deckSource: string;
  const deckPath = resolve(
    __dirname,
    '../../../src/explainer/decks/aiperf-flow-system.flow'
  );

  beforeEach(() => {
    deckSource = readFileSync(deckPath, 'utf-8');
  });

  describe('Schema Validation', () => {
    it('parses .flow source without syntax errors', () => {
      const parsed = parseDocument(deckSource, 'aiperf-flow-system.flow');
      expect(parsed.ok).toBe(true);
    });

    it('defines explainer block with required metadata', () => {
      const parsed = parseDocument(deckSource, 'aiperf-flow-system.flow');
      if (!parsed.ok) {
        throw new Error(`Parse failed: ${parsed.diagnostics.map(d => d.message).join('; ')}`);
      }

      const doc = parsed.value;
      const explainers = doc.explainers ?? [];
      expect(explainers.length).toBeGreaterThan(0);

      const deckExplainer = explainers[0];
      expect(deckExplainer).toBeDefined();
      expect(deckExplainer?.id).toBe('aiperf-flow-system');
      expect(deckExplainer?.metadata.route).toBe('/explainers/aiperf-flow-system');
      expect(deckExplainer?.metadata.topic).toBe('aiperf-architecture');
      expect(deckExplainer?.metadata.eyebrowLabel).toBe('AIPerf Flow System');
    });

    it('contains 9 slides', () => {
      const parsed = parseDocument(deckSource, 'aiperf-flow-system.flow');
      if (!parsed.ok) {
        throw new Error(`Parse failed: ${parsed.diagnostics.map(d => d.message).join('; ')}`);
      }

      const explainers = parsed.value.explainers ?? [];
      const deck = explainers[0];
      expect(deck?.slides.length).toBe(9);
    });
  });

  describe('Slide Content', () => {
    let slides: SlideDefinition[];

    beforeEach(() => {
      const parsed = parseDocument(deckSource, 'aiperf-flow-system.flow');
      if (!parsed.ok) {
        throw new Error(`Parse failed: ${parsed.diagnostics.map(d => d.message).join('; ')}`);
      }

      const explainers = parsed.value.explainers ?? [];
      slides = explainers[0]?.slides ?? [];
    });

    it('ensures all slides have required fields', () => {
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
      const titles = slides.map(s => s.title);
      const uniqueTitles = new Set(titles);
      expect(uniqueTitles.size).toBe(titles.length);
    });

    it('narration is non-empty and substantive', () => {
      slides.forEach((slide, index) => {
        expect(slide.narration.trim().length).toBeGreaterThan(20,
          `Slide ${index} narration too brief`
        );
      });
    });

    it('each slide has relevant bullet points', () => {
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
      const expectedTopics = [
        'Architecture',      // Slide 1
        'Lifecycle',          // Slide 2
        'Admission',          // Slide 3
        'Transport',          // Slide 4
        'Worker',             // Slide 5
        'Stream',             // Slide 6
        'Measurement',        // Slide 7
        'Visualization',      // Slide 8
        'Integration'         // Slide 9
      ];

      slides.forEach((slide, index) => {
        const expectedKeyword = expectedTopics[index];
        const titleAndNarration = (slide.title + ' ' + slide.narration).toLowerCase();
        expect(titleAndNarration).toContain(expectedKeyword.toLowerCase(),
          `Slide ${index} does not cover expected topic: ${expectedKeyword}`
        );
      });
    });

    it('some slides define embedded scenes', () => {
      const slidesWithScenes = slides.filter(s => s.sceneId);
      expect(slidesWithScenes.length).toBeGreaterThan(0,
        'At least some slides should reference embedded scenes'
      );
    });
  });

  describe('Scene Rendering', () => {
    it('defines embedded scenes for visualization', () => {
      const parsed = parseDocument(deckSource, 'aiperf-flow-system.flow');
      if (!parsed.ok) {
        throw new Error(`Parse failed: ${parsed.diagnostics.map(d => d.message).join('; ')}`);
      }

      const doc = parsed.value;
      const scenes = doc.scenes ?? [];
      expect(scenes.length).toBeGreaterThan(0,
        'Deck should define at least one embedded scene'
      );
    });

    it('embedded scenes have required structure', () => {
      const parsed = parseDocument(deckSource, 'aiperf-flow-system.flow');
      if (!parsed.ok) {
        throw new Error(`Parse failed: ${parsed.diagnostics.map(d => d.message).join('; ')}`);
      }

      const doc = parsed.value;
      const scenes = doc.scenes ?? [];

      scenes.forEach((scene, index) => {
        expect(scene.id, `Scene ${index} missing id`).toBeTruthy();
        expect(scene.title, `Scene ${index} missing title`).toBeTruthy();
        expect(scene.summary, `Scene ${index} missing summary`).toBeTruthy();
        expect(scene.narration, `Scene ${index} missing narration`).toBeTruthy();
      });
    });

    it('scenes reference defined tokens', () => {
      const parsed = parseDocument(deckSource, 'aiperf-flow-system.flow');
      if (!parsed.ok) {
        throw new Error(`Parse failed: ${parsed.diagnostics.map(d => d.message).join('; ')}`);
      }

      const doc = parsed.value;
      const tokenIds = (doc.tokens ?? []).map(t => t.id);

      // Should have color tokens for visualization
      const colorTokens = ['background', 'surface', 'request', 'transport', 'observer', 'evidence'];
      colorTokens.forEach(token => {
        expect(tokenIds).toContain(token,
          `Missing token definition: ${token}`
        );
      });
    });
  });

  describe('Narration Integration', () => {
    let slides: SlideDefinition[];

    beforeEach(() => {
      const parsed = parseDocument(deckSource, 'aiperf-flow-system.flow');
      if (!parsed.ok) {
        throw new Error(`Parse failed: ${parsed.diagnostics.map(d => d.message).join('; ')}`);
      }

      const explainers = parsed.value.explainers ?? [];
      slides = explainers[0]?.slides ?? [];
    });

    it('narration covers AIPerf lifecycle concepts', () => {
      const allNarration = slides.map(s => s.narration).join(' ');
      const concepts = ['request', 'admission', 'transport', 'worker', 'stream', 'observer', 'evidence', 'measurement'];

      concepts.forEach(concept => {
        expect(allNarration.toLowerCase()).toContain(concept.toLowerCase(),
          `Narration should cover: ${concept}`
        );
      });
    });

    it('narration explains clock-aware execution', () => {
      const narration = slides.map(s => s.narration).join(' ');
      expect(narration.toLowerCase()).toContain('clock');
    });

    it('narration mentions stable evidence', () => {
      const narration = slides.map(s => s.narration).join(' ');
      expect(narration.toLowerCase()).toContain('evidence');
    });

    it('slide captions are concise summaries', () => {
      slides.forEach((slide, index) => {
        expect(slide.caption.trim().length).toBeGreaterThan(10,
          `Slide ${index} caption too brief`
        );
        expect(slide.caption.split(' ').length).toBeLessThan(20,
          `Slide ${index} caption too long`
        );
      });
    });

    it('some slides define terminology', () => {
      const slidesWithTerms = slides.filter(s => s.term);
      expect(slidesWithTerms.length).toBeGreaterThan(0,
        'At least some slides should define key terminology'
      );
    });

    it('term definitions are educational', () => {
      slides.forEach((slide, index) => {
        if (slide.term) {
          expect(slide.term.word).toBeTruthy();
          expect(slide.term.meaning).toBeTruthy();
          expect(slide.term.meaning.length).toBeGreaterThan(15,
            `Slide ${index} term definition too brief`
          );
        }
      });
    });
  });

  describe('Compilation and Integration', () => {
    it('deck compiles to valid Flow IR', () => {
      const result = compileSource({
        source: deckSource,
        sourceName: 'aiperf-flow-system.flow',
        capabilities: {
          'core.rect': '^1.0.0',
          'core.connector': '^1.0.0',
          'core.camera': '^1.0.0',
          'core.timeline': '^1.0.0',
          'core.inspect': '^1.0.0',
          'viz.queue': '^1.0.0',
          'viz.waterfall': '^1.0.0',
        },
        strict: true,
      });

      if (!result.ok) {
        const messages = result.diagnostics.map(d => `${d.code}: ${d.message}`).join('\n');
        throw new Error(`Compilation failed:\n${messages}`);
      }

      expect(result.value).toBeDefined();
      expect(result.value.id).toBe('aiperf-flow-system');
      expect(result.value.scenes.length).toBeGreaterThan(0);
    });

    it('compiled deck has proper structure', () => {
      const result = compileSource({
        source: deckSource,
        sourceName: 'aiperf-flow-system.flow',
        capabilities: {
          'core.rect': '^1.0.0',
          'core.connector': '^1.0.0',
          'core.camera': '^1.0.0',
          'core.timeline': '^1.0.0',
          'core.inspect': '^1.0.0',
          'viz.queue': '^1.0.0',
          'viz.waterfall': '^1.0.0',
        },
        strict: false,
      });

      if (!result.ok) {
        throw new Error('Compilation failed');
      }

      const ir = result.value;
      expect(ir.id).toBe('aiperf-flow-system');
      expect(ir.title).toContain('AIPerf');
      expect(ir.scenes.length).toBeGreaterThan(0);

      // Verify scene content
      ir.scenes.forEach(scene => {
        expect(scene.id).toBeTruthy();
        expect(scene.title).toBeTruthy();
        expect(scene.summary).toBeTruthy();
      });
    });
  });

  describe('Accessibility and Fallback', () => {
    it('includes narration for audio and semantic contexts', () => {
      const parsed = parseDocument(deckSource, 'aiperf-flow-system.flow');
      if (!parsed.ok) {
        throw new Error('Parse failed');
      }

      const explainers = parsed.value.explainers ?? [];
      const slides = explainers[0]?.slides ?? [];

      slides.forEach(slide => {
        expect(slide.narration.trim().length).toBeGreaterThan(0);
      });
    });

    it('scenes include fallback text for non-interactive contexts', () => {
      const parsed = parseDocument(deckSource, 'aiperf-flow-system.flow');
      if (!parsed.ok) {
        throw new Error('Parse failed');
      }

      const doc = parsed.value;
      const scenes = doc.scenes ?? [];

      scenes.forEach((scene, index) => {
        expect(scene.fallback).toBeTruthy(
          `Scene ${index} should have fallback text`
        );
      });
    });
  });
});
