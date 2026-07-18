/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { SLURM_VELO_DECK, BUILTIN_DECKS, loadBuiltinDecks } from '../../../src/explainer/decks/loader.js';
import { ExplainerRegistry } from '../../../src/explainer/registry.js';
import type { ExplainerDefinition } from '@aiperf/flow-compiler';

describe('Explainer Deck Loader: SLURM+Velo', () => {
  beforeEach(() => {
    ExplainerRegistry.clear();
  });

  afterEach(() => {
    ExplainerRegistry.clear();
  });

  describe('SLURM_VELO_DECK structure', () => {
    it('has correct metadata', () => {
      expect(SLURM_VELO_DECK.id).toBe('slurm-velo');
      expect(SLURM_VELO_DECK.route).toBe('/explainers/slurm-velo');
      expect(SLURM_VELO_DECK.topic).toBe('distributed-execution');
      expect(SLURM_VELO_DECK.eyebrowLabel).toBe('Cluster Orchestration');
      expect(SLURM_VELO_DECK.startGateTitle).toBe('Ready to learn SLURM + Velo?');
    });

    it('contains exactly 16 slides', () => {
      expect(SLURM_VELO_DECK.slides).toHaveLength(16);
    });

    it('has scenesById map for scene integration', () => {
      expect(SLURM_VELO_DECK.scenesById).toBeDefined();
      expect(SLURM_VELO_DECK.scenesById).toBeInstanceOf(Map);
    });
  });

  describe('Slide structure validation', () => {
    it('each slide has required fields', () => {
      SLURM_VELO_DECK.slides.forEach((slide) => {
        expect(slide.eyebrow).toBeDefined();
        expect(slide.title).toBeDefined();
        expect(slide.lede).toBeDefined();
        expect(slide.narration).toBeDefined();
        expect(slide.points).toBeDefined();
        expect(Array.isArray(slide.points)).toBe(true);
        expect(slide.caption).toBeDefined();
        expect(slide.narration.trim().length).toBeGreaterThan(0);
      });
    });

    it('each slide has at least 2 bullet points', () => {
      SLURM_VELO_DECK.slides.forEach((slide) => {
        expect(slide.points.length).toBeGreaterThanOrEqual(2);
      });
    });

    it('glossary terms defined on relevant slides', () => {
      const slidesWithTerms = SLURM_VELO_DECK.slides.filter((s) => s.term);
      expect(slidesWithTerms.length).toBeGreaterThan(0);
      slidesWithTerms.forEach((slide) => {
        expect(slide.term?.word).toBeDefined();
        expect(slide.term?.meaning).toBeDefined();
      });
    });
  });

  describe('Registry integration', () => {
    it('deck can be registered and retrieved', () => {
      ExplainerRegistry.register(SLURM_VELO_DECK);
      expect(ExplainerRegistry.getDeck('slurm-velo')).toBeDefined();
      expect(ExplainerRegistry.getDeckByRoute('/explainers/slurm-velo')).toBeDefined();
    });
  });

  describe('Builtin decks list', () => {
    it('includes SLURM+Velo deck', () => {
      expect(BUILTIN_DECKS).toContain(SLURM_VELO_DECK);
    });

    it('loads all builtin decks successfully', () => {
      const registry = { register: (deck: ExplainerDefinition) => ExplainerRegistry.register(deck) };
      loadBuiltinDecks(registry);
      expect(ExplainerRegistry.getDeck('slurm-velo')).toBeDefined();
      expect(ExplainerRegistry.getDeck('aiperf-flow-system')).toBeDefined();
    });
  });

  describe('Narration coverage', () => {
    it('all 16 narration texts are distinct', () => {
      const narrations = SLURM_VELO_DECK.slides.map((s) => s.narration);
      const uniqueNarrations = new Set(narrations);
      expect(uniqueNarrations.size).toBe(16);
    });
  });

  describe('Content completeness', () => {
    it('covers SLURM, Velo, and communication patterns', () => {
      const allContent = SLURM_VELO_DECK.slides
        .map((s) => s.title + ' ' + s.narration)
        .join(' ');

      expect(allContent).toContain('SLURM');
      expect(allContent).toContain('Velo');
      expect(allContent).toContain('fan');
      expect(allContent).toContain('controller');
      expect(allContent).toContain('cell');
    });
  });
});
