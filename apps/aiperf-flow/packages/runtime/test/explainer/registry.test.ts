import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { ExplainerRegistry } from '../../src/explainer/registry.js';
import type { ExplainerDefinition } from '@aiperf/flow-compiler';

describe('ExplainerRegistry', () => {
  afterEach(() => {
    // Clear registry between tests
    ExplainerRegistry.clear();
  });

  it('registers and retrieves explainer deck', () => {
    const deck: ExplainerDefinition = {
      id: 'test',
      route: '/test',
      topic: 'intro',
      eyebrowLabel: 'Test',
      startGateTitle: 'Go?',
      slides: [],
      scenesById: new Map(),
    };

    ExplainerRegistry.register(deck);
    const retrieved = ExplainerRegistry.getDeck('test');

    expect(retrieved).toEqual(deck);
  });

  it('rejects duplicate deck IDs', () => {
    const deck1: ExplainerDefinition = {
      id: 'dup',
      route: '/dup',
      topic: 'intro',
      eyebrowLabel: 'Dup',
      startGateTitle: '?',
      slides: [],
      scenesById: new Map(),
    };

    const deck2: ExplainerDefinition = {
      id: 'dup',
      route: '/dup2',
      topic: 'intro',
      eyebrowLabel: 'Dup2',
      startGateTitle: '?',
      slides: [],
      scenesById: new Map(),
    };

    ExplainerRegistry.register(deck1);
    expect(() => ExplainerRegistry.register(deck2)).toThrow(/duplicate/i);
  });

  it('rejects duplicate routes', () => {
    const deck1: ExplainerDefinition = {
      id: 'deck1',
      route: '/duplicate',
      topic: 'intro',
      eyebrowLabel: 'D1',
      startGateTitle: '?',
      slides: [],
      scenesById: new Map(),
    };

    const deck2: ExplainerDefinition = {
      id: 'deck2',
      route: '/duplicate',
      topic: 'intro',
      eyebrowLabel: 'D2',
      startGateTitle: '?',
      slides: [],
      scenesById: new Map(),
    };

    ExplainerRegistry.register(deck1);
    expect(() => ExplainerRegistry.register(deck2)).toThrow(/route.*already/i);
  });

  it('returns all registered decks', () => {
    const deck1: ExplainerDefinition = {
      id: 'deck1',
      route: '/deck1',
      topic: 'intro',
      eyebrowLabel: 'D1',
      startGateTitle: '?',
      slides: [],
      scenesById: new Map(),
    };

    const deck2: ExplainerDefinition = {
      id: 'deck2',
      route: '/deck2',
      topic: 'intro',
      eyebrowLabel: 'D2',
      startGateTitle: '?',
      slides: [],
      scenesById: new Map(),
    };

    ExplainerRegistry.register(deck1);
    ExplainerRegistry.register(deck2);
    const all = ExplainerRegistry.getAllDecks();

    expect(all).toHaveLength(2);
    expect(all.map(d => d.id)).toContain('deck1');
    expect(all.map(d => d.id)).toContain('deck2');
  });

  it('provides route-to-ID mapping', () => {
    const deck: ExplainerDefinition = {
      id: 'test',
      route: '/test-route',
      topic: 'intro',
      eyebrowLabel: 'Test',
      startGateTitle: '?',
      slides: [],
      scenesById: new Map(),
    };

    ExplainerRegistry.register(deck);
    const routeMap = ExplainerRegistry.getRouteMap();

    expect(routeMap.get('/test-route')).toBe('test');
  });
});
