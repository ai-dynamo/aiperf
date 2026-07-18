import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { readFileSync } from 'fs';
import { join } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { ExplainerRegistry } from '../../src/explainer/registry.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

describe('dynosim.flow deck', () => {
  const flowPath = join(
    __dirname,
    '../../src/explainer/decks/dynosim.flow'
  );

  let flowContent: string;

  beforeEach(() => {
    // Read the flow file
    flowContent = readFileSync(flowPath, 'utf-8');
  });

  afterEach(() => {
    ExplainerRegistry.clear();
  });

  describe('Schema validation', () => {
    it('should have valid flow header', () => {
      expect(flowContent).toMatch(/flow\s+"Dynosim:/i);
      expect(flowContent).toMatch(/as\s+dynosim-explainer/i);
    });

    it('should declare language version', () => {
      expect(flowContent).toMatch(/language\s+1/);
    });

    it('should have required dependencies', () => {
      expect(flowContent).toMatch(/require\s+core\.rect/);
      expect(flowContent).toMatch(/require\s+core\.connector/);
      expect(flowContent).toMatch(/require\s+core\.camera/);
      expect(flowContent).toMatch(/require\s+core\.timeline/);
      expect(flowContent).toMatch(/require\s+core\.inspect/);
    });

    it('should define color tokens', () => {
      expect(flowContent).toMatch(/token\s+background\s*=/);
      expect(flowContent).toMatch(/token\s+surface\s*=/);
      expect(flowContent).toMatch(/token\s+dynosim\s*=/);
      expect(flowContent).toMatch(/token\s+clock\s*=/);
      expect(flowContent).toMatch(/token\s+engine\s*=/);
    });
  });

  describe('Scene definitions', () => {
    it('should have 18 scenes', () => {
      const sceneMatches = flowContent.match(/scene\s+"[^"]+"\s+as\s+[a-z-]+\s*\{/g);
      expect(sceneMatches).toBeDefined();
      expect(sceneMatches?.length).toBe(18);
    });

    it('should have all expected scene IDs', () => {
      const expectedScenes = [
        'why-dynosim',
        'feature-gate',
        'config-seam',
        'routing',
        'composition',
        'offline-mode',
        'online-mode',
        'clock-compare',
        'event-queues',
        'sim-pump',
        'ordering-rule',
        'step-bounds',
        'submission',
        'token-path',
        'metrics',
        'delivery-modes',
        'completion',
        'recap',
      ];

      expectedScenes.forEach(sceneId => {
        expect(flowContent).toContain(`as ${sceneId}`);
      });
    });

    it('should have summary for each scene', () => {
      const summaryMatches = flowContent.match(/summary\s+"[^"]+"/g);
      expect(summaryMatches).toBeDefined();
      expect(summaryMatches?.length).toBeGreaterThanOrEqual(19);
    });

    it('should have visual elements in each scene', () => {
      // Check for rect elements (basic UI building block)
      const rectMatches = flowContent.match(/rect\s+[a-z-]+\s*\{/g);
      expect(rectMatches).toBeDefined();
      expect(rectMatches?.length).toBeGreaterThan(50); // Multiple rects per scene
    });

    it('should have connectors for relationships', () => {
      const connectorMatches = flowContent.match(/connector\s+[a-z-]+\s*\{/g);
      expect(connectorMatches).toBeDefined();
      expect(connectorMatches?.length).toBeGreaterThan(5);
    });

    it('should have camera definitions', () => {
      const cameraMatches = flowContent.match(/camera\s+[a-z-]+\s*\{/g);
      expect(cameraMatches).toBeDefined();
      expect(cameraMatches?.length).toBe(19);
    });

    it('should have timeline definitions', () => {
      const timelineMatches = flowContent.match(/timeline\s+[a-z-]+\s*\{/g);
      expect(timelineMatches).toBeDefined();
      expect(timelineMatches?.length).toBe(19);
    });
  });

  describe('Narration completeness', () => {
    it('should have narration for each scene', () => {
      const narrateMatches = flowContent.match(/narrate\s+"[^"]+"/g);
      expect(narrateMatches).toBeDefined();
      // Should have at least 19 narration blocks (one per scene)
      expect(narrateMatches?.length).toBeGreaterThanOrEqual(19);
    });

    it('should have non-empty narration', () => {
      const narrateMatches = flowContent.matchAll(/narrate\s+"([^"]+)"/g);
      const narrations = Array.from(narrateMatches);

      expect(narrations.length).toBeGreaterThan(0);
      narrations.forEach(match => {
        expect(match[1]).toBeTruthy();
        expect(match[1].length).toBeGreaterThan(10); // Non-trivial narration
      });
    });

    it('should cover all key dynosim topics in narration', () => {
      const combinedNarration = flowContent
        .match(/narrate\s+"([^"]+)"/g)
        ?.join(' ')
        .toLowerCase() || '';

      expect(combinedNarration).toContain('clock');
      expect(combinedNarration).toContain('engine');
      expect(combinedNarration).toContain('virtual');
      expect(combinedNarration).toContain('deterministic');
      expect(combinedNarration).toContain('observer');
    });
  });

  describe('Content accuracy', () => {
    it('should reference SimClock and RealClock', () => {
      expect(flowContent).toContain('SimClock');
      expect(flowContent).toContain('RealClock');
    });

    it('should reference dynosim_offline and dynosim_online', () => {
      expect(flowContent).toContain('dynosim_offline');
      expect(flowContent).toContain('dynosim_online');
    });

    it('should reference SteppableReplay', () => {
      expect(flowContent).toContain('SteppableReplay');
    });

    it('should reference RequestObserver', () => {
      expect(flowContent).toContain('RequestObserver');
    });

    it('should reference DynosimSink', () => {
      expect(flowContent).toContain('DynosimSink');
    });

    it('should reference EngineHost', () => {
      expect(flowContent).toContain('EngineHost');
    });

    it('should discuss Clock abstraction', () => {
      expect(flowContent).toContain('Clock');
      expect(flowContent).toContain('clock-compare');
    });

    it('should discuss event queues', () => {
      expect(flowContent).toContain('event-queue');
      expect(flowContent).toContain('clock-queue');
      expect(flowContent).toContain('source-queue');
    });

    it('should discuss the sim pump', () => {
      expect(flowContent).toContain('sim-pump');
      expect(flowContent).toContain('poll');
      expect(flowContent).toContain('advance');
    });

    it('should discuss ordering and determinism', () => {
      expect(flowContent).toContain('ordering-rule');
      expect(flowContent).toContain('arrival-before-pass');
      expect(flowContent).toContain('deterministic');
    });

    it('should discuss metrics accumulation', () => {
      expect(flowContent).toContain('TTFT');
      expect(flowContent).toContain('ITL');
      expect(flowContent).toContain('NativeMetricsObserver');
    });

    it('should discuss terminal handling and streaming', () => {
      expect(flowContent).toContain('terminal');
      expect(flowContent).toContain('streaming');
      expect(flowContent).toContain('coalescing');
    });
  });

  describe('Scene structure', () => {
    it('should have reading order for each scene', () => {
      const readingOrderMatches = flowContent.match(/reading-order\s+[^;]+;/g);
      expect(readingOrderMatches).toBeDefined();
      expect(readingOrderMatches?.length).toBeGreaterThanOrEqual(19);
    });

    it('should have fallback text for accessibility', () => {
      const fallbackMatches = flowContent.match(/fallback\s+"[^"]+"/g);
      expect(fallbackMatches).toBeDefined();
      expect(fallbackMatches?.length).toBeGreaterThan(50); // Multiple fallbacks per scene
    });

    it('should have role annotations for accessibility', () => {
      const roleMatches = flowContent.match(/role\s+"[^"]+"/g);
      expect(roleMatches).toBeDefined();
      expect(roleMatches?.length).toBeGreaterThan(50);
    });

    it('should have description annotations', () => {
      const descriptionMatches = flowContent.match(/description\s+"[^"]+"/g);
      expect(descriptionMatches).toBeDefined();
      expect(descriptionMatches?.length).toBeGreaterThan(50);
    });
  });

  describe('Timeline definitions', () => {
    it('should have reveal animations', () => {
      expect(flowContent).toMatch(/at\s+\d+\s+reveal\s+[a-z-]+\s+duration\s+\d+/);
    });

    it('should have trace animations', () => {
      expect(flowContent).toMatch(/at\s+\d+\s+trace\s+[a-z-]+\s+duration\s+\d+/);
    });

    it('should have non-zero durations', () => {
      const durationMatches = flowContent.matchAll(/duration\s+(\d+)/g);
      const durations = Array.from(durationMatches).map(m => parseInt(m[1]));

      durations.forEach(duration => {
        expect(duration).toBeGreaterThan(0);
      });
    });
  });

  describe('Design tokens and colors', () => {
    it('should use defined color tokens', () => {
      expect(flowContent).toMatch(/token\(background\)/);
      expect(flowContent).toMatch(/token\(surface\)/);
      expect(flowContent).toMatch(/token\(dynosim\)/);
      expect(flowContent).toMatch(/token\(clock\)/);
    });

    it('should have valid hex color definitions', () => {
      const tokenMatches = flowContent.matchAll(/token\s+[a-z-]+\s*=\s*"(#[0-9a-f]{6})"/gi);
      const tokens = Array.from(tokenMatches);

      expect(tokens.length).toBeGreaterThan(0);
      tokens.forEach(match => {
        expect(match[1]).toMatch(/^#[0-9a-f]{6}$/i);
      });
    });
  });

  describe('Visual hierarchy', () => {
    it('should have header elements in scenes', () => {
      expect(flowContent).toMatch(/label\s+"[^"]*heading[^"]*"/i);
    });

    it('should have content grouping', () => {
      const groupMatches = flowContent.match(/role\s+"group"/g);
      expect(groupMatches).toBeDefined();
      expect(groupMatches?.length).toBeGreaterThan(30);
    });

    it('should have note/caption elements', () => {
      const noteMatches = flowContent.match(/role\s+"note"/g);
      expect(noteMatches).toBeDefined();
      expect(noteMatches?.length).toBeGreaterThan(10);
    });
  });

  describe('Responsive design', () => {
    it('should have responsive layout patterns', () => {
      // The scene widths should be relative to container
      expect(flowContent).toMatch(/width\s+1200/);
      expect(flowContent).toMatch(/height\s+\d+/);
    });

    it('should have consistent spacing', () => {
      // Check for x, y coordinate patterns
      expect(flowContent).toMatch(/x\s+\d+/);
      expect(flowContent).toMatch(/y\s+\d+/);
    });
  });

  describe('Slide metadata', () => {
    it('should cover all topics from original explainer', () => {
      const topics = [
        'Why Dynosim',
        'Feature gate',
        'Config seam',
        'Routing',
        'Composition',
        'Offline mode',
        'Online mode',
        'Clock compare',
        'Event queues',
        'Sim pump',
        'Ordering rule',
        'Step bounds',
        'Submission',
        'Token path',
        'Metrics',
        'Delivery modes',
        'Completion',
        'Recap',
      ];

      topics.forEach(topic => {
        expect(flowContent.toLowerCase()).toContain(topic.toLowerCase());
      });
    });

    it('should follow logical progression', () => {
      // Check that scenes appear in order
      const whyIndex = flowContent.indexOf('why-dynosim');
      const featureIndex = flowContent.indexOf('feature-gate');
      const recapIndex = flowContent.indexOf('as recap');

      expect(whyIndex).toBeLessThan(featureIndex);
      expect(featureIndex).toBeLessThan(recapIndex);
    });
  });

  describe('Technical accuracy', () => {
    it('should mention key components', () => {
      const components = [
        'Application',
        'BenchmarkRun',
        'DirectRequest',
        'SimStep',
        'RunOutcome',
      ];

      components.forEach(component => {
        expect(flowContent).toContain(component);
      });
    });

    it('should reference correct timing concepts', () => {
      const concepts = [
        'wall-clock',
        'virtual time',
        'deterministic',
        'arrivals',
        'phase',
        'executor',
      ];

      concepts.forEach(concept => {
        expect(flowContent.toLowerCase()).toContain(concept.toLowerCase());
      });
    });

    it('should explain transport-neutral patterns', () => {
      expect(flowContent).toContain('transport-neutral');
      expect(flowContent).toContain('RequestObserver');
      expect(flowContent).toContain('RequestSink');
    });
  });

  describe('Completeness', () => {
    it('should have no unclosed braces', () => {
      let openCount = (flowContent.match(/\{/g) || []).length;
      let closeCount = (flowContent.match(/\}/g) || []).length;
      expect(openCount).toBe(closeCount);
    });

    it('should have no unclosed strings', () => {
      // Count unescaped quotes at line boundaries
      const lines = flowContent.split('\n');
      lines.forEach(line => {
        const quoteCount = (line.match(/(?<!\\)"/g) || []).length;
        // Should be even on most lines (though comments might violate this)
        // Just check for obvious issues with block-level quotes
        if (
          line.includes('narrate') ||
          line.includes('summary') ||
          line.includes('label')
        ) {
          // These should have paired quotes
          const str = line.trim();
          if (str && !str.startsWith('//')) {
            // Don't enforce on comments
            expect(quoteCount % 2).toBe(0);
          }
        }
      });
    });

    it('should have SPDX header', () => {
      expect(flowContent).toContain('SPDX-FileCopyrightText');
      expect(flowContent).toContain('SPDX-License-Identifier: Apache-2.0');
    });

    it('should have copyright notice', () => {
      expect(flowContent).toContain('NVIDIA');
      expect(flowContent).toContain('2025-2026');
    });
  });
});
