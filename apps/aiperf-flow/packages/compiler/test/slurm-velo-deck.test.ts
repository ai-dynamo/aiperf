/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { describe, it, expect, beforeAll } from 'vitest';
import { parseDocument } from '@aiperf/flow-language';
import { collectSymbols } from '../src/symbols.js';
import { expandSymbolInvocations } from '../src/expand-symbols.js';
import { link } from '../src/link.js';
import { validate } from '../src/validate.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

/**
 * Compiler test for slurm-velo.flow explainer deck.
 * Validates:
 * - Parses without syntax errors
 * - Symbol collection succeeds
 * - Symbol expansion works
 * - Linking completes
 * - Validation passes
 */
describe('slurm-velo.flow compilation', () => {
  let sourceCode: string;

  beforeAll(() => {
    // Load the .flow source file
    const flowPath = join(
      __dirname,
      '../../runtime/src/explainer/decks/slurm-velo.flow'
    );
    sourceCode = readFileSync(flowPath, 'utf-8');
  });

  it('parses the source file successfully', () => {
    const result = parseDocument(sourceCode, 'slurm-velo.flow');
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value).toBeDefined();
      expect(result.value.explainers).toBeDefined();
      expect(result.value.explainers!.length).toBeGreaterThan(0);
    }
  });

  it('collects symbols without errors', () => {
    const parseResult = parseDocument(sourceCode, 'slurm-velo.flow');
    expect(parseResult.ok).toBe(true);

    if (parseResult.ok) {
      const symbolResult = collectSymbols(parseResult.value);
      expect(symbolResult.ok).toBe(true);
      if (symbolResult.ok) {
        expect(symbolResult.value).toBeDefined();
      }
    }
  });

  it('expands symbol invocations', () => {
    const parseResult = parseDocument(sourceCode, 'slurm-velo.flow');
    expect(parseResult.ok).toBe(true);

    if (parseResult.ok) {
      const symbolResult = collectSymbols(parseResult.value);
      expect(symbolResult.ok).toBe(true);

      if (symbolResult.ok) {
        const expandResult = expandSymbolInvocations(
          parseResult.value,
          symbolResult.value
        );
        expect(expandResult.ok).toBe(true);
        if (expandResult.ok) {
          expect(expandResult.value).toBeDefined();
        }
      }
    }
  });

  it('links without errors', () => {
    const parseResult = parseDocument(sourceCode, 'slurm-velo.flow');
    expect(parseResult.ok).toBe(true);

    if (parseResult.ok) {
      const symbolResult = collectSymbols(parseResult.value);
      if (symbolResult.ok) {
        const expandResult = expandSymbolInvocations(
          parseResult.value,
          symbolResult.value
        );
        if (expandResult.ok) {
          const linkResult = link(expandResult.value);
          expect(linkResult.ok).toBe(true);
          if (linkResult.ok) {
            expect(linkResult.value).toBeDefined();
          }
        }
      }
    }
  });

  it('validates without errors', () => {
    const parseResult = parseDocument(sourceCode, 'slurm-velo.flow');
    expect(parseResult.ok).toBe(true);

    if (parseResult.ok) {
      const symbolResult = collectSymbols(parseResult.value);
      if (symbolResult.ok) {
        const expandResult = expandSymbolInvocations(
          parseResult.value,
          symbolResult.value
        );
        if (expandResult.ok) {
          const linkResult = link(expandResult.value);
          if (linkResult.ok) {
            const validateResult = validate(
              linkResult.value,
              {},
              false
            );
            expect(validateResult.ok).toBe(true);
            if (validateResult.ok) {
              expect(validateResult.value).toBeDefined();
            }
          }
        }
      }
    }
  });

  it('full compilation pipeline succeeds', () => {
    // Parse
    const parseResult = parseDocument(sourceCode, 'slurm-velo.flow');
    if (!parseResult.ok) {
      console.error('Parse failed:', parseResult.diagnostics);
      expect(parseResult.ok).toBe(true);
      return;
    }

    // Collect symbols
    const symbolResult = collectSymbols(parseResult.value);
    if (!symbolResult.ok) {
      console.error('Symbol collection failed:', symbolResult.diagnostics);
      expect(symbolResult.ok).toBe(true);
      return;
    }

    // Expand symbols
    const expandResult = expandSymbolInvocations(
      parseResult.value,
      symbolResult.value
    );
    if (!expandResult.ok) {
      console.error('Symbol expansion failed:', expandResult.diagnostics);
      expect(expandResult.ok).toBe(true);
      return;
    }

    // Link
    const linkResult = link(expandResult.value);
    if (!linkResult.ok) {
      console.error('Linking failed:', linkResult.diagnostics);
      expect(linkResult.ok).toBe(true);
      return;
    }

    // Validate
    const validateResult = validate(linkResult.value, {}, false);
    if (!validateResult.ok) {
      console.error('Validation failed:', validateResult.diagnostics);
      expect(validateResult.ok).toBe(true);
      return;
    }

    expect(validateResult.value).toBeDefined();
  });

  it('produces diagnostics array in results', () => {
    const parseResult = parseDocument(sourceCode, 'slurm-velo.flow');

    expect(parseResult.diagnostics).toBeDefined();
    expect(Array.isArray(parseResult.diagnostics)).toBe(true);
  });

  it('source code is well-formed', () => {
    // Basic structural checks
    const openBraces = (sourceCode.match(/\{/g) || []).length;
    const closeBraces = (sourceCode.match(/\}/g) || []).length;

    expect(openBraces).toBe(closeBraces);
    expect(openBraces).toBeGreaterThan(0);
  });

  it('contains valid deck ID', () => {
    expect(sourceCode).toContain('explainer "slurm-velo"');
  });
});
