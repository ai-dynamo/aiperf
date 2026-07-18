#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Compile-time build script: converts .flow explainer files to TypeScript ExplainerDefinition objects.
 * Runs at build time to embed .flow source data into the runtime.
 * Ensures byte-exact visual rendering from .flow files.
 */

import { readFileSync, writeFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(__dirname, '..');

const EXPLAINER_DECKS = [
  {
    id: 'rust-architecture',
    path: 'packages/runtime/src/explainer/decks/rust-architecture.flow',
    topic: 'system-architecture',
  },
  {
    id: 'slurm-velo',
    path: 'packages/runtime/src/explainer/decks/slurm-velo.flow',
    topic: 'distributed-execution',
  },
  {
    id: 'dynosim',
    path: 'packages/runtime/src/explainer/decks/dynosim.flow',
    topic: 'simulation',
  },
  {
    id: 'aiperf-flow-system',
    path: 'packages/runtime/src/explainer/decks/aiperf-flow-system.flow',
    topic: 'flow-system',
  },
];

function extractSlidesFromFlow(source) {
  const slides = [];

  // Try Format 1: `slide "title" { ... }` with brace matching
  const slideMatches = [...source.matchAll(/slide\s+"([^"]+)"\s*\{/g)];

  if (slideMatches.length > 0) {
    // Format 1 (slurm-velo, dynosim, aiperf-flow-system style)
    for (const match of slideMatches) {
      const slideTitle = match[1];
      const startPos = match.index + match[0].length;

      // Find matching closing brace
      let braceDepth = 1;
      let endPos = startPos;
      while (braceDepth > 0 && endPos < source.length) {
        if (source[endPos] === '{') braceDepth++;
        if (source[endPos] === '}') braceDepth--;
        endPos++;
      }

      const slideContent = source.substring(startPos, endPos - 1);

      const eyebrowMatch = slideContent.match(/eyebrow:\s+"([^"]*?)"/);
      const narrationMatch = slideContent.match(/narration:\s+"([^"]*?)"/);
      const pointsMatch = slideContent.match(/points:\s*\[([\s\S]*?)\]/);
      const captionMatch = slideContent.match(/caption:\s+"([^"]*?)"/);
      const ledeMatch = slideContent.match(/lede:\s+"([^"]*?)"/);

      const points = pointsMatch ? parsePoints(pointsMatch[1]) : [];

      // Extract @scene block for native .flow rendering
      const sceneMatch = slideContent.match(/render:\s+@scene\s*\{([\s\S]*)\}/);
      const sceneBlock = sceneMatch ? `{${sceneMatch[1]}}` : null;

      // Generate ID from title
      const slideId = slideTitle
        .toLowerCase()
        .replace(/\s+/g, '-')
        .replace(/[^a-z0-9-]/g, '');

      slides.push({
        id: slideId,
        eyebrow: eyebrowMatch?.[1] || '',
        title: slideTitle,
        lede: ledeMatch?.[1] || '',
        narration: narrationMatch?.[1] || '',
        points,
        caption: captionMatch?.[1] || '',
        term: undefined,
        // Include raw @scene block for native rendering
        ...(sceneBlock && { sceneBlock }),
      });
    }

    return slides;
  }

  // Otherwise try Format 2: object literal style (rust-architecture)
  const slideSections = source.split(/(?=\n\s+id:\s+")/);

  for (const section of slideSections) {
    if (!section.includes('render: @scene')) continue;

    const idMatch = section.match(/id:\s+"([^"]+)"/);
    const eyebrowMatch = section.match(/eyebrow:\s+"([^"]*?)"/);
    const titleMatch = section.match(/title:\s+"([^"]*?)"/);
    const ledeMatch = section.match(/lede:\s+"([^"]*?)"/);
    const narrationMatch = section.match(/narration:\s+`([^`]*?)`/s);
    const pointsMatch = section.match(/points:\s*\[([\s\S]*?)\]/);
    const captionMatch = section.match(/caption:\s+"([^"]*?)"/);

    if (!idMatch) continue;

    const points = pointsMatch ? parsePoints(pointsMatch[1]) : [];

    slides.push({
      id: idMatch[1],
      eyebrow: eyebrowMatch?.[1] || '',
      title: titleMatch?.[1] || '',
      lede: ledeMatch?.[1] || '',
      narration: narrationMatch?.[1] || '',
      points,
      caption: captionMatch?.[1] || '',
      term: undefined,
    });
  }

  return slides;
}

function parsePoints(pointsStr) {
  const matches = pointsStr.match(/"([^"]*)"/g);
  return (matches || []).map(m => m.slice(1, -1));
}

function extractGlossaryFromFlow(source) {
  const terms = [];
  const termPattern = /term:\s*\{\s*word:\s+"([^"]+)"\s*,\s*meaning:\s+"([^"]+)"\s*\}/g;

  let match;
  while ((match = termPattern.exec(source)) !== null) {
    terms.push({
      word: match[1],
      meaning: match[2],
    });
  }

  return terms;
}

// Compile all .flow files
const compiledDecks = [];

for (const deck of EXPLAINER_DECKS) {
  const flowPath = resolve(projectRoot, deck.path);
  console.log(`Compiling ${deck.id} from ${deck.path}...`);

  const source = readFileSync(flowPath, 'utf-8');
  const slides = extractSlidesFromFlow(source);
  const glossary = extractGlossaryFromFlow(source);

  compiledDecks.push({
    id: deck.id,
    topic: deck.topic,
    slides,
    glossary,
  });

  console.log(`  ✓ ${slides.length} slides extracted`);
}

// Generate TypeScript output
const output = `// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * AUTO-GENERATED: Compiled from .flow explainer source files.
 * Run: node scripts/compile-explainer-flows.mjs
 *
 * This file contains TypeScript representations of all .flow explainer decks,
 * compiled at build time to ensure byte-exact rendering from .flow source.
 */

import type { ExplainerDefinition } from '../runtime/src/explainer/registry';

${compiledDecks
  .map(
    deck => `
export const ${deck.id.replace(/-/g, '_').toUpperCase()}_DECK: ExplainerDefinition = {
  id: '${deck.id}',
  topic: '${deck.topic}',
  slides: ${JSON.stringify(deck.slides, null, 2)},
  glossary: ${JSON.stringify(deck.glossary, null, 2)},
};
`.trim()
  )
  .join('\n\n')}

export const COMPILED_EXPLAINER_DECKS = [
  ${compiledDecks.map(d => `${d.id.replace(/-/g, '_').toUpperCase()}_DECK`).join(',\n  ')},
] as const;
`;

const outputPath = resolve(projectRoot, 'packages/runtime/src/explainer/compiled-decks.ts');
writeFileSync(outputPath, output, 'utf-8');

console.log(`\n✓ Compiled ${compiledDecks.length} explainer decks to ${outputPath}`);
console.log(`  Total slides: ${compiledDecks.reduce((sum, d) => sum + d.slides.length, 0)}`);
