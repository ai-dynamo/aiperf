#!/usr/bin/env node

import { chromium } from '@playwright/test';
import fs from 'fs';
import { PNG } from 'pngjs';
import pixelmatch from 'pixelmatch';

async function loadPNG(filepath) {
  return new Promise((resolve, reject) => {
    fs.createReadStream(filepath)
      .pipe(new PNG())
      .on('parsed', function() { resolve(this); })
      .on('error', reject);
  });
}

async function comparePNGs(file1, file2) {
  const img1 = await loadPNG(file1);
  const img2 = await loadPNG(file2);

  if (img1.width !== img2.width || img1.height !== img2.height) {
    return {
      error: `Size mismatch: ${img1.width}x${img1.height} vs ${img2.width}x${img2.height}`,
      diff: 100,
      pixels: 0,
      diffPixels: img1.width * img1.height
    };
  }

  const diff = new PNG({ width: img1.width, height: img1.height });
  const diffPixels = pixelmatch(img1.data, img2.data, diff.data, img1.width, img1.height, { threshold: 0.1 });

  const totalPixels = img1.width * img1.height;
  const diffPercent = (diffPixels / totalPixels) * 100;

  return {
    width: img1.width,
    height: img1.height,
    totalPixels,
    diffPixels,
    diff: parseFloat(diffPercent.toFixed(2))
  };
}

async function main() {
  console.log('Comparing slides...\n');
  console.log('EXPLAINERS APP vs AIPERF-FLOW\n');
  console.log('─'.repeat(80));

  // Compare slides
  const comparisons = [];

  for (let i = 1; i <= 3; i++) {
    const explainersFile = `/tmp/explainers-slide-${i}.png`;
    const aiperfFile = `/tmp/aiperf-flow-slide-${i}.png`;

    if (!fs.existsSync(explainersFile) || !fs.existsSync(aiperfFile)) {
      console.log(`Slide ${i}: File missing`);
      continue;
    }

    try {
      const result = await comparePNGs(explainersFile, aiperfFile);
      comparisons.push({ slide: i, ...result });

      if (result.error) {
        console.log(`\nSlide ${i}:`);
        console.log(`  Error: ${result.error}`);
      } else {
        console.log(`\nSlide ${i}:`);
        console.log(`  Dimensions: ${result.width} × ${result.height}`);
        console.log(`  Diff pixels: ${result.diffPixels} / ${result.totalPixels}`);
        console.log(`  Difference: ${result.diff}%`);

        if (result.diff === 0) {
          console.log(`  Status: ✓ IDENTICAL`);
        } else if (result.diff < 1) {
          console.log(`  Status: ⚠ Near-identical (minor rendering differences)`);
        } else if (result.diff < 5) {
          console.log(`  Status: ◐ Similar (some visual differences)`);
        } else {
          console.log(`  Status: ◯ Different (significant divergence)`);
        }
      }
    } catch (error) {
      console.log(`\nSlide ${i}: Error - ${error.message}`);
    }
  }

  console.log('\n' + '─'.repeat(80));
  console.log('\nSUMMARY:');
  console.log(`  Slides compared: ${comparisons.length}`);

  const identical = comparisons.filter(c => c.diff === 0);
  const similar = comparisons.filter(c => c.diff > 0 && c.diff < 5);
  const different = comparisons.filter(c => c.diff >= 5);

  console.log(`  Identical (0% diff): ${identical.length}`);
  console.log(`  Similar (<5% diff): ${similar.length}`);
  console.log(`  Different (≥5% diff): ${different.length}`);

  if (identical.length > 0) {
    console.log(`\n  Identical slides: ${identical.map(c => `#${c.slide}`).join(', ')}`);
  }
  if (similar.length > 0) {
    console.log(`  Similar slides: ${similar.map(c => `#${c.slide} (${c.diff}%)`).join(', ')}`);
  }
  if (different.length > 0) {
    console.log(`  Different slides: ${different.map(c => `#${c.slide} (${c.diff}%)`).join(', ')}`);
  }
}

main().catch(console.error);
