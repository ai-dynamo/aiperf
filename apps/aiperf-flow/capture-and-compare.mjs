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
      diffPixels: img1.width * img1.height,
      width: img1.width,
      height: img1.height,
      totalPixels: img1.width * img1.height
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

async function captureExplainersSlides() {
  console.log('📸 Capturing EXPLAINERS app slides...');
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });

  const slides = [];

  try {
    // Navigate to explainers app - it's on 5173
    console.log('   Navigating to http://localhost:5173/');
    await page.goto('http://localhost:5173/', { waitUntil: 'networkidle', timeout: 30000 });
    await page.waitForTimeout(1500);

    // Click "Rust architecture from scratch"
    console.log('   Looking for Rust Architecture button...');
    const buttons = await page.locator('button, a').all();
    let clicked = false;
    for (const btn of buttons) {
      const text = await btn.textContent();
      if (text && text.toLowerCase().includes('rust') && text.toLowerCase().includes('architecture')) {
        console.log(`   ✓ Clicking: "${text}"`);
        await btn.click();
        clicked = true;
        await page.waitForTimeout(2000);
        break;
      }
    }

    if (!clicked) {
      console.error('   ✗ Could not find Rust architecture button');
      // Try to log what buttons we found
      for (const btn of buttons) {
        const text = await btn.textContent();
        if (text && text.length > 0 && text.length < 100) {
          console.log(`     Button: "${text}"`);
        }
      }
      await browser.close();
      return slides;
    }

    // Capture slides
    for (let i = 0; i < 4; i++) {
      try {
        // Dismiss audio dialog if present
        const audioBtn = await page.locator('button:has-text("Play without audio")').first();
        if (await audioBtn.isVisible({ timeout: 500 }).catch(() => false)) {
          console.log(`   Dismissing audio dialog for slide ${i + 1}...`);
          await audioBtn.click();
          await page.waitForTimeout(1000);
        }

        // Wait for slide to render
        await page.waitForTimeout(800);

        // Take screenshot
        const screenshot = await page.screenshot();
        slides.push({ app: 'explainers', slideNum: i + 1, data: screenshot });
        console.log(`   ✓ Captured slide ${i + 1}`);

        // Check for slide indicator
        const slideIndicator = await page.locator('text=/Slide \\d+ of \\d+/').textContent().catch(() => '');
        if (slideIndicator) {
          console.log(`     (${slideIndicator})`);
        }

        // Go to next slide
        const nextBtn = await page.locator('button').filter({ hasText: /Next|→|>/ }).first();
        if (await nextBtn.isEnabled({ timeout: 500 }).catch(() => false)) {
          await nextBtn.click();
          await page.waitForTimeout(1500);
        } else {
          console.log(`   End of slides reached at slide ${i + 1}`);
          break;
        }
      } catch (error) {
        console.log(`   ✗ Error on slide ${i + 1}: ${error.message}`);
        break;
      }
    }

  } catch (error) {
    console.error('✗ Error capturing explainers app:', error.message);
  } finally {
    await browser.close();
  }

  return slides;
}

async function captureAiPerfFlowSlides() {
  console.log('\n📸 Capturing AIPERF-FLOW app slides...');
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });

  const slides = [];

  try {
    // Navigate to aiperf-flow - it's on 5188
    console.log('   Navigating to http://127.0.0.1:5188/');
    await page.goto('http://127.0.0.1:5188/', { waitUntil: 'networkidle', timeout: 30000 });
    await page.waitForTimeout(2000);

    // Dismiss audio consent dialog if present
    const audioConsentBtn = await page.locator('button:has-text("Play without audio")').first();
    if (await audioConsentBtn.isVisible({ timeout: 1000 }).catch(() => false)) {
      console.log('   Dismissing audio consent dialog...');
      await audioConsentBtn.click();
      await page.waitForTimeout(500);
    }

    // Click explainers button (📚 Explainers)
    console.log('   Looking for Explainers button...');
    const explainersBtn = await page.locator('button[aria-label*="explainer"], button:has-text("Explainers"), button:has-text("📚")').first();
    if (await explainersBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      console.log('   ✓ Clicking Explainers button...');
      await explainersBtn.click();
      await page.waitForTimeout(1500);
    }

    // Find and click Rust Architecture deck
    console.log('   Looking for Rust Architecture deck...');
    await page.waitForTimeout(500);

    // Try multiple selectors
    let rustArchBtn = await page.locator('button:has-text("Rust")').filter({ hasText: /Architecture/i }).first();
    if (!(await rustArchBtn.isVisible({ timeout: 500 }).catch(() => false))) {
      rustArchBtn = await page.locator('a:has-text("Rust")').filter({ hasText: /Architecture/i }).first();
    }
    if (!(await rustArchBtn.isVisible({ timeout: 500 }).catch(() => false))) {
      rustArchBtn = await page.locator('button, a').filter({ hasText: /Rust.*Architecture|architecture.*Rust/i }).first();
    }

    if (await rustArchBtn.isVisible({ timeout: 1000 }).catch(() => false)) {
      console.log('   ✓ Clicking Rust Architecture deck...');
      await rustArchBtn.click();
      await page.waitForTimeout(2000);
    } else {
      console.log('   ✗ Could not find Rust Architecture button');
      await browser.close();
      return slides;
    }

    // Capture slides
    for (let i = 0; i < 4; i++) {
      try {
        // Wait for slide to render
        await page.waitForTimeout(800);

        // Take screenshot
        const screenshot = await page.screenshot();
        slides.push({ app: 'aiperf-flow', slideNum: i + 1, data: screenshot });
        console.log(`   ✓ Captured slide ${i + 1}`);

        // Check for slide indicator
        const slideIndicator = await page.locator('text=/Slide \\d+ of \\d+/').textContent().catch(() => '');
        if (slideIndicator) {
          console.log(`     (${slideIndicator})`);
        }

        // Go to next slide
        const nextBtn = await page.locator('button:has-text("Next")').last();
        if (await nextBtn.isEnabled({ timeout: 500 }).catch(() => false)) {
          await nextBtn.click();
          await page.waitForTimeout(1500);
        } else {
          console.log(`   End of slides reached at slide ${i + 1}`);
          break;
        }
      } catch (error) {
        console.log(`   ✗ Error on slide ${i + 1}: ${error.message}`);
        break;
      }
    }

  } catch (error) {
    console.error('✗ Error capturing aiperf-flow:', error.message);
  } finally {
    await browser.close();
  }

  return slides;
}

async function main() {
  console.log('═'.repeat(80));
  console.log('SLIDE COMPARISON: Explainers vs AIPerf-Flow');
  console.log('═'.repeat(80));

  const explainersSlides = await captureExplainersSlides();
  const aiperfflowSlides = await captureAiPerfFlowSlides();

  // Save slides
  console.log('\n💾 Saving screenshots...');
  for (const slide of explainersSlides) {
    const filename = `/tmp/explainers-rust-arch-slide-${slide.slideNum}.png`;
    fs.writeFileSync(filename, slide.data);
    console.log(`   ✓ ${filename}`);
  }

  for (const slide of aiperfflowSlides) {
    const filename = `/tmp/aiperf-flow-rust-arch-slide-${slide.slideNum}.png`;
    fs.writeFileSync(filename, slide.data);
    console.log(`   ✓ ${filename}`);
  }

  // Compare slides
  console.log('\n' + '═'.repeat(80));
  console.log('PIXEL-BY-PIXEL COMPARISON');
  console.log('═'.repeat(80));

  const comparisons = [];
  const maxSlides = Math.max(explainersSlides.length, aiperfflowSlides.length);

  for (let i = 1; i <= maxSlides; i++) {
    const explainersFile = `/tmp/explainers-rust-arch-slide-${i}.png`;
    const aiperfFile = `/tmp/aiperf-flow-rust-arch-slide-${i}.png`;

    if (!fs.existsSync(explainersFile) || !fs.existsSync(aiperfFile)) {
      console.log(`\n❌ Slide ${i}: File missing`);
      if (!fs.existsSync(explainersFile)) console.log(`   Missing: ${explainersFile}`);
      if (!fs.existsSync(aiperfFile)) console.log(`   Missing: ${aiperfFile}`);
      continue;
    }

    try {
      const result = await comparePNGs(explainersFile, aiperfFile);
      comparisons.push({ slide: i, ...result });

      console.log(`\n📊 Slide ${i}:`);
      if (result.error) {
        console.log(`   ❌ Error: ${result.error}`);
      } else {
        console.log(`   Dimensions: ${result.width} × ${result.height}`);
        console.log(`   Diff pixels: ${result.diffPixels} / ${result.totalPixels}`);
        console.log(`   Difference: ${result.diff}%`);

        if (result.diff === 0) {
          console.log(`   Status: ✅ IDENTICAL (0% diff)`);
        } else if (result.diff < 1) {
          console.log(`   Status: ⚠️ Near-identical (<1% diff - minor rendering variations)`);
        } else if (result.diff < 5) {
          console.log(`   Status: 🔶 Similar (1-5% diff - some visual differences)`);
        } else if (result.diff < 10) {
          console.log(`   Status: 🔴 Different (5-10% diff - noticeable changes)`);
        } else {
          console.log(`   Status: ❌ Significantly different (>10% diff)`);
        }
      }
    } catch (error) {
      console.log(`\n❌ Slide ${i}: Error - ${error.message}`);
    }
  }

  // Summary
  console.log('\n' + '═'.repeat(80));
  console.log('SUMMARY');
  console.log('═'.repeat(80));
  console.log(`Slides compared: ${comparisons.length}`);

  const identical = comparisons.filter(c => c.diff === 0);
  const nearIdentical = comparisons.filter(c => c.diff > 0 && c.diff < 1);
  const similar = comparisons.filter(c => c.diff >= 1 && c.diff < 5);
  const different = comparisons.filter(c => c.diff >= 5 && c.diff < 10);
  const veryDifferent = comparisons.filter(c => c.diff >= 10);

  console.log(`  ✅ Identical (0% diff): ${identical.length} slide(s)`);
  console.log(`  ⚠️ Near-identical (<1% diff): ${nearIdentical.length} slide(s)`);
  console.log(`  🔶 Similar (1-5% diff): ${similar.length} slide(s)`);
  console.log(`  🔴 Different (5-10% diff): ${different.length} slide(s)`);
  console.log(`  ❌ Very different (≥10% diff): ${veryDifferent.length} slide(s)`);

  if (identical.length > 0) {
    console.log(`\n  ✅ Identical slides: ${identical.map(c => `#${c.slide}`).join(', ')}`);
  }
  if (nearIdentical.length > 0) {
    console.log(`  ⚠️ Near-identical slides: ${nearIdentical.map(c => `#${c.slide} (${c.diff}%)`).join(', ')}`);
  }
  if (similar.length > 0) {
    console.log(`  🔶 Similar slides: ${similar.map(c => `#${c.slide} (${c.diff}%)`).join(', ')}`);
  }
  if (different.length > 0) {
    console.log(`  🔴 Different slides: ${different.map(c => `#${c.slide} (${c.diff}%)`).join(', ')}`);
  }
  if (veryDifferent.length > 0) {
    console.log(`  ❌ Very different slides: ${veryDifferent.map(c => `#${c.slide} (${c.diff}%)`).join(', ')}`);
  }

  console.log('\n' + '═'.repeat(80));
}

main().catch(console.error);
