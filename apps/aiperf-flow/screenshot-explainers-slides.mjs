#!/usr/bin/env node

import { chromium } from '@playwright/test';
import fs from 'fs';

async function captureExplainersAppSlides() {
  console.log('Capturing explainers app slides...');
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2 });

  const slides = [];

  try {
    // Navigate to explainers app
    await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' });
    await page.waitForTimeout(1500);

    // Click "Rust architecture from scratch"
    const buttons = await page.locator('button, a').all();
    let clicked = false;
    for (const btn of buttons) {
      const text = await btn.textContent();
      if (text && text.toLowerCase().includes('rust') && text.toLowerCase().includes('architecture')) {
        console.log(`  Clicking: "${text}"`);
        await btn.click();
        clicked = true;
        await page.waitForTimeout(2000);
        break;
      }
    }

    if (!clicked) {
      console.error('  Could not find Rust architecture button');
      return slides;
    }

    // Dismiss audio dialog and capture slides
    for (let i = 0; i < 4; i++) {
      // Dismiss audio dialog if present
      const audioBtn = await page.locator('button:has-text("Play without audio")').first();
      if (await audioBtn.isVisible({ timeout: 1000 }).catch(() => false)) {
        await audioBtn.click();
        await page.waitForTimeout(1000);
      }

      // Wait for slide to render
      await page.waitForTimeout(800);

      // Take screenshot
      const screenshot = await page.screenshot();
      slides.push({ app: 'explainers', slideNum: i + 1, data: screenshot });
      console.log(`  ✓ Captured slide ${i + 1}`);

      // Go to next slide
      const nextBtn = await page.locator('button, a').filter({ hasText: /Next|→|>/ }).first();
      if (await nextBtn.isEnabled({ timeout: 1000 }).catch(() => false)) {
        await nextBtn.click();
        await page.waitForTimeout(1500);
      } else {
        break; // No more slides
      }
    }

  } catch (error) {
    console.error('Error capturing explainers app:', error.message);
  } finally {
    await browser.close();
  }

  return slides;
}

async function captureAiPerfFlowSlides() {
  console.log('Capturing aiperf-flow explainer slides...');
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2 });

  const slides = [];

  try {
    // Navigate to aiperf-flow
    await page.goto('http://127.0.0.1:5188/', { waitUntil: 'networkidle' });
    await page.waitForTimeout(2000);

    // Dismiss audio consent dialog if present
    const audioConsentBtn = await page.locator('button:has-text("Play without audio")').first();
    if (await audioConsentBtn.isVisible({ timeout: 1000 }).catch(() => false)) {
      console.log('  Dismissing audio consent dialog...');
      await audioConsentBtn.click();
      await page.waitForTimeout(500);
    }

    // Click explainers button
    const explainersBtn = await page.locator('button[aria-label="Open explainer decks"]').first();
    if (await explainersBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      console.log('  Clicking explainers button...');
      await explainersBtn.click();
      await page.waitForTimeout(1500);
    }

    // Find and click Rust Architecture deck
    await page.waitForTimeout(500);
    const rustArchBtn = await page.locator('button, a').filter({ hasText: /Rust.*Architecture/i }).first();
    if (await rustArchBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      console.log('  Clicking Rust Architecture deck...');
      await rustArchBtn.click();
      await page.waitForTimeout(2000);
    }

    // Capture slides
    for (let i = 0; i < 4; i++) {
      // Wait for slide to render
      await page.waitForTimeout(800);

      // Take screenshot
      const screenshot = await page.screenshot();
      slides.push({ app: 'aiperf-flow', slideNum: i + 1, data: screenshot });
      console.log(`  ✓ Captured slide ${i + 1}`);

      // Go to next slide
      const nextBtn = await page.locator('button:has-text("Next")').last();
      if (await nextBtn.isEnabled({ timeout: 1000 }).catch(() => false)) {
        await nextBtn.click();
        await page.waitForTimeout(1500);
      } else {
        break;
      }
    }

  } catch (error) {
    console.error('Error capturing aiperf-flow:', error.message);
  } finally {
    await browser.close();
  }

  return slides;
}

async function main() {
  console.log('Starting explainer slide capture...\n');

  const explainersSlides = await captureExplainersAppSlides();
  console.log('');
  const aiperfflowSlides = await captureAiPerfFlowSlides();

  // Save slides
  console.log('\nSaving screenshots...');
  for (const slide of explainersSlides) {
    const filename = `/tmp/explainers-slide-${slide.slideNum}.png`;
    fs.writeFileSync(filename, slide.data);
    console.log(`  ✓ ${filename}`);
  }

  for (const slide of aiperfflowSlides) {
    const filename = `/tmp/aiperf-flow-slide-${slide.slideNum}.png`;
    fs.writeFileSync(filename, slide.data);
    console.log(`  ✓ ${filename}`);
  }

  console.log('\n✓ All slides captured');
}

main().catch(console.error);
