#!/usr/bin/env node

import { chromium } from '@playwright/test';
import fs from 'fs';

async function captureExplainersAppSlide() {
  console.log('Capturing explainers app...');
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2 });

  try {
    // Navigate to explainers app
    await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' });
    await page.waitForTimeout(1500);

    // Click "Rust architecture from scratch"
    const rustBtn = await page.locator('button, a').filter({ hasText: /Rust.*architecture.*scratch|architecture.*from.*scratch/ }).first();

    if (await rustBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      console.log('  Clicking Rust architecture button...');
      await rustBtn.click();
      await page.waitForTimeout(2000);
    } else {
      console.log('  Could not find/click Rust architecture button, using all buttons');
      const buttons = await page.locator('button, a').all();
      for (const btn of buttons) {
        const text = await btn.textContent();
        if (text && text.toLowerCase().includes('rust') && text.toLowerCase().includes('architecture')) {
          console.log(`  Found button: "${text}"`);
          await btn.click();
          await page.waitForTimeout(2000);
          break;
        }
      }
    }

    // Wait for slide content
    await page.waitForTimeout(1000);

    // Take screenshot
    const screenshot = await page.screenshot({ path: '/tmp/explainers-app-rust-arch-slide-1.png' });
    console.log('✓ Explainers app screenshot: /tmp/explainers-app-rust-arch-slide-1.png');

  } catch (error) {
    console.error('Error capturing explainers app:', error.message);
  } finally {
    await browser.close();
  }
}

async function captureAiPerfFlowExplainer() {
  console.log('Capturing aiperf-flow explainer...');
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2 });

  try {
    // Navigate to aiperf-flow
    await page.goto('http://127.0.0.1:5188/', { waitUntil: 'networkidle' });
    await page.waitForTimeout(2000);

    // Dismiss audio consent dialog if present
    try {
      const audioConsentBtn = await page.locator('button:has-text("Play without audio")').first();
      if (await audioConsentBtn.isVisible({ timeout: 1000 }).catch(() => false)) {
        console.log('  Dismissing audio consent dialog...');
        await audioConsentBtn.click();
        await page.waitForTimeout(500);
      }
    } catch (e) {
      // No audio consent dialog, continue
    }

    // Look for explainers button
    const explainersBtn = await page.locator('button[aria-label="Open explainer decks"]').first();
    if (await explainersBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      console.log('  Clicking explainers button...');
      await explainersBtn.click();
      await page.waitForTimeout(1500);
    } else {
      console.log('  Could not find explainers button');
    }

    // Now look for and click Rust Architecture deck
    await page.waitForTimeout(500);
    const rustArchBtn = await page.locator('button, a').filter({ hasText: /Rust.*Architecture/i }).first();
    if (await rustArchBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      console.log('  Clicking Rust Architecture deck...');
      await rustArchBtn.click();
      await page.waitForTimeout(2000);
    } else {
      console.log('  Could not find Rust Architecture deck');
    }

    // Wait for slide to render
    await page.waitForTimeout(1000);

    // Take screenshot
    const screenshot = await page.screenshot({ path: '/tmp/aiperf-flow-rust-arch-slide-1.png' });
    console.log('✓ AIPerf Flow screenshot: /tmp/aiperf-flow-rust-arch-slide-1.png');

  } catch (error) {
    console.error('Error capturing aiperf-flow:', error.message);
  } finally {
    await browser.close();
  }
}

async function main() {
  console.log('Starting explainer screenshot comparison...\n');

  await captureExplainersAppSlide();
  console.log('');
  await captureAiPerfFlowExplainer();

  console.log('\n✓ Screenshots captured successfully');
  console.log('Files created:');
  console.log('  /tmp/explainers-app-rust-arch-slide-1.png');
  console.log('  /tmp/aiperf-flow-rust-arch-slide-1.png');
}

main().catch(console.error);
