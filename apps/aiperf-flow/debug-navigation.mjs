#!/usr/bin/env node

import { chromium } from '@playwright/test';

async function debugNavigation() {
  console.log('Debugging aiperf-flow navigation...\n');
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });

  try {
    // Navigate to aiperf-flow
    console.log('Navigating to http://127.0.0.1:5188/');
    await page.goto('http://127.0.0.1:5188/', { waitUntil: 'networkidle', timeout: 30000 });
    await page.waitForTimeout(2000);

    // Dismiss audio consent dialog if present
    const audioConsentBtn = await page.locator('button:has-text("Play without audio")').first();
    if (await audioConsentBtn.isVisible({ timeout: 1000 }).catch(() => false)) {
      console.log('Dismissing audio consent dialog...');
      await audioConsentBtn.click();
      await page.waitForTimeout(500);
    }

    // Click explainers button
    console.log('Clicking Explainers button...');
    const explainersBtn = await page.locator('button[aria-label*="explainer"], button:has-text("Explainers"), button:has-text("📚")').first();
    if (await explainersBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      await explainersBtn.click();
      await page.waitForTimeout(1500);
    }

    // Find and click Rust Architecture deck
    console.log('Clicking Rust Architecture deck...');
    let rustArchBtn = await page.locator('button:has-text("Rust")').filter({ hasText: /Architecture/i }).first();
    if (await rustArchBtn.isVisible({ timeout: 500 }).catch(() => false)) {
      await rustArchBtn.click();
      await page.waitForTimeout(2000);
    }

    // Now debug the presentation state
    console.log('\n=== INITIAL STATE ===');
    await inspectState(page);

    // Click Next 4 times and inspect state each time
    for (let i = 0; i < 4; i++) {
      console.log(`\n=== CLICKING NEXT (attempt ${i + 1}) ===`);

      const nextBtn = await page.locator('button:has-text("Next")').last();
      console.log(`Next button found: ${nextBtn ? 'yes' : 'no'}`);

      if (nextBtn) {
        const isEnabled = await nextBtn.isEnabled({ timeout: 500 }).catch(() => false);
        const isVisible = await nextBtn.isVisible({ timeout: 500 }).catch(() => false);
        console.log(`Next button enabled: ${isEnabled}, visible: ${isVisible}`);

        if (isEnabled && isVisible) {
          await nextBtn.click();
          console.log('Clicked Next button');
        } else {
          console.log('Next button not enabled or visible - stopping');
          break;
        }
      }

      await page.waitForTimeout(1500);
      await inspectState(page);
    }

  } catch (error) {
    console.error('Error:', error.message);
  } finally {
    await browser.close();
  }
}

async function inspectState(page) {
  // Get slide indicator text
  const slideIndicator = await page.locator('text=/Slide \\d+ of \\d+/').textContent().catch(() => 'not found');
  console.log(`Slide indicator: ${slideIndicator}`);

  // Get slide content title/heading
  const heading = await page.locator('h1, h2').first().textContent().catch(() => 'not found');
  console.log(`Main heading: ${heading}`);

  // Check what's currently in the main diagram area
  const whiteBox = await page.locator('text=/aiperf binary|profile parent|request-rate|warmup/').first().textContent().catch(() => 'not found');
  console.log(`Key content: ${whiteBox?.substring(0, 50) || 'not found'}`);

  // List all buttons to understand navigation
  const buttons = await page.locator('button').all();
  const navButtons = [];
  for (const btn of buttons) {
    const text = await btn.textContent().catch(() => '');
    if (text && (text.includes('Next') || text.includes('Previous') || text.includes('→') || text.includes('←'))) {
      navButtons.push(text);
    }
  }
  console.log(`Nav buttons visible: ${navButtons.join(', ') || 'none'}`);

  // Check for any data attributes or IDs that might indicate current slide
  const presentationArea = await page.locator('[data-slide], [data-step], [data-index]').first();
  if (presentationArea) {
    const attrs = await presentationArea.evaluate(el => {
      return Array.from(el.attributes)
        .filter(attr => attr.name.includes('data') || attr.name.includes('aria'))
        .map(attr => `${attr.name}="${attr.value}"`)
        .join(', ');
    }).catch(() => 'no attrs');
    console.log(`Presentation area attrs: ${attrs}`);
  }
}

debugNavigation().catch(console.error);
