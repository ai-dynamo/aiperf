import { chromium } from 'playwright';
import fs from 'fs';

const baseDir = '/tmp';

async function screenshot(name, url, selector, delay = 1500) {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  page.setViewportSize({ width: 1280, height: 800 });
  
  try {
    console.log(`Loading ${url}...`);
    await page.goto(url, { waitUntil: 'networkidle', timeout: 15000 });
    await page.waitForTimeout(delay);
    
    const path = `${baseDir}/auth-${name}.png`;
    await page.screenshot({ path, fullPage: false });
    console.log(`✓ ${name}: ${path}`);
    return path;
  } catch (e) {
    console.error(`✗ ${name}: ${e.message}`);
  } finally {
    await browser.close();
  }
}

(async () => {
  try {
    console.log('=== LEGACY EXPLAINERS APP (http://localhost:5173) ===');
    await screenshot('explainers-home', 'http://localhost:5173', 'main');
    
    const browser1 = await chromium.launch({ headless: true });
    const page1 = await browser1.newPage();
    page1.setViewportSize({ width: 1280, height: 800 });
    
    console.log('\nNavigating to Rust Architecture in explainers...');
    await page1.goto('http://localhost:5173', { waitUntil: 'networkidle', timeout: 15000 });
    await page1.waitForTimeout(800);
    
    // Look for link to Rust Architecture
    const allText = await page1.evaluate(() => document.body.innerText);
    console.log('Page text preview:', allText.substring(0, 200));
    
    try {
      // Try finding and clicking Rust Architecture link
      const link = page1.locator('a, button').filter({ hasText: /Rust.*Architecture|Architecture.*Rust/ }).first();
      if (await link.isVisible()) {
        console.log('Found Rust Architecture link, clicking...');
        await link.click();
        await page1.waitForTimeout(1000);
      }
    } catch (e) {
      console.log('Could not find direct link, trying alternate selectors');
    }
    
    await page1.screenshot({ path: `${baseDir}/auth-explainers-slide-1.png`, fullPage: false });
    console.log(`✓ explainers-slide-1: ${baseDir}/auth-explainers-slide-1.png`);
    
    // Try arrow key navigation
    await page1.keyboard.press('ArrowRight');
    await page1.waitForTimeout(600);
    await page1.screenshot({ path: `${baseDir}/auth-explainers-slide-2.png`, fullPage: false });
    console.log(`✓ explainers-slide-2: ${baseDir}/auth-explainers-slide-2.png`);
    
    await page1.keyboard.press('ArrowRight');
    await page1.waitForTimeout(600);
    await page1.screenshot({ path: `${baseDir}/auth-explainers-slide-3.png`, fullPage: false });
    console.log(`✓ explainers-slide-3: ${baseDir}/auth-explainers-slide-3.png`);
    
    await browser1.close();
    
    console.log('\n=== AIPERF-FLOW APP (http://localhost:5188) ===');
    await screenshot('aiperf-home', 'http://localhost:5188', 'main');
    
    const browser2 = await chromium.launch({ headless: true });
    const page2 = await browser2.newPage();
    page2.setViewportSize({ width: 1280, height: 800 });
    
    console.log('\nNavigating to Rust Architecture in aiperf-flow...');
    await page2.goto('http://localhost:5188', { waitUntil: 'networkidle', timeout: 15000 });
    await page2.waitForTimeout(800);
    
    // Click on Explainers section
    try {
      const explainersLink = page2.locator('a, button').filter({ hasText: /Explainers|📚/ }).first();
      if (await explainersLink.isVisible()) {
        console.log('Found Explainers link, clicking...');
        await explainersLink.click();
        await page2.waitForTimeout(1000);
      }
    } catch (e) {
      console.log('Could not find Explainers link');
    }
    
    // Click on Rust Architecture
    try {
      const rustLink = page2.locator('a, button').filter({ hasText: /Rust.*Architecture|Architecture.*Rust/ }).first();
      if (await rustLink.isVisible()) {
        console.log('Found Rust Architecture link, clicking...');
        await rustLink.click();
        await page2.waitForTimeout(1000);
      }
    } catch (e) {
      console.log('Could not find Rust Architecture link');
    }
    
    await page2.screenshot({ path: `${baseDir}/auth-aiperf-slide-1.png`, fullPage: false });
    console.log(`✓ aiperf-slide-1: ${baseDir}/auth-aiperf-slide-1.png`);
    
    await page2.keyboard.press('ArrowRight');
    await page2.waitForTimeout(600);
    await page2.screenshot({ path: `${baseDir}/auth-aiperf-slide-2.png`, fullPage: false });
    console.log(`✓ aiperf-slide-2: ${baseDir}/auth-aiperf-slide-2.png`);
    
    await page2.keyboard.press('ArrowRight');
    await page2.waitForTimeout(600);
    await page2.screenshot({ path: `${baseDir}/auth-aiperf-slide-3.png`, fullPage: false });
    console.log(`✓ aiperf-slide-3: ${baseDir}/auth-aiperf-slide-3.png`);
    
    await browser2.close();
    
    console.log('\n=== SCREENSHOTS COMPLETE ===');
    console.log('Files ready for pixelmatch comparison');
    
  } catch (e) {
    console.error('Fatal error:', e);
    process.exit(1);
  }
})();
