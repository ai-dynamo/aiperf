import { chromium } from 'playwright';

const baseDir = '/tmp';

(async () => {
  console.log('=== PROPER NAVIGATION TO RUST ARCHITECTURE DECK ===\n');
  
  // LEGACY APP: Click on Rust architecture from scratch
  console.log('LEGACY EXPLAINERS (http://localhost:5173)');
  const browser1 = await chromium.launch({ headless: true });
  const page1 = await browser1.newPage();
  page1.setViewportSize({ width: 1280, height: 800 });
  
  await page1.goto('http://localhost:5173', { waitUntil: 'networkidle', timeout: 15000 });
  await page1.waitForTimeout(800);
  
  // Find and click on Rust architecture card
  const cards = await page1.$$('div[role="button"], a');
  for (const card of cards) {
    const text = await card.textContent();
    if (text.includes('Rust architecture from scratch')) {
      console.log('Found Rust architecture card');
      await card.click();
      break;
    }
  }
  
  await page1.waitForTimeout(1200);
  
  // Now we should be on the deck - take slide 1
  await page1.screenshot({ path: `${baseDir}/legacy-rust-slide-1.png`, fullPage: false });
  console.log('✓ Captured slide 1');
  
  await page1.keyboard.press('ArrowRight');
  await page1.waitForTimeout(600);
  await page1.screenshot({ path: `${baseDir}/legacy-rust-slide-2.png`, fullPage: false });
  console.log('✓ Captured slide 2');
  
  await page1.keyboard.press('ArrowRight');
  await page1.waitForTimeout(600);
  await page1.screenshot({ path: `${baseDir}/legacy-rust-slide-3.png`, fullPage: false });
  console.log('✓ Captured slide 3');
  
  await browser1.close();
  
  // AIPERF-FLOW APP: Navigate to Rust architecture
  console.log('\nAIPERF-FLOW (http://localhost:5188)');
  const browser2 = await chromium.launch({ headless: true });
  const page2 = await browser2.newPage();
  page2.setViewportSize({ width: 1280, height: 800 });
  
  await page2.goto('http://localhost:5188', { waitUntil: 'networkidle', timeout: 15000 });
  await page2.waitForTimeout(800);
  
  // First, dismiss any audio consent dialog
  try {
    const audioButtons = await page2.$$('button');
    for (const btn of audioButtons) {
      const text = await btn.textContent();
      if (text.includes('Play without audio')) {
        console.log('Audio dialog appeared, clicking "Play without audio"');
        await btn.click({ force: true });
        break;
      }
    }
  } catch (e) {
    console.log('No audio dialog to dismiss');
  }
  
  await page2.waitForTimeout(800);
  
  // Try to navigate to explainers section
  const allElements = await page2.$$('a, button, div[role="button"]');
  let foundExplainers = false;
  for (const el of allElements) {
    const text = await el.textContent();
    if (text.includes('Explainers') || text.includes('📚')) {
      console.log('Found Explainers link');
      try {
        await el.click({ force: true });
        foundExplainers = true;
        break;
      } catch (e) {
        // Try next one
      }
    }
  }
  
  if (foundExplainers) {
    await page2.waitForTimeout(1200);
  }
  
  // Click on Rust architecture card
  const allElementsAgain = await page2.$$('div, a, button');
  for (const el of allElementsAgain) {
    const text = await el.textContent();
    if (text.includes('Rust architecture from scratch')) {
      console.log('Found Rust architecture card');
      try {
        await el.click({ force: true });
        break;
      } catch (e) {
        // Try next
      }
    }
  }
  
  await page2.waitForTimeout(1500);
  
  // Handle audio dialog that may appear again
  try {
    const audioButtons2 = await page2.$$('button');
    for (const btn of audioButtons2) {
      const text = await btn.textContent();
      if (text.includes('Play without audio')) {
        console.log('Audio dialog appeared again, clicking "Play without audio"');
        await btn.click({ force: true });
        break;
      }
    }
  } catch (e) {}
  
  await page2.waitForTimeout(1200);
  
  // Take slide 1
  await page2.screenshot({ path: `${baseDir}/aiperf-rust-slide-1.png`, fullPage: false });
  console.log('✓ Captured slide 1');
  
  await page2.keyboard.press('ArrowRight');
  await page2.waitForTimeout(600);
  await page2.screenshot({ path: `${baseDir}/aiperf-rust-slide-2.png`, fullPage: false });
  console.log('✓ Captured slide 2');
  
  await page2.keyboard.press('ArrowRight');
  await page2.waitForTimeout(600);
  await page2.screenshot({ path: `${baseDir}/aiperf-rust-slide-3.png`, fullPage: false });
  console.log('✓ Captured slide 3');
  
  await browser2.close();
  
  console.log('\n=== SCREENSHOTS READY FOR PIXEL COMPARISON ===');
})();
