import { chromium } from 'playwright';

const baseDir = '/tmp';

(async () => {
  console.log('=== CORRECTED NAVIGATION - ENTERING ACTUAL DECK SLIDES ===\n');
  
  // LEGACY APP
  console.log('LEGACY EXPLAINERS (http://localhost:5173)');
  const browser1 = await chromium.launch({ headless: true });
  const page1 = await browser1.newPage();
  page1.setViewportSize({ width: 1280, height: 800 });
  
  await page1.goto('http://localhost:5173', { waitUntil: 'networkidle', timeout: 15000 });
  await page1.waitForTimeout(800);
  
  // Click on Rust architecture card
  const cards = await page1.$$('div[role="button"], a');
  for (const card of cards) {
    const text = await card.textContent();
    if (text.includes('Rust architecture from scratch')) {
      console.log('Clicking Rust architecture card');
      await card.click();
      break;
    }
  }
  
  await page1.waitForTimeout(1200);
  
  // Take actual slides
  await page1.screenshot({ path: `${baseDir}/final-legacy-slide-1.png`, fullPage: false });
  console.log('✓ Slide 1');
  
  await page1.keyboard.press('ArrowRight');
  await page1.waitForTimeout(600);
  await page1.screenshot({ path: `${baseDir}/final-legacy-slide-2.png`, fullPage: false });
  console.log('✓ Slide 2');
  
  await page1.keyboard.press('ArrowRight');
  await page1.waitForTimeout(600);
  await page1.screenshot({ path: `${baseDir}/final-legacy-slide-3.png`, fullPage: false });
  console.log('✓ Slide 3');
  
  await browser1.close();
  
  // AIPERF-FLOW APP: Must click "VIEW DECK" button
  console.log('\nAIPERF-FLOW (http://localhost:5188)');
  const browser2 = await chromium.launch({ headless: true });
  const page2 = await browser2.newPage();
  page2.setViewportSize({ width: 1280, height: 800 });
  
  await page2.goto('http://localhost:5188', { waitUntil: 'networkidle', timeout: 15000 });
  await page2.waitForTimeout(800);
  
  // Dismiss audio consent if present
  try {
    const audioButtons = await page2.$$('button');
    for (const btn of audioButtons) {
      const text = await btn.textContent();
      if (text.includes('Play without audio')) {
        console.log('Dismissing audio dialog');
        await btn.click({ force: true });
        break;
      }
    }
  } catch (e) {}
  
  await page2.waitForTimeout(800);
  
  // Navigate to explainers
  const allElements = await page2.$$('a, button, div[role="button"]');
  for (const el of allElements) {
    const text = await el.textContent();
    if (text.includes('Explainers')) {
      console.log('Clicking Explainers');
      try {
        await el.click({ force: true });
        break;
      } catch (e) {}
    }
  }
  
  await page2.waitForTimeout(1200);
  
  // Click "VIEW DECK" button for Rust Architecture
  const buttons = await page2.$$('button');
  let clickedViewDeck = false;
  for (let i = 0; i < buttons.length; i++) {
    const text = await buttons[i].textContent();
    if (text.includes('VIEW DECK')) {
      // Get the parent card text to verify it's Rust Architecture
      const parent = await buttons[i].evaluate(el => el.closest('[class*="card"], div[class*="Card"]')?.textContent || '');
      if (parent.includes('Rust Architecture')) {
        console.log('Clicking VIEW DECK for Rust Architecture');
        await buttons[i].click({ force: true });
        clickedViewDeck = true;
        break;
      }
    }
  }
  
  if (!clickedViewDeck) {
    console.log('Could not find VIEW DECK button, trying first VIEW DECK');
    for (const btn of buttons) {
      const text = await btn.textContent();
      if (text.includes('VIEW DECK')) {
        await btn.click({ force: true });
        break;
      }
    }
  }
  
  await page2.waitForTimeout(1500);
  
  // Handle audio preference if it appears
  try {
    const audioButtons2 = await page2.$$('button');
    for (const btn of audioButtons2) {
      const text = await btn.textContent();
      if (text.includes('Play without audio')) {
        console.log('Dismissing audio dialog again');
        await btn.click({ force: true });
        break;
      }
    }
  } catch (e) {}
  
  await page2.waitForTimeout(1200);
  
  // Take slides
  await page2.screenshot({ path: `${baseDir}/final-aiperf-slide-1.png`, fullPage: false });
  console.log('✓ Slide 1');
  
  await page2.keyboard.press('ArrowRight');
  await page2.waitForTimeout(600);
  await page2.screenshot({ path: `${baseDir}/final-aiperf-slide-2.png`, fullPage: false });
  console.log('✓ Slide 2');
  
  await page2.keyboard.press('ArrowRight');
  await page2.waitForTimeout(600);
  await page2.screenshot({ path: `${baseDir}/final-aiperf-slide-3.png`, fullPage: false });
  console.log('✓ Slide 3');
  
  await browser2.close();
  
  console.log('\n=== READY FOR FINAL COMPARISON ===');
})();
