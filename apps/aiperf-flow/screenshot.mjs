import { chromium } from 'playwright';

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

async function takeScreenshots() {
  const browser = await chromium.launch({ headless: true });

  try {
    console.log('\nTaking screenshot of aiperf-flow explainer deck picker...');
    const page2 = await browser.newPage({
      viewport: { width: 1280, height: 1024 }
    });
    await page2.goto('http://127.0.0.1:5188/', { waitUntil: 'networkidle' });
    await sleep(2000);

    // Dismiss audio consent backdrop if present
    try {
      const backdrop = page2.locator('.aiperf-flow__audio-consent-backdrop');
      const dismissButton = backdrop.locator('button').first();
      if (await dismissButton.isVisible()) {
        console.log('Dismissing audio consent...');
        await dismissButton.click();
        await sleep(500);
      }
    } catch (e) {
      console.log('No audio consent backdrop found');
    }

    // Look for the "Explainers" button and click it
    const buttons = await page2.locator('button').all();
    let found = false;
    for (let i = 0; i < buttons.length; i++) {
      const text = await buttons[i].textContent();
      if (text && (text.includes('Explainers') || text.includes('📚'))) {
        console.log('Found Explainers button, clicking...');
        await buttons[i].click({ force: true });
        await sleep(1500);
        await page2.screenshot({ path: '/tmp/aiperf-flow-deck-picker.png', fullPage: false });
        console.log('✓ Saved /tmp/aiperf-flow-deck-picker.png');
        found = true;
        break;
      }
    }
    if (!found) {
      console.log('⚠ Explainers button not found');
    }
    await page2.close();

  } finally {
    await browser.close();
  }
}

takeScreenshots().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
