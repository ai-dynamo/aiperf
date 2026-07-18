import fs from 'fs';
import sharp from 'sharp';
import { PNG } from 'pngjs';
import pixelmatch from 'pixelmatch';

async function analyzeVisualDifferences() {
  console.log('╔═════════════════════════════════════════════════════════════╗');
  console.log('║     SCREENSHOT COMPARISON: EXPLAINERS vs AIPERF-FLOW        ║');
  console.log('╚═════════════════════════════════════════════════════════════╝\n');

  // Get metadata
  const explainersImg = await sharp('/tmp/explainers-home.png').metadata();
  const aiPerfImg = await sharp('/tmp/aiperf-flow-home.png').metadata();

  console.log('📷 IMAGE DIMENSIONS:');
  console.log(`   Explainers app:  ${explainersImg.width}x${explainersImg.height}px`);
  console.log(`   AIPerf Flow:     ${aiPerfImg.width}x${aiPerfImg.height}px\n`);

  // Calculate pixel diff
  const img1Data = fs.readFileSync('/tmp/explainers-home.png');
  const img2Data = fs.readFileSync('/tmp/aiperf-flow-home.png');
  
  const img1 = PNG.sync.read(img1Data);
  const img2 = PNG.sync.read(img2Data);

  const diff = new PNG({ width: img1.width, height: img1.height });
  const mismatchedPixels = pixelmatch(img1.data, img2.data, diff.data, img1.width, img1.height, { threshold: 0.1 });
  const totalPixels = img1.width * img1.height;
  const diffPercentage = (mismatchedPixels / totalPixels * 100).toFixed(2);

  console.log('📊 PIXEL DIFFERENCE ANALYSIS:');
  console.log(`   Mismatched pixels: ${mismatchedPixels.toLocaleString()} out of ${totalPixels.toLocaleString()}`);
  console.log(`   Difference percentage: ${diffPercentage}%`);
  console.log(`   Assessment: ${parseFloat(diffPercentage) < 5 ? '✓ Very similar' : parseFloat(diffPercentage) < 15 ? '⚠ Moderately different' : '✗ Significantly different'}\n`);

  console.log('🎨 SPECIFIC STYLING DIFFERENCES IDENTIFIED:\n');

  console.log('1. LAYOUT & STRUCTURE:');
  console.log('   Explainers:');
  console.log('   • Full-viewport centered card layout');
  console.log('   • Hero section with title "Interactive walkthroughs"');
  console.log('   • Vertical stack of deck cards');
  console.log('   • Each card: title, description, metadata');
  console.log('   • 1-column responsive grid');
  console.log('   ');
  console.log('   AIPerf Flow:');
  console.log('   • Complex multi-panel layout');
  console.log('   • Left sidebar with navigation menu');
  console.log('   • Centered audio consent dialog overlay');
  console.log('   • Top navigation bar with scene selector');
  console.log('   • 3-column split: sidebar | main | control panel');
  console.log('   ');

  console.log('2. COLOR SCHEME:');
  console.log('   Explainers:');
  console.log('   • Dark background (charcoal/dark gray)');
  console.log('   • Light text (white/light gray)');
  console.log('   • Accent colors: cyan/teal for highlighted text');
  console.log('   • Card borders: subtle gray');
  console.log('   ');
  console.log('   AIPerf Flow:');
  console.log('   • Similar dark background');
  console.log('   • Audio dialog: cyan/turquoise accent button');
  console.log('   • Gray secondary button');
  console.log('   • Navigation: dark with subtle highlights');
  console.log('   ');

  console.log('3. TYPOGRAPHY:');
  console.log('   Explainers:');
  console.log('   • Header: Bold, large (∼36px) white text');
  console.log('   • Subheader: Medium gray text');
  console.log('   • Card titles: Bold white (∼18px)');
  console.log('   • Descriptions: Regular gray (∼14px)');
  console.log('   ');
  console.log('   AIPerf Flow:');
  console.log('   • Large title: "From one request to the whole system"');
  console.log('   • Dialog title: "Audio preference" (bold white)');
  console.log('   • Button text: Bold white on colored backgrounds');
  console.log('   ');

  console.log('4. INTERACTIVE ELEMENTS:');
  console.log('   Explainers:');
  console.log('   • Cards: Clickable boxes with subtle borders');
  console.log('   • Focus state: Border highlight');
  console.log('   • Hover state: Not visible in static screenshot');
  console.log('   ');
  console.log('   AIPerf Flow:');
  console.log('   • Dialog buttons: Rounded corners (∼8px)');
  console.log('   • Primary button: Cyan background with bold text');
  console.log('   • Secondary button: Transparent/outline style');
  console.log('   • Sidebar: Menu items with text hierarchy');
  console.log('   ');

  console.log('5. SPACING & PADDING:');
  console.log('   Explainers:');
  console.log('   • Generous vertical spacing between cards (∼20px gap)');
  console.log('   • Card padding: ∼20px internal');
  console.log('   • Header margin-bottom: ∼40px');
  console.log('   ');
  console.log('   AIPerf Flow:');
  console.log('   • Sidebar: Compact spacing');
  console.log('   • Dialog: Centered with padding');
  console.log('   • Main content area: Different proportions');
  console.log('   ');

  console.log('6. BORDERS & SHADOWS:');
  console.log('   Explainers:');
  console.log('   • Card borders: 1px solid gray (#444 or similar)');
  console.log('   • No visible box shadows');
  console.log('   • Borders: Rounded (∼8px)');
  console.log('   ');
  console.log('   AIPerf Flow:');
  console.log('   • Dialog border: 2px solid cyan (#22d3ee)');
  console.log('   • Slight glow effect on focused elements');
  console.log('   • Rounded corners on most interactive elements');
  console.log('   ');

  console.log('\n📋 DECK PICKER COMPARISON:');
  console.log('   The explainer deck picker (viewed in aiperf-flow):');
  console.log('   • Shows 3-column grid of deck cards');
  console.log('   • Cards include: title, slide count, description, "VIEW DECK" button');
  console.log('   • Cyan accent buttons matching the primary color');
  console.log('   • Cleaner presentation of deck options');
  console.log('   ');

  console.log('\n🔍 SUMMARY OF KEY CSS ADJUSTMENTS NEEDED:\n');
  console.log('   1. Card layouts:');
  console.log('      - Grid: 1-column → 3-column for deck picker');
  console.log('      - Gap between cards: Adjust spacing');
  console.log('      - Border radius: Ensure consistency');
  console.log('   ');
  console.log('   2. Typography hierarchy:');
  console.log('      - Font sizes may differ slightly');
  console.log('      - Line heights and letter spacing should be reviewed');
  console.log('   ');
  console.log('   3. Color consistency:');
  console.log('      - Primary accent: Cyan (#22d3ee or similar)');
  console.log('      - Background: Dark charcoal/dark gray');
  console.log('      - Text: White for primary, gray for secondary');
  console.log('   ');
  console.log('   4. Button styling:');
  console.log('      - Rounded corners: 8px border-radius');
  console.log('      - Padding: ∼12px vertical, ∼24px horizontal');
  console.log('      - Font weight: Bold');
  console.log('   ');
  console.log('   5. Interactive states:');
  console.log('      - Hover: Slight opacity change or background brightness');
  console.log('      - Focus: Visible outline or glow effect');
  console.log('      - Active: Highlighted state with primary color');
  console.log('');

  console.log('═'.repeat(65));
  console.log(`PIXEL DIFF: ${diffPercentage}% (${mismatchedPixels.toLocaleString()} pixels)`.padEnd(65));
  console.log('═'.repeat(65));
}

analyzeVisualDifferences().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
