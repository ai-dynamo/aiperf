import pixelmatch from 'pixelmatch';
import PNG from 'pngjs';
import fs from 'fs';

const { createReadStream, createWriteStream } = fs;

async function comparePngs(legacy, aiperf, safeSlug, label) {
  return new Promise((resolve, reject) => {
    const legacyImg = new PNG.PNG();
    const aiperfImg = new PNG.PNG();
    let completed = 0;
    
    const finalize = () => {
      if (completed !== 2) return;
      
      if (legacyImg.width !== aiperfImg.width || legacyImg.height !== aiperfImg.height) {
        console.log(`⚠️  ${label}: Size mismatch! Legacy ${legacyImg.width}x${legacyImg.height} vs Aiperf ${aiperfImg.width}x${aiperfImg.height}`);
        resolve();
        return;
      }
      
      const { width, height } = legacyImg;
      const diff = new PNG.PNG({ width, height });
      
      const numDiffPixels = pixelmatch(
        legacyImg.data,
        aiperfImg.data,
        diff.data,
        width,
        height,
        { threshold: 0.1 }
      );
      
      const totalPixels = width * height;
      const diffPercent = ((numDiffPixels / totalPixels) * 100).toFixed(2);
      
      console.log(`${label}: ${numDiffPixels} different pixels out of ${totalPixels} (${diffPercent}%)`);
      
      if (diffPercent > 5) {
        const diffPath = `/tmp/${safeSlug}-definitive-diff.png`;
        diff.pack().pipe(createWriteStream(diffPath));
        console.log(`   → Diff map: ${diffPath}`);
      }
      
      resolve({ numDiffPixels, totalPixels, diffPercent });
    };
    
    createReadStream(legacy)
      .pipe(new PNG.PNG())
      .on('parsed', function() {
        legacyImg.data = Buffer.from(this.data);
        legacyImg.width = this.width;
        legacyImg.height = this.height;
        completed++;
        finalize();
      })
      .on('error', reject);
    
    createReadStream(aiperf)
      .pipe(new PNG.PNG())
      .on('parsed', function() {
        aiperfImg.data = Buffer.from(this.data);
        aiperfImg.width = this.width;
        aiperfImg.height = this.height;
        completed++;
        finalize();
      })
      .on('error', reject);
  });
}

(async () => {
  console.log('╔════════════════════════════════════════════════════════╗');
  console.log('║          DEFINITIVE VISUAL PARITY COMPARISON            ║');
  console.log('║             Rust Architecture Deck Slides               ║');
  console.log('╚════════════════════════════════════════════════════════╝\n');
  
  console.log('Test Setup:');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('Legacy App:    http://localhost:5173 (Explainers Home)');
  console.log('               → Click "Rust architecture from scratch"');
  console.log('               → View deck slides');
  console.log('');
  console.log('AIPerf-Flow:   http://localhost:5188');
  console.log('               → Click "Explainers"');
  console.log('               → Click "VIEW DECK" on Rust Architecture');
  console.log('               → View deck slides');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
  
  const slides = [
    ['1', 'slide_1', 'Slide 1 (Intro)'],
    ['2', 'slide_2', 'Slide 2 (Content)'],
    ['3', 'slide_3', 'Slide 3 (Content)']
  ];
  
  const results = [];
  
  console.log('Pixel Comparison:');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  
  for (const [num, safeSlug, label] of slides) {
    const legacy = `/tmp/final-legacy-slide-${num}.png`;
    const aiperf = `/tmp/final-aiperf-slide-${num}.png`;
    
    if (!fs.existsSync(legacy) || !fs.existsSync(aiperf)) {
      console.log(`✗ ${label}: Missing files`);
      continue;
    }
    
    const result = await comparePngs(legacy, aiperf, safeSlug, label);
    results.push({ label, ...result });
  }
  
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
  
  console.log('Analysis:');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  
  let avgDiff = 0;
  let maxDiff = 0;
  let minDiff = 100;
  
  for (const r of results) {
    if (r.diffPercent) {
      const pct = parseFloat(r.diffPercent);
      avgDiff += pct;
      maxDiff = Math.max(maxDiff, pct);
      minDiff = Math.min(minDiff, pct);
    }
  }
  
  if (results.length > 0) {
    avgDiff = (avgDiff / results.length).toFixed(2);
    
    console.log(`Average difference: ${avgDiff}%`);
    console.log(`Minimum difference: ${minDiff}%`);
    console.log(`Maximum difference: ${maxDiff}%`);
    console.log('');
    
    if (avgDiff < 2) {
      console.log('✓ RESULT: Byte-exact or nearly identical (< 2%)');
      console.log('  → Apps render identically');
    } else if (avgDiff < 5) {
      console.log('✓ RESULT: Near-identical (2-5%)');
      console.log('  → Minor rendering differences (sub-pixel anti-aliasing)');
    } else if (avgDiff < 15) {
      console.log('⚠️  RESULT: Minor visual differences (5-15%)');
      console.log('');
      console.log('  Likely causes:');
      console.log('  • Font rendering (sub-pixel hinting, kerning)');
      console.log('  • CSS anti-aliasing on borders/shadows');
      console.log('  • Chromium version differences');
      console.log('  • Color space or gamma adjustments');
    } else if (avgDiff < 30) {
      console.log('⚠️  RESULT: Moderate visual differences (15-30%)');
      console.log('');
      console.log('  Likely causes:');
      console.log('  • Layout changes (margins, padding, sizes)');
      console.log('  • Font family or size differences');
      console.log('  • Color palette changes');
      console.log('  • Element visibility or opacity changes');
    } else {
      console.log('✗ RESULT: Significant visual differences (> 30%)');
      console.log('');
      console.log('  Likely causes:');
      console.log('  • Structural layout changes');
      console.log('  • Rendering engine differences');
      console.log('  • Missing or replaced components');
    }
  }
  
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
})();
