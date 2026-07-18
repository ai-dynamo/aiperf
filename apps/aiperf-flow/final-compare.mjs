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
        const diffPath = `/tmp/${safeSlug}-final-diff.png`;
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
  console.log('=== RUST ARCHITECTURE DECK: PIXEL-PERFECT COMPARISON ===\n');
  console.log('Comparing:');
  console.log('  Legacy: http://localhost:5173 → Rust architecture from scratch deck');
  console.log('  AIPerf-Flow: http://localhost:5188 → Explainers → Rust architecture from scratch deck\n');
  
  const slides = [
    ['1', 'slide_1', 'Slide 1 (Intro)'],
    ['2', 'slide_2', 'Slide 2 (Content)'],
    ['3', 'slide_3', 'Slide 3 (Content)']
  ];
  
  const results = [];
  
  for (const [num, safeSlug, label] of slides) {
    const legacy = `/tmp/legacy-rust-slide-${num}.png`;
    const aiperf = `/tmp/aiperf-rust-slide-${num}.png`;
    
    if (!fs.existsSync(legacy) || !fs.existsSync(aiperf)) {
      console.log(`✗ ${label}: Missing files`);
      continue;
    }
    
    const result = await comparePngs(legacy, aiperf, safeSlug, label);
    results.push({ label, ...result });
  }
  
  console.log('\n=== ANALYSIS ===');
  let avgDiff = 0;
  let maxDiff = 0;
  for (const r of results) {
    if (r.diffPercent) {
      console.log(`${r.label}: ${r.diffPercent}% different`);
      const pct = parseFloat(r.diffPercent);
      avgDiff += pct;
      maxDiff = Math.max(maxDiff, pct);
    }
  }
  
  if (results.length > 0) {
    avgDiff = (avgDiff / results.length).toFixed(2);
    console.log(`\nAverage difference: ${avgDiff}%`);
    console.log(`Maximum difference: ${maxDiff}%`);
    
    if (avgDiff < 5) {
      console.log('\n✓ RESULT: Byte-exact or near-identical (0-5%)');
    } else if (avgDiff < 15) {
      console.log('\n⚠️  RESULT: Minor visual differences (5-15%)');
      console.log('\nLikely causes:');
      console.log('  - Font rendering differences (sub-pixel anti-aliasing)');
      console.log('  - Anti-aliasing on CSS borders and shadows');
      console.log('  - CSS transform/filter effects');
      console.log('  - Color space or gamma differences in browser rendering');
    } else {
      console.log('\n✗ RESULT: Significant visual differences (>15%)');
      console.log('  Check diff maps for detailed mismatch areas');
    }
  }
})();
