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
        const diffPath = `/tmp/${safeSlug}-diff.png`;
        diff.pack().pipe(createWriteStream(diffPath));
        console.log(`   → Diff map saved to ${diffPath}`);
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
  console.log('=== PIXEL-PERFECT COMPARISON ===\n');
  
  const slides = [
    ['slide-1', 'slide_1', 'Slide 1 (Intro/Title)'],
    ['slide-2', 'slide_2', 'Slide 2 (Content)'],
    ['slide-3', 'slide_3', 'Slide 3 (Content)']
  ];
  
  const results = [];
  
  for (const [slug, safeSlug, label] of slides) {
    const legacy = `/tmp/auth-explainers-${slug}.png`;
    const aiperf = `/tmp/auth-aiperf-${slug}.png`;
    
    if (!fs.existsSync(legacy) || !fs.existsSync(aiperf)) {
      console.log(`✗ ${label}: Missing files`);
      continue;
    }
    
    const result = await comparePngs(legacy, aiperf, safeSlug, label);
    results.push({ label, ...result });
  }
  
  console.log('\n=== SUMMARY ===');
  let avgDiff = 0;
  for (const r of results) {
    if (r.diffPercent) {
      console.log(`${r.label}: ${r.diffPercent}% different`);
      avgDiff += parseFloat(r.diffPercent);
    }
  }
  
  if (results.length > 0) {
    avgDiff = (avgDiff / results.length).toFixed(2);
    console.log(`\nAverage difference: ${avgDiff}%`);
    
    if (avgDiff < 5) {
      console.log('✓ Byte-exact or near-identical (0-5%)');
    } else if (avgDiff < 15) {
      console.log('⚠️  Minor visual differences (5-15%)');
    } else {
      console.log('✗ Significant visual differences (>15%)');
    }
  }
})();
