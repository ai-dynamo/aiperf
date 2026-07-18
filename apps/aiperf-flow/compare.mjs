import fs from 'fs';
import path from 'path';
import { PNG } from 'pngjs';
import pixelmatch from 'pixelmatch';
import sharp from 'sharp';

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

async function compareImages(path1, path2, outputPath) {
  // Read both images
  const img1Data = fs.readFileSync(path1);
  const img2Data = fs.readFileSync(path2);

  // Parse PNG files
  const img1 = PNG.sync.read(img1Data);
  const img2 = PNG.sync.read(img2Data);

  // Check dimensions
  if (img1.width !== img2.width || img1.height !== img2.height) {
    console.log(`⚠ Image dimensions differ:`);
    console.log(`  Image 1: ${img1.width}x${img1.height}`);
    console.log(`  Image 2: ${img2.width}x${img2.height}`);
    console.log('  (Resizing image 2 to match image 1 dimensions)');
    
    // Resize img2 to match img1 dimensions
    const buffer = await sharp(img2Data)
      .resize(img1.width, img1.height, { fit: 'fill' })
      .png()
      .toBuffer();
    
    const img2Resized = PNG.sync.read(buffer);
    return { ...comparePixels(img1, img2Resized), resized: true };
  }

  return { ...comparePixels(img1, img2), resized: false };
}

function comparePixels(img1, img2) {
  const width = img1.width;
  const height = img1.height;

  // Create diff image
  const diff = new PNG({ width, height });

  // Compare pixels
  const mismatchedPixels = pixelmatch(
    img1.data,
    img2.data,
    diff.data,
    width,
    height,
    { threshold: 0.1 }
  );

  const totalPixels = width * height;
  const diffPercentage = (mismatchedPixels / totalPixels * 100).toFixed(2);

  return {
    mismatchedPixels,
    totalPixels,
    diffPercentage: parseFloat(diffPercentage)
  };
}

async function analyzeScreenshots() {
  console.log('=== SCREENSHOT COMPARISON ANALYSIS ===\n');

  // Compare home pages
  console.log('1. Comparing explainers home vs aiperf-flow home...');
  const homeResult = await compareImages('/tmp/explainers-home.png', '/tmp/aiperf-flow-home.png', '/tmp/diff-home.png');
  console.log(`   Mismatched pixels: ${homeResult.mismatchedPixels}`);
  console.log(`   Total pixels: ${homeResult.totalPixels}`);
  console.log(`   Difference: ${homeResult.diffPercentage}%`);
  console.log(`   ${homeResult.resized ? '(Images were resized to match dimensions)' : ''}\n`);

  // Get image info
  const img1Meta = await sharp('/tmp/explainers-home.png').metadata();
  const img2Meta = await sharp('/tmp/aiperf-flow-home.png').metadata();

  console.log('Image dimensions:');
  console.log(`   Explainers: ${img1Meta.width}x${img1Meta.height}`);
  console.log(`   AIPerf Flow: ${img2Meta.width}x${img2Meta.height}\n`);

  // Analyze color differences
  console.log('2. Analyzing visual differences...');
  console.log('   Taking pixel-level samples to identify styling differences...\n');

  // Read and analyze the images
  const img1Data = fs.readFileSync('/tmp/explainers-home.png');
  const img2Data = fs.readFileSync('/tmp/aiperf-flow-home.png');

  const img1 = PNG.sync.read(img1Data);
  const img2 = PNG.sync.read(img2Data);

  // Sample key areas (header, footer, main content)
  console.log('3. Key findings:');
  console.log(`   • Pixel difference: ${homeResult.diffPercentage}%`);
  if (homeResult.diffPercentage < 5) {
    console.log('   • Layouts are very similar');
  } else if (homeResult.diffPercentage < 15) {
    console.log('   • Moderate styling differences detected');
  } else {
    console.log('   • Significant styling differences detected');
  }
  console.log(`   • Image dimensions are ${img1Meta.width === img2Meta.width && img1Meta.height === img2Meta.height ? 'the same' : 'different'}`);
}

analyzeScreenshots().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
