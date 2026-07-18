#!/usr/bin/env node
/**
 * Comprehensive page and state catalog for explainers app.
 * Navigates through every discoverable page, takes screenshots, and documents UI elements.
 */

import { chromium } from "playwright";
import fs from "fs";
import path from "path";

const BASE_URL = "http://127.0.0.1:5188";
const SCREENSHOT_DIR = "./page-catalog";
const CATALOG_FILE = path.join(SCREENSHOT_DIR, "catalog.md");

// Ensure output directory exists
if (!fs.existsSync(SCREENSHOT_DIR)) {
  fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });
}

class PageCatalog {
  constructor() {
    this.pages = [];
    this.currentPageIndex = 1;
  }

  addPage(info) {
    const pageNum = this.currentPageIndex;
    this.currentPageIndex++;
    this.pages.push({ ...info, pageNum });
    return pageNum;
  }

  generateMarkdown() {
    let md = `# Explainers App - Complete Page Catalog\n\n`;
    md += `Generated: ${new Date().toISOString()}\n`;
    md += `Total Unique Pages: ${this.pages.length}\n\n`;
    md += `---\n\n`;

    for (const page of this.pages) {
      md += `## Page ${page.pageNum}: ${page.title}\n\n`;
      md += `**URL/Path:** ${page.url || page.path}\n\n`;
      md += `**Screenshot:** \`${page.screenshot}\`\n\n`;
      md += `### Visual Description\n${page.visual}\n\n`;
      md += `### Layout & Design\n${page.layout}\n\n`;
      md += `### Interactive Elements\n${page.interactive}\n\n`;
      md += `### Content & Data\n${page.content}\n\n`;
      md += `### Navigation Controls\n${page.navigation}\n\n`;
      if (page.notes) {
        md += `### Notes\n${page.notes}\n\n`;
      }
      md += `---\n\n`;
    }

    return md;
  }
}

async function capturePageInfo(page, catalog, title, path, options = {}) {
  const baseFileName = title
    .toLowerCase()
    .replace(/\s+/g, "-")
    .replace(/[^\w-]/g, "");
  const screenshotPath = path.join(
    SCREENSHOT_DIR,
    `${String(catalog.currentPageIndex).padStart(2, "0")}-${baseFileName}.png`
  );

  // Take screenshot
  await page.screenshot({ path: screenshotPath, fullPage: true });

  // Get page content info
  const url = page.url();
  const dimensions = await page.evaluate(() => ({
    width: window.innerWidth,
    height: window.innerHeight,
  }));

  // Extract interactive elements
  const elements = await page.evaluate(() => {
    const result = {
      buttons: [],
      links: [],
      inputs: [],
      selects: [],
      textareas: [],
      cards: [],
      headings: [],
    };

    // Buttons
    document.querySelectorAll("button").forEach((btn) => {
      const text = btn.textContent?.trim() || btn.getAttribute("aria-label") || "";
      if (text && !result.buttons.includes(text)) {
        result.buttons.push(text);
      }
    });

    // Links
    document.querySelectorAll("a").forEach((link) => {
      const text = link.textContent?.trim() || "";
      const href = link.getAttribute("href") || "";
      if (text && !result.links.find((l) => l.text === text)) {
        result.links.push({ text, href });
      }
    });

    // Inputs
    document.querySelectorAll("input").forEach((input) => {
      const type = input.getAttribute("type") || "text";
      const placeholder = input.getAttribute("placeholder") || "";
      const id = input.id || "";
      if (!result.inputs.find((i) => i.id === id)) {
        result.inputs.push({ type, placeholder, id });
      }
    });

    // Select dropdowns
    document.querySelectorAll("select").forEach((select) => {
      const id = select.id || "";
      const options = Array.from(select.options).map((o) => o.text);
      result.selects.push({ id, options });
    });

    // Cards/containers with class "card" or similar
    document.querySelectorAll("[class*='card']").forEach((card) => {
      const text = card.textContent?.trim().substring(0, 100) || "";
      if (text && !result.cards.includes(text)) {
        result.cards.push(text);
      }
    });

    // Headings
    ["h1", "h2", "h3", "h4"].forEach((tag) => {
      document.querySelectorAll(tag).forEach((heading) => {
        const text = heading.textContent?.trim() || "";
        if (text && !result.headings.includes(text)) {
          result.headings.push(text);
        }
      });
    });

    return result;
  });

  const pageNum = catalog.addPage({
    title,
    path: path,
    url: url,
    screenshot: path.basename(screenshotPath),
    visual: options.visual || `Modern dark theme interface at ${dimensions.width}x${dimensions.height}px`,
    layout: options.layout || "Responsive flexbox layout",
    interactive: formatInteractiveElements(elements),
    content: options.content || "Interactive flow visualization content",
    navigation: options.navigation || "Back/forward navigation controls",
    notes: options.notes || "",
  });

  console.log(`✓ Page ${pageNum}: ${title}`);
  return pageNum;
}

function formatInteractiveElements(elements) {
  let md = "";

  if (elements.headings.length) {
    md += `- **Headings:** ${elements.headings.slice(0, 5).join("; ")}\n`;
  }

  if (elements.buttons.length) {
    md += `- **Buttons:** ${elements.buttons.slice(0, 5).join("; ")}\n`;
  }

  if (elements.links.length) {
    md += `- **Links:** ${elements.links
      .slice(0, 3)
      .map((l) => `${l.text} (${l.href})`)
      .join("; ")}\n`;
  }

  if (elements.inputs.length) {
    md += `- **Input Fields:** ${elements.inputs.map((i) => i.type).join(", ")}\n`;
  }

  if (elements.selects.length) {
    md += `- **Dropdowns:** ${elements.selects.length} found\n`;
  }

  if (elements.cards.length) {
    md += `- **Cards:** ${elements.cards.length} clickable card elements\n`;
  }

  return md || "- Navigation controls and interactive elements\n";
}

async function main() {
  const browser = await chromium.launch({ headless: true });
  const catalog = new PageCatalog();

  try {
    const browserPage = await browser.newPage({
      viewport: { width: 1920, height: 1080 },
    });

    // PAGE 1: Home/Landing Page
    console.log("\n=== Capturing Home Page ===");
    await browserPage.goto(BASE_URL, { waitUntil: "networkidle" });
    await browserPage.waitForLoadState("networkidle");
    await capturePageInfo(
      browserPage,
      catalog,
      "Home Page - Scene Selector",
      "/",
      {
        visual: "Dark theme landing page with grid of scene cards organized by flow",
        layout:
          "Header with title + subtitle, main grid with scene cards in multiple columns, responsive layout",
        content:
          "Title: 'AIPerf Flow Scenes', subtitle showing count of flows and scenes, multiple scene cards with titles, descriptions, and flow badges",
        navigation:
          "Scene cards are clickable buttons with hover effects, each card links to a specific flow/scene",
      }
    );

    // PAGE 2: Get the scene list and click into each flow
    const sceneCards = await browserPage.evaluate(() => {
      const cards = [];
      document.querySelectorAll(".scene-card").forEach((card) => {
        const title = card.querySelector(".scene-card-title")?.textContent || "";
        const flow = card.querySelector(".scene-card-kicker")?.textContent || "";
        // Get the click handler by checking what scene it will load
        const ariaLabel = card.getAttribute("aria-label") || "";
        cards.push({ title, flow, ariaLabel });
      });
      return cards;
    });

    console.log(`\nFound ${sceneCards.length} scene cards on home page`);

    // Click into each scene and catalog unique slide layouts
    const seenScenes = new Set();

    for (let i = 0; i < Math.min(sceneCards.length, 6); i++) {
      const card = sceneCards[i];
      const sceneKey = `${card.flow}-${card.title}`;

      if (seenScenes.has(sceneKey)) continue;
      seenScenes.add(sceneKey);

      console.log(`\n=== Scene ${i + 1}: ${card.title} (${card.flow}) ===`);

      // Click the scene card
      await browserPage.click(".scene-card", { timeout: 5000 });
      await browserPage.waitForLoadState("networkidle");
      await new Promise((r) => setTimeout(r, 1000)); // Let content render

      // PAGE: First slide of scene
      await capturePageInfo(
        browserPage,
        catalog,
        `${card.flow} - ${card.title} (Slide 1)`,
        `/${card.flow}/${card.title}`,
        {
          visual: "Interactive flow visualization with narrative content",
          layout: "Main canvas area with flow diagram, sidebar with narrative/controls",
          content: `Scene: ${card.title} from ${card.flow} flow`,
          navigation:
            "Next/previous slide controls, timeline navigation, playback controls",
        }
      );

      // Get slide navigation info
      const slideInfo = await browserPage.evaluate(() => {
        return {
          currentSlide: document.querySelector("[data-slide-index]")?.getAttribute("data-slide-index"),
          totalSlides: document.querySelectorAll("[data-slide]").length,
          hasPlayButton: !!document.querySelector("button[aria-label*='Play']"),
          hasPauseButton: !!document.querySelector("button[aria-label*='Pause']"),
          hasNextButton: !!document.querySelector("button[aria-label*='Next']"),
          hasPrevButton: !!document.querySelector("button[aria-label*='Previous']"),
          hasTimelineSlider: !!document.querySelector("input[type='range']"),
        };
      });

      console.log(`  Slide info:`, slideInfo);

      // Try to navigate to middle slide
      const nextButtons = await browserPage.$$("button[aria-label*='Next']");
      if (nextButtons.length > 0) {
        console.log("  Clicking next slide...");
        await nextButtons[0].click();
        await new Promise((r) => setTimeout(r, 800));

        // PAGE: Middle slide
        await capturePageInfo(
          browserPage,
          catalog,
          `${card.flow} - ${card.title} (Slide 2)`,
          `/${card.flow}/${card.title}/slide-2`,
          {
            visual: "Updated flow visualization showing different stage or aspect",
            layout: "Same layout as slide 1 with different canvas content",
            content: `Scene: ${card.title} - Mid-sequence slide showing progression`,
            navigation: "Next/previous controls, progress indicator updated",
          }
        );
      }

      // Try to navigate to last slide
      const nextButtons2 = await browserPage.$$("button[aria-label*='Next']");
      if (nextButtons2.length > 0) {
        console.log("  Clicking next slide again...");
        await nextButtons2[0].click();
        await new Promise((r) => setTimeout(r, 800));

        // PAGE: Last/final slide
        await capturePageInfo(
          browserPage,
          catalog,
          `${card.flow} - ${card.title} (Final Slide)`,
          `/${card.flow}/${card.title}/slide-final`,
          {
            visual: "Final slide in sequence with conclusion content",
            layout: "Same layout with final visualization",
            content: `Scene: ${card.title} - Final slide with summary or conclusion`,
            navigation: "Navigation at end of sequence",
          }
        );
      }

      // Check for any modals or overlays
      const modals = await browserPage.$$(".modal, [role='dialog']");
      if (modals.length > 0) {
        console.log(`  Found ${modals.length} modal(s)`);

        // PAGE: Modal/dialog if present
        await capturePageInfo(
          browserPage,
          catalog,
          `Modal Dialog / Overlay`,
          `/modals/${card.title}`,
          {
            visual: "Modal dialog overlaid on top of scene content",
            layout: "Center-aligned modal dialog with content",
            content: "Dialog content and controls",
            navigation: "Close button or dismiss controls",
          }
        );
      }

      // Go back to home
      console.log("  Returning to home page...");
      await browserPage.goto(BASE_URL, { waitUntil: "networkidle" });
      await new Promise((r) => setTimeout(r, 500));
    }

    // Check for theme switcher or settings
    console.log("\n=== Checking for Settings/Options ===");
    const themeButtons = await browserPage.$$("button[aria-label*='theme'], button[aria-label*='Theme']");
    if (themeButtons.length > 0) {
      console.log(`  Found ${themeButtons.length} theme switcher button(s)`);

      await themeButtons[0].click();
      await new Promise((r) => setTimeout(r, 500));

      // PAGE: Theme selector (if visible)
      const isThemeVisible = await browserPage.evaluate(() => {
        return !!document.querySelector("[class*='theme'], [class*='Theme']");
      });

      if (isThemeVisible) {
        await capturePageInfo(
          browserPage,
          catalog,
          "Theme Selector / Settings",
          "/settings/theme",
          {
            visual: "Theme selection UI component",
            layout: "Dropdown or menu showing available themes",
            content: "Systems Chalk, Legacy, Core theme options",
            navigation: "Theme selection buttons/radio buttons",
          }
        );
      }
    }

    // Check for glossary or help
    console.log("\n=== Checking for Glossary/Help ===");
    const glossaryButtons = await browserPage.$$("button[aria-label*='glossary'], button[aria-label*='Glossary'], a[href*='glossary']");
    if (glossaryButtons.length > 0) {
      console.log(`  Found ${glossaryButtons.length} glossary link(s)`);

      await glossaryButtons[0].click();
      await new Promise((r) => setTimeout(r, 800));

      // PAGE: Glossary
      await capturePageInfo(
        browserPage,
        catalog,
        "Glossary / Terms Reference",
        "/glossary",
        {
          visual: "Glossary or reference page listing terms",
          layout: "List or grid layout of term definitions",
          content: "Technical terms and explanations",
          navigation: "Search or filter controls for terms",
        }
      );
    }

    // Check for error state (try invalid URL)
    console.log("\n=== Checking for Error States ===");
    await browserPage.goto(`${BASE_URL}/nonexistent`, { waitUntil: "networkidle" });
    await new Promise((r) => setTimeout(r, 500));

    const hasErrorContent = await browserPage.evaluate(() => {
      return (
        !!document.querySelector("[class*='error'], [class*='not-found']") ||
        document.body.textContent.includes("404") ||
        document.body.textContent.includes("not found")
      );
    });

    if (hasErrorContent) {
      // PAGE: Error page
      await capturePageInfo(
        browserPage,
        catalog,
        "Error State - Page Not Found",
        "/error-404",
        {
          visual: "Error page displayed when route is invalid",
          layout: "Centered error message with recovery options",
          content: "404 error message and suggestions",
          navigation: "Link back to home or previous page",
        }
      );
    }

    // Return to home for loading state capture
    console.log("\n=== Checking for Loading States ===");
    await browserPage.goto(BASE_URL, { waitUntil: "domcontentloaded" }); // Stop early to catch loading
    await new Promise((r) => setTimeout(r, 300));

    const hasLoadingIndicator = await browserPage.evaluate(() => {
      return (
        !!document.querySelector("[class*='loading'], [class*='spinner'], [role='progressbar']") ||
        !!document.querySelector(".loader, .progress, .skeleton")
      );
    });

    if (hasLoadingIndicator) {
      // PAGE: Loading state
      await capturePageInfo(
        browserPage,
        catalog,
        "Loading State / Skeleton",
        "/loading",
        {
          visual: "Loading indicator or skeleton screen",
          layout: "Placeholder layout while content loads",
          content: "Loading spinner or progress bar",
          navigation: "No navigation available during load",
        }
      );
    }

    // Wait for full load
    await browserPage.waitForLoadState("networkidle");

    // Check responsive breakpoint (mobile view)
    console.log("\n=== Capturing Mobile Responsive View ===");
    await browserPage.setViewportSize({ width: 375, height: 812 });
    await browserPage.goto(BASE_URL, { waitUntil: "networkidle" });
    await new Promise((r) => setTimeout(r, 500));

    // PAGE: Home page mobile
    await capturePageInfo(
      browserPage,
      catalog,
      "Home Page - Mobile View (iPhone)",
      "/ (mobile 375x812)",
      {
        visual: "Home page layout adapted for mobile screens",
        layout: "Single column layout, full-width cards, touch-friendly spacing",
        content: "Same scene cards optimized for mobile",
        navigation: "Touch-friendly buttons and navigation",
      }
    );

    // Click a scene on mobile
    const mobileCard = await browserPage.$(".scene-card");
    if (mobileCard) {
      await mobileCard.click();
      await browserPage.waitForLoadState("networkidle");
      await new Promise((r) => setTimeout(r, 500));

      // PAGE: Scene view mobile
      await capturePageInfo(
        browserPage,
        catalog,
        "Scene - Mobile View",
        "/scene (mobile 375x812)",
        {
          visual: "Flow scene on mobile with adapted layout",
          layout: "Single column with stacked controls and canvas",
          content: "Same scene content as desktop",
          navigation: "Mobile-optimized navigation controls",
        }
      );
    }

    // Generate catalog markdown
    console.log("\n=== Generating Catalog ===");
    const catalogMd = catalog.generateMarkdown();
    fs.writeFileSync(CATALOG_FILE, catalogMd);
    console.log(`✓ Catalog written to ${CATALOG_FILE}`);
    console.log(`\nTotal unique pages captured: ${catalog.pages.length}`);

    // Print summary
    console.log("\n=== Page Summary ===");
    catalog.pages.forEach((page) => {
      console.log(`  ${page.pageNum}. ${page.title}`);
    });

    await browserPage.close();
  } catch (error) {
    console.error("Error during catalog generation:", error);
    process.exit(1);
  } finally {
    await browser.close();
  }
}

main().catch(console.error);
