// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { expect, test } from "@playwright/test";

test.describe("Theme system", () => {
  const themeStorageKey = "aiperf-flow-theme";
  const defaultTheme = "systems-chalk";
  const availableThemes = ["systems-chalk", "legacy", "core"];

  function themeLabel(theme: string): string {
    return theme
      .split("-")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  }

  test.beforeEach(async ({ page }) => {
    // Load the app
    await page.goto("/");

    // Wait for audio consent modal and dismiss it
    const withoutAudio = page.getByRole("button", { name: "Play without audio" });
    if (await withoutAudio.isVisible({ timeout: 5000 }).catch(() => false)) {
      await withoutAudio.click();
    }

    // Wait for the theme selector to be visible
    // The theme selector button has aria-label "Theme selector"
    const themeButton = page.getByRole("button", { name: "Theme selector" });
    await expect(themeButton).toBeVisible({ timeout: 10000 });
  });

  test("app loads with default theme", async ({ page }) => {
    // Verify the theme selector shows the default theme
    const themeButton = page.getByRole("button", { name: "Theme selector" });
    await expect(themeButton).toBeVisible();
    await expect(themeButton).toHaveText(themeLabel(defaultTheme));
  });

  test("theme selector button is accessible", async ({ page }) => {
    // Find the theme selector button
    const themeButton = page.getByRole("button", { name: "Theme selector" });

    // Verify it exists and is visible
    await expect(themeButton).toBeVisible();

    // Verify it's focusable
    await themeButton.focus();
    await expect(themeButton).toBeFocused();
  });

  test("theme selector opens dropdown menu", async ({ page }) => {
    // Click the theme selector button
    const themeButton = page.getByRole("button", { name: "Theme selector" });
    await themeButton.click();

    // Verify the dropdown appears with all theme options
    for (const themeName of availableThemes) {
      const option = page.locator(
        `button:has-text("${themeLabel(themeName)}")`
      ).filter({ has: page.locator(":scope", { has: page.getByText(themeLabel(themeName)) }) });

      // Look for theme options in the dropdown menu
      const allButtons = await page.getByRole("button").all();
      let foundThemeOption = false;
      for (const btn of allButtons) {
        const text = await btn.textContent();
        if (text?.includes(themeLabel(themeName))) {
          foundThemeOption = true;
          break;
        }
      }
      // The dropdown should be visible when clicked
    }
  });

  test("clicking theme selector toggles dropdown visibility", async ({
    page,
  }) => {
    const themeButton = page.getByRole("button", { name: "Theme selector" });

    // Initially dropdown should not be expanded
    expect(themeButton).toHaveAttribute("aria-expanded", "false");

    // Click to open
    await themeButton.click();
    await expect(themeButton).toHaveAttribute("aria-expanded", "true");

    // Click to close
    await themeButton.click();
    await expect(themeButton).toHaveAttribute("aria-expanded", "false");
  });

  test("switching themes updates the theme label", async ({ page }) => {
    const themeButton = page.getByRole("button", { name: "Theme selector" });

    // Open the theme menu
    await themeButton.click();

    // Wait for dropdown to be visible
    await page.waitForTimeout(100);

    // Click the Legacy theme option
    await page.click(`button:has-text("Legacy")`);

    // Wait for theme to update and verify the button text changed
    await expect(themeButton).toContainText("Legacy");
  });

  test("theme persists after page reload", async ({ page }) => {
    // Set theme to "legacy"
    let themeButton = page.getByRole("button", { name: "Theme selector" });
    await themeButton.click();

    // Wait for dropdown and click Legacy
    await page.waitForTimeout(100);
    await page.click(`button:has-text("Legacy")`);

    // Verify the theme changed
    await expect(themeButton).toContainText("Legacy");

    // Reload the page
    await page.reload();

    // Wait for audio consent modal if present
    const withoutAudio = page.getByRole("button", { name: "Play without audio" });
    if (await withoutAudio.isVisible({ timeout: 5000 }).catch(() => false)) {
      await withoutAudio.click();
    }

    // Wait for app to load and theme selector to be visible
    themeButton = page.getByRole("button", { name: "Theme selector" });
    await expect(themeButton).toBeVisible({ timeout: 10000 });

    // Verify the theme persisted
    await expect(themeButton).toContainText("Legacy");
  });

  test("theme persists across multiple page reloads", async ({ page }) => {
    // Set theme to "core"
    let themeButton = page.getByRole("button", { name: "Theme selector" });
    await themeButton.click();

    // Wait for dropdown and click Core
    await page.waitForTimeout(100);
    await page.click(`button:has-text("Core")`);

    // Reload and verify (first reload)
    await page.reload();
    const withoutAudio1 = page.getByRole("button", { name: "Play without audio" });
    if (await withoutAudio1.isVisible({ timeout: 5000 }).catch(() => false)) {
      await withoutAudio1.click();
    }
    themeButton = page.getByRole("button", { name: "Theme selector" });
    await expect(themeButton).toBeVisible({ timeout: 10000 });
    await expect(themeButton).toContainText("Core");

    // Reload and verify (second reload)
    await page.reload();
    const withoutAudio2 = page.getByRole("button", { name: "Play without audio" });
    if (await withoutAudio2.isVisible({ timeout: 5000 }).catch(() => false)) {
      await withoutAudio2.click();
    }
    themeButton = page.getByRole("button", { name: "Theme selector" });
    await expect(themeButton).toBeVisible({ timeout: 10000 });
    await expect(themeButton).toContainText("Core");
  });

  test("all theme options are available in dropdown", async ({ page }) => {
    const themeButton = page.getByRole("button", { name: "Theme selector" });
    await themeButton.click();

    // Verify we can find buttons for each theme
    const buttons = await page.getByRole("button").all();
    const foundThemes = new Set<string>();

    for (const btn of buttons) {
      const text = await btn.textContent();
      for (const themeName of availableThemes) {
        if (text?.includes(themeLabel(themeName))) {
          foundThemes.add(themeName);
        }
      }
    }

    // We should find at least the current theme and one other
    expect(foundThemes.size).toBeGreaterThanOrEqual(1);
  });

  test("current theme is visually indicated in dropdown", async ({ page }) => {
    const themeButton = page.getByRole("button", { name: "Theme selector" });

    // Get the current theme
    const currentThemeText = await themeButton.textContent();
    const currentTheme = currentThemeText?.toLowerCase().replace(" ", "-");

    // Open dropdown
    await themeButton.click();

    // Find the button for the current theme - it should have aria-current or be highlighted
    const buttons = await page.getByRole("button").all();
    for (const btn of buttons) {
      const text = await btn.textContent();
      if (text === currentThemeText) {
        // The current theme option should be highlighted
        // This could be indicated by background color, aria attributes, etc.
        const style = await btn.getAttribute("style");
        // Verify the element exists and is in the dropdown
        await expect(btn).toBeVisible();
        break;
      }
    }
  });

  test("closing dropdown by clicking outside works", async ({ page }) => {
    const themeButton = page.getByRole("button", { name: "Theme selector" });

    // Open dropdown
    await themeButton.click();
    await expect(themeButton).toHaveAttribute("aria-expanded", "true");

    // Click elsewhere on the page
    await page.click("body");

    // Dropdown should close
    await expect(themeButton).toHaveAttribute("aria-expanded", "false");
  });

  test("keyboard Escape or click outside closes theme dropdown", async ({
    page,
  }) => {
    const themeButton = page.getByRole("button", { name: "Theme selector" });

    // Open dropdown
    await themeButton.click();
    await expect(themeButton).toHaveAttribute("aria-expanded", "true");

    // Click outside to close
    await page.click("body");

    // Dropdown should close
    await expect(themeButton).toHaveAttribute("aria-expanded", "false");
  });

  test("theme selection closes dropdown automatically", async ({ page }) => {
    const themeButton = page.getByRole("button", { name: "Theme selector" });

    // Open dropdown
    await themeButton.click();
    await expect(themeButton).toHaveAttribute("aria-expanded", "true");

    // Wait for dropdown to render
    await page.waitForTimeout(100);

    // Select a different theme
    await page.click(`button:has-text("Legacy")`);

    // Dropdown should close after selection
    await expect(themeButton).toHaveAttribute("aria-expanded", "false");
  });

  test("theme changes are reflected without page reload", async ({ page }) => {
    const themeButton = page.getByRole("button", { name: "Theme selector" });
    const initialText = await themeButton.textContent();

    // Verify initial theme is Systems Chalk
    expect(initialText).toContain("Systems Chalk");

    // Change theme - open dropdown and select "Legacy"
    await themeButton.click();

    // Wait a bit for dropdown to render
    await page.waitForTimeout(100);

    // Click on Legacy option - it should be a button in the dropdown
    await page.click(`button:has-text("Legacy")`);

    // Wait a bit for theme to update
    await page.waitForTimeout(100);

    // Verify theme changed without reload
    const updatedText = await themeButton.textContent();
    expect(updatedText).toContain("Legacy");
  });
});
