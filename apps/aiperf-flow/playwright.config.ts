// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { defineConfig, devices } from "@playwright/test";

const previewOrigin = "http://127.0.0.1:5188";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 2 : 0,
  reporter: "list",
  use: {
    ...devices["Desktop Chrome"],
    baseURL: previewOrigin,
    locale: "en-US",
    timezoneId: "UTC",
    colorScheme: "dark",
    reducedMotion: "reduce",
    trace: "retain-on-failure",
    launchOptions:
      process.env.PLAYWRIGHT_EXECUTABLE_PATH === undefined
        ? undefined
        : { executablePath: process.env.PLAYWRIGHT_EXECUTABLE_PATH },
  },
  projects: [
    {
      name: "chromium",
      use: {
        browserName: "chromium",
      },
    },
  ],
  webServer: {
    command: "npm run dev",
    url: previewOrigin,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
