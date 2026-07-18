// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { readFileSync } from "node:fs";
import { join } from "node:path";

// Inject the CSS into jsdom for tests
const css = readFileSync(join(process.cwd(), "preview/styles.css"), "utf8");
const style = document.createElement("style");
style.textContent = css;
document.head.appendChild(style);
