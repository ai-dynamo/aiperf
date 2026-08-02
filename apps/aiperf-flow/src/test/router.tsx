/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Testing Library, with a router around every render.
//!
//! `TopBar` links home, and every deck renders a `TopBar`, so those components genuinely require
//! router context — rendering them without one is a configuration the app never uses. Rather than
//! wrap each call site by hand, a file swaps its `@testing-library/react` import for this one and
//! keeps `render`, `screen`, `fireEvent` and the rest exactly as before.

import { render as testingLibraryRender, type RenderOptions } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import type { ReactElement, ReactNode } from "react";

export * from "@testing-library/react";

function Wrapper({ children }: { children: ReactNode }): React.JSX.Element {
  return <MemoryRouter>{children}</MemoryRouter>;
}

/** `render`, with a `MemoryRouter` supplied. Accepts a caller's own wrapper if one is given. */
export function render(
  ui: ReactElement,
  options?: Omit<RenderOptions, "wrapper">,
): ReturnType<typeof testingLibraryRender> {
  return testingLibraryRender(ui, { wrapper: Wrapper, ...options });
}
