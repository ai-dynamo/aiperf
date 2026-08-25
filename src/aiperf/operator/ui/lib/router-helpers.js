// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export function normalizePath(path) {
  return path.startsWith('/') ? path : `/${path}`;
}

export function replaceHash(win, path) {
  const target = normalizePath(path);
  const hash = `#${target}`;
  if (win.location.hash === hash) return;
  win.history.replaceState(null, '', hash);
}
