// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
export function runHref(namespace, name, epoch = null) {
  const base = `#/jobs/${encodeURIComponent(namespace)}/${encodeURIComponent(name)}`;
  return epoch == null ? base : `${base}/runs/${encodeURIComponent(epoch)}`;
}
