// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export function runHref(namespace, name, epoch = null) {
  const base = `#/jobs/${encodeURIComponent(namespace)}/${encodeURIComponent(name)}`;
  return epoch == null ? base : `${base}/runs/${encodeURIComponent(epoch)}`;
}

function compareEpochsNewestFirst(a, b) {
  const aEpoch = String(a?.epoch ?? '');
  const bEpoch = String(b?.epoch ?? '');
  const aNumber = Number(aEpoch);
  const bNumber = Number(bEpoch);
  if (Number.isFinite(aNumber) && Number.isFinite(bNumber)) {
    return bNumber - aNumber;
  }
  return bEpoch.localeCompare(aEpoch);
}

export function buildRunSelectorRows({ namespace, name, epochs, current, hasLive, isRunning = false }) {
  const sorted = [...(epochs || [])].sort(compareEpochsNewestFirst);
  const liveRowVisible = hasLive && isRunning;
  const latestEp = sorted.find(e => e?.isLatest)?.epoch ?? sorted[0]?.epoch;
  const selectedEpoch = current == null && !liveRowVisible ? String(latestEp) : current;
  const rows = [];
  if (liveRowVisible) {
    rows.push({
      kind: 'live',
      epoch: null,
      label: 'Live',
      selected: current == null,
      href: runHref(namespace, name),
      fileCount: null,
      mtimeEpoch: null,
      isLatest: false,
    });
  }
  for (const epoch of sorted) {
    rows.push({
      kind: 'epoch',
      epoch: String(epoch.epoch),
      label: String(epoch.epoch),
      selected: selectedEpoch === String(epoch.epoch),
      href: runHref(namespace, name, epoch.epoch),
      fileCount: epoch.fileCount ?? null,
      mtimeEpoch: epoch.mtimeEpoch ?? null,
      isLatest: Boolean(epoch.isLatest) && sorted.length > 1,
    });
  }
  return rows;
}
