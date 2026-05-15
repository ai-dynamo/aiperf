export function runHref(namespace, name, epoch = null) {
  const base = `#/jobs/${encodeURIComponent(namespace)}/${encodeURIComponent(name)}`;
  return epoch == null ? base : `${base}/runs/${encodeURIComponent(epoch)}`;
}

export function buildRunSelectorRows({ namespace, name, epochs, current, hasLive, isRunning = false }) {
  const sorted = [...(epochs || [])].sort((a, b) => String(b?.epoch ?? '').localeCompare(String(a?.epoch ?? '')));
  const latestEp = sorted.find(e => e?.isLatest)?.epoch;
  const rows = [];
  if (hasLive) {
    // Running jobs: row points at the no-epoch URL so the page opens the
    // controller WebSocket for the live stream. Non-running jobs: there is
    // no live stream, so pin directly to /runs/<latest> — otherwise the
    // click bounces through a stale no-epoch render and run-scoped child
    // fetches (e.g. profile_export.jsonl) hit the legacy endpoint and 409.
    const liveEpoch = isRunning || latestEp == null ? null : String(latestEp);
    rows.push({
      kind: 'live',
      epoch: liveEpoch,
      label: isRunning ? 'Live' : 'Latest',
      selected: liveEpoch != null ? current === liveEpoch : current == null,
      href: runHref(namespace, name, liveEpoch),
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
      selected: current === String(epoch.epoch),
      href: runHref(namespace, name, epoch.epoch),
      fileCount: epoch.fileCount ?? null,
      mtimeEpoch: epoch.mtimeEpoch ?? null,
      isLatest: Boolean(epoch.isLatest),
    });
  }
  return rows;
}
