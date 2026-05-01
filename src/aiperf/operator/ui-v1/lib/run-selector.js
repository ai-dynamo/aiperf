export function runHref(namespace, name, epoch = null) {
  const base = `#/jobs/${encodeURIComponent(namespace)}/${encodeURIComponent(name)}`;
  return epoch == null ? base : `${base}/runs/${encodeURIComponent(epoch)}`;
}

export function buildRunSelectorRows({ namespace, name, epochs, current, hasLive }) {
  const sorted = [...(epochs || [])].sort((a, b) => String(b?.epoch ?? '').localeCompare(String(a?.epoch ?? '')));
  const rows = [];
  if (hasLive) {
    rows.push({
      kind: 'live',
      epoch: null,
      label: 'Live / latest',
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
      selected: current === String(epoch.epoch),
      href: runHref(namespace, name, epoch.epoch),
      fileCount: epoch.fileCount ?? null,
      mtimeEpoch: epoch.mtimeEpoch ?? null,
      isLatest: Boolean(epoch.isLatest),
    });
  }
  return rows;
}
