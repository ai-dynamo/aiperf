export function normalizePath(path) {
  return path.startsWith('/') ? path : `/${path}`;
}

export function replaceHash(win, path) {
  const target = normalizePath(path);
  const hash = `#${target}`;
  if (win.location.hash === hash) return;
  win.history.replaceState(null, '', hash);
}
