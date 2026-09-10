// Local harness: serves the operator UI from source and replays real API
// responses captured from the live cluster, so the UI can be inspected and
// screenshotted without a cluster, a build, or a deploy.
//
// Fixtures are keyed by exact request path (see fixtures/api.json). Anything
// not captured returns an empty-but-valid shape rather than a 404, because a
// 404 makes the SPA render an error card and hides the thing under test.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const UI_DIR = path.resolve(HERE, '../../src/aiperf/operator/ui');
const FIXTURES = JSON.parse(fs.readFileSync(path.join(HERE, 'fixtures/api.json'), 'utf8'));
const PORT = Number(process.env.PORT || 8099);

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
};

function sendJson(res, body, status = 200) {
  const buf = Buffer.from(JSON.stringify(body));
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'content-length': buf.length,
    'access-control-allow-origin': '*',
  });
  res.end(buf);
}

// Shape-appropriate empties. Returning `{}` for a list route makes the page
// throw on `.map`, which would look like a UI bug in the screenshot.
function emptyFor(pathname) {
  if (pathname.endsWith('/children')) return { children: [] };
  if (pathname.endsWith('/cells')) return { cells: [], dimensions: [], source: 'archived' };
  if (pathname.endsWith('/epochs')) return { epochs: [] };
  if (pathname.startsWith('/api/v1/sweeps')) return { sweeps: [] };
  if (pathname.startsWith('/api/v1/jobs')) return { jobs: [] };
  if (pathname.startsWith('/api/v1/results')) return { results: [] };
  return {};
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  const pathname = decodeURIComponent(url.pathname);

  if (pathname.startsWith('/api/')) {
    const hit = FIXTURES[pathname];
    if (hit && !hit.__error__) return sendJson(res, hit);
    return sendJson(res, emptyFor(pathname));
  }

  const rel = pathname === '/' ? 'index.html' : pathname.replace(/^\/+/, '');
  const file = path.resolve(UI_DIR, rel);
  // Containment check: never serve outside the UI dir.
  if (!file.startsWith(UI_DIR)) {
    res.writeHead(403).end('forbidden');
    return;
  }
  fs.readFile(file, (err, buf) => {
    if (err) {
      // SPA fallback so hash routes deep-link cleanly.
      return fs.readFile(path.join(UI_DIR, 'index.html'), (e2, idx) => {
        if (e2) { res.writeHead(404).end('not found'); return; }
        res.writeHead(200, { 'content-type': MIME['.html'] });
        res.end(idx);
      });
    }
    res.writeHead(200, { 'content-type': MIME[path.extname(file)] ?? 'application/octet-stream' });
    res.end(buf);
  });
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`ui-verify serving ${UI_DIR} on http://127.0.0.1:${PORT}`);
  console.log(`fixtures: ${Object.keys(FIXTURES).length} endpoints`);
});
