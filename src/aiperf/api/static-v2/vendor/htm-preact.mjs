// HTM 3.1.1 (Apache-2.0) bound to the local Preact 10.25.4 runtime.
import htm from './htm.mjs';
import { Component, h, render } from './preact.mjs';

const html = htm.bind(h);

export { Component, h, html, render };
