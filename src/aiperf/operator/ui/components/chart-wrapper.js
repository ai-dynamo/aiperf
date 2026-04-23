import { html } from 'htm/preact';
import { useRef, useEffect } from 'preact/hooks';

/**
 * Compute a fast fingerprint of chart data to detect actual changes.
 * Extracts only the numeric values that matter for rendering.
 */
function dataFingerprint(data) {
  if (!data?.datasets) return '';
  return data.datasets.map(ds =>
    (ds.label ?? '') + ':' + (ds.data ?? []).map(pt =>
      typeof pt === 'object' ? `${pt.x},${pt.y}` : pt
    ).join(';')
  ).join('|');
}

/**
 * Chart.js lifecycle wrapper for Preact.
 * Chart.js is loaded as UMD via <script> in index.html, accessed as window.Chart.
 *
 * Only calls chart.update() when the data fingerprint actually changes,
 * preventing unnecessary redraws during polling cycles with no new data.
 *
 * @param {{ type: string, data: object, options?: object, height?: number }} props
 */
export function ChartWrapper({ type, data, options = {}, height = 300 }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);
  const prevFingerprintRef = useRef('');
  const prevOptionsRef = useRef('');

  // Create chart on mount and whenever `type` changes (Chart.js cannot mutate
  // chart type in place — it must be destroyed and recreated).
  useEffect(() => {
    if (!canvasRef.current) return;
    if (!window.Chart) {
      console.warn('ChartWrapper: window.Chart not available - Chart.js not loaded');
      return;
    }

    chartRef.current = new window.Chart(canvasRef.current, {
      type,
      data,
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300 },
        ...options,
      },
    });
    prevFingerprintRef.current = dataFingerprint(data);
    prevOptionsRef.current = JSON.stringify(options);

    return () => {
      if (chartRef.current) {
        chartRef.current.destroy();
        chartRef.current = null;
      }
    };
  }, [type]); // eslint-disable-line react-hooks/exhaustive-deps - data/options handled by their own effects

  // Update data only when fingerprint changes
  useEffect(() => {
    if (!chartRef.current) return;
    const fp = dataFingerprint(data);
    if (fp === prevFingerprintRef.current) return;
    prevFingerprintRef.current = fp;
    chartRef.current.data = data;
    chartRef.current.update();
  }, [data]);

  // Update options only when serialized form changes
  useEffect(() => {
    if (!chartRef.current) return;
    const optStr = JSON.stringify(options);
    if (optStr === prevOptionsRef.current) return;
    prevOptionsRef.current = optStr;
    chartRef.current.options = { responsive: true, maintainAspectRatio: false, animation: { duration: 300 }, ...options };
    chartRef.current.update();
  }, [options]);

  return html`
    <div class="chart-container" style=${'height: ' + height + 'px'}>
      <canvas ref=${canvasRef} />
    </div>
  `;
}
