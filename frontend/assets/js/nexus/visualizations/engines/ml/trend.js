/**
 * Trend Analysis Visualization
 * Renders trend line charts for trend engine results.
 *
 * @module nexus/visualizations/engines/ml/trend
 */

import { VIZ_COLORS } from '../../core/viz-utils.js';
import { ensurePlotly } from '../../core/plotly-helpers.js';

// ============================================================================
// HTML Section Builder
// ============================================================================

/**
 * Build HTML section for Trend visualization.
 * @param {Object} data - Engine result data
 * @param {string} vizId - Unique visualization ID
 * @returns {string} HTML string
 */
export function buildSection(data, vizId) {
    if (!data) return '';
    return `
        <div class="engine-viz-section">
            <h5>Trend Analysis</h5>
            <div class="trend-container" id="trend-${vizId}"></div>
        </div>
    `;
}

// ============================================================================
// Chart Renderer
// ============================================================================

/**
 * Render Trend visualization charts.
 * @param {Object} data - Engine result data
 * @param {string} vizId - Unique visualization ID
 */
export function render(data, vizId) {
    renderTrendChart(data, `trend-${vizId}`);
}

/**
 * Render trend line chart.
 * @param {Object} data - Trend engine result data
 * @param {string} containerId - Container element ID
 */
function renderTrendChart(data, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !ensurePlotly(container, 'Plotly not loaded')) return;

    const dates = data?.dates || data?.x_data || data?.x || [];
    const values = data?.values || data?.y_data || data?.y || [];
    const trend = data?.trend_line || data?.trend || [];

    if (!dates.length) {
        container.innerHTML = '<p style="color: #64748b; text-align: center; padding: 2rem;">No trend data available</p>';
        return;
    }

    const traces = [
        {
            type: 'scatter',
            mode: 'lines',
            x: dates,
            y: values,
            name: 'Actual',
            line: { color: VIZ_COLORS.primary, width: 2 }
        }
    ];

    if (trend && trend.length) {
        traces.push({
            type: 'scatter',
            mode: 'lines',
            x: dates,
            y: trend,
            name: 'Trend',
            line: { color: VIZ_COLORS.warning, width: 2, dash: 'dot' }
        });
    }

    Plotly.newPlot(container, traces, {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: VIZ_COLORS.textMuted, size: 11 },
        margin: { l: 50, r: 20, t: 20, b: 40 },
        xaxis: { gridcolor: VIZ_COLORS.border },
        yaxis: { gridcolor: VIZ_COLORS.border },
        showlegend: true,
        legend: { x: 0, y: 1, bgcolor: 'transparent' }
    }, { responsive: true, displayModeBar: false });
}
