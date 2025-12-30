/**
 * Chaos Analysis Visualization
 * Renders heatmaps for chaos engine results.
 *
 * @module nexus/visualizations/engines/advanced/chaos
 */

import { VIZ_COLORS } from '../../core/viz-utils.js';
import { ensurePlotly } from '../../core/plotly-helpers.js';

export function buildSection(data, vizId) {
    if (!data) return '';
    return `
        <div class="engine-viz-section">
            <h5>Chaos Analysis</h5>
            <div class="chaos-heatmap-container" id="chaos-heatmap-${vizId}" style="height: 400px;"></div>
        </div>
    `;
}

export function render(data, vizId) {
    renderChaosHeatmap(data, `chaos-heatmap-${vizId}`);
}

function renderChaosHeatmap(data, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !ensurePlotly(container, 'Plotly not loaded')) return;

    const matrix = data?.correlation_matrix || data?.chaos_matrix || data?.heatmap || null;
    const labels = data?.labels || data?.variables || [];

    if (!matrix) {
        container.innerHTML = '<p style="color: #64748b; text-align: center; padding: 2rem;">No chaos analysis data available</p>';
        return;
    }

    Plotly.newPlot(container, [{
        type: 'heatmap',
        z: matrix,
        x: labels.length ? labels : undefined,
        y: labels.length ? labels : undefined,
        colorscale: [
            [0, VIZ_COLORS.error],
            [0.5, VIZ_COLORS.surface],
            [1, VIZ_COLORS.success]
        ],
        zmin: -1,
        zmax: 1,
        hovertemplate: '%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
    }], {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: VIZ_COLORS.textMuted, size: 11 },
        margin: { l: 100, r: 40, t: 30, b: 100 },
        title: { text: 'Correlation Heatmap', font: { size: 12 } },
        xaxis: { tickangle: -45 },
        yaxis: {}
    }, { responsive: true, displayModeBar: false });
}
