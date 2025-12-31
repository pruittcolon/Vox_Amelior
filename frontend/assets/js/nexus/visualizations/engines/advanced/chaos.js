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
    console.log('[Chaos] Rendering with data keys:', Object.keys(data || {}));
    renderChaosHeatmap(data, `chaos-heatmap-${vizId}`);
}

function renderChaosHeatmap(data, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !ensurePlotly(container, 'Plotly not loaded')) return;

    // Try multiple data sources - backend sends mutual_information as dict
    let matrix = null;
    let labels = [];

    // Option 1: Direct matrix (legacy format)
    if (data?.correlation_matrix || data?.chaos_matrix || data?.heatmap) {
        matrix = data.correlation_matrix || data.chaos_matrix || data.heatmap;
        labels = data.labels || data.variables || [];
    }

    // Option 2: Convert mutual_information dict to matrix (actual backend format)
    else if (data?.mutual_information && typeof data.mutual_information === 'object') {
        const mi = data.mutual_information;
        const cols = data.numeric_columns || [];

        if (cols.length >= 2) {
            // Build symmetric matrix from dict with "col1__col2" keys
            const n = cols.length;
            matrix = Array(n).fill(null).map(() => Array(n).fill(0));
            labels = cols;

            for (let i = 0; i < n; i++) {
                matrix[i][i] = 1; // Diagonal = 1
                for (let j = i + 1; j < n; j++) {
                    const key = `${cols[i]}__${cols[j]}`;
                    const val = mi[key] || 0;
                    matrix[i][j] = val;
                    matrix[j][i] = val; // Symmetric
                }
            }
        }
    }

    if (!matrix || matrix.length === 0) {
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
        zmin: 0,
        zmax: 1,
        hovertemplate: '%{x} vs %{y}<br>MI: %{z:.3f}<extra></extra>'
    }], {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: VIZ_COLORS.textMuted, size: 11 },
        margin: { l: 100, r: 40, t: 30, b: 100 },
        title: { text: 'Mutual Information Heatmap', font: { size: 12 } },
        xaxis: { tickangle: -45 },
        yaxis: {}
    }, { responsive: true, displayModeBar: false });
}
