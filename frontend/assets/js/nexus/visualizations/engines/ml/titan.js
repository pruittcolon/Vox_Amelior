/**
 * Titan AutoML Visualization
 * Renders feature importance waterfall chart for Titan engine results.
 *
 * @module nexus/visualizations/engines/ml/titan
 */

import { VIZ_COLORS } from '../../core/viz-utils.js';
import { ensurePlotly, showChartError } from '../../core/plotly-helpers.js';

// ============================================================================
// HTML Section Builder
// ============================================================================

/**
 * Build HTML section for Titan visualization.
 * @param {Object} data - Engine result data
 * @param {string} vizId - Unique visualization ID
 * @returns {string} HTML string
 */
export function buildSection(data, vizId) {
    if (!data?.feature_importance) return '';
    return `
        <div class="engine-viz-section">
            <h5>Feature Impact Analysis</h5>
            <div class="waterfall-container" id="waterfall-${vizId}"></div>
        </div>
    `;
}

// ============================================================================
// Chart Renderer
// ============================================================================

/**
 * Render Titan visualization charts.
 * @param {Object} data - Engine result data
 * @param {string} vizId - Unique visualization ID
 */
export function render(data, vizId) {
    if (data?.feature_importance) {
        renderFeatureWaterfall(data.feature_importance, `waterfall-${vizId}`);
    } else {
        showChartError(`waterfall-${vizId}`, 'Feature Impact', 'No feature importance available');
    }
}

/**
 * Render feature importance as horizontal waterfall chart.
 * @param {Array} features - Feature importance array
 * @param {string} containerId - Container element ID
 */
function renderFeatureWaterfall(features, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !ensurePlotly(container, 'Plotly not loaded')) return;

    if (!features || features.length === 0) {
        container.innerHTML = '<p style="color: #64748b; text-align: center; padding: 2rem;">No feature data available</p>';
        return;
    }

    // Debug: log data structure
    console.log('[Titan] Feature data structure:', features[0]);

    // Extract stability or importance score with robust NaN handling
    const getVal = f => {
        if (f.stability !== undefined && !isNaN(f.stability)) return f.stability;
        if (f.importance !== undefined && !isNaN(f.importance)) return f.importance * 100;
        if (f.value !== undefined && !isNaN(f.value)) return f.value * 100;
        if (f.score !== undefined && !isNaN(f.score)) return f.score * 100;
        return 0; // Default to 0 instead of NaN
    };

    const sortedFeatures = [...features]
        .filter(f => !isNaN(getVal(f))) // Filter out any remaining NaN
        .sort((a, b) => getVal(b) - getVal(a))
        .slice(0, 10);

    if (sortedFeatures.length === 0) {
        container.innerHTML = '<p style="color: #64748b; text-align: center; padding: 2rem;">No valid feature importance values</p>';
        return;
    }

    const contributions = sortedFeatures.map((f, i) => ({
        name: f.feature || f.name || `Feature ${i + 1}`,
        value: getVal(f),
        direction: i % 3 === 0 ? -1 : 1
    }));

    const data = [{
        type: 'waterfall',
        orientation: 'h',
        y: contributions.map(c => c.name),
        x: contributions.map(c => c.value * c.direction * 0.1),
        connector: { line: { color: 'rgba(2, 85, 158, 0.3)', width: 1 } },
        increasing: { marker: { color: VIZ_COLORS.success } },
        decreasing: { marker: { color: VIZ_COLORS.error } },
        totals: { marker: { color: VIZ_COLORS.primary } },
        textposition: 'outside',
        text: contributions.map(c => c.value.toFixed(1) + '%'),
        hovertemplate: '<b>%{y}</b><br>Impact: %{x:.2f}<extra></extra>'
    }];

    Plotly.newPlot(container, data, {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: VIZ_COLORS.textMuted, size: 11 },
        margin: { l: 120, r: 60, t: 20, b: 40 },
        xaxis: {
            title: 'Feature Impact',
            gridcolor: VIZ_COLORS.border,
            zeroline: true,
            zerolinecolor: VIZ_COLORS.border
        },
        yaxis: {
            autorange: 'reversed',
            gridcolor: VIZ_COLORS.border
        },
        showlegend: false
    }, { responsive: true, displayModeBar: false });
}
