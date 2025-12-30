/**
 * Statistical Analysis Visualization
 * Renders box plots for statistical engine results.
 *
 * @module nexus/visualizations/engines/ml/statistical
 */

import { VIZ_COLORS, getClusterColor, escapeHtml } from '../../core/viz-utils.js';
import { ensurePlotly } from '../../core/plotly-helpers.js';

// ============================================================================
// HTML Section Builder
// ============================================================================

/**
 * Build HTML section for Statistical visualization.
 * @param {Object} data - Engine result data
 * @param {string} vizId - Unique visualization ID
 * @returns {string} HTML string
 */
export function buildSection(data, vizId) {
    if (!data) return '';
    return `
        <div class="engine-viz-section">
            <h5>Statistical Overview</h5>
            <div class="boxplot-container" id="boxplot-${vizId}"></div>
        </div>
    `;
}

// ============================================================================
// Chart Renderer
// ============================================================================

/**
 * Render Statistical visualization charts.
 * @param {Object} data - Engine result data
 * @param {string} vizId - Unique visualization ID
 */
export function render(data, vizId) {
    renderBoxPlots(data, `boxplot-${vizId}`);
}

/**
 * Render box plots for numeric columns.
 * @param {Object} data - Statistical engine result data
 * @param {string} containerId - Container element ID
 */
function renderBoxPlots(data, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !ensurePlotly(container, 'Plotly not loaded')) return;

    // Extract column statistics
    let columns = [];

    if (data?.column_stats) {
        columns = Object.entries(data.column_stats).map(([name, stats]) => ({
            name,
            ...stats
        }));
    } else if (data?.statistics) {
        columns = Object.entries(data.statistics).map(([name, stats]) => ({
            name,
            ...stats
        }));
    } else if (data?.summary) {
        columns = Object.entries(data.summary).slice(0, 10).map(([name, stats]) => ({
            name,
            ...stats
        }));
    }

    if (!columns.length) {
        container.innerHTML = '<p style="color: #64748b; text-align: center; padding: 2rem;">No statistical data available for box plots</p>';
        return;
    }

    // Limit to 10 columns for readability
    const displayColumns = columns.slice(0, 10);

    const traces = displayColumns.map((col, idx) => {
        // Build box plot from summary statistics if raw values not available
        const boxData = {
            type: 'box',
            name: col.name.length > 15 ? col.name.substring(0, 12) + '...' : col.name,
            marker: { color: getClusterColor(idx) },
            boxmean: true
        };

        if (col.values && Array.isArray(col.values)) {
            boxData.y = col.values;
        } else {
            // Use lowerfence/upperfence/median/q1/q3 for synthetic box
            boxData.lowerfence = [col.min ?? col.percentile_5 ?? 0];
            boxData.q1 = [col.q1 ?? col.percentile_25 ?? col.min ?? 0];
            boxData.median = [col.median ?? col.percentile_50 ?? col.mean ?? 0];
            boxData.q3 = [col.q3 ?? col.percentile_75 ?? col.max ?? 0];
            boxData.upperfence = [col.max ?? col.percentile_95 ?? 0];
            boxData.mean = col.mean !== undefined ? [col.mean] : undefined;
        }

        return boxData;
    });

    const layout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: VIZ_COLORS.textMuted, size: 11 },
        margin: { l: 50, r: 20, t: 20, b: 80 },
        xaxis: {
            tickangle: -45,
            gridcolor: VIZ_COLORS.border
        },
        yaxis: {
            title: 'Value',
            gridcolor: VIZ_COLORS.border
        },
        showlegend: false
    };

    Plotly.newPlot(container, traces, layout, { responsive: true, displayModeBar: false });
}
