/**
 * Graphs Visualization
 * Renders universal graph visualizations for the graphs engine results.
 *
 * @module nexus/visualizations/engines/ml/graphs
 */

import { VIZ_COLORS, escapeHtml } from '../../core/viz-utils.js';
import { ensurePlotly } from '../../core/plotly-helpers.js';

// ============================================================================
// HTML Section Builder
// ============================================================================

/**
 * Build HTML section for Graphs visualization.
 * @param {Object} data - Engine result data
 * @param {string} vizId - Unique visualization ID
 * @returns {string} HTML string
 */
export function buildSection(data, vizId) {
    if (!data) return '';

    const graphCount = data?.total_graphs || data?.graphs?.length || 0;
    if (graphCount === 0) return '';

    return `
        <div class="engine-viz-section">
            <h5>Generated Visualizations (${graphCount})</h5>
            <div class="graphs-grid" id="graphs-${vizId}"></div>
        </div>
    `;
}

// ============================================================================
// Chart Renderer
// ============================================================================

/**
 * Render Graphs visualization.
 * @param {Object} data - Engine result data
 * @param {string} vizId - Unique visualization ID
 */
export function render(data, vizId) {
    const container = document.getElementById(`graphs-${vizId}`);
    if (!container) return;

    const graphs = data?.graphs || [];

    if (!graphs.length) {
        container.innerHTML = `<p style="color: #64748b; text-align: center; padding: 2rem;">Generated ${data?.total_graphs || 0} visualizations - view in detailed report</p>`;
        return;
    }

    // Display summary of graph types
    const graphSummary = {};
    graphs.forEach(g => {
        const type = g.type || 'unknown';
        graphSummary[type] = (graphSummary[type] || 0) + 1;
    });

    container.innerHTML = `
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem;">
            ${Object.entries(graphSummary).map(([type, count]) => `
                <div style="background: ${VIZ_COLORS.surface}; padding: 1rem; border-radius: 0.5rem; text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: 600; color: ${VIZ_COLORS.primary};">${count}</div>
                    <div style="font-size: 0.85rem; color: ${VIZ_COLORS.textMuted};">${escapeHtml(type.replace(/_/g, ' '))}</div>
                </div>
            `).join('')}
        </div>
    `;
}
