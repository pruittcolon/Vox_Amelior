/**
 * Anomaly Detection Visualization
 * Renders anomaly score distribution histogram for anomaly engine results.
 *
 * @module nexus/visualizations/engines/ml/anomaly
 */

import { VIZ_COLORS } from '../../core/viz-utils.js';
import { ensurePlotly } from '../../core/plotly-helpers.js';

// ============================================================================
// HTML Section Builder
// ============================================================================

/**
 * Build HTML section for Anomaly visualization.
 * @param {Object} data - Engine result data
 * @param {string} vizId - Unique visualization ID
 * @returns {string} HTML string
 */
export function buildSection(data, vizId) {
    if (!data) return '';
    return `
        <div class="engine-viz-section">
            <h5>Anomaly Score Distribution</h5>
            <div class="anomaly-dist-container" id="anomaly-${vizId}"></div>
        </div>
    `;
}

// ============================================================================
// Chart Renderer
// ============================================================================

/**
 * Render Anomaly visualization charts.
 * @param {Object} data - Engine result data
 * @param {string} vizId - Unique visualization ID
 */
export function render(data, vizId) {
    renderAnomalyDistribution(data, `anomaly-${vizId}`);
}

/**
 * Render anomaly score distribution as histogram.
 * @param {Object} data - Anomaly engine result data
 * @param {string} containerId - Container element ID
 */
function renderAnomalyDistribution(data, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !ensurePlotly(container, 'Plotly not loaded')) return;

    // Extract anomaly scores from various possible locations
    let scores = data?.scores || data?.anomaly_scores || [];
    let threshold = data?.threshold || data?.contamination_threshold || null;
    const anomalyIndices = data?.anomaly_indices || [];

    // Check nested method_results for scores (common structure from ML service)
    if (!scores.length && data?.method_results) {
        const methods = ['isolation_forest', 'ensemble', 'lof', 'zscore'];
        for (const method of methods) {
            const mr = data.method_results[method];
            if (mr?.scores?.length) {
                scores = mr.scores;
                threshold = mr.contamination || threshold;
                break;
            }
        }
    }

    if (!scores.length && data?.results) {
        scores = data.results.map(r => r.score || r.anomaly_score || 0);
    }

    if (!scores.length) {
        container.innerHTML = '<p style="color: #64748b; text-align: center; padding: 2rem;">No anomaly score data available</p>';
        return;
    }

    const traces = [{
        type: 'histogram',
        x: scores,
        name: 'Score Distribution',
        marker: {
            color: VIZ_COLORS.primaryLight,
            line: { color: VIZ_COLORS.primary, width: 1 }
        },
        opacity: 0.8,
        nbinsx: 30
    }];

    const layout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: VIZ_COLORS.textMuted, size: 11 },
        margin: { l: 50, r: 20, t: 20, b: 40 },
        xaxis: {
            title: 'Anomaly Score',
            gridcolor: VIZ_COLORS.border
        },
        yaxis: {
            title: 'Frequency',
            gridcolor: VIZ_COLORS.border
        },
        bargap: 0.05,
        showlegend: false
    };

    // Add threshold line if available
    if (threshold !== null) {
        layout.shapes = [{
            type: 'line',
            x0: threshold,
            x1: threshold,
            y0: 0,
            y1: 1,
            yref: 'paper',
            line: {
                color: VIZ_COLORS.error,
                width: 2,
                dash: 'dash'
            }
        }];

        layout.annotations = [{
            x: threshold,
            y: 1,
            yref: 'paper',
            text: 'Threshold',
            showarrow: true,
            arrowhead: 2,
            ax: 30,
            ay: -20,
            font: { color: VIZ_COLORS.error, size: 10 }
        }];
    }

    Plotly.newPlot(container, traces, layout, { responsive: true, displayModeBar: false });
}
