/**
 * Predictive Analysis Visualization
 * Renders forecast charts with confidence intervals for predictive engine results.
 *
 * @module nexus/visualizations/engines/ml/predictive
 */

import { VIZ_COLORS } from '../../core/viz-utils.js';
import { ensurePlotly } from '../../core/plotly-helpers.js';

// ============================================================================
// HTML Section Builder
// ============================================================================

/**
 * Build HTML section for Predictive visualization.
 * @param {Object} data - Engine result data
 * @param {string} vizId - Unique visualization ID
 * @returns {string} HTML string
 */
export function buildSection(data, vizId) {
    if (!data) return '';
    return `
        <div class="engine-viz-section">
            <h5>Predictive Forecast</h5>
            <div class="forecast-container" id="forecast-${vizId}"></div>
        </div>
    `;
}

// ============================================================================
// Chart Renderer
// ============================================================================

/**
 * Render Predictive visualization charts.
 * @param {Object} data - Engine result data
 * @param {string} vizId - Unique visualization ID
 */
export function render(data, vizId) {
    renderForecastChart(data, `forecast-${vizId}`);
}

/**
 * Render forecast line chart with confidence intervals.
 * @param {Object} data - Predictive engine result data
 * @param {string} containerId - Container element ID
 */
function renderForecastChart(data, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !ensurePlotly(container, 'Plotly not loaded')) return;

    const historicalDates = data?.historical_dates || data?.dates || [];
    const historicalValues = data?.historical_values || data?.values || [];
    const forecastDates = data?.forecast_dates || data?.x_data || data?.x || [];
    const forecastValues = data?.forecast_values || data?.y_data || data?.y || [];
    const lower = data?.lower_bound || [];
    const upper = data?.upper_bound || [];

    if (!forecastDates.length && !historicalDates.length) {
        container.innerHTML = '<p style="color: #64748b; text-align: center; padding: 2rem;">No forecast data available</p>';
        return;
    }

    const traces = [];

    // Historical line
    if (historicalDates.length) {
        traces.push({
            type: 'scatter',
            mode: 'lines',
            x: historicalDates,
            y: historicalValues,
            name: 'Historical',
            line: { color: VIZ_COLORS.primary, width: 2 }
        });
    }

    // Forecast line
    if (forecastDates.length) {
        traces.push({
            type: 'scatter',
            mode: 'lines',
            x: forecastDates,
            y: forecastValues,
            name: 'Forecast',
            line: { color: VIZ_COLORS.accent, width: 2, dash: 'dot' }
        });
    }

    // Confidence interval
    if (lower.length && upper.length) {
        traces.push({
            type: 'scatter',
            x: forecastDates,
            y: upper,
            line: { color: 'rgba(16, 185, 129, 0.1)' },
            showlegend: false,
            hoverinfo: 'skip'
        });
        traces.push({
            type: 'scatter',
            x: forecastDates,
            y: lower,
            fill: 'tonexty',
            fillcolor: 'rgba(16, 185, 129, 0.15)',
            line: { color: 'rgba(16, 185, 129, 0.1)' },
            name: 'Confidence'
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
