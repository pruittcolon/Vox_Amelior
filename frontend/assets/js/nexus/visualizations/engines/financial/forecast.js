/**
 * Revenue Forecasting Visualization
 * Renders forecast line charts for revenue forecasting engine results.
 *
 * @module nexus/visualizations/engines/financial/forecast
 */

import { VIZ_COLORS } from '../../core/viz-utils.js';
import { ensurePlotly } from '../../core/plotly-helpers.js';

export function buildSection(data, vizId) {
    if (!data) return '';
    return `
        <div class="engine-viz-section fin-viz-premium">
            <h5>Revenue Forecast</h5>
            <div class="fin-chart-container-lg" id="forecast-${vizId}" style="height: 400px;"></div>
        </div>
    `;
}

export function render(data, vizId) {
    renderRevenueForecast(data, `forecast-${vizId}`);
}

function renderRevenueForecast(data, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !ensurePlotly(container, 'Plotly not loaded')) return;

    const historical = data?.historical || data?.historical_values || [];
    const forecast = data?.forecast || data?.forecast_values || [];
    const dates = data?.dates || data?.periods || [];
    const forecastDates = data?.forecast_dates || dates.slice(-forecast.length) || [];
    const lower = data?.lower_bound || [];
    const upper = data?.upper_bound || [];

    if (!historical.length && !forecast.length) {
        container.innerHTML = '<p style="color: #64748b; text-align: center; padding: 2rem;">No revenue forecast data available</p>';
        return;
    }

    const historicalDates = dates.slice(0, historical.length);
    const traces = [];

    if (historical.length) {
        traces.push({
            type: 'scatter',
            mode: 'lines',
            x: historicalDates,
            y: historical,
            name: 'Historical',
            line: { color: VIZ_COLORS.primary, width: 2 }
        });
    }

    if (forecast.length) {
        traces.push({
            type: 'scatter',
            mode: 'lines',
            x: forecastDates,
            y: forecast,
            name: 'Forecast',
            line: { color: VIZ_COLORS.accent, width: 2, dash: 'dot' }
        });
    }

    if (lower.length && upper.length) {
        traces.push({
            type: 'scatter',
            x: forecastDates,
            y: upper,
            line: { color: 'transparent' },
            showlegend: false,
            hoverinfo: 'skip'
        });
        traces.push({
            type: 'scatter',
            x: forecastDates,
            y: lower,
            fill: 'tonexty',
            fillcolor: 'rgba(16, 185, 129, 0.15)',
            line: { color: 'transparent' },
            name: 'Confidence'
        });
    }

    Plotly.newPlot(container, traces, {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: VIZ_COLORS.textMuted, size: 11 },
        margin: { l: 60, r: 20, t: 30, b: 40 },
        title: { text: 'Revenue Forecast', font: { size: 12 } },
        xaxis: { title: 'Period', gridcolor: VIZ_COLORS.border },
        yaxis: { title: 'Revenue ($)', gridcolor: VIZ_COLORS.border },
        showlegend: true,
        legend: { x: 0, y: 1, bgcolor: 'transparent' }
    }, { responsive: true, displayModeBar: false });
}
