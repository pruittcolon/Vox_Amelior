/**
 * Plotly.js Helper Functions
 * Guards, sanitizers, and utilities for Plotly charts.
 *
 * @module nexus/visualizations/core/plotly-helpers
 */

import { isLowPowerMode, escapeHtml } from './viz-utils.js';

// ============================================================================
// Plotly Container Helpers
// ============================================================================

/**
 * Resolve Plotly container - handles both string IDs and elements.
 * @param {string|HTMLElement} containerOrId - Container reference
 * @returns {HTMLElement|null} DOM element or null
 */
export function resolvePlotlyContainer(containerOrId) {
    if (typeof containerOrId === 'string') {
        return document.getElementById(containerOrId);
    }
    return containerOrId;
}

/**
 * Sanitize Plotly data array to prevent errors.
 * @param {Array} data - Plotly data array
 * @returns {Array} Sanitized data array
 */
export function sanitizePlotlyData(data) {
    if (!data) return [];
    if (!Array.isArray(data)) return [data];
    return data.filter(trace => trace && typeof trace === 'object');
}

/**
 * Sanitize Plotly layout object to prevent errors.
 * @param {Object} layout - Plotly layout object
 * @returns {Object} Sanitized layout object
 */
export function sanitizePlotlyLayout(layout) {
    if (!layout || typeof layout !== 'object') return {};
    const safeLayout = { ...layout };

    if (typeof safeLayout.title === 'string') {
        safeLayout.title = { text: safeLayout.title };
    }

    if (Array.isArray(safeLayout.annotations)) {
        safeLayout.annotations = safeLayout.annotations
            .filter(Boolean)
            .map((annotation) => {
                if (!annotation || typeof annotation !== 'object') return null;
                const safe = { ...annotation };
                if (!('xanchor' in safe)) safe.xanchor = 'auto';
                if (!('yanchor' in safe)) safe.yanchor = 'auto';
                return safe;
            })
            .filter(Boolean);
    } else if (safeLayout.annotations) {
        safeLayout.annotations = [safeLayout.annotations].filter(Boolean);
    }

    if (Array.isArray(safeLayout.shapes)) {
        safeLayout.shapes = safeLayout.shapes.filter(Boolean);
    }

    if (Array.isArray(safeLayout.images)) {
        safeLayout.images = safeLayout.images.filter(Boolean);
    }

    return safeLayout;
}

// ============================================================================
// Plotly Guards
// ============================================================================

let plotlyGuardsInstalled = false;

/**
 * Install guards on Plotly.newPlot to prevent common errors.
 * Should be called once on page load.
 */
export function installPlotlyGuards() {
    if (typeof Plotly === 'undefined' || plotlyGuardsInstalled) return;

    const originalNewPlot = Plotly.newPlot;
    Plotly.newPlot = function (container, data, layout, config) {
        const safeContainer = resolvePlotlyContainer(container) || container;
        const safeData = sanitizePlotlyData(data);
        const safeLayout = sanitizePlotlyLayout(layout);
        const safeConfig = { ...(config || {}) };

        if (isLowPowerMode() && safeConfig.staticPlot === undefined) {
            safeConfig.staticPlot = true;
        }
        if (isLowPowerMode() && safeConfig.scrollZoom === undefined) {
            safeConfig.scrollZoom = false;
        }

        const result = originalNewPlot.call(this, safeContainer, safeData, safeLayout, safeConfig);
        if (result && typeof result.catch === 'function') {
            result.catch((err) => {
                console.error('Plotly render failed:', err);
            });
        }
        return result;
    };

    plotlyGuardsInstalled = true;
}

/**
 * Check if Plotly is available and show error message if not.
 * @param {HTMLElement} container - Container element
 * @param {string} [message] - Error message to display
 * @returns {boolean} True if Plotly is available
 */
export function ensurePlotly(container, message) {
    if (typeof Plotly === 'undefined') {
        container.innerHTML = `<div class="demo-data-notice">${escapeHtml(message || 'Chart library not loaded')}</div>`;
        return false;
    }
    return true;
}

/**
 * Show an error message in a chart container.
 * @param {string} containerId - Container element ID
 * @param {string} chartName - Name of the chart for error message
 * @param {string} message - Error message
 */
export function showChartError(containerId, chartName, message) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `<div class="demo-data-notice">${escapeHtml(chartName)}: ${escapeHtml(message)}</div>`;
    }
}

// Install guards when module loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', installPlotlyGuards);
} else {
    installPlotlyGuards();
}
