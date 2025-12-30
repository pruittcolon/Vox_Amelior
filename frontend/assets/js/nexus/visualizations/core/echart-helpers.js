/**
 * ECharts Helper Functions
 * Utilities for ECharts initialization and management.
 *
 * @module nexus/visualizations/core/echart-helpers
 */

import { escapeHtml, echartInstanceCache } from './viz-utils.js';

// ============================================================================
// ECharts Helpers
// ============================================================================

/**
 * Check if ECharts is available and show error message if not.
 * @param {HTMLElement} container - Container element
 * @param {string} [message] - Error message to display
 * @returns {boolean} True if ECharts is available
 */
export function ensureECharts(container, message) {
    if (typeof echarts === 'undefined') {
        container.innerHTML = `<div class="demo-data-notice">${escapeHtml(message || 'ECharts not loaded')}</div>`;
        return false;
    }
    return true;
}

/**
 * Initialize an ECharts instance in a container.
 * Disposes of any existing instance first.
 * @param {string} containerId - Container element ID
 * @param {Object} [theme] - Optional ECharts theme
 * @returns {Object|null} ECharts instance or null if container not found
 */
export function initEChart(containerId, theme) {
    const container = document.getElementById(containerId);
    if (!container) return null;

    if (!ensureECharts(container)) return null;

    // Dispose existing instance if present
    const existing = echarts.getInstanceByDom(container);
    if (existing) {
        existing.dispose();
    }

    const instance = echarts.init(container, theme);
    echartInstanceCache.set(containerId, instance);
    return instance;
}

/**
 * Get or create an ECharts instance for a container.
 * @param {string} containerId - Container element ID
 * @returns {Object|null} ECharts instance or null
 */
export function getOrCreateEChart(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return null;

    if (!ensureECharts(container)) return null;

    let instance = echarts.getInstanceByDom(container);
    if (!instance) {
        instance = echarts.init(container);
        echartInstanceCache.set(containerId, instance);
    }
    return instance;
}

/**
 * Resize all registered ECharts instances.
 */
export function resizeAllECharts() {
    if (typeof echarts === 'undefined') return;

    echartInstanceCache.forEach((instance, id) => {
        if (instance && typeof instance.resize === 'function') {
            try {
                instance.resize();
            } catch (e) {
                // Ignore resize errors
            }
        }
    });

    // Also try to find any instances not in cache
    document.querySelectorAll('.echart-container').forEach(node => {
        const instance = echarts.getInstanceByDom(node);
        if (instance) {
            try {
                instance.resize();
            } catch (e) {
                // Ignore resize errors
            }
        }
    });
}
