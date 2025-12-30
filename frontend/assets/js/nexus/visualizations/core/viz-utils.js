/**
 * Nexus Visualization Utilities
 * Shared constants, formatters, and helper functions used across all engine visualizations.
 *
 * @module nexus/visualizations/core/viz-utils
 */

// ============================================================================
// Visualization Color Palette (matches premium theme)
// ============================================================================

export const VIZ_COLORS = {
    primary: '#02559e',
    primaryLight: '#0284c7',
    primaryDark: '#1e40af',
    accent: '#10b981',
    accentLight: '#34d399',
    success: '#10b981',
    warning: '#f59e0b',
    error: '#ef4444',
    info: '#3b82f6',
    background: '#ffffff',
    surface: '#f8fafc',
    text: '#1e293b',
    textMuted: '#64748b',
    border: '#e2e8f0',
    // Category colors
    ml: '#06b6d4',
    financial: '#f59e0b',
    operations: '#8b5cf6',
    advanced: '#ec4899'
};

// ============================================================================
// Performance Limits
// ============================================================================

export const PERFORMANCE_LIMITS = {
    clusterPoints: 1500,
    clusterPointsLow: 800,
    networkNodes: 120,
    networkNodesLow: 60,
    networkLinks: 240,
    networkLinksLow: 120,
    bubblePoints: 600,
    scatterPoints: 1000
};

// ============================================================================
// Caches
// ============================================================================

export const vizSpecCache = new Map();
export const chartInstanceCache = new Map();
export const echartInstanceCache = new Map();

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Escape HTML special characters for safe display.
 * @param {string} text - Text to escape
 * @returns {string} Escaped text
 */
export function escapeHtml(text) {
    if (text === null || text === undefined) return '';
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

/**
 * Get nested value from object using dot notation path.
 * @param {Object} obj - Object to get value from
 * @param {string} path - Dot-separated path (e.g., "summary.total")
 * @returns {*} Value at path or undefined
 */
export function getNestedValue(obj, path) {
    if (!obj || !path) return undefined;
    return path.split('.').reduce((acc, key) => (acc && acc[key] !== undefined ? acc[key] : undefined), obj);
}

/**
 * Format number with locale-aware formatting.
 * @param {number} value - Number to format
 * @param {number} [digits=2] - Maximum fraction digits
 * @returns {string} Formatted number string
 */
export function formatNumber(value, digits = 2) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return '-';
    return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
}

/**
 * Check if low power mode is enabled (for reduced animations).
 * @returns {boolean} True if low power mode is active
 */
export function isLowPowerMode() {
    return Boolean(window.NexusPerformance?.lowPower);
}

/**
 * Register a chart instance for cleanup/resize handling.
 * @param {string} containerId - Container element ID
 * @param {Object} chart - Chart instance
 */
export function registerChartInstance(containerId, chart) {
    if (!containerId || !chart) return;
    const existing = chartInstanceCache.get(containerId);
    if (existing && typeof existing.destroy === 'function') {
        existing.destroy();
    }
    chartInstanceCache.set(containerId, chart);
}

/**
 * Register an ECharts instance for resize handling.
 * @param {string} containerId - Container element ID
 * @param {Object} instance - ECharts instance
 */
export function registerEChartInstance(containerId, instance) {
    if (!containerId || !instance) return;
    echartInstanceCache.set(containerId, instance);
}

/**
 * Get cluster color for consistent visualization styling.
 * @param {number} clusterId - Cluster identifier (0-based)
 * @returns {string} Hex color code
 */
export function getClusterColor(clusterId) {
    const colors = ['#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316'];
    return colors[clusterId % colors.length];
}

/**
 * Normalize visualization type string to canonical form.
 * @param {string} type - Raw type string
 * @returns {string} Normalized type
 */
export function normalizeVisualizationType(type) {
    const raw = String(type || '').toLowerCase();
    if (raw === 'scatter') return 'scatter_plot';
    if (raw === 'radar') return 'radar_chart';
    return raw;
}

/**
 * Get CSS size class based on visualization type.
 * @param {string} type - Visualization type
 * @returns {string} CSS class name
 */
export function getSizeClass(type) {
    const normalized = normalizeVisualizationType(type);
    const smallTypes = ['metric_cards', 'metric_card', 'text_summary', 'gauge', 'gauge_chart'];
    const tallTypes = [
        'heatmap',
        'line_chart_with_forecast',
        'radar_chart',
        'network_graph',
        'bubble_chart',
        'pareto',
        'waterfall'
    ];
    if (smallTypes.includes(normalized)) return 'viz-short';
    if (tallTypes.includes(normalized)) return 'viz-tall';
    return '';
}
