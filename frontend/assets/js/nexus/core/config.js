/**
 * NexusAI Core Configuration
 * Central configuration for API endpoints, visualization colors, and settings.
 *
 * @module nexus/core/config
 */

// ============================================================================
// API Configuration
// ============================================================================

/**
 * Base URL for API requests. Defaults to current origin.
 * @type {string}
 */
export const API_BASE = window.location.origin;

/**
 * Retrieves CSRF token from cookies for secure API requests.
 * @returns {string|null} The CSRF token or null if not found.
 */
export function getCsrfToken() {
    const cookies = document.cookie.split(';');
    for (const cookie of cookies) {
        const [name, value] = cookie.trim().split('=');
        if (name === 'ws_csrf') {
            return value;
        }
    }
    return null;
}

/**
 * Returns headers object including CSRF token if available.
 * @returns {Object} Headers for API requests.
 */
export function getAuthHeaders() {
    const headers = {
        'Content-Type': 'application/json'
    };
    const csrfToken = getCsrfToken();
    if (csrfToken) {
        headers['X-CSRF-Token'] = csrfToken;
    }
    return headers;
}

// ============================================================================
// Visualization Color Palette
// ============================================================================

/**
 * Color palette for ECharts, Plotly, and CSS styling.
 * Matches the Nexus dark theme design system.
 * @type {Object}
 */
export const VIZ_COLORS = {
    // Primary
    primary: '#06b6d4',
    primaryLight: '#22d3ee',
    primaryDark: '#0891b2',

    // Accent
    accent: '#10b981',
    accentLight: '#34d399',

    // Status
    success: '#10b981',
    warning: '#f59e0b',
    error: '#ef4444',
    info: '#3b82f6',

    // Backgrounds
    background: '#0a0f1c',
    surface: '#111827',

    // Text
    text: '#e5e7eb',
    textMuted: '#6b7280',

    // Category-specific (for engine categorization)
    ml: '#06b6d4',
    financial: '#f59e0b',
    operations: '#8b5cf6',
    advanced: '#ec4899'
};

// ============================================================================
// Engine Categories
// ============================================================================

/**
 * Engine category metadata for grouping and display.
 * @type {Object}
 */
export const ENGINE_CATEGORIES = {
    ml: {
        name: 'ML & Analytics',
        icon: '🤖',
        color: VIZ_COLORS.ml,
        engineCount: 7
    },
    financial: {
        name: 'Financial Intelligence',
        icon: '💰',
        color: VIZ_COLORS.financial,
        engineCount: 12
    },
    operations: {
        name: 'Operations',
        icon: '📦',
        color: VIZ_COLORS.operations,
        engineCount: 0  // Currently no dedicated operations category in engine list
    },
    advanced: {
        name: 'Advanced AI Lab',
        icon: '🔬',
        color: VIZ_COLORS.advanced,
        engineCount: 3
    }
};

// ============================================================================
// Analysis Settings
// ============================================================================

/**
 * Default settings for analysis runs.
 * @type {Object}
 */
export const DEFAULT_SETTINGS = {
    maxFileSize: 50 * 1024 * 1024, // 50MB
    acceptedFormats: [
        '.csv', '.tsv', '.txt',
        '.json', '.jsonl', '.ndjson',
        '.xlsx', '.xls',
        '.parquet',
        '.db', '.sqlite', '.sqlite3'
    ],
    defaultForecastPeriods: 30,
    minForecastPeriods: 7,
    maxForecastPeriods: 90
};

/**
 * Session storage keys for persistence.
 * @type {Object}
 */
export const STORAGE_KEYS = {
    analysisSession: 'nemo_analysis_session',
    nexusFilename: 'nexus_filename',
    nexusColumns: 'nexus_columns',
    multiRunState: 'nemo_multi_run_state'
};
