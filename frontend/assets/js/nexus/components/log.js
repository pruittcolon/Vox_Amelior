/**
 * NexusAI Activity Log Component
 * Real-time activity log with timing display.
 *
 * @module nexus/components/log
 */

// ============================================================================
// State
// ============================================================================

let logContainer = null;
let timingStart = null;
let lastLogTime = Date.now();

// ============================================================================
// Public API
// ============================================================================

/**
 * Initialize the log component.
 * @param {string|HTMLElement} containerSelector
 */
export function initLog(containerSelector) {
    logContainer = typeof containerSelector === 'string'
        ? document.querySelector(containerSelector)
        : containerSelector;
}

/**
 * Start timing for a new operation.
 * @returns {number} Start timestamp
 */
export function startTiming() {
    timingStart = Date.now();
    lastLogTime = Date.now();
    return timingStart;
}

/**
 * Get elapsed time since timing started.
 * @returns {number} Milliseconds elapsed
 */
export function getElapsedTime() {
    return timingStart ? Date.now() - timingStart : 0;
}

/**
 * Add a log entry.
 * @param {string} message - Log message
 * @param {'info'|'success'|'error'|'warning'} [type='info'] - Log type
 * @param {number} [duration] - Optional duration in ms
 */
export function log(message, type = 'info', duration = null) {
    if (!logContainer) {
        console.log(`[${type.toUpperCase()}] ${message}`);
        return;
    }

    const timestamp = new Date().toLocaleTimeString();
    const icons = {
        info: 'ℹ️',
        success: '✅',
        error: '❌',
        warning: '⚠️'
    };

    const entry = document.createElement('div');
    entry.className = `log-entry log-${type}`;

    // Calculate timing string
    let timingStr = '';
    const now = Date.now();
    if (duration !== null) {
        timingStr = `<span class="log-timing">(${formatDuration(duration)})</span>`;
    } else if (lastLogTime && type !== 'info') {
        const elapsed = now - lastLogTime;
        if (elapsed > 100) {
            timingStr = `<span class="log-timing">(+${formatDuration(elapsed)})</span>`;
        }
    }
    lastLogTime = now;

    entry.innerHTML = `
    <span class="log-time">[${timestamp}]</span>
    ${icons[type] || ''}
    <span class="log-message">${escapeHtml(message)}</span>
    ${timingStr}
  `;

    logContainer.appendChild(entry);
    logContainer.scrollTop = logContainer.scrollHeight;
}

/**
 * Clear all log entries.
 */
export function clearLog() {
    if (logContainer) {
        logContainer.innerHTML = '';
    }
    timingStart = null;
    lastLogTime = Date.now();
}

/**
 * Add a separator line to the log.
 */
export function logSeparator() {
    if (!logContainer) return;

    const separator = document.createElement('div');
    separator.className = 'log-separator';
    separator.textContent = '─'.repeat(40);
    logContainer.appendChild(separator);
}

// ============================================================================
// Formatting
// ============================================================================

/**
 * Format duration in human-readable format.
 * @param {number} ms - Duration in milliseconds
 * @returns {string}
 */
export function formatDuration(ms) {
    if (ms < 1000) return `${Math.round(ms)}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(2)}s`;
    const mins = Math.floor(ms / 60000);
    const secs = ((ms % 60000) / 1000).toFixed(1);
    return `${mins}m ${secs}s`;
}

/**
 * Escape HTML for safe display.
 * @param {string} text
 * @returns {string}
 */
function escapeHtml(text) {
    if (!text) return '';
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}
