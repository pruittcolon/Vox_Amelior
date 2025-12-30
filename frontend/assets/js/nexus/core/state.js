/**
 * NexusAI State Management
 * Manages analysis session state with localStorage persistence for pause/resume.
 *
 * @module nexus/core/state
 */

import { STORAGE_KEYS } from './config.js';

// ============================================================================
// Analysis Session State
// ============================================================================

/**
 * @typedef {Object} AnalysisSession
 * @property {string|null} id - Session identifier
 * @property {string|null} filename - Uploaded file name
 * @property {string|null} startedAt - ISO timestamp when analysis started
 * @property {string|null} stoppedAt - ISO timestamp when analysis was paused
 * @property {'idle'|'running'|'paused'|'completed'} status - Current status
 * @property {string[]} completedEngines - Names of completed engines
 * @property {string[]} pendingEngines - Names of pending engines
 * @property {Object<string, Object>} results - Engine name to result mapping
 * @property {Object<string, Array>} chats - Engine name to chat history
 * @property {number} currentEngineIndex - Index of current/next engine
 */

/**
 * Default analysis session state.
 * @returns {AnalysisSession}
 */
function getDefaultSession() {
    return {
        id: null,
        filename: null,
        startedAt: null,
        stoppedAt: null,
        status: 'idle',
        completedEngines: [],
        pendingEngines: [],
        results: {},
        chats: {},
        currentEngineIndex: 0
    };
}

/**
 * Current analysis session (in-memory state).
 * @type {AnalysisSession}
 */
let analysisSession = getDefaultSession();

/**
 * Flag to indicate if analysis should stop.
 * @type {boolean}
 */
let analysisStopped = false;

// ============================================================================
// Session Accessors
// ============================================================================

/**
 * Get current session state.
 * @returns {AnalysisSession}
 */
export function getSession() {
    return analysisSession;
}

/**
 * Check if analysis is stopped.
 * @returns {boolean}
 */
export function isAnalysisStopped() {
    return analysisStopped;
}

/**
 * Set analysis stopped flag.
 * @param {boolean} stopped
 */
export function setAnalysisStopped(stopped) {
    analysisStopped = stopped;
}

/**
 * Reset session to defaults for new analysis.
 * @param {string} filename - The uploaded file name
 * @param {string[]} engineNames - List of all engine names
 * @returns {AnalysisSession}
 */
export function initSession(filename, engineNames) {
    analysisStopped = false;
    analysisSession = {
        id: `session_${Date.now()}`,
        filename: filename,
        startedAt: new Date().toISOString(),
        stoppedAt: null,
        status: 'running',
        completedEngines: [],
        pendingEngines: [...engineNames],
        results: {},
        chats: {},
        currentEngineIndex: 0
    };
    return analysisSession;
}

/**
 * Update session with completed engine result.
 * @param {string} engineName - Name of the engine
 * @param {Object} result - Engine result data
 */
export function recordEngineResult(engineName, result) {
    analysisSession.results[engineName] = result;
    analysisSession.completedEngines.push(engineName);
    analysisSession.pendingEngines = analysisSession.pendingEngines.filter(
        name => name !== engineName
    );
    analysisSession.currentEngineIndex += 1;
}

/**
 * Add chat message for an engine.
 * @param {string} engineName
 * @param {'user'|'assistant'} role
 * @param {string} message
 */
export function addChatMessage(engineName, role, message) {
    if (!analysisSession.chats[engineName]) {
        analysisSession.chats[engineName] = [];
    }
    analysisSession.chats[engineName].push({
        role,
        message,
        timestamp: new Date().toISOString()
    });
}

/**
 * Mark session as paused.
 */
export function pauseSession() {
    analysisSession.status = 'paused';
    analysisSession.stoppedAt = new Date().toISOString();
    analysisStopped = true;
}

/**
 * Mark session as completed.
 */
export function completeSession() {
    analysisSession.status = 'completed';
}

// ============================================================================
// LocalStorage Persistence
// ============================================================================

/**
 * Save current session to localStorage.
 */
export function saveSessionToStorage() {
    try {
        localStorage.setItem(
            STORAGE_KEYS.analysisSession,
            JSON.stringify(analysisSession)
        );
    } catch (err) {
        console.warn('Failed to save session to localStorage:', err);
    }
}

/**
 * Load session from localStorage.
 * @returns {AnalysisSession|null}
 */
export function loadSessionFromStorage() {
    try {
        const saved = localStorage.getItem(STORAGE_KEYS.analysisSession);
        if (saved) {
            return JSON.parse(saved);
        }
    } catch (err) {
        console.warn('Failed to load session from localStorage:', err);
    }
    return null;
}

/**
 * Clear session from localStorage.
 */
export function clearSessionStorage() {
    try {
        localStorage.removeItem(STORAGE_KEYS.analysisSession);
    } catch (err) {
        console.warn('Failed to clear session storage:', err);
    }
}

/**
 * Cancel and completely reset the current session.
 * Clears both in-memory state and localStorage.
 */
export function cancelSession() {
    analysisSession = getDefaultSession();
    analysisStopped = false;
    clearSessionStorage();
}

/**
 * Restore session from storage into memory.
 * @param {AnalysisSession} savedSession
 */
export function restoreSession(savedSession) {
    analysisSession = savedSession;
    analysisStopped = false;
}

// ============================================================================
// Upload State
// ============================================================================

/**
 * @typedef {Object} UploadState
 * @property {string|null} filename - Uploaded file name
 * @property {string[]} columns - Column names from uploaded file
 */

/** @type {UploadState} */
let uploadState = {
    filename: null,
    columns: []
};

/**
 * Get current upload state.
 * @returns {UploadState}
 */
export function getUploadState() {
    return uploadState;
}

/**
 * Set upload state after successful file upload.
 * @param {string} filename
 * @param {string[]} columns
 */
export function setUploadState(filename, columns) {
    uploadState = { filename, columns };

    // Also save to sessionStorage for page navigation
    try {
        sessionStorage.setItem(STORAGE_KEYS.nexusFilename, filename);
        sessionStorage.setItem(STORAGE_KEYS.nexusColumns, JSON.stringify(columns));
    } catch (err) {
        console.warn('Failed to save upload state:', err);
    }
}

/**
 * Clear upload state.
 */
export function clearUploadState() {
    uploadState = { filename: null, columns: [] };
    try {
        sessionStorage.removeItem(STORAGE_KEYS.nexusFilename);
        sessionStorage.removeItem(STORAGE_KEYS.nexusColumns);
    } catch (err) {
        console.warn('Failed to clear upload state:', err);
    }
}

// ============================================================================
// Column Selection State
// ============================================================================

/**
 * @typedef {Object} ColumnSelection
 * @property {'gemma'|'manual'} mode - Selection mode
 * @property {string|null} target - Target column for prediction
 * @property {string[]} features - Feature columns for analysis
 * @property {string[]} excludedTargets - Previously tried targets with poor results
 */

/** @type {ColumnSelection} */
let columnSelection = {
    mode: 'gemma',
    target: null,
    features: [],
    excludedTargets: []
};

/**
 * Get current column selection.
 * @returns {ColumnSelection}
 */
export function getColumnSelection() {
    return columnSelection;
}

/**
 * Set column selection mode.
 * @param {'gemma'|'manual'} mode
 */
export function setSelectionMode(mode) {
    columnSelection.mode = mode;
}

/**
 * Set target column.
 * @param {string} target
 */
export function setTargetColumn(target) {
    columnSelection.target = target;
}

/**
 * Set feature columns.
 * @param {string[]} features
 */
export function setFeatureColumns(features) {
    columnSelection.features = features;
}

/**
 * Add a target to the excluded list.
 * @param {string} target
 */
export function excludeTarget(target) {
    if (!columnSelection.excludedTargets.includes(target)) {
        columnSelection.excludedTargets.push(target);
    }
}

/**
 * Reset column selection.
 */
export function resetColumnSelection() {
    columnSelection = {
        mode: 'gemma',
        target: null,
        features: [],
        excludedTargets: []
    };
}
