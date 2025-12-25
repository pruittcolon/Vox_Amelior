/**
 * NexusAI Engine Runner
 * Handles sequential engine execution with pause/resume support.
 *
 * @module nexus/engines/engine-runner
 */

import { runEngine, getGemmaSummary } from '../core/api.js';
import {
    getSession,
    initSession,
    recordEngineResult,
    pauseSession,
    completeSession,
    saveSessionToStorage,
    loadSessionFromStorage,
    restoreSession,
    isAnalysisStopped,
    setAnalysisStopped,
    getUploadState
} from '../core/state.js';
import { ALL_ENGINES, getAllEngineNames, ENGINE_COUNT } from './engine-definitions.js';
import { createEngineCard, displayEngineResults, updateEngineCardStatus } from './engine-results.js';

// ============================================================================
// Event Callbacks
// ============================================================================

/**
 * Callback definitions for UI updates.
 * @type {Object}
 */
let callbacks = {
    onProgress: null,      // (completed, total, engineName) => void
    onEngineStart: null,   // (engine, index) => void
    onEngineComplete: null, // (engine, result, duration) => void
    onEngineError: null,   // (engine, error, duration) => void
    onAllComplete: null,   // (stats) => void
    onLog: null            // (message, type, duration) => void
};

/**
 * Register event callbacks.
 * @param {Object} cbs - Callback functions
 */
export function registerCallbacks(cbs) {
    callbacks = { ...callbacks, ...cbs };
}

/**
 * Internal log helper.
 * @param {string} message
 * @param {string} type
 * @param {number} [duration]
 */
function log(message, type = 'info', duration = null) {
    if (callbacks.onLog) {
        callbacks.onLog(message, type, duration);
    }
}

// ============================================================================
// Engine Execution
// ============================================================================

/**
 * Check if there's a saved session that can be resumed.
 * @returns {{canResume: boolean, session: Object|null, completedCount: number}}
 */
export function checkForResumableSession() {
    const uploadState = getUploadState();
    if (!uploadState.filename) {
        return { canResume: false, session: null, completedCount: 0 };
    }

    const savedSession = loadSessionFromStorage();
    if (
        savedSession &&
        savedSession.filename === uploadState.filename &&
        savedSession.status === 'paused'
    ) {
        return {
            canResume: true,
            session: savedSession,
            completedCount: savedSession.completedEngines.length
        };
    }

    return { canResume: false, session: null, completedCount: 0 };
}

/**
 * Start a fresh analysis run.
 * @param {Object} options
 * @param {boolean} [options.useVectorization=false]
 * @returns {Promise<void>}
 */
export async function startAnalysis(options = {}) {
    const uploadState = getUploadState();
    if (!uploadState.filename) {
        throw new Error('No file uploaded');
    }

    // Initialize fresh session
    initSession(uploadState.filename, getAllEngineNames());
    setAnalysisStopped(false);

    log(`🧪 Starting comprehensive analysis with all ${ENGINE_COUNT} engines...`, 'info');
    log(`📁 Testing on: ${uploadState.filename}`, 'info');

    await runEngineLoop(0, options);
}

/**
 * Resume a paused analysis.
 * @param {Object} savedSession - The session to resume
 * @param {Object} options
 * @returns {Promise<void>}
 */
export async function resumeAnalysis(savedSession, options = {}) {
    restoreSession(savedSession);
    setAnalysisStopped(false);

    const session = getSession();
    log(`🔄 Resuming analysis from engine ${session.currentEngineIndex + 1}/${ENGINE_COUNT}`, 'info');

    await runEngineLoop(session.currentEngineIndex, options);
}

/**
 * Stop the current analysis (pause).
 */
export function stopAnalysis() {
    setAnalysisStopped(true);
    pauseSession();
    saveSessionToStorage();
    log('⏸️ Analysis paused. You can resume later.', 'warning');
}

/**
 * Main engine execution loop.
 * @param {number} startIndex - Index to start from
 * @param {Object} options
 * @returns {Promise<void>}
 */
async function runEngineLoop(startIndex, options = {}) {
    const session = getSession();
    const stats = {
        success: session.completedEngines.length,
        error: 0,
        total: ENGINE_COUNT,
        startTime: performance.now()
    };

    for (let i = startIndex; i < ALL_ENGINES.length; i++) {
        // Check if stopped
        if (isAnalysisStopped()) {
            session.currentEngineIndex = i;
            saveSessionToStorage();
            log(`⏸️ Analysis paused at engine ${i + 1}/${ENGINE_COUNT}`, 'warning');
            return;
        }

        const engine = ALL_ENGINES[i];

        // Skip already completed
        if (session.completedEngines.includes(engine.name)) {
            continue;
        }

        const engineStartTime = performance.now();

        // Notify start
        if (callbacks.onEngineStart) {
            callbacks.onEngineStart(engine, i);
        }
        if (callbacks.onProgress) {
            callbacks.onProgress(i, ENGINE_COUNT, engine.display);
        }

        log(`⏳ Running ${engine.display}...`, 'info');

        try {
            // Execute engine
            const data = await runEngine(engine.name, {
                useVectorization: options.useVectorization || false
            });

            const duration = performance.now() - engineStartTime;

            // Get Gemma summary
            log(`🤖 Getting Gemma summary for ${engine.display}...`, 'info');
            const summary = await getGemmaSummary(engine.name, engine.display, data);

            // Build result
            const result = {
                status: 'success',
                data: data,
                gemmaSummary: summary,
                dataSize: JSON.stringify(data).length,
                duration: duration
            };

            // Record in session
            recordEngineResult(engine.name, result);
            saveSessionToStorage();

            // Log and notify
            log(`✅ ${engine.display} - Analysis Complete`, 'success', duration);

            if (callbacks.onEngineComplete) {
                callbacks.onEngineComplete(engine, result, duration);
            }

            stats.success++;
        } catch (err) {
            const duration = performance.now() - engineStartTime;

            const errorResult = {
                status: 'error',
                error: err.message,
                duration: duration
            };

            recordEngineResult(engine.name, errorResult);
            saveSessionToStorage();

            log(`❌ ${engine.display} - Failed: ${err.message}`, 'error', duration);

            if (callbacks.onEngineError) {
                callbacks.onEngineError(engine, err, duration);
            }

            stats.error++;
        }

        // Small delay for UI updates
        await new Promise(r => setTimeout(r, 100));
    }

    // All complete
    completeSession();
    saveSessionToStorage();

    const totalDuration = performance.now() - stats.startTime;
    stats.totalTime = totalDuration;

    log(`🎉 All ${ENGINE_COUNT} engines analysis complete!`, 'success', totalDuration);
    log(`📊 Results: ${stats.success} succeeded, ${stats.error} failed`, 'info');

    if (callbacks.onAllComplete) {
        callbacks.onAllComplete(stats);
    }
}

/**
 * Get current progress.
 * @returns {{completed: number, total: number, percent: number}}
 */
export function getProgress() {
    const session = getSession();
    const completed = session.completedEngines.length;
    return {
        completed,
        total: ENGINE_COUNT,
        percent: Math.round((completed / ENGINE_COUNT) * 100)
    };
}
