/**
 * NexusAI API Utilities
 * API wrappers for file upload, engine execution, and Gemma chat.
 *
 * @module nexus/core/api
 */

import { API_BASE, getAuthHeaders } from './config.js';
import { setUploadState, getUploadState, getColumnSelection } from './state.js';

// ============================================================================
// Request Timeout Configuration
// ============================================================================

/**
 * Default timeout for API requests in milliseconds.
 * @type {number}
 */
const REQUEST_TIMEOUT_MS = 60000; // 60 seconds
const GEMMA_TIMEOUT_MS = 90000; // 90 seconds for LLM responses
const GEMMA_MAX_RETRIES = 2;
const GEMMA_RETRY_BASE_MS = 500;
let gemmaQueue = Promise.resolve();

/**
 * GPU VRAM threshold in GB - if below this, warn before starting analysis.
 * @type {number}
 */
const VRAM_WARNING_THRESHOLD_GB = 1.5;

/**
 * Fetch with timeout wrapper to prevent UI freezes.
 * @param {string} url - Request URL
 * @param {Object} options - Fetch options
 * @param {number} [timeoutMs=60000] - Timeout in milliseconds
 * @returns {Promise<Response>}
 * @throws {Error} If request times out
 */
async function fetchWithTimeout(url, options = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    try {
        const response = await fetch(url, {
            ...options,
            signal: controller.signal
        });
        clearTimeout(timeoutId);
        return response;
    } catch (err) {
        clearTimeout(timeoutId);
        if (err.name === 'AbortError') {
            throw new Error(`Request timed out after ${timeoutMs / 1000}s`);
        }
        throw err;
    }
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function fetchJsonWithRetry(url, options, timeoutMs, retries = GEMMA_MAX_RETRIES) {
    let attempt = 0;

    while (attempt <= retries) {
        try {
            const response = await fetchWithTimeout(url, options, timeoutMs);
            if (!response.ok) {
                throw new Error(`Request failed (${response.status})`);
            }
            const text = await response.text();
            try {
                return JSON.parse(text);
            } catch (err) {
                throw new Error('Invalid JSON response');
            }
        } catch (err) {
            if (attempt >= retries) {
                throw err;
            }
            const delay = GEMMA_RETRY_BASE_MS * (2 ** attempt);
            await sleep(delay);
            attempt += 1;
        }
    }

    throw new Error('Request failed');
}

function enqueueGemmaRequest(task) {
    const next = gemmaQueue.then(task, task);
    gemmaQueue = next.catch(() => { });
    return next;
}

// ============================================================================
// GPU Health Monitoring
// ============================================================================

/**
 * Check GPU status before running analysis.
 * @returns {Promise<{available: boolean, vramFreeGb: number, warning: string|null}>}
 */
export async function checkGpuHealth() {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE}/api/ml/gpu-status`,
            { method: 'GET', credentials: 'include' },
            5000 // Quick 5s timeout for health check
        );

        if (!response.ok) {
            return { available: false, vramFreeGb: 0, warning: 'GPU status unavailable' };
        }

        const data = await response.json();

        // Check if CUDA is available
        if (!data.cuda_available) {
            return { available: false, vramFreeGb: 0, warning: 'GPU not available - running on CPU' };
        }

        // Check VRAM if devices exist
        if (data.devices && data.devices.length > 0) {
            const totalVram = data.devices[0].total_memory_gb || 0;
            // Estimate free VRAM (conservative: assume 50% used if no free data)
            const estimatedFreeVram = totalVram * 0.5;

            if (estimatedFreeVram < VRAM_WARNING_THRESHOLD_GB) {
                return {
                    available: true,
                    vramFreeGb: estimatedFreeVram,
                    warning: `Low VRAM available (~${estimatedFreeVram.toFixed(1)}GB). Analysis may be slow.`
                };
            }

            return { available: true, vramFreeGb: estimatedFreeVram, warning: null };
        }

        return { available: true, vramFreeGb: 0, warning: null };
    } catch (err) {
        console.warn('GPU health check failed:', err.message);
        return { available: false, vramFreeGb: 0, warning: 'GPU health check failed: ' + err.message };
    }
}

/**
 * Check GPU coordinator status.
 * @returns {Promise<{owner: string, available: boolean}>}
 */
export async function checkGpuCoordinatorStatus() {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE}/api/gpu-coordinator/gpu/state`,
            {
                method: 'GET',
                headers: getAuthHeaders(),
                credentials: 'include'
            },
            5000
        );

        if (!response.ok) {
            // Try legacy endpoint
            const legacyResponse = await fetchWithTimeout(
                `${API_BASE}/api/gpu-coordinator/status`,
                { method: 'GET', credentials: 'include' },
                5000
            );
            if (legacyResponse.ok) {
                const data = await legacyResponse.json();
                return {
                    owner: data.lock_status?.current_owner || 'unknown',
                    available: data.lock_status?.state === 'transcription'
                };
            }
            return { owner: 'unknown', available: true };
        }

        const data = await response.json();
        return {
            owner: data.owner || 'transcription',
            available: data.owner === 'transcription'
        };
    } catch (err) {
        console.warn('GPU coordinator check failed:', err.message);
        return { owner: 'unknown', available: true }; // Assume available on error
    }
}

// ============================================================================
// File Upload
// ============================================================================

/**
 * Upload a file to the server.
 * @param {File} file - The file to upload
 * @returns {Promise<{filename: string, columns: string[], row_count: number}>}
 * @throws {Error} If upload fails
 */
export async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetchWithTimeout(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
        credentials: 'include'
    }, 120000); // 2 minute timeout for large file uploads

    const data = await response.json();

    if (data.error || data.detail) {
        throw new Error(data.error || data.detail);
    }

    // Update state with upload info
    setUploadState(data.filename, data.columns || []);

    return {
        filename: data.filename,
        columns: data.columns || [],
        row_count: data.row_count || data.rows || 0
    };
}

// ============================================================================
// Engine Execution
// ============================================================================

/**
 * Run a single analysis engine.
 * @param {string} engineName - The engine identifier
 * @param {Object} options - Additional options
 * @param {string} [options.filename] - Override filename
 * @param {string} [options.targetColumn] - Specific target column
 * @param {boolean} [options.useVectorization] - Enable Gemma vectorization
 * @returns {Promise<Object>} Engine result data
 * @throws {Error} If engine execution fails
 */
export async function runEngine(engineName, options = {}) {
    const uploadState = getUploadState();
    const columnSelection = getColumnSelection();

    const filename = options.filename || uploadState.filename;
    if (!filename) {
        throw new Error('No file uploaded');
    }

    const endpoint = `/analytics/run-engine/${engineName}`;

    const response = await fetchWithTimeout(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: getAuthHeaders(),
        credentials: 'include',
        body: JSON.stringify({
            filename: filename,
            target_column: options.targetColumn || columnSelection.target || null,
            config: null,
            use_vectorization: options.useVectorization || false
        })
    }, REQUEST_TIMEOUT_MS); // 60s timeout per engine

    const data = await response.json();

    if (data.error) {
        throw new Error(data.error);
    }

    return data;
}

/**
 * Classify columns in a dataset using the backend statistical classifier.
 * @param {string} filename - The name of the file to classify
 * @returns {Promise<Object>} classification results (target, features, ignored)
 */
export async function classifyColumns(filename) {
    const response = await fetchWithTimeout(`${API_BASE}/analytics/classify-columns`, {
        method: 'POST',
        headers: getAuthHeaders(),
        credentials: 'include',
        body: JSON.stringify({ filename })
    }, 30000); // 30s timeout

    const data = await response.json();

    if (data.error) {
        throw new Error(data.error);
    }

    return data.classification;
}

// ============================================================================
// Gemma AI Chat
// ============================================================================

/**
 * Send a prompt to Gemma and get a response.
 * @param {string} prompt - The prompt to send
 * @param {Object} options - Additional options
 * @param {number} [options.maxTokens=500] - Maximum response tokens
 * @returns {Promise<string>} Gemma's response text
 */
export async function askGemma(prompt, options = {}) {
    // Global skip check - when toggle is checked, don't call Gemma API
    const skipGemma = document.getElementById('skip-gemma-toggle')?.checked || false;
    if (skipGemma) {
        console.log('[askGemma] Skipped - toggle is checked');
        return '';
    }

    const maxTokens = options.maxTokens || 500;
    const sessionId = options.sessionId || null;
    const timeoutMs = options.timeoutMs || GEMMA_TIMEOUT_MS;

    try {
        const requestBody = {
            messages: [{ role: 'user', content: prompt }],
            max_tokens: maxTokens
        };

        // If session_id provided, GPU is held for entire session (for analytics)
        if (sessionId) {
            requestBody.session_id = sessionId;
        }

        const data = await enqueueGemmaRequest(() => fetchJsonWithRetry(
            `${API_BASE}/api/public/chat`,
            {
                method: 'POST',
                headers: getAuthHeaders(),
                credentials: 'include',
                body: JSON.stringify(requestBody)
            },
            timeoutMs
        ));

        return data?.message || data?.response || '';
    } catch (err) {
        console.warn('Gemma chat failed:', err);
        return '';
    }
}

/**
 * Ask Gemma to recommend columns for analysis.
 * @param {string[]} columns - Available column names
 * @param {string[]} excludedTargets - Previously failed target columns
 * @returns {Promise<{target: string, features: string[]}|null>}
 */
export async function askGemmaForColumns(columns, excludedTargets = []) {
    // Skip check - return sensible defaults when Gemma is disabled
    const skipGemma = document.getElementById('skip-gemma-toggle')?.checked || false;
    if (skipGemma) {
        console.log('[askGemmaForColumns] Skipped - using first numeric-looking column as target');
        const availableColumns = columns.filter(col => !excludedTargets.includes(col));
        // Default: first column as target, next 5 as features
        return {
            target: availableColumns[0] || 'target',
            features: availableColumns.slice(1, 6)
        };
    }

    const availableColumns = columns.filter(col => !excludedTargets.includes(col));
    const columnsStr = availableColumns.map((col, i) => `${i + 1}. ${col}`).join('\n');

    const exclusionNote = excludedTargets.length > 0
        ? `\n\nIMPORTANT: Do NOT select these columns as the target (they were already tried and gave poor results): ${excludedTargets.join(', ')}`
        : '';

    const prompt = `You are analyzing a dataset with these columns:

${columnsStr}

Task: Select the best TARGET column for prediction (what we want to predict) and the FEATURE columns to use (input variables).

Rules:
- The target should be something meaningful to predict (not an ID)
- Features should be potential predictors of the target
- Do NOT include the target column in the features list${exclusionNote}

Respond in this EXACT format (no extra text):
target: [column name]
features: [column1, column2, column3, ...]`;

    const responseText = await askGemma(prompt);

    // Parse response
    const targetMatch = responseText.match(/target:\s*(.+)/i);
    const featuresMatch = responseText.match(/features:\s*(.+)/i);

    if (targetMatch && featuresMatch) {
        const target = targetMatch[1].trim().replace(/[\[\]"']/g, '');
        const featuresRaw = featuresMatch[1]
            .replace(/[\[\]]/g, '')
            .split(',')
            .map(f => f.trim().replace(/["']/g, ''));

        // Filter to valid columns and exclude target
        const features = featuresRaw.filter(f => columns.includes(f) && f !== target);

        return { target, features };
    }

    return null;
}

/**
 * Get Gemma summary for engine results.
 * @param {string} engineName - Engine identifier
 * @param {string} engineDisplay - Engine display name
 * @param {Object} data - Engine result data
 * @param {string} [sessionId] - Session ID for GPU retention during analytics
 * @returns {Promise<string>} Summary text
 */
export async function getGemmaSummary(engineName, engineDisplay, data, sessionId = null) {
    // Skip check - return fallback summary when Gemma is disabled
    const skipGemma = document.getElementById('skip-gemma-toggle')?.checked || false;
    if (skipGemma) {
        console.log(`[getGemmaSummary] Skipped for ${engineName} - using fallback`);
        return buildFallbackSummary(engineName, data);
    }

    // Extract key metrics based on data structure
    const dataStr = JSON.stringify(data).substring(0, 2000);

    const prompt = `Summarize this ${engineDisplay} analysis result in 2-3 sentences. Focus on the key insights and business implications:

${dataStr}

Be concise and focus on actionable insights.`;

    const options = { maxTokens: 200 };
    if (sessionId) {
        options.sessionId = sessionId;
    }

    const summary = await askGemma(prompt, options);
    return summary || buildFallbackSummary(engineName, data);
}

export function buildFallbackSummary(engineName, data) {
    if (!data || typeof data !== 'object') {
        return 'Analysis complete. Review the details below.';
    }

    if (data.status === 'requires_time_data') {
        return `This engine requires time-series data. ${data.recommendation || 'Add a date or timestamp column to enable forecasting.'}`;
    }

    if (data.error) {
        return `Analysis encountered an issue: ${data.error}.`;
    }

    const findings = [];

    if (data.best_model) {
        findings.push(`Best model: ${data.best_model}.`);
    }

    if (data.accuracy !== undefined) {
        findings.push(`Accuracy: ${(data.accuracy * 100).toFixed(1)}%.`);
    }

    if (data.cv_score !== undefined) {
        findings.push(`CV score: ${(data.cv_score * 100).toFixed(1)}%.`);
    }

    if (data.n_clusters !== undefined) {
        findings.push(`Identified ${data.n_clusters} clusters.`);
    }

    if (data.anomaly_count !== undefined) {
        findings.push(`Detected ${data.anomaly_count} anomalies.`);
    }

    if (data.summary && typeof data.summary === 'object') {
        if (data.summary.total_revenue) {
            findings.push(`Total revenue: $${Number(data.summary.total_revenue).toLocaleString()}.`);
        }
        if (data.summary.total_cost) {
            findings.push(`Total cost: $${Number(data.summary.total_cost).toLocaleString()}.`);
        }
    }

    if (Array.isArray(data.insights) && data.insights.length) {
        findings.push(data.insights[0]);
    }

    if (data.total_graphs) {
        findings.push(`Generated ${data.total_graphs} visualizations.`);
    }

    if (!findings.length) {
        return 'Analysis complete. Review the details below.';
    }

    return findings.slice(0, 2).join(' ');
}

// ============================================================================
// Health Check
// ============================================================================

/**
 * Check if the API is available.
 * @returns {Promise<boolean>}
 */
export async function checkApiHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`, {
            method: 'GET',
            credentials: 'include'
        });
        return response.ok;
    } catch {
        return false;
    }
}
