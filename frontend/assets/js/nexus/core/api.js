/**
 * NexusAI API Utilities
 * API wrappers for file upload, engine execution, and Gemma chat.
 *
 * @module nexus/core/api
 */

import { API_BASE, getAuthHeaders } from './config.js';
import { setUploadState, getUploadState, getColumnSelection } from './state.js';

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

    const response = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
        credentials: 'include'
    });

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

    const response = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: getAuthHeaders(),
        credentials: 'include',
        body: JSON.stringify({
            filename: filename,
            target_column: options.targetColumn || columnSelection.target || null,
            config: null,
            use_vectorization: options.useVectorization || false
        })
    });

    const data = await response.json();

    if (data.error) {
        throw new Error(data.error);
    }

    return data;
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
    const maxTokens = options.maxTokens || 500;

    try {
        const response = await fetch(`${API_BASE}/api/public/chat`, {
            method: 'POST',
            headers: getAuthHeaders(),
            credentials: 'include',
            body: JSON.stringify({
                messages: [{ role: 'user', content: prompt }],
                max_tokens: maxTokens
            })
        });

        const data = await response.json();
        return data.message || data.response || '';
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
 * @returns {Promise<string>} Summary text
 */
export async function getGemmaSummary(engineName, engineDisplay, data) {
    // Extract key metrics based on data structure
    const dataStr = JSON.stringify(data).substring(0, 2000);

    const prompt = `Summarize this ${engineDisplay} analysis result in 2-3 sentences. Focus on the key insights and business implications:

${dataStr}

Be concise and focus on actionable insights.`;

    const summary = await askGemma(prompt, { maxTokens: 200 });
    return summary || 'Analysis complete. Review the details below.';
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
