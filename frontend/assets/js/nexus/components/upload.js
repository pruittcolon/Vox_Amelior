/**
 * NexusAI Upload Component
 * Drag and drop file upload with progress feedback.
 *
 * @module nexus/components/upload
 */

import { uploadFile } from '../core/api.js';
import { setUploadState, clearUploadState, getUploadState } from '../core/state.js';
import { DEFAULT_SETTINGS } from '../core/config.js';

// ============================================================================
// State
// ============================================================================

let uploadCallbacks = {
    onUploadStart: null,  // (file) => void
    onUploadProgress: null, // (percent) => void
    onUploadSuccess: null, // ({filename, columns, row_count}) => void
    onUploadError: null   // (error) => void
};

// ============================================================================
// Public API
// ============================================================================

/**
 * Register upload event callbacks.
 * @param {Object} callbacks
 */
export function registerUploadCallbacks(callbacks) {
    uploadCallbacks = { ...uploadCallbacks, ...callbacks };
}

/**
 * Initialize upload area with drag/drop handlers.
 * @param {string|HTMLElement} uploadAreaSelector - Selector or element
 * @param {string|HTMLElement} fileInputSelector - Selector or element
 */
export function initUploadArea(uploadAreaSelector, fileInputSelector) {
    const uploadArea = typeof uploadAreaSelector === 'string'
        ? document.querySelector(uploadAreaSelector)
        : uploadAreaSelector;

    const fileInput = typeof fileInputSelector === 'string'
        ? document.querySelector(fileInputSelector)
        : fileInputSelector;

    if (!uploadArea || !fileInput) {
        console.warn('Upload area or file input not found');
        return;
    }

    // Drag events
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files[0]) {
            handleFileUpload(e.dataTransfer.files[0], uploadArea);
        }
    });

    // Click to browse
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files[0]) {
            handleFileUpload(e.target.files[0], uploadArea);
        }
    });
}

/**
 * Handle file upload process.
 * @param {File} file
 * @param {HTMLElement} uploadArea
 */
async function handleFileUpload(file, uploadArea) {
    // Validate file size
    if (file.size > DEFAULT_SETTINGS.maxFileSize) {
        const maxMB = DEFAULT_SETTINGS.maxFileSize / (1024 * 1024);
        if (uploadCallbacks.onUploadError) {
            uploadCallbacks.onUploadError(new Error(`File too large. Maximum size is ${maxMB}MB.`));
        }
        return;
    }

    // Notify start
    if (uploadCallbacks.onUploadStart) {
        uploadCallbacks.onUploadStart(file);
    }

    // Update UI to loading state
    uploadArea.innerHTML = `
    <div class="nexus-upload-icon">⏳</div>
    <div class="nexus-upload-title">Uploading ${escapeHtml(file.name)}...</div>
    <div class="nexus-upload-subtitle">Please wait</div>
  `;

    try {
        const result = await uploadFile(file);

        // Update UI to success state
        uploadArea.innerHTML = `
      <div class="nexus-upload-icon">✅</div>
      <div class="nexus-upload-title">${escapeHtml(result.filename)}</div>
      <div class="nexus-upload-subtitle">
        ${result.columns.length} columns • ${result.row_count.toLocaleString()} rows
      </div>
      <div style="margin-top: var(--nexus-space-4, 1rem);">
        <span class="nexus-btn nexus-btn-ghost nexus-btn-sm" onclick="event.stopPropagation(); window.NexusUI.resetUpload()">
          🔄 Upload Different File
        </span>
      </div>
    `;

        if (uploadCallbacks.onUploadSuccess) {
            uploadCallbacks.onUploadSuccess(result);
        }
    } catch (err) {
        // Update UI to error state
        uploadArea.innerHTML = `
      <div class="nexus-upload-icon">❌</div>
      <div class="nexus-upload-title">Upload Failed</div>
      <div class="nexus-upload-subtitle" style="color: var(--nexus-error-400, #f87171);">
        ${escapeHtml(err.message)}
      </div>
      <div style="margin-top: var(--nexus-space-4, 1rem);">
        <span class="nexus-btn nexus-btn-ghost nexus-btn-sm" onclick="event.stopPropagation(); window.NexusUI.resetUpload()">
          🔄 Try Again
        </span>
      </div>
    `;

        if (uploadCallbacks.onUploadError) {
            uploadCallbacks.onUploadError(err);
        }
    }
}

/**
 * Reset upload area to initial state.
 * @param {string|HTMLElement} uploadAreaSelector
 */
export function resetUploadArea(uploadAreaSelector) {
    const uploadArea = typeof uploadAreaSelector === 'string'
        ? document.querySelector(uploadAreaSelector)
        : uploadAreaSelector;

    if (!uploadArea) return;

    clearUploadState();

    uploadArea.innerHTML = `
    <div class="nexus-upload-icon">📊</div>
    <div class="nexus-upload-title">Drop your database file here</div>
    <div class="nexus-upload-subtitle">or click to browse • Max 50MB</div>
    <div class="nexus-upload-formats">
      <span class="nexus-upload-format">.csv</span>
      <span class="nexus-upload-format">.json</span>
      <span class="nexus-upload-format">.xlsx</span>
      <span class="nexus-upload-format">.sqlite</span>
      <span class="nexus-upload-format">.parquet</span>
    </div>
  `;
}

/**
 * Check if a file is currently uploaded.
 * @returns {boolean}
 */
export function hasUploadedFile() {
    const state = getUploadState();
    return !!state.filename;
}

/**
 * Get current uploaded file info.
 * @returns {{filename: string|null, columns: string[]}}
 */
export function getUploadedFileInfo() {
    return getUploadState();
}

// ============================================================================
// Utility
// ============================================================================

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
