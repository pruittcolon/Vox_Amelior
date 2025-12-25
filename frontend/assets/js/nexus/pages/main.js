/**
 * NexusAI Main Entry Point
 * Initializes all components and wires up the analysis page.
 *
 * @module nexus/pages/main
 */

// Core modules
import { API_BASE, getAuthHeaders, STORAGE_KEYS } from '../core/config.js';
import {
    getSession,
    getUploadState,
    setUploadState,
    clearUploadState,
    loadSessionFromStorage,
    saveSessionToStorage,
    clearSessionStorage,
    initSession,
    setAnalysisStopped
} from '../core/state.js';
import { uploadFile, runEngine, getGemmaSummary } from '../core/api.js';

// Engine modules
import { ALL_ENGINES, ENGINE_COUNT, getEngineByName } from '../engines/engine-definitions.js';
import { registerCallbacks, startAnalysis, resumeAnalysis, stopAnalysis, getProgress } from '../engines/engine-runner.js';
import { createEngineCard, displayEngineResults, updateEngineCardStatus, formatDuration } from '../engines/engine-results.js';

// Component modules
import { initLog, log, startTiming, getElapsedTime, clearLog } from '../components/log.js';
import { initDashboard, resetDashboard, trackEnginePerformance } from '../components/dashboard.js';

// ============================================================================
// Global UI Interface (for onclick handlers in HTML)
// ============================================================================

window.NexusUI = {
    // Reset upload area
    resetUpload() {
        const uploadArea = document.getElementById('upload-area');
        const analyzeBtn = document.getElementById('analyze-btn');

        clearUploadState();

        if (analyzeBtn) {
            analyzeBtn.disabled = true;
            analyzeBtn.innerHTML = '⚡ Start Full Analysis';
        }

        if (uploadArea) {
            uploadArea.innerHTML = `
        <div class="upload-icon">📊</div>
        <div class="upload-title">Drop your database file here</div>
        <div class="upload-subtitle">or click to browse • Max 50MB</div>
        <div class="upload-formats">
          <span class="upload-format">.csv</span>
          <span class="upload-format">.json</span>
          <span class="upload-format">.xlsx</span>
          <span class="upload-format">.sqlite</span>
          <span class="upload-format">.parquet</span>
        </div>
      `;
        }

        // Hide progress bar
        const progressContainer = document.getElementById('progress-container');
        if (progressContainer) progressContainer.classList.remove('active');
    },

    // Toggle engine card expansion
    toggleEngineCard(engineName) {
        const cards = document.querySelectorAll(`[id^="engine-card-${engineName}"]`);
        cards.forEach(card => card.classList.toggle('expanded'));
    },

    // Toggle raw data visibility
    toggleRawData(id) {
        const content = document.getElementById(id);
        const icon = document.getElementById('icon-' + id);
        if (content && icon) {
            const isHidden = content.style.display === 'none';
            content.style.display = isHidden ? 'block' : 'none';
            icon.textContent = isHidden ? '▼' : '▶';
        }
    },

    // Switch engine category tab
    switchCategory(category) {
        // Update buttons
        document.querySelectorAll('.engine-category-btn').forEach(btn => btn.classList.remove('active'));
        event.target.closest('.engine-category-btn').classList.add('active');

        // Show/hide content
        document.querySelectorAll('.engine-category-content').forEach(content => {
            content.style.display = 'none';
            content.classList.remove('active');
        });
        const targetContent = document.getElementById(`category-${category}`);
        if (targetContent) {
            targetContent.style.display = 'block';
            targetContent.classList.add('active');
        }
    },

    // Send follow-up question
    async sendFollowUp(engineName) {
        const input = document.getElementById(`input-${engineName}`);
        const messages = document.getElementById(`messages-${engineName}`);
        if (!input || !messages || !input.value.trim()) return;

        const question = input.value.trim();
        input.value = '';

        // Add user message
        messages.innerHTML += `
      <div class="followup-msg user">
        <span class="msg-label">You:</span>
        <span class="msg-text">${escapeHtml(question)}</span>
      </div>
    `;

        // Get context from session
        const session = getSession();
        const result = session.results[engineName];

        // Ask Gemma
        try {
            const response = await fetch(`${API_BASE}/api/public/chat`, {
                method: 'POST',
                headers: getAuthHeaders(),
                credentials: 'include',
                body: JSON.stringify({
                    messages: [{ role: 'user', content: `Based on this ${engineName} analysis result:\n${JSON.stringify(result?.data).substring(0, 1500)}\n\nUser question: ${question}` }],
                    max_tokens: 300
                })
            });
            const data = await response.json();
            const answer = data.message || data.response || 'Unable to get response.';

            messages.innerHTML += `
        <div class="followup-msg assistant">
          <span class="msg-label">🤖 Gemma:</span>
          <span class="msg-text">${escapeHtml(answer)}</span>
        </div>
      `;
        } catch (err) {
            messages.innerHTML += `
        <div class="followup-msg assistant">
          <span class="msg-label">🤖 Gemma:</span>
          <span class="msg-text">Error: ${err.message}</span>
        </div>
      `;
        }

        messages.scrollTop = messages.scrollHeight;
    },

    // Stop analysis
    stopAnalysis() {
        stopAnalysis();
        document.getElementById('stop-btn').style.display = 'none';
        document.getElementById('resume-btn').style.display = 'inline-block';
    },

    // Resume analysis
    async resumeAnalysis() {
        const savedSession = loadSessionFromStorage();
        if (savedSession) {
            document.getElementById('stop-btn').style.display = 'inline-block';
            document.getElementById('resume-btn').style.display = 'none';
            await resumeAnalysis(savedSession, { useVectorization: document.getElementById('use-vectors')?.checked });
        }
    },

    // Clear analysis
    clearAnalysis() {
        if (!confirm('Clear all analysis results?')) return;
        clearSessionStorage();
        ['all', 'ml', 'financial', 'advanced'].forEach(cat => {
            const container = document.getElementById(`${cat}-engines-results`);
            if (container) container.innerHTML = '';
        });
        document.getElementById('engines-progress').textContent = '0/22';
        document.getElementById('engines-status').textContent = 'Cleared';
        document.getElementById('engines-total-time').textContent = '0.0s';
        document.getElementById('all-engines-section').style.display = 'none';
        document.getElementById('analyze-btn').disabled = false;
        log('🗑️ Analysis cleared', 'info');
    }
};

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Initialize log
    initLog('#log');

    // DOM elements
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const analyzeBtn = document.getElementById('analyze-btn');

    // Setup upload handlers
    if (uploadArea && fileInput) {
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
                handleFileUpload(e.dataTransfer.files[0]);
            }
        });

        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files[0]) {
                handleFileUpload(e.target.files[0]);
            }
        });
    }

    // Setup analyze button
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', runFullAnalysis);
    }

    // Setup dropdown hover
    document.querySelectorAll('.nexus-dropdown').forEach(dropdown => {
        dropdown.addEventListener('mouseenter', () => {
            dropdown.querySelector('.nexus-dropdown-menu').style.display = 'block';
        });
        dropdown.addEventListener('mouseleave', () => {
            dropdown.querySelector('.nexus-dropdown-menu').style.display = 'none';
        });
    });

    // Load recent analyses
    loadRecentAnalyses();

    // Register engine runner callbacks
    registerCallbacks({
        onProgress: updateProgressUI,
        onEngineStart: handleEngineStart,
        onEngineComplete: handleEngineComplete,
        onEngineError: handleEngineError,
        onAllComplete: handleAllComplete,
        onLog: log
    });
});

// ============================================================================
// File Upload Handler
// ============================================================================

async function handleFileUpload(file) {
    const uploadArea = document.getElementById('upload-area');
    const analyzeBtn = document.getElementById('analyze-btn');

    uploadArea.innerHTML = `
    <div class="upload-icon">⏳</div>
    <div class="upload-title">Uploading ${escapeHtml(file.name)}...</div>
    <div class="upload-subtitle">Please wait</div>
  `;

    try {
        const result = await uploadFile(file);

        uploadArea.innerHTML = `
      <div class="upload-icon">✅</div>
      <div class="upload-title">${escapeHtml(result.filename)}</div>
      <div class="upload-subtitle">
        ${result.columns.length} columns • ${result.row_count.toLocaleString()} rows
      </div>
      <div style="margin-top: 1rem;">
        <span class="vox-btn vox-btn-ghost" style="font-size: 0.85rem;" onclick="event.stopPropagation(); window.NexusUI.resetUpload()">
          🔄 Upload Different File
        </span>
      </div>
    `;

        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = `⚡ Start Full Analysis on ${escapeHtml(result.filename)}`;

        log(`✅ Uploaded: ${result.filename} (${result.columns.length} cols, ${result.row_count.toLocaleString()} rows)`, 'success');
    } catch (err) {
        uploadArea.innerHTML = `
      <div class="upload-icon">❌</div>
      <div class="upload-title">Upload Failed</div>
      <div class="upload-subtitle" style="color: var(--vox-error);">${escapeHtml(err.message)}</div>
      <div style="margin-top: 1rem;">
        <span class="vox-btn vox-btn-ghost" style="font-size: 0.85rem;" onclick="event.stopPropagation(); window.NexusUI.resetUpload()">
          🔄 Try Again
        </span>
      </div>
    `;
        log(`❌ Upload failed: ${err.message}`, 'error');
    }
}

// ============================================================================
// Analysis Execution
// ============================================================================

async function runFullAnalysis() {
    const uploadState = getUploadState();
    if (!uploadState.filename) return;

    const useVectorization = document.getElementById('use-vectors')?.checked || false;

    // Show results section
    document.getElementById('all-engines-section').style.display = 'block';
    document.getElementById('engines-database-name').textContent = uploadState.filename;

    // Clear previous results
    ['all', 'ml', 'financial', 'advanced'].forEach(cat => {
        const container = document.getElementById(`${cat}-engines-results`);
        if (container) container.innerHTML = '';
    });

    // Reset dashboard
    resetDashboard();
    initDashboard();

    // Disable analyze button
    document.getElementById('analyze-btn').disabled = true;

    // Show and reset progress bar
    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const progressStats = document.getElementById('progress-stats');
    const progressEngine = document.getElementById('progress-engine');

    if (progressContainer) {
        progressContainer.classList.add('active');
        progressBar.style.width = '0%';
        progressStats.textContent = '0/22 engines complete';
        progressEngine.textContent = 'Starting...';
    }

    // Clear and start log
    clearLog();
    startTiming();

    log(`🧪 Starting comprehensive analysis with all ${ENGINE_COUNT} engines...`, 'info');
    log(`📁 Database: ${uploadState.filename}`, 'info');

    // Start analysis
    await startAnalysis({ useVectorization });
}

// ============================================================================
// Engine Callbacks
// ============================================================================

function updateProgressUI(completed, total, engineName) {
    const elapsed = formatDuration(getElapsedTime());
    const percentage = Math.round((completed / total) * 100);

    // Update summary stats
    document.getElementById('engines-progress').textContent = `${completed}/${total}`;
    document.getElementById('engines-status').textContent = `Running ${engineName}...`;
    document.getElementById('engines-total-time').textContent = elapsed;

    // Update progress bar
    const progressBar = document.getElementById('progress-bar');
    const progressStats = document.getElementById('progress-stats');
    const progressEngine = document.getElementById('progress-engine');

    if (progressBar) {
        progressBar.style.width = `${percentage}%`;
        if (completed === total) {
            progressBar.classList.remove('animated');
        }
    }
    if (progressStats) progressStats.textContent = `${completed}/${total} engines complete`;
    if (progressEngine) progressEngine.textContent = `🔄 Running: ${engineName}`;
}

function handleEngineStart(engine, index) {
    // Create cards in both "all" and category containers
    const card = createEngineCard(engine, index);
    const cardClone = card.cloneNode(true);

    document.getElementById('all-engines-results').appendChild(card);
    document.getElementById(`${engine.category}-engines-results`).appendChild(cardClone);

    updateEngineCardStatus(card, 'running');
    updateEngineCardStatus(cardClone, 'running');
}

function handleEngineComplete(engine, result, duration) {
    // Find and update cards
    const cards = document.querySelectorAll(`#engine-card-${engine.name}`);
    cards.forEach(card => {
        displayEngineResults(card, result);
        updateEngineCardStatus(card, result.status === 'success' ? 'success' : 'fallback');
    });

    // Track for dashboard
    trackEnginePerformance(engine, duration, 'success', result.dataSize);

    // Update category stats
    updateCategoryStats();
}

function handleEngineError(engine, error, duration) {
    const cards = document.querySelectorAll(`#engine-card-${engine.name}`);
    cards.forEach(card => {
        displayEngineResults(card, { status: 'error', error: error.message, duration });
        updateEngineCardStatus(card, 'error');
    });

    trackEnginePerformance(engine, duration, 'error', 0);
    updateCategoryStats();
}

function handleAllComplete(stats) {
    const elapsed = formatDuration(stats.totalTime);
    document.getElementById('engines-status').textContent = 'Complete!';
    document.getElementById('engines-total-time').textContent = elapsed;
    document.getElementById('analyze-btn').disabled = false;

    // Update progress bar to complete
    const progressBar = document.getElementById('progress-bar');
    const progressStats = document.getElementById('progress-stats');
    const progressEngine = document.getElementById('progress-engine');

    if (progressBar) {
        progressBar.style.width = '100%';
        progressBar.classList.remove('animated');
    }
    if (progressStats) progressStats.textContent = `${ENGINE_COUNT}/${ENGINE_COUNT} engines complete`;
    if (progressEngine) progressEngine.textContent = '✅ Analysis Complete!';

    log(`🎉 All ${ENGINE_COUNT} engines complete!`, 'success', stats.totalTime);
    log(`📊 Results: ${stats.success} succeeded, ${stats.error} failed`, 'info');
}

function updateCategoryStats() {
    const session = getSession();
    const categories = ['all', 'ml', 'financial', 'advanced'];

    categories.forEach(cat => {
        let success = 0, error = 0, pending = 0;
        const engines = cat === 'all' ? ALL_ENGINES : ALL_ENGINES.filter(e => e.category === cat);

        engines.forEach(engine => {
            const result = session.results[engine.name];
            if (!result) pending++;
            else if (result.status === 'success') success++;
            else error++;
        });

        const successEl = document.getElementById(`${cat}-success-count`);
        const errorEl = document.getElementById(`${cat}-error-count`);
        const pendingEl = document.getElementById(`${cat}-pending-count`);

        if (successEl) successEl.textContent = `✓ ${success}`;
        if (errorEl) errorEl.textContent = `✗ ${error}`;
        if (pendingEl) pendingEl.textContent = `⏳ ${pending}`;
    });
}

// ============================================================================
// Recent Analyses
// ============================================================================

function loadRecentAnalyses() {
    try {
        const saved = localStorage.getItem(STORAGE_KEYS.analysisSession);
        if (saved) {
            const session = JSON.parse(saved);
            if (session.filename && Object.keys(session.results).length > 0) {
                document.getElementById('recent-section').style.display = 'block';
                const grid = document.getElementById('recent-grid');
                grid.innerHTML = `
          <div class="nexus-recent-card" onclick="window.NexusUI.resumeAnalysis()">
            <div class="nexus-recent-card-header">
              <span class="nexus-recent-card-icon">📄</span>
              <span class="nexus-recent-card-title">${escapeHtml(session.filename)}</span>
            </div>
            <div class="nexus-recent-card-meta">
              <span>${Object.keys(session.results).length}/${ENGINE_COUNT} engines</span>
              <span>${session.status}</span>
            </div>
          </div>
        `;
            }
        }
    } catch (e) {
        console.log('No recent analyses found');
    }
}

// ============================================================================
// Utilities
// ============================================================================

function escapeHtml(text) {
    if (!text) return '';
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

// Expose scroll function globally
window.scrollToUpload = function () {
    document.getElementById('quick-start').scrollIntoView({ behavior: 'smooth' });
};
