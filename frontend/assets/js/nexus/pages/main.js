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
import { uploadFile, runEngine, getGemmaSummary, checkGpuHealth, checkGpuCoordinatorStatus, askGemma, classifyColumns } from '../core/api.js';

// Engine modules
import { ALL_ENGINES, ENGINE_COUNT, getEngineByName } from '../engines/engine-definitions.js';
import { registerCallbacks, startAnalysis, resumeAnalysis, stopAnalysis, cancelAnalysis, getProgress } from '../engines/engine-runner.js';
import { createEngineCard, displayEngineResults, updateEngineCardStatus, formatDuration } from '../engines/engine-results.js';

// Component modules
import { initLog, log, startTiming, getElapsedTime, clearLog } from '../components/log.js';
import { initDashboard, resetDashboard, trackEnginePerformance } from '../components/dashboard.js';
import { initColumnMapper, populateColumnMapper } from '../components/column-mapper.js';

// ============================================================================
// Performance Mode (Low GPU / Low RAM)
// ============================================================================

const performanceState = window.NexusPerformance || { lowPower: false, reason: '' };
window.NexusPerformance = performanceState;

function enableLowPowerMode(reason) {
    if (!performanceState.lowPower) {
        performanceState.lowPower = true;
    }
    if (reason) {
        if (performanceState.reason) {
            if (!performanceState.reason.includes(reason)) {
                performanceState.reason += `; ${reason}`;
            }
        } else {
            performanceState.reason = reason;
        }
    }
}

function evaluateClientPerformance() {
    const hints = [];
    const deviceMemory = navigator.deviceMemory || 0;
    const cores = navigator.hardwareConcurrency || 0;
    const prefersReducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches;

    if (deviceMemory && deviceMemory <= 6) hints.push(`device memory ${deviceMemory}GB`);
    if (cores && cores <= 4) hints.push(`${cores} CPU cores`);
    if (prefersReducedMotion) hints.push('reduced motion');

    if (hints.length) {
        enableLowPowerMode(hints.join(', '));
    }
}

evaluateClientPerformance();

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
        const cards = document.querySelectorAll(`.engine-result-card[data-engine="${engineName}"]`);
        cards.forEach(card => card.classList.toggle('expanded'));
        setTimeout(() => window.NexusViz?.resizeVisibleCharts?.(), 150);
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
        document.querySelectorAll('.engine-category-btn').forEach(btn => {
            btn.classList.remove('active', 'vox-btn-primary');
            btn.classList.add('vox-btn-ghost');
        });
        const activeBtn = document.querySelector(`.engine-category-btn[data-category="${category}"]`);
        if (activeBtn) {
            activeBtn.classList.remove('vox-btn-ghost');
            activeBtn.classList.add('active', 'vox-btn-primary');
        }

        // Show/hide content
        ['all', 'ml', 'financial', 'advanced'].forEach(cat => {
            const content = document.getElementById(`category-${cat}`);
            if (content) {
                content.style.display = cat === category ? 'block' : 'none';
            }
        });

        window.NexusViz?.resizeVisibleCharts?.();
    },

    // Send follow-up question
    async sendFollowUp(engineName, cardId) {
        const inputId = cardId ? `input-${cardId}` : `input-${engineName}`;
        const messagesId = cardId ? `messages-${cardId}` : `messages-${engineName}`;
        const input = document.getElementById(inputId);
        const messages = document.getElementById(messagesId);
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

        // Ask Gemma (if not skipped)
        const skipGemma = document.getElementById('skip-gemma-toggle')?.checked || false;

        if (skipGemma) {
            messages.innerHTML += `
        <div class="followup-msg assistant">
          <span class="msg-label">System:</span>
          <span class="msg-text">Gemma AI is disabled in Fast Mode. Uncheck "Skip Gemma Summaries" to enable follow-up questions.</span>
        </div>
      `;
            messages.scrollTop = messages.scrollHeight;
            return;
        }

        try {
            const context = result?.data ? JSON.stringify(result.data).substring(0, 1500) : 'No engine data available.';
            const prompt = `Based on this ${engineName} analysis result:\n${context}\n\nUser question: ${question}`;
            const answer = await askGemma(prompt, { maxTokens: 300 });

            messages.innerHTML += `
        <div class="followup-msg assistant">
          <span class="msg-label">🤖 Gemma:</span>
          <span class="msg-text">${escapeHtml(answer || 'Unable to get response.')}</span>
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

    // =========================================================================
    // Analysis Control State Management
    // =========================================================================

    // Show running state controls (Pause + Stop)
    showRunningControls() {
        document.getElementById('analyze-btn').style.display = 'none';
        document.getElementById('running-controls').classList.add('active');
        document.getElementById('paused-controls').classList.remove('active');
        document.getElementById('analysis-status-indicator').style.display = 'block';
        document.getElementById('running-indicator').style.display = 'inline-flex';
        document.getElementById('paused-indicator').style.display = 'none';
        document.getElementById('analysis-helper-text').textContent = 'Analysis in progress... You can pause or stop at any time.';
    },

    // Show paused state controls (Resume + Cancel)
    showPausedControls() {
        document.getElementById('analyze-btn').style.display = 'none';
        document.getElementById('running-controls').classList.remove('active');
        document.getElementById('paused-controls').classList.add('active');
        document.getElementById('analysis-status-indicator').style.display = 'block';
        document.getElementById('running-indicator').style.display = 'none';
        document.getElementById('paused-indicator').style.display = 'inline-flex';
        document.getElementById('analysis-helper-text').textContent = 'Analysis paused. Resume to continue or cancel to start fresh.';
    },

    // Show idle state controls (Start button)
    showIdleControls() {
        const analyzeBtn = document.getElementById('analyze-btn');
        analyzeBtn.style.display = 'inline-flex';
        // Reset button text to default
        analyzeBtn.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" style="margin-right: 8px;"><polygon points="5 3 19 12 5 21 5 3"/></svg>Start Full Analysis';

        document.getElementById('running-controls').classList.remove('active');
        document.getElementById('paused-controls').classList.remove('active');
        document.getElementById('analysis-status-indicator').style.display = 'none';
        document.getElementById('analysis-helper-text').textContent = 'All 22 engines will analyze your data sequentially with AI summaries';
    },

    // Pause analysis (stop but allow resume)
    pauseAnalysis() {
        stopAnalysis();
        this.showPausedControls();
        const statusEl = document.getElementById('engines-status');
        if (statusEl) statusEl.textContent = 'Paused';
    },

    // Resume a paused analysis
    async resumeAnalysis() {
        const savedSession = loadSessionFromStorage();
        if (savedSession) {
            this.showRunningControls();
            const skipGemma = document.getElementById('skip-gemma-toggle')?.checked || false;
            await resumeAnalysis(savedSession, {
                useVectorization: document.getElementById('use-vectors')?.checked,
                skipGemma
            });
        } else {
            console.warn('[NexusUI] No saved session to resume');
            this.showIdleControls();
        }
    },

    // Stop and cancel analysis completely
    stopAndCancelAnalysis() {
        if (!confirm('Stop analysis and discard all progress?')) return;
        cancelAnalysis();
        this.showIdleControls();
        ['all', 'ml', 'financial', 'advanced'].forEach(cat => {
            const container = document.getElementById(`${cat}-engines-results`);
            if (container) container.innerHTML = '';
        });
        const progressEl = document.getElementById('engines-progress');
        const statusEl = document.getElementById('engines-status');
        const timeEl = document.getElementById('engines-total-time');
        if (progressEl) progressEl.textContent = '0/22';
        if (statusEl) statusEl.textContent = 'Stopped';
        if (timeEl) timeEl.textContent = '0.0s';
        document.getElementById('all-engines-section').style.display = 'none';
        document.getElementById('progress-container').classList.remove('active');
    },

    // Clear/Cancel analysis (alias for clearAnalysis used by Cancel button)
    clearAnalysis() {
        if (!confirm('Clear all analysis results and reset session?')) return;
        cancelAnalysis();
        this.showIdleControls();
        ['all', 'ml', 'financial', 'advanced'].forEach(cat => {
            const container = document.getElementById(`${cat}-engines-results`);
            if (container) container.innerHTML = '';
        });
        const progressEl = document.getElementById('engines-progress');
        const statusEl = document.getElementById('engines-status');
        const timeEl = document.getElementById('engines-total-time');
        if (progressEl) progressEl.textContent = '0/22';
        if (statusEl) statusEl.textContent = 'Ready';
        if (timeEl) timeEl.textContent = '0.0s';
        document.getElementById('all-engines-section').style.display = 'none';
        document.getElementById('progress-container').classList.remove('active');
        document.getElementById('analyze-btn').disabled = false;
    },

    // Re-run a single engine analysis
    async retestEngine(engineName) {
        const session = getSession();
        if (!session.filename) {
            console.warn('[NexusUI] No active session, cannot retest engine');
            return;
        }

        // Find the engine definition
        const engine = ALL_ENGINES.find(e => e.name === engineName);
        if (!engine) {
            console.error(`[NexusUI] Engine not found: ${engineName}`);
            return;
        }

        // Update card status to running
        const cards = document.querySelectorAll(`.engine-result-card[data-engine="${engineName}"]`);
        cards.forEach(card => {
            const statusEl = card.querySelector('.engine-status');
            if (statusEl) {
                statusEl.classList.remove('success', 'error', 'pending', 'fallback');
                statusEl.classList.add('running');
                statusEl.textContent = 'Re-running...';
            }
            const bodyEl = card.querySelector('.engine-card-body');
            if (bodyEl) {
                bodyEl.innerHTML = '<div class="loading-spinner">Re-analyzing your data with AI engine...</div>';
            }
        });

        try {
            // Import and call runSingleEngine
            const { runSingleEngine } = await import('../engines/engine-runner.js');
            const skipGemma = document.getElementById('skip-gemma-toggle')?.checked || false;

            await runSingleEngine(engine, session.filename, {
                useVectorization: document.getElementById('use-vectors')?.checked,
                skipGemma
            });
        } catch (err) {
            console.error(`[NexusUI] Error retesting ${engineName}:`, err);
            cards.forEach(card => {
                const statusEl = card.querySelector('.engine-status');
                if (statusEl) {
                    statusEl.classList.remove('running');
                    statusEl.classList.add('error');
                    statusEl.textContent = 'Failed';
                }
                const bodyEl = card.querySelector('.engine-card-body');
                if (bodyEl) {
                    bodyEl.innerHTML = `<div class="engine-error"><span class="error-icon">X</span><span class="error-message">${err.message}</span></div>`;
                }
            });
        }
    },

    // Pause the current analysis run (allows resuming later)
    pauseAnalysis() {
        stopAnalysis();
        const pauseBtn = document.getElementById('pause-btn');
        const resumeBtn = document.getElementById('resume-btn');
        if (pauseBtn) pauseBtn.style.display = 'none';
        if (resumeBtn) resumeBtn.style.display = 'inline-block';
        document.getElementById('engines-status').textContent = 'Paused';
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

    // Check for resumable session on page load
    const savedSession = loadSessionFromStorage();
    if (savedSession && savedSession.completedEngines && savedSession.completedEngines.length > 0 &&
        savedSession.completedEngines.length < ENGINE_COUNT) {
        // There's a partially completed session - show paused state
        console.log('[Nexus] Found resumable session: ' + savedSession.completedEngines.length + '/' + ENGINE_COUNT + ' engines complete');
        window.NexusUI.showPausedControls();
        document.getElementById('analysis-helper-text').textContent =
            'Previous session found (' + savedSession.completedEngines.length + '/' + ENGINE_COUNT + ' engines). Resume to continue.';
    }

    // Initialize Column Mapper
    initColumnMapper('#column-mapper-root');
});

// ============================================================================
// File Upload Handler
// ============================================================================

async function handleFileUpload(file) {
    const uploadArea = document.getElementById('upload-area');
    const analyzeBtn = document.getElementById('analyze-btn');

    uploadArea.innerHTML = `
    <div class="upload-icon">...</div>
    <div class="upload-title">Uploading ${escapeHtml(file.name)}...</div>
    <div class="upload-subtitle">Please wait</div>
  `;

    try {
        const result = await uploadFile(file);

        // CRITICAL: Save upload state so runFullAnalysis() can access it
        setUploadState(result.filename, result.columns);

        uploadArea.innerHTML = `
      <div class="upload-icon">OK</div>
      <div class="upload-title">${escapeHtml(result.filename)}</div>
      <div class="upload-subtitle">
        ${result.columns.length} columns - ${result.row_count.toLocaleString()} rows
      </div>
      <div style="margin-top: 1rem;">
        <span class="vox-btn vox-btn-ghost" style="font-size: 0.85rem;" onclick="event.stopPropagation(); window.NexusUI.resetUpload()">
          Upload Different File
        </span>
      </div>
    `;

        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = `Start Full Analysis on ${escapeHtml(result.filename)} `;

        // Show single engine test panel for individual testing
        if (window.showSingleEnginePanel) {
            const defaultTarget = result.columns.length > 0 ? result.columns[0] : 'value';
            window.showSingleEnginePanel(result.filename, defaultTarget);
        }

        // --- NEW: Run Statistical Classifier & Populate Drag-and-Drop UI ---
        try {
            log('🤖 Auto-classifying columns...', 'info');
            const classification = await classifyColumns(result.filename);

            // Show the mapper container
            const mapperRoot = document.getElementById('column-mapper-root');
            if (mapperRoot) mapperRoot.style.display = 'block';

            // Populate with data
            populateColumnMapper(result.columns, classification);
            log('✨ Columns auto-mapped to Target/Features', 'success');
        } catch (classErr) {
            console.error(classErr);
            log(`⚠️ Column classification failed: ${classErr.message} `, 'warning');
            // Fallback: just show all as ignored or features? 
            // The mapper handles empty/default state gracefully usually.
        }
        // -------------------------------------------------------------------

        log(`Uploaded: ${result.filename} (${result.columns.length} cols, ${result.row_count.toLocaleString()} rows)`, 'success');
    } catch (err) {
        uploadArea.innerHTML = `
    < div class="upload-icon" >❌</div >
      <div class="upload-title">Upload Failed</div>
      <div class="upload-subtitle" style="color: var(--vox-error);">${escapeHtml(err.message)}</div>
      <div style="margin-top: 1rem;">
        <span class="vox-btn vox-btn-ghost" style="font-size: 0.85rem;" onclick="event.stopPropagation(); window.NexusUI.resetUpload()">
          🔄 Try Again
        </span>
      </div>
`;
        log(`❌ Upload failed: ${err.message} `, 'error');
    }
}

// ============================================================================
// Analysis Execution
// ============================================================================

// Track if we've shown the mapper for this session
let mapperConfirmed = false;

async function runFullAnalysis() {
    const uploadState = getUploadState();
    if (!uploadState.filename) return;

    // Check if this is a re-run (session exists with completed engines)
    const existingSession = getSession();
    const isRerun = existingSession && existingSession.completedEngines &&
        existingSession.completedEngines.length > 0;

    // Show column mapper on re-run if not already confirmed
    const mapperContainer = document.getElementById('column-mapper-root');
    if (isRerun && mapperContainer && !mapperConfirmed) {
        // Show the mapper for reconfiguration
        mapperContainer.style.display = 'block';

        // Change button text to indicate confirmation needed
        const analyzeBtn = document.getElementById('analyze-btn');
        analyzeBtn.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" style="margin-right: 8px;"><path d="M9 16.2L4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4L9 16.2z"/></svg>Confirm & Start Analysis';

        // Mark that user needs to confirm
        mapperConfirmed = true;

        // Scroll to mapper
        mapperContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });

        return; // Wait for user to click again to confirm
    }

    // Reset for next re-run
    mapperConfirmed = false;

    const useVectorization = document.getElementById('use-vectors')?.checked || false;

    // GPU Health Check - informational only, does not block analysis
    try {
        const [gpuHealth, coordinatorStatus] = await Promise.all([
            checkGpuHealth(),
            checkGpuCoordinatorStatus()
        ]);

        // Log GPU status (informational only)
        if (gpuHealth.available) {
            log(`GPU available(${gpuHealth.vramFreeGb.toFixed(1)}GB estimated free)`, 'info');
        } else if (gpuHealth.warning) {
            log(`GPU info: ${gpuHealth.warning} `, 'info');
        }

        if (!gpuHealth.available || gpuHealth.vramFreeGb < 4 || gpuHealth.warning) {
            const reason = gpuHealth.warning || 'GPU not available';
            enableLowPowerMode(reason);
        }

        if (!coordinatorStatus.available && coordinatorStatus.owner !== 'unknown') {
            log(`GPU owned by ${coordinatorStatus.owner} - ML engines will use CPU`, 'info');
        }

        if (performanceState.lowPower) {
            log(`Low graphics mode enabled(${performanceState.reason || 'resource constraints'})`, 'info');
        }
    } catch (err) {
        // GPU check failures are non-blocking - just log and continue
        console.log('GPU check skipped:', err.message);
    }

    // Show results section
    document.getElementById('all-engines-section').style.display = 'block';
    document.getElementById('engines-database-name').textContent = uploadState.filename;

    // Clear previous results
    ['all', 'ml', 'financial', 'advanced'].forEach(cat => {
        const container = document.getElementById(`${cat} -engines - results`);
        if (container) container.innerHTML = '';
    });

    // Reset dashboard
    resetDashboard();
    initDashboard();

    // Disable analyze button and show running controls
    document.getElementById('analyze-btn').disabled = true;
    window.NexusUI.showRunningControls();

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

    log(`Starting comprehensive analysis with all ${ENGINE_COUNT} engines...`, 'info');
    log(`Database: ${uploadState.filename} `, 'info');

    // Read Skip Gemma toggle
    const skipGemma = document.getElementById('skip-gemma-toggle')?.checked || false;
    if (skipGemma) {
        log(`[Skip Gemma] Gemma summaries disabled - using fallback summaries`, 'info');
    }

    // Start analysis
    await startAnalysis({ useVectorization, skipGemma });
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
    const card = createEngineCard(engine, index, 'all');
    const categoryCard = createEngineCard(engine, index, engine.category);

    document.getElementById('all-engines-results').appendChild(card);
    document.getElementById(`${engine.category}-engines-results`).appendChild(categoryCard);

    // AUTO-EXPAND: Expand cards immediately when engine starts running
    // This ensures users see the loading state without needing to click
    card.classList.add('expanded');
    categoryCard.classList.add('expanded');

    updateEngineCardStatus(card, 'running');
    updateEngineCardStatus(categoryCard, 'running');
}

function handleEngineComplete(engine, result, duration) {
    // Find and update cards
    const cards = document.querySelectorAll(`.engine-result-card[data-engine="${engine.name}"]`);
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
    const cards = document.querySelectorAll(`.engine-result-card[data-engine="${engine.name}"]`);
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

    // Reset to idle controls (show Start button again)
    window.NexusUI.showIdleControls();

    // Update progress bar to complete
    const progressBar = document.getElementById('progress-bar');
    const progressStats = document.getElementById('progress-stats');
    const progressEngine = document.getElementById('progress-engine');

    if (progressBar) {
        progressBar.style.width = '100%';
        progressBar.classList.remove('animated');
    }
    if (progressStats) progressStats.textContent = `${ENGINE_COUNT}/${ENGINE_COUNT} engines complete`;
    if (progressEngine) progressEngine.textContent = 'Analysis Complete!';

    log(`All ${ENGINE_COUNT} engines complete!`, 'success', stats.totalTime);
    log(`Results: ${stats.success} succeeded, ${stats.error} failed`, 'info');
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
                const recentSection = document.getElementById('recent-section');
                const grid = document.getElementById('recent-grid');

                if (!recentSection || !grid) {
                    console.log('Recent section elements not found');
                    return;
                }

                recentSection.style.display = 'block';
                grid.innerHTML = `
          <div class="vox-card" style="padding: 1rem; cursor: pointer; transition: all 0.2s ease;" 
               onmouseover="this.style.borderColor='var(--vox-primary)'; this.style.transform='translateY(-2px)'"
               onmouseout="this.style.borderColor=''; this.style.transform=''"
               onclick="window.NexusUI.resumeAnalysis()">
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
              <span style="font-size: 1.5rem;">📄</span>
              <span style="font-weight: 600; color: var(--vox-grey-800);">${escapeHtml(session.filename)}</span>
            </div>
            <div style="display: flex; gap: 1rem; font-size: 0.85rem; color: var(--vox-grey-500);">
              <span>🔄 ${Object.keys(session.results).length}/${ENGINE_COUNT} engines</span>
              <span>📊 ${session.status === 'paused' ? 'Paused' : session.status}</span>
            </div>
          </div>
        `;
            }
        }
    } catch (e) {
        console.log('No recent analyses found:', e);
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
