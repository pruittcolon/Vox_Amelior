/**
 * Predictions Page - State Management
 * Central state store for upload data, analysis results, and user selections
 * 
 * Multi-Run Analysis System:
 * - Each engine can have up to 20 runs stored
 * - Navigation arrows allow browsing between runs ("1 of 3")
 * - Session ID links all runs for comparison
 */

const PredictionsState = {
  // Session tracking for multi-run persistence
  sessionId: null,
  
  // Upload & File Data
  uploadedFilename: null,
  uploadedColumns: [],
  dataColumns: [],
  
  // Analysis Results (current display)
  analysisResults: null,
  forecastResults: null,
  
  // Multi-Run Storage: { engineName: [{ runId, targetColumn, result, timestamp }] }
  runs: {},
  
  // Current run index per engine: { engineName: 0 }
  currentRunIndex: {},
  
  // Column Selection
  selectionMode: 'gemma', // 'gemma' or 'manual'
  gemmaSelection: null,
  manualTarget: null,
  manualFeatures: [],
  excludedTargets: [],
  
  // Forecast State
  detectedTimeColumn: null,
  detectedTargetColumn: null,
  hasTimeColumn: false,
  
  // Gemma Explanation State
  gemmaExplanationPending: false,
  lastGemmaExplanation: null,
  
  // Chart instances
  charts: {},
  
  // Maximum runs per engine
  MAX_RUNS_PER_ENGINE: 20,
  
  // Generate unique session ID
  generateSessionId() {
    return `session_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  },
  
  // Initialize or get session ID
  initSession() {
    if (!this.sessionId) {
      this.sessionId = this.generateSessionId();
    }
    return this.sessionId;
  },
  
  // Initialize state
  reset() {
    this.sessionId = null;
    this.uploadedFilename = null;
    this.uploadedColumns = [];
    this.analysisResults = null;
    this.gemmaSelection = null;
    this.manualTarget = null;
    this.manualFeatures = [];
    this.excludedTargets = [];
    this.forecastResults = null;
    this.hasTimeColumn = false;
    this.gemmaExplanationPending = false;
    this.lastGemmaExplanation = null;
    
    // Reset multi-run state
    this.runs = {};
    this.currentRunIndex = {};
    
    // Destroy existing charts
    Object.values(this.charts).forEach(chart => {
      if (chart && typeof chart.destroy === 'function') {
        chart.destroy();
      }
    });
    this.charts = {};
  },
  
  // Set uploaded data
  setUploadData(filename, columns, rowCount) {
    this.uploadedFilename = filename;
    this.uploadedColumns = columns;
    this.dataColumns = columns;
    this.manualTarget = null;
    this.manualFeatures = [];
    this.excludedTargets = [];
    this.forecastResults = null;
    this.hasTimeColumn = false;
    return this;
  },
  
  // Get current state as JSON
  toJSON() {
    return {
      uploadedFilename: this.uploadedFilename,
      uploadedColumns: this.uploadedColumns,
      analysisResults: this.analysisResults ? 'exists' : null,
      selectionMode: this.selectionMode,
      gemmaSelection: this.gemmaSelection,
      manualTarget: this.manualTarget,
      manualFeatures: this.manualFeatures,
      excludedTargets: this.excludedTargets,
      hasTimeColumn: this.hasTimeColumn,
      sessionId: this.sessionId,
      runCounts: Object.fromEntries(
        Object.entries(this.runs).map(([k, v]) => [k, v.length])
      )
    };
  },
  
  // ===== MULTI-RUN MANAGEMENT =====
  
  /**
   * Add a new run for an engine
   * @param {string} engineName - The engine identifier
   * @param {object} runData - { targetColumn, result, runId? }
   * @returns {object} The added run with metadata
   */
  addRun(engineName, runData) {
    if (!this.runs[engineName]) {
      this.runs[engineName] = [];
    }
    
    const run = {
      runId: runData.runId || `run_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`,
      targetColumn: runData.targetColumn,
      result: runData.result,
      timestamp: new Date().toISOString(),
      sessionId: this.sessionId
    };
    
    // Add to beginning (newest first)
    this.runs[engineName].unshift(run);
    
    // Enforce 20 run limit
    if (this.runs[engineName].length > this.MAX_RUNS_PER_ENGINE) {
      this.runs[engineName] = this.runs[engineName].slice(0, this.MAX_RUNS_PER_ENGINE);
    }
    
    // Set current index to newest (0)
    this.currentRunIndex[engineName] = 0;
    
    return run;
  },
  
  /**
   * Get all runs for an engine
   * @param {string} engineName - The engine identifier
   * @returns {array} Array of runs (newest first)
   */
  getRuns(engineName) {
    return this.runs[engineName] || [];
  },
  
  /**
   * Get the current run for an engine
   * @param {string} engineName - The engine identifier
   * @returns {object|null} The current run or null
   */
  getCurrentRun(engineName) {
    const runs = this.getRuns(engineName);
    const index = this.currentRunIndex[engineName] || 0;
    return runs[index] || null;
  },
  
  /**
   * Get current run index for an engine
   * @param {string} engineName - The engine identifier
   * @returns {number} Current index (0-based)
   */
  getCurrentRunIndex(engineName) {
    return this.currentRunIndex[engineName] || 0;
  },
  
  /**
   * Get run count for an engine
   * @param {string} engineName - The engine identifier
   * @returns {number} Total number of runs
   */
  getRunCount(engineName) {
    return this.getRuns(engineName).length;
  },
  
  /**
   * Navigate to a specific run
   * @param {string} engineName - The engine identifier
   * @param {number} index - The run index to navigate to
   * @returns {object|null} The run at that index
   */
  setCurrentRunIndex(engineName, index) {
    const runs = this.getRuns(engineName);
    if (index >= 0 && index < runs.length) {
      this.currentRunIndex[engineName] = index;
      return runs[index];
    }
    return null;
  },
  
  /**
   * Navigate to previous run (older)
   * @param {string} engineName - The engine identifier
   * @returns {object|null} The previous run or null if at end
   */
  previousRun(engineName) {
    const currentIndex = this.getCurrentRunIndex(engineName);
    const runs = this.getRuns(engineName);
    if (currentIndex < runs.length - 1) {
      return this.setCurrentRunIndex(engineName, currentIndex + 1);
    }
    return null;
  },
  
  /**
   * Navigate to next run (newer)
   * @param {string} engineName - The engine identifier
   * @returns {object|null} The next run or null if at beginning
   */
  nextRun(engineName) {
    const currentIndex = this.getCurrentRunIndex(engineName);
    if (currentIndex > 0) {
      return this.setCurrentRunIndex(engineName, currentIndex - 1);
    }
    return null;
  },
  
  /**
   * Check if can navigate to previous (older) run
   * @param {string} engineName - The engine identifier
   * @returns {boolean}
   */
  hasPreviousRun(engineName) {
    const currentIndex = this.getCurrentRunIndex(engineName);
    const runCount = this.getRunCount(engineName);
    return currentIndex < runCount - 1;
  },
  
  /**
   * Check if can navigate to next (newer) run
   * @param {string} engineName - The engine identifier
   * @returns {boolean}
   */
  hasNextRun(engineName) {
    return this.getCurrentRunIndex(engineName) > 0;
  },
  
  /**
   * Get all used target columns for this session
   * @returns {array} Array of target column names already analyzed
   */
  getUsedTargets() {
    const used = new Set();
    Object.values(this.runs).forEach(engineRuns => {
      engineRuns.forEach(run => {
        if (run.targetColumn) {
          used.add(run.targetColumn);
        }
      });
    });
    return Array.from(used);
  },
  
  /**
   * Get navigation display string
   * @param {string} engineName - The engine identifier
   * @returns {string} e.g. "1 of 3"
   */
  getNavigationDisplay(engineName) {
    const runCount = this.getRunCount(engineName);
    if (runCount === 0) return '';
    const currentIndex = this.getCurrentRunIndex(engineName);
    // Display 1-indexed (user sees "1 of 3" for index 0 of 3 runs)
    return `${currentIndex + 1} of ${runCount}`;
  },
  
  /**
   * Sync runs from backend history API
   * @param {string} engineName - The engine identifier
   * @param {array} backendRuns - Runs from the history API
   */
  syncRunsFromBackend(engineName, backendRuns) {
    this.runs[engineName] = backendRuns.map(run => ({
      runId: run.run_id,
      targetColumn: run.target_column,
      result: run.result,
      timestamp: run.created_at,
      sessionId: run.session_id
    }));
    this.currentRunIndex[engineName] = 0;
  },
  
  /**
   * Load session from backend
   * @param {string} sessionId - Session ID to restore
   */
  async restoreSession(sessionId) {
    try {
      const response = await fetch(`${API_BASE}/analytics/history/${sessionId}`);
      if (!response.ok) throw new Error('Session not found');
      
      const data = await response.json();
      this.sessionId = sessionId;
      this.uploadedFilename = data.session?.filename || this.uploadedFilename;
      
      // Group runs by engine
      const runsByEngine = {};
      data.runs.forEach(run => {
        if (!runsByEngine[run.engine_name]) {
          runsByEngine[run.engine_name] = [];
        }
        runsByEngine[run.engine_name].push(run);
      });
      
      // Sync each engine's runs
      Object.entries(runsByEngine).forEach(([engineName, runs]) => {
        this.syncRunsFromBackend(engineName, runs);
      });
      
      return true;
    } catch (err) {
      console.error('Failed to restore session:', err);
      return false;
    }
  }
};

/**
 * DOM element cache for faster access
 */
const DOM = {
  uploadArea: null,
  fileInput: null,
  analyzeBtn: null,
  testAllBtn: null,
  statusBar: null,
  progressFill: null,
  resultsSection: null,
  log: null,
  
  init() {
    this.uploadArea = document.getElementById('upload-area');
    this.fileInput = document.getElementById('file-input');
    this.analyzeBtn = document.getElementById('analyze-btn');
    this.testAllBtn = document.getElementById('test-all-btn');
    this.statusBar = document.getElementById('status-bar');
    this.progressFill = document.getElementById('progress-fill');
    this.resultsSection = document.getElementById('results-section');
    this.log = document.getElementById('log');
    return this;
  }
};

/**
 * Logging utility
 */
const Logger = {
  log(msg, type = 'info') {
    const ts = new Date().toLocaleTimeString();
    const icons = { info: 'ℹ️', success: '✅', error: '❌', warning: '⚠️' };
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerHTML = `<span class="log-time">[${ts}]</span> ${icons[type] || ''} <span class="log-message">${msg}</span>`;
    DOM.log.appendChild(entry);
    DOM.log.scrollTop = DOM.log.scrollHeight;
  },
  
  setStatus(text, type, percent) {
    DOM.statusBar.className = 'status-bar ' + (type || '');
    if (type === 'loading') {
      DOM.statusBar.innerHTML = `<div class="spinner"></div><span>${text}</span>`;
    } else {
      const icons = { success: '✅', error: '❌' };
      DOM.statusBar.innerHTML = `<span>${icons[type] || ''} ${text}</span>`;
    }
    if (percent !== undefined) {
      DOM.progressFill.style.width = percent + '%';
    }
  },
  
  errorToString(err) {
    if (!err) return 'Unknown error';
    if (typeof err === 'string') return err;
    if (err.message) return err.message;
    if (err.detail) return this.errorToString(err.detail);
    if (err.error) return this.errorToString(err.error);
    try { return JSON.stringify(err); } catch (e) { return String(err); }
  }
};

/**
 * API Base URL
 */
const API_BASE = window.location.origin;

export { PredictionsState, DOM, Logger, API_BASE };
