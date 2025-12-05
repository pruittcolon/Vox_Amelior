/**
 * Predictions Page - Main Entry Point
 * Orchestrates all modules and initializes the application
 */

import { PredictionsState, DOM, Logger, API_BASE } from './predictions-state.js';
import Upload from './predictions-upload.js';
import ColumnSelection from './predictions-columns.js';
import RunNavigation from './predictions-runs.js';

// Tab switching
function switchTab(tabName) {
  document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  
  const tabContent = document.getElementById('tab-' + tabName);
  if (tabContent) tabContent.classList.add('active');
  
  document.querySelectorAll('.tab-btn').forEach(btn => {
    if (btn.textContent.toLowerCase().includes(tabName.toLowerCase().slice(0, 4))) {
      btn.classList.add('active');
    }
  });
  
  // Re-render charts if needed
  if ((tabName === 'features' || tabName === 'models' || tabName === 'predictions') && PredictionsState.analysisResults) {
    setTimeout(renderCharts, 100);
  }
  
  if (tabName === 'forecasting' && PredictionsState.forecastResults) {
    setTimeout(renderForecastChart, 100);
  }
}

// Stub functions for now - will be expanded in feature modules
async function runAnalysis() {
  Logger.log('Analysis execution not yet implemented in modular version', 'warning');
}

async function testAllEngines() {
  Logger.log('All engines test not yet implemented in modular version', 'warning');
}

function renderCharts() {
  Logger.log('Chart rendering queued', 'info');
}

function renderForecastChart() {
  Logger.log('Forecast chart rendering queued', 'info');
}

// Initialize application
async function init() {
  try {
    // Initialize DOM cache
    DOM.init();
    
    // Initialize session for multi-run tracking
    PredictionsState.initSession();
    
    // Initialize modules
    Upload.init();
    ColumnSelection.init();
    RunNavigation.init();
    
    // Attach global functions for HTML onclick handlers
    window.switchTab = switchTab;
    window.runAnalysis = runAnalysis;
    window.testAllEngines = testAllEngines;
    window.testAgain = RunNavigation.testAgain.bind(RunNavigation);
    window.setSelectionMode = ColumnSelection.setSelectionMode.bind(ColumnSelection);
    window.onTargetChange = ColumnSelection.onTargetChange.bind(ColumnSelection);
    window.selectAllFeatures = Upload.populateColumnSelectors.bind(Upload); // TODO: Fix this
    window.deselectAllFeatures = ColumnSelection.deselectAllFeatures.bind(ColumnSelection);
    
    // Attach event listeners for main buttons
    if (DOM.analyzeBtn) {
      DOM.analyzeBtn.addEventListener('click', runAnalysis);
    }
    
    if (DOM.testAllBtn) {
      DOM.testAllBtn.addEventListener('click', testAllEngines);
    }
    
    // Check URL for session restore
    const urlParams = new URLSearchParams(window.location.search);
    const sessionToRestore = urlParams.get('session');
    if (sessionToRestore) {
      Logger.log(`📂 Restoring session: ${sessionToRestore}`, 'info');
      await PredictionsState.restoreSession(sessionToRestore);
    }
    
    Logger.log('🚀 Predictions page initialized', 'success');
    Logger.log(`📊 Session ID: ${PredictionsState.sessionId}`, 'info');
    Logger.setStatus('👋 Upload a dataset to begin', 'info', 0);
    
  } catch (err) {
    Logger.log(`Initialization error: ${err.message}`, 'error');
    Logger.setStatus('Failed to initialize', 'error', 0);
  }
}

// Wait for DOM to be ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

// Export for testing
export { switchTab, runAnalysis, testAllEngines, PredictionsState };
