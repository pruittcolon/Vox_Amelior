/**
 * Predictions Page - Multi-Run Navigation
 * Handles run navigation arrows, "Test Again", and per-engine reruns
 */

import { PredictionsState, Logger, API_BASE } from './predictions-state.js';

const RunNavigation = {
  
  /**
   * Create navigation HTML for an engine card header
   * @param {string} engineName - The engine identifier
   * @returns {string} HTML for navigation arrows
   */
  createNavigationHTML(engineName) {
    const runCount = PredictionsState.getRunCount(engineName);
    if (runCount === 0) return '';
    
    const display = PredictionsState.getNavigationDisplay(engineName);
    const hasPrev = PredictionsState.hasPreviousRun(engineName);
    const hasNext = PredictionsState.hasNextRun(engineName);
    
    return `
      <div class="run-navigation" data-engine="${engineName}">
        <button class="nav-arrow prev ${hasPrev ? '' : 'disabled'}" 
                onclick="event.stopPropagation(); RunNavigation.navigatePrevious('${engineName}')"
                title="Previous run (older)"
                ${hasPrev ? '' : 'disabled'}>
          ◀
        </button>
        <span class="run-counter">${display}</span>
        <button class="nav-arrow next ${hasNext ? '' : 'disabled'}"
                onclick="event.stopPropagation(); RunNavigation.navigateNext('${engineName}')"
                title="Next run (newer)"
                ${hasNext ? '' : 'disabled'}>
          ▶
        </button>
        <button class="rerun-btn"
                onclick="event.stopPropagation(); RunNavigation.rerunEngine('${engineName}')"
                title="Run this engine again with a different target">
          🔄 Rerun
        </button>
      </div>
    `;
  },
  
  /**
   * Update navigation display for an engine
   * @param {string} engineName - The engine identifier
   */
  updateNavigation(engineName) {
    const navContainer = document.querySelector(`.run-navigation[data-engine="${engineName}"]`);
    if (!navContainer) return;
    
    const runCount = PredictionsState.getRunCount(engineName);
    if (runCount === 0) {
      navContainer.style.display = 'none';
      return;
    }
    
    navContainer.style.display = 'flex';
    
    const display = PredictionsState.getNavigationDisplay(engineName);
    const hasPrev = PredictionsState.hasPreviousRun(engineName);
    const hasNext = PredictionsState.hasNextRun(engineName);
    
    const prevBtn = navContainer.querySelector('.nav-arrow.prev');
    const nextBtn = navContainer.querySelector('.nav-arrow.next');
    const counter = navContainer.querySelector('.run-counter');
    
    if (counter) counter.textContent = display;
    
    if (prevBtn) {
      prevBtn.disabled = !hasPrev;
      prevBtn.classList.toggle('disabled', !hasPrev);
    }
    
    if (nextBtn) {
      nextBtn.disabled = !hasNext;
      nextBtn.classList.toggle('disabled', !hasNext);
    }
  },
  
  /**
   * Navigate to previous (older) run
   * @param {string} engineName - The engine identifier
   */
  navigatePrevious(engineName) {
    const run = PredictionsState.previousRun(engineName);
    if (run) {
      this.displayRun(engineName, run);
      this.updateNavigation(engineName);
      Logger.log(`📊 Viewing older run for ${engineName}: target "${run.targetColumn}"`, 'info');
    }
  },
  
  /**
   * Navigate to next (newer) run
   * @param {string} engineName - The engine identifier
   */
  navigateNext(engineName) {
    const run = PredictionsState.nextRun(engineName);
    if (run) {
      this.displayRun(engineName, run);
      this.updateNavigation(engineName);
      Logger.log(`📊 Viewing newer run for ${engineName}: target "${run.targetColumn}"`, 'info');
    }
  },
  
  /**
   * Display a specific run's results in the engine card
   * @param {string} engineName - The engine identifier
   * @param {object} run - The run data to display
   */
  displayRun(engineName, run) {
    const bodyEl = document.getElementById(`body-${engineName}`);
    if (!bodyEl || !run || !run.result) return;
    
    // Update the card body with the run's result
    // This will call the appropriate render function based on engine type
    if (typeof window.renderEngineResult === 'function') {
      window.renderEngineResult(engineName, run.result, run.targetColumn);
    } else {
      // Fallback: just show the raw result
      bodyEl.innerHTML = `
        <div class="run-result">
          <div class="run-meta">
            <span class="target-badge">Target: ${run.targetColumn}</span>
            <span class="timestamp">${new Date(run.timestamp).toLocaleString()}</span>
          </div>
          <pre class="result-json">${JSON.stringify(run.result, null, 2)}</pre>
        </div>
      `;
    }
    
    // Expand the card to show results
    const card = document.getElementById(`engine-card-${engineName}`);
    if (card) card.classList.add('expanded');
  },
  
  /**
   * Rerun a specific engine with a new target column
   * Uses Gemma to select from remaining columns
   * @param {string} engineName - The engine identifier
   */
  async rerunEngine(engineName) {
    try {
      const usedTargets = PredictionsState.getUsedTargets();
      const allColumns = PredictionsState.dataColumns || PredictionsState.uploadedColumns;
      const remainingColumns = allColumns.filter(col => !usedTargets.includes(col));
      
      if (remainingColumns.length === 0) {
        Logger.log(`⚠️ All columns have been analyzed for ${engineName}. No columns remaining.`, 'warning');
        alert('All columns have already been analyzed. Upload a new dataset or clear history.');
        return;
      }
      
      Logger.log(`🔄 Rerunning ${engineName}... Asking Gemma to select from ${remainingColumns.length} remaining columns`, 'info');
      
      // Show loading state
      const bodyEl = document.getElementById(`body-${engineName}`);
      if (bodyEl) {
        bodyEl.innerHTML = `
          <div class="loading-spinner">
            🤖 Gemma selecting new target column...
            <div class="remaining-columns">Available: ${remainingColumns.slice(0, 5).join(', ')}${remainingColumns.length > 5 ? '...' : ''}</div>
          </div>
        `;
      }
      
      // Call rerun API endpoint
      const response = await fetch(`${API_BASE}/analytics/rerun`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId: PredictionsState.sessionId,
          engineName: engineName,
          filename: PredictionsState.uploadedFilename,
          allColumns: allColumns,
          usedTargets: usedTargets
        })
      });
      
      if (!response.ok) {
        throw new Error(`Rerun failed: ${response.statusText}`);
      }
      
      const result = await response.json();
      
      // Add the new run to state
      const run = PredictionsState.addRun(engineName, {
        runId: result.runId,
        targetColumn: result.targetColumn,
        result: result.result
      });
      
      // Display the new run
      this.displayRun(engineName, run);
      this.updateNavigation(engineName);
      
      Logger.log(`✅ ${engineName} rerun complete with target "${result.targetColumn}"`, 'success');
      
    } catch (err) {
      Logger.log(`❌ Rerun failed for ${engineName}: ${err.message}`, 'error');
      const bodyEl = document.getElementById(`body-${engineName}`);
      if (bodyEl) {
        bodyEl.innerHTML = `<div class="error-message">❌ Rerun failed: ${err.message}</div>`;
      }
    }
  },
  
  /**
   * Test Again - Run all engines with new target columns
   * Gemma selects from remaining columns for each engine
   */
  async testAgain() {
    try {
      const usedTargets = PredictionsState.getUsedTargets();
      const allColumns = PredictionsState.dataColumns || PredictionsState.uploadedColumns;
      const remainingColumns = allColumns.filter(col => !usedTargets.includes(col));
      
      if (remainingColumns.length === 0) {
        Logger.log('⚠️ All columns have been analyzed. Upload a new dataset.', 'warning');
        alert('All columns have been analyzed. Upload a new dataset to test again.');
        return;
      }
      
      Logger.log(`🔄 Test Again: ${remainingColumns.length} columns remaining to analyze`, 'info');
      
      // Call test-again API endpoint
      const response = await fetch(`${API_BASE}/analytics/test-again`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId: PredictionsState.sessionId,
          filename: PredictionsState.uploadedFilename,
          allColumns: allColumns,
          usedTargets: usedTargets
        })
      });
      
      if (!response.ok) {
        throw new Error(`Test Again failed: ${response.statusText}`);
      }
      
      const result = await response.json();
      
      // The result contains the new target column selected by Gemma
      Logger.log(`🤖 Gemma selected new target: "${result.targetColumn}"`, 'info');
      
      // Trigger the full analysis with the new target
      if (typeof window.runAnalysisWithTarget === 'function') {
        await window.runAnalysisWithTarget(result.targetColumn);
      } else {
        // Fallback: update state and trigger standard analysis
        PredictionsState.gemmaSelection = {
          target: result.targetColumn,
          features: result.features || allColumns.filter(c => c !== result.targetColumn)
        };
        if (typeof window.testAllEngines === 'function') {
          await window.testAllEngines();
        }
      }
      
    } catch (err) {
      Logger.log(`❌ Test Again failed: ${err.message}`, 'error');
    }
  },
  
  /**
   * Inject navigation into all engine cards
   */
  injectNavigationToCards() {
    document.querySelectorAll('.engine-result-card').forEach(card => {
      const engineName = card.id?.replace('engine-card-', '');
      if (!engineName) return;
      
      const header = card.querySelector('.engine-card-header');
      if (!header) return;
      
      // Check if navigation already exists
      if (header.querySelector('.run-navigation')) return;
      
      // Add navigation HTML
      const navHTML = this.createNavigationHTML(engineName);
      if (navHTML) {
        header.insertAdjacentHTML('beforeend', navHTML);
      }
    });
  },
  
  /**
   * Initialize run navigation
   */
  init() {
    // Make RunNavigation globally available for onclick handlers
    window.RunNavigation = this;
    
    // Observe for new engine cards being added
    const observer = new MutationObserver(() => {
      this.injectNavigationToCards();
    });
    
    const resultsSection = document.getElementById('results-section');
    if (resultsSection) {
      observer.observe(resultsSection, { childList: true, subtree: true });
    }
    
    Logger.log('📊 Run navigation system initialized', 'info');
  }
};

export default RunNavigation;
