/**
 * NexusAI Engine Results
 * Result card creation, display formatting, and key finding extraction.
 *
 * @module nexus/engines/engine-results
 */

import { VIZ_COLORS } from '../core/config.js';

// ============================================================================
// Card Creation
// ============================================================================

/**
 * Create an engine result card element.
 * @param {Object} engine - Engine definition
 * @param {number} index - Engine index (0-based)
 * @returns {HTMLElement}
 */
export function createEngineCard(engine, index, instanceId = 'all') {
  const card = document.createElement('div');
  card.className = 'engine-result-card';
  const cardId = `${instanceId}-${engine.name}-${index}`;
  card.id = `engine-card-${cardId}`;
  card.dataset.engine = engine.name;
  card.dataset.cardId = cardId;
  card.dataset.category = engine.category;

  card.innerHTML = `
    <div class="engine-card-header" onclick="window.NexusUI.toggleEngineCard('${engine.name}')">
      <span class="engine-icon">${engine.icon}</span>
      <h3>${index + 1}. ${engine.display}</h3>
      <div class="engine-header-controls">
        <button class="engine-retest-btn" onclick="event.stopPropagation(); window.NexusUI.retestEngine('${engine.name}')" title="Re-run this analysis">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M1 4v6h6"/>
            <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/>
          </svg>
        </button>
        <span class="engine-status pending" id="status-${cardId}">Pending</span>
      </div>
    </div>
    <div class="engine-card-body" id="body-${cardId}">
      <div class="loading-spinner">Analyzing your data with AI engine...</div>
    </div>
  `;

  return card;
}


/**
 * Update card status indicator.
 * @param {HTMLElement|string} card - Card element or engine name
 * @param {'pending'|'running'|'success'|'error'|'fallback'} status
 */
export function updateEngineCardStatus(card, status) {
  // Support both element and name
  if (typeof card === 'string') {
    document.querySelectorAll(`.engine-result-card[data-engine="${card}"]`).forEach(cardEl => {
      applyStatus(cardEl, status);
    });
    return;
  }

  const cardEl = card;
  applyStatus(cardEl, status);
}

function applyStatus(cardEl, status) {
  if (!cardEl) return;

  const statusEl = cardEl.querySelector('.engine-status');
  if (!statusEl) return;

  // Remove all status classes
  statusEl.classList.remove('pending', 'running', 'success', 'error', 'fallback');

  // Add new status
  statusEl.classList.add(status);

  // Update text
  const statusText = {
    pending: '⏳ Pending',
    running: '⏳ Running...',
    success: '✅ Complete',
    error: '❌ Failed',
    fallback: '⚠️ Partial'
  };

  statusEl.textContent = statusText[status] || status;
}

/**
 * Display results in an engine card.
 * @param {HTMLElement|string} card - Card element or engine name
 * @param {Object} result - Engine result object
 */
export function displayEngineResults(card, result) {
  const cardEl = typeof card === 'string'
    ? document.querySelector(`.engine-result-card[data-engine="${card}"]`)
    : card;

  if (!cardEl) return;

  const bodyEl = cardEl.querySelector('.engine-card-body');
  if (!bodyEl) return;

  if (result.status === 'error') {
    bodyEl.innerHTML = `
      <div class="engine-error">
        <span class="error-icon">❌</span>
        <span class="error-message">${escapeHtml(result.error)}</span>
      </div>
    `;
    return;
  }

  // Build results HTML
  const durationStr = formatDuration(result.duration);
  const dataSizeStr = formatBytes(result.dataSize);
  const engineName = cardEl.dataset.engine;
  const cardId = cardEl.dataset.cardId || `${engineName}-${Date.now()}`;
  const vizId = `viz-${cardId}`;

  let html = `
    <div class="engine-summary">
      <div class="gemma-summary">
        <span class="gemma-icon">🤖</span>
        <div class="gemma-content">
          <div class="gemma-text">${renderMarkdown(result.gemmaSummary || 'Analysis complete. Review the details below for insights.')}</div>
          <div class="followup-messages" id="messages-${cardId}"></div>
          <div class="gemma-chat-inline">
            <input type="text" placeholder="Ask a follow-up question..." id="input-${cardId}" 
                   onkeydown="if(event.key==='Enter') window.NexusUI.sendFollowUp('${engineName}', '${cardId}')">
            <button onclick="window.NexusUI.sendFollowUp('${engineName}', '${cardId}')">Send</button>
          </div>
        </div>
      </div>
      <div class="engine-meta">
        <span class="meta-item">⏱️ ${durationStr}</span>
        <span class="meta-item">📦 ${dataSizeStr}</span>
      </div>
    </div>
  `;

  const vizSection = window.NexusViz?.buildVizSection?.(engineName, result.data, vizId);
  if (vizSection) html += vizSection;

  // Add key findings if available
  const findings = extractKeyFindings(result.data);
  if (findings.length > 0) {
    html += `
      <div class="key-findings">
        <h4>Key Findings</h4>
        <ul>
          ${findings.map(f => `<li>${escapeHtml(f)}</li>`).join('')}
        </ul>
      </div>
    `;
  }

  // Add expandable raw data section
  const dataId = `raw-data-${cardEl.id}`;
  html += `
    <div class="raw-data-section">
      <button class="toggle-raw-btn" onclick="window.NexusUI.toggleRawData('${dataId}')">
        <span id="icon-${dataId}">▶</span> View Raw Data
      </button>
      <pre class="raw-data" id="${dataId}" style="display: none;">${escapeHtml(JSON.stringify(result.data, null, 2))}</pre>
    </div>
  `;

  // Chat input is now inline inside gemma-summary, so no separate followup section needed

  bodyEl.innerHTML = html;

  // AUTO-EXPAND: Expand card when results are displayed
  cardEl.classList.add('expanded');

  // Render visualizations after DOM update
  setTimeout(() => {
    window.NexusViz?.renderEngineVisualizations?.(engineName, result.data, vizId);
    // Smooth scroll into view (nearest to avoid jarring jumps)
    cardEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, 150);
}

/**
 * Create chat input HTML for follow-up questions.
 * @param {string} engineName
 * @returns {string}
 */
function createChatInput(engineName) {
  return `
    <div class="followup-section">
      <div class="followup-messages" id="messages-${engineName}"></div>
      <div class="followup-input">
        <input type="text"
               placeholder="Ask a follow-up question..."
               id="input-${engineName}"
               onkeydown="if(event.key==='Enter') window.NexusUI.sendFollowUp('${engineName}')">
        <button onclick="window.NexusUI.sendFollowUp('${engineName}')">Send</button>
      </div>
    </div>
  `;
}

// ============================================================================
// Key Finding Extraction
// ============================================================================

/**
 * Extract key findings from engine result data.
 * @param {Object} data - Raw engine result data
 * @returns {string[]} Array of finding strings
 */
export function extractKeyFindings(data) {
  const findings = [];

  if (!data) return findings;

  // Best model
  if (data.best_model) {
    findings.push(`Best model: ${data.best_model}`);
  }

  // Accuracy
  if (data.accuracy !== undefined) {
    findings.push(`Accuracy: ${(data.accuracy * 100).toFixed(1)}%`);
  }
  if (data.cv_score !== undefined) {
    findings.push(`CV Score: ${(data.cv_score * 100).toFixed(1)}%`);
  }

  // Clusters
  if (data.n_clusters !== undefined) {
    findings.push(`Found ${data.n_clusters} clusters`);
  }

  // Anomalies
  if (data.anomaly_count !== undefined) {
    findings.push(`Detected ${data.anomaly_count} anomalies`);
  }
  if (data.anomalies_count !== undefined) {
    findings.push(`Detected ${data.anomalies_count} anomalies`);
  }

  // Financial metrics
  if (data.total_cost !== undefined) {
    findings.push(`Total cost: $${formatNumber(data.total_cost)}`);
  }
  if (data.potential_savings !== undefined) {
    findings.push(`Potential savings: $${formatNumber(data.potential_savings)}`);
  }
  if (data.roi !== undefined) {
    findings.push(`ROI: ${(data.roi * 100).toFixed(1)}%`);
  }

  // Trend
  if (data.trend_direction) {
    findings.push(`Trend: ${data.trend_direction}`);
  }

  // Feature importance
  if (data.top_features && Array.isArray(data.top_features)) {
    const topThree = data.top_features.slice(0, 3).map(f =>
      typeof f === 'string' ? f : f.feature || f.name
    );
    if (topThree.length > 0) {
      findings.push(`Top features: ${topThree.join(', ')}`);
    }
  }

  return findings.slice(0, 5); // Limit to 5 findings
}

// ============================================================================
// Formatting Utilities
// ============================================================================

/**
 * Format duration in human-readable format.
 * @param {number} ms - Duration in milliseconds
 * @returns {string}
 */
export function formatDuration(ms) {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(2)}s`;
  const mins = Math.floor(ms / 60000);
  const secs = ((ms % 60000) / 1000).toFixed(1);
  return `${mins}m ${secs}s`;
}

/**
 * Format bytes in human-readable format.
 * @param {number} bytes
 * @returns {string}
 */
export function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

/**
 * Format number with commas.
 * @param {number} num
 * @returns {string}
 */
export function formatNumber(num) {
  return num.toLocaleString();
}

/**
 * Escape HTML special characters.
 * @param {string} text
 * @returns {string}
 */
export function escapeHtml(text) {
  if (!text) return '';
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/**
 * Render markdown-like formatting to HTML.
 * @param {string} text
 * @returns {string}
 */
export function renderMarkdown(text) {
  if (!text) return '';

  let html = escapeHtml(text);

  // Headers - light colors for dark background
  html = html.replace(/^### (.+)$/gm, '<h4 style="margin: 0.8em 0 0.4em; color: #67e8f9; font-size: 0.95em;">$1</h4>');
  html = html.replace(/^## (.+)$/gm, '<h3 style="margin: 1em 0 0.5em; color: #06b6d4; font-size: 1.05em;">$1</h3>');

  // Bold - use light cyan for dark background
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong style="color: #67e8f9; font-weight: 600;">$1</strong>');
  html = html.replace(/__([^_]+)__/g, '<strong style="color: #67e8f9; font-weight: 600;">$1</strong>');

  // Italic
  html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  html = html.replace(/_([^_]+)_/g, '<em>$1</em>');

  // Bullet points
  html = html.replace(/^\s*[\*\-]\s+(.+)$/gm, '<li style="margin: 0.3em 0; padding-left: 0.5em;">$1</li>');

  // Wrap consecutive <li> in <ul>
  html = html.replace(/(<li[^>]*>.*?<\/li>\s*)+/g, (match) => {
    return `<ul style="margin: 0.5em 0; padding-left: 1.2em; list-style-type: disc;">${match}</ul>`;
  });

  // Inline code - light styling for dark background
  html = html.replace(/`([^`]+)`/g, '<code style="background: rgba(6, 182, 212, 0.2); padding: 0.15em 0.4em; border-radius: 4px; font-size: 0.9em; color: #67e8f9;">$1</code>');

  // Line breaks
  html = html.replace(/\n\n+/g, '</p><p style="margin: 0.8em 0;">');
  html = html.replace(/\n/g, '<br>');

  // CRITICAL: Use LIGHT text color for dark Gemma bubble background
  return `<div style="line-height: 1.7; color: #e2e8f0;">${html}</div>`;
}

// ============================================================================
// Category Stats
// ============================================================================

/**
 * Calculate stats per category from session results.
 * @param {Object} results - Engine name to result mapping
 * @param {Object[]} engines - Engine definitions
 * @returns {Object<string, {success: number, error: number, pending: number}>}
 */
export function calculateCategoryStats(results, engines) {
  const stats = {
    all: { success: 0, error: 0, pending: 0 },
    ml: { success: 0, error: 0, pending: 0 },
    financial: { success: 0, error: 0, pending: 0 },
    advanced: { success: 0, error: 0, pending: 0 }
  };

  engines.forEach(engine => {
    const result = results[engine.name];
    if (!result) {
      stats[engine.category].pending++;
      stats.all.pending++;
    } else if (result.status === 'success') {
      stats[engine.category].success++;
      stats.all.success++;
    } else {
      stats[engine.category].error++;
      stats.all.error++;
    }
  });

  return stats;
}
