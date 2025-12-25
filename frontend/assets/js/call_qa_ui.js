/**
 * Call QA Dashboard UI - Agent QA metrics, leaderboard, and compliance flags
 * Part of the Call QA & Vectorization System
 * 
 * Dependencies: banking_core.js (getCsrfToken)
 * Used by: banking.html
 */

// =================================================================
// QA DASHBOARD MAIN FUNCTIONS
// =================================================================

/**
 * Load the QA Dashboard with all components
 */
async function loadQADashboard() {
    console.log('[QA Dashboard] Loading...');

    // Load all components in parallel
    await Promise.all([
        loadQAStats(),
        loadAgentLeaderboard(),
        loadComplianceFlags(),
        loadRecentQAResults()
    ]);

    console.log('[QA Dashboard] Load complete');
}

/**
 * Load QA statistics summary
 */
async function loadQAStats() {
    try {
        const resp = await fetch('/api/v1/calls/qa/stats', {
            headers: {
                'Accept': 'application/json',
                'X-CSRF-Token': getCsrfToken()
            }
        });

        if (!resp.ok) {
            console.warn('[QA Stats] Endpoint not available');
            return;
        }

        const data = await resp.json();

        // Update stat cards
        updateStatCard('qa-avg-overall', data.avg_overall, 'Overall Score');
        updateStatCard('qa-avg-compliance', data.avg_compliance, 'Compliance');
        updateStatCard('qa-calls-analyzed', data.calls_analyzed, 'Calls Analyzed');
        updateStatCard('qa-flags-count', data.flags_count, 'Flags');

    } catch (e) {
        console.error('[QA Stats] Failed:', e);
    }
}

/**
 * Update a stat card value and trend
 */
function updateStatCard(elementId, value, label) {
    const el = document.getElementById(elementId);
    if (el) {
        if (typeof value === 'number') {
            el.textContent = Number.isInteger(value) ? value : value.toFixed(1);
        } else {
            el.textContent = value || '--';
        }
    }
}

// =================================================================
// AGENT LEADERBOARD
// =================================================================

/**
 * Load agent QA leaderboard
 * @param {string} period - Time period: 7d, 30d, 90d
 * @param {number} limit - Max agents to show
 */
async function loadAgentLeaderboard(period = '7d', limit = 10) {
    const tableEl = document.getElementById('qa-leaderboard-table');
    const loadingEl = document.getElementById('qa-leaderboard-loading');

    if (loadingEl) loadingEl.style.display = 'block';

    try {
        const resp = await fetch(`/api/v1/calls/qa/leaderboard?period=${period}&limit=${limit}`, {
            headers: {
                'Accept': 'application/json',
                'X-CSRF-Token': getCsrfToken()
            }
        });

        if (!resp.ok) {
            throw new Error(`HTTP ${resp.status}`);
        }

        const data = await resp.json();

        if (loadingEl) loadingEl.style.display = 'none';

        if (!data.leaderboard || data.leaderboard.length === 0) {
            if (tableEl) {
                tableEl.innerHTML = `
                    <tbody>
                        <tr><td colspan="6" class="text-center" style="padding: 20px; color: #888;">
                            No QA data available yet. Process some calls to see the leaderboard.
                        </td></tr>
                    </tbody>
                `;
            }
            return;
        }

        // Render leaderboard table
        renderLeaderboardTable(data.leaderboard);

    } catch (e) {
        console.error('[QA Leaderboard] Failed:', e);
        if (loadingEl) loadingEl.style.display = 'none';
        if (tableEl) {
            tableEl.innerHTML = `
                <tbody>
                    <tr><td colspan="6" class="text-center" style="padding: 20px; color: #dc3545;">
                        Failed to load leaderboard: ${e.message}
                    </td></tr>
                </tbody>
            `;
        }
    }
}

/**
 * Render leaderboard table HTML
 */
function renderLeaderboardTable(agents) {
    const tableEl = document.getElementById('qa-leaderboard-table');
    if (!tableEl) return;

    const tbody = agents.map((agent, idx) => {
        const rank = idx + 1;
        const rankEmoji = rank === 1 ? '🥇' : rank === 2 ? '🥈' : rank === 3 ? '🥉' : rank;
        const overallColor = getScoreColor(agent.avg_overall);

        return `
            <tr onclick="loadAgentDetails('${agent.agent_id}')" style="cursor: pointer;">
                <td style="font-weight: bold;">${rankEmoji}</td>
                <td>${agent.agent_id}</td>
                <td style="font-weight: bold; color: ${overallColor};">${formatScore(agent.avg_overall)}</td>
                <td>${formatScore(agent.avg_professionalism)}</td>
                <td>${formatScore(agent.avg_compliance)}</td>
                <td>${agent.calls_analyzed || 0}</td>
            </tr>
        `;
    }).join('');

    tableEl.innerHTML = `
        <thead>
            <tr>
                <th>#</th>
                <th>Agent</th>
                <th>Overall</th>
                <th>Prof.</th>
                <th>Comp.</th>
                <th>Calls</th>
            </tr>
        </thead>
        <tbody>${tbody}</tbody>
    `;
}

/**
 * Get color for a QA score
 */
function getScoreColor(score) {
    if (score === null || score === undefined) return '#888';
    if (score >= 8) return '#28a745'; // Green
    if (score >= 6) return '#17a2b8'; // Blue
    if (score >= 4) return '#ffc107'; // Orange
    return '#dc3545'; // Red
}

/**
 * Format score for display
 */
function formatScore(score) {
    if (score === null || score === undefined) return '--';
    return typeof score === 'number' ? score.toFixed(1) : score;
}

// =================================================================
// COMPLIANCE FLAGS
// =================================================================

/**
 * Load compliance flags (calls requiring review)
 * @param {number} days - Days to look back
 */
async function loadComplianceFlags(days = 7) {
    const listEl = document.getElementById('qa-flags-list');

    try {
        const resp = await fetch(`/api/v1/calls/qa/flags?days=${days}`, {
            headers: {
                'Accept': 'application/json',
                'X-CSRF-Token': getCsrfToken()
            }
        });

        if (!resp.ok) {
            throw new Error(`HTTP ${resp.status}`);
        }

        const data = await resp.json();

        if (!data.flagged || data.flagged.length === 0) {
            if (listEl) {
                listEl.innerHTML = `
                    <div class="empty-state" style="padding: 20px; text-align: center; color: #888;">
                        ✅ No compliance issues in the last ${days} days
                    </div>
                `;
            }
            return;
        }

        // Render flags
        renderComplianceFlags(data.flagged);

    } catch (e) {
        console.error('[QA Flags] Failed:', e);
        if (listEl) {
            listEl.innerHTML = `
                <div class="error-state" style="padding: 20px; text-align: center; color: #dc3545;">
                    Failed to load compliance flags
                </div>
            `;
        }
    }
}

/**
 * Render compliance flags list
 */
function renderComplianceFlags(flags) {
    const listEl = document.getElementById('qa-flags-list');
    if (!listEl) return;

    listEl.innerHTML = flags.map(flag => `
        <div class="flag-item" style="padding: 12px; border-left: 4px solid ${getSeverityColor(flag.severity)}; margin-bottom: 8px; background: #fff3cd20;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <strong>Call ${flag.call_id?.slice(0, 8) || 'Unknown'}</strong>
                <span class="badge" style="background: ${getSeverityColor(flag.severity)}; color: white; padding: 2px 8px; border-radius: 4px;">
                    ${flag.severity || 'Medium'}
                </span>
            </div>
            <div style="font-size: 0.9em; color: #666; margin-top: 4px;">
                ${flag.review_reason || 'Flagged for review'}
            </div>
            <div style="font-size: 0.8em; color: #888; margin-top: 4px;">
                Agent: ${flag.agent_id || 'Unknown'} | ${formatTimestamp(flag.analyzed_at)}
            </div>
            <button onclick="reviewFlag('${flag.id}')" class="btn btn-sm" style="margin-top: 8px; padding: 4px 12px; font-size: 0.8em;">
                Review
            </button>
        </div>
    `).join('');
}

/**
 * Get color for severity level
 */
function getSeverityColor(severity) {
    switch ((severity || '').toLowerCase()) {
        case 'critical': return '#dc3545';
        case 'high': return '#fd7e14';
        case 'medium': return '#ffc107';
        case 'low': return '#28a745';
        default: return '#6c757d';
    }
}

/**
 * Format timestamp for display
 */
function formatTimestamp(ts) {
    if (!ts) return '';
    const date = new Date(ts);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// =================================================================
// RECENT QA RESULTS
// =================================================================

/**
 * Load recent QA analysis results
 */
async function loadRecentQAResults() {
    const containerEl = document.getElementById('qa-recent-results');
    if (!containerEl) return;

    try {
        const resp = await fetch('/api/v1/calls?days=7&limit=10', {
            headers: {
                'Accept': 'application/json',
                'X-CSRF-Token': getCsrfToken()
            }
        });

        if (!resp.ok) return;
        const data = await resp.json();

        // Render recent calls with QA scores
        // This would show calls that have been QA analyzed

    } catch (e) {
        console.error('[QA Recent] Failed:', e);
    }
}

// =================================================================
// QA PROCESSING (MANUAL TRIGGER)
// =================================================================

/**
 * Manually trigger QA processing for a call
 * @param {string} callId - Call ID
 * @param {string} transcript - Full transcript
 * @param {string} agentId - Agent ID (optional)
 */
async function processCallQA(callId, transcript, agentId = null) {
    const resultsEl = document.getElementById('qa-process-results');

    if (!callId || !transcript) {
        alert('Call ID and transcript are required');
        return;
    }

    // Show loading state
    if (resultsEl) {
        resultsEl.innerHTML = `
            <div style="text-align: center; padding: 40px;">
                <div class="spinner"></div>
                <p style="margin-top: 12px;">Analyzing transcript with Gemma AI...</p>
                <p style="font-size: 0.8em; color: #888;">This may take 30-60 seconds</p>
            </div>
        `;
    }

    try {
        const resp = await fetch('/api/v1/calls/qa/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRF-Token': typeof getCsrfToken === 'function' ? getCsrfToken() : ''
            },
            credentials: 'include',
            body: JSON.stringify({
                call_id: callId,
                agent_id: agentId,
                transcript: transcript
            })
        });

        if (!resp.ok) {
            const error = await resp.json();
            throw new Error(error.detail || `HTTP ${resp.status}`);
        }

        const data = await resp.json();

        // Render results
        renderQAResults(data);

    } catch (e) {
        console.error('[QA Process] Failed:', e);
        if (resultsEl) {
            resultsEl.innerHTML = `
                <div style="text-align: center; padding: 40px; color: #dc3545;">
                    <strong>Analysis Failed</strong>
                    <p>${e.message}</p>
                </div>
            `;
        }
    }
}

/**
 * Render QA processing results
 */
function renderQAResults(data) {
    const resultsEl = document.getElementById('qa-process-results');
    if (!resultsEl) return;

    const avgScores = data.avg_scores || {};
    const chunks = data.chunks || [];

    resultsEl.innerHTML = `
        <div class="qa-results">
            <!-- Summary Cards -->
            <div class="score-cards" style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px;">
                ${renderScoreCard('Overall', avgScores.overall, getScoreColor(avgScores.overall))}
                ${renderScoreCard('Professionalism', avgScores.professionalism, getScoreColor(avgScores.professionalism))}
                ${renderScoreCard('Compliance', avgScores.compliance, getScoreColor(avgScores.compliance))}
                ${renderScoreCard('Customer Svc', avgScores.customer_service, getScoreColor(avgScores.customer_service))}
            </div>
            
            <!-- Processing Stats -->
            <div style="background: #f8f9fa; padding: 12px; border-radius: 8px; margin-bottom: 20px;">
                <strong>Processing Summary</strong><br>
                Chunks analyzed: ${chunks.length}<br>
                Processing time: ${data.total_processing_time_sec?.toFixed(1) || '--'}s<br>
                Flags raised: ${data.compliance_flags_count || 0}<br>
                Requires review: ${data.requires_review_count || 0}
            </div>
            
            <!-- Chunk Details (collapsible) -->
            <details>
                <summary style="cursor: pointer; font-weight: bold;">View Chunk Details (${chunks.length})</summary>
                <div style="margin-top: 12px;">
                    ${chunks.map((chunk, idx) => renderChunkCard(chunk, idx)).join('')}
                </div>
            </details>
        </div>
    `;
}

/**
 * Render individual score card
 */
function renderScoreCard(label, score, color) {
    return `
        <div style="background: white; border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; text-align: center;">
            <div style="font-size: 2em; font-weight: bold; color: ${color};">${formatScore(score)}</div>
            <div style="font-size: 0.9em; color: #666;">${label}</div>
        </div>
    `;
}

/**
 * Render chunk analysis card
 */
function renderChunkCard(chunk, idx) {
    const scores = chunk.scores || {};
    return `
        <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <strong>Chunk ${idx + 1}</strong>
                <span style="color: ${getScoreColor(scores.overall || 5)}; font-weight: bold;">
                    Overall: ${formatScore(scores.overall || 5)}
                </span>
            </div>
            <div style="font-size: 0.9em; color: #666; margin-bottom: 8px;">
                ${chunk.chunk_text?.slice(0, 150) || ''}...
            </div>
            <div style="font-size: 0.8em; display: flex; gap: 12px;">
                <span>Prof: ${formatScore(scores.professionalism)}</span>
                <span>Comp: ${formatScore(scores.compliance)}</span>
                <span>Svc: ${formatScore(scores.customer_service)}</span>
                <span>Proto: ${formatScore(scores.protocol_adherence)}</span>
            </div>
            ${chunk.requires_review ? `<div style="margin-top: 8px; color: #ffc107;">⚠️ Flagged for review</div>` : ''}
        </div>
    `;
}

// =================================================================
// AGENT DETAILS
// =================================================================

/**
 * Load detailed metrics for a specific agent
 */
async function loadAgentDetails(agentId) {
    try {
        const resp = await fetch(`/api/v1/calls/agents/${agentId}/qa-metrics?period=7d`, {
            headers: {
                'Accept': 'application/json',
                'X-CSRF-Token': getCsrfToken()
            }
        });

        if (!resp.ok) return;
        const data = await resp.json();

        // Show in modal or side panel
        console.log('[QA] Agent details:', data);

    } catch (e) {
        console.error('[QA Agent] Failed:', e);
    }
}

/**
 * Review a flagged item
 */
async function reviewFlag(flagId) {
    alert(`Review functionality for flag ${flagId} - coming soon`);
    // TODO: Open review modal
}

// =================================================================
// INITIALIZATION
// =================================================================

// Make functions globally available
window.loadQADashboard = loadQADashboard;
window.loadAgentLeaderboard = loadAgentLeaderboard;
window.loadComplianceFlags = loadComplianceFlags;
window.processCallQA = processCallQA;
window.loadAgentDetails = loadAgentDetails;
window.reviewFlag = reviewFlag;
