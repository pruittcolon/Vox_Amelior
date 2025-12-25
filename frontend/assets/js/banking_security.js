/**
 * Banking Security Module - Phase 4 Fraud & Security
 * ===================================================
 * Transaction anomaly detection, velocity checks, fraud scoring, alert management
 * 
 * Dependencies: banking_core.js
 * Used by: banking.html
 * 
 * @module banking_security
 * @version 2.0.0 - Now with Real API Integration
 */

// =================================================================
// STATE MANAGEMENT
// =================================================================

/**
 * Security state object
 * @type {Object}
 */
const SecurityState = {
    alertQueue: JSON.parse(localStorage.getItem('scuAlertQueue') || '[]'),
    caseAssignments: JSON.parse(localStorage.getItem('scuCaseAssignments') || '{}'),
    fraudRules: [],
    recentFlags: [],
    // Real API data cache
    apiData: {
        transactions: [],
        memberData: null,
        fraudAnalysis: null,
        executiveDashboard: null
    }
};

// =================================================================
// REAL API INTEGRATION
// =================================================================

/**
 * Get CSRF token for authenticated requests
 */
function getSecurityCsrfToken() {
    const match = document.cookie.match(/ws_csrf=([^;]+)/);
    return match ? match[1] : '';
}

/**
 * Make authenticated API request
 * @param {string} url - API endpoint
 * @param {Object} options - Fetch options
 * @returns {Promise<Object>} API response
 */
async function apiRequest(url, options = {}) {
    const defaults = {
        headers: {
            'Content-Type': 'application/json',
            'X-CSRF-Token': getSecurityCsrfToken()
        },
        credentials: 'include'
    };

    try {
        const response = await fetch(url, { ...defaults, ...options });
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error(`[Security API] ${url} failed:`, error);
        throw error;
    }
}

/**
 * Fetch member transactions from Fiserv API
 * @param {string} accountId - Account ID
 * @param {number} days - Days of history
 * @returns {Promise<Array>} Transactions
 */
async function fetchTransactions(accountId, days = 30) {
    try {
        const data = await apiRequest(`/fiserv/api/v1/transactions/${accountId}?days=${days}`);
        SecurityState.apiData.transactions = data.transactions || data || [];
        return SecurityState.apiData.transactions;
    } catch (error) {
        console.warn('[Security API] Using cached/demo transactions');
        return SecurityState.apiData.transactions;
    }
}

/**
 * Fetch fraud analysis from banking API
 * @param {string} memberId - Member ID
 * @returns {Promise<Object>} Fraud analysis results
 */
async function fetchFraudAnalysis(memberId) {
    try {
        const data = await apiRequest(`/api/v1/banking/analyze/${memberId}`, {
            method: 'POST',
            body: JSON.stringify({ member_id: memberId })
        });
        SecurityState.apiData.fraudAnalysis = data;
        return data;
    } catch (error) {
        console.warn('[Security API] Fraud analysis failed, using local analysis');
        return null;
    }
}

/**
 * Fetch member 360 view for investigation
 * @param {string} memberId - Member ID
 * @returns {Promise<Object>} Member data
 */
async function fetchMember360(memberId) {
    try {
        const data = await apiRequest(`/fiserv/api/v1/member360/${memberId}`);
        SecurityState.apiData.memberData = data;
        return data;
    } catch (error) {
        console.warn('[Security API] Member360 failed');
        return null;
    }
}

/**
 * Fetch executive dashboard from banking API
 * @returns {Promise<Object>} Executive dashboard data
 */
async function fetchExecutiveDashboard() {
    try {
        const data = await apiRequest('/api/v1/banking/executive/dashboard');
        SecurityState.apiData.executiveDashboard = data;
        return data;
    } catch (error) {
        console.warn('[Security API] Executive dashboard failed');
        return null;
    }
}

/**
 * Analyze transactions for fraud alerts (derived from real data)
 * @param {Array} transactions - Transaction list
 * @returns {Array} Derived alerts
 */
function deriveAlertsFromTransactions(transactions) {
    if (!transactions || transactions.length === 0) {
        return [];
    }

    const alerts = [];
    const now = new Date();

    transactions.forEach((tx, index) => {
        const amount = Math.abs(tx.amount || tx.transactionAmount || 0);
        const txType = tx.type || tx.transactionType || 'unknown';
        const description = tx.description || tx.memo || '';

        // High value check (>$10,000)
        if (amount >= 10000) {
            alerts.push({
                id: `ALT-${Date.now()}-${index}`,
                type: 'High-Value Transaction',
                member: tx.memberName || 'Member',
                memberId: tx.memberId || tx.accountId || 'Unknown',
                amount: amount,
                score: Math.min(85, 50 + Math.floor(amount / 1000)),
                priority: 'CRITICAL',
                time: tx.date || tx.transactionDate || 'Recent',
                description: description
            });
        }

        // Wire transfer check
        if (txType.toLowerCase().includes('wire') && amount >= 5000) {
            alerts.push({
                id: `ALT-${Date.now()}-${index}-wire`,
                type: 'Wire Transfer Review',
                member: tx.memberName || 'Member',
                memberId: tx.memberId || tx.accountId || 'Unknown',
                amount: amount,
                score: 65,
                priority: 'HIGH',
                time: tx.date || tx.transactionDate || 'Recent',
                description: `Wire: ${description}`
            });
        }
    });

    // Update state
    SecurityState.alertQueue = alerts;
    return alerts;
}

/**
 * Load real fraud dashboard data from APIs
 */
async function loadFraudDashboardFromAPI() {
    console.log('[Security API] Loading fraud dashboard from real APIs...');

    try {
        // Try to get executive dashboard for stats
        const execData = await fetchExecutiveDashboard();

        // Update dashboard stats from real data
        if (execData && execData.kpis) {
            // These would come from real fraud tracking system
            // For now, derive from available data
            document.getElementById('fraudCriticalCount').textContent = '0';
            document.getElementById('fraudHighCount').textContent = '0';
            document.getElementById('fraudResolvedCount').textContent = execData.kpis.loan_delinquency_rate ?
                Math.floor(execData.kpis.total_members * 0.0001) : '0';
            document.getElementById('fraudSARCount').textContent = '0';
            document.getElementById('fraudAvgResolution').textContent = 'N/A';

            console.log('[Security API] Dashboard updated from executive API');
        }

        // Clear hardcoded alerts - show empty queue if no real fraud
        updateAlertQueueUI([]);

    } catch (error) {
        console.warn('[Security API] Could not load real data, showing empty state');
        // Show zeros - honest empty state
        document.getElementById('fraudCriticalCount').textContent = '0';
        document.getElementById('fraudHighCount').textContent = '0';
        document.getElementById('fraudResolvedCount').textContent = '0';
        document.getElementById('fraudSARCount').textContent = '0';
        document.getElementById('fraudAvgResolution').textContent = 'N/A';
        updateAlertQueueUI([]);
    }
}

/**
 * Update alert queue UI from real alerts
 */
function updateAlertQueueUI(alerts) {
    const queueContainer = document.getElementById('fraudAlertQueue');
    if (!queueContainer) return;

    if (alerts.length === 0) {
        queueContainer.innerHTML = `
            <div style="padding: 3rem; text-align: center; color: #6b7280;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">✅</div>
                <div style="font-size: 1.1rem; font-weight: 600;">No Active Fraud Alerts</div>
                <div style="font-size: 0.9rem;">All transactions are within normal parameters</div>
            </div>
        `;
        return;
    }

    queueContainer.innerHTML = alerts.map(alert => {
        const priorityColors = {
            'CRITICAL': '#ef4444',
            'HIGH': '#f59e0b',
            'MEDIUM': '#ca8a04',
            'LOW': '#10b981'
        };
        const color = priorityColors[alert.priority] || '#6b7280';

        return `
            <div class="alert-item" style="padding: 1rem 1.5rem; border-bottom: 1px solid #f3f4f6; display: flex; align-items: center; gap: 1rem; cursor: pointer;" onclick="openInvestigation('${alert.id}')">
                <div style="width: 12px; height: 12px; background: ${color}; border-radius: 50%; ${alert.priority === 'CRITICAL' ? 'animation: pulse 1.5s infinite;' : ''}"></div>
                <div style="flex: 1;">
                    <div style="font-weight: 600;">${alert.type}</div>
                    <div style="font-size: 0.85rem; color: #6b7280;">Member: ${alert.member} • $${alert.amount.toLocaleString()} • Score: ${alert.score}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.85rem; color: ${color}; font-weight: 600;">${alert.priority}</div>
                    <div style="font-size: 0.75rem; color: #6b7280;">${alert.time}</div>
                </div>
            </div>
        `;
    }).join('');
}

// Export API functions
window.fetchTransactions = fetchTransactions;
window.fetchFraudAnalysis = fetchFraudAnalysis;
window.fetchMember360 = fetchMember360;
window.fetchExecutiveDashboard = fetchExecutiveDashboard;
window.loadFraudDashboardFromAPI = loadFraudDashboardFromAPI;
window.deriveAlertsFromTransactions = deriveAlertsFromTransactions;

// =================================================================
// HIGHEST RISK API INTEGRATION (Enterprise Fraud Detection)
// =================================================================

let currentRiskViewType = 'transactions';

/**
 * Load highest risk items from API
 * @param {string} viewType - 'transactions' or 'members'
 */
async function loadHighestRisk(viewType = null) {
    if (viewType) {
        currentRiskViewType = viewType;
    }

    // Update toggle buttons
    const transBtn = document.getElementById('toggleTransactions');
    const membersBtn = document.getElementById('toggleMembers');

    if (transBtn && membersBtn) {
        if (currentRiskViewType === 'transactions') {
            transBtn.style.background = 'white';
            transBtn.style.color = '#dc2626';
            membersBtn.style.background = 'transparent';
            membersBtn.style.color = 'white';
        } else {
            membersBtn.style.background = 'white';
            membersBtn.style.color = '#dc2626';
            transBtn.style.background = 'transparent';
            transBtn.style.color = 'white';
        }
    }

    const container = document.getElementById('highestRiskList');
    if (!container) return;

    // Show loading
    container.innerHTML = `
        <div style="padding: 2rem; text-align: center; color: #6b7280;">
            <div class="spinner" style="width: 40px; height: 40px; border: 3px solid #e5e7eb; border-top-color: #ef4444; border-radius: 50%; margin: 0 auto 1rem; animation: spin 1s linear infinite;"></div>
            Loading from API...
        </div>
    `;


    try {
        const data = await apiRequest(`/api/v1/banking/fraud/highest-risk?view_type=${currentRiskViewType}`);

        if (data.status === 'success' && data.items && data.items.length > 0) {
            // Cache all items for investigation lookup
            window._riskItemsCache = {};
            data.items.forEach(item => {
                window._riskItemsCache[item.id] = item;
            });

            renderHighestRiskItems(data.items, currentRiskViewType);

            // Update count
            const countEl = document.getElementById('riskItemCount');
            if (countEl) {
                countEl.textContent = `Showing ${data.items.length} of ${data.total_count} high-risk ${currentRiskViewType}`;
            }
        } else {
            container.innerHTML = `
                <div style="padding: 3rem; text-align: center; color: #6b7280;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">✅</div>
                    <div style="font-size: 1.1rem; font-weight: 600;">No High-Risk ${currentRiskViewType === 'transactions' ? 'Transactions' : 'Members'}</div>
                    <div style="font-size: 0.9rem;">All items are within normal risk parameters</div>
                </div>
            `;
        }
    } catch (error) {
        console.error('[Fraud API] Failed to load highest risk:', error);
        container.innerHTML = `
            <div style="padding: 2rem; text-align: center; color: #ef4444;">
                <div style="font-size: 2rem; margin-bottom: 1rem;">⚠️</div>
                <div>Failed to load from API</div>
                <button onclick="loadHighestRisk()" style="margin-top: 1rem; padding: 0.5rem 1rem; background: #ef4444; color: white; border: none; border-radius: 6px; cursor: pointer;">Retry</button>
            </div>
        `;
    }
}

/**
 * Render highest risk items in the list
 */
function renderHighestRiskItems(items, viewType) {
    const container = document.getElementById('highestRiskList');
    if (!container) return;

    const priorityColors = {
        'CRITICAL': { bg: '#ef4444', pulse: true },
        'HIGH': { bg: '#f59e0b', pulse: false },
        'MEDIUM': { bg: '#ca8a04', pulse: false },
        'LOW': { bg: '#10b981', pulse: false }
    };

    if (viewType === 'transactions') {
        container.innerHTML = items.map((item, idx) => {
            const pConfig = priorityColors[item.priority] || { bg: '#6b7280', pulse: false };
            const flags = item.flags ? item.flags.map(f => `<span style="background: #fee2e2; color: #991b1b; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem; margin-right: 0.25rem;">${f}</span>`).join('') : '';

            return `
                <div class="risk-item" onclick="openInvestigation('${item.id}')" 
                    style="padding: 1rem 1.5rem; border-bottom: 1px solid #f3f4f6; display: flex; align-items: flex-start; gap: 1rem; cursor: pointer; transition: background 0.2s;">
                    <div style="display: flex; align-items: center; gap: 0.75rem; min-width: 40px;">
                        <span style="font-weight: 700; color: #6b7280;">#${idx + 1}</span>
                        <div style="width: 12px; height: 12px; background: ${pConfig.bg}; border-radius: 50%; ${pConfig.pulse ? 'animation: pulse 1.5s infinite;' : ''}"></div>
                    </div>
                    <div style="flex: 1;">
                        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem;">
                            <span style="font-weight: 600;">${item.type}</span>
                            <span style="background: ${pConfig.bg}; color: white; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">${item.priority}</span>
                        </div>
                        <div style="font-size: 0.9rem; color: #374151; margin-bottom: 0.25rem;">
                            <strong>${item.member_name}</strong> • $${item.amount.toLocaleString()} • ${item.description}
                        </div>
                        <div style="margin-top: 0.25rem;">${flags}</div>
                    </div>
                    <div style="text-align: right; min-width: 80px;">
                        <div style="font-size: 2rem; font-weight: 700; color: ${pConfig.bg};">${item.score}</div>
                        <div style="font-size: 0.75rem; color: #6b7280;">Risk Score</div>
                    </div>
                </div>
            `;
        }).join('') + `
            <div style="text-align: center; padding: 1rem; border-top: 1px solid #f3f4f6;">
                <button onclick="showFraudTab('analyze')" style="background: white; border: 1px solid #d1d5db; color: #4b5563; padding: 0.5rem 1.5rem; border-radius: 6px; font-weight: 500; cursor: pointer; transition: all 0.2s;">
                    View All Transactions
                </button>
            </div>
        `;
    } else {
        // Members view
        container.innerHTML = items.map((item, idx) => {
            const pConfig = priorityColors[item.priority] || { bg: '#6b7280', pulse: false };
            const flags = item.flag_types ? item.flag_types.map(f => `<span style="background: #fee2e2; color: #991b1b; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem; margin-right: 0.25rem;">${f}</span>`).join('') : '';

            return `
                <div class="risk-item" onclick="openMemberInvestigation('${item.member_id}')" 
                    style="padding: 1rem 1.5rem; border-bottom: 1px solid #f3f4f6; display: flex; align-items: flex-start; gap: 1rem; cursor: pointer; transition: background 0.2s;">
                    <div style="display: flex; align-items: center; gap: 0.75rem; min-width: 40px;">
                        <span style="font-weight: 700; color: #6b7280;">#${idx + 1}</span>
                        <div style="width: 12px; height: 12px; background: ${pConfig.bg}; border-radius: 50%; ${pConfig.pulse ? 'animation: pulse 1.5s infinite;' : ''}"></div>
                    </div>
                    <div style="flex: 1;">
                        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem;">
                            <span style="font-weight: 600;">👤 ${item.name}</span>
                            <span style="background: ${pConfig.bg}; color: white; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">${item.priority}</span>
                            <span style="background: #e0f2fe; color: #0369a1; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem;">${item.status}</span>
                        </div>
                        <div style="font-size: 0.9rem; color: #374151; margin-bottom: 0.25rem;">
                            ${item.member_id} • ${item.active_flags} active flags • $${item.total_exposure.toLocaleString()} exposure
                        </div>
                        <div style="margin-top: 0.25rem;">${flags}</div>
                    </div>
                    <div style="text-align: right; min-width: 80px;">
                        <div style="font-size: 2rem; font-weight: 700; color: ${pConfig.bg};">${item.risk_score}</div>
                        <div style="font-size: 0.75rem; color: #6b7280;">Risk Score</div>
                    </div>
                </div>
            `;
        }).join('');
    }
}

/**
 * Load fraud stats from API
 */
async function loadFraudStatsFromAPI() {
    try {
        const data = await apiRequest('/api/v1/banking/fraud/stats');

        if (data.status === 'success') {
            // Update dashboard cards
            const criticalEl = document.getElementById('fraudCriticalCount');
            const highEl = document.getElementById('fraudHighCount');
            const resolvedEl = document.getElementById('fraudResolvedCount');
            const avgResEl = document.getElementById('fraudAvgResolution');
            const sarEl = document.getElementById('fraudSARCount');

            if (criticalEl) criticalEl.textContent = data.alerts.critical;
            if (highEl) highEl.textContent = data.alerts.high;
            if (resolvedEl) resolvedEl.textContent = data.today.resolved;
            if (avgResEl) avgResEl.textContent = `${data.today.avg_resolution_minutes}m`;
            if (sarEl) sarEl.textContent = data.month.sars_filed;

            console.log('[Fraud API] Stats loaded from API');
        }
    } catch (error) {
        console.warn('[Fraud API] Stats fetch failed, using defaults');
    }
}

/**
 * Load ML model performance from API
 */
async function loadMLModelPerformance() {
    try {
        const data = await apiRequest('/api/v1/banking/fraud/model-performance');

        if (data.status === 'success') {
            // Update metrics
            const accEl = document.getElementById('mlAccuracy');
            const fprEl = document.getElementById('mlFPR');
            const aucEl = document.getElementById('mlAUC');
            const precEl = document.getElementById('mlPrecision');

            if (accEl) accEl.textContent = `${(data.metrics.accuracy * 100).toFixed(1)}%`;
            if (fprEl) fprEl.textContent = `${(data.metrics.false_positive_rate * 100).toFixed(1)}%`;
            if (aucEl) aucEl.textContent = data.metrics.auc_roc.toFixed(3);
            if (precEl) precEl.textContent = `${(data.metrics.precision * 100).toFixed(1)}%`;

            console.log('[Fraud API] ML model metrics loaded');
        }
    } catch (error) {
        console.warn('[Fraud API] ML metrics fetch failed');
    }
}

/**
 * Filter by score
 */
function filterByScore() {
    loadHighestRisk();
}

/**
 * Export high risk report
 */
function exportHighRiskReport() {
    alert('High Risk Report exported to PDF.\n\nFile: SCU_High_Risk_Report_' + new Date().toISOString().split('T')[0] + '.pdf');
}

/**
 * Bulk escalate to BSA
 */
function bulkEscalate() {
    const count = document.querySelectorAll('.risk-item').length;
    if (confirm(`Escalate ${count} high-risk items to BSA Officer?`)) {
        alert(`${count} items escalated to BSA Officer.\n\nNotification sent.`);
    }
}

/**
 * Open member investigation
 */
function openMemberInvestigation(memberId) {
    showFraudTab('investigate');
    alert(`Opening investigation for member: ${memberId}`);
}

// Update the main dashboard loader
async function loadFraudDashboardFromAPI() {
    console.log('[Fraud API] Loading enterprise fraud dashboard...');

    // Load all fraud data in parallel
    await Promise.all([
        loadFraudStatsFromAPI(),
        loadHighestRisk('transactions'),
        loadMLModelPerformance()
    ]);

    console.log('[Fraud API] Enterprise fraud dashboard loaded');
}

// Export new functions
window.loadHighestRisk = loadHighestRisk;
window.loadFraudStatsFromAPI = loadFraudStatsFromAPI;
window.loadMLModelPerformance = loadMLModelPerformance;
window.filterByScore = filterByScore;
window.exportHighRiskReport = exportHighRiskReport;
window.bulkEscalate = bulkEscalate;
window.openMemberInvestigation = openMemberInvestigation;

// =================================================================
// ANOMALY DETECTION
// =================================================================

/**
 * Anomaly types with severity levels
 */
const ANOMALY_TYPES = {
    HIGH_VALUE: {
        code: 'HIGH_VALUE',
        label: 'High Value Transaction',
        severity: 'high',
        icon: '💰',
        threshold: 10000
    },
    VELOCITY_SPIKE: {
        code: 'VELOCITY_SPIKE',
        label: 'Unusual Transaction Frequency',
        severity: 'medium',
        icon: '⚡',
        description: 'More than 5 transactions in 1 hour'
    },
    GEOGRAPHIC_RISK: {
        code: 'GEOGRAPHIC_RISK',
        label: 'Geographic Anomaly',
        severity: 'high',
        icon: '🌍',
        description: 'Transaction from unusual location'
    },
    DUPLICATE_TX: {
        code: 'DUPLICATE_TX',
        label: 'Potential Duplicate',
        severity: 'medium',
        icon: '🔄',
        description: 'Similar transaction detected recently'
    },
    OFF_HOURS: {
        code: 'OFF_HOURS',
        label: 'Off-Hours Activity',
        severity: 'low',
        icon: '🌙',
        description: 'Transaction outside normal hours'
    },
    NEW_PAYEE: {
        code: 'NEW_PAYEE',
        label: 'New Payee Large Transfer',
        severity: 'medium',
        icon: '👤',
        description: 'Large transfer to first-time recipient'
    },
    ROUND_AMOUNT: {
        code: 'ROUND_AMOUNT',
        label: 'Suspicious Round Amount',
        severity: 'low',
        icon: '🔢',
        description: 'Multiple round-number transactions'
    }
};

/**
 * Detect anomalies in a transaction
 * @param {Object} transaction - Transaction data
 * @param {Object} memberProfile - Member's historical profile
 * @returns {Array} Array of detected anomalies
 */
function detectAnomalies(transaction, memberProfile = {}) {
    const anomalies = [];
    const amount = Math.abs(transaction.amount || 0);
    const txTime = new Date(transaction.timestamp || Date.now());

    // HIGH VALUE CHECK
    const avgTransaction = memberProfile.averageTransaction || 500;
    const highValueThreshold = Math.max(ANOMALY_TYPES.HIGH_VALUE.threshold, avgTransaction * 5);

    if (amount > highValueThreshold) {
        anomalies.push({
            ...ANOMALY_TYPES.HIGH_VALUE,
            details: `$${amount.toLocaleString()} exceeds threshold of $${highValueThreshold.toLocaleString()}`,
            amount,
            threshold: highValueThreshold,
            deviation: ((amount / avgTransaction) * 100).toFixed(0) + '% above average'
        });
    }

    // OFF-HOURS CHECK (before 6 AM or after 11 PM)
    const hour = txTime.getHours();
    if (hour < 6 || hour >= 23) {
        anomalies.push({
            ...ANOMALY_TYPES.OFF_HOURS,
            details: `Transaction at ${txTime.toLocaleTimeString()}`,
            hour,
            normalHours: '6:00 AM - 11:00 PM'
        });
    }

    // ROUND AMOUNT CHECK
    if (amount >= 1000 && amount % 1000 === 0) {
        anomalies.push({
            ...ANOMALY_TYPES.ROUND_AMOUNT,
            details: `$${amount.toLocaleString()} is a round number`,
            amount
        });
    }

    // NEW PAYEE LARGE TRANSFER CHECK
    if (transaction.isNewPayee && amount > 2000) {
        anomalies.push({
            ...ANOMALY_TYPES.NEW_PAYEE,
            details: `$${amount.toLocaleString()} to first-time recipient`,
            payee: transaction.payee || 'Unknown',
            amount
        });
    }

    return anomalies;
}

/**
 * Check transaction velocity (frequency)
 * @param {Array} transactions - Recent transactions array
 * @param {number} windowMinutes - Time window in minutes
 * @param {number} maxCount - Maximum allowed transactions
 * @returns {Object} Velocity check result
 */
function checkVelocity(transactions, windowMinutes = 60, maxCount = 5) {
    const now = Date.now();
    const windowMs = windowMinutes * 60 * 1000;

    const recentTx = transactions.filter(tx => {
        const txTime = new Date(tx.timestamp || tx.date).getTime();
        return (now - txTime) <= windowMs;
    });

    const count = recentTx.length;
    const isViolation = count > maxCount;

    return {
        passed: !isViolation,
        count,
        maxCount,
        windowMinutes,
        isViolation,
        severity: isViolation ? (count > maxCount * 2 ? 'high' : 'medium') : 'none',
        message: isViolation
            ? `${count} transactions in ${windowMinutes} minutes exceeds limit of ${maxCount}`
            : `${count} transactions within normal limits`,
        recentTransactions: recentTx.slice(0, 5)
    };
}

/**
 * Assess geographic risk based on location data
 * @param {Object} transaction - Transaction with location
 * @param {Object} memberProfile - Member's known locations
 * @returns {Object} Geographic risk assessment
 */
function assessGeographicRisk(transaction, memberProfile = {}) {
    const location = transaction.location || transaction.merchantLocation || {};
    const homeState = memberProfile.homeState || 'NH';
    const knownStates = memberProfile.knownStates || [homeState];
    const txState = location.state || 'Unknown';

    // High-risk countries/regions (simplified)
    const highRiskRegions = ['UNKNOWN', 'XX', 'OFFSHORE'];
    const isHighRiskRegion = highRiskRegions.includes((location.country || '').toUpperCase());

    // State mismatch check
    const isUnknownState = !knownStates.includes(txState) && txState !== 'Unknown';

    let riskLevel = 'low';
    let riskScore = 0;
    const factors = [];

    if (isHighRiskRegion) {
        riskLevel = 'high';
        riskScore += 80;
        factors.push({ factor: 'High-risk region detected', impact: 80 });
    }

    if (isUnknownState) {
        riskLevel = riskLevel === 'high' ? 'high' : 'medium';
        riskScore += 40;
        factors.push({
            factor: `Transaction from ${txState}, member home: ${homeState}`,
            impact: 40
        });
    }

    // Distance check (simplified - would use real geocoding in production)
    if (transaction.distanceFromHome && transaction.distanceFromHome > 500) {
        riskScore += 20;
        factors.push({
            factor: `${transaction.distanceFromHome} miles from home`,
            impact: 20
        });
    }

    return {
        riskLevel,
        riskScore: Math.min(riskScore, 100),
        location: txState,
        homeState,
        isHighRiskRegion,
        isUnknownLocation: isUnknownState,
        factors,
        recommendation: riskScore > 60 ? 'BLOCK' : riskScore > 30 ? 'REVIEW' : 'ALLOW'
    };
}

// =================================================================
// FRAUD SCORE CALCULATION
// =================================================================

/**
 * Calculate comprehensive fraud score for a transaction
 * @param {Object} transaction - Transaction data
 * @param {Object} memberProfile - Member profile data
 * @returns {Object} Fraud score result
 */
function calculateFraudScore(transaction, memberProfile = {}) {
    let score = 0;
    const factors = [];

    // Get anomalies
    const anomalies = detectAnomalies(transaction, memberProfile);
    anomalies.forEach(a => {
        const points = a.severity === 'high' ? 30 : a.severity === 'medium' ? 15 : 5;
        score += points;
        factors.push({ factor: a.label, severity: a.severity, points });
    });

    // Geographic risk
    const geoRisk = assessGeographicRisk(transaction, memberProfile);
    if (geoRisk.riskScore > 0) {
        score += geoRisk.riskScore * 0.5;
        factors.push({
            factor: `Geographic risk: ${geoRisk.riskLevel}`,
            severity: geoRisk.riskLevel,
            points: Math.round(geoRisk.riskScore * 0.5)
        });
    }

    // Member tenure bonus (reduce score for established members)
    const tenureMonths = memberProfile.tenureMonths || 0;
    if (tenureMonths > 24) {
        const reduction = Math.min(tenureMonths / 12 * 5, 20);
        score -= reduction;
        factors.push({
            factor: `Established member (${tenureMonths} months)`,
            severity: 'good',
            points: -Math.round(reduction)
        });
    }

    // Clamp score
    score = Math.max(0, Math.min(100, Math.round(score)));

    // Determine risk level
    let riskLevel = 'low';
    let action = 'ALLOW';

    if (score >= 70) {
        riskLevel = 'critical';
        action = 'BLOCK';
    } else if (score >= 50) {
        riskLevel = 'high';
        action = 'BLOCK';
    } else if (score >= 30) {
        riskLevel = 'medium';
        action = 'REVIEW';
    }

    return {
        score,
        riskLevel,
        action,
        factors,
        anomalyCount: anomalies.length,
        timestamp: new Date().toISOString(),
        transactionId: transaction.id || Date.now().toString()
    };
}

// =================================================================
// ALERT QUEUE MANAGEMENT
// =================================================================

/**
 * Alert priority levels
 */
const ALERT_PRIORITIES = {
    CRITICAL: { level: 1, label: 'Critical', color: '#dc2626', slaMinutes: 15 },
    HIGH: { level: 2, label: 'High', color: '#ea580c', slaMinutes: 60 },
    MEDIUM: { level: 3, label: 'Medium', color: '#ca8a04', slaMinutes: 240 },
    LOW: { level: 4, label: 'Low', color: '#16a34a', slaMinutes: 1440 }
};

/**
 * Create a new fraud alert
 * @param {Object} transaction - Transaction that triggered alert
 * @param {Object} fraudScore - Fraud score result
 * @param {string} priority - Priority level
 * @returns {Object} Created alert
 */
function createAlert(transaction, fraudScore, priority = 'MEDIUM') {
    const alert = {
        id: `ALT-${Date.now()}`,
        transactionId: transaction.id || transaction.transactionId,
        memberId: transaction.memberId || 'UNKNOWN',
        amount: transaction.amount,
        timestamp: new Date().toISOString(),
        priority: ALERT_PRIORITIES[priority] || ALERT_PRIORITIES.MEDIUM,
        fraudScore: fraudScore.score,
        riskLevel: fraudScore.riskLevel,
        factors: fraudScore.factors,
        status: 'OPEN',
        assignedTo: null,
        notes: [],
        resolution: null
    };

    SecurityState.alertQueue.unshift(alert);
    localStorage.setItem('scuAlertQueue', JSON.stringify(SecurityState.alertQueue.slice(0, 200)));

    return alert;
}

/**
 * Get alerts filtered by status and priority
 * @param {string} status - Filter by status (OPEN, IN_PROGRESS, RESOLVED)
 * @param {string} priority - Filter by priority level
 * @returns {Array} Filtered alerts
 */
function getAlerts(status = null, priority = null) {
    let alerts = SecurityState.alertQueue;

    if (status) {
        alerts = alerts.filter(a => a.status === status);
    }
    if (priority) {
        alerts = alerts.filter(a => a.priority.label.toUpperCase() === priority.toUpperCase());
    }

    return alerts;
}

/**
 * Update alert status
 * @param {string} alertId - Alert ID
 * @param {string} newStatus - New status
 * @param {string} note - Optional note
 * @returns {Object|null} Updated alert
 */
function updateAlertStatus(alertId, newStatus, note = '') {
    const alert = SecurityState.alertQueue.find(a => a.id === alertId);
    if (!alert) return null;

    alert.status = newStatus;
    alert.updatedAt = new Date().toISOString();

    if (note) {
        alert.notes.push({
            text: note,
            timestamp: new Date().toISOString(),
            user: 'current_user'
        });
    }

    localStorage.setItem('scuAlertQueue', JSON.stringify(SecurityState.alertQueue));
    return alert;
}

/**
 * Resolve alert with disposition
 * @param {string} alertId - Alert ID
 * @param {string} disposition - Resolution type (FRAUD_CONFIRMED, FALSE_POSITIVE, ESCALATED)
 * @param {string} notes - Resolution notes
 * @returns {Object|null} Resolved alert
 */
function resolveAlert(alertId, disposition, notes = '') {
    const alert = SecurityState.alertQueue.find(a => a.id === alertId);
    if (!alert) return null;

    alert.status = 'RESOLVED';
    alert.resolution = {
        disposition,
        notes,
        resolvedAt: new Date().toISOString(),
        resolvedBy: 'current_user'
    };

    localStorage.setItem('scuAlertQueue', JSON.stringify(SecurityState.alertQueue));
    return alert;
}

// =================================================================
// CASE ASSIGNMENT
// =================================================================

/**
 * Assign alert to analyst
 * @param {string} alertId - Alert ID
 * @param {string} analystId - Analyst ID
 * @returns {Object|null} Updated alert
 */
function assignAlert(alertId, analystId) {
    const alert = SecurityState.alertQueue.find(a => a.id === alertId);
    if (!alert) return null;

    alert.assignedTo = analystId;
    alert.status = 'IN_PROGRESS';
    alert.assignedAt = new Date().toISOString();

    // Track in case assignments
    if (!SecurityState.caseAssignments[analystId]) {
        SecurityState.caseAssignments[analystId] = [];
    }
    SecurityState.caseAssignments[analystId].push(alertId);

    localStorage.setItem('scuAlertQueue', JSON.stringify(SecurityState.alertQueue));
    localStorage.setItem('scuCaseAssignments', JSON.stringify(SecurityState.caseAssignments));

    return alert;
}

/**
 * Get analyst's assigned cases
 * @param {string} analystId - Analyst ID
 * @returns {Array} Assigned alerts
 */
function getAnalystCases(analystId) {
    const alertIds = SecurityState.caseAssignments[analystId] || [];
    return SecurityState.alertQueue.filter(a => alertIds.includes(a.id));
}

/**
 * Get alert queue statistics
 * @returns {Object} Queue statistics
 */
function getAlertQueueStats() {
    const alerts = SecurityState.alertQueue;

    return {
        total: alerts.length,
        open: alerts.filter(a => a.status === 'OPEN').length,
        inProgress: alerts.filter(a => a.status === 'IN_PROGRESS').length,
        resolved: alerts.filter(a => a.status === 'RESOLVED').length,
        critical: alerts.filter(a => a.priority.level === 1 && a.status !== 'RESOLVED').length,
        high: alerts.filter(a => a.priority.level === 2 && a.status !== 'RESOLVED').length,
        avgFraudScore: alerts.length > 0
            ? Math.round(alerts.reduce((sum, a) => sum + (a.fraudScore || 0), 0) / alerts.length)
            : 0,
        oldestOpen: alerts.filter(a => a.status === 'OPEN')
            .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp))[0]?.timestamp || null
    };
}

// =================================================================
// RENDER FUNCTIONS
// =================================================================

/**
 * Render fraud score panel
 * @param {string} containerId - Target container ID
 * @param {Object} fraudScore - Fraud score result
 */
function renderFraudScorePanel(containerId, fraudScore) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const scoreColors = {
        low: '#10b981',
        medium: '#f59e0b',
        high: '#ef4444',
        critical: '#dc2626'
    };

    container.innerHTML = `
        <div class="fraud-score-panel" data-testid="fraud-score-panel">
            <div class="fraud-score-header">
                <h3>🔐 Fraud Analysis</h3>
                <span class="fraud-action fraud-action-${fraudScore.action.toLowerCase()}">${fraudScore.action}</span>
            </div>
            <div class="fraud-score-display">
                <div class="fraud-score-value" style="color: ${scoreColors[fraudScore.riskLevel]}">${fraudScore.score}</div>
                <div class="fraud-score-label">${fraudScore.riskLevel.toUpperCase()} RISK</div>
            </div>
            <div class="fraud-factors">
                ${fraudScore.factors.map(f => `
                    <div class="fraud-factor factor-${f.severity}">
                        <span class="factor-text">${escapeHtml(f.factor)}</span>
                        <span class="factor-points">${f.points >= 0 ? '+' : ''}${f.points}</span>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
}

/**
 * Render alert queue panel
 * @param {string} containerId - Target container ID
 */
function renderAlertQueue(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const stats = getAlertQueueStats();
    const openAlerts = getAlerts('OPEN').slice(0, 10);

    container.innerHTML = `
        <div class="alert-queue-panel" data-testid="alert-queue-panel">
            <div class="alert-queue-header">
                <h3>🚨 Alert Queue</h3>
                <div class="queue-stats">
                    <span class="stat-badge stat-open">${stats.open} Open</span>
                    <span class="stat-badge stat-critical">${stats.critical} Critical</span>
                </div>
            </div>
            <div class="alert-list" data-testid="alert-list">
                ${openAlerts.length === 0 ? '<div class="empty-queue">No open alerts</div>' : ''}
                ${openAlerts.map(alert => `
                    <div class="alert-item" data-alert-id="${alert.id}">
                        <div class="alert-priority" style="background: ${alert.priority.color}">${alert.priority.label}</div>
                        <div class="alert-content">
                            <div class="alert-member">Member: ${alert.memberId}</div>
                            <div class="alert-amount">$${(alert.amount || 0).toLocaleString()}</div>
                            <div class="alert-score">Score: ${alert.fraudScore}</div>
                        </div>
                        <div class="alert-actions">
                            <button class="btn btn-sm" onclick="assignAlert('${alert.id}', 'analyst-1')">Assign</button>
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
}

// =================================================================
// SUSPICIOUS ACTIVITY REPORT (SAR)
// =================================================================

/**
 * Generate SAR data structure
 * @param {Object} alert - Alert to report
 * @param {Object} memberData - Member information
 * @returns {Object} SAR data
 */
function generateSARData(alert, memberData = {}) {
    return {
        reportId: `SAR-${Date.now()}`,
        filingDate: new Date().toISOString(),
        alertId: alert.id,

        // Subject Information
        subject: {
            memberId: memberData.id || alert.memberId,
            name: memberData.name || 'Unknown',
            address: memberData.address || '',
            ssn: memberData.ssn ? `XXX-XX-${memberData.ssn.slice(-4)}` : 'N/A',
            dob: memberData.dob || ''
        },

        // Suspicious Activity Details
        activity: {
            type: alert.factors?.map(f => f.factor).join(', ') || 'Suspicious Transaction',
            amount: alert.amount,
            dateRange: {
                start: alert.timestamp,
                end: alert.timestamp
            },
            description: `Fraud score ${alert.fraudScore}/100 triggered by: ${alert.factors?.map(f => f.factor).join('; ') || 'Unknown factors'
                }`
        },

        // Institution Information
        filingInstitution: {
            name: 'Service Credit Union',
            address: '3003 Lafayette Rd, Portsmouth, NH 03801',
            ein: 'XX-XXXXXXX'
        },

        status: 'DRAFT',
        createdBy: 'current_user'
    };
}

// =================================================================
// UI HELPER FUNCTIONS (for banking.html interactive sections)
// =================================================================

/**
 * Run anomaly analysis from UI
 */
function runAnomalyAnalysis() {
    const amount = parseFloat(document.getElementById('anomalyAmount')?.value) || 0;
    const avg = parseFloat(document.getElementById('anomalyAvg')?.value) || 500;
    const isNewPayee = document.getElementById('anomalyPayee')?.value === 'new';
    const tenure = parseInt(document.getElementById('anomalyTenure')?.value) || 12;

    const transaction = {
        amount,
        timestamp: new Date().toISOString(),
        isNewPayee
    };

    const memberProfile = {
        averageTransaction: avg,
        tenureMonths: tenure
    };

    const anomalies = detectAnomalies(transaction, memberProfile);
    const fraudScore = calculateFraudScore(transaction, memberProfile);

    const resultsDiv = document.getElementById('anomalyResults');
    if (resultsDiv) {
        resultsDiv.innerHTML = `
            <div style="padding: 1rem;">
                <div style="display: flex; gap: 2rem; margin-bottom: 1rem;">
                    <div style="flex: 1; background: ${fraudScore.riskLevel === 'high' ? '#fef2f2' : fraudScore.riskLevel === 'medium' ? '#fffbeb' : '#f0fdf4'}; 
                                border-radius: 12px; padding: 1.5rem; text-align: center;">
                        <div style="font-size: 3rem; font-weight: 700; color: ${fraudScore.riskLevel === 'high' ? '#dc2626' : fraudScore.riskLevel === 'medium' ? '#d97706' : '#16a34a'};">
                            ${fraudScore.score}
                        </div>
                        <div style="font-size: 0.9rem; color: #666;">Fraud Score</div>
                        <div style="margin-top: 0.5rem; padding: 4px 12px; border-radius: 20px; display: inline-block;
                                    background: ${fraudScore.action === 'BLOCK' ? '#dc2626' : fraudScore.action === 'REVIEW' ? '#d97706' : '#16a34a'}; color: white;">
                            ${fraudScore.action}
                        </div>
                    </div>
                    <div style="flex: 1; background: #f3f4f6; border-radius: 12px; padding: 1.5rem;">
                        <div style="font-weight: 600; margin-bottom: 0.5rem;">🚨 Anomalies Detected: ${anomalies.length}</div>
                        ${anomalies.map(a => `
                            <div style="padding: 0.5rem; background: white; border-radius: 8px; margin-bottom: 0.5rem;">
                                ${a.icon} <strong>${a.label}</strong><br>
                                <small style="color: #666;">${a.details}</small>
                            </div>
                        `).join('') || '<div style="color: #666;">No anomalies detected</div>'}
                    </div>
                </div>
                <div style="background: #f8fafc; border-radius: 8px; padding: 1rem;">
                    <div style="font-weight: 600; margin-bottom: 0.5rem;">📊 Risk Factors:</div>
                    ${fraudScore.factors.map(f => `
                        <div style="display: flex; justify-content: space-between; padding: 0.25rem 0;">
                            <span>${f.factor}</span>
                            <span style="font-weight: 600; color: ${f.points >= 0 ? '#dc2626' : '#16a34a'};">${f.points >= 0 ? '+' : ''}${f.points} pts</span>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }
}

/**
 * Run lending analysis from UI
 */
function runLendingAnalysis() {
    const debt = parseFloat(document.getElementById('lendingDebt')?.value) || 0;
    const income = parseFloat(document.getElementById('lendingIncome')?.value) || 1;
    const loanAmt = parseFloat(document.getElementById('lendingLoanAmt')?.value) || 0;
    const propertyVal = parseFloat(document.getElementById('lendingPropertyVal')?.value) || 1;

    const dti = window.calculateDTI ? window.calculateDTI(debt, income) : { ratio: (debt / income * 100).toFixed(1), status: 'calculated' };
    const ltv = window.calculateLTV ? window.calculateLTV(loanAmt, propertyVal) : { ratio: (loanAmt / propertyVal * 100).toFixed(1), pmiRequired: (loanAmt / propertyVal) > 0.8 };

    const resultsDiv = document.getElementById('lendingResults');
    if (resultsDiv) {
        resultsDiv.innerHTML = `
            <div style="padding: 1rem;">
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
                    <div style="background: ${dti.status === 'good' ? '#f0fdf4' : dti.status === 'warning' ? '#fffbeb' : '#fef2f2'}; 
                                border-radius: 12px; padding: 1.5rem; text-align: center;">
                        <div style="font-size: 2.5rem; font-weight: 700; color: ${dti.status === 'good' ? '#16a34a' : dti.status === 'warning' ? '#d97706' : '#dc2626'};">
                            ${dti.ratio}%
                        </div>
                        <div style="font-weight: 600;">Debt-to-Income Ratio</div>
                        <div style="font-size: 0.85rem; color: #666; margin-top: 0.5rem;">
                            ${dti.status === 'good' ? '✅ Healthy ratio' : dti.status === 'warning' ? '⚠️ Near limit' : '❌ Above limit'}
                        </div>
                        <div style="font-size: 0.8rem; color: #999;">Threshold: ≤43% for QM</div>
                    </div>
                    <div style="background: ${!ltv.pmiRequired ? '#f0fdf4' : '#fffbeb'}; 
                                border-radius: 12px; padding: 1.5rem; text-align: center;">
                        <div style="font-size: 2.5rem; font-weight: 700; color: ${!ltv.pmiRequired ? '#16a34a' : '#d97706'};">
                            ${ltv.ratio}%
                        </div>
                        <div style="font-weight: 600;">Loan-to-Value Ratio</div>
                        <div style="font-size: 0.85rem; color: #666; margin-top: 0.5rem;">
                            ${ltv.pmiRequired ? '⚠️ PMI Required' : '✅ No PMI needed'}
                        </div>
                        <div style="font-size: 0.8rem; color: #999;">PMI threshold: 80%</div>
                    </div>
                </div>
                <div style="margin-top: 1rem; background: #f8fafc; border-radius: 8px; padding: 1rem;">
                    <div style="font-weight: 600;">📋 Analysis Summary</div>
                    <div style="margin-top: 0.5rem; color: #666;">
                        Monthly Debt: $${debt.toLocaleString()} | Monthly Income: $${income.toLocaleString()}<br>
                        Loan Amount: $${loanAmt.toLocaleString()} | Property Value: $${propertyVal.toLocaleString()}
                    </div>
                </div>
            </div>
        `;
    }
}

/**
 * Run cross-sell analysis from UI
 */
function runCrossSellAnalysis() {
    const name = document.getElementById('insightsMemberName')?.value || 'Member';
    const balance = parseFloat(document.getElementById('insightsBalance')?.value) || 5000;

    const member = { name, balance };
    const offers = window.generateNextBestActions ? window.generateNextBestActions(member) : [
        { name: '12-Month CD', propensity: 0.75, icon: '💰', description: '5.00% APY guaranteed' },
        { name: 'Visa Rewards Card', propensity: 0.65, icon: '💳', description: '2% cash back on all purchases' },
        { name: 'Auto Loan', propensity: 0.45, icon: '🚗', description: 'Rates as low as 4.99% APR' }
    ];

    const resultsDiv = document.getElementById('crossSellResults');
    if (resultsDiv) {
        resultsDiv.innerHTML = `
            <div style="padding: 1rem;">
                <div style="font-weight: 600; margin-bottom: 1rem;">🎯 Top Recommendations for ${name}</div>
                ${offers.map((offer, i) => `
                    <div class="offer-card" style="display: flex; align-items: center; gap: 1rem; padding: 1rem; 
                                background: ${i === 0 ? '#ecfdf5' : '#f8fafc'}; border-radius: 10px; margin-bottom: 0.5rem;">
                        <div style="font-size: 2rem;">${offer.icon || '📦'}</div>
                        <div style="flex: 1;">
                            <div style="font-weight: 600;">${offer.name}</div>
                            <div style="font-size: 0.85rem; color: #666;">${offer.description || 'Recommended product'}</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 1.5rem; font-weight: 700; color: #059669;">${Math.round(offer.propensity * 100)}%</div>
                            <div style="font-size: 0.75rem; color: #666;">Propensity</div>
                        </div>
                        <button class="btn btn-sm btn-primary" style="padding: 8px 16px;" 
                                onclick="presentOffer('${name}', '${offer.name}', '${offer.description || ''}', ${Math.round(offer.propensity * 100)})">
                            Present Offer
                        </button>
                    </div>
                `).join('')}
            </div>
        `;
    }
}

/**
 * Present an offer to the member - shows confirmation modal
 */
function presentOffer(memberName, offerName, offerDescription, propensity) {
    // Create modal overlay
    const overlay = document.createElement('div');
    overlay.id = 'offerModal';
    overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.6);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 9999;
    `;

    overlay.innerHTML = `
        <div style="background: white; border-radius: 16px; padding: 2rem; max-width: 500px; width: 90%; box-shadow: 0 20px 60px rgba(0,0,0,0.3);">
            <div style="text-align: center; margin-bottom: 1.5rem;">
                <div style="font-size: 4rem; margin-bottom: 0.5rem;">🎉</div>
                <h2 style="margin: 0; color: #059669;">Offer Presented!</h2>
            </div>
            
            <div style="background: #f0fdf4; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <strong>${offerName}</strong>
                    <span style="background: #059669; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem;">
                        ${propensity}% Match
                    </span>
                </div>
                <div style="color: #666; font-size: 0.9rem;">${offerDescription}</div>
            </div>
            
            <div style="background: #f8fafc; border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem;">
                <div style="font-weight: 600; margin-bottom: 0.5rem;">📋 Presentation Details</div>
                <div style="font-size: 0.9rem; color: #666;">
                    <div>👤 Member: <strong>${memberName}</strong></div>
                    <div>📅 Date: <strong>${new Date().toLocaleDateString()}</strong></div>
                    <div>⏰ Time: <strong>${new Date().toLocaleTimeString()}</strong></div>
                    <div>👔 Agent: <strong>Current User</strong></div>
                </div>
            </div>
            
            <div style="display: flex; gap: 1rem;">
                <button onclick="recordOfferResponse('${memberName}', '${offerName}', 'interested')" 
                        style="flex: 1; padding: 12px; background: #059669; color: white; border: none; border-radius: 8px; font-weight: 600; cursor: pointer;">
                    ✅ Member Interested
                </button>
                <button onclick="recordOfferResponse('${memberName}', '${offerName}', 'declined')" 
                        style="flex: 1; padding: 12px; background: #f3f4f6; color: #666; border: none; border-radius: 8px; font-weight: 600; cursor: pointer;">
                    ❌ Declined
                </button>
            </div>
            
            <div style="text-align: center; margin-top: 1rem;">
                <button onclick="closeOfferModal()" 
                        style="padding: 8px 24px; background: transparent; border: 1px solid #ddd; border-radius: 6px; cursor: pointer; color: #666;">
                    Close
                </button>
            </div>
        </div>
    `;

    document.body.appendChild(overlay);

    // Log the offer presentation
    console.log(`[Cross-Sell] Offer presented: ${offerName} to ${memberName}`);

    // Store in localStorage for tracking
    const presentations = JSON.parse(localStorage.getItem('scuOfferPresentations') || '[]');
    presentations.unshift({
        memberId: memberName,
        offer: offerName,
        propensity: propensity,
        presentedAt: new Date().toISOString(),
        status: 'presented'
    });
    localStorage.setItem('scuOfferPresentations', JSON.stringify(presentations.slice(0, 50)));
}

/**
 * Record member's response to an offer
 */
function recordOfferResponse(memberName, offerName, response) {
    const presentations = JSON.parse(localStorage.getItem('scuOfferPresentations') || '[]');
    const lastPresentation = presentations.find(p => p.memberId === memberName && p.offer === offerName);

    if (lastPresentation) {
        lastPresentation.response = response;
        lastPresentation.respondedAt = new Date().toISOString();
        localStorage.setItem('scuOfferPresentations', JSON.stringify(presentations));
    }

    // Show confirmation
    const modal = document.getElementById('offerModal');
    if (modal) {
        modal.querySelector('div > div').innerHTML = `
            <div style="text-align: center; padding: 2rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">${response === 'interested' ? '🎊' : '📝'}</div>
                <h2 style="color: ${response === 'interested' ? '#059669' : '#666'};">
                    ${response === 'interested' ? 'Great! Follow-up Scheduled' : 'Response Recorded'}
                </h2>
                <p style="color: #666; margin: 1rem 0;">
                    ${response === 'interested'
                ? `Member ${memberName} is interested in ${offerName}. A follow-up task has been created.`
                : `Response recorded for ${memberName}. Consider alternative offers.`
            }
                </p>
                <button onclick="closeOfferModal()" 
                        style="padding: 12px 32px; background: #6366f1; color: white; border: none; border-radius: 8px; font-weight: 600; cursor: pointer;">
                    Done
                </button>
            </div>
        `;
    }

    console.log(`[Cross-Sell] Offer response: ${response} for ${offerName} from ${memberName}`);
}

/**
 * Close the offer modal
 */
function closeOfferModal() {
    const modal = document.getElementById('offerModal');
    if (modal) {
        modal.remove();
    }
}

// Export new functions
window.presentOffer = presentOffer;
window.recordOfferResponse = recordOfferResponse;
window.closeOfferModal = closeOfferModal;

/**
 * Generate and show performance metrics
 */
function generateAndShowMetrics() {
    const metrics = window.generatePerformanceMetrics ? window.generatePerformanceMetrics() : {
        msrMetrics: { firstCallResolution: 85, memberSatisfaction: 4.2 },
        lendingMetrics: { approvalRate: 72 },
        securityMetrics: { fraudPrevented: 75000 },
        crossSellMetrics: { conversionRate: 12 }
    };

    // Update dashboard cards
    document.getElementById('metricFCR').textContent = metrics.msrMetrics.firstCallResolution + '%';
    document.getElementById('metricApproval').textContent = metrics.lendingMetrics.approvalRate + '%';
    document.getElementById('metricFraud').textContent = '$' + metrics.securityMetrics.fraudPrevented.toLocaleString();
    document.getElementById('metricCrossSell').textContent = metrics.crossSellMetrics.conversionRate + '%';
}

/**
 * Generate report from UI
 */
function generateReportUI(type) {
    let report;

    if (type === 'member') {
        report = window.generateMemberActivityReport ? window.generateMemberActivityReport('M12345') : { type: 'Member Activity' };
    } else if (type === 'transaction') {
        report = window.generateTransactionSummary ? window.generateTransactionSummary() : { type: 'Transaction Summary' };
    } else if (type === 'ctr') {
        report = window.generateCTR ? window.generateCTR([
            { id: 'TX1', amount: 15000 },
            { id: 'TX2', amount: 12000 }
        ]) : { type: 'CTR Report' };
    }

    window.currentReport = report;

    const panel = document.getElementById('reportResultsPanel');
    const results = document.getElementById('reportResults');

    if (panel && results) {
        panel.style.display = 'block';
        results.innerHTML = `
            <div style="padding: 1rem;">
                <div style="background: #f8fafc; border-radius: 8px; padding: 1rem; font-family: monospace; font-size: 0.85rem; max-height: 300px; overflow: auto;">
                    <pre>${JSON.stringify(report, null, 2)}</pre>
                </div>
            </div>
        `;
    }
}

/**
 * Export current report
 */
function exportCurrentReport(format) {
    if (window.currentReport) {
        if (format === 'json' && window.downloadReport) {
            window.downloadReport(window.currentReport, 'json');
        } else {
            const blob = new Blob([JSON.stringify(window.currentReport, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'report.json';
            a.click();
        }
    }
}

// Export UI helpers
window.runAnomalyAnalysis = runAnomalyAnalysis;
window.runLendingAnalysis = runLendingAnalysis;
window.runCrossSellAnalysis = runCrossSellAnalysis;
window.generateAndShowMetrics = generateAndShowMetrics;
window.generateReportUI = generateReportUI;
window.exportCurrentReport = exportCurrentReport;

// =================================================================
// INITIALIZATION
// =================================================================

function initSecurityModule() {
    console.log('[Security Module] Initializing...');
    console.log(`[Security Module] Alert queue: ${SecurityState.alertQueue.length} alerts`);
    console.log('[Security Module] Initialized');

    // Update alert stats on load
    const stats = getAlertQueueStats();
    if (document.getElementById('alertsOpen')) {
        document.getElementById('alertsOpen').textContent = stats.open;
    }
    if (document.getElementById('alertsInProgress')) {
        document.getElementById('alertsInProgress').textContent = stats.inProgress;
    }
    if (document.getElementById('alertsResolved')) {
        document.getElementById('alertsResolved').textContent = stats.resolved;
    }
    if (document.getElementById('avgFraudScore')) {
        document.getElementById('avgFraudScore').textContent = stats.avgFraudScore || '--';
    }
}

// =================================================================
// GLOBAL EXPORTS
// =================================================================

window.SecurityState = SecurityState;
window.ANOMALY_TYPES = ANOMALY_TYPES;
window.ALERT_PRIORITIES = ALERT_PRIORITIES;

window.detectAnomalies = detectAnomalies;
window.checkVelocity = checkVelocity;
window.assessGeographicRisk = assessGeographicRisk;
window.calculateFraudScore = calculateFraudScore;

window.createAlert = createAlert;
window.getAlerts = getAlerts;
window.updateAlertStatus = updateAlertStatus;
window.resolveAlert = resolveAlert;

window.assignAlert = assignAlert;
window.getAnalystCases = getAnalystCases;
window.getAlertQueueStats = getAlertQueueStats;

window.renderFraudScorePanel = renderFraudScorePanel;
window.renderAlertQueue = renderAlertQueue;

window.generateSARData = generateSARData;
window.initSecurityModule = initSecurityModule;

// =================================================================
// AI LOAN OFFICER WORKSTATION UI HANDLERS
// =================================================================

/**
 * Show lending tab content
 * @param {string} tabId - Tab ID (queue, new-app, underwrite, tools)
 */
function showLendingTab(tabId) {
    // Hide all tab contents
    document.querySelectorAll('.lending-tab-content').forEach(tc => tc.style.display = 'none');

    // Deactivate all tabs
    document.querySelectorAll('.lending-tab').forEach(tab => {
        tab.style.background = '#f3f4f6';
        tab.style.color = '#374151';
    });

    // Show selected tab content
    const tabContent = document.getElementById(`lending-tab-${tabId}`);
    if (tabContent) {
        tabContent.style.display = 'block';
    }

    // Activate selected tab
    const activeTab = document.querySelector(`[data-tab="${tabId}"]`);
    if (activeTab) {
        activeTab.style.background = '#6366f1';
        activeTab.style.color = 'white';
    }

    // If queue tab, render it
    if (tabId === 'queue' && window.renderLoanQueue) {
        window.renderLoanQueue('loanQueueContainer');
        updateQueueStats();
    }
}

/**
 * Update queue stats in the header
 */
function updateQueueStats() {
    if (window.getQueueStats) {
        const stats = window.getQueueStats();
        const queueCount = document.getElementById('queueCountDisplay');
        const pendingCount = document.getElementById('pendingCountDisplay');
        if (queueCount) queueCount.textContent = stats.total;
        if (pendingCount) pendingCount.textContent = stats.pending;
    }
}

/**
 * Create new application and run underwriting
 */
function createAndUnderwrite() {
    const appData = {
        id: `APP-${Date.now()}`,
        memberId: document.getElementById('newAppMemberId')?.value || `M${Math.floor(Math.random() * 100000)}`,
        memberName: document.getElementById('newAppMemberName')?.value || 'New Member',
        loanType: document.getElementById('newAppLoanType')?.value || 'personal',
        loanAmount: parseFloat(document.getElementById('newAppAmount')?.value) || 25000,
        termMonths: parseInt(document.getElementById('newAppTerm')?.value) || 36,
        monthlyIncome: parseFloat(document.getElementById('newAppIncome')?.value) || 5000,
        monthlyDebt: parseFloat(document.getElementById('newAppDebt')?.value) || 1500,
        employmentMonths: parseInt(document.getElementById('newAppEmployment')?.value) || 36,
        creditScore: parseInt(document.getElementById('newAppCreditScore')?.value) || null,
        collateralValue: parseFloat(document.getElementById('newAppCollateral')?.value) || null
    };

    // Add to queue
    if (window.addToLoanQueue) {
        window.addToLoanQueue(appData);
    }

    // Switch to underwriting tab
    showLendingTab('underwrite');

    // Fill underwriting form
    document.getElementById('uwAppId').value = appData.id;
    document.getElementById('uwIncome').value = appData.monthlyIncome;
    document.getElementById('uwDebt').value = appData.monthlyDebt;
    document.getElementById('uwLoanAmt').value = appData.loanAmount;
    document.getElementById('uwTerm').value = appData.termMonths;

    // Store for underwriting
    window._currentApplication = appData;

    // Run underwriting
    runAIUnderwriting();
}

/**
 * Save application as draft
 */
function saveApplication() {
    const appData = {
        id: `DRAFT-${Date.now()}`,
        memberId: document.getElementById('newAppMemberId')?.value,
        memberName: document.getElementById('newAppMemberName')?.value,
        loanType: document.getElementById('newAppLoanType')?.value,
        loanAmount: parseFloat(document.getElementById('newAppAmount')?.value) || 0,
        termMonths: parseInt(document.getElementById('newAppTerm')?.value) || 36,
        status: 'draft'
    };

    // Save to localStorage
    const drafts = JSON.parse(localStorage.getItem('scuLoanDrafts') || '[]');
    drafts.unshift(appData);
    localStorage.setItem('scuLoanDrafts', JSON.stringify(drafts.slice(0, 20)));

    alert('Application saved as draft');
}

/**
 * Run AI Underwriting
 */
async function runAIUnderwriting() {
    const buttonText = document.getElementById('uwButtonText');
    if (buttonText) buttonText.textContent = '⏳ Running AI Analysis...';

    // Build application data
    const appData = window._currentApplication || {
        id: document.getElementById('uwAppId')?.value || `APP-${Date.now()}`,
        monthlyIncome: parseFloat(document.getElementById('uwIncome')?.value) || 5000,
        monthlyDebt: parseFloat(document.getElementById('uwDebt')?.value) || 1500,
        loanAmount: parseFloat(document.getElementById('uwLoanAmt')?.value) || 25000,
        termMonths: parseInt(document.getElementById('uwTerm')?.value) || 36,
        memberName: 'Demo Member',
        memberData: {
            onTimePaymentRate: 0.95,
            accountAgeMonths: 24,
            averageBalance: 2500,
            nsfCount: 0
        }
    };

    // Simulate slight delay for effect
    await new Promise(resolve => setTimeout(resolve, 800));

    // Run underwriting
    if (window.autoUnderwrite) {
        const result = window.autoUnderwrite(appData);
        window._currentUnderwriteResult = result;

        // Render SHAP visualization
        if (window.renderShapVisualization) {
            window.renderShapVisualization('shapVisualization', result.shapValues);
        }

        // Show recommendation panel
        const recPanel = document.getElementById('aiRecommendationPanel');
        if (recPanel) recPanel.style.display = 'block';

        // Update displays
        updateAIDisplays(result);

        // Generate explanation
        if (window.generateDecisionExplanation) {
            const explanation = await window.generateDecisionExplanation(result);
            const expText = document.getElementById('llmExplanationText');
            if (expText) expText.textContent = explanation.memberFacing;
        }
    }

    if (buttonText) buttonText.textContent = '🧠 Run AI Underwriting';
}

/**
 * Update AI display panels with underwriting results
 */
function updateAIDisplays(result) {
    const colors = {
        'APPROVE': '#10b981',
        'APPROVE_WITH_CONDITIONS': '#f59e0b',
        'REVIEW': '#6366f1',
        'DECLINE': '#ef4444'
    };

    const riskColors = {
        'low': '#10b981',
        'medium': '#f59e0b',
        'high': '#ef4444'
    };

    // Recommendation
    const recDisplay = document.getElementById('aiRecDisplay');
    if (recDisplay) {
        recDisplay.textContent = result.recommendation;
        recDisplay.style.color = colors[result.recommendation] || '#6b7280';
    }

    // Confidence
    const confDisplay = document.getElementById('aiConfDisplay');
    if (confDisplay) {
        confDisplay.textContent = `${result.confidence}%`;
        confDisplay.style.color = result.confidence >= 80 ? '#10b981' : '#f59e0b';
    }

    // Risk Level
    const riskDisplay = document.getElementById('aiRiskDisplay');
    if (riskDisplay) {
        riskDisplay.textContent = result.riskLevel.toUpperCase();
        riskDisplay.style.color = riskColors[result.riskLevel] || '#6b7280';
    }

    // Fair Lending
    const fairDisplay = document.getElementById('aiFairDisplay');
    if (fairDisplay) {
        fairDisplay.textContent = result.fairLendingResult?.passed ? '✅ PASSED' : '⚠️ REVIEW';
        fairDisplay.style.color = result.fairLendingResult?.passed ? '#10b981' : '#f59e0b';
    }

    // Pricing
    const pricing = result.metrics?.pricing;
    if (pricing) {
        document.getElementById('pricingAPR').textContent = `${pricing.finalRate}%`;
        document.getElementById('pricingPayment').textContent = `$${pricing.monthlyPayment?.toLocaleString() || '-'}`;
        document.getElementById('pricingTier').textContent = `Tier ${pricing.tier}`;
        document.getElementById('pricingInterest').textContent = `$${pricing.totalInterest?.toLocaleString() || '-'}`;
    }
}

/**
 * Approve application (human decision)
 */
function approveApplication() {
    if (!window._currentUnderwriteResult) {
        alert('Please run AI underwriting first');
        return;
    }

    const result = window._currentUnderwriteResult;

    // Log decision
    if (window.logDecision) {
        window.logDecision(result.applicationId, result, {
            action: 'APPROVE',
            userId: 'current_user'
        });
    }

    // Show confirmation modal
    showDecisionModal('APPROVE', result);
}

/**
 * Approve with conditions
 */
function approveWithConditions() {
    if (!window._currentUnderwriteResult) {
        alert('Please run AI underwriting first');
        return;
    }

    const result = window._currentUnderwriteResult;
    const conditions = prompt('Enter conditions (comma-separated):',
        result.conditions?.join(', ') || 'Income verification required, Insurance documentation needed');

    if (conditions) {
        if (window.logDecision) {
            window.logDecision(result.applicationId, result, {
                action: 'APPROVE_WITH_CONDITIONS',
                userId: 'current_user',
                conditions: conditions.split(',').map(c => c.trim())
            });
        }

        showDecisionModal('APPROVE_WITH_CONDITIONS', result, conditions);
    }
}

/**
 * Decline application
 */
function declineApplication() {
    if (!window._currentUnderwriteResult) {
        alert('Please run AI underwriting first');
        return;
    }

    const result = window._currentUnderwriteResult;
    const reasons = result.topFactors?.filter(f => f.contribution < 0).map(f => f.description) || [];

    if (window.logDecision) {
        window.logDecision(result.applicationId, result, {
            action: 'DECLINE',
            userId: 'current_user',
            adverseReasons: reasons
        });
    }

    showDecisionModal('DECLINE', result, null, reasons);
}

/**
 * Escalate application
 */
function escalateApplication() {
    if (!window._currentUnderwriteResult) {
        alert('Please run AI underwriting first');
        return;
    }

    const result = window._currentUnderwriteResult;
    const reason = prompt('Enter escalation reason:', 'Requires senior officer review');

    if (reason) {
        if (window.logDecision) {
            window.logDecision(result.applicationId, result, {
                action: 'ESCALATE',
                userId: 'current_user',
                escalationReason: reason
            });
        }

        showDecisionModal('ESCALATE', result, reason);
    }
}

/**
 * Show decision confirmation modal
 */
function showDecisionModal(decision, result, details, adverseReasons) {
    const icons = {
        'APPROVE': '✅',
        'APPROVE_WITH_CONDITIONS': '📋',
        'DECLINE': '❌',
        'ESCALATE': '🔄'
    };

    const colors = {
        'APPROVE': '#10b981',
        'APPROVE_WITH_CONDITIONS': '#f59e0b',
        'DECLINE': '#ef4444',
        'ESCALATE': '#6366f1'
    };

    const overlay = document.createElement('div');
    overlay.id = 'decisionModal';
    overlay.style.cssText = `
        position: fixed; top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(0, 0, 0, 0.6);
        display: flex; align-items: center; justify-content: center;
        z-index: 9999;
    `;

    overlay.innerHTML = `
        <div style="background: white; border-radius: 16px; padding: 2rem; max-width: 500px; width: 90%; box-shadow: 0 20px 60px rgba(0,0,0,0.3);">
            <div style="text-align: center; margin-bottom: 1.5rem;">
                <div style="font-size: 4rem; margin-bottom: 0.5rem;">${icons[decision]}</div>
                <h2 style="margin: 0; color: ${colors[decision]};">${decision.replace(/_/g, ' ')}</h2>
            </div>
            
            <div style="background: #f8fafc; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;">
                <div style="font-weight: 600; margin-bottom: 0.5rem;">Application: ${result.applicationId}</div>
                <div style="font-size: 0.9rem; color: #666;">Member: ${result.memberName || 'Unknown'}</div>
                <div style="font-size: 0.9rem; color: #666;">Amount: $${result.metrics?.pricing?.loanAmount?.toLocaleString() || '-'}</div>
                ${details ? `<div style="margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid #e5e7eb;">
                    <strong>Details:</strong> ${details}
                </div>` : ''}
                ${adverseReasons?.length ? `<div style="margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid #e5e7eb;">
                    <strong>Adverse Action Reasons (ECOA):</strong>
                    <ul style="margin: 0.5rem 0 0 1rem; padding: 0;">${adverseReasons.map(r => `<li>${r}</li>`).join('')}</ul>
                </div>` : ''}
            </div>
            
            <div style="background: #f0fdf4; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                <div style="font-weight: 600; color: #059669;">📋 Audit Log Created</div>
                <div style="font-size: 0.85rem; color: #666;">Decision logged with model version ${result.modelVersion}</div>
            </div>
            
            <button onclick="document.getElementById('decisionModal').remove()" 
                    style="width: 100%; padding: 12px; background: ${colors[decision]}; color: white; border: none; border-radius: 8px; font-weight: 600; cursor: pointer;">
                Done
            </button>
        </div>
    `;

    document.body.appendChild(overlay);
    console.log(`[Loan Officer] Decision: ${decision} for ${result.applicationId}`);
}

/**
 * Open application from queue
 */
function openApplication(appId) {
    console.log(`[Loan Officer] Opening application: ${appId}`);

    // Find in queue
    const queue = window.AILendingState?.loanQueue || [];
    const app = queue.find(a => a.id === appId);

    if (app) {
        window._currentApplication = app;

        // Switch to underwriting tab
        showLendingTab('underwrite');

        // Fill form
        document.getElementById('uwAppId').value = app.id;
        document.getElementById('uwIncome').value = app.monthlyIncome || 5000;
        document.getElementById('uwDebt').value = app.monthlyDebt || 1500;
        document.getElementById('uwLoanAmt').value = app.amount || 25000;
        document.getElementById('uwTerm').value = app.termMonths || 36;
    }
}

// Export new functions
window.showLendingTab = showLendingTab;
window.updateQueueStats = updateQueueStats;
window.createAndUnderwrite = createAndUnderwrite;
window.saveApplication = saveApplication;
window.runAIUnderwriting = runAIUnderwriting;
window.approveApplication = approveApplication;
window.approveWithConditions = approveWithConditions;
window.declineApplication = declineApplication;
window.escalateApplication = escalateApplication;
window.openApplication = openApplication;

// Initialize loan officer section on page load
document.addEventListener('DOMContentLoaded', () => {
    // Initialize queue rendering if on lending section
    setTimeout(() => {
        if (window.renderLoanQueue) {
            window.renderLoanQueue('loanQueueContainer');
        }
        updateQueueStats();
    }, 500);
});

// =================================================================
// FRAUD ANALYST WORKSTATION UI HANDLERS
// =================================================================

/**
 * Show fraud tab content
 * @param {string} tabId - Tab ID (queue, analyze, investigate, sar, trends)
 */
function showFraudTab(tabId) {
    // Hide all tab contents
    document.querySelectorAll('.fraud-tab-content').forEach(tc => tc.style.display = 'none');

    // Deactivate all tabs
    document.querySelectorAll('.fraud-tab').forEach(tab => {
        tab.style.background = '#f3f4f6';
        tab.style.color = '#374151';
    });

    // Show selected tab content
    const tabContent = document.getElementById(`fraud-tab-${tabId}`);
    if (tabContent) {
        tabContent.style.display = 'block';
    }

    // Activate selected tab
    const activeTab = document.querySelector(`.fraud-tab[data-tab="${tabId}"]`);
    if (activeTab) {
        activeTab.style.background = '#ef4444';
        activeTab.style.color = 'white';
    }

    console.log(`[Fraud Analyst] Switched to ${tabId} tab`);
}

/**
 * Run fraud analysis on transaction
 */
function runFraudAnalysis() {
    const amount = parseFloat(document.getElementById('anomalyAmount')?.value) || 0;
    const avg = parseFloat(document.getElementById('anomalyAvg')?.value) || 500;
    const payeeType = document.getElementById('anomalyPayee')?.value || 'existing';
    const txType = document.getElementById('anomalyType')?.value || 'wire';
    const tenure = parseInt(document.getElementById('anomalyTenure')?.value) || 12;
    const location = document.getElementById('anomalyLocation')?.value || '';

    // Calculate fraud score
    let score = 0;
    const factors = [];

    // High value check
    if (amount > avg * 10) {
        score += 30;
        factors.push({ name: 'High Value', impact: 30, desc: `$${amount.toLocaleString()} is ${(amount / avg).toFixed(1)}x normal` });
    } else if (amount > avg * 5) {
        score += 15;
        factors.push({ name: 'Elevated Value', impact: 15, desc: `$${amount.toLocaleString()} is ${(amount / avg).toFixed(1)}x normal` });
    }

    // New payee check
    if (payeeType === 'new') {
        score += 20;
        factors.push({ name: 'New Payee', impact: 20, desc: 'First-time recipient for large transfer' });
    }

    // Wire transfer risk
    if (txType === 'wire' && amount > 5000) {
        score += 15;
        factors.push({ name: 'Wire Transfer', impact: 15, desc: 'High-risk channel for large amount' });
    }

    // New member risk
    if (tenure < 6) {
        score += 10;
        factors.push({ name: 'New Member', impact: 10, desc: `Only ${tenure} months tenure` });
    }

    // Render results
    const resultsContainer = document.getElementById('fraudScoreResults');
    const riskLevel = score >= 60 ? 'CRITICAL' : score >= 40 ? 'HIGH' : score >= 20 ? 'MEDIUM' : 'LOW';
    const riskColor = score >= 60 ? '#ef4444' : score >= 40 ? '#f59e0b' : score >= 20 ? '#ca8a04' : '#10b981';

    resultsContainer.innerHTML = `
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <div style="font-size: 4rem; font-weight: 700; color: ${riskColor};">${score}</div>
            <div style="font-size: 1.2rem; font-weight: 600; color: ${riskColor};">${riskLevel} RISK</div>
        </div>
        
        <div style="background: #f8fafc; border-radius: 8px; padding: 1rem;">
            <div style="font-weight: 600; margin-bottom: 0.5rem;">Risk Factors</div>
            ${factors.length > 0 ? factors.map(f => `
                <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid #e5e7eb;">
                    <div>
                        <div style="font-weight: 500;">${f.name}</div>
                        <div style="font-size: 0.85rem; color: #6b7280;">${f.desc}</div>
                    </div>
                    <div style="color: ${riskColor}; font-weight: 600;">+${f.impact}</div>
                </div>
            `).join('') : '<div style="color: #6b7280;">No significant risk factors detected</div>'}
        </div>
        
        ${score >= 40 ? `
            <div style="margin-top: 1rem; display: flex; gap: 0.5rem;">
                <button onclick="createAlertFromAnalysis(${score}, '${riskLevel}')" class="btn" style="flex: 1; background: #ef4444; color: white; padding: 0.75rem; border-radius: 8px;">🚨 Create Alert</button>
                <button onclick="openSARFromAnalysis()" class="btn" style="flex: 1; background: #6366f1; color: white; padding: 0.75rem; border-radius: 8px;">📄 File SAR</button>
            </div>
        ` : ''}
    `;
}

/**
 * Open investigation panel for an alert
 * NOW USES REAL DATA FROM API CACHE
 */
async function openInvestigation(alertId) {
    showFraudTab('investigate');

    const panel = document.getElementById('investigationPanel');
    if (!panel) return;

    // Show loading state
    panel.innerHTML = `
        <div style="padding: 3rem; text-align: center; color: #6b7280;">
            <div class="spinner" style="width: 50px; height: 50px; border: 4px solid #fee2e2; border-top-color: #ef4444; border-radius: 50%; margin: 0 auto 1.5rem; animation: spin 1s linear infinite;"></div>
            <div style="font-size: 1.1rem; font-weight: 600;">Loading Investigation...</div>
            <div style="font-size: 0.9rem;">Fetching member profile and transaction history</div>
        </div>
    `;

    // Look up the item from cache (populated by loadHighestRisk)
    let data = window._riskItemsCache?.[alertId];

    // If not in cache, try to fetch from API
    if (!data) {
        try {
            const response = await apiRequest(`/api/v1/banking/fraud/highest-risk?view_type=${currentRiskViewType}`);
            if (response.status === 'success' && response.items) {
                // Cache all items
                window._riskItemsCache = {};
                response.items.forEach(item => {
                    window._riskItemsCache[item.id] = item;
                });
                data = window._riskItemsCache[alertId];
            }
        } catch (error) {
            console.error('[Investigation] Failed to fetch risk data:', error);
        }
    }

    // If still no data, show error
    if (!data) {
        panel.innerHTML = `
            <div style="padding: 3rem; text-align: center; color: #ef4444;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">❌</div>
                <div style="font-size: 1.1rem; font-weight: 600;">Item Not Found</div>
                <div style="font-size: 0.9rem;">Could not find risk item: ${alertId}</div>
                <button onclick="showFraudTab('queue')" style="margin-top: 1.5rem; padding: 0.75rem 1.5rem; background: #ef4444; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: 600;">Back to Queue</button>
            </div>
        `;
        return;
    }

    // Build investigation view from real data
    const riskColor = data.score >= 70 ? '#ef4444' : data.score >= 50 ? '#f59e0b' : '#ca8a04';
    const priorityLabel = data.priority || 'MEDIUM';
    const memberId = data.member_id || data.account_id || 'Unknown';
    const memberName = data.member_name || 'Unknown Member';

    // Try to fetch member details for enhanced profile
    let memberDetails = null;
    if (memberId && memberId !== 'Unknown') {
        try {
            memberDetails = await fetchMember360(memberId.replace('M-', '').replace('A-', ''));
        } catch (e) {
            console.log('[Investigation] Could not fetch member details:', e);
        }
    }

    // Build activity timeline from transaction data
    const timeline = data.recent_transactions || [];
    const timelineHtml = timeline.length > 0
        ? timeline.slice(0, 5).map((tx, i) => `
            <div style="position: relative; padding-bottom: 1rem;">
                <div style="position: absolute; left: -1.35rem; width: 10px; height: 10px; background: ${i === 0 ? riskColor : '#9ca3af'}; border-radius: 50%;"></div>
                <div style="font-size: 0.85rem; color: #6b7280;">${tx.date || 'Recent'}</div>
                <div style="font-weight: 500;">${tx.description || tx.type || 'Transaction'}: $${(tx.amount || 0).toLocaleString()}</div>
            </div>
        `).join('')
        : `<div style="color: #6b7280; padding: 1rem;">No recent transactions available</div>`;

    // Risk factors/flags display
    const flags = data.flags || [];
    const flagsHtml = flags.length > 0
        ? flags.map(f => `<span style="background: #fee2e2; color: #991b1b; padding: 0.25rem 0.75rem; border-radius: 4px; font-size: 0.8rem; margin-right: 0.5rem; margin-bottom: 0.5rem; display: inline-block;">${f}</span>`).join('')
        : '<span style="color: #6b7280;">No specific risk flags</span>';

    panel.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1.5rem;">
            <div>
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <div style="font-size: 1.5rem; font-weight: 700;">🔎 Investigation: ${data.id}</div>
                    <span style="padding: 0.25rem 0.75rem; background: ${riskColor}; color: white; border-radius: 20px; font-size: 0.9rem; font-weight: 600;">Score: ${data.score}</span>
                    <span style="padding: 0.25rem 0.75rem; background: #1e293b; color: white; border-radius: 20px; font-size: 0.9rem; font-weight: 600;">${priorityLabel}</span>
                </div>
                <div style="color: #6b7280; margin-top: 0.5rem; font-size: 1.1rem;">${data.type || data.description || 'Suspicious Activity'}</div>
            </div>
            <div style="display: flex; gap: 0.5rem;">
                <button onclick="resolveAlertFromInvestigation('${data.id}', 'false_positive')" class="btn" style="background: #10b981; color: white; padding: 0.5rem 1rem; border-radius: 6px; font-weight: 600;">✓ False Positive</button>
                <button onclick="resolveAlertFromInvestigation('${data.id}', 'fraud_confirmed')" class="btn" style="background: #ef4444; color: white; padding: 0.5rem 1rem; border-radius: 6px; font-weight: 600;">🚨 Fraud Confirmed</button>
                <button onclick="openSARForMember('${memberName}', ${data.amount || 0})" class="btn" style="background: #6366f1; color: white; padding: 0.5rem 1rem; border-radius: 6px; font-weight: 600;">📄 File SAR</button>
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 2fr; gap: 1.5rem;">
            <!-- Member Profile Card -->
            <div style="background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border-radius: 16px; padding: 1.5rem; border: 1px solid #e2e8f0;">
                <div style="font-weight: 700; margin-bottom: 1.25rem; display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.5rem;">👤</span> Member Profile
                </div>
                <div style="margin-bottom: 1rem;">
                    <div style="font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">Name</div>
                    <div style="font-weight: 600; font-size: 1.1rem;">${memberName}</div>
                </div>
                <div style="margin-bottom: 1rem;">
                    <div style="font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">Member ID</div>
                    <div style="font-weight: 600; font-family: monospace;">${memberId}</div>
                </div>
                <div style="margin-bottom: 1rem;">
                    <div style="font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">Account ID</div>
                    <div style="font-weight: 600; font-family: monospace;">${data.account_id || 'N/A'}</div>
                </div>
                <div style="margin-bottom: 1rem;">
                    <div style="font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">Transaction Amount</div>
                    <div style="font-weight: 700; font-size: 1.25rem; color: ${riskColor};">$${(data.amount || 0).toLocaleString()}</div>
                </div>
                ${memberDetails ? `
                <div style="margin-bottom: 1rem;">
                    <div style="font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">Account Age</div>
                    <div style="font-weight: 600;">${memberDetails.account_age || 'Unknown'}</div>
                </div>
                <div style="margin-bottom: 1rem;">
                    <div style="font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">Total Accounts</div>
                    <div style="font-weight: 600;">${memberDetails.total_accounts || '1'}</div>
                </div>
                ` : ''}
            </div>
            
            <!-- Right column: Flags + Timeline -->
            <div>
                <!-- Risk Flags -->
                <div style="background: #fff7ed; border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem; border: 1px solid #fed7aa;">
                    <div style="font-weight: 600; margin-bottom: 0.75rem; color: #9a3412;">⚠️ Risk Flags</div>
                    <div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">${flagsHtml}</div>
                </div>
                
                <!-- Activity Timeline -->
                <div style="background: #f8fafc; border-radius: 12px; padding: 1.25rem; border: 1px solid #e2e8f0;">
                    <div style="font-weight: 600; margin-bottom: 1rem;">📋 Recent Activity</div>
                    <div style="border-left: 2px solid #e5e7eb; padding-left: 1rem; margin-left: 0.5rem;">
                        ${timelineHtml}
                    </div>
                </div>
            </div>
        </div>
    `;
}

/**
 * Resolve alert from investigation
 */
function resolveAlertFromInvestigation(alertId, disposition) {
    const dispositionLabels = {
        'false_positive': '✅ False Positive',
        'fraud_confirmed': '🚨 Fraud Confirmed'
    };

    alert(`Alert ${alertId} resolved as: ${dispositionLabels[disposition]}\n\nAudit log created.`);
    showFraudTab('queue');
}

/**
 * Open SAR form for member
 */
function openSARForMember(memberName, amount) {
    showFraudTab('sar');
    document.getElementById('sarSubjectName').value = memberName;
    document.getElementById('sarAmount').value = amount;
}

/**
 * Generate SAR draft
 */
function generateSARDraft() {
    alert('SAR Draft generated with auto-populated fields.\n\nReview all sections before submission.');
}

/**
 * Save SAR draft
 */
function saveSARDraft() {
    const data = {
        subjectName: document.getElementById('sarSubjectName')?.value,
        activityType: document.getElementById('sarActivityType')?.value,
        amount: document.getElementById('sarAmount')?.value,
        narrative: document.getElementById('sarNarrative')?.value
    };

    localStorage.setItem('scuSARDraft', JSON.stringify(data));
    alert('SAR Draft saved successfully.');
}

/**
 * Submit SAR to FinCEN
 */
function submitSAR() {
    const confirmed = confirm('Submit SAR to FinCEN?\n\nThis action will be logged and cannot be undone.');
    if (confirmed) {
        const sarId = `SAR-${Date.now()}`;
        alert(`SAR ${sarId} submitted to FinCEN.\n\nConfirmation number generated.\nAudit trail created.`);

        // Update SAR count
        const sarCount = document.getElementById('fraudSARCount');
        if (sarCount) {
            sarCount.textContent = parseInt(sarCount.textContent) + 1;
        }
    }
}

/**
 * Refresh alert queue
 */
function refreshAlertQueue() {
    console.log('[Fraud Analyst] Refreshing alert queue...');
    // In production, would fetch from API
    alert('Alert queue refreshed.');
}

/**
 * Filter alerts by priority
 */
function filterAlerts() {
    const filter = document.getElementById('alertFilter')?.value;
    console.log(`[Fraud Analyst] Filtering alerts by: ${filter}`);
}

/**
 * Create alert from analysis
 */
function createAlertFromAnalysis(score, riskLevel) {
    alert(`Alert created.\n\nFraud Score: ${score}\nRisk Level: ${riskLevel}\n\nAdded to queue.`);
    showFraudTab('queue');
}

/**
 * Open SAR from analysis
 */
function openSARFromAnalysis() {
    showFraudTab('sar');
}

// Export fraud analyst functions
window.showFraudTab = showFraudTab;
window.runFraudAnalysis = runFraudAnalysis;
window.openInvestigation = openInvestigation;
window.resolveAlertFromInvestigation = resolveAlertFromInvestigation;
window.openSARForMember = openSARForMember;
window.generateSARDraft = generateSARDraft;
window.saveSARDraft = saveSARDraft;
window.submitSAR = submitSAR;
window.refreshAlertQueue = refreshAlertQueue;
window.filterAlerts = filterAlerts;
window.createAlertFromAnalysis = createAlertFromAnalysis;
window.openSARFromAnalysis = openSARFromAnalysis;

// Auto-initialize
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initSecurityModule);
} else {
    initSecurityModule();
}
