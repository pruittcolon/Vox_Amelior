/**
 * Banking Cross-Sell Module - Phase 2 Productivity Features
 * ==========================================================
 * Next Best Action, Cross-Sell Propensity, At-Risk Flags, Referral Capture
 * 
 * Dependencies: banking_core.js, banking_member.js
 * Used by: banking.html
 * 
 * @module banking_crosssell
 * @version 1.0.0
 */

// =================================================================
// STATE MANAGEMENT
// =================================================================

/**
 * Cross-Sell state object
 * @type {Object}
 */
const CrossSellState = {
    currentOffers: [],
    referrals: JSON.parse(localStorage.getItem('scuReferrals') || '[]'),
    alertThresholds: JSON.parse(localStorage.getItem('scuAlertThresholds') || '{}'),
    riskFlags: []
};

// =================================================================
// NEXT BEST ACTION
// =================================================================

/**
 * Product offers database (simulated)
 */
const PRODUCT_OFFERS = [
    { id: 'auto_loan', name: 'Auto Loan', description: 'Rates as low as 4.99% APR', icon: '🚗', category: 'lending' },
    { id: 'credit_card', name: 'Visa Rewards Card', description: '2% cash back on all purchases', icon: '💳', category: 'cards' },
    { id: 'home_equity', name: 'Home Equity Line', description: 'Tap into your home equity', icon: '🏠', category: 'lending' },
    { id: 'savings_cd', name: '12-Month CD', description: '5.00% APY guaranteed', icon: '💰', category: 'savings' },
    { id: 'checking_plus', name: 'Checking+', description: 'Premium checking with rewards', icon: '✨', category: 'deposit' },
    { id: 'insurance', name: 'Life Insurance', description: 'Protect your family', icon: '🛡️', category: 'insurance' }
];

/**
 * Generate Next Best Action recommendations based on member data
 * @param {Object} member - Member data object
 * @returns {Array} Array of recommended offers
 */
function generateNextBestActions(member) {
    const recommendations = [];

    // Simple rule-based recommendations (would be ML in production)
    if (member) {
        // If member has checking but no savings CD
        recommendations.push({
            ...PRODUCT_OFFERS[3], // CD
            reason: 'Member has idle funds - suggest CD for better returns',
            propensity: 0.75
        });

        // Credit card offer
        recommendations.push({
            ...PRODUCT_OFFERS[1], // Credit Card
            reason: 'No credit card on file - cross-sell opportunity',
            propensity: 0.65
        });

        // Auto loan based on patterns
        recommendations.push({
            ...PRODUCT_OFFERS[0], // Auto Loan
            reason: 'Recent auto-related transactions detected',
            propensity: 0.55
        });
    }

    CrossSellState.currentOffers = recommendations;
    return recommendations;
}

/**
 * Render Next Best Action panel
 * @param {string} containerId - Target container ID
 * @param {Object} member - Member data
 */
function renderNextBestActionPanel(containerId, member) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const offers = generateNextBestActions(member);

    container.innerHTML = `
        <div class="nba-panel" data-testid="nba-panel">
            <div class="nba-header">
                <h3>🎯 Next Best Action</h3>
                <span class="nba-count">${offers.length} recommendations</span>
            </div>
            <div class="nba-offers">
                ${offers.map((offer, idx) => `
                    <div class="nba-offer" data-testid="nba-offer-${idx}" data-offer-id="${offer.id}">
                        <div class="nba-offer-icon">${offer.icon}</div>
                        <div class="nba-offer-content">
                            <div class="nba-offer-name">${escapeHtml(offer.name)}</div>
                            <div class="nba-offer-desc">${escapeHtml(offer.description)}</div>
                            <div class="nba-offer-reason">${escapeHtml(offer.reason)}</div>
                        </div>
                        <div class="nba-offer-actions">
                            <div class="propensity-badge">${Math.round(offer.propensity * 100)}%</div>
                            <button class="btn btn-sm btn-primary" onclick="presentOffer('${offer.id}')">Present</button>
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
}

/**
 * Present an offer to member and log it
 * @param {string} offerId - Offer ID
 */
function presentOffer(offerId) {
    const offer = PRODUCT_OFFERS.find(o => o.id === offerId);
    if (!offer) return;

    // Show referral capture modal
    showReferralCapture(offerId, offer.name);

    console.log(`[CrossSell] Presented offer: ${offer.name}`);
}

// =================================================================
// REFERRAL CAPTURE
// =================================================================

/**
 * Show referral capture modal
 * @param {string} offerId - Offer ID
 * @param {string} offerName - Offer name for display
 */
function showReferralCapture(offerId, offerName) {
    const modal = document.createElement('div');
    modal.className = 'modal-overlay';
    modal.id = 'referralModal';
    modal.innerHTML = `
        <div class="modal-content" data-testid="referral-modal">
            <div class="modal-header">
                <h3>📝 Capture Referral Outcome</h3>
                <button class="modal-close" onclick="closeReferralModal()">&times;</button>
            </div>
            <div class="modal-body">
                <p class="modal-subtitle">Recording outcome for: <strong>${escapeHtml(offerName)}</strong></p>
                <div class="referral-options">
                    <button class="referral-btn accepted" onclick="logReferral('${offerId}', 'accepted')" data-testid="referral-accepted">
                        ✅ Accepted
                    </button>
                    <button class="referral-btn declined" onclick="logReferral('${offerId}', 'declined')" data-testid="referral-declined">
                        ❌ Declined
                    </button>
                    <button class="referral-btn followup" onclick="logReferral('${offerId}', 'followup')" data-testid="referral-followup">
                        📌 Follow-up
                    </button>
                </div>
                <div class="referral-notes">
                    <label class="form-label">Notes (optional)</label>
                    <textarea id="referralNotes" class="form-input" rows="2" placeholder="Add any notes..."></textarea>
                </div>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
}

/**
 * Close referral modal
 */
function closeReferralModal() {
    const modal = document.getElementById('referralModal');
    if (modal) modal.remove();
}

/**
 * Log referral outcome
 * @param {string} offerId - Offer ID
 * @param {string} outcome - Outcome (accepted/declined/followup)
 */
function logReferral(offerId, outcome) {
    const notes = document.getElementById('referralNotes')?.value || '';
    const memberId = window.MemberState?.currentMember?.PartyId || 'unknown';

    const referral = {
        id: Date.now().toString(),
        offerId,
        outcome,
        notes,
        memberId,
        timestamp: new Date().toISOString(),
        createdBy: 'current_user'
    };

    CrossSellState.referrals.unshift(referral);
    localStorage.setItem('scuReferrals', JSON.stringify(CrossSellState.referrals.slice(0, 100)));

    console.log(`[CrossSell] Logged referral: ${offerId} -> ${outcome}`);

    closeReferralModal();
    showNotification('success', `Referral logged: ${outcome}`);
}

/**
 * Get referral history
 * @param {string} memberId - Optional member ID filter
 * @returns {Array} Referral history
 */
function getReferralHistory(memberId) {
    if (memberId) {
        return CrossSellState.referrals.filter(r => r.memberId === memberId);
    }
    return CrossSellState.referrals;
}

// =================================================================
// AT-RISK FLAGS
// =================================================================

/**
 * At-risk flag types
 */
const RISK_TYPES = {
    NSF: { label: 'NSF Risk', icon: '⚠️', severity: 'high', color: '#ef4444' },
    LOW_BALANCE: { label: 'Low Balance', icon: '💸', severity: 'medium', color: '#f59e0b' },
    DECLINING_ACTIVITY: { label: 'Declining Activity', icon: '📉', severity: 'medium', color: '#f59e0b' },
    CHURN_RISK: { label: 'Churn Risk', icon: '🚪', severity: 'high', color: '#ef4444' },
    OVERDRAFT: { label: 'Overdraft', icon: '❗', severity: 'high', color: '#ef4444' }
};

/**
 * Check member for risk flags
 * @param {Object} member - Member data
 * @returns {Array} Array of risk flags
 */
function checkRiskFlags(member) {
    const flags = [];

    // Simulated risk detection (would use actual data in production)
    if (member) {
        // Random demo flags for testing
        if (Math.random() > 0.7) {
            flags.push({ ...RISK_TYPES.LOW_BALANCE, details: 'Balance below $100' });
        }
        if (Math.random() > 0.8) {
            flags.push({ ...RISK_TYPES.NSF, details: '2 NSF events in last 30 days' });
        }
        if (Math.random() > 0.85) {
            flags.push({ ...RISK_TYPES.CHURN_RISK, details: 'Churn score: 0.72' });
        }
    }

    CrossSellState.riskFlags = flags;
    return flags;
}

/**
 * Render risk flags panel
 * @param {string} containerId - Target container ID
 * @param {Object} member - Member data
 */
function renderRiskFlagsPanel(containerId, member) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const flags = checkRiskFlags(member);

    if (flags.length === 0) {
        container.innerHTML = `
            <div class="risk-panel" data-testid="risk-panel">
                <div class="risk-header">
                    <h3>🛡️ Risk Status</h3>
                </div>
                <div class="risk-clear">
                    <span class="risk-clear-icon">✅</span>
                    <span>No risk flags detected</span>
                </div>
            </div>
        `;
        return;
    }

    container.innerHTML = `
        <div class="risk-panel risk-panel-warning" data-testid="risk-panel">
            <div class="risk-header">
                <h3>⚠️ Risk Flags (${flags.length})</h3>
            </div>
            <div class="risk-flags">
                ${flags.map((flag, idx) => `
                    <div class="risk-flag risk-${flag.severity}" data-testid="risk-flag-${idx}">
                        <span class="risk-flag-icon">${flag.icon}</span>
                        <div class="risk-flag-content">
                            <div class="risk-flag-label">${flag.label}</div>
                            <div class="risk-flag-details">${escapeHtml(flag.details)}</div>
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
}

// =================================================================
// QUICK ACTIONS
// =================================================================

/**
 * Quick action definitions
 */
const QUICK_ACTIONS = [
    { id: 'add_note', label: 'Add Note', icon: '📝', action: 'showMemberNotes' },
    { id: 'view_transactions', label: 'Transactions', icon: '💳', action: 'navigateTransactions' },
    { id: 'start_transfer', label: 'Transfer', icon: '💸', action: 'navigateTransfers' },
    { id: 'send_message', label: 'Message', icon: '✉️', action: 'showSecureMessage' },
    { id: 'schedule_callback', label: 'Callback', icon: '📞', action: 'scheduleCallback' }
];

/**
 * Render quick actions bar
 * @param {string} containerId - Target container ID
 */
function renderQuickActions(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = `
        <div class="quick-actions" data-testid="quick-actions">
            ${QUICK_ACTIONS.map(action => `
                <button class="quick-action-btn" 
                        onclick="executeQuickAction('${action.id}')"
                        data-testid="quick-action-${action.id}"
                        title="${action.label}">
                    <span class="quick-action-icon">${action.icon}</span>
                    <span class="quick-action-label">${action.label}</span>
                </button>
            `).join('')}
        </div>
    `;
}

/**
 * Execute a quick action
 * @param {string} actionId - Action ID
 */
function executeQuickAction(actionId) {
    const action = QUICK_ACTIONS.find(a => a.id === actionId);
    if (!action) return;

    console.log(`[QuickAction] Executing: ${action.label}`);

    switch (actionId) {
        case 'add_note':
            const memberId = window.MemberState?.currentMember?.PartyId;
            if (memberId && typeof showMemberNotes === 'function') {
                showMemberNotes(memberId);
            }
            break;
        case 'view_transactions':
            if (typeof showSection === 'function') {
                showSection('transactions');
            }
            break;
        case 'start_transfer':
            if (typeof showSection === 'function') {
                showSection('transfers');
            }
            break;
        case 'send_message':
            showSecureMessageForm();
            break;
        case 'schedule_callback':
            showCallbackScheduler();
            break;
    }
}

/**
 * Show secure message form (placeholder)
 */
function showSecureMessageForm() {
    showNotification('info', 'Secure messaging feature coming soon');
}

/**
 * Show callback scheduler (placeholder)
 */
function showCallbackScheduler() {
    showNotification('info', 'Callback scheduling feature coming soon');
}

// =================================================================
// ALERT THRESHOLDS
// =================================================================

/**
 * Get alert thresholds for current user
 * @returns {Object} Alert thresholds
 */
function getAlertThresholds() {
    return {
        lowBalanceThreshold: CrossSellState.alertThresholds.lowBalance || 100,
        nsfCountThreshold: CrossSellState.alertThresholds.nsfCount || 2,
        churnScoreThreshold: CrossSellState.alertThresholds.churnScore || 0.7
    };
}

/**
 * Save alert thresholds
 * @param {Object} thresholds - Threshold values
 */
function saveAlertThresholds(thresholds) {
    CrossSellState.alertThresholds = {
        lowBalance: thresholds.lowBalanceThreshold || 100,
        nsfCount: thresholds.nsfCountThreshold || 2,
        churnScore: thresholds.churnScoreThreshold || 0.7
    };
    localStorage.setItem('scuAlertThresholds', JSON.stringify(CrossSellState.alertThresholds));
    console.log('[AlertThresholds] Saved:', CrossSellState.alertThresholds);
}

/**
 * Render alert threshold settings
 * @param {string} containerId - Target container ID
 */
function renderAlertThresholdSettings(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const thresholds = getAlertThresholds();

    container.innerHTML = `
        <div class="threshold-settings" data-testid="threshold-settings">
            <h4>⚙️ Alert Thresholds</h4>
            <div class="threshold-row">
                <label class="form-label">Low Balance Alert ($)</label>
                <input type="number" id="lowBalanceThreshold" class="form-input form-input-sm" 
                       value="${thresholds.lowBalanceThreshold}" min="0" step="50">
            </div>
            <div class="threshold-row">
                <label class="form-label">NSF Count Threshold</label>
                <input type="number" id="nsfCountThreshold" class="form-input form-input-sm" 
                       value="${thresholds.nsfCountThreshold}" min="1" max="10">
            </div>
            <div class="threshold-row">
                <label class="form-label">Churn Score Threshold</label>
                <input type="number" id="churnScoreThreshold" class="form-input form-input-sm" 
                       value="${thresholds.churnScoreThreshold}" min="0" max="1" step="0.1">
            </div>
            <button class="btn btn-primary btn-sm" onclick="saveThresholdsFromForm()" data-testid="save-thresholds">
                Save Thresholds
            </button>
        </div>
    `;
}

/**
 * Save thresholds from form
 */
function saveThresholdsFromForm() {
    saveAlertThresholds({
        lowBalanceThreshold: parseFloat(document.getElementById('lowBalanceThreshold')?.value) || 100,
        nsfCountThreshold: parseInt(document.getElementById('nsfCountThreshold')?.value) || 2,
        churnScoreThreshold: parseFloat(document.getElementById('churnScoreThreshold')?.value) || 0.7
    });
    showNotification('success', 'Alert thresholds saved');
}

// =================================================================
// UTILITY FUNCTIONS
// =================================================================

/**
 * Show notification toast
 * @param {string} type - Type (success/error/info/warning)
 * @param {string} message - Message to display
 */
function showNotification(type, message) {
    const toast = document.createElement('div');
    toast.className = `notification notification-${type}`;
    toast.innerHTML = `
        <span class="notification-icon">${type === 'success' ? '✅' : type === 'error' ? '❌' : type === 'warning' ? '⚠️' : 'ℹ️'}</span>
        <span class="notification-message">${escapeHtml(message)}</span>
    `;
    document.body.appendChild(toast);

    setTimeout(() => toast.classList.add('show'), 10);
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

/**
 * Escape HTML (reuse from banking_member if available)
 */
function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// =================================================================
// INITIALIZATION
// =================================================================

/**
 * Initialize cross-sell module
 */
function initCrossSellModule() {
    console.log('[CrossSell Module] Initializing...');
    console.log('[CrossSell Module] Initialized');
}

// =================================================================
// GLOBAL EXPORTS
// =================================================================

window.CrossSellState = CrossSellState;
window.PRODUCT_OFFERS = PRODUCT_OFFERS;
window.RISK_TYPES = RISK_TYPES;
window.QUICK_ACTIONS = QUICK_ACTIONS;

window.generateNextBestActions = generateNextBestActions;
window.renderNextBestActionPanel = renderNextBestActionPanel;
window.presentOffer = presentOffer;

window.showReferralCapture = showReferralCapture;
window.closeReferralModal = closeReferralModal;
window.logReferral = logReferral;
window.getReferralHistory = getReferralHistory;

window.checkRiskFlags = checkRiskFlags;
window.renderRiskFlagsPanel = renderRiskFlagsPanel;

window.renderQuickActions = renderQuickActions;
window.executeQuickAction = executeQuickAction;

window.getAlertThresholds = getAlertThresholds;
window.saveAlertThresholds = saveAlertThresholds;
window.renderAlertThresholdSettings = renderAlertThresholdSettings;
window.saveThresholdsFromForm = saveThresholdsFromForm;

window.showNotification = showNotification;
window.initCrossSellModule = initCrossSellModule;

// Auto-initialize
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initCrossSellModule);
} else {
    initCrossSellModule();
}
