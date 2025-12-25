/**
 * Banking Extended APIs Module - Phase 1 Features
 * ================================================
 * Stop Check management and Card Management functionality
 * 
 * Dependencies: banking_core.js (for getCsrfToken, API_BASE)
 * Used by: banking.html
 * 
 * @module banking_extended
 * @version 1.0.0
 */

// =================================================================
// STATE MANAGEMENT
// =================================================================

const ExtendedState = {
    stopPayments: [],
    cards: [],
    selectedCard: null,
    currentAccountId: null
};

// =================================================================
// STOP PAYMENT FUNCTIONS
// =================================================================

/**
 * Add a new stop payment request
 */
async function addStopPayment() {
    const accountId = document.getElementById('stopAccountId').value.trim();
    const checkStart = document.getElementById('stopCheckStart').value.trim();
    const checkEnd = document.getElementById('stopCheckEnd').value.trim();
    const amount = document.getElementById('stopAmount').value;
    const payee = document.getElementById('stopPayee').value.trim();
    const reason = document.getElementById('stopReason').value;

    if (!accountId) {
        alert('Please enter an account number');
        return;
    }

    if (!checkStart) {
        alert('Please enter a check number or range start');
        return;
    }

    const spinner = document.getElementById('stopSpinner');
    if (spinner) spinner.style.display = 'inline-block';

    try {
        const payload = {
            account_id: accountId,
            check_number_start: checkStart,
            check_number_end: checkEnd || checkStart,
            amount: amount ? parseFloat(amount) : null,
            payee: payee || null,
            reason: reason,
            expiration_days: 180 // Standard 6-month expiration
        };

        // For demo purposes, simulate API call
        // In production: const response = await fetch('/fiserv/api/v1/stopcheck/add', {...})

        console.log('[StopCheck] Adding stop payment:', payload);

        // Simulate success
        const newStop = {
            id: `SP-${Date.now()}`,
            ...payload,
            status: 'Active',
            created_at: new Date().toISOString(),
            expires_at: new Date(Date.now() + 180 * 24 * 60 * 60 * 1000).toISOString()
        };

        ExtendedState.stopPayments.unshift(newStop);
        ExtendedState.currentAccountId = accountId;

        // Update UI
        updateStopPaymentStats();
        renderStopPayments();
        clearStopForm();

        showNotification('Stop payment added successfully', 'success');

    } catch (error) {
        console.error('[StopCheck] Error adding stop payment:', error);
        alert('Failed to add stop payment: ' + error.message);
    } finally {
        if (spinner) spinner.style.display = 'none';
    }
}

/**
 * Load stop payments for an account
 */
async function loadStopPayments() {
    const accountId = document.getElementById('stopAccountId').value.trim();

    if (!accountId) {
        alert('Please enter an account number first');
        return;
    }

    ExtendedState.currentAccountId = accountId;

    try {
        // For demo, generate sample data
        // In production: const response = await fetch(`/fiserv/api/v1/stopcheck/list/${accountId}`)

        console.log('[StopCheck] Loading stop payments for account:', accountId);

        // Sample demo data
        ExtendedState.stopPayments = [
            {
                id: 'SP-001',
                account_id: accountId,
                check_number_start: '1001',
                check_number_end: '1001',
                amount: 500.00,
                payee: 'ABC Services',
                reason: 'dispute',
                status: 'Active',
                created_at: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(),
                expires_at: new Date(Date.now() + 175 * 24 * 60 * 60 * 1000).toISOString()
            },
            {
                id: 'SP-002',
                account_id: accountId,
                check_number_start: '1050',
                check_number_end: '1060',
                amount: null,
                payee: null,
                reason: 'lost',
                status: 'Active',
                created_at: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(),
                expires_at: new Date(Date.now() + 150 * 24 * 60 * 60 * 1000).toISOString()
            }
        ];

        updateStopPaymentStats();
        renderStopPayments();

    } catch (error) {
        console.error('[StopCheck] Error loading stop payments:', error);
    }
}

/**
 * Cancel a stop payment
 * @param {string} stopId - Stop payment ID
 */
async function cancelStopPayment(stopId) {
    if (!confirm('Are you sure you want to cancel this stop payment?')) {
        return;
    }

    try {
        // In production: await fetch(`/fiserv/api/v1/stopcheck/${stopId}`, { method: 'DELETE' })

        ExtendedState.stopPayments = ExtendedState.stopPayments.filter(s => s.id !== stopId);
        updateStopPaymentStats();
        renderStopPayments();
        showNotification('Stop payment cancelled', 'info');

    } catch (error) {
        console.error('[StopCheck] Error cancelling stop payment:', error);
        alert('Failed to cancel stop payment: ' + error.message);
    }
}

/**
 * Update stop payment statistics
 */
function updateStopPaymentStats() {
    const active = ExtendedState.stopPayments.filter(s => s.status === 'Active').length;
    const expiringSoon = ExtendedState.stopPayments.filter(s => {
        const expiresAt = new Date(s.expires_at);
        const daysUntilExpiry = (expiresAt - new Date()) / (1000 * 60 * 60 * 24);
        return daysUntilExpiry <= 30 && daysUntilExpiry > 0;
    }).length;

    document.getElementById('activeStopCount').textContent = active;
    document.getElementById('expiringStopCount').textContent = expiringSoon;
    document.getElementById('stoppedThisMonth').textContent = Math.floor(Math.random() * 10); // Demo
}

/**
 * Render stop payments list
 */
function renderStopPayments() {
    const container = document.getElementById('stopPaymentsList');
    if (!container) return;

    if (ExtendedState.stopPayments.length === 0) {
        container.innerHTML = '<div class="results-empty">No stop payments found for this account</div>';
        return;
    }

    container.innerHTML = ExtendedState.stopPayments.map(stop => {
        const checkRange = stop.check_number_end !== stop.check_number_start
            ? `${stop.check_number_start} - ${stop.check_number_end}`
            : stop.check_number_start;

        const expiresAt = new Date(stop.expires_at);
        const daysLeft = Math.ceil((expiresAt - new Date()) / (1000 * 60 * 60 * 24));
        const isExpiringSoon = daysLeft <= 30;

        return `
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 1rem; background: white; border-radius: 8px; margin-bottom: 0.5rem; border: 1px solid ${isExpiringSoon ? '#fbbf24' : '#e5e7eb'};">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <div style="width: 48px; height: 48px; background: ${stop.status === 'Active' ? '#fef3c7' : '#f3f4f6'}; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 1.5rem;">
                        🛑
                    </div>
                    <div>
                        <div style="font-weight: 600; color: #1f2937;">Check #${checkRange}</div>
                        <div style="font-size: 0.85rem; color: #6b7280;">
                            ${stop.payee ? `Payee: ${stop.payee}` : 'Any payee'}
                            ${stop.amount ? ` • Amount: $${stop.amount.toFixed(2)}` : ''}
                        </div>
                        <div style="font-size: 0.75rem; color: ${isExpiringSoon ? '#d97706' : '#6b7280'};">
                            Expires: ${expiresAt.toLocaleDateString()} (${daysLeft} days)
                        </div>
                    </div>
                </div>
                <div style="display: flex; gap: 0.5rem;">
                    <span style="padding: 0.25rem 0.75rem; background: ${stop.status === 'Active' ? '#dcfce7' : '#f3f4f6'}; color: ${stop.status === 'Active' ? '#166534' : '#6b7280'}; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">
                        ${stop.status}
                    </span>
                    <button onclick="cancelStopPayment('${stop.id}')" style="padding: 0.25rem 0.75rem; background: #fee2e2; color: #dc2626; border: none; border-radius: 4px; font-size: 0.75rem; cursor: pointer;">
                        Cancel
                    </button>
                </div>
            </div>
        `;
    }).join('');
}

/**
 * Clear stop payment form
 */
function clearStopForm() {
    document.getElementById('stopAccountId').value = '';
    document.getElementById('stopCheckStart').value = '';
    document.getElementById('stopCheckEnd').value = '';
    document.getElementById('stopAmount').value = '';
    document.getElementById('stopPayee').value = '';
    document.getElementById('stopReason').value = 'lost';
}

// =================================================================
// CARD MANAGEMENT FUNCTIONS
// =================================================================

/**
 * Lookup cards for a member
 */
async function lookupCards() {
    const memberId = document.getElementById('cardMemberId').value.trim();
    const last4 = document.getElementById('cardLast4').value.trim();

    if (!memberId) {
        alert('Please enter a member ID or account number');
        return;
    }

    const spinner = document.getElementById('cardSpinner');
    if (spinner) spinner.style.display = 'inline-block';

    try {
        // For demo, generate sample card data
        // In production: const response = await fetch(`/fiserv/api/v1/cards/${memberId}`)

        console.log('[Cards] Looking up cards for:', memberId);

        // Sample demo data
        ExtendedState.cards = [
            {
                id: 'CARD-001',
                card_number: '************4532',
                last4: '4532',
                type: 'Debit',
                brand: 'Visa',
                status: 'Active',
                expires: '12/26',
                daily_limit: 5000,
                atm_limit: 500,
                is_primary: true
            },
            {
                id: 'CARD-002',
                card_number: '************8721',
                last4: '8721',
                type: 'Credit',
                brand: 'Mastercard',
                status: 'Active',
                expires: '03/25',
                credit_limit: 15000,
                available_credit: 12500,
                is_primary: false
            },
            {
                id: 'CARD-003',
                card_number: '************1199',
                last4: '1199',
                type: 'Debit',
                brand: 'Visa',
                status: 'Blocked',
                expires: '08/27',
                daily_limit: 2000,
                atm_limit: 300,
                is_primary: false
            }
        ];

        // Filter by last 4 if provided
        if (last4) {
            ExtendedState.cards = ExtendedState.cards.filter(c => c.last4.includes(last4));
        }

        updateCardStats();
        renderCards();

    } catch (error) {
        console.error('[Cards] Error loading cards:', error);
        alert('Failed to load cards: ' + error.message);
    } finally {
        if (spinner) spinner.style.display = 'none';
    }
}

/**
 * Update card statistics
 */
function updateCardStats() {
    const active = ExtendedState.cards.filter(c => c.status === 'Active').length;
    const blocked = ExtendedState.cards.filter(c => c.status === 'Blocked').length;
    const lost = ExtendedState.cards.filter(c => c.status === 'Lost' || c.status === 'Stolen').length;

    const expiringSoon = ExtendedState.cards.filter(c => {
        const parts = c.expires.split('/');
        const expMonth = parseInt(parts[0]);
        const expYear = parseInt('20' + parts[1]);
        const expDate = new Date(expYear, expMonth - 1, 1);
        const daysUntilExpiry = (expDate - new Date()) / (1000 * 60 * 60 * 24);
        return daysUntilExpiry <= 90 && daysUntilExpiry > 0;
    }).length;

    document.getElementById('activeCardCount').textContent = active;
    document.getElementById('blockedCardCount').textContent = blocked;
    document.getElementById('lostCardCount').textContent = lost;
    document.getElementById('expiringCardCount').textContent = expiringSoon;
}

/**
 * Render cards list
 */
function renderCards() {
    const container = document.getElementById('cardsList');
    if (!container) return;

    document.getElementById('cardResultCount').textContent = `${ExtendedState.cards.length} cards found`;

    if (ExtendedState.cards.length === 0) {
        container.innerHTML = '<div class="results-empty">No cards found for this member</div>';
        return;
    }

    container.innerHTML = ExtendedState.cards.map(card => {
        const statusColor = {
            'Active': '#10b981',
            'Blocked': '#f59e0b',
            'Lost': '#ef4444',
            'Stolen': '#dc2626',
            'Expired': '#6b7280'
        }[card.status] || '#6b7280';

        const brandIcon = {
            'Visa': '💳',
            'Mastercard': '💳',
            'Amex': '💳',
            'Discover': '💳'
        }[card.brand] || '💳';

        return `
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 1rem; background: ${ExtendedState.selectedCard === card.id ? '#eff6ff' : 'white'}; border-radius: 8px; margin-bottom: 0.5rem; border: 2px solid ${ExtendedState.selectedCard === card.id ? '#3b82f6' : '#e5e7eb'}; cursor: pointer; transition: all 0.2s;"
                 onclick="selectCard('${card.id}')"
                 onmouseover="this.style.borderColor='#3b82f6'"
                 onmouseout="this.style.borderColor='${ExtendedState.selectedCard === card.id ? '#3b82f6' : '#e5e7eb'}'">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <div style="width: 64px; height: 40px; background: linear-gradient(135deg, #1e293b, #334155); border-radius: 6px; display: flex; align-items: center; justify-content: center; color: white; font-size: 0.75rem; font-weight: 600;">
                        ${card.brand}
                    </div>
                    <div>
                        <div style="font-weight: 600; color: #1f2937; display: flex; align-items: center; gap: 0.5rem;">
                            ${card.card_number}
                            ${card.is_primary ? '<span style="font-size: 0.65rem; padding: 0.15rem 0.5rem; background: #6366f1; color: white; border-radius: 4px;">PRIMARY</span>' : ''}
                        </div>
                        <div style="font-size: 0.85rem; color: #6b7280;">
                            ${card.type} • Expires ${card.expires}
                            ${card.credit_limit ? ` • Limit: $${card.credit_limit.toLocaleString()}` : ''}
                        </div>
                    </div>
                </div>
                <div style="text-align: right;">
                    <span style="padding: 0.25rem 0.75rem; background: ${statusColor}22; color: ${statusColor}; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">
                        ${card.status}
                    </span>
                    ${card.available_credit !== undefined ? `
                        <div style="font-size: 0.8rem; color: #6b7280; margin-top: 0.25rem;">
                            Available: $${card.available_credit.toLocaleString()}
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    }).join('');
}

/**
 * Select a card for actions
 * @param {string} cardId - Card ID
 */
function selectCard(cardId) {
    ExtendedState.selectedCard = cardId;
    const card = ExtendedState.cards.find(c => c.id === cardId);

    if (card) {
        document.getElementById('selectedCardNumber').textContent = card.card_number;
        document.getElementById('cardActionsPanel').style.display = 'block';
    }

    renderCards();
}

/**
 * Set card status (block, unblock, lost, stolen)
 * @param {string} status - New status
 */
async function setCardStatus(status) {
    if (!ExtendedState.selectedCard) {
        alert('Please select a card first');
        return;
    }

    const statusLabels = {
        'block': 'Block',
        'unblock': 'Activate',
        'lost': 'Report Lost',
        'stolen': 'Report Stolen'
    };

    if (!confirm(`Are you sure you want to ${statusLabels[status]} this card?`)) {
        return;
    }

    try {
        // In production: await fetch(`/fiserv/api/v1/cards/${ExtendedState.selectedCard}/status`, { method: 'PUT', body: { status } })

        const card = ExtendedState.cards.find(c => c.id === ExtendedState.selectedCard);
        if (card) {
            const statusMap = {
                'block': 'Blocked',
                'unblock': 'Active',
                'lost': 'Lost',
                'stolen': 'Stolen'
            };
            card.status = statusMap[status];
        }

        updateCardStats();
        renderCards();
        showNotification(`Card ${statusLabels[status].toLowerCase()}ed successfully`, 'success');

    } catch (error) {
        console.error('[Cards] Error updating card status:', error);
        alert('Failed to update card status: ' + error.message);
    }
}

/**
 * Order replacement card
 */
function orderReplacementCard() {
    if (!ExtendedState.selectedCard) {
        alert('Please select a card first');
        return;
    }

    const card = ExtendedState.cards.find(c => c.id === ExtendedState.selectedCard);
    if (!card) return;

    const rushOrder = confirm('Order rush delivery? (Additional $15 fee)');

    alert(`Replacement card ordered for ${card.card_number}\n\nDelivery: ${rushOrder ? '2-3 business days (Rush)' : '7-10 business days (Standard)'}\n\nNew card will be mailed to address on file.`);

    showNotification('Replacement card ordered', 'success');
}

/**
 * Set travel notice
 */
function setTravelNotice() {
    if (!ExtendedState.selectedCard) {
        alert('Please select a card first');
        return;
    }

    const destination = prompt('Enter travel destination(s):');
    if (!destination) return;

    const startDate = prompt('Start date (MM/DD/YYYY):');
    if (!startDate) return;

    const endDate = prompt('End date (MM/DD/YYYY):');
    if (!endDate) return;

    alert(`Travel notice set!\n\nDestination: ${destination}\nDates: ${startDate} - ${endDate}\n\nCard ${ExtendedState.selectedCard} is enabled for travel.`);

    showNotification('Travel notice added', 'success');
}

/**
 * View card limits
 */
function viewCardLimits() {
    if (!ExtendedState.selectedCard) {
        alert('Please select a card first');
        return;
    }

    const card = ExtendedState.cards.find(c => c.id === ExtendedState.selectedCard);
    if (!card) return;

    let limitsInfo = `Card Limits for ${card.card_number}\n\n`;

    if (card.type === 'Debit') {
        limitsInfo += `Daily Purchase Limit: $${card.daily_limit?.toLocaleString() || 'N/A'}\n`;
        limitsInfo += `Daily ATM Limit: $${card.atm_limit?.toLocaleString() || 'N/A'}\n`;
    } else {
        limitsInfo += `Credit Limit: $${card.credit_limit?.toLocaleString() || 'N/A'}\n`;
        limitsInfo += `Available Credit: $${card.available_credit?.toLocaleString() || 'N/A'}\n`;
    }

    alert(limitsInfo);
}

// =================================================================
// UTILITY FUNCTIONS
// =================================================================

/**
 * Show notification toast
 * @param {string} message - Notification message
 * @param {string} type - Type (success, info, error)
 */
function showNotification(message, type = 'info') {
    // Create toast element
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed;
        top: 80px;
        right: 20px;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        color: white;
        font-weight: 500;
        z-index: 10000;
        animation: slideIn 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    `;

    switch (type) {
        case 'success':
            toast.style.background = 'linear-gradient(135deg, #10b981, #059669)';
            break;
        case 'error':
            toast.style.background = 'linear-gradient(135deg, #ef4444, #dc2626)';
            break;
        default:
            toast.style.background = 'linear-gradient(135deg, #3b82f6, #1d4ed8)';
    }

    toast.textContent = message;
    document.body.appendChild(toast);

    // Remove after 3 seconds
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// =================================================================
// GLOBAL EXPORTS
// =================================================================

window.ExtendedState = ExtendedState;
window.addStopPayment = addStopPayment;
window.loadStopPayments = loadStopPayments;
window.cancelStopPayment = cancelStopPayment;
window.clearStopForm = clearStopForm;
window.lookupCards = lookupCards;
window.selectCard = selectCard;
window.setCardStatus = setCardStatus;
window.orderReplacementCard = orderReplacementCard;
window.setTravelNotice = setTravelNotice;
window.viewCardLimits = viewCardLimits;
window.showNotification = showNotification;

console.log('[Extended APIs] Module loaded - Stop Check & Card Management ready');
