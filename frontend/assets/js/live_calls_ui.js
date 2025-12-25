/**
 * Live Calls UI - Screen pop, verification gate, and call simulation
 * Extracted from banking.html for modularity
 * 
 * Dependencies: banking_core.js (getCsrfToken)
 * Used by: banking.html
 */

// CTI API Base URL
const CTI_API_BASE = '/api/v1/cti';

// Current call state
let currentCall = null;

// Verification method prompts
const verificationPrompts = {
    'dob': 'What is your date of birth?',
    'last4ssn': 'What are the last 4 digits of your SSN?',
    'last4card': 'What are the last 4 digits of your card number?',
    'address': 'What is your full address on file?',
    'otp': 'A one-time code has been sent to your registered device'
};

// =================================================================
// SCREEN POP
// =================================================================

/**
 * Display screen pop with member data
 * @param {Object} pop - Screen pop data from CTI
 */
function showScreenPop(pop) {
    if (!pop) {
        console.warn('[LiveCalls] No screen pop data');
        return;
    }

    const panel = document.getElementById('screenPopPanel');
    if (!panel) {
        console.error('[LiveCalls] screenPopPanel not found');
        return;
    }

    // Force panel visibility
    panel.style.display = 'block';

    // Update caller info
    const callerName = document.getElementById('callerName');
    const callerPhone = document.getElementById('callerPhone');
    const callerMemberId = document.getElementById('callerMemberId');

    if (callerName) callerName.textContent = pop.member_name || 'Unknown Caller';
    if (callerPhone) callerPhone.textContent = formatPhoneNumber(pop.phone) || pop.ani || '';
    if (callerMemberId) callerMemberId.textContent = pop.member_id || 'Not identified';

    // Update status badge
    const badge = document.getElementById('callStatusBadge');
    if (badge) {
        badge.textContent = 'INCOMING';
        badge.style.background = '#ffc107';
    }

    // Update account summary if available
    const accountSummary = document.getElementById('accountSummaryContent');
    if (accountSummary && pop.fiserv_data) {
        accountSummary.innerHTML = renderAccountSummary(pop.fiserv_data);
    }

    // Update call history if available
    const callHistory = document.getElementById('callHistoryContent');
    if (callHistory && pop.call_history) {
        callHistory.innerHTML = renderCallHistory(pop.call_history);
    }

    // Show verification panel
    const verifPanel = document.getElementById('verificationPanel');
    if (verifPanel) {
        verifPanel.style.display = 'block';
    }

    console.log('[LiveCalls] Screen pop displayed:', pop);
}

/**
 * Render account summary HTML
 */
function renderAccountSummary(fiservData) {
    if (!fiservData || !fiservData.accounts) {
        return '<p style="color:#888;">No account data available</p>';
    }

    return fiservData.accounts.map(acct => `
        <div class="account-item" style="padding: 8px; border-bottom: 1px solid #eee;">
            <strong>${acct.type || 'Account'}</strong> ...${acct.last4 || '****'}<br>
            <span style="font-size: 1.2em; color: ${acct.balance >= 0 ? '#28a745' : '#dc3545'};">
                $${(acct.balance || 0).toLocaleString()}
            </span>
        </div>
    `).join('');
}

/**
 * Render call history HTML
 */
function renderCallHistory(history) {
    if (!history || history.length === 0) {
        return '<p style="color:#888;">No previous calls</p>';
    }

    return history.slice(0, 5).map(call => `
        <div style="padding: 4px 0; font-size: 0.9em;">
            ${call.date || 'Unknown'} - ${call.reason || 'General inquiry'}
        </div>
    `).join('');
}

/**
 * Format phone number for display
 */
function formatPhoneNumber(phone) {
    if (!phone) return '';
    const cleaned = phone.replace(/\D/g, '');
    if (cleaned.length === 10) {
        return `(${cleaned.slice(0, 3)}) ${cleaned.slice(3, 6)}-${cleaned.slice(6)}`;
    } else if (cleaned.length === 11) {
        return `+${cleaned[0]} (${cleaned.slice(1, 4)}) ${cleaned.slice(4, 7)}-${cleaned.slice(7)}`;
    }
    return phone;
}

// =================================================================
// VERIFICATION GATE
// =================================================================

/**
 * Update verification prompt label when method changes
 */
function updateVerificationPrompt() {
    const method = document.getElementById('verifyMethod')?.value;
    const label = document.getElementById('verifyPromptLabel');
    if (label) {
        label.textContent = verificationPrompts[method] || '';
    }
}

/**
 * Submit verification attempt
 */
async function submitVerification() {
    if (!currentCall) {
        alert('No active call to verify');
        return;
    }

    const method = document.getElementById('verifyMethod')?.value;
    const providedValue = document.getElementById('verifyValue')?.value;

    if (!method || !providedValue) {
        alert('Please select a verification method and enter the response');
        return;
    }

    const statusEl = document.getElementById('verifyStatus');
    if (statusEl) statusEl.textContent = 'Verifying...';

    try {
        const response = await fetch(`${CTI_API_BASE}/calls/${currentCall.call_id}/verify`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRF-Token': typeof getCsrfToken === 'function' ? getCsrfToken() : ''
            },
            credentials: 'include',
            body: JSON.stringify({
                verification_method: method,
                provided_value: providedValue
            })
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();

        if (data.verified) {
            currentCall.verified = true;

            if (statusEl) {
                statusEl.textContent = '✅ Verified!';
                statusEl.style.color = '#28a745';
            }

            // Update UI
            const badge = document.getElementById('callStatusBadge');
            if (badge) {
                badge.textContent = 'VERIFIED';
                badge.style.background = '#28a745';
            }

            // Hide verification panel, show full access
            const verifPanel = document.getElementById('verificationPanel');
            if (verifPanel) verifPanel.style.display = 'none';

            const fullAccess = document.getElementById('fullAccessPanel');
            if (fullAccess) fullAccess.style.display = 'block';

        } else {
            if (statusEl) {
                statusEl.textContent = '❌ Verification failed: ' + (data.reason || 'Unknown error');
                statusEl.style.color = '#dc3545';
            }
        }

    } catch (e) {
        console.error('Verification error:', e);
        if (statusEl) {
            statusEl.textContent = '❌ Error: ' + e.message;
            statusEl.style.color = '#dc3545';
        }
    }
}

/**
 * Send OTP code to member
 */
async function sendOtpCode() {
    if (!currentCall?.call_id) {
        alert('No active call');
        return;
    }

    alert('OTP code sent to member\'s registered device');
    // TODO: Implement actual OTP sending via backend
}

/**
 * Access member data (requires verification)
 */
async function accessMemberData(action) {
    if (!currentCall?.verified) {
        alert('Member must be verified before accessing data');
        return;
    }

    console.log(`[LiveCalls] Accessing member data: ${action}`);
    alert(`Accessing ${action} for verified member`);
    // TODO: Implement data access based on action
}

// =================================================================
// CALL SIMULATION (Development/Testing)
// =================================================================

/**
 * Simulate inbound call (for testing)
 */
async function simulateInboundCall() {
    const ani = document.getElementById('simCallAni')?.value || '+16035551234';
    const memberId = document.getElementById('simMemberId')?.value || null;

    try {
        const callSid = 'SIM-' + Date.now();
        const response = await fetch(`${CTI_API_BASE}/webhook/call_started`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRF-Token': typeof getCsrfToken === 'function' ? getCsrfToken() : ''
            },
            credentials: 'include',
            body: JSON.stringify({
                event: 'call_started',
                call_sid: callSid,
                ani: ani,
                dnis: '+18005551000',
                direction: 'inbound',
                queue_id: 'MSR_QUEUE_1',
                source_system: 'simulator'
            })
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();

        // Store current call
        currentCall = {
            call_id: data.call_id,
            call_sid: callSid,
            verified: false
        };

        // Show screen pop
        showScreenPop(data.screen_pop);

    } catch (e) {
        console.error('Simulate call error:', e);
        alert('Failed to simulate call: ' + e.message);
    }
}

/**
 * Simulate call answer
 */
async function simulateCallAnswer() {
    if (!currentCall) {
        alert('No active call to answer');
        return;
    }

    try {
        const response = await fetch(`${CTI_API_BASE}/webhook/call_answered`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRF-Token': typeof getCsrfToken === 'function' ? getCsrfToken() : ''
            },
            credentials: 'include',
            body: JSON.stringify({
                event: 'call_answered',
                call_sid: currentCall.call_sid,
                agent_id: 'MSR-Demo',
                agent_extension: '1001',
                source_system: 'simulator'
            })
        });

        if (response.ok) {
            const badge = document.getElementById('callStatusBadge');
            if (badge) {
                badge.textContent = 'IN PROGRESS';
                badge.style.background = 'var(--scu-blue)';
            }
        }
    } catch (e) {
        console.error('Answer call error:', e);
    }
}

/**
 * Simulate call end
 */
async function simulateCallEnd() {
    if (!currentCall) {
        alert('No active call to end');
        return;
    }

    try {
        await fetch(`${CTI_API_BASE}/webhook/call_ended`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRF-Token': typeof getCsrfToken === 'function' ? getCsrfToken() : ''
            },
            credentials: 'include',
            body: JSON.stringify({
                event: 'call_ended',
                call_sid: currentCall.call_sid,
                duration_seconds: 180,
                disposition: 'completed',
                source_system: 'simulator'
            })
        });

        // Reset UI
        const panel = document.getElementById('screenPopPanel');
        if (panel) panel.style.display = 'none';

        const badge = document.getElementById('callStatusBadge');
        if (badge) {
            badge.textContent = 'READY';
            badge.style.background = '#28a745';
        }

        currentCall = null;

    } catch (e) {
        console.error('End call error:', e);
    }
}

// =================================================================
// ACTIVE CALLS QUEUE
// =================================================================

/**
 * Refresh active calls list
 */
async function refreshActiveCalls() {
    const queueEl = document.getElementById('activeCallsQueue');
    if (!queueEl) return;

    try {
        const response = await fetch(`${CTI_API_BASE}/calls/active`, {
            headers: { 'Accept': 'application/json' },
            credentials: 'include'
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();

        if (!data.calls || data.calls.length === 0) {
            queueEl.innerHTML = '<p style="color:#888; text-align:center;">No active calls</p>';
            return;
        }

        queueEl.innerHTML = data.calls.map(call => `
            <div class="call-item" onclick="loadCallDetails('${call.call_id}')" style="padding: 8px; border-bottom: 1px solid #eee; cursor: pointer;">
                <strong>${call.ani || 'Unknown'}</strong>
                <span style="float: right; font-size: 0.8em; color: #888;">${call.duration || '0:00'}</span><br>
                <span style="font-size: 0.9em; color: ${call.verified ? '#28a745' : '#ffc107'};">
                    ${call.verified ? '✅ Verified' : '⏳ Pending'}
                </span>
            </div>
        `).join('');

    } catch (e) {
        console.error('Refresh calls error:', e);
        queueEl.innerHTML = '<p style="color:#dc3545;">Failed to load calls</p>';
    }
}

/**
 * Load call details
 */
async function loadCallDetails(callId) {
    console.log(`[LiveCalls] Loading details for call ${callId}`);
    // TODO: Load and display call details
}

// =================================================================
// INITIALIZATION
// =================================================================

// Make functions globally available
window.showScreenPop = showScreenPop;
window.formatPhoneNumber = formatPhoneNumber;
window.updateVerificationPrompt = updateVerificationPrompt;
window.submitVerification = submitVerification;
window.sendOtpCode = sendOtpCode;
window.accessMemberData = accessMemberData;
window.simulateInboundCall = simulateInboundCall;
window.simulateCallAnswer = simulateCallAnswer;
window.simulateCallEnd = simulateCallEnd;
window.refreshActiveCalls = refreshActiveCalls;
window.loadCallDetails = loadCallDetails;
