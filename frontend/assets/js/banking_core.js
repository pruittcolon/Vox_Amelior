/**
 * Banking Core - Core utilities, navigation, and authentication functions
 * Extracted from banking.html for modularity
 * 
 * Dependencies: None
 * Used by: banking.html
 */

// SECURITY: All requests go through Gateway, not directly to services
// Gateway proxies /fiserv/* to fiserv-service:8015
const API_BASE = '/fiserv';
let currentSection = 'party';
let currentRole = 'msr';

// =================================================================
// TYPEAHEAD & DEBOUNCE UTILITIES (Phase 2)
// =================================================================
let _searchDebounceTimer = null;
const SEARCH_DEBOUNCE_MS = 300; // 300ms debounce for typeahead

/**
 * Debounce utility - delays function execution until after wait period
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in ms
 * @returns {Function} Debounced function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Instant typeahead search - triggered on keyup with debounce
 * Searches across name, phone, email with minimum 2 characters
 */
async function typeaheadSearch() {
    const nameInput = document.getElementById('partyName');
    const phoneInput = document.getElementById('partyPhone');
    const emailInput = document.getElementById('partyEmail');
    const accountInput = document.getElementById('partyAccount');

    const name = nameInput?.value || '';
    const phone = phoneInput?.value || '';
    const email = emailInput?.value || '';
    const accountId = accountInput?.value || '';

    // Require at least 2 characters in any field to trigger search
    const hasInput = name.length >= 2 || phone.length >= 4 || email.length >= 3 || accountId.length >= 3;

    if (!hasInput) {
        // Clear results if input is too short
        const resultsDiv = document.getElementById('partyResults');
        if (resultsDiv) {
            resultsDiv.innerHTML = '<div class="results-hint" style="color: #9ca3af; text-align: center; padding: 2rem;">Type to search members...</div>';
        }
        return;
    }

    // Show loading state
    const resultsDiv = document.getElementById('partyResults');
    if (resultsDiv) {
        resultsDiv.innerHTML = '<div class="results-loading" style="text-align: center; padding: 2rem;"><div class="spinner" style="display: inline-block; width: 24px; height: 24px; border: 3px solid #e5e7eb; border-top-color: #3b82f6; border-radius: 50%; animation: spin 0.8s linear infinite;"></div><div style="margin-top: 0.5rem; color: #6b7280;">Searching...</div></div>';
    }

    // Execute search
    await searchParty();
}

// Debounced version for typeahead
const debouncedTypeahead = debounce(typeaheadSearch, SEARCH_DEBOUNCE_MS);

// Export for global access
window.typeaheadSearch = typeaheadSearch;
window.debouncedTypeahead = debouncedTypeahead;

/**
 * Get CSRF token from cookie for API requests
 * @returns {string} CSRF token or empty string
 */
function getCsrfToken() {
    const match = document.cookie.match(/ws_csrf=([^;]+)/);
    return match ? match[1] : '';
}

// =================================================================
// NAVIGATION
// =================================================================

/**
 * Show a section by ID, hide all others
 * @param {string} section - Section ID (without 'section-' prefix)
 * @param {Event} evt - Click event (optional)
 */
function showSection(section, evt) {
    document.querySelectorAll('.section').forEach(s => s.style.display = 'none');
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

    // Clear dynamic result containers to prevent stale data from persisting
    const resultContainers = [
        'crossSellResults',
        'anomalyResults',
        'lendingResults',
        'reportResults'
    ];
    resultContainers.forEach(id => {
        const container = document.getElementById(id);
        if (container) {
            container.innerHTML = '<div class="results-empty">Enter details and click the action button</div>';
        }
    });

    const sectionEl = document.getElementById(`section-${section}`);
    if (sectionEl) {
        sectionEl.style.display = 'block';
    } else {
        console.error(`Section not found: section-${section}`);
    }

    if (evt && evt.target) {
        evt.target.closest('.nav-item').classList.add('active');
    }
    currentSection = section;
}

// =================================================================
// API USAGE & TOKEN MANAGEMENT
// =================================================================

/**
 * Fetch and display API usage statistics
 */
async function refreshUsage() {
    try {
        const resp = await fetch(`${API_BASE}/api/v1/usage`);
        const data = await resp.json();

        const count = data.total_calls;
        const remaining = data.calls_remaining;
        const percent = data.usage_percent;

        document.getElementById('apiCount').textContent = `${count}/1000`;
        document.getElementById('apiBar').style.width = `${percent}%`;

        // Color coding
        const bar = document.getElementById('apiBar');
        bar.classList.remove('warning', 'danger');
        if (percent > 80) bar.classList.add('danger');
        else if (percent > 50) bar.classList.add('warning');

        // Update stats
        document.getElementById('statTokens').textContent = data.token_refreshes;
        document.getElementById('statParty').textContent = data.party_calls;
        document.getElementById('statAccount').textContent = data.account_calls;
        document.getElementById('statTx').textContent = data.transaction_calls;

    } catch (e) {
        console.error('Failed to fetch usage:', e);
    }
}

/**
 * Fetch and display OAuth token status
 */
async function refreshTokenStatus() {
    try {
        const resp = await fetch(`${API_BASE}/api/v1/token`);
        const data = await resp.json();

        const dot = document.getElementById('tokenDot');
        const text = document.getElementById('tokenText');

        if (data.status === 'valid') {
            dot.classList.remove('expired');
            const expiresIn = Math.round(data.expires_in_seconds);
            text.textContent = `Token valid (${Math.floor(expiresIn / 60)}m ${expiresIn % 60}s)`;
        } else if (data.status === 'no_token') {
            dot.classList.add('expired');
            text.textContent = 'No token';
        } else {
            dot.classList.add('expired');
            text.textContent = 'Token expired';
        }
    } catch (e) {
        document.getElementById('tokenText').textContent = 'Offline';
        document.getElementById('tokenDot').classList.add('expired');
    }
}

/**
 * Refresh OAuth token
 */
async function refreshToken() {
    try {
        const resp = await fetch(`${API_BASE}/api/v1/token/refresh`, { method: 'POST' });
        const data = await resp.json();
        alert('Token refreshed successfully!');
        refreshTokenStatus();
        refreshUsage();
    } catch (e) {
        alert('Failed to refresh token: ' + e.message);
    }
}

// =================================================================
// CSRF TOKEN
// =================================================================

/**
 * Get CSRF token from cookie
 * @returns {string} CSRF token or empty string
 */
function getCsrfToken() {
    const match = document.cookie.match(/ws_csrf=([^;]+)/);
    return match ? match[1] : '';
}

// =================================================================
// FISERV PARTY/ACCOUNT FUNCTIONS
// =================================================================

/**
 * Search for party (member) in Fiserv
 */
async function searchParty() {
    const name = document.getElementById('partyName').value;
    const phone = document.getElementById('partyPhone').value;
    const email = document.getElementById('partyEmail').value;
    const accountId = document.getElementById('partyAccount').value;

    const payload = {};
    if (name) payload.name = name;
    if (phone) payload.phone = phone;
    if (email) payload.email = email;
    if (accountId) payload.account_id = accountId;

    try {
        const resp = await fetch(`${API_BASE}/api/v1/party/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRF-Token': getCsrfToken()
            },
            body: JSON.stringify(payload),
            credentials: 'include'
        });
        const data = await resp.json();

        const resultsDiv = document.getElementById('partyResults');
        const member360Panel = document.getElementById('member360Panel');

        if (data.success && data.parties && data.parties.length > 0) {
            document.getElementById('partyResultCount').textContent = `${data.count} members found`;

            // Render clickable member cards
            resultsDiv.innerHTML = data.parties.map(p => `
                <div class="result-item" 
                     style="cursor: pointer; padding: 1rem; border-radius: 8px; transition: all 0.2s; border: 1px solid #e5e7eb; margin-bottom: 0.5rem;"
                     onclick="selectMemberFromSearch('${p.party_id}', ${JSON.stringify(p).replace(/"/g, '&quot;')})"
                     onmouseover="this.style.background='#f0f9ff'; this.style.borderColor='#3b82f6'"
                     onmouseout="this.style.background='white'; this.style.borderColor='#e5e7eb'"
                     data-testid="member-result-${p.party_id}">
                    <div style="display: flex; align-items: center; gap: 1rem;">
                        <div style="width: 48px; height: 48px; background: linear-gradient(135deg, #6366f1, #4f46e5); color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600;">
                            ${(p.name || 'U').split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2)}
                        </div>
                        <div style="flex: 1;">
                            <div style="font-weight: 600; color: #1f2937;">${p.name || 'Unknown'}</div>
                            <div style="font-size: 0.85rem; color: #6b7280;">ID: ${p.party_id}</div>
                        </div>
                        <div style="text-align: right; font-size: 0.85rem;">
                            ${p.email ? `<div>📧 ${p.email}</div>` : ''}
                            ${p.phone ? `<div>📞 ${p.phone}</div>` : ''}
                        </div>
                        <div style="color: #3b82f6; font-size: 1.25rem;">→</div>
                    </div>
                </div>
            `).join('');

            // Hide Member 360 panel when showing new results
            if (member360Panel) {
                member360Panel.style.display = 'none';
            }
        } else {
            document.getElementById('partyResultCount').textContent = '0 members found';
            resultsDiv.innerHTML = `<div class="results-empty">No members found matching criteria<br><small>${data.error || data.raw?.message || ''}</small></div>`;
            if (member360Panel) {
                member360Panel.style.display = 'none';
            }
        }

        refreshUsage();
    } catch (e) {
        alert('Search failed: ' + e.message);
    }
}

/**
 * Select a member from search results and load Member 360
 * @param {string} partyId - Party ID
 * @param {Object} partyData - Party data from search
 */
function selectMemberFromSearch(partyId, partyData) {
    // Show the Member 360 panel
    const member360Panel = document.getElementById('member360Panel');
    if (member360Panel) {
        member360Panel.style.display = 'block';
    }

    // Convert party data to member format and load Member 360
    const memberData = {
        PartyId: partyData.party_id || partyId,
        id: partyData.party_id || partyId,
        FullName: partyData.name,
        name: partyData.name,
        Phone: partyData.phone,
        PhoneNumber: partyData.phone,
        Email: partyData.email,
        EmailAddress: partyData.email,
        Address: partyData.address,
        Status: 'Active',
        Tier: 'Standard',
        Accounts: [],
        Relationships: []
    };

    // Call the Member 360 render function
    if (typeof renderMember360 === 'function') {
        renderMember360(memberData);
    }

    // Add to recent members
    if (typeof addToRecentMembers === 'function') {
        addToRecentMembers({
            id: memberData.PartyId,
            name: memberData.FullName,
            accountId: ''
        });
    }

    // Scroll to Member 360 panel
    if (member360Panel) {
        member360Panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    console.log(`[Party] Selected member: ${memberData.FullName} (${partyId})`);
}

/**
 * Clear party search form
 */
function clearPartyForm() {
    document.getElementById('partyName').value = '';
    document.getElementById('partyPhone').value = '';
    document.getElementById('partyEmail').value = '';
    document.getElementById('partyAccount').value = '';
}

/**
 * Lookup account details in Fiserv
 */
async function lookupAccount() {
    const accountId = document.getElementById('accountId').value;
    const accountType = document.getElementById('accountType').value;

    if (!accountId) {
        alert('Please enter an account number');
        return;
    }

    try {
        const resp = await fetch(`${API_BASE}/api/v1/account/lookup`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRF-Token': getCsrfToken()
            },
            body: JSON.stringify({ account_id: accountId, account_type: accountType }),
            credentials: 'include'
        });
        const data = await resp.json();

        const resultsDiv = document.getElementById('accountResults');
        if (data.success) {
            const acct = data.account;
            resultsDiv.innerHTML = `
                <div class="result-item">
                    <strong>Account ${acct.account_id}</strong> (${acct.account_type})<br>
                    Status: ${acct.status || 'N/A'}<br>
                    Product: ${acct.product_id || 'N/A'}<br>
                    Opened: ${acct.open_date || 'N/A'}
                </div>
            `;
        } else {
            resultsDiv.innerHTML = `<div class="results-empty">Account not found: ${data.error || ''}</div>`;
        }

        refreshUsage();
    } catch (e) {
        alert('Lookup failed: ' + e.message);
    }
}

/**
 * List transactions for an account
 */
async function listTransactions() {
    const accountId = document.getElementById('txAccountId').value;
    const days = document.getElementById('txDays').value;

    if (!accountId) {
        alert('Please enter an account number');
        return;
    }

    try {
        const resp = await fetch(`${API_BASE}/api/v1/transactions/${accountId}?days=${days}`, {
            credentials: 'include'
        });
        const data = await resp.json();

        const resultsDiv = document.getElementById('txResults');
        if (data.success && data.transactions && data.transactions.length > 0) {
            resultsDiv.innerHTML = data.transactions.slice(0, 20).map(tx => `
                <div class="result-item">
                    <strong>${tx.date || 'N/A'}</strong> - ${tx.description || 'Transaction'}<br>
                    Amount: $${(tx.amount || 0).toFixed(2)} (${tx.dr_cr || 'N/A'})
                </div>
            `).join('');
        } else {
            resultsDiv.innerHTML = `<div class="results-empty">No transactions found</div>`;
        }

        refreshUsage();
    } catch (e) {
        alert('Failed to fetch transactions: ' + e.message);
    }
}

// =================================================================
// ROLE VIEW CONFIGURATION
// =================================================================

const roleViewConfig = {
    msr: {
        // ALL visible sidebar tabs + role-specific sections
        sections: ['party', 'account', 'transactions', 'transfers', 'cases', 'anomaly', 'ml-profit', 'realtime', 'executive-dashboard', 'status', 'role-insights'],
        defaultSection: 'party',
        sidebarTitle: '🏦 MSR Dashboard',
        themeColor: '#0074BA'
    },
    loan_officer: {
        // ALL visible sidebar tabs + role-specific sections
        sections: ['party', 'account', 'transactions', 'transfers', 'cases', 'anomaly', 'ml-profit', 'realtime', 'executive-dashboard', 'role-insights', 'ml-credit'],
        defaultSection: 'ml-profit',
        sidebarTitle: '💼 Loan Officer Dashboard',
        themeColor: '#10b981'
    },
    fraud_analyst: {
        // ALL visible sidebar tabs + role-specific sections
        sections: ['party', 'account', 'transactions', 'transfers', 'cases', 'anomaly', 'ml-profit', 'realtime', 'executive-dashboard', 'alerts'],
        defaultSection: 'anomaly',
        sidebarTitle: '🔍 Fraud Analyst Dashboard',
        themeColor: '#ef4444'
    },
    executive: {
        // ALL visible sidebar tabs + role-specific sections
        sections: ['party', 'account', 'transactions', 'transfers', 'cases', 'anomaly', 'ml-profit', 'realtime', 'executive-dashboard', 'branch-performance', 'qa-dashboard'],
        defaultSection: 'executive-dashboard',
        sidebarTitle: '📊 Executive Dashboard',
        themeColor: '#8b5cf6'
    },
    call_intelligence: {
        // ALL visible sidebar tabs + role-specific sections
        sections: ['party', 'account', 'transactions', 'transfers', 'cases', 'anomaly', 'ml-profit', 'realtime', 'executive-dashboard', 'call-portal', 'call-summary', 'call-problems', 'live-calls', 'qa-dashboard'],
        defaultSection: 'live-calls',
        sidebarTitle: '📞 Call Intelligence Dashboard',
        themeColor: '#06b6d4'
    }
};

/**
 * Switch role view - updates navigation, theme, and default section
 * @param {string} role - Role identifier
 */
function switchRoleView(role) {
    currentRole = role;
    const config = roleViewConfig[role];
    if (!config) return;

    // Update sidebar nav items visibility
    const allNavItems = document.querySelectorAll('.nav-item');
    allNavItems.forEach(item => {
        const onclick = item.getAttribute('onclick') || '';
        const match = onclick.match(/showSection\('([^']+)'/);
        if (match) {
            const sectionId = match[1];
            if (config.sections.includes(sectionId)) {
                item.style.display = 'flex';
            } else {
                item.style.display = 'none';
            }
        }
    });

    // Update theme color accent
    document.documentElement.style.setProperty('--role-accent', config.themeColor);

    // Update active tab logic with SLIDING ANIMATION
    moveRoleSlider(role);

    // Show default section for this role
    showSection(config.defaultSection);

    // Update active nav item
    allNavItems.forEach(item => item.classList.remove('active'));
    const defaultNav = document.querySelector(`[onclick*="showSection('${config.defaultSection}'"]`);
    if (defaultNav) defaultNav.classList.add('active');

    // Render role-specific visualizations if function exists
    if (typeof renderRoleVisualizations === 'function') {
        renderRoleVisualizations(role);
    }

    // Load real API data for role-specific sections
    if (role === 'fraud_analyst' && typeof loadFraudDashboardFromAPI === 'function') {
        setTimeout(() => loadFraudDashboardFromAPI(), 100);
    }

    if (role === 'executive' && typeof loadExecutiveDashboardFromAPI === 'function') {
        setTimeout(() => loadExecutiveDashboardFromAPI(), 100);
    }

    if (role === 'loan_officer' && typeof loadLoanOfficerDashboardFromAPI === 'function') {
        setTimeout(() => loadLoanOfficerDashboardFromAPI(), 100);
    }

    console.log(`[Role] Switched to ${role} view, loading real API data...`);
}

/**
 * [NEW] Moves the role slider to the active button
 * @param {string} role 
 */
function moveRoleSlider(role) {
    const activeTab = document.querySelector(`.role-tab[data-role="${role}"]`);
    const slider = document.getElementById('roleSlider');

    if (activeTab && slider) {
        // Update active class on buttons
        document.querySelectorAll('.role-tab').forEach(t => t.classList.remove('active'));
        activeTab.classList.add('active');

        // Move slider
        slider.style.width = `${activeTab.offsetWidth}px`;
        slider.style.left = `${activeTab.offsetLeft}px`;
    }
}

// Initialize slider position on load and handle URL hash navigation
document.addEventListener('DOMContentLoaded', () => {
    moveRoleSlider('msr');

    // Handle URL hash for deep linking from Fiserv page
    const hash = window.location.hash;
    if (hash && hash.startsWith('#section-')) {
        const sectionName = hash.replace('#section-', '');
        showSection(sectionName);

        // Find and activate the corresponding nav item
        const navItem = document.querySelector(`[onclick*="showSection('${sectionName}',"]`);
        if (navItem) {
            document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
            navItem.classList.add('active');
        }

        console.log(`[Navigation] Deep linked to section: ${sectionName}`);
    }
});

// Also handle hash changes for single-page navigation
window.addEventListener('hashchange', () => {
    const hash = window.location.hash;
    if (hash && hash.startsWith('#section-')) {
        const sectionName = hash.replace('#section-', '');
        showSection(sectionName);
    }
});

// Make functions globally available
window.showSection = showSection;
window.refreshUsage = refreshUsage;
window.refreshTokenStatus = refreshTokenStatus;
window.refreshToken = refreshToken;
window.getCsrfToken = getCsrfToken;
window.searchParty = searchParty;
window.selectMemberFromSearch = selectMemberFromSearch;
window.clearPartyForm = clearPartyForm;
window.lookupAccount = lookupAccount;
window.listTransactions = listTransactions;
window.switchRoleView = switchRoleView;
window.roleViewConfig = roleViewConfig;
