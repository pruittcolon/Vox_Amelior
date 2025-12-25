/**
 * Banking UI - handles view rendering and interactions
 * Complements banking_core.js (logic) and banking.css (visuals)
 */

document.addEventListener('DOMContentLoaded', () => {
    // Initialize
    refreshUsage();
    refreshTokenStatus();

    // Check for active role in localStorage or default
    const savedRole = localStorage.getItem('fiserv_role') || 'msr';

    // Preserve hash navigation - don't switch if deep-linking
    const hash = window.location.hash;
    if (!hash || !hash.startsWith('#section-')) {
        switchRoleView(savedRole);
    } else {
        // Ensure theme colors are still applied even if we skip full switch
        const config = roleViewConfig[savedRole];
        if (config) {
            document.documentElement.style.setProperty('--role-accent', config.themeColor);
            currentRole = savedRole;
            moveRoleSlider(savedRole);
        }
    }

    // Initial load for active section
    const currentNav = document.querySelector('.nav-item.active');
    if (currentNav) {
        const onclick = currentNav.getAttribute('onclick');
        if (onclick) {
            const match = onclick.match(/showSection\('([^']+)'/);
            if (match) showSection(match[1]);
        }
    }
});

// Load Gemma Chat if not already loaded
function ensureGemmaLoaded() {
    // Logic to init Gemma chat would go here
    console.log('Gemma Chat initialized');
}

// Additional UI-specific helpers
function toggleSpinner(id, show) {
    const el = document.getElementById(id);
    if (el) el.style.display = show ? 'inline-block' : 'none';
}

function showToast(message, type = 'info') {
    // Simple alert for now, could be upgraded to a toast notification
    alert(message);
}

// =============================================================================
// MEMBER 360 VISUALIZATION (Phase 2)
// =============================================================================

/**
 * Render Member 360 panel with premium glassmorphism styling
 * @param {Object} member - Member data object
 */
function renderMember360(member) {
    const panel = document.getElementById('member360Panel');
    if (!panel) return;

    // Calculate mock health score based on available data
    const healthScore = calculateHealthScore(member);
    const healthColor = healthScore >= 70 ? '#10b981' : healthScore >= 40 ? '#f59e0b' : '#ef4444';

    // Generate initials for avatar
    const initials = (member.FullName || member.name || 'U')
        .split(' ')
        .map(n => n[0])
        .join('')
        .toUpperCase()
        .slice(0, 2);

    panel.innerHTML = `
        <div class="glass-card member-360-container" style="
            background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 24px;
            padding: 2rem;
            animation: fadeInUp 0.4s ease;
        ">
            <!-- Header with Avatar and Quick Actions -->
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 2rem;">
                <div style="display: flex; align-items: center; gap: 1.5rem;">
                    <div class="member-avatar" style="
                        width: 80px; height: 80px;
                        background: linear-gradient(135deg, #6366f1, #8b5cf6);
                        border-radius: 50%;
                        display: flex; align-items: center; justify-content: center;
                        font-size: 1.75rem; font-weight: 700; color: white;
                        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.3);
                    ">${initials}</div>
                    <div>
                        <h2 style="font-size: 1.75rem; font-weight: 700; color: var(--text-primary); margin: 0;">
                            ${member.FullName || member.name || 'Unknown Member'}
                        </h2>
                        <p style="color: var(--text-secondary); margin: 0.25rem 0 0;">
                            ID: ${member.PartyId || member.id} • ${member.Status || 'Active'} • ${member.Tier || 'Standard'}
                        </p>
                    </div>
                </div>
                <div style="display: flex; gap: 0.75rem;">
                    <button class="btn btn-secondary" style="padding: 0.5rem 1rem;" onclick="showToast('Opening contact...')">
                        📞 Call
                    </button>
                    <button class="btn btn-secondary" style="padding: 0.5rem 1rem;" onclick="showToast('Opening email...')">
                        ✉️ Email
                    </button>
                    <button class="btn btn-primary" style="padding: 0.5rem 1rem;" onclick="loadMemberAccounts('${member.PartyId || member.id}')">
                        🏦 View Accounts
                    </button>
                </div>
            </div>

            <!-- Stats Grid -->
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.5rem; margin-bottom: 2rem;">
                <!-- Health Score Card -->
                <div class="stat-card glass-card" style="
                    padding: 1.25rem; text-align: center;
                    background: rgba(255,255,255,0.05);
                    border: 1px solid rgba(255,255,255,0.1);
                ">
                    <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-tertiary); margin-bottom: 0.5rem;">
                        Health Score
                    </div>
                    <div style="position: relative; width: 80px; height: 80px; margin: 0 auto;">
                        <svg viewBox="0 0 100 100" style="transform: rotate(-90deg);">
                            <circle cx="50" cy="50" r="40" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="8"/>
                            <circle cx="50" cy="50" r="40" fill="none" stroke="${healthColor}" stroke-width="8"
                                stroke-dasharray="${healthScore * 2.51} 251"
                                stroke-linecap="round"
                                style="transition: stroke-dasharray 1s ease;"/>
                        </svg>
                        <div style="
                            position: absolute; inset: 0;
                            display: flex; align-items: center; justify-content: center;
                            font-size: 1.25rem; font-weight: 700; color: ${healthColor};
                        ">${healthScore}</div>
                    </div>
                </div>

                <!-- Contact Info Card -->
                <div class="stat-card glass-card" style="padding: 1.25rem; background: rgba(255,255,255,0.05);">
                    <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-tertiary); margin-bottom: 0.75rem;">
                        Contact
                    </div>
                    <div style="font-size: 0.9rem; color: var(--text-secondary);">
                        ${member.Phone || member.PhoneNumber || 'No phone'}<br>
                        ${member.Email || member.EmailAddress || 'No email'}
                    </div>
                </div>

                <!-- Address Card -->
                <div class="stat-card glass-card" style="padding: 1.25rem; background: rgba(255,255,255,0.05);">
                    <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-tertiary); margin-bottom: 0.75rem;">
                        Address
                    </div>
                    <div style="font-size: 0.9rem; color: var(--text-secondary);">
                        ${member.Address || 'No address on file'}
                    </div>
                </div>

                <!-- Products Card -->
                <div class="stat-card glass-card" style="padding: 1.25rem; background: rgba(255,255,255,0.05);">
                    <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-tertiary); margin-bottom: 0.75rem;">
                        Products
                    </div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: var(--text-accent);">
                        ${(member.Accounts || []).length || '--'}
                    </div>
                    <div style="font-size: 0.85rem; color: var(--text-tertiary);">Active accounts</div>
                </div>
            </div>

            <!-- Account Cards Section -->
            <div id="member360Accounts" style="margin-top: 1.5rem;">
                <div style="font-size: 0.85rem; color: var(--text-tertiary); text-align: center; padding: 1rem;">
                    Click "View Accounts" to load account details
                </div>
            </div>
        </div>
    `;

    panel.style.display = 'block';
}

/**
 * Calculate financial health score (0-100)
 * @param {Object} member - Member data
 * @returns {number} Health score
 */
function calculateHealthScore(member) {
    let score = 50; // Base score

    // Boost for having contact info
    if (member.Phone || member.PhoneNumber) score += 10;
    if (member.Email || member.EmailAddress) score += 10;
    if (member.Address) score += 5;

    // Boost for active accounts
    const accountCount = (member.Accounts || []).length;
    score += Math.min(accountCount * 5, 20);

    // Cap at 100
    return Math.min(score, 100);
}

/**
 * Load member accounts from API
 * @param {string} partyId - Member party ID
 */
async function loadMemberAccounts(partyId) {
    const container = document.getElementById('member360Accounts');
    if (!container) return;

    container.innerHTML = '<div style="text-align: center; padding: 1rem; color: var(--text-tertiary);">Loading accounts...</div>';

    try {
        const resp = await fetch(`/fiserv/api/v1/party/${partyId}/accounts`, {
            headers: { 'X-CSRF-Token': getCsrfToken() },
            credentials: 'include'
        });
        const data = await resp.json();

        if (data.success && data.accounts && data.accounts.length > 0) {
            container.innerHTML = `
                <div style="font-size: 0.85rem; font-weight: 600; color: var(--text-secondary); margin-bottom: 1rem;">
                    ${data.accounts.length} Account(s) Found
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem;">
                    ${data.accounts.map(acc => `
                        <div class="glass-card" style="padding: 1rem; background: rgba(255,255,255,0.05); cursor: pointer; transition: all 0.2s;"
                             onclick="lookupAccountFromId('${acc.account_id || acc.AcctId}')"
                             onmouseover="this.style.transform='translateY(-2px)'; this.style.borderColor='var(--text-accent)';"
                             onmouseout="this.style.transform='translateY(0)'; this.style.borderColor='rgba(255,255,255,0.1)';">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                <span style="font-weight: 600; color: var(--text-primary);">${acc.account_type || acc.AcctType || 'Account'}</span>
                                <span style="font-size: 0.85rem; color: var(--text-tertiary);">${acc.account_id || acc.AcctId}</span>
                            </div>
                            <div style="font-size: 1.25rem; font-weight: 700; color: var(--text-accent);">
                                $${(acc.balance || acc.Available || 0).toLocaleString('en-US', { minimumFractionDigits: 2 })}
                            </div>
                        </div>
                    `).join('')}
                </div>
            `;
        } else {
            container.innerHTML = '<div style="text-align: center; padding: 1rem; color: var(--text-tertiary);">No accounts found for this member</div>';
        }
    } catch (e) {
        container.innerHTML = `<div style="text-align: center; padding: 1rem; color: var(--neon-rose);">Failed to load accounts: ${e.message}</div>`;
    }
}

// Export for global access
window.renderMember360 = renderMember360;
window.loadMemberAccounts = loadMemberAccounts;
window.calculateHealthScore = calculateHealthScore;
