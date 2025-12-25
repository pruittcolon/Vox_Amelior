/**
 * Banking Member Module - Phase 1 MSR Features
 * =============================================
 * Member 360 view, notes, recent members, transaction filters, CSV export
 * 
 * Dependencies: banking_core.js (for getCsrfToken, API_BASE)
 * Used by: banking.html
 * 
 * @module banking_member
 * @version 1.0.0
 */

// =================================================================
// STATE MANAGEMENT
// =================================================================

/**
 * Global state for member-related data
 * @type {Object}
 */
const MemberState = {
    currentMember: null,
    recentMembers: JSON.parse(localStorage.getItem('scuRecentMembers') || '[]'),
    memberNotes: {},
    transactionFilters: {
        dateFrom: null,
        dateTo: null,
        minAmount: null,
        maxAmount: null,
        type: 'all'
    },
    transactions: []
};

// =================================================================
// RECENT MEMBERS
// =================================================================

/**
 * Add member to recent list (max 10, most recent first)
 * @param {Object} member - Member object with id, name, accountId
 */
function addToRecentMembers(member) {
    if (!member || !member.id) return;

    // Remove duplicate if exists
    MemberState.recentMembers = MemberState.recentMembers.filter(m => m.id !== member.id);

    // Add to front
    MemberState.recentMembers.unshift({
        id: member.id,
        name: member.name || `Member ${member.id}`,
        accountId: member.accountId || '',
        accessedAt: new Date().toISOString()
    });

    // Keep only last 10
    MemberState.recentMembers = MemberState.recentMembers.slice(0, 10);

    // Persist to localStorage
    localStorage.setItem('scuRecentMembers', JSON.stringify(MemberState.recentMembers));

    // Update UI
    renderRecentMembers();
}

/**
 * Render recent members list in sidebar
 */
function renderRecentMembers() {
    const container = document.getElementById('recentMembersPanel');
    if (!container) return;

    if (MemberState.recentMembers.length === 0) {
        container.innerHTML = '<div class="empty-state">No recent members</div>';
        return;
    }

    container.innerHTML = MemberState.recentMembers.map((m, idx) => `
        <div class="recent-member-item" onclick="loadMember('${m.id}')" data-testid="recent-member-${idx}">
            <div class="recent-member-name">${escapeHtml(m.name)}</div>
            <div class="recent-member-time">${formatRelativeTime(m.accessedAt)}</div>
        </div>
    `).join('');
}

// =================================================================
// MEMBER 360 VIEW
// =================================================================

/**
 * Load and display Member 360 summary
 * @param {string} memberId - Member ID to load
 */
async function loadMember360(memberId) {
    if (!memberId) {
        console.warn('[Member360] No member ID provided');
        return;
    }

    const container = document.getElementById('member360Panel');
    if (container) {
        container.innerHTML = '<div class="loading-state">Loading member data...</div>';
    }

    try {
        const csrfToken = typeof getCsrfToken === 'function' ? getCsrfToken() : '';
        const response = await fetch(`/fiserv/api/v1/party/search?name=${encodeURIComponent(memberId)}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRF-Token': csrfToken
            },
            credentials: 'include'
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const data = await response.json();
        const member = data.parties?.[0] || data.data?.[0] || data;

        MemberState.currentMember = member;
        addToRecentMembers({
            id: member.PartyId || member.id || memberId,
            name: member.FullName || member.name || `Member ${memberId}`,
            accountId: member.PrimaryAccountId || ''
        });

        renderMember360(member);

    } catch (error) {
        console.error('[Member360] Error loading member:', error);
        if (container) {
            container.innerHTML = `<div class="error-state">Error loading member: ${escapeHtml(error.message)}</div>`;
        }
    }
}

/**
 * Render Member 360 summary panel - Enterprise Enhanced Version
 * @param {Object} member - Member data object
 */
function renderMember360(member) {
    const container = document.getElementById('member360Panel');
    if (!container) return;

    const name = member.FullName || member.name || 'Unknown';
    const phone = member.Phone || member.PhoneNumber || 'N/A';
    const email = member.Email || member.EmailAddress || 'N/A';
    const memberId = member.PartyId || member.id || '';
    const status = member.Status || 'Active';
    const tier = member.Tier || member.tier || 'Standard';

    // Calculate health score (0-100 based on available data)
    const healthScore = calculateMemberHealthScore(member);
    const healthClass = healthScore >= 70 ? 'good' : healthScore >= 40 ? 'warning' : 'danger';
    const healthLabel = healthScore >= 70 ? 'Healthy' : healthScore >= 40 ? 'At Risk' : 'Critical';

    // Get accounts summary
    const accounts = member.Accounts || member.accounts || [];
    const totalBalance = accounts.reduce((sum, acc) => sum + (acc.Balance || acc.balance || 0), 0);
    const accountCount = accounts.length || '—';

    // Get relationships
    const relationships = member.Relationships || member.relationships || [];

    // Next Best Action (would come from ML service)
    const nba = member.NextBestAction || getDefaultNBA(member);

    container.innerHTML = `
        <div class="member-360-card" data-testid="member-360-card">
            <!-- Header with Avatar and Key Info -->
            <div class="member-360-header" style="display: flex; gap: 1.5rem; margin-bottom: 1.5rem; padding-bottom: 1.5rem; border-bottom: 1px solid #e5e7eb;">
                <div class="member-avatar" style="width: 72px; height: 72px; background: linear-gradient(135deg, #6366f1, #4f46e5); color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; font-weight: 700;">
                    ${getInitials(name)}
                </div>
                <div class="member-info" style="flex: 1;">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <div>
                            <h2 class="member-name" style="margin: 0 0 0.25rem 0; font-size: 1.4rem; font-weight: 700; color: #1f2937;">${escapeHtml(name)}</h2>
                            <div class="member-id" style="font-size: 0.85rem; color: #6b7280;">ID: ${escapeHtml(memberId)}</div>
                        </div>
                        <div style="text-align: right;">
                            <span class="member-status status-${status.toLowerCase()}" style="display: inline-block; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600; background: ${status === 'Active' ? '#dcfce7' : '#fef3c7'}; color: ${status === 'Active' ? '#166534' : '#92400e'};">
                                ${escapeHtml(status)}
                            </span>
                            <div style="margin-top: 0.5rem; font-size: 0.75rem; padding: 0.15rem 0.5rem; background: #f3f4f6; border-radius: 8px; color: #374151;">
                                ⭐ ${escapeHtml(tier)} Member
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Quick Stats Row -->
            <div class="member-360-stats" style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem;">
                <div style="background: linear-gradient(135deg, #6366f1, #4f46e5); color: white; padding: 1rem; border-radius: 12px; text-align: center;">
                    <div style="font-size: 0.75rem; opacity: 0.9;">Products</div>
                    <div style="font-size: 1.5rem; font-weight: 700;">${accountCount}</div>
                </div>
                <div style="background: linear-gradient(135deg, #10b981, #059669); color: white; padding: 1rem; border-radius: 12px; text-align: center;">
                    <div style="font-size: 0.75rem; opacity: 0.9;">Total Balance</div>
                    <div style="font-size: 1.25rem; font-weight: 700;">${formatCurrency(totalBalance)}</div>
                </div>
                <div style="background: ${healthClass === 'good' ? 'linear-gradient(135deg, #22c55e, #16a34a)' : healthClass === 'warning' ? 'linear-gradient(135deg, #f59e0b, #d97706)' : 'linear-gradient(135deg, #ef4444, #dc2626)'}; color: white; padding: 1rem; border-radius: 12px; text-align: center;">
                    <div style="font-size: 0.75rem; opacity: 0.9;">Health Score</div>
                    <div style="font-size: 1.5rem; font-weight: 700;">${healthScore}</div>
                </div>
                <div style="background: linear-gradient(135deg, #8b5cf6, #7c3aed); color: white; padding: 1rem; border-radius: 12px; text-align: center;">
                    <div style="font-size: 0.75rem; opacity: 0.9;">Tenure</div>
                    <div style="font-size: 1.25rem; font-weight: 700;">${member.TenureYears || member.tenure || '—'}y</div>
                </div>
            </div>

            <!-- Contact Details -->
            <div class="member-360-details" style="background: #f8fafc; border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem;">
                <div style="font-weight: 600; margin-bottom: 0.75rem; color: #374151;">📋 Contact Information</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem;">
                    <div class="detail-row" style="display: flex; justify-content: space-between;">
                        <span class="detail-label" style="color: #6b7280;">📞 Phone</span>
                        <span class="detail-value" style="font-weight: 500; color: #1f2937;">${escapeHtml(phone)}</span>
                    </div>
                    <div class="detail-row" style="display: flex; justify-content: space-between;">
                        <span class="detail-label" style="color: #6b7280;">📧 Email</span>
                        <span class="detail-value" style="font-weight: 500; color: #1f2937;">${escapeHtml(email)}</span>
                    </div>
                    <div class="detail-row" style="display: flex; justify-content: space-between;">
                        <span class="detail-label" style="color: #6b7280;">📍 Address</span>
                        <span class="detail-value" style="font-weight: 500; color: #1f2937;">${escapeHtml(member.Address || member.address || 'N/A')}</span>
                    </div>
                    <div class="detail-row" style="display: flex; justify-content: space-between;">
                        <span class="detail-label" style="color: #6b7280;">🎂 DOB</span>
                        <span class="detail-value" style="font-weight: 500; color: #1f2937;">${member.DateOfBirth ? formatDate(member.DateOfBirth) : 'N/A'}</span>
                    </div>
                </div>
            </div>

            <!-- Accounts/Products Section -->
            <div class="member-360-accounts" style="background: #f8fafc; border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem;" id="member360Accounts">
                <div style="font-weight: 600; margin-bottom: 0.75rem; color: #374151;">🏦 Products & Accounts</div>
                <div id="member360AccountsList">
                    ${accounts.length > 0 ? renderAccountsList(accounts) : '<div style="text-align: center; color: #6b7280; padding: 1rem;">Loading accounts...</div>'}
                </div>
            </div>

            <!-- Relationships Section -->
            ${relationships.length > 0 ? `
            <div class="member-360-relationships" style="background: #f8fafc; border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem;">
                <div style="font-weight: 600; margin-bottom: 0.75rem; color: #374151;">👥 Household & Relationships</div>
                <div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">
                    ${relationships.map(rel => `
                        <div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.5rem 0.75rem; background: white; border-radius: 8px; border: 1px solid #e5e7eb;">
                            <span style="font-weight: 500;">${escapeHtml(rel.Name || rel.name)}</span>
                            <span style="font-size: 0.75rem; padding: 0.15rem 0.5rem; background: #e0e7ff; color: #4338ca; border-radius: 4px;">${escapeHtml(rel.Type || rel.type)}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
            ` : ''}

            <!-- Next Best Action -->
            <div class="member-360-nba" style="background: linear-gradient(135deg, #dbeafe, #e0e7ff); border: 2px solid #6366f1; border-radius: 12px; padding: 1rem; margin-bottom: 1.5rem;">
                <div style="display: flex; align-items: start; gap: 1rem;">
                    <div style="font-size: 2rem;">🎯</div>
                    <div style="flex: 1;">
                        <div style="font-weight: 700; color: #4338ca; margin-bottom: 0.25rem;">Next Best Action</div>
                        <div style="font-size: 1.1rem; font-weight: 600; color: #1f2937; margin-bottom: 0.5rem;">${escapeHtml(nba.action)}</div>
                        <div style="font-size: 0.9rem; color: #4b5563; margin-bottom: 0.75rem;">${escapeHtml(nba.reason)}</div>
                        <div style="display: flex; gap: 0.5rem;">
                            <span style="font-size: 0.75rem; padding: 0.25rem 0.5rem; background: #4f46e5; color: white; border-radius: 4px;">${nba.confidence}% confidence</span>
                            <span style="font-size: 0.75rem; padding: 0.25rem 0.5rem; background: #10b981; color: white; border-radius: 4px;">Est. value: ${formatCurrency(nba.value)}</span>
                        </div>
                    </div>
                    <button class="btn btn-primary btn-sm" onclick="executeNBA('${memberId}', '${nba.action}')" style="white-space: nowrap;">
                        Present Offer
                    </button>
                </div>
            </div>

            <!-- Actions -->
            <div class="member-360-actions" style="display: flex; flex-wrap: wrap; gap: 0.5rem; padding-top: 1rem; border-top: 1px solid #e5e7eb;">
                <button class="btn btn-sm" onclick="showMemberNotes('${memberId}')" style="background: #f3f4f6; color: #374151; padding: 0.5rem 1rem; border-radius: 8px; border: none; cursor: pointer;">📝 Notes</button>
                <button class="btn btn-sm" onclick="loadMemberTransactions('${memberId}')" style="background: #f3f4f6; color: #374151; padding: 0.5rem 1rem; border-radius: 8px; border: none; cursor: pointer;">💳 Transactions</button>
                <button class="btn btn-sm" onclick="showMemberTimeline('${memberId}')" style="background: #f3f4f6; color: #374151; padding: 0.5rem 1rem; border-radius: 8px; border: none; cursor: pointer;">📅 Timeline</button>
                <button class="btn btn-sm" onclick="askGemmaAboutMember('${memberId}')" style="background: #6366f1; color: white; padding: 0.5rem 1rem; border-radius: 8px; border: none; cursor: pointer;">🤖 Ask Gemma</button>
            </div>
        </div>
    `;

    // Load accounts asynchronously if not already loaded
    if (accounts.length === 0) {
        loadMemberAccounts(memberId);
    }
}

/**
 * Calculate member health score based on available data
 * @param {Object} member - Member data
 * @returns {number} Health score 0-100
 */
function calculateMemberHealthScore(member) {
    let score = 50; // Base score

    // Positive factors
    if (member.Status === 'Active') score += 15;
    if (member.TenureYears > 5) score += 10;
    if ((member.Accounts || []).length > 2) score += 10;
    if (member.DigitalEnrolled) score += 5;
    if (!member.DelinquentFlag) score += 10;

    // Negative factors
    if (member.ChurnRisk > 0.5) score -= 20;
    if (member.RecentComplaint) score -= 10;
    if (member.OverdraftCount > 3) score -= 15;

    return Math.max(0, Math.min(100, Math.round(score)));
}

/**
 * Get default Next Best Action based on member profile
 * @param {Object} member - Member data
 * @returns {Object} NBA recommendation
 */
function getDefaultNBA(member) {
    const accounts = member.Accounts || [];
    const hasChecking = accounts.some(a => a.Type === 'DDA' || a.type === 'checking');
    const hasCreditCard = accounts.some(a => a.Type === 'CARD' || a.type === 'credit_card');
    const hasLoan = accounts.some(a => a.Type === 'LOAN' || a.type === 'loan');

    if (!hasCreditCard) {
        return {
            action: 'Credit Card Upgrade',
            reason: 'Member has checking account with consistent deposits but no credit card. High propensity for SCU Visa.',
            confidence: 78,
            value: 1200
        };
    }
    if (!hasLoan && (member.TenureYears || 0) > 2) {
        return {
            action: 'Personal Loan Offer',
            reason: 'Long-tenured member with good standing. Pre-approved for personal loan up to $15,000.',
            confidence: 65,
            value: 2500
        };
    }
    return {
        action: 'Financial Review',
        reason: 'Schedule a financial wellness check-in to discuss member goals and product optimization.',
        confidence: 55,
        value: 500
    };
}

/**
 * Render accounts list for Member 360
 * @param {Array} accounts - Array of account objects
 * @returns {string} HTML for accounts list
 */
function renderAccountsList(accounts) {
    if (!accounts || accounts.length === 0) {
        return '<div style="text-align: center; color: #6b7280; padding: 1rem;">No accounts found</div>';
    }

    return accounts.map(acc => {
        const accType = acc.Type || acc.type || 'Unknown';
        const accNum = acc.AccountNumber || acc.accountNumber || acc.id || '****';
        const balance = acc.Balance || acc.balance || 0;
        const typeIcon = getAccountTypeIcon(accType);
        const typeLabel = getAccountTypeLabel(accType);

        return `
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.75rem; background: white; border-radius: 8px; margin-bottom: 0.5rem; border: 1px solid #e5e7eb;">
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <span style="font-size: 1.25rem;">${typeIcon}</span>
                    <div>
                        <div style="font-weight: 500; color: #1f2937;">${typeLabel}</div>
                        <div style="font-size: 0.8rem; color: #6b7280;">****${accNum.toString().slice(-4)}</div>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-weight: 600; color: ${balance >= 0 ? '#10b981' : '#ef4444'};">${formatCurrency(balance)}</div>
                    <div style="font-size: 0.75rem; color: #6b7280;">Available</div>
                </div>
            </div>
        `;
    }).join('');
}

/**
 * Get icon for account type
 * @param {string} type - Account type code
 * @returns {string} Emoji icon
 */
function getAccountTypeIcon(type) {
    const icons = {
        'DDA': '💳', 'checking': '💳',
        'SDA': '💰', 'savings': '💰',
        'CDA': '📜', 'certificate': '📜',
        'LOAN': '🏦', 'loan': '🏦',
        'CARD': '💳', 'credit_card': '💳',
        'MORTGAGE': '🏠', 'mortgage': '🏠'
    };
    return icons[type] || '📋';
}

/**
 * Get display label for account type
 * @param {string} type - Account type code
 * @returns {string} Display label
 */
function getAccountTypeLabel(type) {
    const labels = {
        'DDA': 'Checking', 'checking': 'Checking',
        'SDA': 'Savings', 'savings': 'Savings',
        'CDA': 'Certificate', 'certificate': 'Certificate',
        'LOAN': 'Loan', 'loan': 'Loan',
        'CARD': 'Credit Card', 'credit_card': 'Credit Card',
        'MORTGAGE': 'Mortgage', 'mortgage': 'Mortgage'
    };
    return labels[type] || type;
}

/**
 * Load member accounts from Fiserv API
 * @param {string} memberId - Member ID
 */
async function loadMemberAccounts(memberId) {
    try {
        const csrfToken = typeof getCsrfToken === 'function' ? getCsrfToken() : '';
        const response = await fetch(`/fiserv/accounts/${encodeURIComponent(memberId)}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRF-Token': csrfToken
            },
            credentials: 'include'
        });

        if (response.ok) {
            const data = await response.json();
            const accounts = data.accounts || data.data || [];

            if (MemberState.currentMember) {
                MemberState.currentMember.Accounts = accounts;
            }

            const container = document.getElementById('member360AccountsList');
            if (container) {
                container.innerHTML = renderAccountsList(accounts);
            }
        }
    } catch (error) {
        console.warn('[Member360] Could not load accounts:', error);
    }
}

/**
 * Show member timeline (activity history)
 * @param {string} memberId - Member ID
 */
function showMemberTimeline(memberId) {
    const container = document.getElementById('partyResults') || document.getElementById('member360Panel');
    if (!container) return;

    // Generate sample timeline events (would come from API in production)
    const events = generateSampleTimeline(memberId);

    container.innerHTML = `
        <div class="member-timeline" style="padding: 1rem;">
            <h3 style="margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
                📅 Activity Timeline
                <button class="btn btn-sm" onclick="loadMember360('${memberId}')" style="margin-left: auto; font-size: 0.75rem;">← Back</button>
            </h3>
            <div class="timeline-events" style="border-left: 2px solid #e5e7eb; padding-left: 1.5rem; margin-left: 0.5rem;">
                ${events.map(e => `
                    <div class="timeline-event" style="position: relative; padding-bottom: 1.5rem;">
                        <div style="position: absolute; left: -2rem; top: 0; width: 12px; height: 12px; background: ${e.color}; border-radius: 50%; border: 2px solid white;"></div>
                        <div style="font-size: 0.75rem; color: #6b7280; margin-bottom: 0.25rem;">${formatDate(e.date)}</div>
                        <div style="font-weight: 500; color: #1f2937;">${escapeHtml(e.title)}</div>
                        <div style="font-size: 0.9rem; color: #4b5563;">${escapeHtml(e.description)}</div>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
}

/**
 * Generate sample timeline events (demo data)
 * @param {string} memberId - Member ID
 * @returns {Array} Timeline events
 */
function generateSampleTimeline(memberId) {
    return [
        { date: new Date().toISOString(), title: 'Called Member Services', description: 'Inquiry about auto loan rates', color: '#3b82f6' },
        { date: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(), title: 'Mobile Deposit', description: 'Check deposit: $1,250.00', color: '#10b981' },
        { date: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(), title: 'Credit Card Payment', description: 'Auto-pay processed: $450.00', color: '#10b981' },
        { date: new Date(Date.now() - 14 * 24 * 60 * 60 * 1000).toISOString(), title: 'Address Updated', description: 'New address: 123 Main St, Portsmouth NH', color: '#6366f1' },
        { date: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(), title: 'Account Opened', description: 'High-Yield Savings opened with $500 deposit', color: '#8b5cf6' }
    ];
}

/**
 * Execute Next Best Action
 * @param {string} memberId - Member ID
 * @param {string} action - Action to execute
 */
function executeNBA(memberId, action) {
    console.log(`[NBA] Executing ${action} for member ${memberId}`);
    alert(`Opening ${action} workflow for member ${memberId}...\\n\\nIn production, this would open the appropriate product application or referral workflow.`);
}

/**
 * Ask Gemma AI about a member
 * @param {string} memberId - Member ID
 */
function askGemmaAboutMember(memberId) {
    // Check if Gemma chat function exists
    if (typeof askGemmaQuestion === 'function') {
        askGemmaQuestion(`Summarize member ${memberId}'s profile, account health, and provide recommendations.`);
    } else {
        alert('Gemma AI is not available. Please ensure the AI service is running.');
    }
}

// =================================================================
// MEMBER NOTES
// =================================================================

/**
 * Load notes for a member from localStorage
 * @param {string} memberId - Member ID
 * @returns {Array} Array of note objects
 */
function getMemberNotes(memberId) {
    const notes = JSON.parse(localStorage.getItem(`scuNotes_${memberId}`) || '[]');
    return notes;
}

/**
 * Save a note for a member
 * @param {string} memberId - Member ID
 * @param {string} text - Note text
 * @param {string} tag - Optional tag (general, important, followup)
 */
function saveMemberNote(memberId, text, tag = 'general') {
    if (!memberId || !text.trim()) return;

    const notes = getMemberNotes(memberId);
    notes.unshift({
        id: Date.now().toString(),
        text: text.trim(),
        tag: tag,
        createdAt: new Date().toISOString(),
        createdBy: 'current_user' // Would be replaced with actual user
    });

    localStorage.setItem(`scuNotes_${memberId}`, JSON.stringify(notes));
    renderMemberNotes(memberId);
}

/**
 * Show notes panel for a member
 * @param {string} memberId - Member ID
 */
function showMemberNotes(memberId) {
    const notes = getMemberNotes(memberId);
    const container = document.getElementById('memberNotesPanel') ||
        document.getElementById('partyResults');

    if (!container) return;

    container.innerHTML = `
        <div class="notes-panel" data-testid="notes-panel">
            <h3>Member Notes</h3>
            <div class="note-input-group">
                <textarea id="newNoteText" class="form-input" placeholder="Add a note..." rows="3"></textarea>
                <div class="note-input-actions">
                    <select id="noteTag" class="form-input form-input-sm">
                        <option value="general">General</option>
                        <option value="important">⚠️ Important</option>
                        <option value="followup">📌 Follow-up</option>
                    </select>
                    <button class="btn btn-primary btn-sm" onclick="saveMemberNote('${memberId}', document.getElementById('newNoteText').value, document.getElementById('noteTag').value)">
                        Add Note
                    </button>
                </div>
            </div>
            <div class="notes-list" data-testid="notes-list">
                ${notes.length === 0 ? '<div class="empty-state">No notes yet</div>' :
            notes.map(n => `
                    <div class="note-item note-${n.tag}">
                        <div class="note-header">
                            <span class="note-tag">${getTagLabel(n.tag)}</span>
                            <span class="note-date">${formatRelativeTime(n.createdAt)}</span>
                        </div>
                        <div class="note-text">${escapeHtml(n.text)}</div>
                    </div>
                  `).join('')}
            </div>
        </div>
    `;
}

/**
 * Render notes for a member
 * @param {string} memberId - Member ID
 */
function renderMemberNotes(memberId) {
    showMemberNotes(memberId); // Re-render the notes panel
}

// =================================================================
// TRANSACTION FILTERS
// =================================================================

/**
 * Apply transaction filters and re-render
 */
function applyTransactionFilters() {
    const dateFrom = document.getElementById('filterDateFrom')?.value || null;
    const dateTo = document.getElementById('filterDateTo')?.value || null;
    const minAmount = parseFloat(document.getElementById('filterMinAmount')?.value) || null;
    const maxAmount = parseFloat(document.getElementById('filterMaxAmount')?.value) || null;
    const type = document.getElementById('filterType')?.value || 'all';

    MemberState.transactionFilters = { dateFrom, dateTo, minAmount, maxAmount, type };

    const filtered = filterTransactions(MemberState.transactions, MemberState.transactionFilters);
    renderFilteredTransactions(filtered);
}

/**
 * Filter transactions based on criteria
 * @param {Array} transactions - Array of transactions
 * @param {Object} filters - Filter criteria
 * @returns {Array} Filtered transactions
 */
function filterTransactions(transactions, filters) {
    return transactions.filter(tx => {
        const txDate = new Date(tx.PostedDate || tx.date);
        const txAmount = Math.abs(tx.Amount || tx.amount || 0);
        const txType = tx.Type || tx.type || 'unknown';

        if (filters.dateFrom && txDate < new Date(filters.dateFrom)) return false;
        if (filters.dateTo && txDate > new Date(filters.dateTo)) return false;
        if (filters.minAmount && txAmount < filters.minAmount) return false;
        if (filters.maxAmount && txAmount > filters.maxAmount) return false;
        if (filters.type !== 'all' && txType.toLowerCase() !== filters.type.toLowerCase()) return false;

        return true;
    });
}

/**
 * Render filtered transactions
 * @param {Array} transactions - Filtered transactions
 */
function renderFilteredTransactions(transactions) {
    const container = document.getElementById('txResults');
    if (!container) return;

    if (transactions.length === 0) {
        container.innerHTML = '<div class="empty-state">No transactions match filters</div>';
        return;
    }

    container.innerHTML = `
        <div class="transactions-grid">
            ${transactions.map(tx => `
                <div class="transaction-row">
                    <div class="tx-date">${formatDate(tx.PostedDate || tx.date)}</div>
                    <div class="tx-desc">${escapeHtml(tx.Description || tx.description || 'N/A')}</div>
                    <div class="tx-amount ${(tx.Amount || tx.amount) >= 0 ? 'positive' : 'negative'}">
                        ${formatCurrency(tx.Amount || tx.amount)}
                    </div>
                </div>
            `).join('')}
        </div>
    `;
}

/**
 * Render transaction filter panel
 */
function renderTransactionFilters() {
    const container = document.getElementById('transactionFiltersPanel');
    if (!container) return;

    container.innerHTML = `
        <div class="filters-panel" data-testid="filters-panel">
            <div class="filter-row">
                <div class="filter-group">
                    <label class="form-label">From Date</label>
                    <input type="date" id="filterDateFrom" class="form-input form-input-sm">
                </div>
                <div class="filter-group">
                    <label class="form-label">To Date</label>
                    <input type="date" id="filterDateTo" class="form-input form-input-sm">
                </div>
            </div>
            <div class="filter-row">
                <div class="filter-group">
                    <label class="form-label">Min Amount</label>
                    <input type="number" id="filterMinAmount" class="form-input form-input-sm" placeholder="0.00">
                </div>
                <div class="filter-group">
                    <label class="form-label">Max Amount</label>
                    <input type="number" id="filterMaxAmount" class="form-input form-input-sm" placeholder="9999.99">
                </div>
            </div>
            <div class="filter-row">
                <div class="filter-group">
                    <label class="form-label">Type</label>
                    <select id="filterType" class="form-input form-input-sm">
                        <option value="all">All Types</option>
                        <option value="debit">Debit</option>
                        <option value="credit">Credit</option>
                        <option value="transfer">Transfer</option>
                    </select>
                </div>
                <div class="filter-actions">
                    <button class="btn btn-primary btn-sm" onclick="applyTransactionFilters()" data-testid="apply-filters">Apply</button>
                    <button class="btn btn-secondary btn-sm" onclick="clearTransactionFilters()">Clear</button>
                </div>
            </div>
        </div>
    `;
}

/**
 * Clear all transaction filters
 */
function clearTransactionFilters() {
    document.getElementById('filterDateFrom').value = '';
    document.getElementById('filterDateTo').value = '';
    document.getElementById('filterMinAmount').value = '';
    document.getElementById('filterMaxAmount').value = '';
    document.getElementById('filterType').value = 'all';

    MemberState.transactionFilters = { dateFrom: null, dateTo: null, minAmount: null, maxAmount: null, type: 'all' };
    renderFilteredTransactions(MemberState.transactions);
}

// =================================================================
// CSV EXPORT
// =================================================================

/**
 * Export transactions to CSV
 */
function exportTransactionsCSV() {
    const transactions = filterTransactions(MemberState.transactions, MemberState.transactionFilters);

    if (transactions.length === 0) {
        alert('No transactions to export');
        return;
    }

    const headers = ['Date', 'Description', 'Amount', 'Type', 'Category'];
    const rows = transactions.map(tx => [
        tx.PostedDate || tx.date || '',
        `"${(tx.Description || tx.description || '').replace(/"/g, '""')}"`,
        tx.Amount || tx.amount || 0,
        tx.Type || tx.type || '',
        tx.Category || tx.category || ''
    ]);

    const csvContent = [
        headers.join(','),
        ...rows.map(r => r.join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `transactions_${new Date().toISOString().split('T')[0]}.csv`;
    link.click();

    console.log(`[Export] Downloaded ${transactions.length} transactions as CSV`);
}

// =================================================================
// UTILITY FUNCTIONS
// =================================================================

/**
 * Escape HTML to prevent XSS
 * @param {string} str - Input string
 * @returns {string} Escaped string
 */
function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

/**
 * Get initials from name
 * @param {string} name - Full name
 * @returns {string} Initials (max 2 chars)
 */
function getInitials(name) {
    if (!name) return '?';
    return name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2);
}

/**
 * Format date for display
 * @param {string} dateStr - Date string
 * @returns {string} Formatted date
 */
function formatDate(dateStr) {
    if (!dateStr) return 'N/A';
    return new Date(dateStr).toLocaleDateString('en-US', {
        month: 'short', day: 'numeric', year: 'numeric'
    });
}

/**
 * Format relative time (e.g., "2 hours ago")
 * @param {string} dateStr - ISO date string
 * @returns {string} Relative time string
 */
function formatRelativeTime(dateStr) {
    const now = new Date();
    const then = new Date(dateStr);
    const diffMs = now - then;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return formatDate(dateStr);
}

/**
 * Format currency
 * @param {number} amount - Amount in dollars
 * @returns {string} Formatted currency string
 */
function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(amount || 0);
}

/**
 * Get display label for note tag
 * @param {string} tag - Tag identifier
 * @returns {string} Display label
 */
function getTagLabel(tag) {
    const labels = {
        'general': '📋 General',
        'important': '⚠️ Important',
        'followup': '📌 Follow-up'
    };
    return labels[tag] || tag;
}

// =================================================================
// INITIALIZATION
// =================================================================

/**
 * Initialize member module on page load
 */
function initMemberModule() {
    console.log('[Member Module] Initializing...');
    renderRecentMembers();

    // Check for memberId URL parameter (screen pop)
    const urlParams = new URLSearchParams(window.location.search);
    const memberId = urlParams.get('memberId');
    if (memberId) {
        console.log(`[Member Module] Screen pop: Loading member ${memberId}`);
        loadMember360(memberId);
    }

    console.log('[Member Module] Initialized');
}

// =================================================================
// GLOBAL EXPORTS
// =================================================================

window.MemberState = MemberState;
window.addToRecentMembers = addToRecentMembers;
window.renderRecentMembers = renderRecentMembers;
window.loadMember360 = loadMember360;
window.getMemberNotes = getMemberNotes;
window.saveMemberNote = saveMemberNote;
window.showMemberNotes = showMemberNotes;
window.applyTransactionFilters = applyTransactionFilters;
window.clearTransactionFilters = clearTransactionFilters;
window.exportTransactionsCSV = exportTransactionsCSV;
window.initMemberModule = initMemberModule;

// Member 360 Enhanced Exports
window.calculateMemberHealthScore = calculateMemberHealthScore;
window.getDefaultNBA = getDefaultNBA;
window.renderAccountsList = renderAccountsList;
window.loadMemberAccounts = loadMemberAccounts;
window.showMemberTimeline = showMemberTimeline;
window.executeNBA = executeNBA;
window.askGemmaAboutMember = askGemmaAboutMember;

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initMemberModule);
} else {
    initMemberModule();
}
