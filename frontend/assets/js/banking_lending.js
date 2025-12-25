/**
 * Banking Lending Module - Phase 3 Lending Intelligence
 * ======================================================
 * Affordability calculators, Nemo Score, Pricing Engine, Document Checklist
 * 
 * Dependencies: banking_core.js, banking_member.js
 * Used by: banking.html
 * 
 * @module banking_lending
 * @version 1.0.0
 */

// =================================================================
// STATE MANAGEMENT
// =================================================================

/**
 * Lending state object
 * @type {Object}
 */
const LendingState = {
    currentApplication: null,
    pipeline: JSON.parse(localStorage.getItem('scuLendingPipeline') || '[]'),
    pricingRules: [],
    documentChecklists: {}
};

// =================================================================
// AFFORDABILITY CALCULATORS
// =================================================================

/**
 * Calculate Debt-to-Income ratio
 * @param {number} monthlyDebt - Total monthly debt payments
 * @param {number} monthlyIncome - Total monthly gross income
 * @returns {Object} DTI calculation result
 */
function calculateDTI(monthlyDebt, monthlyIncome) {
    if (!monthlyIncome || monthlyIncome <= 0) {
        return { ratio: 0, status: 'error', message: 'Income must be greater than 0' };
    }

    const ratio = (monthlyDebt / monthlyIncome) * 100;
    let status = 'good';
    let message = 'Acceptable DTI';

    if (ratio > 50) {
        status = 'high_risk';
        message = 'DTI exceeds 50% - High risk';
    } else if (ratio > 43) {
        status = 'elevated';
        message = 'DTI exceeds 43% - Review required';
    } else if (ratio > 36) {
        status = 'moderate';
        message = 'DTI between 36-43% - Monitor';
    }

    return {
        ratio: Math.round(ratio * 100) / 100,
        monthlyDebt,
        monthlyIncome,
        status,
        message,
        maxRecommendedPayment: monthlyIncome * 0.43 - monthlyDebt
    };
}

/**
 * Calculate Loan-to-Value ratio
 * @param {number} loanAmount - Loan principal amount
 * @param {number} propertyValue - Appraised property value
 * @returns {Object} LTV calculation result
 */
function calculateLTV(loanAmount, propertyValue) {
    if (!propertyValue || propertyValue <= 0) {
        return { ratio: 0, status: 'error', message: 'Property value must be greater than 0' };
    }

    const ratio = (loanAmount / propertyValue) * 100;
    let status = 'good';
    let message = 'Standard LTV';
    let pmiRequired = false;

    if (ratio > 97) {
        status = 'high_risk';
        message = 'LTV exceeds 97% - Very high risk';
        pmiRequired = true;
    } else if (ratio > 80) {
        status = 'elevated';
        message = 'LTV exceeds 80% - PMI required';
        pmiRequired = true;
    }

    return {
        ratio: Math.round(ratio * 100) / 100,
        loanAmount,
        propertyValue,
        status,
        message,
        pmiRequired,
        equityPercent: 100 - ratio
    };
}

/**
 * Calculate Debt Service Coverage Ratio (for commercial/investment)
 * @param {number} netOperatingIncome - Annual NOI
 * @param {number} annualDebtService - Annual debt payments
 * @returns {Object} DSCR calculation result
 */
function calculateDSCR(netOperatingIncome, annualDebtService) {
    if (!annualDebtService || annualDebtService <= 0) {
        return { ratio: 0, status: 'error', message: 'Debt service must be greater than 0' };
    }

    const ratio = netOperatingIncome / annualDebtService;
    let status = 'good';
    let message = 'Healthy DSCR';

    if (ratio < 1.0) {
        status = 'high_risk';
        message = 'DSCR below 1.0 - Cannot cover debt';
    } else if (ratio < 1.2) {
        status = 'elevated';
        message = 'DSCR below 1.2 - Marginal coverage';
    } else if (ratio < 1.5) {
        status = 'moderate';
        message = 'DSCR between 1.2-1.5 - Acceptable';
    }

    return {
        ratio: Math.round(ratio * 100) / 100,
        netOperatingIncome,
        annualDebtService,
        status,
        message,
        cushionAmount: netOperatingIncome - annualDebtService
    };
}

/**
 * Render affordability calculator panel
 * @param {string} containerId - Target container ID
 */
function renderAffordabilityCalculator(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = `
        <div class="affordability-panel" data-testid="affordability-panel">
            <h3>📊 Affordability Calculator</h3>
            
            <div class="calc-tabs">
                <button class="calc-tab active" onclick="showCalcTab('dti')" data-testid="calc-tab-dti">DTI</button>
                <button class="calc-tab" onclick="showCalcTab('ltv')" data-testid="calc-tab-ltv">LTV</button>
                <button class="calc-tab" onclick="showCalcTab('dscr')" data-testid="calc-tab-dscr">DSCR</button>
            </div>
            
            <div id="calc-dti" class="calc-content active" data-testid="calc-content-dti">
                <div class="form-row">
                    <div class="form-group">
                        <label class="form-label">Monthly Income ($)</label>
                        <input type="number" id="dtiIncome" class="form-input" placeholder="5000" value="5000">
                    </div>
                    <div class="form-group">
                        <label class="form-label">Monthly Debt ($)</label>
                        <input type="number" id="dtiDebt" class="form-input" placeholder="1500" value="1500">
                    </div>
                </div>
                <button class="btn btn-primary" onclick="runDTICalc()" data-testid="calc-dti-btn">Calculate DTI</button>
                <div id="dtiResult" class="calc-result" data-testid="dti-result"></div>
            </div>
            
            <div id="calc-ltv" class="calc-content" data-testid="calc-content-ltv">
                <div class="form-row">
                    <div class="form-group">
                        <label class="form-label">Loan Amount ($)</label>
                        <input type="number" id="ltvLoan" class="form-input" placeholder="200000" value="200000">
                    </div>
                    <div class="form-group">
                        <label class="form-label">Property Value ($)</label>
                        <input type="number" id="ltvValue" class="form-input" placeholder="250000" value="250000">
                    </div>
                </div>
                <button class="btn btn-primary" onclick="runLTVCalc()" data-testid="calc-ltv-btn">Calculate LTV</button>
                <div id="ltvResult" class="calc-result" data-testid="ltv-result"></div>
            </div>
            
            <div id="calc-dscr" class="calc-content" data-testid="calc-content-dscr">
                <div class="form-row">
                    <div class="form-group">
                        <label class="form-label">Net Operating Income (Annual $)</label>
                        <input type="number" id="dscrNOI" class="form-input" placeholder="50000" value="50000">
                    </div>
                    <div class="form-group">
                        <label class="form-label">Annual Debt Service ($)</label>
                        <input type="number" id="dscrDebt" class="form-input" placeholder="35000" value="35000">
                    </div>
                </div>
                <button class="btn btn-primary" onclick="runDSCRCalc()" data-testid="calc-dscr-btn">Calculate DSCR</button>
                <div id="dscrResult" class="calc-result" data-testid="dscr-result"></div>
            </div>
        </div>
    `;
}

function showCalcTab(tab) {
    document.querySelectorAll('.calc-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.calc-content').forEach(c => c.classList.remove('active'));
    document.querySelector(`.calc-tab[onclick*="${tab}"]`)?.classList.add('active');
    document.getElementById(`calc-${tab}`)?.classList.add('active');
}

function runDTICalc() {
    const income = parseFloat(document.getElementById('dtiIncome').value) || 0;
    const debt = parseFloat(document.getElementById('dtiDebt').value) || 0;
    const result = calculateDTI(debt, income);
    document.getElementById('dtiResult').innerHTML = formatCalcResult('DTI', result);
}

function runLTVCalc() {
    const loan = parseFloat(document.getElementById('ltvLoan').value) || 0;
    const value = parseFloat(document.getElementById('ltvValue').value) || 0;
    const result = calculateLTV(loan, value);
    document.getElementById('ltvResult').innerHTML = formatCalcResult('LTV', result);
}

function runDSCRCalc() {
    const noi = parseFloat(document.getElementById('dscrNOI').value) || 0;
    const debt = parseFloat(document.getElementById('dscrDebt').value) || 0;
    const result = calculateDSCR(noi, debt);
    document.getElementById('dscrResult').innerHTML = formatCalcResult('DSCR', result);
}

function formatCalcResult(type, result) {
    const statusColors = {
        'good': '#10b981',
        'moderate': '#f59e0b',
        'elevated': '#ef4444',
        'high_risk': '#dc2626',
        'error': '#6b7280'
    };

    return `
        <div class="result-card result-${result.status}">
            <div class="result-header">
                <span class="result-type">${type} Ratio</span>
                <span class="result-value" style="color: ${statusColors[result.status]}">${result.ratio}${type === 'DSCR' ? 'x' : '%'}</span>
            </div>
            <div class="result-status" style="color: ${statusColors[result.status]}">${result.message}</div>
        </div>
    `;
}

// =================================================================
// NEMO SCORE (Internal Credit Proxy)
// =================================================================

/**
 * Calculate Nemo Score - internal credit proxy based on member behavior
 * @param {Object} memberData - Member transaction/account data
 * @returns {Object} Nemo Score result
 */
function calculateNemoScore(memberData) {
    let score = 500; // Base score
    const factors = [];

    // Payment history factor (+/- up to 150 points)
    const onTimePayments = memberData.onTimePaymentRate || 0.9;
    const paymentPoints = Math.round((onTimePayments - 0.5) * 300);
    score += paymentPoints;
    factors.push({
        name: 'Payment History',
        impact: paymentPoints,
        description: `${Math.round(onTimePayments * 100)}% on-time payments`
    });

    // Account longevity (+/- up to 100 points)
    const accountAgeMonths = memberData.accountAgeMonths || 12;
    const longevityPoints = Math.min(accountAgeMonths * 2, 100);
    score += longevityPoints;
    factors.push({
        name: 'Account Longevity',
        impact: longevityPoints,
        description: `${accountAgeMonths} months with SCU`
    });

    // Balance stability (+/- up to 75 points)
    const avgBalance = memberData.averageBalance || 1000;
    const balancePoints = Math.min(avgBalance / 100, 75);
    score += Math.round(balancePoints);
    factors.push({
        name: 'Balance Stability',
        impact: Math.round(balancePoints),
        description: `$${avgBalance.toLocaleString()} average balance`
    });

    // NSF/Overdraft penalty (up to -100 points)
    const nsfCount = memberData.nsfCount || 0;
    const nsfPenalty = Math.min(nsfCount * 25, 100);
    score -= nsfPenalty;
    if (nsfPenalty > 0) {
        factors.push({
            name: 'NSF/Overdraft',
            impact: -nsfPenalty,
            description: `${nsfCount} NSF events in past year`
        });
    }

    // Clamp score to 300-850 range
    score = Math.max(300, Math.min(850, score));

    // Determine tier
    let tier = 'Fair';
    if (score >= 750) tier = 'Excellent';
    else if (score >= 700) tier = 'Good';
    else if (score >= 650) tier = 'Fair';
    else tier = 'Needs Improvement';

    return {
        score: Math.round(score),
        tier,
        factors,
        rateAdjustment: score >= 700 ? 0 : (700 - score) * 0.01,
        timestamp: new Date().toISOString()
    };
}

/**
 * Render Nemo Score panel
 * @param {string} containerId - Target container ID
 * @param {Object} memberData - Member data for scoring
 */
function renderNemoScorePanel(containerId, memberData) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const score = calculateNemoScore(memberData || {});

    const tierColors = {
        'Excellent': '#10b981',
        'Good': '#3b82f6',
        'Fair': '#f59e0b',
        'Needs Improvement': '#ef4444'
    };

    container.innerHTML = `
        <div class="nemo-score-panel" data-testid="nemo-score-panel">
            <div class="nemo-score-header">
                <h3>🐠 Nemo Score</h3>
                <span class="nemo-score-badge">${score.tier}</span>
            </div>
            <div class="nemo-score-display">
                <div class="nemo-score-value" style="color: ${tierColors[score.tier]}">${score.score}</div>
                <div class="nemo-score-range">300 - 850</div>
            </div>
            <div class="nemo-score-factors">
                ${score.factors.map(f => `
                    <div class="score-factor">
                        <span class="factor-name">${f.name}</span>
                        <span class="factor-impact ${f.impact >= 0 ? 'positive' : 'negative'}">
                            ${f.impact >= 0 ? '+' : ''}${f.impact}
                        </span>
                    </div>
                `).join('')}
            </div>
            ${score.rateAdjustment > 0 ? `
                <div class="rate-adjustment-notice">
                    ⚠️ Rate adjustment: +${(score.rateAdjustment * 100).toFixed(2)}%
                </div>
            ` : ''}
        </div>
    `;
}

// =================================================================
// PRICING ENGINE
// =================================================================

/**
 * Default pricing rules
 */
const PRICING_RULES = [
    { scoreMin: 750, scoreMax: 850, baseRate: 5.99, adjustment: 0, tier: 'A' },
    { scoreMin: 700, scoreMax: 749, baseRate: 6.49, adjustment: 0.5, tier: 'B' },
    { scoreMin: 650, scoreMax: 699, baseRate: 7.49, adjustment: 1.5, tier: 'C' },
    { scoreMin: 600, scoreMax: 649, baseRate: 8.99, adjustment: 3.0, tier: 'D' },
    { scoreMin: 300, scoreMax: 599, baseRate: 10.99, adjustment: 5.0, tier: 'E' }
];

/**
 * Calculate loan pricing based on Nemo Score
 * @param {number} nemoScore - Member's Nemo Score
 * @param {number} loanAmount - Requested loan amount
 * @param {number} termMonths - Loan term in months
 * @returns {Object} Pricing result
 */
function calculateLoanPricing(nemoScore, loanAmount, termMonths) {
    const rule = PRICING_RULES.find(r => nemoScore >= r.scoreMin && nemoScore <= r.scoreMax)
        || PRICING_RULES[PRICING_RULES.length - 1];

    const annualRate = rule.baseRate + rule.adjustment;
    const monthlyRate = annualRate / 100 / 12;

    // Calculate monthly payment (standard amortization)
    const payment = loanAmount * (monthlyRate * Math.pow(1 + monthlyRate, termMonths))
        / (Math.pow(1 + monthlyRate, termMonths) - 1);

    const totalPayments = payment * termMonths;
    const totalInterest = totalPayments - loanAmount;

    return {
        tier: rule.tier,
        baseRate: rule.baseRate,
        adjustment: rule.adjustment,
        finalRate: annualRate,
        monthlyPayment: Math.round(payment * 100) / 100,
        totalPayments: Math.round(totalPayments * 100) / 100,
        totalInterest: Math.round(totalInterest * 100) / 100,
        nemoScore,
        loanAmount,
        termMonths
    };
}

/**
 * Render pricing engine panel
 * @param {string} containerId - Target container ID
 */
function renderPricingEngine(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = `
        <div class="pricing-panel" data-testid="pricing-panel">
            <h3>💰 Loan Pricing Engine</h3>
            <div class="form-row">
                <div class="form-group">
                    <label class="form-label">Nemo Score</label>
                    <input type="number" id="pricingScore" class="form-input" value="720" min="300" max="850">
                </div>
                <div class="form-group">
                    <label class="form-label">Loan Amount ($)</label>
                    <input type="number" id="pricingAmount" class="form-input" value="25000">
                </div>
                <div class="form-group">
                    <label class="form-label">Term (Months)</label>
                    <select id="pricingTerm" class="form-input">
                        <option value="12">12 months</option>
                        <option value="24">24 months</option>
                        <option value="36" selected>36 months</option>
                        <option value="48">48 months</option>
                        <option value="60">60 months</option>
                        <option value="72">72 months</option>
                    </select>
                </div>
            </div>
            <button class="btn btn-primary" onclick="runPricingCalc()" data-testid="pricing-calc-btn">Calculate Pricing</button>
            <div id="pricingResult" class="pricing-result" data-testid="pricing-result"></div>
        </div>
    `;
}

function runPricingCalc() {
    const score = parseInt(document.getElementById('pricingScore').value) || 720;
    const amount = parseFloat(document.getElementById('pricingAmount').value) || 25000;
    const term = parseInt(document.getElementById('pricingTerm').value) || 36;

    const result = calculateLoanPricing(score, amount, term);

    document.getElementById('pricingResult').innerHTML = `
        <div class="pricing-card">
            <div class="pricing-header">
                <span class="pricing-tier tier-${result.tier}">Tier ${result.tier}</span>
                <span class="pricing-rate">${result.finalRate.toFixed(2)}% APR</span>
            </div>
            <div class="pricing-details">
                <div class="pricing-row">
                    <span>Monthly Payment</span>
                    <span class="pricing-value">$${result.monthlyPayment.toLocaleString()}</span>
                </div>
                <div class="pricing-row">
                    <span>Total Interest</span>
                    <span class="pricing-value">$${result.totalInterest.toLocaleString()}</span>
                </div>
                <div class="pricing-row">
                    <span>Total Payments</span>
                    <span class="pricing-value">$${result.totalPayments.toLocaleString()}</span>
                </div>
            </div>
        </div>
    `;
}

// =================================================================
// DOCUMENT CHECKLIST
// =================================================================

/**
 * Document requirements by loan type
 */
const DOC_REQUIREMENTS = {
    'auto': [
        { id: 'income', name: 'Proof of Income', required: true },
        { id: 'id', name: 'Valid Government ID', required: true },
        { id: 'residence', name: 'Proof of Residence', required: true },
        { id: 'insurance', name: 'Auto Insurance Quote', required: true },
        { id: 'purchase', name: 'Purchase Agreement/Bill of Sale', required: true }
    ],
    'personal': [
        { id: 'income', name: 'Proof of Income', required: true },
        { id: 'id', name: 'Valid Government ID', required: true },
        { id: 'residence', name: 'Proof of Residence', required: true }
    ],
    'mortgage': [
        { id: 'income', name: 'Proof of Income (2 years)', required: true },
        { id: 'id', name: 'Valid Government ID', required: true },
        { id: 'w2', name: 'W-2 Forms (2 years)', required: true },
        { id: 'tax', name: 'Tax Returns (2 years)', required: true },
        { id: 'bank', name: 'Bank Statements (2 months)', required: true },
        { id: 'appraisal', name: 'Property Appraisal', required: true },
        { id: 'title', name: 'Title Report', required: true },
        { id: 'insurance', name: 'Homeowners Insurance', required: true }
    ]
};

/**
 * Get document checklist for loan type
 * @param {string} loanType - Type of loan (auto, personal, mortgage)
 * @returns {Array} Document checklist
 */
function getDocumentChecklist(loanType) {
    return DOC_REQUIREMENTS[loanType] || DOC_REQUIREMENTS['personal'];
}

/**
 * Render document checklist
 * @param {string} containerId - Target container ID
 * @param {string} loanType - Loan type
 */
function renderDocumentChecklist(containerId, loanType = 'auto') {
    const container = document.getElementById(containerId);
    if (!container) return;

    const docs = getDocumentChecklist(loanType);

    container.innerHTML = `
        <div class="doc-checklist-panel" data-testid="doc-checklist-panel">
            <h3>📋 Document Checklist</h3>
            <div class="doc-type-select">
                <select id="docLoanType" class="form-input" onchange="updateDocChecklist()">
                    <option value="auto" ${loanType === 'auto' ? 'selected' : ''}>Auto Loan</option>
                    <option value="personal" ${loanType === 'personal' ? 'selected' : ''}>Personal Loan</option>
                    <option value="mortgage" ${loanType === 'mortgage' ? 'selected' : ''}>Mortgage</option>
                </select>
            </div>
            <div class="doc-list" data-testid="doc-list">
                ${docs.map(doc => `
                    <div class="doc-item" data-doc-id="${doc.id}">
                        <input type="checkbox" id="doc-${doc.id}" class="doc-checkbox" onchange="updateDocProgress()">
                        <label for="doc-${doc.id}" class="doc-label">
                            ${doc.name}
                            ${doc.required ? '<span class="doc-required">*</span>' : ''}
                        </label>
                    </div>
                `).join('')}
            </div>
            <div class="doc-progress" data-testid="doc-progress">
                <div class="doc-progress-bar">
                    <div class="doc-progress-fill" id="docProgressFill" style="width: 0%"></div>
                </div>
                <span class="doc-progress-text" id="docProgressText">0/${docs.length} documents</span>
            </div>
        </div>
    `;
}

function updateDocChecklist() {
    const loanType = document.getElementById('docLoanType').value;
    renderDocumentChecklist('documentChecklistPanel', loanType);
}

function updateDocProgress() {
    const checkboxes = document.querySelectorAll('.doc-checkbox');
    const checked = document.querySelectorAll('.doc-checkbox:checked').length;
    const total = checkboxes.length;
    const percent = Math.round((checked / total) * 100);

    document.getElementById('docProgressFill').style.width = `${percent}%`;
    document.getElementById('docProgressText').textContent = `${checked}/${total} documents`;
}

// =================================================================
// PIPELINE VIEW
// =================================================================

/**
 * Pipeline stages
 */
const PIPELINE_STAGES = [
    { id: 'application', name: 'Application', color: '#94a3b8' },
    { id: 'review', name: 'Under Review', color: '#3b82f6' },
    { id: 'approved', name: 'Approved', color: '#10b981' },
    { id: 'funded', name: 'Funded', color: '#8b5cf6' },
    { id: 'declined', name: 'Declined', color: '#ef4444' }
];

/**
 * Add application to pipeline
 * @param {Object} application - Application data
 */
function addToPipeline(application) {
    application.id = application.id || Date.now().toString();
    application.stage = application.stage || 'application';
    application.createdAt = application.createdAt || new Date().toISOString();

    LendingState.pipeline.unshift(application);
    localStorage.setItem('scuLendingPipeline', JSON.stringify(LendingState.pipeline.slice(0, 50)));
}

/**
 * Move application to different stage
 * @param {string} appId - Application ID
 * @param {string} newStage - New stage ID
 */
function moveApplicationStage(appId, newStage) {
    const app = LendingState.pipeline.find(a => a.id === appId);
    if (app) {
        app.stage = newStage;
        app.updatedAt = new Date().toISOString();
        localStorage.setItem('scuLendingPipeline', JSON.stringify(LendingState.pipeline));
    }
}

/**
 * Get applications by stage
 * @param {string} stage - Stage ID
 * @returns {Array} Applications in stage
 */
function getApplicationsByStage(stage) {
    return LendingState.pipeline.filter(a => a.stage === stage);
}

/**
 * Render pipeline view
 * @param {string} containerId - Target container ID
 */
function renderPipelineView(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = `
        <div class="pipeline-view" data-testid="pipeline-view">
            <div class="pipeline-header">
                <h3>📊 Loan Pipeline</h3>
                <span class="pipeline-count">${LendingState.pipeline.length} applications</span>
            </div>
            <div class="pipeline-columns">
                ${PIPELINE_STAGES.filter(s => s.id !== 'declined').map(stage => `
                    <div class="pipeline-column" data-stage="${stage.id}">
                        <div class="pipeline-column-header" style="border-color: ${stage.color}">
                            <span>${stage.name}</span>
                            <span class="stage-count">${getApplicationsByStage(stage.id).length}</span>
                        </div>
                        <div class="pipeline-cards">
                            ${getApplicationsByStage(stage.id).map(app => `
                                <div class="pipeline-card" data-app-id="${app.id}">
                                    <div class="app-name">${escapeHtml(app.memberName || 'Unknown')}</div>
                                    <div class="app-type">${escapeHtml(app.loanType || 'Loan')}</div>
                                    <div class="app-amount">$${(app.amount || 0).toLocaleString()}</div>
                                </div>
                            `).join('') || '<div class="empty-column">No applications</div>'}
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
}

// =================================================================
// UTILITY FUNCTIONS
// =================================================================

function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// =================================================================
// INITIALIZATION
// =================================================================

function initLendingModule() {
    console.log('[Lending Module] Initializing...');
    console.log('[Lending Module] Initialized');
}

// =================================================================
// GLOBAL EXPORTS
// =================================================================

window.LendingState = LendingState;
window.PRICING_RULES = PRICING_RULES;
window.DOC_REQUIREMENTS = DOC_REQUIREMENTS;
window.PIPELINE_STAGES = PIPELINE_STAGES;

window.calculateDTI = calculateDTI;
window.calculateLTV = calculateLTV;
window.calculateDSCR = calculateDSCR;
window.renderAffordabilityCalculator = renderAffordabilityCalculator;
window.showCalcTab = showCalcTab;
window.runDTICalc = runDTICalc;
window.runLTVCalc = runLTVCalc;
window.runDSCRCalc = runDSCRCalc;

window.calculateNemoScore = calculateNemoScore;
window.renderNemoScorePanel = renderNemoScorePanel;

window.calculateLoanPricing = calculateLoanPricing;
window.renderPricingEngine = renderPricingEngine;
window.runPricingCalc = runPricingCalc;

window.getDocumentChecklist = getDocumentChecklist;
window.renderDocumentChecklist = renderDocumentChecklist;
window.updateDocChecklist = updateDocChecklist;
window.updateDocProgress = updateDocProgress;

window.addToPipeline = addToPipeline;
window.moveApplicationStage = moveApplicationStage;
window.getApplicationsByStage = getApplicationsByStage;
window.renderPipelineView = renderPipelineView;

window.initLendingModule = initLendingModule;

// Auto-initialize
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initLendingModule);
} else {
    initLendingModule();
}
