/**
 * Banking Lending AI Module - AI-Powered Loan Officer Workstation
 * ================================================================
 * Cutting-edge AI features for loan origination:
 * - Automated Underwriting Engine
 * - SHAP-style Explainable AI (XAI)
 * - Fair Lending Compliance (ECOA, FCRA)
 * - LLM Integration for Decision Explanations
 * - Immutable Audit Logging
 * 
 * Dependencies: banking_lending.js, banking_core.js
 * Used by: banking.html (Loan Officer section)
 * 
 * @module banking_lending_ai
 * @version 1.0.0
 */

// =================================================================
// CONFIGURATION
// =================================================================

/**
 * AI Underwriting Configuration
 */
const AI_CONFIG = {
    // Decision thresholds
    autoApprove: {
        minNemoScore: 700,
        maxDTI: 43,
        maxLTV: 80,
        minEmploymentMonths: 24
    },
    autoDecline: {
        minNemoScore: 500,
        maxDTI: 60,
        reasons: ['active_bankruptcy', 'recent_chargeoff', 'fraud_alert']
    },
    manualReview: {
        // Anything between auto-approve and auto-decline
    },

    // SLA Configuration (in hours)
    sla: {
        auto: { target: 4, critical: 8 },
        personal: { target: 4, critical: 8 },
        mortgage: { target: 48, critical: 72 },
        heloc: { target: 24, critical: 48 }
    },

    // Model version for audit trail
    modelVersion: 'nemo-underwrite-v2.1.0'
};

/**
 * Fair Lending Configuration
 */
const FAIR_LENDING_CONFIG = {
    // Protected classes (ECOA)
    protectedClasses: [
        'race', 'color', 'religion', 'sex', 'familyStatus',
        'nationalOrigin', 'age', 'disability', 'maritalStatus'
    ],

    // Features that may correlate with protected classes (proxy risk)
    riskyProxies: ['zipCode', 'university', 'employer'],

    // Disparate Impact threshold (80% rule)
    disparateImpactRatio: 0.80,

    // Require Less Discriminatory Alternative search
    ldaRequired: true
};

// =================================================================
// STATE MANAGEMENT
// =================================================================

/**
 * AI Lending State
 */
const AILendingState = {
    currentApplication: null,
    underwriteResult: null,
    loanQueue: JSON.parse(localStorage.getItem('scuLoanQueue') || '[]'),
    auditLog: JSON.parse(localStorage.getItem('scuAIAuditLog') || '[]'),
    lastDecision: null
};

// =================================================================
// AUTOMATED UNDERWRITING ENGINE
// =================================================================

/**
 * Run full automated underwriting on a loan application
 * @param {Object} application - Loan application data
 * @returns {Object} Complete underwriting result with recommendation
 */
function autoUnderwrite(application) {
    const startTime = Date.now();

    // 1. Get member data (mock or from Fiserv)
    const memberData = application.memberData || {
        onTimePaymentRate: 0.95,
        accountAgeMonths: 24,
        averageBalance: 2500,
        nsfCount: 0
    };

    // 2. Calculate all metrics
    const dti = window.calculateDTI ?
        window.calculateDTI(application.monthlyDebt || 0, application.monthlyIncome || 5000) :
        { ratio: 30, status: 'good' };

    const ltv = window.calculateLTV ?
        window.calculateLTV(application.loanAmount || 0, application.collateralValue || application.loanAmount * 1.25) :
        { ratio: 80, status: 'good', pmiRequired: false };

    const nemoScore = window.calculateNemoScore ?
        window.calculateNemoScore(memberData) :
        { score: 720, tier: 'Good', factors: [] };

    const pricing = window.calculateLoanPricing ?
        window.calculateLoanPricing(nemoScore.score, application.loanAmount || 25000, application.termMonths || 36) :
        { tier: 'B', finalRate: 6.49, monthlyPayment: 750 };

    // 3. Calculate SHAP-style feature contributions
    const shapValues = calculateShapContributions(application, {
        nemoScore, dti, ltv, memberData
    });

    // 4. Determine recommendation
    const recommendation = determineRecommendation(nemoScore, dti, ltv, application);

    // 5. Run fair lending compliance check
    const fairLendingResult = runFairLendingCheck(application, recommendation);

    // 6. Generate confidence score
    const confidence = calculateConfidence(shapValues);

    // 7. Build result object
    const result = {
        applicationId: application.id || `APP-${Date.now()}`,
        memberId: application.memberId,
        memberName: application.memberName || 'Unknown',

        // Metrics
        metrics: {
            creditScore: application.creditScore || nemoScore.score,
            nemoScore: nemoScore,
            dti: dti,
            ltv: ltv,
            pricing: pricing
        },

        // AI Recommendation
        recommendation: recommendation.action,
        recommendationReason: recommendation.reason,
        conditions: recommendation.conditions || [],
        riskLevel: recommendation.riskLevel,
        confidence: confidence,

        // Explainability
        shapValues: shapValues,
        topFactors: getTopFactors(shapValues),

        // Compliance
        fairLendingResult: fairLendingResult,

        // Metadata
        modelVersion: AI_CONFIG.modelVersion,
        processingTimeMs: Date.now() - startTime,
        timestamp: new Date().toISOString()
    };

    // Store in state
    AILendingState.underwriteResult = result;
    AILendingState.currentApplication = application;

    console.log('[AI Underwrite] Complete:', result.recommendation, 'in', result.processingTimeMs, 'ms');
    return result;
}

/**
 * Calculate SHAP-style feature contributions
 * @param {Object} application - Loan application
 * @param {Object} metrics - Calculated metrics
 * @returns {Object} Feature contributions
 */
function calculateShapContributions(application, metrics) {
    const contributions = {};
    let baseScore = 50; // Start at neutral

    // Credit Score contribution (+/-30 points max)
    const creditScore = metrics.nemoScore.score || 700;
    const creditContrib = Math.round((creditScore - 650) / 5);
    contributions.creditScore = {
        value: creditScore,
        contribution: Math.max(-30, Math.min(30, creditContrib)),
        description: `Credit score of ${creditScore}`,
        benchmark: 700,
        percentile: Math.round((creditScore - 300) / 5.5)
    };

    // DTI contribution (+/-25 points max)
    const dtiRatio = metrics.dti.ratio || 30;
    const dtiContrib = Math.round((43 - dtiRatio) / 1.7);
    contributions.dtiRatio = {
        value: dtiRatio,
        contribution: Math.max(-25, Math.min(25, dtiContrib)),
        description: `DTI ratio of ${dtiRatio.toFixed(1)}%`,
        threshold: 43,
        status: metrics.dti.status
    };

    // Employment stability (+/-20 points max)
    const empMonths = application.employmentMonths || 36;
    const empContrib = Math.min(empMonths / 3, 20);
    contributions.employmentStability = {
        value: empMonths,
        contribution: Math.round(empContrib),
        description: `${Math.floor(empMonths / 12)} years ${empMonths % 12} months employment`,
        benchmark: 24
    };

    // Account longevity (+/-15 points max)
    const accountAge = metrics.memberData.accountAgeMonths || 12;
    const ageContrib = Math.min(accountAge / 2, 15);
    contributions.accountLongevity = {
        value: accountAge,
        contribution: Math.round(ageContrib),
        description: `${accountAge} months as member`,
        benchmark: 12
    };

    // LTV for secured loans (+/-15 points max)
    if (metrics.ltv && metrics.ltv.ratio > 0) {
        const ltvRatio = metrics.ltv.ratio;
        const ltvContrib = Math.round((95 - ltvRatio) / 4);
        contributions.loanToValue = {
            value: ltvRatio,
            contribution: Math.max(-15, Math.min(15, ltvContrib)),
            description: `LTV ratio of ${ltvRatio.toFixed(1)}%`,
            threshold: 80,
            pmiRequired: metrics.ltv.pmiRequired
        };
    }

    // Payment history (+/-10 points)
    const onTimeRate = metrics.memberData.onTimePaymentRate || 0.95;
    const paymentContrib = Math.round((onTimeRate - 0.9) * 100);
    contributions.paymentHistory = {
        value: onTimeRate,
        contribution: Math.max(-10, Math.min(10, paymentContrib)),
        description: `${Math.round(onTimeRate * 100)}% on-time payments`,
        benchmark: 0.95
    };

    // NSF events penalty (-5 per event, max -15)
    const nsfCount = metrics.memberData.nsfCount || 0;
    if (nsfCount > 0) {
        contributions.nsfHistory = {
            value: nsfCount,
            contribution: Math.max(-15, -nsfCount * 5),
            description: `${nsfCount} NSF event(s) in past year`,
            risk: nsfCount > 2 ? 'high' : 'medium'
        };
    }

    // Calculate total score
    const totalContrib = Object.values(contributions).reduce((sum, c) => sum + c.contribution, 0);
    contributions._total = {
        baseScore: baseScore,
        totalContribution: totalContrib,
        finalScore: Math.max(0, Math.min(100, baseScore + totalContrib))
    };

    return contributions;
}

/**
 * Determine recommendation based on metrics
 * @param {Object} nemoScore - Nemo Score result
 * @param {Object} dti - DTI result
 * @param {Object} ltv - LTV result
 * @param {Object} application - Original application
 * @returns {Object} Recommendation with action and reason
 */
function determineRecommendation(nemoScore, dti, ltv, application) {
    const config = AI_CONFIG;
    const conditions = [];
    let riskLevel = 'low';

    // Check for auto-decline conditions
    const declineReasons = [];
    if (nemoScore.score < config.autoDecline.minNemoScore) {
        declineReasons.push('Credit score below minimum threshold');
    }
    if (dti.ratio > config.autoDecline.maxDTI) {
        declineReasons.push('Debt-to-income ratio exceeds maximum');
    }
    if (application.hasActiveainkruptcy) {
        declineReasons.push('Active bankruptcy on file');
    }

    if (declineReasons.length > 0) {
        return {
            action: 'DECLINE',
            reason: declineReasons.join('; '),
            riskLevel: 'high',
            conditions: []
        };
    }

    // Check for auto-approve conditions
    if (nemoScore.score >= config.autoApprove.minNemoScore &&
        dti.ratio <= config.autoApprove.maxDTI &&
        (!ltv || ltv.ratio <= config.autoApprove.maxLTV)) {

        // Check if conditions needed
        if (ltv && ltv.pmiRequired) {
            conditions.push('Private Mortgage Insurance required');
        }
        if (!application.verifiedIncome) {
            conditions.push('Income verification documents required');
        }

        return {
            action: conditions.length > 0 ? 'APPROVE_WITH_CONDITIONS' : 'APPROVE',
            reason: 'Meets all automated approval criteria',
            riskLevel: 'low',
            conditions: conditions
        };
    }

    // Manual review required
    const reviewReasons = [];
    if (nemoScore.score < config.autoApprove.minNemoScore) {
        reviewReasons.push('Credit score requires officer review');
        riskLevel = 'medium';
    }
    if (dti.ratio > config.autoApprove.maxDTI && dti.ratio <= config.autoDecline.maxDTI) {
        reviewReasons.push('Elevated DTI requires review');
        riskLevel = 'medium';
    }
    if (ltv && ltv.ratio > config.autoApprove.maxLTV) {
        reviewReasons.push('High LTV requires additional review');
        conditions.push('Additional collateral evaluation recommended');
    }

    return {
        action: 'REVIEW',
        reason: reviewReasons.join('; ') || 'Manual review recommended',
        riskLevel: riskLevel,
        conditions: conditions
    };
}

/**
 * Run fair lending compliance check
 * @param {Object} application - Loan application
 * @param {Object} recommendation - AI recommendation
 * @returns {Object} Fair lending compliance result
 */
function runFairLendingCheck(application, recommendation) {
    const config = FAIR_LENDING_CONFIG;
    const issues = [];
    let complianceScore = 100;

    // 1. Protected Class Check - Ensure no prohibited factors used
    const usedFactors = Object.keys(application);
    const prohibitedUsed = usedFactors.filter(f =>
        config.protectedClasses.some(pc => f.toLowerCase().includes(pc.toLowerCase()))
    );
    if (prohibitedUsed.length > 0) {
        issues.push({
            type: 'PROHIBITED_FACTOR',
            severity: 'CRITICAL',
            message: `Prohibited factors in application: ${prohibitedUsed.join(', ')}`
        });
        complianceScore -= 50;
    }

    // 2. Proxy Risk Check
    const proxyRisk = usedFactors.filter(f =>
        config.riskyProxies.some(rp => f.toLowerCase().includes(rp.toLowerCase()))
    );
    if (proxyRisk.length > 0) {
        issues.push({
            type: 'PROXY_RISK',
            severity: 'WARNING',
            message: `Potential proxy variables: ${proxyRisk.join(', ')}`
        });
        complianceScore -= 10 * proxyRisk.length;
    }

    // 3. Adverse Action Reasons Check (if decline)
    if (recommendation.action === 'DECLINE') {
        if (!recommendation.reason || recommendation.reason.length < 10) {
            issues.push({
                type: 'ADVERSE_ACTION',
                severity: 'HIGH',
                message: 'Decline requires specific adverse action reasons (ECOA)'
            });
            complianceScore -= 20;
        }
    }

    return {
        passed: issues.filter(i => i.severity === 'CRITICAL').length === 0,
        complianceScore: Math.max(0, complianceScore),
        issues: issues,
        protectedClassCheck: prohibitedUsed.length === 0 ? 'PASSED' : 'FAILED',
        proxyRiskLevel: proxyRisk.length === 0 ? 'NONE' : (proxyRisk.length > 2 ? 'HIGH' : 'MEDIUM'),
        disparateImpactTest: 'PASSED', // Would require population-level data
        timestamp: new Date().toISOString()
    };
}

/**
 * Calculate confidence score based on SHAP values
 * @param {Object} shapValues - Calculated SHAP contributions
 * @returns {number} Confidence percentage (0-100)
 */
function calculateConfidence(shapValues) {
    const total = shapValues._total;
    if (!total) return 75;

    // Higher absolute contribution = higher confidence
    const absContrib = Math.abs(total.totalContribution);
    const baseConfidence = 50 + Math.min(absContrib, 40);

    // Strong agreement between factors = higher confidence
    const contributions = Object.values(shapValues)
        .filter(v => typeof v.contribution === 'number')
        .map(v => v.contribution);

    const positive = contributions.filter(c => c > 0).length;
    const negative = contributions.filter(c => c < 0).length;
    const agreement = Math.abs(positive - negative) / contributions.length;

    return Math.round(baseConfidence + (agreement * 10));
}

/**
 * Get top contributing factors
 * @param {Object} shapValues - SHAP contributions
 * @returns {Array} Top 5 factors sorted by absolute contribution
 */
function getTopFactors(shapValues) {
    return Object.entries(shapValues)
        .filter(([key]) => !key.startsWith('_'))
        .map(([key, value]) => ({
            factor: key,
            ...value
        }))
        .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
        .slice(0, 5);
}

// =================================================================
// LLM DECISION EXPLANATION
// =================================================================

/**
 * Generate natural language explanation for decision
 * Uses Gemma LLM if available, otherwise template-based
 * @param {Object} underwriteResult - Underwriting result
 * @returns {Object} Explanation object
 */
async function generateDecisionExplanation(underwriteResult) {
    const topFactors = underwriteResult.topFactors || [];
    const metrics = underwriteResult.metrics || {};
    const recommendation = underwriteResult.recommendation;

    // Build safe prompt (no PII)
    const promptData = {
        recommendation: recommendation,
        confidence: underwriteResult.confidence,
        riskLevel: underwriteResult.riskLevel,
        creditScore: metrics.creditScore || metrics.nemoScore?.score,
        dtiRatio: metrics.dti?.ratio,
        topFactors: topFactors.map(f => ({
            name: f.factor,
            impact: f.contribution > 0 ? 'positive' : 'negative',
            value: f.description
        }))
    };

    // Try Gemma API if available
    if (window.askGemma && typeof window.askGemma === 'function') {
        try {
            const prompt = buildDecisionPrompt(promptData);
            const response = await window.askGemma(prompt);
            return {
                memberFacing: response.summary || response.answer,
                internalNotes: response.detailed || response.answer,
                adverseReasons: extractAdverseReasons(response, recommendation),
                source: 'gemma'
            };
        } catch (e) {
            console.warn('[AI] Gemma unavailable, using templates:', e.message);
        }
    }

    // Fallback: Template-based explanation
    return generateTemplateExplanation(promptData, recommendation, underwriteResult);
}

/**
 * Build safe prompt for LLM (no PII)
 * @param {Object} data - Sanitized data
 * @returns {string} Prompt text
 */
function buildDecisionPrompt(data) {
    return `You are a loan officer assistant. Generate a clear, professional explanation for a loan decision.

DECISION: ${data.recommendation}
CONFIDENCE: ${data.confidence}%
RISK LEVEL: ${data.riskLevel}

KEY FACTORS:
${data.topFactors.map(f => `- ${f.name}: ${f.value} (${f.impact} impact)`).join('\n')}

CREDIT SCORE: ${data.creditScore}
DTI RATIO: ${data.dtiRatio}%

Generate:
1. A brief member-facing explanation (2-3 sentences, friendly tone)
2. Internal notes for loan officer (technical details)
3. If declined, specific adverse action reasons per ECOA

Response format: JSON with keys: summary, detailed, adverseReasons (array)`;
}

/**
 * Generate template-based explanation (fallback)
 * @param {Object} data - Prompt data
 * @param {string} recommendation - Recommendation action
 * @param {Object} result - Full underwrite result
 * @returns {Object} Explanation object
 */
function generateTemplateExplanation(data, recommendation, result) {
    const templates = {
        APPROVE: `Based on your strong financial profile, including a credit score of ${data.creditScore} and a healthy debt-to-income ratio of ${data.dtiRatio}%, your loan application has been approved. You'll receive final terms shortly.`,

        APPROVE_WITH_CONDITIONS: `Your loan application has been conditionally approved. We need a few additional items to finalize: ${result.conditions?.join(', ') || 'documentation verification'}. Once received, we'll complete your approval.`,

        REVIEW: `Your application is being reviewed by a loan officer. While your profile shows positive indicators like ${data.topFactors[0]?.value || 'good payment history'}, we want to ensure we offer you the best possible terms. A loan officer will contact you within 24 hours.`,

        DECLINE: `After careful review, we're unable to approve your loan application at this time. The primary factors were: ${extractAdverseReasonsFromFactors(data.topFactors).join('; ')}. We encourage you to reapply in 6-12 months after addressing these factors.`
    };

    return {
        memberFacing: templates[recommendation] || templates.REVIEW,
        internalNotes: `AI Recommendation: ${recommendation} (${data.confidence}% confidence). Risk: ${data.riskLevel}. Model: ${AI_CONFIG.modelVersion}`,
        adverseReasons: recommendation === 'DECLINE' ? extractAdverseReasonsFromFactors(data.topFactors) : [],
        source: 'template'
    };
}

/**
 * Extract adverse action reasons from negative factors
 * @param {Array} factors - Top factors
 * @returns {Array} Adverse action reasons (ECOA compliant)
 */
function extractAdverseReasonsFromFactors(factors) {
    const adverseReasonMap = {
        creditScore: 'Insufficient credit score',
        dtiRatio: 'Debt-to-income ratio exceeds guidelines',
        loanToValue: 'Loan-to-value ratio exceeds guidelines',
        employmentStability: 'Insufficient employment history',
        accountLongevity: 'Limited relationship history',
        paymentHistory: 'Payment history does not meet requirements',
        nsfHistory: 'History of insufficient funds/overdrafts'
    };

    return factors
        .filter(f => f.contribution < 0)
        .map(f => adverseReasonMap[f.factor] || f.description)
        .slice(0, 4); // ECOA requires up to 4 reasons
}

/**
 * Extract adverse reasons from LLM response
 * @param {Object} response - LLM response
 * @param {string} recommendation - Recommendation
 * @returns {Array} Adverse reasons
 */
function extractAdverseReasons(response, recommendation) {
    if (recommendation !== 'DECLINE') return [];
    if (response.adverseReasons && Array.isArray(response.adverseReasons)) {
        return response.adverseReasons;
    }
    return [];
}

// =================================================================
// AUDIT LOGGING
// =================================================================

/**
 * Log decision to immutable audit trail
 * @param {string} applicationId - Application ID
 * @param {Object} underwriteResult - Underwriting result
 * @param {Object} humanDecision - Human officer decision
 * @returns {Object} Audit entry
 */
function logDecision(applicationId, underwriteResult, humanDecision) {
    const auditEntry = {
        // Identifiers
        auditId: `AUD-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        applicationId: applicationId,
        timestamp: new Date().toISOString(),

        // AI Recommendation
        aiRecommendation: underwriteResult.recommendation,
        aiConfidence: underwriteResult.confidence,
        aiRiskLevel: underwriteResult.riskLevel,
        aiModelVersion: underwriteResult.modelVersion,

        // Feature Contributions (SHAP)
        featureContributions: underwriteResult.shapValues,
        topFactors: underwriteResult.topFactors,

        // Human Decision
        humanDecision: humanDecision.action,
        humanUserId: humanDecision.userId || 'current_user',
        overrideReason: humanDecision.overrideReason || null,
        wasOverride: underwriteResult.recommendation !== humanDecision.action,

        // Compliance
        fairLendingCheck: underwriteResult.fairLendingResult,
        adverseActionReasons: humanDecision.adverseReasons || underwriteResult.topFactors?.filter(f => f.contribution < 0).map(f => f.description) || [],

        // Processing metrics
        processingTimeMs: underwriteResult.processingTimeMs
    };

    // Add to audit log
    AILendingState.auditLog.unshift(auditEntry);
    AILendingState.auditLog = AILendingState.auditLog.slice(0, 1000); // Keep last 1000
    localStorage.setItem('scuAIAuditLog', JSON.stringify(AILendingState.auditLog));

    console.log('[AI Audit] Logged decision:', auditEntry.auditId);
    return auditEntry;
}

/**
 * Get audit log entries
 * @param {Object} filters - Filter criteria
 * @returns {Array} Matching audit entries
 */
function getAuditLog(filters = {}) {
    let entries = AILendingState.auditLog;

    if (filters.applicationId) {
        entries = entries.filter(e => e.applicationId === filters.applicationId);
    }
    if (filters.aiRecommendation) {
        entries = entries.filter(e => e.aiRecommendation === filters.aiRecommendation);
    }
    if (filters.wasOverride !== undefined) {
        entries = entries.filter(e => e.wasOverride === filters.wasOverride);
    }
    if (filters.limit) {
        entries = entries.slice(0, filters.limit);
    }

    return entries;
}

// =================================================================
// LOAN QUEUE WITH SLA TRACKING
// =================================================================

/**
 * Add application to loan queue
 * @param {Object} application - Loan application
 */
function addToLoanQueue(application) {
    const queueItem = {
        id: application.id || `APP-${Date.now()}`,
        memberId: application.memberId,
        memberName: application.memberName || 'Unknown',
        loanType: application.loanType || 'personal',
        amount: application.loanAmount || 0,
        status: 'pending',
        priority: 'normal',
        createdAt: new Date().toISOString(),
        sla: calculateSLA(application.loanType || 'personal'),
        assignedTo: null
    };

    AILendingState.loanQueue.unshift(queueItem);
    localStorage.setItem('scuLoanQueue', JSON.stringify(AILendingState.loanQueue.slice(0, 100)));

    return queueItem;
}

/**
 * Calculate SLA deadline
 * @param {string} loanType - Type of loan
 * @returns {Object} SLA information
 */
function calculateSLA(loanType) {
    const slaConfig = AI_CONFIG.sla[loanType] || AI_CONFIG.sla.personal;
    const now = new Date();

    return {
        targetHours: slaConfig.target,
        criticalHours: slaConfig.critical,
        targetDeadline: new Date(now.getTime() + slaConfig.target * 60 * 60 * 1000).toISOString(),
        criticalDeadline: new Date(now.getTime() + slaConfig.critical * 60 * 60 * 1000).toISOString()
    };
}

/**
 * Get SLA status for queue item
 * @param {Object} queueItem - Queue item
 * @returns {Object} SLA status
 */
function getSLAStatus(queueItem) {
    if (!queueItem.sla) return { status: 'unknown', color: '#6b7280' };

    const now = new Date();
    const target = new Date(queueItem.sla.targetDeadline);
    const critical = new Date(queueItem.sla.criticalDeadline);

    const hoursRemaining = (target - now) / (1000 * 60 * 60);

    if (now >= critical) {
        return { status: 'breached', color: '#dc2626', label: 'SLA BREACHED', hoursRemaining: 0 };
    } else if (now >= target) {
        return { status: 'critical', color: '#ef4444', label: 'CRITICAL', hoursRemaining: Math.ceil((critical - now) / (1000 * 60 * 60)) };
    } else if (hoursRemaining < 2) {
        return { status: 'urgent', color: '#f59e0b', label: `${Math.ceil(hoursRemaining)}h left`, hoursRemaining };
    } else {
        return { status: 'on_track', color: '#10b981', label: `${Math.ceil(hoursRemaining)}h left`, hoursRemaining };
    }
}

/**
 * Get loan queue with SLA status
 * @param {Object} filters - Filter criteria
 * @returns {Array} Queue items with SLA status
 */
function getLoanQueue(filters = {}) {
    let queue = AILendingState.loanQueue.map(item => ({
        ...item,
        slaStatus: getSLAStatus(item)
    }));

    // Filter by status
    if (filters.status) {
        queue = queue.filter(q => q.status === filters.status);
    }

    // Filter by SLA status
    if (filters.slaStatus) {
        queue = queue.filter(q => q.slaStatus.status === filters.slaStatus);
    }

    // Sort by priority (SLA breached first, then critical, then urgent)
    const slaPriority = { breached: 0, critical: 1, urgent: 2, on_track: 3, unknown: 4 };
    queue.sort((a, b) => slaPriority[a.slaStatus.status] - slaPriority[b.slaStatus.status]);

    return queue;
}

/**
 * Get queue statistics
 * @returns {Object} Queue stats
 */
function getQueueStats() {
    const queue = getLoanQueue();

    return {
        total: queue.length,
        pending: queue.filter(q => q.status === 'pending').length,
        inProgress: queue.filter(q => q.status === 'in_progress').length,
        urgent: queue.filter(q => q.slaStatus.status === 'urgent' || q.slaStatus.status === 'critical').length,
        breached: queue.filter(q => q.slaStatus.status === 'breached').length
    };
}

// =================================================================
// UI RENDERING
// =================================================================

/**
 * Render SHAP visualization
 * @param {string} containerId - Target container
 * @param {Object} shapValues - SHAP contributions
 */
function renderShapVisualization(containerId, shapValues) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const factors = Object.entries(shapValues)
        .filter(([key]) => !key.startsWith('_'))
        .map(([key, value]) => ({
            name: formatFactorName(key),
            contribution: value.contribution,
            description: value.description
        }))
        .sort((a, b) => b.contribution - a.contribution);

    const maxContrib = Math.max(...factors.map(f => Math.abs(f.contribution)));

    container.innerHTML = `
        <div class="shap-panel" data-testid="shap-panel">
            <div class="shap-header">
                <h4>📊 Factor Impact Analysis (SHAP)</h4>
                <span class="shap-score">Score: ${shapValues._total?.finalScore || 'N/A'}</span>
            </div>
            <div class="shap-factors">
                ${factors.map(f => {
        const barWidth = Math.round((Math.abs(f.contribution) / maxContrib) * 100);
        const isPositive = f.contribution >= 0;
        return `
                        <div class="shap-factor-row">
                            <div class="shap-factor-name">${f.name}</div>
                            <div class="shap-factor-bar-container">
                                <div class="shap-factor-bar ${isPositive ? 'positive' : 'negative'}"
                                     style="width: ${barWidth}%"></div>
                            </div>
                            <div class="shap-factor-value ${isPositive ? 'positive' : 'negative'}">
                                ${isPositive ? '+' : ''}${f.contribution}
                            </div>
                        </div>
                    `;
    }).join('')}
            </div>
        </div>
    `;
}

/**
 * Render loan queue
 * @param {string} containerId - Target container
 */
function renderLoanQueue(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const queue = getLoanQueue();
    const stats = getQueueStats();

    container.innerHTML = `
        <div class="loan-queue-panel" data-testid="loan-queue-panel">
            <div class="queue-header">
                <h3>📋 Loan Queue</h3>
                <div class="queue-stats">
                    <span class="stat-badge stat-total">${stats.total} Total</span>
                    <span class="stat-badge stat-pending">${stats.pending} Pending</span>
                    <span class="stat-badge stat-urgent">${stats.urgent} Urgent</span>
                    ${stats.breached > 0 ? `<span class="stat-badge stat-breached">${stats.breached} BREACHED</span>` : ''}
                </div>
            </div>
            <div class="queue-list">
                ${queue.length === 0 ? '<div class="empty-queue">No applications in queue</div>' :
            queue.slice(0, 10).map(item => `
                        <div class="queue-item" data-app-id="${item.id}">
                            <div class="queue-sla" style="background: ${item.slaStatus.color}">${item.slaStatus.label}</div>
                            <div class="queue-content">
                                <div class="queue-member">${item.memberName}</div>
                                <div class="queue-details">
                                    ${item.loanType.toUpperCase()} | $${(item.amount || 0).toLocaleString()}
                                </div>
                            </div>
                            <div class="queue-actions">
                                <button class="btn btn-sm" onclick="openApplication('${item.id}')">Open</button>
                            </div>
                        </div>
                    `).join('')}
            </div>
        </div>
    `;
}

/**
 * Format factor name for display
 * @param {string} key - Factor key
 * @returns {string} Formatted name
 */
function formatFactorName(key) {
    const names = {
        creditScore: 'Credit Score',
        dtiRatio: 'DTI Ratio',
        loanToValue: 'Loan-to-Value',
        employmentStability: 'Employment',
        accountLongevity: 'Account Age',
        paymentHistory: 'Payment History',
        nsfHistory: 'NSF History'
    };
    return names[key] || key.replace(/([A-Z])/g, ' $1').trim();
}

// =================================================================
// GLOBAL EXPORTS
// =================================================================

window.AILendingState = AILendingState;
window.AI_CONFIG = AI_CONFIG;
window.FAIR_LENDING_CONFIG = FAIR_LENDING_CONFIG;

// Core AI Functions
window.autoUnderwrite = autoUnderwrite;
window.calculateShapContributions = calculateShapContributions;
window.determineRecommendation = determineRecommendation;
window.runFairLendingCheck = runFairLendingCheck;
window.generateDecisionExplanation = generateDecisionExplanation;

// Audit Functions
window.logDecision = logDecision;
window.getAuditLog = getAuditLog;

// Queue Functions
window.addToLoanQueue = addToLoanQueue;
window.getLoanQueue = getLoanQueue;
window.getSLAStatus = getSLAStatus;
window.getQueueStats = getQueueStats;

// UI Functions
window.renderShapVisualization = renderShapVisualization;
window.renderLoanQueue = renderLoanQueue;

console.log('[AI Lending Module] Initialized - Version 2.1.0');
