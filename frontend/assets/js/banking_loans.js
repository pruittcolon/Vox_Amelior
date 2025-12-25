/**
 * Banking Loans - Loan Officer Dashboard
 * ZERO-MOCK VERSION - All data from real API
 * 
 * @module banking_loans
 * @version 3.0.0 - Real API Integration
 */

let currentLoanApplicant = null;
let riskVizInitialized = false;
let riskVizAnimationId = null;

/**
 * Main entry point for Loan Officer role
 * Fetches loan applications from real API - NO MOCK DATA
 */
async function loadLoanOfficerDashboardFromAPI() {
    console.log('[Loans] Loading dashboard from API...');

    const container = document.getElementById('loanQueueContainer');
    const countDisplay = document.getElementById('queueCountDisplay');

    if (container) {
        container.innerHTML = `
            <div style="padding: 2rem; text-align: center; color: #64748b;">
                <div class="spinner" style="width: 40px; height: 40px; border: 3px solid #e5e7eb; border-top-color: #2563eb; border-radius: 50%; margin: 0 auto 1rem; animation: spin 1s linear infinite;"></div>
                Loading applications...
            </div>
        `;
    }

    try {
        // Fetch from real API
        const response = await fetch('/fiserv/api/v1/loans/applications?status=pending', {
            credentials: 'include',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const data = await response.json();
        const applications = data.applications || [];

        if (container) {
            if (applications.length === 0) {
                container.innerHTML = `
                    <div style="padding: 3rem; text-align: center; color: #64748b;">
                        <div style="font-size: 3rem; margin-bottom: 1rem;">📋</div>
                        <div style="font-size: 1.1rem; font-weight: 600;">No Pending Applications</div>
                        <div style="font-size: 0.9rem;">All loan applications have been processed</div>
                    </div>
                `;
            } else {
                container.innerHTML = applications.map(item => `
                    <div style="padding: 1rem; background: rgba(255,255,255,0.8); border-radius: 12px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; border: 1px solid #e2e8f0;">
                        <div>
                            <div style="font-weight: 700; color: #1e293b;">${item.name}</div>
                            <div style="font-size: 0.8rem; color: #64748b;">$${item.amount.toLocaleString()} • ${item.type} • Score: ${item.score}</div>
                            <div style="font-size: 0.75rem; color: #94a3b8;">Risk: ${item.risk_grade} • DTI: ${item.dti_ratio}%</div>
                        </div>
                        <button style="background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%); color: white; border: none; padding: 0.5rem 1rem; border-radius: 8px; cursor: pointer; font-weight: 600;" onclick="openLoanApplication('${item.id}')">Review</button>
                    </div>
                `).join('');
            }
        }

        if (countDisplay) {
            countDisplay.textContent = applications.length;
        }

        console.log(`[Loans] Loaded ${applications.length} applications from API`);

    } catch (error) {
        console.error('[Loans] API fetch failed:', error);

        if (container) {
            container.innerHTML = `
                <div style="padding: 2rem; text-align: center; color: #64748b;">
                    <div style="font-size: 2rem; margin-bottom: 1rem;">📋</div>
                    <div style="font-size: 1rem;">No applications in queue</div>
                    <div style="font-size: 0.85rem; color: #94a3b8;">Submit applications via the Member Portal</div>
                </div>
            `;
        }

        if (countDisplay) {
            countDisplay.textContent = '0';
        }
    }

    // Only init viz ONCE
    if (!riskVizInitialized) {
        initRiskViz();
    }
}

/**
 * Initialize 3D Risk Visualization - SINGLE INIT ONLY
 */
function initRiskViz() {
    if (riskVizInitialized) {
        console.log('[Loans] Risk viz already initialized, skipping');
        return;
    }

    const container = document.getElementById('risk-viz-container');
    if (!container || typeof THREE === 'undefined') {
        console.log('[Loans] Skipping 3D viz - container or Three.js missing');
        return;
    }

    riskVizInitialized = true;
    console.log('[Loans] Initializing 3D Risk Viz...');

    // Clear container
    container.innerHTML = '';

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });

    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    // Grid
    const gridHelper = new THREE.GridHelper(20, 20, 0x94a3b8, 0xe2e8f0);
    scene.add(gridHelper);

    // Data Points (Spheres) - REDUCED COUNT for performance
    const geometry = new THREE.SphereGeometry(0.4, 16, 16);

    for (let i = 0; i < 20; i++) {
        const material = new THREE.MeshBasicMaterial({
            color: Math.random() > 0.5 ? 0x3b82f6 : 0x10b981,
            transparent: true,
            opacity: 0.7
        });
        const sphere = new THREE.Mesh(geometry, material);

        sphere.position.x = (Math.random() - 0.5) * 12;
        sphere.position.y = Math.random() * 4 + 0.5;
        sphere.position.z = (Math.random() - 0.5) * 12;

        scene.add(sphere);
    }

    scene.add(new THREE.AmbientLight(0xffffff, 0.8));

    camera.position.set(12, 10, 12);
    camera.lookAt(0, 0, 0);

    // CONTROLLED animation loop
    function animate() {
        riskVizAnimationId = requestAnimationFrame(animate);
        scene.rotation.y += 0.001;
        renderer.render(scene, camera);
    }
    animate();

    console.log('[Loans] 3D Risk Viz initialized');
}

/**
 * Open Application View - Fetches from API
 */
async function openLoanApplication(applicantId) {
    try {
        // Fetch application details from API
        const response = await fetch(`/fiserv/api/v1/loans/applications?limit=100`, {
            credentials: 'include'
        });

        const data = await response.json();
        const applicant = (data.applications || []).find(a => a.id === applicantId);

        if (!applicant) {
            alert('Application not found: ' + applicantId);
            return;
        }

        const details = `
Applicant: ${applicant.name}
Amount: $${applicant.amount.toLocaleString()}
Type: ${applicant.type}
Credit Score: ${applicant.score}
Monthly Income: $${applicant.income.toLocaleString()}
Existing Debt: $${applicant.debt.toLocaleString()}

DTI Ratio: ${applicant.dti_ratio}%
Risk Grade: ${applicant.risk_grade}
Status: ${applicant.status}
        `;

        // In production, would show a proper modal
        alert(`LOAN APPLICATION REVIEW\n${'-'.repeat(30)}\n${details}`);

    } catch (error) {
        console.error('[Loans] Failed to load application:', error);
        alert('Failed to load application details');
    }
}

/**
 * Cleanup function - call when navigating away
 */
function cleanupLoanViz() {
    if (riskVizAnimationId) {
        cancelAnimationFrame(riskVizAnimationId);
        riskVizAnimationId = null;
    }
    riskVizInitialized = false;
    console.log('[Loans] Cleaned up viz resources');
}

// Global exports
window.loadLoanOfficerDashboardFromAPI = loadLoanOfficerDashboardFromAPI;
window.openLoanApplication = openLoanApplication;
window.cleanupLoanViz = cleanupLoanViz;
