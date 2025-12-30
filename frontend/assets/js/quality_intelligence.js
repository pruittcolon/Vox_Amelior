/**
 * Quality Intelligence Dashboard Module
 * 
 * Self-contained 3D Quality Intelligence visualization module.
 * Dynamically injects HTML, initializes Three.js, and renders quality data.
 * 
 * Following modular microservices architecture for frontend components.
 * 
 * @author Enterprise Analytics Team
 * @security CSP-compliant
 */

(function (global) {
    'use strict';

    // =========================================
    // Configuration
    // =========================================
    const CONFIG = {
        containerId: 'quality-3d-dashboard',
        viewerId: 'quality-3d-viewer',
        apiBase: '',  // Uses relative URLs
        // Local vendor files (CSP compliant)
        threejsPath: 'assets/js/vendor/three.min.js',
        orbitControlsPath: 'assets/js/vendor/OrbitControls.js',
        cssPath: 'assets/css/nexus/quality-3d.css',
    };

    // =========================================
    // State
    // =========================================
    let state = {
        initialized: false,
        visible: false,
        threeJsLoaded: false,
        viewer: null,
        currentFile: null,
    };

    // =========================================
    // HTML Template
    // =========================================
    const DASHBOARD_HTML = `
    <div id="${CONFIG.containerId}" class="quality-dashboard-module" style="margin-top: 1.5rem; display: none; padding: 0; overflow: hidden; border-radius: 1rem; background: linear-gradient(135deg, #0f0f17 0%, #1a1a2e 100%); border: 1px solid rgba(99, 102, 241, 0.2);">
        <!-- Dashboard Header -->
        <div style="padding: 1.25rem 1.5rem; border-bottom: 1px solid rgba(99, 102, 241, 0.2);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="font-size: 1.25rem; font-weight: 700; color: #ffffff; margin: 0 0 0.25rem 0; display: flex; align-items: center; gap: 0.5rem;">
                        🏔️ Quality Intelligence Dashboard
                        <span style="font-size: 0.75rem; padding: 0.25rem 0.5rem; background: linear-gradient(135deg, #22c55e, #16a34a); border-radius: 9999px; color: white;">✨ 3D</span>
                    </h3>
                    <p style="color: #888; margin: 0; font-size: 0.85rem;">Explore your data quality in immersive 3D visualization</p>
                </div>
                <div style="display: flex; gap: 0.5rem;">
                    <button class="btn btn-ghost btn-sm" onclick="QualityIntelligence.hide()" style="color: #888; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); padding: 0.5rem 0.75rem; border-radius: 0.5rem; cursor: pointer;">
                        ✕ Close
                    </button>
                </div>
            </div>
        </div>

        <!-- 3D Viewer Container -->
        <div class="quality-3d-container" id="${CONFIG.viewerId}" style="height: 500px; position: relative; background: #0a0a0f;">
            <!-- Three.js renders here -->
            
            <!-- Loading Indicator -->
            <div id="quality-loading" style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; color: #888;">
                <div style="width: 40px; height: 40px; border: 3px solid rgba(99, 102, 241, 0.2); border-top-color: #6366f1; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 1rem;"></div>
                <p style="margin: 0;">Initializing 3D visualization...</p>
            </div>
            
            <!-- Savings Panel Overlay -->
            <div class="savings-panel" id="quality-savings-panel" style="display: none; position: absolute; top: 1rem; left: 1rem; max-width: 280px; padding: 1rem; background: rgba(15, 15, 25, 0.95); border: 1px solid rgba(34, 197, 94, 0.3); border-radius: 0.75rem; backdrop-filter: blur(10px);">
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.75rem;">
                    <div style="width: 36px; height: 36px; display: flex; align-items: center; justify-content: center; background: rgba(34, 197, 94, 0.2); border-radius: 0.5rem; font-size: 1.25rem;">💰</div>
                    <div>
                        <div style="font-size: 0.75rem; color: #888;">Estimated Annual Savings</div>
                        <div id="quality-savings-total" style="font-size: 1.5rem; font-weight: 700; color: #22c55e;">$0</div>
                    </div>
                </div>
                <div id="quality-savings-issues" style="border-top: 1px solid rgba(255,255,255,0.1); padding-top: 0.75rem;">
                    <!-- Populated dynamically -->
                </div>
            </div>

            <!-- Quality Legend -->
            <div class="quality-legend" style="position: absolute; bottom: 1rem; right: 1rem; padding: 0.75rem 1rem; background: rgba(15, 15, 25, 0.9); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 0.5rem; backdrop-filter: blur(10px);">
                <div style="font-size: 0.7rem; color: #888; text-transform: uppercase; margin-bottom: 0.5rem;">Quality Score</div>
                <div style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.75rem; color: #c0c0c0; margin-bottom: 0.35rem;">
                    <div style="width: 10px; height: 10px; border-radius: 2px; background: #22c55e;"></div>
                    <span>8-10: High (ML Ready)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.75rem; color: #c0c0c0; margin-bottom: 0.35rem;">
                    <div style="width: 10px; height: 10px; border-radius: 2px; background: #f59e0b;"></div>
                    <span>6-7: Medium (Review)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.75rem; color: #c0c0c0;">
                    <div style="width: 10px; height: 10px; border-radius: 2px; background: #ef4444;"></div>
                    <span>1-5: Low (Exclude)</span>
                </div>
            </div>
        </div>

        <!-- Info Panel -->
        <div style="display: flex; gap: 1rem; padding: 1rem 1.5rem; background: #0f0f17; border-top: 1px solid rgba(99, 102, 241, 0.1);">
            <div style="flex: 1; padding: 0.75rem 1rem; background: rgba(15, 15, 25, 0.9); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 0.5rem;">
                <div style="font-size: 0.7rem; color: #888; text-transform: uppercase;">ML Readiness</div>
                <div id="quality-stat-ml-readiness" style="font-size: 1.25rem; font-weight: 700; background: linear-gradient(135deg, #22c55e, #16a34a); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">-%</div>
            </div>
            <div style="flex: 1; padding: 0.75rem 1rem; background: rgba(15, 15, 25, 0.9); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 0.5rem;">
                <div style="font-size: 0.7rem; color: #888; text-transform: uppercase;">Avg Quality</div>
                <div id="quality-stat-avg-quality" style="font-size: 1.25rem; font-weight: 700; background: linear-gradient(135deg, #6366f1, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">-/10</div>
            </div>
            <div style="flex: 1; padding: 0.75rem 1rem; background: rgba(15, 15, 25, 0.9); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 0.5rem;">
                <div style="font-size: 0.7rem; color: #888; text-transform: uppercase;">Problem Rows</div>
                <div id="quality-stat-problem-rows" style="font-size: 1.25rem; font-weight: 700; background: linear-gradient(135deg, #ef4444, #dc2626); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">-</div>
            </div>
            <div style="flex: 1; padding: 0.75rem 1rem; background: rgba(15, 15, 25, 0.9); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 0.5rem;">
                <div style="font-size: 0.7rem; color: #888; text-transform: uppercase;">Data Points</div>
                <div id="quality-stat-data-points" style="font-size: 1.25rem; font-weight: 700; color: #e0e0e0;">-</div>
            </div>
        </div>

        <!-- Recommendations -->
        <div style="padding: 1.25rem 1.5rem; background: rgba(255,255,255,0.02);">
            <h4 style="font-size: 0.85rem; font-weight: 600; color: #e0e0e0; margin: 0 0 0.75rem 0;">📋 AI Recommendations</h4>
            <div id="quality-recommendations" style="display: flex; flex-direction: column; gap: 0.5rem;">
                <p style="color: #666; margin: 0; font-size: 0.85rem;">Analyzing data quality...</p>
            </div>
        </div>
    </div>

    <style>
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .quality-dashboard-module {
            transition: all 0.3s ease;
        }
        .quality-dashboard-module canvas {
            display: block;
        }
    </style>
    `;

    // =========================================
    // CSS Loader
    // =========================================
    function loadCSS() {
        if (document.querySelector(`link[href*="quality-3d.css"]`)) return;

        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.href = CONFIG.cssPath;
        document.head.appendChild(link);
    }

    // =========================================
    // Three.js Loader
    // =========================================
    async function loadThreeJS() {
        if (state.threeJsLoaded || typeof THREE !== 'undefined') {
            state.threeJsLoaded = true;
            return true;
        }

        return new Promise((resolve) => {
            // Load Three.js from local vendor
            const threeScript = document.createElement('script');
            threeScript.src = CONFIG.threejsPath;
            threeScript.onload = () => {
                // Load OrbitControls from local vendor
                const controlsScript = document.createElement('script');
                controlsScript.src = CONFIG.orbitControlsPath;
                controlsScript.onload = () => {
                    state.threeJsLoaded = true;
                    console.log('✅ Three.js loaded from local vendor');
                    resolve(true);
                };
                controlsScript.onerror = () => {
                    console.warn('⚠️ OrbitControls failed to load');
                    state.threeJsLoaded = true;
                    resolve(true);
                };
                document.head.appendChild(controlsScript);
            };
            threeScript.onerror = () => {
                console.error('❌ Three.js failed to load');
                resolve(false);
            };
            document.head.appendChild(threeScript);
        });
    }

    // =========================================
    // HTML Injection
    // =========================================
    function injectHTML(targetSelector) {
        // Find or create container
        let target;
        if (targetSelector) {
            target = document.querySelector(targetSelector);
        }

        // If no target, look for common containers
        if (!target) {
            target = document.querySelector('#scoring-results-section')?.parentElement ||
                document.querySelector('.gemma-shell') ||
                document.querySelector('main') ||
                document.body;
        }

        // Check if already injected
        if (document.getElementById(CONFIG.containerId)) {
            return document.getElementById(CONFIG.containerId);
        }

        // Inject HTML
        const wrapper = document.createElement('div');
        wrapper.innerHTML = DASHBOARD_HTML;
        const dashboard = wrapper.firstElementChild;

        // Insert AFTER database-excel-viewer (data table) so 3D is below the chart
        const databaseViewer = document.querySelector('#database-excel-viewer');
        if (databaseViewer && databaseViewer.parentElement) {
            databaseViewer.parentElement.insertBefore(dashboard, databaseViewer.nextSibling);
        } else {
            // Fallback: after scoring results
            const scoringResults = document.querySelector('#scoring-results-section');
            if (scoringResults && scoringResults.parentElement) {
                scoringResults.parentElement.insertBefore(dashboard, scoringResults.nextSibling);
            } else {
                target.appendChild(dashboard);
            }
        }

        console.log('✅ Quality Intelligence Dashboard injected');
        return dashboard;
    }

    // =========================================
    // Three.js Viewer Class
    // =========================================
    class Quality3DViewer {
        constructor(containerId) {
            this.containerId = containerId;
            this.container = null;
            this.scene = null;
            this.camera = null;
            this.renderer = null;
            this.controls = null;
            this.isRunning = false;
            this.particlePositions = null;

            // Interactivity
            this.raycaster = null;
            this.mouse = null;
            this.tooltip = null;
            this.pillars = [];
            this.terrainData = [];
            this.hoveredObject = null;
        }

        init() {
            if (typeof THREE === 'undefined') {
                console.error('Three.js not loaded');
                return false;
            }

            this.container = document.getElementById(this.containerId);
            if (!this.container) {
                console.error('Container not found:', this.containerId);
                return false;
            }

            // Hide loading indicator
            const loading = document.getElementById('quality-loading');
            if (loading) loading.style.display = 'none';

            // Scene
            this.scene = new THREE.Scene();
            this.scene.background = new THREE.Color(0x0a0a0f);
            this.scene.fog = new THREE.Fog(0x0a0a0f, 50, 200);

            // Camera
            const aspect = this.container.clientWidth / this.container.clientHeight;
            this.camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 1000);
            this.camera.position.set(35, 25, 35);
            this.camera.lookAt(0, 5, 0);

            // Renderer
            this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
            this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
            this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            this.renderer.shadowMap.enabled = true;
            this.container.insertBefore(this.renderer.domElement, this.container.firstChild);

            // Controls
            if (typeof THREE.OrbitControls !== 'undefined') {
                this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
                this.controls.enableDamping = true;
                this.controls.dampingFactor = 0.05;
                this.controls.minDistance = 15;
                this.controls.maxDistance = 100;
                this.controls.maxPolarAngle = Math.PI / 2.1;
            }

            // Lighting
            this._setupLighting();

            // Grid
            const grid = new THREE.GridHelper(50, 50, 0x1a1a25, 0x1a1a25);
            grid.position.y = -0.01;
            this.scene.add(grid);

            // Raycaster for interactivity
            this.raycaster = new THREE.Raycaster();
            this.mouse = new THREE.Vector2();

            // Create tooltip element
            this._createTooltip();

            // Mouse event handlers
            this.container.addEventListener('mousemove', (e) => this._onMouseMove(e));
            this.container.addEventListener('click', (e) => this._onClick(e));

            // Resize handler
            window.addEventListener('resize', () => this._onResize());

            console.log('🎬 Quality3DViewer initialized');
            return true;
        }

        _createTooltip() {
            this.tooltip = document.createElement('div');
            this.tooltip.style.cssText = `
                position: absolute;
                padding: 0.75rem 1rem;
                background: rgba(15, 15, 25, 0.95);
                border: 1px solid rgba(99, 102, 241, 0.4);
                border-radius: 0.5rem;
                color: #e0e0e0;
                font-size: 0.8rem;
                pointer-events: none;
                display: none;
                z-index: 100;
                backdrop-filter: blur(8px);
                box-shadow: 0 4px 20px rgba(0,0,0,0.4);
                max-width: 250px;
            `;
            this.container.appendChild(this.tooltip);
        }

        _onMouseMove(event) {
            if (!this.raycaster || !this.camera) return;

            const rect = this.container.getBoundingClientRect();
            this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

            this.raycaster.setFromCamera(this.mouse, this.camera);

            // Check intersections with pillars
            const intersects = this.raycaster.intersectObjects(this.pillars);

            if (intersects.length > 0) {
                const obj = intersects[0].object;
                if (obj.userData && obj.userData.rowData) {
                    this.container.style.cursor = 'pointer';
                    this._showTooltip(event, obj.userData.rowData);

                    // Highlight effect
                    if (this.hoveredObject !== obj) {
                        if (this.hoveredObject) this.hoveredObject.material.emissiveIntensity = 0.1;
                        obj.material.emissiveIntensity = 0.5;
                        this.hoveredObject = obj;
                    }
                    return;
                }
            }

            // No intersection
            if (this.hoveredObject) {
                this.hoveredObject.material.emissiveIntensity = 0.1;
                this.hoveredObject = null;
            }
            this.container.style.cursor = 'grab';
            this._hideTooltip();
        }

        _onClick(event) {
            if (!this.raycaster || !this.camera) return;

            const rect = this.container.getBoundingClientRect();
            this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

            this.raycaster.setFromCamera(this.mouse, this.camera);
            const intersects = this.raycaster.intersectObjects(this.pillars);

            if (intersects.length > 0 && intersects[0].object.userData.rowData) {
                const data = intersects[0].object.userData.rowData;
                console.log('📊 Row clicked:', data);
                // Future: Open detail panel or highlight in database viewer
            }
        }

        _showTooltip(event, data) {
            if (!this.tooltip) return;

            const score = data.quality_score || 0;
            const category = score >= 8 ? '🟢 High Quality' :
                score >= 6 ? '🟡 Medium Quality' : '🔴 Low Quality';

            this.tooltip.innerHTML = `
                <div style="font-weight: 600; margin-bottom: 0.5rem; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 0.35rem;">
                    Row ${data.row_index + 1} ${category}
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.25rem;">
                    <span style="color: #888;">Overall:</span>
                    <span style="font-weight: 600; color: ${this._getScoreColor(score)};">${score.toFixed(1)}/10</span>
                    <span style="color: #888;">Q1 Anomaly:</span>
                    <span>${data.q1?.toFixed(1) || 'N/A'}</span>
                    <span style="color: #888;">Q2 Business:</span>
                    <span>${data.q2?.toFixed(1) || 'N/A'}</span>
                    <span style="color: #888;">Q3 Validity:</span>
                    <span>${data.q3?.toFixed(1) || 'N/A'}</span>
                    <span style="color: #888;">Q4 Complete:</span>
                    <span>${data.q4?.toFixed(1) || 'N/A'}</span>
                    <span style="color: #888;">Q5 Consistent:</span>
                    <span>${data.q5?.toFixed(1) || 'N/A'}</span>
                </div>
            `;

            const rect = this.container.getBoundingClientRect();
            this.tooltip.style.left = (event.clientX - rect.left + 15) + 'px';
            this.tooltip.style.top = (event.clientY - rect.top + 15) + 'px';
            this.tooltip.style.display = 'block';
        }

        _hideTooltip() {
            if (this.tooltip) this.tooltip.style.display = 'none';
        }

        _getScoreColor(score) {
            if (score >= 8) return '#22c55e';
            if (score >= 6) return '#f59e0b';
            return '#ef4444';
        }

        _setupLighting() {
            const ambient = new THREE.AmbientLight(0x404050, 0.5);
            this.scene.add(ambient);

            const directional = new THREE.DirectionalLight(0xffffff, 1.0);
            directional.position.set(50, 50, 25);
            directional.castShadow = true;
            this.scene.add(directional);

            const blueLight = new THREE.PointLight(0x6366f1, 0.5, 50);
            blueLight.position.set(-20, 10, -20);
            this.scene.add(blueLight);

            const purpleLight = new THREE.PointLight(0xa855f7, 0.5, 50);
            purpleLight.position.set(20, 10, 20);
            this.scene.add(purpleLight);
        }

        _onResize() {
            if (!this.container || !this.camera || !this.renderer) return;
            const width = this.container.clientWidth;
            const height = this.container.clientHeight;
            this.camera.aspect = width / height;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(width, height);
        }

        start() {
            if (this.isRunning) return;
            this.isRunning = true;
            this._animate();
        }

        stop() {
            this.isRunning = false;
        }

        _animate() {
            if (!this.isRunning) return;
            requestAnimationFrame(() => this._animate());
            if (this.controls) this.controls.update();
            if (this.particlePositions) this._animateParticles();
            if (this.renderer && this.scene && this.camera) {
                this.renderer.render(this.scene, this.camera);
            }
        }

        _animateParticles() {
            const particles = this.scene.getObjectByName('particles');
            if (!particles) return;
            const positions = particles.geometry.attributes.position.array;
            const time = Date.now() * 0.001;
            for (let i = 0; i < positions.length; i += 3) {
                positions[i] += Math.sin(time + i) * 0.01;
                positions[i + 1] += Math.cos(time + i * 0.5) * 0.005;
                if (positions[i] > 20) positions[i] = -20;
                if (positions[i] < -20) positions[i] = 20;
            }
            particles.geometry.attributes.position.needsUpdate = true;
        }

        renderTerrain(terrainData) {
            // Remove existing terrain and pillars
            const existingTerrain = this.scene.getObjectByName('terrain');
            if (existingTerrain) this.scene.remove(existingTerrain);

            this.pillars.forEach(p => this.scene.remove(p));
            this.pillars = [];

            if (!terrainData || !terrainData.length) return;

            // Store terrain data for interactivity
            this.terrainData = terrainData;

            const size = 40;
            const segments = Math.min(50, Math.ceil(Math.sqrt(terrainData.length)));
            const geometry = new THREE.PlaneGeometry(size, size, segments - 1, segments - 1);
            const positions = geometry.attributes.position;
            const colors = [];

            for (let i = 0; i < positions.count; i++) {
                const dataIndex = Math.floor(i / positions.count * terrainData.length);
                const point = terrainData[dataIndex] || { quality_score: 5 };
                const height = (point.quality_score / 10) * 8;
                positions.setZ(i, height);
                const color = this._getQualityColor(point.quality_score);
                colors.push(color.r, color.g, color.b);
            }

            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
            geometry.computeVertexNormals();

            const material = new THREE.MeshStandardMaterial({
                vertexColors: true, roughness: 0.7, metalness: 0.1, side: THREE.DoubleSide
            });

            const mesh = new THREE.Mesh(geometry, material);
            mesh.rotation.x = -Math.PI / 2;
            mesh.receiveShadow = true;
            mesh.name = 'terrain';
            this.scene.add(mesh);

            // Create interactive pillars (limit to 100 for performance)
            this._createPillars(terrainData.slice(0, 100));
        }

        _createPillars(data) {
            const gridSize = Math.ceil(Math.sqrt(data.length));
            const spacing = 35 / gridSize;
            const startX = -17.5 + spacing / 2;
            const startZ = -17.5 + spacing / 2;

            data.forEach((point, i) => {
                const row = Math.floor(i / gridSize);
                const col = i % gridSize;
                const x = startX + col * spacing;
                const z = startZ + row * spacing;
                const height = Math.max(0.5, (point.quality_score / 10) * 6);

                const geometry = new THREE.CylinderGeometry(0.3, 0.4, height, 8);
                const color = this._getQualityColor(point.quality_score);
                const material = new THREE.MeshStandardMaterial({
                    color: color,
                    emissive: color,
                    emissiveIntensity: 0.1,
                    roughness: 0.5,
                    metalness: 0.3
                });

                const pillar = new THREE.Mesh(geometry, material);
                pillar.position.set(x, height / 2, z);
                pillar.castShadow = true;
                pillar.userData.rowData = point;

                this.scene.add(pillar);
                this.pillars.push(pillar);
            });
        }

        renderParticles(riskDistribution) {
            const existing = this.scene.getObjectByName('particles');
            if (existing) this.scene.remove(existing);

            const { high = 0, medium = 0, low = 0 } = riskDistribution;
            const total = high + medium + low;
            if (!total) return;

            const count = 500;
            const positions = new Float32Array(count * 3);
            const colors = new Float32Array(count * 3);

            for (let i = 0; i < count; i++) {
                positions[i * 3] = (Math.random() - 0.5) * 40;
                positions[i * 3 + 1] = 2 + Math.random() * 10;
                positions[i * 3 + 2] = (Math.random() - 0.5) * 40;

                const rand = Math.random() * total;
                let color = rand < high ? new THREE.Color(0x22c55e) :
                    rand < high + medium ? new THREE.Color(0xf59e0b) :
                        new THREE.Color(0xef4444);
                colors[i * 3] = color.r;
                colors[i * 3 + 1] = color.g;
                colors[i * 3 + 2] = color.b;
            }

            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

            const material = new THREE.PointsMaterial({
                size: 0.3, vertexColors: true, transparent: true, opacity: 0.8,
                blending: THREE.AdditiveBlending
            });

            const particles = new THREE.Points(geometry, material);
            particles.name = 'particles';
            this.scene.add(particles);
            this.particlePositions = positions;
        }

        _getQualityColor(score) {
            if (score >= 8) return new THREE.Color(0x22c55e);
            if (score >= 6) return new THREE.Color(0xf59e0b);
            if (score >= 4) return new THREE.Color(0xf97316);
            return new THREE.Color(0xef4444);
        }

        dispose() {
            this.stop();
            if (this.renderer) {
                this.renderer.dispose();
                if (this.container && this.renderer.domElement) {
                    this.container.removeChild(this.renderer.domElement);
                }
            }
        }
    }

    // =========================================
    // Public API
    // =========================================
    const QualityIntelligence = {
        /**
         * Initialize the module
         * @param {string} targetSelector - Optional CSS selector for container
         */
        async init(targetSelector) {
            if (state.initialized) return true;

            loadCSS();
            await loadThreeJS();
            injectHTML(targetSelector);

            state.initialized = true;
            console.log('✅ QualityIntelligence module initialized');
            return true;
        },

        /**
         * Show the 3D dashboard and load data
         * @param {string} insightsFile - Name of the _insights.csv file
         */
        async show(insightsFile) {
            if (!state.initialized) {
                await this.init();
            }

            const dashboard = document.getElementById(CONFIG.containerId);
            if (!dashboard) {
                console.error('Dashboard container not found');
                return;
            }

            // Show dashboard
            dashboard.style.display = 'block';
            state.visible = true;
            state.currentFile = insightsFile;

            // Scroll to dashboard
            dashboard.scrollIntoView({ behavior: 'smooth' });

            // Initialize 3D viewer
            if (!state.viewer) {
                state.viewer = new Quality3DViewer(CONFIG.viewerId);
                if (!state.viewer.init()) {
                    console.error('Failed to initialize 3D viewer');
                    return;
                }
                state.viewer.start();
            }

            // Load data
            if (insightsFile) {
                await this.loadData(insightsFile);
            }
        },

        /**
         * Hide the dashboard
         */
        hide() {
            const dashboard = document.getElementById(CONFIG.containerId);
            if (dashboard) {
                dashboard.style.display = 'none';
            }
            if (state.viewer) {
                state.viewer.stop();
            }
            state.visible = false;
        },

        /**
         * Load and render quality data
         * @param {string} insightsFile - Name of the _insights.csv file
         */
        async loadData(insightsFile) {
            if (!insightsFile) return;

            try {
                // Fetch quality analysis
                const response = await fetch(`${CONFIG.apiBase}/quality-insights/analyze`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    credentials: 'include',
                    body: JSON.stringify({ insights_file: insightsFile })
                });

                if (!response.ok) {
                    throw new Error(`Analysis failed: ${response.status}`);
                }

                const data = await response.json();

                // Update 3D visualization
                if (state.viewer) {
                    state.viewer.renderTerrain(data.terrain_data || []);
                    state.viewer.renderParticles(data.risk_distribution || {});
                }

                // Update stats
                this._updateStats(data);

                // Update recommendations
                this._updateRecommendations(data.recommendations || []);

                // Load business savings
                await this._loadSavings(insightsFile);

                console.log('📊 Quality data loaded:', data);

            } catch (error) {
                console.error('Failed to load quality data:', error);
                this._showError('Failed to load quality data: ' + error.message);
            }
        },

        /**
         * Update stats display
         */
        _updateStats(data) {
            const setEl = (id, value) => {
                const el = document.getElementById(id);
                if (el) el.textContent = value;
            };

            setEl('quality-stat-ml-readiness', (data.ml_readiness || 0).toFixed(1) + '%');
            setEl('quality-stat-avg-quality', (data.avg_overall || 0).toFixed(1) + '/10');
            setEl('quality-stat-problem-rows', (data.problem_row_count || 0).toLocaleString());
            setEl('quality-stat-data-points', (data.row_count || 0).toLocaleString());
        },

        /**
         * Update recommendations display
         */
        _updateRecommendations(recommendations) {
            const container = document.getElementById('quality-recommendations');
            if (!container) return;

            if (!recommendations.length) {
                container.innerHTML = '<p style="color: #666; margin: 0; font-size: 0.85rem;">No recommendations available.</p>';
                return;
            }

            container.innerHTML = recommendations.slice(0, 5).map(rec => `
                <div style="padding: 0.6rem 0.75rem; background: rgba(255,255,255,0.03); border-radius: 0.5rem; border-left: 3px solid #6366f1; font-size: 0.85rem; color: #c0c0c0;">
                    ${rec}
                </div>
            `).join('');
        },

        /**
         * Load business savings data
         */
        async _loadSavings(insightsFile) {
            try {
                const response = await fetch(`${CONFIG.apiBase}/quality-insights/business-savings`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    credentials: 'include',
                    body: JSON.stringify({ insights_file: insightsFile, use_llm: false })
                });

                if (!response.ok) return;

                const data = await response.json();

                // Show savings panel
                const panel = document.getElementById('quality-savings-panel');
                if (panel) panel.style.display = 'block';

                // Update total
                const total = document.getElementById('quality-savings-total');
                if (total) total.textContent = data.estimated_annual_savings || '$0';

                // Update issues
                const issues = document.getElementById('quality-savings-issues');
                if (issues && data.high_impact_issues) {
                    issues.innerHTML = data.high_impact_issues.slice(0, 3).map(issue => `
                        <div style="display: flex; justify-content: space-between; padding: 0.35rem 0; font-size: 0.8rem;">
                            <span style="color: #e0e0e0;">${issue.issue}</span>
                            <span style="color: #f59e0b; font-weight: 600;">${issue.impact}</span>
                        </div>
                    `).join('');
                }

            } catch (e) {
                console.warn('Could not load business savings:', e);
            }
        },

        /**
         * Show error message
         */
        _showError(message) {
            const container = document.getElementById('quality-recommendations');
            if (container) {
                container.innerHTML = `<p style="color: #ef4444; margin: 0; font-size: 0.85rem;">❌ ${message}</p>`;
            }
        },

        /**
         * Check if visible
         */
        isVisible() {
            return state.visible;
        },

        /**
         * Dispose resources
         */
        dispose() {
            if (state.viewer) {
                state.viewer.dispose();
                state.viewer = null;
            }
            const dashboard = document.getElementById(CONFIG.containerId);
            if (dashboard) dashboard.remove();
            state.initialized = false;
            state.visible = false;
        }
    };

    // =========================================
    // Export to global scope
    // =========================================
    global.QualityIntelligence = QualityIntelligence;

    // Legacy function names for backward compatibility
    global.showQuality3D = function (insightsFile) {
        QualityIntelligence.show(insightsFile);
    };
    global.hideQuality3D = function () {
        QualityIntelligence.hide();
    };

    console.log('📦 QualityIntelligence module loaded');

})(typeof window !== 'undefined' ? window : this);
