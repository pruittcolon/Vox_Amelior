/**
 * Quality 3D Viewer - Main Entry Point
 * 
 * Orchestrates all 3D visualization components for the Quality Intelligence Dashboard.
 * Loads data from API and renders 3D scene.
 * 
 * @author Enterprise Analytics Team
 * @security CSP-compliant, JWT-authenticated API calls
 */

// Note: In production, use ES6 modules with bundler
// For now, we load Three.js from CDN and use window globals

class Quality3DViewer {
    constructor(containerId) {
        this.containerId = containerId;
        this.container = null;

        // Three.js components
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;

        // Visualization components (will be initialized after scene)
        this.terrain = null;
        this.pillars = null;
        this.particles = null;
        this.stars = null;

        // State
        this.isInitialized = false;
        this.isAnimating = false;
        this.currentData = null;
        this.lowPower = false;

        // Configuration
        this.config = {
            backgroundColor: 0x0a0a0f,
            cameraFov: 60,
            cameraNear: 0.1,
            cameraFar: 1000,
        };
    }

    /**
     * Initialize the 3D viewer
     */
    async init() {
        // Check for Three.js
        if (typeof THREE === 'undefined') {
            console.error('Three.js not loaded');
            return false;
        }

        this.container = document.getElementById(this.containerId);
        if (!this.container) {
            console.error(`Container #${this.containerId} not found`);
            return false;
        }

        this.lowPower = Boolean(window.NexusPerformance?.lowPower);

        // Create scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(this.config.backgroundColor);
        this.scene.fog = new THREE.Fog(this.config.backgroundColor, 50, 200);

        // Create camera
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(
            this.config.cameraFov,
            aspect,
            this.config.cameraNear,
            this.config.cameraFar
        );
        this.camera.position.set(35, 25, 35);
        this.camera.lookAt(0, 5, 0);

        // Create renderer
        this.renderer = new THREE.WebGLRenderer({
            antialias: !this.lowPower,
            alpha: true,
            powerPreference: this.lowPower ? 'low-power' : 'high-performance'
        });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(this.lowPower ? 1 : Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = !this.lowPower;
        this.container.appendChild(this.renderer.domElement);

        // Create controls (if OrbitControls available)
        if (typeof THREE.OrbitControls !== 'undefined') {
            this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
            this.controls.enableDamping = true;
            this.controls.dampingFactor = 0.05;
            this.controls.minDistance = 15;
            this.controls.maxDistance = 100;
            this.controls.maxPolarAngle = Math.PI / 2.1;
        }

        // Setup lighting
        this._setupLighting();

        // Add grid
        this._addGrid();

        // Handle resize
        window.addEventListener('resize', () => this._onResize());

        this.isInitialized = true;
        console.log('🎬 Quality 3D Viewer initialized');

        return true;
    }

    /**
     * Setup scene lighting
     */
    _setupLighting() {
        // Ambient
        const ambient = new THREE.AmbientLight(0x404050, this.lowPower ? 0.6 : 0.5);
        this.scene.add(ambient);

        // Main directional light
        const directional = new THREE.DirectionalLight(0xffffff, this.lowPower ? 0.6 : 1.0);
        directional.position.set(50, 50, 25);
        directional.castShadow = !this.lowPower;
        if (!this.lowPower) {
            directional.shadow.mapSize.width = 2048;
            directional.shadow.mapSize.height = 2048;
        }
        this.scene.add(directional);

        if (!this.lowPower) {
            // Accent lights
            const blueLight = new THREE.PointLight(0x6366f1, 0.5, 50);
            blueLight.position.set(-20, 10, -20);
            this.scene.add(blueLight);

            const purpleLight = new THREE.PointLight(0xa855f7, 0.5, 50);
            purpleLight.position.set(20, 10, 20);
            this.scene.add(purpleLight);
        }

        // Hemisphere
        const hemi = new THREE.HemisphereLight(0x606080, 0x404040, this.lowPower ? 0.2 : 0.3);
        this.scene.add(hemi);
    }

    /**
     * Add reference grid
     */
    _addGrid() {
        const grid = new THREE.GridHelper(50, 50, 0x1a1a25, 0x1a1a25);
        grid.position.y = -0.01;
        this.scene.add(grid);
    }

    /**
     * Handle resize
     */
    _onResize() {
        if (!this.container || !this.camera || !this.renderer) return;

        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    /**
     * Load data from API and render visualization
     * @param {string} insightsFile - Name of the insights CSV file
     */
    async loadData(insightsFile) {
        if (!this.isInitialized) {
            await this.init();
        }

        try {
            // Fetch quality analysis
            const analysisResponse = await fetch('/quality-insights/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ insights_file: insightsFile }),
            });

            if (!analysisResponse.ok) {
                throw new Error(`Analysis failed: ${analysisResponse.status}`);
            }

            const analysisData = await analysisResponse.json();
            this.currentData = analysisData;

            // Render terrain
            this._renderTerrain(analysisData.terrain_data || []);

            // Render column pillars
            this._renderPillars(analysisData.column_scores || {});

            // Render particles
            this._renderParticles(analysisData.risk_distribution || {});

            // Fetch and render savings
            try {
                const savingsResponse = await fetch('/quality-insights/business-savings', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    credentials: 'include',
                    body: JSON.stringify({ insights_file: insightsFile, use_llm: false }),
                });

                if (savingsResponse.ok) {
                    const savingsData = await savingsResponse.json();
                    this._renderSavings(savingsData);
                }
            } catch (e) {
                console.warn('Could not load savings data:', e);
            }

            // Start animation
            this.start();

            console.log('📊 3D visualization loaded');
            return analysisData;

        } catch (error) {
            console.error('Failed to load data:', error);
            throw error;
        }
    }

    /**
     * Render terrain from data
     */
    _renderTerrain(terrainData) {
        // Clear existing terrain
        const existing = this.scene.getObjectByName('terrain');
        if (existing) this.scene.remove(existing);

        if (!terrainData.length) return;

        // Create height-mapped plane
        const size = 40;
        const segments = Math.min(50, Math.ceil(Math.sqrt(terrainData.length)));
        const geometry = new THREE.PlaneGeometry(size, size, segments - 1, segments - 1);

        const positions = geometry.attributes.position;
        const colors = [];

        // Map data to grid
        for (let i = 0; i < positions.count; i++) {
            const dataIndex = Math.floor(i / positions.count * terrainData.length);
            const point = terrainData[dataIndex] || { quality_score: 5 };
            const height = (point.quality_score / 10) * 8;

            positions.setZ(i, height);

            // Color based on quality
            const color = this._getQualityColor(point.quality_score);
            colors.push(color.r, color.g, color.b);
        }

        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        geometry.computeVertexNormals();

        const material = new THREE.MeshStandardMaterial({
            vertexColors: true,
            roughness: 0.7,
            metalness: 0.1,
            side: THREE.DoubleSide,
        });

        const mesh = new THREE.Mesh(geometry, material);
        mesh.rotation.x = -Math.PI / 2;
        mesh.receiveShadow = true;
        mesh.name = 'terrain';

        this.scene.add(mesh);
    }

    /**
     * Render column pillars
     */
    _renderPillars(columnScores) {
        // Clear existing pillars
        this.scene.children
            .filter(c => c.name?.startsWith('pillar-'))
            .forEach(c => this.scene.remove(c));

        const columns = Object.entries(columnScores);
        if (!columns.length) return;

        const startX = -20;
        const spacing = 3;
        const maxHeight = 10;

        columns.forEach(([name, score], index) => {
            const height = (score / 10) * maxHeight;
            const color = this._getQualityColor(score);

            const geometry = new THREE.CylinderGeometry(0.4, 0.5, height, 16);
            const material = new THREE.MeshStandardMaterial({
                color: color,
                emissive: color,
                emissiveIntensity: score < 5 ? 0.3 : 0.1,
                roughness: 0.3,
                metalness: 0.5,
            });

            const pillar = new THREE.Mesh(geometry, material);
            pillar.position.set(startX + index * spacing, height / 2, 18);
            pillar.castShadow = true;
            pillar.name = `pillar-${name}`;

            this.scene.add(pillar);
        });
    }

    /**
     * Render particle system
     */
    _renderParticles(riskDistribution) {
        // Clear existing particles
        const existing = this.scene.getObjectByName('particles');
        if (existing) this.scene.remove(existing);

        if (this.lowPower) return;

        const { high = 0, medium = 0, low = 0 } = riskDistribution;
        const total = high + medium + low;
        if (!total) return;

        const particleCount = 500;
        const positions = new Float32Array(particleCount * 3);
        const colors = new Float32Array(particleCount * 3);

        for (let i = 0; i < particleCount; i++) {
            // Position
            positions[i * 3] = (Math.random() - 0.5) * 40;
            positions[i * 3 + 1] = 2 + Math.random() * 10;
            positions[i * 3 + 2] = (Math.random() - 0.5) * 40;

            // Color based on distribution
            const rand = Math.random() * total;
            let color;
            if (rand < high) {
                color = new THREE.Color(0x22c55e);  // Green
            } else if (rand < high + medium) {
                color = new THREE.Color(0xf59e0b);  // Yellow
            } else {
                color = new THREE.Color(0xef4444);  // Red
            }

            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
            size: 0.3,
            vertexColors: true,
            transparent: true,
            opacity: 0.8,
            blending: THREE.AdditiveBlending,
        });

        const particles = new THREE.Points(geometry, material);
        particles.name = 'particles';
        this.scene.add(particles);

        // Store for animation
        this.particlePositions = positions;
    }

    /**
     * Render savings stars
     */
    _renderSavings(savingsData) {
        // Clear existing
        const existing = this.scene.getObjectByName('savings-group');
        if (existing) this.scene.remove(existing);

        if (this.lowPower) {
            const group = new THREE.Group();
            group.name = 'savings-group';

            const centralGeometry = new THREE.SphereGeometry(1.2, 12, 12);
            const centralMaterial = new THREE.MeshBasicMaterial({
                color: 0x22c55e,
                transparent: true,
                opacity: 0.75,
            });
            const central = new THREE.Mesh(centralGeometry, centralMaterial);
            central.position.set(0, 18, 0);
            group.add(central);
            this.scene.add(group);
            return;
        }

        const group = new THREE.Group();
        group.name = 'savings-group';

        // Central savings sphere
        const totalSavings = savingsData.estimated_annual_savings || '$0';
        const centralGeometry = new THREE.SphereGeometry(1.5, 32, 32);
        const centralMaterial = new THREE.MeshBasicMaterial({
            color: 0x22c55e,
            transparent: true,
            opacity: 0.8,
        });
        const central = new THREE.Mesh(centralGeometry, centralMaterial);
        central.position.set(0, 18, 0);
        group.add(central);

        // Recommendation stars
        const recommendations = savingsData.recommendations || [];
        recommendations.forEach((rec, i) => {
            const angle = (i / recommendations.length) * Math.PI * 2;
            const radius = 8;

            const starGeometry = new THREE.SphereGeometry(0.5, 16, 16);
            const starMaterial = new THREE.MeshBasicMaterial({
                color: 0xfbbf24,
                transparent: true,
                opacity: 0.9,
            });
            const star = new THREE.Mesh(starGeometry, starMaterial);
            star.position.set(
                Math.cos(angle) * radius,
                16 + Math.random() * 4,
                Math.sin(angle) * radius
            );
            group.add(star);
        });

        this.scene.add(group);
    }

    /**
     * Get color for quality score
     */
    _getQualityColor(score) {
        if (score >= 8) return new THREE.Color(0x22c55e);
        if (score >= 6) return new THREE.Color(0xf59e0b);
        if (score >= 4) return new THREE.Color(0xf97316);
        return new THREE.Color(0xef4444);
    }

    /**
     * Start animation loop
     */
    start() {
        if (this.isAnimating) return;
        if (this.lowPower) {
            if (this.controls) {
                this.controls.update();
            }
            if (this.renderer && this.scene && this.camera) {
                this.renderer.render(this.scene, this.camera);
            }
            return;
        }
        this.isAnimating = true;
        this._animate();
    }

    /**
     * Stop animation
     */
    stop() {
        this.isAnimating = false;
    }

    /**
     * Animation loop
     */
    _animate() {
        if (!this.isAnimating) return;

        requestAnimationFrame(() => this._animate());

        // Update controls
        if (this.controls) {
            this.controls.update();
        }

        // Animate particles
        if (this.particlePositions) {
            this._animateParticles();
        }

        // Render
        this.renderer.render(this.scene, this.camera);
    }

    /**
     * Animate particles
     */
    _animateParticles() {
        const particles = this.scene.getObjectByName('particles');
        if (!particles) return;

        const positions = particles.geometry.attributes.position.array;
        const time = Date.now() * 0.001;

        for (let i = 0; i < positions.length; i += 3) {
            positions[i] += Math.sin(time + i) * 0.01;
            positions[i + 1] += Math.cos(time + i * 0.5) * 0.005;
            positions[i + 2] += Math.sin(time * 0.5 + i) * 0.01;

            // Wrap around
            if (positions[i] > 20) positions[i] = -20;
            if (positions[i] < -20) positions[i] = 20;
            if (positions[i + 2] > 20) positions[i + 2] = -20;
            if (positions[i + 2] < -20) positions[i + 2] = 20;
        }

        particles.geometry.attributes.position.needsUpdate = true;
    }

    /**
     * Dispose of resources
     */
    dispose() {
        this.stop();

        if (this.renderer) {
            this.renderer.dispose();
            if (this.container && this.renderer.domElement) {
                this.container.removeChild(this.renderer.domElement);
            }
        }

        window.removeEventListener('resize', () => this._onResize());
        console.log('🗑️ Quality 3D Viewer disposed');
    }
}

// Export for global use
window.Quality3DViewer = Quality3DViewer;
