/**
 * Particle Flow - Data Point Animation System
 * 
 * Creates flowing particles that visualize data movement between columns.
 * Particle density and color indicate data quality.
 * 
 * @author Enterprise Analytics Team
 */

import * as THREE from 'three';

/**
 * ParticleFlow Class
 * 
 * GPU-accelerated particle system for data visualization.
 */
export class ParticleFlow {
    constructor(sceneManager) {
        this.sceneManager = sceneManager;
        this.particleSystem = null;
        this.particles = [];
        this.isAnimating = false;

        // Configuration
        this.config = {
            particleCount: 1000,
            particleSize: 0.15,
            flowSpeed: 0.02,
            spreadX: 40,
            spreadZ: 40,
            heightMin: 0.5,
            heightMax: 12,
        };
    }

    /**
     * Generate particle system from quality distribution
     * @param {Object} riskDistribution - {high, medium, low} counts
     */
    generate(riskDistribution) {
        this.clear();

        const { high = 0, medium = 0, low = 0 } = riskDistribution;
        const total = high + medium + low;

        if (total === 0) {
            console.warn('No risk distribution data');
            return;
        }

        // Calculate particle counts per category
        const { particleCount } = this.config;
        const highCount = Math.floor((high / total) * particleCount);
        const mediumCount = Math.floor((medium / total) * particleCount);
        const lowCount = particleCount - highCount - mediumCount;

        // Create particles
        this.particles = [];

        // High quality particles (green, high)
        for (let i = 0; i < highCount; i++) {
            this.particles.push(this._createParticle('high', i / highCount));
        }

        // Medium quality particles (yellow, mid)
        for (let i = 0; i < mediumCount; i++) {
            this.particles.push(this._createParticle('medium', i / mediumCount));
        }

        // Low quality particles (red, low)
        for (let i = 0; i < lowCount; i++) {
            this.particles.push(this._createParticle('low', i / lowCount));
        }

        // Create buffer geometry
        const geometry = this._createGeometry();
        const material = this._createMaterial();

        this.particleSystem = new THREE.Points(geometry, material);
        this.particleSystem.name = 'particle-flow';

        if (this.sceneManager) {
            this.sceneManager.addObject('particles', this.particleSystem);
        }

        // Start animation
        this.startAnimation();

        console.log(`✨ Created ${this.particles.length} particles`);
    }

    /**
     * Create a single particle with initial properties
     */
    _createParticle(quality, progress) {
        const { spreadX, spreadZ, heightMin, heightMax } = this.config;

        // Color based on quality
        let color;
        let baseHeight;

        switch (quality) {
            case 'high':
                color = new THREE.Color(0x22c55e);
                baseHeight = heightMax * 0.7;
                break;
            case 'medium':
                color = new THREE.Color(0xf59e0b);
                baseHeight = heightMax * 0.4;
                break;
            case 'low':
                color = new THREE.Color(0xef4444);
                baseHeight = heightMax * 0.2;
                break;
        }

        return {
            position: new THREE.Vector3(
                (Math.random() - 0.5) * spreadX,
                baseHeight + Math.random() * 3,
                (Math.random() - 0.5) * spreadZ
            ),
            velocity: new THREE.Vector3(
                (Math.random() - 0.5) * 0.1,
                (Math.random() - 0.5) * 0.05,
                (Math.random() - 0.5) * 0.1
            ),
            color: color,
            quality: quality,
            age: Math.random() * 100,
            lifetime: 100 + Math.random() * 100,
        };
    }

    /**
     * Create buffer geometry from particles
     */
    _createGeometry() {
        const geometry = new THREE.BufferGeometry();

        const positions = new Float32Array(this.particles.length * 3);
        const colors = new Float32Array(this.particles.length * 3);
        const sizes = new Float32Array(this.particles.length);

        this.particles.forEach((p, i) => {
            positions[i * 3] = p.position.x;
            positions[i * 3 + 1] = p.position.y;
            positions[i * 3 + 2] = p.position.z;

            colors[i * 3] = p.color.r;
            colors[i * 3 + 1] = p.color.g;
            colors[i * 3 + 2] = p.color.b;

            sizes[i] = this.config.particleSize;
        });

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

        return geometry;
    }

    /**
     * Create particle material
     */
    _createMaterial() {
        // Create circular particle texture
        const canvas = document.createElement('canvas');
        canvas.width = 32;
        canvas.height = 32;
        const ctx = canvas.getContext('2d');

        const gradient = ctx.createRadialGradient(16, 16, 0, 16, 16, 16);
        gradient.addColorStop(0, 'rgba(255,255,255,1)');
        gradient.addColorStop(0.3, 'rgba(255,255,255,0.8)');
        gradient.addColorStop(1, 'rgba(255,255,255,0)');

        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, 32, 32);

        const texture = new THREE.CanvasTexture(canvas);

        return new THREE.PointsMaterial({
            size: this.config.particleSize * 3,
            vertexColors: true,
            transparent: true,
            opacity: 0.8,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
            map: texture,
        });
    }

    /**
     * Start particle animation
     */
    startAnimation() {
        if (this.isAnimating) return;
        this.isAnimating = true;
        this._animate();
    }

    /**
     * Stop particle animation
     */
    stopAnimation() {
        this.isAnimating = false;
    }

    /**
     * Animation loop
     */
    _animate() {
        if (!this.isAnimating || !this.particleSystem) return;

        const positions = this.particleSystem.geometry.attributes.position.array;
        const { flowSpeed, spreadX, spreadZ, heightMin, heightMax } = this.config;

        this.particles.forEach((p, i) => {
            // Update age
            p.age += 1;

            // Apply velocity
            p.position.add(p.velocity);

            // Add some turbulence
            p.position.x += Math.sin(p.age * 0.05) * 0.02;
            p.position.y += Math.cos(p.age * 0.03) * 0.01;

            // Wrap around boundaries
            if (p.position.x > spreadX / 2) p.position.x = -spreadX / 2;
            if (p.position.x < -spreadX / 2) p.position.x = spreadX / 2;
            if (p.position.z > spreadZ / 2) p.position.z = -spreadZ / 2;
            if (p.position.z < -spreadZ / 2) p.position.z = spreadZ / 2;

            // Keep within height bounds
            if (p.position.y < heightMin) p.position.y = heightMin;
            if (p.position.y > heightMax) p.position.y = heightMax;

            // Update buffer
            positions[i * 3] = p.position.x;
            positions[i * 3 + 1] = p.position.y;
            positions[i * 3 + 2] = p.position.z;
        });

        this.particleSystem.geometry.attributes.position.needsUpdate = true;

        requestAnimationFrame(() => this._animate());
    }

    /**
     * Set particle flow direction
     */
    setFlowDirection(direction) {
        const dir = direction.normalize().multiplyScalar(this.config.flowSpeed);
        this.particles.forEach(p => {
            p.velocity.copy(dir);
            p.velocity.add(new THREE.Vector3(
                (Math.random() - 0.5) * 0.05,
                0,
                (Math.random() - 0.5) * 0.05
            ));
        });
    }

    /**
     * Update particles based on new distribution
     */
    update(riskDistribution) {
        this.generate(riskDistribution);
    }

    /**
     * Clear particle system
     */
    clear() {
        this.stopAnimation();
        if (this.sceneManager) {
            this.sceneManager.removeObject('particles');
        }
        this.particleSystem = null;
        this.particles = [];
    }
}

export default ParticleFlow;
