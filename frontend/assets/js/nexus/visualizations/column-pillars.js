/**
 * Column Pillars - 3D Column Quality Visualization
 * 
 * Creates cylinders for each data column where height = quality score.
 * Includes glow effects for problematic columns.
 * 
 * @author Enterprise Analytics Team
 */

import * as THREE from 'three';

/**
 * ColumnPillars Class
 * 
 * Visualizes column-level quality as 3D pillars.
 */
export class ColumnPillars {
    constructor(sceneManager) {
        this.sceneManager = sceneManager;
        this.pillars = new Map();
        this.labels = new Map();

        // Configuration
        this.config = {
            radius: 0.5,
            maxHeight: 10,
            spacing: 3,
            startX: -20,
            baseY: 0,
            startZ: 18,
        };
    }

    /**
     * Generate pillars from column scores
     * @param {Object} columnScores - {column_name: score}
     */
    generate(columnScores) {
        if (!columnScores || Object.keys(columnScores).length === 0) {
            console.warn('No column scores provided');
            return;
        }

        // Clean up existing pillars
        this.clear();

        const columns = Object.entries(columnScores);

        columns.forEach(([name, score], index) => {
            const pillar = this._createPillar(name, score, index);
            this.pillars.set(name, pillar);

            if (this.sceneManager) {
                this.sceneManager.addObject(`pillar-${name}`, pillar);
            }
        });

        console.log(`📊 Generated ${columns.length} column pillars`);
    }

    /**
     * Create a single pillar
     */
    _createPillar(name, score, index) {
        const { radius, maxHeight, spacing, startX, baseY, startZ } = this.config;

        // Calculate height based on score (1-10)
        const height = (score / 10) * maxHeight;

        // Create geometry
        const geometry = new THREE.CylinderGeometry(
            radius * 0.8,  // Top radius
            radius,        // Bottom radius
            height,
            16,            // Segments
            1
        );

        // Get color based on score
        const color = this._getColorForScore(score);

        // Create material with emissive for glow effect
        const material = new THREE.MeshStandardMaterial({
            color: color,
            emissive: color,
            emissiveIntensity: score < 5 ? 0.3 : 0.1,  // Glow more for low scores
            roughness: 0.3,
            metalness: 0.5,
        });

        const pillar = new THREE.Mesh(geometry, material);

        // Position
        const x = startX + (index * spacing);
        const y = baseY + height / 2;
        const z = startZ;

        pillar.position.set(x, y, z);
        pillar.castShadow = true;
        pillar.receiveShadow = true;
        pillar.name = `pillar-${name}`;

        // Store metadata
        pillar.userData = {
            columnName: name,
            score: score,
            type: 'column-pillar',
        };

        // Add base ring for visual grounding
        const baseRing = this._createBaseRing(x, z, color);
        pillar.add(baseRing);

        // Add top cap with glow for problematic columns
        if (score < 6) {
            const glow = this._createGlowEffect(radius, height, color);
            pillar.add(glow);
        }

        return pillar;
    }

    /**
     * Create base ring for pillar
     */
    _createBaseRing(x, z, color) {
        const { radius } = this.config;

        const ringGeometry = new THREE.RingGeometry(radius, radius * 1.5, 16);
        const ringMaterial = new THREE.MeshBasicMaterial({
            color: color,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.5,
        });

        const ring = new THREE.Mesh(ringGeometry, ringMaterial);
        ring.rotation.x = -Math.PI / 2;
        ring.position.y = -this.config.maxHeight / 2;

        return ring;
    }

    /**
     * Create glow effect for low-quality columns
     */
    _createGlowEffect(radius, height, color) {
        const glowGeometry = new THREE.SphereGeometry(radius * 1.5, 16, 16);
        const glowMaterial = new THREE.MeshBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.3,
        });

        const glow = new THREE.Mesh(glowGeometry, glowMaterial);
        glow.position.y = height / 2 + radius;

        return glow;
    }

    /**
     * Get color for quality score
     */
    _getColorForScore(score) {
        if (score >= 8) {
            return new THREE.Color(0x22c55e);  // Green
        } else if (score >= 6) {
            return new THREE.Color(0xf59e0b);  // Yellow
        } else if (score >= 4) {
            return new THREE.Color(0xf97316);  // Orange
        } else {
            return new THREE.Color(0xef4444);  // Red
        }
    }

    /**
     * Highlight a specific column
     */
    highlight(columnName) {
        const pillar = this.pillars.get(columnName);
        if (!pillar) return;

        // Create highlight ring around pillar
        const { radius, maxHeight, spacing, startX, startZ } = this.config;

        const ringGeometry = new THREE.TorusGeometry(radius * 2, 0.1, 8, 32);
        const ringMaterial = new THREE.MeshBasicMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: 0.8,
        });

        const ring = new THREE.Mesh(ringGeometry, ringMaterial);
        ring.rotation.x = Math.PI / 2;
        ring.position.copy(pillar.position);
        ring.position.y = maxHeight + 1;

        if (this.sceneManager) {
            this.sceneManager.addObject('pillar-highlight', ring);
        }

        // Animate the ring
        let scale = 1;
        const animate = () => {
            scale = 1 + 0.1 * Math.sin(Date.now() * 0.005);
            ring.scale.set(scale, scale, scale);
            if (this.sceneManager?.getObject('pillar-highlight')) {
                requestAnimationFrame(animate);
            }
        };
        animate();
    }

    /**
     * Clear highlight
     */
    clearHighlight() {
        if (this.sceneManager) {
            this.sceneManager.removeObject('pillar-highlight');
        }
    }

    /**
     * Update pillar heights with animation
     */
    update(columnScores) {
        Object.entries(columnScores).forEach(([name, newScore]) => {
            const pillar = this.pillars.get(name);
            if (!pillar) return;

            const targetHeight = (newScore / 10) * this.config.maxHeight;
            const currentHeight = pillar.geometry.parameters.height;

            // Animate height change
            this._animateHeight(pillar, currentHeight, targetHeight);
        });
    }

    /**
     * Animate pillar height change
     */
    _animateHeight(pillar, fromHeight, toHeight) {
        const duration = 500;
        const startTime = Date.now();

        const animate = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Ease out
            const eased = 1 - Math.pow(1 - progress, 3);
            const height = fromHeight + (toHeight - fromHeight) * eased;

            pillar.scale.y = height / fromHeight;
            pillar.position.y = this.config.baseY + height / 2;

            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };

        animate();
    }

    /**
     * Clear all pillars
     */
    clear() {
        this.pillars.forEach((pillar, name) => {
            if (this.sceneManager) {
                this.sceneManager.removeObject(`pillar-${name}`);
            }
        });
        this.pillars.clear();
        this.clearHighlight();
    }

    /**
     * Get pillar data for a column
     */
    getPillarData(columnName) {
        const pillar = this.pillars.get(columnName);
        return pillar ? pillar.userData : null;
    }
}

export default ColumnPillars;
