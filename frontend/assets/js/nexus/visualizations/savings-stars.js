/**
 * Savings Stars - ROI Recommendation Constellation
 * 
 * Visualizes business savings recommendations as a star field.
 * Larger stars = higher ROI, connected by lines for related actions.
 * 
 * @author Enterprise Analytics Team
 */

import * as THREE from 'three';

/**
 * SavingsStars Class
 * 
 * Creates a constellation visualization of ROI recommendations.
 */
export class SavingsStars {
    constructor(sceneManager) {
        this.sceneManager = sceneManager;
        this.stars = new Map();
        this.connections = [];
        this.group = null;

        // Configuration
        this.config = {
            baseRadius: 0.3,
            maxRadius: 1.2,
            starY: 15,  // Height above terrain
            spreadRadius: 15,
            connectionOpacity: 0.3,
        };
    }

    /**
     * Generate star constellation from savings data
     * @param {Object} savingsData - Business savings analysis result
     */
    generate(savingsData) {
        this.clear();

        const recommendations = savingsData.recommendations || [];
        const highImpact = savingsData.high_impact_issues || [];

        if (recommendations.length === 0 && highImpact.length === 0) {
            console.warn('No savings data to visualize');
            return;
        }

        this.group = new THREE.Group();
        this.group.name = 'savings-constellation';

        // Create stars for recommendations
        recommendations.forEach((rec, index) => {
            const star = this._createStar(rec, index, 'recommendation');
            this.stars.set(`rec-${index}`, star);
            this.group.add(star);
        });

        // Create stars for high-impact issues
        highImpact.forEach((issue, index) => {
            const star = this._createStar(issue, index + recommendations.length, 'issue');
            this.stars.set(`issue-${index}`, star);
            this.group.add(star);
        });

        // Create connections between related elements
        this._createConnections();

        // Add central savings display
        const total = savingsData.estimated_annual_savings || '$0';
        const centralStar = this._createCentralStar(total);
        this.group.add(centralStar);

        if (this.sceneManager) {
            this.sceneManager.addObject('savings-stars', this.group);
        }

        // Start twinkling animation
        this._startTwinkling();

        console.log(`⭐ Created ${this.stars.size} savings stars`);
    }

    /**
     * Create a single star
     */
    _createStar(data, index, type) {
        const { baseRadius, maxRadius, starY, spreadRadius } = this.config;

        // Calculate star size based on ROI or impact
        const roiValue = this._extractROI(data);
        const size = baseRadius + (roiValue / 5) * (maxRadius - baseRadius);

        // Create star geometry (sphere with glow)
        const geometry = new THREE.SphereGeometry(size, 16, 16);

        // Color: gold for recommendations, orange for issues
        const color = type === 'recommendation'
            ? new THREE.Color(0xfbbf24)  // Gold
            : new THREE.Color(0xf97316); // Orange

        const material = new THREE.MeshBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.9,
        });

        const star = new THREE.Mesh(geometry, material);

        // Position in circular arrangement
        const angle = (index / 8) * Math.PI * 2;
        const radius = spreadRadius * (0.5 + Math.random() * 0.5);

        star.position.set(
            Math.cos(angle) * radius,
            starY + Math.random() * 3,
            Math.sin(angle) * radius
        );

        // Add glow effect
        const glowGeometry = new THREE.SphereGeometry(size * 1.5, 16, 16);
        const glowMaterial = new THREE.MeshBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.3,
        });
        const glow = new THREE.Mesh(glowGeometry, glowMaterial);
        star.add(glow);

        // Store metadata
        star.userData = {
            type: type,
            data: data,
            baseOpacity: 0.9,
            phase: Math.random() * Math.PI * 2,
        };

        return star;
    }

    /**
     * Create central star showing total savings
     */
    _createCentralStar(totalSavings) {
        const { starY } = this.config;

        // Large central sphere
        const geometry = new THREE.SphereGeometry(2, 32, 32);
        const material = new THREE.MeshBasicMaterial({
            color: 0x22c55e,  // Green for savings
            transparent: true,
            opacity: 0.8,
        });

        const star = new THREE.Mesh(geometry, material);
        star.position.set(0, starY + 2, 0);

        // Outer glow rings
        for (let i = 0; i < 3; i++) {
            const ringGeometry = new THREE.TorusGeometry(3 + i, 0.1, 8, 32);
            const ringMaterial = new THREE.MeshBasicMaterial({
                color: 0x22c55e,
                transparent: true,
                opacity: 0.3 - i * 0.1,
            });
            const ring = new THREE.Mesh(ringGeometry, ringMaterial);
            ring.rotation.x = Math.PI / 2;
            ring.position.y = -i * 0.5;
            star.add(ring);
        }

        star.userData = {
            type: 'total',
            savings: totalSavings,
            rotationSpeed: 0.002,
        };

        return star;
    }

    /**
     * Create connections between stars
     */
    _createConnections() {
        const starPositions = Array.from(this.stars.values()).map(s => s.position);

        // Connect nearby stars
        for (let i = 0; i < starPositions.length; i++) {
            for (let j = i + 1; j < starPositions.length; j++) {
                const dist = starPositions[i].distanceTo(starPositions[j]);

                if (dist < 15) {  // Only connect close stars
                    const points = [starPositions[i], starPositions[j]];
                    const geometry = new THREE.BufferGeometry().setFromPoints(points);
                    const material = new THREE.LineBasicMaterial({
                        color: 0x6366f1,
                        transparent: true,
                        opacity: this.config.connectionOpacity,
                    });

                    const line = new THREE.Line(geometry, material);
                    this.connections.push(line);
                    this.group.add(line);
                }
            }
        }
    }

    /**
     * Extract ROI value from recommendation/issue
     */
    _extractROI(data) {
        // Try to parse ROI from string like "3.2x in 6 months"
        const roiStr = data.roi || data.impact || '';
        const match = roiStr.match(/(\d+\.?\d*)/);
        return match ? parseFloat(match[1]) : 2.0;
    }

    /**
     * Start twinkling animation
     */
    _startTwinkling() {
        const animate = () => {
            if (!this.group) return;

            this.stars.forEach(star => {
                const phase = star.userData.phase;
                const base = star.userData.baseOpacity;
                const twinkle = 0.1 * Math.sin(Date.now() * 0.003 + phase);

                star.material.opacity = base + twinkle;
            });

            // Rotate central star rings
            const centralStar = this.group.children.find(c => c.userData?.type === 'total');
            if (centralStar) {
                centralStar.rotation.y += centralStar.userData.rotationSpeed;
            }

            requestAnimationFrame(animate);
        };

        animate();
    }

    /**
     * Highlight a specific recommendation
     */
    highlight(key) {
        const star = this.stars.get(key);
        if (!star) return;

        // Pulse animation
        const originalScale = star.scale.clone();
        let pulseTime = 0;

        const pulse = () => {
            pulseTime += 0.1;
            const scale = 1 + 0.3 * Math.sin(pulseTime);
            star.scale.set(scale, scale, scale);

            if (pulseTime < Math.PI * 4) {
                requestAnimationFrame(pulse);
            } else {
                star.scale.copy(originalScale);
            }
        };

        pulse();
    }

    /**
     * Update with new savings data
     */
    update(savingsData) {
        this.generate(savingsData);
    }

    /**
     * Clear constellation
     */
    clear() {
        if (this.sceneManager) {
            this.sceneManager.removeObject('savings-stars');
        }
        this.stars.clear();
        this.connections = [];
        this.group = null;
    }

    /**
     * Get star data
     */
    getStarData(key) {
        const star = this.stars.get(key);
        return star ? star.userData : null;
    }
}

export default SavingsStars;
