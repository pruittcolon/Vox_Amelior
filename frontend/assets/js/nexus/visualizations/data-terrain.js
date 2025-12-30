/**
 * Data Terrain - 3D Height-Mapped Quality Mesh
 * 
 * Generates a 3D terrain where height corresponds to quality scores.
 * Mountains = High Quality, Valleys = Low Quality
 * 
 * @author Enterprise Analytics Team
 * @security CSP-compliant
 */

import * as THREE from 'three';

/**
 * Quality color gradient stops
 */
const QUALITY_COLORS = {
    low: new THREE.Color(0xef4444),    // Red (1-5)
    medium: new THREE.Color(0xf59e0b), // Yellow (6-7)
    high: new THREE.Color(0x22c55e),   // Green (8-10)
};

/**
 * DataTerrain Class
 * 
 * Creates a 3D height-mapped mesh from quality scores.
 */
export class DataTerrain {
    constructor(sceneManager) {
        this.sceneManager = sceneManager;
        this.mesh = null;
        this.wireframe = null;
        this.terrainData = [];

        // Configuration
        this.config = {
            gridSize: 50,        // Number of vertices per side
            terrainSize: 40,     // World units
            heightScale: 8,      // Max height multiplier
            smoothing: 2,        // Smoothing passes
        };
    }

    /**
     * Generate terrain from quality data
     * @param {Array} data - Array of {row_index, quality_score, color_category, q1-q5}
     */
    generate(data) {
        if (!data || data.length === 0) {
            console.warn('No terrain data provided');
            return null;
        }

        this.terrainData = data;

        // Clean up existing terrain
        this.clear();

        // Create geometry
        const geometry = this._createGeometry(data);

        // Create materials
        const material = this._createMaterial();
        const wireframeMaterial = this._createWireframeMaterial();

        // Create mesh
        this.mesh = new THREE.Mesh(geometry, material);
        this.mesh.receiveShadow = true;
        this.mesh.castShadow = true;
        this.mesh.rotation.x = -Math.PI / 2;  // Rotate to horizontal
        this.mesh.name = 'terrain';

        // Create wireframe overlay for low-quality regions
        this.wireframe = new THREE.Mesh(geometry.clone(), wireframeMaterial);
        this.wireframe.rotation.x = -Math.PI / 2;
        this.wireframe.position.y = 0.01;  // Slight offset
        this.wireframe.name = 'terrain-wireframe';

        // Add to scene
        if (this.sceneManager) {
            this.sceneManager.addObject('terrain', this.mesh);
            this.sceneManager.addObject('terrain-wireframe', this.wireframe);
        }

        console.log(`🏔️ Terrain generated from ${data.length} data points`);
        return this.mesh;
    }

    /**
     * Create terrain geometry from data
     */
    _createGeometry(data) {
        const { gridSize, terrainSize, heightScale } = this.config;

        // Create plane geometry
        const geometry = new THREE.PlaneGeometry(
            terrainSize,
            terrainSize,
            gridSize - 1,
            gridSize - 1
        );

        // Create height map from data
        const heightMap = this._createHeightMap(data, gridSize);

        // Apply heights to vertices
        const positions = geometry.attributes.position;
        const colors = [];

        for (let i = 0; i < positions.count; i++) {
            const x = Math.floor(i % gridSize);
            const y = Math.floor(i / gridSize);

            // Get height from height map
            const height = heightMap[y][x];
            const normalizedHeight = height / 10;  // 0-1 range

            // Apply height
            positions.setZ(i, normalizedHeight * heightScale);

            // Calculate color based on quality
            const color = this._getColorForQuality(height);
            colors.push(color.r, color.g, color.b);
        }

        // Apply colors
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

        // Update normals for lighting
        geometry.computeVertexNormals();

        return geometry;
    }

    /**
     * Create height map grid from sparse data points
     */
    _createHeightMap(data, gridSize) {
        // Initialize grid with base value
        const heights = Array(gridSize).fill(null).map(() =>
            Array(gridSize).fill(5.0)  // Default to medium quality
        );

        // Map data points to grid
        const dataPerCell = Math.ceil(data.length / (gridSize * gridSize));

        data.forEach((point, index) => {
            // Map data index to grid position
            const gridIndex = Math.floor(index / dataPerCell);
            const x = gridIndex % gridSize;
            const y = Math.floor(gridIndex / gridSize) % gridSize;

            if (x < gridSize && y < gridSize) {
                heights[y][x] = point.quality_score || 5.0;
            }
        });

        // Apply smoothing
        for (let pass = 0; pass < this.config.smoothing; pass++) {
            this._smoothHeightMap(heights, gridSize);
        }

        return heights;
    }

    /**
     * Smooth height map for natural terrain appearance
     */
    _smoothHeightMap(heights, gridSize) {
        const smoothed = Array(gridSize).fill(null).map(() =>
            Array(gridSize).fill(0)
        );

        for (let y = 0; y < gridSize; y++) {
            for (let x = 0; x < gridSize; x++) {
                let sum = 0;
                let count = 0;

                // Sample 3x3 neighborhood
                for (let dy = -1; dy <= 1; dy++) {
                    for (let dx = -1; dx <= 1; dx++) {
                        const nx = x + dx;
                        const ny = y + dy;

                        if (nx >= 0 && nx < gridSize && ny >= 0 && ny < gridSize) {
                            sum += heights[ny][nx];
                            count++;
                        }
                    }
                }

                smoothed[y][x] = sum / count;
            }
        }

        // Copy back
        for (let y = 0; y < gridSize; y++) {
            for (let x = 0; x < gridSize; x++) {
                heights[y][x] = smoothed[y][x];
            }
        }
    }

    /**
     * Get color for quality value
     */
    _getColorForQuality(quality) {
        if (quality >= 8) {
            // High quality - green
            const t = (quality - 8) / 2;  // 0-1 within range
            return QUALITY_COLORS.high.clone();
        } else if (quality >= 6) {
            // Medium quality - yellow to green blend
            const t = (quality - 6) / 2;
            return new THREE.Color().lerpColors(
                QUALITY_COLORS.medium,
                QUALITY_COLORS.high,
                t
            );
        } else if (quality >= 4) {
            // Low-medium - red to yellow blend
            const t = (quality - 4) / 2;
            return new THREE.Color().lerpColors(
                QUALITY_COLORS.low,
                QUALITY_COLORS.medium,
                t
            );
        } else {
            // Very low - red
            return QUALITY_COLORS.low.clone();
        }
    }

    /**
     * Create main terrain material
     */
    _createMaterial() {
        return new THREE.MeshStandardMaterial({
            vertexColors: true,
            roughness: 0.7,
            metalness: 0.1,
            flatShading: false,
            side: THREE.DoubleSide,
        });
    }

    /**
     * Create wireframe overlay material
     */
    _createWireframeMaterial() {
        return new THREE.MeshBasicMaterial({
            color: 0x6366f1,
            wireframe: true,
            transparent: true,
            opacity: 0.15,
        });
    }

    /**
     * Update terrain with new data
     */
    update(data) {
        this.generate(data);
    }

    /**
     * Highlight a specific region
     * @param {number} rowIndex - Row to highlight
     */
    highlightRow(rowIndex) {
        // Find the data point
        const point = this.terrainData.find(p => p.row_index === rowIndex);
        if (!point) return;

        // Create highlight marker
        const markerGeometry = new THREE.CylinderGeometry(0.3, 0.3, 2, 8);
        const markerMaterial = new THREE.MeshBasicMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: 0.8,
        });

        const marker = new THREE.Mesh(markerGeometry, markerMaterial);

        // Calculate position based on row index
        const gridSize = this.config.gridSize;
        const terrainSize = this.config.terrainSize;
        const dataPerCell = Math.ceil(this.terrainData.length / (gridSize * gridSize));

        const gridIndex = Math.floor(rowIndex / dataPerCell);
        const x = (gridIndex % gridSize) / gridSize * terrainSize - terrainSize / 2;
        const z = Math.floor(gridIndex / gridSize) / gridSize * terrainSize - terrainSize / 2;
        const y = (point.quality_score / 10) * this.config.heightScale + 1;

        marker.position.set(x, y, z);

        if (this.sceneManager) {
            this.sceneManager.addObject('row-highlight', marker);
        }
    }

    /**
     * Clear highlight
     */
    clearHighlight() {
        if (this.sceneManager) {
            this.sceneManager.removeObject('row-highlight');
        }
    }

    /**
     * Clear terrain
     */
    clear() {
        if (this.sceneManager) {
            this.sceneManager.removeObject('terrain');
            this.sceneManager.removeObject('terrain-wireframe');
        }
        this.mesh = null;
        this.wireframe = null;
    }

    /**
     * Get statistics about the terrain
     */
    getStats() {
        if (!this.terrainData.length) return null;

        const scores = this.terrainData.map(p => p.quality_score);
        const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
        const min = Math.min(...scores);
        const max = Math.max(...scores);

        return {
            dataPoints: this.terrainData.length,
            averageQuality: avg.toFixed(1),
            minQuality: min.toFixed(1),
            maxQuality: max.toFixed(1),
            highQualityCount: scores.filter(s => s >= 8).length,
            lowQualityCount: scores.filter(s => s < 6).length,
        };
    }
}

export default DataTerrain;
