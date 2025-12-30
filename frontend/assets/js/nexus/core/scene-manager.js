/**
 * Scene Manager - Three.js Core Module
 * 
 * Handles Three.js scene setup, camera, renderer, and animation loop.
 * Flagship 3D visualization system for Quality Intelligence Dashboard.
 * 
 * @author Enterprise Analytics Team
 * @security CSP-compliant, no external scripts
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

/**
 * Scene Manager Class
 * 
 * Manages the Three.js scene lifecycle including:
 * - Scene, camera, and renderer setup
 * - Lighting system
 * - Animation loop
 * - Resize handling
 * - Object management
 */
export class SceneManager {
    constructor(containerId) {
        this.containerId = containerId;
        this.container = null;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.animationId = null;
        this.objects = new Map();
        this.isRunning = false;

        // Configuration
        this.config = {
            backgroundColor: 0x0a0a0f,
            fogColor: 0x0a0a0f,
            fogNear: 50,
            fogFar: 200,
            cameraFov: 60,
            cameraNear: 0.1,
            cameraFar: 1000,
        };
    }

    /**
     * Initialize the Three.js scene
     */
    init() {
        this.container = document.getElementById(this.containerId);
        if (!this.container) {
            console.error(`Container #${this.containerId} not found`);
            return false;
        }

        // Create scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(this.config.backgroundColor);
        this.scene.fog = new THREE.Fog(
            this.config.fogColor,
            this.config.fogNear,
            this.config.fogFar
        );

        // Create camera
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(
            this.config.cameraFov,
            aspect,
            this.config.cameraNear,
            this.config.cameraFar
        );
        this.camera.position.set(30, 25, 30);
        this.camera.lookAt(0, 0, 0);

        // Create renderer
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true,
        });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.container.appendChild(this.renderer.domElement);

        // Create controls
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.minDistance = 10;
        this.controls.maxDistance = 150;
        this.controls.maxPolarAngle = Math.PI / 2.1;

        // Setup lighting
        this._setupLighting();

        // Add grid helper
        this._addGrid();

        // Handle resize
        window.addEventListener('resize', () => this._onResize());

        console.log('🎬 Scene Manager initialized');
        return true;
    }

    /**
     * Setup scene lighting
     */
    _setupLighting() {
        // Ambient light for base illumination
        const ambient = new THREE.AmbientLight(0x404050, 0.5);
        this.scene.add(ambient);

        // Main directional light (sun)
        const directional = new THREE.DirectionalLight(0xffffff, 1.0);
        directional.position.set(50, 50, 25);
        directional.castShadow = true;
        directional.shadow.mapSize.width = 2048;
        directional.shadow.mapSize.height = 2048;
        directional.shadow.camera.near = 0.5;
        directional.shadow.camera.far = 200;
        directional.shadow.camera.left = -50;
        directional.shadow.camera.right = 50;
        directional.shadow.camera.top = 50;
        directional.shadow.camera.bottom = -50;
        this.scene.add(directional);

        // Accent lights for visual interest
        const blueLight = new THREE.PointLight(0x6366f1, 0.5, 50);
        blueLight.position.set(-20, 10, -20);
        this.scene.add(blueLight);

        const purpleLight = new THREE.PointLight(0xa855f7, 0.5, 50);
        purpleLight.position.set(20, 10, 20);
        this.scene.add(purpleLight);

        // Hemisphere light for sky/ground gradient
        const hemi = new THREE.HemisphereLight(0x606080, 0x404040, 0.3);
        this.scene.add(hemi);
    }

    /**
     * Add reference grid
     */
    _addGrid() {
        const gridHelper = new THREE.GridHelper(50, 50, 0x1a1a25, 0x1a1a25);
        gridHelper.position.y = -0.01;
        this.scene.add(gridHelper);
        this.objects.set('grid', gridHelper);
    }

    /**
     * Handle window resize
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
     * Start the animation loop
     */
    start() {
        if (this.isRunning) return;
        this.isRunning = true;
        this._animate();
        console.log('▶️ Animation started');
    }

    /**
     * Stop the animation loop
     */
    stop() {
        if (!this.isRunning) return;
        this.isRunning = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        console.log('⏹️ Animation stopped');
    }

    /**
     * Animation loop
     */
    _animate() {
        if (!this.isRunning) return;

        this.animationId = requestAnimationFrame(() => this._animate());

        // Update controls
        if (this.controls) {
            this.controls.update();
        }

        // Update any animated objects
        this._updateAnimatedObjects();

        // Render
        if (this.renderer && this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera);
        }
    }

    /**
     * Update animated objects (override in subclass or add callbacks)
     */
    _updateAnimatedObjects() {
        // Placeholder for custom animations
        // Children can override or add to this.animationCallbacks
    }

    /**
     * Add an object to the scene
     * @param {string} key - Unique identifier
     * @param {THREE.Object3D} object - Three.js object
     */
    addObject(key, object) {
        if (this.objects.has(key)) {
            this.removeObject(key);
        }
        this.scene.add(object);
        this.objects.set(key, object);
    }

    /**
     * Remove an object from the scene
     * @param {string} key - Object identifier
     */
    removeObject(key) {
        const object = this.objects.get(key);
        if (object) {
            this.scene.remove(object);
            if (object.geometry) object.geometry.dispose();
            if (object.material) {
                if (Array.isArray(object.material)) {
                    object.material.forEach(m => m.dispose());
                } else {
                    object.material.dispose();
                }
            }
            this.objects.delete(key);
        }
    }

    /**
     * Get an object by key
     * @param {string} key - Object identifier
     */
    getObject(key) {
        return this.objects.get(key);
    }

    /**
     * Clear all objects
     */
    clearObjects() {
        const keys = Array.from(this.objects.keys());
        keys.forEach(key => {
            if (key !== 'grid') {
                this.removeObject(key);
            }
        });
    }

    /**
     * Move camera to a position
     * @param {THREE.Vector3} position - Target position
     * @param {THREE.Vector3} lookAt - Point to look at
     * @param {number} duration - Animation duration in ms
     */
    moveCameraTo(position, lookAt, duration = 1000) {
        const startPos = this.camera.position.clone();
        const startTime = Date.now();

        const animate = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Ease out cubic
            const eased = 1 - Math.pow(1 - progress, 3);

            this.camera.position.lerpVectors(startPos, position, eased);
            this.controls.target.lerp(lookAt, eased * 0.1);

            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };

        animate();
    }

    /**
     * Set background color
     * @param {number} color - Hex color
     */
    setBackgroundColor(color) {
        this.scene.background = new THREE.Color(color);
        this.config.backgroundColor = color;
    }

    /**
     * Dispose of all resources
     */
    dispose() {
        this.stop();

        // Clear all objects
        const keys = Array.from(this.objects.keys());
        keys.forEach(key => this.removeObject(key));

        // Dispose renderer
        if (this.renderer) {
            this.renderer.dispose();
            if (this.container && this.renderer.domElement) {
                this.container.removeChild(this.renderer.domElement);
            }
        }

        // Remove resize listener
        window.removeEventListener('resize', () => this._onResize());

        console.log('🗑️ Scene Manager disposed');
    }
}

export default SceneManager;
