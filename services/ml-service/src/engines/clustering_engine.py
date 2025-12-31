"""
Clustering Engine - Schema-Agnostic Version

Unsupervised clustering with multiple algorithms:
- K-Means: Centroid-based clustering
- DBSCAN: Density-based clustering
- Hierarchical/Agglomerative: Tree-based clustering
- Gaussian Mixture Models: Probabilistic clustering

Includes auto-k selection (elbow method, silhouette) and cluster profiling.
NOW SCHEMA-AGNOSTIC: Automatically detects numeric features via semantic intelligence.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from typing import Any

import numpy as np
import pandas as pd
from core import ColumnMapper, ColumnProfiler, EngineRequirements, SemanticType
from core.premium_models import ConfigParameter
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ClusteringEngine:
    """Schema-agnostic multi-algorithm clustering engine"""

    def __init__(self):
        self.name = "Clustering Engine (Schema-Agnostic)"
        self.profiler = ColumnProfiler()
        self.mapper = ColumnMapper()
        self.models = {}
        self.scalers = {}

    def get_engine_info(self) -> dict[str, str]:
        """Get engine display information."""
        return {"name": "clustering", "display_name": "Clustering Analysis", "icon": "🔮", "task_type": "detection"}

    def get_config_schema(self) -> list[ConfigParameter]:
        """Get configuration schema for UI."""
        return [
            ConfigParameter(
                name="algorithm",
                type="select",
                default="auto",
                range=["auto", "kmeans", "dbscan", "hierarchical"],
                description="Clustering algorithm to use",
            ),
            ConfigParameter(
                name="n_clusters",
                type="int",
                default=3,
                range=[2, 20],
                description="Number of clusters (for k-means/hierarchical)",
            ),
        ]

    def get_methodology_info(self) -> dict[str, Any]:
        """Get methodology documentation."""
        return {
            "name": "Multi-Algorithm Clustering",
            "url": "https://scikit-learn.org/stable/modules/clustering.html",
            "steps": [
                {
                    "step_number": 1,
                    "title": "Feature Selection",
                    "description": "Identify numeric features via schema intelligence",
                },
                {
                    "step_number": 2,
                    "title": "Scaling",
                    "description": "Standardize features for distance-based algorithms",
                },
                {
                    "step_number": 3,
                    "title": "Optimal K Selection",
                    "description": "Use elbow method and silhouette scores",
                },
                {
                    "step_number": 4,
                    "title": "Clustering",
                    "description": "Apply selected algorithm (K-Means, DBSCAN, Hierarchical)",
                },
                {"step_number": 5, "title": "Profiling", "description": "Generate cluster profiles and metrics"},
            ],
            "limitations": ["Assumes meaningful numeric features exist", "K-means assumes spherical clusters"],
            "assumptions": ["Features are appropriately scaled", "Outliers may affect results"],
        }

    def get_requirements(self) -> EngineRequirements:
        """Define semantic requirements for clustering."""
        return EngineRequirements(
            required_semantics=[SemanticType.NUMERIC_CONTINUOUS],
            optional_semantics={},
            required_entities=[],
            min_rows=10,  # Need enough data for meaningful clusters
        )

    def analyze(self, df: pd.DataFrame, config: dict[str, Any] = None, **kwargs) -> dict[str, Any]:
        """
        Standard analyze() interface that wraps cluster().
        Required for compatibility with testing framework.
        
        Accepts **kwargs for compatibility with engine runner (target_column, etc.)
        """
        try:
            return self.cluster(df, config)
        except ValueError as e:
            # Return graceful error instead of crashing
            return {
                "status": "error",
                "error": str(e),
                "engine": "clustering",
                "recommendation": "Upload a dataset with numeric columns for clustering analysis."
            }

    def cluster(self, df: pd.DataFrame, config: dict[str, Any] = None) -> dict[str, Any]:
        """
        Run schema-agnostic clustering analysis.

        Args:
            df: Input dataframe
            config:
                - features: Hint for columns to use for clustering
                - algorithm: 'kmeans', 'dbscan', 'hierarchical', 'auto'
                - n_clusters: Number of clusters (for k-means/hierarchical)
                - auto_k_range: Range for auto k selection [min, max]
                - skip_profiling: Skip schema intelligence (default: False)

        Returns:
            Cluster assignments, profiling, and column mappings
        """
        # Handle None config
        config = config or {}

        # SCHEMA INTELLIGENCE
        if not config.get("skip_profiling", False):
            profiles = self.profiler.profile_dataset(df)

            # Auto-detect numeric features
            numeric_cols = [
                col
                for col, prof in profiles.items()
                if prof.semantic_type in [SemanticType.NUMERIC_CONTINUOUS, SemanticType.NUMERIC_DISCRETE]
            ]

            features = config.get("features") or numeric_cols
        else:
            features = config.get("features")
            if not features:
                # Use all numeric columns
                features = df.select_dtypes(include=[np.number]).columns.tolist()

        if not features:
            raise ValueError("No numeric features found for clustering")

        algorithm = config.get("algorithm", "auto")
        n_clusters = config.get("n_clusters", 3)
        auto_k_range = config.get("auto_k_range", [2, 10])

        # Prepare data
        X = df[features].dropna()

        if len(X) < 10:
            raise ValueError("Need at least 10 data points for clustering")

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers["latest"] = scaler

        # Auto-select algorithm if requested
        if algorithm == "auto":
            # Default to k-means with auto-k
            algorithm = "kmeans"
            n_clusters = self._find_optimal_k(X_scaled, auto_k_range)

        # Run clustering
        if algorithm == "kmeans":
            result = self._kmeans_clustering(X_scaled, n_clusters)
        elif algorithm == "dbscan":
            result = self._dbscan_clustering(X_scaled)
        elif algorithm == "hierarchical":
            result = self._hierarchical_clustering(X_scaled, n_clusters)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        labels = result["labels"]

        # Profile clusters
        cluster_profiles = self._profile_clusters(X, labels, features)

        # Calculate metrics
        metrics = self._calculate_metrics(X_scaled, labels)

        # Generate PCA coordinates for 3D visualization (pass feature names for loadings)
        pca_data = self._generate_pca_coordinates(X_scaled, labels, features)

        return {
            "algorithm": algorithm,
            "n_clusters": result["n_clusters"],
            "labels": labels.tolist(),
            "cluster_profiles": cluster_profiles,
            "metrics": metrics,
            "visualizations": self._generate_cluster_visualizations(X, labels, features),
            "pca_3d": pca_data,
            "column_mappings": {"features": features},
            "profiling_used": not config.get("skip_profiling", False),
            "metadata": {"total_points": len(X), "features_used": features},
        }

    def _find_optimal_k(self, X: np.ndarray, k_range: list[int]) -> int:
        """Find optimal number of clusters using elbow method"""
        inertias = []
        silhouettes = []

        k_min, k_max = k_range
        k_max = min(k_max, len(X) - 1)  # Can't have more clusters than points

        for k in range(k_min, k_max + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)

            inertias.append(kmeans.inertia_)

            if k > 1:
                silhouettes.append(silhouette_score(X, labels))

        # Find elbow (simple heuristic: max second derivative)
        if len(inertias) > 2:
            diffs = np.diff(inertias)
            second_diffs = np.diff(diffs)
            elbow_idx = np.argmax(second_diffs) + 1
            optimal_k = k_min + elbow_idx
        else:
            # Fallback: use silhouette
            optimal_k = k_min + np.argmax(silhouettes) if silhouettes else k_min

        return optimal_k

    def _kmeans_clustering(self, X: np.ndarray, n_clusters: int) -> dict:
        """K-Means clustering"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        self.models["latest"] = kmeans

        return {
            "labels": labels,
            "n_clusters": n_clusters,
            "centroids": kmeans.cluster_centers_.tolist(),
            "inertia": float(kmeans.inertia_),
        }

    def _dbscan_clustering(self, X: np.ndarray) -> dict:
        """DBSCAN clustering"""
        # Auto-tune eps using heuristic
        eps = 0.5
        min_samples = max(2, int(len(X) * 0.01))

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        return {
            "labels": labels,
            "n_clusters": n_clusters,
            "noise_points": int((labels == -1).sum()),
            "eps": eps,
            "min_samples": min_samples,
        }

    def _hierarchical_clustering(self, X: np.ndarray, n_clusters: int) -> dict:
        """Hierarchical/Agglomerative clustering"""
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels = model.fit_predict(X)

        return {"labels": labels, "n_clusters": n_clusters, "linkage": "ward"}

    def _profile_clusters(self, X: pd.DataFrame, labels: np.ndarray, features: list[str]) -> list[dict]:
        """Profile each cluster"""
        X_with_labels = X.copy()
        X_with_labels["cluster"] = labels

        profiles = []

        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:  # Noise in DBSCAN
                continue

            cluster_data = X_with_labels[X_with_labels["cluster"] == cluster_id]

            profile = {
                "cluster_id": int(cluster_id),
                "size": len(cluster_data),
                "percentage": float(len(cluster_data) / len(X) * 100),
                "feature_stats": {},
            }

            for feature in features:
                profile["feature_stats"][feature] = {
                    "mean": float(cluster_data[feature].mean()),
                    "std": float(cluster_data[feature].std()),
                    "min": float(cluster_data[feature].min()),
                    "max": float(cluster_data[feature].max()),
                }

            profiles.append(profile)

        return profiles

    def _calculate_metrics(self, X: np.ndarray, labels: np.ndarray) -> dict:
        """Calculate clustering quality metrics"""
        metrics = {}

        # Filter out noise points (label -1 from DBSCAN)
        mask = labels != -1
        X_clean = X[mask]
        labels_clean = labels[mask]

        n_clusters = len(set(labels_clean))

        if n_clusters > 1 and len(X_clean) > n_clusters:
            try:
                metrics["silhouette_score"] = float(silhouette_score(X_clean, labels_clean))
            except:
                metrics["silhouette_score"] = None

            try:
                metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(X_clean, labels_clean))
            except:
                metrics["calinski_harabasz_score"] = None

            try:
                metrics["davies_bouldin_score"] = float(davies_bouldin_score(X_clean, labels_clean))
            except:
                metrics["davies_bouldin_score"] = None

        return metrics

    def _generate_pca_coordinates(
        self, X_scaled: np.ndarray, labels: np.ndarray, feature_names: list[str] = None
    ) -> dict[str, Any]:
        """
        Generate PCA-transformed 3D coordinates for cluster visualization.

        Args:
            X_scaled: Scaled feature matrix
            labels: Cluster labels
            feature_names: Original feature names for component loadings

        Returns:
            Dictionary with PCA coordinates, component loadings, and metadata
        """
        try:
            # Determine number of components (max 3 for 3D viz)
            n_components = min(3, X_scaled.shape[1])

            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)

            # Build points list with coordinates and cluster assignments
            points = []
            for i in range(len(X_pca)):
                point = {
                    "x": float(X_pca[i, 0]) if n_components >= 1 else 0.0,
                    "y": float(X_pca[i, 1]) if n_components >= 2 else 0.0,
                    "z": float(X_pca[i, 2]) if n_components >= 3 else 0.0,
                    "cluster": int(labels[i]),
                }
                points.append(point)

            # Build component loadings (which features contribute to each PC)
            component_loadings = []
            if feature_names:
                for pc_idx in range(n_components):
                    loadings = pca.components_[pc_idx]
                    # Sort by absolute contribution
                    feature_contributions = [
                        {
                            "feature": feature_names[i],
                            "loading": float(loadings[i]),
                            "abs_loading": abs(float(loadings[i])),
                        }
                        for i in range(len(feature_names))
                    ]
                    feature_contributions.sort(key=lambda x: x["abs_loading"], reverse=True)
                    component_loadings.append(
                        {
                            "component": f"PC{pc_idx + 1}",
                            "variance_explained": float(pca.explained_variance_ratio_[pc_idx]),
                            "top_features": feature_contributions[:5],  # Top 5 contributors
                        }
                    )

            return {
                "points": points,
                "explained_variance": pca.explained_variance_ratio_.tolist(),
                "total_variance_explained": float(sum(pca.explained_variance_ratio_)),
                "n_components": n_components,
                "component_loadings": component_loadings,
            }
        except Exception as e:
            logger.warning(f"PCA visualization failed: {e}")
            return {
                "points": [],
                "explained_variance": [],
                "total_variance_explained": 0.0,
                "n_components": 0,
                "component_loadings": [],
                "error": str(e),
            }

    def _generate_cluster_visualizations(self, X: pd.DataFrame, labels: np.ndarray, features: list[str]) -> list[dict]:
        """Generate visualization metadata"""
        visualizations = []

        # Scatter plot (first 2 features)
        if len(features) >= 2:
            visualizations.append(
                {
                    "type": "scatter_plot",
                    "title": f"Clusters: {features[0]} vs {features[1]}",
                    "data": {
                        "x": X[features[0]].tolist(),
                        "y": X[features[1]].tolist(),
                        "labels": labels.tolist(),
                        "x_label": features[0],
                        "y_label": features[1],
                    },
                    "description": "Clusters colored by assignment",
                }
            )

        # Cluster size bar chart
        cluster_sizes = pd.Series(labels).value_counts().to_dict()
        visualizations.append(
            {
                "type": "bar_chart",
                "title": "Cluster Sizes",
                "data": {
                    "labels": [f"Cluster {k}" for k in cluster_sizes.keys()],
                    "values": list(cluster_sizes.values()),
                },
                "description": "Number of points in each cluster",
            }
        )

        return visualizations
