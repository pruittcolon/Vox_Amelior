import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class SegmentationEngine:
    """
    Performs automated user segmentation using K-Means clustering.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def segment(self, k: int = 3, features: List[str] = None) -> Dict[str, Any]:
        """
        Run K-Means clustering.
        
        Args:
            k: Number of clusters.
            features: List of column names to use for clustering. 
                      If None, uses all numeric columns.
                      
        Returns:
            Dict containing cluster profiles, counts, and sample assignments.
        """
        try:
            # 1. Select Features
            if features:
                # Validate features exist
                missing = [f for f in features if f not in self.df.columns]
                if missing:
                    return {"error": f"Features not found: {missing}"}
                X = self.df[features].copy()
            else:
                # Auto-select numeric features
                X = self.df.select_dtypes(include=[np.number]).copy()
                # Remove ID-like columns (simple heuristic: if unique count == row count)
                # Also remove likely target columns if known (but we don't know them here easily without schema)
                # For MVP, just taking all numerics is okay, but let's try to be slightly smarter
                # Drop columns with "id" in name
                drop_cols = [c for c in X.columns if "id" in c.lower()]
                if drop_cols:
                    X = X.drop(columns=drop_cols)
                    
            if X.empty:
                return {"error": "No numeric features available for segmentation."}
                
            feature_names = X.columns.tolist()
            
            # 2. Preprocessing
            # Impute
            imputer = SimpleImputer(strategy='mean')
            X_imputed = imputer.fit_transform(X)
            
            # Scale
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imputed)
            
            # 3. Clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            # 4. Analysis
            # Add cluster labels to original (or copy)
            df_labeled = self.df.copy()
            df_labeled['cluster'] = clusters
            
            # Calculate Profiles (Mean of features per cluster)
            # We use the original scale values for interpretability
            # But we need to handle the imputed values if we want to be precise. 
            # For profiles, using the original df (with NaNs ignored by groupby mean) is usually fine/better.
            
            # Group by cluster and calculate mean for the used features
            profiles = df_labeled.groupby('cluster')[feature_names].mean().to_dict(orient='index')
            
            # Counts
            counts = df_labeled['cluster'].value_counts().to_dict()
            
            # Sample Assignments (first 10 rows)
            samples = df_labeled[['cluster']].head(10).to_dict(orient='records')
            
            # Convert numpy types to python native for JSON serialization
            profiles_serializable = {}
            for cluster_id, stats in profiles.items():
                profiles_serializable[int(cluster_id)] = {k: float(v) for k, v in stats.items()}
                
            counts_serializable = {int(k): int(v) for k, v in counts.items()}
            
            return {
                "k": k,
                "features": feature_names,
                "profiles": profiles_serializable,
                "counts": counts_serializable,
                "samples": samples
            }
            
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            return {"error": str(e)}
