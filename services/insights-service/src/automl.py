import logging
import pandas as pd
import numpy as np
import json
from typing import List, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

from shared.crypto.db_encryption import create_encrypted_db

logger = logging.getLogger(__name__)

class AutoMLService:
    def __init__(self, db_path: str, db_key: str):
        self.db_path = db_path
        self.db_key = db_key
        self.df = None
        self.label_encoders = {}

    def load_data(self):
        """Load data from encrypted SQLite into Pandas DataFrame"""
        try:
            db = create_encrypted_db(self.db_path, self.db_key)
            conn = db.connect()
            
            # Fetch core analysis fields
            query = """
                SELECT 
                    speaker, 
                    emotion, 
                    emotion_confidence, 
                    pace_wpm, 
                    pitch_mean, 
                    pitch_std, 
                    volume_rms, 
                    word_count, 
                    filler_count, 
                    created_at 
                FROM transcript_segments
            """
            self.df = pd.read_sql_query(query, conn)
            db.close()
            
            # Drop empty rows or rows with critical missing data
            self.df.dropna(subset=['speaker', 'emotion'], inplace=True)
            
            # Preprocessing: Handle missing numeric values
            numeric_cols = ['pace_wpm', 'pitch_mean', 'pitch_std', 'volume_rms', 'word_count', 'filler_count']
            for col in numeric_cols:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)

            return {"status": "success", "rows": len(self.df), "columns": list(self.df.columns)}
        except Exception as e:
            logger.error(f"Data load failed: {e}")
            return {"status": "error", "message": str(e)}

    def generate_hypotheses(self) -> List[Dict[str, Any]]:
        """Analyze the dataframe and suggest ML experiments"""
        if self.df is None or self.df.empty:
            return []

        hypotheses = []

        # Hypothesis 1: Predict Emotion (Classification)
        # Features: Audio metrics + Speaker
        hypotheses.append({
            "id": "predict_emotion",
            "type": "classification",
            "target": "emotion",
            "features": ["pace_wpm", "pitch_mean", "pitch_std", "volume_rms", "word_count"],
            "description": "Can we predict the speaker's emotion based on their voice pitch, speed, and volume?",
            "model_type": "Random Forest"
        })

        # Hypothesis 2: Predict Speaker Identity (Classification)
        # Features: Audio metrics
        hypotheses.append({
            "id": "predict_speaker",
            "type": "classification",
            "target": "speaker",
            "features": ["pitch_mean", "pitch_std", "pace_wpm"],
            "description": "Can we identify the speaker purely by their voice characteristics?",
            "model_type": "Random Forest"
        })

        # Hypothesis 3: Predict Pace (Regression)
        # Features: Emotion
        hypotheses.append({
            "id": "predict_pace",
            "type": "regression",
            "target": "pace_wpm",
            "features": ["emotion", "word_count"],
            "description": "Does the emotion affect how fast the speaker talks?",
            "model_type": "Linear Regression"
        })

        return hypotheses

    def run_experiment(self, experiment_id: str):
        """Run a specific AutoML experiment"""
        if self.df is None:
            self.load_data()
            
        if self.df.empty:
            return {"status": "error", "message": "No data available"}

        # Define experiment config
        config = next((h for h in self.generate_hypotheses() if h["id"] == experiment_id), None)
        if not config:
            return {"status": "error", "message": "Unknown experiment ID"}

        target = config["target"]
        features = config["features"]
        model_type = config["model_type"]
        task_type = config["type"]

        # Prepare Data
        X = self.df[features].copy()
        y = self.df[target].copy()

        # Encode categorical features/target
        encoders = {}
        
        # Encode Target if classification
        if task_type == "classification":
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
            encoders[target] = le
            classes = list(le.classes_)
        
        # Encode categorical Features
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                encoders[col] = le

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Select Model
        if task_type == "classification":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = LinearRegression()

        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)

        # Evaluate
        result = {
            "experiment_id": experiment_id,
            "target": target,
            "model": model_type,
            "rows_used": len(self.df),
            "status": "success"
        }

        if task_type == "classification":
            acc = accuracy_score(y_test, y_pred)
            result["accuracy"] = round(acc * 100, 2)
            result["metric"] = "Accuracy"
            
            # Feature Importance
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                result["feature_importance"] = [
                    {"feature": f, "importance": round(i, 3)} 
                    for f, i in zip(features, importances)
                ]
                # Sort by importance
                result["feature_importance"].sort(key=lambda x: x["importance"], reverse=True)
                
        else:
            score = r2_score(y_test, y_pred)
            result["r2_score"] = round(score, 3)
            result["metric"] = "R² Score"

        return result
