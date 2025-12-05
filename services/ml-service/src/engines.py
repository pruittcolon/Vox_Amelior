import pandas as pd
import numpy as np
import re
import json
from typing import Dict, List, Any, Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle

# ==============================================================================
# PHASE 2: VECTOR ENGINE ("The Memory")
# ==============================================================================

class VectorEngine:
    _model = None
    
    @classmethod
    def get_model(cls):
        if cls._model is None:
            # Use a small, fast model optimized for CPU
            print("🧠 Loading Embedding Model (all-MiniLM-L6-v2)...")
            cls._model = SentenceTransformer('all-MiniLM-L6-v2')
        return cls._model

    def __init__(self, index_path: str = None):
        self.index = None
        self.texts = [] # Keep track of texts corresponding to vectors
        if index_path and os.path.exists(index_path):
            self.load(index_path)

    def embed_text(self, texts: List[str]) -> np.ndarray:
        model = self.get_model()
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings

    def build_index(self, texts: List[str], embeddings: np.ndarray):
        self.texts = texts
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        return self.index

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self.index:
            return []
        
        model = self.get_model()
        q_vec = model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(q_vec, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.texts):
                results.append({
                    "text": self.texts[idx],
                    "score": float(1 / (1 + distances[0][i])), # Convert L2 distance to similarity score
                    "index": int(idx)
                })
        return results

    def save(self, path: str):
        # Save FAISS index
        faiss.write_index(self.index, path)
        # Save texts metadata
        with open(path + ".meta", "wb") as f:
            pickle.dump(self.texts, f)

    def load(self, path: str):
        self.index = faiss.read_index(path)
        if os.path.exists(path + ".meta"):
            with open(path + ".meta", "rb") as f:
                self.texts = pickle.load(f)

# ==============================================================================
# PHASE 5: QUERY ENGINE ("The Voice")
# ==============================================================================

class QueryEngine:
    def __init__(self, df: pd.DataFrame, schema: Dict[str, str], vector_engine: VectorEngine = None):
        self.df = df
        self.schema = schema
        self.vector_engine = vector_engine
        self.cols_by_type = {}
        for col, t in schema.items():
            self.cols_by_type.setdefault(t, []).append(col)

    def _detect_intent(self, question: str) -> str:
        """
        Simple heuristic to decide between RETRIEVAL (text search) and ANALYTICS (agg).
        """
        q_lower = question.lower()
        analytics_keywords = ['average', 'mean', 'sum', 'total', 'count', 'max', 'min', 'highest', 'lowest', 'how many', 'trend']
        
        if any(k in q_lower for k in analytics_keywords):
            return "ANALYTICS"
        return "RETRIEVAL"

    def _execute_analytics(self, question: str) -> str:
        """
        Executes simple pandas aggregations based on regex templates.
        Safe execution without eval().
        """
        q_lower = question.lower()
        
        # 1. Average/Mean
        if 'average' in q_lower or 'mean' in q_lower:
            for col in self.cols_by_type.get("METRIC", []) + self.cols_by_type.get("MONEY_IN", []):
                if col.lower() in q_lower:
                    val = self.df[col].mean()
                    return f"The average {col} is {val:.2f}."
        
        # 2. Sum/Total
        if 'sum' in q_lower or 'total' in q_lower:
            for col in self.cols_by_type.get("METRIC", []) + self.cols_by_type.get("MONEY_IN", []):
                if col.lower() in q_lower:
                    val = self.df[col].sum()
                    return f"The total {col} is {val:.2f}."
                    
        # 3. Max/Highest
        if 'max' in q_lower or 'highest' in q_lower:
            for col in self.cols_by_type.get("METRIC", []) + self.cols_by_type.get("MONEY_IN", []):
                if col.lower() in q_lower:
                    val = self.df[col].max()
                    # Try to find the row identifier
                    id_cols = self.cols_by_type.get("ID", [])
                    if id_cols:
                        row = self.df.loc[self.df[col].idxmax()]
                        return f"The highest {col} is {val:.2f} (ID: {row[id_cols[0]]})."
                    return f"The highest {col} is {val:.2f}."

        # 4. Count/How many
        if 'count' in q_lower or 'how many' in q_lower:
             return f"The dataset has {len(self.df)} rows."

        return "I couldn't identify the specific calculation you asked for. I support average, sum, max, and count."

    def answer_question(self, question: str) -> Dict[str, Any]:
        intent = self._detect_intent(question)
        context = ""
        
        if intent == "RETRIEVAL" and self.vector_engine:
            # Search for relevant text
            results = self.vector_engine.search(question, k=3)
            if results:
                context = "Found these relevant records:\n" + "\n".join([f"- {r['text']}" for r in results])
            else:
                context = "No relevant text records found via semantic search."
                
        elif intent == "ANALYTICS":
            context = self._execute_analytics(question)
            
        else:
            context = "I wasn't sure how to analyze this, so I'm looking at the general dataset summary."

        return {
            "question": question,
            "intent": intent,
            "context": context
        }

# ==============================================================================
# PHASE 4: TIME SERIES ("The Future")
# ==============================================================================

class TimeSeriesEngine:
    def __init__(self, df: pd.DataFrame, schema: Dict[str, str]):
        self.df = df
        self.schema = schema
        self.cols_by_type = {}
        for col, t in schema.items():
            self.cols_by_type.setdefault(t, []).append(col)

    def detect_time_col(self) -> Optional[str]:
        """Finds the best candidate for a time column."""
        time_cols = self.cols_by_type.get("TIME", [])
        if time_cols:
            return time_cols[0]
        # Fallback: Check for datetime dtype
        for col in self.df.columns:
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                return col
        return None

    def analyze_forecast(self, horizon: int = 30) -> Dict[str, Any]:
        """
        Generates a forecast using a 'Prophet-lite' approach (RandomForest + Date Features).
        """
        time_col = self.detect_time_col()
        metric_cols = self.cols_by_type.get("METRIC", []) + self.cols_by_type.get("MONEY_IN", [])
        
        if not time_col or not metric_cols:
            return {"error": "Need Time and Metric columns for forecasting."}
            
        target_col = metric_cols[0]
        
        # Prepare Data
        df = self.df.copy()
        try:
            df[time_col] = pd.to_datetime(df[time_col])
        except:
            return {"error": f"Could not parse date column '{time_col}'."}
            
        # Aggregate to daily (or reasonable frequency) to handle duplicates
        df = df.groupby(time_col)[target_col].sum().reset_index()
        df = df.sort_values(time_col)
        
        # Feature Engineering
        df['day_of_week'] = df[time_col].dt.dayofweek
        df['month'] = df[time_col].dt.month
        df['day_of_year'] = df[time_col].dt.dayofyear
        df['year'] = df[time_col].dt.year
        df['timestamp'] = df[time_col].astype(int) // 10**9 # Unix timestamp
        
        # Lags (Autoregression)
        for lag in [1, 7, 30]:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
            
        # Drop NaNs created by lags
        df_train = df.dropna()
        
        if len(df_train) < 50:
             return {"error": "Not enough historical data for forecasting (need > 50 points)."}

        features = ['day_of_week', 'month', 'day_of_year', 'year', 'timestamp', 'lag_1', 'lag_7', 'lag_30']
        X = df_train[features]
        y = df_train[target_col]
        
        # Train Model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Forecast Loop (Recursive)
        last_date = df[time_col].max()
        future_dates = [last_date + pd.Timedelta(days=x) for x in range(1, horizon + 1)]
        forecast_values = []
        
        # We need to maintain a rolling window of history to compute lags for future points
        # This is a simplified recursive strategy
        current_history = df.copy()
        
        for date in future_dates:
            # Build feature row
            feat_row = {
                'day_of_week': date.dayofweek,
                'month': date.month,
                'day_of_year': date.dayofyear,
                'year': date.year,
                'timestamp': int(date.timestamp())
            }
            
            # Compute lags from current_history
            # We need to find the value from 1, 7, 30 days ago in current_history
            # If exact date missing, use ffill or nearest (simplified here: nearest valid index)
            def get_lag_val(days_ago):
                target_date = date - pd.Timedelta(days=days_ago)
                # Find closest date in history
                # This is slow but functional for 'lite' version
                closest_idx = (current_history[time_col] - target_date).abs().idxmin()
                return current_history.loc[closest_idx, target_col]

            feat_row['lag_1'] = get_lag_val(1)
            feat_row['lag_7'] = get_lag_val(7)
            feat_row['lag_30'] = get_lag_val(30)
            
            X_future = pd.DataFrame([feat_row])
            pred = model.predict(X_future)[0]
            forecast_values.append(pred)
            
            # Append prediction to history so next step can use it as lag
            new_row = pd.DataFrame({time_col: [date], target_col: [pred]})
            current_history = pd.concat([current_history, new_row], ignore_index=True)

        # Insights
        total_forecast = sum(forecast_values)
        avg_forecast = total_forecast / len(forecast_values)
        current_avg = df[target_col].mean()
        trend = "increasing" if avg_forecast > current_avg else "decreasing"
        
        insights = [
            f"🔮 **Forecast**: Predicted {target_col} for next {horizon} days.",
            f"📈 **Trend**: The metric is expected to be **{trend}** (Avg: {avg_forecast:.2f} vs Hist: {current_avg:.2f})."
        ]
        
        # Chart
        # Combine History + Forecast
        # Limit history to last 90 days for readability
        history_cut = df.tail(90)
        
        labels = history_cut[time_col].dt.strftime('%Y-%m-%d').tolist() + [d.strftime('%Y-%m-%d') for d in future_dates]
        data_hist = history_cut[target_col].tolist() + [None] * len(future_dates)
        data_pred = [None] * len(history_cut) + forecast_values
        
        chart = {
            "id": "forecast_line",
            "title": f"Forecast: {target_col}",
            "type": "line",
            "labels": labels,
            "datasets": [
                {"label": "Historical", "data": data_hist, "borderColor": "#3b82f6"},
                {"label": "Forecast", "data": data_pred, "borderColor": "#10b981", "borderDash": [5, 5]}
            ]
        }
        
        return {"summary": "Forecast Analysis", "insights": insights, "charts": [chart]}

    def analyze_seasonality(self) -> Dict[str, Any]:
        """
        Analyzes seasonality by aggregating by Day of Week and Month.
        """
        time_col = self.detect_time_col()
        metric_cols = self.cols_by_type.get("METRIC", []) + self.cols_by_type.get("MONEY_IN", [])
        
        if not time_col or not metric_cols:
            return {"error": "Need Time and Metric columns for seasonality."}
            
        target_col = metric_cols[0]
        df = self.df.copy()
        try:
            df[time_col] = pd.to_datetime(df[time_col])
        except:
            return {"error": f"Could not parse date column '{time_col}'."}
            
        # Day of Week Analysis
        df['DayOfWeek'] = df[time_col].dt.day_name()
        # Sort order
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['DayOfWeek'] = pd.Categorical(df['DayOfWeek'], categories=days_order, ordered=True)
        
        dow_agg = df.groupby('DayOfWeek')[target_col].mean()
        
        # Month Analysis
        df['Month'] = df[time_col].dt.month_name()
        months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        df['Month'] = pd.Categorical(df['Month'], categories=months_order, ordered=True)
        
        month_agg = df.groupby('Month')[target_col].mean()
        
        # Insights
        best_day = dow_agg.idxmax()
        best_month = month_agg.idxmax()
        
        insights = [
            f"📅 **Seasonality**: {target_col} peaks on **{best_day}s**.",
            f"🗓️ **Monthly Trend**: The strongest month is usually **{best_month}**."
        ]
        
        # Charts
        chart_dow = {
            "id": "seasonality_dow",
            "title": "Average by Day of Week",
            "type": "bar",
            "labels": days_order,
            "data": [dow_agg.get(d, 0) for d in days_order],
            "label": f"Avg {target_col}"
        }
        
        return {"summary": "Seasonality Analysis", "insights": insights, "charts": [chart_dow]}

# ==============================================================================
# PHASE 3: BUSINESS RECIPES ("The Skills")
# ==============================================================================

class RecipeEngine:
    def __init__(self, df: pd.DataFrame, schema: Dict[str, str]):
        self.df = df
        self.schema = schema
        self.cols_by_type = {}
        for col, t in schema.items():
            self.cols_by_type.setdefault(t, []).append(col)

    def analyze_churn(self) -> Dict[str, Any]:
        """
        Predicts churn probability using RandomForest.
        Requires a TARGET column (binary) and some numeric/categorical features.
        """
        target_cols = self.cols_by_type.get("TARGET", [])
        if not target_cols:
            # Fallback: look for 'churn' in column names
            target_cols = [c for c in self.df.columns if "churn" in c.lower()]
        
        if not target_cols:
            return {"error": "No target/churn column found."}
        
        target_col = target_cols[0]
        
        # Prepare Data
        df_clean = self.df.copy()
        
        # Drop ID columns and Time columns (simple baseline)
        drop_cols = self.cols_by_type.get("ID", []) + self.cols_by_type.get("TIME", []) + self.cols_by_type.get("PII", [])
        feature_cols = [c for c in df_clean.columns if c != target_col and c not in drop_cols]
        
        # Simple Preprocessing
        le = LabelEncoder()
        for col in feature_cols:
            if df_clean[col].dtype == 'object':
                df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        
        # Handle missing
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(df_clean[feature_cols])
        X = pd.DataFrame(X_imputed, columns=feature_cols)
        y = df_clean[target_col]
        
        # Encode target if needed
        if y.dtype == 'object':
            y = le.fit_transform(y.astype(str))
            
        # Train Model
        clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        clf.fit(X, y)
        
        # --- SHAP Integration (Phase 1.1) ---
        try:
            from .explainability import SHAPAnalyzer
            analyzer = SHAPAnalyzer(clf, X)
            
            # Generate utility-filtered insights
            shap_insights = analyzer.generate_insights(X, target_col)
            
            if shap_insights:
                # Format insights for display
                insights = [f"Target: Analyzing churn based on '{target_col}'."]
                for item in shap_insights[:3]: # Top 3 actionable insights
                    feat = item['feature']
                    score = item['importance']
                    # Prescriptive text based on feature name (simple heuristic for demo)
                    action = "monitor"
                    if "price" in feat.lower() or "charge" in feat.lower():
                        action = "optimize pricing for"
                    elif "call" in feat.lower() or "support" in feat.lower():
                        action = "improve support experience for"
                    elif "tenure" in feat.lower():
                        action = "focus on early-stage retention for"
                        
                    insights.append(f"💡 **{feat}**: {action} this factor (Impact: {score:.3f}).")
            else:
                # Fallback if SHAP fails or filters everything
                insights = ["No significant actionable drivers found."]
                
        except Exception as e:
            print(f"SHAP Error: {e}")
            # Fallback to basic importance
            importances = clf.feature_importances_
            indices = np.argsort(importances)[::-1]
            top_features = [(feature_cols[i], importances[i]) for i in indices[:2]]
            insights = [
                f"Target: Analyzing churn based on '{target_col}'.",
                f"Key Drivers: {top_features[0][0]} and {top_features[1][0]}."
            ]

        # Chart (Basic Feature Importance for now, SHAP charts next)
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        chart = {
            "id": "churn_drivers",
            "title": "Top Churn Drivers (SHAP-Enhanced)",
            "type": "bar",
            "labels": [feature_cols[i] for i in indices[:5]],
            "data": [float(importances[i]) for i in indices[:5]],
            "label": "Importance Score"
        }
        
        return {"summary": "Churn Analysis (SHAP)", "insights": insights, "charts": [chart]}

    def simulate_churn(self, perturbations: Dict[str, float]) -> Dict[str, Any]:
        """
        Run a What-If simulation on the churn model.
        Re-trains the model (stateless) and applies perturbations.
        """
        # 1. Re-run basic prep to get X, y, and model
        # NOTE: In a real prod system, we'd load a cached model. 
        # Here we reuse the logic from analyze_churn but return the objects.
        # For simplicity in Phase 1.2, we'll just duplicate the minimal training logic here
        # or refactor analyze_churn to be split. 
        # Let's duplicate minimal logic for safety/speed in this phase.
        
        target_cols = self.cols_by_type.get("TARGET", [])
        if not target_cols:
            return {"error": "No target column found for churn simulation."}
            
        target_col = target_cols[0]
        df_clean = self.df.copy()
        
        # Drop ID/Time/PII
        drop_cols = self.cols_by_type.get("ID", []) + self.cols_by_type.get("TIME", []) + self.cols_by_type.get("PII", [])
        feature_cols = [c for c in df_clean.columns if c != target_col and c not in drop_cols]
        
        # Encode
        le = LabelEncoder()
        for col in feature_cols:
            if df_clean[col].dtype == 'object':
                df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                
        # Impute
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(df_clean[feature_cols])
        X = pd.DataFrame(X_imputed, columns=feature_cols)
        y = df_clean[target_col]
        
        if y.dtype == 'object':
            y = le.fit_transform(y.astype(str))
            
        # Train
        clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        clf.fit(X, y)
        
        # 2. Run Simulation
        try:
            from .simulations import WhatIfSimulator
            simulator = WhatIfSimulator(clf, X, target_col)
            result = simulator.simulate(perturbations)
            return result
        except ImportError:
             # Fallback if relative import fails (e.g. running as script)
            from src.simulations import WhatIfSimulator
            simulator = WhatIfSimulator(clf, X, target_col)
            result = simulator.simulate(perturbations)
            return result

    def analyze_cohorts(self) -> Dict[str, Any]:
        """
        Generates a retention matrix.
        Requires TIME (transaction date) and ID (user id).
        """
        time_cols = self.cols_by_type.get("TIME", [])
        id_cols = self.cols_by_type.get("ID", [])
        
        if not time_cols or not id_cols:
            return {"error": "Need Time and ID columns for cohort analysis."}
            
        date_col = time_cols[0]
        user_col = id_cols[0]
        
        df = self.df.copy()
        
        # Ensure date
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except:
            return {"error": f"Could not parse date column '{date_col}'."}
            
        # Create CohortMonth (first transaction month)
        df['TransactionMonth'] = df[date_col].dt.to_period('M')
        df['CohortMonth'] = df.groupby(user_col)['TransactionMonth'].transform('min')
        
        # Calculate Cohort Index (months since first transaction)
        def get_date_int(df, column):
            year = df[column].dt.year
            month = df[column].dt.month
            return year, month

        transaction_year, transaction_month = get_date_int(df, 'TransactionMonth')
        cohort_year, cohort_month = get_date_int(df, 'CohortMonth')
        
        years_diff = transaction_year - cohort_year
        months_diff = transaction_month - cohort_month
        
        df['CohortIndex'] = years_diff * 12 + months_diff + 1
        
        # Count active users in each cohort/index
        grouping = df.groupby(['CohortMonth', 'CohortIndex'])
        cohort_data = grouping[user_col].apply(pd.Series.nunique).reset_index()
        
        # Pivot
        cohort_counts = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values=user_col)
        
        # Retention Rate
        cohort_sizes = cohort_counts.iloc[:,0]
        retention = cohort_counts.divide(cohort_sizes, axis=0)
        
        # Format for Heatmap Chart
        # We'll take the first 12 months and last 12 cohorts to keep it readable
        retention_cut = retention.iloc[-12:, :12]
        
        labels_x = [str(c) for c in retention_cut.columns]
        labels_y = [str(r) for r in retention_cut.index]
        data_matrix = retention_cut.fillna(0).values.tolist()
        
        insights = [
            f"📅 **Cohorts**: Analyzed retention for {len(cohort_sizes)} monthly cohorts.",
            f"🔻 **Drop-off**: Average Month 1 retention is {retention.iloc[:,1].mean():.1%}."
        ]
        
        chart = {
            "id": "cohort_retention",
            "title": "User Retention by Cohort",
            "type": "heatmap", # Frontend needs to support heatmap or we map to grid
            "labels_x": labels_x,
            "labels_y": labels_y,
            "data": data_matrix,
            "label": "Retention Rate"
        }
        
        return {"summary": "Cohort Analysis", "insights": insights, "charts": [chart]}

    def analyze_anomalies(self) -> Dict[str, Any]:
        """
        Detects anomalies in numeric metrics using IsolationForest.
        """
        metric_cols = self.cols_by_type.get("METRIC", []) + self.cols_by_type.get("MONEY_IN", [])
        # Exclude Coordinates if possible
        coord_cols = self.cols_by_type.get("COORDINATE", [])
        target_cols = [c for c in metric_cols if c not in coord_cols]
        
        if not target_cols:
            # Fallback if only coordinates are numeric
            target_cols = metric_cols
            
        if not target_cols:
            return {"error": "No numeric metrics found for anomaly detection."}
            
        target_col = target_cols[0]
        data = self.df[[target_col]].dropna()
        
        clf = IsolationForest(contamination=0.05, random_state=42)
        data['anomaly'] = clf.fit_predict(data[[target_col]])
        
        anomalies = data[data['anomaly'] == -1]
        
        insights = [
            f"🚨 **Anomalies**: Detected {len(anomalies)} outliers in '{target_col}'.",
            f"📉 **Range**: Normal values are mostly between {data[data['anomaly']==1][target_col].min():.2f} and {data[data['anomaly']==1][target_col].max():.2f}."
        ]
        
        # Chart: Scatter with anomalies highlighted? 
        # Or just a histogram/line with outliers marked.
        # Let's do a line chart if we have time, otherwise scatter index vs value
        chart = {
            "id": "anomaly_scatter",
            "title": f"Anomalies in {target_col}",
            "type": "scatter",
            "labels": data.index.tolist(), # X-axis (Index)
            "data": data[target_col].tolist(),
            "highlight_indices": anomalies.index.tolist(), # Frontend needs to handle this
            "label": target_col
        }
        
        return {"summary": "Anomaly Detection", "insights": insights, "charts": [chart]}

# ==============================================================================
# PHASE 1: SEMANTIC SCHEMA INFERENCE ("The Brain")
# ==============================================================================

class SemanticMapper:
    """
    Maps raw column names to semantic business concepts.
    """
    
    SEMANTIC_TYPES = {
        "MONEY_IN": ["sales", "revenue", "profit", "income", "price", "amount", "billing"],
        "MONEY_OUT": ["cost", "expense", "burn", "spend", "loss", "salary", "payout"],
        "TIME": ["date", "time", "year", "month", "day", "hour", "duration", "timestamp", "week"],
        "CATEGORY": ["department", "region", "city", "state", "country", "type", "category", "segment", "gender", "source", "company", "product", "discount band"],
        "ID": ["id", "uuid", "guid", "pk"],
        "TARGET": ["churn", "status", "target", "label", "class"],
        "PII": ["email", "phone", "ssn", "social security", "credit card", "card number", "password", "secret", "token", "iban", "api_key", "bearer", "jwt", "address"],
        "COORDINATE": ["latitude", "longitude", "lat", "long", "lng", "gps"]
    }

    PII_PATTERNS = {
        "EMAIL": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        "PHONE": r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        "SSN": r'\d{3}-\d{2}-\d{4}',
        "CREDIT_CARD": r'\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}',
        "IBAN": r'[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}',
        "API_KEY": r'(?i)(api_key|apikey|secret|token).{0,10}([a-zA-Z0-9]{20,})',
        "JWT": r'ey[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*',
        "ADDRESS": r'\d+\s+[A-Za-z]+\s+(St|Ave|Rd|Blvd|Lane|Dr|Way|Court|Plaza)'
    }

    @staticmethod
    def infer_schema(df: pd.DataFrame) -> Dict[str, str]:
        """
        Returns a dict mapping column_name -> semantic_type (or 'UNKNOWN')
        """
        mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            inferred_type = "UNKNOWN"
            
            # 1. Heuristic Matching
            for s_type, keywords in SemanticMapper.SEMANTIC_TYPES.items():
                if any(k in col_lower for k in keywords):
                    inferred_type = s_type
                    break
            
            # 2. Data Type Refinement
            dtype = df[col].dtype
            
            # Check for PII content in ALL columns (including ID, Numeric, Text)
            # unless it's already flagged as PII by name
            # SKIP floats (metrics) to avoid false positives with phone/CC regex on long decimals
            if inferred_type != "PII" and not pd.api.types.is_float_dtype(dtype):
                # Scan deeper (100 rows)
                sample_values = df[col].dropna().astype(str).head(100).tolist()
                for val in sample_values:
                    # Skip short values to avoid false positives
                    if len(val) < 4: continue
                    
                    for pii_type, pattern in SemanticMapper.PII_PATTERNS.items():
                        if re.search(pattern, val):
                            inferred_type = "PII"
                            break
                    if inferred_type == "PII":
                        break

            if inferred_type == "UNKNOWN" or inferred_type == "TEXT":
                if pd.api.types.is_numeric_dtype(dtype):
                    # If it has 'id' in name, likely ID, else METRIC
                    if "id" in col_lower:
                        inferred_type = "ID"
                    else:
                        inferred_type = "METRIC"
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    inferred_type = "TIME"
                elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
                    if df[col].nunique() < len(df) * 0.5: # Low cardinality -> Category
                        inferred_type = "CATEGORY"
                    else:
                        inferred_type = "TEXT"

            mapping[col] = inferred_type
            
        return mapping

# ==============================================================================
# PHASE 2: ANALYTIC ENGINES ("The Tools")
# ==============================================================================

class AnalyticEngine:
    def __init__(self, df: pd.DataFrame, schema: Dict[str, str]):
        self.df = df
        self.schema = schema
        self.cols_by_type = {}
        self.vector_engine = None # Lazy load
        for col, t in schema.items():
            self.cols_by_type.setdefault(t, []).append(col)

    def preprocess(self):
        # Robust preprocessing that handles currency strings, etc.
        df_clean = self.df.copy()
        
        # 1. Clean Currency/Strings in Numerical Columns
        for col in df_clean.columns:
            # If inferred as MONEY or METRIC but dtype is object/string, clean it
            if self.schema.get(col) in ["MONEY_IN", "MONEY_OUT", "METRIC"] and df_clean[col].dtype == 'object':
                try:
                    # Remove $, ,, spaces, parentheses
                    df_clean[col] = df_clean[col].astype(str).str.replace(r'[$,\(\) ]', '', regex=True)
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                except:
                    pass

        num_cols = df_clean.select_dtypes(include=np.number).columns
        cat_cols = df_clean.select_dtypes(exclude=np.number).columns
        
        if len(num_cols) > 0:
            df_clean[num_cols] = SimpleImputer(strategy='mean').fit_transform(df_clean[num_cols])
        
        if len(cat_cols) > 0:
            df_clean[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df_clean[cat_cols])
            
        return df_clean

    def analyze_profitability(self):
        """Analyzes Money In/Out by Category - ENHANCED with multiple diverse insights."""
        df_clean = self.preprocess()
        
        money_cols = self.cols_by_type.get("MONEY_IN", []) + self.cols_by_type.get("MONEY_OUT", [])
        cat_cols = self.cols_by_type.get("CATEGORY", [])
        
        if not money_cols or not cat_cols:
            return None

        target_money = money_cols[0]
        target_cat = cat_cols[0]
        
        # Multiple aggregations for diverse insights
        grouped = df_clean.groupby(target_cat)[target_money].sum().sort_values(ascending=False)
        top_5 = grouped.head(5)
        bottom_5 = grouped.tail(5)
        
        # Calculate statistics
        total = grouped.sum()
        mean_value = grouped.mean()
        std_value = grouped.std()
        
        insights = [
            f"🏆 **Top Performer**: {top_5.index[0]} accounts for ${top_5.values[0]:,.2f} ({top_5.values[0]/total*100:.1f}% of total revenue).",
            f"📊 **Concentration Risk**: Top 3 categories represent {(top_5.head(3).sum()/total*100):.1f}% of all {target_money}.",
            f"⚠️ **Underperformer**: {bottom_5.index[-1]} generates only ${bottom_5.values[-1]:,.2f}, which is {(bottom_5.values[-1]/mean_value*100):.1f}% of average.",
            f"💡 **Opportunity**: Bottom 5 categories combined (${bottom_5.sum():,.2f}) are only {(bottom_5.sum()/total*100):.1f}% of total.",
            f"📈 **Variability**: Revenue varies widely (σ=${std_value:,.2f}), suggesting different business dynamics across categories.",
            f"🎯 **Quick Win**: Improving {bottom_5.index[-2]} by 20% would add ${bottom_5.values[-2]*0.2:,.2f} with likely less effort than chasing new markets."
        ]
        
        # Chart 1: Top 5 Performers
        chart_top = {
            "id": "top_performers",
            "type": "bar",
            "title": f"Top 5 {target_cat} by {target_money}",
            "labels": [str(x) for x in top_5.index.tolist()],
            "data": top_5.values.tolist(),
            "insight_ref": [0, 1],  # References insights 1 and 2
            "color_scheme": "success"
        }
        
        # Chart 2: Bottom 5 for comparison
        chart_bottom = {
            "id": "underperformers",
            "type": "bar",
            "title": f"Bottom 5 {target_cat} (Opportunities)",
            "labels": [str(x) for x in bottom_5.index.tolist()],
            "data": bottom_5.values.tolist(),
            "insight_ref": [2, 3],  # References insights 3 and 4
            "color_scheme": "warning"
        }
        
        # Chart 3: Distribution pie
        chart_distribution = {
            "id": "concentration",
            "type": "doughnut",
            "title": "Revenue Concentration",
            "labels": ["Top 3", "Others"],
            "data": [top_5.head(3).sum(), grouped.sum() - top_5.head(3).sum()],
            "insight_ref": [1],  # References insight 2 (concentration risk)
            "color_scheme": "primary"
        }
        
        charts = [chart_top, chart_bottom, chart_distribution]
        
        return {"summary": "Profitability Analysis", "insights": insights, "charts": charts}

    def analyze_trends(self):
        """Analyzes Metrics over Time."""
        # Use preprocessed data to clean metric values
        df_clean = self.preprocess()
        
        time_cols = self.cols_by_type.get("TIME", [])
        metric_cols = self.cols_by_type.get("MONEY_IN", []) + self.cols_by_type.get("METRIC", [])
        
        if not time_cols or not metric_cols:
            # Fallback: Try to parse string columns as dates if not already found
            for col in self.df.columns:
                if "date" in col.lower():
                    try:
                        df_clean[col] = pd.to_datetime(self.df[col])
                        time_cols = [col]
                        break
                    except:
                        pass
            if not time_cols:
                return None

        target_time = time_cols[0]
        target_metric = metric_cols[0]
        
        # Ensure time is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_clean[target_time]):
             df_clean[target_time] = pd.to_datetime(df_clean[target_time], errors='coerce')
             df_clean = df_clean.dropna(subset=[target_time])

        # Resample/Group by time
        df_sorted = df_clean.sort_values(by=target_time)
        
        try:
            df_sorted.set_index(target_time, inplace=True)
            # Resample to monthly if data spans > 60 days
            span = (df_sorted.index.max() - df_sorted.index.min()).days
            if span > 60:
                trend = df_sorted[target_metric].resample('M').sum()
            else:
                trend = df_sorted[target_metric] # Raw
        except:
            trend = df_sorted[target_metric]

        # Fill missing trend parts
        trend = trend.fillna(0)

        # If trend is too long, sample it
        if len(trend) > 20:
            trend = trend.iloc[::len(trend)//20]

        insights = [
            f"⏰ **Time Span**: Analysis covers {(df_sorted.index.max() - df_sorted.index.min()).days} days from {df_sorted.index.min().strftime('%Y-%m-%d')} to {df_sorted.index.max().strftime('%Y-%m-%d')}.",
            f"🔺 **Peak Performance**: Highest {target_metric} was {trend.max():,.2f} on {trend.idxmax().strftime('%Y-%m-%d' if hasattr(trend.idxmax(), 'strftime') else '%s')}.",
            f"🔻 **Lowest Point**: Minimum {target_metric} of {trend.min():,.2f} occurred on {trend.idxmin().strftime('%Y-%m-%d' if hasattr(trend.idxmin(), 'strftime') else '%s')}.",
            f"📊 **Volatility**: {target_metric} ranges from {trend.min():,.2f} to {trend.max():,.2f}, a span of {(trend.max()-trend.min())/trend.mean()*100:.1f}% of the mean.",
            f"💹 **Average Performance**: Mean {target_metric} is {trend.mean():,.2f} with standard deviation of {trend.std():,.2f}.",
            f"🔎 **Recent Trend**: Last recorded value ({trend.iloc[-1]:,.2f}) is {((trend.iloc[-1]/trend.mean()-1)*100):+.1f}% vs. overall average.",
            f"🎯 **Actionable**: {'Consider investigating the spike' if trend.max() > trend.mean() + 2*trend.std() else 'Trend appears stable'} - {'recent decline warrants attention' if len(trend) > 3 and trend.iloc[-1] < trend.iloc[-3] else 'momentum is positive'}."
        ]
        
        # Chart 1: Main trend line
        chart_trend = {
            "id": "trend_line",
            "type": "line",
            "title": f"{target_metric} Over Time",
            "labels": [str(d) for d in trend.index],
            "data": [float(v) for v in trend.values],
            "insight_ref": [0, 1, 2, 5],
            "color_scheme": "primary"
        }
        
        # Chart 2: Peak vs Valley comparison
        chart_peaks = {
            "id": "peak_valley",
            "type": "bar",
            "title": "Performance Extremes",
            "labels": ["Peak", "Average", "Valley"],
            "data": [float(trend.max()), float(trend.mean()), float(trend.min())],
            "insight_ref": [1, 2, 4],
            "color_scheme": "mixed"
        }
        
        # Chart 3: Recent trend (last 10 points)
        recent_trend = trend.tail(min(10, len(trend)))
        chart_recent = {
            "id": "recent_momentum",
            "type": "line",
            "title": "Recent Momentum",
            "labels": [str(d) for d in recent_trend.index],
            "data": [float(v) for v in recent_trend.values],
            "insight_ref": [5, 6],
            "color_scheme": "success" if len(recent_trend) > 1 and recent_trend.iloc[-1] > recent_trend.iloc[0] else "warning"
        }
        
        charts = [chart_trend, chart_peaks, chart_recent]
        
        return {"summary": "Trend Analysis", "insights": insights, "charts": charts}

    def analyze_anomalies(self):
        """Finds outliers in numerical data."""
        df_clean = self.preprocess()
        
        # Get all numeric columns first
        num_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
        
        # Filter out Coordinates and IDs for better business anomalies
        exclude_cols = self.cols_by_type.get("COORDINATE", []) + self.cols_by_type.get("ID", [])
        business_cols = [c for c in num_cols if c not in exclude_cols]
        
        # Selection Strategy:
        # 1. Use Business Metrics (Money, Count, etc.) if >= 2 available
        # 2. Use Coordinates if available (Geospatial outlier detection)
        # 3. Fallback to any numeric
        
        mode = "generic"
        selected_cols = []
        
        if len(business_cols) >= 2:
            selected_cols = business_cols[:2] # Pick top 2 business metrics
            mode = "business"
        elif self.cols_by_type.get("COORDINATE") and len(self.cols_by_type.get("COORDINATE")) >= 2:
            selected_cols = self.cols_by_type.get("COORDINATE")[:2]
            mode = "geo"
        elif len(num_cols) >= 2:
            selected_cols = num_cols[:2]
        else:
            return None
            
        col_x = selected_cols[0]
        col_y = selected_cols[1]
        
        model = IsolationForest(contamination=0.05, random_state=42)
        df_clean['anomaly'] = model.fit_predict(df_clean[selected_cols])
        
        anomalies = df_clean[df_clean['anomaly'] == -1]
        
        if mode == "geo":
            insights = [
                f"🌍 **Geographic Outliers**: Found {len(anomalies)} locations that are far from the main clusters.",
                f"📍 **Location**: Analyzing '{col_x}' vs '{col_y}'.",
                f"📏 **Spread**: Outliers likely represent remote areas or data entry errors in coordinates."
            ]
        else:
            insights = [
                f"🔴 **Anomalies Detected**: Found **{len(anomalies)} anomalies** ({len(anomalies)/len(df_clean)*100:.1f}% of {len(df_clean)} total records).",
                f"💡 **What are Anomalies?**: These are data points that deviate significantly from normal patterns - could be fraud, errors, or exceptional cases.",
                f"🔍 **Analysis Dimensions**: Used {len(selected_cols)} numerical features ({col_x}, {col_y}) to detect outliers via Isolation Forest.",
                f"📊 **Severity Distribution**: Anomaly scores range from extreme ({model.score_samples(df_clean[selected_cols]).min():.3f}) to moderate thresholds.",
                f"⚡ **Investigation Priority**: Focus on anomalies with extreme values in **{col_x}** (max: {anomalies[col_x].max():.2f}) and **{col_y}** (max: {anomalies[col_y].max():.2f}).",
                f"🎯 **Common Patterns**: {'High clustering suggests systemic issues' if len(anomalies) > len(df_clean)*0.1 else 'Scattered anomalies suggest isolated incidents'}.",
                f"✅ **Next Steps**: {'Review data quality processes' if len(anomalies) > 100 else 'Manually inspect each case'} - potential value in anomalies: {'Very High' if len(anomalies) < 10 else 'Mixed'}."
            ]
        
        # Safe sampling logic
        normal_sample_size = min(100, len(df_clean[df_clean['anomaly'] == 1]))
        anomaly_sample_size = min(50, len(anomalies))
        
        chart_data = {
            "type": "scatter",
            "labels": ["Normal", "Anomaly"],
            "datasets": [
                {
                    "label": "Normal",
                    "data": df_clean[df_clean['anomaly'] == 1].sample(normal_sample_size, replace=False).to_dict(orient='records'), 
                    "parsing": {"xAxisKey": col_x, "yAxisKey": col_y}
                },
                {
                    "label": "Anomaly",
                    "data": anomalies.sample(anomaly_sample_size, replace=False).to_dict(orient='records'),
                    "parsing": {"xAxisKey": col_x, "yAxisKey": col_y},
            "backgroundColor": "red"
                }
            ]
        }
        
        chart_data["id"] = "anomaly_scatter"
        chart_data["title"] = "Geographic Outliers" if mode == "geo" else "Anomaly Detection"
        
        return {"summary": chart_data["title"], "insights": insights, "charts": [chart_data]}
    
    def analyze_correlations(self):
        """Find surprising correlations between variables."""
        df_clean = self.preprocess()
        num_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
        
        if len(num_cols) < 2:
            return None
        
        # Calculate correlation matrix
        corr_matrix = df_clean[num_cols].corr()
        
        # Find strongest correlations (excluding diagonal)
        strong_corrs = []
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.3:  # Only meaningful correlations
                    strong_corrs.append((corr_matrix.index[i], corr_matrix.columns[j], corr_val))
        
        strong_corrs = sorted(strong_corrs, key=lambda x: abs(x[2]), reverse=True)[:5]
        
        insights = []
        if len(strong_corrs) > 0:
            top_corr = strong_corrs[0]
            insights.append(f"🔗 **Strongest Relationship**: {top_corr[0]} and {top_corr[1]} are {('highly correlated' if top_corr[2] > 0 else 'inversely correlated')} (r={top_corr[2]:.3f}).")
            
            positive_corrs = [c for c in strong_corrs if c[2] > 0]
            negative_corrs = [c for c in strong_corrs if c[2] < 0]
            
            if positive_corrs:
                insights.append(f"📈 **Positive Drivers**: Found {len(positive_corrs)} positive correlations - when one goes up, the other follows.")
            if negative_corrs:
                insights.append(f"📉 **Inverse Relationships**: {len(negative_corrs)} negative correlations detected - potential trade-offs or constraints.")
            
            insights.append(f"💡 **Actionable**: Strong correlation between {top_corr[0]} and {top_corr[1]} means optimizing one will likely influence the other.")
            insights.append(f"🎯 **Caution**: Correlation ≠ causation. Further testing needed to confirm causal relationships.")
            
            # Surprising correlations (unexpected based on semantics)
            money_cols = self.cols_by_type.get("MONEY_IN", []) + self.cols_by_type.get("MONEY_OUT", [])
            time_sensitive = [c for c in strong_corrs if any(col in money_cols for col in c[:2])]
            if time_sensitive:
                insights.append(f"💰 **Revenue Impact**: {len(time_sensitive)} financial metrics show correlation with other variables.")
        else:
            insights.append("🤔 **Weak Correlations**: No strong relationships found between variables (all |r| < 0.3).")
            insights.append("📊 **Independence**: Variables appear largely independent - changes in one won't predictably affect others.")
        
        charts = []
        if strong_corrs:
            chart_data = {
                "id": "correlation_bar",
                "title": "Top Correlations",
                "type": "bar",
                "labels": [f"{c[0]} vs {c[1]}" for c in strong_corrs],
                "data": [abs(c[2]) for c in strong_corrs],
                "label": "Correlation Strength (|r|)"
            }
            charts.append(chart_data)
        
        return {"summary": "Correlation Analysis", "insights": insights, "charts": charts}
    
    def analyze_distributions(self):
        """Analyze the distribution shape of key metrics."""
        df_clean = self.preprocess()
        metric_cols = self.cols_by_type.get("METRIC", []) + self.cols_by_type.get("MONEY_IN", [])[:2]
        
        if not metric_cols:
            return None
        
        target_col = metric_cols[0]
        data = df_clean[target_col].dropna()
        
        if len(data) < 10:
            return None
        
        # Statistical properties
        mean_val = data.mean()
        median_val = data.median()
        std_val = data.std()
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        skewness = data.skew()
        
        insights = [
            f"📊 **Central Tendency**: Mean is {mean_val:,.2f}, Median is {median_val:,.2f} ({'roughly symmetric' if abs(mean_val - median_val) < std_val*0.1 else 'skewed distribution'}).",
            f"📏 **Spread**: Data ranges from {data.min():,.2f} to {data.max():,.2f}, with std deviation of {std_val:,.2f}.",
            f"📦 **Middle 50%**: Half of all values fall between {q1:,.2f} and {q3:,.2f} (IQR = {iqr:,.2f}).",
        ]
        
        # Skewness interpretation
        if abs(skewness) < 0.5:
            insights.append(f"⚖️ **Shape**: Distribution is fairly symmetric (skew={skewness:.2f}), similar to normal distribution.")
        elif skewness > 0.5:
            insights.append(f"📈 **Right-Skewed**: Most values are on the low end with a long tail of high values (skew={skewness:.2f}) - a few outliers pull the average up.")
        else:
            insights.append(f"📉 **Left-Skewed**: Most values are on the high end with some low outliers (skew={skewness:.2f}).")
        
        # Outlier detection via IQR
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        outliers = data[(data < lower_fence) | (data > upper_fence)]
        insights.append(f"🚨 **Outliers**: {len(outliers)} values ({len(outliers)/len(data)*100:.1f}%) fall outside the typical range [{lower_fence:,.2f}, {upper_fence:,.2f}].")
        
        # Practical implications
        if skewness > 1:
            insights.append(f"💡 **Implication**: Heavy right skew suggests most entities are similar, with a few high performers. Focus on replicating top performers' strategies.")
        elif mean_val > median_val * 1.2:
            insights.append(f"🎯 **Actionable**: Mean ({mean_val:,.2f}) significantly exceeds median ({median_val:,.2f}) - address low performers to raise median.")
        else:
            insights.append(f"✅ **Healthy**: Distribution is well-balanced without extreme concentration issues.")
        
        # Create histogram data
        hist, bins = np.histogram(data, bins=min(20, len(data)//10 + 1))
        chart_data = {
            "id": "distribution_hist",
            "title": f"Distribution of {target_col}",
            "type": "bar",
            "labels": [f"{float(b):.1f}" for b in bins[:-1]],
            "data": [int(x) for x in hist],
            "label": "Frequency"
        }
        
        return {"summary": "Distribution Analysis", "insights": insights, "charts": [chart_data]}

    def analyze_text_clusters(self):
        """Performs Semantic Clustering on text data using TF-IDF + PCA + KMeans."""
        df_clean = self.preprocess()
        text_cols = self.cols_by_type.get("TEXT", [])
        
        if not text_cols:
            return None
            
        # Use the longest text column (likely to contain the most signal)
        target_col = max(text_cols, key=lambda c: df_clean[c].astype(str).str.len().mean())
        
        # Sample if too large (TF-IDF is expensive)
        if len(df_clean) > 5000:
            df_sample = df_clean.sample(5000, random_state=42)
        else:
            df_sample = df_clean
            
        raw_text = df_sample[target_col].astype(str).fillna("")
        
        # 1. Vectorize (Convert text to numbers)
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(raw_text)
        except ValueError:
            return None # Empty vocabulary or stop words issue
            
        # 2. Cluster (Group similar vectors)
        num_clusters = 5
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(tfidf_matrix)
        
        # 3. Dimensionality Reduction (Squash to 2D for visualization)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(tfidf_matrix.toarray())
        
        # 4. Extract Keywords per Cluster
        feature_names = vectorizer.get_feature_names_out()
        cluster_keywords = {}
        
        for i in range(num_clusters):
            # Get center of cluster
            center = kmeans.cluster_centers_[i]
            # Get top 5 words
            top_indices = center.argsort()[-5:][::-1]
            keywords = [feature_names[ind] for ind in top_indices]
            cluster_keywords[i] = ", ".join(keywords)
            
        # 5. Build Insights
        insights = [
            f"🧠 **Semantic Clustering**: Analyzed {len(df_sample)} text records from '{target_col}' and grouped them into {num_clusters} thematic clusters.",
            f"🔍 **Technique**: Used TF-IDF Vectorization (to capture keywords) combined with K-Means clustering and PCA for visualization.",
        ]
        
        # Add insight for each cluster
        counts = pd.Series(clusters).value_counts()
        for i in range(min(3, num_clusters)):
            cluster_id = counts.index[i]
            size = counts.iloc[i]
            keywords = cluster_keywords[cluster_id]
            insights.append(f"📂 **Cluster {cluster_id + 1}** ({size} items): Dominated by terms like **'{keywords}'**.")
            
        insights.append(f"💡 **Business Value**: This technique automatically organizes unstructured text into topics, useful for tagging support tickets, categorizing reviews, or spotting emerging themes.")

        # 6. Build Chart (Scatter Plot)
        chart_data = {
            "id": "semantic_map",
            "title": f"Semantic Map of {target_col}",
            "type": "scatter",
            "labels": [f"Cluster {c+1}" for c in sorted(list(set(clusters)))],
            "datasets": []
        }
        
        # Create a dataset for each cluster so they have different colors
        for i in range(num_clusters):
            mask = clusters == i
            cluster_coords = coords[mask]
            # Sample points for chart if too many
            if len(cluster_coords) > 100:
                indices = np.random.choice(len(cluster_coords), 100, replace=False)
                cluster_coords = cluster_coords[indices]
                
            chart_data["datasets"].append({
                "label": f"Cluster {i+1} ({cluster_keywords[i][:20]}...)",
                "data": [{"x": float(pt[0]), "y": float(pt[1])} for pt in cluster_coords],
                "backgroundColor": ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF"][i % 5]
            })
            
        return {"summary": "Semantic Clustering Analysis", "insights": insights, "charts": [chart_data]}

    def analyze_cohorts(self):
        """
        Performs Cohort Analysis (Retention Heatmap).
        Requires: Time column + User ID column.
        """
        df_clean = self.preprocess()
        
        # Identify columns
        time_cols = self.cols_by_type.get("TIME", [])
        id_cols = self.cols_by_type.get("ID", [])
        
        if not time_cols or not id_cols:
            return None
            
        # Use first time col and first ID col (heuristic)
        date_col = time_cols[0]
        user_col = id_cols[0]
        
        # Ensure date column is datetime
        try:
            # Check if numeric and large (likely timestamp)
            if pd.api.types.is_numeric_dtype(df_clean[date_col]):
                mean_val = df_clean[date_col].mean()
                if mean_val > 1e11: # Milliseconds
                    df_clean[date_col] = pd.to_datetime(df_clean[date_col], unit='ms')
                elif mean_val > 1e8: # Seconds
                    df_clean[date_col] = pd.to_datetime(df_clean[date_col], unit='s')
                else:
                    df_clean[date_col] = pd.to_datetime(df_clean[date_col])
            else:
                df_clean[date_col] = pd.to_datetime(df_clean[date_col])
        except:
            return None
            
        # 1. Create CohortMonth (Month of first activity for each user)
        # Group by user, take min date, truncate to month
        first_activity = df_clean.groupby(user_col)[date_col].transform('min')
        df_clean['CohortMonth'] = first_activity.dt.to_period('M')
        
        # 2. Create ActivityMonth (Month of current activity)
        df_clean['ActivityMonth'] = df_clean[date_col].dt.to_period('M')
        
        # 3. Calculate CohortIndex (Months since first activity)
        # We can't subtract periods directly in all pandas versions easily to get int, 
        # so we do (year_diff * 12 + month_diff)
        def diff_month(d1, d2):
            return (d1.year - d2.year) * 12 + d1.month - d2.month

        # Need to convert Period to Timestamp for calculation if using apply, 
        # or just extract year/month from Period
        # Faster vector approach:
        df_clean['CohortIndex'] = (df_clean['ActivityMonth'].dt.year - df_clean['CohortMonth'].dt.year) * 12 + \
                                  (df_clean['ActivityMonth'].dt.month - df_clean['CohortMonth'].dt.month)
                                  
        # 4. Count unique users in each (CohortMonth, CohortIndex)
        cohort_data = df_clean.groupby(['CohortMonth', 'CohortIndex'])[user_col].nunique().reset_index()
        
        # 5. Pivot: Rows=CohortMonth, Cols=CohortIndex, Values=UserCount
        cohort_counts = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values=user_col)
        
        # 6. Calculate Retention (as percentage of Month 0)
        cohort_sizes = cohort_counts.iloc[:, 0]
        retention = cohort_counts.divide(cohort_sizes, axis=0)
        
        # Limit to last 12 months / 12 indexes to keep chart readable
        retention = retention.iloc[-12:, :13] # Last 12 cohorts, first 13 months (0-12)
        
        # 7. Build Insights
        avg_retention_m1 = retention.iloc[:, 1].mean() if 1 in retention.columns else 0
        insights = [
            f"📅 **Cohort Analysis**: Tracked user retention over time based on '{date_col}'.",
            f"👥 **Retention Rate**: On average, {avg_retention_m1:.1%} of users return in the first month after acquisition.",
        ]
        
        if avg_retention_m1 > 0.5:
            insights.append("✅ **High Stickiness**: Users are highly engaged and keep coming back.")
        elif avg_retention_m1 < 0.1:
            insights.append("⚠️ **Churn Risk**: Low retention in Month 1. Consider re-engagement campaigns.")
            
        # 8. Build Chart (Heatmap)
        # Chart.js doesn't have a native heatmap, so we'll use a Bubble Chart or a specialized grid.
        # Actually, for the "Predictions" UI, we can use a Matrix chart if available, 
        # or hack a Scatter chart where X=CohortIndex, Y=CohortMonth, Color=Value.
        
        # Let's use a Scatter chart structured as a Heatmap
        # X: Month 0, 1, 2...
        # Y: Cohort Month (Index 0, 1, 2...)
        # R: Fixed size
        # Color: Based on retention value
        
        datasets = []
        cohort_labels = [str(c) for c in retention.index]
        
        # We'll flatten the data for the chart
        heatmap_data = []
        for y_idx, cohort_date in enumerate(retention.index):
            for x_idx, month_idx in enumerate(retention.columns):
                val = retention.loc[cohort_date, month_idx]
                if pd.notna(val) and val > 0:
                    heatmap_data.append({
                        "x": int(month_idx),
                        "y": y_idx, # We will map this to label on client side or just use index
                        "v": float(val),
                        "cohort": str(cohort_date)
                    })

        # Since Chart.js scatter doesn't support categorical Y easily without plugin,
        # We will try to return a "Heatmap" type and hope the frontend handles it 
        # OR we stick to a Line chart showing retention curves for top 3 cohorts.
        
        # Let's do Retention Curves (Line Chart) for the last 5 cohorts. 
        # It's cleaner and supported by standard Chart.js.
        
        chart_data = {
            "id": "retention_curves",
            "title": "User Retention by Cohort",
            "type": "line",
            "labels": [f"Month {i}" for i in range(len(retention.columns))],
            "datasets": []
        }
        
        # Add a line for each of the last 5 cohorts
        colors = ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF"]
        for i in range(min(5, len(retention))):
            # Go backwards from newest (but skip the very last one if it has no data yet)
            idx = len(retention) - 1 - i
            cohort_name = str(retention.index[idx])
            data_row = retention.iloc[idx].fillna(0).tolist()
            
            chart_data["datasets"].append({
                "label": cohort_name,
                "data": data_row,
                "borderColor": colors[i % len(colors)],
                "fill": False
            })
            
        return {"summary": "Cohort Retention Analysis", "insights": insights, "charts": [chart_data]}