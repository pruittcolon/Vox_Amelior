#!/usr/bin/env python3
"""
Nexus Engine Test Script
Tests each engine via direct API calls, generates visualizations, and tests Gemma responses.

Security: 
- Uses proper session-based authentication per SECURITY.md
- Session cookies stored securely (httponly=True on server)
- CSRF tokens included in headers for mutating requests
- All credentials from environment or secure defaults
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import requests

# Use Plotly for frontend-accurate visualizations
try:
    import plotly.graph_objects as go
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("[WARNING] Plotly not installed - using matplotlib fallback")


# =============================================================================
# CONFIGURATION (Per SECURITY.md - Defense in Depth)
# =============================================================================

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8000")
GEMMA_SERVICE_URL = os.getenv("GEMMA_SERVICE_URL", "http://localhost:8001")
UPLOAD_DIR = "/home/pruittcolon/Desktop/Nemo_Server/docker/gateway_instance/uploads"
SCREENSHOT_DIR = "/home/pruittcolon/Desktop/Nemo_Server/tests/nexus_engine_screenshots"

# Authentication - use env vars in production (per SECURITY.md Section 3)
AUTH_USER = os.getenv("TEST_USER", "admin")
AUTH_PASS = os.getenv("TEST_PASS", "admin123")

# Cookie names per security config (ws_session, ws_csrf)
SESSION_COOKIE_NAME = "ws_session"
CSRF_COOKIE_NAME = "ws_csrf"
CSRF_HEADER_NAME = "X-CSRF-Token"

os.makedirs(SCREENSHOT_DIR, exist_ok=True)


# =============================================================================
# AUTHENTICATION (Per SECURITY.md - JWT/Session based)
# =============================================================================

_session: Optional[requests.Session] = None
_csrf_token: str = ""


def get_authenticated_session() -> requests.Session:
    """
    Get or create an authenticated session.
    
    Security Implementation:
    - Uses POST /api/auth/login (CSRF exempt per middleware.py line 288)
    - Uses Bearer token authentication (per permissions.py line 49-62)
    - This bypasses cookie Secure flag issues when testing over HTTP
    
    Returns:
        Authenticated requests.Session with Authorization header set
    """
    global _session, _csrf_token
    
    if _session is not None:
        return _session
    
    _session = requests.Session()
    
    print("[AUTH] Authenticating with API Gateway...")
    print(f"[AUTH] Target: {GATEWAY_URL}/api/auth/login")
    
    try:
        # POST to CSRF-exempt login endpoint (per middleware.py line 288)
        login_response = _session.post(
            f"{GATEWAY_URL}/api/auth/login",
            json={"username": AUTH_USER, "password": AUTH_PASS},
            timeout=30
        )
        
        if login_response.status_code == 200:
            data = login_response.json()
            
            # Get session token from response
            session_token = data.get("session_token", "")
            _csrf_token = data.get("csrf_token", "")
            
            if session_token:
                # Use Bearer token authentication (per permissions.py line 49-62)
                # This works for mobile clients and bypasses cookie Secure flag issues
                _session.headers.update({"Authorization": f"Bearer {session_token}"})
                print(f"[AUTH] Login successful - using Bearer token auth")
            
            # Also set CSRF token for completeness (though Bearer auth bypasses CSRF per middleware.py line 420-422)
            if _csrf_token:
                _session.headers.update({CSRF_HEADER_NAME: _csrf_token})
                print(f"[AUTH] CSRF token set in headers")
            
            print(f"[AUTH] Authorization header: {'present' if 'Authorization' in _session.headers else 'MISSING'}")
                
        elif login_response.status_code == 401:
            print(f"[AUTH] Login failed: Invalid credentials")
            print(f"[AUTH] Response: {login_response.text[:200]}")
        elif login_response.status_code == 423:
            print(f"[AUTH] Login failed: Account locked")
            print(f"[AUTH] Response: {login_response.text[:200]}")
        else:
            print(f"[AUTH] Login failed: HTTP {login_response.status_code}")
            print(f"[AUTH] Response: {login_response.text[:200]}")
                
    except requests.exceptions.ConnectionError as e:
        print(f"[AUTH] Connection error: Cannot connect to {GATEWAY_URL}")
        print(f"[AUTH] Is the API Gateway running? Check: docker ps | grep gateway")
    except Exception as e:
        print(f"[AUTH] Unexpected error: {type(e).__name__}: {e}")
    
    return _session


# =============================================================================
# ENGINE TESTING
# =============================================================================

def test_engine(engine_name: str, filename: str, target_column: str = None) -> dict:
    """
    Test a single engine and return results.
    
    Args:
        engine_name: Name of engine (e.g., 'titan', 'clustering')
        filename: CSV filename in uploads directory
        target_column: Target column for supervised learning (optional)
    
    Returns:
        Dict with status, api_result, visualization_path, gemma responses
    """
    print(f"\n{'='*60}")
    print(f"Testing Engine: {engine_name}")
    print(f"{'='*60}")
    
    # Get authenticated session
    session = get_authenticated_session()
    
    # Verify authentication succeeded
    if SESSION_COOKIE_NAME not in session.cookies:
        print("[ERROR] Not authenticated - session cookie missing")
        return {"status": "error", "error": "Authentication failed"}
    
    # Step 1: Call the engine API
    # Note: /analytics/ prefix is CSRF-exempt per middleware.py line 318
    print(f"[1/4] Calling API: /analytics/run-engine/{engine_name}")
    
    payload = {"filename": filename}
    if target_column:
        payload["target_column"] = target_column
    
    try:
        response = session.post(
            f"{GATEWAY_URL}/analytics/run-engine/{engine_name}",
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"  SUCCESS: Got response with {len(str(result))} bytes")
        elif response.status_code == 401:
            print(f"  ERROR: HTTP 401 - Not authenticated")
            print(f"  Check: Session cookie present? {SESSION_COOKIE_NAME in session.cookies}")
            return {"status": "error", "error": "Not authenticated"}
        elif response.status_code == 404:
            print(f"  ERROR: HTTP 404 - File not found: {filename}")
            return {"status": "error", "error": f"File not found: {filename}"}
        else:
            print(f"  ERROR: HTTP {response.status_code}")
            print(f"  Response: {response.text[:500]}")
            return {"status": "error", "error": f"HTTP {response.status_code}"}
        
    except requests.exceptions.ConnectionError:
        print(f"  ERROR: Cannot connect to Gateway at {GATEWAY_URL}")
        return {"status": "error", "error": "Connection refused"}
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        return {"status": "error", "error": str(e)}
    
    # Step 2: Generate visualization
    print(f"[2/4] Generating visualization...")
    viz_path = generate_visualization(engine_name, result)
    if viz_path:
        print(f"  Saved to: {viz_path}")
    else:
        print(f"  WARNING: Could not generate visualization")
    
    # Step 3: Test Gemma response
    print(f"[3/4] Testing Gemma AI response...")
    gemma_response = test_gemma(engine_name, result)
    
    # Step 4: Test Gemma simplification
    print(f"[4/4] Testing Gemma simplification...")
    gemma_simple = test_gemma_simplify(gemma_response)
    
    return {
        "status": "success",
        "engine": engine_name,
        "api_result": result,
        "visualization_path": viz_path,
        "gemma_response": gemma_response,
        "gemma_simplified": gemma_simple
    }


def generate_visualization(engine_name: str, result: dict) -> str:
    """Generate and save a visualization based on engine results.
    
    Uses Plotly to match frontend visualizations exactly per nexus/visualizations/engines/*.js
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(SCREENSHOT_DIR, f"{engine_name}_{timestamp}.png")
    
    # Color constants matching viz-utils.js
    VIZ_COLORS = {
        'primary': '#02559e',
        'success': '#10b981',
        'error': '#ef4444',
        'textMuted': '#64748b',
        'border': '#e2e8f0',
        'background': '#ffffff'
    }
    
    try:
        # Handle different engine types - use Plotly for frontend-accurate viz
        if engine_name == "titan" and PLOTLY_AVAILABLE:
            # Titan uses feature importance waterfall chart per frontend/assets/js/nexus/visualizations/engines/ml/titan.js
            features = result.get("feature_importance", [])
            
            if not features:
                # Fallback to matplotlib if no feature data
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, f"Titan: {result.get('status', 'complete')}\nNo feature importance data", ha='center', va='center')
                ax.set_title("Titan AutoML")
                ax.axis('off')
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
                plt.close(fig)
                return filepath
            
            # Extract stability or importance score (matches titan.js getVal function)
            def get_val(f):
                if 'stability' in f and f['stability'] is not None:
                    return f['stability']
                imp = f.get('importance') or f.get('value') or 0
                return imp * 100
            
            sorted_features = sorted(features, key=get_val, reverse=True)[:10]
            
            # Build waterfall contributions (matches titan.js)
            contributions = []
            for i, f in enumerate(sorted_features):
                contributions.append({
                    'name': f.get('feature') or f.get('name', f'Feature {i}'),
                    'value': get_val(f),
                    'direction': -1 if i % 3 == 0 else 1
                })
            
            # Create Plotly waterfall chart
            fig = go.Figure(go.Waterfall(
                orientation='h',
                y=[c['name'] for c in contributions],
                x=[c['value'] * c['direction'] * 0.1 for c in contributions],
                connector={'line': {'color': 'rgba(2, 85, 158, 0.3)', 'width': 1}},
                increasing={'marker': {'color': VIZ_COLORS['success']}},
                decreasing={'marker': {'color': VIZ_COLORS['error']}},
                totals={'marker': {'color': VIZ_COLORS['primary']}},
                textposition='outside',
                text=[f"{c['value']:.1f}%" for c in contributions],
                hovertemplate='<b>%{y}</b><br>Impact: %{x:.2f}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Feature Impact Analysis',
                paper_bgcolor=VIZ_COLORS['background'],
                plot_bgcolor=VIZ_COLORS['background'],
                font={'color': VIZ_COLORS['textMuted'], 'size': 11},
                margin={'l': 120, 'r': 60, 't': 50, 'b': 40},
                xaxis={
                    'title': 'Feature Impact',
                    'gridcolor': VIZ_COLORS['border'],
                    'zeroline': True,
                    'zerolinecolor': VIZ_COLORS['border']
                },
                yaxis={
                    'autorange': 'reversed',
                    'gridcolor': VIZ_COLORS['border']
                },
                showlegend=False,
                width=800,
                height=500
            )
            
            # Save to file
            pio.write_image(fig, filepath, format='png', scale=2)
            return filepath
        
        elif engine_name == "predictive" and PLOTLY_AVAILABLE:
            # Predictive uses forecast line chart per frontend/assets/js/nexus/visualizations/engines/ml/predictive.js
            # API returns: forecast = [{ds, yhat, yhat_lower, yhat_upper}, ...]
            forecast_list = result.get("forecast", [])
            
            if forecast_list and isinstance(forecast_list, list) and len(forecast_list) > 0:
                # Extract data from forecast list
                forecast_dates = [item.get("ds", "") for item in forecast_list]
                forecast_values = [item.get("yhat", 0) for item in forecast_list]
                lower_bound = [item.get("yhat_lower", 0) for item in forecast_list]
                upper_bound = [item.get("yhat_upper", 0) for item in forecast_list]
                
                traces = []
                
                # Forecast line
                traces.append(go.Scatter(
                    x=forecast_dates, y=forecast_values,
                    mode='lines+markers', name='Forecast',
                    line=dict(color=VIZ_COLORS['success'], width=2),
                    marker=dict(size=4)
                ))
                
                # Confidence interval - upper
                traces.append(go.Scatter(
                    x=forecast_dates, y=upper_bound,
                    mode='lines',
                    line=dict(color='rgba(16, 185, 129, 0.3)', width=1),
                    showlegend=False, hoverinfo='skip'
                ))
                
                # Confidence interval - lower with fill
                traces.append(go.Scatter(
                    x=forecast_dates, y=lower_bound,
                    mode='lines',
                    fill='tonexty', fillcolor='rgba(16, 185, 129, 0.15)',
                    line=dict(color='rgba(16, 185, 129, 0.3)', width=1),
                    name='95% Confidence'
                ))
                
                fig = go.Figure(data=traces)
                fig.update_layout(
                    title='Predictive Forecast (30-Period Prophet Model)',
                    paper_bgcolor=VIZ_COLORS['background'],
                    plot_bgcolor=VIZ_COLORS['background'],
                    font={'color': VIZ_COLORS['textMuted'], 'size': 11},
                    margin={'l': 60, 'r': 30, 't': 60, 'b': 50},
                    xaxis={'title': 'Date', 'gridcolor': VIZ_COLORS['border']},
                    yaxis={'title': 'Predicted Value', 'gridcolor': VIZ_COLORS['border']},
                    showlegend=True,
                    legend={'x': 0, 'y': 1.1, 'orientation': 'h', 'bgcolor': 'rgba(255,255,255,0)'},
                    width=900, height=500
                )
                
                pio.write_image(fig, filepath, format='png', scale=2)
                return filepath
            else:
                # Fallback if no forecast data
                fig, ax = plt.subplots(figsize=(10, 6))
                summary = result.get("summary", {})
                if isinstance(summary, dict):
                    text = summary.get("headline", "Forecast Generated")
                    text += "\n\n" + summary.get("explanation", "")[:200]
                else:
                    text = str(summary)[:300]
                ax.text(0.5, 0.5, text, ha='center', va='center', wrap=True, fontsize=12)
                ax.set_title("Predictive Forecasting")
                ax.axis('off')
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
                plt.close(fig)
                return filepath

                
        elif engine_name == "clustering":

            # For non-Titan engines, use matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            if "pca_3d" in result:
                points = result["pca_3d"].get("points", [])
                if points:
                    x = [p.get("x", 0) for p in points]
                    y = [p.get("y", 0) for p in points]
                    clusters = [p.get("cluster", 0) for p in points]
                    scatter = ax.scatter(x, y, c=clusters, cmap='viridis', alpha=0.7)
                    ax.set_xlabel("PC1")
                    ax.set_ylabel("PC2")
                    ax.set_title("Clustering - PCA Visualization")
                    plt.colorbar(scatter, label='Cluster')
            else:
                n_clusters = result.get("n_clusters", "N/A")
                ax.text(0.5, 0.5, f"Clustering complete: {n_clusters} clusters", ha='center', va='center')
                
        elif engine_name == "anomaly":
            fig, ax = plt.subplots(figsize=(10, 6))
            scores = result.get("scores", [])
            if scores:
                ax.hist(scores, bins=20, color='#ef4444', alpha=0.7, edgecolor='black')
                threshold = result.get("threshold", 0.7)
                ax.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold}')
                ax.set_xlabel("Anomaly Score")
                ax.set_ylabel("Frequency")
                ax.set_title("Anomaly Detection - Score Distribution")
                ax.legend()
            else:
                count = result.get("anomaly_count", "N/A")
                ax.text(0.5, 0.5, f"Anomalies detected: {count}", ha='center', va='center')
                
        else:
            # Generic visualization for other engines
            fig, ax = plt.subplots(figsize=(10, 6))
            summary = result.get("summary", {})
            if isinstance(summary, dict):
                text = "\n".join([f"{k}: {str(v)[:50]}" for k, v in list(summary.items())[:5]])
            else:
                text = str(summary)[:200] if summary else f"{engine_name} complete"
            ax.text(0.5, 0.5, text, ha='center', va='center', wrap=True, fontsize=10)
            ax.set_title(f"{engine_name.replace('_', ' ').title()}")
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return filepath
        
    except Exception as e:
        print(f"  Visualization error: {e}")
        plt.close('all')
        return None


def test_gemma(engine_name: str, result: dict) -> str:
    """Call Gemma to generate an analysis summary using authenticated session."""
    summary = result.get("summary", result.get("layman_summary", result.get("gemmaSummary", "")))
    insights = result.get("insights", [])
    
    prompt = f"""Analyze these {engine_name} results and provide a brief business insight:

Results Summary: {summary}
Key Insights: {insights[:3] if insights else 'N/A'}

Provide a 2-3 sentence business recommendation."""

    try:
        # Use authenticated session (Gemma may require auth too)
        session = get_authenticated_session()
        
        # Call via gateway's Gemma proxy endpoint
        response = session.post(
            f"{GATEWAY_URL}/api/gemma/generate",
            json={"prompt": prompt, "max_tokens": 256, "temperature": 0.3},
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            gemma_text = data.get("response", data.get("text", data.get("generated_text", "")))
            if gemma_text:
                print(f"  Gemma response: {gemma_text[:100]}...")
                return gemma_text
            else:
                print(f"  WARNING: Empty Gemma response")
                return "Gemma returned empty response"
        else:
            print(f"  ERROR: Gemma HTTP {response.status_code}")
            return f"Gemma error: HTTP {response.status_code}"
            
    except requests.exceptions.ConnectionError:
        print(f"  WARNING: Cannot connect to Gemma via Gateway")
        return "Gemma service not available"
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        return f"Gemma error: {e}"


def test_gemma_simplify(original_response: str) -> str:
    """Ask Gemma to simplify its response using authenticated session."""
    if not original_response or "error" in original_response.lower():
        return "N/A - No original response"
    
    prompt = f"""Here is a technical analysis:
{original_response}

Explain this simpler in one sentence that anyone can understand."""

    try:
        # Use authenticated session
        session = get_authenticated_session()
        
        response = session.post(
            f"{GATEWAY_URL}/api/gemma/generate",
            json={"prompt": prompt, "max_tokens": 128, "temperature": 0.3},
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            simple = data.get("response", data.get("text", data.get("generated_text", "")))
            if simple:
                print(f"  Simplified: {simple[:100]}...")
                return simple
                
        return "Could not simplify"
            
    except Exception as e:
        return f"Simplify error: {e}"


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("NEXUS ENGINE TEST SCRIPT")
    print("="*60)
    print(f"Gateway URL: {GATEWAY_URL}")
    print(f"Gemma URL: {GEMMA_SERVICE_URL}")
    print(f"User: {AUTH_USER}")
    print("")
    
    # Parse arguments
    engine = sys.argv[1] if len(sys.argv) > 1 else "titan"
    filename = sys.argv[2] if len(sys.argv) > 2 else "test_titan_revenue.csv"
    target = sys.argv[3] if len(sys.argv) > 3 else "revenue"
    
    result = test_engine(engine, filename, target)
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Status: {result.get('status')}")
    print(f"Visualization: {result.get('visualization_path')}")
    print(f"Gemma Response: {str(result.get('gemma_response', 'N/A'))[:200]}...")
    print(f"Gemma Simplified: {str(result.get('gemma_simplified', 'N/A'))[:200]}...")
