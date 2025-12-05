
import requests
import json
import sys
import os
import time

# Configuration
GATEWAY_URL = "http://localhost:8000"
# Direct fallback if gateway auth fails or is tricky
ML_SERVICE_URL = "http://localhost:8006" 

def print_step(step_num, description):

    print(f"\n{'='*60}", flush=True)

    print(f"STEP {step_num}: {description}", flush=True)

    print(f"{'='*60}", flush=True)



# Helper for requests with timeout

def safe_post(url, **kwargs):

    try:

        return requests.post(url, timeout=30, **kwargs)

    except requests.exceptions.Timeout:

        print(f"❌ Request timed out: {url}", flush=True)

        raise

    except Exception as e:

        print(f"❌ Request failed: {e}", flush=True)

        raise

def test_vectorization_flow():
    """
    Simulates the user flow:
    1. Upload database (LLM Vectorization page)
    2. Trigger Vectorization (LLM Vectorization page)
    3. Run Analysis with Vectorization (ML Service page)
    """
    
    filename = "text_prediction_data.csv"
    filepath = os.path.join("data/test_datasets", filename)
    
    if not os.path.exists(filepath):
        print(f"❌ Test file not found: {filepath}")
        return

    # --- Step 1: Upload ---
    print_step(1, f"Uploading {filename}...")
    
    # We'll try the gateway first. If it requires auth we might hit 401.
    # The main.py shows /upload is exempt from auth? 
    # L401: self.exempt_paths = { ... "/upload", "/api/upload" ... }
    # So we can use /api/upload without auth.
    
    url = f"{GATEWAY_URL}/api/upload"
    try:
        with open(filepath, 'rb') as f:
            files = {'file': (filename, f, 'text/csv')}
            response = safe_post(url, files=files)
            
        if response.status_code == 200:
            data = response.json()
            uploaded_filename = data.get('filename', filename)
            print(f"✅ Upload successful: {uploaded_filename}", flush=True)
            print(f"   Rows: {data.get('row_count')}", flush=True)
            print(f"   Columns: {data.get('columns')}", flush=True)
        else:
            print(f"❌ Upload failed: {response.status_code} {response.text}", flush=True)
            return
    except Exception as e:
        print(f"❌ Connection error during upload: {e}", flush=True)
        print("   Ensure services are running (./start.sh)", flush=True)
        return

    # --- Step 2: Vectorize ---
    print_step(2, f"Vectorizing {uploaded_filename}...")
    
    # The endpoint /api/vectorize/{filename} REQUIRES auth (Depends(require_auth))
    
    # Authenticate first
    print("   🔐 Authenticating to get session token...", flush=True)
    login_url = f"{GATEWAY_URL}/api/auth/login"
    session = requests.Session()
    try:
        # Try default creds
        login_resp = session.post(login_url, json={"username": "admin", "password": "password"}, timeout=10) 
        if login_resp.status_code != 200:
             login_resp = session.post(login_url, json={"username": "demo", "password": "demo"}, timeout=10)
        
        if login_resp.status_code == 200:
            print("   ✅ Login successful", flush=True)
        else:
            print(f"   ⚠️ Login failed ({login_resp.status_code}). Trying to hit ML service directly for vectorization...", flush=True)
            # Fallback to direct ML service
            url = f"{ML_SERVICE_URL}/embed/{uploaded_filename}"
            response = safe_post(url)
            if response.status_code == 200:
                print(f"   ✅ Vectorization successful (Direct): {response.json().get('message')}", flush=True)
            else:
                print(f"   ❌ Vectorization failed: {response.status_code} {response.text}", flush=True)
                return
    except Exception as e:
        print(f"   ❌ Auth error: {e}", flush=True)
        return

    # If login worked, try gateway
    if login_resp.status_code == 200:
        url = f"{GATEWAY_URL}/api/vectorize/{uploaded_filename}"
        try:
            response = session.post(url, timeout=30)
            if response.status_code == 200:
                print(f"✅ Vectorization successful: {response.json().get('message')}", flush=True)
            else:
                print(f"❌ Vectorization failed: {response.status_code} {response.text}", flush=True)
                return
        except Exception as e:
            print(f"❌ Vectorization request failed: {e}", flush=True)
            return

    # --- Step 3: Run Analysis with Vectorization ---
    print_step(3, "Running Titan Premium Analysis (with Vectorization)...")
    
    url = f"{GATEWAY_URL}/api/analytics/titan-premium"
    payload = {
        "filename": uploaded_filename,
        "target_column": "satisfaction_score",
        "use_vectorization": True,
        "n_variants": 3 # Keep it fast
    }
    
    print(f"   POST {url}", flush=True)
    # print(f"   Payload: {json.dumps(payload, indent=2)}", flush=True)
    
    try:
        if login_resp.status_code == 200:
            response = session.post(url, json=payload, timeout=120) # Titan can be slow
        else:
            # Direct fallback
            url = f"{ML_SERVICE_URL}/analytics/titan-premium"
            response = safe_post(url, json=payload)
            
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Analysis Successful!", flush=True)
            print(f"   Engine: {result.get('engine_display_name')}", flush=True)
            
            # Handle variants list
            variants = result.get('variants', [])
            best_variant = variants[0] if variants else {}
            
            print(f"   Best Model: {best_variant.get('model_name')}", flush=True)
            print(f"   Score: {best_variant.get('cv_score')}", flush=True)
            
            # Verify vectorization was used
            features = result.get('feature_importance', [])
            vector_features = [f['name'] for f in features if f['name'].startswith('vec_')]
            
            if vector_features:
                print(f"\n   💎 VECTORIZATION CONFIRMED!", flush=True)
                print(f"   Found {len(vector_features)} vector features used in the model:", flush=True)
                print(f"   {vector_features[:5]}...", flush=True)
                
                # Check if vector features are important
                top_features = [f['name'] for f in features[:3]]
                print(f"   Top 3 Drivers: {top_features}", flush=True)
                if any(f.startswith('vec_') for f in top_features):
                    print("   🚀 Vector features are top drivers! It did better!", flush=True)
            else:
                print("\n   ⚠️ No vector features found in results.", flush=True)
                
        else:
            print(f"❌ Analysis failed: {response.status_code} {response.text}", flush=True)

    except Exception as e:
        print(f"❌ Analysis error: {e}", flush=True)
