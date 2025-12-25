import csv
import json
import os
import shutil
import sqlite3
import sys
import time

import requests

# Try to import jwt for token generation (e.g. inside container)
try:
    import datetime

    import jwt

    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

# Configuration
API_URL = os.getenv("ML_SERVICE_URL", "http://localhost:8006")  # Default to internal container port
INGEST_URL = f"{API_URL}/ingest"
SECRET_PATH = "/run/secrets/jwt_secret_primary"
TEMP_DIR = "temp_verification_samples"


def get_auth_headers():
    """Confirms authentication headers based on environment."""
    token = os.getenv("AUTH_TOKEN")

    # 1. Try explicit Env Var
    if token:
        print("🔑 Using AUTH_TOKEN from environment.")
        return {"X-Service-Token": token}

    # 2. Try to generate from secret (Container mode)
    if os.path.exists(SECRET_PATH) and JWT_AVAILABLE:
        try:
            with open(SECRET_PATH) as f:
                secret = f.read().strip()

            payload = {
                "sub": "verification-script",
                "aud": "ml-service",
                "iat": datetime.datetime.utcnow(),
                "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=5),
            }
            token = jwt.encode(payload, secret, algorithm="HS256")
            print("🔑 Generated JWT from local secret file (Container Mode).")
            return {"X-Service-Token": token}
        except Exception as e:
            print(f"⚠️ Found secret but failed to generate token: {e}")

    # 3. Fallback (might work if auth is disabled or testing Gateway with cookie?)
    print("⚠️ No auth token found. Request might fail with 401.")
    return {}


def create_sample_files():
    """Generates sample files for testing."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    files = []

    # 1. CSV
    csv_path = os.path.join(TEMP_DIR, "test_dataset.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "category", "amount", "date"])
        writer.writerow([1, "Sales", 150.50, "2023-01-01"])
        writer.writerow([2, "Marketing", 200.00, "2023-01-02"])
        writer.writerow([3, "IT", 50.25, "2023-01-03"])
    files.append(csv_path)

    # 2. JSON
    json_path = os.path.join(TEMP_DIR, "test_dataset.json")
    data = [
        {"id": 4, "category": "HR", "amount": 120.0, "date": "2023-01-04"},
        {"id": 5, "category": "Legal", "amount": 300.0, "date": "2023-01-05"},
    ]
    with open(json_path, "w") as f:
        json.dump(data, f)
    files.append(json_path)

    # 3. SQLite
    db_path = os.path.join(TEMP_DIR, "test_dataset.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE expenses (id INTEGER, category TEXT, amount REAL, date TEXT)")
    cursor.execute("INSERT INTO expenses VALUES (6, 'Ops', 90.0, '2023-01-06')")
    cursor.execute("INSERT INTO expenses VALUES (7, 'R&D', 500.0, '2023-01-07')")
    conn.commit()
    conn.close()
    files.append(db_path)

    return files


def run_tests():
    print("🚀 Starting Normalization Verification")
    print(f"Target: {INGEST_URL}")

    headers = get_auth_headers()
    files = create_sample_files()
    success_count = 0

    print(f"\n📂 Testing {len(files)} files...")

    for file_path in files:
        filename = os.path.basename(file_path)
        print(f"\n--- Uploading: {filename} ---")

        try:
            with open(file_path, "rb") as f:
                start_time = time.time()
                response = requests.post(
                    INGEST_URL, files={"file": (filename, f, "application/octet-stream")}, headers=headers, timeout=10
                )
                duration = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                print(f"✅ SUCCESS ({duration:.2f}s)")
                print(f"   Profile Filename: {data.get('filename')}")
                print(f"   Rows: {data.get('total_rows')}")
                print(f"   Columns: {data.get('columns')}")
                success_count += 1
            else:
                print(f"❌ FAILED: Status {response.status_code}")
                try:
                    print(f"   Response: {response.json()}")
                except:
                    print(f"   Response: {response.text[:200]}...")

        except requests.exceptions.ConnectionError:
            print(f"❌ CONNECTION ERROR. Is the service running at {API_URL}?")
            print("   If running with Docker from host, allow port mapping or run this script inside container.")
            break
        except Exception as e:
            print(f"❌ ERROR: {e}")

    # Cleanup
    try:
        shutil.rmtree(TEMP_DIR)
        print(f"\n🧹 Cleaned up {TEMP_DIR}")
    except:
        pass

    print(f"\n📊 Summary: {success_count}/{len(files)} Tests Passed")

    if success_count == len(files):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    # Hint for user
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage:")
        print("  python verify_normalization.py")
        print("\nEnvironment Variables:")
        print("  ML_SERVICE_URL: URL of the ML service (default: http://localhost:8006)")
        print("  AUTH_TOKEN: Optional JWT/Token for X-Service-Token header")
        print("\nDocker Usage (Recommended):")
        print("  docker cp verify_normalization.py refactored_ml_service:/tmp/")
        print("  docker exec refactored_ml_service python3 /tmp/verify_normalization.py")
        sys.exit(0)

    run_tests()
