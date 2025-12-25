import csv
import json
import os
import shutil
import sqlite3

import requests

# API Config
API_URL = "http://localhost:8000"
INGEST_URL = f"{API_URL}/ingest"


def create_sample_files():
    samples_dir = "temp_samples_standalone"
    os.makedirs(samples_dir, exist_ok=True)
    files = []

    # 1. CSV
    csv_path = os.path.join(samples_dir, "test.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name", "value"])
        writer.writerow([1, "Item 1", 10.5])
        writer.writerow([2, "Item 2", 20.0])
    files.append(csv_path)

    # 2. JSON
    json_path = os.path.join(samples_dir, "test.json")
    data = [{"id": 1, "name": "Item 1", "value": 10.5}, {"id": 2, "name": "Item 2", "value": 20.0}]
    with open(json_path, "w") as f:
        json.dump(data, f)
    files.append(json_path)

    # 3. SQLite
    db_path = os.path.join(samples_dir, "test.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE items (id INTEGER, name TEXT, value REAL)")
    cursor.execute("INSERT INTO items VALUES (1, 'Item 1', 10.5)")
    cursor.execute("INSERT INTO items VALUES (2, 'Item 2', 20.0)")
    conn.commit()
    conn.close()
    files.append(db_path)

    return files, samples_dir


def test_ingest():
    print(f"Checking API connectivity at {API_URL}...")
    try:
        # Check health or just docs to verify connectivity
        requests.get(f"{API_URL}/docs", timeout=5)
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to {API_URL}. Is the ML service running?")
        return

    files, samples_dir = create_sample_files()
    success_count = 0

    print(f"\nTesting ingestion for {len(files)} files...")

    for file_path in files:
        filename = os.path.basename(file_path)
        print(f"--> Uploading {filename}...")

        try:
            with open(file_path, "rb") as f:
                response = requests.post(INGEST_URL, files={"file": (filename, f, "application/octet-stream")})

            if response.status_code == 200:
                print(f"✅ Success for {filename}")
                data = response.json()
                print(f"   Profile Filename: {data.get('filename')}")
                print(f"   Detected Columns: {data.get('columns')}")
                success_count += 1
            else:
                print(f"❌ Failed for {filename}: {response.status_code}")
                try:
                    print(f"   Detail: {response.json()}")
                except:
                    print(f"   Response: {response.text[:200]}")
        except Exception as e:
            print(f"❌ Error uploading {filename}: {e}")

    # Cleanup
    try:
        shutil.rmtree(samples_dir)
        print("\nCleaned up temp files.")
    except:
        pass

    print("\nFor full verification, including Excel/Parquet, manual UI testing is recommended.")
    print(f"Standalone Test Summary: {success_count}/{len(files)} passed.")


if __name__ == "__main__":
    test_ingest()
