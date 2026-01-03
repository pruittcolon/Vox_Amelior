import os
import sys
import time
import json
import pandas as pd
import glob
from llama_cpp import Llama

# Configuration
# Adjust this path if necessary based on where the CSVs are found
DATASET_DIR = "/home/pruittcolon/Desktop/Nemo_Server/docker/gateway_instance/uploads"
TEST_DIR = "/home/pruittcolon/Desktop/Nemo_Server/test_gemma"
MODEL_PATH = "/home/pruittcolon/Desktop/Nemo_Server/models/gemma-3-4b-it-UD-Q4_K_XL.gguf"
OUTPUT_FILE = os.path.join(TEST_DIR, "gemma3_benchmark_results.md")

def load_gemma3():
    print(f"Loading Gemma 3 4B from {MODEL_PATH}...")
    try:
        # Load with 4k context, offloading layers if possible (though -1 usually handles it)
        llm = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=-1,
            n_ctx=4096,
            verbose=False
        )
        return llm
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def test_dataset(llm, filepath):
    filename = os.path.basename(filepath)
    print(f"\nProcessing {filename}...")
    
    try:
        df = pd.read_csv(filepath)
        # Limit columns for context window
        cols_to_use = df.columns[:50]
        
        columns_info = []
        for col in cols_to_use:
            dtype = str(df[col].dtype)
            columns_info.append(f"- {col} (type={dtype})")
            
        columns_str = "\n".join(columns_info)
        
        # System instructions for the chat model
        system_content = (
            "You are an expert data scientist. Your task is to identify the single best 'target' column "
            "for predictive modeling from the provided list, and identify valid 'features' (predictors). "
            "Exclude IDs, dates, and the target itself from features. "
            "Return JSON only."
        )

        user_content = f"Analyze this dataset '{filename}' and classify its columns.\n\nColumns:\n{columns_str}"
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        start_time = time.time()
        
        # Run inference with JSON constraint
        output = llm.create_chat_completion(
            messages=messages,
            response_format={
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string"},
                        "features": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["target", "features"]
                }
            },
            max_tokens=512,
            temperature=0.0
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        content = output['choices'][0]['message']['content']
        
        try:
            result = json.loads(content)
            target = result.get("target", "N/A")
            feature_count = len(result.get("features", []))
            print(f"  Success: Target='{target}' with {feature_count} features")
        except json.JSONDecodeError:
            print(f"  JSON Error. Raw: {content[:100]}...")
            result = {"target": "JSON_ERROR", "features": [], "raw": content}

        return result, duration
        
    except Exception as e:
        print(f"  Error: {e}")
        return {"error": str(e)}, 0

def main():
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)
        
    print("Starting Gemma 3 4B Benchmark...")
    llm = load_gemma3()
    if not llm:
        sys.exit(1)
        
    # Get 10 random csv files if possible, or just all of them
    files = glob.glob(os.path.join(DATASET_DIR, "*.csv"))
    if not files:
        # Fallback to current dir or check where files are
        print(f"No files found in {DATASET_DIR}, trying recursive search...")
        files = glob.glob(os.path.join("/home/pruittcolon/Desktop/Nemo_Server", "**/*.csv"), recursive=True)
        # Filter out some non-datasets
        files = [f for f in files if "unit1" in f or "test_datasets" in f][:10]
    else:
        files = files[:10]

    print(f"Found {len(files)} datasets to test.")
    
    with open(OUTPUT_FILE, "w") as f:
        f.write("# Gemma 3 4B Benchmark Results\n\n")
        f.write("| Dataset | Detected Target | Time (s) | Feature Count |\n")
        f.write("|---------|-----------------|----------|---------------|\n")
    
    for filepath in files:
        result, duration = test_dataset(llm, filepath)
        
        filename = os.path.basename(filepath)
        target = result.get("target", "N/A")
        feat_count = len(result.get("features", []))
        
        res_entry = f"| {filename} | {target} | {duration:.3f} | {feat_count} |"
        print(res_entry)
        
        with open(OUTPUT_FILE, "a") as f:
            f.write(res_entry + "\n")
            
    print(f"\nResults saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
