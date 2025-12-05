import os
import json
import logging
import threading
import shutil
from pathlib import Path
from typing import List, Dict, Any

# Imports for Training
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch

# Imports for DB
try:
    from pysqlcipher3 import dbapi2 as sqlite3
except ImportError:
    import sqlite3

logger = logging.getLogger(__name__)

class PersonalizationManager:
    def __init__(self, db_path: str, db_key: str, models_path: str):
        self.db_path = db_path
        self.db_key = db_key
        self.models_path = models_path
        self.data_dir = Path("/app/data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data_file = self.data_dir / "minilm_pairs.jsonl"
        self.output_path = Path(models_path) / "finetuned" / "minilm-personal-v1"
        self.base_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.is_running = False
        self.lock = threading.Lock()

    def run_pipeline(self):
        if self.is_running:
            logger.warning("Personalization pipeline already running")
            return {"status": "running", "message": "Already running"}
        
        thread = threading.Thread(target=self._execute_task)
        thread.start()
        return {"status": "started", "message": "Personalization started"}

    def _execute_task(self):
        with self.lock:
            self.is_running = True
            try:
                logger.info("[PERSONALIZATION] Starting export...")
                count = self.export_semantic_pairs()
                logger.info(f"[PERSONALIZATION] Exported {count} pairs")
                
                if count < 10:
                    logger.info("[PERSONALIZATION] Not enough data, using dummy data for demo")
                    self._create_dummy_data()
                
                logger.info("[PERSONALIZATION] Starting training...")
                self.train_model()
                logger.info("[PERSONALIZATION] Training complete")
                
            except Exception as e:
                logger.error(f"[PERSONALIZATION] Failed: {e}", exc_info=True)
            finally:
                self.is_running = False

    def get_db_connection(self):
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"DB not found at {self.db_path}")
            
        conn = sqlite3.connect(self.db_path)
        if self.db_key:
            conn.execute(f"PRAGMA key = '{self.db_key}'")
            conn.execute("PRAGMA cipher = 'aes-256-cbc'")
            conn.execute("PRAGMA kdf_iter = 64000")
        return conn

    def export_semantic_pairs(self) -> int:
        conn = self.get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT transcript_id, speaker, text, start_time
                FROM transcript_segments
                ORDER BY transcript_id, start_time
            """)
            rows = cursor.fetchall()
        except Exception as e:
            logger.error(f"Export query failed: {e}")
            conn.close()
            return 0
        
        conn.close()

        transcripts: Dict[str, List[Any]] = {}
        for r in rows:
            tid, spk, txt, start = r
            if not txt or len(txt.strip()) < 5: 
                continue
            if tid not in transcripts:
                transcripts[tid] = []
            transcripts[tid].append({"speaker": spk, "text": txt, "start": start})

        pairs = []
        for tid, segs in transcripts.items():
            for i in range(len(segs) - 1):
                s1 = segs[i]
                for j in range(i + 1, min(i + 5, len(segs))):
                    s2 = segs[j]
                    if s1['speaker'] == s2['speaker']:
                        # Relaxed time constraint for demo
                        if abs(s2['start'] - s1['start']) < 300:
                            pairs.append([s1['text'], s2['text']])
        
        with open(self.data_file, 'w') as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        
        return len(pairs)

    def _create_dummy_data(self):
        pairs = [
            ["Hello", "Hi there"],
            ["How are you?", "Doing well"],
            ["System check", "All systems nominal"],
            ["Vectorize this", "Embedding generated"]
        ] * 10
        with open(self.data_file, 'w') as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")

    def train_model(self):
        # Force CPU inside thread to be safe
        # But if rag-service runs as PID 1, env vars might persist.
        # We'll pass device='cpu' explicitly.
        
        train_examples = []
        with open(self.data_file, 'r') as f:
            for line in f:
                try:
                    pair = json.loads(line)
                    if len(pair) == 2:
                        train_examples.append(InputExample(texts=pair))
                except:
                    pass
        
        logger.info(f"Training on {len(train_examples)} examples")
        
        model = SentenceTransformer(self.base_model_name, cache_folder=self.models_path, device="cpu")
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
        train_loss = losses.MultipleNegativesRankingLoss(model)
        
        # Ensure output dir exists
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=5,
            output_path=str(self.output_path),
            show_progress_bar=False
        )
