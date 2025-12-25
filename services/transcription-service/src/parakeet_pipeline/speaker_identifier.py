"""Speaker identification using TitaNet embeddings with AS-Norm and Strict VAD."""

from __future__ import annotations

import logging
import os
import tempfile

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


# --- Strict VAD Logic (Copied from Batch Script) ---
def detect_speech_segments(
    audio: np.ndarray, sr: int = 16000, frame_size: float = 0.025, energy_threshold: float = 0.02
):
    frame_samples = int(frame_size * sr)
    hop_samples = frame_samples // 2

    segments = []
    speech_start = None
    min_speech_duration = int(0.3 * sr)
    min_silence_duration = int(0.3 * sr)
    silence_count = 0

    for i in range(0, len(audio) - frame_samples, hop_samples):
        frame = audio[i : i + frame_samples]
        energy = np.sqrt(np.mean(frame**2))
        if energy > energy_threshold:
            silence_count = 0
            if speech_start is None:
                speech_start = i
        else:
            if speech_start is not None:
                silence_count += hop_samples
                if silence_count >= min_silence_duration:
                    end_sample = i - silence_count + hop_samples
                    if end_sample - speech_start >= min_speech_duration:
                        segments.append((speech_start, end_sample))
                    speech_start = None
                    silence_count = 0
    if speech_start is not None:
        end_sample = len(audio)
        if end_sample - speech_start >= min_speech_duration:
            segments.append((speech_start, end_sample))
    return segments


def extract_speech_audio(audio: np.ndarray, sr: int = 16000) -> np.ndarray | None:
    try:
        segments = detect_speech_segments(audio, sr)
        if not segments:
            return None
        speech = np.concatenate([audio[s:e] for s, e in segments])
        # FINAL CHECK: Must have at least 1.0s of speech
        if len(speech) < 1.0 * sr:
            return None
        return speech
    except:
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten()
    b = b.flatten()
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


class SpeakerIdentifier:
    """Identifies speakers using TitaNet + AS-Norm + Strict VAD."""

    COHORT_PATH = "/gateway_instance/enrollment/cohort_embeddings.npy"

    # AS-Norm Threshold (Z-Score)
    # 3.5 means score is 3.5 standard deviations above the mean of the background cohort.
    # This is statistically very significant.
    AS_NORM_THRESHOLD = 3.5

    def __init__(self, enrollment_dir: str, titanet_model=None):
        self.enrollment_dir = enrollment_dir
        self.titanet_model = titanet_model
        self.enrolled_embeddings: dict[str, np.ndarray] = {}
        self.cohort_embeddings: np.ndarray | None = None

        self._load_enrollments()
        self._load_cohort()

    def _load_enrollments(self) -> None:
        if not os.path.exists(self.enrollment_dir):
            return
        for filename in os.listdir(self.enrollment_dir):
            if filename.endswith("_embedding.npy"):
                name = filename.replace("_embedding.npy", "").capitalize()
                # ONLY load trusted names if requested, but folder should be clean now
                filepath = os.path.join(self.enrollment_dir, filename)
                try:
                    emb = np.load(filepath)
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                    self.enrolled_embeddings[name] = emb
                    logger.info(f"Loaded {name}")
                except Exception as e:
                    logger.error(f"Failed {filename}: {e}")

    def _load_cohort(self) -> None:
        if os.path.exists(self.COHORT_PATH):
            try:
                self.cohort_embeddings = np.load(self.COHORT_PATH)
                logger.info(f"Loaded Cohort: {self.cohort_embeddings.shape}")
            except Exception as e:
                logger.error(f"Failed Cohort: {e}")
        else:
            logger.warning("Cohort file not found! AS-Norm disabled.")

    def identify_from_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> tuple[str, float]:
        if not self.enrolled_embeddings or self.titanet_model is None:
            return "Unknown", 0.0

        try:
            # 1. STRICT VAD cleaning
            clean_speech = extract_speech_audio(audio_data, sample_rate)
            if clean_speech is None:
                # logger.debug("Start/end silence filtered or too short")
                return "Unknown", 0.0

            # 2. Extract Embedding
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, clean_speech, sample_rate)
                tmp_path = tmp.name

            embedding = self._extract_embedding(tmp_path)
            os.unlink(tmp_path)

            if embedding is None:
                return "Unknown", 0.0

            # 3. AS-Norm Match
            return self._find_best_match(embedding)

        except Exception as e:
            logger.error(f"ID error: {e}")
            return "Unknown", 0.0

    def identify_from_file(self, audio_path: str) -> tuple[str, float]:
        if not os.path.exists(audio_path):
            return "Unknown", 0.0

        # Here we should also apply VAD if possible, but reading file first
        try:
            audio, sr = sf.read(audio_path)
            return self.identify_from_audio(audio, sr)
        except:
            return "Unknown", 0.0

    def _extract_embedding(self, audio_path: str) -> np.ndarray | None:
        if self.titanet_model is None:
            return None
        try:
            emb = None
            if hasattr(self.titanet_model, "get_embedding"):
                emb = self.titanet_model.get_embedding(audio_path)
            elif hasattr(self.titanet_model, "infer_file"):
                embeddings, _ = self.titanet_model.infer_file(audio_path)
                emb = embeddings[0] if len(embeddings) > 0 else None

            if hasattr(emb, "cpu"):
                emb = emb.cpu().numpy()
            if emb is not None:
                emb = emb.flatten()
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
            return emb
        except:
            return None

    def _find_best_match(self, embedding: np.ndarray) -> tuple[str, float]:
        best_match = "Unknown"
        best_score = -99.9  # Z-score, can be negative

        # Calculate stats against cohort
        if self.cohort_embeddings is not None:
            # Calculate Similarity of (Input vs Cohort)
            # Shapes: (192,) vs (200, 192) -> (200,)
            cohort_sims = np.dot(self.cohort_embeddings, embedding)
            c_mean = np.mean(cohort_sims)
            c_std = np.std(cohort_sims)
        else:
            # Fallback if no cohort
            c_mean = 0.0
            c_std = 1.0

        stats = []
        for name, enrolled_emb in self.enrolled_embeddings.items():
            raw_sim = cosine_similarity(embedding, enrolled_emb)

            # Adaptive Score Normalization
            # "How many standard deviations better than the average person is this?"
            if c_std > 0:
                z_score = (raw_sim - c_mean) / c_std
            else:
                z_score = 0.0

            stats.append(f"{name}: Z={z_score:.2f} (Sim={raw_sim:.2f})")

            if z_score > best_score:
                best_score = z_score
                best_match = name

        logger.info(f"ID Scores: {', '.join(stats)} | Mean={c_mean:.2f}, Std={c_std:.2f}")

        if best_score > self.AS_NORM_THRESHOLD:
            logger.info(f"✅ MATCH: {best_match} (Z={best_score:.2f})")
            return best_match, float(best_score)  # Return Z-score as confidence
        else:
            logger.info(f"❌ Unknown (Best Z={best_score:.2f} < {self.AS_NORM_THRESHOLD})")
            return "Unknown", 0.0

    def get_enrolled_speakers(self) -> list[str]:
        return list(self.enrolled_embeddings.keys())
