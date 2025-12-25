"""High-level Parakeet transcription pipeline."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import torch
from nemo.collections.asr.models import EncDecCTCModelBPE

from .chunker import ConvertedAudio, convert_audio_to_wav, split_audio_into_chunks
from .diarizer import PyannoteDiarizer, SpeakerSegment, assign_speakers

try:  # Optional dependency
    from .nemo_sortformer import NemoSortformerDiarizer
except Exception:  # pragma: no cover - fallback if NeMo extras missing
    NemoSortformerDiarizer = None

logger = logging.getLogger(__name__)


@dataclass
class ParakeetSegment:
    """Speech segment produced by the Parakeet pipeline."""

    start: float
    end: float
    text: str
    speaker: str | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class ParakeetTranscription:
    """Full transcription result."""

    text: str
    segments: list[ParakeetSegment] = field(default_factory=list)
    duration: float = 0.0


class ParakeetPipeline:
    """Encapsulates the Parakeet-TDT ASR model and optional Pyannote diarization."""

    def __init__(
        self,
        model_id: str = "nvidia/parakeet-tdt-0.6b-v2",
        chunk_duration: int = 300,
        device: str | None = None,
        enable_diarization: bool = True,
        diarization_device: str = "cpu",
        diarization_num_speakers: int | None = None,
    ) -> None:
        self.model_id = model_id
        self.chunk_duration = chunk_duration

        # Respect START_ON_CPU for GPU coordination
        start_on_cpu = os.getenv("START_ON_CPU", "false").lower() == "true"
        if start_on_cpu:
            self.model_device = "cpu"
            logger.info("START_ON_CPU=true, loading Parakeet on CPU for GPU coordination")
        else:
            self.model_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading Parakeet model %s on %s", self.model_id, self.model_device)
        self.model: EncDecCTCModelBPE = EncDecCTCModelBPE.from_pretrained(self.model_id)
        self.model = self.model.to(self.model_device)
        self.model.eval()
        logger.info("Parakeet model loaded successfully")

        self.diarizer_backend = os.getenv("DIARIZER_BACKEND", "pyannote").lower()
        self.diarizer: object | None = None

        # Diarizer should also start on CPU if requested
        diarizer_device = "cpu" if start_on_cpu else (diarization_device or self.model_device)

        if enable_diarization:
            if self.diarizer_backend == "nemo":
                self.diarizer = self._init_nemo_diarizer(diarization_num_speakers, diarizer_device)
            else:
                self.diarizer = self._init_pyannote_diarizer(diarizer_device, diarization_num_speakers)

        if self.diarizer:
            logger.info("Diarization backend enabled: %s", self.diarizer_backend)
        else:
            logger.info("Diarization disabled or unavailable (requested backend=%s)", self.diarizer_backend)

    # ------------------------------------------------------------------#
    # Device management
    # ------------------------------------------------------------------#
    def to(self, device: str) -> None:
        """Move the ASR model and diarizer to the requested device."""
        target = device
        if target == "cuda" and not torch.cuda.is_available():
            target = "cpu"

        if target != self.model_device:
            logger.info("Moving Parakeet model to %s", target)
            self.model = self.model.to(target)
            self.model_device = target

        if self.diarizer and hasattr(self.diarizer, "to"):
            self.diarizer.to(target)

        if target == "cpu":
            # Force proper CUDA memory release
            import gc

            gc.collect()  # Force garbage collection first
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Wait for any pending operations
                torch.cuda.empty_cache()  # Release cached memory
                # Log memory freed
                allocated = torch.cuda.memory_allocated() / 1024**2
                logger.info("VRAM released, current allocation: %.1f MB", allocated)

    # ------------------------------------------------------------------#
    # Direct Audio Transcription (for streaming)
    # ------------------------------------------------------------------#
    def transcribe_audio_array(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> str:
        """
        Transcribe audio array directly using temp file.

        Args:
            audio: numpy array of audio samples (float32, mono)
            sample_rate: audio sample rate (default 16000)

        Returns:
            Transcribed text string
        """
        import tempfile

        import numpy as np
        import soundfile as sf

        if audio is None or len(audio) == 0:
            return ""

        try:
            # Ensure audio is float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Ensure 1D
            if audio.ndim > 1:
                audio = audio.squeeze()

            # Write to temp file - NeMo transcribe expects file paths
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            sf.write(tmp_path, audio, sample_rate)

            try:
                with torch.no_grad():
                    hypotheses = self.model.transcribe([tmp_path])

                if hypotheses and len(hypotheses) > 0:
                    result = hypotheses[0]
                    if isinstance(result, str):
                        return result.strip()
                    elif hasattr(result, "text"):
                        return result.text.strip()
                    return str(result).strip()
                return ""
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        except Exception as exc:
            logger.error("transcribe_audio_array failed: %s", exc)
            return ""

    # ------------------------------------------------------------------#
    # Transcription (file-based)
    def transcribe_file(self, audio_path: str) -> ParakeetTranscription:
        """
        Transcribe an audio file using Parakeet and optional diarization.

        Args:
            audio_path: path to an audio file (any format supported by ffmpeg)
        """
        converted: ConvertedAudio = convert_audio_to_wav(audio_path)
        chunk_paths, duration = split_audio_into_chunks(converted.path, self.chunk_duration)

        full_text_parts: list[str] = []
        segments: list[ParakeetSegment] = []

        try:
            for index, chunk_path in enumerate(chunk_paths):
                offset = index * self.chunk_duration
                chunk_segments = self._transcribe_chunk(chunk_path, offset)
                if not chunk_segments:
                    continue

                full_text_parts.append(" ".join(seg.text for seg in chunk_segments if seg.text))
                segments.extend(chunk_segments)

            if self.diarizer:
                diarized_segments: list[SpeakerSegment] = self.diarizer.diarize(converted.path)
                assign_speakers(segments, diarized_segments)

            # Normalise speaker labels
            for seg in segments:
                seg.speaker = _normalise_speaker_label(seg.speaker)

            full_text = " ".join(part for part in full_text_parts if part).strip()
            return ParakeetTranscription(
                text=full_text,
                segments=segments,
                duration=duration,
            )
        finally:
            # Clean up temporary assets
            for chunk in chunk_paths:
                if chunk != converted.path and os.path.exists(chunk):
                    try:
                        os.unlink(chunk)
                    except OSError:
                        pass
            converted.cleanup()

    def _transcribe_chunk(self, chunk_path: str, offset: float) -> list[ParakeetSegment]:
        """
        Transcribe an individual chunk and return segments with offsets applied.
        """
        try:
            hypotheses = self.model.transcribe(
                [chunk_path],
                return_hypotheses=True,
                timestamps=True,
            )
        except Exception as exc:
            logger.error("Parakeet transcription failed for %s: %s", chunk_path, exc)
            return []

        if not hypotheses:
            return []

        hypothesis = hypotheses[0]
        text = getattr(hypothesis, "text", "") or ""
        text = text.strip()
        segments: list[ParakeetSegment] = []

        timestamp_info = getattr(hypothesis, "timestamp", None)
        segment_entries = []
        if isinstance(timestamp_info, dict):
            segment_entries = timestamp_info.get("segment") or timestamp_info.get("segments") or []

        if segment_entries:
            for entry in segment_entries:
                start = float(entry.get("start", 0.0)) + offset
                end = float(entry.get("end", start)) + offset
                segment_text = entry.get("text") or entry.get("segment") or text
                segments.append(
                    ParakeetSegment(
                        start=start,
                        end=end,
                        text=segment_text.strip(),
                    )
                )
        else:
            words = text.split()
            approx_duration = max(1.0, len(words) / 2.5)
            segments.append(
                ParakeetSegment(
                    start=offset,
                    end=offset + approx_duration,
                    text=text,
                )
            )

        # Ensure segments are ordered and contiguous
        segments.sort(key=lambda seg: seg.start)
        return segments

    # ------------------------------------------------------------------#
    # Diarizer initialisation helpers
    # ------------------------------------------------------------------#
    def _init_pyannote_diarizer(
        self,
        device: str,
        num_speakers: int | None,
    ) -> PyannoteDiarizer | None:
        diarizer = PyannoteDiarizer(
            device=device,
            num_speakers=num_speakers,
        )
        if diarizer and diarizer.pipeline:
            return diarizer
        logger.warning("Pyannote diarization unavailable (missing token or dependency)")
        return None

    def _init_nemo_diarizer(self, num_speakers: int | None, device: str = "cuda") -> object | None:
        if NemoSortformerDiarizer is None:
            logger.warning("NeMo diarizer unavailable (nemo_toolkit not installed)")
            return None

        model_path = os.getenv("SORTFORMER_MODEL")
        if not model_path:
            logger.warning("SORTFORMER_MODEL env var not set; cannot enable NeMo diarizer")
            return None

        max_spks = num_speakers or int(os.getenv("SORTFORMER_MAX_SPKS", "4"))
        # device argument overrides env var if provided (for start_on_cpu support)
        target_device = device or os.getenv("SORTFORMER_DEVICE", "cuda")

        try:
            return NemoSortformerDiarizer(
                nemo_model_path=model_path,
                max_speakers=max_spks,
                device=target_device,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to initialise NeMo diarizer: %s", exc)
            return None


def _normalise_speaker_label(label: str | None) -> str | None:
    if not label:
        return None
    label = label.lower()
    if label.startswith("speaker_"):
        suffix = label.split("_", 1)[1]
        if suffix.startswith("speaker_"):
            suffix = suffix.split("_", 1)[1]
        if suffix.startswith("spk"):
            suffix = suffix.replace("spk", "")
        suffix = suffix.replace("speaker", "")
        suffix = suffix.strip("_")
        try:
            idx = int(suffix)
        except ValueError:
            return f"speaker_{suffix}"
        return f"speaker_{idx}"
    return label
