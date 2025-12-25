"""
Real-Time Streaming Transcription via WebSocket

Provides a WebSocket endpoint for continuous audio streaming with:
- Immediate ASR transcription (< 1 second latency)
- Deferred diarization (every 30 seconds)
- Real-time n8n voice command triggering
"""

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import WebSocket, WebSocketDisconnect

from .speaker_training_collector import ALLOWED_SPEAKERS as TRAINING_SPEAKERS, get_training_collector

logger = logging.getLogger(__name__)

# Configuration
CHUNK_DURATION_MS = int(os.getenv("STREAMING_CHUNK_MS", "500"))  # Expected chunk size from client
DIARIZATION_INTERVAL_SEC = int(os.getenv("STREAMING_DIARIZATION_INTERVAL", "30"))
MIN_TRANSCRIPTION_SEC = float(os.getenv("STREAMING_MIN_TRANSCRIPTION_SEC", "2.0"))  # Buffer audio for better ASR
OVERLAP_SEC = float(os.getenv("STREAMING_OVERLAP_SEC", "0.5"))  # Overlap for context continuity
SAMPLE_RATE = 16000  # Expected sample rate


class NeMoStreamingASR:
    """
    Real-time streaming ASR using NeMo's official utilities.

    Based on NeMo's FrameBatchASR with proper:
    - Stateful decoding (preserves decoder state between chunks)
    - Feature buffering (4 second context window)
    - Alignment merging (handles chunk boundaries)

    For Parakeet RNNT models, this provides proper real-time streaming.
    """

    def __init__(
        self,
        model,
        frame_len: float = 1.6,  # Duration of each chunk (seconds)
        total_buffer: float = 4.0,  # Total context window (seconds)
        sample_rate: int = 16000,
    ):
        self.model = model
        self.frame_len = frame_len
        self.total_buffer = total_buffer
        self.sr = sample_rate

        # Audio accumulation
        self.audio_buffer: list[np.ndarray] = []
        self.buffer_duration: float = 0.0

        # Frame sizes in samples
        self.n_frame_len = int(frame_len * sample_rate)
        self.n_total_buffer = int(total_buffer * sample_rate)

        # Sliding window buffer for context
        self.context_buffer = np.zeros(self.n_total_buffer, dtype=np.float32)

        # Track previous transcription for incremental output
        self.prev_full_text: str = ""
        self.emitted_text: str = ""

        # Minimum samples before transcribing (1.6s = one frame)
        self.min_samples = self.n_frame_len

        logger.info(f"[NeMoStreamingASR] Initialized: frame={frame_len}s, buffer={total_buffer}s")

    def add_audio(self, audio_data: np.ndarray) -> None:
        """Add audio chunk to accumulator."""
        self.audio_buffer.append(audio_data)
        self.buffer_duration += len(audio_data) / self.sr

    def should_transcribe(self) -> bool:
        """Check if we have enough audio for transcription."""
        return self.buffer_duration >= self.frame_len

    def get_transcription(self) -> str:
        """
        Transcribe accumulated audio using sliding window approach.
        Returns only the NEW text (incremental output).
        """
        if not self.audio_buffer:
            return ""

        # Combine all accumulated audio
        new_audio = np.concatenate(self.audio_buffer)

        # Slide context buffer and add new audio
        if len(new_audio) >= self.n_total_buffer:
            # New audio is longer than buffer - just use newest portion
            self.context_buffer = new_audio[-self.n_total_buffer :].astype(np.float32)
        else:
            # Slide left and append new audio
            shift = len(new_audio)
            self.context_buffer[:-shift] = self.context_buffer[shift:]
            self.context_buffer[-shift:] = new_audio.astype(np.float32)

        # Clear accumulator
        self.audio_buffer.clear()
        self.buffer_duration = 0.0

        # Transcribe the full context buffer
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, self.context_buffer, self.sr)
            try:
                result = self.model.transcribe([tmp.name])
                full_text = result[0].text if hasattr(result[0], "text") else result[0]
            except Exception as e:
                logger.error(f"[NeMoStreamingASR] Transcription error: {e}")
                full_text = ""
            finally:
                os.unlink(tmp.name)

        # Extract new portion using stable prefix property
        new_text = self._extract_new_text(full_text)
        return new_text

    def _extract_new_text(self, full_text: str) -> str:
        """
        Extract only NEW text using RNNT's stable prefix property.
        RNNT produces consistent prefixes - "one two" stays "one two".
        """
        full_text = full_text.strip()

        if not full_text:
            return ""

        if not self.prev_full_text:
            # First transcription
            self.prev_full_text = full_text
            self.emitted_text = full_text
            return full_text

        # Check if new text extends old text (stable prefix)
        prev_lower = self.prev_full_text.lower().strip()
        full_lower = full_text.lower().strip()

        if full_lower.startswith(prev_lower):
            # Perfect case: new text extends old
            new_portion = full_text[len(self.prev_full_text) :].strip()
            if new_portion:
                self.prev_full_text = full_text
                self.emitted_text += " " + new_portion
                return new_portion
            return ""

        # Model revised earlier text - find longest common prefix
        common_len = 0
        min_len = min(len(prev_lower), len(full_lower))
        for i in range(min_len):
            if prev_lower[i] == full_lower[i]:
                common_len = i + 1
            else:
                break

        # Emit text after what we've already emitted
        emitted_lower = self.emitted_text.lower().strip()
        if full_lower.startswith(emitted_lower):
            new_portion = full_text[len(self.emitted_text) :].strip()
        else:
            new_portion = full_text[common_len:].strip()

        self.prev_full_text = full_text
        if new_portion and not self.emitted_text.lower().endswith(new_portion.lower()):
            self.emitted_text += " " + new_portion if self.emitted_text else new_portion
            return new_portion

        return ""

    def reset(self) -> None:
        """Reset state for new session."""
        self.audio_buffer.clear()
        self.buffer_duration = 0.0
        self.context_buffer = np.zeros(self.n_total_buffer, dtype=np.float32)
        self.prev_full_text = ""
        self.emitted_text = ""
        logger.info("[NeMoStreamingASR] State reset")


# Compatibility aliases for existing code
class FrameASR(NeMoStreamingASR):
    """Alias for backwards compatibility."""

    pass


class StreamingASRState:
    """
    Simple streaming ASR state - compatible with existing code.
    Uses a sliding buffer approach for real-time transcription.
    """

    def __init__(self, min_duration: float = 1.6):
        self.min_duration = min_duration
        self.sr = SAMPLE_RATE  # Use module-level sample rate
        self.audio_buffer: list[np.ndarray] = []
        self.buffer_duration: float = 0.0

        # Context buffer for sliding window (4 seconds)
        self.total_buffer = 4.0
        self.n_total_buffer = int(self.total_buffer * self.sr)
        self.context_buffer = np.zeros(self.n_total_buffer, dtype=np.float32)

        # Track previous transcription for incremental output
        self.prev_full_text: str = ""
        self.emitted_text: str = ""

    def add_chunk(self, audio_data: np.ndarray) -> None:
        """Add audio chunk to buffer."""
        self.audio_buffer.append(audio_data)
        self.buffer_duration += len(audio_data) / self.sr

    def should_transcribe(self) -> bool:
        """Check if we have enough audio for transcription."""
        return self.buffer_duration >= self.min_duration

    def get_audio_for_transcription(self) -> tuple:
        """
        Get audio for transcription using sliding window.
        Returns: (audio_array, duration, 0.0)
        """
        if not self.audio_buffer:
            return None, 0.0, 0.0

        # Combine accumulated audio
        new_audio = np.concatenate(self.audio_buffer)
        duration = self.buffer_duration

        # Update sliding context buffer
        if len(new_audio) >= self.n_total_buffer:
            self.context_buffer = new_audio[-self.n_total_buffer :].astype(np.float32)
        else:
            shift = len(new_audio)
            self.context_buffer[:-shift] = self.context_buffer[shift:]
            self.context_buffer[-shift:] = new_audio.astype(np.float32)

        # Clear accumulator
        self.audio_buffer.clear()
        self.buffer_duration = 0.0

        # Return the full context buffer
        return self.context_buffer.copy(), self.total_buffer, 0.0

    def update_context(self, new_text: str) -> str:
        """Extract new portion of text using stable prefix property."""
        full_text = new_text.strip() if new_text else ""

        if not full_text:
            return ""

        if not self.prev_full_text:
            self.prev_full_text = full_text
            self.emitted_text = full_text
            return full_text

        # Check if new text extends old (stable prefix)
        prev_lower = self.prev_full_text.lower().strip()
        full_lower = full_text.lower().strip()

        if full_lower.startswith(prev_lower):
            new_portion = full_text[len(self.prev_full_text) :].strip()
            if new_portion:
                self.prev_full_text = full_text
                self.emitted_text += " " + new_portion
                return new_portion
            return ""

        # Model revised text - find common prefix
        common_len = 0
        for i in range(min(len(prev_lower), len(full_lower))):
            if prev_lower[i] == full_lower[i]:
                common_len = i + 1
            else:
                break

        emitted_lower = self.emitted_text.lower().strip()
        if full_lower.startswith(emitted_lower):
            new_portion = full_text[len(self.emitted_text) :].strip()
        else:
            new_portion = full_text[common_len:].strip()

        self.prev_full_text = full_text
        if new_portion and not self.emitted_text.lower().endswith(new_portion.lower()):
            self.emitted_text += " " + new_portion if self.emitted_text else new_portion
            return new_portion

        return ""

    def reset(self) -> None:
        """Reset state for new session."""
        self.audio_buffer.clear()
        self.buffer_duration = 0.0
        self.context_buffer = np.zeros(self.n_total_buffer, dtype=np.float32)
        self.prev_full_text = ""
        self.emitted_text = ""


@dataclass
class StreamingSession:
    """Tracks state for a single streaming transcription session."""

    session_id: str
    websocket: WebSocket
    start_time: float = field(default_factory=time.time)

    # Audio buffer for diarization (accumulates 30 seconds)
    audio_chunks: list[np.ndarray] = field(default_factory=list)
    total_audio_duration: float = 0.0
    last_diarization_time: float = field(default_factory=time.time)

    # Stateful ASR for real-time streaming with overlapping windows
    asr_state: StreamingASRState | None = None

    # Transcription segments waiting for diarization
    pending_segments: list[dict[str, Any]] = field(default_factory=list)

    # Offset for segment timestamps
    time_offset: float = 0.0

    def __post_init__(self):
        """Initialize stateful ASR after dataclass init."""
        if self.asr_state is None:
            self.asr_state = StreamingASRState()

    def add_audio(self, audio_data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
        """Add audio chunk to diarization buffer."""
        self.audio_chunks.append(audio_data)
        self.total_audio_duration += len(audio_data) / sample_rate

    def get_combined_audio(self) -> np.ndarray:
        """Get all buffered audio as single array."""
        if not self.audio_chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(self.audio_chunks)

    def should_diarize(self) -> bool:
        """Check if we have enough audio for diarization."""
        return self.total_audio_duration >= DIARIZATION_INTERVAL_SEC

    def clear_buffer(self) -> None:
        """Clear the audio buffer after diarization."""
        self.time_offset += self.total_audio_duration
        self.audio_chunks.clear()
        self.total_audio_duration = 0.0
        self.pending_segments.clear()
        self.last_diarization_time = time.time()


class StreamingTranscriber:
    """
    Manages real-time streaming transcription.

    Usage:
        transcriber = StreamingTranscriber(parakeet_pipeline, n8n_callback)
        await transcriber.handle_websocket(websocket)
    """

    def __init__(
        self,
        parakeet_pipeline,  # ParakeetPipeline instance
        n8n_callback=None,  # Async function to send transcripts to n8n
        emotion_callback=None,  # Async function to get emotion for text
        speaker_identifier=None,  # SpeakerIdentifier for named speaker mapping
    ):
        self.parakeet = parakeet_pipeline
        self.n8n_callback = n8n_callback
        self.emotion_callback = emotion_callback
        self.speaker_identifier = speaker_identifier
        self.active_sessions: dict[str, StreamingSession] = {}

    async def handle_websocket(self, websocket: WebSocket) -> None:
        """
        Main WebSocket handler for streaming transcription.

        Protocol:
        - Client sends: binary audio chunks (PCM 16-bit mono 16kHz)
        - Server sends: JSON messages with transcription results
        """
        await websocket.accept()

        session_id = str(uuid.uuid4())
        session = StreamingSession(session_id=session_id, websocket=websocket)
        self.active_sessions[session_id] = session

        logger.info(f"[STREAMING] Session {session_id} started")

        # Send session start confirmation
        await self._send_message(
            websocket,
            {
                "type": "session_start",
                "session_id": session_id,
                "config": {
                    "sample_rate": SAMPLE_RATE,
                    "chunk_duration_ms": CHUNK_DURATION_MS,
                    "diarization_interval_sec": DIARIZATION_INTERVAL_SEC,
                },
            },
        )

        try:
            while True:
                # Receive audio chunk (binary)
                data = await websocket.receive()

                if "bytes" in data:
                    await self._process_audio_chunk(session, data["bytes"])
                elif "text" in data:
                    # Handle control messages
                    await self._handle_control_message(session, data["text"])

        except WebSocketDisconnect:
            logger.info(f"[STREAMING] Session {session_id} disconnected")
        except Exception as e:
            logger.error(f"[STREAMING] Session {session_id} error: {e}")
            await self._send_message(websocket, {"type": "error", "message": str(e)})
        finally:
            # Cleanup: run final diarization if there's buffered audio
            if session.audio_chunks:
                await self._run_diarization(session)

            del self.active_sessions[session_id]
            logger.info(f"[STREAMING] Session {session_id} ended")

    async def _process_audio_chunk(self, session: StreamingSession, audio_bytes: bytes) -> None:
        """Process incoming audio chunk using stateful streaming ASR with overlapping windows."""

        try:
            # Convert bytes to numpy array (assuming 16-bit PCM mono)
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            if len(audio_data) == 0:
                return

            # Add to diarization buffer (30-second window)
            session.add_audio(audio_data)

            # Add to stateful ASR buffer (uses overlapping windows)
            session.asr_state.add_chunk(audio_data)

            # Only transcribe when we have enough audio (2+ seconds)
            if session.asr_state.should_transcribe():
                # Get audio with overlap from previous chunk for context
                buffered_audio, buffer_duration, overlap_duration = session.asr_state.get_audio_for_transcription()

                if buffered_audio is None or len(buffered_audio) == 0:
                    return

                # Calculate timestamp for this segment (excluding overlap)
                chunk_end = session.time_offset + session.total_audio_duration
                chunk_start = chunk_end - buffer_duration

                # Quick transcription using Parakeet (no diarization)
                raw_text = await self._quick_transcribe(buffered_audio)

                if raw_text and raw_text.strip():
                    # Deduplicate overlapping text from previous transcription
                    text = session.asr_state.update_context(raw_text.strip())

                    if text:  # Only send if we have new text after deduplication
                        segment = {
                            "start": chunk_start,
                            "end": chunk_end,
                            "text": text,
                            "speaker": None,  # Will be filled by diarization
                            "emotion": None,
                        }

                        # Add to pending segments for diarization
                        session.pending_segments.append(segment)

                        # Send immediate transcript (no speaker yet)
                        await self._send_message(
                            session.websocket,
                            {
                                "type": "transcript",
                                "segment": segment,
                                "is_final": False,  # Will be updated with diarization
                            },
                        )

                        # Trigger n8n for voice commands (real-time!)
                        if self.n8n_callback:
                            asyncio.create_task(self.n8n_callback(text, session.session_id))

                        logger.info(f"[STREAMING] Raw ASR: '{raw_text}' -> Final: '{text}'")

            # Check if we should run diarization
            if session.should_diarize():
                await self._run_diarization(session)

        except Exception as e:
            logger.error(f"[STREAMING] Error processing chunk: {e}")

    async def _quick_transcribe(self, audio_data: np.ndarray) -> str:
        """
        Fast transcription without diarization.
        Uses direct audio array transcription (no temp file I/O).
        """
        if self.parakeet is None:
            return ""

        try:
            # Use direct audio transcription - no temp files!
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.parakeet.transcribe_audio_array,
                audio_data,
                SAMPLE_RATE,
            )

            return result if result else ""

        except Exception as e:
            logger.error(f"[STREAMING] Quick transcribe error: {e}")
            return ""

    async def _run_diarization(self, session: StreamingSession) -> None:
        """
        Run diarization on buffered audio and update segments with speaker labels.
        """
        if not session.audio_chunks:
            return

        logger.info(
            f"[STREAMING] Running diarization for session {session.session_id} "
            f"({session.total_audio_duration:.1f}s of audio)"
        )

        try:
            combined_audio = session.get_combined_audio()

            if len(combined_audio) == 0:
                return

            # Save to temp file for diarization
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, combined_audio, SAMPLE_RATE)
                tmp_path = tmp.name

            # Run diarization in thread pool
            loop = asyncio.get_event_loop()

            if self.parakeet and self.parakeet.diarizer:
                diarized_segments = await loop.run_in_executor(None, self.parakeet.diarizer.diarize, tmp_path)

                # Match pending segments with diarization results
                enriched_segments = self._assign_speakers(
                    session.pending_segments, diarized_segments, session.time_offset
                )

                # Identify speakers by name using enrolled embeddings (Pruitt, Ericah, etc.)
                if self.speaker_identifier:
                    await self._identify_speakers_in_segments(enriched_segments, combined_audio, session.time_offset)

                # Send diarization update immediately (don't wait for emotion analysis)
                await self._send_message(
                    session.websocket,
                    {
                        "type": "diarization_update",
                        "segments": enriched_segments,
                        "buffer_start": session.time_offset,
                        "buffer_end": session.time_offset + session.total_audio_duration,
                    },
                )

                logger.info(f"[STREAMING] Diarization complete: {len(enriched_segments)} segments")

                # Run emotion analysis in background (fire-and-forget) to not block
                if self.emotion_callback:
                    asyncio.create_task(self._add_emotions_async(session, enriched_segments))

            # Cleanup
            import os

            os.unlink(tmp_path)

            # Clear buffer for next batch
            session.clear_buffer()

        except Exception as e:
            logger.error(f"[STREAMING] Diarization error: {e}")

    async def _add_emotions_async(self, session: StreamingSession, segments: list[dict[str, Any]]) -> None:
        """Add emotion analysis to segments in background (fire-and-forget)."""
        try:
            for seg in segments:
                if seg.get("text") and self.emotion_callback:
                    try:
                        seg["emotion"] = await self.emotion_callback(seg["text"])
                    except Exception as e:
                        logger.debug(f"[STREAMING] Emotion callback failed: {e}")

            # Optionally send emotion update to client
            # (uncomment if you want emotions sent separately)
            # await self._send_message(session.websocket, {
            #     "type": "emotion_update",
            #     "segments": segments,
            # })
        except Exception as e:
            logger.debug(f"[STREAMING] Background emotion task failed: {e}")

    async def _identify_speakers_in_segments(
        self,
        segments: list[dict[str, Any]],
        full_audio: np.ndarray,
        time_offset: float,
    ) -> None:
        """
        Identify speakers in segments using enrolled embeddings.

        APPROACH: Aggregate all audio from the same diarized speaker (speaker_0, speaker_1)
        before identification. This creates longer audio samples for more reliable embeddings.
        """
        if not self.speaker_identifier:
            logger.debug("[STREAMING] Speaker identifier not available")
            return

        logger.info(f"[STREAMING] Starting speaker identification for {len(segments)} segments")

        try:
            loop = asyncio.get_event_loop()

            # Step 1: Group audio slices by generic speaker ID (speaker_0, speaker_1)
            speaker_audio: dict[str, list[np.ndarray]] = {}

            for seg in segments:
                generic_speaker = seg.get("speaker", "unknown")
                if generic_speaker == "unknown" or not generic_speaker:
                    continue

                # Extract audio slice for this segment
                seg_start = seg.get("start", 0) - time_offset
                seg_end = seg.get("end", 0) - time_offset

                # Convert to sample indices
                start_sample = max(0, int(seg_start * SAMPLE_RATE))
                end_sample = min(len(full_audio), int(seg_end * SAMPLE_RATE))

                if end_sample <= start_sample:
                    continue

                audio_slice = full_audio[start_sample:end_sample]

                if generic_speaker not in speaker_audio:
                    speaker_audio[generic_speaker] = []
                speaker_audio[generic_speaker].append(audio_slice)

            logger.info(f"[STREAMING] Aggregated audio for {len(speaker_audio)} speakers: {list(speaker_audio.keys())}")

            # Step 2: Identify each aggregated speaker
            speaker_names: dict[str, tuple[str, float]] = {}  # generic_id -> (name, confidence)

            for generic_id, audio_slices in speaker_audio.items():
                # Concatenate all audio slices for this speaker
                combined_audio = np.concatenate(audio_slices)
                duration = len(combined_audio) / SAMPLE_RATE

                # Require at least 2 seconds of combined audio for reliable identification
                if duration < 2.0:
                    logger.info(
                        f"[STREAMING] Speaker {generic_id}: only {duration:.1f}s audio, skipping identification"
                    )
                    continue

                logger.info(
                    f"[STREAMING] Identifying {generic_id} from {duration:.1f}s combined audio ({len(audio_slices)} segments)"
                )

                # DEBUG: Save live audio for analysis
                import os

                debug_dir = "/gateway_instance/training_data/debug"
                os.makedirs(debug_dir, exist_ok=True)
                debug_path = os.path.join(debug_dir, f"live_{generic_id}_{int(time.time())}.wav")
                try:
                    sf.write(debug_path, combined_audio, SAMPLE_RATE)
                    logger.info(f"[STREAMING] DEBUG: Saved live audio to {debug_path}")
                except Exception as e:
                    logger.warning(f"[STREAMING] DEBUG: Failed to save: {e}")

                # Identify speaker from combined audio (run in thread pool)
                speaker_name, confidence = await loop.run_in_executor(
                    None, self.speaker_identifier.identify_from_audio, combined_audio, SAMPLE_RATE
                )

                if speaker_name != "Unknown":
                    speaker_names[generic_id] = (speaker_name, confidence)
                    logger.info(
                        f"[STREAMING] ✅ {generic_id} identified as: {speaker_name} (confidence: {confidence:.2f})"
                    )

                    # Save audio segment for training if high-confidence and allowed speaker
                    if speaker_name.lower() in TRAINING_SPEAKERS and confidence >= 0.7:
                        training_collector = get_training_collector()
                        # Get text from segments for this speaker
                        segment_texts = [seg.get("text", "") for seg in segments if seg.get("speaker") == generic_id]
                        combined_text = " ".join(segment_texts)

                        asyncio.create_task(
                            training_collector.save_segment(
                                speaker=speaker_name,
                                audio_data=combined_audio,
                                sample_rate=SAMPLE_RATE,
                                confidence=confidence,
                                text=combined_text,
                                session_id=segments[0].get("start", 0) if segments else 0,
                            )
                        )
                        logger.info(f"[STREAMING] 📁 Queued training data save for {speaker_name}")
                else:
                    logger.info(f"[STREAMING] ❌ {generic_id}: no match found")

            # Step 3: Apply identified names to all segments from that speaker
            for seg in segments:
                generic_id = seg.get("speaker")
                if generic_id in speaker_names:
                    name, conf = speaker_names[generic_id]
                    seg["speaker"] = name
                    seg["speaker_confidence"] = conf

        except Exception as e:
            logger.error(f"[STREAMING] Speaker identification failed: {e}")
            import traceback

            logger.debug(traceback.format_exc())

    def _assign_speakers(
        self, transcript_segments: list[dict[str, Any]], diarized_segments: list, time_offset: float
    ) -> list[dict[str, Any]]:
        """
        Assign speaker labels to transcript segments based on diarization overlap.
        """
        result = []

        for seg in transcript_segments:
            seg_start = seg["start"] - time_offset
            seg_end = seg["end"] - time_offset

            best_speaker = None
            best_overlap = 0.0

            for diar_seg in diarized_segments:
                diar_start = getattr(diar_seg, "start", 0)
                diar_end = getattr(diar_seg, "end", 0)
                diar_speaker = getattr(diar_seg, "speaker", "unknown")

                # Calculate overlap
                overlap_start = max(seg_start, diar_start)
                overlap_end = min(seg_end, diar_end)
                overlap = max(0, overlap_end - overlap_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = diar_speaker

            enriched = seg.copy()
            enriched["speaker"] = best_speaker or "unknown"
            result.append(enriched)

        return result

    async def _handle_control_message(self, session: StreamingSession, message: str) -> None:
        """Handle control messages from client."""
        import json

        try:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "force_diarize":
                # Client requests immediate diarization
                await self._run_diarization(session)

            elif msg_type == "ping":
                await self._send_message(session.websocket, {"type": "pong"})

            elif msg_type == "end_session":
                # Graceful session end
                if session.audio_chunks:
                    await self._run_diarization(session)
                await session.websocket.close()

        except json.JSONDecodeError:
            logger.warning(f"[STREAMING] Invalid control message: {message}")

    async def _send_message(self, websocket: WebSocket, data: dict[str, Any]) -> None:
        """Send JSON message to client."""
        import json

        try:
            await websocket.send_text(json.dumps(data))
        except Exception as e:
            logger.error(f"[STREAMING] Failed to send message: {e}")


# Global streaming transcriber instance
_streaming_transcriber: StreamingTranscriber | None = None


def get_streaming_transcriber() -> StreamingTranscriber | None:
    """Get the global streaming transcriber instance."""
    return _streaming_transcriber


def init_streaming_transcriber(
    parakeet_pipeline,
    n8n_callback=None,
    emotion_callback=None,
    speaker_identifier=None,
) -> StreamingTranscriber:
    """Initialize the global streaming transcriber."""
    global _streaming_transcriber
    _streaming_transcriber = StreamingTranscriber(
        parakeet_pipeline=parakeet_pipeline,
        n8n_callback=n8n_callback,
        emotion_callback=emotion_callback,
        speaker_identifier=speaker_identifier,
    )
    logger.info("[STREAMING] Streaming transcriber initialized")
    return _streaming_transcriber
