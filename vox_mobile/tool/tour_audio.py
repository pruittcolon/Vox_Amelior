"""Speech for the on-device tour (CI only).

A short household conversation between two voices (Piper TTS: "ryan" and
"amy"), cut into chunks the way the speech detector cuts them and saved as
16 kHz float32, the format of the listening service's speech queue. The tour
test downloads them from this machine (the emulator reaches it at 10.0.2.2),
drops them in the queue and starts listening, so the real models transcribe,
tell the voices apart and hear the tone, exactly as with the microphone.

Usage: python3 tour_audio.py <voices dir> <out dir>
"""

import json
import os
import sys
import wave

import numpy as np
from piper import PiperVoice
from scipy.signal import resample_poly

# Who says what. Lines in one tuple share a chunk (people talking back to
# back with no pause, which the speaker-change model has to split).
SCRIPT = [
    [("ryan", "Hey, did you see the electric bill that came in the mail today?")],
    [("amy", "I did. It is almost twice what we paid last month, and I honestly do not know how we are going to cover it.")],
    [("ryan", "We could stop eating out for a few weeks. That alone would save us a couple hundred dollars.")],
    [("amy", "That helps, but the rent is due on Friday as well.")],
    [("ryan", "I will call the landlord tomorrow morning and ask if we can pay a few days late.")],
    [("amy", "Thank you. I really hate worrying about money all the time.")],
    [("ryan", "I know. We will figure it out together, like we always do.")],
    [("amy", "By the way, the dog needs his shots at the vet on Tuesday.")],
    [
        ("ryan", "Can you take him? I have a dentist appointment at nine that morning."),
        ("amy", "Sure, I will take him after work, around five."),
    ],
    [("ryan", "Great. And let us go for a long walk this weekend if the weather is nice.")],
    [
        ("amy", "I would love that. Maybe we can try the trail by the lake."),
        ("ryan", "Perfect. I will pack some sandwiches and a thermos of coffee."),
    ],
    [("amy", "Sounds like a plan. I am going to start dinner now.")],
]

RATE = 16000
GAP_IN_CHUNK = 0.35  # seconds between two people in one chunk
GAP_BETWEEN = 0.9  # silence between chunks
PAD = 0.25  # quiet kept around speech, as the speech detector does


def synth(voice, text):
    """Text as 16 kHz float32 samples."""
    path = "/tmp/tour_line.wav"
    with wave.open(path, "wb") as w:
        if hasattr(voice, "synthesize_wav"):
            voice.synthesize_wav(text, w)
        else:  # older piper-tts
            voice.synthesize(text, w)
    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    g = np.gcd(rate, RATE)
    return resample_poly(pcm, RATE // g, rate // g).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * RATE), dtype=np.float32)


def main():
    voices_dir, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    voices = {
        name: PiperVoice.load(os.path.join(voices_dir, f"en_US-{name}-medium.onnx"))
        for name in ("ryan", "amy")
    }
    chunks = []
    at = 0.0
    for i, lines in enumerate(SCRIPT):
        parts = [silence(PAD)]
        for j, (who, text) in enumerate(lines):
            if j:
                parts.append(silence(GAP_IN_CHUNK))
            parts.append(synth(voices[who], text))
        parts.append(silence(PAD))
        audio = np.concatenate(parts)
        # Same level as a phone a metre or two away.
        audio *= 0.5 / max(1e-6, float(np.max(np.abs(audio))))
        name = f"{i:02d}.f32"
        audio.astype("<f4").tofile(os.path.join(out, name))
        seconds = len(audio) / RATE
        chunks.append({
            "file": name,
            "offset_ms": int(at * 1000),
            "duration_ms": int(seconds * 1000),
            "lines": [{"voice": who, "text": text} for who, text in lines],
        })
        at += seconds + GAP_BETWEEN
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump({"rate": RATE, "chunks": chunks}, f, indent=1)
    print(f"{len(chunks)} chunks, {at:.1f} s of conversation")


if __name__ == "__main__":
    main()
