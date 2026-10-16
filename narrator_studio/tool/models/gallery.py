#!/usr/bin/env python3
"""Build the Voice Gallery: short reference clips Pocket TTS can speak with.

Only permissively licensed voices from kyutai/tts-voices are used:
  voice-donations  CC0     228 volunteers who donated their voice for TTS
  vctk             CC-BY   VCTK corpus speakers (UK, US, Irish, Scottish, Indian... accents)
  voice-zero       CC0     LibriVox audiobook narrators
  alba-mackenna    CC-BY   voice-acted characters
  unmute (own)     CC0     Kyutai's own recordings
Non-commercial sets (expresso, ears) are left out.

Each clip is cleaned (the `_enhanced` version when there is one), cut to the
best 10 seconds, resampled to 24 kHz mono 16-bit and tagged with its pitch.

  gallery.py --out DIR      writes clips + gallery.json into DIR
"""

import argparse
import hashlib
import json
import os
import re
import urllib.parse
import urllib.request

import numpy as np
import soundfile as sf

REPO = "kyutai/tts-voices"
RAW = f"https://huggingface.co/{REPO}/resolve/main/"
SR = 24000
CLIP_SECONDS = 10.0

SETS = {
    "voice-zero": {"license": "CC0", "source": "LibriVox narrator (Voice-Zero)"},
    "alba-mackenna": {"license": "CC-BY-4.0", "source": "Voice-acted by Alba MacKenna"},
    "unmute-prod-website": {"license": "CC0", "source": "Kyutai recording"},
    "vctk": {"license": "CC-BY-4.0", "source": "VCTK corpus (University of Edinburgh)"},
    "voice-donations": {"license": "CC0", "source": "Unmute voice donation"},
}
# Kyutai's own recordings in unmute-prod-website (the others there are not CC0).
UNMUTE_OWN = {"default_voice.wav", "developer-1.mp3"}

FRIENDLY = {
    "voice-zero/bill_boerst.wav": "Bill (LibriVox narrator)",
    "voice-zero/caro_davy.wav": "Caro (LibriVox narrator)",
    "voice-zero/peter_yearsley.wav": "Peter (LibriVox narrator)",
    "voice-zero/stuart_bell.wav": "Stuart (LibriVox narrator)",
    "alba-mackenna/announcer.wav": "Alba: Announcer",
    "alba-mackenna/merchant.wav": "Alba: Merchant",
    "alba-mackenna/casual.wav": "Alba: Casual",
    "alba-mackenna/a-moment-by.wav": "Alba: Calm",
    "unmute-prod-website/default_voice.wav": "Unmute default",
    "unmute-prod-website/developer-1.mp3": "The Developer",
}


def list_files():
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}") as r:
        info = json.load(r)
    return [s["rfilename"] for s in info["siblings"]]


def pick(files):
    """One file per voice: the enhanced version if it exists."""
    chosen = []
    names = set(files)
    for f in sorted(files):
        top = f.split("/")[0]
        if top not in SETS or not re.search(r"\.(wav|mp3)$", f) or f.endswith("_enhanced.wav"):
            continue
        if top == "unmute-prod-website" and f.split("/", 1)[1] not in UNMUTE_OWN:
            continue
        enhanced = re.sub(r"\.(wav|mp3)$", "_enhanced.wav", f)
        chosen.append((f, enhanced if enhanced in names else f))
    return chosen


def vctk_info():
    """speaker id -> (gender, age, accent, region) from the HF dataset viewer, if reachable."""
    out = {}
    for attempt in range(4):
        _vctk_try(out)
        if out:
            break
        import time

        time.sleep(15 * (attempt + 1))
    print(f"vctk speaker info for {len(out)} speakers")
    return out


def _vctk_try(out):
    try:
        where = urllib.parse.quote("\"text_id\"='023'")
        for offset in range(0, 200, 100):
            url = (
                "https://datasets-server.huggingface.co/filter?dataset=CSTR-Edinburgh/vctk"
                f"&config=main&split=train&where={where}&offset={offset}&length=100"
            )
            with urllib.request.urlopen(url, timeout=60) as r:
                rows = json.load(r).get("rows", [])
            for row in rows:
                d = row["row"]
                out[d["speaker_id"]] = (d.get("gender"), d.get("age"), d.get("accent"), d.get("region"))
            if len(rows) < 100:
                break
    except Exception as e:  # best effort: pitch tags still work
        print("vctk speaker info unavailable:", repr(e))


def resample(x, sr_in, sr_out):
    if sr_in == sr_out:
        return x
    n = int(round(len(x) * sr_out / sr_in))
    t = np.arange(n) * (sr_in / sr_out)
    # Light low-pass first when going down so the clip doesn't alias.
    if sr_out < sr_in:
        k = int(sr_in / sr_out * 2) | 1
        x = np.convolve(x, np.hanning(k) / np.hanning(k).sum(), mode="same")
    return np.interp(t, np.arange(len(x)), x).astype(np.float32)


def best_window(x, seconds):
    """The loudest-on-average window (most speech, fewest gaps)."""
    n = int(seconds * SR)
    if len(x) <= n:
        return x
    energy = np.convolve(x.astype(np.float64) ** 2, np.ones(SR // 10), mode="same")
    csum = np.concatenate([[0], np.cumsum(energy)])
    starts = np.arange(0, len(x) - n, SR // 10)
    scores = csum[starts + n] - csum[starts]
    s = int(starts[np.argmax(scores)])
    return x[s : s + n]


def median_pitch(x):
    """Median F0 of voiced 40 ms frames (autocorrelation), in Hz."""
    frame = int(0.04 * SR)
    f0s = []
    for i in range(0, len(x) - frame, frame):
        w = x[i : i + frame]
        if np.sqrt(np.mean(w**2)) < 0.02:
            continue
        w = w - w.mean()
        ac = np.correlate(w, w, mode="full")[frame - 1 :]
        lo, hi = SR // 400, SR // 60
        seg = ac[lo:hi]
        best = float(np.max(seg))
        if best <= 0.3 * ac[0]:
            continue
        # The first peak close to the best one is the period (the global
        # maximum is often a multiple of it, which would halve the pitch).
        for k in range(1, len(seg) - 1):
            if seg[k] >= 0.9 * best and seg[k] >= seg[k - 1] and seg[k] >= seg[k + 1]:
                f0s.append(SR / (lo + k))
                break
    return float(np.median(f0s)) if f0s else 0.0


def tone(f0):
    if f0 <= 0:
        return "unknown"
    if f0 < 110:
        return "deep"
    if f0 < 150:
        return "low"
    if f0 < 190:
        return "medium"
    return "high"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="for quick tests")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    picks = pick(list_files())
    if a.limit:
        picks = picks[: a.limit]
    vctk = vctk_info()
    voices = []
    for original, use in picks:
        top = original.split("/")[0]
        stem = re.sub(r"\.(wav|mp3)$", "", original.split("/", 1)[1])
        vid = f"{top}-{stem}".replace("_", "-").lower()
        tmp = os.path.join(a.out, "_dl" + os.path.splitext(use)[1])
        try:
            urllib.request.urlretrieve(RAW + use, tmp)
            x, sr = sf.read(tmp, dtype="float32", always_2d=True)
        except Exception as e:
            print("skip", use, repr(e))
            continue
        x = resample(x.mean(axis=1), sr, SR)
        x = best_window(x, CLIP_SECONDS)
        peak = float(np.max(np.abs(x))) or 1.0
        x = (x / peak * 0.9).astype(np.float32)
        if len(x) < SR * 3:
            print("skip (too short)", use)
            continue
        asset = f"voice--{vid}.wav"
        path = os.path.join(a.out, asset)
        sf.write(path, x, SR, subtype="PCM_16")
        f0 = median_pitch(x)
        entry = {
            "id": vid,
            "set": top,
            "name": FRIENDLY.get(original, ""),
            "asset": asset,
            "size": os.path.getsize(path),
            "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
            "seconds": round(len(x) / SR, 1),
            "pitch": round(f0),
            "tone": tone(f0),
            "license": SETS[top]["license"],
            "source": SETS[top]["source"],
        }
        if top == "vctk":
            spk = stem.split("_")[0]
            g = vctk.get(spk)
            entry["name"] = f"VCTK {spk}"
            if g:
                gender, age, accent, region = g
                entry.update({"gender": gender, "age": age, "accent": accent, "region": region})
        if top == "voice-donations":
            entry["name"] = f"Donor {stem[:4]}"
        voices.append(entry)
        print(f"{vid:40s} {entry['seconds']:5}s f0={entry['pitch']:4d} {entry['tone']}", flush=True)
    for leftover in ("_dl.wav", "_dl.mp3"):
        if os.path.exists(os.path.join(a.out, leftover)):
            os.remove(os.path.join(a.out, leftover))
    with open(os.path.join(a.out, "gallery.json"), "w") as f:
        json.dump({"version": 1, "voices": voices}, f, indent=1)
    print(f"{len(voices)} gallery voices")


if __name__ == "__main__":
    main()
