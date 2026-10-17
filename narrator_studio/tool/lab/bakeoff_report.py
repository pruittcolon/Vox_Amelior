#!/usr/bin/env python3
"""Collects the bake-off renders into M4A files to listen to on a phone, and
a results table (how much work each model needs, next to Kokoro's).

  bakeoff_report.py --in DIR --out DIR
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ORDER = ["kokoro", "supertonic", "pocket", "chatterbox", "neutts", "soprano", "vibevoice"]

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--previous", default="", help="the last release's files: keeps engines that did not run now")
ap.add_argument("--asr", default="", help="sherpa-onnx Whisper folder: checks every line says its words")
a = ap.parse_args()


def words(t):
    return re.sub(r"[^a-z0-9' ]+", " ", t.lower().replace("-", " ")).split()


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


asr = None
if a.asr:
    import sherpa_onnx
    import soundfile as sf

    asr = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=glob.glob(os.path.join(a.asr, "*encoder.int8.onnx"))[0],
        decoder=glob.glob(os.path.join(a.asr, "*decoder.int8.onnx"))[0],
        tokens=glob.glob(os.path.join(a.asr, "*tokens.txt"))[0],
        language="en",
        task="transcribe",
        num_threads=4,
    )


def check(r):
    """Word error rate over the six lines (high = mumbles, skips or rambles)."""
    if asr is None:
        return None
    from bakeoff import LINES

    total = 0.0
    for i, (_, text) in enumerate(LINES):
        f = glob.glob(os.path.join(a.inp, "**", f"{r['file']}.line{i}.wav"), recursive=True)
        if not f:
            return None
        x, sr = sf.read(f[0], dtype="float32")
        st = asr.create_stream()
        st.accept_waveform(sr, x)
        asr.decode_stream(st)
        total += wer(text, st.result.text)
    return total / len(LINES)
os.makedirs(a.out, exist_ok=True)

rows = [json.load(open(p)) for p in glob.glob(os.path.join(a.inp, "**", "*.json"), recursive=True)]
for r in rows:
    r["wav"] = glob.glob(os.path.join(a.inp, "**", r["file"] + ".wav"), recursive=True)[0]
    r["wer"] = check(r)
fresh = {r["engine"] for r in rows}
old = os.path.join(a.previous, "results.json")
if a.previous and os.path.exists(old):
    for r in json.load(open(old)):
        m4a = glob.glob(os.path.join(a.previous, f"*-{r['file']}.m4a"))
        if r["engine"] not in fresh and m4a:
            r["wav"] = m4a[0]  # already encoded; ffmpeg re-encodes it unchanged in sound
            rows.append(r)
rows.sort(key=lambda r: (ORDER.index(r["engine"]) if r["engine"] in ORDER else 99, r["label"]))
kokoro = next((r["rtf"] for r in rows if r["engine"] == "kokoro"), None)

lines = [
    "# Voice bake-off",
    "",
    "The same six podcast lines (three speakers) in every model. Listen on your phone and pick by ear.",
    "",
    "Work = seconds of computing per second of speech on a 4-core cloud CPU (lower is faster).",
    "Kokoro plays live on a phone at about 2x real time, so a model needing up to about 2x Kokoro's work "
    "can play live; more than that means export first, then listen.",
    "",
    "Words wrong = how many words Whisper (a speech recogniser) did not hear as written: a few percent is normal; "
    "much more means the model mumbled, skipped or rambled.",
    "",
    "| # | Model | Work (CI) | vs Kokoro | Words wrong | Rate | Notes |",
    "|---|---|---|---|---|---|---|",
]
for i, r in enumerate(rows, 1):
    name = f"{i:02d}-{r['file']}.m4a"
    src = r.pop("wav")
    if src.endswith(".m4a"):
        shutil.copy(src, os.path.join(a.out, name))
    else:
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-c:a", "aac", "-b:a", "160k", os.path.join(a.out, name)],
            check=True,
        )
    vs = f"{r['rtf'] / kokoro:.1f}x" if kokoro else "?"
    w = "?" if r.get("wer") is None else f"{100 * r['wer']:.0f}%"
    lines.append(f"| {i:02d} | {r['label']} (`{name}`) | {r['rtf']:.2f} | {vs} | {w} | {r['sample_rate'] // 1000} kHz | {r['notes']} |")
missing = sorted(set(ORDER) - {r["engine"] for r in rows})
if missing:
    lines += ["", "Did not run this time: " + ", ".join(missing) + " (see the workflow log)."]
open(os.path.join(a.out, "results.md"), "w").write("\n".join(lines) + "\n")
json.dump(rows, open(os.path.join(a.out, "results.json"), "w"), indent=1)
print("\n".join(lines))
summary = os.environ.get("GITHUB_STEP_SUMMARY")
if summary:
    open(summary, "a").write("\n".join(lines) + "\n")
