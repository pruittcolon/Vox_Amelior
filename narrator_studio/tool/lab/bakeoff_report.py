#!/usr/bin/env python3
"""Collects the bake-off renders into M4A files to listen to on a phone, and
a results table (how much work each model needs, next to Kokoro's).

  bakeoff_report.py --in DIR --out DIR
"""
import argparse
import glob
import json
import os
import subprocess

ORDER = ["kokoro", "supertonic", "pocket", "chatterbox", "vibevoice", "neutts", "soprano"]

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

rows = [json.load(open(p)) for p in glob.glob(os.path.join(a.inp, "**", "*.json"), recursive=True)]
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
    "| # | Model | Work (CI) | vs Kokoro | Rate | Notes |",
    "|---|---|---|---|---|---|",
]
for i, r in enumerate(rows, 1):
    wav = glob.glob(os.path.join(a.inp, "**", r["file"] + ".wav"), recursive=True)[0]
    name = f"{i:02d}-{r['file']}.m4a"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", wav, "-c:a", "aac", "-b:a", "160k", os.path.join(a.out, name)],
        check=True,
    )
    vs = f"{r['rtf'] / kokoro:.1f}x" if kokoro else "?"
    lines.append(f"| {i:02d} | {r['label']} (`{name}`) | {r['rtf']:.2f} | {vs} | {r['sample_rate'] // 1000} kHz | {r['notes']} |")
missing = sorted(set(ORDER) - {r["engine"] for r in rows})
if missing:
    lines += ["", "Did not run this time: " + ", ".join(missing) + " (see the workflow log)."]
open(os.path.join(a.out, "results.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
summary = os.environ.get("GITHUB_STEP_SUMMARY")
if summary:
    open(summary, "a").write("\n".join(lines) + "\n")
