#!/usr/bin/env python3
"""Checks that ffprobe sees the chapters the app wrote (titles and starts).

  ffprobe_chapters.py --m4a FILE --chapters FILE.json [--decode]
"""
import argparse
import json
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--m4a", required=True)
ap.add_argument("--chapters", required=True)
ap.add_argument("--decode", action="store_true", help="also decode the audio to the end")
a = ap.parse_args()

want = json.load(open(a.chapters))
out = subprocess.run(
    ["ffprobe", "-v", "error", "-show_chapters", "-show_format", "-of", "json", a.m4a],
    check=True, capture_output=True, text=True,
).stdout
got = json.loads(out)
chapters = got.get("chapters", [])
print(f"{a.m4a}: ffprobe sees {len(chapters)} chapters, {float(got['format']['duration']):.1f} s")
ok = len(chapters) == len(want)
for w, c in zip(want, chapters):
    start = float(c["start_time"])
    title = c.get("tags", {}).get("title", "")
    print(f"  {start:8.3f} s  {title}")
    if title != w["title"] or abs(start - w["ms"] / 1000) > 0.01:
        print(f"  expected {w['ms'] / 1000:.3f} s  {w['title']}")
        ok = False
if a.decode:
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", a.m4a, "-map", "0:a", "-f", "null", "-"], capture_output=True, text=True)
    if r.returncode != 0 or r.stderr.strip():
        print("audio decode problem:", r.stderr.strip())
        ok = False
    else:
        print("  audio decodes cleanly")
sys.exit(0 if ok else 1)
