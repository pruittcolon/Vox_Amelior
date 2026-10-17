#!/usr/bin/env python3
"""Listens to a rendered episode with Whisper and compares it with the script.

  tool/listen.py --wav demo.wav --text demo.wav.txt --asr WHISPER_DIR [--max-wer 0.2]

Long audio is cut at the quietest moment between 18 and 26 seconds, so no
word is split, then each piece is transcribed.
"""

import argparse
import os
import re
import sys

import numpy as np
import sherpa_onnx
import soundfile as sf


def words(s):
    return re.sub(r"[^a-z0-9' ]", " ", s.lower().replace("-", " ")).split()


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


def pieces(x, sr):
    start = 0
    win = sr // 10
    while start < len(x):
        if len(x) - start <= 26 * sr:
            yield x[start:]
            return
        lo, hi = start + 18 * sr, start + 26 * sr
        seg = x[lo:hi]
        energy = np.convolve(seg.astype(np.float64) ** 2, np.ones(win), mode="valid")
        cut = lo + int(np.argmin(energy)) + win // 2
        yield x[start:cut]
        start = cut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--asr", required=True)
    ap.add_argument("--max-wer", type=float, default=0.2)
    a = ap.parse_args()

    def find(suffix):
        return next(os.path.join(a.asr, f) for f in sorted(os.listdir(a.asr)) if f.endswith(suffix))

    rec = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=find("encoder.int8.onnx"),
        decoder=find("decoder.int8.onnx"),
        tokens=find("tokens.txt"),
        language="en",
        task="transcribe",
        num_threads=4,
    )
    x, sr = sf.read(a.wav, dtype="float32")
    heard = []
    for p in pieces(x, sr):
        s = rec.create_stream()
        s.accept_waveform(sr, p)
        rec.decode_stream(s)
        heard.append(s.result.text.strip())
    hyp = " ".join(heard)
    ref = open(a.text).read()
    score = wer(ref, hyp)
    report = f"Whisper heard {len(words(hyp))} words of {len(words(ref))}; word error rate {score:.1%}"
    print(report)
    print("\nHEARD:\n" + "\n".join(heard))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(f"## Sample podcast listening test\n\n{report} (limit {a.max_wer:.0%}).\n")
    if score > a.max_wer:
        sys.exit(f"Too many words misheard ({score:.1%} > {a.max_wer:.0%})")


if __name__ == "__main__":
    main()
