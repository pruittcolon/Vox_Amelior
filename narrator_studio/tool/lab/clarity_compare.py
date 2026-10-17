#!/usr/bin/env python3
"""Checks the app's Clarity upscaler (Dart, tool/clarity_check.dart) against
the numpy pipeline in voicelab.py, which is itself checked against the
LavaSR-ONNX reference implementation.

  clarity_compare.py prepare --src voice24k.wav --out in48.wav
  clarity_compare.py compare --clarity DIR --in in48.wav --dart out.wav [--min-snr 30]
"""

import argparse
import math
import os
import sys

import numpy as np
import soundfile as sf
from scipy import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def band_energy(x, sr, lo, hi):
    f, _, z = signal.stft(x, sr, nperseg=2048, noverlap=1536)
    p = np.abs(z) ** 2
    return float(p[(f >= lo) & (f < hi)].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "compare"])
    ap.add_argument("--src")
    ap.add_argument("--out")
    ap.add_argument("--clarity")
    ap.add_argument("--in", dest="inp")
    ap.add_argument("--dart")
    ap.add_argument("--min-snr", type=float, default=30.0)
    a = ap.parse_args()

    if a.cmd == "prepare":
        x, sr = sf.read(a.src, dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
        if sr != 24000:
            x = signal.resample_poly(x, 24000, sr).astype(np.float32)
        # Band-limit to 8 kHz, like a Pocket TTS voice, so Clarity has work to do.
        x = signal.resample_poly(signal.resample_poly(x, 2, 3), 3, 2).astype(np.float32)
        x48 = signal.resample_poly(x, 2, 1).astype(np.float32)
        sf.write(a.out, x48, 48000, subtype="FLOAT")
        print(f"prepared {len(x48) / 48000:.1f} s at 48 kHz (content below 8 kHz)")
        return

    import voicelab

    x, sr = sf.read(a.inp, dtype="float32")
    d, dsr = sf.read(a.dart, dtype="float32")
    assert sr == dsr == 48000, (sr, dsr)
    ref, _ = voicelab.Clarity(a.clarity).enhance_v2(x)
    peak = float(np.max(np.abs(ref)))
    if peak > 0.99:
        ref = ref * (0.99 / peak)  # the app keeps the same headroom
    n = min(len(ref), len(d))
    err = ref[:n] - d[:n]
    snr = 10 * math.log10(float(np.sum(ref[:n] ** 2)) / max(float(np.sum(err**2)), 1e-20))
    air_in = band_energy(x, sr, 12500, 24000) / max(band_energy(x, sr, 0, 8000), 1e-20)
    air_out = band_energy(d, sr, 12500, 24000) / max(band_energy(d, sr, 0, 8000), 1e-20)
    low = band_energy(d[:n] - x[:n], sr, 0, 6000) / max(band_energy(x, sr, 0, 6000), 1e-20)
    print(f"Dart vs numpy Clarity: SNR {snr:.1f} dB")
    print(f"energy above 12.5 kHz (re 0-8 kHz): {10 * math.log10(max(air_in, 1e-20)):.1f} dB -> {10 * math.log10(max(air_out, 1e-20)):.1f} dB")
    print(f"change below 6 kHz: {10 * math.log10(max(low, 1e-20)):.1f} dB (the voice itself must stay as it was)")
    ok = snr >= a.min_snr and air_out > air_in * 10 and low < 1e-2
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(
                f"## Clarity check\n\nDart vs numpy SNR **{snr:.1f} dB**; energy above 12.5 kHz "
                f"{10 * math.log10(max(air_in, 1e-20)):.1f} dB -> {10 * math.log10(max(air_out, 1e-20)):.1f} dB; "
                f"change below 6 kHz {10 * math.log10(max(low, 1e-20)):.1f} dB\n"
            )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
