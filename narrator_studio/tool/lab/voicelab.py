#!/usr/bin/env python3
"""Voice lab: which free on-device voices sound best, and does Clarity help?

Renders one passage with every candidate (Kokoro fp32, Supertonic 3 int8 vs
fp32 at several step counts, Pocket TTS int8 vs fp32, Kitten), runs the
Clarity upscaler (LavaSR bandwidth extension, 24 kHz -> 48 kHz) on the 24 kHz
voices, and scores everything:

  utmos   naturalness predicted by UTMOS22 (1-5, judged at 16 kHz)
  nisqa   overall quality predicted by NISQA (1-5, judged at 48 kHz, so it
          hears the top octave) and its "coloration" dimension
  wer     word error rate of Whisper listening to it
  bw      frequency below which 99% / 99.9% of the speech energy lies
  air     energy above 8 kHz relative to below, in dB
  rtf     seconds of compute per second of audio on this machine

  voicelab.py --models DIR --asr DIR --out DIR [--quick]
"""

import argparse
import html
import json
import math
import os
import re
import time

import numpy as np
import sherpa_onnx
import soundfile as sf
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

TEXT = (
    "Light from the sun looks white, but it is really a mixture of every colour. "
    "When sunlight enters the atmosphere, it bumps into tiny molecules of nitrogen and oxygen. "
    "Blue light has a shorter wavelength, so it is scattered far more than red light. "
    "Isn't that strange? The sky is blue for the same reason sunsets glow orange."
)

THREADS = 4


# ---------------------------------------------------------------------------
# Clarity: LavaSR bandwidth extension, the exact pipeline the app runs.
# Features: log-mel (80 bands, Slaney, filterbank built for 44.1 kHz, up to
# 8 kHz) of the 48 kHz signal; a Vocos backbone + ISTFT head predicts the full
# 48 kHz spectrum; we keep the real voice below `cutoff` and take the predicted
# spectrum above it, crossfading over `width` Hz, then inverse STFT.

N_FFT = 2048
HOP = 512
PAD = (N_FFT - HOP) // 2


def hann(n):
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float32)


def hz_to_mel(f):
    f = np.asarray(f, dtype=np.float64)
    lin = f / (200.0 / 3)
    log = 15.0 + np.log(np.maximum(f, 1e-9) / 1000.0) / (np.log(6.4) / 27.0)
    return np.where(f >= 1000.0, log, lin)


def mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    lin = m * (200.0 / 3)
    log = 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0))
    return np.where(m >= 15.0, log, lin)


def mel_filterbank(sr=44100, n_fft=N_FFT, n_mels=80, fmin=0.0, fmax=8000.0):
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    pts = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        lo = (freqs - pts[i]) / max(pts[i + 1] - pts[i], 1e-12)
        hi = (pts[i + 2] - freqs) / max(pts[i + 2] - pts[i + 1], 1e-12)
        fb[i] = np.maximum(0, np.minimum(lo, hi))
    fb *= (2.0 / (pts[2:] - pts[:-2]))[:, None]
    return fb.astype(np.float32)


def stft(x):
    xp = np.pad(x, (PAD, PAD), mode="reflect")
    frames = sliding_window_view(xp, N_FFT)[::HOP] * hann(N_FFT)
    return np.fft.rfft(frames, axis=1).T  # [1025, T]


def istft(spec, n):
    w = hann(N_FFT)
    frames = np.fft.irfft(spec.T, n=N_FFT, axis=1) * w
    t = frames.shape[0]
    size = (t - 1) * HOP + N_FFT
    y = np.zeros(size, dtype=np.float64)
    env = np.zeros(size, dtype=np.float64)
    for i in range(t):
        y[i * HOP : i * HOP + N_FFT] += frames[i]
        env[i * HOP : i * HOP + N_FFT] += w * w
    y = (y / np.maximum(env, 1e-8))[PAD : PAD + n]
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y.astype(np.float32)


class Clarity:
    def __init__(self, folder):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = THREADS
        self.backbone = ort.InferenceSession(os.path.join(folder, "enhancer_backbone.onnx"), so)
        self.head = ort.InferenceSession(os.path.join(folder, "enhancer_spec_head.onnx"), so)
        self.fb = mel_filterbank()
        for s in (self.backbone, self.head):
            print("clarity io:", [(i.name, i.shape) for i in s.get_inputs()], [(o.name, o.shape) for o in s.get_outputs()])

    def predict(self, x48):
        spec = stft(x48)
        mel = np.log(np.clip(self.fb @ np.abs(spec), 1e-5, None)).astype(np.float32)[None]
        hidden = self.backbone.run(None, {self.backbone.get_inputs()[0].name: mel})[0]
        re_, im_ = self.head.run(None, {self.head.get_inputs()[0].name: hidden})
        return spec, (re_[0] + 1j * im_[0])

    def enhance(self, x48, cutoff=11000.0, width=1000.0):
        """First version: fixed crossover, complex crossfade (leaves a notch)."""
        spec, pred = self.predict(x48)
        f = np.arange(spec.shape[0]) * 48000.0 / N_FFT
        t = np.clip((f - (cutoff - width / 2)) / width, 0, 1)
        m = (t * t * (3 - 2 * t))[:, None]
        return istft(spec * (1 - m) + pred[:, : spec.shape[1]] * m, len(x48))

    def enhance_v2(self, x48, width=800.0, skip_above=10500.0):
        """What the app runs. Only voices whose spectrum stops short (a cliff
        below skip_above, like Pocket TTS at ~7.8 kHz) are touched; the crossover
        ends at the cliff; in the crossfade, magnitudes blend under the voice's
        own phase (no cancellation notch); the added band is level-matched per
        frame to the real voice just below the crossover."""
        edge = voice_edge(stft(x48))
        if edge >= skip_above:
            print(f"clarity2: voice reaches {edge:.0f} Hz, left as it is", flush=True)
            return x48.copy(), edge
        spec, pred = self.predict(x48)
        pred = pred[:, : spec.shape[1]]
        bins, frames = spec.shape
        hz = 48000.0 / N_FFT
        f = np.arange(bins) * hz
        cutoff = edge - width / 2
        lo = cutoff - width / 2
        mag_s, mag_p = np.abs(spec), np.abs(pred)
        band = (f >= lo - 2000) & (f < lo)
        es = (mag_s[band] ** 2).sum(0)
        ep = (mag_p[band] ** 2).sum(0)
        g = np.sqrt(es / np.maximum(ep, 1e-12))
        g = np.where(ep > 1e-12, np.clip(g, 0.5, 2.0), 1.0)
        g = np.convolve(g, np.ones(5) / 5, mode="same")  # smooth over ~50 ms
        t = np.clip((f - lo) / width, 0, 1)
        m = (t * t * (3 - 2 * t))[:, None]
        phase = np.where(mag_s > 1e-9, spec / np.maximum(mag_s, 1e-9), pred / np.maximum(mag_p, 1e-9))
        blend = ((1 - m) * mag_s + m * g[None, :] * mag_p) * phase
        out = np.where(m <= 0, spec, np.where(m >= 1, pred * g[None, :], blend))
        print(f"clarity2: voice ends at {cutoff + width / 2:.0f} Hz, crossover {lo:.0f}-{lo + width:.0f} Hz, gain {g.mean():.2f}", flush=True)
        return istft(out, len(x48)), cutoff


def voice_edge(spec, min_drop_db=15.0):
    """Where the voice's own spectrum falls off a cliff (Hz): the largest drop
    in its average log spectrum over 1 kHz, searched between 5 and 11.8 kHz.
    A voice that only falls away at the 12 kHz limit of 24 kHz audio returns
    12000 (it has its top end; nothing to rebuild)."""
    hz = 48000.0 / N_FFT
    power = (np.abs(spec) ** 2).mean(1)
    level = 10 * np.log10(np.convolve(power, np.ones(5) / 5, mode="same") + 1e-20)
    k_lo, k_hi = int(round(5000 / hz)), int(round(11800 / hz))
    w0, w1 = int(round(100 / hz)), int(round(600 / hz))
    best, edge = 0.0, 12000.0
    for k in range(k_lo, k_hi + 1):
        drop = level[k - w1 : k - w0 + 1].mean() - level[k + w0 : k + w1 + 1].mean()
        if drop > best:
            best, edge = drop, k * hz
    return float(edge) if best >= min_drop_db else 12000.0


def check_against_reference(clarity, ref_dir, x48):
    """Our features must match the LavaSR-ONNX reference implementation."""
    import sys

    sys.path.insert(0, ref_dir)
    import lavasr_core  # noqa: E402

    ours = np.log(np.clip(clarity.fb @ np.abs(stft(x48)), 1e-5, None))
    fe = lavasr_core.MelSpectrogramFrontend(44100, N_FFT, HOP, 80, fmin=0.0, fmax=8000.0)
    theirs = fe(x48[None])[0]
    err = float(np.max(np.abs(ours - theirs)))
    print(f"clarity features vs reference: max abs diff {err:.2e}", flush=True)
    return err


# ---------------------------------------------------------------------------
# Judges.


def words(s):
    return re.sub(r"[^a-z0-9' ]", " ", s.lower().replace("colour", "color")).split()


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


def speech_frames(x, sr):
    f, _, z = signal.stft(x, sr, nperseg=2048, noverlap=1536)
    p = np.abs(z) ** 2
    e = p.sum(0)
    keep = e > e.max() * 1e-3
    return f, p[:, keep]


def bandwidth(x, sr):
    f, p = speech_frames(x, sr)
    spec = p.sum(1)
    cum = np.cumsum(spec) / max(spec.sum(), 1e-20)
    lo = spec[f < 8000].sum()
    hi = spec[f >= 8000].sum()
    return {
        "bw99": int(f[min(np.searchsorted(cum, 0.99), len(f) - 1)]),
        "bw999": int(f[min(np.searchsorted(cum, 0.999), len(f) - 1)]),
        "air": round(10 * math.log10(max(hi, 1e-20) / max(lo, 1e-20)), 1),
    }


class Judges:
    def __init__(self, asr_dir):
        import torch

        torch.set_num_threads(THREADS)
        self.torch = torch
        self.utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
        from torchmetrics.functional.audio.nisqa import non_intrusive_speech_quality_assessment

        self.nisqa = non_intrusive_speech_quality_assessment

        def find(suffix):
            return os.path.join(asr_dir, next(f for f in sorted(os.listdir(asr_dir)) if f.endswith(suffix)))

        self.asr = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=find("encoder.int8.onnx"),
            decoder=find("decoder.int8.onnx"),
            tokens=find("tokens.txt"),
            language="en",
            task="transcribe",
            num_threads=THREADS,
        )

    def score(self, x, sr, text):
        t = self.torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        with self.torch.no_grad():
            u = float(self.utmos(t[None], sr).item())
            n = self.nisqa(t, sr).tolist()
        s = self.asr.create_stream()
        s.accept_waveform(sr, x)
        self.asr.decode_stream(s)
        heard = s.result.text.strip()
        out = {
            "utmos": round(u, 3),
            "nisqa": round(n[0], 3),
            "noisiness": round(n[1], 2),
            "coloration": round(n[3], 2),
            "wer": round(wer(text, heard), 3),
            "heard": heard,
        }
        out.update(bandwidth(x, sr))
        return out


# ---------------------------------------------------------------------------
# Engines.


def first(root, *names):
    for n in names:
        if os.path.exists(os.path.join(root, n)):
            return os.path.join(root, n)
    raise FileNotFoundError(f"none of {names} in {root}")


def kokoro(root):
    return sherpa_onnx.OfflineTts(
        sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                    model=first(root, "model.onnx"),
                    voices=os.path.join(root, "voices.bin"),
                    tokens=os.path.join(root, "tokens.txt"),
                    data_dir=os.path.join(root, "espeak-ng-data"),
                    lexicon=os.path.join(root, "lexicon-us-en.txt"),
                ),
                num_threads=THREADS,
                debug=False,
            ),
            max_num_sentences=1,
        )
    )


def supertonic(root, onnx_root, suffix):
    return sherpa_onnx.OfflineTts(
        sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                supertonic=sherpa_onnx.OfflineTtsSupertonicModelConfig(
                    duration_predictor=os.path.join(onnx_root, f"duration_predictor{suffix}.onnx"),
                    text_encoder=os.path.join(onnx_root, f"text_encoder{suffix}.onnx"),
                    vector_estimator=os.path.join(onnx_root, f"vector_estimator{suffix}.onnx"),
                    vocoder=os.path.join(onnx_root, f"vocoder{suffix}.onnx"),
                    tts_json=os.path.join(root, "tts.json"),
                    unicode_indexer=os.path.join(root, "unicode_indexer.bin"),
                    voice_style=os.path.join(root, "voice.bin"),
                ),
                num_threads=THREADS,
                debug=False,
            )
        )
    )


def pocket(root, int8):
    q = ".int8" if int8 else ""
    return sherpa_onnx.OfflineTts(
        sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                pocket=sherpa_onnx.OfflineTtsPocketModelConfig(
                    lm_flow=first(root, f"lm_flow{q}.onnx"),
                    lm_main=first(root, f"lm_main{q}.onnx"),
                    encoder=first(root, "encoder.onnx"),
                    decoder=first(root, f"decoder{q}.onnx"),
                    text_conditioner=first(root, "text_conditioner.onnx"),
                    vocab_json=os.path.join(root, "vocab.json"),
                    token_scores_json=os.path.join(root, "token_scores.json"),
                ),
                num_threads=THREADS,
                debug=False,
            )
        )
    )


def kitten(root):
    model = next(f for f in sorted(os.listdir(root)) if f.endswith(".onnx"))
    return sherpa_onnx.OfflineTts(
        sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                kitten=sherpa_onnx.OfflineTtsKittenModelConfig(
                    model=os.path.join(root, model),
                    voices=os.path.join(root, "voices.bin"),
                    tokens=os.path.join(root, "tokens.txt"),
                    data_dir=os.path.join(root, "espeak-ng-data"),
                ),
                num_threads=THREADS,
                debug=False,
            ),
            max_num_sentences=1,
        )
    )


def kokoro_names(root):
    import onnx

    m = onnx.load(os.path.join(root, "model.onnx"), load_external_data=False)
    meta = {p.key: p.value for p in m.metadata_props}
    return [n for n in meta.get("speaker_names", "").split(",") if n]


def gen_config(**kw):
    g = sherpa_onnx.GenerationConfig()
    extra = kw.pop("extra", None)
    for k, v in kw.items():
        setattr(g, k, v)
    if extra:
        g.extra = {k: str(v) for k, v in extra.items()}  # a string map in C++
    return g


# ---------------------------------------------------------------------------


class Lab:
    def __init__(self, out, judges, clarity):
        self.out = out
        self.judges = judges
        self.clarity = clarity
        self.rows = []
        os.makedirs(os.path.join(out, "wav"), exist_ok=True)

    def save(self, label, x, sr):
        path = os.path.join("wav", re.sub(r"[^A-Za-z0-9_.-]", "_", label) + ".wav")
        sf.write(os.path.join(self.out, path), x, sr, subtype="PCM_16")
        return path

    def add(self, group, label, fn, clarity_cutoffs=(11000.0,)):
        try:
            t0 = time.time()
            audio = fn()
            took = time.time() - t0
            x = np.asarray(audio.samples, dtype=np.float32)
            sr = audio.sample_rate
        except Exception as e:  # keep going: one broken candidate must not stop the lab
            print("FAILED", label, repr(e), flush=True)
            self.rows.append({"group": group, "label": label, "error": repr(e)})
            return None
        dur = len(x) / sr
        row = {"group": group, "label": label, "sr": sr, "seconds": round(dur, 2), "rtf": round(took / max(dur, 1e-6), 3)}
        row.update(self.judges.score(x, sr, TEXT))
        row["wav"] = self.save(label, x, sr)
        self.rows.append(row)
        print(json.dumps(row), flush=True)
        if self.clarity is not None and sr == 24000:
            x48 = signal.resample_poly(x, 2, 1).astype(np.float32)
            variants = [(f"+clarity{int(c / 1000)}k", lambda c=c: self.clarity.enhance(x48, cutoff=c)) for c in clarity_cutoffs]
            variants.append(("+clarity2", lambda: self.clarity.enhance_v2(x48)[0]))
            for tag, run in variants:
                t0 = time.time()
                y = run()
                took = time.time() - t0
                peak = float(np.max(np.abs(y)))
                if peak > 0.99:
                    y = y * (0.99 / peak)
                lab = f"{label} {tag}"
                r = {
                    "group": group,
                    "label": lab,
                    "sr": 48000,
                    "seconds": row["seconds"],
                    "rtf": round(took / max(dur, 1e-6), 4),
                }
                r.update(self.judges.score(y, 48000, TEXT))
                r["wav"] = self.save(lab, y, 48000)
                self.rows.append(r)
                print(json.dumps(r), flush=True)
        return x, sr


def spectrograms(out, rows, labels, name):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    picked = [r for lab in labels for r in rows if r.get("label") == lab and "wav" in r]
    if not picked:
        return
    fig, axes = plt.subplots(len(picked), 1, figsize=(11, 2.3 * len(picked)), squeeze=False)
    for ax, r in zip(axes[:, 0], picked):
        x, sr = sf.read(os.path.join(out, r["wav"]), dtype="float32")
        x = x[: sr * 6]
        f, t, z = signal.stft(x, sr, nperseg=1024, noverlap=768)
        db = 20 * np.log10(np.abs(z) + 1e-7)
        ax.pcolormesh(t, f / 1000, db, vmin=db.max() - 90, vmax=db.max(), shading="auto", cmap="magma")
        ax.set_ylim(0, 24)
        ax.set_ylabel("kHz")
        ax.set_title(f"{r['label']}  (UTMOS {r['utmos']}, NISQA {r['nisqa']}, bw99 {r['bw99']} Hz)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=80)
    plt.close(fig)


def report(out, rows):
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(rows, f, indent=1)
    lines = []
    groups = []
    for r in rows:
        if r["group"] not in groups:
            groups.append(r["group"])
    for g in groups:
        lines += [f"### {g}", "", "| voice | UTMOS | NISQA | colour | WER | bw99 | bw99.9 | air dB | RTF |", "|---|---|---|---|---|---|---|---|---|"]
        rs = [r for r in rows if r["group"] == g]
        rs.sort(key=lambda r: -(r.get("utmos", 0) + r.get("nisqa", 0)))
        for r in rs:
            if "error" in r:
                lines.append(f"| {r['label']} | error: {r['error'][:80]} | | | | | | | |")
                continue
            lines.append(
                f"| {r['label']} | {r['utmos']} | {r['nisqa']} | {r['coloration']} | {r['wer']} | {r['bw99']} | {r['bw999']} | {r['air']} | {r['rtf']} |"
            )
        lines.append("")
    table = "\n".join(lines)
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(table)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("## Voice lab\n\n" + table + "\n")
    # A page to listen side by side.
    cells = []
    for r in rows:
        if "wav" not in r:
            continue
        cells.append(
            f"<tr><td>{html.escape(r['group'])}</td><td>{html.escape(r['label'])}</td>"
            f"<td>{r['utmos']}</td><td>{r['nisqa']}</td><td>{r['bw99']}</td>"
            f"<td><audio controls preload=none src='{r['wav']}'></audio></td></tr>"
        )
    with open(os.path.join(out, "index.html"), "w") as f:
        f.write(
            "<!doctype html><meta charset=utf-8><title>Voice lab</title>"
            "<style>body{font:14px system-ui;margin:16px}td{padding:4px 8px;border-bottom:1px solid #ddd}</style>"
            "<table><tr><th>group</th><th>voice</th><th>UTMOS</th><th>NISQA</th><th>bw99</th><th></th></tr>"
            + "".join(cells)
            + "</table>"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--asr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true", help="a few voices per engine")
    ap.add_argument("--groups", default="kokoro,supertonic,pocket,kitten", help="which engines to run")
    ap.add_argument("--focus", default="", help="'clarity': only the voices the app uses, Clarity v1 vs v2")
    a = ap.parse_args()
    m = a.models
    os.makedirs(a.out, exist_ok=True)

    judges = Judges(a.asr)
    clarity = None
    cdir = os.path.join(m, "clarity")
    if os.path.isdir(cdir):
        clarity = Clarity(cdir)
    lab = Lab(a.out, judges, clarity)

    groups = set(a.groups.split(","))
    focus = a.focus == "clarity"
    if focus:
        groups = {"kokoro", "pocket"}

    # ---- Kokoro fp32 (24 kHz) -----------------------------------------------
    kdir = os.path.join(m, "kokoro")
    names = kokoro_names(kdir)
    tts = kokoro(kdir) if "kokoro" in groups else None
    english = [n for n in names if n[:2] in ("af", "am", "bf", "bm")] if tts else []
    if a.quick:
        english = ["af_heart", "am_michael", "bm_george", "bf_emma"]
    if focus:
        english = ["af_heart", "af_sarah", "af_kore", "am_fenrir", "am_puck", "bf_isabella", "bm_fable", "bm_george"]
    for n in english:
        sid = names.index(n)
        extra = {"lang": "en"} if n.startswith("b") else None
        cut = (11000.0,) if focus else (8000.0, 10000.0, 11000.0) if n in ("af_heart", "bm_george") else (11000.0,)
        got = lab.add("kokoro", f"kokoro {n}", lambda: tts.generate(TEXT, gen_config(sid=sid, speed=1.0, extra=extra)), cut)
        if n == "af_heart" and got is not None and clarity is not None and os.path.isdir(os.path.join(m, "lavasr-ref")):
            check_against_reference(clarity, os.path.join(m, "lavasr-ref"), signal.resample_poly(got[0], 2, 1).astype(np.float32))
    del tts

    # ---- Supertonic 3 (44.1 kHz): int8 (what the app ships) vs fp32 ----------
    sdir = os.path.join(m, "supertonic")
    voices = ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"]
    pick = [0, 5] if a.quick else range(10)
    variants = [("int8", sdir, ".int8", (8,)), ("fp32", os.path.join(m, "supertonic-fp32"), "", (5, 8, 16, 32))]
    for q, onnx_root, suffix, steps_list in variants if "supertonic" in groups else []:
        try:
            tts = supertonic(sdir, onnx_root, suffix)
        except Exception as e:
            lab.rows.append({"group": "supertonic", "label": f"supertonic {q}", "error": repr(e)})
            continue
        for sid in pick:
            for steps in steps_list:
                if q == "fp32" and steps != 16 and sid not in (0, 5):
                    continue  # the step sweep on one female and one male voice
                lab.add(
                    "supertonic",
                    f"supertonic {q} {voices[sid]} steps{steps}",
                    lambda: tts.generate(TEXT, gen_config(sid=sid, speed=1.0, num_steps=steps, extra={"lang": "en", "seed": 7})),
                )
        del tts

    # ---- Pocket TTS (24 kHz, clones a reference clip) ------------------------
    refs = sorted(f for f in os.listdir(os.path.join(m, "refs")) if f.endswith(".wav"))
    if a.quick:
        refs = refs[:2]
    pockets = (("int8", os.path.join(m, "pocket")), ("fp32", os.path.join(m, "pocket-fp32")))
    if focus:
        pockets = pockets[1:]
    for q, root in pockets if "pocket" in groups else ():
        try:
            tts = pocket(root, q == "int8")
        except Exception as e:
            lab.rows.append({"group": "pocket", "label": f"pocket {q}", "error": repr(e)})
            continue
        for ref_name in refs:
            ref, rsr = sf.read(os.path.join(m, "refs", ref_name), dtype="float32")
            if ref.ndim > 1:
                ref = ref[:, 0]
            settings = [(5, 0.7)] if q == "int8" else [(10, 0.5)] if focus else [(5, 0.7), (10, 0.7), (10, 0.5)]
            for steps, temp in settings:
                lab.add(
                    "pocket",
                    f"pocket {q} {ref_name[:-4]} steps{steps} t{temp}",
                    lambda: tts.generate(
                        TEXT,
                        gen_config(
                            reference_audio=ref,
                            reference_sample_rate=rsr,
                            num_steps=steps,
                            speed=1.0,
                            extra={"max_reference_audio_len": 12, "seed": 7, "temperature": temp},
                        ),
                    ),
                )
        del tts

    # ---- Kitten mini (24 kHz) ------------------------------------------------
    kt = os.path.join(m, "kitten")
    if os.path.isdir(kt) and "kitten" in groups:
        try:
            tts = kitten(kt)
            for sid in range(min(tts.num_speakers, 2 if a.quick else 8)):
                lab.add("kitten", f"kitten {sid}", lambda: tts.generate(TEXT, gen_config(sid=sid, speed=1.0)), ())
            del tts
        except Exception as e:
            lab.rows.append({"group": "kitten", "label": "kitten", "error": repr(e)})

    report(a.out, lab.rows)
    if focus:
        spectrograms(
            a.out,
            lab.rows,
            ["kokoro af_heart", "kokoro af_heart +clarity11k", "kokoro af_heart +clarity2", "kokoro bm_george", "kokoro bm_george +clarity2"],
            "spectrograms-1.png",
        )
        spectrograms(
            a.out,
            lab.rows,
            [
                "pocket fp32 alba-mackenna-a-moment-by steps10 t0.5",
                "pocket fp32 alba-mackenna-a-moment-by steps10 t0.5 +clarity11k",
                "pocket fp32 alba-mackenna-a-moment-by steps10 t0.5 +clarity2",
                "pocket fp32 voice-zero-caro-davy steps10 t0.5",
                "pocket fp32 voice-zero-caro-davy steps10 t0.5 +clarity2",
            ],
            "spectrograms-2.png",
        )
        return
    spectrograms(
        a.out,
        lab.rows,
        [
            "kokoro af_heart",
            "kokoro af_heart +clarity8k",
            "kokoro af_heart +clarity11k",
            "supertonic int8 F1 steps8",
            "supertonic fp32 F1 steps16",
        ],
        "spectrograms-1.png",
    )
    first_ref = refs[0][:-4] if refs else ""
    spectrograms(
        a.out,
        lab.rows,
        [
            "kokoro bm_george",
            "kokoro bm_george +clarity11k",
            f"pocket int8 {first_ref} steps5 t0.7",
            f"pocket fp32 {first_ref} steps10 t0.7",
            f"pocket fp32 {first_ref} steps10 t0.7 +clarity11k",
        ],
        "spectrograms-2.png",
    )


if __name__ == "__main__":
    main()
