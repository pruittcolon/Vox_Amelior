#!/usr/bin/env python3
"""Check every voice pack really speaks: synthesize, then let Whisper listen.

Writes WAVs plus a results.json/markdown table (speed and word error rate per
voice) so a bad pack or a wrong speaker id is caught before the app ships it.

  smoke.py --work DIR --asr DIR --out DIR [--all-kokoro]
"""

import argparse
import json
import os
import re
import time

import numpy as np
import sherpa_onnx
import soundfile as sf

TEXT = (
    "Welcome to the show. Today we explore why the sky is blue, "
    "and how a physicist would explain it to a curious child."
)


def words(s):
    return re.sub(r"[^a-z0-9' ]", " ", s.lower()).split()


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


class Listener:
    def __init__(self, asr_dir):
        def find(suffix):
            for f in sorted(os.listdir(asr_dir)):
                if f.endswith(suffix):
                    return os.path.join(asr_dir, f)
            raise SystemExit(f"no *{suffix} in {asr_dir}")

        self.rec = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=find("encoder.int8.onnx"),
            decoder=find("decoder.int8.onnx"),
            tokens=find("tokens.txt"),
            language="en",
            task="transcribe",
            num_threads=2,
        )

    def hear(self, samples, sr):
        s = self.rec.create_stream()
        s.accept_waveform(sr, samples)
        self.rec.decode_stream(s)
        return s.result.text.strip()


def kokoro_names(model_path):
    import onnx

    m = onnx.load(model_path, load_external_data=False)
    meta = {p.key: p.value for p in m.metadata_props}
    print("kokoro metadata:", {k: (v if len(v) < 300 else v[:300] + "...") for k, v in meta.items()})
    names = meta.get("speaker_names", "")
    return [n for n in names.split(",") if n], meta


def first(root, *names):
    for n in names:
        if os.path.exists(os.path.join(root, n)):
            return os.path.join(root, n)
    raise SystemExit(f"none of {names} in {root}")


def run(tts, gen, label, out, listener, results, text=TEXT):
    t0 = time.time()
    audio = gen(tts)
    took = time.time() - t0
    samples = np.asarray(audio.samples, dtype=np.float32)
    dur = len(samples) / audio.sample_rate
    path = os.path.join(out, re.sub(r"[^A-Za-z0-9_.-]", "_", label) + ".wav")
    sf.write(path, samples, audio.sample_rate, subtype="PCM_16")
    heard = listener.hear(samples, audio.sample_rate) if dur > 0 else ""
    row = {
        "voice": label,
        "seconds": round(dur, 2),
        "rtf": round(took / max(dur, 1e-6), 3),
        "peak": round(float(np.max(np.abs(samples))) if len(samples) else 0.0, 3),
        "wer": round(wer(text, heard), 3),
        "heard": heard,
    }
    results.append(row)
    print(json.dumps(row), flush=True)


LONG = (
    "Light from the sun looks white, but it is really a mixture of every colour. "
    "When sunlight enters the atmosphere, it bumps into tiny molecules of nitrogen and oxygen. "
    "Blue light has a shorter wavelength, so it is scattered far more than red light, "
    "and that scattered blue light reaches our eyes from every direction of the sky."
)


def bench(work):
    """Real-time factor for each engine and thread count (lower is faster)."""
    rows = []
    k = os.path.join(work, "kokoro-int8-multi-lang-v1_0")
    variants = [("kokoro int8", k, "model.int8.onnx")]
    k32 = os.path.join(work, "kokoro-multi-lang-v1_0")
    if os.path.isdir(k32):
        variants.append(("kokoro fp32", k32, "model.onnx"))
    for label, root, model in variants:
        for threads in (1, 2, 4):
            tts = sherpa_onnx.OfflineTts(
                sherpa_onnx.OfflineTtsConfig(
                    model=sherpa_onnx.OfflineTtsModelConfig(
                        kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                            model=os.path.join(root, model),
                            voices=os.path.join(root, "voices.bin"),
                            tokens=os.path.join(root, "tokens.txt"),
                            data_dir=os.path.join(root, "espeak-ng-data"),
                            lexicon=os.path.join(root, "lexicon-us-en.txt"),
                        ),
                        num_threads=threads,
                        debug=False,
                    ),
                    max_num_sentences=1,
                )
            )
            t0 = time.time()
            audio = tts.generate(LONG, sid=3, speed=1.0)
            rtf = (time.time() - t0) / (len(audio.samples) / audio.sample_rate)
            rows.append((label, threads, round(rtf, 3)))
            print("bench", label, threads, round(rtf, 3), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--asr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--all-kokoro", action="store_true")
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    listener = Listener(a.asr)
    results = []

    # ---- Kokoro -----------------------------------------------------------
    k = os.path.join(a.work, "kokoro-int8-multi-lang-v1_0")
    model = first(k, "model.int8.onnx", "model.onnx")
    names, _ = kokoro_names(model)
    print("kokoro speakers:", len(names), names)
    for mode, lexicon, lang in (
        ("lexicon", ",".join(p for p in (os.path.join(k, "lexicon-us-en.txt"),) if os.path.exists(p)), ""),
        ("espeak", "", "en-us"),
    ):
        tts = sherpa_onnx.OfflineTts(
            sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                        model=model,
                        voices=os.path.join(k, "voices.bin"),
                        tokens=os.path.join(k, "tokens.txt"),
                        data_dir=os.path.join(k, "espeak-ng-data"),
                        lexicon=lexicon,
                        lang=lang,
                    ),
                    num_threads=2,
                    debug=False,
                ),
                max_num_sentences=1,
            )
        )
        print(f"kokoro {mode}: sample_rate={tts.sample_rate} speakers={tts.num_speakers}", flush=True)
        if mode == "lexicon" and a.all_kokoro:
            sids = range(tts.num_speakers)
        else:
            want = ["af_heart", "bm_george", "am_michael", "bf_emma", "ff_siwis", "hm_omega"]
            sids = [names.index(n) for n in want if n in names] or [0]
        for sid in sids:
            label = f"kokoro-{mode}-{sid}-{names[sid] if sid < len(names) else sid}"
            run(tts, lambda t, sid=sid: t.generate(TEXT, sid=sid, speed=1.0), label, a.out, listener, results)

    # ---- Pocket TTS (clones the reference clip) -----------------------------
    p = os.path.join(a.work, "sherpa-onnx-pocket-tts-int8-2026-01-26")
    tts = sherpa_onnx.OfflineTts(
        sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                pocket=sherpa_onnx.OfflineTtsPocketModelConfig(
                    lm_flow=first(p, "lm_flow.int8.onnx", "lm_flow.onnx"),
                    lm_main=first(p, "lm_main.int8.onnx", "lm_main.onnx"),
                    encoder=first(p, "encoder.onnx", "encoder.int8.onnx"),
                    decoder=first(p, "decoder.int8.onnx", "decoder.onnx"),
                    text_conditioner=first(p, "text_conditioner.onnx", "text_conditioner.int8.onnx"),
                    vocab_json=os.path.join(p, "vocab.json"),
                    token_scores_json=os.path.join(p, "token_scores.json"),
                ),
                num_threads=2,
                debug=False,
            )
        )
    )
    print(f"pocket: sample_rate={tts.sample_rate}", flush=True)
    wavs = sorted(f for f in os.listdir(os.path.join(p, "test_wavs")) if f.endswith(".wav"))
    for wav in wavs:
        ref, sr = sf.read(os.path.join(p, "test_wavs", wav), dtype="float32")
        if ref.ndim > 1:
            ref = ref[:, 0]
        print(f"pocket ref {wav}: {len(ref) / sr:.1f}s at {sr} Hz", flush=True)

        def gen(t, ref=ref, sr=sr):
            g = sherpa_onnx.GenerationConfig()
            g.reference_audio = ref
            g.reference_sample_rate = sr
            g.num_steps = 5
            return t.generate(TEXT, g)

        run(tts, gen, f"pocket-{wav[:-4]}", a.out, listener, results)

    # ---- Supertonic ---------------------------------------------------------
    s = os.path.join(a.work, "sherpa-onnx-supertonic-3-tts-int8-2026-05-11")
    try:
        tts = sherpa_onnx.OfflineTts(
            sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    supertonic=sherpa_onnx.OfflineTtsSupertonicModelConfig(
                        duration_predictor=os.path.join(s, "duration_predictor.int8.onnx"),
                        text_encoder=os.path.join(s, "text_encoder.int8.onnx"),
                        vector_estimator=os.path.join(s, "vector_estimator.int8.onnx"),
                        vocoder=os.path.join(s, "vocoder.int8.onnx"),
                        tts_json=os.path.join(s, "tts.json"),
                        unicode_indexer=os.path.join(s, "unicode_indexer.bin"),
                        voice_style=os.path.join(s, "voice.bin"),
                    ),
                    num_threads=2,
                    debug=False,
                )
            )
        )
        print(f"supertonic: sample_rate={tts.sample_rate} speakers={tts.num_speakers}", flush=True)
        for sid in range(max(1, tts.num_speakers)):

            def gen(t, sid=sid):
                g = sherpa_onnx.GenerationConfig()
                g.sid = sid
                g.speed = 1.0
                g.num_steps = 8
                return t.generate(TEXT, g)

            run(tts, gen, f"supertonic-{sid}", a.out, listener, results)
    except Exception as e:  # keep the other packs' results
        print("supertonic failed:", repr(e))
        results.append({"voice": "supertonic", "error": repr(e)})

    bench_rows = bench(a.work) if a.bench else []

    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    lines = ["| voice | sec | RTF | peak | WER | heard |", "|---|---|---|---|---|---|"]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['voice']} | error | | | | {r['error']} |")
        else:
            lines.append(
                f"| {r['voice']} | {r['seconds']} | {r['rtf']} | {r['peak']} | {r['wer']} | {r['heard'][:90]} |"
            )
    if bench_rows:
        lines += ["", "| speed test | threads | RTF |", "|---|---|---|"]
        lines += [f"| {r[0]} | {r[1]} | {r[2]} |" for r in bench_rows]
    table = "\n".join(lines)
    print(table)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("## Voice smoke test\n\n" + table + "\n")


if __name__ == "__main__":
    main()
