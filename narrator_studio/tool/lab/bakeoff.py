#!/usr/bin/env python3
"""Voice bake-off: the same podcast lines in every candidate model, to listen to.

The automatic judges (UTMOS, NISQA) once picked Supertonic, and people found
it robotic. So this renders a real exchange (three speakers, questions,
reactions) with each model and measures how much work it takes, for a person
to choose by ear.

  bakeoff.py --engine NAME --models DIR --out DIR

Engines: kokoro, supertonic, pocket (sherpa-onnx, what the app runs today),
chatterbox (Resemble's official ONNX, fp32/q8/q4), soprano, neutts, vibevoice.
Each writes OUT/<label>.wav and OUT/<label>.json.
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LINES = [
    ("HOST", "Welcome to Curious Minds, the show where we take one everyday question and chase it all the way down."),
    ("CO-HOST", "And I'm the one who keeps asking, but why? Today's question sounds simple. Why is the sky blue?"),
    ("GUEST", "Here's the key. Short blue waves are scattered far more than long red waves. Roughly five times more."),
    ("CO-HOST", "Wait, really? Five times?"),
    ("GUEST", "Really. So blue light gets bounced around the whole sky, and it reaches your eyes from wherever you look."),
    ("HOST", "I love that. The blue sky and the red sunset are the same story, just seen from different ends."),
]

# Reference clips for the models that clone a voice (from the app's Gallery:
# Kyutai studio recordings and a LibriVox narrator, all CC0).
REFS = {
    "HOST": "unmute-prod-website-developer-1",
    "CO-HOST": "unmute-prod-website-default-voice",
    "GUEST": "voice-zero-bill-boerst",
}

THREADS = 4


def slug(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", s).strip("-").lower()


def mono(x):
    x = np.squeeze(np.asarray(x, dtype=np.float32))
    if x.ndim > 1:  # (channels, samples) or (samples, channels)
        x = x[0] if x.shape[0] < x.shape[-1] else x[:, 0]
    return x.reshape(-1)


def level(x):
    """Same loudness for every line (speech RMS -20 dBFS, peak at most -1 dBFS)."""
    rms = float(np.sqrt(np.mean(x**2))) if len(x) else 0.0
    if rms > 1e-6:
        x = x * (0.1 / rms)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak > 0.89:
        x = x * (0.89 / peak)
    return x.astype(np.float32)


def trim(x, sr):
    a = np.abs(x)
    on = np.where(a > 0.01 * max(float(a.max()), 1e-6))[0]
    if len(on) == 0:
        return x
    pad = int(0.04 * sr)
    return x[max(0, on[0] - pad) : on[-1] + pad]


def ref_clip(models, speaker, sr=None):
    import librosa

    path = os.path.join(models, "refs", REFS[speaker] + ".wav")
    x, r = librosa.load(path, sr=sr)
    return x.astype(np.float32), r, path


class Out:
    def __init__(self, out):
        self.out = out
        os.makedirs(out, exist_ok=True)

    def save(self, label, engine, pieces, sr, compute, notes="", setup=0.0):
        gap = np.zeros(int(0.35 * sr), dtype=np.float32)
        parts = []
        for x in pieces:
            parts += [level(trim(mono(x), sr)), gap]
        audio = np.concatenate(parts[:-1]) if parts else np.zeros(1, np.float32)
        seconds = len(audio) / sr
        speech = sum(len(trim(mono(x), sr)) for x in pieces) / sr
        name = slug(label)
        sf.write(os.path.join(self.out, name + ".wav"), audio, sr, subtype="PCM_16")
        # Each line alone too, for the speech recogniser that checks the words.
        for i, x in enumerate(pieces):
            sf.write(os.path.join(self.out, f"{name}.line{i}.wav"), level(trim(mono(x), sr)), sr, subtype="PCM_16")
        row = {
            "label": label,
            "engine": engine,
            "file": name,
            "sample_rate": sr,
            "seconds": round(seconds, 2),
            "compute_seconds": round(compute, 2),
            "setup_seconds": round(setup, 2),
            "rtf": round(compute / max(speech, 1e-6), 3),
            "notes": notes,
        }
        json.dump(row, open(os.path.join(self.out, name + ".json"), "w"), indent=1)
        print(json.dumps(row), flush=True)


def timed(fn):
    t0 = time.time()
    x = fn()
    return x, time.time() - t0


# ---- what the app runs today (sherpa-onnx) ----------------------------------


def run_sherpa(a, out):
    import voicelab as vl

    vl.THREADS = THREADS
    m = a.models
    names = vl.kokoro_names(os.path.join(m, "kokoro"))
    tts = vl.kokoro(os.path.join(m, "kokoro"))
    cast = {"HOST": "am_michael", "CO-HOST": "af_bella", "GUEST": "bm_george"}
    pieces, total = [], 0.0
    for spk, text in LINES:
        n = cast[spk]
        extra = {"lang": "en"} if n.startswith("b") else None
        audio, took = timed(lambda: tts.generate(text, vl.gen_config(sid=names.index(n), speed=1.0, extra=extra)))
        pieces.append(audio.samples)
        total += took
    out.save("Kokoro (the app's Natural set)", "kokoro", pieces, 24000, total, "Michael, Bella, George. Fast: plays live.")
    del tts

    tts = vl.supertonic(os.path.join(m, "supertonic"), os.path.join(m, "supertonic-fp32"), "")
    sids = {"HOST": 5, "CO-HOST": 1, "GUEST": 7}  # M1, F2, M3
    pieces, total, sr = [], 0.0, 44100
    for spk, text in LINES:
        cfg = vl.gen_config(sid=sids[spk], speed=1.0, num_steps=8, extra={"lang": "en", "seed": 7})
        audio, took = timed(lambda: tts.generate(text, cfg))
        pieces.append(audio.samples)
        sr = audio.sample_rate
        total += took
    out.save("Supertonic HD (the app's Studio set)", "supertonic", pieces, sr, total, "Miles, Faye, Martin.")
    del tts

    tts = vl.pocket(os.path.join(m, "pocket-fp32"), False)
    pieces, total = [], 0.0
    for spk, text in LINES:
        ref, rsr, _ = ref_clip(m, spk)
        cfg = vl.gen_config(
            reference_audio=ref,
            reference_sample_rate=rsr,
            num_steps=10,
            speed=1.0,
            extra={"max_reference_audio_len": 12, "seed": 7, "temperature": 0.5},
        )
        audio, took = timed(lambda: tts.generate(text, cfg))
        pieces.append(audio.samples)
        total += took
    out.save("Pocket TTS (cloned voices)", "pocket", pieces, 24000, total, "Kyutai, 100M. Clones the reference clips.")


# ---- Chatterbox Turbo (Resemble AI, MIT), official ONNX ---------------------


def run_chatterbox(a, out):
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    repo = "ResembleAI/chatterbox-turbo-ONNX"
    sr = 24000
    start, stop, silence = 6561, 6562, 4299
    tok = AutoTokenizer.from_pretrained(repo)

    sizes = {}

    def graph(name, dtype):
        f = f"{name}{'' if dtype == 'fp32' else '_quantized' if dtype == 'q8' else f'_{dtype}'}.onnx"
        # Real files side by side: ONNX Runtime refuses weights that live
        # elsewhere (the Hugging Face cache keeps them as blob symlinks).
        local = os.path.join(os.path.expanduser("~"), "chatterbox-turbo")
        path = hf_hub_download(repo, subfolder="onnx", filename=f, local_dir=local)
        data = hf_hub_download(repo, subfolder="onnx", filename=f"{f}_data", local_dir=local)
        sizes[(dtype, name)] = os.path.getsize(path) + os.path.getsize(data)
        return path

    def session(path):
        o = ort.SessionOptions()
        o.intra_op_num_threads = THREADS
        o.inter_op_num_threads = 1
        return ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])

    for dtype in a.dtypes.split(","):
        t0 = time.time()
        enc = session(graph("speech_encoder", dtype))
        emb = session(graph("embed_tokens", dtype))
        lm = session(graph("language_model", dtype))
        dec = session(graph("conditional_decoder", dtype))
        setup = time.time() - t0
        # What a phone would download: everything but the speech encoder,
        # whose output for each voice can be computed once and shipped.
        mb = sum(v for (d, n), v in sizes.items() if d == dtype and n != "speech_encoder") / 1e6
        # The voice of each speaker is encoded once (the app would ship this).
        voices = {}
        t1 = time.time()
        for spk in REFS:
            ref, _, _ = ref_clip(a.models, spk, sr=sr)
            voices[spk] = enc.run(None, {"audio_values": ref[np.newaxis, :]})
        encode = time.time() - t1
        lm_inputs = [i for i in lm.get_inputs() if "past_key_values" in i.name]
        pieces, total, tokens = [], 0.0, 0
        for spk, text in LINES:
            cond_emb, prompt_token, spk_emb, spk_feat = voices[spk]
            t2 = time.time()
            ids = tok(text, return_tensors="np")["input_ids"].astype(np.int64)
            generated = np.array([[start]], dtype=np.int64)
            past = None
            for step in range(1024):
                embeds = emb.run(None, {"input_ids": ids})[0]
                if step == 0:
                    embeds = np.concatenate((cond_emb, embeds), axis=1)
                    n = embeds.shape[1]
                    past = {
                        i.name: np.zeros([1, 16, 0, 64], dtype=np.float16 if i.type == "tensor(float16)" else np.float32)
                        for i in lm_inputs
                    }
                    mask = np.ones((1, n), dtype=np.int64)
                    pos = np.arange(n, dtype=np.int64).reshape(1, -1)
                logits, *present = lm.run(None, dict(inputs_embeds=embeds, attention_mask=mask, position_ids=pos, **past))
                logits = logits[:, -1, :].astype(np.float32)
                # Repetition penalty 1.2, then greedy (the reference settings).
                seen = np.take_along_axis(logits, generated, axis=1)
                np.put_along_axis(logits, generated, np.where(seen < 0, seen * 1.2, seen / 1.2), axis=1)
                ids = np.argmax(logits, axis=-1, keepdims=True).astype(np.int64)
                generated = np.concatenate((generated, ids), axis=-1)
                if int(ids[0, 0]) == stop:
                    break
                mask = np.concatenate([mask, np.ones((1, 1), dtype=np.int64)], axis=1)
                pos = pos[:, -1:] + 1
                for j, k in enumerate(past):
                    past[k] = present[j]
            speech = generated[:, 1:-1]
            tokens += speech.shape[1]
            codes = np.concatenate([prompt_token, speech, np.full((1, 3), silence, dtype=np.int64)], axis=1)
            wav = dec.run(None, dict(speech_tokens=codes, speaker_embeddings=spk_emb, speaker_features=spk_feat))[0]
            total += time.time() - t2
            pieces.append(wav.squeeze())
        out.save(
            f"Chatterbox Turbo {dtype}",
            "chatterbox",
            pieces,
            sr,
            total,
            f"Resemble AI, 350M, MIT. Cloned voices. {tokens} speech tokens; voices encoded once in {encode:.1f} s; "
            f"{mb:.0f} MB to download without the voice encoder.",
            setup,
        )
        del enc, emb, lm, dec


# ---- Soprano 1.1 (80M, Apache-2.0) ------------------------------------------


def run_soprano(a, out):
    from soprano import SopranoTTS

    t0 = time.time()
    model = SopranoTTS(backend="auto", device="cpu")
    setup = time.time() - t0
    pieces, total = [], 0.0
    for _, text in LINES:
        audio, took = timed(lambda: model.infer(text))
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        pieces.append(audio)
        total += took
    out.save("Soprano 1.1", "soprano", pieces, 32000, total, "80M, one voice for every speaker.", setup)


# ---- NeuTTS (Neuphonic) ------------------------------------------------------


def run_neutts(a, out):
    from neutts import NeuTTS

    samples = a.neutts_samples
    voices = {"HOST": "dave", "CO-HOST": "jo", "GUEST": "paul"}
    for backbone in a.backbones.split(","):
        t0 = time.time()
        try:
            tts = NeuTTS(backbone_repo=backbone, backbone_device="cpu", codec_repo="neuphonic/neucodec", codec_device="cpu")
        except Exception as e:  # e.g. a gated repo: no use for a free in-app download either
            print("SKIPPED", backbone, repr(e)[:300], flush=True)
            continue
        refs = {}
        for spk, v in voices.items():
            text = open(os.path.join(samples, v + ".txt")).read().strip()
            refs[spk] = (tts.encode_reference(os.path.join(samples, v + ".wav")), text)
        setup = time.time() - t0
        pieces, total = [], 0.0
        for spk, text in LINES:
            codes, ref_text = refs[spk]
            audio, took = timed(lambda: tts.infer(text, codes, ref_text))
            pieces.append(audio)
            total += took
        out.save(f"NeuTTS {backbone.split('/')[-1]}", "neutts", pieces, 24000, total, "Dave, Jo, Paul (its sample voices).", setup)
        del tts


# ---- VibeVoice Realtime 0.5B (Microsoft) -------------------------------------


def run_vibevoice(a, out):
    import copy

    import torch
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

    torch.set_num_threads(THREADS)
    path = "microsoft/VibeVoice-Realtime-0.5B"
    t0 = time.time()
    processor = VibeVoiceStreamingProcessor.from_pretrained(path)
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        path, torch_dtype=torch.float32, device_map="cpu", attn_implementation="sdpa"
    )
    model.eval()
    model.set_ddpm_inference_steps(num_steps=5)
    setup = time.time() - t0
    voices = {"HOST": "en-Carter_man", "CO-HOST": "en-Emma_woman", "GUEST": "en-Frank_man"}
    prompts = {}
    for spk, v in voices.items():
        # Microsoft's own voice prompts from the official repo (pickled caches).
        prompts[spk] = torch.load(os.path.join(a.vibevoice_voices, v + ".pt"), map_location="cpu", weights_only=False)
    pieces, total = [], 0.0
    for spk, text in LINES:
        t1 = time.time()
        inputs = processor.process_input_with_cached_prompt(
            text=text, cached_prompt=prompts[spk], padding=True, return_tensors="pt", return_attention_mask=True
        )
        with torch.no_grad():
            got = model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=1.5,
                tokenizer=processor.tokenizer,
                generation_config={"do_sample": False},
                verbose=False,
                all_prefilled_outputs=copy.deepcopy(prompts[spk]),
            )
        total += time.time() - t1
        pieces.append(got.speech_outputs[0].float().cpu().numpy())
    out.save(
        "VibeVoice Realtime 0.5B",
        "vibevoice",
        pieces,
        24000,
        total,
        "Microsoft, MIT. Carter, Emma, Frank (about 6 English voices, no cloning). PyTorch on CPU, unoptimised.",
        setup,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--models", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtypes", default="fp32,q4")  # q8 ran 8x slower than q4 on CPU and rambled
    # neutts-air is gated on Hugging Face (needs an account that accepted its terms).
    ap.add_argument("--backbones", default="neuphonic/neutts-nano,neuphonic/neutts-air")
    ap.add_argument("--neutts-samples", default="")
    ap.add_argument("--vibevoice-voices", default="")
    a = ap.parse_args()
    out = Out(a.out)
    {
        "sherpa": run_sherpa,
        "chatterbox": run_chatterbox,
        "soprano": run_soprano,
        "neutts": run_neutts,
        "vibevoice": run_vibevoice,
    }[a.engine](a, out)


if __name__ == "__main__":
    main()
