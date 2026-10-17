#!/usr/bin/env python3
"""Makes the VibeVoice pack small enough to download, and measures what each
saving costs: the same text and diffusion noise through every variant,
compared with full precision.

  shrink.py --pack DIR --check DIR --out DIR

Variants:
  f16w   every large weight stored as float16 (cast back to float32 when the
         model loads): half the download, the same arithmetic.
  lm8    as f16w, but the two language models' matrix weights in int8
         (dynamic quantization): faster on a phone CPU, slightly coarser.
The chosen pack goes to OUT; results.json records every number.
"""

import argparse
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import onnx
import soundfile as sf
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vv_pack  # noqa: E402

GRAPHS = ["text_lm", "tts_lm", "head", "decoder"]
TEXT = "Here's the key. Short blue waves are scattered far more than long red waves, roughly five times more."


def save(m, path):
    for f in (path, path + ".data"):
        if os.path.exists(f):
            os.remove(f)
    onnx.save(m, path, save_as_external_data=True, all_tensors_to_one_file=True, location=os.path.basename(path) + ".data")


def f16_weights(src, dst, min_size=4096):
    """Large float initializers -> float16 + Cast; ONNX Runtime folds the casts at load."""
    m = onnx.load(src)
    g = m.graph
    casts, keep, halves = [], [], []
    for init in g.initializer:
        if init.data_type == TensorProto.FLOAT and int(np.prod(init.dims)) >= min_size:
            a = numpy_helper.to_array(init)
            h = numpy_helper.from_array(a.astype(np.float16), init.name + "__f16")
            halves.append(h)
            casts.append(helper.make_node("Cast", [h.name], [init.name], to=TensorProto.FLOAT, name=init.name + "__cast"))
        else:
            keep.append(init)
    del g.initializer[:]
    g.initializer.extend(keep + halves)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)
    save(m, dst)


def nbits(src, dst, bits):
    """Weight-only quantization (MatMulNBits): the matrices are stored in
    `bits` bits per weight, in blocks of 32 with their own scale; the
    arithmetic stays float, unlike dynamic quantization."""
    from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer

    cfg = DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, accuracy_level=4, bits=bits)
    q = MatMulNBitsQuantizer(onnx.load(src), block_size=32, is_symmetric=True, accuracy_level=4, algo_config=cfg)
    q.process()
    save(q.model.model, dst)


def int8_lm(src, dst):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    tmp = dst + ".tmp.onnx"
    quantize_dynamic(src, tmp, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"], per_channel=True, use_external_data_format=True)
    m = onnx.load(tmp)
    save(m, dst)
    for f in os.listdir(os.path.dirname(tmp)):
        if f.startswith(os.path.basename(tmp)):
            os.remove(os.path.join(os.path.dirname(tmp), f))


def size_mb(d):
    return sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))) / 1e6


def copy_rest(src, dst):
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        s = os.path.join(src, f)
        if f.split(".")[0] in GRAPHS and f.endswith((".onnx", ".onnx.data")):
            continue
        if os.path.isdir(s):
            shutil.copytree(s, os.path.join(dst, f), dirs_exist_ok=True)
        else:
            shutil.copy(s, dst)


def snr(ref, got):
    n = min(len(ref), len(got))
    e = ref[:n] - got[:n]
    return 10 * math.log10(float(np.sum(ref[:n] ** 2)) / max(float(np.sum(e**2)), 1e-20))


def run(pack_dir, voice, noise):
    pack = vv_pack.Pack(pack_dir)
    v = vv_pack.read_voice(os.path.join(pack_dir, "voices", voice + ".bin"))
    t0 = time.time()
    lat, audio = pack.generate(v, TEXT, lambda i: noise[i])
    return lat, audio, time.time() - t0


def whisper(d):
    import glob

    import sherpa_onnx

    return sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=glob.glob(os.path.join(d, "*encoder.int8.onnx"))[0],
        decoder=glob.glob(os.path.join(d, "*decoder.int8.onnx"))[0],
        tokens=glob.glob(os.path.join(d, "*tokens.txt"))[0],
        language="en",
        task="transcribe",
        num_threads=4,
    )


def word_errors(asr, audio, text):
    import re

    def words(t):
        return re.sub(r"[^a-z0-9' ]+", " ", t.lower()).split()

    st = asr.create_stream()
    st.accept_waveform(24000, audio.astype(np.float32))
    asr.decode_stream(st)
    r, h = words(text), words(st.result.text)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)], len(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--check", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/tmp/vv-variants")
    ap.add_argument("--asr", default="", help="sherpa-onnx Whisper folder for the word check")
    a = ap.parse_args()

    variants = {}
    f16 = os.path.join(a.work, "f16w")
    copy_rest(a.pack, f16)
    for g in GRAPHS:
        f16_weights(os.path.join(a.pack, g + ".onnx"), os.path.join(f16, g + ".onnx"))
    variants["f16w"] = f16
    lm8 = os.path.join(a.work, "lm8")
    copy_rest(a.pack, lm8)
    for g in ("head", "decoder"):
        for f in (g + ".onnx", g + ".onnx.data"):
            shutil.copy(os.path.join(f16, f), lm8)
    for g in ("text_lm", "tts_lm"):
        int8_lm(os.path.join(a.pack, g + ".onnx"), os.path.join(lm8, g + ".onnx"))
    variants["lm8"] = lm8
    for bits in (8, 4):
        d = os.path.join(a.work, f"w{bits}")
        try:
            copy_rest(a.pack, d)
            for f in ("decoder.onnx", "decoder.onnx.data"):
                shutil.copy(os.path.join(f16, f), d)
            for g in ("text_lm", "tts_lm", "head"):
                nbits(os.path.join(a.pack, g + ".onnx"), os.path.join(d, g + ".onnx"), bits)
            vv_pack.Pack(d)  # does ONNX Runtime run it?
            variants[f"w{bits}"] = d
        except Exception as e:  # e.g. a runtime without 8-bit MatMulNBits
            print(f"w{bits}: not available: {e!r}"[:400], flush=True)

    results = {"fp32": {"mb": round(size_mb(a.pack))}}
    for name, d in variants.items():
        results[name] = {"mb": round(size_mb(d))}
    for voice in ("carter", "emma"):
        noise = np.fromfile(os.path.join(a.check, f"{voice}-noise.f32"), "<f4").reshape(-1, vv_pack.LATENT)
        ref_lat, ref, took = run(a.pack, voice, noise)
        results["fp32"][voice] = {"seconds": round(len(ref) / 24000, 2), "work": round(took / (len(ref) / 24000), 2)}
        for name, d in variants.items():
            lat, audio, took = run(d, voice, noise)
            n = min(len(lat), len(ref_lat))
            results[name][voice] = {
                "frames": f"{len(lat)} vs {len(ref_lat)}",
                "latent_snr": round(snr(ref_lat[:n].reshape(-1), lat[:n].reshape(-1)), 1),
                "audio_snr": round(snr(ref, audio), 1),
                "work": round(took / (len(audio) / 24000), 2),
            }
            sf.write(os.path.join(a.check, f"{voice}-{name}.wav"), audio, 24000)
        print(json.dumps(results, indent=1), flush=True)
    # Free-running speech (each variant's own noise) through Whisper: does it
    # still say every word?
    asr = whisper(a.asr) if a.asr else None
    lines = [
        "Welcome to Curious Minds, the show where we chase one everyday question all the way down.",
        "So blue light gets bounced around the whole sky, and reaches your eyes from wherever you look.",
    ]
    for name, d in [("fp32", a.pack), *variants.items()]:
        pack = vv_pack.Pack(d)
        errors, words, secs, work = 0.0, 0, 0.0, 0.0
        for voice in ("carter", "emma"):
            v = vv_pack.read_voice(os.path.join(d, "voices", voice + ".bin"))
            rng = np.random.default_rng(7)
            for text in lines:
                t0 = time.time()
                _, audio = pack.generate(v, text, lambda i: rng.standard_normal(vv_pack.LATENT).astype(np.float32))
                work += time.time() - t0
                secs += len(audio) / 24000
                if asr is not None:
                    e, n = word_errors(asr, audio, text)
                    errors += e
                    words += n
            sf.write(os.path.join(a.check, f"free-{voice}-{name}.wav"), audio, 24000)
        results[name]["free_work"] = round(work / secs, 2)
        if words:
            results[name]["wer"] = round(100 * errors / words, 1)
        print(name, json.dumps(results[name]), flush=True)
    json.dump(results, open(os.path.join(a.check, "shrink.json"), "w"), indent=1)

    # The pack to ship: the fastest variant that stays close to full precision
    # (same noise -> nearly the same frames) and says every word as well.
    base_wer = results["fp32"].get("wer", 0.0)

    def good(name):
        r = results[name]
        close = all(r[v]["latent_snr"] >= 20 for v in ("carter", "emma"))
        clear = r.get("wer", 0.0) <= base_wer + 3
        return close and clear

    candidates = [n for n in variants if n != "f16w" and good(n)]
    chosen = min(candidates, key=lambda n: results[n]["free_work"]) if candidates else "f16w"
    print(f"shipping {chosen}: {results[chosen]['mb']} MB", flush=True)
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    shutil.copytree(variants[chosen], a.out)
    json.dump({"variant": chosen, **results}, open(os.path.join(a.out, "shrink.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
