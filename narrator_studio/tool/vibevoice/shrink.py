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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--check", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/tmp/vv-variants")
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

    results = {"fp32": {"mb": round(size_mb(a.pack))}}
    for name, d in variants.items():
        results[name] = {"mb": round(size_mb(d))}
    import soundfile as sf

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
    json.dump(results, open(os.path.join(a.check, "shrink.json"), "w"), indent=1)

    # The pack to ship: int8 language models if they stay close to full
    # precision, else float16-stored weights everywhere.
    close = all(results["lm8"][v]["latent_snr"] >= 20 for v in ("carter", "emma"))
    chosen = "lm8" if close else "f16w"
    print(f"shipping {chosen}: {results[chosen]['mb']} MB", flush=True)
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    shutil.copytree(variants[chosen], a.out)
    json.dump({"variant": chosen, **results}, open(os.path.join(a.out, "shrink.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
