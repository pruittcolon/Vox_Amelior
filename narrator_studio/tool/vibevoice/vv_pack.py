"""The VibeVoice pack as the app uses it, and the generation loop on ONNX
Runtime: the reference the Dart port (lib/core/tts/vibevoice.dart) follows
step for step.

Pack layout (DIR):
  text_lm.onnx   the 4 lower Qwen2 layers: (x, pos, past k/v...) -> (h, present k/v...)
  tts_lm.onnx    the 20 upper layers + final norm, same interface
  head.onnx      diffusion head: (noisy [B,64], t [B], cond [B,896]) -> v [B,64]
  decoder.onnx   acoustic decoder: latents [1,64,T] -> audio [1,1,T*3200] (24 kHz)
  embed.f16      token embeddings, float16 [vocab, 896]
  small.f32      tts_input_types, acoustic_connector, tts_eos_classifier (see SMALL)
  tokenizer.json Qwen2.5 tokenizer
  voices/<name>.bin + voices.json   prompt caches (see write_voice)
  vibevoice.json dims, scheduler coefficients, latent scaling
"""

import json
import os
import struct

import numpy as np

HIDDEN = 896
LATENT = 64
KV_HEADS = 2
HEAD_DIM = 64
TEXT_WINDOW = 5
SPEECH_WINDOW = 6
CFG = 1.5
MAX_POSITIONS = 8192

# small.f32: float32 arrays in this order.
SMALL = [
    ("types", (2, HIDDEN)),
    ("conn_fc1_w", (HIDDEN, LATENT)),
    ("conn_fc1_b", (HIDDEN,)),
    ("conn_norm_w", (HIDDEN,)),
    ("conn_fc2_w", (HIDDEN, HIDDEN)),
    ("conn_fc2_b", (HIDDEN,)),
    ("eos_fc1_w", (HIDDEN, HIDDEN)),
    ("eos_fc1_b", (HIDDEN,)),
    ("eos_fc2_w", (1, HIDDEN)),
    ("eos_fc2_b", (1,)),
]


def write_small(path, arrays):
    with open(path, "wb") as f:
        for name, shape in SMALL:
            a = np.asarray(arrays[name], dtype="<f4")
            assert a.shape == shape, (name, a.shape, shape)
            f.write(a.tobytes())


def read_small(path):
    raw = np.fromfile(path, dtype="<f4")
    out, at = {}, 0
    for name, shape in SMALL:
        n = int(np.prod(shape))
        out[name] = raw[at : at + n].reshape(shape)
        at += n
    assert at == len(raw)
    return out


def write_voice(path, lm, tts, neg, neg_last):
    """lm/tts/neg: lists of (k, v), each [2, L, 64] float32. Header: magic,
    then the three lengths and layer counts, then k,v per layer, then the
    negative branch's last hidden state [896]."""
    with open(path, "wb") as f:
        f.write(b"VVP1")
        for kv in (lm, tts, neg):
            f.write(struct.pack("<ii", len(kv), kv[0][0].shape[1]))
        for kv in (lm, tts, neg):
            for k, v in kv:
                f.write(np.ascontiguousarray(k, dtype="<f4").tobytes())
                f.write(np.ascontiguousarray(v, dtype="<f4").tobytes())
        f.write(np.asarray(neg_last, dtype="<f4").reshape(HIDDEN).tobytes())


def read_voice(path):
    raw = open(path, "rb").read()
    assert raw[:4] == b"VVP1"
    dims = struct.unpack("<6i", raw[4:28])
    at = 28
    parts = []
    for layers, length in zip(dims[0::2], dims[1::2]):
        kv = []
        n = KV_HEADS * length * HEAD_DIM
        for _ in range(layers):
            k = np.frombuffer(raw, "<f4", n, at).reshape(1, KV_HEADS, length, HEAD_DIM)
            at += 4 * n
            v = np.frombuffer(raw, "<f4", n, at).reshape(1, KV_HEADS, length, HEAD_DIM)
            at += 4 * n
            kv.append((k, v))
        parts.append(kv)
    neg_last = np.frombuffer(raw, "<f4", HIDDEN, at)
    return parts[0], parts[1], parts[2], neg_last


class Pack:
    def __init__(self, d, threads=4, models=None):
        import onnxruntime as ort

        self.dir = d
        self.cfg = json.load(open(os.path.join(d, "vibevoice.json")))
        o = ort.SessionOptions()
        o.intra_op_num_threads = threads
        o.inter_op_num_threads = 1
        models = models or {}

        def s(name):
            return ort.InferenceSession(os.path.join(d, models.get(name, name + ".onnx")), o, providers=["CPUExecutionProvider"])

        self.text_lm, self.tts_lm, self.head, self.decoder = s("text_lm"), s("tts_lm"), s("head"), s("decoder")
        self.small = read_small(os.path.join(d, "small.f32"))
        vocab = self.cfg["vocab"]
        self.embed = np.memmap(os.path.join(d, "embed.f16"), dtype="<f2", mode="r", shape=(vocab, HIDDEN))
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))

    def tokens(self, text):
        return self.tok.encode(text.strip() + "\n", add_special_tokens=False).ids

    def lm(self, sess, x, pos, kv):
        feeds = {"x": x.astype(np.float32), "pos": np.asarray(pos, np.float32)[None, :]}
        for i, (k, v) in enumerate(kv):
            feeds[f"past_k{i}"] = k
            feeds[f"past_v{i}"] = v
        out = sess.run(None, feeds)
        return out[0], [(out[1 + 2 * i], out[2 + 2 * i]) for i in range(len(kv))]

    def connector(self, z):
        s = self.small
        x = s["conn_fc1_w"] @ z + s["conn_fc1_b"]
        x = x / np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-6).astype(np.float32) * s["conn_norm_w"]
        return s["conn_fc2_w"] @ x + s["conn_fc2_b"]

    def eos(self, h):
        s = self.small
        x = np.maximum(s["eos_fc1_w"] @ h + s["eos_fc1_b"], 0)
        return float((s["eos_fc2_w"] @ x + s["eos_fc2_b"])[0]) > 0  # sigmoid > 0.5

    def sample(self, cond, neg, noise):
        sch = self.cfg["scheduler"]
        x = noise.astype(np.float32)
        prev = np.zeros(LATENT, np.float32)
        c = np.stack([cond, neg]).astype(np.float32)
        for k, t in enumerate(sch["timesteps"]):
            v = self.head.run(None, {"noisy": np.stack([x, x]), "t": np.array([t, t], np.float32), "cond": c})[0]
            v = v[1] + CFG * (v[0] - v[1])
            x0 = sch["alpha"][k] * x - sch["sigma"][k] * v
            x = sch["cs"][k] * x + sch["c0"][k] * x0 + sch["c1"][k] * prev
            prev = x0
        return x

    def generate(self, voice, text, noise):
        """noise(i) -> the i-th Gaussian [64] vector. Returns (latents [T,64], audio)."""
        lm_kv, tts_kv, neg_kv, neg_h = voice
        lm_pos, tts_pos, neg_pos = lm_kv[0][0].shape[2], tts_kv[0][0].shape[2], neg_kv[0][0].shape[2]
        types = self.small["types"]
        ids = self.tokens(text)
        latents, w, done, tts_h = [], 0, False, None
        while not done:
            win = ids[w * TEXT_WINDOW : (w + 1) * TEXT_WINDOW]
            w += 1
            if win:
                n = len(win)
                x = np.asarray(self.embed[win], np.float32)[None]
                h, lm_kv = self.lm(self.text_lm, x, range(lm_pos, lm_pos + n), lm_kv)
                lm_pos += n
                h, tts_kv = self.lm(self.tts_lm, h + types[1], range(tts_pos, tts_pos + n), tts_kv)
                tts_pos += n
                tts_h = h[0, -1]
            elif tts_h is None:
                break
            for _ in range(SPEECH_WINDOW):
                z = self.sample(tts_h, neg_h, noise(len(latents)))
                latents.append(z)
                e = (self.connector(z) + types[0])[None, None]
                if tts_pos >= MAX_POSITIONS:
                    done = True
                    break
                h, tts_kv = self.lm(self.tts_lm, e, [tts_pos], tts_kv)
                tts_pos += 1
                tts_h = h[0, -1]
                h, neg_kv = self.lm(self.tts_lm, e, [neg_pos], neg_kv)
                neg_pos += 1
                neg_h = h[0, -1]
                if self.eos(tts_h):
                    done = True
                    break
        lat = np.stack(latents)
        scaled = lat / self.cfg["scaling"] - self.cfg["bias"]
        audio = self.decoder.run(None, {"latents": scaled.T[None].astype(np.float32)})[0].reshape(-1)
        return lat, audio
