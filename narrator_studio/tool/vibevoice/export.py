#!/usr/bin/env python3
"""Converts Microsoft's VibeVoice-Realtime-0.5B into the app's ONNX pack and
proves each piece matches the original.

  export.py --voices DIR --out PACK [--check-dir DIR]

1. Every network is exported (text LM, TTS LM, diffusion head, acoustic
   decoder) and compared with the PyTorch module on the same inputs.
2. The scheduler is reduced to per-step coefficients and compared with the
   original scheduler.
3. The whole loop on ONNX Runtime (vv_pack.Pack.generate) is compared with
   Microsoft's generate() given the very same diffusion noise.
"""

import argparse
import copy
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vv_pack  # noqa: E402

MODEL = "microsoft/VibeVoice-Realtime-0.5B"
VOICES = {  # app name -> Microsoft's preset
    "carter": "en-Carter_man",
    "davis": "en-Davis_man",
    "emma": "en-Emma_woman",
    "frank": "en-Frank_man",
    "grace": "en-Grace_woman",
    "mike": "en-Mike_man",
}
TEST = "Here's the key. Short blue waves are scattered far more than long red waves, roughly five times more."


def log(*a):
    print(*a, flush=True)


def rms(x, w, eps):
    v = x.float().pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(v + eps) * w


def rotate(x):
    return torch.cat([-x[..., HALF:], x[..., :HALF]], dim=-1)


HALF = vv_pack.HEAD_DIM // 2


class Layers(torch.nn.Module):
    """Qwen2 decoder layers with an explicit cache: what the ONNX graphs are."""

    def __init__(self, qwen, final_norm, theta, eps):
        super().__init__()
        self.layers = qwen.layers
        self.norm = qwen.norm if final_norm else None
        self.eps = eps
        inv = 1.0 / (theta ** (torch.arange(0, vv_pack.HEAD_DIM, 2, dtype=torch.float64) / vv_pack.HEAD_DIM))
        self.register_buffer("inv", inv.float(), persistent=False)

    def forward(self, x, pos, keep, *past):
        """x [B,S,896], pos [B,S], keep [B,P+S] (1 = a real position, 0 = padding);
        past/present k,v [B,2,P,64] per layer. Rows are independent sequences
        (the voice and its guidance branch), left-padded to one cache length."""
        b, s = x.shape[0], x.shape[1]
        p = past[0].shape[2]
        f = pos[:, :, None] * self.inv[None, None, :]
        emb = torch.cat([f, f], dim=-1)
        cos, sin = emb.cos()[:, None], emb.sin()[:, None]
        i = torch.arange(s, dtype=torch.float32)[:, None] + p
        j = torch.arange(p + s, dtype=torch.float32)[None, :]
        causal = torch.where(j <= i, 0.0, -1e9)
        bias = causal[None, None] + (1.0 - keep)[:, None, None, :] * -1e9
        nh = 14
        rep = nh // vv_pack.KV_HEADS
        present = []
        for n, layer in enumerate(self.layers):
            a = layer.self_attn
            h = rms(x, layer.input_layernorm.weight, self.eps)
            q = a.q_proj(h).view(b, s, nh, vv_pack.HEAD_DIM).transpose(1, 2)
            k = a.k_proj(h).view(b, s, vv_pack.KV_HEADS, vv_pack.HEAD_DIM).transpose(1, 2)
            v = a.v_proj(h).view(b, s, vv_pack.KV_HEADS, vv_pack.HEAD_DIM).transpose(1, 2)
            q = q * cos + rotate(q) * sin
            k = k * cos + rotate(k) * sin
            k = torch.cat([past[2 * n], k], dim=2)
            v = torch.cat([past[2 * n + 1], v], dim=2)
            present += [k, v]
            t = k.shape[2]
            kk = k[:, :, None].expand(b, vv_pack.KV_HEADS, rep, t, vv_pack.HEAD_DIM).reshape(b, nh, t, vv_pack.HEAD_DIM)
            vv = v[:, :, None].expand(b, vv_pack.KV_HEADS, rep, t, vv_pack.HEAD_DIM).reshape(b, nh, t, vv_pack.HEAD_DIM)
            w = torch.softmax(q @ kk.transpose(-1, -2) / math.sqrt(vv_pack.HEAD_DIM) + bias, dim=-1)
            o = (w @ vv).transpose(1, 2).reshape(b, s, nh * vv_pack.HEAD_DIM)
            x = x + a.o_proj(o)
            h = rms(x, layer.post_attention_layernorm.weight, self.eps)
            m = layer.mlp
            x = x + m.down_proj(F.silu(m.gate_proj(h)) * m.up_proj(h))
        if self.norm is not None:
            x = self.norm(x)
        return (x, *present)


class Head(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, noisy, t, cond):
        return self.head(noisy, t, condition=cond)


class Decoder(torch.nn.Module):
    def __init__(self, tok):
        super().__init__()
        self.tok = tok

    def forward(self, latents):
        return self.tok.decode(latents)


def cache_layers(c):
    """(k, v) per layer from a transformers cache, old or new style."""
    if hasattr(c, "key_cache") and len(getattr(c, "key_cache", [])):
        return list(zip(c.key_cache, c.value_cache))
    return [(l.keys, l.values) for l in c.layers]


def export(module, args, names, outs, dynamic, path):
    torch.onnx.export(
        module,
        args,
        path,
        input_names=names,
        output_names=outs,
        dynamic_axes=dynamic,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    import onnx

    m = onnx.load(path)
    onnx.save(m, path, save_as_external_data=True, all_tensors_to_one_file=True, location=os.path.basename(path) + ".data")
    log(f"exported {os.path.basename(path)}: {sum(os.path.getsize(p) for p in glob.glob(path + '*')) / 1e6:.0f} MB")


def compare(name, a, b, tol):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    err = float(np.max(np.abs(a - b)))
    scale = float(np.max(np.abs(b))) or 1.0
    log(f"{name}: max |diff| {err:.2e} (values up to {scale:.2e})")
    if err > tol * scale:
        raise SystemExit(f"{name} does not match (tolerance {tol})")


def lm_feeds(x, pos, keep, kv):
    feeds = {"x": x, "pos": np.asarray(pos, np.float32), "keep": np.asarray(keep, np.float32)}
    for i, (k, v) in enumerate(kv):
        feeds[f"past_k{i}"], feeds[f"past_v{i}"] = k, v
    return feeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", required=True, help="VibeVoice/demo/voices/streaming_model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-dir", default="", help="where to write the comparison audio")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "voices"), exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)

    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

    processor = VibeVoiceStreamingProcessor.from_pretrained(MODEL)
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        MODEL, torch_dtype=torch.float32, device_map="cpu", attn_implementation="sdpa"
    )
    model.eval()
    model.set_ddpm_inference_steps(num_steps=5)
    inner = model.model
    dec_cfg = model.config.decoder_config
    theta, eps = float(dec_cfg.rope_theta), float(dec_cfg.rms_norm_eps)

    def params(m):
        return sum(p.numel() for p in m.parameters()) / 1e6

    log(
        f"parameters (M): text LM {params(inner.language_model):.0f} (embeddings "
        f"{inner.language_model.embed_tokens.weight.numel() / 1e6:.0f}), TTS LM {params(inner.tts_language_model):.0f}, "
        f"head {params(inner.prediction_head):.0f}, decoder {params(inner.acoustic_tokenizer.decoder):.0f}"
    )

    text_lm = Layers(inner.language_model, False, theta, eps).eval()
    tts_lm = Layers(inner.tts_language_model, True, theta, eps).eval()

    # ---- 1. the language models, checked against transformers ------------
    with torch.no_grad():
        x = torch.randn(1, 9, vv_pack.HIDDEN) * 0.1
        pos = torch.arange(9, dtype=torch.float32)[None]
        for name, mine, hf, layers in (
            ("text LM", text_lm, inner.language_model, 4),
            ("TTS LM", tts_lm, inner.tts_language_model, 20),
        ):
            ref = hf(inputs_embeds=x, position_ids=pos.long(), use_cache=False).last_hidden_state
            empty = [torch.zeros(1, vv_pack.KV_HEADS, 0, vv_pack.HEAD_DIM)] * (2 * layers)
            got = mine(x, pos, torch.ones(1, 9), *empty)[0]
            compare(f"{name} vs transformers", got, ref, 1e-4)
            # In two pieces through the cache: the same as all at once.
            first = mine(x[:, :5], pos[:, :5], torch.ones(1, 5), *empty)
            second = mine(x[:, 5:], pos[:, 5:], torch.ones(1, 9), *first[1:])[0]
            compare(f"{name} through its cache", second, ref[:, 5:], 1e-4)
            # Two sequences in one batch, the shorter one left-padded: each row
            # comes out as if it were alone. Row 0: x after 5 positions; row 1:
            # y after 2 positions (3 of padding in front).
            y = torch.randn(1, 3, vv_pack.HIDDEN) * 0.1
            alone = mine(y, torch.arange(3, dtype=torch.float32)[None], torch.ones(1, 3), *empty)
            short = mine(y[:, :2], torch.arange(2, dtype=torch.float32)[None], torch.ones(1, 2), *empty)
            joint = [
                torch.cat([a0[:, :, :5], torch.cat([torch.zeros_like(a1[:, :, :1]).expand(1, vv_pack.KV_HEADS, 3, vv_pack.HEAD_DIM), a1], dim=2)], dim=0)
                for a0, a1 in zip(first[1:], short[1:])
            ]
            keep = torch.tensor([[1.0] * 5 + [1.0], [0.0] * 3 + [1.0] * 2 + [1.0]])
            step = torch.cat([x[:, 5:6], y[:, 2:3]], dim=0)
            both = mine(step, torch.tensor([[5.0], [2.0]]), keep, *joint)[0]
            compare(f"{name} batched row 0", both[0, 0], ref[0, 5], 1e-4)
            compare(f"{name} batched row 1", both[1, 0], alone[0][0, 2], 1e-4)

    # ---- 2. export ------------------------------------------------------------
    import onnxruntime as ort

    def lm_export(mod, layers, path):
        names = ["x", "pos", "keep"] + [f"past_{kv}{i}" for i in range(layers) for kv in "kv"]
        outs = ["h"] + [f"present_{kv}{i}" for i in range(layers) for kv in "kv"]
        dyn = {"x": {0: "b", 1: "s"}, "pos": {0: "b", 1: "s"}, "keep": {0: "b", 1: "t"}, "h": {0: "b", 1: "s"}}
        for i in range(layers):
            for kv in "kv":
                dyn[f"past_{kv}{i}"] = {0: "b", 2: "p"}
                dyn[f"present_{kv}{i}"] = {0: "b", 2: "t"}
        ex = (
            torch.zeros(2, 3, vv_pack.HIDDEN),
            torch.arange(4, 7, dtype=torch.float32)[None].expand(2, 3).contiguous(),
            torch.ones(2, 7),
        ) + tuple(torch.zeros(2, vv_pack.KV_HEADS, 4, vv_pack.HEAD_DIM) for _ in range(2 * layers))
        export(mod, ex, names, outs, dyn, path)

    lm_export(text_lm, 4, os.path.join(a.out, "text_lm.onnx"))
    lm_export(tts_lm, 20, os.path.join(a.out, "tts_lm.onnx"))
    head = Head(inner.prediction_head).eval()
    export(
        head,
        (torch.zeros(2, vv_pack.LATENT), torch.tensor([999.0, 999.0]), torch.zeros(2, vv_pack.HIDDEN)),
        ["noisy", "t", "cond"],
        ["v"],
        {"noisy": {0: "b"}, "t": {0: "b"}, "cond": {0: "b"}, "v": {0: "b"}},
        os.path.join(a.out, "head.onnx"),
    )
    # VibeVoice trims transposed-convolution output with an absolute end index
    # (x.shape[-1] - right), which tracing would freeze at the example's
    # length. Trimming from the end gives the same result for any length.
    import vibevoice.modular.modular_vibevoice_tokenizer as vt

    def unpad1d(x, paddings):
        left, right = paddings
        return x[..., left : (-right if right > 0 else None)]

    vt.unpad1d = unpad1d
    # Its stride-1 convolutions pad by (ideal length - length), always 0, but
    # traced it becomes (example length - length) and crops longer input.
    extra_padding = vt.get_extra_padding_for_conv1d

    def no_extra_for_stride_1(x, kernel_size, stride, padding_total=0):
        return 0 if stride == 1 else extra_padding(x, kernel_size, stride, padding_total)

    vt.get_extra_padding_for_conv1d = no_extra_for_stride_1
    decoder = Decoder(inner.acoustic_tokenizer).eval()
    export(
        decoder,
        (torch.zeros(1, vv_pack.LATENT, 5),),
        ["latents"],
        ["audio"],
        {"latents": {2: "t"}, "audio": {2: "n"}},
        os.path.join(a.out, "decoder.onnx"),
    )

    # ---- 3. the ONNX graphs against PyTorch ------------------------------------
    o = ort.SessionOptions()
    o.intra_op_num_threads = 4
    sess = {n: ort.InferenceSession(os.path.join(a.out, n + ".onnx"), o) for n in ("text_lm", "tts_lm", "head", "decoder")}
    with torch.no_grad():
        for name, mine, layers in (("text_lm", text_lm, 4), ("tts_lm", tts_lm, 20)):
            past = [torch.randn(2, vv_pack.KV_HEADS, 7, vv_pack.HEAD_DIM) for _ in range(2 * layers)]
            x = torch.randn(2, 5, vv_pack.HIDDEN) * 0.1
            pos = torch.tensor([[7.0, 8, 9, 10, 11], [3.0, 4, 5, 6, 7]])
            keep = torch.tensor([[1.0] * 12, [0.0] * 4 + [1.0] * 8])
            ref = mine(x, pos, keep, *past)
            got = sess[name].run(None, lm_feeds(x.numpy(), pos.numpy(), keep.numpy(), [(past[2 * i].numpy(), past[2 * i + 1].numpy()) for i in range(layers)]))
            compare(f"{name}.onnx hidden", got[0], ref[0], 1e-4)
            compare(f"{name}.onnx cache", got[-1], ref[-1], 1e-4)
        noisy, t, cond = torch.randn(2, vv_pack.LATENT), torch.tensor([799.0, 799.0]), torch.randn(2, vv_pack.HIDDEN)
        compare("head.onnx", sess["head"].run(None, {"noisy": noisy.numpy(), "t": t.numpy(), "cond": cond.numpy()})[0], head(noisy, t, cond), 1e-4)
        lat = torch.randn(1, vv_pack.LATENT, 12)
        ref = decoder(lat)
        compare("decoder.onnx", sess["decoder"].run(None, {"latents": lat.numpy()})[0], ref, 1e-3)
        # Streaming decode (what Microsoft's loop does, frame by frame) vs all at once.
        from vibevoice.modular.modular_vibevoice_tokenizer import VibeVoiceTokenizerStreamingCache

        cache = VibeVoiceTokenizerStreamingCache()
        chunks = [
            inner.acoustic_tokenizer.decode(lat[:, :, i : i + 1], cache=cache, sample_indices=torch.LongTensor([0]), use_cache=True)
            for i in range(lat.shape[2])
        ]
        streamed = torch.cat(chunks, dim=-1)
        n = min(streamed.shape[-1], ref.shape[-1])
        d = (ref[..., :n] - streamed[..., :n]).numpy()
        snr = 10 * math.log10(float((ref[..., :n].numpy() ** 2).sum()) / max(float((d**2).sum()), 1e-20))
        log(f"decoder all at once vs frame by frame: {ref.shape[-1]} vs {streamed.shape[-1]} samples, SNR {snr:.1f} dB")

    # ---- 4. small weights, embeddings, tokenizer, scaling --------------------
    sd = {
        "types": inner.tts_input_types.weight,
        "conn_fc1_w": inner.acoustic_connector.fc1.weight,
        "conn_fc1_b": inner.acoustic_connector.fc1.bias,
        "conn_norm_w": inner.acoustic_connector.norm.weight,
        "conn_fc2_w": inner.acoustic_connector.fc2.weight,
        "conn_fc2_b": inner.acoustic_connector.fc2.bias,
        "eos_fc1_w": model.tts_eos_classifier.fc1.weight,
        "eos_fc1_b": model.tts_eos_classifier.fc1.bias,
        "eos_fc2_w": model.tts_eos_classifier.fc2.weight,
        "eos_fc2_b": model.tts_eos_classifier.fc2.bias,
    }
    vv_pack.write_small(os.path.join(a.out, "small.f32"), {k: v.detach().float().numpy() for k, v in sd.items()})
    emb = inner.language_model.embed_tokens.weight.detach().float().numpy()
    emb.astype("<f2").tofile(os.path.join(a.out, "embed.f16"))
    processor.tokenizer.backend_tokenizer.save(os.path.join(a.out, "tokenizer.json"))

    # The scheduler as per-step coefficients (DPM-Solver++ 2M, v-prediction):
    #   x0 = alpha*x - sigma*v;  x' = cs*x + c0*x0 + c1*x0_previous
    sch = copy.deepcopy(inner.noise_scheduler)
    sch.set_timesteps(5)
    sig = [float(s) for s in sch.sigmas]
    coef = {"timesteps": [float(t) for t in sch.timesteps], "alpha": [], "sigma": [], "cs": [], "c0": [], "c1": []}

    def a_s(s):
        al = 1.0 / math.sqrt(s * s + 1.0)
        return al, s * al

    steps = len(coef["timesteps"])
    for k in range(steps):
        al_s, si_s = a_s(sig[k])
        al_t, si_t = a_s(sig[k + 1])
        coef["alpha"].append(al_s)
        coef["sigma"].append(si_s)
        lam = lambda al, si: math.log(al) - math.log(si) if si > 0 else math.inf  # noqa: E731
        h = lam(al_t, si_t) - lam(al_s, si_s)
        e = math.exp(-h) - 1.0
        cs = si_t / si_s
        order = 1 if k == 0 or k == steps - 1 else 2
        if order == 1:
            coef["cs"].append(cs)
            coef["c0"].append(-al_t * e)
            coef["c1"].append(0.0)
        else:
            al_p, si_p = a_s(sig[k - 1])
            r0 = (lam(al_s, si_s) - lam(al_p, si_p)) / h
            coef["cs"].append(cs)
            coef["c0"].append(-al_t * e * (1 + 0.5 / r0))
            coef["c1"].append(al_t * e * 0.5 / r0)
    # Same result as the original scheduler on random data?
    test = copy.deepcopy(inner.noise_scheduler)
    test.set_timesteps(5)
    x = torch.randn(2, vv_pack.LATENT)
    mine = x[0].numpy().astype(np.float64)
    prev = np.zeros(vv_pack.LATENT)
    for k, t in enumerate(test.timesteps):
        v = torch.randn(2, vv_pack.LATENT)
        x = test.step(v, t, x).prev_sample
        x0 = coef["alpha"][k] * mine - coef["sigma"][k] * v[0].numpy()
        mine = coef["cs"][k] * mine + coef["c0"][k] * x0 + coef["c1"][k] * prev
        prev = x0
    compare("scheduler coefficients vs DPMSolverMultistepScheduler", mine, x[0].numpy(), 1e-4)

    cfg = {
        "model": MODEL,
        "vocab": int(emb.shape[0]),
        "hidden": vv_pack.HIDDEN,
        "latent": vv_pack.LATENT,
        "text_layers": 4,
        "tts_layers": 20,
        "sample_rate": 24000,
        "hop": 3200,
        "cfg_scale": vv_pack.CFG,
        "scaling": float(inner.speech_scaling_factor),
        "bias": float(inner.speech_bias_factor),
        "scheduler": coef,
    }
    json.dump(cfg, open(os.path.join(a.out, "vibevoice.json"), "w"), indent=1)

    # ---- 5. voices ---------------------------------------------------------
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_outputs import BaseModelOutputWithPast

    prompts = {}
    meta = []
    for name, preset in VOICES.items():
        p = os.path.join(a.voices, preset + ".pt")
        with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
            try:
                pr = torch.load(p, map_location="cpu", weights_only=True)
            except Exception:
                pr = torch.load(p, map_location="cpu", weights_only=False)  # Microsoft's own file
        prompts[name] = pr

        def kv(c):
            return [(k.float().numpy()[0], v.float().numpy()[0]) for k, v in cache_layers(c)]

        neg_last = pr["neg_tts_lm"].last_hidden_state[0, -1].float().numpy()
        vv_pack.write_voice(os.path.join(a.out, "voices", name + ".bin"), kv(pr["lm"].past_key_values), kv(pr["tts_lm"].past_key_values), kv(pr["neg_tts_lm"].past_key_values), neg_last)
        meta.append({"id": name, "preset": preset, "gender": "woman" if "woman" in preset else "man"})
    json.dump(meta, open(os.path.join(a.out, "voices.json"), "w"), indent=1)

    # ---- 6. the whole loop vs Microsoft's generate(), same noise ------------
    pack = vv_pack.Pack(a.out)
    if a.check_dir:
        os.makedirs(a.check_dir, exist_ok=True)
        cases = [
            TEST,
            "Hello! I can read your books, notes and podcast scripts. Shall we begin?",
            "It's 1905, and Einstein's paper says E equals m c squared; isn't that wild?",
            "  Two  spaces,\ttabs\nand newlines\n\nhere.  ",
            "Wait, really?! Five times... (well, roughly) — 50% more: $3.50 & 12,000 km/s.",
            "Café, naïve, résumé, Zürich, São Paulo, 東京, Привет, 🎙️ emoji.",
            "don't won't they're I'm we'll you'd It'S DON'T",
            "Numbers: 0 1 22 333 4444 55555 3.14159 1e10 2026-10-17",
        ]
        json.dump([{"text": t, "ids": pack.tokens(t)} for t in cases], open(os.path.join(a.check_dir, "tokenizer_cases.json"), "w"), ensure_ascii=False, indent=1)
    for name in ("carter", "emma"):
        noise = []
        real_randn = torch.randn

        def recording(*shape, **kw):
            r = real_randn(*shape, **kw)
            noise.append(r[0].clone())
            return r

        inputs = processor.process_input_with_cached_prompt(
            text=TEST, cached_prompt=prompts[name], padding=True, return_tensors="pt", return_attention_mask=True
        )
        torch.randn = recording
        t0 = time.time()
        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=vv_pack.CFG,
                    tokenizer=processor.tokenizer,
                    generation_config={"do_sample": False},
                    verbose=False,
                    show_progress_bar=False,
                    all_prefilled_outputs=copy.deepcopy(prompts[name]),
                )
        finally:
            torch.randn = real_randn
        took_ref = time.time() - t0
        ref = out.speech_outputs[0].float().numpy().reshape(-1)
        assert inputs["tts_text_ids"][0].tolist() == pack.tokens(TEST), "tokenizer differs"
        voice = vv_pack.read_voice(os.path.join(a.out, "voices", name + ".bin"))
        t0 = time.time()
        lat, got = pack.generate(voice, TEST, lambda i: noise[i].numpy())
        took = time.time() - t0
        n = min(len(ref), len(got))
        err = ref[:n] - got[:n]
        snr = 10 * math.log10(float(np.sum(ref[:n] ** 2)) / max(float(np.sum(err**2)), 1e-20))
        log(
            f"{name}: Microsoft {len(ref) / 24000:.2f} s in {took_ref:.1f} s; ONNX loop {len(got) / 24000:.2f} s "
            f"({len(lat)} frames) in {took:.1f} s; SNR {snr:.1f} dB"
        )
        if a.check_dir:
            import soundfile as sf

            os.makedirs(a.check_dir, exist_ok=True)
            sf.write(os.path.join(a.check_dir, f"{name}-microsoft.wav"), ref, 24000)
            sf.write(os.path.join(a.check_dir, f"{name}-onnx.wav"), got, 24000)
            np.stack([v.numpy() for v in noise]).astype("<f4").tofile(os.path.join(a.check_dir, f"{name}-noise.f32"))
            lat.astype("<f4").tofile(os.path.join(a.check_dir, f"{name}-latents.f32"))
        if abs(len(ref) - len(got)) > 2 * 3200 or snr < 15:
            raise SystemExit(f"the ONNX loop does not reproduce Microsoft's output for {name}")
    log("pack:", sorted(os.listdir(a.out)))


if __name__ == "__main__":
    main()
