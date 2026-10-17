#!/usr/bin/env bash
# Downloads every candidate for the voice lab into DIR:
#   kokoro/ supertonic/ pocket/   what the app ships (from the narrator-models release)
#   supertonic-fp32/              Supertone's full-precision originals (Hugging Face)
#   pocket-fp32/                  the full-precision Pocket TTS pack (sherpa-onnx)
#   kitten/                       Kitten TTS mini v0.8 (sherpa-onnx)
#   clarity/                      LavaSR bandwidth extension, ONNX (Topping1/LavaSR-ONNX)
#   lavasr-ref/                   the reference Python implementation, to check ours
#   refs/                         reference clips for Pocket TTS
set -euo pipefail
DIR=${1:?usage: fetch_lab.sh DIR}
BASE=https://github.com/pruittcolon/Vox_Amelior/releases/download/narrator-models
SHERPA=https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models
HF=https://huggingface.co/Supertone/supertonic-3/resolve/main/onnx
LAVA=https://github.com/Topping1/LavaSR-ONNX/releases/download/Alpha-v0.1
get() { # url, destination
  if [ ! -s "$2" ]; then
    mkdir -p "$(dirname "$2")"
    curl -sSfL --retry 4 -o "$2.part" "$1" && mv "$2.part" "$2"
  fi
}
tarball() { # name, destination folder
  if [ ! -d "$2" ]; then
    mkdir -p "$DIR/tmp"
    curl -sSfL --retry 4 "$SHERPA/$1.tar.bz2" | tar xj -C "$DIR/tmp"
    mv "$DIR/tmp/$1" "$2"
  fi
}

get "$BASE/kokoro-fp32--model.onnx" "$DIR/kokoro/model.onnx"
for f in voices.bin tokens.txt lexicon-us-en.txt lexicon-gb-en.txt; do get "$BASE/kokoro--$f" "$DIR/kokoro/$f"; done
if [ ! -d "$DIR/kokoro/espeak-ng-data" ]; then
  get "$BASE/kokoro--espeak-ng-data.zip" "$DIR/kokoro/espeak.zip"
  (cd "$DIR/kokoro" && unzip -q espeak.zip && rm espeak.zip)
fi
for f in duration_predictor.int8.onnx text_encoder.int8.onnx vector_estimator.int8.onnx vocoder.int8.onnx tts.json unicode_indexer.bin voice.bin; do
  get "$BASE/supertonic--$f" "$DIR/supertonic/$f"
done
for f in duration_predictor text_encoder vector_estimator vocoder; do
  get "$HF/$f.onnx" "$DIR/supertonic-fp32/$f.onnx"
done
for f in lm_flow.int8.onnx lm_main.int8.onnx encoder.onnx decoder.int8.onnx text_conditioner.onnx vocab.json token_scores.json; do
  get "$BASE/pocket--$f" "$DIR/pocket/$f"
done
tarball sherpa-onnx-pocket-tts-2026-01-26 "$DIR/pocket-fp32"
tarball kitten-mini-en-v0_8 "$DIR/kitten"
for f in enhancer_backbone.onnx enhancer_backbone.onnx.data enhancer_spec_head.onnx enhancer_spec_head.onnx.data; do
  get "$LAVA/$f" "$DIR/clarity/$f"
done
get https://raw.githubusercontent.com/Topping1/LavaSR-ONNX/main/lavasr_core.py "$DIR/lavasr-ref/lavasr_core.py"
get "$BASE/pocket--bria.wav" "$DIR/refs/bria.wav"
for v in voice-zero-caro-davy voice-zero-peter-yearsley alba-mackenna-a-moment-by; do
  get "$BASE/voice--$v.wav" "$DIR/refs/$v.wav"
done
rm -rf "$DIR/tmp"
ls -l "$DIR"/pocket-fp32 "$DIR"/kitten
du -sh "$DIR"/*
