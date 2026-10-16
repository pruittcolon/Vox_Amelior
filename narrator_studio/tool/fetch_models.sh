#!/usr/bin/env bash
# Downloads the Kokoro (fast) and Pocket TTS files from the narrator-models
# release into DIR/kokoro and DIR/pocket, laid out like the app stores them.
set -euo pipefail
DIR=${1:?usage: fetch_models.sh DIR}
BASE=https://github.com/pruittcolon/Vox_Amelior/releases/download/narrator-models
get() { # asset, destination
  if [ ! -s "$2" ]; then
    mkdir -p "$(dirname "$2")"
    curl -sSfL --retry 4 -o "$2.part" "$BASE/$1" && mv "$2.part" "$2"
  fi
}
get kokoro-fp32--model.onnx "$DIR/kokoro/model.onnx"
for f in voices.bin tokens.txt lexicon-us-en.txt lexicon-gb-en.txt; do get "kokoro--$f" "$DIR/kokoro/$f"; done
if [ ! -d "$DIR/kokoro/espeak-ng-data" ]; then
  get kokoro--espeak-ng-data.zip "$DIR/kokoro/espeak.zip"
  (cd "$DIR/kokoro" && unzip -q espeak.zip && rm espeak.zip)
fi
for f in lm_flow.int8.onnx lm_main.int8.onnx encoder.onnx decoder.int8.onnx text_conditioner.onnx vocab.json token_scores.json; do
  get "pocket--$f" "$DIR/pocket/$f"
done
get pocket--bria.wav "$DIR/pocket/test_wavs/bria.wav"
du -sh "$DIR"/*
