#!/usr/bin/env bash
# Downloads Kokoro, Supertonic HD, Pocket TTS HD and Clarity from the
# narrator-models release into DIR/<pack>, laid out like the app stores them.
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
for f in encoder.onnx text_conditioner.onnx vocab.json token_scores.json; do get "pocket--$f" "$DIR/pocket/$f"; done
for f in lm_flow.onnx lm_main.onnx decoder.onnx; do get "pocket-fp32--$f" "$DIR/pocket/$f"; done
get pocket--bria.wav "$DIR/pocket/test_wavs/bria.wav"
for f in tts.json unicode_indexer.bin voice.bin; do get "supertonic--$f" "$DIR/supertonic/$f"; done
for f in duration_predictor.onnx text_encoder.onnx vector_estimator.onnx vocoder.onnx; do get "supertonic-fp32--$f" "$DIR/supertonic/$f"; done
for f in enhancer_backbone.onnx enhancer_backbone.onnx.data enhancer_spec_head.onnx enhancer_spec_head.onnx.data; do
  get "clarity--$f" "$DIR/clarity/$f"
done
du -sh "$DIR"/*
