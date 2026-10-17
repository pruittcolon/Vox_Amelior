#!/usr/bin/env bash
# Stages Qualcomm's AI-chip runtime for the APK (64-bit ARM only):
#   libonnxruntime_qnn.so   ONNX Runtime 1.29 with the QNN execution provider
#                           (renamed, so it never clashes with sherpa-onnx's own
#                           libonnxruntime.so; the app opens it by name)
#   libQnnHtp*.so           the Hexagon NPU backend, with the v79 stub and skel
#                           for the Snapdragon 8 Elite (Galaxy S25)
#   libQnnGpu.so            the Adreno GPU backend
# Sources: Maven Central (com.microsoft.onnxruntime:onnxruntime-android-qnn and
# the com.qualcomm.qti:qnn-runtime version it was built against).
set -euo pipefail
ORT=1.29.0
QNN=2.42.0
OUT=${1:-android/app/src/main/jniLibs/arm64-v8a}
M=https://repo1.maven.org/maven2
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
curl -sSfL --retry 4 -o "$TMP/ort.aar" "$M/com/microsoft/onnxruntime/onnxruntime-android-qnn/$ORT/onnxruntime-android-qnn-$ORT.aar"
curl -sSfL --retry 4 -o "$TMP/qnn.aar" "$M/com/qualcomm/qti/qnn-runtime/$QNN/qnn-runtime-$QNN.aar"
mkdir -p "$OUT" "$TMP/ort" "$TMP/qnn"
unzip -q -o "$TMP/ort.aar" 'jni/arm64-v8a/libonnxruntime.so' -d "$TMP/ort"
unzip -q -o "$TMP/qnn.aar" -d "$TMP/qnn" \
  'jni/arm64-v8a/libQnnHtp.so' 'jni/arm64-v8a/libQnnHtpPrepare.so' 'jni/arm64-v8a/libQnnHtpV79Stub.so' \
  'jni/arm64-v8a/libQnnHtpV79Skel.so' 'jni/arm64-v8a/libQnnSystem.so' 'jni/arm64-v8a/libQnnGpu.so'
cp "$TMP/ort/jni/arm64-v8a/libonnxruntime.so" "$OUT/libonnxruntime_qnn.so"
patchelf --set-soname libonnxruntime_qnn.so "$OUT/libonnxruntime_qnn.so"
cp "$TMP"/qnn/jni/arm64-v8a/*.so "$OUT/"
ls -l "$OUT"
