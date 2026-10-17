#!/usr/bin/env bash
# Runs integration_test/device_test.dart on the emulator CI started
# (narrator-device.yml). Serves the voice files to the app, grants what a
# person would allow when asked, screenshots on "TOUR_SHOT <name>", keeps logs.
set -uo pipefail
PKG=com.vox.narrator
OUT=build/device
MODELS=${MODELS:?set MODELS to the folder with the release assets}
mkdir -p "$OUT/screenshots"

# The voice files, at http://10.0.2.2:8765 from inside the emulator.
python3 -m http.server 8765 --directory "$MODELS" > "$OUT/http.txt" 2>&1 &

adb logcat -c || true
adb logcat -v time '*:W' 'flutter:I' 'AudioTrack:I' > "$OUT/logcat.txt" 2>&1 &
adb shell settings put global hide_error_dialogs 1 || true

( adb logcat -v raw -s flutter:I | while IFS= read -r line; do
    case "$line" in
      *TOUR_SHOT\ *)
        name="${line##*TOUR_SHOT }"
        name="${name//[^A-Za-z0-9_-]/}"
        adb exec-out screencap -p > "$OUT/screenshots/$name.png" ;;
    esac
  done ) &

( for _ in $(seq 1 1800); do
    if adb shell pm list packages 2>/dev/null | grep -q "$PKG"; then
      sleep 2
      adb shell pm grant "$PKG" android.permission.POST_NOTIFICATIONS || true
      adb shell pm grant "$PKG" android.permission.RECORD_AUDIO || true
      echo "granted permissions"
      break
    fi
    sleep 1
  done ) &

flutter test integration_test/device_test.dart -d emulator-5554 -r expanded 2>&1 | tee "$OUT/test.txt"
status=${PIPESTATUS[0]}
adb exec-out screencap -p > "$OUT/screenshots/zz-last.png" || true
adb shell dumpsys meminfo "$PKG" > "$OUT/meminfo.txt" 2>&1 || true
grep "DEVICE " "$OUT/test.txt" > "$OUT/summary.txt" || true
exit "$status"
