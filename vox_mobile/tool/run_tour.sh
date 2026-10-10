#!/usr/bin/env bash
# Runs the on-device tour on the emulator that CI started (app-tour.yml).
# Serves the recorded conversation to the app, grants what a person would
# allow when asked, takes a screenshot whenever the tour asks for one, and
# keeps the device log.
set -uo pipefail
PKG=com.voxamelior.vox_amelior_mobile
OUT=build/tour
mkdir -p "$OUT/screenshots"

# The conversation, at http://10.0.2.2:8765 from inside the emulator.
python3 -m http.server 8765 --directory build/tour_audio > "$OUT/http.txt" 2>&1 &

adb logcat -c || true
adb logcat -v time > "$OUT/logcat.txt" 2>&1 &

# Screenshots on request: the tour logs "TOUR_SHOT <name>" and waits.
( adb logcat -v raw -s flutter:I | while IFS= read -r line; do
    case "$line" in
      *TOUR_SHOT\ *)
        name="${line##*TOUR_SHOT }"
        name="${name//[^A-Za-z0-9_-]/}"
        adb exec-out screencap -p > "$OUT/screenshots/$name.png" ;;
    esac
  done ) &

# Microphone and notifications (asked for when listening starts) and
# background activity: Android accepts these once the app is installed.
( for _ in $(seq 1 1800); do
    if adb shell pm list packages 2>/dev/null | grep -q "$PKG"; then
      sleep 2
      adb shell pm grant "$PKG" android.permission.RECORD_AUDIO || true
      adb shell pm grant "$PKG" android.permission.POST_NOTIFICATIONS || true
      adb shell dumpsys deviceidle whitelist +"$PKG" || true
      echo "granted permissions"
      break
    fi
    sleep 1
  done ) &

flutter drive --profile \
  --driver=test_driver/integration_test.dart \
  --target=integration_test/app_tour_test.dart \
  -d emulator-5554 2>&1 | tee "$OUT/drive.txt"
status=${PIPESTATUS[0]}

adb exec-out screencap -p > "$OUT/screenshots/zz-last.png" || true
adb shell dumpsys meminfo "$PKG" > "$OUT/meminfo.txt" 2>&1 || true
exit "$status"
