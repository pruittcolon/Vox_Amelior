# Even Suite

A friendlier version of the open-source [Even Realities EvenDemoApp](https://github.com/even-realities/EvenDemoApp)
for the G1 glasses, with Claude and Termux built in. No API keys, and no packages beyond the demo's own
(only `crclib` is used; speech, settings and HTTP come from Android and the Dart SDK).

## What it does

| | |
|---|---|
| **Glasses** | Find, connect and auto-reconnect your G1 (remembers the pair, keeps trying if the link drops). |
| **Claude** | Talk or type to Claude Code running in Termux (through Agent Hub). Replies appear on the glasses, 5 lines a page. |
| **Hands-free** | Say “Claude” (“clod”, “claud”, “cloud” at the start also work), your message, then “send”. Turns itself off after 2–60 minutes; the notification has a Stop button. |
| **Talk button** | Tap the mic, speak, say “send” (or tap the arrow). Or hold the left TouchBar and let go to send. |
| **Scrolling** | Tap left/right TouchBar, the ▲ ▼ buttons, the volume keys (optional), or say “Claude, scroll down”. |
| **Terminal** | Watch and drive the Claude Code / shell sessions in Termux, with Esc, Enter, arrows, Ctrl‑C and 1/2/3 keys; mirror it to the glasses. |
| **Demo features** | Text pages, notifications (allowed-apps list + send), and BMP images, sent fast with flow-controlled Bluetooth writes. |

## Install

1. On the phone, open the latest release **even-suite-latest** of this repo and install `even-suite.apk`.
2. In Termux, Agent Hub must be at `~/agent-hub` (see `termux/agent-hub/`). Start it with `all`, or let the app do it:
   - add `allow-external-apps = true` to `~/.termux/termux.properties`, then `termux-reload-settings`;
   - in Android Settings › Apps › Even Suite › Permissions, allow **Run commands in Termux**;
   - tap **Start in Termux** on the Glasses tab.

## Develop

Build and test on GitHub, not on the phone: `.github/workflows/even-suite.yml` runs the Agent Hub API
tests (fake Claude, private tmux), `flutter analyze`, all app tests against simulated glasses and a live
test bridge, renders every screen to PNG (artifact `even-suite-screenshots`), then builds the signed APK.
