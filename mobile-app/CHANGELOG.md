## Changelog

### 1.2.0
- Added Google AI Mode Handler for voice-triggered commands via Android Intents
- Supports: timers, alarms, calls, messages, web search, navigation
- No wake word required - commands detected directly ("set timer for 5 minutes")
- Native Android platform channel (`GoogleAssistantChannel.kt`)
- Command parser supports word numbers ("two hours", "twenty minutes")
- Auto-exit after 30 seconds of inactivity
- 40 unit tests for command parsing
- Falls back to native Google Assistant for unrecognized commands

### 1.1.0
- Added Chat Mode with on‑glasses guidance and 10‑minute auto‑exit.
- Added Interview Mode (uses optional local assets; degrades gracefully if absent).
- Switched Roku remote to on‑glasses text UI.
- Added Quick Note BLE trigger in `Proto.sendQuickNoteBasic`.
- Resolved authorization loop (attach session cookie to WhisperServer requests).
- Resolved vocabulary game loop/hangs (removed external audio; vibration feedback; lifecycle guards).
- Added app‑level `.gitignore` for `.env`, Flutter build artifacts, and private interview files.

