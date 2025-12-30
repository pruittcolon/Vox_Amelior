# Even Demo Client

A Flutter demo client that showcases how the WhisperServer backend connects to smart glasses
for real-time transcription, memory recall, and multimodal feedback.

The project bundles BLE control flows, Deepgram/Whisper integrations, Roku remote commands,
and a lightweight UI for triggering backend workflows.

---

## Quick Start

```bash
cd EvenDemoApp-main
flutter pub get
flutter run  # choose your target device/emulator
```

The repository ships with a non-secret `.env` asset so the app loads without extra steps.
Update the values before distributing real builds.

---

## Configuration

Environment variables are loaded from the bundled `.env` file (see `pubspec.yaml`).

| Key | Purpose | Default |
| --- | ------- | ------- |
| `MEMORY_SERVER_BASE` | WhisperServer FastAPI endpoint | `http://YOUR_SERVER_IP:8000` |
| `WHISPER_SERVER_BASE` | Audio ingestion endpoint | `http://YOUR_SERVER_IP:8000` |
| `ASR_SERVER_BASE` | Optional ASR service | `http://YOUR_SERVER_IP:8000` |
| `WHISPER_CHUNK_SECS` | Audio chunk size in seconds | `30` |
| `DEEPGRAM_API_KEY` | Deepgram realtime transcription | _blank_ |
| `OPENAI_API_KEY` | Follow-up responses | _blank_ |
| `VOICEMONKEY_TRIGGER_URL` | Alexa Voice Monkey | _blank_ |
| `ROKU_BASE_URL` | Roku remote integration | _blank_ |
| `N8N_SERVICE_URL` | n8n Integration Service (Port 8011) | `http://YOUR_SERVER_IP:8011` |

To customize per environment, create a copy (e.g. `.env.prod`) and load it manually in
`lib/main.dart` before building release builds.
> **Note**: The backend now uses `/api/v1` prefixes. Ensure your `.env` reflects this.

---

## Scripts & Commands

| Command | Description |
| ------- | ----------- |
| `flutter run` | Run on the selected device/emulator |
| `flutter build apk` | Produce an Android APK |
| `flutter test` | Execute widget/unit tests |
| `flutter analyze` | Lint the project |

---

## Project Structure

```
lib/
├─ app.dart            # Root widget and theme wiring
├─ controllers/        # GetX controllers for features and BLE interaction
├─ services/           # API, BLE, Deepgram/Whisper helpers, protocol serializers
├─ views/              # Feature screens (BMP, vocabulary, voice enrollment, etc.)
├─ widgets/            # Reusable UI widgets and dialog components
└─ utils/              # Utility extensions and formatters
```

Assets live under `assets/` and include BMP samples, animations, sounds, and vocabulary files.

---

## Testing & Quality

flutter test
flutter analyze


## Security Features

- **Secure Credential Storage**: Uses `flutter_secure_storage` for session tokens
  - Android: Encrypted SharedPreferences with AES encryption + KeyStore
  - iOS: Keychain Services with accessibility options
- **Session Management**: Tokens cleared on logout
- **No Plaintext Storage**: Credentials never stored in SharedPreferences

## Known Limitations

- BLE flows assume the dual-arm connection model documented by the hardware vendor.
- Many features rely on optional cloud APIs (Deepgram, OpenAI, Voice Monkey). Leave the env
  values empty to disable them in offline demos.

---

Contributions are welcome—keep PRs focused, add Flutter tests where it makes sense, and make
sure the backend API contract is respected.

---

## Features (Summary)

- Authorization loop fix: attaches session cookie to WhisperServer for transcription/health.
- Vocabulary game reliability: uses vibration feedback; guards against lifecycle races.
- Chat Mode: wake with “Chat”, remembers context, auto‑exit after 10 minutes, say “Terminate” to exit.
- Interview Mode: wake with “Interview”, optionally uses `assets/interview/` files if present.
- Roku remote: renders a text UI on the glasses.
- Quick Note: `Proto.sendQuickNoteBasic()` BLE trigger.

---

## Google AI Mode (v1.2.0)

Voice-triggered commands executed via native Android Intents. **No wake word required** - press the glasses button and speak a command directly.

### Supported Commands

| Command | Example | Action |
|---------|---------|--------|
| Timer | "set timer for 5 minutes" | Opens Clock app with timer |
| Alarm | "set alarm for 7 AM" | Opens Clock app with alarm |
| Call | "call mom" | Opens Phone dialer |
| Message | "text john hello" | Opens Messages app |
| Search | "search for weather" | Opens browser search |
| Navigation | "navigate to downtown" | Opens Google Maps |

### Word Numbers Supported

The parser understands word numbers: "two hours", "twenty minutes", "a minute"

### How It Works

```
User speaks → Deepgram STT → CommandParser → Android Intent → Native App
```

All processing is **local** (no API limits, no cloud calls for execution).

### Files

| File | Purpose |
|------|---------|
| `GoogleAssistantChannel.kt` | Native Android platform channel |
| `google_assistant_service.dart` | Flutter service wrapper |
| `command_parser.dart` | NL command parsing |
| `google_ai_mode_handler.dart` | Mode handler integration |

## Quick Start (Supplement)

```
cd EvenDemoApp-main
cp .env.example .env   # update values for your setup
flutter pub get
flutter run
```

## Optional Interview Data (Do Not Commit)

Create locally if desired:
- `assets/interview/resume.txt`
- `assets/interview/projects.txt`

These files are ignored by git and are not required for the app to run.

## Changelog

See `CHANGELOG.md` for recent updates.
