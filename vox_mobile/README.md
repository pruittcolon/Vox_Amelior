# Vox Amelior — Android

A private, always-on assistant that runs entirely on your phone. It listens,
writes down who said what, and answers questions about it later.

| Feature | How |
|---|---|
| Speech to text | NVIDIA Parakeet TDT 0.6B (int8) via sherpa-onnx |
| Speech detection | Silero VAD |
| Who is speaking | NVIDIA TitaNet voiceprints; unknown voices become "Guest N" until you name them |
| Memory | SQLite on the phone with full-text search, grouped into conversations |
| Questions | Google Gemma 3n (E4B or E2B) via LiteRT-LM, answering from your transcripts |
| "Hey Vox, …" | Spoken questions are answered with a notification (and read aloud) |
| Automations | Rules on phrases/speakers → webhooks (signed, retried), notifications, notes |

Nothing is uploaded. The internet is used only to download models and to call
webhooks you configure.

## Install

1. On your phone, open the repository's **Releases** page and download
   `vox-amelior.apk` from **Vox Amelior for Android (latest build)**.
2. Open it and allow installing apps from your browser.
3. Needs a 64-bit phone on Android 8 or newer. Gemma needs about 6 GB of RAM
   (choose E2B on smaller phones).

## First run

1. **Download speech models** (~520 MB). Keep Vox open until they finish; a
   paused or interrupted download resumes where it stopped.
2. **Assistant (optional):** Gemma is gated by Google.
   - Create a free Hugging Face account, open the Gemma model page from the app,
     and accept the licence.
   - Create a *Read* token at huggingface.co/settings/tokens and paste it in
     the app. It is stored in Android's encrypted storage.
   - Download Gemma (3–4 GB).
3. **People → Add person.** Record 3–5 samples (or import WAV files) for each
   person, e.g. you and your wife.
4. **Live → Start listening.** Allow microphone and notifications, and allow
   Vox to ignore battery optimisation so Android does not stop it.

Tap any line in the transcript to correct who said it; Vox learns from
corrections. Voices it does not know appear under **People → Voices Vox has
heard**; name them once and their past lines are relabelled.

## Automations

Placeholders in URLs, bodies and notifications: `{{text}}`, `{{speaker}}`,
`{{match}}`, `{{command}}`, `{{time}}`, `{{date}}`, `{{rule}}`. Add `|json` or
`|url` to escape, e.g. `{"msg": "{{text|json}}"}`.

With a signing secret, requests carry `X-Vox-Timestamp` and
`X-Vox-Signature: sha256=HMAC(secret, "<timestamp>.<body>")`.
Plain `http://` must be allowed per rule (for devices on your home network).
Failed deliveries retry with backoff (up to 6 attempts) and are listed under
**Automations → Deliveries**.

## Development

```bash
flutter pub get
flutter analyze
flutter test
flutter build apk --release --target-platform android-arm64
```

Real-model tests (Linux x64) run the actual Parakeet, Silero and TitaNet models:

```bash
export VOX_MODELS_DIR=/path/with/encoder.int8.onnx,decoder...,titanet.onnx,silero_vad.onnx,test.wav
export SHERPA_LIB_DIR=$PUB_CACHE/hosted/pub.dev/sherpa_onnx_linux-<ver>/linux/x64
LD_LIBRARY_PATH=$SHERPA_LIB_DIR flutter test test/integration
```

Code layout: `lib/core` (database), `lib/data` (repositories), `lib/pipeline`
(audio → transcript), `lib/speakers` (voiceprints), `lib/assistant`
(retrieval + prompting), `lib/automation` (rules, webhooks), `lib/models`
(downloads), `lib/native` (sherpa-onnx, Gemma, microphone adapters),
`lib/service` (always-on foreground service), `lib/ui` (screens).

## Known limitations

- English only (Parakeet v2).
- Text appears after each sentence, not word by word.
- Two people talking over each other in one sentence are attributed to one voice.
- Voice questions are answered while the app is running in the background; if
  Android has closed it, tap the notification to get the answer.
- The database is not encrypted at rest; rely on the phone's own encryption and
  screen lock. Webhook secrets are stored in the app's private database.
