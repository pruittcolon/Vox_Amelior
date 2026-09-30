# Vox Amelior — Android

A private, always-on assistant that runs entirely on your phone. It listens,
writes down who said what, and answers questions about it later — or acts on
them.

| Feature | How |
|---|---|
| Speech to text | NVIDIA **Parakeet TDT 0.6B v2** (int8) via sherpa-onnx; writes punctuation and capitals. (The 1.1B RNNT model is no longer the default: it wrote no punctuation and was less accurate.) |
| Speech detection | Silero VAD |
| Who is speaking | NVIDIA TitaNet voiceprints with up to 5 automatic voice patterns per person (close up, across the room, …); unknown voices become "Guest N" until named |
| Speaker changes | NVIDIA **Nemotron 3 Diarization** (Sortformer family, converted to ONNX by CI): quick back-and-forth is split into one line per person, and people talking at the same time are marked (108 MB, downloads after the speech models) |
| Teaching voices | Tap any line: pick the right person (the sample moves with it), "Not [name]" (a guest; similar voices stop getting that name, the person's voiceprint is untouched), or "New person…" |
| Memory | SQLite on the phone with full-text search, grouped into days and conversations |
| Assistant | Google **Gemma 4 E4B** via LiteRT-LM (E2B or any `.litertlm` URL selectable) |
| Agent mode | Gemma 4 can search conversations, read a day/week timeline, save notes, set reminders, run automations, pause listening |
| Time-aware questions | "last week" (previous Mon–Sun), "this week", "yesterday evening", "on Monday", "3 days ago", "in March", … — pick a period on the Ask tab for ready-made questions |
| Go through everything | Reviews a whole period part by part (e.g. all logical fallacies last week, all promises this month); each part's answer is saved, findings are counted by the app and link to the exact line |
| Gemma capacity | "Test this phone" measures the largest context Gemma handles; review parts use half of it |
| Voice clips | Optional: keep each line's audio (WAV) with its text for training later — everyone or chosen people, with a size limit |
| Look | Light/dark, 8 colours, text size and corner style (More → Appearance) |
| "Hey Vox, …" | Spoken questions are answered with a notification and read aloud |
| Places | Listen only at home (or pause at work) using low-power location |
| Automations | Rules on phrases/speakers → signed webhooks with retries, notifications, notes |

Nothing is uploaded. The internet is used only to download models and to call
webhooks you configure.

## How the work is scheduled

Everything heavy runs in one background service with a single work queue:

```
microphone → speech detector → speech queue (on disk) ─┐
                                                        ├─► one job at a time ─► transcript + speaker
questions (voice or app) ───────────────────────────────┘                     └► Gemma answer
```

Transcription and Gemma never run at the same moment, so the phone is not
overloaded. While Gemma answers, new speech waits safely on disk and is
transcribed right after. The queue also survives the service being restarted.

Priority: a question someone is waiting for → transcription → review parts
(one small step at a time, only when nothing else is waiting). When Vox is not
listening, reviews run in the app instead; a lease in the database hands them
between the two without doing any part twice.

## Context size and reviews

Gemma reads a fixed amount at once (its context window: instructions,
transcript, tool results and its reply together). Phones differ, so
**Settings → Gemma capacity → Test this phone** tries 2,048 → 32,768 tokens,
filling the window with text that starts with a code word and checking Gemma
can still repeat it (this catches crashes, errors and silently dropped text).
If a size crashes the app, the next start remembers it as too big. Everything
else is sized from the result; each review part is half of it.

## Install

1. On your phone, open the repository's **Releases** page and download
   `vox-amelior.apk` from **Vox Amelior for Android (latest build)**.
2. Open it and allow installing from your browser.
3. Needs a 64-bit phone on Android 8 or newer. Gemma 4 E4B needs about 8 GB of
   RAM; choose E2B on phones with less.

## First run

1. **Models** (shown automatically): download the speech models (~1.2 GB) and
   Gemma 4 (3.7 GB). Downloads continue in the background, show progress on
   every tab, and resume if interrupted. No Hugging Face token is needed.
2. **People → Add person**: record 3–5 samples (or import WAV files) for each
   person.
3. **Now → Start listening**: allow microphone and notifications. Tap *Allow*
   on the battery card so Android does not stop Vox.
4. Optional: **More → Places** to save Home and listen only there.

Browse everything in **Timeline** (pick a day, open a conversation, tap a line
to fix who said it). Ask in **Ask** or say "Hey Vox, …".

## Automations

Placeholders: `{{text}}`, `{{speaker}}`, `{{match}}`, `{{command}}`, `{{time}}`,
`{{date}}`, `{{rule}}`; add `|json` or `|url` to escape. With a signing secret,
requests carry `X-Vox-Timestamp` and
`X-Vox-Signature: sha256=HMAC(secret, "<timestamp>.<body>")`. Plain `http://`
must be allowed per rule (home-network devices). Redirects are not followed.
Gemma can run enabled automations by name in agent mode.

## Development

```bash
flutter pub get && flutter analyze && flutter test
flutter build apk --release --target-platform android-arm64
```

Real-model tests (Linux x64) run the actual sherpa-onnx models:

```bash
export VOX_MODELS_DIR=/dir/with/encoder.int8.onnx,decoder.int8.onnx,joiner.int8.onnx,tokens.txt,silero_vad.onnx,titanet.onnx,test.wav
export SHERPA_LIB_DIR=$PUB_CACHE/hosted/pub.dev/sherpa_onnx_linux-<ver>/linux/x64
LD_LIBRARY_PATH=$SHERPA_LIB_DIR flutter test test/integration
```

Layout: `lib/core` (database), `lib/data` (repositories), `lib/pipeline`
(capture, disk queue, scheduler, transcription), `lib/speakers`,
`lib/assistant` (time windows, retrieval, prompts, agent tools),
`lib/automation`, `lib/location`, `lib/models` (downloads), `lib/native`
(sherpa-onnx, Gemma, microphone), `lib/service` (always-on runtime), `lib/ui`.

## Known limitations

- English only (Parakeet TDT 0.6B v2; the older 1.1B RNNT is no longer the default).
- Text appears after each sentence, not word by word.
- Two people talking over each other in one sentence are attributed to one voice.
- Android does not allow starting the microphone after a reboot; open Vox once.
- The database is not encrypted at rest; rely on the phone's encryption and lock
  screen. Webhook secrets are stored in the app's private database.
