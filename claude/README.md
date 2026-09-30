# Vox Amelior for Android — everything in one place

Working notes for the **Android app in `vox_mobile/`**: what it is, how every
part works, which models it uses and how they were converted, how it is built
and released, what was decided and why, and what to know before changing it.

Written for the next person (or the next Claude session) picking this up cold.

> This folder is documentation only. Nothing in `claude/` is built, shipped or
> imported, and changing it does **not** trigger an APK build (the build only
> watches `vox_mobile/**`).

## Get the code and build it

Everything is in this repository on `main`:

| Path | What |
|---|---|
| [`vox_mobile/`](../vox_mobile) | The whole Android app: [`lib/`](../vox_mobile/lib) (code), [`test/`](../vox_mobile/test) (tests), [`android/`](../vox_mobile/android), [`pubspec.yaml`](../vox_mobile/pubspec.yaml) |
| [`.github/workflows/build-android.yml`](../.github/workflows/build-android.yml) | How CI builds and publishes the APK |
| [`.github/workflows/export-parakeet-rnnt.yml`](../.github/workflows/export-parakeet-rnnt.yml) | Converts NVIDIA Parakeet RNNT 1.1B (no longer the app's default; kept as is) |
| [`.github/workflows/export-diarizer.yml`](../.github/workflows/export-diarizer.yml) | Converts NVIDIA Nemotron 3 Diarization for the phone |
| `claude/README.md` | This document |

```bash
git clone https://github.com/pruittcolon/Vox_Amelior.git
cd Vox_Amelior/vox_mobile
# Flutter 3.47.5 (Dart 3.13), Java 17, Android SDK
flutter pub get
flutter analyze && flutter test
flutter build apk --release --target-platform android-arm64
# → build/app/outputs/flutter-apk/app-release.apk
```

No keys or tokens are needed: the app downloads its models itself on first
run (public files, checked by SHA-256). Ready-made APK:
[`vox-android-latest`](https://github.com/pruittcolon/Vox_Amelior/releases/tag/vox-android-latest).

---

## 1. What it is

A **local-first, Android-only** Flutter app. It listens all day, writes down who
said what, and answers questions about it later — or acts on it. Nothing leaves
the phone. The internet is used only to download models and to call webhooks the
user configures.

| Area | What it does |
|---|---|
| Transcription | Always-on. Silero VAD finds speech, **NVIDIA Parakeet TDT 0.6B v2** (int8, via sherpa-onnx; writes punctuation and capitals) writes it down |
| Who is speaking | **NVIDIA TitaNet** voiceprints; up to 5 automatic patterns per person; unknown voices are "Guest N" until named |
| Speaker changes | **NVIDIA Nemotron 3 Diarization** (Sortformer family) splits a line where the speaker changes and marks people talking at once |
| Memory | SQLite with full-text search, grouped into days and conversations |
| Assistant | **Google Gemma 4 E4B** (or E2B / any `.litertlm`) through flutter_gemma / LiteRT-LM, with agent tools |
| Time-aware questions | "last week", "yesterday evening", "3 days ago", "in March" are understood automatically; the Ask tab offers ready-made questions for the chosen period |
| Reviews | Rolls through a whole period part by part (e.g. every logical fallacy last week) and keeps each part's answer |
| Voice clips | Optionally keeps each line's audio for training later |
| Automations | Phrase / speaker rules → signed webhooks, notifications, notes |
| Places | Listen only at home (or pause at work) using low-power location |
| Looks | Light/dark, 8 accent colours, text size, corner style |

Tested on a **Samsung S25**. Needs a 64-bit Android 8+ phone; Gemma 4 E4B wants
about 8 GB RAM (E2B for less).

### Where this sits in the repo

| Path | What |
|---|---|
| `vox_mobile/` | **This app.** Standalone, no server needed |
| `mobile-app/` | An older Flutter companion for the server platform. Separate; untouched |
| `frontend/`, `docker/`, `k8s/`, `docs/`, root `README.md` | The original server-side platform (microservices, web UI). Unrelated to the Android app |
| `.github/workflows/` | `build-android.yml`, `export-parakeet-rnnt.yml`, `export-diarizer.yml` |
| `claude/` | This documentation |

---

## 2. Releases and downloads (what exists on GitHub)

| Release tag | Made by | Contents |
|---|---|---|
| `vox-android-latest` | `build-android.yml` (every push touching `vox_mobile/**`) | `vox-amelior.apk` (~218 MB, arm64). Deleted and recreated each build, so the link is stable |
| `parakeet-rnnt-1.1b-int8` | `export-parakeet-rnnt.yml` | Parakeet RNNT 1.1B converted for sherpa-onnx (no longer used by the app) |
| `nemotron-3-diarization-onnx` | `export-diarizer.yml` | Nemotron 3 Diarization converted to ONNX |

Install: on the phone open the repo's **Releases**, download `vox-amelior.apk`
from **Vox Amelior for Android (latest build)**, open it, allow installing from
the browser.

### Models the app downloads

All hashes and sizes live in `lib/models/model_catalog.dart`; a unit test checks
the catalog is complete. Downloads resume, are verified by SHA-256, and need **no
Hugging Face token** (everything is public or hosted on this repo's releases).

| Model | Size | Source |
|---|---|---|
| Parakeet TDT 0.6B v2 int8 (one `.tar.bz2`; the app keeps `encoder.int8.onnx`, `decoder.int8.onnx`, `joiner.int8.onnx`, `tokens.txt`) | 482,468,385 B download, ~661 MB unpacked | sherpa-onnx releases (`asr-models`, sha256 `157c157b…e1ad`) |
| Parakeet TDT 0.6B v2 **fp16** (optional; one `.tar.bz2`, keeps `encoder.fp16.onnx`, `decoder.fp16.onnx`, `joiner.fp16.onnx`, `tokens.txt`) | 1,120,982,957 B download | sherpa-onnx releases (`asr-models`, sha256 `37f67a1a…4cf0`); chosen in Settings → Microphone & hearing |
| Silero VAD | 643,854 B | sherpa-onnx releases |
| TitaNet small (voiceprints) | 40,257,283 B | sherpa-onnx releases |
| **Speaker changes** — Nemotron 3 Diarization int8 | 107,759,677 B, sha256 `f468ec63…1886` | this repo, `nemotron-3-diarization-onnx` |
| Gemma 4 E4B `.litertlm` | 3,659,530,240 B | `litert-community` on Hugging Face |
| Gemma 4 E2B `.litertlm` (faster) | 2,588,147,712 B | `litert-community` on Hugging Face |
| Custom | any `.litertlm` URL | user supplied |

The speech models (~1.2 GB) are "essential". **Speaker changes** is an extra: it
is queued automatically once the speech models are ready and can also be
downloaded from the Models screen.

---

## 3. Architecture

### 3.1 One background service, one work queue

Everything heavy runs in a foreground service (`lib/service/`). The UI talks to
it through `ServiceController` / `protocol.dart` and shares files through
`shared_files.dart`.

```
microphone → VAD (speech capture) → speech queue (on disk) ─┐
                                                             ├─► ComputeScheduler ─► transcript + speaker
questions (voice or app) ────────────────────────────────────┘                    └► Gemma answer
```

`ComputeScheduler` runs **one job at a time** so the phone is never overloaded.
Priority:

1. **Exclusive** — a question someone is waiting for, the context test.
2. **Transcription** of queued speech.
3. **Background** — review parts, one small step at a time, only when nothing
   else is waiting.

Speech that arrives while Gemma is answering waits safely on disk (`chunk_queue`)
and is transcribed right after; the queue survives the service restarting.

When Vox is not listening, reviews run inside the app instead. A **lease** row in
the database hands work between the two workers (`'service'` / `'app'`) without
ever doing a part twice.

### 3.2 Code map (`vox_mobile/lib`, ~15,400 lines)

| Folder | Purpose |
|---|---|
| `core/` | SQLite (`database.dart`, schema v4, migrations), `log`, `clock`, `idle_timeout` |
| `data/` | Repositories: `transcript_repository`, `speaker_repository`, `clip_store`, `models` |
| `pipeline/` | `speech_capture`, `chunk_queue`, `segment_processor`, `compute_scheduler`, `listening_pipeline`, `speaker_turns`, `engines` (interfaces) |
| `speakers/` | `speaker_identifier`, `voice_patterns`, `enrollment_service`, `embedding_engine`, `vector_math` |
| `assistant/` | `assistant_service`, `prompt_builder`, `agent_tools`, `time_window`, `query_parser`, `retriever`, `context_budget`, `context_probe`, `review_*`, `wake_command` |
| `automation/` | `rule`, `rule_engine`, `template`, `webhook_dispatcher`, `http_sender`, `action_executor` |
| `location/` | `location_monitor`, `location_policy` |
| `models/` | catalog, resumable downloader, archive extractor, installer, store |
| `native/` | Everything that touches native code: `sherpa_engines`, `gemma_llm_engine`, `onnx_runtime` (FFI), `sortformer_diarizer`, `pcm_source`, `voiceprint_worker`, `local_notifier` |
| `service/` | `listening_service`, `listening_runtime`, `service_controller`, `protocol`, `shared_files` |
| `settings/` | `app_settings`, `service_config` |
| `app/` | `app_services` (app-side wiring), `model_downloads`, `assistant_client`, `token_store` |
| `ui/` | Screens (Now, Timeline, Ask, People, More, Models, Settings, Reviews, Voice clips, Appearance, Capacity, Places, Automations, prompt editor) |

Engines are behind small interfaces (`EmbeddingEngine`, `AsrEngine`, `VadEngine`,
`DiarizationEngine`, `LlmEngine`) so the whole pipeline is tested with fakes and
only `lib/native/` needs a device.

### 3.3 Database (SQLite, `user_version` = 4)

Tables: `speakers`, `speaker_samples`, `speaker_negatives`, `unknown_clusters`,
`conversations`, `segments` (+ FTS5 `segments_fts`), `rules`, `outbox`,
`assistant_requests`, `notes`, `reminders`, `voice_clips`, `review_runs`,
`review_chunks`, `review_items`.

| Version | Added |
|---|---|
| v1–v2 | Core transcript, speakers, rules, outbox, requests, notes, reminders |
| v3 | patterns, `speaker_samples.segment_id`, `speaker_negatives`, `voice_clips`, review tables |
| v4 | `segments.overlap` (talking at once) |

The migration re-reads `user_version` **inside** its `BEGIN IMMEDIATE`
transaction, so two processes opening an old database cannot both migrate it.
`repositories_test` upgrades an old-shaped database and opens it twice.

---

## 4. Speech and speakers

### 4.1 Transcription

Silero VAD → `SpeechCapture` cuts segments (max ~20 s) → `ChunkQueue` on disk →
Parakeet. English only; text appears after each sentence, not word by word.

Sensitivity: **Settings → Speech detection sensitivity** (`vadThreshold`, slider
0.2–0.8, default 0.5). Lower (about 0.3) picks up quieter or farther speech; higher ignores more
noise. The user chose to tune this by hand rather than change the default.

Parakeet **TDT 0.6B v2** is the ready-made int8 export from the sherpa-onnx releases. It replaced the RNNT 1.1B model (§7.2), which wrote no punctuation or capitals and was less accurate. On upgrade the app deletes the old ~1.1 GB model folder; the setup screen then offers the 482 MB speech download again (tap **Download speech models**).

### 4.2 Who is speaking (voiceprints)

TitaNet gives each stretch of speech a 192-number voiceprint. The identifier
compares it with every person's stored samples.

| Setting | Default | Meaning |
|---|---|---|
| `matchThreshold` | 0.55 | Minimum similarity to accept a person |
| `matchMargin` | 0.04 | Best must beat the runner-up by this much |
| `guestThreshold` | 0.6 | Unknown voices group into the same "Guest N" above this |

Unknown voices are clustered into Guests so the same stranger keeps the same
label until someone names them.

### 4.3 How it learns (only when a person corrects a line)

Nothing is learned silently. Labelling is what teaches it, and it is stored
permanently (SQLite) — months of use keep working.

- **Samples.** Each corrected line stores its voiceprint against the chosen
  person. Up to **1,000** samples per person (`maxSamplesPerSpeaker`); when over
  the cap the oldest samples learned from corrections are dropped first.
- **Enrollment samples are pinned.** The 3–5 samples recorded when adding a
  person are never dropped, however many later samples arrive, so the original
  reference cannot be pushed out.
- **Patterns.** A person's samples are grouped (adaptive k-means) into up to **5
  patterns** (close up, across the room, …). k is the largest value for which
  every group still has ≥ 10 samples, so groups cannot over-split. A **switch**
  turns patterns off (`multiPatterns`). The identifier scores against the best
  pattern, not one blended average, so a rare voice style is not diluted.
- **Fixing a line moves its sample.** Re-labelling a line takes its sample from
  the old person to the new one; nothing is double-counted.
- **"Not [name]".** Available for *any* person, not just one. It makes the line a
  Guest, and stores a **negative** voiceprint against that person. Similar voices
  then stop matching that person (`negativeFloor` 0.6: a voice is vetoed only when
  it scores at least that high against a negative *and* higher than it scores
  against the person; newest 100 negatives kept per person). The person's own
  voiceprint is never touched. Safeguard: when a line is later confirmed as that
  person, any negative that sounds like it (similarity ≥ 0.75 —
  `negativeConflict`) is deleted, so a mistaken "Not" cannot lock someone out.
- **"New person…"** creates a person from the line (and from that guest's past
  speech) and relabels it.
- **Replay safety test** (`voice_learning_test`): replays lines through the
  identifier to prove patterns never move a line from one person to another.

### 4.4 Speaker changes (diarization)

Voiceprints name whoever spoke a segment, but a segment can hold two people
answering each other. **Nemotron 3 Diarization** decides *when* each voice is
active, at 10 ms resolution, for up to 8 voices in one stretch.

`SegmentProcessor.process()` now returns a **list** of saved lines:

1. If a diarizer is loaded, `splitSpeakers` is on and the audio is ≥ 2 s, run it.
2. `SpeakerTurns` turns the 0–1 activity per slot into turns.
3. Each slot is named from its **solo** audio (overlap excluded) via TitaNet.
4. A slot that cannot be named (solo audio shorter than `minEmbeddingSeconds`) is not a different person: it is merged into its longer neighbour. Neighbouring slots that are the *same person* are then joined (`joinSame`).
5. Each turn is transcribed on its own slice and saved with its own time,
   speaker, and `overlap` flag; voice clips get that turn's own audio.

Turn rules (`SpeakerTurns`): threshold 0.5, ignore bursts < 0.24 s, bridge gaps
< 0.24 s, merge turns < 1.5 s into a neighbour, mark overlap only when two
speakers overlap ≥ 0.24 s (a quick hand-over is not overlap).

**It can never lose a line.** Without the model, switched off
(**Settings → Split lines when the speaker changes**), or if it throws, the line
is saved whole exactly as before. Overlapped lines show "talking at once" under
their time, and Gemma / Reviews see `(over someone else)` after them.

How the model runs without sherpa-onnx support: see §7.3.

---

## 5. Gemma assistant

### 5.1 Engine

`GemmaLlmEngine` (flutter_gemma + LiteRT-LM) with hard-won safety:

- one lock per session with a 5-minute timeout; `_active` count — **never reload
  or unload under a running chat**;
- `withIdleTimeout` (not `Stream.timeout`, which never completes under fake-async
  tests) so a stuck stream fails instead of hanging;
- automatic retries; **CPU fallback** through `recover()` after "code 13" style
  GPU errors;
- context size pinned during the capacity probe.

### 5.2 Agent mode

Tools Gemma can call: `search_conversations`, `get_timeline`, `save_note`,
`list_notes`, `set_reminder`, `run_automation`, `pause_listening`. Tool results
are trimmed to the context budget.

### 5.3 Time-aware questions

`time_window` / `query_parser` understand: last week (previous Mon–Sun), this
week, yesterday (evening), on Monday, 3 days ago, in March… The Ask tab has a
period picker with suggested questions Gemma can answer from that period.

### 5.4 Editing the prompt

The tune button on **Ask** ("How Gemma answers") and an entry in **Settings** open
the prompt editor for Gemma's instructions. Reviews have per-template prompts (§6).

### 5.5 Context size and the phone test

A phone's usable context is much smaller than Gemma's advertised window. **Settings
→ Gemma capacity → Test this phone** (`ContextProbe`) tries 2,048 → 32,768 tokens
(`2048, 4096, 8192, 16384, 32768`), filling the window with text that starts with
a code word and checking Gemma can still repeat it. That catches crashes, errors
and silently dropped text.

- A crash writes a marker with the process id; the next start remembers that size
  as too big.
- The probe has a 12-minute idle timeout and waits for exclusive access.
- Default (untested) context: **4,096**.
- Reply budget = 15 % of context, clamped to 256–1,024 tokens; tool spec costs
  ~700 tokens, system ~350; `charsPerToken` = 3.2.
- **Recommended part size = ½ the tested context** (so instructions, notes and the
  reply fit). Editable, capped by `maxChunkTokens = context − reply − 700`.

---

## 6. Reviews (roll through a period)

For questions that don't fit one prompt ("how many logical fallacies did I say
last week?"):

1. Pick a period (any preset or custom dates) and a template.
2. The transcript is split into parts of the recommended size (e.g. 1,000 of
   10,000 lines).
3. Gemma answers each part; **every answer is stored** and shown in **Reviews**.
4. **List** templates: the app parses and counts findings itself (Gemma is not
   trusted to count), and each finding links to the exact line. **Summary**
   templates merge answers in rounds.

Built-in templates: `fallacies` (prefilled with a working output format),
`promises`, `decisions`, `disagreements`, `facts`, `summary`, and `custom`.
Custom templates are saved in settings.

Runs are database-driven (`review_runs` → `review_chunks` → `review_items`) with
leases, so they resume after a restart and hand off between the service and the
app. They use the background lane, so a question always jumps the queue.

---

## 7. How the models were converted (CI)

### 7.1 Why CI

The sandbox that built this has no GPU and a small disk. The conversions need
NeMo (large) and several GB, so they run on GitHub Actions and publish to a
release. Each job **verifies the converted model against the original before
publishing**, so a bad conversion fails the job instead of shipping.

### 7.2 Parakeet RNNT 1.1B → sherpa-onnx (`export-parakeet-rnnt.yml`)

No longer the app's default: the app uses Parakeet TDT 0.6B v2 (§4.1). The workflow and its release stay as they are.

NeMo `nvidia/parakeet-rnnt-1.1b` → ONNX (encoder / decoder / joiner) → dynamic
int8 quantization → transcribe a real speech sample with sherpa-onnx and check the
words → publish `parakeet-rnnt-1.1b-int8`. The encoder is split into `.onnx` +
`.weights` because of ONNX's 2 GB limit.

### 7.3 Nemotron 3 Diarization → ONNX (`export-diarizer.yml`)

`nvidia/Nemotron-3-Diarization`: ~100 M parameters, `SortformerEncLabelModel`, 31
Transformer layers with RoPE, frame stacking ×8, output upsampled to **10 ms**,
`(1, frames, 8)` speaker probabilities, OpenMDW licence.

**sherpa-onnx has no Sortformer**, so the app runs the ONNX file itself through
the ONNX Runtime **C API over dart:ffi** (`lib/native/onnx_runtime.dart`). sherpa
1.13.8 already bundles `libonnxruntime.so` (ORT 1.28.2) for Android arm64 and
Linux x64; the app loads the same library. Function indices used (API version 17)
are listed in that file; the integration test proves they match ORT 1.28.2.

Export recipe — every step is checked against NeMo's own output:

1. Install NeMo **from source** (`NVIDIA-NeMo/Speech` main). PyPI NeMo lacks RoPE.
2. `MelFeatures` reproduces NeMo's filterbank front end with a **conv-based STFT**
   (complex STFT can't be exported). Checked: features differ < 2e-3
   (measured 6e-5 – 2e-4).
3. Replace NeMo's **FlexAttention** (vmap-built block masks are not traceable)
   with a dense padding mask and `scaled_dot_product_attention` — identical maths.
   Checked: predictions differ < 1e-5 from NeMo.
4. A custom export forward (`Diarizer`) pads with **tensor ops**. NeMo pads inside
   `FeatureStacking` behind a Python `if` that tracing froze to the example length
   (first fully exported model failed with a Reshape error at any other length).
5. `torch.onnx.export` (opset 17, `dynamo=False`, dynamic audio length), then int8
   dynamic quantization of **MatMul only**.
6. Verify on **eight lengths** (2 s – 45 s): shapes match, fp32 ONNX vs torch
   < 1e-3 (measured ≈ 1e-6), int8 makes the same speaker decisions on 99.2–100 %.
7. Sanity: the four-speaker recording separates into ≥ 2 speakers, one speaker
   stays one, int8 agreement > 97 %.
8. Publish `diarizer.int8.onnx`, `diarizer.json`, `SHA256SUMS.txt`.

**After a re-export**, copy the new byte size and sha256 from the release's
`SHA256SUMS.txt` into `_diarizerBytes` / `_diarizerSha` in `model_catalog.dart`;
the catalog test fails until they match.

---

## 8. Voice clips (training data)

**More → Voice clips**: off / everyone / chosen people, with a size limit
(default 2 GB), per-person delete, delete everything, and live usage. When the
limit is reached Vox **stops saving** clips; nothing is deleted automatically.
Clips are WAV files linked to their transcript line; deleting a line clears the
clip's `segment_id` link. Filenames are collision-proof. This is only the collection
step — nothing trains a model yet.

---

## 9. Appearance

**More → Appearance**: system / light / dark, 8 accent colours (`kAccentChoices`),
text size, corner style (square / soft / rounded). `main.dart` builds the theme from
settings, so changes apply instantly.

---

## 10. Settings reference (`AppSettings`, saved as JSON)

| Setting | Default |
|---|---|
| `wakePhrases` | hey vox, ok vox, vox |
| `matchThreshold` / `matchMargin` / `guestThreshold` | 0.55 / 0.04 / 0.6 |
| `vadThreshold` | 0.5 |
| `retentionDays` | 90 |
| `speakReplies` | true |
| `llmId` (+ custom model fields) | gemma-4-e4b-it |
| `agentMode` | true |
| `instructions` | '' (built-in prompt) |
| `locationMode` (`off`, listen only at places, pause at places) + `places` | off |
| `multiPatterns` | true |
| `splitSpeakers` | true |
| `clipMode` (`off`/`everyone`/`chosen`), `clipPeople`, `clipLimitMb` | off, none, 2048 |
| `contextTokens`, `contextTested`, `contextTestNote`, `reviewChunkTokens` | 4096, 0, '', 0 (= ½ of context) |
| `customTemplates` | none |
| `themeMode`, `accent`, `textScale`, `corners` (`square`/`soft`/`rounded`) | system, Indigo, 1.0, rounded |

Speech model paths are in `ServiceConfig`/`SpeechModelPaths` (the diarizer path is
optional and set only when installed).

---

## 11. Automations (webhooks)

Rules match phrases and/or speakers, then send a webhook, notify, or save a note.
Placeholders: `{{text}}`, `{{speaker}}`, `{{match}}`, `{{command}}`, `{{time}}`,
`{{date}}`, `{{rule}}`; add `|json` or `|url` to escape. With a signing secret,
requests carry `X-Vox-Timestamp` and
`X-Vox-Signature: sha256=HMAC(secret, "<timestamp>.<body>")`. Plain `http://` must
be allowed per rule (home-network devices). Redirects are not followed. Delivery
uses an outbox with retries. Gemma can run enabled automations by name.

---

## 12. Build, test, release

### 12.1 Local

```bash
cd vox_mobile
flutter pub get && flutter analyze && flutter test        # Flutter 3.47.5 / Dart 3.13
flutter build apk --release --target-platform android-arm64
```

Currently **217 tests pass, 6 are skipped** (real-model tests that need model
files). `flutter analyze` is clean.

### 12.2 Real-model tests (Linux x64)

```bash
export VOX_MODELS_DIR=/dir/with/encoder.int8.onnx,decoder.int8.onnx,joiner.int8.onnx,tokens.txt,silero_vad.onnx,titanet.onnx,test.wav
export SHERPA_LIB_DIR=$PUB_CACHE/hosted/pub.dev/sherpa_onnx_linux-1.13.8/linux/x64
export VOX_DIARIZER=/path/to/diarizer.int8.onnx
export VOX_MULTI_WAV=/path/to/16k-multi-speaker.wav      # optional
LD_LIBRARY_PATH=$SHERPA_LIB_DIR flutter test test/integration
```

`diarizer_test.dart` checks: one voice = one speaker with correct timing; two
people back-to-back split near the change (the test joins 6 s of one recording to
our speaker; measured turn change 6.49 s, allowed within 1 s of 6.0 s);
and the diarizer coexists with sherpa's sessions in one process, leaving voiceprints
unchanged. Measured here: 7.4 s of audio in ~0.13 s on a laptop CPU.

### 12.3 CI

`build-android.yml` runs on every push to the branch that touches `vox_mobile/**`
(or the workflow): `flutter analyze` → `flutter test` → `flutter build apk` →
publish `vox-android-latest`. `concurrency.cancel-in-progress` means **a newer push
cancels a running build** — batch changes, and don't push trivia while a build
you care about is running. The export workflows are on their own triggers.

Pushing only `claude/` (this folder) or `.github/workflows/export-*.yml` does not
rebuild the APK.

### 12.4 Test suites

`agent`, `assistant`, `automation`, `enrollment_service`, `gemma_resilience`,
`listening_pipeline`, `location_policy`, `models`, `repositories`, `review`,
`scheduler`, `screens_smoke`, `settings`, `speaker_identifier`, `speaker_turns`,
`time_window`, `vector_math`, `voice_learning`, `wake_command`, `widgets`, plus
`integration/` (real models). Fakes live in `test/support/fakes.dart`.

---

## 13. First run (user guide)

1. **Models** (shown automatically): speech models (~1.2 GB) and Gemma 4 (3.7 GB).
   Downloads continue in the background, resume if interrupted, and show progress
   on every tab. **Speaker changes** (108 MB) follows automatically.
2. **People → Add person**: record 3–5 samples (or import WAVs) per person.
3. **Now → Start listening**: allow microphone and notifications; tap **Allow** on
   the battery card so Android doesn't stop Vox.
4. Optional: **More → Places** to listen only at home.
5. Optional but recommended: **Settings → Gemma capacity → Test this phone** once,
   so reviews are sized for the S25.

Browse in **Timeline** (tap a line to fix who said it), ask in **Ask** or say
"Hey Vox, …".

---

## 14. Problems that were hit, and their fixes

| Symptom | Cause | Fix |
|---|---|---|
| Mic stuck on "permissions" | `pcm_source` called `hasPermission()`, which reports false in the background service (no Activity) even when granted | removed it; the UI checks permission before starting |
| Gemma "code: 13" / stuck | GPU init failure and a hung stream | idle timeout, retries, CPU fallback via `recover()`, never reload under an active chat |
| Review vs probe clash | both wanted the model | probe is exclusive; reviews are background; `kickReviews` only after refresh |
| Old database opened twice migrated twice | version read outside the transaction | re-read `user_version` in `BEGIN IMMEDIATE` |
| Clip filename collisions, dangling `segment_id` | naive names | unique names; null `segment_id` on line delete |
| Old profiles had no patterns | patterns added later | `ensurePatterns` on load |
| k-means over-split people | fixed k | adaptive k (largest k with every group ≥ 10) |
| `Stream.timeout` never fires in widget tests | fake-async | `withIdleTimeout` |
| Smoke-test overflows on long names | rigid rows | `Expanded` / `Wrap` / `Flexible` |
| Dart FFI generics | `asFunction` needs concrete types | explicit typedefs in `onnx_runtime.dart` |
| Diarizer export: no RoPE | PyPI NeMo too old | install NeMo from source |
| Diarizer export: `create_block_mask` crash | FlexAttention not traceable | dense mask + SDPA |
| Diarizer ONNX: Reshape error at other lengths | tracing froze `FeatureStacking`'s pad `if` | pad with tensor ops in the export forward |

---

## 15. Known limits

- English only (Parakeet TDT 0.6B v2).
- Text arrives per sentence, not word by word.
- Diarization only splits **within** one segment (≤ ~20 s). Segments under 2 s are
  not split; a turn under 1.5 s is merged into its neighbour, so a quick "yeah"
  may share a line. A line is only cut when every part is at least 1.5 s and has at
  least 2 words, and the parts are different *named* people; otherwise it stays as
  the fast pass saved it. Up to 8 voices per stretch.
- Overlapping speech is **marked**, but each word still goes to one line: when two
  people talk over each other, the transcript is only as good as Parakeet is at
  mixed audio.
- The `vox_mobile/README.md` limitations list still says overlap is attributed to
  one voice; it predates diarization.
- Android forbids starting the microphone after a reboot; open Vox once.
- The database is **not encrypted at rest**; rely on the phone's encryption and lock
  screen. Webhook secrets are stored in the app's private database.
- Phone context is small; use the capacity test and let reviews chunk.
- No model is trained on-device yet; voice clips only collect data.

---

## 16. Working notes for whoever continues (human or Claude)

- **Branch:** all work is on `claude/laughing-cerf-zer2dj`. Don't push elsewhere.
  No pull request unless asked.
- **Secrets:** a Hugging Face token is not needed and must never be committed or sent
  anywhere except huggingface.co. Don't put personal emails in commits or code.
- **Commits** end with the `Co-Authored-By` and `Claude-Session` trailers the
  harness specifies. Don't put model names in commits, code or docs.
- **Batch changes.** The owner tests on the phone from the release APK and dislikes
  waiting on repeated builds: finish a whole change, run `flutter analyze` and
  `flutter test`, do a once-over for bugs, *then* push.
- **Answer first.** When asked a question, answer it before doing anything. When told
  "just explain" / "don't do", change nothing.
- **Style:** short, plain replies. Beautiful and easy-to-use UI is a stated goal —
  preferences should make looks easy to change.
- **Shell quirks:** `.github/` is git-ignored here, so use `git add -f` for workflow
  files. Flutter is not on the default `PATH` in the sandbox.
- **Verify, don't assume:** export jobs compare against the original model at every
  step; keep it that way when adding models. Prefer proving with the real model on
  real audio over reasoning about it.

### Ideas not done

- Train or fine-tune voiceprints from saved clips.
- Stream the diarizer for very long segments.
- A one-tap "test speaker changes" on the phone that reports speed.
- Update `vox_mobile/README.md` (limitations list) — left alone here so this
  documentation change does not trigger another APK build.

---

## 17. History of this build (short)

1. Core: database, models and downloads, speaker ID, pipeline, rules, assistant
   (pure Dart, fully tested).
2. Native adapters (sherpa-onnx, Gemma, microphone), foreground service, screens.
3. CI: build and publish the APK; phone testing on an S25.
4. Parakeet RNNT 1.1B converted in CI (later replaced by TDT 0.6B v2); mic-permission fix.
5. Voice clips, voice learning (patterns, pinned enrollment, "Not [name]", "New
   person…", replay safety), Gemma resilience, prompt editor, time-aware questions,
   rolling reviews with a phone context test, appearance, error review pass.
6. Speech sensitivity tuned by the user via the existing setting.
7. Nemotron 3 Diarization converted in CI, run through the ONNX Runtime C API, and
   wired into the pipeline: split lines at speaker changes, mark overlap.

---

## 18. Two-stage lines (fast text, then speakers per chunk)

Each line is handled in two passes (`SegmentProcessor.transcribe` / `refine`):

1. **Fast**, about 2 s after a sentence ends, the same as before speaker
   splitting: Parakeet transcribes the whole sentence once (with word start
   times) and TitaNet gives a first voice match. The line is saved and shown right
   away, and rules **without** a person fire on it.
2. **Chunk**: recent lines are gathered until a 3 s pause or 30 s of speech. The
   diarizer runs over the whole chunk (more context, better separation). A line
   where the speaker changes is cut at word start times (no re-transcription),
   and each part is named from that voice's solo audio across the chunk. The
   first part updates the fast line in place and the other parts are added. Rules
   **about a person** fire on these finished lines. Each rule sees a line once.

Lines corrected by hand before the chunk pass are left alone. Without the
diarizer (or with splitting switched off) a line finishes immediately. If the
service stops mid-chunk, lines keep their fast labels.

The **wake word is off by default** (the old "hey vox" default is migrated to
off). Add one in Settings → Wake word to ask Gemma out loud.

---

## 19. Settings you can tune

**Settings** opens a home page with five categories, each showing a live summary. Every slider can be
dragged, nudged with − / +, or typed exactly (tap the value), applies when you let go, and shows a reset
arrow when it differs from the default. Listening settings apply **while Vox is running** (the service
re-reads them on `reload`; changing sensitivity, pause or short-sound length restarts only the speech
detector).

| Setting | Default | Range | What it does |
|---|---|---|---|
| Mic boost (`micGain`) | +15% (1.15) | −50% … +300% | Multiplies every sample (clamped at full scale) before detection, transcription and voice matching |
| Sensitivity (`vadThreshold`, shown inverted) | 50% | 20–80% | How easily a sound counts as speech |
| Pause that ends a sentence (`pauseSeconds`) | 0.6 s | 0.3–1.5 s | Silence needed to close a line |
| Ignore short sounds (`minSpeechSeconds`) | 0.3 s | 0.1–1.0 s | Drops coughs and clicks |
| Speech model (`speechModel`) | Standard (int8) | int8 / fp16 | fp16 downloads on first selection; int8 stays as the fallback until it is ready |
| How sure before naming (`matchThreshold`) | 55% | 30–90% | Higher = fewer wrong names |
| Lead over runner-up (`matchMargin`) | 4% | 0–30% | Best match must beat the second by this much |
| Grouping unknown voices (`guestThreshold`) | 60% | 30–90% | How alike strangers must sound to share a "Guest" |
| Shortest part of a line (`splitMinSeconds`) | 1.5 s | 0.8–3.0 s | A line is only split when every part is this long |
| Fewest words in a part (`splitMinWords`) | 2 | 1–5 | …and has this many words |

Presets on the microphone card set boost and sensitivity together: **Sensitive** (+50%, 60%),
**Balanced** (+15%, 50%), **Noise-proof** (none, 35%). **Restore recommended settings** (Settings home)
resets all of the above except the speech model.

**Live level meter.** While the microphone card (Settings, or the *Mic* button on Now) is open, the
app sends `meter` to the service, which sends a `level` event about five times a second (loudness after
the boost on a −60…0 dBFS scale, a peak, and whether the sound is clipping). The bar shows a "good"
zone and says whether to raise or lower the boost. Nothing is sent when no screen is showing it.

**Speech model switch.** `AppSettings.asrAsset` picks the recognizer; `SpeechModelPaths.fromStore`
uses it when installed, otherwise the standard int8 model. When the service sees a new encoder path it
swaps the recognizer without restarting (`SegmentProcessor.swapAsr`).

