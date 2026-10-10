# Naming voices, switching models, and the emulator tour

## Naming voices

Before this round, a line's sheet named only that line. Naming a voice with 40 lines meant opening the conversation, scrolling, and tapping 40 times.

- **A whole voice in one tap.** A guest voice (a cluster of lines Vox could not name) is named all at once: every line it said, in every conversation. The person learns from all of them, using the existing `assignClusterToSpeaker` and `promoteCluster`.
- **Undo.** `SpeakerRepository.nameVoice`, `nameVoiceAsNew` and `undoNaming` keep a snapshot of the guest and its line ids. Undo restores the guest and gives back the lines that were not corrected since. It removes the samples the naming added, and removes a person the naming created.
- **What you need to recognise a voice.** `TranscriptRepository.voicesToName` returns, per voice: its line count, when it was last heard, its clearest lines (longer ones first, then the newest) and the named people it was heard with.
- **Where naming happens:**
  - **A conversation** lists its voices without a name at the top, each with "Who's this?". A guest's name on its lines is also a button.
  - **The line sheet** is compact chips. It offers "Name this voice" (the whole voice) first, then "just this line is…".
  - **People and Now** offer "Who's this?" for every unnamed voice, one after another, with "Not now" to skip one.
  - **Timeline cards** say how many voices in that conversation still need a name.
- **Long conversations** build only the lines on screen. A line opened from search, Now or a review starts a quarter of the way down the screen; the lines before it are laid out upwards (a `CustomScrollView` with a `center` sliver).

## Models in one place

- **Grouped by job:** hearing speech, tone of voice, search by meaning, assistant. Each choice is a radio row that shows its size and state: in use, downloading (and "switches when ready"), downloaded, or failed.
- **Tap to switch.** A missing model downloads first, and the current one keeps working until the new one is ready. Downloads over 1 GB are confirmed first. Models no longer in use can be deleted.
- **Search by meaning sizes** (EmbeddingGemma 300M, the onnx-community export pinned to one commit):

  | Size | File | Download | CI speed (x86) |
  |---|---|---|---|
  | Small | `model_q4.onnx` (4-bit) | 202 MB | 34 ms per line |
  | Standard (default) | `model_quantized.onnx` (8-bit) | 314 MB | 70 ms per line |
  | Full precision | `model.onnx` (fp32) | 1.24 GB | 37 ms per line |

  Each size keeps its own vectors, so switching prepares every line again. Switching back to a size that is still installed is instant. Deleting a size deletes its vectors. CI (`verify-embeddings.yml`) runs every size through the app's Dart pipeline against Hugging Face's reference: identical token ids, vectors at cosine ≥ 0.9996, and the same related-versus-unrelated separation.

## Smoothness

- **Hidden tabs wait.** While listening, a line arrives every few seconds. Tabs that are not shown, and pages covered by another, now only note that they are out of date (`RefreshWhenShown`). They reload once when shown. Before this, Insights recomputed all its statistics, and People its lists, on every line.
- **Unfiltered meaning search** no longer lists every line id first; it skips only the few TV and background lines.

## The emulator tour (CI: `app-tour.yml`)

The real app runs on an Android emulator (x86_64, KVM, profile build), driven the way a person would use it:

1. First-run setup with the real model downloads.
2. Tone and search switched on in Models.
3. Listening: a recorded two-person conversation (Piper TTS voices "ryan" and "amy") is placed in the speech queue, and the real Parakeet, TitaNet, Sortformer and SenseVoice models process it.
4. The two voices named through "Who's this?", choosing each name by what the voice said.
5. Timeline, a conversation, and search by meaning, then the same search again after switching to the small search model.
6. Insights, Ask, People, every settings page, and dark mode.
7. A year-long archive (about 24,000 lines), with frame times measured while scrolling the busiest screens, opening a 400-line conversation, and while a new line arrives.

The tour logs `TOUR_SHOT <name>`. `tool/run_tour.sh` then takes a real screenshot with `adb`, so screenshots show exactly what a phone shows. Screenshots, frame timings (`tour_perf.json`), the device log and the drive log are kept as the run's `vox-tour` artifact.
