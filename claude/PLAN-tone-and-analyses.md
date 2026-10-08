# Plan: tone of voice, sorting by emotion, Gemma analyses, easy transfer

Goal: go back through what was said at home and understand it. Find who said what, how it was said, and what happened in arguments. Results are saved in a list you can come back to, and moving data in and out of the app should be easy.

## 1. Tone of voice per line (emotion)

**How:** each line's audio is classified on the phone by SenseVoice Small (FunAudioLLM), run through sherpa-onnx, which the app already ships.
- It returns an emotion tag: happy, sad, angry, neutral, fearful, disgusted or surprised.
- It also returns a sound tag: laughter, music, applause, crying, coughing or sneezing.
- It is an optional download of 163 MB (int8 export `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17`).

**Why this model:**
- It judges the *voice*, not the words. "Fine." said angrily is angry, and a text model would miss that.
- It needs no new native library or tokenizer, because sherpa-onnx 1.13.8 already exposes `emotion` and `event` on every result.
- It is much more accurate on real speech than the tiny distilled models; the 10 MB emotion2vec student detects only about 14 % of the teacher's non-calm moments.
- It also hears **music and background noise**, which helps with the "movie playing" problem.

Rejected alternatives:
- Text-only sentiment (DistilBERT or RoBERTa) would need a Dart tokenizer and misses tone.
- Asking Gemma per line is far too slow and uses too much battery.
- wav2vec2 emotion classifiers (95 MB) offer only 4 to 7 classes and no sound events.

**Where it runs:** after each line is saved (`SegmentProcessor._afterSave`), in the background service that already does the transcription. That includes lines split at speaker changes. Audio is capped at 15 s per line. A failure never loses the line.

**Storage:** schema v6 adds the columns `segments.emotion` and `segments.sound`. These are new columns only, so updating keeps all data.

## 2. Sort and filter by emotion

- Conversation screen: each line shows a small tone chip (😠 angry, 😊 happy, ...). A tone filter row sits next to the people chips.
- Timeline: a "Tone" filter finds conversations with angry or sad lines, and conversation cards show a mood summary such as "3 angry · 1 sad".
- Search (`SegmentQuery.emotions`) can be narrowed to tones.

## 3. Gemma analyses that are kept in a list ("Reviews", extended)

The app already has Reviews: Gemma reads transcripts part by part, writes findings in a fixed format that the app parses (exact counts, each linked to its line), and keeps every run in a list that survives restarts. It already has templates for fallacies, disagreements, promises, decisions and summaries. New in this round:

- **Choose lines, not only a period:**
  - "The last N lines" (for example 100) said by the chosen people, such as Pruitt and Ericah, in time order. Both sides of the conversation are kept.
  - Optionally only lines with certain tones, for example angry or sad.
  - TV and background voices are always left out.
- **Tone given to Gemma:** each line Gemma reads carries its tone, e.g. `[12] 20:31 Pruitt (angry): ...`. Gemma can therefore tell a fight from a joke.
- **New built-in templates:**
  - *Fights & tension*: what started it, how it escalated, how it ended.
  - *Who said what in arguments*: each person's position.
  - *Kind words*: appreciation and support, so the analysis isn't only negative.
- **Answer list:** every run and its findings stay in the Reviews list, and can be copied or shared as text.

## 4. Transfer and copy/paste

- Every line can be selected and copied: long-press it, then choose **Copy line**.
- A conversation can be shared through Android's share menu.
- "Save all transcripts to a file" and "Import transcripts" already exist (round 3). The export now includes the tone tags, and the import reads them back.
- Updates now install over the old app (permanent signing key), so data stays.

## 5. Tests (run in CI on every push)

- The v5 → v6 upgrade keeps every line; the new columns start empty.
- Tone stored per line, including for split lines; a failing tone model never loses a line.
- Tone tags parsed from SenseVoice output, covering unknown or empty tags and `<|EMO_UNKNOWN|>`.
- Emotion filters in search and in conversations; mood summary counts.
- Review scope "last N lines by these people": the right lines, oldest first, background excluded, tone filter.
- Review prompt lines include the tone.
- The export/import round trip keeps the tone.
- Widget tests: tone chips and filter, and the review creator's new scope controls.

## Later (not built yet)

- A full backup file that also holds voiceprints and audio clips.
- Mood over time (a chart per person per week).
- Auto-marking a voice as "background" when SenseVoice keeps hearing music under it.
