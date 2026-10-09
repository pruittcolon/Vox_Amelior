# Search by meaning, hybrid search and RAG

## Model

**Google EmbeddingGemma 300M** is the best open embedding model under 500M parameters (MTEB multilingual) and is built for phones.

| What | Choice | Why |
|---|---|---|
| Export | `onnx-community/embeddinggemma-300m-ONNX`, `model_quantized` (int8, 309 MB) | It is not gated. Google's LiteRT build needs a Hugging Face login and license approval. int8 is the safest quantization; fp16 activations are unsupported, and q4 needs newer operators. |
| Pinning | One Hugging Face commit, with a SHA-256 for each file | The download can never change underneath the app. |
| Runtime | The ONNX Runtime 1.28.2 already inside sherpa-onnx, through the app's own FFI bindings | No second native library and no APK size increase. The contrib operators the model needs (RotaryEmbedding, MultiHeadAttention) were checked in the phone's library. |
| Tokenizer | Pure-Dart `dart_sentencepiece_tokenizer` on `tokenizer.model` | It is what flutter_gemma uses for EmbeddingGemma. Ids are `[BOS=2, …prompt+text…, EOS=1]`. |
| Prompts | query: `task: search result \| query: `; document: `title: none \| text: ` | From the model card. Retrieval quality drops without them. |
| Where it runs | A worker isolate in the app, started on first use and unloaded after 2 minutes idle | The UI never stutters, and the ~400 MB of memory is freed when search is not in use. The model's memory is native, so unloading asks the worker to release it and waits for the worker to end; killing the worker would leak the whole model. |

## Storing vectors

- **Matryoshka:** keep the first 256 of 768 dimensions and re-normalize. EmbeddingGemma is trained for this, and the quality loss is small.
- **int8:** one scale per vector, 260 bytes a line (100,000 lines is about 26 MB). Similarity stays within about 1% (tested).
- **Table and triggers:** vectors live in `segment_vectors` (schema v7). Triggers drop a vector when its line's text changes (a speaker split), so the line is embedded again; deleted lines take their vector with them.
- **What is embedded:** the line itself. A line under 6 words gets the line before it as context. One-word lines are not embedded.

## Search

- **Words:** SQLite FTS5 with Porter stemming, ranked by BM25 (already in the app).
- **Meaning:** cosine similarity between the query and every stored vector, which is fast enough at phone scale (no ANN index needed). Lines below 0.30 are never matches. Lines more than 0.12 below the best match are dropped too. EmbeddingGemma's scores are compressed (in CI, related lines scored 0.44–0.62 and unrelated ones up to 0.43), so a cut relative to the best match separates them better than any fixed threshold.
- **Query vector:** the last query's vector is kept. Results refresh as new lines arrive, and switching mode re-ranks, without running the model again. While a refresh runs, the current results stay on screen.
- **Fusion:** Reciprocal Rank Fusion with k = 60. It uses only ranks, so BM25 and cosine scores never have to share a scale. It is the standard method in Elastic, Azure AI Search and Weaviate.
- **Filters:** people, tone and period apply to both halves. TV and background voices are always left out.
- **Fallbacks:** smart search falls back to words if the model fails or is not installed.

## RAG (Ask Gemma)

- The app finds the best lines for a question by meaning (up to 12, within a 6 s limit).
- **When Gemma runs in the app:** these "hints" go straight to the retriever.
- **When Gemma runs in the listening service:** the hints are stored on the request (`assistant_requests.hint_ids`), because the service has no embedder.
- The retriever keeps only hints that match the question's person and period. It fuses them with the keyword hits (RRF), adds surrounding lines for context, and trims to Gemma's context budget.
- Questions spoken with the wake word stay keyword-only, since the service has no embedder.

## Indexing

New lines are embedded in batches of 8, newest first, while the app is open: at startup, when data changes, when the app resumes, and when you search. The indexer rests at least as long as each batch took, so it never uses more than half the CPU, and transcription keeps up.

- **Failures:** if the model fails, indexing waits 5 minutes before trying again by itself, so a model that will not load is not reloaded with every new line. Settings shows where it stopped and why, and a tap retries at once.
- **Switching off:** turning meaning search off and on again in the middle of a batch is not a failure. Indexing carries on with the new model.

## Verification

- **Unit and screen tests:** a deterministic fake embedder covers storage, triggers, filters, fusion, fallbacks, the indexer, RAG hints and the search UI.
- **CI `verify-embeddings.yml`:** runs the real model through the Dart pipeline and compares it with Hugging Face's own tokenizer and ONNX Runtime in Python. Token ids must be identical, vectors must reach cosine > 0.999 (on test lines that include emoji, accents, numbers and CJK text), and related lines must score clearly higher than unrelated ones after compact storage.
