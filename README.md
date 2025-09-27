# Obvix Wellness by Obvix Labs

**Obvix Wellness** is an experimental emotional chat companion with a strong emphasis on safety, designed and built by Obvix Labs. It integrates a Flask backend API, advanced hybrid retrieval augmented generation (RAG) using MongoDB, Google Gemini 2.5 Flash router for detailed state analysis, and a local Ollama-served chat model, all orchestrated to provide sensitive, therapeutic-style responses.

---

## Project Overview

* **Project Name:** Obvix Wellness
* **Organization:** Obvix Labs
* **Model Hubs:**

  * Instruction-tuned adapter: [hf.co/thtskaran/emoai](https://hf.co/thtskaran/emoai)
  * Pretrained adapter (PT base): [hf.co/thtskaran/ob-wl-g3-pt](https://hf.co/thtskaran/ob-wl-g3-pt)
* **Disclaimer:** This prototype is *not medical advice* nor a substitute for professional care. All safety logic should be validated before real user deployment.

---

## Key Components & Architecture

* **Backend API:** Flask
* **Database:** MongoDB (stores chat history, semantic and episodic memories, therapeutic knowledge base chunks)
* **Router and Embeddings:** Google Gemini 2.5 Flash for conversation state analysis and embedding generation
* **Local Chat Model Serving:** **Ollama (mandatory)** with custom Modelfiles supporting lightweight, quantized LoRA adapters
* **Models:**

  * LoRA adapters and quantized GGUF models optimized for efficient inference hosted on Hugging Face
  * Alpaca-style templates and stop tokens are pre-configured
* **Hybrid Retrieval:** Combines cosine similarity with BM25, enhanced with topic bias towards therapeutic guideline content (primarily CBT/DBT)
* **Therapeutic Knowledge Base:** Implemented using Google Drive files as an optional, automatically syncable source of therapy-related documents and rulebooks

---

## Runtime Flow

1. User sends a `POST /chat` request with JSON `{ user_id, message }`.
2. The message is logged into MongoDB chat history.
3. Router LLM (Gemini 2.5 Flash) performs:

   * Crisis detection (suicidality, psychosis, abuse, dysregulation)
   * Extraction of semantic and episodic memories
   * Sentiment, emotion, sarcasm, and volatility analysis
4. Messages flagged as crisis bypass the full pipeline and receive a predefined compassionate response template immediately.
5. For regular messages, hybrid retrieval fetches contextual memories and optional therapeutic knowledge base chunks from Google Drive (if configured), weighted by relevance and CBT/DBT topic boost.
6. The local Ollama-served chat model receives a structured JSON prompt including memories, retrieved KB, and emotion summaries, then generates a JSON response.
7. The assistant reply is saved and returned to the user.

**Note:** The router LLM currently performs a single call per message. Future plans involve batching to reduce API calls and latency.

---

## Installation & Setup

### System Requirements

* Python 3.10+
* MongoDB 6.x+ for persistent storage
* Outbound HTTPS access for Google API (embeddings, router)
* **Ollama installation and run** is *mandatory* for local chat model inference

### Installation (Ubuntu example)

```bash
sudo apt update && sudo apt install -y python3-dev python3-venv build-essential
```

### Set up environment and dependencies

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install flask pymongo python-dotenv transformers google-api-python-client google-auth pypdf python-docx
```

### Ollama Model Serving

* Ollama is a **required** dependency to run any chat model inference. No alternative serving mechanism is currently supported.
* Load Modelfiles tailored for your chosen model:
  Example to set up the Pretrained (PT) model:

  ```bash
  ollama create ob-wl-pt -f training/Modelfile-pt
  ollama run ob-wl-pt
  ```
* The PT Modelfile directs the model with concise therapeutic instructions avoiding excess sycophancy, resulting from skipping heavy RLHF phases. It works best embedded in the full orchestration with router logic rather than standalone chat.

---

## Environment Variable Configuration

Example `.env` file:

```ini
GEMINI_KEY=YOUR_GEMINI_API_KEY
OLLAMA_URL=http://127.0.0.1:11434
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=emoai
LOG_FILE=logs.log
PORT=5001
FLASK_DEBUG=1

# Therapeutic Knowledge Base (Google Drive KB)
GDRIVE_FOLDER_ID=
GOOGLE_CREDENTIALS_PATH=client.json

# Retrieval tuning
KB_OVERLAP=0
KB_TOPK=3
MEMORY_TOPK=5
CHUNK_SIZE=1500

# Model and API configuration
EMBED_MODEL=gemini-embedding-001
ROUTER_MODEL=gemini-2.5-flash
CHAT_MODEL=ob-wl-pt
MAX_TOKENS=135
```

---

## Model Training & Artifact Summary

### Instruction-Tuned Adapter (Gemma3-4b base)

* Trained on roughly 14,000 therapeutic conversation rows
* LoRA fine-tuning done with Unsloth FastLanguageModel and TRL SFTTrainer with assistant-only loss
* Early stopping and cosine learning rate scheduler in use
* Validation perplexity: 1.5503 after 3 epochs
* Artifacts available: LoRA adapters, 4-bit GGUF models, and Modelfile for Ollama deployment at [hf.co/thtskaran/emoai](https://hf.co/thtskaran/emoai)

### Pretrained Adapter (Gemma3-4b PT base)

* Skips resource-heavy RLHF phase to reduce sycophantic behavior, promoting groundedness
* Validation perplexity: 1.7113
* LoRA adapters and 8-bit GGUF models with Modelfile provided, hosted at [hf.co/thtskaran/ob-wl-g3-pt](https://hf.co/thtskaran/ob-wl-g3-pt)
* Best used embedded within orchestration; no standalone direct chat Modelfile provided for PT adapter
* Future work planned to fine-tune adapters on broader conversational datasets for improved chat style and generalization

---

## Therapeutic Knowledge Base (Google Drive)

* Optional integration of a therapeutic knowledge base via Google Drive documents shared with a service account
* Auto syncs via Drive Changes API in an incremental background thread: uploads, modifications, deletions reflected in knowledge chunks and embeddings
* Documents chunked (~1500 characters plus configurable overlap), embedded with Gemini embeddings, stored in MongoDB
* Retrieval weighted heavily towards CBT/DBT relevant content with topic boost factors
* Drives therapy-specific guidance and contextual facts into the chat interaction

---

## Technical Addendum

### API Reference

#### `POST /chat`

**Request body**

```json
{
  "user_id": "string (required)",
  "message": "string (required)"
}
```

**Success response**

```json
{
  "message": "assistant reply (≤2 sentences, ≤40 words)",
  "meta": { "analysis": { /* minimal router JSON */ } }
}
```

**Errors**

* `400` – `{"error":"user_id and non-empty message required"}`
* `500` – `{"error":"router_llm_failed" | "context_retrieval_failed" | "ollama_failed", "details":"..."}`
* All responses are JSON.

**Special behaviors**

* **Greeting short-circuit:** very short greetings return a warm “Hey … how are you doing today?” without running the full pipeline.
* **Crisis short-circuit:** if the router flags a crisis (`suicidality|psychosis|abuse|dysregulation`), the endpoint returns a predefined supportive template immediately.

---

### Memory System

* **Collections**

  * `semantic_memory`: long-lived user facts (`name|preference|relation|profession`) with optional embeddings.
  * `episodic_memory`: high-level event summaries (+ optional emotion mention) with embeddings.
  * `chat_history`: all user/assistant turns.
  * `kb_sources`: Google Drive file metadata and content hash.
  * `kb_chunks`: chunked text + embeddings for retrieval.
  * `settings`: generic app settings (e.g., Drive changes page token).

* **Write paths**

  * Router returns minimal memory JSON; the app **persists semantic** (upsert by `{user_id,type,value}`) and **episodic** (append) memories and **adds embeddings** immediately.

* **Indexes (created at boot)**

  * `chat_history(user_id, time desc)`
  * `semantic_memory(user_id)`
  * `episodic_memory(user_id)`
  * `kb_chunks(source_id)`

* **Top-K memory retrieval**

  * Retrieves **both** semantic and episodic memories, ranks with hybrid scoring (see Retrieval Math), filters low-relevance items, returns `MEMORY_TOPK`.

---

### Emotion / Sarcasm Analysis

* **Optional pipelines** (loaded best-effort; app runs if unavailable):

  * Sentiment: `cardiffnlp/twitter-roberta-base-sentiment-latest`
  * Emotion: `j-hartmann/emotion-english-distilroberta-base` (returns all scores)
  * Sarcasm: `helinivan/english-sarcasm-detector`

* **Emotional velocity**

  * Compute sentiment per message, then an **EMA of absolute deltas** (`alpha=0.6`) over recent user turns + current one.
  * Buckets:

    * `<0.25` → *calm and stable*
    * `<0.55` → *showing gentle shifts*
    * `<0.85` → *somewhat agitated or volatile*
    * `≥0.85` → *highly volatile and changing rapidly*
  * Output is a plain-English summary used as context.

---

### Retrieval Scoring Math

**Tokenization**

* Lowercased alphanumerics, tokens of length ≥3.

**Cosine similarity**

* Standard normalized dot product on **Gemini embeddings**.

**BM25-lite**

$$
\text{score}=\frac{\mathrm{tf}\times(k_1+1)}{\mathrm{tf}+k_1\cdot(1-b+b\cdot \frac{dl}{avgdl})}
$$

* Memories: `k1=1.2`, `b=0.75`, `avgdl=50`
* KB chunks: `avgdl=400` (others same)

**Hybrid scores**

* **Memories:** `0.6 * cosine + 0.4 * BM25` → filter `> 0.1` → top-K.
* **KB:** `0.5 * cosine + 0.2 * BM25 + topic_boost`

  * `topic_boost = 0.4 * (# matched CBT/DBT tags)`

---

### Knowledge Base (Google Drive) Sync & Chunking

* **Supported sources**

  * Google Docs (exported as text), PDF (`pypdf`), DOCX (`python-docx`), Markdown, plain text; other types fallback to UTF-8 decode.

* **Chunking**

  * Chunk size: `CHUNK_SIZE` (default 1500 chars).
  * Overlap: `KB_OVERLAP`.

* **Tagging**

  * Title tokens become tags; `cbt` / `dbt` auto-added when present in title.

* **Embeddings**

  * Each chunk embedded via **Gemini** and stored in `kb_chunks`.

* **Drive watcher**

  * Polls the Changes API approximately every **20s**.
  * Only processes files in the configured folder.
  * Handles new/updated/deleted files; uses **content hash** to skip unchanged files.
  * Service account requires Drive **readonly** scope; share the folder with that account.

---

### Router & System Prompt Contracts

* **Router (Gemini)**

  * Returns strict JSON:

    * `crisis: { flag: bool, type: "none|suicidality|psychosis|abuse|dysregulation" }`
    * `memories: { semantic: [{ type, value, relation?, name? }], episodic: [{ summary }] }`
  * Minimal facts only; crisis true **only** on imminent risk signals.

* **System prompt for the Ollama model**

  * Receives a structured JSON context:

    * `chat_history`: last 5 user/assistant pairs
    * `relevant_memories`: hybrid-ranked memory snippets
    * `emotion_analysis`: one-line state + velocity summary
    * `kb_snippets`: short clinical handbook snippets (CBT/DBT-leaning)
    * `user_input`: current user message
  * Must return **strict JSON**: `{"message":"..."}`.
  * The app enforces response brevity independently of model compliance.

---

### Local Generation (Ollama)

* **Endpoint**: `POST {OLLAMA_URL}/api/generate`
* **Model**: from `CHAT_MODEL` (default `ob-wl-pt`)
* **Options**:

  * `num_predict = MAX_TOKENS` (default 135)
  * `temperature = 0.2`
* **Output handling**:

  * Robust JSON extraction (handles accidental code fences, nested JSON, or plain text).
  * Safe fallback message when parsing fails.
  * **Brevity enforcement**: ≤2 sentences, ≤40 words.

---

### Environment Variables (Brief Explaination)



* **Core**

  * `GEMINI_KEY` (required)
  * `OLLAMA_URL` (default `http://127.0.0.1:11434`)
  * `MONGODB_URI` (default `mongodb://localhost:27017`)
  * `MONGODB_DB` (default `emoai`)
  * `PORT` (default `5001`)
  * `FLASK_DEBUG` (`1/true` enables Flask reloader & threaded debug server)

* **Models**

  * `EMBED_MODEL` (default `gemini-embedding-001`)
  * `ROUTER_MODEL` (default `gemini-2.5-flash`)
  * `CHAT_MODEL` (default `ob-wl-pt`)
  * `MAX_TOKENS` (default `135`)

* **KB / Drive**

  * `GDRIVE_FOLDER_ID`
  * `GOOGLE_CREDENTIALS_PATH` (service account JSON path)
  * `KB_OVERLAP` (default `0`)
  * `KB_TOPK` (default `3`)
  * `MEMORY_TOPK` (default `5`)
  * `CHUNK_SIZE` (fixed at 1500 by code;)

* **Logging**

  * `LOG_FILE` (default `<repo>/logs.log` `/prompts.log` (for model-prompt analysis))

---

### Logging & Observability

* **Rotating logs**: `logs.log` (1 MB × 3), used by Flask/Werkzeug.
* **Chat trace log**: JSONL records to `test.txt` (prompts, retrieval counts, model outputs length, etc.).
* **Request/response hooks**: logs method, path, `user_id` (if present), and status codes.

---

### Installation Notes (Transformers Models)

If you intend to enable sentiment/emotion/sarcasm analysis:

* Ensure PyTorch is installed (CPU or CUDA).
* First run will download the three models listed above.
* The app **gracefully degrades** (skips these analyses) if pipelines fail to initialize.

---

### Tuning Knobs

* **Retrieval**

  * `KB_TOPK` / `MEMORY_TOPK`: balance context size vs. latency.
  * `KB_OVERLAP` / `CHUNK_SIZE`: control chunk boundaries.
  * Topic biasing: CBT/DBT tag presence boosts KB scores by `0.4` per matched tag.

* **Response style**

  * `MAX_TOKENS` and post-generation brevity enforcement shape reply length and density.

---

### UX Behaviors

* **Greeting detection**: very short greetings (≤3 tokens) from a curated list short-circuit unless “problem keywords” are present.
* **Strict JSON contract**: the model must return `{"message":"..."}`; the app will parse robustly and fall back safely if needed.
* **Consistent brevity**: enforced even if the model returns longer text.

---

## Current Limitations & Roadmap

- Fully reactive system currently; ongoing work to evolve proactive, intent-aware, and safety-layered orchestration  
- Planned improvements include:  
  - Autonomous topic detection replacing keyword heuristics with zero-shot classification  
  - Classical ML model pre-filter for fast crisis detection (shadow mode initially)  
  - Intent classification for routing (small-talk, information requests, therapeutic inquiries, personal questions)  
  - Dynamic retrieval policy adjustments post-intent classification  
  - Rolling online adaptation smoothing user mood and intent signals  
  - Batch request support for router LLM calls to lower latency and cost  
  - Migration to advanced vector DB backends (Qdrant/Weaviate)  
  - Enhanced safety features: NSFW classifiers, red-team phrase matchers, model guardrails  
  - Dynamic model selection depending on conversation context and type  
  - Periodic embedding backfill jobs for legacy data and model upgrades  
  - Improved hyperparameter tuning and multi-domain chat data-adapter training  

***

## Contribution & Contact

Contributors and collaborators are invited to participate and help evolve Obvix Wellness.

- Please open Pull Requests on the code repositories, or  
- Contact directly via email: [hello@Karanprasad.com](mailto:hello@Karanprasad.com)

***

This project aims to advance a safe, emotionally intelligent AI companion integrating real-time retrieval, advanced conversational orchestration, and continuous research to improve groundedness and therapeutic efficacy.