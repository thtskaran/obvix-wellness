# Obvix Wellness (made by Obvix Labs)

An experimental, safety-first emotional chat companion. Backend is a Flask API with lightweight RAG (MongoDB), a Gemini 2.5 Flash router for state analysis and embeddings that support memories, and an Ollama-served local chat model. Trained adapters and a 4-bit quantized prototype are available on Hugging Face.

Important: This prototype is not medical advice and not a replacement for professional care.



## Using the Ollama Modelfile

- ollama create emoai-sarah -f training/Modelfile
- ollama run emoai-sarah

Notes:
- The Modelfile pulls from hf.co/thtskaran/emoai. Ensure your GPU/CPU meets requirements for the chosen GGUF.
- Stop tokens and a lightweight Alpaca-style template are pre-configured.

## Training summary

- Data: ~14,000 conversation rows
- Method: Unsloth LoRA fine-tuning on Google Colab A100
- Duration: ~3 hours end-to-end (prep + SFT)
- Key stack:
  - Unsloth FastLanguageModel for efficient 4-bit load + LoRA
  - TRL SFTTrainer with packing and assistant-only loss
  - Early stopping and cosine scheduler
- Artifacts:
  - LoRA adapters pushed to: https://hf.co/thtskaran/emoai
  - A prototype 4-bit .gguf for easy inference via llama.cpp/Ollama

Important compatibility note:
- Always merge a LoRA with the same base model family it was trained on.
- If you change base families (e.g., from Llama to Gemma), retrain or use matching adapters.

## Merging and quantization (notebook flow)

- Loading a base model and training LoRA with Unsloth.
- Merging LoRA into base weights (Peft merge_and_unload).
- Converting merged HF weights to GGUF via llama.cpp.
- Optional quantization (e.g., Q4_K_M).
- Uploading GGUF back to the HF repo.

If you only want to infer locally:
- Use the provided Modelfile with Ollama (see above).
- Or directly download the GGUF from the HF repo.


## License

- Content and model artifacts in this project are released under CC BY-NC-SA (Attribution-NonCommercial-ShareAlike).
- You may remix and share non-commercially with attribution and the same license. Commercial use requires separate permission.

## Acknowledgments

- Unsloth, TRL, Transformers, llama.cpp
- Hugging Face Hub (hosting adapters and GGUF)
- Google AI Studio (Gemini) for routing and embeddings
- Google Drive API
- Community models used in prototyping

## Project

- Name: Obvix Wellness (made by Obvix Labs)
- HF model hub: https://hf.co/thtskaran/emoai

## Quick Start

1. Install system deps (Ubuntu example)
   - sudo apt update && sudo apt install -y python3-dev python3-venv build-essential
   - Install and start MongoDB (or point to remote cluster)

2. Clone & set up
   - python3 -m venv .venv && source .venv/bin/activate
   - pip install --upgrade pip
   - pip install flask pymongo python-dotenv transformers google-api-python-client google-auth pypdf python-docx

3. (Optional) Pull / run Ollama model
   - Install Ollama: https://ollama.ai
   - ollama pull gemma3:12b  (or your custom fine-tuned model)
   - Ensure it matches CHAT_MODEL in app.py

4. Create .env (see below), then:
   - python app.py
   - Server: http://localhost:5001

## .env Example

```
GEMINI_KEY=YOUR_GEMINI_API_KEY
OLLAMA_URL=http://127.0.0.1:11434
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=emoai
LOG_FILE=logs.log
PORT=5001
FLASK_DEBUG=1

# Optional Knowledge Base (Google Drive)
GDRIVE_FOLDER_ID=
GOOGLE_CREDENTIALS_PATH=client.json

# Retrieval tuning
KB_OVERLAP=0
KB_TOPK=3
MEMORY_TOPK=5
```

## Architecture Overview (Runtime Flow)

1. User POSTs /chat with { user_id, message }.
2. Message stored in chat_history.
3. Gemini Router (ROUTER_MODEL) called → returns JSON:
   - crisis flag + type
   - semantic memory candidates (facts: name, preference, relation, profession)
   - episodic summaries
4. Semantic + episodic memories stored (with on-demand embeddings).
5. Crisis shortcut: if flagged → immediate compassionate template reply.
6. Context building:
   - Memory retrieval: hybrid (0.6 * cosine + 0.4 * BM25) with threshold.
   - KB retrieval: hybrid (0.5 * cosine + 0.2 * BM25 + topic_boost) with strong CBT/DBT topic bias.
   - Emotion + volatility summary (sentiment EMA + emotion + sarcasm).
7. Prompt assembled as structured JSON inside AASHA_SYS instructions.
8. Local Ollama model (CHAT_MODEL) generates JSON {"message": "..."}.
9. Assistant reply stored and returned.

## Mongo Collections

- chat_history: {user_id, role, text, time}
- semantic_memory: stable user facts (+ embedding)
- episodic_memory: short event summaries (+ embedding)
- kb_sources: metadata per Drive file
- kb_chunks: chunked text + embedding + tags
- settings: internal state (e.g., Drive changes page token)

Indexes auto-created on startup:
- chat_history: (user_id, time)
- semantic_memory: user_id
- episodic_memory: user_id
- kb_chunks: source_id

## Retrieval Details

Memory:
- Hybrid score = 0.6 cosine + 0.4 BM25
- Filter threshold > 0.1
- Returns top K (MEMORY_TOPK env)

Knowledge Base:
- Chunks embedded at ingestion
- Topic detection (CBT/DBT phrases) adds 0.4 * matches to score
- Score = 0.5 cosine + 0.2 BM25 + topic_boost
- Returns KB_TOPK

Chunking:
- CHUNK_SIZE = 1500 chars
- Overlap configurable via KB_OVERLAP (default 0)

## Emotion & Velocity

- Sentiment model (twitter-roberta) → normalized scalar sequence
- Exponential moving average of absolute deltas → volatility bucket
- Emotion classifier selects primary label (confidence > 0.4)
- Sarcasm detector optional (score > 0.6)
- Output: single sentence summary consumed by model

## Crisis Handling

Router sets crisis.flag only for imminent signals (suicidality, psychosis, abuse, dysregulation).
If true → bypass full generation → predefined compassionate template reply.

## Google Drive Knowledge Sync (Optional)

1. Provide service account JSON at GOOGLE_CREDENTIALS_PATH.
2. Share target folder with service account email.
3. Set GDRIVE_FOLDER_ID.
4. Background thread (watch_drive_changes_loop):
   - Uses Drive Changes API incremental polling
   - On change: re-chunks file, re-embeds, upserts
   - On removal/trashed: deletes related kb_chunks

## Endpoint

POST /chat
Request:
```
{
  "user_id": "user123",
  "message": "I keep catastrophizing before meetings."
}
```

Success Response:
```
{
  "message": "You’ve been really hard on yourself before meetings. Want to look at one thought together and gently reframe it?",
  "meta": {
    "analysis": {
      "crisis": {"flag": false, "type": "none"},
      "memories": { "semantic": [...], "episodic": [...] }
    }
  }
}
```

Error Response:
```
{ "error": "router_llm_failed", "details": "..." }
```

## Logging

- Main rotating log: LOG_FILE (default logs.log)
- Structured trace: test.txt (chat_trace logger)
- Each step annotated: router_call_start, kb_retrieval_done, ollama_call_ok, etc.

## Embeddings

- Gemini embedding endpoint model: gemini-embedding-001
- Failures gracefully degrade (memory retrieval still uses BM25 portion)

## Local Model (Ollama)

- CHAT_MODEL = gemma3:12b (change via env if desired)
- Must return JSON; fallback regex extraction if malformed
- Adjust temperature etc. by modifying payload in ollama_generate

## Safety & Limitations

- Not a diagnostic or therapeutic system.
- No PII redaction implemented—deploy behind controlled access.
- Crisis detection heuristic depends on router model quality.

## Example curl

```
curl -X POST http://localhost:5001/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u1","message":"I feel overwhelmed and keep overthinking"}'
```

## Troubleshooting

- 500 router_llm_failed: Check GEMINI_KEY validity / quota.
- ollama_failed: Ensure model pulled and Ollama daemon running.
- context_retrieval_failed: Often embedding call/network; check connectivity.
- No KB results: Confirm Drive folder shared & service account permissions.
- High latency: Reduce KB_TOPK / MEMORY_TOPK or switch to smaller local model.

## Minimal Requirements

- Python 3.10+
- MongoDB 6.x+
- Outbound HTTPS to Google APIs (embeddings + router)
- Optional: Ollama GPU or CPU capacity for selected model

## Extending

- Add new topic tags in detect_guideline_topics
- Swap in vector DB (e.g., Qdrant) by replacing retrieval sections
- Add rate limiting or auth in Flask before_request

## Disclaimer

Prototype only. Not a substitute for professional help. Verify all safety logic before any real user exposure.

## Future Scope (Planned Enhancements)

### 1. Autonomous Topic Detection (Replacing Keyword Heuristics)
Current: detect_guideline_topics uses keyword lists.
Planned: Zero-shot classification to infer CBT / DBT relevance even without explicit terms.

Example (future replacement):
```python
from transformers import pipeline
try:
    topic_clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
except Exception:
    topic_clf = None

def detect_guideline_topics_autonomous(text: str):
    if not topic_clf:
        return []
    labels = ["cognitive behavioral therapy", "dialectical behavior therapy"]
    try:
        res = topic_clf(text, labels, multi_label=True)
        out = []
        for label, score in zip(res["labels"], res["scores"]):
            if score > 0.40:
                if "cognitive" in label: out.append("cbt")
                elif "dialectical" in label: out.append("dbt")
        return list(set(out))
    except Exception:
        return []
```
Fallback: retain existing keyword function as backup if model unavailable.

Impact:
- Higher recall on distorted thinking descriptions
- Cleaner KB retrieval biasing

### 2. Classical ML Crisis Pre-Filter
Add a fast Logistic Regression (or linear SVM) that runs before router:
Features:
- TF-IDF of message
- Counts of high-risk lexicon
- Sentiment scalar + (optional) top emotion score
Action:
- If probability > threshold (e.g., 0.65) → tag message and force crisis-sensitive branch (still confirm via router).
Benefits: Lower latency early detection; layered safety.

### 3. Intent Pre-Router (Efficiency Layer)
Multiclass Naive Bayes over intents:
- therapeutic_inquiry
- personal_question
- information_request
- small_talk
Routing Policy (example):
- small_talk → skip KB + memory retrieval (fast path)
- personal_question → prioritize memory retrieval
- information_request → emphasize KB
- therapeutic_inquiry → full pipeline (current behavior)

Pseudo-hook (future in /chat start):
```python
intent = intent_model.predict([user_msg])[0]
if intent == "small_talk":
    # build minimal prompt (no retrieval) → generate
```

### 4. Adjustable Retrieval Policies
After intent classification:
- Dynamically set KB_TOPK / MEMORY_TOPK
- Potential caching of embeddings for repeated short-talk sessions

### 5. Lightweight Online Adaptation
Store rolling intent/mood distributions per user for:
- Temporal smoothing (e.g., volatility trend)
- Adaptive thresholds (e.g., lower crisis trigger after escalating language spikes)

### 6. Evaluation & Monitoring
Planned:
- Confusion matrix logging for crisis pre-filter (shadow mode first)
- Intent distribution dashboard
- Drift alerts: sudden drop in zero-shot topic assignment rates

### 7. Optional Vector DB Migration
Abstract current Mongo retrieval layer to allow plugging Qdrant / Weaviate for:
- ANN search
- Filtering by tags + hybrid scoring server-side

### 8. Safety Hardening
- Add prompt guardrail classifier (NSFW / disallowed)
- Red-team phrase pattern matcher before LLM call

### 9. Model Selection Strategy
Future: dynamic CHAT_MODEL swap:
- Smaller model for small_talk
- Larger / safer model for crisis-adjacent or therapeutic_inquiry

### 10. Batch Embedding Backfill
Periodic job to:
- Recompute embeddings for legacy semantic / episodic rows with missing vectors
- Normalize embedding dimensionality after model upgrade

These upgrades transition the system from reactive keyword heuristics to proactive, intent-aware, safety-layered orchestration while controlling latency and cost.
