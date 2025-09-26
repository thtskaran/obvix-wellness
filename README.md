# Obvix Wellness by Obvix Labs 

**Obvix Wellness** is an experimental emotional chat companion with a strong emphasis on safety, designed and built by Obvix Labs. It integrates a Flask backend API, advanced hybrid retrieval augmented generation (RAG) using MongoDB, Google Gemini 2.5 Flash router for detailed state analysis, and a local Ollama-served chat model, all orchestrated to provide sensitive, therapeutic-style responses.

***

## Project Overview

- **Project Name:** Obvix Wellness  
- **Organization:** Obvix Labs  
- **Model Hubs:**  
  - Instruction-tuned adapter: [hf.co/thtskaran/emoai](https://hf.co/thtskaran/emoai)  
  - Pretrained adapter (PT base): [hf.co/thtskaran/ob-wl-g3-pt](https://hf.co/thtskaran/ob-wl-g3-pt)  
- **Disclaimer:** This prototype is *not medical advice* nor a substitute for professional care. All safety logic should be validated before real user deployment.

***

## Key Components & Architecture

- **Backend API:** Flask  
- **Database:** MongoDB (stores chat history, semantic and episodic memories, therapeutic knowledge base chunks)  
- **Router and Embeddings:** Google Gemini 2.5 Flash for conversation state analysis and embedding generation  
- **Local Chat Model Serving:** **Ollama (mandatory)** with custom Modelfiles supporting lightweight, quantized LoRA adapters  
- **Models:**  
  - LoRA adapters and quantized GGUF models optimized for efficient inference hosted on Hugging Face  
  - Alpaca-style templates and stop tokens are pre-configured  
- **Hybrid Retrieval:** Combines cosine similarity with BM25, enhanced with topic bias towards therapeutic guideline content (primarily CBT/DBT)  
- **Therapeutic Knowledge Base:** Implemented using Google Drive files as an optional, automatically syncable source of therapy-related documents and rulebooks  

***

## Runtime Flow

1. User sends a `POST /chat` request with JSON `{ user_id, message }`.  
2. The message is logged into MongoDB chat history.  
3. Router LLM (Gemini 2.5 Flash) performs:  
   - Crisis detection (suicidality, psychosis, abuse, dysregulation)  
   - Extraction of semantic and episodic memories  
   - Sentiment, emotion, sarcasm, and volatility analysis  
4. Messages flagged as crisis bypass the full pipeline and receive a predefined compassionate response template immediately.  
5. For regular messages, hybrid retrieval fetches contextual memories and optional therapeutic knowledge base chunks from Google Drive (if configured), weighted by relevance and CBT/DBT topic boost.  
6. The local Ollama-served chat model receives a structured JSON prompt including memories, retrieved KB, and emotion summaries, then generates a JSON response.  
7. The assistant reply is saved and returned to the user.  

**Note:** The router LLM currently performs a single call per message. Future plans involve batching to reduce API calls and latency.

***

## Installation & Setup

### System Requirements

- Python 3.10+  
- MongoDB 6.x+ for persistent storage  
- Outbound HTTPS access for Google API (embeddings, router)  
- **Ollama installation and run** is *mandatory* for local chat model inference  

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

- Ollama is a **required** dependency to run any chat model inference. No alternative serving mechanism is currently supported.  
- Load Modelfiles tailored for your chosen model:  
  Example to set up the Pretrained (PT) model:  
  ```bash
  ollama create ob-wl-pt -f training/Modelfile-pt
  ollama run ob-wl-pt
  ```
- The PT Modelfile directs the model with concise therapeutic instructions avoiding excess sycophancy, resulting from skipping heavy RLHF phases. It works best embedded in the full orchestration with router logic rather than standalone chat.

***

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

***

## Model Training & Artifact Summary

### Instruction-Tuned Adapter (Gemma3-4b base)

- Trained on roughly 14,000 therapeutic conversation rows  
- LoRA fine-tuning done with Unsloth FastLanguageModel and TRL SFTTrainer with assistant-only loss  
- Early stopping and cosine learning rate scheduler in use  
- Validation perplexity: 1.5503 after 3 epochs  
- Artifacts available: LoRA adapters, 4-bit GGUF models, and Modelfile for Ollama deployment at [hf.co/thtskaran/emoai](https://hf.co/thtskaran/emoai)  

### Pretrained Adapter (Gemma3-4b PT base)

- Skips resource-heavy RLHF phase to reduce sycophantic behavior, promoting groundedness  
- Validation perplexity: 1.7113  
- LoRA adapters and 8-bit GGUF models with Modelfile provided, hosted at [hf.co/thtskaran/ob-wl-g3-pt](https://hf.co/thtskaran/ob-wl-g3-pt)  
- Best used embedded within orchestration; no standalone direct chat Modelfile provided for PT adapter  
- Future work planned to fine-tune adapters on broader conversational datasets for improved chat style and generalization

***

## Therapeutic Knowledge Base (Google Drive)

- Optional integration of a therapeutic knowledge base via Google Drive documents shared with a service account  
- Auto syncs via Drive Changes API in an incremental background thread: uploads, modifications, deletions reflected in knowledge chunks and embeddings  
- Documents chunked (~1500 characters plus configurable overlap), embedded with Gemini embeddings, stored in MongoDB  
- Retrieval weighted heavily towards CBT/DBT relevant content with topic boost factors  
- Drives therapy-specific guidance and contextual facts into the chat interaction

***

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