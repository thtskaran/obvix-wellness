# Obvix Wellness (made by Obvix Labs)

An experimental, safety-first emotional chat companion. Backend is a Flask API with lightweight RAG (MongoDB), a Gemini 2.5 Flash router for state analysis and embeddings that support memories, and an Ollama-served local chat model. Trained adapters and a 4-bit quantized prototype are available on Hugging Face.

Important: This prototype is not medical advice and not a replacement for professional care.

## What’s in this repo
- Flask API: service entrypoint in app.py (Mongo-backed memories, router-driven FSMs).
- Training notebook: training/trainijng gemma3.ipynb (LoRA with Unsloth, merge, and GGUF conversion flow).
- Ollama Modelfile: training/Modelfile (prompt template + system message for local inference).
- Env config: .env (set your own keys and endpoints).
- External LLM: Gemini 2.5 Flash for routing (analysis JSON) and embeddings used by memories and state logic.
- Google Drive KB: Sync a Drive folder of therapeutic texts (PDF, DOCX, MD, Google Docs) into a vector KB for lightweight grounding.

## Model artifacts
- Hugging Face model repo (LoRA + GGUF prototype): https://hf.co/thtskaran/emoai
  - Includes a prototype 4-bit quantized .gguf and LoRA adapters.
  - Use the provided Modelfile to run the model locally via Ollama.

## Quickstart

Prerequisites:
- Python 3.10+
- MongoDB (local or Atlas)
- Ollama (for local inference): https://ollama.com
- A Google AI Studio API key (GEMINI_KEY) for routing and embeddings (Gemini 2.5 Flash + text-embedding-004)
- Google Drive read access via a Service Account (client.json in project root or set GOOGLE_CREDENTIALS_PATH)

Install:
- Create and fill a .env (example below).
- pip install -U flask pymongo requests python-dotenv transitions transformers
- pip install -U google-api-python-client google-auth google-auth-httplib2 python-docx pypdf

Environment (.env):
- Do not commit secrets. Example:
  - MONGODB_URI=mongodb+srv://<user>:<pass>@<cluster>/<db>
  - GEMINI_KEY=AIza...
  - OLLAMA_URL=http://localhost:11434
  - OLLAMA_MODEL=emoai-sarah
  - GDRIVE_FOLDER_ID=your_drive_folder_id
  - GOOGLE_CREDENTIALS_PATH=./client.json
  - KB_TOPK=2
  - KB_OVERLAP=200

Run API:
- python app.py
- The server listens on PORT (default 5000).

Example request:
- POST /chat
- JSON: { "user_id": "u1", "message": "rough day, feeling stuck" }
- Response: { "message": "...", "meta": { "fsm": {...}, "conversation": "..." } }

## Google Drive Knowledge Base

Purpose:
- Maintain a global, non-clinical therapeutic corpus (e.g., CBT/DBT books, rulebooks, guidelines) for light grounding of the local 4B model.
- The bot stays conversational and non-clinical; KB snippets are only used to keep language and facts steady.

Setup:
- Place your service account JSON at GOOGLE_CREDENTIALS_PATH (default ./client.json).
- Share the Drive folder (GDRIVE_FOLDER_ID) with the service account email (Viewer).
- Put text-friendly files in the folder: PDF, DOCX, MD/TXT, and Google Docs.
  - No OCR is performed; files should be text-based.

Real-time sync:
- The server starts a background Drive Changes watcher on boot and keeps the KB in sync in near real time.
- Changes in the configured folder (add/update/remove) automatically update MongoDB:
  - kb_sources: one per Drive file
  - kb_chunks: vector chunks (1500 chars, overlap controlled by KB_OVERLAP)
- Everything from the files is chunked as-is (no stripping), then embedded with Gemini text-embedding-004.

Manual full sync:
- POST /kb/sync with optional JSON { "force": true } to reindex everything.

Retrieval:
- On each /chat call, the top KB snippets are added as short KB[...] lines in context (e.g., “KB[CBT_rulebook]: ...”).
- The prompt explicitly instructs the model to remain non-clinical and to not cite sources.
- Filenames matter. For example, “CBT_rulebook.pdf” implies CBT content (not DBT); this guides retrieval via title tags.

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

## API architecture (high level)

- Router (Gemini 2.5 Flash) analyzes user input → crisis flags, emotion/engagement, conversation mode, and memory updates.
- External LLM usage: Offloaded tasks include routing/analysis JSON and embeddings that support memories (RAG) and FSM state tracking; chat generation remains local via Ollama.
- FSMs:
  - DuoFSM: emotion (positive/neutral/negative) and engagement (engaged/withdrawn/looping/intimate).
  - ConversationFSM: casual_chat/light_support/deeper_exploration/skill_offering/crisis_support/cool_down/idle.
- Memories:
  - Semantic (facts like name, preferences).
  - Episodic (short event summaries with embeddings).
- Knowledge Base:
  - Global KB from Drive (kb_sources, kb_chunks) with embeddings for lightweight grounding.
- Local chat:
  - Prompt constructed with short context and minimal “benevolent friction”.
  - Served via Ollama model specified by OLLAMA_MODEL.

## Safety and scope

- Not a clinician. Avoids irreversible advice, romance/parasocial tones, and keeps responses brief and grounded.
- If acute risk is detected, the agent provides a brief safety-first message and nudges toward human support.
- You are responsible for your own deployment’s compliance and guardrails.

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
