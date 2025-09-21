# Obvix Wellness (made by Obvix Labs)

An experimental, safety-first emotional chat companion. Backend is a Flask API with lightweight RAG (MongoDB), a small router model for state analysis, and an Ollama-served local chat model. Trained adapters and a 4-bit quantized prototype are available on Hugging Face.

Important: This prototype is not medical advice and not a replacement for professional care.

## What’s in this repo
- Flask API: service entrypoint in app.py (Mongo-backed memories, router-driven FSMs).
- Training notebook: training/trainijng gemma3.ipynb (LoRA with Unsloth, merge, and GGUF conversion flow).
- Ollama Modelfile: training/Modelfile (prompt template + system message for local inference).
- Env config: .env (set your own keys and endpoints).

## Model artifacts
- Hugging Face model repo (LoRA + GGUF prototype): https://hf.co/thtskaran/emoai
  - Includes a prototype 4-bit quantized .gguf and LoRA adapters.
  - Use the provided Modelfile to run the model locally via Ollama.

## Quickstart

Prerequisites:
- Python 3.10+
- MongoDB (local or Atlas)
- Ollama (for local inference): https://ollama.com
- An OpenAI API key (for routing and embeddings)

Install:
- Create and fill a .env (example below).
- pip install -U flask pymongo requests python-dotenv transitions transformers

Environment (.env):
- Do not commit secrets. Example:
  - MONGODB_URI=mongodb+srv://<user>:<pass>@<cluster>/<db>
  - OPENAI_API_KEY=sk-...
  - OLLAMA_URL=http://localhost:11434
  - OLLAMA_MODEL=emoai-sarah

Run API:
- python app.py
- The server listens on PORT (default 5000).

Example request:
- POST /chat
- JSON: { "user_id": "u1", "message": "rough day, feeling stuck" }
- Response: { "message": "...", "meta": { "fsm": {...}, "conversation": "..." } }

## Using the Ollama Modelfile

This repo includes training/Modelfile so you can create a local model:

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

The notebook covers:
- Loading a base model and training LoRA with Unsloth.
- Merging LoRA into base weights (Peft merge_and_unload).
- Converting merged HF weights to GGUF via llama.cpp.
- Optional quantization (e.g., Q4_K_M).
- Uploading GGUF back to the HF repo.

If you only want to infer locally:
- Use the provided Modelfile with Ollama (see above).
- Or directly download the GGUF from the HF repo.

## API architecture (high level)

- Router (small OpenAI model) analyses user input → crisis flags, emotion/engagement, conversation mode, and memory updates.
- FSMs:
  - DuoFSM: emotion (positive/neutral/negative) and engagement (engaged/withdrawn/looping/intimate).
  - ConversationFSM: casual_chat/light_support/deeper_exploration/skill_offering/crisis_support/cool_down/idle.
- Memories:
  - Semantic (facts like name, preferences).
  - Episodic (short event summaries with embeddings).
  - Diff logs (state changes).
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
- Community models used in prototyping

## Project

- Name: Obvix Wellness (made by Obvix Labs)
- HF model hub: https://hf.co/thtskaran/emoai
