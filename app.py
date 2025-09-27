import os, re, math, json, requests
from datetime import datetime
from flask import Flask, request, jsonify
from pymongo import MongoClient
from bson import ObjectId
from transformers import pipeline
from dotenv import load_dotenv

import logging
from logging.handlers import RotatingFileHandler
from werkzeug.exceptions import HTTPException

from io import BytesIO
import tempfile
import hashlib
import threading
import time
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError
from docx import Document
from pypdf import PdfReader

load_dotenv()


GEMINI_KEY    = os.getenv("GEMINI_KEY")
OLLAMA_URL    = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
MONGO_URI     = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME       = os.getenv("MONGODB_DB", "emoai")
LOG_FILE      = os.getenv("LOG_FILE", os.path.join(os.path.dirname(__file__), "logs.log"))
PORT          = int(os.getenv("PORT", "5001"))
FLASK_DEBUG   = os.getenv("FLASK_DEBUG", "1").strip().lower() in ("1", "true", "t", "yes", "y", "on")

GDRIVE_FOLDER_ID        = os.getenv("GDRIVE_FOLDER_ID")
GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", os.path.join(os.path.dirname(__file__), "client.json"))
KB_OVERLAP              = int(os.getenv("KB_OVERLAP") or "0")
KB_TOPK                 = int(os.getenv("KB_TOPK", "3")) 
MEMORY_TOPK             = int(os.getenv("MEMORY_TOPK", "5")) # Top K for combined memories
CHUNK_SIZE              = 1500

EMBED_MODEL   = "gemini-embedding-001"
ROUTER_MODEL  = "gemini-2.5-flash"
CHAT_MODEL    = "ob-wl-pt"
MAX_TOKENS    = int(os.getenv("MAX_TOKENS", "135"))

app = Flask(__name__)
logger = logging.getLogger(__name__)


def setup_logging(flask_app: Flask):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True) if os.path.dirname(LOG_FILE) else None
    handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)

    flask_app.logger.setLevel(logging.INFO)
    flask_app.logger.addHandler(handler)

    werk = logging.getLogger("werkzeug")
    werk.setLevel(logging.INFO)
    werk.addHandler(handler)

setup_logging(app)

chat_logger = logging.getLogger("chat_trace")
if not chat_logger.handlers:
    chat_logger.setLevel(logging.INFO)
    _test_log_path = os.path.join(os.path.dirname(__file__), "prompts.log")
    try:
        fh = logging.FileHandler(_test_log_path)
        fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        chat_logger.addHandler(fh)
    except Exception:
        app.logger.exception("Failed to initialize chat_trace logger")

def chat_trace(uid, event, meta=None):
    try:
        rec = {"uid": uid, "event": event, "meta": meta or {}}
        chat_logger.info(json.dumps(rec, ensure_ascii=False))
    except Exception:
        pass

@app.before_request
def _log_request():
    try:
        uid = (request.get_json(silent=True) or {}).get("user_id")
    except Exception:
        uid = None
    app.logger.info(f">>> {request.method} {request.path} uid={uid}")

@app.after_request
def _log_response(resp):
    app.logger.info(f"<<< {resp.status_code} {request.method} {request.path}")
    return resp

@app.errorhandler(Exception)
def _handle_exception(e):
    if isinstance(e, HTTPException):
        app.logger.warning(f"HTTPException {e.code} on {request.method} {request.path}: {e.description}")
        return e
    app.logger.exception("Unhandled exception during request")
    return jsonify({"error": "internal_error"}), 500


mc = MongoClient(MONGO_URI)
db = mc[DB_NAME]
col_semantic      = db.semantic_memory   # user facts (name, prefs, relations, work)
col_episodic      = db.episodic_memory   # high-level event summaries
col_kb_src        = db.kb_sources        # records per Drive file
col_kb            = db.kb_chunks         # chunked embeddings
col_settings      = db.settings          # generic settings (e.g., Drive changes page token)
col_chat_history  = db.chat_history      # stores all user/assistant turns


try:
    sent_clf = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
except Exception:
    sent_clf = None

try:
    emo_clf  = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
except Exception:
    emo_clf = None

try:
    sarc_clf = pipeline("text-classification", model="helinivan/english-sarcasm-detector")
except Exception:
    sarc_clf = None


def get_embedding(text: str):
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_KEY required")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBED_MODEL}:embedContent?key={GEMINI_KEY}"
    payload = {"content": {"parts": [{"text": text}]}}
    resp = requests.post(url, json=payload, timeout=45)
    resp.raise_for_status()
    data = resp.json()
    emb = ((data or {}).get("embedding") or {}).get("values")
    if not emb:
        raise RuntimeError("embedding_failed")
    return emb

def cosine(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return dot / (na*nb) if na and nb else 0.0

def _tokenize_simple(text: str):
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2]

def _bm25lite_score(query_tokens, doc_tokens, k1=1.2, b=0.75, avgdl=50.0): 
    if not doc_tokens or not query_tokens:
        return 0.0
    tf = sum(doc_tokens.count(q) for q in set(query_tokens))
    dl = max(1.0, float(len(doc_tokens)))
    denom = tf + k1 * (1 - b + b * (dl / avgdl))
    return (tf * (k1 + 1)) / denom if denom > 0 else 0.0


def retrieve_relevant_memories(user_id: str, query_text: str, k=MEMORY_TOPK):
    """
    Retrieves top-K semantic and episodic memories based on hybrid relevancy score.
    Combines cosine similarity (semantic) and BM25 (lexical) for ranking.
    """
    qemb = None
    try:
        qemb = get_embedding(query_text)
    except Exception:
        app.logger.warning(f"Could not get embedding for memory retrieval uid={user_id}")

    qtok = _tokenize_simple(query_text)
    
    all_memories = []

    # 1. Fetch all memories for user
    for doc in col_semantic.find({"user_id": user_id}):
        if doc["type"] == "name": text = f"User's name is {doc['value']}."
        elif doc["type"] == "preference": text = f"User likes {doc['value']}."
        elif doc["type"] == "relation": text = f"User has a {doc.get('relation')} named {doc.get('name')}."
        elif doc["type"] == "profession": text = f"User works as a {doc['value']}."
        else: continue
        all_memories.append({"text": text, "embedding": doc.get("embedding")}) # Assumes semantic memories could have embeddings

    for doc in col_episodic.find({"user_id": user_id}):
        text = f"Previously: {doc.get('summary','')}"
        if doc.get("emotions"): text += f" (user felt {doc['emotions'][0]})."
        all_memories.append({"text": text, "embedding": doc.get("embedding")})
    
    if not all_memories:
        return []

    # 2. Score and rank each memory
    ranked = []
    for mem in all_memories:
        mem_text = mem.get("text", "")
        mem_emb = mem.get("embedding")
        
        cos_score = cosine(qemb, mem_emb) if qemb and mem_emb else 0.0
        mem_tok = _tokenize_simple(mem_text)
        bm_score = _bm25lite_score(qtok, mem_tok)
        
        hybrid_score = 0.6 * cos_score + 0.4 * bm_score
        if hybrid_score > 0.1: # Threshold to filter out irrelevant memories
            ranked.append((hybrid_score, mem_text))

    # 3. Return top-k memories
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [text for score, text in ranked[:k]]



AASHA_SYS = (
    "You are Aasha — a warm, grounded, everyday chat companion. "
    "Talk naturally with the user like a supportive friend. Keep replies to 1–2 short sentences (max 40 words total). "
    "You will receive structured context.\n\n"
    "INPUTS YOU MAY RECEIVE:\n"
    "- chat_history: the last 5 turns as a JSON list of {user, assistant} pairs (most-recent last).\n"
    "- relevant_memories: a list of the user's most relevant facts and past event summaries related to the current topic.\n"
    "- emotion_analysis: a plain English summary of the user's current emotional state and recent volatility.\n"
    "- kb_snippets: short, relevant knowledge snippets from clinical handbooks (e.g., CBT, DBT) to help you provide supportive, informed guidance.\n\n"
    "BEHAVIOR:\n"
    "- Be friendly, succinct, and real; avoid clinical tone, but you can use concepts from the kb_snippets if they seem helpful.\n"
    "- If user_input is just a greeting or very short, reply with a warm hello and one brief question; do not give advice.\n"
    "- Use memories and history for continuity. If a memory about a specific person is retrieved, it's a strong signal they are important right now.\n"
    "- Adapt your tone based on the emotion_analysis. If the user's state is volatile, be extra calm and grounding.\n"
    "- Ask at most one brief clarifying question only if needed.\n"
    "- No meta-chatter; do not prefix with 'Aasha:' or similar.\n\n"
    "OUTPUT:\n"
    "Return STRICT JSON only: {\"message\":\"...\"} — one short conversational reply. "
    "No markdown, no extra keys, no code fences."
)


_EMOTION_AROUSAL = {
    "joy": 0.40, "happy": 0.40, "love": 0.45, "surprise": 0.65, "neutral": 0.10, 
    "fear": 0.85, "anger": 0.90, "disgust": 0.80, "sadness": 0.70
}

def _sentiment_to_unit(label: str, score: float) -> float:
    l = (label or "").lower()
    s = max(0.0, min(1.0, float(score or 0.0)))
    if "pos" in l: return s
    if "neg" in l: return -s
    return 0.0

def _emotion_arousal(label: str, score: float) -> float:
    base = _EMOTION_AROUSAL.get((label or "").lower(), 0.25)
    return base * max(0.0, min(1.0, float(score or 0.0)))

def analyze_emotion_and_velocity(user_id: str, current_text: str, lookback: int = 6) -> str:
    """
    Computes emotional velocity and returns a plain English summary for the LLM.
    """
    history = col_chat_history.find({"user_id": user_id, "role": "user"}).sort("time", -1).limit(lookback)
    recent_user_msgs = [doc["text"] for doc in history][::-1] # oldest to newest
    series = recent_user_msgs + [current_text]

    primary_emotion = "neutral"
    sarcasm_detected = False
    
    if emo_clf:
        try:
            e_all = emo_clf(current_text)[0]
            top = max(e_all, key=lambda x: x["score"])
            if top["score"] > 0.4: # Confidence threshold
                primary_emotion = top["label"]
        except Exception: pass

    if sarc_clf:
        try:
            s = sarc_clf(current_text)[0]
            if "sarcas" in (s["label"] or "").lower() and s["score"] > 0.6:
                sarcasm_detected = True
        except Exception: pass

    # Calculate velocity
    stream = []
    if sent_clf:
        for m in series:
            try: stream.append(_sentiment_to_unit(sent_clf(m)[0]["label"], sent_clf(m)[0].get("score", 0.0)))
            except Exception: stream.append(0.0)
    else: stream = [0.0] * len(series)

    alpha = 0.6
    ema = 0.0
    for i in range(1, len(stream)):
        delta = abs(stream[i] - stream[i - 1])
        ema = alpha * delta + (1 - alpha) * ema

    # Convert to plain english
    if ema < 0.25: velocity_desc = "calm and stable"
    elif ema < 0.55: velocity_desc = "showing gentle shifts"
    elif ema < 0.85: velocity_desc = "somewhat agitated or volatile"
    else: velocity_desc = "highly volatile and changing rapidly"

    summary = f"User seems to be feeling {primary_emotion}. Their emotional state has been {velocity_desc}."
    if sarcasm_detected:
        summary += " Note: Sarcasm may be present."
    
    return summary


def detect_guideline_topics(text: str):
    """
    More robust detector for clinical guideline topics.
    Returns topic tags to bias retrieval (e.g., ['cbt', 'dbt']).
    """
    t = (text or "").lower()
    tags = []
    # CBT Keywords ,  will move to Zero-Shot Text Classification model
    if any(w in t for w in ["cbt", "cognitive distortion", "automatic thought", "thought record", "reframing", "overthinking", "anxious about", "catastrophizing", "all or nothing"]):
        tags.append("cbt")
    # DBT Keywords will move to Zero-Shot Text Classification model
    if any(w in t for w in ["dbt", "distress tolerance", "wise mind", "opposite action", "radical acceptance", "emotion regulation", "intense emotion", "overwhelmed", "self-harm"]):
        tags.append("dbt")
    
    return list(sorted(set(tags)))

def retrieve_kb(query_text: str, k=KB_TOPK):
    """
    Hybrid retrieval with stronger topic biasing for clinical handbooks.
    """
    topic_tags = detect_guideline_topics(query_text)
    try:
        qemb = get_embedding(query_text)
    except Exception:
        qemb = None

    qtok = _tokenize_simple(query_text)
    
    # Broaden search if specific topics are detected
    search_limit = 5000 if topic_tags else 3000
    cur = col_kb.find({"embedding": {"$ne": None}}, {"text": 1, "embedding": 1, "title": 1, "tags": 1}).limit(search_limit)

    scored = []
    for doc in cur:
        text = doc.get("text", "")[:800]
        emb = doc.get("embedding")
        title = (doc.get("title") or "").lower()
        tags = [str(t).lower() for t in (doc.get("tags") or [])]

        cos = cosine(qemb, emb) if qemb and emb else 0.0
        dtok = _tokenize_simple(text)
        bm = _bm25lite_score(qtok, dtok, avgdl=400.0)

        # Apply topic boosting
        topic_boost = 0.0
        matched_topics = [tag for tag in topic_tags if tag in tags or tag in title]
        if matched_topics:
            topic_boost = 0.4 * len(matched_topics) 

        
        hybrid = 0.5 * cos + 0.2 * bm + topic_boost
        scored.append((hybrid, text))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:k]]


ROUTER_SYS = (
    "You are an analysis router for a friendly, grounded chat companion. "
    "Return STRICT JSON ONLY with keys: "
    "{ 'crisis': {'flag': bool, 'type': 'none|suicidality|psychosis|abuse|dysregulation'}, "
    # Embeddings will be added in the main app, so router focuses on text
    "'memories': {'semantic': [ { 'type':'name|preference|relation|profession', 'value': str, 'relation': str|null, 'name': str|null } ], "
    "'episodic': [ { 'summary': str } ] } } "
    "Rules: keep it minimal; summarize episodic at high level; include only key facts; "
    "set crisis true ONLY if the message indicates imminent risk; prefer 'none' otherwise."
)

def call_router(user_id: str, user_message: str):
    if not GEMINI_KEY: raise RuntimeError("GEMINI_KEY required")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{ROUTER_MODEL}:generateContent?key={GEMINI_KEY}"
    body = {
        "systemInstruction": { "parts": [{"text": ROUTER_SYS}] },
        "contents": [{"role": "user", "parts": [{"text": json.dumps({"user_id": user_id, "message": user_message}, ensure_ascii=False)}] }],
        "generationConfig": {"response_mime_type": "application/json", "temperature": 0.2}
    }
    r = requests.post(url, json=body, timeout=45); r.raise_for_status()
    data = r.json()
    try: txt = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e: raise RuntimeError(f"router_parse_failed: {data}") from e
    return json.loads(txt)


def crisis_reply(kind: str):
    replies = {
        "suicidality": "I'm really sorry you're feeling this much pain. You deserve support right now. If you can, please reach out to someone you trust or a local crisis line. I can stay with you in this chat while you consider that.",
        "psychosis": "I'm sorry you're going through this. It may help to get medical support. If you can, consider contacting someone you trust or local services. I'm here to listen.",
        "abuse": "I'm so sorry you're dealing with that. You don’t deserve to be hurt. If you're able, reaching out to trusted people or local services could help. I'm here with you.",
        "dysregulation": "I can tell this feels intense. I'm here with you. If talking it through helps, we can take it one small step at a time."
    }
    return replies.get(kind, "I'm here with you. You're not alone, and it's okay to ask for help close to you right now.")


def build_companion_prompt(user_id, user_message, kb_snippets, relevant_memories, emotion_analysis):
    history_docs = col_chat_history.find({"user_id": user_id}).sort("time", -1).limit(10)
    conv = list(history_docs)[::-1] 
    
    pairs = []
    i = 0
    while i < len(conv) - 1:
        if conv[i]["role"] == "user" and conv[i + 1]["role"] == "assistant":
            pairs.append({"user": conv[i]["text"], "assistant": conv[i + 1]["text"]})
            i += 2
        else:
            i += 1
    history_pairs = pairs

    instruction = {
        "chat_history": history_pairs,
        "relevant_memories": relevant_memories,
        "emotion_analysis": emotion_analysis,
        "kb_snippets": kb_snippets,
        "user_input": user_message
    }

    prompt = (
        f"{AASHA_SYS}\n\n"
        "CONTEXT (JSON):\n"
        + json.dumps(instruction, ensure_ascii=False, indent=2)
        + "\n\nAasha -> Return STRICT JSON only as {\"message\":\"...\"}:"
    )
    return prompt


def ollama_generate(prompt: str):
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": CHAT_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": MAX_TOKENS,
            "temperature": 0.7
        }
    }
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    try:
        j = r.json()
        return (j.get("response") or "").strip()
    except Exception:
        return r.text.strip()

def extract_message(raw: str) -> str:
    s = (raw or "").strip()
    try:
        # Strip code fences if present
        if s.startswith("```"):
            s = re.sub(r"^```[a-zA-Z0-9]*\n", "", s)
            s = re.sub(r"\n```$", "", s).strip()

        # Try parsing JSON directly
        data = None
        try:
            data = json.loads(s)
        except Exception:
            data = None

        # Case 1: JSON object
        if isinstance(data, dict):
            msg = data.get("message") or data.get("text") or ""
            if isinstance(msg, str) and msg.strip():
                return msg.strip()

        # Case 2: JSON was a string (possibly JSON-encoded again)
        if isinstance(data, str):
            inner = data.strip()
            try:
                inner_obj = json.loads(inner)
                if isinstance(inner_obj, dict) and "message" in inner_obj:
                    return str(inner_obj.get("message", "")).strip()
            except Exception:
                if inner:
                    return inner

        # Fallback: regex for message field (handles "message": "..." or 'message': '...')
        match = re.search(r'["\']message["\']\s*:\s*["\'](.*?)["\']', s, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Final fallback: plain text, remove any "Aasha:" prefix
        return s.replace("Aasha:", "").strip()
    except Exception:
        # Never raise from extractor
        return s.replace("Aasha:", "").strip()

# Enforce brevity and detect low-content greetings
def enforce_brevity(text: str, max_words: int = 40, max_sentences: int = 2) -> str:
    if not text:
        return text
    norm = re.sub(r"\s+", " ", text).strip()
    # Keep at most max_sentences
    sentences = re.split(r"(?<=[.!?])\s+", norm)
    clipped = " ".join(sentences[:max_sentences])
    # Then cap by words
    words = clipped.split()
    if len(words) > max_words:
        clipped = " ".join(words[:max_words]).rstrip(",;:") + "..."
    return clipped

def is_low_content_greeting(text: str) -> bool:
    if not text:
        return False
    s = re.sub(r"[\s\W_]+", " ", text.strip().lower()).strip()
    if not s:
        return False
    greetings = {
        "hi", "hey", "hello", "yo", "heya", "hiya", "hey there", "hola",
        "sup", "whats up", "what's up", "good morning", "good afternoon", "good evening"
    }
    if s in greetings:
        return True
    tokens = s.split()
    if len(tokens) <= 3 and any(g in s for g in greetings):
        # Treat as greeting if there isn't clear problem content
        if not any(k in s for k in ["help", "issue", "problem", "because", "anxious", "stress", "sad", "angry", "sister"]):
            return True
    return False

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    user_id = str(data.get("user_id", "")).strip()
    user_msg = str(data.get("message", "")).strip()

    if not user_id or not user_msg:
        return jsonify({"error": "user_id and non-empty message required"}), 400

    app.logger.info(f"/chat start uid={user_id}")
    chat_trace(user_id, "request_received", {"msg_len": len(user_msg)})

    col_chat_history.insert_one({"user_id": user_id, "role": "user", "text": user_msg, "time": datetime.utcnow()})

    # Greeting short-circuit: keep it friendly and very brief
    if is_low_content_greeting(user_msg):
        try:
            name_doc = col_semantic.find_one({"user_id": user_id, "type": "name"})
            first_name = None
            if name_doc:
                v = str(name_doc.get("value", "")).strip()
                first_name = v.split()[0] if v else None
        except Exception:
            first_name = None
        msg = f"Hey {first_name}! How are you doing today?" if first_name else "Hey! How are you doing today?"
        col_chat_history.insert_one({"user_id": user_id, "role": "assistant", "text": msg, "time": datetime.utcnow()})
        chat_trace(user_id, "greeting_short_circuit", {})
        return jsonify({"message": msg, "meta": {"short_circuit": "greeting"}})

    # 1) Router LLM → analysis JSON
    chat_trace(user_id, "router_call_start", {"model": ROUTER_MODEL})
    try:
        analysis = call_router(user_id, user_msg)
        chat_trace(user_id, "router_call_ok", {
            "crisis": analysis.get("crisis"), "mem_sem": len(analysis.get("memories", {}).get("semantic", [])), "mem_epi": len(analysis.get("memories", {}).get("episodic", []))
        })
    except Exception as e:
        chat_trace(user_id, "router_call_failed", {"error": str(e)})
        app.logger.exception("router_llm_failed")
        return jsonify({"error": "router_llm_failed", "details": str(e)}), 500

    # 2) Persist memories from analysis (with embeddings)
    for s in analysis.get("memories", {}).get("semantic", []):
        doc = {"user_id": user_id, "type": s["type"], "value": s.get("value")}
        text_for_embedding = ""
        if s["type"] == "relation":
            doc["relation"] = s.get("relation"); doc["name"] = s.get("name")
            text_for_embedding = f"User has a {s.get('relation')} named {s.get('name')}."
        else: text_for_embedding = f"A fact about the user: {s.get('value')}"
        try: doc["embedding"] = get_embedding(text_for_embedding)
        except Exception: doc["embedding"] = None
        filt = {"user_id": user_id, "type": s["type"], "value": s.get("value")}
        col_semantic.update_one(filt, {"$set": doc}, upsert=True)

    for ev in analysis.get("memories", {}).get("episodic", []):
        summ = (ev.get("summary", "") or "")[:500]
        edoc = {"user_id": user_id, "summary": summ, "time": datetime.utcnow()}
        try: edoc["embedding"] = get_embedding(summ)
        except Exception: pass
        col_episodic.insert_one(edoc)

    # 3) Crisis short-circuit
    cr = analysis.get("crisis", {})
    if cr.get("flag"):
        chat_trace(user_id, "crisis_short_circuit", {"type": cr.get("type")})
        msg = crisis_reply(cr.get("type", "none"))
        col_chat_history.insert_one({"user_id": user_id, "role": "assistant", "text": msg, "time": datetime.utcnow()})
        chat_trace(user_id, "response_sent", {"status": "crisis"})
        return jsonify({"message": msg, "meta": {"crisis": cr}})

    # 4) Retrieve all context using new V2 functions
    try:
        chat_trace(user_id, "kb_retrieval_start", {"topk": KB_TOPK})
        kb_snippets = retrieve_kb(user_msg, k=KB_TOPK)
        chat_trace(user_id, "kb_retrieval_done", {"returned": len(kb_snippets)})
        
        chat_trace(user_id, "memory_retrieval_start", {"topk": MEMORY_TOPK})
        relevant_memories = retrieve_relevant_memories(user_id, user_msg, k=MEMORY_TOPK)
        chat_trace(user_id, "memory_retrieval_done", {"returned": len(relevant_memories)})

        emotion_analysis = analyze_emotion_and_velocity(user_id, user_msg)
        chat_trace(user_id, "emotion_analysis_done", {"analysis": emotion_analysis})
    except Exception as e:
        chat_trace(user_id, "context_retrieval_failed", {"error": str(e)})
        app.logger.exception(f"Context retrieval failed for uid={user_id}")
        return jsonify({"error": "context_retrieval_failed", "details": str(e)}), 500

    # Build Aasha prompt V2
    prompt = build_companion_prompt(user_id, user_msg, kb_snippets, relevant_memories, emotion_analysis)
    chat_trace(user_id, "prompt_built", {"chars": len(prompt)})
    try: chat_trace(user_id, "ollama_prompt", {"prompt": prompt})
    except Exception: pass

    # 5) Generate
    chat_trace(user_id, "ollama_call_start", {"model": CHAT_MODEL})
    try:
        raw_response = ollama_generate(prompt)
        chat_trace(user_id, "ollama_call_ok", {"raw_chars": len(raw_response)})
        message = extract_message(raw_response)
        if not message:
            # Safe fallback to avoid 500s when model output is malformed
            message = "I’m here with you. Tell me a bit more so I can help."
        # Enforce brevity regardless of model behavior
        message = enforce_brevity(message, max_words=40, max_sentences=2)
        chat_trace(user_id, "message_extracted", {"chars": len(message)})
    except Exception as e:
        chat_trace(user_id, "ollama_call_failed", {"error": str(e)})
        app.logger.exception("ollama_failed")
        return jsonify({"error": "ollama_failed", "details": str(e)}), 500

    # 6) Log assistant response to persistent history
    col_chat_history.insert_one({"user_id": user_id, "role": "assistant", "text": message, "time": datetime.utcnow()})
    chat_trace(user_id, "conversation_appended", {})

    app.logger.info(f"/chat done uid={user_id}")
    chat_trace(user_id, "response_sent", {"status": "ok"})
    return jsonify({"message": message, "meta": {"analysis": analysis}})

# -------------------------
# Google Drive KB ingestion/sync 
# -------------------------
def build_drive_service():
    if not GDRIVE_FOLDER_ID: raise RuntimeError("GDRIVE_FOLDER_ID not set")
    if not os.path.isfile(GOOGLE_CREDENTIALS_PATH): raise RuntimeError(f"Missing Google credentials file at {GOOGLE_CREDENTIALS_PATH}")
    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    creds = service_account.Credentials.from_service_account_file(GOOGLE_CREDENTIALS_PATH, scopes=scopes)
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def list_drive_files(service, folder_id):
    files = []
    page_token = None
    q = f"'{folder_id}' in parents and trashed=false"
    fields = "nextPageToken, files(id,name,mimeType,modifiedTime,md5Checksum,size,parents)"
    while True:
        resp = service.files().list(q=q, fields=fields, pageToken=page_token).execute()
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token: break
    return files

def download_drive_file_text(service, file_id, mime_type):
    if mime_type == "application/vnd.google-apps.document":
        data = service.files().export(fileId=file_id, mimeType="text/plain").execute()
        return data.decode("utf-8", errors="ignore")
    req = service.files().get_media(fileId=file_id)
    buf = BytesIO(); downloader = MediaIoBaseDownload(buf, req)
    done = False
    while not done: _, done = downloader.next_chunk()
    data = buf.getvalue()
    if mime_type in ("text/markdown", "text/plain"): return data.decode("utf-8", errors="ignore")
    if mime_type in ("application/pdf",):
        try:
            reader = PdfReader(BytesIO(data)); txt = [p.extract_text() or "" for p in reader.pages]
            return "\n".join(txt)
        except Exception: return ""
    if mime_type in ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"):
        try: return "\n".join(p.text for p in Document(BytesIO(data)).paragraphs)
        except Exception: return ""
    try: return data.decode("utf-8", errors="ignore")
    except Exception: return ""

def chunk_text(text, max_chars=1800, overlap=200):
    text = (text or ""); chunks, i, n = [], 0, len(text)
    if not text: return []
    while i < n:
        j = min(i + max_chars, n); chunks.append(text[i:j])
        if j == n: break
        i = max(0, j - overlap)
    return chunks

def title_tags(title: str):
    t = (title or "").strip().lower(); parts = re.split(r"[\s_\-\.\(\)\[\]:]+", t)
    return list({p for p in parts if len(p) >= 3})

def upsert_kb_for_file(file_meta, text):
    fid = file_meta["id"]; title = file_meta.get("name", ""); mime  = file_meta.get("mimeType", ""); mtime = file_meta.get("modifiedTime", "")
    content_hash = hashlib.sha256((text or "").encode("utf-8")).hexdigest()
    src = col_kb_src.find_one({"file_id": fid})
    if src and src.get("modifiedTime") == mtime and src.get("content_hash") == content_hash:
        return {"status": "skip", "file_id": fid, "title": title}
    col_kb.delete_many({"source_id": fid})
    tags = title_tags(title); lowered = title.lower()
    if "cbt" in lowered or "cognitive" in lowered: tags.append("cbt")
    if "dbt" in lowered or "dialectical" in lowered: tags.append("dbt")
    tags = list(sorted(set(tags)))
    chunks = chunk_text(text, max_chars=CHUNK_SIZE, overlap=KB_OVERLAP); now = datetime.utcnow()
    for idx, ch in enumerate(chunks):
        try: emb = get_embedding(ch)
        except Exception: emb = None
        doc = {"source_id": fid, "title": title, "mimeType": mime, "chunk_index": idx, "text": ch, "embedding": emb, "tags": tags, "time": now}
        col_kb.insert_one(doc)
    col_kb_src.update_one({"file_id": fid}, {"$set": {"file_id": fid, "title": title, "mimeType": mime, "modifiedTime": mtime, "content_hash": content_hash, "chunk_count": len(chunks), "tags": tags, "time": now}}, upsert=True)
    return {"status": "updated", "file_id": fid, "title": title, "chunks": len(chunks)}

def delete_kb_for_file(file_id: str):
    col_kb.delete_many({"source_id": file_id})
    col_kb_src.delete_one({"file_id": file_id})

def _settings_get(key, default=None):
    doc = col_settings.find_one({"_id": key}); return (doc or {}).get("val", default)
def _settings_set(key, val):
    col_settings.update_one({"_id": key}, {"$set": {"val": val, "time": datetime.utcnow()}}, upsert=True)
def get_drive_start_page_token(service):
    res = service.changes().getStartPageToken().execute(); return res.get("startPageToken")

def watch_drive_changes_loop():
    if not GDRIVE_FOLDER_ID: app.logger.warning("Drive watcher disabled: GDRIVE_FOLDER_ID not set"); return
    try: service = build_drive_service()
    except Exception: app.logger.exception("Drive watcher: failed to build service"); return
    token_key = "drive_changes_start_page_token"; page_token = _settings_get(token_key)
    if not page_token:
        try: page_token = get_drive_start_page_token(service); _settings_set(token_key, page_token); app.logger.info(f"Drive watcher: initialized start page token {page_token}")
        except Exception: app.logger.exception("Drive watcher: failed to get start page token"); return
    fields = "newStartPageToken,nextPageToken,changes(fileId,removed,file(id,name,mimeType,parents,modifiedTime,md5Checksum,trashed))"
    while True:
        try:
            pt = page_token
            while pt:
                resp = service.changes().list(pageToken=pt, spaces="drive", fields=fields, pageSize=100, includeItemsFromAllDrives=True, supportsAllDrives=True).execute()
                for ch in resp.get("changes", []):
                    fid = ch.get("fileId"); file = ch.get("file") or {}; removed = ch.get("removed") or file.get("trashed")
                    if removed: delete_kb_for_file(fid); continue
                    if GDRIVE_FOLDER_ID not in (file.get("parents") or []): continue
                    try: text = download_drive_file_text(service, fid, file.get("mimeType")); upsert_kb_for_file(file, text)
                    except Exception: app.logger.exception(f"Drive watcher: upsert failed for {fid}")
                new_token = resp.get("newStartPageToken"); next_token = resp.get("nextPageToken")
                if next_token: pt = next_token
                else:
                    if new_token: page_token = new_token; _settings_set(token_key, page_token); app.logger.info(f"Drive watcher: updated page token to {page_token}")
                    break
        except Exception: app.logger.exception("Drive watcher: error while polling changes")
        time.sleep(20)


if __name__ == "__main__":
    # Create indexes for performance
    col_chat_history.create_index([("user_id", 1), ("time", -1)])
    col_semantic.create_index("user_id")
    col_episodic.create_index("user_id")
    col_kb.create_index("source_id")

    app.logger.info(f"Starting server on 0.0.0.0:{PORT} debug={FLASK_DEBUG}")
    threading.Thread(target=watch_drive_changes_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, threaded=True, debug=FLASK_DEBUG, use_reloader=FLASK_DEBUG)

