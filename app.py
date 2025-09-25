# app.py
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
import hashlib
import threading
import time
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from docx import Document
from pypdf import PdfReader

load_dotenv()

# -------------------------
# Config / Env
# -------------------------
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
KB_TOPK                 = int(os.getenv("KB_TOPK", "2"))
CHUNK_SIZE              = 1500  # fixed chunk size per request

EMBED_MODEL   = "gemini-embedding-001"
ROUTER_MODEL  = "gemini-2.5-flash"
CHAT_MODEL    = "gemma3:12b"

app = Flask(__name__)
logger = logging.getLogger(__name__)

# -------------------------
# Logging setup
# -------------------------
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

# Add a simple dedicated logger for /chat traces to test.txt
chat_logger = logging.getLogger("chat_trace")
if not chat_logger.handlers:
    chat_logger.setLevel(logging.INFO)
    _test_log_path = os.path.join(os.path.dirname(__file__), "test.txt")
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

# -------------------------
# DB / Collections
# -------------------------
mc = MongoClient(MONGO_URI)
db = mc[DB_NAME]
col_semantic  = db.semantic_memory   # user facts (name, prefs, relations, work)
col_episodic  = db.episodic_memory   # high-level event summaries
col_kb_src    = db.kb_sources        # records per Drive file
col_kb        = db.kb_chunks         # chunked embeddings
col_settings  = db.settings          # generic settings (e.g., Drive changes page token)

# -------------------------
# Classifiers (best-effort)
# -------------------------
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

# -------------------------
# Ephemeral in-memory state
# -------------------------
# USER[user_id] = {
#   "conversation": [ { "role": "user"|"assistant", "text": str, "ts": iso, "signals": {...} } ],
# }
USER = {}

# -------------------------
# Embeddings / Similarity
# -------------------------
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

# -------------------------
# Tokenization / BM25-lite
# -------------------------
def _tokenize_simple(text: str):
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2]

def _bm25lite_score(query_tokens, doc_tokens, k1=1.2, b=0.75, avgdl=200.0):
    if not doc_tokens or not query_tokens:
        return 0.0
    qtf = {t: query_tokens.count(t) for t in set(query_tokens)}
    dtf = {t: doc_tokens.count(t) for t in set(doc_tokens)}
    dl = len(doc_tokens)
    score = 0.0
    for t, qfreq in qtf.items():
        tf = dtf.get(t, 0)
        if tf == 0:
            continue
        # A tiny IDF-like constant since we don't have a full corpus DF here
        idf = 1.5
        denom = tf + k1 * (1 - b + b * (dl / avgdl))
        score += idf * ((tf * (k1 + 1)) / (denom + 1e-9)) * (1 + math.log1p(qfreq))
    return float(score)

# -------------------------
# Memories Retrieval (contextual)
# -------------------------
def retrieve_episodic_memories(user_id: str, query_text: str, k=5):
    """
    Return top-k episodic summaries most relevant to the query by cosine similarity (if embeddings present),
    otherwise lexical similarity.
    """
    try:
        qemb = get_embedding(query_text)
    except Exception:
        qemb = None

    episodic = list(col_episodic.find({"user_id": user_id}, {"summary": 1, "embedding": 1}).limit(200))
    ranked = []
    qtok = _tokenize_simple(query_text)
    for ev in episodic:
        ev_emb = ev.get("embedding")
        if qemb and ev_emb:
            score = cosine(qemb, ev_emb)
        else:
            score = _bm25lite_score(qtok, _tokenize_simple(ev.get("summary", "")))
        ranked.append((score, {"summary": ev.get("summary", "").strip()}))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [ev for _, ev in ranked[:k]]

def retrieve_semantic_facts(user_id: str, query_text: str, k=5):
    """
    Rank semantic facts by cosine relevance to the query; fallback to lexical.
    Returns top-k compact dicts.
    """
    facts = list(col_semantic.find({"user_id": user_id}, {"type": 1, "value": 1, "relation": 1, "name": 1, "embedding": 1}).limit(200))
    if not facts:
        return []

    try:
        qemb = get_embedding(query_text)
    except Exception:
        qemb = None

    ranked = []
    qtok = _tokenize_simple(query_text)
    for f in facts:
        if f.get("type") == "relation":
            txt = f"{f.get('relation','')} {f.get('name','')}".strip()
        else:
            txt = (f.get("value") or f.get("name") or "").strip()
        if qemb and f.get("embedding"):
            cos = cosine(qemb, f["embedding"])
            score = cos
        else:
            score = _bm25lite_score(qtok, _tokenize_simple(txt))
        ranked.append((score, {
            "type": f.get("type"),
            **({"value": f.get("value")} if "value" in f else {}),
            **({"relation": f.get("relation")} if "relation" in f else {}),
            **({"name": f.get("name")} if "name" in f else {}),
        }))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:k]]

def get_topk_memories(user_id: str, query_text: str, k=5):
    """
    Combine episodic + semantic, re-rank shallowly by lexical similarity against query, return up to k.
    """
    episodic = retrieve_episodic_memories(user_id, query_text, k=k)
    semantic = retrieve_semantic_facts(user_id, query_text, k=k)
    candidates = []

    # Flatten into comparable text + payload
    for ev in episodic:
        candidates.append(("episodic", ev.get("summary", ""), ev))
    for sf in semantic:
        if sf.get("type") == "relation":
            txt = f"{sf.get('relation','')} {sf.get('name','')}".strip()
        else:
            txt = (sf.get("value") or sf.get("name") or "").strip()
        candidates.append(("semantic", txt, sf))

    qtok = _tokenize_simple(query_text)
    rescored = []
    for typ, text, payload in candidates:
        score = _bm25lite_score(qtok, _tokenize_simple(text))
        rescored.append((score, typ, text, payload))
    rescored.sort(key=lambda x: x[0], reverse=True)

    out = []
    for _, typ, text, payload in rescored[:k]:
        if typ == "episodic":
            out.append({"summary": text})
        else:
            out.append(payload)
    return out

# -------------------------
# System Prompt
# -------------------------
AASHA_SYS = (
    "You are Aasha — a warm, grounded, everyday chat companion. "
    "Talk naturally with the user like a supportive friend. Keep replies concise and human. "
    "You will receive structured context.\n\n"
    "INPUTS YOU MAY RECEIVE:\n"
    "- chat_history: the last 5 turns as a JSON list of {user, assistant} pairs (most-recent last).\n"
    "- semantic_memories: brief user facts (name, preferences, relations, work roles).\n"
    "- episodic_memories: short summaries of notable past events and feelings.\n"
    "- emotion_signals: {sentiment, primary_emotion, sarcasm} gathered for the current user message.\n"
    "- rolling_state: {avg_sentiment, velocity, label} computed from the last 10 user messages.\n"
    "- kb_snippets: short, relevant knowledge snippets retrieved from the user's Google Drive corpus.\n\n"
    "BEHAVIOR:\n"
    "- Be friendly, succinct, and real; avoid clinical tone.\n"
    "- Use memories and recent history only when they clearly improve continuity or clarity.\n"
    "- If label indicates tension/volatility, keep a steady, calm tone; if steady/positive, keep it light and flowing.\n"
    "- Ask at most one brief clarifying question only if needed.\n"
    "- No meta-chatter; do not prefix with 'Aasha:' or similar.\n\n"
    "OUTPUT:\n"
    "Return STRICT JSON only: {\"message\":\"...\"} — one short conversational reply. "
    "No markdown, no extra keys, no code fences."
)

# -------------------------
# Signals & Rolling State
# -------------------------
def _sentiment_to_unit(label: str, score: float) -> float:
    l = (label or "").lower()
    s = max(0.0, min(1.0, float(score or 0.0)))
    if "pos" in l: return s
    if "neg" in l: return -s
    return 0.0

def analyze_message(text: str) -> dict:
    # sentiment
    sent_label, sent_score = "neutral", 0.0
    if sent_clf:
        try:
            s = sent_clf(text)[0]
            sent_label, sent_score = s["label"], float(s.get("score", 0.0))
        except Exception:
            pass

    # emotion (take top)
    emo_label, emo_score = "neutral", 0.0
    if emo_clf:
        try:
            e_all = emo_clf(text)[0]
            top = max(e_all, key=lambda x: x["score"])
            emo_label, emo_score = top["label"], float(top["score"])
        except Exception:
            pass

    sarcasm_flag = False
    if sarc_clf:
        try:
            s = sarc_clf(text)[0]
            sarcasm_flag = "sarcas" in (s["label"] or "").lower()
        except Exception:
            pass

    sent_unit = _sentiment_to_unit(sent_label, sent_score)
    return {
        "sentiment": {"label": sent_label, "score": float(round(sent_score, 3)), "unit": float(round(sent_unit, 3))},
        "primary_emotion": {"label": emo_label, "score": float(round(emo_score, 3))},
        "sarcasm": sarcasm_flag
    }

def _label_from_avg_vel(avg: float, vel: float) -> str:
    # Vel thresholds: low<0.15, med<0.45, high>=0.45 (mean |Δ| in [-2,2] but practically ~0..1)
    # Avg sentiment thresholds: neg<=-0.25, pos>=0.25, else neutral
    mood = "neutral"
    if avg <= -0.25:
        mood = "negative"
    elif avg >= 0.25:
        mood = "positive"

    if vel < 0.15:
        pace = "steady"
    elif vel < 0.45:
        pace = "shifting"
    else:
        pace = "spiking"

    # Map combinations to friendly English
    table = {
        ("positive", "steady"): "steady",
        ("positive", "shifting"): "energized",
        ("positive", "spiking"): "amped",
        ("neutral",  "steady"): "steady",
        ("neutral",  "shifting"): "edgy",
        ("neutral",  "spiking"): "volatile",
        ("negative", "steady"): "low",
        ("negative", "shifting"): "tense",
        ("negative", "spiking"): "volatile",
    }
    return table.get((mood, pace), "steady")

def compute_state_from_last10(user_id: str) -> dict:
    convo = USER.get(user_id, {}).get("conversation", [])
    last_user_msgs = [t for t in convo if t["role"] == "user"][-10:]
    if not last_user_msgs:
        return {"avg_sentiment": 0.0, "velocity": 0.0, "label": "steady"}

    units = [float(t.get("signals", {}).get("sentiment", {}).get("unit", 0.0)) for t in last_user_msgs]
    # average sentiment
    avg = sum(units) / max(1, len(units))
    # mean absolute delta
    if len(units) >= 2:
        deltas = [abs(units[i] - units[i-1]) for i in range(1, len(units))]
        vel = sum(deltas) / len(deltas)
    else:
        vel = 0.0
    # clamp
    avg = max(-1.0, min(1.0, avg))
    vel = max(0.0, min(2.0, vel))
    label = _label_from_avg_vel(avg, vel)
    return {"avg_sentiment": float(round(avg, 3)), "velocity": float(round(vel, 3)), "label": label}

def append_conversation(user_id: str, role: str, text: str, signals: dict | None = None):
    state = USER.setdefault(user_id, {"conversation": []})
    state["conversation"].append({
        "role": role,
        "text": text,
        "ts": datetime.utcnow().isoformat(),
        "signals": signals or {}
    })
    # keep last 200 turns total
    if len(state["conversation"]) > 200:
        state["conversation"] = state["conversation"][-200:]

def get_recent_turns(user_id: str, n_pairs: int = 5):
    """
    Return up to n_pairs most recent (user, assistant) pairs as list of dicts: [{user, assistant}], most-recent last.
    """
    convo = USER.get(user_id, {}).get("conversation", [])
    # Build pairs scanning from end
    pairs = []
    current_pair = {"user": None, "assistant": None}
    for msg in reversed(convo):
        if msg["role"] == "assistant" and current_pair["assistant"] is None:
            current_pair["assistant"] = msg["text"]
        elif msg["role"] == "user":
            if current_pair["user"] is None:
                current_pair["user"] = msg["text"]
                pairs.append(current_pair)
                current_pair = {"user": None, "assistant": None}
    pairs = list(reversed(pairs))  # chronological
    return pairs[-n_pairs:]

# -------------------------
# KB Retrieval (placeholder stubs)
# -------------------------
def retrieve_kb_snippets(user_id: str, query_text: str, topk: int = KB_TOPK):
    """
    If you have KB populated in col_kb, implement hybrid retrieval here.
    For now returns [] to keep prompt tight.
    """
    return []

# -------------------------
# Router (optional) — you can keep or remove
# -------------------------
def router_analysis(user_id: str, user_message: str) -> dict | None:
    if not GEMINI_KEY:
        return None
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{ROUTER_MODEL}:generateContent?key={GEMINI_KEY}"
        system = (
            "You are an analysis router for a friendly, grounded chat companion. "
            "Return STRICT JSON ONLY with keys: { 'crisis': {'flag': bool, 'type': 'none|suicidality|psychosis|abuse|dysregulation'}, "
            "'memories': { 'semantic': [ { 'type':'name|preference|relation|profession', 'value': str, 'relation': str|null, 'name': str|null } ], "
            "'episodic': [ { 'summary': str } ] }, 'graph': { 'edges': [ ['node_a','node_b'] ] } } "
            "Rules: keep it minimal; summarize episodic at high level; include only key facts; set crisis true ONLY if the message indicates imminent risk; "
            "prefer 'none' otherwise; avoid romantic/parasocial attachments; avoid irreversible advice."
        )
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": json.dumps({
                    "user_id": user_id,
                    "message": user_message,
                    "signals": {}
                })}]}
            ],
            "generationConfig": {"response_mime_type": "application/json", "temperature": 0.2}
        }
        # we stuff system via safety; Gemini doesn't have system param: omitted here; acceptable as optional
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Try parsing text parts to JSON
        text = ""
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            text = ""
        return json.loads(text) if text else None
    except Exception:
        return None

# -------------------------
# Ollama call
# -------------------------
def ollama_generate_chat(system_prompt: str, context_json: dict) -> str:
    """
    Sends a single-turn generate request to Ollama with the system + packed context JSON.
    """
    prompt = f"{system_prompt}\n\nCONTEXT (JSON):\n{json.dumps(context_json, ensure_ascii=False)}\n\nAasha -> Return STRICT JSON only as {{\"message\":\"...\"}}:"
    url = f"{OLLAMA_URL.rstrip('/')}/api/generate"
    payload = {
        "model": CHAT_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7}
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    out = data.get("response", "")
    # Try to extract {"message": "..."} from fences or raw
    out_stripped = out.strip()
    if out_stripped.startswith("```"):
        # remove code fences
        out_stripped = re.sub(r"^```[a-zA-Z]*\n", "", out_stripped)
        out_stripped = re.sub(r"\n```$", "", out_stripped)
    try:
        obj = json.loads(out_stripped)
        return obj.get("message", "").strip() or out
    except Exception:
        return out

# -------------------------
# API Endpoints
# -------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "model": CHAT_MODEL})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_id = (data or {}).get("user_id") or "anon"
    user_msg = (data or {}).get("message") or ""
    if not user_msg:
        return jsonify({"error": "empty_message"}), 400

    chat_trace(user_id, "request_received", {"keys": list((data or {}).keys()), "msg_len": len(user_msg)})

    # Optional router (non-blocking)
    try:
        chat_trace(user_id, "router_call_start", {"model": ROUTER_MODEL})
        r = router_analysis(user_id, user_msg)
        if r:
            chat_trace(user_id, "router_response_preview", {"chars": len(json.dumps(r)), "preview": json.dumps(r)[:200]})
    except Exception:
        pass

    # Analyze current message with transformers pipelines
    signals = analyze_message(user_msg)
    append_conversation(user_id, "user", user_msg, signals)

    # Rolling state from last 10 user msgs (excluding current already appended—intentionally included)
    rolling = compute_state_from_last10(user_id)

    # Retrieve memories (Top-K 5 total)
    chat_trace(user_id, "kb_retrieval_start", {"topk": 5})
    try:
        top_mem = get_topk_memories(user_id, user_msg, k=5)
    except Exception:
        top_mem = []
    chat_trace(user_id, "kb_retrieval_done", {"returned": len(top_mem)})

    # Build last 5 turns
    turns = get_recent_turns(user_id, n_pairs=5)

    # You can split memories into semantic/episodic buckets for the LLM if you prefer;
    # here we keep a single compact structure and also provide the raw split counts.
    # For clarity with the current system prompt, we mirror original keys as possible:
    # We'll try to separate based on shapes we produced earlier.
    semantic_mems, episodic_mems = [], []
    for m in top_mem:
        if "summary" in m:
            episodic_mems.append({"summary": m["summary"]})
        else:
            semantic_mems.append(m)

    # KB snippets (placeholder)
    kb_snips = retrieve_kb_snippets(user_id, user_msg, topk=KB_TOPK)

    packed = {
        "chat_history": turns,  # last 5 turns only
        "semantic_memories": semantic_mems,  # subset of total memories
        "episodic_memories": episodic_mems,
        "emotion_signals": {
            "sentiment": {"label": signals["sentiment"]["label"], "score": signals["sentiment"]["score"]},
            "primary_emotion": signals["primary_emotion"],
            "sarcasm": signals["sarcasm"]
        },
        "rolling_state": rolling,  # from last 10 user msgs
        "kb_snippets": kb_snips,
        "user_input": user_msg
    }

    chat_trace(user_id, "prompt_built", {"chars": len(json.dumps(packed))})
    chat_trace(user_id, "ollama_prompt", {"prompt": AASHA_SYS})

    # Call Ollama
    try:
        chat_trace(user_id, "ollama_call_start", {"model": CHAT_MODEL})
        message = ollama_generate_chat(AASHA_SYS, packed)
        # Minimal sanity: ensure we got something JSON-like per spec
        try:
            # If the model already returned {"message": "..."} we keep it;
            # otherwise we wrap it to comply with the contract.
            obj = json.loads(message)
            final_text = obj.get("message", "")
            if not final_text:
                final_text = message if isinstance(message, str) else json.dumps(message, ensure_ascii=False)
                obj = {"message": final_text}
        except Exception:
            obj = {"message": message}
        assistant_text = obj.get("message", "").strip()
    except Exception as e:
        app.logger.exception("Ollama call failed")
        assistant_text = "Sorry—I'm having trouble responding right now."

    append_conversation(user_id, "assistant", assistant_text, signals=None)
    chat_trace(user_id, "response_sent", {"status": "ok", "len": len(assistant_text)})

    # Return the assistant JSON contract expected by your frontend
    return jsonify({"message": assistant_text, "state": rolling})

# -------------------------
# Google Drive ingestion (skeleton)
# -------------------------
def _get_drive_service():
    if not os.path.exists(GOOGLE_CREDENTIALS_PATH):
        return None
    try:
        creds = service_account.Credentials.from_service_account_file(
            GOOGLE_CREDENTIALS_PATH,
            scopes=["https://www.googleapis.com/auth/drive.readonly"]
        )
        return build("drive", "v3", credentials=creds)
    except Exception:
        return None

def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = KB_OVERLAP):
    text = text or ""
    out = []
    i = 0
    n = len(text)
    while i < n:
        out.append(text[i:i+chunk_size])
        i += max(1, chunk_size - overlap)
    return out

def _embed_and_store_chunks(user_id: str, file_id: str, chunks: list[str], meta: dict):
    for idx, ch in enumerate(chunks):
        try:
            emb = get_embedding(ch)
        except Exception:
            emb = None
        doc = {
            "user_id": user_id,
            "file_id": file_id,
            "chunk_index": idx,
            "text": ch,
            "embedding": emb,
            "meta": meta,
            "created_at": datetime.utcnow()
        }
        col_kb.insert_one(doc)

@app.route("/ingest_drive", methods=["POST"])
def ingest_drive():
    """
    Body: { "user_id": "...", "folder_id": "optional override" }
    Minimal skeleton; expand per your needs.
    """
    data = request.get_json(force=True)
    user_id = (data or {}).get("user_id") or "anon"
    folder_id = (data or {}).get("folder_id") or GDRIVE_FOLDER_ID
    svc = _get_drive_service()
    if not svc or not folder_id:
        return jsonify({"error": "drive_not_configured"}), 400

    q = f"'{folder_id}' in parents and (mimeType='application/vnd.google-apps.document' or mimeType='application/vnd.openxmlformats-officedocument.wordprocessingml.document' or mimeType='application/pdf') and trashed = false"
    files = svc.files().list(q=q, fields="files(id, name, mimeType, modifiedTime)").execute().get("files", [])
    total = 0
    for f in files:
        fid, name, mt = f["id"], f["name"], f["mimeType"]
        # Export/download
        try:
            if mt == "application/pdf":
                request_dl = svc.files().get_media(fileId=fid)
                fh = BytesIO()
                downloader = MediaIoBaseDownload(fh, request_dl)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                fh.seek(0)
                reader = PdfReader(fh)
                text = "\n".join([p.extract_text() or "" for p in reader.pages])
            else:
                # Try exporting Google Doc as docx
                if mt == "application/vnd.google-apps.document":
                    request_exp = svc.files().export_media(fileId=fid, mimeType="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
                    fh = BytesIO(request_exp.execute())
                else:
                    request_dl = svc.files().get_media(fileId=fid)
                    fh = BytesIO()
                    downloader = MediaIoBaseDownload(fh, request_dl)
                    done = False
                    while not done:
                        status, done = downloader.next_chunk()
                    fh.seek(0)
                # parse docx
                doc = Document(fh)
                text = "\n".join([p.text for p in doc.paragraphs])
        except Exception:
            continue

        chunks = _chunk_text(text, CHUNK_SIZE, KB_OVERLAP)
        _embed_and_store_chunks(user_id, fid, chunks, {"name": name, "mimeType": mt, "modifiedTime": f.get("modifiedTime")})
        total += 1

    return jsonify({"ingested_files": total})

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=FLASK_DEBUG)
