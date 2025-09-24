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
USER = {}  # user_id -> {"conversation": [...]}

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
# Memories Retrieval (RAG light)
# -------------------------
def retrieve_memories(user_id: str, query_text: str, k=3):
    qemb = None
    episodic = list(col_episodic.find({"user_id": user_id}))
    if episodic:
        try:
            qemb = get_embedding(query_text)
        except Exception:
            qemb = None

    ranked = []
    for ev in episodic:
        ev_emb = ev.get("embedding")
        score = cosine(qemb, ev_emb) if qemb and ev_emb else 0.0
        ranked.append((score, ev))
    ranked.sort(key=lambda x: x[0], reverse=True)

    ev_summaries = []
    for _, ev in ranked[:k]:
        emo = ev.get("emotions", [])
        emostr = f" (felt {emo[0]})" if emo else ""
        ev_summaries.append(f"Previously: {ev.get('summary','')}{emostr}.")

    facts = []
    for f in col_semantic.find({"user_id": user_id}).limit(10):
        if f["type"] == "name":
            facts.append(f"User name: {f['value']}.")
        elif f["type"] == "preference":
            facts.append(f"Likes: {f['value']}.")
        elif f["type"] == "relation":
            nm = f.get("name")
            facts.append(f"Relation: {f.get('relation')}{f' named {nm}' if nm else ''}.")
        elif f["type"] == "profession":
            facts.append(f"Work: {f['value']}.")
    return facts + ev_summaries

def get_first_name(user_id: str):
    doc = col_semantic.find_one({"user_id": user_id, "type": "name"})
    if not doc:
        return None
    val = (doc.get("value") or "").strip()
    return val.split()[0] if val else None

# -------------------------
# Aasha System Prompt
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
    "- emotional_velocity: a scalar ~[0..2] showing recent intensity/volatility (higher = faster shifts).\n"
    "- kb_snippets: short, relevant knowledge snippets retrieved from the user's Google Drive corpus.\n\n"
    "BEHAVIOR:\n"
    "- Be friendly, succinct, and real; avoid clinical tone.\n"
    "- Use memories and recent history only when they clearly improve continuity or clarity.\n"
    "- If emotional_velocity is high, keep a steady, calm tone; if low, keep it light and flowing.\n"
    "- Ask at most one brief clarifying question only if needed.\n"
    "- No meta-chatter; do not prefix with 'Aasha:' or similar.\n\n"
    "OUTPUT:\n"
    "Return STRICT JSON only: {\"message\":\"...\"} — one short conversational reply. "
    "No markdown, no extra keys, no code fences."
)

# -------------------------
# Emotional Velocity helpers
# -------------------------
_EMOTION_AROUSAL = {
    "joy": 0.40, "happy": 0.40, "love": 0.45,
    "surprise": 0.65, "anticipation": 0.50,
    "neutral": 0.10, "calm": 0.10,
    "trust": 0.30, "contentment": 0.25,
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

def compute_emotional_velocity(user_id: str, current_text: str, lookback: int = 6) -> dict:
    state = USER.get(user_id, {"conversation": []})
    recent_user_msgs = [t["text"] for t in state.get("conversation", []) if t["role"] == "user"][-lookback:]
    series = recent_user_msgs + [current_text]

    sent_label, sent_score = "neutral", 0.0
    if sent_clf:
        try:
            s = sent_clf(current_text)[0]
            sent_label, sent_score = s["label"], float(s.get("score", 0.0))
        except Exception:
            pass

    emo_label, emo_score = "neutral", 0.0
    if emo_clf:
        try:
            e_all = emo_clf(current_text)[0]
            top = max(e_all, key=lambda x: x["score"])
            emo_label, emo_score = top["label"], float(top["score"])
        except Exception:
            pass

    sarcasm_flag = False
    if sarc_clf:
        try:
            s = sarc_clf(current_text)[0]
            sarcasm_flag = "sarcas" in (s["label"] or "").lower()
        except Exception:
            pass

    stream = []
    if sent_clf:
        for m in series:
            try:
                r = sent_clf(m)[0]
                stream.append(_sentiment_to_unit(r["label"], r.get("score", 0.0)))
            except Exception:
                stream.append(0.0)
    else:
        stream = [0.0] * len(series)

    alpha = 0.6
    ema = 0.0
    for i in range(1, len(stream)):
        delta = abs(stream[i] - stream[i - 1])
        ema = alpha * delta + (1 - alpha) * ema

    arousal = _emotion_arousal(emo_label, emo_score)
    sarcasm_bump = 0.15 if sarcasm_flag else 0.0
    velocity = min(2.0, max(0.0, ema + arousal + sarcasm_bump))

    return {
        "velocity": float(round(velocity, 4)),
        "signals": {
            "sentiment": {"label": sent_label, "score": sent_score},
            "primary_emotion": {"label": emo_label, "score": emo_score},
            "sarcasm": sarcasm_flag
        }
    }

# -------------------------
# KB Retrieval (hybrid semantic + contextual)
# -------------------------
def _tokenize_simple(text: str):
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2]

def _bm25lite_score(query_tokens, doc_tokens, k1=1.2, b=0.75, avgdl=400.0):
    if not doc_tokens or not query_tokens:
        return 0.0
    tf = sum(doc_tokens.count(q) for q in set(query_tokens))
    dl = max(1.0, float(len(doc_tokens)))
    denom = tf + k1 * (1 - b + b * (dl / avgdl))
    return (tf * (k1 + 1)) / denom if denom > 0 else 0.0

def detect_guideline_topics(text: str):
    """
    Lightweight detector for therapy guideline pulls.
    Returns topic tags to bias retrieval (e.g., ['cbt', 'dbt']).
    """
    t = (text or "").lower()
    tags = []
    # lexical hints
    if any(w in t for w in ["cognitive distortion", "automatic thoughts", "thought record", "reframing", "cbt"]):
        tags.append("cbt")
    if any(w in t for w in ["distress tolerance", "wise mind", "opposite action", "radical acceptance", "dbt", "emotion regulation", "interpersonal effectiveness"]):
        tags.append("dbt")
    # emotion-based heuristic: high anger/fear + help-ish language → cbt/dbt
    # (kept minimal; model-independent)
    return list(sorted(set(tags)))

def retrieve_kb(query_text: str, k=KB_TOPK, topic_tags=None):
    """
    Hybrid retrieval over KB (Mongo col_kb):
      1) Cosine with Gemini embeddings (semantic)
      2) BM25-lite keyword score (contextual)
      3) Title/tag boosts (+ extra boost for requested topics like ['cbt','dbt'])
      4) Hybrid rerank
    Returns top-k short snippets.
    """
    topic_tags = topic_tags or []
    try:
        qemb = get_embedding(query_text)
    except Exception:
        qemb = None

    qtok = _tokenize_simple(query_text)
    cur = col_kb.find({"embedding": {"$ne": None}}, {"text": 1, "embedding": 1, "title": 1, "tags": 1}).limit(3000)

    scored = []
    for doc in cur:
        text = doc.get("text", "")[:800]
        emb = doc.get("embedding")
        title = (doc.get("title") or "")
        tags = [str(t).lower() for t in (doc.get("tags") or [])]

        cos = cosine(qemb, emb) if qemb and emb else 0.0
        dtok = _tokenize_simple(text)
        bm = _bm25lite_score(qtok, dtok)

        # Base title/tag match
        tmatch = 0.15 if any(t in (title.lower()) for t in qtok) else 0.0
        if tags and any(q in tags for q in qtok):
            tmatch += 0.10

        # Topic bias for guideline pulls (CBT/DBT/etc.)
        topic_boost = 0.0
        if topic_tags:
            # if any requested topic present in title or tags, boost
            if any(tp in title.lower() for tp in topic_tags):
                topic_boost += 0.20
            if any(tp in tags for tp in topic_tags):
                topic_boost += 0.20

        hybrid = 0.72 * cos + 0.25 * bm + tmatch + topic_boost
        scored.append((hybrid, text))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:k]]

# -------------------------
# Router LLM (analysis JSON) — unchanged
# -------------------------
ROUTER_SYS = (
    "You are an analysis router for a friendly, grounded chat companion. "
    "Return STRICT JSON ONLY with keys: "
    "{ 'crisis': {'flag': bool, 'type': 'none|suicidality|psychosis|abuse|dysregulation'}, "
    "'memories': {'semantic': [ { 'type':'name|preference|relation|profession', 'value': str, 'relation': str|null, 'name': str|null } ], "
    "'episodic': [ { 'summary': str } ] }, "
    "'graph': { 'edges': [ ['node_a','node_b'] ] } } "
    "Rules: keep it minimal; summarize episodic at high level; include only key facts; "
    "set crisis true ONLY if the message indicates imminent risk; prefer 'none' otherwise; "
    "avoid romantic/parasocial attachments; avoid irreversible advice."
)

def call_router(user_id: str, user_message: str):
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_KEY required")

    sig = {}
    if sent_clf:
        try:
            s = sent_clf(user_message)[0]
            sig["sentiment"] = s["label"].lower()
        except Exception:
            pass
    if emo_clf:
        try:
            e = emo_clf(user_message)[0]
            top = max(e, key=lambda x: x["score"])
            sig["primary_emotion"] = top["label"].lower()
        except Exception:
            pass
    if sarc_clf:
        try:
            s = sarc_clf(user_message)[0]
            sig["sarcasm"] = ("sarcas" in s["label"].lower())
        except Exception:
            pass

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{ROUTER_MODEL}:generateContent?key={GEMINI_KEY}"
    body = {
        "systemInstruction": { "parts": [{"text": ROUTER_SYS}] },
        "contents": [
            {
                "role": "user",
                "parts": [{
                    "text": json.dumps({"user_id": user_id, "message": user_message, "signals": sig}, ensure_ascii=False)
                }]
            }
        ],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.2
        }
    }

    try:
        chat_trace(user_id, "router_prompt", {
            "system": ROUTER_SYS,
            "payload": {"contents": body.get("contents"), "generationConfig": body.get("generationConfig")}
        })
    except Exception:
        pass

    r = requests.post(url, json=body, timeout=45)
    r.raise_for_status()
    data = r.json()
    try:
        txt = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        raise RuntimeError(f"router_parse_failed: {data}") from e

    try:
        chat_trace(user_id, "router_response_preview", {"chars": len(txt) if isinstance(txt, str) else None, "preview": txt[:500] if isinstance(txt, str) else None})
    except Exception:
        pass

    return json.loads(txt)

# -------------------------
# Crisis short replies
# -------------------------
def crisis_reply(kind: str):
    if kind == "suicidality":
        return ("I'm really sorry you're feeling this much pain. You deserve support right now. "
                "If you can, please reach out to someone you trust or a local crisis line. "
                "I can stay with you in this chat while you consider that.")
    if kind == "psychosis":
        return ("I'm sorry you're going through this. It may help to get medical support. "
                "If you can, consider contacting someone you trust or local services. I'm here to listen.")
    if kind == "abuse":
        return ("I'm so sorry you're dealing with that. You don’t deserve to be hurt. "
                "If you're able, reaching out to trusted people or local services could help. "
                "I'm here with you.")
    if kind == "dysregulation":
        return ("I can tell this feels intense. I'm here with you. "
                "If talking it through helps, we can take it one small step at a time.")
    return ("I'm here with you. You're not alone, and it's okay to ask for help close to you right now.")

# -------------------------
# Build Aasha prompt (history+memories+EV+KB)
# -------------------------
def build_companion_prompt(user_id, user_message, kb_snippets):
    # last 5 turns (pairs)
    conv = USER.get(user_id, {}).get("conversation", [])
    pairs = []
    i = 0
    while i < len(conv) - 1:
        if conv[i]["role"] == "user" and conv[i + 1]["role"] == "assistant":
            pairs.append({"user": conv[i]["text"], "assistant": conv[i + 1]["text"]})
            i += 2
        else:
            i += 1
    history_pairs = pairs[-5:]

    # semantic facts (compact)
    semantic_facts = []
    for f in col_semantic.find({"user_id": user_id}).limit(12):
        if f["type"] == "name":
            semantic_facts.append({"type": "name", "value": f.get("value")})
        elif f["type"] == "preference":
            semantic_facts.append({"type": "preference", "value": f.get("value")})
        elif f["type"] == "relation":
            semantic_facts.append({"type": "relation", "relation": f.get("relation"), "name": f.get("name")})
        elif f["type"] == "profession":
            semantic_facts.append({"type": "profession", "value": f.get("value")})

    # episodic (recent)
    episodic_docs = list(col_episodic.find({"user_id": user_id}).sort("time", -1).limit(8))
    episodic_summaries = [{"summary": d.get("summary", "")} for d in episodic_docs[:3]]

    # emotional velocity for current msg
    ev = compute_emotional_velocity(user_id, user_message)

   
    instruction = {
        "chat_history": history_pairs,
        "semantic_memories": semantic_facts[:10],
        "episodic_memories": episodic_summaries,
        "emotion_signals": ev["signals"],
        "emotional_velocity": ev["velocity"],
        "kb_snippets": kb_snippets,
        "user_input": user_message
    }

    prompt = (
        f"{AASHA_SYS}\n\n"
        "CONTEXT (JSON):\n"
        + json.dumps(instruction, ensure_ascii=False)
        + "\n\nAasha -> Return STRICT JSON only as {\"message\":\"...\"}:"
    )
    return prompt

# -------------------------
# Ollama call + extract
# -------------------------
def ollama_generate(prompt: str):
    url = f"{OLLAMA_URL}/api/generate"
    payload = {"model": CHAT_MODEL, "prompt": prompt, "stream": False}
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    try:
        j = r.json()
        return (j.get("response") or j.get("text") or "").strip()
    except json.JSONDecodeError:
        last = ""
        for line in r.text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                last = obj.get("response") or obj.get("text") or last
            except Exception:
                continue
        return (last or r.text).strip()

def extract_message(raw: str) -> str:
    texts = []
    def add_text(t):
        if not isinstance(t, str): return
        s = t.strip()
        s = re.sub(r"^(EmoAI|AI|Assistant|Aasha)\s*:\s*", "", s, flags=re.I)
        if s: texts.append(s)
    def coerce(obj):
        if isinstance(obj, dict):
            if isinstance(obj.get("message"), str):
                add_text(obj["message"])
            elif isinstance(obj.get("messages"), list):
                for it in obj["messages"]:
                    if isinstance(it, str): add_text(it)
                    elif isinstance(it, dict): add_text(it.get("text") or it.get("content") or it.get("message") or "")
            else:
                add_text(obj.get("text") or obj.get("content"))
        elif isinstance(obj, list):
            for it in obj:
                if isinstance(it, str): add_text(it)
                elif isinstance(it, dict): add_text(it.get("text") or it.get("content") or it.get("message") or "")
    try:
        obj = json.loads(raw); coerce(obj)
    except Exception:
        pass
    if not texts:
        blocks = re.findall(r"```(?:json|JSON)?\s*(.*?)\s*```", raw, flags=re.S)
        for blk in blocks:
            try:
                obj = json.loads(blk.strip()); coerce(obj)
            except Exception:
                continue
    if not texts:
        for m in re.finditer(r"(\{.*?\}|\[.*?\])", raw, flags=re.S):
            chunk = m.group(1).strip()
            if 2 < len(chunk) < 8000:
                try:
                    obj = json.loads(chunk); coerce(obj)
                except Exception:
                    continue
    if not texts:
        chunks = [s.strip() for s in re.split(r"\n\s*\n", raw) if s.strip()]
        if chunks: add_text(chunks[0])
        else: add_text(raw.strip())
    if not texts:
        return "I'm here with you. What's on your mind?"
    first = texts[0].strip()
    first = re.sub(r"\s+\n\s+|\s{2,}", " ", first).strip()
    return first

# -------------------------
# HTTP: /chat
# -------------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}

    uid_raw = data.get("user_id")
    user_id = (str(uid_raw).strip() if uid_raw is not None else "")

    msg_raw = None
    for k in ("chat", "message", "text", "input", "content"):
        if k in data:
            msg_raw = data.get(k)
            break

    if isinstance(msg_raw, (dict, list)):
        if isinstance(msg_raw, dict) and "text" in msg_raw:
            msg_raw = msg_raw["text"]
        else:
            msg_raw = json.dumps(msg_raw, ensure_ascii=False)

    user_msg = msg_raw.strip() if isinstance(msg_raw, str) else None

    if not user_id or not user_msg:
        app.logger.warning("400 bad request: missing user_id or message")
        chat_trace(str(uid_raw) if uid_raw is not None else "", "bad_request", {
            "reason": "missing user_id or message",
            "keys": list(data.keys())
        })
        return jsonify({
            "error": "user_id and non-empty message required",
            "hint": "Provide user_id and one of chat|message|text|input|content"
        }), 400

    app.logger.info(f"/chat start uid={user_id}")
    chat_trace(user_id, "request_received", {"keys": list(data.keys()), "msg_len": len(user_msg)})

    state = USER.setdefault(user_id, {"conversation": []})

    # 1) Router LLM → analysis JSON
    chat_trace(user_id, "router_call_start", {"model": ROUTER_MODEL})
    try:
        analysis = call_router(user_id, user_msg)
        app.logger.info(f"router_ok uid={user_id}")
        chat_trace(user_id, "router_call_ok", {
            "crisis": analysis.get("crisis"),
            "semantic_count": len(analysis.get("memories", {}).get("semantic", [])),
            "episodic_count": len(analysis.get("memories", {}).get("episodic", []))
        })
    except Exception as e:
        chat_trace(user_id, "router_call_failed", {"error": str(e)})
        app.logger.exception("router_llm_failed")
        return jsonify({"error": "router_llm_failed", "details": str(e)}), 500

    # 2) Persist memories from analysis
    sem_upserts = 0
    for s in analysis.get("memories", {}).get("semantic", []):
        doc = {"user_id": user_id, "type": s["type"], "value": s.get("value")}
        if s["type"] == "relation":
            doc["relation"] = s.get("relation")
            doc["name"] = s.get("name")
        filt = {"user_id": user_id, "type": s["type"]}
        if "relation" in doc: filt["relation"] = doc["relation"]
        if "name" in doc:     filt["name"] = doc["name"]
        res = col_semantic.update_one(filt, {"$set": doc}, upsert=True)
        sem_upserts += 1
        chat_trace(user_id, "semantic_upsert", {
            "type": s["type"], "matched": res.matched_count,
            "modified": res.modified_count,
            "upserted_id": str(res.upserted_id) if getattr(res, "upserted_id", None) else None
        })
    if sem_upserts:
        chat_trace(user_id, "semantic_summary", {"upserts": sem_upserts})

    epi_inserts, embed_ok, embed_fail = 0, 0, 0
    for ev in analysis.get("memories", {}).get("episodic", []):
        summ = (ev.get("summary", "") or "")[:500]
        edoc = {"user_id": user_id, "summary": summ, "emotions": [], "time": datetime.utcnow()}
        try:
            edoc["embedding"] = get_embedding(summ); embed_ok += 1
        except Exception:
            embed_fail += 1
        res = col_episodic.insert_one(edoc)
        epi_inserts += 1
        chat_trace(user_id, "episodic_inserted", {"id": str(res.inserted_id), "has_emb": bool(edoc.get("embedding"))})
    if epi_inserts or embed_fail:
        chat_trace(user_id, "episodic_summary", {"inserts": epi_inserts, "embed_ok": embed_ok, "embed_fail": embed_fail})

    # 3) Crisis short-circuit
    cr = analysis.get("crisis", {"flag": False, "type": "none"})
    if cr.get("flag"):
        app.logger.warning(f"crisis_detected uid={user_id} type={cr.get('type')}")
        chat_trace(user_id, "crisis_short_circuit", {"type": cr.get("type")})
        msg = crisis_reply(cr.get("type", "none"))
        message = extract_message(json.dumps({"message": msg}))
        state["conversation"].append({"role": "user", "text": user_msg})
        state["conversation"].append({"role": "assistant", "text": message})
        chat_trace(user_id, "response_sent", {"status": "crisis"})
        return jsonify({"message": message, "meta": {"crisis": cr}})

    # 4) KB + Memories + EV
    try:
        facts_total = col_semantic.count_documents({"user_id": user_id})
        epis_total = col_episodic.count_documents({"user_id": user_id})
    except Exception:
        facts_total = epis_total = None

    # topic detection for CBT/DBT pulls
    guideline_topics = detect_guideline_topics(user_msg)

    chat_trace(user_id, "kb_retrieval_start", {"topk": KB_TOPK, "topics": guideline_topics})
    try:
        kb_lines = retrieve_kb(user_msg, k=KB_TOPK, topic_tags=guideline_topics)
        chat_trace(user_id, "kb_retrieval_done", {"returned": len(kb_lines)})
    except Exception as e:
        kb_lines = []
        chat_trace(user_id, "kb_retrieval_failed", {"error": str(e)})

    mem_lines = retrieve_memories(user_id, user_msg, k=3)
    chat_trace(user_id, "memory_retrieval", {"facts_total": facts_total, "epis_total": epis_total, "returned": len(mem_lines)})

    # Build Aasha prompt
    prompt = build_companion_prompt(user_id, user_msg, kb_lines)
    chat_trace(user_id, "prompt_built", {"chars": len(prompt)})
    try:
        chat_trace(user_id, "ollama_prompt", {"prompt": prompt})
    except Exception:
        pass

    # 5) Generate
    chat_trace(user_id, "ollama_call_start", {"model": CHAT_MODEL})
    try:
        raw = ollama_generate(prompt)
        app.logger.info(f"ollama_ok uid={user_id}")
        chat_trace(user_id, "ollama_call_ok", {"raw_chars": len(raw)})
        try:
            chat_trace(user_id, "ollama_response_preview", {"chars": len(raw) if isinstance(raw, str) else None, "preview": raw[:500] if isinstance(raw, str) else None})
        except Exception:
            pass
    except Exception as e:
        chat_trace(user_id, "ollama_call_failed", {"error": str(e)})
        app.logger.exception("ollama_failed")
        return jsonify({"error": "ollama_failed", "details": str(e)}), 500

    message = extract_message(raw)
    chat_trace(user_id, "message_extracted", {"chars": len(message), "preview": message[:120]})

    # 6) Log conversation
    state["conversation"].append({"role": "user", "text": user_msg})
    state["conversation"].append({"role": "assistant", "text": message})
    chat_trace(user_id, "conversation_appended", {"total_turns": len(state["conversation"])})

    app.logger.info(f"/chat done uid={user_id}")
    chat_trace(user_id, "response_sent", {"status": "ok"})
    return jsonify({"message": message, "meta": {"analysis": analysis}})

# -------------------------
# Google Drive KB ingestion/sync
# -------------------------
def build_drive_service():
    if not GDRIVE_FOLDER_ID:
        raise RuntimeError("GDRIVE_FOLDER_ID not set")
    if not os.path.isfile(GOOGLE_CREDENTIALS_PATH):
        raise RuntimeError(f"Missing Google credentials file at {GOOGLE_CREDENTIALS_PATH}")
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
        if not page_token:
            break
    return files

def download_drive_file_text(service, file_id, mime_type):
    if mime_type == "application/vnd.google-apps.document":
        data = service.files().export(fileId=file_id, mimeType="text/plain").execute()
        return data.decode("utf-8", errors="ignore")

    req = service.files().get_media(fileId=file_id)
    buf = BytesIO()
    downloader = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    data = buf.getvalue()

    if mime_type in ("text/markdown", "text/plain"):
        return data.decode("utf-8", errors="ignore")

    if mime_type in ("application/pdf",):
        try:
            reader = PdfReader(BytesIO(data))
            txt = []
            for page in reader.pages:
                t = page.extract_text() or ""
                if t: txt.append(t)
            return "\n".join(txt)
        except Exception:
            return ""

    if mime_type in ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"):
        try:
            doc = Document(BytesIO(data))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            return ""

    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def chunk_text(text, max_chars=1800, overlap=200):
    text = (text or "")
    if not text: return []
    chunks, i, n = [], 0, len(text)
    while i < n:
        j = min(i + max_chars, n)
        chunks.append(text[i:j])
        if j == n: break
        i = max(0, j - overlap)
    return chunks

def title_tags(title: str):
    t = (title or "").strip().lower()
    parts = re.split(r"[\s_\-\.\(\)\[\]:]+", t)
    parts = [p for p in parts if p]
    return list({p for p in parts if len(p) >= 3})

def upsert_kb_for_file(file_meta, text):
    fid = file_meta["id"]
    title = file_meta.get("name", "")
    mime  = file_meta.get("mimeType", "")
    mtime = file_meta.get("modifiedTime", "")
    content_hash = hashlib.sha256((text or "").encode("utf-8")).hexdigest()

    src = col_kb_src.find_one({"file_id": fid})
    if src and src.get("modifiedTime") == mtime and src.get("content_hash") == content_hash:
        return {"status": "skip", "file_id": fid, "title": title}

    col_kb.delete_many({"source_id": fid})

    # Derive tags from title + explicit CBT/DBT normalizations
    tags = title_tags(title)
    lowered = title.lower()
    if "cbt" in lowered or "cognitive" in lowered: tags += ["cbt"]
    if "dbt" in lowered or "dialectical" in lowered: tags += ["dbt"]
    tags = list(sorted(set(tags)))

    chunks = chunk_text(text, max_chars=CHUNK_SIZE, overlap=KB_OVERLAP)
    now = datetime.utcnow()
    for idx, ch in enumerate(chunks):
        try:
            emb = get_embedding(ch)
        except Exception:
            emb = None
        doc = {
            "source_id": fid,
            "title": title,
            "mimeType": mime,
            "chunk_index": idx,
            "text": ch,
            "embedding": emb,
            "tags": tags,
            "time": now
        }
        col_kb.insert_one(doc)

    col_kb_src.update_one(
        {"file_id": fid},
        {"$set": {
            "file_id": fid,
            "title": title,
            "mimeType": mime,
            "modifiedTime": mtime,
            "content_hash": content_hash,
            "chunk_count": len(chunks),
            "tags": tags,
            "time": now
        }},
        upsert=True
    )
    return {"status": "updated", "file_id": fid, "title": title, "chunks": len(chunks)}

def delete_kb_for_file(file_id: str):
    col_kb.delete_many({"source_id": file_id})
    col_kb_src.delete_one({"file_id": file_id})

# -------------------------
# Drive Changes watcher
# -------------------------
def _settings_get(key, default=None):
    doc = col_settings.find_one({"_id": key})
    return (doc or {}).get("val", default)

def _settings_set(key, val):
    col_settings.update_one({"_id": key}, {"$set": {"val": val, "time": datetime.utcnow()}}, upsert=True)

def get_drive_start_page_token(service):
    res = service.changes().getStartPageToken().execute()
    return res.get("startPageToken")

def watch_drive_changes_loop():
    if not GDRIVE_FOLDER_ID:
        app.logger.warning("Drive watcher disabled: GDRIVE_FOLDER_ID not set")
        return
    try:
        service = build_drive_service()
    except Exception:
        app.logger.exception("Drive watcher: failed to build service")
        return

    token_key = "drive_changes_start_page_token"
    page_token = _settings_get(token_key)
    if not page_token:
        try:
            page_token = get_drive_start_page_token(service)
            _settings_set(token_key, page_token)
            app.logger.info(f"Drive watcher: initialized start page token {page_token}")
        except Exception:
            app.logger.exception("Drive watcher: failed to get start page token")
            return

    fields = "newStartPageToken,nextPageToken,changes(fileId,removed,file(id,name,mimeType,parents,modifiedTime,md5Checksum,trashed))"
    while True:
        try:
            pt = page_token
            while pt:
                resp = service.changes().list(
                    pageToken=pt,
                    spaces="drive",
                    fields=fields,
                    pageSize=100,
                    includeItemsFromAllDrives=True,
                    supportsAllDrives=True
                ).execute()

                for ch in resp.get("changes", []):
                    fid = ch.get("fileId")
                    file = ch.get("file") or {}
                    removed = ch.get("removed") or file.get("trashed")
                    if removed:
                        delete_kb_for_file(fid)
                        continue
                    parents = file.get("parents") or []
                    if GDRIVE_FOLDER_ID not in parents:
                        continue
                    try:
                        text = download_drive_file_text(service, fid, file.get("mimeType"))
                        upsert_kb_for_file(file, text)
                    except Exception:
                        app.logger.exception(f"Drive watcher: upsert failed for {fid}")

                new_token = resp.get("newStartPageToken")
                next_token = resp.get("nextPageToken")
                if next_token:
                    pt = next_token
                else:
                    if new_token:
                        page_token = new_token
                        _settings_set(token_key, page_token)
                        app.logger.info(f"Drive watcher: updated page token to {page_token}")
                    break
        except Exception:
            app.logger.exception("Drive watcher: error while polling changes")
        time.sleep(20)

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    app.logger.info(f"Starting server on 0.0.0.0:{PORT} debug={FLASK_DEBUG}")
    threading.Thread(target=watch_drive_changes_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, threaded=True, debug=FLASK_DEBUG, use_reloader=FLASK_DEBUG)
