import os, re, math, json, requests
from datetime import datetime
from flask import Flask, request, jsonify
from pymongo import MongoClient
from bson import ObjectId
from transformers import pipeline
from transitions import Machine
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
LOG_FILE       = os.getenv("LOG_FILE", os.path.join(os.path.dirname(__file__), "logs.log"))
PORT           = int(os.getenv("PORT", "5001"))
FLASK_DEBUG    = os.getenv("FLASK_DEBUG", "1").strip().lower() in ("1", "true", "t", "yes", "y", "on")


GDRIVE_FOLDER_ID        = os.getenv("GDRIVE_FOLDER_ID")
GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", os.path.join(os.path.dirname(__file__), "client.json"))
KB_OVERLAP              = int(os.getenv("KB_OVERLAP") or "0")
KB_TOPK                 = int(os.getenv("KB_TOPK", "2"))
CHUNK_SIZE              = 1500  # fixed chunk size per request


EMBED_MODEL   = "gemini-embedding-001"      # Gemini embeddings for RAG
ROUTER_MODEL  = "gemini-2.5-flash"        # Router that emits JSON analysis
CHAT_MODEL    = "emoai-sarah"             # Local Ollama model for the actual companion chat


CONV_STATES = [
    "casual_chat",
    "light_support",
    "deeper_exploration",
    "skill_offering",
    "crisis_support",
    "cool_down",
    "idle",
]


app = Flask(__name__)
# Optional module logger for non-request helpers
logger = logging.getLogger(__name__)




def setup_logging(flask_app: Flask):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True) if os.path.dirname(LOG_FILE) else None
    handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)

    # Attach to Flask app logger
    flask_app.logger.setLevel(logging.INFO)
    flask_app.logger.addHandler(handler)

    # Also attach to Werkzeug to capture access logs
    werk = logging.getLogger("werkzeug")
    werk.setLevel(logging.INFO)
    werk.addHandler(handler)

setup_logging(app)

# Optional per-request logging
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

# Error logging while preserving HTTPException behavior
@app.errorhandler(Exception)
def _handle_exception(e):
    if isinstance(e, HTTPException):
        app.logger.warning(f"HTTPException {e.code} on {request.method} {request.path}: {e.description}")
        return e
    app.logger.exception("Unhandled exception during request")
    return jsonify({"error": "internal_error"}), 500

mc = MongoClient(MONGO_URI)
db = mc[DB_NAME]

col_semantic  = db.semantic_memory   # user facts (name, prefs, relations, work)
col_episodic  = db.episodic_memory   # high-level event summaries
col_diff      = db.diff_memory       # state changes over time (emotion/engagement)
# Knowledge base (global corpus)
col_kb_src    = db.kb_sources        # records per Drive file
col_kb        = db.kb_chunks         # chunked embeddings
col_settings = db.settings           # generic settings (e.g., Drive changes page token)


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
# Ephemeral per-user state (in-memory)
# -------------------------

USER = {}  # user_id -> {"conversation": [...], "fsm": DuoFSM(), "conv": ConversationFSM()}

# -------------------------
# Utils
# -------------------------

def get_embedding(text: str):
    """Embed text using Gemini embeddings for RAG retrieval."""
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_KEY required")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBED_MODEL}:embedContent?key={GEMINI_KEY}"
    payload = {
        "content": {
            "parts": [{"text": text}]
        }
    }
    resp = requests.post(url, json=payload, timeout=45)
    resp.raise_for_status()
    data = resp.json()
    # Gemini returns: {"embedding": {"values": [float,...]}}
    emb = ((data or {}).get("embedding") or {}).get("values")
    if not emb:
        raise RuntimeError("embedding_failed")
    return emb

def cosine(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return dot / (na*nb) if na and nb else 0.0

def retrieve_memories(user_id: str, query_text: str, k=3):
    """
    Simple RAG: vector-sim episodic memories + a few semantic facts.
    Keeps context lightweight for the companion model.
    """
    # Vector similarity on episodic
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

    # Semantic facts (lightweight)
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
    """Friendly personalization if we’ve saved a name."""
    doc = col_semantic.find_one({"user_id": user_id, "type": "name"})
    if not doc:
        return None
    val = (doc.get("value") or "").strip()
    return val.split()[0] if val else None

# -------------------------
# KB retrieval (neutral, non-therapeutic)
# -------------------------

def retrieve_kb(query_text: str, k=2):
    """
    Retrieve up to k KB snippets by cosine similarity using stored embeddings.
    Returns a list of small strings. Best-effort; safe to return [].
    """
    try:
        qemb = get_embedding(query_text)
    except Exception:
        return []

    # Only consider chunks that actually have embeddings
    cur = col_kb.find({"embedding": {"$ne": None}}, {"text": 1, "embedding": 1}).limit(2000)
    ranked = []
    for doc in cur:
        emb = doc.get("embedding")
        if not emb:
            continue
        try:
            score = cosine(qemb, emb)
        except Exception:
            score = 0.0
        ranked.append((score, doc.get("text", "")[:400]))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in ranked[:k]]

# -------------------------
# FSM (emotion + engagement)
# -------------------------

class DuoFSM:
    """
    FSM with two tracks:
      - emotion_state: neutral|positive|negative
      - engagement_state: engaged|withdrawn|looping|intimate
    Transitions are applied from the router-LLM JSON, not hand-coded heuristics.
    """
    def __init__(self):
        self.emotion_state = "neutral"
        self.engagement_state = "engaged"

        self.emotion_machine = Machine(
            model=self,
            states=["neutral", "positive", "negative"],
            transitions=[
                {"trigger": "set_emotion", "source": "*", "dest": "negative", "conditions": lambda: self._next == "negative"},
                {"trigger": "set_emotion", "source": "*", "dest": "positive", "conditions": lambda: self._next == "positive"},
                {"trigger": "set_emotion", "source": "*", "dest": "neutral",  "conditions": lambda: self._next == "neutral"},
            ],
            initial="neutral",
            model_attribute="emotion_state",  # use model_attribute with transitions>=0.9
        )

        self.engagement_machine = Machine(
            model=self,
            states=["engaged", "withdrawn", "looping", "intimate"],
            transitions=[
                {"trigger": "set_eng", "source": "*", "dest": "withdrawn", "conditions": lambda: self._next_e == "withdrawn"},
                {"trigger": "set_eng", "source": "*", "dest": "looping",   "conditions": lambda: self._next_e == "looping"},
                {"trigger": "set_eng", "source": "*", "dest": "intimate",  "conditions": lambda: self._next_e == "intimate"},
                {"trigger": "set_eng", "source": "*", "dest": "engaged",   "conditions": lambda: self._next_e == "engaged"},
            ],
            initial="engaged",
            model_attribute="engagement_state",
        )

        self._next   = "neutral"
        self._next_e = "engaged"

    def apply(self, emo_next, eng_next):
        self._next, self._next_e = emo_next, eng_next
        self.set_emotion()
        self.set_eng()

# New: High-level conversation mode FSM
class ConversationFSM:
    """
    High-level conversational mode controller.
    States: casual_chat | light_support | deeper_exploration | skill_offering | crisis_support | cool_down | idle
    Transitions are driven by router JSON (or overridden to crisis_support when crisis=true).
    """
    def __init__(self):
        self.conversation_state = "casual_chat"
        self._next_c = "casual_chat"

        # Build transitions dynamically for readability
        trans = [
            {"trigger": "set_conv", "source": "*", "dest": s, "conditions": (lambda s=s: self._next_c == s)}
            for s in CONV_STATES
        ]

        self.conv_machine = Machine(
            model=self,
            states=CONV_STATES,
            transitions=trans,
            initial="casual_chat",
            model_attribute="conversation_state",
        )

    def apply(self, next_state: str):
        self._next_c = next_state or "casual_chat"
        if self._next_c not in CONV_STATES:
            self._next_c = "casual_chat"
        self.set_conv()

# -------------------------
# Router LLM (analysis JSON)
# -------------------------

ROUTER_SYS = (
    "You are an analysis router for a friendly, grounded chat companion. "
    "Return STRICT JSON ONLY with keys: "
    "{ 'crisis': {'flag': bool, 'type': 'none|suicidality|psychosis|abuse|dysregulation'}, "
    "'fsm': {'emotion':'neutral|positive|negative','engagement':'engaged|withdrawn|looping|intimate'}, "
    "'conversation': {'state':'casual_chat|light_support|deeper_exploration|skill_offering|crisis_support|cool_down|idle', 'confidence': number}, "
    "'memories': {'semantic': [ { 'type':'name|preference|relation|profession', 'value': str, 'relation': str|null, 'name': str|null } ], "
    "'episodic': [ { 'summary': str } ], 'diff': [ { 'kind':'emotion_state_change|engagement_state_change', 'from': str, 'to': str } ] }, "
    "'graph': { 'edges': [ ['node_a','node_b'] ] } } "
    "Rules: keep it minimal; summarize episodic at high level; include only key facts; "
    "set crisis true ONLY if the message indicates imminent risk; prefer 'none' otherwise; "
    "IF crisis.flag is true, set conversation.state='crisis_support'; "
    "avoid romantic/parasocial attachments; avoid irreversible advice."
)

def call_router(user_id: str, user_message: str, last_states):
    """
    Calls the Gemini router model to emit a canonical analysis JSON:
    - crisis flag/type
    - FSM targets
    - memories (semantic/episodic/diff)
    - graph edges
    """
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

    # Gemini generateContent with systemInstruction and JSON-only output
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{ROUTER_MODEL}:generateContent?key={GEMINI_KEY}"
    body = {
        "systemInstruction": {
            "parts": [{"text": ROUTER_SYS}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{
                    "text": json.dumps({
                        "user_id": user_id,
                        "message": user_message,
                        "last_states": last_states,
                        "signals": sig
                    }, ensure_ascii=False)
                }]
            }
        ],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.2
        }
    }
    r = requests.post(url, json=body, timeout=45)
    r.raise_for_status()
    data = r.json()
    try:
        txt = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        raise RuntimeError(f"router_parse_failed: {data}") from e
    return json.loads(txt)

# -------------------------
# Crisis replies (warm, non-clinical, no exercises)
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

def build_companion_prompt(user_id, user_message, analysis_json, memories_for_prompt, convo_tail):
    """
    Build a prompt that asks the Ollama chat model to return exactly one short chat message as JSON.
    Tone: friendly, grounded, non-clinical. No unsolicited techniques or advice.
    """
    persona = (
        "You are Emo AI — a grounded, everyday chat companion (not a clinician). "
        "Sound like a caring friend. Keep it short and natural. "
        "Do NOT suggest techniques, exercises, or action steps unless the user explicitly asks. "
        "Stay on the user's topic; be concise and real."
    )

    fsm = analysis_json.get("fsm", {})
    conv = (analysis_json.get("conversation", {}) or {}).get("state", "casual_chat")
    ctx = [
        f"Emotion={fsm.get('emotion','neutral')}, Engagement={fsm.get('engagement','engaged')}.",
        f"Mode={conv.upper()}."
    ]
    for line in memories_for_prompt[:2]:
        ctx.append(line)

    first_name = get_first_name(user_id)
    name_line = f"FirstName={first_name}." if first_name else "FirstName=null."

    style_rules = (
        "OUTPUT FORMAT: Return STRICT JSON exactly in this shape (and nothing else): "
        "{\"message\": \"...\"}. "
        "A single short string in 'message'. "
        "No other keys. No markdown/code fences. No prefixes like 'EmoAI:'. "
        "STYLE: Warm, succinct, human. "
        "Ask at most one brief clarifying question only if it's necessary to respond helpfully. "
        "Do NOT propose coping strategies, breathing, grounding, journaling, routines, or any how-to advice unless the user asked. "
        "If Mode=CRISIS_SUPPORT, be very brief and compassionate."
    )

    tail = ""
    if convo_tail:
        for pair in convo_tail[-2:]:
            tail += f"User: {pair['u']}\nEmoAI: {pair['a']}\n"

    prompt = (
        f"{persona}\n\n"
        f"{style_rules}\n\n"
        f"Context:\n- {name_line}\n- " + "\n- ".join(ctx) + "\n\n"
        + (f"Recent:\n{tail}\n" if tail else "")
        + "Now craft one chat message as per OUTPUT FORMAT and STYLE.\n\n"
        f"User: {user_message}\n"
        "EmoAI -> JSON only:"
    )
    return prompt

def ollama_generate(prompt: str):
    """Query local Ollama and return raw text (ideally JSON per our prompt)."""
    url = f"{OLLAMA_URL}/api/generate"
    payload = {"model": CHAT_MODEL, "prompt": prompt, "stream": False}
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    try:
        j = r.json()
        return (j.get("response") or j.get("text") or "").strip()
    except json.JSONDecodeError:
        # Fallback for NDJSON/stream-like responses
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

# Single-message extractor: normalize model output to one concise string
def extract_message(raw: str) -> str:
    texts = []

    def add_text(t):
        if not isinstance(t, str):
            return
        s = t.strip()
        s = re.sub(r"^(EmoAI|AI|Assistant)\s*:\s*", "", s, flags=re.I)
        if s:
            texts.append(s)

    def coerce(obj):
        if isinstance(obj, dict):
            if isinstance(obj.get("message"), str):
                add_text(obj["message"])
            elif isinstance(obj.get("messages"), list):
                for it in obj["messages"]:
                    if isinstance(it, str):
                        add_text(it)
                    elif isinstance(it, dict):
                        add_text(it.get("text") or it.get("content") or it.get("message") or "")
            else:
                add_text(obj.get("text") or obj.get("content"))
        elif isinstance(obj, list):
            for it in obj:
                if isinstance(it, str):
                    add_text(it)
                elif isinstance(it, dict):
                    add_text(it.get("text") or it.get("content") or it.get("message") or "")

    # 1) Try direct JSON
    try:
        obj = json.loads(raw)
        coerce(obj)
    except Exception:
        pass

    # 2) Try fenced code blocks ```json ... ```
    if not texts:
        blocks = re.findall(r"```(?:json|JSON)?\s*(.*?)\s*```", raw, flags=re.S)
        for blk in blocks:
            try:
                obj = json.loads(blk.strip())
                coerce(obj)
            except Exception:
                continue

    # 3) Try inline JSON chunks
    if not texts:
        for m in re.finditer(r"(\{.*?\}|\[.*?\])", raw, flags=re.S):
            chunk = m.group(1).strip()
            if 2 < len(chunk) < 8000:
                try:
                    obj = json.loads(chunk)
                    coerce(obj)
                except Exception:
                    continue

    # 4) Fallbacks
    if not texts:
        # pick first paragraph-like chunk
        chunks = [s.strip() for s in re.split(r"\n\s*\n", raw) if s.strip()]
        if chunks:
            add_text(chunks[0])
        else:
            add_text(raw.strip())

    # Finalize
    if not texts:
        return "I'm here with you. What's on your mind?"

    # Prefer the first concise line
    first = texts[0].strip()
    # Collapse internal whitespace
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

    # Be flexible with the message key
    msg_raw = None
    for k in ("chat", "message", "text", "input", "content"):
        if k in data:
            msg_raw = data.get(k)
            break

    # Normalize nested payloads
    if isinstance(msg_raw, (dict, list)):
        if isinstance(msg_raw, dict) and "text" in msg_raw:
            msg_raw = msg_raw["text"]
        else:
            msg_raw = json.dumps(msg_raw, ensure_ascii=False)

    user_msg = msg_raw.strip() if isinstance(msg_raw, str) else None

    if not user_id or not user_msg:
        app.logger.warning("400 bad request: missing user_id or message")
        return jsonify({
            "error": "user_id and non-empty message required",
            "hint": "Provide user_id and one of chat|message|text|input|content"
        }), 400

    app.logger.info(f"/chat start uid={user_id}")

    state = USER.setdefault(user_id, {"conversation": [], "fsm": DuoFSM(), "conv": ConversationFSM()})
    last_states = {
        "emotion": state["fsm"].emotion_state,
        "engagement": state["fsm"].engagement_state,
        "conversation": state["conv"].conversation_state
    }

    # 1) Router LLM → analysis JSON
    try:
        analysis = call_router(user_id, user_msg, last_states)
        app.logger.info(f"router_ok uid={user_id} conv_target={(analysis.get('conversation',{}) or {}).get('state')}")
    except Exception as e:
        app.logger.exception("router_llm_failed")
        return jsonify({"error": "router_llm_failed", "details": str(e)}), 500

    # 2) Persist memories & diffs
    # Semantic (upsert light facts)
    for s in analysis.get("memories", {}).get("semantic", []):
        doc = {"user_id": user_id, "type": s["type"], "value": s.get("value")}
        if s["type"] == "relation":
            doc["relation"] = s.get("relation")
            doc["name"] = s.get("name")
        # naive upsert by (user_id, type, relation, name)
        filt = {"user_id": user_id, "type": s["type"]}
        if "relation" in doc: filt["relation"] = doc["relation"]
        if "name" in doc:     filt["name"] = doc["name"]
        col_semantic.update_one(filt, {"$set": doc}, upsert=True)

    # Episodic (embed on write for RAG)
    for ev in analysis.get("memories", {}).get("episodic", []):
        summ = (ev.get("summary", "") or "")[:500]
        edoc = {
            "user_id": user_id,
            "summary": summ,
            "emotions": [],  # could be added by router if desired
            "time": datetime.utcnow()
        }
        try:
            edoc["embedding"] = get_embedding(summ)
        except Exception:
            pass
        col_episodic.insert_one(edoc)

    # Diff (state change log from router JSON)
    for d in analysis.get("memories", {}).get("diff", []):
        d["user_id"] = user_id
        d["time"] = datetime.utcnow()
        col_diff.insert_one(d)

    # 3) Apply FSM targets from analysis JSON
    fsm = analysis.get("fsm", {})
    state["fsm"].apply(fsm.get("emotion", "neutral"), fsm.get("engagement", "engaged"))

    # Apply ConversationFSM (with crisis override) and log diff
    prev_conv = state["conv"].conversation_state
    conv_target = ((analysis.get("conversation", {}) or {}).get("state") or "casual_chat")
    if analysis.get("crisis", {}).get("flag"):
        conv_target = "crisis_support"
    if conv_target not in CONV_STATES:
        conv_target = "casual_chat"
    if prev_conv != conv_target:
        col_diff.insert_one({
            "user_id": user_id,
            "time": datetime.utcnow(),
            "kind": "conversation_state_change",
            "from": prev_conv,
            "to": conv_target
        })
    state["conv"].apply(conv_target)

    # 4) Crisis short-circuit
    cr = analysis.get("crisis", {"flag": False, "type": "none"})
    if cr.get("flag"):
        app.logger.warning(f"crisis_detected uid={user_id} type={cr.get('type')}")
        msg = crisis_reply(cr.get("type", "none"))
        message = extract_message(json.dumps({"message": msg}))
        state["conversation"].append({"role": "user", "text": user_msg})
        state["conversation"].append({"role": "assistant", "text": message})
        return jsonify({
            "message": message,
            "meta": {
                "crisis": cr,
                "fsm": {
                    "emotion": state["fsm"].emotion_state,
                    "engagement": state["fsm"].engagement_state
                },
                "conversation": state["conv"].conversation_state
            }
        })

    # 5) Build grounded prompt & query Ollama
    mem_lines = retrieve_memories(user_id, user_msg, k=3)

    # Bring in a couple of relevant KB snippets (neutral KB)
    try:
        kb_lines = retrieve_kb(user_msg, k=KB_TOPK)
    except Exception:
        kb_lines = []
    mem_lines = (kb_lines[:KB_TOPK]) + mem_lines

    # last 2 exchanges for continuity
    tail = []
    conv = state["conversation"]
    for i in range(len(conv) - 1):
        if conv[i]["role"] == "user" and conv[i + 1]["role"] == "assistant":
            tail.append({"u": conv[i]["text"], "a": conv[i + 1]["text"]})

    prompt = build_companion_prompt(user_id, user_msg, analysis, mem_lines, tail)

    try:
        raw = ollama_generate(prompt)
        app.logger.info(f"ollama_ok uid={user_id}")
    except Exception as e:
        app.logger.exception("ollama_failed")
        return jsonify({"error": "ollama_failed", "details": str(e)}), 500

    # Normalize to a single message regardless of model quirks
    message = extract_message(raw)

    # 6) Log conversation
    state["conversation"].append({"role": "user", "text": user_msg})
    state["conversation"].append({"role": "assistant", "text": message})

    app.logger.info(f"/chat done uid={user_id} mode={state['conv'].conversation_state} emo={state['fsm'].emotion_state} eng={state['fsm'].engagement_state}")
    return jsonify({
        "message": message,
        "meta": {
            "fsm": {
                "emotion": state["fsm"].emotion_state,
                "engagement": state["fsm"].engagement_state
            },
            "conversation": state["conv"].conversation_state
        }
    })

# -------------------------
# Knowledge Base: Google Drive sync and retrieval
# -------------------------

def build_drive_service():
    """Create a Drive API client using service account credentials."""
    if not GDRIVE_FOLDER_ID:
        raise RuntimeError("GDRIVE_FOLDER_ID not set")
    if not os.path.isfile(GOOGLE_CREDENTIALS_PATH):
        raise RuntimeError(f"Missing Google credentials file at {GOOGLE_CREDENTIALS_PATH}")
    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    creds = service_account.Credentials.from_service_account_file(GOOGLE_CREDENTIALS_PATH, scopes=scopes)
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def list_drive_files(service, folder_id):
    """List files in a Drive folder (no trashed)."""
    files = []
    page_token = None
    q = f"'{folder_id}' in parents and trashed=false"
    fields = "nextPageToken, files(id,name,mimeType,modifiedTime,md5Checksum,size)"
    while True:
        resp = service.files().list(q=q, fields=fields, pageToken=page_token).execute()
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files

def download_drive_file_text(service, file_id, mime_type):
    """Download a Drive file and return UTF-8 text (no OCR)."""
    # Google Docs → export as text/plain
    if mime_type == "application/vnd.google-apps.document":
        data = service.files().export(fileId=file_id, mimeType="text/plain").execute()
        return data.decode("utf-8", errors="ignore")

    # Binary files → get_media
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
                if t:
                    txt.append(t)
            return "\n".join(txt)
        except Exception:
            return ""

    if mime_type in ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"):
        try:
            doc = Document(BytesIO(data))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            return ""

    # Fallback attempt
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def chunk_text(text, max_chars=1800, overlap=200):
    """Simple char-based chunking with overlap."""
    text = (text or "")
    if not text:
        return []
    chunks, i, n = [], 0, len(text)
    while i < n:
        j = min(i + max_chars, n)
        chunks.append(text[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks

def title_tags(title: str):
    """Derive light tags from title (e.g., words >=3 chars)."""
    t = (title or "").strip().lower()
    parts = re.split(r"[\s_\-\.\(\)\[\]:]+", t)
    parts = [p for p in parts if p]
    return list({p for p in parts if len(p) >= 3})

def upsert_kb_for_file(file_meta, text):
    """Upsert chunks for a single Drive file into Mongo KB."""
    fid = file_meta["id"]
    title = file_meta.get("name", "")
    mime  = file_meta.get("mimeType", "")
    mtime = file_meta.get("modifiedTime", "")
    content_hash = hashlib.sha256((text or "").encode("utf-8")).hexdigest()

    src = col_kb_src.find_one({"file_id": fid})
    if src and src.get("modifiedTime") == mtime and src.get("content_hash") == content_hash:
        return {"status": "skip", "file_id": fid, "title": title}

    # Remove old chunks if any
    col_kb.delete_many({"source_id": fid})

    # Chunk + embed
    chunks = chunk_text(text, max_chars=CHUNK_SIZE, overlap=KB_OVERLAP)
    tags = title_tags(title)
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
    """Remove KB records for a removed/trashed file."""
    col_kb.delete_many({"source_id": file_id})
    col_kb_src.delete_one({"file_id": file_id})

# -------------------------
# Drive Changes watcher
# -------------------------

def _settings_get(key, default=None):
    doc = col_settings.find_one({"_id": key})
    return (doc or {}).get("val", default)

def _settings_set(key, val):
    col_settings.update_one(
        {"_id": key},
        {"$set": {"val": val, "time": datetime.utcnow()}},
        upsert=True
    )

def get_drive_start_page_token(service):
    """Fetch current Drive start page token."""
    res = service.changes().getStartPageToken().execute()
    return res.get("startPageToken")

def watch_drive_changes_loop():
    """Background loop: poll Drive Changes API and keep KB in sync in near real time."""
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
                    # If removed/trashed → delete KB entries
                    if removed:
                        delete_kb_for_file(fid)
                        continue
                    # Process only files in our folder
                    parents = file.get("parents") or []
                    if GDRIVE_FOLDER_ID not in parents:
                        # Not in our watched folder → ignore (but optional cleanup if previously indexed)
                        continue
                    # Upsert the changed file
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
        # Poll interval
        time.sleep(20)

# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    app.logger.info(f"Starting server on 0.0.0.0:{PORT} debug={FLASK_DEBUG}")
    # Start Drive real-time watcher (changes API)
    threading.Thread(target=watch_drive_changes_loop, daemon=True).start()

    app.run(host="0.0.0.0", port=PORT, threaded=True, debug=FLASK_DEBUG, use_reloader=FLASK_DEBUG)
