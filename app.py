#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, math, json, requests
from datetime import datetime
from flask import Flask, request, jsonify
from pymongo import MongoClient
from bson import ObjectId
from transformers import pipeline
from transitions import Machine
from dotenv import load_dotenv
# NEW: logging imports
import logging
from logging.handlers import RotatingFileHandler
from werkzeug.exceptions import HTTPException

# -------------------------
# Environment & Config
# -------------------------

# Load environment variables from a .env file (if present)
load_dotenv()

# REPLACED: OpenAI key -> Gemini key
GEMINI_KEY    = os.getenv("GEMINI_KEY")
OLLAMA_URL    = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
MONGO_URI     = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME       = os.getenv("MONGODB_DB", "emoai")
# NEW: logging & debug config
LOG_FILE       = os.getenv("LOG_FILE", os.path.join(os.path.dirname(__file__), "logs.log"))
PORT           = int(os.getenv("PORT", "5000"))
FLASK_DEBUG    = os.getenv("FLASK_DEBUG", "1").strip().lower() in ("1", "true", "t", "yes", "y", "on")

# REPLACED: models to Gemini equivalents
EMBED_MODEL   = "text-embedding-004"      # Gemini embeddings for RAG memories
ROUTER_MODEL  = "gemini-2.5-flash"        # Gemini router that emits JSON analysis
CHAT_MODEL    = "emoai-sarah"             # Local Ollama model for the actual companion chat

# High-level conversation modes
CONV_STATES = [
    "casual_chat",
    "light_support",
    "deeper_exploration",
    "skill_offering",
    "crisis_support",
    "cool_down",
    "idle",
]

# -------------------------
# App & DB
# -------------------------

app = Flask(__name__)

# NEW: setup rotating file logging
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

# -------------------------
# Optional HF signals (sentiment/emotion/sarcasm)
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

    # Semantic facts (lightweight, no regex extraction here)
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
    "You are a safety-first, grounded analysis router for an emotional AI companion. "
    "You must return STRICT JSON ONLY with keys: "
    "{ 'crisis': {'flag': bool, 'type': 'none|suicidality|psychosis|abuse|dysregulation'}, "
    "'fsm': {'emotion':'neutral|positive|negative','engagement':'engaged|withdrawn|looping|intimate'}, "
    "'conversation': {'state':'casual_chat|light_support|deeper_exploration|skill_offering|crisis_support|cool_down|idle', 'confidence': number}, "
    "'memories': {'semantic': [ { 'type':'name|preference|relation|profession', 'value': str, 'relation': str|null, 'name': str|null } ], "
    "'episodic': [ { 'summary': str } ], 'diff': [ { 'kind':'emotion_state_change|engagement_state_change', 'from': str, 'to': str } ] }, "
    "'graph': { 'edges': [ ['node_a','node_b'] ] } } "
    "Rules: keep it minimal; summarize episodic at high level; include only key facts; "
    "set crisis true ONLY if the message indicates imminent risk; prefer 'none' otherwise; "
    "IF crisis.flag is true, set conversation.state='crisis_support'; "
    "avoid romantic/parasocial attachments; avoid irreversible advice; reflect benevolent friction."
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
# Crisis replies (warm, non-clinical)
# -------------------------

def crisis_reply(kind: str):
    if kind == "suicidality":
        return ("I'm really sorry you're feeling this much pain. You deserve support right now. "
                "If you can, please reach out to someone you trust or a local crisis line. "
                "I can sit with you for a moment—want to do a 10-second pause together?")
    if kind == "psychosis":
        return ("I'm sorry you're experiencing this. It might help to get medical support. "
                "While you think about that, could we focus on what’s certain around you for a moment?")
    if kind == "abuse":
        return ("I'm so sorry you're going through that. You don’t deserve to be hurt. "
                "If you're able, consider contacting trusted people or local services for safety. "
                "We can take one tiny step together if you'd like.")
    if kind == "dysregulation":
        return ("I can feel how intense this is. Let’s slow it down for a minute. "
                "Want a quick breathing check-in while you line up support?")
    return ("I'm here with you. You’re not alone, and it’s okay to ask for help close to you right now.")


# -------------------------

def build_companion_prompt(user_id, user_message, analysis_json, memories_for_prompt, convo_tail):
    """
    Build a prompt that asks the Ollama chat model to return exactly one short chat message as JSON.
    Tone: casual, warm, grounded, non-clinical, with benevolent friction used sparingly.
    """
    persona = (
        "You are Emo AI — a grounded, everyday chat companion (not a clinician). "
        "Sound like a caring friend, not a therapist. Keep it short and natural. "
        "Avoid canned phrases (e.g., 'let's take a moment', 'it might help to'). "
        "Use benevolent friction when useful (brief reflect, one small clarify, or a tiny next step). "
        "No romance/parasocial vibes. Never give irreversible advice."
    )

    fsm = analysis_json.get("fsm", {})
    conv = (analysis_json.get("conversation", {}) or {}).get("state", "casual_chat")
    ctx = [f"Emotion={fsm.get('emotion','neutral')}, Engagement={fsm.get('engagement','engaged')}.",
           f"Mode={conv.upper()}."]
    for line in memories_for_prompt[:2]:
        ctx.append(line)

    first_name = get_first_name(user_id)
    name_line = f"FirstName={first_name}." if first_name else "FirstName=null."

    style_rules = (
        "OUTPUT FORMAT: Return STRICT JSON exactly in this shape (and nothing else): "
        "{\"message\": \"...\"}. "
        "A single short string in 'message'. "
        "No other keys. No markdown/code fences. No 'speaker' or 'role' fields. "
        "No prefixes like 'EmoAI:' in the string. "
        "STYLE: Warm, succinct, human. Max one gentle question. "
        "Suggest a tiny next step only if it fits, phrased casually (e.g., 'want to try a 10-sec pause?'). "
        "Avoid clinical words ('CBT', 'exercise', 'grounding technique'). "
        "If FirstName exists, use it once. "
        "Mode guidance: if Mode=SKILL_OFFERING, ask consent before one practical tip; "
        "if Mode=CASUAL_CHAT, keep it light and friendly; "
        "if Mode=CRISIS_SUPPORT, be safety-first and very brief."
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

    # 1) Router LLM → analysis JSON (no regex; safety-first)
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

    # New: Apply ConversationFSM (with crisis override) and log diff
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
# Main
# -------------------------

if __name__ == "__main__":
    # In dev, set threaded=True so HF downloads don’t block other requests
    # Enable auto-reload for live editing when FLASK_DEBUG is true (default)
    app.logger.info(f"Starting server on 0.0.0.0:{PORT} debug={FLASK_DEBUG}")
    app.run(host="0.0.0.0", port=PORT, threaded=True, debug=FLASK_DEBUG, use_reloader=FLASK_DEBUG)
