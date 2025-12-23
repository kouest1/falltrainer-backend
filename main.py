import os
import json
import uuid
import secrets
import string
from datetime import datetime, timedelta
from typing import List, Literal, Optional, Dict, Any

import requests
from jose import jwt
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import Column, String, DateTime, Text
from openai import OpenAI

from database import Base, engine, SessionLocal
from models import User

# -------------------------------------------------------
# DB Model (neu): DuoSession -> in main.py, damit du models.py nicht anfassen musst
# -------------------------------------------------------

class DuoSession(Base):
    __tablename__ = "duo_sessions"

    session_id = Column(String, primary_key=True)               # uuid string
    join_code = Column(String, unique=True, index=True, nullable=False)

    doctor_user_id = Column(String, index=True, nullable=False)
    patient_user_id = Column(String, index=True, nullable=True)

    case_title = Column(String, nullable=True)
    case_description = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)


# -------------------------------------------------------
# Datenbank initialisieren
# -------------------------------------------------------

Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -------------------------------------------------------
# OpenAI-Client
# -------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY ist nicht gesetzt")

client = OpenAI(api_key=OPENAI_API_KEY)

# Plan-Konfiguration: Limit + Modell
PLAN_CONFIG = {
    "free": {
        "limit": 6,
        "model": os.getenv("OPENAI_MODEL_FREE", "gpt-4.5"),
    },
    "plus": {
        "limit": 150,
        "model": os.getenv("OPENAI_MODEL_PLUS", "gpt-5-mini"),
    },
    "premium": {
        "limit": 1000,
        "model": os.getenv("OPENAI_MODEL_PREMIUM", "gpt-5.1"),
    },
}


def current_month_str() -> str:
    """z.B. '2025-11' – so speichern wir die Nutzung pro Monat."""
    return datetime.utcnow().strftime("%Y-%m")


def get_limit_for_plan(plan: str) -> int:
    cfg = PLAN_CONFIG.get(plan, PLAN_CONFIG["free"])
    return int(cfg["limit"])


def get_model_for_plan(plan: str) -> str:
    cfg = PLAN_CONFIG.get(plan, PLAN_CONFIG["free"])
    return str(cfg["model"])


def call_openai(message: str, model_name: str) -> str:
    """
    Schickt den Prompt an OpenAI (Chat Completions) und gibt NUR die Modell-Antwort zurück.
    """
    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": message}],
        temperature=1,
    )
    content = completion.choices[0].message.content
    return content or ""

def call_openai_stream(message: str, model_name: str):
    """
    Streamt Text (Delta-Chunks) aus OpenAI ChatCompletions und yieldet Strings.
    """
    stream = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": message}],
        temperature=1,
        stream=True,
    )

    for event in stream:
        # event.choices[0].delta.content enthält neue Textstücke
        delta = event.choices[0].delta.content
        if delta:
            yield delta


def extract_text_from_responses_api(resp) -> str:
    """
    Holt Text aus OpenAI Responses API Antwort, mit Fallbacks.
    """
    try:
        return resp.output[0].content[0].text or ""
    except Exception:
        return getattr(resp, "output_text", "") or str(resp)


# -------------------------------------------------------
# User + Plan Logik
# -------------------------------------------------------

def get_or_create_user(db: Session, user_id: str) -> User:
    user = db.query(User).filter(User.user_id == user_id).first()
    if user is None:
        user = User(
            user_id=user_id,
            plan="free",
            monthly_usage=0,
            usage_month=None,
            excluded_cases=None,
            notes=None,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


def apply_month_reset(user: User):
    now_month = current_month_str()
    if user.usage_month != now_month:
        user.usage_month = now_month
        user.monthly_usage = 0


# -------------------------------------------------------
# FastAPI App & Konstanten
# -------------------------------------------------------

app = FastAPI()
APPLE_KEYS_URL = "https://appleid.apple.com/auth/keys"


# -------------------------------------------------------
# Schemas (Pydantic)
# -------------------------------------------------------

class AppleAuthPayload(BaseModel):
    token: str  # identityToken von Apple


class AppleAuthResponse(BaseModel):
    userId: str
    plan: str
    monthlyUsage: int


class ReceiptPayload(BaseModel):
    userId: str
    receipt: str  # Base64 aus iOS


class ReceiptResponse(BaseModel):
    plan: str
    monthlyUsage: int
    limit: int


class AskPayload(BaseModel):
    userId: str
    message: str


class AskResponse(BaseModel):
    reply: str
    usage: int
    limit: int


class ExcludedCasesPayload(BaseModel):
    userId: str
    excludedIds: list[str]


class ExcludedCasesResponse(BaseModel):
    excludedIds: list[str]


class NotesLoadPayload(BaseModel):
    userId: str


class NoteUpdatePayload(BaseModel):
    userId: str
    caseId: str
    note: str


class NotesResponse(BaseModel):
    notes: dict[str, str]


class DuoMessage(BaseModel):
    role: Literal["doctor", "patient"]
    content: str


class DuoChatRequest(BaseModel):
    userId: str
    caseTitle: str
    caseDescription: str
    messages: List[DuoMessage]


class DuoChatResponse(BaseModel):
    reply: str
    usage: Optional[int] = None
    limit: Optional[int] = None


class TransactionPayload(BaseModel):
    userId: str
    productId: str
    transactionJWS: Optional[str] = None
    planHint: Optional[str] = None


class TransactionResponse(BaseModel):
    plan: str
    monthlyUsage: int
    limit: int


# ---- Neu: Duo Session REST ----

class DuoSessionCreatePayload(BaseModel):
    userId: str
    caseTitle: Optional[str] = None
    caseDescription: Optional[str] = None


class DuoSessionCreateResponse(BaseModel):
    sessionId: str
    joinCode: str
    expiresAt: str


class DuoSessionJoinPayload(BaseModel):
    userId: str
    joinCode: str


class DuoSessionJoinResponse(BaseModel):
    sessionId: str
    ok: bool


# -------------------------------------------------------
# Helper für Apple Public Key
# -------------------------------------------------------

def get_apple_public_key(kid: str):
    apple_keys = requests.get(APPLE_KEYS_URL).json()["keys"]
    for key in apple_keys:
        if key["kid"] == kid:
            return key
    raise Exception("Apple Public Key nicht gefunden")


# -------------------------------------------------------
# WebSocket Manager (in-memory connections + in-memory history)
# -------------------------------------------------------

class WSManager:
    def __init__(self):
        # session_id -> {"doctor": set(ws), "patient": set(ws)}
        self.sessions: Dict[str, Dict[str, set[WebSocket]]] = {}
        # session_id -> history list[{"role": "...", "content": "..."}]
        self.history: Dict[str, List[Dict[str, str]]] = {}

    async def connect(self, session_id: str, role: str, websocket: WebSocket):
        # ✅ WICHTIG: KEIN websocket.accept() hier!
        self.sessions.setdefault(session_id, {"doctor": set(), "patient": set()})
        self.sessions[session_id][role].add(websocket)
        self.history.setdefault(session_id, [])

    def disconnect(self, session_id: str, role: str, websocket: WebSocket):
        try:
            self.sessions.get(session_id, {}).get(role, set()).discard(websocket)
        except Exception:
            pass

    async def broadcast(self, session_id: str, role: str, payload: Dict[str, Any]):
        targets = list(self.sessions.get(session_id, {}).get(role, set()))
        msg = json.dumps(payload)

        dead: List[WebSocket] = []
        for ws in targets:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)

        # kaputte sockets cleanup
        for ws in dead:
            self.sessions.get(session_id, {}).get(role, set()).discard(ws)

ws_manager = WSManager()


def generate_join_code(length: int = 6) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def build_coach_prompt(case_title: str, case_description: str, history: List[Dict[str, str]]) -> str:
    system_prompt = (
        "Du bist ein erfahrener Neurologe und Lehrarzt.\n"
        "Du siehst den Dialog zwischen einem Patienten und einem Assistenzarzt.\n"
        "Deine Aufgabe ist, knappe, konkrete Vorschlaege zu machen:\n"
        "- Welche Frage sollte der Arzt als naechstes stellen?\n"
        "- Welche koerperliche Untersuchung oder Zusatzdiagnostik bietet sich an?\n"
        "Antworte in 1-3 kurzen Saetzen auf Deutsch, als Vorschlag an den Arzt.\n\n"
        f"Falltitel: {case_title}\n\n"
        f"Fallbeschreibung (medizinischer Hintergrund):\n{case_description}\n"
    )

    lines = []
    for msg in history:
        if msg.get("role") == "patient":
            lines.append(f"Patient: {msg.get('content','')}")
        else:
            lines.append(f"Arzt: {msg.get('content','')}")
    conversation_text = "\n".join(lines) if lines else "(noch kein Dialog)"

    return (
        system_prompt
        + "\n\nBisheriger Dialog:\n"
        + conversation_text
        + "\n\nDein naechster Vorschlag an den Arzt:"
    )


# -------------------------------------------------------
# Endpoints
# -------------------------------------------------------

@app.get("/")
def root():
    return {"message": "Backend läuft!"}


# 1) Login mit Apple
@app.post("/auth/apple", response_model=AppleAuthResponse)
async def auth_apple(payload: AppleAuthPayload, db: Session = Depends(get_db)):
    identity_token = payload.token

    header = jwt.get_unverified_header(identity_token)
    kid = header["kid"]

    pubkey_dict = get_apple_public_key(kid)

    try:
        decoded = jwt.decode(
            identity_token,
            pubkey_dict,
            algorithms=["RS256"],
            audience="com.konstantin.Falltrainer",
            issuer="https://appleid.apple.com",
        )
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Ungültiges Apple Token: {e}")

    apple_user_id = decoded["sub"]

    user = get_or_create_user(db, apple_user_id)
    apply_month_reset(user)
    db.commit()
    db.refresh(user)

    return AppleAuthResponse(
        userId=apple_user_id,
        plan=user.plan,
        monthlyUsage=user.monthly_usage,
    )


# 2) Receipt validieren (Abo)
@app.post("/validateReceipt", response_model=ReceiptResponse)
async def validate_receipt(payload: ReceiptPayload, db: Session = Depends(get_db)):
    user_id = payload.userId
    receipt_data = payload.receipt

    # Sandbox-Endpoint für Tests
    APPLE_VERIFY_URL = "https://sandbox.itunes.apple.com/verifyReceipt"

    response = requests.post(
        APPLE_VERIFY_URL,
        json={
            "receipt-data": receipt_data,
            "password": "DEIN_APP_STORE_SHARED_SECRET",  # TODO: echtes Shared Secret
        },
    )

    result = response.json()

    if result.get("status") != 0:
        raise HTTPException(status_code=400, detail=f"Ungültiger Receipt: {result}")

    latest = result.get("latest_receipt_info") or []
    product_ids = {item.get("product_id") for item in latest if item.get("product_id")}

    if "Premium1000" in product_ids:
        plan = "premium"
    elif "Plus150" in product_ids:
        plan = "plus"
    else:
        plan = "free"

    user = get_or_create_user(db, user_id)
    user.plan = plan
    user.monthly_usage = 0
    user.usage_month = current_month_str()

    db.commit()
    db.refresh(user)

    limit = get_limit_for_plan(plan)

    return ReceiptResponse(
        plan=plan,
        monthlyUsage=user.monthly_usage,
        limit=limit,
    )


@app.post("/validateTransaction", response_model=TransactionResponse)
async def validate_transaction(payload: TransactionPayload, db: Session = Depends(get_db)):
    user_id = payload.userId
    product_id = payload.productId

    if product_id == "Premium1000":
        plan = "premium"
    elif product_id == "Plus150":
        plan = "plus"
    else:
        if payload.planHint in ("free", "plus", "premium"):
            plan = payload.planHint
        else:
            plan = "free"

    user = get_or_create_user(db, user_id)
    apply_month_reset(user)

    user.plan = plan

    db.commit()
    db.refresh(user)

    limit = get_limit_for_plan(plan)

    return TransactionResponse(
        plan=plan,
        monthlyUsage=user.monthly_usage,
        limit=limit,
    )


# 3) KI-Frage stellen
@app.post("/ask", response_model=AskResponse)
async def ask(payload: AskPayload, db: Session = Depends(get_db)):
    user_id = payload.userId
    message = payload.message

    user = get_or_create_user(db, user_id)
    apply_month_reset(user)

    plan = user.plan
    usage = user.monthly_usage
    limit = get_limit_for_plan(plan)
    model_name = get_model_for_plan(plan)

    if usage >= limit:
        raise HTTPException(status_code=403, detail="Limit erreicht")

    try:
        reply = call_openai(message, model_name=model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fehler bei KI-Anfrage: {e}")

    user.monthly_usage += 1
    db.commit()
    db.refresh(user)

    return AskResponse(
        reply=reply,
        usage=user.monthly_usage,
        limit=limit,
    )


# 4) Excluded load
@app.post("/user/excluded/load", response_model=ExcludedCasesResponse)
async def load_excluded_cases(payload: ExcludedCasesPayload, db: Session = Depends(get_db)):
    user = get_or_create_user(db, payload.userId)

    if user.excluded_cases:
        try:
            ids = json.loads(user.excluded_cases)
        except Exception:
            ids = []
    else:
        ids = []

    ids = [str(x) for x in ids]
    return ExcludedCasesResponse(excludedIds=ids)


# 5) Excluded save
@app.post("/user/excluded/save", response_model=ExcludedCasesResponse)
async def save_excluded_cases(payload: ExcludedCasesPayload, db: Session = Depends(get_db)):
    user = get_or_create_user(db, payload.userId)

    user.excluded_cases = json.dumps(payload.excludedIds)
    db.commit()
    db.refresh(user)

    return ExcludedCasesResponse(excludedIds=payload.excludedIds)


# 6) Notes load
@app.post("/user/notes/load", response_model=NotesResponse)
async def load_notes(payload: NotesLoadPayload, db: Session = Depends(get_db)):
    user = get_or_create_user(db, payload.userId)

    if user.notes:
        try:
            notes = json.loads(user.notes)
        except Exception:
            notes = {}
    else:
        notes = {}

    if not isinstance(notes, dict):
        notes = {}

    clean_notes: dict[str, str] = {}
    for k, v in notes.items():
        clean_notes[str(k)] = str(v)

    return NotesResponse(notes=clean_notes)


@app.post("/user/notes/save", response_model=NotesResponse)
async def save_note(payload: NoteUpdatePayload, db: Session = Depends(get_db)):
    user = get_or_create_user(db, payload.userId)

    # Falls der Falltitel im Payload steckt (empfohlen)
    case_title = getattr(payload, "caseTitle", None) or getattr(payload, "case_title", None) or ""

    prompt = (
        "Du spielst in diesem Chat den PATIENTEN in einem neurologischen Trainingsgespräch. "
        "Der Benutzer ist der Arzt.\n\n"
        "WICHTIG:\n"
        "- Sprich so, wie ein echter Patient sprechen würde: ganz normale Alltagssprache.\n"
        "- Keine medizinischen Fachbegriffe, keine Diagnosen, keine Erklärungen.\n"
        "- Antworte nur mit Informationen, die im Falltext stehen. Erfinde nichts dazu.\n"
        "- Wenn du etwas nicht weißt oder es nicht im Falltext steht, sag ehrlich: "
        "\"Das weiß ich nicht\" oder \"Dazu kann ich nichts sagen\".\n"
        "- Nenne keine Untersuchungs- oder Laborergebnisse (CT/MRT/EEG/Blut/LP usw.), "
        "außer sie stehen ausdrücklich im Falltext.\n\n"
        "SO SOLLST DU ANTWORTEN:\n"
        "- In der Ich-Form (\"Ich …\"), freundlich, natürlich.\n"
        "- Kurz und gut vorlesbar: meist 1–3 Sätze.\n"
        "- Gib nur die Infos, nach denen der Arzt gerade fragt. Keine langen Monologe.\n"
        "- Wenn mehrere Fragen kommen: nacheinander kurz beantworten.\n"
        "- Wenn der Arzt ein Wort benutzt, das du nicht verstehst: frag zurück, z.B. "
        "\"Was meinen Sie genau?\"\n\n"
        "UNTERSUCHUNG:\n"
        "- Wenn der Arzt dich bittet, etwas zu machen (z.B. Arme heben, auf einem Bein stehen): "
        "reagiere wie ein Patient (was du dabei merkst).\n"
        "- Sag objektive Befunde (z.B. \"Pupillen sind …\") nur, wenn sie im Falltext stehen.\n\n"
        f"Falltitel: {case_title}\n"
    )

    # Notes laden (falls vorhanden)
    notes = []
    if user.notes:
        try:
            notes = json.loads(user.notes)
        except Exception:
            notes = []
    else:
        notes = {}

    if not isinstance(notes, dict):
        notes = {}

    if payload.note.strip() == "":
        notes.pop(payload.caseId, None)
    else:
        notes[payload.caseId] = payload.note

    user.notes = json.dumps(notes)
    db.commit()
    db.refresh(user)

    clean_notes: dict[str, str] = {}
    for k, v in notes.items():
        clean_notes[str(k)] = str(v)

    return NotesResponse(notes=clean_notes)


# -------------------------------------------------------
# Duo REST: Patient (zählt)
# -------------------------------------------------------
@app.post("/duoChat", response_model=DuoChatResponse)
async def duo_chat(req: DuoChatRequest, db: Session = Depends(get_db)):
    user = get_or_create_user(db, req.userId)
    apply_month_reset(user)

    plan = user.plan
    limit = get_limit_for_plan(plan)
    if user.monthly_usage >= limit:
        raise HTTPException(status_code=403, detail="Limit erreicht")

    model_name = get_model_for_plan(plan)

    system_prompt = (
    "Du spielst in diesem Chat den PATIENTEN in einem neurologischen Trainingsgespräch (Anamnese + Untersuchung). "
    "Der Benutzer ist der Arzt. Du kennst den Falltext, der Arzt nicht.\n\n"
    "WICHTIG (bitte so sprechen, dass man es gut vorlesen kann):\n"
    "- Sprich wie ein echter Patient: normale Alltagssprache, keine Fachbegriffe.\n"
    "- Keine Diagnosen, keine Erklärungen, kein medizinisches Dozieren.\n"
    "- Nutze nur Informationen, die im Falltext stehen. Erfinde nichts dazu.\n"
    "- Wenn etwas nicht im Falltext steht oder du es nicht sicher weißt: sag ehrlich "
    "\"Das weiß ich nicht\" oder \"Dazu kann ich nichts sagen\".\n"
    "- Keine Labor/CT/MRT/EEG/LP-Ergebnisse nennen, außer sie stehen ausdrücklich im Falltext.\n\n"
    "SO SOLLST DU ANTWORTEN:\n"
    "- Immer in der Ich-Form (\"Ich ...\"), freundlich und menschlich.\n"
    "- Kurz, klar, gut vorlesbar: meistens 1–3 Sätze (maximal 4).\n"
    "- Gib nur die Infos, nach denen der Arzt gerade fragt. Keine ungefragten Info-Dumps.\n"
    "- Wenn mehrere Fragen kommen: nacheinander kurz beantworten.\n"
    "- Wenn der Arzt ein Wort benutzt, das du nicht verstehst: frag nach, z.B. "
    "\"Was meinen Sie genau?\".\n\n"
    "UNTERSUCHUNG:\n"
    "- Wenn der Arzt dich bittet, etwas zu machen (z.B. Arme heben, gehen, Finger-Nase): "
    "reagiere wie ein Patient und beschreibe, was du dabei merkst.\n"
    "- Nenne objektive Befunde (z.B. \"Pupillen sind ...\", \"Reflexe sind ...\") nur, wenn sie im Falltext stehen.\n\n"
    f"Falltitel: {req.caseTitle}\n\n"
    f"Fallbeschreibung (nur für dich, Ground Truth):\n{req.caseDescription}\n\n"
    "Regel für jede Antwort:\n"
    "- Antworte als Patient ausschließlich auf die letzte Nachricht des Arztes."
    )

    convo_lines = []
    for msg in req.messages:
        sprecher = "Arzt" if msg.role == "doctor" else "Patient"
        convo_lines.append(f"{sprecher}: {msg.content}")
    conversation_text = "\n".join(convo_lines) if convo_lines else "(noch kein Dialog)"

    prompt = (
        system_prompt
        + "\n\nBisheriger Dialog zwischen Arzt und Patient:\n"
        + conversation_text
        + "\n\nAntwort des Patienten (kurz und alltagssprachlich):"
    )

    try:
        # ✅ Statt Responses-API: nutze deine vorhandene call_openai()-Funktion
        reply_text = call_openai(prompt, model_name=model_name).strip()

        # ✅ Usage zählen (Duo zählt mit)
        user.monthly_usage += 1
        db.commit()
        db.refresh(user)

        return DuoChatResponse(reply=reply_text, usage=user.monthly_usage, limit=limit)

    except Exception as e:
        print("Fehler in /duoChat:", repr(e))
        return DuoChatResponse(
            reply="Entschuldigung, ich kann gerade nicht gut antworten – es gab einen technischen Fehler.",
            usage=None,
            limit=limit,
        )

@app.post("/duoChatStream")
async def duo_chat_stream(req: DuoChatRequest, db: Session = Depends(get_db)):
    user = get_or_create_user(db, req.userId)
    apply_month_reset(user)

    plan = user.plan
    limit = get_limit_for_plan(plan)

    if user.monthly_usage >= limit:
        raise HTTPException(status_code=403, detail="Limit erreicht")

    model_name = get_model_for_plan(plan)

    # ✅ DEIN PROMPT – 1:1
    system_prompt = (
        "Du spielst in diesem Chat den PATIENTEN in einem neurologischen Trainingsgespräch (Anamnese + Untersuchung). "
        "Der Benutzer ist der Arzt. Du kennst den Falltext, der Arzt nicht.\n\n"
        "WICHTIG (bitte so sprechen, dass man es gut vorlesen kann):\n"
        "- Sprich wie ein echter Patient: normale Alltagssprache, keine Fachbegriffe.\n"
        "- Keine Diagnosen, keine Erklärungen, kein medizinisches Dozieren.\n"
        "- Nutze nur Informationen, die im Falltext stehen. Erfinde nichts dazu.\n"
        "- Wenn etwas nicht im Falltext steht oder du es nicht sicher weißt: sag ehrlich "
        "\"Das weiß ich nicht\" oder \"Dazu kann ich nichts sagen\".\n"
        "- Keine Labor/CT/MRT/EEG/LP-Ergebnisse nennen, außer sie stehen ausdrücklich im Falltext.\n\n"
        "SO SOLLST DU ANTWORTEN:\n"
        "- Immer in der Ich-Form (\"Ich ...\"), freundlich und menschlich.\n"
        "- Kurz, klar, gut vorlesbar: meistens 1–3 Sätze (maximal 4).\n"
        "- Gib nur die Infos, nach denen der Arzt gerade fragt. Keine ungefragten Info-Dumps.\n"
        "- Wenn mehrere Fragen kommen: nacheinander kurz beantworten.\n"
        "- Wenn der Arzt ein Wort benutzt, das du nicht verstehst: frag nach, z.B. "
        "\"Was meinen Sie genau?\".\n\n"
        "UNTERSUCHUNG:\n"
        "- Wenn der Arzt dich bittet, etwas zu machen (z.B. Arme heben, gehen, Finger-Nase): "
        "reagiere wie ein Patient und beschreibe, was du dabei merkst.\n"
        "- Nenne objektive Befunde (z.B. \"Pupillen sind ...\", \"Reflexe sind ...\") nur, wenn sie im Falltext stehen.\n\n"
        f"Falltitel: {req.caseTitle}\n\n"
        f"Fallbeschreibung (nur für dich, Ground Truth):\n{req.caseDescription}\n\n"
        "Regel für jede Antwort:\n"
        "- Antworte als Patient ausschließlich auf die letzte Nachricht des Arztes."
    )

    convo_lines = []
    for msg in req.messages:
        sprecher = "Arzt" if msg.role == "doctor" else "Patient"
        convo_lines.append(f"{sprecher}: {msg.content}")
    conversation_text = "\n".join(convo_lines) if convo_lines else "(noch kein Dialog)"

    prompt = (
        system_prompt
        + "\n\nBisheriger Dialog zwischen Arzt und Patient:\n"
        + conversation_text
        + "\n\nAntwort des Patienten (kurz und alltagssprachlich):"
    )

    def sse(obj: dict) -> str:
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    async def gen():
        full = ""
        try:
            # ✅ Streaming deltas
            for delta in call_openai_stream(prompt, model_name=model_name):
                full += delta
                yield sse({"delta": delta})

            # ✅ Usage am Ende zählen
            user.monthly_usage += 1
            db.commit()
            db.refresh(user)

            yield sse({"done": True, "reply": full, "usage": user.monthly_usage, "limit": limit})

        except Exception as e:
            print("Fehler in /duoChatStream:", repr(e))
            yield sse({
                "done": True,
                "reply": "Entschuldigung, ich kann gerade nicht gut antworten – es gab einen technischen Fehler.",
                "usage": None,
                "limit": limit
            })

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no"
    }
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)


# -------------------------------------------------------
# Duo REST: Coach (zählt)
# -------------------------------------------------------
@app.post("/duoDoctorChat", response_model=DuoChatResponse)
async def duo_doctor_chat(req: DuoChatRequest, db: Session = Depends(get_db)):
    user = get_or_create_user(db, req.userId)
    apply_month_reset(user)

    plan = user.plan
    limit = get_limit_for_plan(plan)
    if user.monthly_usage >= limit:
        raise HTTPException(status_code=403, detail="Limit erreicht")

    model_name = get_model_for_plan(plan)

    system_prompt = (
    "Du bist ein erfahrener Neurologe und Lehrarzt.\n"
    "Du siehst den Dialog zwischen einem Patienten und einem Assistenzarzt.\n\n"
    "Deine Aufgabe: Gib dem Arzt sehr knappe, sofort umsetzbare Vorschläge.\n"
    "Sprich ihn direkt an (\"Fragen Sie ...\", \"Untersuchen Sie ...\").\n\n"
    "Regeln:\n"
    "- Schreibe auf Deutsch.\n"
    "- Kurz und klar: 1–3 Sätze.\n"
    "- Keine langen Erklärungen, keine Lehrbuchtexte.\n"
    "- Wenn du etwas vorschlägst, nenne möglichst genau, was er fragen/prüfen soll.\n\n"
    "Was du liefern sollst:\n"
    "- Nächste sinnvolle Frage an den Patienten.\n"
    "- Nächster sinnvoller Untersuchungsschritt (körperlich).\n"
    "- Optional: eine passende Zusatzdiagnostik (nur wenn wirklich naheliegend).\n\n"
    f"Falltitel: {req.caseTitle}\n\n"
    f"Fallbeschreibung (medizinischer Hintergrund):\n{req.caseDescription}\n"
    )

    lines = []
    for msg in req.messages:
        if msg.role == "patient":
            lines.append(f"Patient: {msg.content}")
        else:
            lines.append(f"Arzt: {msg.content}")
    conversation_text = "\n".join(lines) if lines else "(noch kein Dialog)"

    prompt = (
        system_prompt
        + "\n\nBisheriger Dialog:\n"
        + conversation_text
        + "\n\nDein naechster Vorschlag an den Arzt:"
    )

    try:
        reply_text = call_openai(prompt, model_name=model_name).strip()

        user.monthly_usage += 1
        db.commit()
        db.refresh(user)

        return DuoChatResponse(
            reply=reply_text,
            usage=user.monthly_usage,
            limit=limit
        )

    except Exception as e:
        print("Fehler in /duoDoctorChat:", repr(e))
        return DuoChatResponse(
            reply="Ich kann gerade keine sinnvollen Vorschlaege machen - es gab einen technischen Fehler.",
            usage=None,
            limit=limit
        )

# -------------------------------------------------------
# NEW: Duo Session REST (iPad erstellt, iPhone joint)
# -------------------------------------------------------

@app.post("/duo/session/create", response_model=DuoSessionCreateResponse)
async def duo_session_create(payload: DuoSessionCreatePayload, db: Session = Depends(get_db)):
    # doctor session owner
    session_id = str(uuid.uuid4())

    # unique join code
    join_code = None
    for _ in range(10):
        candidate = generate_join_code(6)
        exists = db.query(DuoSession).filter(DuoSession.join_code == candidate).first()
        if not exists:
            join_code = candidate
            break
    if not join_code:
        raise HTTPException(status_code=500, detail="Konnte keinen eindeutigen Code erzeugen")

    now = datetime.utcnow()
    expires_at = now + timedelta(minutes=60)

    s = DuoSession(
        session_id=session_id,
        join_code=join_code,
        doctor_user_id=None,              # ✅ Doctor kommt später per Join rein
        patient_user_id=payload.userId,   # ✅ Ersteller ist Patient
        case_title=payload.caseTitle,
        case_description=payload.caseDescription,
        created_at=now,
        expires_at=expires_at,

    )

    db.add(s)
    db.commit()
    db.refresh(s)

    return DuoSessionCreateResponse(
        sessionId=s.session_id,
        joinCode=s.join_code,
        expiresAt=s.expires_at.isoformat(),
    )


@app.post("/duo/session/join", response_model=DuoSessionJoinResponse)
async def duo_session_join(payload: DuoSessionJoinPayload, db: Session = Depends(get_db)):
    s = db.query(DuoSession).filter(DuoSession.join_code == payload.joinCode).first()
    if not s:
        raise HTTPException(status_code=404, detail="Session-Code nicht gefunden")

    if s.expires_at < datetime.utcnow():
        raise HTTPException(status_code=410, detail="Session ist abgelaufen")

    # ✅ nur 1 doctor pro session
    if s.doctor_user_id and s.doctor_user_id != payload.userId:
        raise HTTPException(status_code=409, detail="Session ist bereits belegt")

    s.doctor_user_id = payload.userId
    db.commit()
    db.refresh(s)

    return DuoSessionJoinResponse(sessionId=s.session_id, ok=True)


# -------------------------------------------------------
# NEW: WebSocket (iPhone -> iPad live)
# -------------------------------------------------------
# Client sendet JSON:
#   { "type": "patient_text", "text": "..." }  (patient)
# Server broadcastet an doctor:
#   { "type": "patient_text", "text": "...", "usage": X, "limit": Y }
# Und automatisch:
#   { "type": "coach_suggestion", "text": "...", "usage": X, "limit": Y }
#
# connect:
#   wss://HOST/ws/duo/{sessionId}?role=doctor&userId=...
#   wss://HOST/ws/duo/{sessionId}?role=patient&userId=...
# -------------------------------------------------------

from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime
import json

@app.websocket("/ws/duo/{session_id}")
async def ws_duo(session_id: str, websocket: WebSocket):
    # ✅ accept GENAU EINMAL – hier!
    await websocket.accept()

    role = websocket.query_params.get("role")  # "doctor" | "patient"
    user_id = websocket.query_params.get("userId")

    if role not in ("doctor", "patient") or not user_id:
        await websocket.close(code=1008)
        return

    db = SessionLocal()
    try:
        s = db.query(DuoSession).filter(DuoSession.session_id == session_id).first()
        if not s or s.expires_at < datetime.utcnow():
            await websocket.close(code=1008)
            return

        # -------------------------------------------------
        # Rollenbesetzung / Berechtigung
        # -------------------------------------------------
        if role == "patient":
            if s.patient_user_id and user_id != s.patient_user_id:
                await websocket.close(code=1008)
                return
            if not s.patient_user_id:
                s.patient_user_id = user_id
                db.commit()

        if role == "doctor":
            if s.doctor_user_id and user_id != s.doctor_user_id:
                await websocket.close(code=1008)
                return
            if not s.doctor_user_id:
                s.doctor_user_id = user_id
                db.commit()

        await ws_manager.connect(session_id, role, websocket)
        await websocket.send_text(json.dumps({
            "type": "connected",
            "sessionId": session_id,
            "role": role
        }))

        # -------------------------------------------------
        # MAIN RECEIVE LOOP
        # -------------------------------------------------
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except Exception:
                await websocket.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))
                continue

            msg_type = data.get("type")

            # ✅ Doctor kann Case setzen
            if msg_type == "init_case" and role == "doctor":
                s.case_title = data.get("caseTitle") or s.case_title
                s.case_description = data.get("caseDescription") or s.case_description
                db.commit()
                await ws_manager.broadcast(session_id, "doctor", {"type": "status", "message": "Case gespeichert"})
                continue

            # ✅ Doctor -> Patient
            if msg_type == "doctor_text" and role == "doctor":
                text = (data.get("text") or "").strip()
                if not text:
                    continue

                ws_manager.history.setdefault(session_id, [])
                ws_manager.history[session_id].append({"role": "doctor", "content": text})

                await ws_manager.broadcast(session_id, "patient", {"type": "doctor_text", "text": text})
                continue

            # ✅ Patient -> Doctor
            if msg_type == "patient_text" and role == "patient":
                text = (data.get("text") or "").strip()
                if not text:
                    continue

                ws_manager.history.setdefault(session_id, [])
                ws_manager.history[session_id].append({"role": "patient", "content": text})

                # Doctor bekommt live Patiententext
                await ws_manager.broadcast(session_id, "doctor", {"type": "patient_text", "text": text})

                # OPTIONAL: Auto-Coach (nur wenn case vorhanden)
                if s.doctor_user_id and s.case_title and s.case_description:
                    doctor = get_or_create_user(db, s.doctor_user_id)
                    apply_month_reset(doctor)

                    plan = doctor.plan
                    limit = get_limit_for_plan(plan)

                    if doctor.monthly_usage >= limit:
                        await ws_manager.broadcast(session_id, "doctor", {
                            "type": "error",
                            "message": "Limit erreicht",
                            "usage": doctor.monthly_usage,
                            "limit": limit
                        })
                        continue

                    # patient_text usage
                    doctor.monthly_usage += 1
                    db.commit()
                    db.refresh(doctor)

                    model_name = get_model_for_plan(plan)
                    coach_prompt = build_coach_prompt(s.case_title, s.case_description, ws_manager.history[session_id])

                    try:
                        resp = client.responses.create(model=model_name, input=coach_prompt)
                        reply_text = extract_text_from_responses_api(resp).strip()
                    except Exception as e:
                        print("Coach WS error:", repr(e))
                        reply_text = "Technischer Fehler beim Coach."

                    # coach_suggestion usage
                    doctor.monthly_usage += 1
                    db.commit()
                    db.refresh(doctor)

                    await ws_manager.broadcast(session_id, "doctor", {
                        "type": "coach_suggestion",
                        "text": reply_text,
                        "usage": doctor.monthly_usage,
                        "limit": limit
                    })
                continue

            # Unknown
            await websocket.send_text(json.dumps({"type": "error", "message": "Unknown event type"}))

    except WebSocketDisconnect:
        pass
    finally:
        try:
            ws_manager.disconnect(session_id, role, websocket)
        except Exception:
            pass
        db.close()


