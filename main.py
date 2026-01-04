import os
import json
import uuid
import secrets
import string
import time
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

    doctor_user_id = Column(String, index=True, nullable=True)  # ✅ FIX: nullable=True (Doctor kommt später)
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
        "model": os.getenv("OPENAI_MODEL_FREE", "gpt-4.1-mini"),
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
    Nicht-streaming Antwort. Wird von /ask, /duoChat und /duoDoctorChat genutzt.
    """
    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": message}],
        temperature=1,
    )
    return completion.choices[0].message.content or ""


def call_openai_stream(message: str, model_name: str):
    stream = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": message}],
        temperature=1,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            yield delta


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

# ✅ FIX: Apple Public Key Caching (1 Stunde)
_apple_keys_cache = None
_apple_keys_cache_time = None


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
    chatHistory: Optional[List[Dict[str, str]]] = None  # Optional: Chat-Verlauf für Kontext
    caseTitle: Optional[str] = None  # Optional: Fall-Titel für Kontext
    caseDescription: Optional[str] = None  # Optional: Fall-Beschreibung für Kontext
    # Optional: Zusätzliche Fall-Informationen für Arzt-KI
    questions: Optional[List[str]] = None
    answers: Optional[List[str]] = None
    extraInfo: Optional[str] = None


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
    # Optional: Nur für Arzt-KI (nicht für Patienten-KI)
    questions: Optional[List[str]] = None
    answers: Optional[List[str]] = None
    extraInfo: Optional[str] = None


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
    caseTitle: str  # Fall-Titel zur Überprüfung


class DuoSessionJoinResponse(BaseModel):
    sessionId: str
    ok: bool

class QuickAnswersCreatePayload(BaseModel):
    userId: str
    sessionId: str


class QuickAnswersResponse(BaseModel):
    sessionId: str
    items: List[Dict[str, str]]
    roleDescription: Optional[str] = None  # Rollenbeschreibung in Ich-Form
    usage: Optional[int] = None
    limit: Optional[int] = None


class DiagnosisValidationPayload(BaseModel):
    userId: str
    caseTitle: str
    caseDescription: str
    diagnosis: str
    history: List[DuoMessage]  # Bisheriger Dialog
    # Optional: Zusätzliche Fall-Informationen für Arzt-KI
    questions: Optional[List[str]] = None
    answers: Optional[List[str]] = None
    extraInfo: Optional[str] = None


class DiagnosisValidationResponse(BaseModel):
    isCorrect: bool
    feedback: str
    correctDiagnosis: Optional[str] = None
    explanation: Optional[str] = None
    usage: Optional[int] = None
    limit: Optional[int] = None


class TherapyValidationPayload(BaseModel):
    userId: str
    caseTitle: str
    caseDescription: str
    diagnosis: str  # Die (korrekte) Diagnose
    therapy: str
    history: List[DuoMessage]  # Bisheriger Dialog
    # Optional: Zusätzliche Fall-Informationen für Arzt-KI
    questions: Optional[List[str]] = None
    answers: Optional[List[str]] = None
    extraInfo: Optional[str] = None


class TherapyValidationResponse(BaseModel):
    isCorrect: bool
    feedback: str
    correctTherapy: Optional[str] = None
    explanation: Optional[str] = None
    usage: Optional[int] = None
    limit: Optional[int] = None


# -------------------------------------------------------
# Helper für Apple Public Key (mit Caching)
# -------------------------------------------------------

def get_apple_public_key(kid: str):
    """✅ FIX: Apple Public Keys werden gecacht (1 Stunde)"""
    global _apple_keys_cache, _apple_keys_cache_time
    now = time.time()
    
    # Cache für 1 Stunde
    if _apple_keys_cache is None or (now - _apple_keys_cache_time) > 3600:
        _apple_keys_cache = requests.get(APPLE_KEYS_URL).json()["keys"]
        _apple_keys_cache_time = now
    
    for key in _apple_keys_cache:
        if key["kid"] == kid:
            return key
    raise Exception("Apple Public Key nicht gefunden")


# -------------------------------------------------------
# WebSocket Manager (in-memory connections + in-memory history)
# -------------------------------------------------------

class WSManager:
    def __init__(self):
        self.quickanswers: Dict[str, List[Dict[str, str]]] = {}
        self.role_descriptions: Dict[str, str] = {}  # ✅ NEU: Cache für Rollenbeschreibungen
        # session_id -> {"doctor": set(ws), "patient": set(ws)}
        self.sessions: Dict[str, Dict[str, set[WebSocket]]] = {}
        # session_id -> history list[{"role": "...", "content": "..."}]
        self.history: Dict[str, List[Dict[str, str]]] = {}
        # ✅ NEU: Rate-Limiting für Patient-Nachrichten
        self.patient_message_count: Dict[str, int] = {}  # session_id -> Anzahl Nachrichten hintereinander
        self.patient_ai_processing: Dict[str, bool] = {}  # session_id -> ob KI-Antwort läuft

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

# =========================
# ✅ Quick Answers (Session Cache + KI Generator)
# =========================

TYPICAL_QUESTIONS: list[str] = [
    "Was führt sie zu uns?",
    "Wie haben die Beschwerden begonnen (plötzlich oder schleichend),und seit wann bestehen die Beschwerden?",
    "Wo genau ist das Problem / welche Körperregion ist betroffen?",
    "Wie würden Sie das Gefühl beschreiben (Schmerz, Taubheit, Schwäche, Schwindel, Sehstörung)?",
    "Wie stark ist es (Skala 0–10)?",
    "Gibt es Auslöser oder Besserung/Verschlechterung?",
    "Gab es ähnliche Episoden früher?",
    "Gibt es Begleitsymptome (Übelkeit, Fieber, Bewusstseinsstörung, Gewichtsverlust)?",
    "Welche Vorerkrankungen haben Sie, nehmen sie Medikamente?",
    "Ist in der Familie eine Erkrankung bekannt?",
    "Allergien?",
    "Rauchen/Alkohol/Drogen/Sport?",
]

def build_role_description_prompt(case_title: str, case_description: str) -> str:
    """Generiert eine Rollenbeschreibung in Ich-Form für den Patienten."""
    return f"""
Du bist der PATIENT in einem neurologischen Trainingsfall.
Deine Aufgabe ist es, eine kurze Rollenbeschreibung in Ich-Form zu erstellen, die dem Patienten hilft, sich in seine Rolle hineinzuversetzen.

REGELN:
- Schreibe in Ich-Form (z.B. "Ich bin ein 45-jähriger Mann...")
- Nutze AUSSCHLIESSLICH Informationen aus dem Falltext
- Erfinde nichts dazu
- Kurz und prägnant: 3-5 Sätze
- Alltagssprache, keine Fachbegriffe
- Fokus auf die wichtigsten Symptome und Umstände

FALL:
Titel: {case_title}

Fallbeschreibung (Ground Truth):
{case_description}

AUFGABE:
Erstelle eine kurze Rollenbeschreibung in Ich-Form, die dem Patienten hilft, sich in seine Rolle hineinzuversetzen.

Antworte NUR mit dem Text der Rollenbeschreibung (ohne zusätzliche Erklärungen oder Formatierung).
""".strip()


def generate_role_description(case_title: str, case_description: str, model_name: str) -> str:
    """Generiert eine Rollenbeschreibung für den Patienten."""
    prompt = build_role_description_prompt(case_title, case_description)
    
    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    
    return (completion.choices[0].message.content or "").strip()


def build_quickanswers_prompt(case_title: str, case_description: str, questions: List[str]) -> str:
    q_lines = "\n".join([f"- {q}" for q in questions])
    return f"""
Du bist der PATIENT in einem neurologischen Trainingsfall.
Der Arzt stellt typische Anamnesefragen. Du antwortest wie ein echter Patient, in Alltagssprache.

REGELN:
- Nutze AUSSCHLIESSLICH Informationen aus dem Falltext.
- Erfinde nichts dazu.
- Wenn es nicht im Falltext steht: "Das weiß ich nicht" / "Dazu kann ich nichts sagen".
- Keine Diagnosen, keine Fachbegriffe.
- Kurz: 1–2 Sätze pro Antwort.

FALL:
Titel: {case_title}

Fallbeschreibung (Ground Truth):
{case_description}

AUFGABE:
Beantworte die folgenden Fragen:

{q_lines}

OUTPUT:
Gib NUR gültiges JSON zurück (ohne Text drumherum):
[
  {{"q":"...","a":"..."}},
  ...
]
""".strip()

def generate_quickanswers(case_title: str, case_description: str, model_name: str) -> List[Dict[str, str]]:
    prompt = build_quickanswers_prompt(case_title, case_description, TYPICAL_QUESTIONS)

    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        # kein temperature -> kompatibler
    )

    text = (completion.choices[0].message.content or "").strip()

    # 1) Direkt JSON versuchen
    data = None
    try:
        data = json.loads(text)
    except Exception:
        # 2) Best-effort JSON extrahieren, falls Modell Text davor/danach liefert
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(text[start:end + 1])
            except Exception:
                data = None

    cleaned: List[Dict[str, str]] = []
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            q = str(item.get("q", "")).strip()
            a = str(item.get("a", "")).strip()
            if q and a:
                cleaned.append({"q": q, "a": a})

    # Fallback: wenn die KI Mist liefert
    if not cleaned:
        cleaned = [{"q": q, "a": "Das weiß ich nicht."} for q in TYPICAL_QUESTIONS]

    return cleaned


def generate_join_code(length: int = 6) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def build_coach_prompt(case_title: str, case_description: str, history: List[Dict[str, str]]) -> str:
    system_prompt = (
        "Du bist ein erfahrener Neurologe und Lehrarzt.\n"
        "Du siehst den Dialog zwischen einem Patienten und einem Assistenzarzt.\n\n"
        "WICHTIG: Formuliere NUR EINE konkrete Frage, die der Arzt direkt an den Patienten stellen kann.\n"
        "Nicht mehrere Vorschläge auf einmal. Schritt für Schritt zur Diagnose.\n\n"
        "Regeln:\n"
        "- Wenn noch wenig Anamnese: formuliere die nächste wichtige Anamnesefrage in der Ich-Form.\n"
        "- Wenn Anamnese ausreichend: formuliere eine Untersuchungsanweisung in der Ich-Form.\n"
        "- Wenn Untersuchung abgeschlossen: schlage die nächste sinnvolle Zusatzdiagnostik vor.\n"
        "- Baue auf dem bisherigen Dialog auf. Wiederhole nicht bereits gestellte Fragen.\n"
        "- Führe logisch zur Diagnose hin.\n\n"
        "Format:\n"
        "- Formuliere die Frage direkt, die der Arzt stellen soll (nicht \"Fragen Sie...\", sondern direkt die Frage).\n"
        "- Beispiele: \"Wo genau haben Sie Schmerzen?\" oder \"Können Sie bitte die Pupillenreaktion auf Licht prüfen?\"\n"
        "- Sehr kurz: 1-2 Sätze.\n"
        "- Auf Deutsch.\n\n"
        f"Falltitel: {case_title}\n\n"
        f"Fallbeschreibung (medizinischer Hintergrund - nur für dich):\n{case_description}\n"
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
        + "\n\nFormuliere NUR EINE konkrete Frage, die der Arzt direkt an den Patienten stellen kann (1-2 Sätze, direkt formuliert, nicht \"Fragen Sie...\"):"
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

    # ✅ FIX: App Store Shared Secret aus Environment-Variable
    # Nur erforderlich, wenn dieser Endpoint verwendet wird (für In-App-Käufe)
    app_store_secret = os.getenv("APP_STORE_SHARED_SECRET", "")
    if not app_store_secret:
        raise HTTPException(
            status_code=500, 
            detail="APP_STORE_SHARED_SECRET nicht gesetzt. Für lokale Tests ohne Receipt-Validation nicht erforderlich."
        )

    # Sandbox-Endpoint für Tests
    APPLE_VERIFY_URL = "https://sandbox.itunes.apple.com/verifyReceipt"

    response = requests.post(
        APPLE_VERIFY_URL,
        json={
            "receipt-data": receipt_data,
            "password": app_store_secret,
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


# 3) KI-Frage stellen (mit optionalem Chat-Verlauf)
@app.post("/ask", response_model=AskResponse)
async def ask(payload: AskPayload, db: Session = Depends(get_db)):
    user_id = payload.userId
    message = payload.message
    chat_history = payload.chatHistory or []
    case_title = payload.caseTitle
    case_description = payload.caseDescription

    user = get_or_create_user(db, user_id)
    apply_month_reset(user)

    plan = user.plan
    usage = user.monthly_usage
    limit = get_limit_for_plan(plan)
    model_name = get_model_for_plan(plan)

    if usage >= limit:
        raise HTTPException(status_code=403, detail="Limit erreicht")

    try:
        # Baue Kontext-Prompt mit Chat-Verlauf
        messages = []
        
        # System-Prompt mit Fall-Informationen (wenn vorhanden)
        system_prompt = "Du bist ein medizinischer Assistent, der Ärzte bei der Diagnose und Therapie unterstützt."
        if case_title and case_description:
            context_parts = [f"Titel: {case_title}", f"Beschreibung: {case_description}"]
            
            # Füge Fragen hinzu (wenn vorhanden)
            if payload.questions:
                questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(payload.questions)])
                context_parts.append(f"Fragen zum Fall:\n{questions_text}")
            
            # Füge Antworten hinzu (wenn vorhanden)
            if payload.answers:
                answers_text = "\n".join([f"{i+1}. {a}" for i, a in enumerate(payload.answers)])
                context_parts.append(f"Musterantworten zum Fall:\n{answers_text}")
            
            # Füge Zusatzinfo hinzu (wenn vorhanden)
            if payload.extraInfo:
                context_parts.append(f"Zusatzinformationen zum Fall:\n{payload.extraInfo}")
            
            system_prompt += f"\n\nAktueller Fall:\n" + "\n".join(context_parts)
        messages.append({"role": "system", "content": system_prompt})
        
        # Chat-Verlauf hinzufügen (wenn vorhanden)
        if chat_history:
            for msg in chat_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                # Konvertiere "doctor"/"patient" zu "user" für OpenAI
                if role in ["doctor", "patient"]:
                    role = "user"
                messages.append({"role": role, "content": content})
        
        # Aktuelle Frage hinzufügen
        messages.append({"role": "user", "content": message})
        
        # OpenAI aufrufen mit Chat-Verlauf
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=1,
        )
        reply = completion.choices[0].message.content or ""
        
        # ✅ FIX: Usage erst NACH erfolgreichem OpenAI-Call
        user.monthly_usage += 1
        db.commit()
        db.refresh(user)
    except Exception as e:
        db.rollback()  # ✅ FIX: Rollback bei Fehler
        raise HTTPException(status_code=500, detail=f"Fehler bei KI-Anfrage: {e}")

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

    # ✅ FIX: Unbenutzten Prompt entfernt, direkt Notes speichern
    # Notes laden (falls vorhanden)
    notes = {}  # ✅ FIX: Direkt als Dict initialisieren
    if user.notes:
        try:
            notes = json.loads(user.notes)
        except Exception:
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
        reply_text = call_openai(prompt, model_name=model_name).strip()
        # ✅ FIX: Usage erst NACH erfolgreichem OpenAI-Call
        user.monthly_usage += 1
        db.commit()
        db.refresh(user)
    except Exception as e:
        db.rollback()  # ✅ FIX: Rollback bei Fehler
        print("Fehler in /duoChat:", repr(e))
        return DuoChatResponse(
            reply="Entschuldigung, ich kann gerade nicht gut antworten – es gab einen technischen Fehler.",
            usage=None,
            limit=limit,
        )

    return DuoChatResponse(reply=reply_text, usage=user.monthly_usage, limit=limit)

@app.post("/duoChatStream")
async def duo_chat_stream(req: DuoChatRequest, db: Session = Depends(get_db)):
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

    # ✅ Usage VOR dem Stream zählen (DB bleibt sauber)
    user.monthly_usage += 1
    db.commit()
    db.refresh(user)
    usage_now = user.monthly_usage

    def sse(obj: dict) -> str:
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    def gen():
        full = ""
        try:
            for delta in call_openai_stream(prompt, model_name=model_name):
                full += delta
                yield sse({"delta": delta})

            yield sse({"done": True, "reply": full, "usage": usage_now, "limit": limit})

        except Exception as e:
            print("Fehler in /duoChatStream:", repr(e))
            yield sse({
                "done": True,
                "reply": "Entschuldigung, ich kann gerade nicht gut antworten – es gab einen technischen Fehler.",
                "usage": usage_now,
                "limit": limit,
                "msg": repr(e)
            })

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
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

    # Baue erweiterten Kontext für Arzt-KI
    context_parts = [
        f"Falltitel: {req.caseTitle}",
        f"Fallbeschreibung (medizinischer Hintergrund - nur für dich):\n{req.caseDescription}"
    ]
    
    # Füge Fragen hinzu (wenn vorhanden)
    if req.questions:
        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(req.questions)])
        context_parts.append(f"\nFragen zum Fall (nur für dich):\n{questions_text}")
    
    # Füge Antworten hinzu (wenn vorhanden)
    if req.answers:
        answers_text = "\n".join([f"{i+1}. {a}" for i, a in enumerate(req.answers)])
        context_parts.append(f"\nMusterantworten zum Fall (nur für dich):\n{answers_text}")
    
    # Füge Zusatzinfo hinzu (wenn vorhanden)
    if req.extraInfo:
        context_parts.append(f"\nZusatzinformationen zum Fall (nur für dich):\n{req.extraInfo}")
    
    context_text = "\n".join(context_parts)
    
    system_prompt = (
    "Du bist ein erfahrener Neurologe und Lehrarzt.\n"
    "Du siehst den Dialog zwischen einem Patienten und einem Assistenzarzt.\n\n"
    "WICHTIG: Formuliere NUR EINE konkrete Frage, die der Arzt direkt an den Patienten stellen kann.\n"
    "Nicht mehrere Vorschläge auf einmal. Schritt für Schritt zur Diagnose.\n\n"
    "Regeln:\n"
    "- Wenn noch wenig Anamnese: formuliere die nächste wichtige Anamnesefrage in der Ich-Form.\n"
    "- Wenn Anamnese ausreichend: formuliere eine Untersuchungsanweisung in der Ich-Form.\n"
    "- Wenn Untersuchung abgeschlossen: schlage die nächste sinnvolle Zusatzdiagnostik vor.\n"
    "- Baue auf dem bisherigen Dialog auf. Wiederhole nicht bereits gestellte Fragen.\n"
    "- Führe logisch zur Diagnose hin.\n\n"
    "Format:\n"
    "- Formuliere die Frage direkt, die der Arzt stellen soll (nicht \"Fragen Sie...\", sondern direkt die Frage).\n"
    "- Beispiele: \"Wo genau haben Sie Schmerzen?\" oder \"Können Sie bitte die Pupillenreaktion auf Licht prüfen?\"\n"
    "- Sehr kurz: 1-2 Sätze.\n"
    "- Auf Deutsch.\n\n"
    f"{context_text}\n"
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
        + "\n\nFormuliere NUR EINE konkrete Frage, die der Arzt direkt an den Patienten stellen kann (1-2 Sätze, direkt formuliert, nicht \"Fragen Sie...\"):"
    )

    try:
        reply_text = call_openai(prompt, model_name=model_name).strip()
        # ✅ FIX: Usage erst NACH erfolgreichem OpenAI-Call
        user.monthly_usage += 1
        db.commit()
        db.refresh(user)
    except Exception as e:
        db.rollback()  # ✅ FIX: Rollback bei Fehler
        print("Fehler in /duoDoctorChat:", repr(e))
        return DuoChatResponse(
            reply="Ich kann gerade keine sinnvollen Vorschlaege machen - es gab einen technischen Fehler.",
            usage=None,
            limit=limit
        )

    return DuoChatResponse(
        reply=reply_text,
        usage=user.monthly_usage,
        limit=limit
    )

@app.post("/duo/quickanswers/create", response_model=QuickAnswersResponse)
async def duo_quickanswers_create(payload: QuickAnswersCreatePayload, db: Session = Depends(get_db)):
    s = db.query(DuoSession).filter(DuoSession.session_id == payload.sessionId).first()
    if not s:
        raise HTTPException(status_code=404, detail="Session nicht gefunden")
    if s.expires_at < datetime.utcnow():
        raise HTTPException(status_code=410, detail="Session abgelaufen")

    if not s.case_title or not s.case_description:
        raise HTTPException(status_code=400, detail="Fall ist in der Session noch nicht gesetzt")

    # ✅ Cache: wenn schon da, direkt zurück (ohne neues Usage)
    cached = ws_manager.quickanswers.get(payload.sessionId)
    cached_role_desc = ws_manager.role_descriptions.get(payload.sessionId)  # ✅ NEU: Role Description Cache
    if cached and len(cached) > 0:
        user = get_or_create_user(db, payload.userId)
        apply_month_reset(user)
        limit = get_limit_for_plan(user.plan)
        return QuickAnswersResponse(
            sessionId=payload.sessionId, 
            items=cached, 
            roleDescription=cached_role_desc,  # ✅ NEU: Role Description zurückgeben
            usage=user.monthly_usage, 
            limit=limit
        )

    user = get_or_create_user(db, payload.userId)
    apply_month_reset(user)

    plan = user.plan
    limit = get_limit_for_plan(plan)
    if user.monthly_usage >= limit:
        raise HTTPException(status_code=403, detail="Limit erreicht")

    model_name = get_model_for_plan(plan)

    try:
        # ✅ NEU: Generiere sowohl Rollenbeschreibung als auch Quick-Answers
        role_desc = generate_role_description(s.case_title, s.case_description, model_name=model_name)
        items = generate_quickanswers(s.case_title, s.case_description, model_name=model_name)
        
        # ✅ FIX: Usage erst NACH erfolgreichem OpenAI-Call (beide Calls zählen als 1)
        user.monthly_usage += 1
        db.commit()
        db.refresh(user)
    except Exception as e:
        db.rollback()  # ✅ FIX: Rollback bei Fehler
        raise HTTPException(status_code=500, detail=f"Fehler bei QuickAnswers-Generierung: {e}")

    # ✅ Cache speichern
    ws_manager.quickanswers[payload.sessionId] = items
    ws_manager.role_descriptions[payload.sessionId] = role_desc  # ✅ NEU: Role Description Cache

    return QuickAnswersResponse(
        sessionId=payload.sessionId, 
        items=items, 
        roleDescription=role_desc,  # ✅ NEU: Role Description zurückgeben
        usage=user.monthly_usage, 
        limit=limit
    )


# -------------------------------------------------------
# NEW: Diagnose-Validierung
# -------------------------------------------------------
@app.post("/duo/validate-diagnosis", response_model=DiagnosisValidationResponse)
async def validate_diagnosis(payload: DiagnosisValidationPayload, db: Session = Depends(get_db)):
    user = get_or_create_user(db, payload.userId)
    apply_month_reset(user)

    plan = user.plan
    limit = get_limit_for_plan(plan)
    if user.monthly_usage >= limit:
        raise HTTPException(status_code=403, detail="Limit erreicht")

    model_name = get_model_for_plan(plan)

    # Dialog-History formatieren
    convo_lines = []
    for msg in payload.history:
        sprecher = "Arzt" if msg.role == "doctor" else "Patient"
        convo_lines.append(f"{sprecher}: {msg.content}")
    conversation_text = "\n".join(convo_lines) if convo_lines else "(noch kein Dialog)"

    # Baue erweiterten Kontext für Arzt-KI
    context_parts = [
        f"Falltitel: {payload.caseTitle}",
        f"Fallbeschreibung (Ground Truth - nur für dich):\n{payload.caseDescription}"
    ]
    
    # Füge Fragen hinzu (wenn vorhanden)
    if payload.questions:
        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(payload.questions)])
        context_parts.append(f"\nFragen zum Fall (nur für dich):\n{questions_text}")
    
    # Füge Antworten hinzu (wenn vorhanden)
    if payload.answers:
        answers_text = "\n".join([f"{i+1}. {a}" for i, a in enumerate(payload.answers)])
        context_parts.append(f"\nMusterantworten zum Fall (nur für dich):\n{answers_text}")
    
    # Füge Zusatzinfo hinzu (wenn vorhanden)
    if payload.extraInfo:
        context_parts.append(f"\nZusatzinformationen zum Fall (nur für dich):\n{payload.extraInfo}")
    
    context_text = "\n".join(context_parts)
    
    prompt = f"""Du bist ein erfahrener Neurologe und Lehrarzt.
Du prüfst die Diagnose eines Assistenzarztes für einen Fall.

{context_text}

Bisheriger Dialog zwischen Arzt und Patient:
{conversation_text}

Vom Arzt gestellte Diagnose: {payload.diagnosis}

AUFGABE:
1. Prüfe ob die Diagnose korrekt ist (vergliche mit der Fallbeschreibung, Fragen, Antworten und Zusatzinformationen).
2. Wenn falsch: Gib die korrekte Diagnose an und erkläre kurz, warum die gestellte Diagnose falsch ist.
3. Wenn richtig: Bestätige dies und gib eine kurze Erklärung.

Antworte NUR mit gültigem JSON (ohne Text drumherum):
{{
  "isCorrect": true/false,
  "feedback": "Kurze Rückmeldung (1-2 Sätze)",
  "correctDiagnosis": "Korrekte Diagnose (nur wenn falsch)",
  "explanation": "Kurze Erklärung warum richtig/falsch (2-3 Sätze)"
}}
"""

    try:
        reply_text = call_openai(prompt, model_name=model_name).strip()
        
        # JSON aus Antwort extrahieren
        data = None
        try:
            data = json.loads(reply_text)
        except Exception:
            # Best-effort JSON extrahieren
            start = reply_text.find("{")
            end = reply_text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(reply_text[start:end + 1])
                except Exception:
                    data = None

        if not data or not isinstance(data, dict):
            # Fallback wenn JSON-Parsing fehlschlägt
            return DiagnosisValidationResponse(
                isCorrect=False,
                feedback="Technischer Fehler bei der Validierung. Bitte versuche es erneut.",
                usage=user.monthly_usage,
                limit=limit
            )

        is_correct = bool(data.get("isCorrect", False))
        feedback = str(data.get("feedback", "Validierung abgeschlossen."))
        correct_diagnosis = data.get("correctDiagnosis")
        explanation = data.get("explanation")

        # ✅ FIX: Usage erst NACH erfolgreichem OpenAI-Call
        user.monthly_usage += 1
        db.commit()
        db.refresh(user)

        return DiagnosisValidationResponse(
            isCorrect=is_correct,
            feedback=feedback,
            correctDiagnosis=correct_diagnosis,
            explanation=explanation,
            usage=user.monthly_usage,
            limit=limit
        )

    except Exception as e:
        db.rollback()  # ✅ FIX: Rollback bei Fehler
        print("Fehler in /duo/validate-diagnosis:", repr(e))
        return DiagnosisValidationResponse(
            isCorrect=False,
            feedback=f"Technischer Fehler: {str(e)}",
            usage=user.monthly_usage,
            limit=limit
        )


# -------------------------------------------------------
# NEW: Therapie-Validierung
# -------------------------------------------------------
@app.post("/duo/validate-therapy", response_model=TherapyValidationResponse)
async def validate_therapy(payload: TherapyValidationPayload, db: Session = Depends(get_db)):
    user = get_or_create_user(db, payload.userId)
    apply_month_reset(user)

    plan = user.plan
    limit = get_limit_for_plan(plan)
    if user.monthly_usage >= limit:
        raise HTTPException(status_code=403, detail="Limit erreicht")

    model_name = get_model_for_plan(plan)

    # Dialog-History formatieren
    convo_lines = []
    for msg in payload.history:
        sprecher = "Arzt" if msg.role == "doctor" else "Patient"
        convo_lines.append(f"{sprecher}: {msg.content}")
    conversation_text = "\n".join(convo_lines) if convo_lines else "(noch kein Dialog)"

    # Baue erweiterten Kontext für Arzt-KI
    context_parts = [
        f"Falltitel: {payload.caseTitle}",
        f"Fallbeschreibung (Ground Truth - nur für dich):\n{payload.caseDescription}"
    ]
    
    # Füge Fragen hinzu (wenn vorhanden)
    if payload.questions:
        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(payload.questions)])
        context_parts.append(f"\nFragen zum Fall (nur für dich):\n{questions_text}")
    
    # Füge Antworten hinzu (wenn vorhanden)
    if payload.answers:
        answers_text = "\n".join([f"{i+1}. {a}" for i, a in enumerate(payload.answers)])
        context_parts.append(f"\nMusterantworten zum Fall (nur für dich):\n{answers_text}")
    
    # Füge Zusatzinfo hinzu (wenn vorhanden)
    if payload.extraInfo:
        context_parts.append(f"\nZusatzinformationen zum Fall (nur für dich):\n{payload.extraInfo}")
    
    context_text = "\n".join(context_parts)
    
    prompt = f"""Du bist ein erfahrener Neurologe und Lehrarzt.
Du prüfst die Therapievorschläge eines Assistenzarztes für einen Fall.

{context_text}

Bisheriger Dialog zwischen Arzt und Patient:
{conversation_text}

Diagnose: {payload.diagnosis}

Vom Arzt vorgeschlagene Therapie: {payload.therapy}

AUFGABE:
1. Prüfe ob die Therapie für die gegebene Diagnose angemessen ist (berücksichtige Fallbeschreibung, Fragen, Antworten und Zusatzinformationen).
2. Wenn unpassend: Gib eine bessere Therapie an und erkläre kurz, warum die vorgeschlagene Therapie nicht optimal ist.
3. Wenn passend: Bestätige dies und gib eine kurze Erklärung.

Antworte NUR mit gültigem JSON (ohne Text drumherum):
{{
  "isCorrect": true/false,
  "feedback": "Kurze Rückmeldung (1-2 Sätze)",
  "correctTherapy": "Bessere Therapie (nur wenn unpassend)",
  "explanation": "Kurze Erklärung warum passend/unpassend (2-3 Sätze)"
}}
"""

    try:
        reply_text = call_openai(prompt, model_name=model_name).strip()
        
        # JSON aus Antwort extrahieren
        data = None
        try:
            data = json.loads(reply_text)
        except Exception:
            # Best-effort JSON extrahieren
            start = reply_text.find("{")
            end = reply_text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(reply_text[start:end + 1])
                except Exception:
                    data = None

        if not data or not isinstance(data, dict):
            # Fallback wenn JSON-Parsing fehlschlägt
            return TherapyValidationResponse(
                isCorrect=False,
                feedback="Technischer Fehler bei der Validierung. Bitte versuche es erneut.",
                usage=user.monthly_usage,
                limit=limit
            )

        is_correct = bool(data.get("isCorrect", False))
        feedback = str(data.get("feedback", "Validierung abgeschlossen."))
        correct_therapy = data.get("correctTherapy")
        explanation = data.get("explanation")

        # ✅ FIX: Usage erst NACH erfolgreichem OpenAI-Call
        user.monthly_usage += 1
        db.commit()
        db.refresh(user)

        return TherapyValidationResponse(
            isCorrect=is_correct,
            feedback=feedback,
            correctTherapy=correct_therapy,
            explanation=explanation,
            usage=user.monthly_usage,
            limit=limit
        )

    except Exception as e:
        db.rollback()  # ✅ FIX: Rollback bei Fehler
        print("Fehler in /duo/validate-therapy:", repr(e))
        return TherapyValidationResponse(
            isCorrect=False,
            feedback=f"Technischer Fehler: {str(e)}",
            usage=user.monthly_usage,
            limit=limit
        )


# -------------------------------------------------------
# NEW: Duo Session REST (iPad erstellt, iPhone joint)
# -------------------------------------------------------

@app.post("/duo/session/create", response_model=DuoSessionCreateResponse)
async def duo_session_create(payload: DuoSessionCreatePayload, db: Session = Depends(get_db)):
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

    # ✅ WICHTIG: Prüfe ob der Fall übereinstimmt
    if s.case_title and s.case_title != payload.caseTitle:
        raise HTTPException(
            status_code=400, 
            detail=f"Fall stimmt nicht überein! Die Session ist für Fall '{s.case_title}', du versuchst mit Fall '{payload.caseTitle}' beizutreten."
        )

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
#   { "type": "request_coach" }  (doctor) - ✅ NEU: Explizite Coach-Anfrage
# Server broadcastet an doctor:
#   { "type": "patient_text", "text": "..." }
#   { "type": "coach_suggestion", "text": "...", "usage": X, "limit": Y }  (nur auf request_coach)
#
# connect:
#   wss://HOST/ws/duo/{sessionId}?role=doctor&userId=...
#   wss://HOST/ws/duo/{sessionId}?role=patient&userId=...
# -------------------------------------------------------

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
        }, ensure_ascii=False))

        print(f"[WS] connected session={session_id} role={role} user={user_id}")

        # -------------------------------------------------
        # MAIN RECEIVE LOOP
        # -------------------------------------------------
        while True:
            raw = await websocket.receive_text()
            print("[WS] RAW:", raw)

            try:
                data = json.loads(raw)
            except Exception:
                await websocket.send_text(json.dumps(
                    {"type": "error", "message": "Invalid JSON"},
                    ensure_ascii=False
                ))
                continue

            msg_type = data.get("type")

            # =========================
            # ✅ Quick Answers anfordern (Patient UI Button)
            # =========================
            if msg_type == "request_quick_answers":
                force = bool(data.get("force", False))
                print(f"[WS] request_quick_answers session={session_id} force={force} by user={user_id}")

                if not s.case_title or not s.case_description:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Kein Falltext vorhanden (case_title/case_description fehlen)."
                    }, ensure_ascii=False))
                    continue

                cached = ws_manager.quickanswers.get(session_id)
                cached_role_desc = ws_manager.role_descriptions.get(session_id)  # ✅ NEU: Role Description Cache
                if cached and not force:
                    await websocket.send_text(json.dumps({
                        "type": "quick_answers",
                        "items": cached,
                        "roleDescription": cached_role_desc  # ✅ NEU: Role Description mitsenden
                    }, ensure_ascii=False))
                    print(f"[WS] quick_answers cache-hit session={session_id} items={len(cached)}")
                    continue

                user = get_or_create_user(db, user_id)
                apply_month_reset(user)
                plan = user.plan
                limit = get_limit_for_plan(plan)

                if user.monthly_usage >= limit:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Limit erreicht",
                        "usage": user.monthly_usage,
                        "limit": limit
                    }, ensure_ascii=False))
                    continue

                try:
                    model_name = get_model_for_plan(plan)
                    # ✅ NEU: Generiere sowohl Rollenbeschreibung als auch Quick-Answers
                    role_desc = generate_role_description(s.case_title, s.case_description, model_name=model_name)
                    items = generate_quickanswers(
                        case_title=s.case_title,
                        case_description=s.case_description,
                        model_name=model_name
                    )
                    # ✅ FIX: Usage erst NACH erfolgreichem OpenAI-Call (beide Calls zählen als 1)
                    user.monthly_usage += 1
                    db.commit()
                    db.refresh(user)
                except Exception as e:
                    db.rollback()  # ✅ FIX: Rollback bei Fehler
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"QuickAnswers Fehler: {repr(e)}",
                        "usage": user.monthly_usage,
                        "limit": limit
                    }, ensure_ascii=False))
                    continue

                # ✅ NEU: Generiere Rollenbeschreibung
                role_desc = generate_role_description(s.case_title, s.case_description, model_name=model_name)
                
                ws_manager.quickanswers[session_id] = items
                ws_manager.role_descriptions[session_id] = role_desc  # ✅ NEU: Role Description Cache
                print(f"[WS] quick_answers generated session={session_id} items={len(items)} usage={user.monthly_usage}/{limit}")

                await websocket.send_text(json.dumps({
                    "type": "quick_answers",
                    "items": items,
                    "roleDescription": role_desc,  # ✅ NEU: Role Description mitsenden
                    "usage": user.monthly_usage,
                    "limit": limit
                }, ensure_ascii=False))
                continue

            # =========================
            # ✅ NEU: Coach explizit anfordern (Doctor UI Button)
            # =========================
            if msg_type == "request_coach" and role == "doctor":
                print(f"[WS] request_coach session={session_id} by user={user_id}")

                if not s.case_title or not s.case_description:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Kein Falltext vorhanden (case_title/case_description fehlen)."
                    }, ensure_ascii=False))
                    continue

                user = get_or_create_user(db, user_id)
                apply_month_reset(user)
                plan = user.plan
                limit = get_limit_for_plan(plan)

                if user.monthly_usage >= limit:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Limit erreicht",
                        "usage": user.monthly_usage,
                        "limit": limit
                    }, ensure_ascii=False))
                    continue

                try:
                    model_name = get_model_for_plan(plan)
                    coach_prompt = build_coach_prompt(s.case_title, s.case_description, ws_manager.history.get(session_id, []))
                    reply_text = call_openai(coach_prompt, model_name=model_name).strip()
                    
                    if reply_text:
                        # ✅ FIX: Usage erst NACH erfolgreichem OpenAI-Call
                        user.monthly_usage += 1
                        db.commit()
                        db.refresh(user)
                        
                        await websocket.send_text(json.dumps({
                            "type": "coach_suggestion",
                            "text": reply_text,
                            "usage": user.monthly_usage,
                            "limit": limit
                        }, ensure_ascii=False))
                except Exception as e:
                    db.rollback()  # ✅ FIX: Rollback bei Fehler
                    print(f"Coach-Fehler: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Coach-Fehler: {repr(e)}",
                        "usage": user.monthly_usage,
                        "limit": limit
                    }, ensure_ascii=False))
                continue

            # =========================
            # ✅ Doctor kann Case setzen
            # =========================
            if msg_type == "init_case" and role == "doctor":
                s.case_title = data.get("caseTitle") or s.case_title
                s.case_description = data.get("caseDescription") or s.case_description
                db.commit()

                ws_manager.quickanswers.pop(session_id, None)

                await ws_manager.broadcast(session_id, "doctor", {"type": "status", "message": "Case gespeichert"})
                await ws_manager.broadcast(session_id, "patient", {"type": "status", "message": "Case gespeichert"})
                print(f"[WS] init_case session={session_id} title_set={bool(s.case_title)} desc_len={len(s.case_description or '')}")
                continue

            # =========================
            # ✅ Doctor -> Patient
            # =========================
            if msg_type == "doctor_text" and role == "doctor":
                text = (data.get("text") or "").strip()
                if not text:
                    continue
                     
                ws_manager.history.setdefault(session_id, [])
                ws_manager.history[session_id].append({"role": "doctor", "content": text})
                
                # ✅ NEU: Rate-Limiting zurücksetzen wenn Arzt eine Nachricht sendet
                ws_manager.patient_message_count[session_id] = 0
                        
                await ws_manager.broadcast(session_id, "patient", {"type": "doctor_text", "text": text})
                continue

            # =========================
            # ✅ KI-Antwort Status (Patient signalisiert Start/Ende)
            # =========================
            if msg_type == "ai_reply_start" and role == "patient":
                ws_manager.patient_ai_processing[session_id] = True
                print(f"[WS] ai_reply_start session={session_id}")
                continue
                
            if msg_type == "ai_reply_end" and role == "patient":
                ws_manager.patient_ai_processing[session_id] = False
                print(f"[WS] ai_reply_end session={session_id}")
                continue

            # =========================
            # ✅ Patient -> Doctor (KEIN automatischer Coach mehr!)
            # =========================
            if msg_type == "patient_text" and role == "patient":
                text = (data.get("text") or "").strip()
                if not text:
                    continue
                
                # ✅ NEU: Rate-Limiting - Prüfe ob KI-Antwort läuft
                if ws_manager.patient_ai_processing.get(session_id, False):
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Bitte warte, bis die KI-Antwort abgeschlossen ist."
                    }, ensure_ascii=False))
                    continue
                
                # ✅ NEU: Rate-Limiting - Prüfe ob mehr als 3 Nachrichten hintereinander
                count = ws_manager.patient_message_count.get(session_id, 0)
                if count >= 3:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Du hast bereits 3 Nachrichten hintereinander gesendet. Bitte warte auf eine Antwort vom Arzt."
                    }, ensure_ascii=False))
                    continue
                
                # ✅ NEU: Nachricht zählen
                ws_manager.patient_message_count[session_id] = count + 1
                     
                ws_manager.history.setdefault(session_id, [])
                ws_manager.history[session_id].append({"role": "patient", "content": text})
                        
                # ✅ FIX: Nur Nachricht senden, KEIN automatischer Coach mehr!
                await ws_manager.broadcast(session_id, "doctor", {"type": "patient_text", "text": text})
                continue

            # --- Ende der bekannten Message-Types ---

            # Dieser Teil darf erst kommen, wenn KEINES der obigen 'if' zutraf
            await websocket.send_text(json.dumps({
                "type": "error", 
                "message": "Unknown event type"
            }, ensure_ascii=False))

    except WebSocketDisconnect:
        print(f"[WS] disconnected session={session_id} role={role} user={user_id}")
    finally:
        try:
            ws_manager.disconnect(session_id, role, websocket)
        except Exception:
            pass
        db.close()


