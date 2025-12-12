import os
import json
from datetime import datetime

import requests
from jose import jwt
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from openai import OpenAI

from database import Base, engine, SessionLocal
from models import User

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
        # Platzhalter – in Render über OPENAI_MODEL_FREE überschreibbar
        "model": os.getenv("OPENAI_MODEL_FREE", "gpt-4.5"),
    },
    "pro": {
        "limit": 300,
        "model": os.getenv("OPENAI_MODEL_PRO", "gpt-5-mini"),
    },
    "premium": {
        "limit": 300,
        "model": os.getenv("OPENAI_MODEL_PREMIUM", "gpt-5.1"),
    },
}


def current_month_str() -> str:
    """z.B. '2025-11' – so speichern wir die Nutzung pro Monat."""
    return datetime.utcnow().strftime("%Y-%m")


def get_limit_for_plan(plan: str) -> int:
    cfg = PLAN_CONFIG.get(plan, PLAN_CONFIG["free"])
    return cfg["limit"]


def get_model_for_plan(plan: str) -> str:
    cfg = PLAN_CONFIG.get(plan, PLAN_CONFIG["free"])
    return cfg["model"]


def call_openai(message: str, model_name: str) -> str:
    """
    Schickt den Prompt an OpenAI und gibt NUR die Modell-Antwort zurück.
    """
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": message}
        ],
        temperature=1,
    )
    content = completion.choices[0].message.content
    return content or ""


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
    receipt: str   # Base64 aus iOS


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


# ---------- NEU: Notizen-Schemas ----------

class NotesLoadPayload(BaseModel):
    userId: str


class NoteUpdatePayload(BaseModel):
    userId: str
    caseId: str   # UUID-String des Falls
    note: str     # kompletter Text der Notiz


class NotesResponse(BaseModel):
    notes: dict[str, str]  # { "caseId": "Notiztext", ... }

from pydantic import BaseModel
from typing import List, Literal, Optional

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
    usage: Optional[int] = None   # falls du Usage/Limits mitschicken willst


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
# Endpoints
# -------------------------------------------------------

@app.get("/")
def root():
    return {"message": "Backend läuft!"}


# 1) Login mit Apple
@app.post("/auth/apple", response_model=AppleAuthResponse)
async def auth_apple(payload: AppleAuthPayload, db: Session = Depends(get_db)):
    identity_token = payload.token

    # JWT Header lesen -> dort steht "kid"
    header = jwt.get_unverified_header(identity_token)
    kid = header["kid"]

    # Passenden Apple Public Key laden
    pubkey_dict = get_apple_public_key(kid)

    # Token mittels python-jose prüfen
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

    # User aus DB holen/anlegen
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

    latest = result["latest_receipt_info"]
    product_ids = {item["product_id"] for item in latest}

    # Produkt-ID → Plan Mapping (an deine IDs angepasst)
    if "Premium1000" in product_ids:
        plan = "premium"
    elif "Plus150" in product_ids:
        plan = "pro"
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


# 3) KI-Frage stellen (mit Limit + Plan-spezifischem Modell)
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

    # Limit prüfen
    if usage >= limit:
        raise HTTPException(status_code=403, detail="Limit erreicht")

    # KI anfragen
    try:
        reply = call_openai(message, model_name=model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fehler bei KI-Anfrage: {e}")

    # Usage hochzählen
    user.monthly_usage += 1
    db.commit()
    db.refresh(user)

    return AskResponse(
        reply=reply,
        usage=user.monthly_usage,
        limit=limit,
    )


# 4) Erledigte / ausgeschlossene Fälle LADEN
@app.post("/user/excluded/load", response_model=ExcludedCasesResponse)
async def load_excluded_cases(
    payload: ExcludedCasesPayload,
    db: Session = Depends(get_db),
):
    # User holen/anlegen
    user = get_or_create_user(db, payload.userId)

    # JSON aus der DB lesen
    if user.excluded_cases:
        try:
            ids = json.loads(user.excluded_cases)
        except Exception:
            ids = []
    else:
        ids = []

    # sicherstellen, dass wir Strings zurückgeben
    ids = [str(x) for x in ids]

    return ExcludedCasesResponse(excludedIds=ids)


# 5) Erledigte / ausgeschlossene Fälle SPEICHERN
@app.post("/user/excluded/save", response_model=ExcludedCasesResponse)
async def save_excluded_cases(
    payload: ExcludedCasesPayload,
    db: Session = Depends(get_db),
):
    # User holen/anlegen
    user = get_or_create_user(db, payload.userId)

    # Liste aus dem Payload nehmen und als JSON sichern
    user.excluded_cases = json.dumps(payload.excludedIds)
    db.commit()
    db.refresh(user)

    return ExcludedCasesResponse(excludedIds=payload.excludedIds)


# 6) Notizen LADEN
@app.post("/user/notes/load", response_model=NotesResponse)
async def load_notes(
    payload: NotesLoadPayload,
    db: Session = Depends(get_db),
):
    user = get_or_create_user(db, payload.userId)

    if user.notes:
        try:
            notes = json.loads(user.notes)
        except Exception:
            notes = {}
    else:
        notes = {}

    # Fallback, falls irgendwas anderes drinsteht
    if not isinstance(notes, dict):
        notes = {}

    # Keys & Values sicher als Strings
    clean_notes: dict[str, str] = {}
    for k, v in notes.items():
        clean_notes[str(k)] = str(v)

    return NotesResponse(notes=clean_notes)


# 7) Notizen SPEICHERN / aktualisieren
@app.post("/user/notes/save", response_model=NotesResponse)
async def save_note(
    payload: NoteUpdatePayload,
    db: Session = Depends(get_db),
):
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

    # leere Notiz = löschen
    if payload.note.strip() == "":
        notes.pop(payload.caseId, None)
    else:
        notes[payload.caseId] = payload.note

    user.notes = json.dumps(notes)
    db.commit()
    db.refresh(user)

    # wieder aufräumen und zurückgeben
    clean_notes: dict[str, str] = {}
    for k, v in notes.items():
        clean_notes[str(k)] = str(v)

    return NotesResponse(notes=clean_notes)
@app.post("/duoChat", response_model=DuoChatResponse)
async def duo_chat(req: DuoChatRequest):
    """
    Spielt den Patienten im Arzt/Patienten-Duo-Modus.
    iOS schickt: userId, caseTitle, caseDescription, messages[role, content]
    """

    # 1) System-Prompt: Rolle + Fallkontext
    system_prompt = (
    "Du bist ein SIMULIERTER PATIENT in einem neurologischen Trainings-Chat (Anamnese + Untersuchung).\n"
    "Der Benutzer ist der Arzt. Du kennst die Fallbeschreibung als Ground Truth, der Arzt nicht.\n\n"
    "HARTER RAHMEN\n"
    "- Du bist medizinischer Laie: keine Fachbegriffe, keine Diagnosen, keine Erklaerungen. Fachbegriffe aus dem Falltext immer in Alltagssprache uebersetzen.\n"
    "- Nutze nur Infos aus der Fallbeschreibung. Nichts erfinden.\n"
    "- Wenn der Arzt etwas fragt, was nicht im Falltext steht oder noch nicht erhoben wurde: sage ehrlich, dass du das nicht weisst / nicht beurteilen kannst / dass es nicht untersucht wurde.\n"
    "- Keine Labor/CT/MRT/EEG/LP-Ergebnisse nennen, ausser sie stehen im Falltext oder wurden angeordnet UND sind im Falltext vorhanden.\n\n"
    "ANTWORTSTIL\n"
    "- Ich-Form, hoeflich, menschlich, eher kurz (1bis4 Saetze).\n"
    "- Pro Nachricht nur neue relevante Infos, die der Arzt erfragt hat. Keine ungefragten Info-Dumps.\n"
    "- Mehrere Fragen: der Reihe nach kurz beantworten.\n"
    "- Unklare Fachwoerter: freundlich nachfragen, z.B. \"Was meinen Sie genau?\"\n\n"
    "UNTERSUCHUNG\n"
    "- Bei Untersuchungsanordnungen reagierst du wie ein Patient (Kooperation, subjektive Eindruecke), ausser im Falltext steht, dass du nicht kooperierst.\n"
    "- Objektive Befunde nur nennen, wenn sie im Falltext stehen.\n\n"
    f"Falltitel: {req.caseTitle}\n\n"
    f"Fallbeschreibung (Ground Truth):\n{req.caseDescription}\n\n"
    "Antworte als Patient nur auf die letzte Arztnachricht."
)
    # 2) bisherigen Dialog in Text gießen
    convo_lines = []
    for msg in req.messages:
        sprecher = "Arzt" if msg.role == "doctor" else "Patient"
        convo_lines.append(f"{sprecher}: {msg.content}")
    conversation_text = "\n".join(convo_lines) if convo_lines else "– noch kein Dialog –"

    prompt = (
        system_prompt
        + "\n\nBisheriger Dialog zwischen Arzt und Patient:\n"
        + conversation_text
        + "\n\nAntwort des Patienten (ein kurzer, natürlicher Satz):"
    )

    # 3) Modell wählen – wie bei dir im PLAN_CONFIG
    model_name = PLAN_CONFIG.get("free", {}).get("model") or "gpt-4.5-mini"

    try:
        # SYNCHRONER Call – KEIN await, weil client = OpenAI(...)
        resp = client.responses.create(
            model=model_name,
            input=prompt,
        )

        # Text aus der Response holen – gleiche Logik wie bei deinem /ask
        try:
            # übliche Struktur der Responses-API
            reply_text = resp.output[0].content[0].text
        except Exception:
            # Fallback, falls du ein anderes Format verwendest
            reply_text = getattr(resp, "output_text", str(resp))

        # Hier könntest du optional usage auswerten, wenn du willst:
        # new_usage = resp.usage.total_tokens  # oder etwas Ähnliches
        new_usage = None

        return DuoChatResponse(reply=reply_text.strip(), usage=new_usage)

    except Exception as e:
        print("Fehler in /duoChat:", repr(e))
        return DuoChatResponse(
            reply="Entschuldigung, ich kann gerade nicht gut antworten – es gab einen technischen Fehler.",
            usage=None,
        )
@app.post("/duoDoctorChat", response_model=DuoChatResponse)
async def duo_doctor_chat(req: DuoChatRequest):
    """
    Coaching für die Arzt-Rolle im Duo-Modus.
    iOS schickt: userId, caseTitle, caseDescription, messages[role, content]
    """

    system_prompt = (
        "Du bist ein erfahrener Neurologe und Lehrarzt. "
        "Du siehst den Dialog zwischen einem Patienten und einem Assistenzarzt.\n"
        "Der Assistenzarzt tippt ein, was der Patient sagt. "
        "Deine Aufgabe ist, knappe, konkrete Vorschläge zu machen:\n"
        "- Welche Frage sollte der Arzt als nächstes stellen?\n"
        "- Welche körperliche Untersuchung oder Zusatzdiagnostik bietet sich an?\n"
        "Antworte in 1–3 kurzen Sätzen auf Deutsch, als Vorschlag an den Arzt.\n\n"
        f"Falltitel: {req.caseTitle}\n\n"
        f"Fallbeschreibung (medizinischer Hintergrund):\n{req.caseDescription}\n"
    )

    # bisherigen Verlauf in Textform
    lines = []
    for msg in req.messages:
        if msg.role == "patient":
            lines.append(f"Patient: {msg.content}")
        else:
            lines.append(f"Arzt-Coach: {msg.content}")
    conversation_text = "\n".join(lines) if lines else "Noch keine Aussagen."

    prompt = (
        system_prompt
        + "\n\nBisheriger Dialog:\n"
        + conversation_text
        + "\n\nDein nächster Vorschlag an den Arzt:"
    )

    model_name = PLAN_CONFIG.get("free", {}).get("model") or "gpt-4.5-mini"

    try:
        resp = client.responses.create(
            model=model_name,
            input=prompt,
        )

        try:
            reply_text = resp.output[0].content[0].text
        except Exception:
            reply_text = getattr(resp, "output_text", str(resp))

        return DuoChatResponse(reply=reply_text.strip(), usage=None)

    except Exception as e:
        print("Fehler in /duoDoctorChat:", repr(e))
        return DuoChatResponse(
            reply="Ich kann gerade keine sinnvollen Vorschläge machen – es gab einen technischen Fehler.",
            usage=None,
        )

