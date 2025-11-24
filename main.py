import os
import json
from datetime import datetime

import requests
from jose import jwt
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import Base, engine, SessionLocal
from models import User

from openai import OpenAI

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


def call_openai(message: str) -> str:
    """
    Schickt den Prompt an OpenAI und gibt NUR die Modell-Antwort zurück.
    """
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",  # oder dein Wunschmodell
        messages=[
            {"role": "user", "content": message}
        ],
        temperature=0.7,
    )
    content = completion.choices[0].message.content
    return content or ""


# -------------------------------------------------------
# User + Plan Logik
# -------------------------------------------------------

PLAN_LIMITS = {
    "free": 6,
    "pro": 150,
    "premium": 1000,
}


def get_or_create_user(db: Session, user_id: str) -> User:
    user = db.query(User).filter(User.user_id == user_id).first()
    if user is None:
        user = User(
            user_id=user_id,
            plan="free",
            monthly_usage=0,
            usage_month=None,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


def apply_month_reset(user: User):
    now_month = datetime.utcnow().strftime("%Y-%m")
    if user.usage_month != now_month:
        user.usage_month = now_month
        user.monthly_usage = 0


def get_limit_for_plan(plan: str) -> int:
    return PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])


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
            "password": "DEIN_APP_STORE_SHARED_SECRET",  # TODO: ersetzen
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
    user.usage_month = datetime.utcnow().strftime("%Y-%m")

    db.commit()
    db.refresh(user)

    limit = get_limit_for_plan(plan)

    return ReceiptResponse(
        plan=plan,
        monthlyUsage=user.monthly_usage,
        limit=limit,
    )


# 3) KI-Frage stellen (mit Limit)
@app.post("/ask", response_model=AskResponse)
async def ask(payload: AskPayload, db: Session = Depends(get_db)):
    user_id = payload.userId
    message = payload.message

    user = get_or_create_user(db, user_id)
    apply_month_reset(user)

    plan = user.plan
    usage = user.monthly_usage
    limit = get_limit_for_plan(plan)

    if usage >= limit:
        raise HTTPException(status_code=403, detail="Limit erreicht")

    # KI anfragen
    try:
        reply = call_openai(message)
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

