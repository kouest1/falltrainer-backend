import os
import json
import requests
from datetime import datetime

from jose import jwt
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session

# -------------------------------------------------------
# Datenbank-Setup (Postgres über DATABASE_URL)
# -------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    # Für lokale Tests kannst du hier z.B. eine SQLite-URL setzen
    # DATABASE_URL = "sqlite:///./test.db"
    raise RuntimeError("DATABASE_URL ist nicht gesetzt")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True, nullable=False)  # von Apple
    plan = Column(String, default="free", nullable=False)              # free/pro/premium
    monthly_usage = Column(Integer, default=0, nullable=False)
    usage_month = Column(String, nullable=True)  # "YYYY-MM"

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


# Tabellen anlegen
Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


PLAN_LIMITS = {
    "free": 6,
    "pro": 150,       # Plus
    "premium": 300,   # kannst du später auf 1000 hochsetzen
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
# FastAPI App
# -------------------------------------------------------

app = FastAPI()

APPLE_KEYS_URL = "https://appleid.apple.com/auth/keys"


class AppleAuthPayload(BaseModel):
    token: str  # identityToken von Apple


class AppleAuthResponse(BaseModel):
    userId: str
    plan: str
    monthlyUsage: int


# Dummy-Datenbank (lassen wir stehen, nutzen aber DB als Wahrheit)
dummy_users = {}


def get_apple_public_key(kid: str):
    apple_keys = requests.get(APPLE_KEYS_URL).json()["keys"]

    for key in apple_keys:
        if key["kid"] == kid:
            return key

    raise Exception("Apple Public Key nicht gefunden")


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

    # Falls neuer User -> anlegen (in richtiger DB)
    user = get_or_create_user(db, apple_user_id)
    apply_month_reset(user)
    db.commit()
    db.refresh(user)

    # Dummy-Map parallel aktualisieren (kann später weg)
    dummy_users[apple_user_id] = {
        "plan": user.plan,
        "monthlyUsage": user.monthly_usage,
    }

    return AppleAuthResponse(
        userId=apple_user_id,
        plan=user.plan,
        monthlyUsage=user.monthly_usage,
    )


@app.get("/")
def root():
    return {"message": "Backend läuft!"}


class ReceiptPayload(BaseModel):
    userId: str
    receipt: str   # Base64 aus iOS


class ReceiptResponse(BaseModel):
    plan: str
    monthlyUsage: int
    limit: int


@app.post("/validateReceipt", response_model=ReceiptResponse)
async def validate_receipt(payload: ReceiptPayload, db: Session = Depends(get_db)):
    user_id = payload.userId
    receipt_data = payload.receipt

    # WICHTIG: Für Tests Sandbox
    APPLE_VERIFY_URL = "https://sandbox.itunes.apple.com/verifyReceipt"

    response = requests.post(
        APPLE_VERIFY_URL,
        json={
            "receipt-data": receipt_data,
            "password": "DEIN_APP_STORE_SHARED_SECRET",
        }
    )

    result = response.json()

    if result.get("status") != 0:
        raise HTTPException(status_code=400, detail=f"Ungültiger Receipt: {result}")

    latest = result["latest_receipt_info"]
    product_ids = {item["product_id"] for item in latest}

    # Produkt → Plan Mapping (an deine Produkt-IDs anpassen!)
    # Beispiel: Plus150, Premium1000
    if "Premium1000" in product_ids:
        plan = "premium"
    elif "Plus150" in product_ids:
        plan = "pro"
    else:
        plan = "free"

    # User in richtiger DB holen / anlegen
    db_user = get_or_create_user(db, user_id)
    db_user.plan = plan
    # Optional: bei Planwechsel Usage zurücksetzen
    db_user.monthly_usage = 0
    db_user.usage_month = datetime.utcnow().strftime("%Y-%m")
    db.commit()
    db.refresh(db_user)

    # Dummy-Map synchron halten
    if user_id not in dummy_users:
        dummy_users[user_id] = {}
    dummy_users[user_id]["plan"] = plan
    if "monthlyUsage" not in dummy_users[user_id]:
        dummy_users[user_id]["monthlyUsage"] = 0

    limit = get_limit_for_plan(plan)

    return ReceiptResponse(
        plan=plan,
        monthlyUsage=db_user.monthly_usage,
        limit=limit,
    )


class AskPayload(BaseModel):
    userId: str
    message: str


class AskResponse(BaseModel):
    reply: str
    usage: int
    limit: int


@app.post("/ask", response_model=AskResponse)
async def ask(payload: AskPayload, db: Session = Depends(get_db)):
    user_id = payload.userId
    message = payload.message

    # User aus richtiger DB holen / anlegen
    user = get_or_create_user(db, user_id)
    apply_month_reset(user)

    plan = user.plan
    usage = user.monthly_usage
    limit = get_limit_for_plan(plan)

    if usage >= limit:
        raise HTTPException(status_code=403, detail="Limit erreicht")

    # KI hier einbauen (später) – aktuell Dummy-Reply
    reply = f"Antwort auf: {message}"

    # Usage hochzählen
    user.monthly_usage += 1
    db.commit()
    db.refresh(user)

    # Dummy-Map optional aktualisieren
    dummy_users[user_id] = {
        "plan": user.plan,
        "monthlyUsage": user.monthly_usage,
    }

    return AskResponse(
        reply=reply,
        usage=user.monthly_usage,
        limit=limit,
    )

