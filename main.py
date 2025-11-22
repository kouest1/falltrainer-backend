import json
import requests
from jose import jwt
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

APPLE_KEYS_URL = "https://appleid.apple.com/auth/keys"


class AppleAuthPayload(BaseModel):
    token: str  # identityToken von Apple


class AppleAuthResponse(BaseModel):
    userId: str
    plan: str
    monthlyUsage: int


# Dummy-Datenbank
dummy_users = {}


def get_apple_public_key(kid: str):
    apple_keys = requests.get(APPLE_KEYS_URL).json()["keys"]

    for key in apple_keys:
        if key["kid"] == kid:
            return key

    raise Exception("Apple Public Key nicht gefunden")


@app.post("/auth/apple", response_model=AppleAuthResponse)
async def auth_apple(payload: AppleAuthPayload):

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

    # Apple User ID extrahieren
    apple_user_id = decoded["sub"]

    # Falls neuer User -> anlegen
    if apple_user_id not in dummy_users:
        dummy_users[apple_user_id] = {
            "plan": "free",
            "monthlyUsage": 0
        }

    user = dummy_users[apple_user_id]

    return AppleAuthResponse(
        userId=apple_user_id,
        plan=user["plan"],
        monthlyUsage=user["monthlyUsage"]
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
async def validate_receipt(payload: ReceiptPayload):

    user_id = payload.userId
    receipt_data = payload.receipt

    # WICHTIG: Für Tests Sandbox
    APPLE_VERIFY_URL = "https://sandbox.itunes.apple.com/verifyReceipt"

    response = requests.post(
        APPLE_VERIFY_URL,
        json={
            "receipt-data": receipt_data,
            "password": "DEIN_APP_STORE_SHARED_SECRET"
        }
    )

    result = response.json()

    if result.get("status") != 0:
        raise HTTPException(status_code=400, detail=f"Ungültiger Receipt: {result}")

    latest = result["latest_receipt_info"]
    product_ids = {item["product_id"] for item in latest}

    # Produkt → Plan Mapping
    if "premium300" in product_ids:
        plan = "premium"
    elif "pro150" in product_ids:
        plan = "pro"
    else:
        plan = "free"

    if user_id not in dummy_users:
        dummy_users[user_id] = {}

    dummy_users[user_id]["plan"] = plan

    # Limits
    limits = {
        "free": 6,
        "pro": 150,
        "premium": 300
    }

    # Falls neuer User → usage setzen
    if "monthlyUsage" not in dummy_users[user_id]:
        dummy_users[user_id]["monthlyUsage"] = 0

    return ReceiptResponse(
        plan=plan,
        monthlyUsage=dummy_users[user_id]["monthlyUsage"],
        limit=limits[plan]
    )
class AskPayload(BaseModel):
    userId: str
    message: str

class AskResponse(BaseModel):
    reply: str
    usage: int
    limit: int

@app.post("/ask", response_model=AskResponse)
async def ask(payload: AskPayload):

    user_id = payload.userId
    message = payload.message

    if user_id not in dummy_users:
        raise HTTPException(status_code=403, detail="User nicht gefunden")

    plan = dummy_users[user_id]["plan"]
    usage = dummy_users[user_id]["monthlyUsage"]

    limits = {
        "free": 6,
        "pro": 150,
        "premium": 300
    }

    limit = limits[plan]

    if usage >= limit:
        raise HTTPException(status_code=403, detail="Limit erreicht")

    # KI hier einbauen (später)
    reply = f"Antwort auf: {message}"

    dummy_users[user_id]["monthlyUsage"] += 1

    return AskResponse(
        reply=reply,
        usage=dummy_users[user_id]["monthlyUsage"],
        limit=limit
    )

