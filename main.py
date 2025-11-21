import json
import requests
import jwt
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import os

app = FastAPI()

import logging
logging.warning("FastAPI wurde gestartet!")

# ---------- MODELS ----------
class AppleAuthPayload(BaseModel):
    token: str  # identityToken vom iPhone


class AppleAuthResponse(BaseModel):
    userId: str
    plan: str
    monthlyUsage: int


# ---------- APPLE PUBLIC KEYS ----------
APPLE_KEYS_URL = "https://appleid.apple.com/auth/keys"


def get_apple_public_key(kid: str):
    # Public Keys von Apple holen
    apple_keys = requests.get(APPLE_KEYS_URL).json()["keys"]

    # passenden Key anhand von "kid" finden
    for key in apple_keys:
        if key["kid"] == kid:
            return key

    raise Exception("Apple Key ID nicht gefunden")


# ---------- USER DATABASE (Dummy) ----------
dummy_users = {
    # Beispiel:
    # "APPLE_USER_12345": {"plan": "free", "monthlyUsage": 10}
}


# ---------- AUTH /apple ----------
@app.post("/auth/apple", response_model=AppleAuthResponse)
async def auth_apple(payload: AppleAuthPayload):

    identity_token = payload.token

    # Header des JWT auslesen (darin steckt der kid)
    header = jwt.get_unverified_header(identity_token)
    kid = header["kid"]

    # Richtigen Public Key holen
    pubkey_dict = get_apple_public_key(kid)

    # Key in ein verifizierbares Format packen
    public_key = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(pubkey_dict))

    # Token validieren & decodieren
    try:
        decoded = jwt.decode(
            identity_token,
            key=public_key,
            audience="com.konstantin.falltrainer.backend",
            algorithms=["RS256"]
        )
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Ungültiges Apple Token: {e}")

    # Apple User ID extrahieren (wichtig!)
    apple_user_id = decoded["sub"]

    # Falls User bei uns noch nicht existiert → anlegen
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
    return {"message": "Backend läuft!", "endpoints": [route.path for route in app.routes]}

print("Registrierte Endpoints:")
for route in app.routes:
    print(" -", route.path)

