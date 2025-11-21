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

