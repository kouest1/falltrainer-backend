from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

class AskPayload(BaseModel):
    message: str

@app.post("/ask")
async def ask(payload: AskPayload):
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "user", "content": payload.message}
        ]
    )
    
    # Neues SDK: so greift man auf den Text zu
    answer = response.choices[0].message.content
    return {"answer": answer}

