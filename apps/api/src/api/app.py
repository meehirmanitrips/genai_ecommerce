# pyrefly: ignore[missing-import]
from fastapi import FastAPI, Request
# pyrefly: ignore[missing-import]
from fastapi.exceptions import HTTPException
from pydantic import BaseModel

from openai import OpenAI
from groq import Groq
from google import genai

from api.core.config import config

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Given a provider, creates a client and returns response from the client
def run_llm(provider, model_name, messages, max_tokens=500):

    if provider == "OpenAI":
        client = OpenAI(
            api_key=config.GROQ_API_KEY,
            base_url=config.BASE_GROQ_URL
        )

        return client.responses.create(
            model=model_name,
            input=messages,
            reasoning={"effort": "low"}
        ).output_text
    elif provider == "Google":
        client = genai.Client(api_key=config.GOOGLE_API_KEY)

        return client.interactions.create(
            model=model_name,
            input=" ".join([message["content"] for message in messages]),
            generation_config={"thinking_level": "low"}
        ).output_text
    else:
        client = Groq(api_key=config.GROQ_API_KEY)

        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=max_tokens
        ).choices[0].message.content


class ChatRequest(BaseModel):
    provider: str
    model_name: str
    messages: list[dict]

class ChatResponse(BaseModel):
    message: str

app = FastAPI()

@app.post("/chat")
def chat(
    request: Request,
    payload: ChatRequest
) -> ChatResponse:

    try:
        logger.info(
            f"Chat request: Provider: {payload.provider}, Model: {payload.model_name}"
        )
        
        result = run_llm(provider=payload.provider, model_name=payload.model_name, messages=payload.messages)

        return ChatResponse(message=result)
    
    except Exception as e:
        logger.error(
            f"Request failed with error: {e}"
        )
        raise HTTPException(status_code=500, detail=str(e))