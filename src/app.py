import streamlit as st
from openai import OpenAI
from groq import Groq
from google import genai
from core.config import config

def run_llm(provider, model_name, messages, max_tokens=500):

    if provider == "OpenAI":
        client = OpenAI(
            api_key=config.GROQ_API_KEY,
            base_url=config.BASE_GROQ_URL
        )
    elif provider == "Google":
        client = genai.Client(api_key=config.GOOGLE_API_KEY)
    else:
        client = Groq(api_key=config.GROQ_API_KEY)

    if provider == "Google":
        return client.interactions.create(
            model=model_name,
            input=messages[0]["content"]
        ).output_text
    elif provider == "OpenAI":
        return client.responses.create(
            model=model_name,
            input=messages
        ).output_text
    else:
        return client.chat.completions.create(
            model=model_name,
            messages=messages
        ).choices[0].message.content



