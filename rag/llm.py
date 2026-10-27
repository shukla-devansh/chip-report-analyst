"""The only file that talks to the LLM. Swap this out to use a different provider."""

import os

from dotenv import load_dotenv

from config import LLM_MODEL

load_dotenv()  # reads GEMINI_API_KEY from the .env file


def has_api_key() -> bool:
    return bool(os.getenv("GEMINI_API_KEY"))


def ask_gemini(prompt: str) -> str:
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
        config={"temperature": 0},  # same question -> same answer
    )
    return response.text
