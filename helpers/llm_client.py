
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI()

def ask_llm(
    prompt: str,
    model: str = "gpt-4o-mini",
    system: str | None = None,
    temperature: float = 0.2,
):
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content
