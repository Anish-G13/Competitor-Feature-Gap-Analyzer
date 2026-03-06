"""
Company sentiment analysis using Gemini 2.5 Flash.
Returns JSON with sentiment, risks, and recommendation.
"""

import json
from typing import Any, Optional

import google.generativeai as genai
import streamlit as st


def _ensure_genai() -> bool:
    if "GOOGLE_API_KEY" not in st.secrets:
        st.error("Add `GOOGLE_API_KEY` to `.streamlit/secrets.toml`.")
        return False
    key = st.secrets["GOOGLE_API_KEY"].strip()
    if not key:
        st.error("Set `GOOGLE_API_KEY` in `.streamlit/secrets.toml`. Get a key at https://aistudio.google.com/apikey")
        return False
    genai.configure(api_key=key)
    return True


def call_gemini(prompt: str) -> Optional[dict[str, Any]]:
    """
    Call Gemini 2.5 Flash and return parsed JSON with keys:
    - sentiment: overall sentiment (e.g. string or object)
    - risks: list of risk items
    - recommendation: string or list of recommendations
    """
    if not _ensure_genai():
        return None

    system = (
        "You are an analyst. Respond only with valid JSON. No markdown code fences or extra text. "
        "Include exactly: \"sentiment\", \"risks\" (array of strings), \"recommendation\" (string or array)."
    )
    full_prompt = f"{system}\n\n{prompt}"

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(full_prompt)
        text = (resp.text or "").strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        data = json.loads(text)
        # Normalize to expected shape
        if "sentiment" not in data:
            data["sentiment"] = None
        if "risks" not in data:
            data["risks"] = []
        if "recommendation" not in data:
            data["recommendation"] = None
        return data
    except json.JSONDecodeError as e:
        st.error(f"Could not parse model response as JSON: {e}")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None
