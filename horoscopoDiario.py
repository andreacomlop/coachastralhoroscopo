import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

from openai import OpenAI


# ---------------------------
# Config (Render env vars)
# ---------------------------
TZ_NAME = os.getenv("TZ", "Europe/Madrid").strip()

ASTRO_USER_ID = os.getenv("ASTRO_USER_ID", "").strip()
ASTRO_API_KEY = os.getenv("ASTRO_API_KEY", "").strip()
HORO_TZ = os.getenv("HORO_TZ", "").strip()  # opcional: "1", "2", "5.5"...
ASTRO_BASE = "https://json.astrologyapi.com/v1"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()


SIGNS_ES = [
    "aries", "tauro", "géminis", "cáncer", "leo", "virgo",
    "libra", "escorpio", "sagitario", "capricornio", "acuario", "piscis",
]

SIGNS_EN = {
    "aries": "aries",
    "tauro": "taurus",
    "géminis": "gemini",
    "cáncer": "cancer",
    "leo": "leo",
    "virgo": "virgo",
    "libra": "libra",
    "escorpio": "scorpio",
    "sagitario": "sagittarius",
    "capricornio": "capricorn",
    "acuario": "aquarius",
    "piscis": "pisces",
}

# Cache en memoria (por día)
CACHE = {}  # { "YYYY-MM-DD": {...} }


# ---------------------------
# App
# ---------------------------
app = Flask(__name__)
CORS(app)


# ---------------------------
# Helpers
# ---------------------------
def today_key_and_label():
    """Devuelve (date_key 'YYYY-MM-DD', label 'DD-MM-YYYY') en TZ."""
    now = datetime.now(ZoneInfo(TZ_NAME))
    return now.strftime("%Y-%m-%d"), now.strftime("%d-%m-%Y")


def timezone_offset_hours() -> float:
    """Offset actual respecto a UTC en horas (ej: 1, 2)."""
    now = datetime.now(ZoneInfo(TZ_NAME))
    offset = now.utcoffset()
    return offset.total_seconds() / 3600 if offset else 0.0


def get_tz_for_astrology() -> float:
    """
    AstrologyAPI acepta timezone como float.
    Si HORO_TZ está en Render, se usa; si no, calculamos el offset real.
    """
    if HORO_TZ:
        try:
            return float(HORO_TZ.replace(",", "."))
        except ValueError:
            pass
    return timezone_offset_hours()


# ---------------------------
# AstrologyAPI (FUENTE del contenido)
# ---------------------------
def astrology_consolidated_daily(sign_en: str, tz_hours: float) -> dict:
    """
    POST https://json.astrologyapi.com/v1/sun_sign_consolidated/daily/<zodiac>
    Auth: Basic (ASTRO_USER_ID, ASTRO_API_KEY)
    """
    if not ASTRO_USER_ID or not ASTRO_API_KEY:
        raise RuntimeError("Faltan ASTRO_USER_ID o ASTRO_API_KEY en Render.")

    url = f"{ASTRO_BASE}/sun_sign_consolidated/daily/{sign_en}"
    payload = {"timezone": tz_hours}

    r = requests.post(url, auth=(ASTRO_USER_ID, ASTRO_API_KEY), json=payload, timeout=25)
    if r.status_code >= 400:
        raise RuntimeError(f"AstrologyAPI error {r.status_code}: {r.text[:300]}")

    return r.json()


def extract_prediction_sections(api_json: dict) -> dict:
    """
    Intenta extraer secciones del consolidated.
    Normalmente viene como:
      { "prediction": { "personal_life": "...", "profession": "...", ... } }
    Si no, devolvemos lo que haya sin romper.
    """
    pred = api_json.get("prediction")

    if isinstance(pred, dict) and pred:
        return pred

    if isinstance(pred, str) and pred.strip():
        return {"general": pred.strip()}

    # fallback: busca en claves habituales
    for k in ["personal_life", "profession", "health", "luck", "emotion", "travel"]:
        if isinstance(api_json.get(k), str) and api_json.get(k).strip():
            # si viene plano, lo empaquetamos
            return {k: api_json.get(k).strip()}

    return {"raw": json.dumps(api_json, ensure_ascii=False)}


# ---------------------------
# OpenAI (SOLO traducción)
# ---------------------------
def translate_es_strict(text: str) -> str:
    """
    Traducción fiel: no añadir, no quitar, no resumir.
    Si no hay OPENAI_API_KEY, devolvemos el original.
    """
    if not text:
        return ""

    if not OPENAI_API_KEY:
        return text

    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = (
        "Traduce al español de España el siguiente texto.\n"
        "Reglas estrictas:\n"
        "- Traducción fiel.\n"
        "- No añadas información.\n"
        "- No elimines información.\n"
        "- No resumas.\n"
        "- No cambies el tono.\n"
        "- Devuelve SOLO la traducción.\n\n"
        f"TEXTO:\n{text}"
    )

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": "Eres un traductor profesional. Respondes solo con la traducción."},
            {"role": "user", "content": prompt},
        ],
    )

    return (resp.choices[0].message.content or "").strip()


def join_sections_for_prediction(sections_es: dict) -> str:
    """
    Devuelve un único texto (por si GoodBarber pinta solo una caja).
    """
    parts = []
    for k, v in sections_es.items():
        label = k.replace("_", " ").strip().capitalize()
        parts.append(f"{label}: {v}")
    return "\n\n".join(parts).strip()


# ---------------------------
# Construcción consolidada (12 signos)
# ---------------------------
def build_consolidated_today(date_key: str, date_label: str) -> dict:
    tz_hours = get_tz_for_astrology()
    signs_out = {}

    for sign_es in SIGNS_ES:
        sign_en = SIGNS_EN[sign_es]

        api_json = astrology_consolidated_daily(sign_en, tz_hours)
        sections_en = extract_prediction_sections(api_json)

        # traducimos cada sección (si hay OpenAI)
        sections_es = {k: translate_es_strict(str(v)) for k, v in sections_en.items()}

        # texto unido para compatibilidad
        prediction = join_sections_for_prediction(sections_es)

        signs_out[sign_es] = {
            "prediction": prediction,
            "prediction_sections": sections_es,
            "prediction_date": date_label,
            "sun_sign": sign_es,
        }

    return {
        "_cached": False,
        "date_key": date_key,
        "signs": signs_out,
    }


# ---------------------------
# Routes
# ---------------------------
@app.get("/")
def home():
    return (
        "<h1>Coach Astral - API Horóscopo</h1>"
        "<p>Endpoint: <code>/api/horoscope/today</code></p>",
        200,
        {"Content-Type": "text/html; charset=utf-8"},
    )


@app.get("/api/horoscope/today")
def api_today():
    force = request.args.get("force", "0").strip() == "1"
    date_key, date_label = today_key_and_label()

    if (not force) and (date_key in CACHE):
        cached = dict(CACHE[date_key])
        cached["_cached"] = True
        return jsonify(cached)

    result = build_consolidated_today(date_key, date_label)
    CACHE[date_key] = result
    return jsonify(result)


if __name__ == "__main__":
    # Local dev
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=True)

