import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

from openai import OpenAI


# ---------------------------
# Config
# ---------------------------
TZ_NAME = os.getenv("TZ", "Europe/Madrid")

# OpenAI SOLO TRADUCE
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

# HoroscopeAPI (Astrology genera el contenido)
HOROSCOPEAPI_KEY = os.getenv("HOROSCOPEAPI_KEY", "").strip()
HOROSCOPEAPI_URL = os.getenv(
    "HOROSCOPEAPI_URL",
    "https://horoscopeapi.com/v1/sun_sign_prediction/daily/zodiac"
).strip()

# Cómo enviar la API key (elige UNA de estas formas con env vars)
# - Header: HOROSCOPEAPI_KEY_HEADER = "X-API-Key" (o "Authorization")
# - Query param: HOROSCOPEAPI_KEY_QUERY = "api_key" (o "key")
HOROSCOPEAPI_KEY_HEADER = os.getenv("HOROSCOPEAPI_KEY_HEADER", "X-API-Key").strip()
HOROSCOPEAPI_KEY_QUERY = os.getenv("HOROSCOPEAPI_KEY_QUERY", "").strip()  # si lo usas, pon nombre del parámetro

SIGNS_ES = [
    "aries",
    "tauro",
    "géminis",
    "cáncer",
    "leo",
    "virgo",
    "libra",
    "escorpio",
    "sagitario",
    "capricornio",
    "acuario",
    "piscis",
]

# Mapeo ES -> EN (HoroscopeAPI suele querer EN)
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


def today_key_and_label():
    """Devuelve (date_key 'YYYY-MM-DD', label 'DD-MM-YYYY') en TZ."""
    now = datetime.now(ZoneInfo(TZ_NAME))
    date_key = now.strftime("%Y-%m-%d")
    label = now.strftime("%d-%m-%Y")
    return date_key, label


def timezone_offset_hours() -> float:
    """Offset horario actual respecto a UTC en horas (ej: 1, 2)."""
    now = datetime.now(ZoneInfo(TZ_NAME))
    offset = now.utcoffset()
    if not offset:
        return 0.0
    return offset.total_seconds() / 3600.0


def fetch_horoscope_one_sign_en(sign_en: str, tz_hours: float) -> dict:
    """
    Llama a HoroscopeAPI y devuelve la respuesta JSON original (EN o el idioma que devuelva).
    """
    if not HOROSCOPEAPI_KEY:
        raise RuntimeError("Falta HOROSCOPEAPI_KEY en variables de entorno (Render).")

    headers = {
        "Content-Type": "application/json",
        "Accept-Language": "en",
    }

    # Auth por header (lo más típico)
    if HOROSCOPEAPI_KEY_HEADER:
        headers[HOROSCOPEAPI_KEY_HEADER] = HOROSCOPEAPI_KEY

    payload = {
        "timezone": tz_hours,
        "sunSign": sign_en,
    }

    params = None
    # Auth por query param (si tu proveedor lo requiere)
    if HOROSCOPEAPI_KEY_QUERY:
        params = {HOROSCOPEAPI_KEY_QUERY: HOROSCOPEAPI_KEY}

    r = requests.post(HOROSCOPEAPI_URL, headers=headers, params=params, json=payload, timeout=25)
    if r.status_code >= 400:
        raise RuntimeError(f"HoroscopeAPI error {r.status_code}: {r.text[:300]}")

    try:
        return r.json()
    except Exception:
        raise RuntimeError(f"HoroscopeAPI no devolvió JSON válido: {r.text[:300]}")


def translate_to_spanish_strict(text: str) -> str:
    """
    Traducción estricta: NO añadir, NO quitar, NO reescribir, NO tono coach.
    """
    if not OPENAI_API_KEY:
        # Si no hay OpenAI, devolvemos el texto tal cual (EN)
        return text

    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = (
        "Traduce al español de España el siguiente texto.\n"
        "Reglas estrictas:\n"
        "- Traducción fiel y literal.\n"
        "- No añadas información.\n"
        "- No elimines información.\n"
        "- No cambies el tono ni resumas.\n"
        "- No introduzcas astrología extra ni consejos nuevos.\n\n"
        f"TEXTO:\n{text}"
    )

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": "Eres un traductor profesional. Devuelves solo la traducción, sin texto extra."},
            {"role": "user", "content": prompt},
        ],
    )

    return (resp.choices[0].message.content or "").strip()


def extract_prediction_text(api_json: dict) -> str:
    """
    Intenta encontrar el campo de predicción dentro del JSON devuelto por HoroscopeAPI.
    Como no tenemos el esquema exacto aquí, lo hacemos robusto.
    Ajusta si ves el nombre exacto.
    """
    # Intentos comunes:
    for key in ["prediction", "horoscope", "text", "description", "daily_prediction"]:
        val = api_json.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    # Si viene anidado:
    for container_key in ["data", "result", "response"]:
        cont = api_json.get(container_key)
        if isinstance(cont, dict):
            for key in ["prediction", "horoscope", "text", "description", "daily_prediction"]:
                val = cont.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()

    # Último recurso: stringify (para que no falle silencioso)
    return json.dumps(api_json, ensure_ascii=False)


def build_consolidated_today(date_key: str, date_label: str) -> dict:
    tz_hours = timezone_offset_hours()

    signs_out = {}

    for sign_es in SIGNS_ES:
        sign_en = SIGNS_EN[sign_es]

        raw = fetch_horoscope_one_sign_en(sign_en, tz_hours)
        prediction_en = extract_prediction_text(raw)

        prediction_es = translate_to_spanish_strict(prediction_en)

        signs_out[sign_es] = {
            "prediction": prediction_es,
            "prediction_date": date_label,
            "sun_sign": sign_es,
            "_source_sign_en": sign_en,  # útil para debug; si no lo quieres, lo quitamos
        }

    return {
        "_cached": False,
        "date_key": date_key,
        "signs": signs_out,
    }


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
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=True)

