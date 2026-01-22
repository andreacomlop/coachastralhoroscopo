import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

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
HORO_TZ = os.getenv("HORO_TZ", "").strip()  # opcional
ASTRO_BASE = "https://json.astrologyapi.com/v1"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

# Tus signos en español (los que expones en el JSON)
SIGNS_ES = [
    "aries", "tauro", "géminis", "cáncer", "leo", "virgo",
    "libra", "escorpio", "sagitario", "capricornio", "acuario", "piscis",
]

# Mapeo al nombre que pide AstrologyAPI
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

# Cache en memoria por día
CACHE_DETAILED = {}  # { "YYYY-MM-DD": {...} }

app = Flask(__name__)
CORS(app)


# ---------------------------
# Helpers
# ---------------------------
def today_key_and_label():
    now = datetime.now(ZoneInfo(TZ_NAME))
    return now.strftime("%Y-%m-%d"), now.strftime("%d-%m-%Y")


def timezone_offset_hours() -> float:
    now = datetime.now(ZoneInfo(TZ_NAME))
    offset = now.utcoffset()
    return offset.total_seconds() / 3600 if offset else 0.0


def get_tz_for_astrology() -> float:
    if HORO_TZ:
        try:
            return float(HORO_TZ.replace(",", "."))
        except ValueError:
            pass
    return timezone_offset_hours()


# ---------------------------
# AstrologyAPI (DETALLADO)
# ---------------------------
def astrology_prediction_daily(sign_en: str, tz_hours: float) -> dict:
    """
    Llama a:
    POST https://json.astrologyapi.com/v1/sun_sign_prediction/daily/{zodiacName}
    """
    if not ASTRO_USER_ID or not ASTRO_API_KEY:
        raise RuntimeError("Faltan ASTRO_USER_ID o ASTRO_API_KEY en Render.")

    url = f"{ASTRO_BASE}/sun_sign_prediction/daily/{sign_en}"
    payload = {"timezone": tz_hours}

    r = requests.post(url, auth=(ASTRO_USER_ID, ASTRO_API_KEY), json=payload, timeout=20)
    if r.status_code >= 400:
        raise RuntimeError(f"AstrologyAPI error {r.status_code}: {r.text[:300]}")

    return r.json()


def extract_sections_en(api_json: dict) -> dict:
    """
    Normaliza el formato venga como:
    A) {"personal_life": "...", "profession": "...", ...}
    B) {"prediction": {"personal_life": "...", ...}, ...}
    """
    keys = ["personal_life", "profession", "health", "emotions", "travel", "luck"]

    # Caso A: claves directas
    out_a = {
        k: api_json.get(k, "").strip()
        for k in keys
        if isinstance(api_json.get(k), str) and api_json.get(k).strip()
    }
    if out_a:
        return out_a

    # Caso B: dentro de prediction
    pred = api_json.get("prediction")
    if isinstance(pred, dict):
        out_b = {
            k: pred.get(k, "").strip()
            for k in keys
            if isinstance(pred.get(k), str) and pred.get(k).strip()
        }
        if out_b:
            return out_b

    # fallback
    return {"raw": json.dumps(api_json, ensure_ascii=False)}


# ---------------------------
# OpenAI – Traducción (cercana) SIN inventar
# ---------------------------
def translate_sections_to_es(sections_en: dict) -> dict:
    """
    Traduce el JSON de secciones a español de España, manteniendo las mismas claves.
    Si no hay OPENAI_API_KEY, devuelve el inglés tal cual.
    """
    if not sections_en:
        return {}

    # Si vino fallback raw, no intentamos “reconstruir”, lo devolvemos tal cual.
    if "raw" in sections_en:
        return sections_en

    if not OPENAI_API_KEY:
        return sections_en

    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = (
        "Traduce al español de España el siguiente JSON de horóscopo.\n\n"
        "ESTILO:\n"
        "- Español natural, cercano y actual.\n"
        "- Que suene a horóscopo bien escrito, no a traducción literal.\n"
        "- Frases fluidas, humanas y fáciles de leer.\n"
        "- Tono cercano pero adulto.\n\n"
        "REGLAS:\n"
        "- Mantén EXACTAMENTE las mismas claves del JSON.\n"
        "- No añadas información nueva.\n"
        "- No elimines contenido.\n"
        "- No cambies el significado.\n"
        "- Devuelve SOLO un JSON válido, sin texto extra.\n\n"
        f"JSON (en inglés):\n{json.dumps(sections_en, ensure_ascii=False)}\n"
    )

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": "Eres un traductor/redactor editorial preciso. Devuelves JSON válido y nada más."
            },
            {"role": "user", "content": prompt},
        ],
    )

    content = (resp.choices[0].message.content or "").strip()

    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            # devolvemos exactamente las mismas claves
            return {k: str(parsed.get(k, "")).strip() for k in sections_en.keys()}
    except Exception:
        pass

    # Fallback si el modelo no devolvió JSON perfecto
    return sections_en


# ---------------------------
# Build detailed
# ---------------------------
def build_one_sign_detailed(sign_es: str, tz_hours: float, date_label: str) -> tuple[str, dict]:
    sign_en = SIGNS_EN[sign_es]

    api_json = astrology_prediction_daily(sign_en, tz_hours)
    sections_en = extract_sections_en(api_json)
    sections_es = translate_sections_to_es(sections_en)

    return sign_es, {
        "sun_sign": sign_es,
        "prediction_date": date_label,
        "sections": sections_es
    }


def build_detailed_today(date_key: str, date_label: str) -> dict:
    tz_hours = get_tz_for_astrology()
    signs_out = {}

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(build_one_sign_detailed, s, tz_hours, date_label) for s in SIGNS_ES]
        for f in as_completed(futures):
            sign_es, data = f.result()
            signs_out[sign_es] = data

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
        "<h1>Coach Astral - API Horóscopo Detallado</h1>"
        "<p>Endpoint:</p>"
        "<ul>"
        "<li><code>/api/horoscope/detailed/today</code></li>"
        "</ul>",
        200,
        {"Content-Type": "text/html; charset=utf-8"},
    )


@app.get("/api/horoscope/detailed/today")
def api_detailed_today():
    # Tu frontend manda ?t=... cuando pulsas "Actualizar".
    force = (request.args.get("force", "0").strip() == "1") or ("t" in request.args)

    date_key, date_label = today_key_and_label()

    if (not force) and (date_key in CACHE_DETAILED):
        cached = dict(CACHE_DETAILED[date_key])
        cached["_cached"] = True
        return jsonify(cached)

    try:
        result = build_detailed_today(date_key, date_label)
        CACHE_DETAILED[date_key] = result
        return jsonify(result)

    except RuntimeError as e:
        msg = str(e)
        if "TRIAL_REQUEST_LIMIT_EXCEEDED" in msg:
            return jsonify({
                "_cached": False,
                "date_key": date_key,
                "error": "ASTRO_TRIAL_LIMIT",
                "message": "Hoy no se ha podido generar el horóscopo completo. Vuelve mañana ✨",
            }), 429

        return jsonify({
            "_cached": False,
            "date_key": date_key,
            "error": "SERVER_ERROR",
            "message": msg[:300],
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=True)

