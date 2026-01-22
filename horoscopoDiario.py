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

CACHE = {}  # { "YYYY-MM-DD": {...} }

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
# AstrologyAPI
# ---------------------------
def astrology_consolidated_daily(sign_en: str, tz_hours: float) -> dict:
    if not ASTRO_USER_ID or not ASTRO_API_KEY:
        raise RuntimeError("Faltan ASTRO_USER_ID o ASTRO_API_KEY en Render.")

    url = f"{ASTRO_BASE}/sun_sign_consolidated/daily/{sign_en}"
    payload = {"timezone": tz_hours}

    r = requests.post(url, auth=(ASTRO_USER_ID, ASTRO_API_KEY), json=payload, timeout=20)
    if r.status_code >= 400:
        raise RuntimeError(f"AstrologyAPI error {r.status_code}: {r.text[:300]}")

    return r.json()


def extract_prediction_sections(api_json: dict) -> dict:
    pred = api_json.get("prediction")
    if isinstance(pred, dict) and pred:
        return pred
    if isinstance(pred, str) and pred.strip():
        return {"general": pred.strip()}
    return {"raw": json.dumps(api_json, ensure_ascii=False)}


def join_sections(sections: dict) -> str:
    parts = []
    for _, v in sections.items():
        parts.append(str(v).strip())
    return "\n\n".join(parts).strip()


# ---------------------------
# OpenAI – adaptación editorial (NO literal)
# ---------------------------
def translate_es_strict(text: str) -> str:
    if not text:
        return ""
    if not OPENAI_API_KEY:
        return text

    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = (
    "A partir del texto de horóscopo que te doy abajo, genera DOS BLOQUES NUEVOS.\n\n"

    "CONTEXTO:\n"
    "El texto es un horóscopo diario por signo. No debes añadir información nueva, "
    "solo interpretar y extraer lo que ya está implícito en él.\n\n"

    "REGLAS GENERALES:\n"
    "- Escribe en español de España.\n"
    "- No inventes hechos, situaciones ni predicciones nuevas.\n"
    "- No menciones astrología técnica ni términos astrológicos.\n"
    "- No menciones servicios premium ni llamadas a la acción.\n"
    "- Usa segunda persona.\n"
    "- Tono cercano, adulto e inteligente.\n\n"

    "BLOQUE 1 · CONSEJO DE TU COACH:\n"
    "- Un único párrafo corto (30-45 palabras).\n"
    "- Enfoque práctico y realista.\n"
    "- Aterriza el mensaje del horóscopo en comportamiento diario.\n"
    "- Debe sonar a alguien que te conoce y te orienta, no a autoayuda vacía. Pon consejos tangibles. Que te hagan reflexionar y actuar. \n\n"

    "BLOQUE 2 · VERDAD INCÓMODA DEL DÍA:\n"
    "- Una sola frase.\n"
    "- Directa, honesta e irónica. Que te queden ganas de compartirla en redes.\n"
    "- Que haga pensar y sonreír a la vez. E incluso puede hacerte emcionar. \n"
    "- Muy compartible (efecto viral), pero sin juzgar.\n"
    "- Debe encajar con el mensaje del horóscopo (debe extraerse del contexto del horóscopo, no inventar, solo interpretar en esta verdad incñomoda, puedes añadir algo de tu cosecha, pero que vaya alineado con el mensaje del horóscopo de dicho signo).\n\n"

    "FORMATO DE SALIDA (OBLIGATORIO):\n"
    "Consejo de tu coach:\n"
    "<texto>\n\n"
    "Verdad incómoda del día:\n"
    "<frase>\n\n"

    )

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.7,
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un redactor editorial experto en horóscopos diarios. "
                    "Transformas textos técnicos en lecturas atractivas y naturales."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    return (resp.choices[0].message.content or "").strip()


# ---------------------------
# Build consolidated
# ---------------------------
def build_one_sign(sign_es: str, tz_hours: float, date_label: str) -> tuple[str, dict]:
    sign_en = SIGNS_EN[sign_es]

    api_json = astrology_consolidated_daily(sign_en, tz_hours)
    sections_en = extract_prediction_sections(api_json)

    text_en = join_sections(sections_en)
    text_es = translate_es_strict(text_en)

    return sign_es, {
        "prediction": text_es,
        "prediction_date": date_label,
        "sun_sign": sign_es,
    }


def build_consolidated_today(date_key: str, date_label: str) -> dict:
    tz_hours = get_tz_for_astrology()
    signs_out = {}

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(build_one_sign, s, tz_hours, date_label) for s in SIGNS_ES]
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

    try:
        result = build_consolidated_today(date_key, date_label)
        CACHE[date_key] = result
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




