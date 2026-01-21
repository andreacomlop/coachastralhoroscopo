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
    for k, v in sections.items():
        label = k.replace("_", " ").strip().capitalize()
        parts.append(f"{label}: {str(v).strip()}")
    return "\n\n".join(parts).strip()


# ---------------------------
# OpenAI translate (solo 1 vez por signo)
# ---------------------------
def translate_es_editorial(text: str, min_chars: int = 900, max_chars: int = 1300) -> str:
    if not text:
        return ""
    if not OPENAI_API_KEY:
        return text

    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = (
        "Reescribe el siguiente horóscopo diario en español de España con estilo editorial.\n\n"
        "Objetivo:\n"
        "- Texto natural, fluido y agradable de leer.\n"
        "- Mantener el significado y las ideas clave del original.\n"
        "- Quitar repeticiones y mejorar ritmo.\n"
        "- Dejar sensación de profundidad (sin prometer nada ni mencionar pagos).\n\n"
        "Reglas estrictas:\n"
        "- No inventes información nueva.\n"
        "- No elimines ideas importantes.\n"
        "- No uses etiquetas o títulos tipo 'General', 'Trabajo', 'Viajes'.\n"
        "- Evita astrología técnica explícita (no menciones aspectos/planetas de forma técnica).\n"
        f"- Longitud final entre {min_chars} y {max_chars} caracteres (aprox.).\n"
        "- Devuelve SOLO el texto final.\n\n"
        f"TEXTO ORIGINAL:\n{text}"
    )

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.6,
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un redactor profesional de horóscopos diarios. "
                    "Escribes en español de España, con estilo cuidado, claro y atractivo."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    return (resp.choices[0].message.content or "").strip()



# ---------------------------
# Build consolidated (rápido)
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
        # si luego quieres, podemos reconstruir secciones en ES, pero esto ya va rápido y limpio
    }


def build_consolidated_today(date_key: str, date_label: str) -> dict:
    tz_hours = get_tz_for_astrology()
    signs_out = {}

    # 12 signos en paralelo (reduce muchísimo el tiempo total)
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

    result = build_consolidated_today(date_key, date_label)
    CACHE[date_key] = result
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=True)

