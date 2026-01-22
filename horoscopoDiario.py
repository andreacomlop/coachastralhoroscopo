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
        "Reescribe el siguiente texto en español de España.\n\n"
        "Objetivo:\n"
        "- Que suene cercano, elegante y fácil de leer.\n"
        "- Que invite a seguir leyendo y deje sensación de que hay más profundidad.\n\n"
        "Reglas importantes:\n"
        "- NO inventes información nueva.\n"
        "- NO menciones astrología técnica.\n"
        "- NO uses encabezados como 'General', 'Trabajo', etc.\n"
        "- NO traduzcas de forma literal.\n"
        "- Integra todo en un solo texto fluido.\n"
        "- Extensión media (aprox. 90–140 palabras).\n"
        "- Tono reflexivo, inspirador y práctico.\n"
        "- Habla en segunda persona.\n"
        "- No menciones servicios premium ni llamadas a la acción.\n\n"
        f"TEXTO BASE:\n{text}"
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
   result = build_daily_horoscope_today(date_key, date_label)
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

# ------------------------------------------------------
# Nuevas funciones y rutas añadidas para horóscopo diario y consejo/verdad
# ------------------------------------------------------

def astrology_daily_prediction(sign_en: str, tz_hours: float) -> dict:
    """
    Llama al endpoint sun_sign_prediction/daily/{sign_en} de AstrologyAPI.
    Devuelve el JSON con las distintas categorías de predicción.
    """
    if not ASTRO_USER_ID or not ASTRO_API_KEY:
        raise RuntimeError("Faltan ASTRO_USER_ID o ASTRO_API_KEY en Render.")
    url = f"{ASTRO_BASE}/sun_sign_prediction/daily/{sign_en}"
    payload = {"timezone": tz_hours}
    r = requests.post(url, auth=(ASTRO_USER_ID, ASTRO_API_KEY), json=payload, timeout=20)
    if r.status_code >= 400:
        raise RuntimeError(f"AstrologyAPI error {r.status_code}: {r.text[:300]}")
    return r.json()

def join_daily_categories(api_json: dict) -> str:
    """
    Une las categorías devueltas por sun_sign_prediction/daily en un único texto.
    Se concatenan las claves en un orden fijo para componer un resumen legible.
    """
    order = ["personal_life", "profession", "health", "emotions", "travel", "luck"]
    parts = []
    for key in order:
        if key in api_json and api_json[key].strip():
            parts.append(api_json[key].strip())
    return "\n\n".join(parts).strip()

def extract_advice_and_truth(prediction: str) -> tuple[str, str]:
    """
    Divide el texto de predicción general en oraciones y toma las dos primeras.
    La primera oración se usa como 'Consejo del día' y la segunda como 'Verdad incómoda'.
    Si solo hay una oración, se usa para ambos campos.
    """
    import re
    sentences = re.split(r"\.\s+", prediction.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return "", ""
    if len(sentences) == 1:
        return sentences[0], sentences[0]
    return sentences[0], sentences[1]

def build_one_sign_daily(sign_es: str, tz_hours: float, date_label: str) -> tuple[str, dict]:
    """
    Construye la predicción diaria para un signo llamando a astrology_daily_prediction,
    combinando las categorías y traduciendo al español.
    Devuelve una tupla con el nombre del signo en español y los datos asociados.
    """
    sign_en = SIGNS_EN[sign_es]
    api_json = astrology_daily_prediction(sign_en, tz_hours)
    text_en = join_daily_categories(api_json)
    text_es = translate_es_strict(text_en)
    return sign_es, {
        "prediction": text_es,
        "prediction_date": date_label,
        "sun_sign": sign_es,
    }

def build_daily_horoscope_today(date_key: str, date_label: str) -> dict:
    """
    Construye el horóscopo diario para todos los signos utilizando
    sun_sign_prediction/daily y traduce el texto al español.
    """
    tz_hours = get_tz_for_astrology()
    signs_out = {}
    from concurrent.futures import as_completed  # ensure as_completed is available
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(build_one_sign_daily, s, tz_hours, date_label) for s in SIGNS_ES]
        for f in as_completed(futures):
            sign_es, data = f.result()
            signs_out[sign_es] = data
    return {
        "_cached": False,
        "date_key": date_key,
        "signs": signs_out,
    }

def build_one_sign_advice_truth(sign_es: str, tz_hours: float, date_label: str) -> tuple[str, dict]:
    """
    Para un signo dado llama al endpoint sign_consolidated/daily, extrae el texto
    general, lo trocea en oraciones y toma las dos primeras como consejo y verdad.
    Traduce ambos al español.
    """
    sign_en = SIGNS_EN[sign_es]
    api_json = astrology_consolidated_daily(sign_en, tz_hours)
    sections_en = extract_prediction_sections(api_json)
    text_en = join_sections(sections_en)
    advice_en, truth_en = extract_advice_and_truth(text_en)
    consejo_es = translate_es_strict(advice_en)
    verdad_es = translate_es_strict(truth_en)
    return sign_es, {
        "consejo": consejo_es,
        "verdad": verdad_es,
        "prediction_date": date_label,
        "sun_sign": sign_es,
    }

def build_advice_truth_today(date_key: str, date_label: str) -> dict:
    """
    Construye la sección de consejo y verdad incómoda para todos los signos.
    Utiliza el endpoint consolidado existente, extrae las dos primeras frases
    y las traduce al español.
    """
    tz_hours = get_tz_for_astrology()
    signs_out = {}
    from concurrent.futures import as_completed
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(build_one_sign_advice_truth, s, tz_hours, date_label) for s in SIGNS_ES]
        for f in as_completed(futures):
            sign_es, data = f.result()
            signs_out[sign_es] = data
    return {
        "_cached": False,
        "date_key": date_key,
        "signs": signs_out,
    }

@app.get("/api/horoscope/daily")
def api_daily_horoscope():
    """
    Devuelve el horóscopo diario basado en sun_sign_prediction/daily.
    Reutiliza el sistema de caché diario (clave: 'daily:{date_key}').
    """
    force = request.args.get("force", "0").strip() == "1"
    date_key, date_label = today_key_and_label()
    cache_key = f"daily:{date_key}"
    if (not force) and (cache_key in CACHE):
        cached = dict(CACHE[cache_key])
        cached["_cached"] = True
        return jsonify(cached)
    try:
        result = build_daily_horoscope_today(date_key, date_label)
        CACHE[cache_key] = result
        return jsonify(result)
    except RuntimeError as e:
        msg = str(e)
        return jsonify({
            "_cached": False,
            "date_key": date_key,
            "error": "API_ERROR",
            "message": msg,
        }), 500

@app.get("/api/horoscope/advice")
def api_advice_truth():
    """
    Devuelve el consejo y la verdad incómoda del día para todos los signos.
    Usa el endpoint consolidado y aplica la misma caché diaria (clave: 'advice:{date_key}').
    """
    force = request.args.get("force", "0").strip() == "1"
    date_key, date_label = today_key_and_label()
    cache_key = f"advice:{date_key}"
    if (not force) and (cache_key in CACHE):
        cached = dict(CACHE[cache_key])
        cached["_cached"] = True
        return jsonify(cached)
    try:
        result = build_advice_truth_today(date_key, date_label)
        CACHE[cache_key] = result
        return jsonify(result)
    except RuntimeError as e:
        msg = str(e)
        return jsonify({
            "_cached": False,
            "date_key": date_key,
            "error": "API_ERROR",
            "message": msg,
        }), 500
