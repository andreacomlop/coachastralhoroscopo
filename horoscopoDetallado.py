import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS


# ---------------------------
# Config (Render env vars)
# ---------------------------
TZ_NAME = os.getenv("TZ", "Europe/Madrid").strip()

ASTRO_USER_ID = os.getenv("ASTRO_USER_ID", "").strip()
ASTRO_API_KEY = os.getenv("ASTRO_API_KEY", "").strip()
HORO_TZ = os.getenv("HORO_TZ", "").strip()  # opcional
ASTRO_BASE = "https://json.astrologyapi.com/v1"

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
    Endpoint daily detailed (6 secciones):
    {
      "personal_life": "...",
      "profession": "...",
      "health": "...",
      "emotions": "...",
      "travel": "...",
      "luck": "..."
    }
    """
    if not ASTRO_USER_ID or not ASTRO_API_KEY:
        raise RuntimeError("Faltan ASTRO_USER_ID o ASTRO_API_KEY en Render.")

    url = f"{ASTRO_BASE}/sun_sign_prediction/daily/{sign_en}"
    payload = {"timezone": tz_hours}

    r = requests.post(url, auth=(ASTRO_USER_ID, ASTRO_API_KEY), json=payload, timeout=20)
    if r.status_code >= 400:
        raise RuntimeError(f"AstrologyAPI error {r.status_code}: {r.text[:300]}")

    return r.json()


def extract_detailed_sections(api_json: dict) -> dict:
    """
    Nos quedamos SOLO con las secciones del daily detailed.
    Si por lo que sea no vienen, devolvemos raw para depurar.
    """
    keys = ["personal_life", "profession", "health", "emotions", "travel", "luck"]
    out = {k: api_json.get(k, "").strip() for k in keys if isinstance(api_json.get(k), str) and api_json.get(k).strip()}
    if out:
        return out
    return {"raw": api_json}


# ---------------------------
# Build detailed
# ---------------------------
def build_one_sign_detailed(sign_es: str, tz_hours: float, date_label: str) -> tuple[str, dict]:
    sign_en = SIGNS_EN[sign_es]
    api_json = astrology_prediction_daily(sign_en, tz_hours)
    sections = extract_detailed_sections(api_json)

    return sign_es, {
        "prediction_date": date_label,
        "sun_sign": sign_es,
        "sections": sections,  # <-- aquí van las 6 secciones
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
        "<p>Endpoints:</p>"
        "<ul>"
        "<li><code>/api/horoscope/detailed/today</code></li>"
        "</ul>",
        200,
        {"Content-Type": "text/html; charset=utf-8"},
    )


@app.get("/api/horoscope/detailed/today")
def api_detailed_today():
    # Si el frontend manda ?t=... cuando pulsas "Actualizar", forzamos refresco.
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
