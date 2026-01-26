# LaLunaDeHoy.py
# Backend Flask para "La Luna de Hoy" (SIN cache)
# - Obtiene moon_phase_report (EN) + lunar_metrics (EN) de AstrologyAPI
# - Traduce el report al español con OpenAI (sin inventar ni añadir contenido)
# - Devuelve JSON listo para GoodBarber: { date, luna_de_hoy, metrics }

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from flask import Flask, jsonify
from flask_cors import CORS
from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, AuthenticationError


# ---------------------------
# Config
# ---------------------------
TZ_NAME = os.getenv("TZ", "Europe/Madrid").strip()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

# AstrologyAPI (json.astrologyapi.com)
ASTRO_USER_ID = os.getenv("ASTRO_USER_ID", "").strip()
ASTRO_API_KEY = os.getenv("ASTRO_API_KEY", "").strip()
ASTRO_BASE = os.getenv("ASTRO_BASE", "https://json.astrologyapi.com/v1").strip()

# Endpoints
ASTRO_MOON_PHASE_URL = os.getenv("ASTRO_MOON_PHASE_URL", f"{ASTRO_BASE}/moon_phase_report").strip()
ASTRO_LUNAR_METRICS_URL = os.getenv("ASTRO_LUNAR_METRICS_URL", f"{ASTRO_BASE}/lunar_metrics").strip()

# Ubicación genérica (no es crítica para fase, pero mantenemos consistencia)
DEFAULT_LAT = float(os.getenv("ASTRO_LAT", "40.4168"))
DEFAULT_LON = float(os.getenv("ASTRO_LON", "-3.7038"))
HOUSE_TYPE = os.getenv("ASTRO_HOUSE_TYPE", "placidus").strip()

client = OpenAI(api_key=OPENAI_API_KEY)

# ✅ Render necesita una variable llamada "app"
app = Flask(__name__)
CORS(app)


# ---------------------------
# Helpers
# ---------------------------
def _tz_offset_hours(dt: datetime) -> float:
    """UTC offset en horas (con decimales)."""
    off = dt.utcoffset()
    if off is None:
        return 0.0
    return off.total_seconds() / 3600.0


def _astro_payload(dt: datetime, lat: float, lon: float) -> dict:
    """Payload estándar para AstrologyAPI."""
    return {
        "day": dt.day,
        "month": dt.month,
        "year": dt.year,
        "hour": dt.hour,
        "min": dt.minute,
        "lat": float(lat),
        "lon": float(lon),
        "tzone": float(_tz_offset_hours(dt)),
        "house_type": HOUSE_TYPE,
    }


def _astro_post(url: str, payload: dict, accept_language: str | None = None, timeout: int = 20):
    """POST a AstrologyAPI con Basic Auth."""
    if not (ASTRO_USER_ID and ASTRO_API_KEY):
        raise RuntimeError("Falta ASTRO_USER_ID o ASTRO_API_KEY en Environment.")

    headers = {}
    if accept_language:
        headers["Accept-Language"] = accept_language

    r = requests.post(
        url,
        auth=(ASTRO_USER_ID, ASTRO_API_KEY),
        json=payload,
        headers=headers,
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def get_moon_today_facts(now: datetime, lat: float, lon: float) -> dict:
    """
    2 llamadas a AstrologyAPI (a las 12:00):
    - moon_phase_report (EN) -> texto
    - lunar_metrics (EN)     -> datos
    """
    midday = now.replace(hour=12, minute=0, second=0, microsecond=0)
    payload = _astro_payload(midday, lat, lon)

    moon_report = _astro_post(ASTRO_MOON_PHASE_URL, payload, accept_language="en", timeout=20)
    lunar_metrics = _astro_post(ASTRO_LUNAR_METRICS_URL, payload, accept_language="en", timeout=20)

    return {
        "moon_phase_report": {
            "considered_date": moon_report.get("considered_date"),
            "moon_phase": moon_report.get("moon_phase"),
            "significance_en": moon_report.get("significance"),
            "report_en": moon_report.get("report"),
        },
        "lunar_metrics": lunar_metrics,
    }


def build_translation_prompt(facts: dict) -> str:
    """
    Prompt de traducción: fiel y natural, sin añadir contenido.
    """
    phase = facts["moon_phase_report"].get("moon_phase") or ""
    sig = facts["moon_phase_report"].get("significance_en") or ""
    rep = facts["moon_phase_report"].get("report_en") or ""

    return "\n".join([
        "A continuación tienes un texto REAL en inglés procedente de una API.",
        "Tu tarea es ÚNICAMENTE traducirlo al español de forma fiel y natural.",
        "",
        "REGLAS:",
        "- Traduce fielmente el contenido.",
        "- No añadas información nueva.",
        "- No elimines ideas.",
        "- No resumas.",
        "- No interpretes.",
        "- No menciones que es una traducción.",
        "",
        "FORMATO DE SALIDA:",
        "- Título breve en español.",
        "- Texto traducido (sin añadir nada).",
        "",
        "CONTENIDO A TRADUCIR:",
        f"Moon phase: {phase}",
        "",
        "Significance:",
        sig,
        "",
        "Report:",
        rep,
    ])


# ---------------------------
# Routes
# ---------------------------
@app.get("/")
def home():
    return jsonify({
        "ok": True,
        "service": "LaLunaDeHoy",
        "endpoints": ["/health", "/moon-today"]
    })


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.get("/moon-today")
def moon_today():
    tz = ZoneInfo(TZ_NAME)
    now = datetime.now(tz)
    today_iso = now.date().isoformat()

    # Validación keys
    if not OPENAI_API_KEY:
        return jsonify({"date": today_iso, "error": "Falta OPENAI_API_KEY en Environment."}), 500
    if not (ASTRO_USER_ID and ASTRO_API_KEY):
        return jsonify({"date": today_iso, "error": "Falta ASTRO_USER_ID o ASTRO_API_KEY en Environment."}), 500

    # 1) Facts (AstrologyAPI)
    try:
        facts = get_moon_today_facts(now, DEFAULT_LAT, DEFAULT_LON)
    except requests.HTTPError as e:
        return jsonify({"date": today_iso, "error": f"AstrologyAPI HTTPError: {str(e)}"}), 502
    except requests.RequestException:
        return jsonify({"date": today_iso, "error": "No se puede conectar con AstrologyAPI ahora mismo."}), 503
    except Exception as e:
        return jsonify({"date": today_iso, "error": f"Error AstrologyAPI: {type(e).__name__}"}), 500

    # 2) OpenAI: traducción
    prompt = build_translation_prompt(facts)

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Eres un traductor profesional EN->ES. Traduces fielmente y con español natural, sin añadir contenido."
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            timeout=45,
        )
        luna_de_hoy_es = resp.choices[0].message.content.strip()

    except AuthenticationError:
        return jsonify({"date": today_iso, "error": "OPENAI_API_KEY inválida o sin permisos."}), 500
    except (APIConnectionError, APITimeoutError):
        return jsonify({"date": today_iso, "error": "No se puede conectar con OpenAI ahora mismo. Prueba en unos minutos."}), 503
    except Exception as e:
        return jsonify({"date": today_iso, "error": f"Error OpenAI: {type(e).__name__}"}), 500

    return jsonify({
        "date": today_iso,
        "luna_de_hoy": luna_de_hoy_es,
        "metrics": facts.get("lunar_metrics", {}),
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
