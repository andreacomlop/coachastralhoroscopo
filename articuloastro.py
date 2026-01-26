import os
import json
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

# Endpoints confirmados por tus capturas
ASTRO_MOON_PHASE_URL = os.getenv("ASTRO_MOON_PHASE_URL", f"{ASTRO_BASE}/moon_phase_report").strip()
ASTRO_PLANETS_URL = os.getenv("ASTRO_PLANETS_URL", f"{ASTRO_BASE}/planets/tropical").strip()

# Ubicación “genérica” (Madrid por defecto) para efemérides colectivas
DEFAULT_LAT = float(os.getenv("ASTRO_LAT", "40.4168"))
DEFAULT_LON = float(os.getenv("ASTRO_LON", "-3.7038"))
HOUSE_TYPE = os.getenv("ASTRO_HOUSE_TYPE", "placidus").strip()

# Idiomas
# - moon_phase_report: en (según docs)
# - planets/tropical: es disponible
PLANETS_LANG = os.getenv("ASTRO_PLANETS_LANG", "es").strip().lower()

client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)
CORS(app)


# ---------------------------
# Helpers
# ---------------------------
def cache_path_for_date(fecha_iso: str) -> str:
    return f"/tmp/cache_articuloastro_{fecha_iso}.json"


def _tz_offset_hours(dt: datetime) -> float:
    """UTC offset en horas (con decimales). Ej: +2.0 en verano."""
    off = dt.utcoffset()
    if off is None:
        return 0.0
    return off.total_seconds() / 3600.0


def _astro_payload(dt: datetime, lat: float, lon: float) -> dict:
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


def _find_body(bodies: list, name: str) -> dict | None:
    name_l = name.lower()
    for b in bodies:
        if isinstance(b, dict) and str(b.get("name", "")).lower() == name_l:
            return b
    return None


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "y", "sí", "si")
    return bool(v)


def get_daily_facts(now: datetime, lat: float, lon: float) -> dict:
    """
    2 llamadas a AstrologyAPI:
    - moon_phase_report (en)
    - planets/tropical (es)
    Ambas a las 12:00 para representar “el día”.
    """
    midday = now.replace(hour=12, minute=0, second=0, microsecond=0)
    payload = _astro_payload(midday, lat, lon)

    # 1) Moon phase report (EN)
    moon = _astro_post(ASTRO_MOON_PHASE_URL, payload, accept_language="en", timeout=20)
    # Esperamos keys tipo: considered_date, moon_phase, significance, report
    moon_phase = moon.get("moon_phase")
    moon_significance = moon.get("significance")
    moon_report = moon.get("report")

    # 2) Planets tropical (ES)
    planets = _astro_post(ASTRO_PLANETS_URL, payload, accept_language=PLANETS_LANG, timeout=20)
    if not isinstance(planets, list):
        raise RuntimeError("Respuesta inesperada en planets/tropical: se esperaba una lista.")

    slow_names = ["Saturn", "Uranus", "Neptune", "Pluto"]
    slow = []
    for nm in slow_names:
        obj = _find_body(planets, nm)
        if obj:
            slow.append({
                "name": nm,
                "sign": obj.get("sign"),
                "isRetro": _to_bool(obj.get("isRetro", False)),
            })

    # Luna por signo (solo como dato duro, no “ahora mismo”)
    moon_obj = _find_body(planets, "Moon")
    moon_sign = moon_obj.get("sign") if moon_obj else None

    return {
        "date": now.date().isoformat(),
        "location": {"lat": lat, "lon": lon, "tz": TZ_NAME},
        "moon_phase_report": {
            "moon_phase": moon_phase,
            "significance": moon_significance,
            "report": moon_report,
        },
        "planets_snapshot": {
            "moon_sign": moon_sign,
            "slow_planets": slow,
        },
    }


def build_daily_article_prompt(fecha_iso: str, facts: dict) -> str:
    moon_phase = facts["moon_phase_report"].get("moon_phase") or "(sin dato)"
    moon_sig = facts["moon_phase_report"].get("significance") or ""
    moon_rep = facts["moon_phase_report"].get("report") or ""

    moon_sign = facts["planets_snapshot"].get("moon_sign")
    slow = facts["planets_snapshot"].get("slow_planets", [])

    slow_lines = []
    for p in slow:
        retro = "retrógrado" if p.get("isRetro") else "directo"
        slow_lines.append(f"- {p['name']}: {p.get('sign')} ({retro})")
    slow_txt = "\n".join(slow_lines) if slow_lines else "- (sin datos)"

    parts = [
        f"Hoy es {fecha_iso}.",
        "",
        "DATOS REALES (no inventes nada fuera de esto):",
        "",
        "1) MOON PHASE REPORT (en inglés, úsalo como base pero NO lo copies literal; reescríbelo en español):",
        f"- Fase lunar: {moon_phase}",
        f"- Significance: {moon_sig}".strip(),
        f"- Report: {moon_rep}".strip(),
        "",
        "2) SNAPSHOT PLANETARIO (datos duros):",
        f"- Signo de la Luna (dato del snapshot): {moon_sign}",
        "Planetas lentos (clima de fondo):",
        slow_txt,
        "",
        "INSTRUCCIONES DE REDACCIÓN:",
        "- Escribe EN ESPAÑOL.",
        "- No uses 'ahora mismo' ni referencias a una hora concreta. Habla en términos de 'durante el día de hoy' / 'a lo largo del día'.",
        "- La fase lunar debe ser el corazón del artículo (clima emocional colectivo).",
        "- Integra Saturno/Urano/Neptuno/Plutón como contexto de fondo (2–3 frases), sin dramatismo y sin predicciones absolutas.",
        "- Si algún planeta lento está retrógrado, menciónalo como matiz colectivo (revisión/ajuste), no como sentencia.",
        "- Nada de horóscopo por signos. Nada de predicciones personales. Nada de destino.",
        "",
        "Extensión: 150–180 palabras.",
        "",
        "Estructura obligatoria:",
        "1. Título atractivo y cercano (sin tecnicismos).",
        "2. Introducción breve: la astrología como 'clima emocional' colectivo.",
        "3. Qué tono trae el día según la fase lunar (explicado fácil).",
        "4. Cómo puede notarse en general (emociones, mente, energía, relaciones) sin predicciones.",
        "5. Clima de fondo: planetas lentos (Saturno/Urano/Neptuno/Plutón) en 2–3 frases.",
        "6. Cierre reflexivo que invite a observarse (sin consejos absolutos).",
        "",
        "Tono y estilo:",
        "- Cercano, humano y educativo.",
        "- Humor suave e ironía inteligente (elegante).",
        "- Emojis sutiles (🌙✨🌀☕️💭) sin exagerar.",
        "- Párrafos cortos.",
    ]
    return "\n".join([p for p in parts if p is not None])


# ---------------------------
# Routes
# ---------------------------
@app.get("/")
def home():
    return jsonify({
        "ok": True,
        "service": "articuloastro",
        "endpoints": ["/health", "/daily-astrology-article"]
    })


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.get("/daily-astrology-article")
def daily_astrology_article():
    tz = ZoneInfo(TZ_NAME)
    now = datetime.now(tz)
    today_iso = now.date().isoformat()
    cache_file = cache_path_for_date(today_iso)

    # 1) Cache del día
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))

    # 2) Validación keys
    if not OPENAI_API_KEY:
        return jsonify({"date": today_iso, "error": "Falta OPENAI_API_KEY en Environment."}), 500
    if not (ASTRO_USER_ID and ASTRO_API_KEY):
        return jsonify({"date": today_iso, "error": "Falta ASTRO_USER_ID o ASTRO_API_KEY en Environment."}), 500

    # 3) Facts desde AstrologyAPI (moon report + planets)
    try:
        facts = get_daily_facts(now, DEFAULT_LAT, DEFAULT_LON)
    except requests.HTTPError as e:
        return jsonify({"date": today_iso, "error": f"AstrologyAPI HTTPError: {str(e)}"}), 502
    except requests.RequestException:
        return jsonify({"date": today_iso, "error": "No se puede conectar con AstrologyAPI ahora mismo."}), 503
    except Exception as e:
        return jsonify({"date": today_iso, "error": f"Error efemérides: {type(e).__name__}"}), 500

    # 4) Prompt y OpenAI
    prompt = build_daily_article_prompt(today_iso, facts)

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Eres un divulgador experto en astrología, claro, didáctico y con humor sutil."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            timeout=45,
        )
        article_text = resp.choices[0].message.content.strip()

    except AuthenticationError:
        return jsonify({"date": today_iso, "error": "OPENAI_API_KEY inválida o sin permisos."}), 500
    except (APIConnectionError, APITimeoutError):
        return jsonify({"date": today_iso, "error": "No se puede conectar con OpenAI ahora mismo. Prueba en unos minutos."}), 503
    except Exception as e:
        return jsonify({"date": today_iso, "error": f"Error inesperado: {type(e).__name__}"}), 500

    data = {
        "date": today_iso,
        "facts": facts,     # útil para debug / confianza
        "article": article_text
    }

    # 5) Guardar cache (silencioso si falla)
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return jsonify(data)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
