import os
import json
import datetime
from pathlib import Path

import requests
from flask import Flask, Response

# OpenAI (SDK nuevo)
from openai import OpenAI

app = Flask(__name__)
app.json.ensure_ascii = False  # por si usas jsonify en algún momento

# ---------------------------
# Config
# ---------------------------
CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)

ASTRO_USER_ID = os.getenv("ASTRO_USER_ID")
ASTRO_API_KEY = os.getenv("ASTRO_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not ASTRO_USER_ID or not ASTRO_API_KEY:
    # No rompemos aquí para que Render "live" arranque,
    # pero el endpoint dará error claro si faltan.
    pass

if not OPENAI_API_KEY:
    pass

# Ajusta esto si tu endpoint real de horóscopo externo es distinto
# (en local ya lo tenías funcionando).
ASTRO_BASE_URL = "https://astrologyapi.com/api/v1"  # <-- si usabas otro, cámbialo aquí

SIGNOS_EN = [
    "aries", "taurus", "gemini", "cancer", "leo", "virgo",
    "libra", "scorpio", "sagittarius", "capricorn", "aquarius", "pisces"
]

SIGNOS_ES = {
    "aries": "Aries",
    "taurus": "Tauro",
    "gemini": "Géminis",
    "cancer": "Cáncer",
    "leo": "Leo",
    "virgo": "Virgo",
    "libra": "Libra",
    "scorpio": "Escorpio",
    "sagittarius": "Sagitario",
    "capricorn": "Capricornio",
    "aquarius": "Acuario",
    "pisces": "Piscis",
}

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ---------------------------
# Helpers
# ---------------------------
def _today_key() -> str:
    # Cache por fecha en horario local del servidor (Render suele ir UTC)
    # Si quieres España sí o sí, lo mejor es fijar TZ en Render.
    return datetime.date.today().isoformat()


def _cache_path(date_key: str) -> Path:
    return CACHE_DIR / f"horoscope_{date_key}.json"


def _load_cache(date_key: str):
    p = _cache_path(date_key)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def _save_cache(date_key: str, data: dict):
    p = _cache_path(date_key)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _json_response(data: dict, status: int = 200) -> Response:
    return Response(
        json.dumps(data, ensure_ascii=False),
        status=status,
        mimetype="application/json; charset=utf-8"
    )


def _astro_auth_headers():
    # AstrologyAPI usa Basic Auth con user_id:api_key
    # Muchas implementaciones usan (user_id, api_key) como auth en requests
    return (ASTRO_USER_ID, ASTRO_API_KEY)


def fetch_12_from_astro() -> dict:
    """
    Devuelve un dict:
    {
      "aries": {"prediction": "...", "prediction_date": "...", "sun_sign":"aries"},
      ...
    }
    """
    if not ASTRO_USER_ID or not ASTRO_API_KEY:
        raise RuntimeError("Faltan ASTRO_USER_ID / ASTRO_API_KEY en variables de entorno")

    results = {}

    # Endpoint típico de astrologyapi para 'sun_sign_prediction/daily/<sign>'
    # Si tu endpoint era otro, dime cuál y lo ajusto.
    for sign in SIGNOS_EN:
        url = f"{ASTRO_BASE_URL}/sun_sign_prediction/daily/{sign}"
        r = requests.post(url, auth=_astro_auth_headers())
        if r.status_code != 200:
            raise RuntimeError(f"Error Astro API ({sign}): {r.status_code} {r.text[:200]}")

        payload = r.json()
        # Suele venir con "prediction" y "prediction_date"
        results[sign] = {
            "sun_sign": sign,
            "prediction_date": payload.get("prediction_date") or payload.get("date") or "",
            "prediction": payload.get("prediction") or "",
        }

    return results


def translate_all_12_with_openai(signs_en_payload: dict) -> dict:
    """
    Recibe el dict con 12 signos en inglés y devuelve dict con 12 signos en español.
    Devuelve:
    {
      "Acuario": {"signo":"Acuario","fecha":"...","texto":"..."},
      ...
    }
    """
    if client is None:
        raise RuntimeError("Falta OPENAI_API_KEY en variables de entorno")

    # Construimos un JSON compacto con lo que hay que traducir
    # (solo texto inglés).
    input_for_model = []
    for sign_en in SIGNOS_EN:
        item = signs_en_payload[sign_en]
        input_for_model.append({
            "sign_en": sign_en,
            "sign_es": SIGNOS_ES[sign_en],
            "prediction_date": item.get("prediction_date", ""),
            "text_en": item.get("prediction", ""),
        })

    system = (
        "Eres traductor profesional al español de España. "
        "Traduce con naturalidad, sin anglicismos, manteniendo el tono de horóscopo diario. "
        "No añadas información nueva. No cambies el significado. "
        "Devuelve SOLO JSON válido."
    )

    user = (
        "Traduce los 12 textos al español (España). "
        "Devuelve un JSON con esta forma EXACTA:\n"
        "{\n"
        '  "date_key": "YYYY-MM-DD",\n'
        '  "signs": {\n'
        '    "Acuario": {"signo":"Acuario","fecha":"...","texto":"..."}\n'
        "  }\n"
        "}\n\n"
        f"date_key = {_today_key()}\n"
        "Datos a traducir (JSON):\n"
        + json.dumps(input_for_model, ensure_ascii=False)
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    content = resp.choices[0].message.content.strip()

    # Intentamos parsear JSON tal cual
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Si el modelo envolvió con ```json ...```, lo limpiamos
        cleaned = content.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned)

    # Validación mínima
    if "signs" not in data or not isinstance(data["signs"], dict):
        raise RuntimeError("OpenAI no devolvió el formato esperado (falta 'signs')")

    return data


# ---------------------------
# Routes
# ---------------------------
@app.get("/")
def home():
    # mini página para comprobar rápido
    html = """
    <html><head><meta charset="utf-8"><title>CoachAstral API</title></head>
    <body style="font-family:Arial;max-width:800px;margin:40px auto;">
      <h2>API CoachAstral Horóscopo</h2>
      <p>Prueba aquí:</p>
      <ul>
        <li><a href="/api/horoscope/today">/api/horoscope/today</a></li>
      </ul>
    </body></html>
    """
    return Response(html, mimetype="text/html; charset=utf-8")


@app.get("/api/horoscope/today")
def api_today():
    date_key = _today_key()

    cached = _load_cache(date_key)
    if cached:
        cached["_cached"] = True
        return _json_response(cached)

    # 1) pedimos los 12 a la API externa (inglés)
    signs_en = fetch_12_from_astro()

    # 2) traducimos de una vez con OpenAI y devolvemos SOLO español
    translated = translate_all_12_with_openai(signs_en)

    # 3) guardamos cache y devolvemos
    translated["_cached"] = False
    _save_cache(date_key, translated)
    return _json_response(translated)


# ---------------------------
# Local run (si lo ejecutas en tu PC)
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)


