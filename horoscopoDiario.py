import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo

from flask import Flask, jsonify, request
from flask_cors import CORS

from openai import OpenAI


# ---------------------------
# Config
# ---------------------------
TZ_NAME = os.getenv("TZ", "Europe/Madrid")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

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

# Cache en memoria (por día)
CACHE = {}  # { "YYYY-MM-DD": { "_cached": bool, "date_key":..., "signs": {...}} }


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


def build_prompt(date_label: str) -> str:
    # Pedimos JSON estricto para evitar sorpresas
    signs_list = ", ".join(SIGNS_ES)
    return f"""
Genera el horóscopo diario SOLO en español para la fecha {date_label}.
Quiero un JSON estrictamente válido (sin markdown, sin comentarios, sin texto extra).

Requisitos:
- Debe incluir los 12 signos EXACTOS en esta lista (minúsculas y tildes tal cual): {signs_list}
- Para cada signo:
  - "prediction": texto en español, tono “coach” pero concreto, 3 a 5 frases, sin mencionar astrología técnica.
  - "prediction_date": "{date_label}"
  - "sun_sign": el nombre del signo (tal cual, en español, de la lista).
- No incluyas nada fuera del JSON.

Formato exacto esperado:

{{
  "date_key": "YYYY-MM-DD",
  "signs": {{
    "aries": {{"prediction": "...", "prediction_date": "{date_label}", "sun_sign": "aries"}},
    ...
  }}
}}
""".strip()


def generate_12_signs_es(date_key: str, date_label: str) -> dict:
    if not OPENAI_API_KEY:
        raise RuntimeError("Falta OPENAI_API_KEY en variables de entorno (Render).")

    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = build_prompt(date_label)

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.7,
        messages=[
            {"role": "system", "content": "Eres un redactor profesional de horóscopos diarios en español. Devuelves JSON válido y nada más."},
            {"role": "user", "content": prompt},
        ],
    )

    content = (resp.choices[0].message.content or "").strip()

    # Parseo robusto: si viene con espacios o saltos, ok; si viene con texto extra, fallará (así lo detectamos)
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"OpenAI no devolvió JSON válido: {e}. Respuesta: {content[:300]}")

    # Validaciones mínimas
    if "signs" not in data or not isinstance(data["signs"], dict):
        raise RuntimeError("JSON inválido: falta 'signs' o no es un objeto.")

    # Forzamos date_key correcto (para que sea consistente aunque el modelo lo invente)
    data["date_key"] = date_key

    # Comprobamos que estén los 12 signos
    missing = [s for s in SIGNS_ES if s not in data["signs"]]
    if missing:
        raise RuntimeError(f"JSON inválido: faltan signos: {missing}")

    # Normalizamos campos por si acaso
    for s in SIGNS_ES:
        item = data["signs"].get(s, {})
        data["signs"][s] = {
            "prediction": str(item.get("prediction", "")).strip(),
            "prediction_date": date_label,
            "sun_sign": s,
        }

    return data


@app.get("/")
def home():
    # Para que la URL principal no dé 404 (en Render se ve más “pro”)
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

    data = generate_12_signs_es(date_key, date_label)
    result = {
        "_cached": False,
        "date_key": date_key,
        "signs": data["signs"],
    }

    CACHE[date_key] = result
    return jsonify(result)


if __name__ == "__main__":
    # Local dev
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=True)

