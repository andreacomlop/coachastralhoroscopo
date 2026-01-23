import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo

from flask import Flask, jsonify
from flask_cors import CORS
from openai import OpenAI

# ---------------------------
# Config (Render env vars)
# ---------------------------
TZ_NAME = os.getenv("TZ", "Europe/Madrid").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)
CORS(app)


def build_daily_astrology_prompt(fecha_iso: str) -> str:
    return f"""
Hoy es {fecha_iso}.

Antes de escribir el artículo, identifica y utiliza únicamente movimientos astrológicos reales que estén ocurriendo hoy (por ejemplo: posición general de la Luna, retrogradaciones activas, aspectos planetarios relevantes a nivel general).

No inventes configuraciones planetarias.
No fuerces eventos que no estén ocurriendo hoy.
Si algún movimiento no es relevante, no lo incluyas.

Con esta información real, escribe un artículo diario de astrología general (no por signos) con un enfoque divulgativo, educativo y cercano, dirigido a personas que no saben astrología pero quieren entender cómo influye en el día a día.

El artículo debe ser LARGO, desarrollado y fácil de leer.

Estructura obligatoria del artículo:
1. Título atractivo y cercano, sin tecnicismos.
2. Introducción breve: la astrología como “clima emocional” colectivo.
3. Qué está pasando hoy en el cielo, explicado fácil.
4. Cómo puede afectar a las personas en general (emociones, mente, energía).
5. Qué significa esto dentro de la astrología (explicación pedagógica).
6. Cierre reflexivo: conciencia, sin predicciones ni destino.

Tono:
- Cercano, humano y educativo
- Con pequeñas pinceladas de ironía inteligente y humor suave
- Nada místico, nada fatalista
- Nada de horóscopo por signos
- Nada de predicciones personales

Puedes usar emojis de forma sutil y elegante (🌙✨🌀☕️💭), sin pasarte.
Párrafos cortos y títulos claros.
"""


def cache_path_for_date(fecha_iso: str) -> str:
    # Render permite escribir en /tmp durante la ejecución
    return f"/tmp/cache_astrology_article_{fecha_iso}.json"


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.get("/daily-astrology-article")
def daily_astrology_article():
    tz = ZoneInfo(TZ_NAME)
    today_iso = datetime.now(tz).date().isoformat()
    cache_file = cache_path_for_date(today_iso)

    # 1) Cache del día
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))

    # 2) Generar con OpenAI
    prompt = build_daily_astrology_prompt(today_iso)

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "Eres un divulgador experto en astrología, claro, didáctico y con humor sutil."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )

    article_text = resp.choices[0].message.content.strip()

    data = {
        "date": today_iso,
        "article": article_text
    }

    # 3) Guardar cache
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return jsonify(data)
