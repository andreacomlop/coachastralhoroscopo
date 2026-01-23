import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo

from flask import Flask, jsonify
from flask_cors import CORS
from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, AuthenticationError


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

Con esta información real, escribe un artículo diario de astrología general (no por signos) con un enfoque divulgativo, educativo y cercano, dirigido a personas que no saben astrología pero quieren aprender.

El artículo debe ser desarrollado y fácil de leer, de unas 100 palabras.

Estructura obligatoria del artículo:

1. Título atractivo y cercano, sin tecnicismos, que invite a leer.
2. Introducción breve, explicando que la astrología funciona como un “clima emocional” colectivo.
3. Qué está pasando hoy en el cielo, explicado de forma sencilla.
4. Cómo puede afectar esto a las personas en general (emociones, mente, energía, relaciones).
5. Qué significa esto dentro de la astrología, con explicación pedagógica (Luna, retrogradaciones, aspectos), como para principiantes.
6. Cierre reflexivo, que invite a observarse, sin predicciones ni consejos absolutos.

Tono y estilo:

- Cercano, humano y educativo.
- Con pequeñas pinceladas de ironía inteligente y humor suave (observacional, elegante).
- Nada místico, nada fatalista.
- Nada de horóscopo por signos.
- Nada de predicciones personales.
- Nada de destino.

Puedes usar emojis de forma sutil y elegante (🌙✨🌀☕️💭), sin exagerar.
Escribe con párrafos cortos, títulos claros y ritmo fluido.
"""


def cache_path_for_date(fecha_iso: str) -> str:
    # Render permite escribir en /tmp durante ejecución
    return f"/tmp/cache_articuloastro_{fecha_iso}.json"


@app.get("/")
def home():
    return jsonify({
        "ok": True,
        "service": "articuloastro",
        "message": "Servicio activo. Usa /daily-astrology-article",
        "endpoints": ["/health", "/daily-astrology-article"]
    })


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

    # 2) Validación rápida de API key (evita errores raros)
    if not OPENAI_API_KEY:
        return jsonify({
            "date": today_iso,
            "error": "Falta OPENAI_API_KEY en Render (Environment)."
        }), 500

    prompt = build_daily_astrology_prompt(today_iso)

    # 3) Generar con OpenAI (con control de errores)
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
        return jsonify({
            "date": today_iso,
            "error": "OPENAI_API_KEY inválida o sin permisos."
        }), 500

    except (APIConnectionError, APITimeoutError):
        return jsonify({
            "date": today_iso,
            "error": "No se puede conectar con OpenAI ahora mismo. Prueba en unos minutos."
        }), 503

    except Exception as e:
        return jsonify({
            "date": today_iso,
            "error": f"Error inesperado: {type(e).__name__}"
        }), 500

    data = {"date": today_iso, "article": article_text}

    # 4) Guardar cache
    try:
        with open(cache_file, "w", encoding="utf-8")_
