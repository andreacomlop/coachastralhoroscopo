import os
import base64
import json
from datetime import datetime, timezone, timedelta

import requests
from flask import Flask, jsonify, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

ASTRO_USER_ID = os.environ.get("ASTRO_USER_ID", "")
ASTRO_API_KEY = os.environ.get("ASTRO_API_KEY", "")
HORO_TZ = float(os.environ.get("HORO_TZ", "1"))  # España = 1 (invierno), 2 (verano)

ASTRO_ENDPOINT = "https://json.astrologyapi.com/v1/sun_sign_consolidated/daily/{}"

SIGNS = [
    "aries","taurus","gemini","cancer","leo","virgo",
    "libra","scorpio","sagittarius","capricorn","aquarius","pisces"
]

CACHE_FILE = "cache_horoscope.json"

def today_key():
    tz = timezone(timedelta(hours=HORO_TZ))
    return datetime.now(tz).strftime("%Y-%m-%d")

def astro_headers():
    auth_string = f"{ASTRO_USER_ID}:{ASTRO_API_KEY}"
    auth_encoded = base64.b64encode(auth_string.encode("utf-8")).decode("utf-8")
    return {
        "Authorization": f"Basic {auth_encoded}",
        "Content-Type": "application/json",
        "Accept-Language": "en"
    }

def load_cache():
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def save_cache(data):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def fetch_one(sign):
    url = ASTRO_ENDPOINT.format(sign)
    payload = {"timezone": HORO_TZ}
    r = requests.post(url, json=payload, headers=astro_headers(), timeout=60)
    r.raise_for_status()
    data = r.json()
    # Respuesta típica: status, sun_sign, prediction_date, prediction
    if not data.get("status"):
        raise RuntimeError(str(data))
    return {
        "sun_sign": data.get("sun_sign", sign),
        "prediction_date": data.get("prediction_date", ""),
        "prediction": data.get("prediction", "")
    }

def get_today_data(force=False):
    key = today_key()
    cached = load_cache()
    if (not force) and cached and cached.get("date_key") == key:
        cached["_cached"] = True
        return cached

    if not ASTRO_USER_ID or not ASTRO_API_KEY:
        raise RuntimeError("Faltan ASTRO_USER_ID / ASTRO_API_KEY en variables de entorno")

    signs_data = {}
    for s in SIGNS:
        signs_data[s] = fetch_one(s)

    result = {
        "date_key": key,
        "signs": signs_data,
        "_cached": False
    }
    save_cache(result)
    return result

# ---------- API JSON ----------
@app.get("/api/horoscope/today")
def api_today():
    data = get_today_data(force=False)
    return jsonify(data)

@app.get("/api/horoscope/refresh")
def api_refresh():
    data = get_today_data(force=True)
    return jsonify(data)

# ---------- HTML bonito (para iframe) ----------
@app.get("/horoscopo")
def horoscopo_html():
    data = get_today_data(force=False)
    date_key = data.get("date_key","")
    signs = data.get("signs",{})

    cards = []
    for sign, info in signs.items():
        text = info.get("prediction", "")
        # Si en el futuro añades prediction_es, se usará automáticamente:
        text = info.get("prediction_es", text)

        cards.append(f"""
          <div class="card">
            <div class="title">{sign.capitalize()}</div>
            <div class="text">{text}</div>
          </div>
        """)

    html = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Horóscopo</title>
<style>
  body {{
    margin:0; padding:18px;
    font-family: Arial, sans-serif;
    background:#0f0f10;
    color:#f5e6b8;
  }}
  h2 {{
    margin:0 0 10px 0;
    text-align:center;
    color:#f5e6b8;
    font-weight:800;
    letter-spacing:.3px;
  }}
  .sub {{
    text-align:center;
    color:#c9a54a;
    margin-bottom:16px;
    font-size:13px;
    opacity:.95;
  }}
  .grid {{
    display:grid;
    grid-template-columns:repeat(auto-fit, minmax(220px, 1fr));
    gap:14px;
  }}
  .card {{
    border:1px solid rgba(201,165,74,.35);
    background:#151518;
    border-radius:14px;
    padding:14px;
  }}
  .title {{
    color:#c9a54a;
    font-size:18px;
    font-weight:800;
    margin-bottom:8px;
    text-transform:capitalize;
  }}
  .text {{
    font-size:14px;
    line-height:1.55;
    color:#f5e6b8;
    white-space:pre-wrap;
  }}
</style>
</head>
<body>
  <h2>Horóscopo de hoy</h2>
  <div class="sub">{date_key} · actualizado automáticamente</div>
  <div class="grid">
    {''.join(cards)}
  </div>
</body>
</html>"""
    return Response(html, mimetype="text/html; charset=utf-8")

# Render usa la variable PORT
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)
