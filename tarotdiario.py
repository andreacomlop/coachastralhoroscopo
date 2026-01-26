import os
import json
import hashlib
import random
import secrets
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# ---------------------------
# Config (Render env vars)
# ---------------------------
TZ_NAME = os.getenv("TZ", "Europe/Madrid").strip()

ASTRO_USER_ID = os.getenv("ASTRO_USER_ID", "").strip()
ASTRO_API_KEY = os.getenv("ASTRO_API_KEY", "").strip()
ASTRO_BASE = "https://json.astrologyapi.com/v1"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

client = OpenAI(api_key=OPENAI_API_KEY)

CACHE_DIR = os.getenv("CACHE_DIR", "/tmp")


def _today_str():
    tz = ZoneInfo(TZ_NAME)
    return datetime.now(tz).strftime("%Y-%m-%d")


def _safe_device_id() -> str:
    """
    device_id debe venir desde GoodBarber como query param: ?device_id=...
    Si no viene, hacemos fallback (evita colisiones masivas, pero no es perfecto).
    """
    did = (request.args.get("device_id") or "").strip()
    if did:
        return did[:128]

    # fallback: user-agent + ip (proxy friendly)
    ua = (request.headers.get("User-Agent") or "").strip()
    ip = (request.headers.get("X-Forwarded-For") or request.remote_addr or "").split(",")[0].strip()
    raw = f"{ua}|{ip}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _cache_path(name: str, device_id: str):
    """
    Cache por día + device_id
    """
    day = _today_str()
    safe = "".join(c for c in device_id if c.isalnum() or c in ("-", "_"))[:64] or "anon"
    return os.path.join(CACHE_DIR, f"{name}_{day}_{safe}.json")


def _read_cache(path: str):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None


def _write_cache(path: str, data: dict):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _astro_post(endpoint: str, payload: dict | None = None):
    url = f"{ASTRO_BASE}/{endpoint.lstrip('/')}"
    r = requests.post(
        url,
        auth=(ASTRO_USER_ID, ASTRO_API_KEY),
        headers={
            "Content-Type": "application/json",
            # Según la documentación, este endpoint solo soporta "en"
            "Accept-Language": "en",
        },
        json=payload or {},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def _draw_tarot_numbers(device_id: str, live: bool) -> dict:
    """
    Devuelve 3 números (1..78) para love/career/finance.

    - Si live=False: determinista por (día + device_id) para que sea estable todo el día.
    - Si live=True: aleatorio real (nueva tirada) para ese device_id.
    """
    if live:
        # Aleatorio real, sin reemplazo
        nums = secrets.SystemRandom().sample(range(1, 79), 3)
    else:
        # Determinista: seed = sha256(day|device_id)
        seed_raw = f"{_today_str()}|{device_id}".encode("utf-8")
        seed_int = int(hashlib.sha256(seed_raw).hexdigest(), 16)
        rng = random.Random(seed_int)
        nums = rng.sample(range(1, 79), 3)

    return {"love": nums[0], "career": nums[1], "finance": nums[2]}


def _translate_tarot_to_es(tarot_en: dict) -> dict:
    """
    Traducción a español natural, estilo Coach Astral.
    Devuelve solo JSON con las claves: amor, trabajo, dinero_y_fortuna
    """
    love = (tarot_en.get("love") or "").strip()
    career = (tarot_en.get("career") or "").strip()
    finance = (tarot_en.get("finance") or "").strip()

    prompt = f"""
Vas a traducir y adaptar a español de España el texto de una lectura de tarot.
Objetivo: que suene natural, correcto, claro y con tono de coaching (cercano, directo, sin exageraciones).
Reglas:
- No inventes información, solo traduce y adapta.
- Evita anglicismos y frases forzadas.
- Mantén el sentido original.
- No uses "etc." ni puntos suspensivos.
- Devuelve ÚNICAMENTE un JSON válido con estas claves exactas:
  - "amor"
  - "trabajo"
  - "dinero_y_fortuna"

Texto (EN):
LOVE:
{love}

CAREER:
{career}

FINANCE:
{finance}
""".strip()

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "Eres un redactor profesional en español (España)."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )

    content = (resp.choices[0].message.content or "").strip()

    # Intentar parsear JSON tal cual
    try:
        data = json.loads(content)
    except Exception:
        # Fallback: si viniera con texto extra, intenta aislar el JSON
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            data = json.loads(content[start : end + 1])
        else:
            # Último recurso: devolver algo mínimo (sin romper la app)
            data = {
                "amor": love,
                "trabajo": career,
                "dinero_y_fortuna": finance,
            }

    # Normaliza campos esperados
    return {
        "amor": (data.get("amor") or "").strip(),
        "trabajo": (data.get("trabajo") or "").strip(),
        "dinero_y_fortuna": (data.get("dinero_y_fortuna") or "").strip(),
    }


@app.get("/api/tarot")
def tarot_daily():
    """
    Tarot diario traducido.
    Cache por día + device_id.
    Puedes forzar live con ?live=1 (solo para ese device_id).
    """
    live = request.args.get("live", "0").strip() == "1"
    device_id = _safe_device_id()

    cache_file = _cache_path("tarot_daily", device_id)

    if not live:
        cached = _read_cache(cache_file)
        if cached:
            return jsonify(cached)

    # 0) Generar números (1..78) requeridos por la API
    nums = _draw_tarot_numbers(device_id=device_id, live=live)

    # 1) Llamar a la API (inglés) con payload requerido
    tarot_en = _astro_post("tarot_predictions", payload=nums)

    # 2) Traducir con OpenAI a tu estructura
    tarot_es = _translate_tarot_to_es(tarot_en)

    # 3) Construir respuesta final
    result = {
        "date": _today_str(),
        "amor": tarot_es["amor"],
        "trabajo": tarot_es["trabajo"],
        "dinero_y_fortuna": tarot_es["dinero_y_fortuna"],
        # opcional debug
        "source_fields": ["love", "career", "finance"],
        "device_id_used": device_id,
        "numbers_used": nums,  # <- quítalo cuando ya esté validado
        "live": live,
    }

    _write_cache(cache_file, result)
    return jsonify(result)


@app.get("/health")
def health():
    return jsonify({"ok": True, "date": _today_str()})
