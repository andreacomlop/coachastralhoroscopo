def build_luna_de_hoy_prompt(fecha_iso: str, facts: dict) -> str:
    moon_phase = facts.get("moon_phase") or "(sin dato)"
    sig = facts.get("significance_en") or ""
    rep = facts.get("report_en") or ""

    return "\n".join([
        f"Hoy es {fecha_iso}.",
        "",
        "A continuación tienes un texto astrológico REAL en inglés procedente de una API.",
        "Tu tarea es ÚNICAMENTE traducirlo al español.",
        "",
        "REGLAS:",
        "- Traduce de forma fiel el contenido.",
        "- No añadas información nueva.",
        "- No elimines ideas.",
        "- No resumas.",
        "- No interpretes.",
        "- No cambies el sentido.",
        "- Usa un español natural y correcto, como si el texto se hubiera escrito originalmente en español.",
        "",
        "CONTENIDO A TRADUCIR:",
        f"Fase lunar: {moon_phase}",
        "",
        f"Significance:",
        sig,
        "",
        f"Report:",
        rep,
        "",
        "FORMATO DE SALIDA:",
        "- Título breve en español.",
        "- Texto traducido, sin añadir explicaciones.",
        "- No menciones que es una traducción.",
    ])
