# cleaner.py
import re
import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)  # ← Agregar esta línea
from config import COUNTRY_TO_GROUP, COUNTRY_TO_CONF

# 🔹 Mapeo de posiciones (inglés/español → español estandarizado)
POSITION_MAP = {
    "GK": "Portero", "GOALKEEPER": "Portero", "PORTERO": "Portero", "PORT": "Portero",
    "DF": "Defensa", "DEFENDER": "Defensa", "DEFENSA": "Defensa", "DEF": "Defensa",
    "CB": "Defensa", "LB": "Defensa", "RB": "Defensa", "LWB": "Defensa", "RWB": "Defensa",
    "MF": "Mediocampista", "MIDFIELDER": "Mediocampista", "MEDIOCAMPISTA": "Mediocampista", 
    "MID": "Mediocampista", "AM": "Mediocampista", "CM": "Mediocampista", "DM": "Mediocampista",
    "FW": "Delantero", "FORWARD": "Delantero", "DELANTERO": "Delantero", "FWD": "Delantero",
    "ST": "Delantero", "LW": "Delantero", "RW": "Delantero", "CF": "Delantero"
}

# 🔹 Mapeo de nombres de países (variaciones → nombre oficial)
COUNTRY_NAME_MAP = {
    "germany": "Alemania", "spain": "España", "france": "Francia",
    "italy": "Italia", "england": "Inglaterra", "netherlands": "Países Bajos",
    "portugal": "Portugal", "brazil": "Brasil", "argentina": "Argentina",
    "usa": "Estados Unidos", "united states": "Estados Unidos",
    "korea republic": "Corea del Sur", "czech republic": "República Checa",
    "cote d'ivoire": "Costa de Marfil", "ivory coast": "Costa de Marfil",
    "czechia": "República Checa", "türkiye": "Turquía", "turkey": "Turquía",
    "cabo verde": "Cabo Verde", "cape verde": "Cabo Verde",
    "dr congo": "República Democrática del Congo", "congo dr": "República Democrática del Congo"
}

def clean_player_name(name: str) -> str:
    """Limpia nombres de jugadores: remueve referencias, normaliza mayúsculas"""
    return (name
            .replace("[", "").replace("]", "")  # Remover [1], [citation needed]
            .strip()
            .title())

def normalize_position(pos_raw: str) -> str:
    """Normaliza posición a valor estandarizado"""
    pos_clean = re.sub(r'[^A-Za-z]', '', pos_raw).upper()
    return POSITION_MAP.get(pos_clean, "Mediocampista")

def clean_club_name(club: str) -> str:
    """Limpia nombres de clubes: remueve paréntesis, normaliza"""
    return (club
            .replace("(", "").replace(")", "")
            .replace("[", "").replace("]", "")
            .strip()
            .title())

def parse_age(age_raw) -> int:
    """Parsea edad a entero, con fallback a mediana"""
    try:
        age = int(age_raw)
        return age if 16 <= age <= 45 else 26
    except:
        return 26  # Edad mediana segura

def clean_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza completa + normalización + ML-ready
    Retorna DataFrame con columnas estandarizadas y sin valores nulos
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "country", "player", "position", "club", "age", "jersey_number",
            "squad_status", "announcement_date", "group", "confederation"
        ])
    
    df_clean = df.copy()
    
    # 1. Limpiar nombres de jugadores
    df_clean["player"] = df_clean["player_raw"].apply(clean_player_name)
    
    # 2. Normalizar posiciones
    df_clean["position"] = df_clean["position_raw"].apply(normalize_position)
    
    # 3. Limpiar clubes
    df_clean["club"] = df_clean["club_raw"].apply(clean_club_name)
    
    # 4. Parsear edad
    df_clean["age"] = df_clean["age_raw"].apply(parse_age)
    
    # 5. Dorsal (0 si no disponible)
    df_clean["jersey_number"] = df_clean["jersey_number_raw"].fillna(0).astype(int)
    
    # 6. Estado de convocatoria (Preliminar/Final)
    df_clean["squad_status"] = df_clean.get("squad_status", "Final")
    
    # 7. Fecha de anuncio
    df_clean["announcement_date"] = df_clean.get("announcement_date", pd.Timestamp.now().strftime("%Y-%m-%d"))
    
    # 8. Asignar grupo y confederación por país
    df_clean["group"] = df_clean["country"].map(COUNTRY_TO_GROUP).fillna("X")
    df_clean["confederation"] = df_clean["country"].map(COUNTRY_TO_CONF).fillna("UEFA")
    
    # 9. Seleccionar columnas finales (ML-ready)
    final_cols = [
        "country", "player", "position", "club", "age", "jersey_number",
        "squad_status", "announcement_date", "group", "confederation"
    ]
    df_final = df_clean[final_cols].drop_duplicates(subset=["country", "player"])
    
    # 10. Ordenar por país → posición → jugador
    df_final = df_final.sort_values(["country", "position", "player"]).reset_index(drop=True)
    
    logger.info(f"Limpieza completada: {len(df_final)} jugadores unicos")
    return df_final

def prepare_for_ml(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara DataFrame para ML: one-hot encoding opcional, sin texto libre
    """
    df_ml = df.copy()
    
    # One-hot encoding para posición (opcional, descomentar si se usa)
    # df_ml = pd.get_dummies(df_ml, columns=["position"], prefix="pos")
    
    # Codificar squad_status como binario
    df_ml["is_final_squad"] = (df_ml["squad_status"] == "Final").astype(int)
    
    # Asegurar tipos numéricos
    df_ml["age"] = pd.to_numeric(df_ml["age"], errors="coerce").fillna(26)
    df_ml["jersey_number"] = pd.to_numeric(df_ml["jersey_number"], errors="coerce").fillna(0)
    
    return df_ml
