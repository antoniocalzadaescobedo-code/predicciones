#!/usr/bin/env python3
"""
Agrega equipos faltantes a la base de datos FIFA
"""
import json
import pandas as pd

# Cargar base de datos existente
with open("fifa_teams_db_es.json", "r", encoding="utf-8") as f:
    db = json.load(f)

df = pd.DataFrame(db)

# Equipos faltantes con ELO ratings razonables
missing_teams = [
    {
        "team_name": "Rusia",
        "iso_code": "RUS",
        "elo_rating": 1650,
        "confederation": "UEFA",
        "wc_qualified": False
    },
    {
        "team_name": "Burundi",
        "iso_code": "BDI",
        "elo_rating": 1250,
        "confederation": "CAF",
        "wc_qualified": False
    },
    {
        "team_name": "Zimbabwe",
        "iso_code": "ZWE",
        "elo_rating": 1300,
        "confederation": "CAF",
        "wc_qualified": False
    }
]

# Agregar equipos faltantes
for team in missing_teams:
    if team["team_name"] not in df["team_name"].values:
        df = pd.concat([df, pd.DataFrame([team])], ignore_index=True)
        print(f"[OK] Agregado: {team['team_name']}")

# Guardar base de datos actualizada
db_updated = df.to_dict(orient="records")
with open("fifa_teams_db_es.json", "w", encoding="utf-8") as f:
    json.dump(db_updated, f, indent=2, ensure_ascii=False)

print(f"[OK] Base de datos actualizada: {len(df)} equipos")
