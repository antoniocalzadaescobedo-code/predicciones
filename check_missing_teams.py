#!/usr/bin/env python3
"""
Verifica qué equipos del calendario no están en la base de datos
"""
import json
from fifa_teams_database import FIFATeamsDatabase

# Cargar calendario
with open("data/fixtures/fixtures_friendlies_2026.json", "r", encoding="utf-8") as f:
    fixtures = json.load(f)

# Cargar base de datos
db = FIFATeamsDatabase("fifa_teams_db_es.json")
team_list = sorted(db.df["team_name"].tolist())

# Verificar equipos faltantes
all_teams = set()
for match in fixtures:
    all_teams.add(match["home"])
    all_teams.add(match["away"])

missing_teams = all_teams - set(team_list)

print(f"Total equipos en calendario: {len(all_teams)}")
print(f"Total equipos en base de datos: {len(team_list)}")
print(f"Equipos faltantes: {len(missing_teams)}")
print("\nEquipos faltantes:")
for team in sorted(missing_teams):
    print(f"  - {team}")

# Verificar cuántos partidos se mostrarían
valid_matches = 0
for match in fixtures:
    if match["home"] in team_list and match["away"] in team_list:
        valid_matches += 1

print(f"\nPartidos que se mostrarán: {valid_matches}/{len(fixtures)}")
