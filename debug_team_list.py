#!/usr/bin/env python3
"""
Debug team_list loading
"""
from fifa_teams_database import FIFATeamsDatabase

db = FIFATeamsDatabase("fifa_teams_db_es.json")
team_list = sorted(db.df["team_name"].tolist())

print(f"Total equipos en base de datos: {len(team_list)}")
print(f"Primeros 10 equipos: {team_list[:10]}")
print(f"Últimos 10 equipos: {team_list[-10:]}")

# Verificar equipos específicos del calendario
test_teams = ["México", "Ghana", "Marruecos", "Nigeria", "Jamaica", "India", "Egipto", "Rusia", "Burundi", "Zimbabwe"]
for team in test_teams:
    in_list = team in team_list
    print(f"{team}: {'OK' if in_list else 'MISSING'}")
