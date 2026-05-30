# test_10_broken.py
from fifa_teams_database import FIFATeamsDatabase

db = FIFATeamsDatabase()

# Lista de nombres en español
test_cases = [
    "Inglaterra", "Costa de Marfil", "República Checa", "Estados Unidos", "Corea del Sur",
    "Arabia Saudita", "Cabo Verde", "Nueva Zelanda", "Fiyi", "Tahití"
]

print("🔍 Validando nombres problemáticos:")
print(f"{'Equipo':<20} | {'Elo':>6} | {'ISO':<6} | {'Estado'}")
print("-" * 45)

for name in test_cases:
    elo = db.get_elo(name, fuzzy=True)
    iso = db.df.loc[db.df['team_name'].str.contains(name, case=False, na=False), 'iso_code']
    iso = iso.values[0] if not iso.empty else "N/A"
    status = "✅" if elo > 1200 else "❌"
    print(f"{name:<20} | {elo:6.1f} | {iso:<6} | {status}")
