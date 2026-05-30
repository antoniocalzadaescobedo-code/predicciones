# config.py
import os
from datetime import datetime

# 🔹 Grupos oficiales FIFA 2026
FIFA_2026_GROUPS = {
    "A": ["México", "Sudáfrica", "Corea del Sur", "República Checa"],
    "B": ["Canadá", "Bosnia y Herzegovina", "Catar", "Suiza"],
    "C": ["Brasil", "Marruecos", "Haití", "Escocia"],
    "D": ["Estados Unidos", "Paraguay", "Australia", "Turquía"],
    "E": ["Alemania", "Curazao", "Costa de Marfil", "Ecuador"],
    "F": ["Países Bajos", "Japón", "Suecia", "Túnez"],
    "G": ["Bélgica", "Egipto", "Irán", "Nueva Zelanda"],
    "H": ["España", "Cabo Verde", "Arabia Saudita", "Uruguay"],
    "I": ["Francia", "Senegal", "Irak", "Noruega"],
    "J": ["Argentina", "Argelia", "Austria", "Jordania"],
    "K": ["Portugal", "República Democrática del Congo", "Uzbekistán", "Colombia"],
    "L": ["Inglaterra", "Croacia", "Ghana", "Panamá"]
}

# 🔹 Confederaciones
CONFEDERATIONS = {
    "UEFA": ["Sudáfrica", "Corea del Sur", "República Checa", "Bosnia y Herzegovina", 
             "Catar", "Suiza", "Haití", "Escocia", "Paraguay", "Australia", "Turquía", 
             "Curazao", "Costa de Marfil", "Ecuador", "Japón", "Suecia", "Túnez", 
             "Egipto", "Irán", "Nueva Zelanda", "Cabo Verde", "Arabia Saudita", 
             "Uruguay", "Senegal", "Irak", "Noruega", "Argelia", "Austria", "Jordania", 
             "República Democrática del Congo", "Uzbekistán", "Colombia", "Croacia", 
             "Ghana", "Panamá", "Inglaterra", "Francia", "España", "Portugal", 
             "Bélgica", "Países Bajos", "Alemania"],
    "CONMEBOL": ["México", "Brasil", "Estados Unidos", "Argentina", "Chile", "Perú", 
                 "Venezuela", "Bolivia", "Ecuador", "Colombia", "Paraguay", "Uruguay"],
    "CONCACAF": ["Canadá", "Estados Unidos", "México", "Haití", "Curazao", "Jamaica", 
                 "Panamá", "Costa Rica", "Honduras", "El Salvador", "Guatemala", "Cuba"],
    "CAF": ["Sudáfrica", "Marruecos", "Egipto", "Senegal", "Nigeria", "Camerún", 
            "Ghana", "Túnez", "Argelia", "Costa de Marfil", "Burkina Faso", "Mali",
            "Cabo Verde", "República Democrática del Congo"],
    "AFC": ["Corea del Sur", "Catar", "Japón", "Australia", "Irán", "Arabia Saudita", 
            "Irak", "Jordania", "Omán", "Uzbekistán", "China", "India"],
    "OFC": ["Nueva Zelanda", "Fiji", "Papúa Nueva Guinea", "Islas Salomón", 
            "Vanuatu", "Nueva Caledonia", "Tahití", "Samoa"]
}

# 🔹 Mapeos automáticos
COUNTRY_TO_GROUP = {team: grp for grp, teams in FIFA_2026_GROUPS.items() for team in teams}
COUNTRY_TO_CONF = {team: conf for conf, teams in CONFEDERATIONS.items() for team in teams}

# 🔹 Headers para scraping ético
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "es, en-US;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "keep-alive"
}

# 🔹 Fuentes de datos (prioridad: Wikipedia → FIFA → fallback)
SQUAD_SOURCES = {
    "wikipedia": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads",
    "fifa_teams": "https://www.fifa.com/fifaplus/en/tournaments/mens/worldcup/canadamexicousa2026/teams"
}

# 🔹 Configuración de output
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🔹 Configuración de monitor
MONITOR_CHECK_INTERVAL_HOURS = 6
MONITOR_HASH_FILE = os.path.join(OUTPUT_DIR, ".squads_hash.json")

# 🔹 Configuración de retries
MAX_RETRIES = 3
RETRY_MIN_WAIT = 2
RETRY_MAX_WAIT = 10
REQUEST_TIMEOUT = 15
