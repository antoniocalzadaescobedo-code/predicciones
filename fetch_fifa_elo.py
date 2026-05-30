"""
FETCH FIFA ELO - Descarga y limpieza automática de ratings Elo
==============================================================

Obtiene ratings actualizados de EloRatings.net, limpia nombres, asigna códigos ISO y exporta a CSV/JSON.
"""

import pandas as pd
import requests
from difflib import get_close_matches
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def fetch_fifa_elo_data(output_csv="fifa_teams_elo.csv", output_json="fifa_teams_db.json"):
    print("🌍 Descargando base de datos FIFA + Elo Ratings...")
    
    # Fuente pública confiable (actualizada tras cada partido oficial)
    url = "https://www.eloratings.net/"
    headers = {"User-Agent": "Mozilla/5.0 (WC2026 Predictor)"}
    
    try:
        # 1. Extraer tabla principal con pandas
        tables = pd.read_html(url, attrs={"class": "maintable"}, header=0)
        df = tables[0]
    except Exception as e:
        print(f"⚠️ No se pudo extraer tabla HTML: {e}")
        print("🔄 Usando fallback estático...")
        # Fallback seguro - crear estructura mínima con datos conocidos
        df = pd.DataFrame({
            "Team": ["Argentina", "France", "Brazil", "England", "Belgium", "Spain", "Netherlands", "Portugal", "Germany", "Italy"],
            "Elo": [2167, 2125, 2119, 2076, 2070, 2068, 2058, 2055, 2047, 2045],
            "Confederation": ["CONMEBOL", "UEFA", "CONMEBOL", "UEFA", "UEFA", "UEFA", "UEFA", "UEFA", "UEFA", "UEFA"]
        })
    
    # 2. Limpieza y estandarización
    df = df.rename(columns={
        "Team": "team_name",
        "Elo": "elo_rating",
        "Confederation": "confederation"
    })
    
    # Eliminar filas sin rating válido
    df = df[pd.to_numeric(df["elo_rating"], errors="coerce").notna()]
    df["elo_rating"] = df["elo_rating"].astype(float)
    
    # 3. Asignación de ISO 3166-1 alpha-3 (mapeo manual para casos críticos)
    iso_map = {
        "Argentina": "ARG", "Brasil": "BRA", "Syria": "SYR", "Portugal": "POR",
        "Spain": "ESP", "Germany": "GER", "France": "FRA", "England": "ENG",
        "Mexico": "MEX", "United States": "USA", "Japan": "JPN", "Australia": "AUS",
        "Saudi Arabia": "KSA", "Iran": "IRN", "South Korea": "KOR", "Morocco": "MAR",
        "Senegal": "SEN", "Nigeria": "NGA", "Ghana": "GHA", "Egypt": "EGY",
        "Peru": "PER", "Colombia": "COL", "Uruguay": "URU", "Chile": "CHI",
        "Ecuador": "ECU", "Paraguay": "PAR", "Bolivia": "BOL", "Venezuela": "VEN",
        "Canada": "CAN", "Costa Rica": "CRC", "Panama": "PAN", "Jamaica": "JAM",
        "Honduras": "HND", "El Salvador": "SLV", "Guatemala": "GTM", "Qatar": "QAT",
        "UAE": "UAE", "Iraq": "IRQ", "Uzbekistan": "UZB", "China": "CHN",
        "India": "IND", "Thailand": "THA", "Vietnam": "VIE", "Indonesia": "IDN",
        "Philippines": "PHI", "Malaysia": "MAS", "Singapore": "SIN", "New Zealand": "NZL",
        "Fiji": "FIJ", "New Caledonia": "NCL", "Tahiti": "TAH", "Papua New Guinea": "PNG",
        "Netherlands": "NED", "Belgium": "BEL", "Italy": "ITA", "Croatia": "CRO",
        "Switzerland": "SUI", "Denmark": "DEN", "Poland": "POL", "Sweden": "SWE",
        "Norway": "NOR", "Austria": "AUT", "Czech Republic": "CZE", "Serbia": "SRB",
        "Ukraine": "UKR", "Turkey": "TUR", "Russia": "RUS", "Greece": "GRC",
        "Romania": "ROU", "Hungary": "HUN", "Slovakia": "SVK", "Slovenia": "SVN",
        "Wales": "WAL", "Scotland": "SCO", "Ireland": "IRL", "Northern Ireland": "NIR"
    }
    
    def resolve_iso(name):
        if name in iso_map:
            return iso_map[name]
        # Fuzzy match para variaciones (ej: "Côte d'Ivoire", "Ivory Coast")
        matches = get_close_matches(name, iso_map.keys(), n=1, cutoff=0.7)
        return iso_map.get(matches[0], "XXX") if matches else "XXX"
    
    df["iso_code"] = df["team_name"].apply(resolve_iso)
    
    # 4. Flags y metadatos
    df["wc_qualified"] = df["team_name"].isin([
        "Argentina", "Brazil", "France", "Germany", "Spain", "England",
        "Portugal", "Netherlands", "Belgium", "Italy", "Denmark", "Switzerland",
        "Croatia", "Serbia", "Poland", "Wales", "Scotland", "Ukraine", "Sweden",
        "Norway", "Romania", "Hungary", "Czech Republic", "Slovakia",
        "Mexico", "United States", "Canada", "Costa Rica", "Panama", "Jamaica",
        "Japan", "South Korea", "Australia", "Iran", "Saudi Arabia", "Qatar",
        "Morocco", "Senegal", "Tunisia", "Nigeria", "Ghana", "Cameroon", "Egypt",
        "South Africa", "Mali", "Burkina Faso", "Cape Verde", "Algeria"
    ])  # Lista ejemplo 2026. Ajustar cuando salga clasificación oficial.
    
    df["last_updated"] = datetime.now().strftime("%Y-%m-%d")
    
    # 5. Exportar
    df = df[["team_name", "iso_code", "elo_rating", "confederation", "wc_qualified", "last_updated"]]
    df.to_csv(output_csv, index=False, encoding="utf-8")
    
    # JSON para carga rápida en producción
    df_json = df.to_dict(orient="records")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(df_json, f, indent=2, ensure_ascii=False)
        
    print(f"✅ Base exportada: {output_csv} ({len(df)} selecciones)")
    print(f"✅ JSON listo: {output_json}")
    return df

if __name__ == "__main__":
    fetch_fifa_elo_data()
