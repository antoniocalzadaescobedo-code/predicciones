"""
FIFA TEAMS DATABASE - Módulo de integración con búsqueda fuzzy
==============================================================

Conecta la base descargada con el predictor. Incluye búsqueda fuzzy, cálculo de elo_diff y fallback seguro.
"""

import pandas as pd
import numpy as np
from difflib import get_close_matches
import os

class FIFATeamsDatabase:
    def __init__(self, db_path="fifa_teams_db_es.json"):
        self.db_path = db_path
        self.df = self._load_db()
        self._build_name_map()
        
    def _load_db(self):
        """Carga base estática o genera si no existe"""
        if not os.path.exists(self.db_path):
            print("🔄 Generando base FIFA estática en español (204 selecciones)...")
            from fifa_elo_static_es import generate_fifa_elo_database_es
            generate_fifa_elo_database_es()
        
        df = pd.read_json(self.db_path)
        
        # Validación mínima
        required_cols = ["team_name", "iso_code", "elo_rating", "confederation", "wc_qualified"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Base corrupta. Faltan columnas: {set(required_cols) - set(df.columns)}")
        
        # 🔥 Cargar ELO actualizado si existe (prioridad sobre el JSON base)
        if os.path.exists("elo_actualizado.json"):
            try:
                import json
                with open("elo_actualizado.json", "r", encoding="utf-8") as f:
                    updated_elos = json.load(f)
                # Actualizar solo los equipos que existen en la DB
                for team, elo in updated_elos.items():
                    if team in df["team_name"].values:
                        idx = df[df["team_name"] == team].index[0]
                        df.at[idx, "elo_rating"] = elo
                print(f"[OK] ELO actualizado cargado desde elo_actualizado.json")
            except Exception as e:
                print(f"[WARNING] No se pudo cargar ELO actualizado: {e}")
        
        print(f"[OK] Base FIFA cargada: {len(df)} selecciones | Elo rango: {df['elo_rating'].min():.0f}-{df['elo_rating'].max():.0f}")
        return df
    
    def _build_name_map(self):
        """Construye mapeo de nombres en inglés a español"""
        self.name_map = {
            "England": "Inglaterra",
            "Ivory Coast": "Costa de Marfil",
            "Czech Republic": "República Checa",
            "USA": "Estados Unidos",
            "South Korea": "Corea del Sur",
            "Saudi Arabia": "Arabia Saudita",
            "Iran": "Irán",
            "Syria": "Siria",
            "Germany": "Alemania",
            "Spain": "España",
            "France": "Francia",
            "Brazil": "Brasil",
            "Argentina": "Argentina",
            "Netherlands": "Países Bajos",
            "Belgium": "Bélgica",
            "Italy": "Italia",
            "Portugal": "Portugal",
            "Croatia": "Croacia",
            "Denmark": "Dinamarca",
            "Switzerland": "Suiza",
            "Serbia": "Serbia",
            "Poland": "Polonia",
            "Ukraine": "Ucrania",
            "Austria": "Austria",
            "Hungary": "Hungría",
            "Scotland": "Escocia",
            "Turkey": "Turquía",
            "Sweden": "Suecia",
            "Wales": "Gales",
            "Norway": "Noruega",
            "Romania": "Rumania",
            "Greece": "Grecia",
            "Slovakia": "Eslovaquia",
            "Ireland": "Irlanda",
            "Finland": "Finlandia",
            "Bulgaria": "Bulgaria",
            "Northern Ireland": "Irlanda del Norte",
            "Iceland": "Islandia",
            "Bosnia and Herzegovina": "Bosnia y Herzegovina",
            "Albania": "Albania",
            "Montenegro": "Montenegro",
            "Slovenia": "Eslovenia",
            "North Macedonia": "Macedonia del Norte",
            "Georgia": "Georgia",
            "Luxembourg": "Luxemburgo",
            "Armenia": "Armenia",
            "Kosovo": "Kosovo",
            "Cyprus": "Chipre",
            "Azerbaijan": "Azerbaiyán",
            "Belarus": "Bielorrusia",
            "Kazakhstan": "Kazajistán",
            "Faroe Islands": "Islas Feroe",
            "Moldova": "Moldavia",
            "Estonia": "Estonia",
            "Latvia": "Letonia",
            "Lithuania": "Lituania",
            "Malta": "Malta",
            "Andorra": "Andorra",
            "Liechtenstein": "Liechtenstein",
            "Gibraltar": "Gibraltar",
            "San Marino": "San Marino",
            "Mexico": "México",
            "United States": "Estados Unidos",
            "Canada": "Canadá",
            "Panama": "Panamá",
            "Costa Rica": "Costa Rica",
            "Jamaica": "Jamaica",
            "Honduras": "Honduras",
            "El Salvador": "El Salvador",
            "Guatemala": "Guatemala",
            "Trinidad and Tobago": "Trinidad y Tobago",
            "Haiti": "Haití",
            "Cuba": "Cuba",
            "Curaçao": "Curazao",
            "Nicaragua": "Nicaragua",
            "Dominican Republic": "República Dominicana",
            "Suriname": "Surinam",
            "French Guiana": "Guayana Francesa",
            "Belize": "Belice",
            "Antigua and Barbuda": "Antigua y Barbuda",
            "Saint Kitts and Nevis": "San Cristóbal y Nieves",
            "Barbados": "Barbados",
            "Grenada": "Granada",
            "Saint Vincent and the Grenadines": "San Vicente y las Granadinas",
            "Dominica": "Dominica",
            "US Virgin Islands": "Islas Vírgenes de EE.UU.",
            "British Virgin Islands": "Islas Vírgenes Británicas",
            "Anguilla": "Anguila",
            "Montserrat": "Montserrat",
            "Turks and Caicos Islands": "Turcas y Caicos",
            "Bermuda": "Bermudas",
            "Puerto Rico": "Puerto Rico",
            "Japan": "Japón",
            "South Korea": "Corea del Sur",
            "Iraq": "Irak",
            "United Arab Emirates": "Emiratos Árabes",
            "Uzbekistan": "Uzbekistán",
            "China": "China",
            "Oman": "Omán",
            "Jordan": "Jordania",
            "Bahrain": "Bahréin",
            "Palestine": "Palestina",
            "Lebanon": "Líbano",
            "India": "India",
            "Thailand": "Tailandia",
            "Vietnam": "Vietnam",
            "Kyrgyzstan": "Kirguistán",
            "Philippines": "Filipinas",
            "Malaysia": "Malasia",
            "Indonesia": "Indonesia",
            "Singapore": "Singapur",
            "Turkmenistan": "Turkmenistán",
            "Tajikistan": "Tayikistán",
            "Hong Kong": "Hong Kong",
            "Yemen": "Yemen",
            "Afghanistan": "Afganistán",
            "Myanmar": "Myanmar",
            "Cambodia": "Camboya",
            "Laos": "Laos",
            "Macau": "Macao",
            "Mongolia": "Mongolia",
            "Bhutan": "Bután",
            "Brunei": "Brunéi",
            "Timor-Leste": "Timor-Leste",
            "Pakistan": "Pakistán",
            "Nepal": "Nepal",
            "Bangladesh": "Bangladés",
            "Maldives": "Maldivas",
            "Sri Lanka": "Sri Lanka",
            "Guam": "Guam",
            "Northern Mariana Islands": "Islas Marianas del Norte",
            "Morocco": "Marruecos",
            "Senegal": "Senegal",
            "Nigeria": "Nigeria",
            "Egypt": "Egipto",
            "Tunisia": "Túnez",
            "Algeria": "Argelia",
            "Cameroon": "Camerún",
            "Ivory Coast": "Costa de Marfil",
            "Ghana": "Ghana",
            "Mali": "Malí",
            "Burkina Faso": "Burkina Faso",
            "South Africa": "Sudáfrica",
            "Cape Verde": "Cabo Verde",
            "Guinea": "Guinea",
            "Zambia": "Zambia",
            "Uganda": "Uganda",
            "Gabon": "Gabón",
            "Congo": "Congo",
            "DR Congo": "RD Congo",
            "Niger": "Níger",
            "Mauritania": "Mauritania",
            "Benin": "Benín",
            "Togo": "Togo",
            "Zimbabwe": "Zimbabue",
            "Kenya": "Kenia",
            "Mozambique": "Mozambique",
            "Tanzania": "Tanzania",
            "Rwanda": "Ruanda",
            "Madagascar": "Madagascar",
            "Angola": "Angola",
            "Namibia": "Namibia",
            "Botswana": "Botsuana",
            "Lesotho": "Lesoto",
            "Eswatini": "Eswatini",
            "Malawi": "Malawi",
            "Comoros": "Comoras",
            "Sudan": "Sudán",
            "South Sudan": "Sudán del Sur",
            "Ethiopia": "Etiopía",
            "Eritrea": "Eritrea",
            "Somalia": "Somalia",
            "Djibouti": "Djibouti",
            "Chad": "Chad",
            "Central African Republic": "República Centroafricana",
            "Guinea-Bissau": "Guinea-Bisáu",
            "Gambia": "Gambia",
            "Liberia": "Liberia",
            "Sierra Leone": "Sierra Leona",
            "Equatorial Guinea": "Guinea Ecuatorial",
            "São Tomé and Príncipe": "Santo Tomé y Príncipe",
            "Seychelles": "Seychelles",
            "Mauritius": "Mauricio",
            "New Zealand": "Nueva Zelanda",
            "New Caledonia": "Nueva Caledonia",
            "Fiji": "Fiyi",
            "Papua New Guinea": "Papúa Nueva Guinea",
            "Tahiti": "Tahití",
            "Solomon Islands": "Islas Salomón",
            "Vanuatu": "Vanuatu",
            "Samoa": "Samoa",
            "American Samoa": "Samoa Americana",
            "Tonga": "Tonga",
            "Cook Islands": "Islas Cook",
            "Kiribati": "Kiribati",
            "Tuvalu": "Tuvalu",
            "Palau": "Palau"
        }
    
    def _normalize_name(self, name):
        """Normaliza nombre usando mapeo o retorna original"""
        return self.name_map.get(name, name)
    
    def get_elo(self, team_name, fuzzy=True):
        """Retorna Elo rating de un equipo con fuzzy matching"""
        # Normalizar nombre (inglés → español)
        normalized_name = self._normalize_name(team_name)
        
        if normalized_name in self.df["team_name"].values:
            return self.df.loc[self.df["team_name"] == normalized_name, "elo_rating"].values[0]
        
        if fuzzy:
            matches = get_close_matches(normalized_name, self.df["team_name"].values, n=1, cutoff=0.75)
            if matches:
                return self.df.loc[self.df["team_name"] == matches[0], "elo_rating"].values[0]
        return 1200.0  # Default para selecciones sin datos
        
    def get_elo_diff(self, home, away, neutral=False, home_advantage=100.0):
        """Calcula elo_diff ajustado para tu predictor"""
        h_elo = self.get_elo(home)
        a_elo = self.get_elo(away)
        
        diff = h_elo - a_elo
        if not neutral:
            diff += home_advantage
        return diff
    
    def get_all_teams(self):
        return self.df["team_name"].tolist()
    
    def is_qualified(self, team_name):
        row = self.df[self.df["team_name"] == team_name]
        if row.empty:
            # Fuzzy fallback
            matches = get_close_matches(team_name, self.df["team_name"].values, n=1, cutoff=0.75)
            if matches:
                return bool(self.df.loc[self.df["team_name"] == matches[0], "wc_qualified"].values[0])
        return bool(row["wc_qualified"].values[0]) if not row.empty else False
