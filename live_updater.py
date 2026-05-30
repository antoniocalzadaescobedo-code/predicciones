#!/usr/bin/env python3
"""
Live Form Engine - Actualización Automática para TODOS los usuarios
- Se ejecuta al cargar la app (con caché de 6h)
- Guarda ELO actualizado en disco para persistencia entre sesiones
- Fallback a CSV local si falla la API
- Indicador visual de "datos en vivo"
"""

import os
import json
import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from fifa_teams_database import FIFATeamsDatabase
from data_pipeline import DataPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class LiveFormUpdater:
    CACHE_HOURS = 6  # Actualizar máximo cada 6 horas
    API_URL = "https://api.football-data.org/v4/matches"
    
    def __init__(self, api_key: Optional[str] = None):
        # Prioridad: env var > parámetro > fallback sin key
        self.api_key = (
            api_key or 
            os.getenv("FOOTBALL_DATA_API_KEY") or 
            os.getenv("FOOTBALL_DATA_API_TOKEN")
        )
        self.db = FIFATeamsDatabase("fifa_teams_db_es.json")
        self.pipeline = DataPipeline()
        
        # Rutas de caché y persistencia
        self.cache_file = "data/elo_cache_status.json"
        self.elo_output = "elo_actualizado.json"  # Este es el que lee fifa_teams_database.py
        os.makedirs("data", exist_ok=True)
        
    def _should_update(self) -> bool:
        """Verifica si es hora de actualizar (caché de 6h)"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r") as f:
                    data = json.load(f)
                    last = datetime.fromisoformat(data["last_update"])
                    hours_since = (datetime.now() - last).total_seconds() / 3600
                    if hours_since < self.CACHE_HOURS:
                        logger.info(f"[OK] Cache valido ({hours_since:.1f}h < {self.CACHE_HOURS}h)")
                        return False
        except Exception as e:
            logger.warning(f"[WARNING] Error leyendo cache: {e}")
        return True
        
    def _save_cache_status(self):
        """Guarda timestamp de última actualización"""
        with open(self.cache_file, "w") as f:
            json.dump({"last_update": datetime.now().isoformat()}, f)
            
    def _get_last_update_display(self) -> str:
        """Retorna string legible de última actualización"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r") as f:
                    data = json.load(f)
                    last = datetime.fromisoformat(data["last_update"])
                    return last.strftime("%d/%m %H:%M")
        except:
            pass
        return "Nunca"
    
    def fetch_recent_friendlies(self, days_back: int = 7) -> pd.DataFrame:
        """Obtiene amistosos recientes con fallback robusto"""
        matches = []
        
        # 🔹 INTENTO 1: API football-data.org (si hay key)
        if self.api_key:
            try:
                headers = {"X-Auth-Token": self.api_key}
                date_from = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
                date_to = datetime.now().strftime("%Y-%m-%d")
                
                resp = requests.get(
                    self.API_URL, 
                    headers=headers, 
                    params={"dateFrom": date_from, "dateTo": date_to, "status": "FINISHED"},
                    timeout=15
                )
                resp.raise_for_status()
                data = resp.json()
                
                for m in data.get("matches", []):
                    comp = m.get("competition", {})
                    if comp.get("type") == "FRIENDLY" or "amistoso" in comp.get("name", "").lower():
                        home = m["homeTeam"]["name"]
                        away = m["awayTeam"]["name"]
                        home_goals = m.get("score", {}).get("fullTime", {}).get("home", 0)
                        away_goals = m.get("score", {}).get("fullTime", {}).get("away", 0)
                        
                        if home_goals is not None and away_goals is not None:
                            matches.append({
                                "Date": m["utcDate"][:10],
                                "home_team": home,
                                "away_team": away,
                                "FTHG": home_goals,
                                "FTAG": away_goals,
                                "FTR": "H" if home_goals > away_goals else ("A" if away_goals > home_goals else "D"),
                                "Competition": "FRIENDLY",
                                "Round": "Amistoso"
                            })
                logger.info(f"✅ API: {len(matches)} amistosos encontrados")
            except Exception as e:
                logger.warning(f"⚠️ API falló: {e}")
                
        # 🔹 INTENTO 2: Fallback CSV local (siempre disponible)
        if not matches and os.path.exists("data/amistosos_reales.csv"):
            try:
                df = pd.read_csv("data/amistosos_reales.csv")
                # Renombrar columnas para coincidir con data_pipeline
                df = df.rename(columns={
                    "HomeTeam": "home_team", 
                    "AwayTeam": "away_team", 
                    "Date": "date",
                    "FTHG": "home_goals",
                    "FTAG": "away_goals"
                })
                df["date"] = pd.to_datetime(df["date"])
                cutoff = datetime.now() - timedelta(days=days_back)
                recent = df[df["date"] >= cutoff]
                matches = recent.to_dict("records")
                logger.info(f"✅ CSV Fallback: {len(matches)} amistosos cargados")
            except Exception as e:
                logger.error(f"❌ Fallback falló: {e}")
                
        return pd.DataFrame(matches) if matches else pd.DataFrame()
    
    def normalize_team_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza nombres para coincidir con fifa_teams_db_es.json"""
        name_map = {
            "Germany": "Alemania", "Spain": "España", "France": "Francia",
            "Italy": "Italia", "England": "Inglaterra", "Netherlands": "PaísesBajos",
            "Portugal": "Portugal", "Brazil": "Brasil", "Argentina": "Argentina",
            "USA": "Estados Unidos", "United States": "Estados Unidos",
            "Korea Republic": "Corea del Sur", "Czech Republic": "República Checa",
            "Cote d'Ivoire": "Costa de Marfil", "Ivory Coast": "Costa de Marfil",
            "Czechia": "República Checa", "Türkiye": "Turquía", "Turkey": "Turquía",
            "Cabo Verde": "Cabo Verde", "Cape Verde": "Cabo Verde",
            "DR Congo": "República Democrática del Congo", "Congo DR": "República Democrática del Congo",
            "Netherlands": "Países Bajos"
        }
        df["home_team"] = df["home_team"].map(name_map).fillna(df["home_team"])
        df["away_team"] = df["away_team"].map(name_map).fillna(df["away_team"])
        return df
    
    def update_elo_from_matches(self, matches_df: pd.DataFrame) -> Dict:
        """Actualiza ELO y guarda en el archivo que lee la app"""
        if matches_df.empty:
            return {"status": "no_data", "teams_updated": 0}
            
        matches_df = self.normalize_team_names(matches_df)
        valid_teams = set(self.db.df["team_name"].values)
        matches_df = matches_df[
            matches_df["home_team"].isin(valid_teams) & 
            matches_df["away_team"].isin(valid_teams)
        ]
        
        if matches_df.empty:
            return {"status": "no_valid_teams", "teams_updated": 0}
            
        # Actualizar ELO usando pipeline existente
        self.pipeline.df = matches_df
        nuevos_elos = self.pipeline.actualizar_elo_masivo(k_factor=25)
        
        # 🔥 GUARDAR EN EL ARCHIVO QUE LEE LA APP
        self.pipeline.exportar_nuevos_elos(self.elo_output)
        self._save_cache_status()
        
        return {
            "status": "success",
            "teams_updated": len(nuevos_elos),
            "matches_processed": len(matches_df),
            "last_update": self._get_last_update_display()
        }
    
    def run_auto_update(self, days_back: int = 7) -> Dict:
        """Ejecuta actualización automática (con caché)"""
        if not self._should_update():
            return {
                "status": "cached",
                "last_update": self._get_last_update_display(),
                "message": f"Datos actualizados hace <{self.CACHE_HOURS}h"
            }
            
        logger.info("🔄 Iniciando actualización automática...")
        matches = self.fetch_recent_friendlies(days_back)
        return self.update_elo_from_matches(matches)


# Instancia global para uso en Streamlit
updater = LiveFormUpdater()

if __name__ == "__main__":
    result = updater.run_auto_update(days_back=7)
    print("📊 Resultado:", result)
