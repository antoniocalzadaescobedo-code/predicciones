#!/usr/bin/env python3
"""
Pipeline de Datos - Fase 2C
Carga datos históricos desde CSV y actualiza ELO automáticamente.
Compatible con Football-Data.co.uk formato.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json

class DataPipeline:
    """Gestiona carga de datos y actualización de ratings ELO."""
    
    def __init__(self, db_path: str = "fifa_teams_db_es.json"):
        self.db_path = db_path
        self.df = None
        self.elo_history = {}
        
    def load_csv(self, csv_path: str, league: str = "internacional") -> pd.DataFrame:
        """
        Carga datos desde CSV (formato Football-Data.co.uk).
        
        Columnas esperadas:
        - Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR
        - Opcional: HS, AS, HC, AC (shots, corners)
        """
        try:
            df = pd.read_csv(csv_path)
            
            # Estandarizar nombres de columnas
            column_map = {
                'Date': 'date',
                'HomeTeam': 'home_team',
                'AwayTeam': 'away_team',
                'FTHG': 'home_goals',
                'FTAG': 'away_goals',
                'FTR': 'result',  # H/D/A
                'Competition': 'league'
            }
            
            df = df.rename(columns=column_map)
            # Si no existe columna Competition, usar el parámetro league
            if 'league' not in df.columns:
                df['league'] = league
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # Calcular métricas adicionales
            if 'home_goals' not in df.columns or 'away_goals' not in df.columns:
                return df
            df['total_goals'] = df['home_goals'] + df['away_goals']
            df['goal_diff'] = df['home_goals'] - df['away_goals']
            
            self.df = df
            print(f"✅ Cargadas {len(df)} partidos de {league}")
            return df
            
        except Exception as e:
            print(f"❌ Error cargando CSV: {e}")
            return None
    
    def actualizar_elo_masivo(self, k_factor: float = 30) -> Dict[str, float]:
        """
        Actualiza ratings ELO para todos los equipos basándose en 
        los partidos cargados en el DataFrame.
        """
        if self.df is None:
            raise ValueError("No hay datos cargados. Ejecuta load_csv() primero.")
        
        # Cargar ELO inicial desde DB
        from fifa_teams_database import FIFATeamsDatabase
        db = FIFATeamsDatabase(self.db_path)
        
        elo_ratings = {}
        for team in set(self.df['home_team'].tolist() + self.df['away_team'].tolist()):
            elo_ratings[team] = db.get_elo(team)
        
        # Ordenar por fecha
        df_sorted = self.df.sort_values('date')
        
        # Actualizar ELO partido por partido
        for _, match in df_sorted.iterrows():
            home = match['home_team']
            away = match['away_team']
            home_goals = match.get('home_goals', 0)
            away_goals = match.get('away_goals', 0)
            
            # Calcular resultado esperado
            elo_home = elo_ratings[home]
            elo_away = elo_ratings[away]
            
            # Probabilidad esperada (fórmula ELO estándar)
            expected_home = 1 / (1 + 10**((elo_away - elo_home) / 400))
            expected_away = 1 - expected_home
            
            # Resultado real
            if home_goals > away_goals:
                actual_home = 1.0
                actual_away = 0.0
            elif home_goals < away_goals:
                actual_home = 0.0
                actual_away = 1.0
            else:
                actual_home = 0.5
                actual_away = 0.5
            
            # Ajuste por diferencia de goles
            goal_diff = abs(home_goals - away_goals)
            goal_bonus = np.log(goal_diff + 1) * 0.1 if goal_diff > 0 else 0
            
            # Actualizar ELO
            delta_home = k_factor * (actual_home - expected_home) * (1 + goal_bonus)
            delta_away = k_factor * (actual_away - expected_away) * (1 + goal_bonus)
            
            elo_ratings[home] += delta_home
            elo_ratings[away] += delta_away
        
        self.elo_history = elo_ratings
        print(f"[OK] ELO actualizado para {len(elo_ratings)} equipos")
        return elo_ratings
    
    def calcular_forma_reciente(self, team: str, last_n: int = 5) -> Dict[str, float]:
        """
        Calcula métricas de forma reciente para un equipo.
        Retorna: puntos_por_partido, goles_a_favor_prom, goles_en_contra_prom, win_rate
        """
        if self.df is None:
            return {"puntos_por_partido": 0.5, "goles_a_favor_prom": 1.0, 
                    "goles_en_contra_prom": 1.0, "win_rate": 0.33}
        
        # Filtrar partidos del equipo
        team_matches = self.df[
            (self.df['home_team'] == team) | (self.df['away_team'] == team)
        ].sort_values('date', ascending=False).head(last_n)
        
        if len(team_matches) == 0:
            return {"puntos_por_partido": 0.5, "goles_a_favor_prom": 1.0, 
                    "goles_en_contra_prom": 1.0, "win_rate": 0.33}
        
        puntos = 0
        goles_a_favor = 0
        goles_en_contra = 0
        victorias = 0
        
        for _, match in team_matches.iterrows():
            es_local = match['home_team'] == team
            gf = match.get('home_goals', 0) if es_local else match.get('away_goals', 0)
            gc = match.get('away_goals', 0) if es_local else match.get('home_goals', 0)
            
            goles_a_favor += gf
            goles_en_contra += gc
            
            if gf > gc:
                puntos += 3
                victorias += 1
            elif gf == gc:
                puntos += 1
        
        n_partidos = len(team_matches)
        
        return {
            "puntos_por_partido": puntos / n_partidos / 3,  # Normalizado 0-1
            "goles_a_favor_prom": goles_a_favor / n_partidos,
            "goles_en_contra_prom": goles_en_contra / n_partidos,
            "win_rate": victorias / n_partidos
        }
    
    def calcular_h2h(self, team1: str, team2: str) -> Dict[str, float]:
        """
        Calcula historial directo entre dos equipos.
        Retorna: win_rate_team1, win_rate_team2, draw_rate, avg_goals
        """
        if self.df is None:
            return {"win_rate_team1": 0.33, "win_rate_team2": 0.33, 
                    "draw_rate": 0.34, "avg_goals": 2.5}
        
        h2h_matches = self.df[
            ((self.df['home_team'] == team1) & (self.df['away_team'] == team2)) |
            ((self.df['home_team'] == team2) & (self.df['away_team'] == team1))
        ]
        
        if len(h2h_matches) == 0:
            return {"win_rate_team1": 0.33, "win_rate_team2": 0.33, 
                    "draw_rate": 0.34, "avg_goals": 2.5}
        
        team1_wins = 0
        team2_wins = 0
        draws = 0
        total_goals = 0
        
        for _, match in h2h_matches.iterrows():
            total_goals += match.get('home_goals', 0) + match.get('away_goals', 0)
            
            if match.get('home_goals', 0) > match.get('away_goals', 0):
                if match['home_team'] == team1:
                    team1_wins += 1
                else:
                    team2_wins += 1
            elif match.get('home_goals', 0) < match.get('away_goals', 0):
                if match['away_team'] == team1:
                    team1_wins += 1
                else:
                    team2_wins += 1
            else:
                draws += 1
        
        n_matches = len(h2h_matches)
        
        return {
            "win_rate_team1": team1_wins / n_matches,
            "win_rate_team2": team2_wins / n_matches,
            "draw_rate": draws / n_matches,
            "avg_goals": total_goals / n_matches,
            "total_matches": n_matches
        }
    
    def exportar_nuevos_elos(self, output_path: str = "elo_actualizado.json"):
        """Exporta los ratings ELO actualizados a JSON."""
        if not self.elo_history:
            raise ValueError("No hay ELO actualizado. Ejecuta actualizar_elo_masivo() primero.")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.elo_history, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] ELO exportado a {output_path}")
    
    def generar_reporte(self, top_n: int = 10) -> pd.DataFrame:
        """Genera reporte de equipos con mejor ELO y forma reciente."""
        if not self.elo_history:
            raise ValueError("No hay datos. Ejecuta actualizar_elo_masivo() primero.")
        
        # Crear DataFrame con ELO
        df_elo = pd.DataFrame([
            {"team": team, "elo": elo}
            for team, elo in self.elo_history.items()
        ]).sort_values('elo', ascending=False)
        
        # Agregar forma reciente
        formas = []
        for team in df_elo['team']:
            forma = self.calcular_forma_reciente(team)
            formas.append(forma)
        
        df_elo['forma'] = [f['puntos_por_partido'] for f in formas]
        df_elo['win_rate'] = [f['win_rate'] for f in formas]
        
        return df_elo.head(top_n)


# ------------------------------------------------------------
# FUNCIÓN DE EJEMPLO DE USO
# ------------------------------------------------------------
if __name__ == "__main__":
    # Ejemplo de uso
    pipeline = DataPipeline()
    
    # Cargar datos (ejemplo con CSV de Football-Data.co.uk)
    # pipeline.load_csv("internacional_2024-2025.csv", league="internacional")
    
    # Actualizar ELO
    # nuevos_elos = pipeline.actualizar_elo_masivo(k_factor=30)
    
    # Calcular forma de un equipo
    # forma_argentina = pipeline.calcular_forma_reciente("Argentina", last_n=5)
    # print(f"Forma Argentina: {forma_argentina}")
    
    # Calcular H2H
    # h2h = pipeline.calcular_h2h("Argentina", "Brasil")
    # print(f"H2H ARG vs BRA: {h2h}")
    
    # Generar reporte
    # reporte = pipeline.generar_reporte(top_n=20)
    # print(reporte)
    
    print("📦 Pipeline de datos listo. Configura tu CSV y ejecuta.")
