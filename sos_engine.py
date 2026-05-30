#!/usr/bin/env python3
"""
Strength of Schedule Engine - Basado en accordingto.ca methodology
Ajusta predicciones según dificultad de calendario y rendimiento real vs esperado.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

# Referencia: Mediana ELO de los 48 clasificados al Mundial 2026
MEDIAN_ELO_WORLD_CUP = 1775
TOP_100_ELO_THRESHOLD = 1400

class SOSEngine:
    """Motor para ajustar predicciones según SOS y performance delta."""
    
    def __init__(self, csv_path: str = "data/sos_2026.csv"):
        self.df = None
        self._load_data(csv_path)
    
    def _load_data(self, csv_path: str):
        """Carga datos SOS desde CSV."""
        try:
            self.df = pd.read_csv(csv_path)
            # Normalizar nombres de equipos para matching
            self.df['team_normalized'] = self.df['Team'].str.lower().str.strip()
        except FileNotFoundError:
            print(f"⚠️ No se encontró {csv_path}. Usando ajustes por defecto.")
            self.df = pd.DataFrame()
    
    def get_team_sos_data(self, team_name: str) -> Optional[Dict]:
        """Obtiene datos SOS para un equipo específico."""
        if self.df is None or self.df.empty:
            return None
        
        normalized = team_name.lower().strip()
        match = self.df[self.df['team_normalized'] == normalized]
        
        if match.empty:
            return None
        
        row = match.iloc[0]
        return {
            "elo": row["ELO"],
            "sos_score": row["SOS_Score"],
            "delta_elo": row["Delta_ELO"],
            "delta_schedule": row["Delta_Schedule"],
            "games_vs_top100": row["Games_Played_vs_Top100"],
            "status": row["Status"],
            "confederation": row["Confederation"]
        }
    
    def calculate_form_adjustment(self, team_name: str) -> float:
        """
        Calcula ajuste de probabilidad basado en Δ ELO (forma real).
        
        Retorna:
            float: Ajuste entre -0.10 y +0.10 (-10% a +10%)
        """
        data = self.get_team_sos_data(team_name)
        if not data or pd.isna(data["delta_elo"]):
            return 0.0
        
        delta = data["delta_elo"]
        games = data["games_vs_top100"]
        
        # Penalizar muestras pequeñas (<4 partidos vs top-100)
        if games < 4:
            delta *= 0.5  # Reducir impacto a la mitad
        
        # Mapear Δ ELO a ajuste de probabilidad
        # Δ > +20% → +8%, Δ > +10% → +5%, Δ < -20% → -8%, etc.
        if delta > 0.20:
            adjustment = 0.08
        elif delta > 0.10:
            adjustment = 0.05
        elif delta > 0.03:
            adjustment = 0.02
        elif delta < -0.20:
            adjustment = -0.08
        elif delta < -0.10:
            adjustment = -0.05
        elif delta < -0.03:
            adjustment = -0.02
        else:
            adjustment = 0.0
        
        return np.clip(adjustment, -0.10, 0.10)
    
    def calculate_schedule_context_adjustment(self, team_name: str) -> float:
        """
        Calcula ajuste basado en dificultad del calendario (SOS Score).
        
        Equipos con SOS muy bajo (calendario difícil) pueden estar subvalorados.
        Equipos con SOS muy alto (calendario fácil) pueden estar sobrevalorados.
        
        Retorna:
            float: Ajuste entre -0.05 y +0.05
        """
        data = self.get_team_sos_data(team_name)
        if not data or pd.isna(data["sos_score"]):
            return 0.0
        
        sos = data["sos_score"]
        
        # Ajuste basado en percentiles de SOS
        if sos < 0.20:  # Calendario extremadamente difícil (CONMEBOL)
            return 0.05
        elif sos < 0.35:  # Difícil
            return 0.03
        elif sos > 0.75:  # Calendario muy fácil (OFC, algunos CAF)
            return -0.05
        elif sos > 0.65:  # Fácil
            return -0.03
        else:
            return 0.0
    
    def get_confederation_baseline(self, confederation: str) -> Dict[str, float]:
        """
        Retorna ajustes base por confederación (para equipos sin datos SOS).
        """
        baselines = {
            "CONMEBOL": {"form_adjustment": 0.03, "schedule_adjustment": 0.03},
            "UEFA": {"form_adjustment": 0.00, "schedule_adjustment": 0.00},
            "CAF": {"form_adjustment": 0.00, "schedule_adjustment": -0.02},
            "AFC": {"form_adjustment": -0.02, "schedule_adjustment": -0.02},
            "CONCACAF": {"form_adjustment": -0.03, "schedule_adjustment": -0.03},
            "OFC": {"form_adjustment": -0.05, "schedule_adjustment": -0.05},
        }
        return baselines.get(confederation, {"form_adjustment": 0.0, "schedule_adjustment": 0.0})
    
    def apply_adjustments_to_prediction(self, team_name: str, 
                                       base_prob: float,
                                       is_home: bool = True) -> float:
        """
        Aplica todos los ajustes a una probabilidad base.
        
        Args:
            team_name: Nombre del equipo
            base_prob: Probabilidad base del modelo
            is_home: Si el equipo juega como local (para ajuste de localía)
        
        Returns:
            float: Probabilidad ajustada (entre 0.0 y 1.0)
        """
        # Ajuste por forma (Δ ELO)
        form_adj = self.calculate_form_adjustment(team_name)
        
        # Ajuste por contexto de calendario (SOS)
        schedule_adj = self.calculate_schedule_context_adjustment(team_name)
        
        # Ajuste total
        total_adjustment = form_adj + schedule_adj
        
        # Aplicar ajuste (con límite para mantener probabilidades válidas)
        adjusted = base_prob + total_adjustment
        return np.clip(adjusted, 0.05, 0.95)  # Mantener en rango razonable


# Función helper para uso directo
def adjust_prediction(team: str, base_prob: float, 
                     sos_engine: Optional[SOSEngine] = None) -> float:
    """Ajusta una probabilidad usando SOS Engine."""
    if sos_engine is None:
        sos_engine = SOSEngine()
    return sos_engine.apply_adjustments_to_prediction(team, base_prob)


if __name__ == "__main__":
    # Demo de uso
    engine = SOSEngine()
    
    equipos_test = ["Argentina", "Portugal", "Curacao", "New Zealand"]
    
    print("📊 Demo de Ajustes SOS")
    print("=" * 60)
    for team in equipos_test:
        base = 0.60  # Probabilidad base de ejemplo
        adjusted = adjust_prediction(team, base, engine)
        data = engine.get_team_sos_data(team)
        
        if data:
            print(f"{team:20s}: {base:.1%} → {adjusted:.1%} "
                  f"(ΔELO: {data['delta_elo']:+.1%}, SOS: {data['sos_score']:.1%})")
        else:
            print(f"{team:20s}: {base:.1%} → {adjusted:.1%} (sin datos SOS)")
