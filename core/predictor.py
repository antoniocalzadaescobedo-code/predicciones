"""
Predictor de fútbol que utiliza un modelo entrenado (Random Forest)
basado en features: Elo, forma reciente, diferencia de goles, localía, días descanso.
Si el modelo no existe, se usa una heurística simple.
"""

import joblib
import pandas as pd
import os
import numpy as np
from pathlib import Path

class WorldCupPredictor:
    def __init__(self, draw_calibration_factor=1.20):
        self.draw_calibration_factor = draw_calibration_factor
        self.model = None
        self.feature_cols = None
        self.teams_state = None
        
        core_path = Path(__file__).parent
        model_path = core_path / 'trained_model.pkl'
        cols_path = core_path / 'feature_columns.pkl'
        state_path = core_path / 'teams_final_state.pkl'
        
        if model_path.exists() and cols_path.exists():
            self.model = joblib.load(model_path)
            self.feature_cols = joblib.load(cols_path)
            if state_path.exists():
                self.teams_state = joblib.load(state_path)
            print("✅ Predictor: modelo y estado de equipos cargados.")
        else:
            print("⚠️ Predictor: modelo no encontrado. Heurística activa.")
    
    def _get_team_features(self, team_name):
        if self.teams_state and team_name in self.teams_state:
            state = self.teams_state[team_name]
            elo = state['elo']
            hist = state['history']
            if not hist:
                return elo, 0.5, 0.0, 14
            puntos = sum(m[3] for m in hist) / len(hist)
            gd = sum(m[4] for m in hist) / len(hist)
            # Días desde el último partido (simulado para el futuro)
            return elo, puntos, gd, 14
        return 1500, 0.5, 0.0, 14
    
    def predict(self, home_team, away_team, stage="Group Stage"):
        if self.model is None:
            return self._heuristic_predict(home_team, away_team)
        
        h_elo, h_form, h_gd, h_days = self._get_team_features(home_team)
        a_elo, a_form, a_gd, a_days = self._get_team_features(away_team)
        
        features = [h_elo - a_elo, h_form, a_form, h_gd, a_gd, 1.0, h_days, a_days]
        X = pd.DataFrame([features], columns=self.feature_cols)
        
        probas = self.model.predict_proba(X)[0]
        # 0=empate, 1=local, 2=visitante
        p_emp, p_loc, p_vis = probas[0], probas[1], probas[2]
        
        p_emp *= self.draw_calibration_factor
        total = p_loc + p_emp + p_vis
        return [p_loc/total, p_emp/total, p_vis/total]
    
    def _heuristic_predict(self, home_team, away_team):
        return [0.45, 0.25, 0.30]
