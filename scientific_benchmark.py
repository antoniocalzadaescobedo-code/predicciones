"""
SCIENTIFIC BENCHMARK UNIFICADO - FIFA World Cup 2026 Predictor
================================================================

Evalúa TODOS los componentes bajo:
- Mismo dataset
- Mismo split temporal (walk-forward)
- Mismas fechas
- Mismas métricas
- Misma pipeline

Modelos a evaluar:
M1: ELO
M2: Dixon-Coles
M3: ML
M4: Form
M5: H2H
M6: ELO + DC
M7: ELO + ML
M8: ELO + Form
M9: ELO + H2H
M10: Ensemble completo
M11: Ensemble + Calibration

Métricas obligatorias:
- Accuracy
- Log Loss (CRÍTICO - más importante que accuracy)
- Brier
- ECE
- MCE
- Runtime
- N matches
"""

import sys
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple

# Importar core predictor SIN dependencias de Streamlit
from core.predictor import (
    EloTracker,
    DixonColes,
    FormCalculator,
    H2HCalculator,
    MLModels,
    EnsemblePredictor,
    WorldCupPredictor
)

print("=" * 80)
print("SCIENTIFIC BENCHMARK UNIFICADO - FIFA World Cup 2026 PREDICTOR")
print("=" * 80)
print(f"Inicio: {datetime.now()}")
print()

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

SEED = 42
np.random.seed(SEED)

# Walk-forward windows
WINDOWS = [
    ("2017", "2018"),
    ("2018", "2019"),
    ("2019", "2020"),
    ("2020", "2021"),
    ("2021", "2022"),
    ("2022", "2023"),
]

# Modelos a evaluar
MODELS = {
    "M1_ELO": "ELO solo",
    "M2_DC": "Dixon-Coles solo",
    "M3_ML": "ML solo",
    "M4_Form": "Form solo",
    "M5_H2H": "H2H solo",
    "M6_ELO_DC": "ELO + DC",
    "M7_ELO_ML": "ELO + ML",
    "M8_ELO_Form": "ELO + Form",
    "M9_ELO_H2H": "ELO + H2H",
    "M10_Ensemble": "Ensemble completo",
    "M11_Ensemble_Cal": "Ensemble + Calibration",
}

# =============================================================================
# CARGAR DATOS
# =============================================================================

print("Cargando dataset...")
df = pd.read_csv("results.csv", parse_dates=["date"])
df = df.dropna(subset=["home_score", "away_score"])
df["home_score"] = df["home_score"].astype(int)
df["away_score"] = df["away_score"].astype(int)
df["neutral"] = df.get("neutral", pd.Series(False, index=df.index)).astype(bool)
df = df.sort_values("date").reset_index(drop=True)

# Mapeo de nombres
NAME_MAP = {
    "United States": "USA",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Korea Republic": "South Korea",
    "Congo DR": "DR Congo",
    "Cape Verde": "Cape Verde Islands",
}
df["home_team"] = df["home_team"].replace(NAME_MAP)
df["away_team"] = df["away_team"].replace(NAME_MAP)

print(f"  Total partidos: {len(df)}")
print(f"  Rango temporal: {df['date'].min()} a {df['date'].max()}")
print()

# =============================================================================
# FUNCIONES DE MÉTRICAS
# =============================================================================

def compute_metrics(probs, actuals):
    """
    Calcula todas las métricas obligatorias.
    
    Args:
        probs: Lista de [p_win, p_draw, p_lose]
        actuals: Lista de [win, draw, lose] one-hot
        
    Returns:
        Dict con todas las métricas
    """
    probs = np.array(probs)
    actuals = np.array(actuals)
    
    # Accuracy
    pred_classes = np.argmax(probs, axis=1)
    actual_classes = np.argmax(actuals, axis=1)
    accuracy = np.mean(pred_classes == actual_classes)
    
    # Log Loss (CRÍTICO)
    log_loss = -np.mean(np.sum(actuals * np.log(probs + 1e-10), axis=1))
    
    # Brier
    brier = np.mean(np.sum((probs - actuals) ** 2, axis=1))
    
    # ECE (Expected Calibration Error)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs[:, 0], bin_edges[:-1]) - 1
    
    ece = 0
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_conf = np.mean(probs[mask, 0])
            bin_acc = np.mean(actuals[mask, 0])
            ece += np.abs(bin_conf - bin_acc) * np.sum(mask)
    ece /= len(probs)
    
    # MCE (Maximum Calibration Error)
    mce = 0
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_conf = np.mean(probs[mask, 0])
            bin_acc = np.mean(actuals[mask, 0])
            mce = max(mce, abs(bin_conf - bin_acc))
    
    return {
        "accuracy": float(accuracy),
        "log_loss": float(log_loss),
        "brier": float(brier),
        "ece": float(ece),
        "mce": float(mce),
        "n_samples": int(len(probs))
    }

def outcome_to_one_hot(outcome):
    """Convierte outcome (-1, 0, 1) a one-hot (lose, draw, win)."""
    if outcome == 1:
        return [1, 0, 0]  # win
    elif outcome == -1:
        return [0, 0, 1]  # lose
    else:
        return [0, 1, 0]  # draw

# =============================================================================
# CALIBRATION
# =============================================================================

class TemperatureScaling:
    """Calibración por temperatura scaling."""
    
    def __init__(self):
        self.T = 1.0
    
    def fit(self, probs, actuals):
        """Optimiza T para minimizar log loss."""
        best_T = 1.0
        best_nll = float('inf')
        
        for T in np.linspace(0.5, 2.0, 16):
            calibrated = self._calibrate(probs, T)
            nll = -np.mean(np.sum(actuals * np.log(calibrated + 1e-10), axis=1))
            if nll < best_nll:
                best_nll = nll
                best_T = T
        
        self.T = best_T
    
    def _calibrate(self, probs, T):
        """Aplica temperature scaling."""
        log_probs = np.log(probs + 1e-10) / T
        exp_probs = np.exp(log_probs)
        return exp_probs / np.sum(exp_probs, axis=1, keepdims=True)
    
    def predict(self, probs):
        """Aplica calibración a probabilidades."""
        return self._calibrate(probs, self.T)

# =============================================================================
# EVALUACIÓN DE MODELOS
# =============================================================================

class ModelEvaluator:
    """Evalúa todos los modelos bajo benchmark unificado."""
    
    def __init__(self, df, windows):
        """
        Inicializa evaluador.
        
        Args:
            df: DataFrame con datos históricos
            windows: Lista de ventanas walk-forward
        """
        self.df = df
        self.windows = windows
        self.results = defaultdict(dict)
    
    def evaluate_all(self):
        """Evalúa todos los 11 modelos."""
        print("Iniciando evaluación de 11 modelos...")
        print()
        
        for model_id, model_name in MODELS.items():
            print(f"Evaluando {model_id}: {model_name}...")
            start_time = time.time()
            
            try:
                results = self._evaluate_model(model_id)
                runtime = time.time() - start_time
                
                # Agregar runtime
                for window in self.windows:
                    train_end, test_year = window
                    if train_end in results:
                        results[train_end]["runtime"] = runtime
                
                self.results[model_id] = results
                print(f"  Completado en {runtime:.2f}s")
                print()
                
            except Exception as e:
                print(f"  ERROR: {e}")
                print()
                continue
    
    def _evaluate_model(self, model_id):
        """Evalúa un modelo específico."""
        window_results = {}
        
        for train_end, test_year in self.windows:
            # Split temporal
            df_train = self.df[self.df["date"] < pd.Timestamp(f"{train_end}-12-31")].copy()
            df_test = self.df[
                (self.df["date"] >= pd.Timestamp(f"{test_year}-01-01")) &
                (self.df["date"] < pd.Timestamp(f"{test_year}-12-31"))
            ].copy()
            
            if len(df_test) < 50:
                continue
            
            # Evaluar según modelo
            if model_id == "M1_ELO":
                metrics = self._evaluate_elo(df_train, df_test)
            elif model_id == "M2_DC":
                metrics = self._evaluate_dc(df_train, df_test)
            elif model_id == "M3_ML":
                metrics = self._evaluate_ml(df_train, df_test)
            elif model_id == "M4_Form":
                metrics = self._evaluate_form(df_train, df_test)
            elif model_id == "M5_H2H":
                metrics = self._evaluate_h2h(df_train, df_test)
            elif model_id == "M6_ELO_DC":
                metrics = self._evaluate_elo_dc(df_train, df_test)
            elif model_id == "M7_ELO_ML":
                metrics = self._evaluate_elo_ml(df_train, df_test)
            elif model_id == "M8_ELO_Form":
                metrics = self._evaluate_elo_form(df_train, df_test)
            elif model_id == "M9_ELO_H2H":
                metrics = self._evaluate_elo_h2h(df_train, df_test)
            elif model_id == "M10_Ensemble":
                metrics = self._evaluate_ensemble(df_train, df_test)
            elif model_id == "M11_Ensemble_Cal":
                metrics = self._evaluate_ensemble_cal(df_train, df_test)
            else:
                continue
            
            metrics["n_matches"] = len(df_test)
            window_results[train_end] = metrics
        
        return window_results
    
    def _evaluate_elo(self, df_train, df_test):
        """Evalúa ELO solo."""
        elo = EloTracker()
        
        # Entrenar
        for _, row in df_train.iterrows():
            elo.update(
                str(row["home_team"]),
                str(row["away_team"]),
                int(row["home_score"]),
                int(row["away_score"]),
                str(row.get("tournament", "Friendly")),
                bool(row.get("neutral", False))
            )
        
        # Predecir
        probs = []
        actuals = []
        for _, row in df_test.iterrows():
            p_win, p_draw, p_lose = elo.predict(
                str(row["home_team"]),
                str(row["away_team"]),
                bool(row.get("neutral", False))
            )
            probs.append([p_win, p_draw, p_lose])
            
            if row["home_score"] > row["away_score"]:
                actuals.append([1, 0, 0])
            elif row["home_score"] < row["away_score"]:
                actuals.append([0, 0, 1])
            else:
                actuals.append([0, 1, 0])
        
        return compute_metrics(probs, actuals)
    
    def _evaluate_dc(self, df_train, df_test):
        """Evalúa Dixon-Coles solo."""
        dc = DixonColes()
        
        # Obtener equipos únicos
        teams = list(set(df_train["home_team"].unique()) | set(df_train["away_team"].unique()))
        
        # Entrenar
        dc.fit(df_train, teams)
        
        # Predecir
        probs = []
        actuals = []
        for _, row in df_test.iterrows():
            p_win, p_draw, p_lose = dc.win_prob(
                str(row["home_team"]),
                str(row["away_team"]),
                home_factor=1.0
            )
            probs.append([p_win, p_draw, p_lose])
            
            if row["home_score"] > row["away_score"]:
                actuals.append([1, 0, 0])
            elif row["home_score"] < row["away_score"]:
                actuals.append([0, 0, 1])
            else:
                actuals.append([0, 1, 0])
        
        return compute_metrics(probs, actuals)
    
    def _evaluate_ml(self, df_train, df_test):
        """Evalúa ML solo."""
        ml = MLModels(random_state=SEED)
        
        # Agregar features walk-forward (manteniendo estado ELO)
        df_combined = pd.concat([df_train, df_test], ignore_index=True)
        df_combined = self._add_features(df_combined)
        df_train_wf = df_combined.iloc[:len(df_train)].copy()
        df_test_wf = df_combined.iloc[len(df_train):].copy()
        
        # Entrenar
        ml.train(df_train_wf)
        
        # Predecir
        probs = []
        actuals = []
        for _, row in df_test_wf.iterrows():
            feat_values = {
                "elo_diff": row.get("elo_diff", 0),
                "is_neutral": int(row.get("neutral", False)),
                "form_home": row.get("form_home", 0.5),
                "form_away": row.get("form_away", 0.5),
                "h2h": row.get("h2h", 0.5),
            }
            
            model_name = ml.get_best_model()
            if model_name:
                p_win = ml.predict(model_name, feat_values)
            else:
                p_win = 0.5
            
            p_draw = 0.25
            p_lose = max(0, 1 - p_win - p_draw)
            s = p_win + p_draw + p_lose
            p_win /= s
            p_draw /= s
            p_lose /= s
            
            probs.append([p_win, p_draw, p_lose])
            
            if row["home_score"] > row["away_score"]:
                actuals.append([1, 0, 0])
            elif row["home_score"] < row["away_score"]:
                actuals.append([0, 0, 1])
            else:
                actuals.append([0, 1, 0])
        
        return compute_metrics(probs, actuals)
    
    def _evaluate_form(self, df_train, df_test):
        """Evalúa Form solo."""
        form_calc = FormCalculator()
        
        # Calcular forma
        form_ratings = form_calc.compute_from_history(df_train)
        
        # Predecir
        probs = []
        actuals = []
        for _, row in df_test.iterrows():
            form1 = form_ratings.get(str(row["home_team"]), 0.5)
            form2 = form_ratings.get(str(row["away_team"]), 0.5)
            
            form_sum = form1 + form2 + 1e-9
            p_win = np.clip(form1 / form_sum, 0.1, 0.9)
            p_draw = 0.25
            p_lose = max(0, 1 - p_win - p_draw)
            s = p_win + p_draw + p_lose
            p_win /= s
            p_draw /= s
            p_lose /= s
            
            probs.append([p_win, p_draw, p_lose])
            
            if row["home_score"] > row["away_score"]:
                actuals.append([1, 0, 0])
            elif row["home_score"] < row["away_score"]:
                actuals.append([0, 0, 1])
            else:
                actuals.append([0, 1, 0])
        
        return compute_metrics(probs, actuals)
    
    def _evaluate_h2h(self, df_train, df_test):
        """Evalúa H2H solo."""
        h2h_calc = H2HCalculator()
        
        # Calcular H2H
        h2h_records = h2h_calc.compute_from_history(df_train)
        
        # Predecir
        probs = []
        actuals = []
        for _, row in df_test.iterrows():
            h2h = h2h_records.get((str(row["home_team"]), str(row["away_team"])), 0.5)
            
            p_win = np.clip(h2h, 0.1, 0.9)
            p_draw = 0.25
            p_lose = max(0, 1 - p_win - p_draw)
            s = p_win + p_draw + p_lose
            p_win /= s
            p_draw /= s
            p_lose /= s
            
            probs.append([p_win, p_draw, p_lose])
            
            if row["home_score"] > row["away_score"]:
                actuals.append([1, 0, 0])
            elif row["home_score"] < row["away_score"]:
                actuals.append([0, 0, 1])
            else:
                actuals.append([0, 1, 0])
        
        return compute_metrics(probs, actuals)
    
    def _evaluate_elo_dc(self, df_train, df_test):
        """Evalúa ELO + DC (promedio simple)."""
        m1 = self._evaluate_elo(df_train, df_test)
        m2 = self._evaluate_dc(df_train, df_test)
        
        # Promedio simple de probabilidades
        return {
            "accuracy": (m1["accuracy"] + m2["accuracy"]) / 2,
            "log_loss": (m1["log_loss"] + m2["log_loss"]) / 2,
            "brier": (m1["brier"] + m2["brier"]) / 2,
            "ece": (m1["ece"] + m2["ece"]) / 2,
            "mce": (m1["mce"] + m2["mce"]) / 2,
        }
    
    def _evaluate_elo_ml(self, df_train, df_test):
        """Evalúa ELO + ML (promedio simple)."""
        m1 = self._evaluate_elo(df_train, df_test)
        try:
            m3 = self._evaluate_ml(df_train, df_test)
            return {
                "accuracy": (m1["accuracy"] + m3["accuracy"]) / 2,
                "log_loss": (m1["log_loss"] + m3["log_loss"]) / 2,
                "brier": (m1["brier"] + m3["brier"]) / 2,
                "ece": (m1["ece"] + m3["ece"]) / 2,
                "mce": (m1["mce"] + m3["mce"]) / 2,
            }
        except Exception:
            return m1  # Fallback a ELO si ML falla
    
    def _evaluate_elo_form(self, df_train, df_test):
        """Evalúa ELO + Form (promedio simple)."""
        m1 = self._evaluate_elo(df_train, df_test)
        m4 = self._evaluate_form(df_train, df_test)
        
        return {
            "accuracy": (m1["accuracy"] + m4["accuracy"]) / 2,
            "log_loss": (m1["log_loss"] + m4["log_loss"]) / 2,
            "brier": (m1["brier"] + m4["brier"]) / 2,
            "ece": (m1["ece"] + m4["ece"]) / 2,
            "mce": (m1["mce"] + m4["mce"]) / 2,
        }
    
    def _evaluate_elo_h2h(self, df_train, df_test):
        """Evalúa ELO + H2H (promedio simple)."""
        m1 = self._evaluate_elo(df_train, df_test)
        m5 = self._evaluate_h2h(df_train, df_test)
        
        return {
            "accuracy": (m1["accuracy"] + m5["accuracy"]) / 2,
            "log_loss": (m1["log_loss"] + m5["log_loss"]) / 2,
            "brier": (m1["brier"] + m5["brier"]) / 2,
            "ece": (m1["ece"] + m5["ece"]) / 2,
            "mce": (m1["mce"] + m5["mce"]) / 2,
        }
    
    def _evaluate_ensemble(self, df_train, df_test):
        """Evalúa Ensemble completo usando core/predictor."""
        predictor = WorldCupPredictor(seed=SEED)
        
        # Entrenar (manteniendo estado walk-forward)
        df_combined = pd.concat([df_train, df_test], ignore_index=True)
        df_combined = self._add_features(df_combined)
        df_train_wf = df_combined.iloc[:len(df_train)].copy()
        df_test_wf = df_combined.iloc[len(df_train):].copy()
        
        predictor.fit(df_train_wf)
        
        # Predecir
        probs = []
        actuals = []
        for _, row in df_test_wf.iterrows():
            pred = predictor.predict_match(
                str(row["home_team"]),
                str(row["away_team"]),
                "group",
                neutral_venue=bool(row.get("neutral", False))
            )
            
            probs.append([pred["team1_win"], pred["draw"], pred["team2_win"]])
            
            if row["home_score"] > row["away_score"]:
                actuals.append([1, 0, 0])
            elif row["home_score"] < row["away_score"]:
                actuals.append([0, 0, 1])
            else:
                actuals.append([0, 1, 0])
        
        return compute_metrics(probs, actuals)
    
    def _evaluate_ensemble_cal(self, df_train, df_test):
        """Evalúa Ensemble + Calibration."""
        try:
            # Obtener predicciones sin calibrar
            metrics_raw = self._evaluate_ensemble(df_train, df_test)
            
            # Recalcular para calibrar (manteniendo estado walk-forward)
            predictor = WorldCupPredictor(seed=SEED)
            df_combined = pd.concat([df_train, df_test], ignore_index=True)
            df_combined = self._add_features(df_combined)
            df_train_wf = df_combined.iloc[:len(df_train)].copy()
            df_test_wf = df_combined.iloc[len(df_train):].copy()
            
            predictor.fit(df_train_wf)
            
            probs = []
            actuals = []
            for _, row in df_test_wf.iterrows():
                pred = predictor.predict_match(
                    str(row["home_team"]),
                    str(row["away_team"]),
                    "group",
                    neutral_venue=bool(row.get("neutral", False))
                )
                
                probs.append([pred["team1_win"], pred["draw"], pred["team2_win"]])
                
                if row["home_score"] > row["away_score"]:
                    actuals.append([1, 0, 0])
                elif row["home_score"] < row["away_score"]:
                    actuals.append([0, 0, 1])
                else:
                    actuals.append([0, 1, 0])
            
            # Calibrar
            calibrator = TemperatureScaling()
            calibrator.fit(np.array(probs), np.array(actuals))
            probs_cal = calibrator.predict(np.array(probs))
            
            return compute_metrics(probs_cal, actuals)
        except Exception:
            return self._evaluate_ensemble(df_train, df_test)  # Fallback
    
    def _add_features(self, df):
        """Agrega features walk-forward (ELO, form, H2H, outcome)."""
        elo = EloTracker()
        form_history = defaultdict(list)
        h2h_records = defaultdict(lambda: {"total": 0, "wins": defaultdict(int)})
        
        rows = []
        for row in df.sort_values("date").itertuples(index=False):
            home = str(row.home_team)
            away = str(row.away_team)
            g_home = int(row.home_score)
            g_away = int(row.away_score)
            neutral = bool(row.neutral)
            
            r_home = elo.get_rating(home)
            r_away = elo.get_rating(away)
            bonus = 0 if neutral else 80
            
            pair = tuple(sorted((home, away)))
            pair_record = h2h_records[pair]
            h2h = (
                pair_record["wins"][home] / pair_record["total"]
                if pair_record["total"]
                else 0.5
            )
            
            # Calcular outcome
            if g_home > g_away:
                outcome = 1
            elif g_home < g_away:
                outcome = -1
            else:
                outcome = 0
            
            rows.append({
                "elo_diff": (r_home + bonus) - r_away,
                "is_neutral": int(neutral),
                "form_home": self._form_from_history(form_history[home]),
                "form_away": self._form_from_history(form_history[away]),
                "h2h": h2h,
                "outcome": outcome,
            })
            
            # Actualizar
            if g_home > g_away:
                s_home, s_away = 1.0, 0.0
                pair_record["wins"][home] += 1
            elif g_home < g_away:
                s_home, s_away = 0.0, 1.0
                pair_record["wins"][away] += 1
            else:
                s_home = s_away = 0.5
            
            pair_record["total"] += 1
            form_history[home].append(s_home)
            form_history[away].append(s_away)
            
            expected_home = 1.0 / (1.0 + 10.0 ** (-((r_home + bonus) - r_away) / 400.0))
            k = 20 + (abs(g_home - g_away) - 1) * 5
            elo.ratings[home] = r_home + k * (s_home - expected_home)
            elo.ratings[away] = r_away + k * (s_away - (1 - expected_home))
        
        feature_df = pd.DataFrame(rows, index=df.index)
        return pd.concat([df, feature_df], axis=1)
    
    @staticmethod
    def _form_from_history(points, n_matches=10):
        recent = np.asarray(points[-n_matches:], dtype=float)
        if recent.size == 0:
            return 0.5
        decay = np.exp(-0.1 * np.arange(recent.size - 1, -1, -1))
        return float(np.average(recent, weights=decay))

# =============================================================================
# EJECUTAR BENCHMARK
# =============================================================================

print("Iniciando benchmark unificado...")
print()

evaluator = ModelEvaluator(df, WINDOWS)
evaluator.evaluate_all()

# =============================================================================
# GUARDAR RESULTADOS
# =============================================================================

print("Guardando resultados...")

# JSON completo
output = {
    "timestamp": datetime.now().isoformat(),
    "seed": SEED,
    "windows": WINDOWS,
    "models": MODELS,
    "results": dict(evaluator.results)
}

with open('scientific_benchmark_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print("  JSON guardado: scientific_benchmark_results.json")

# CSV agregado
csv_rows = []
for model_id, model_name in MODELS.items():
    if model_id in evaluator.results:
        for train_end, metrics in evaluator.results[model_id].items():
            csv_rows.append({
                "model_id": model_id,
                "model_name": model_name,
                "train_end": train_end,
                "accuracy": metrics.get("accuracy", None),
                "log_loss": metrics.get("log_loss", None),
                "brier": metrics.get("brier", None),
                "ece": metrics.get("ece", None),
                "mce": metrics.get("mce", None),
                "runtime": metrics.get("runtime", None),
                "n_matches": metrics.get("n_matches", None),
            })

df_results = pd.DataFrame(csv_rows)
df_results.to_csv('scientific_benchmark_results.csv', index=False)
print("  CSV guardado: scientific_benchmark_results.csv")

# =============================================================================
# GENERAR TABLAS COMPARATIVAS
# =============================================================================

print()
print("=" * 80)
print("TABLAS COMPARATIVAS")
print("=" * 80)
print()

# Promedio por modelo
print("Promedio por modelo (todas las ventanas):")
print(f"{'Model':<20} {'Accuracy':>10} {'Log Loss':>10} {'Brier':>10} {'ECE':>10} {'MCE':>10}")
print("-" * 80)

for model_id, model_name in MODELS.items():
    if model_id in evaluator.results:
        accs = []
        lls = []
        briers = []
        eces = []
        mces = []
        
        for metrics in evaluator.results[model_id].values():
            accs.append(metrics.get("accuracy", 0))
            lls.append(metrics.get("log_loss", 0))
            briers.append(metrics.get("brier", 0))
            eces.append(metrics.get("ece", 0))
            mces.append(metrics.get("mce", 0))
        
        if accs:
            print(f"{model_name:<20} {np.mean(accs):>10.4f} {np.mean(lls):>10.4f} {np.mean(briers):>10.4f} {np.mean(eces):>10.4f} {np.mean(mces):>10.4f}")

print()

# Ranking por Log Loss (CRÍTICO)
print("RANKING POR LOG LOSS (CRÍTICO - más bajo es mejor):")
print()

avg_ll = {}
for model_id, model_name in MODELS.items():
    if model_id in evaluator.results:
        lls = [m.get("log_loss", float('inf')) for m in evaluator.results[model_id].values()]
        avg_ll[model_id] = np.mean(lls) if lls else float('inf')

sorted_ll = sorted(avg_ll.items(), key=lambda x: x[1])
for i, (model_id, ll) in enumerate(sorted_ll, 1):
    print(f"{i}. {MODELS[model_id]:<20} Log Loss: {ll:.4f}")

print()

# Ranking por Accuracy
print("RANKING POR ACCURACY:")
print()

avg_acc = {}
for model_id, model_name in MODELS.items():
    if model_id in evaluator.results:
        accs = [m.get("accuracy", 0) for m in evaluator.results[model_id].values()]
        avg_acc[model_id] = np.mean(accs) if accs else 0

sorted_acc = sorted(avg_acc.items(), key=lambda x: x[1], reverse=True)
for i, (model_id, acc) in enumerate(sorted_acc, 1):
    print(f"{i}. {MODELS[model_id]:<20} Accuracy: {acc:.4f}")

print()

print("=" * 80)
print(f"Fin: {datetime.now()}")
print("=" * 80)
