"""
================================================================================
  evaluation.py — Capa de evaluación científica con calibración temporal
  FIFA World Cup 2026 Predictor
================================================================================
  Arquitectura:
    LiveEloTracker      → ratings sin leakage, partido a partido
    PredictionEvaluator → backtesting temporal, walk-forward, calibración, métricas
    CalibrationAnalyzer → bins expected vs observed, ECE, MCE, draws
    ModelComparison     → compara ELO puro / ELO Calibrado / ELO+DC / Ensemble
================================================================================
"""

from __future__ import annotations

import os
import sys
import math
import json
import pickle
import hashlib
import argparse
import warnings
from collections import defaultdict
from dataclasses  import dataclass, asdict, field
from typing       import Dict, List, Optional, Tuple, Callable

# Evitar UnicodeEncodeError en terminales de Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy  as np
import pandas as pd

from calibration import TemperatureScalingCalibrator

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES GLOBALES
# ──────────────────────────────────────────────────────────────────────────────

ELO_BASE          = 1500
HOME_ADVANTAGE    = 100       # puntos ELO bonus al local
UPSET_THRESHOLD   = 0.35      # underdog si p(win) < 35%
LOG_CLIP          = 1e-7      # evitar log(0)
MIN_MATCHES_EVAL  = 20        # mínimo para reportar una métrica
N_CALIBRATION_BINS = 10

# K base por tipo de torneo — idéntico a elo_football.py para consistencia
TOURNAMENT_K: Dict[str, int] = {
    "FIFA World Cup":               60,
    "UEFA European Championship":   45,
    "Copa America":                 45,
    "Africa Cup of Nations":        45,
    "AFC Asian Cup":                45,
    "CONCACAF Gold Cup":            45,
    "FIFA World Cup qualification": 50,
    "UEFA Euro qualification":      40,
    "UEFA Nations League":          38,
    "CONCACAF Nations League":      35,
    "Friendly":                     25,
}
K_DEFAULT = 35

# Normalización de nombres históricos
NAME_MAP: Dict[str, str] = {
    "United States":       "USA",
    "IR Iran":             "Iran",
    "Côte d'Ivoire":       "Ivory Coast",
    "Korea Republic":      "South Korea",
    "Congo DR":            "DR Congo",
    "Cape Verde":          "Cape Verde Islands",
    "Czech Republic":      "Czechia",
    "North Macedonia":     "Macedonia",
    "Bosnia-Herzegovina":  "Bosnia and Herzegovina",
}

# ──────────────────────────────────────────────────────────────────────────────
# ESTRUCTURAS DE DATOS
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MatchRecord:
    """Un partido histórico con predicción y resultado real."""
    date:       str
    home:       str
    away:       str
    tournament: str
    year:       int
    p_home:     float          # P(victoria local)
    p_draw:     float          # P(empate)
    p_away:     float          # P(victoria visitante)
    actual:     int            # 1=local, 0=empate, -1=visitante
    g_home:     int
    g_away:     int
    elo_diff:   float          # elo_home − elo_away en el momento del partido
    neutral:    bool
    model_tag:  str = "elo"    # qué modelo generó la predicción

    def probs(self) -> np.ndarray:
        return np.array([self.p_home, self.p_draw, self.p_away])

    def one_hot(self) -> np.ndarray:
        if   self.actual ==  1: return np.array([1., 0., 0.])
        elif self.actual ==  0: return np.array([0., 1., 0.])
        else:                   return np.array([0., 0., 1.])

    def predicted_outcome(self) -> int:
        idx = int(np.argmax(self.probs()))
        return [1, 0, -1][idx]

    def is_correct(self) -> bool:
        return self.predicted_outcome() == self.actual

    def is_upset(self) -> bool:
        """True si ganó el equipo con p < UPSET_THRESHOLD."""
        p_fav = max(self.p_home, self.p_away)
        if p_fav < (1 - UPSET_THRESHOLD):
            return False   # no hay favorito claro
        if self.actual ==  1 and self.p_home < UPSET_THRESHOLD:
            return True
        if self.actual == -1 and self.p_away < UPSET_THRESHOLD:
            return True
        return False

    def has_clear_favorite(self) -> bool:
        return min(self.p_home, self.p_away) < UPSET_THRESHOLD


@dataclass
class MetricSet:
    """Conjunto completo de métricas para un segmento de datos."""
    brier:             float = float("nan")
    log_loss:          float = float("nan")
    accuracy:          float = float("nan")   # % resultado correcto
    upset_rate:        float = float("nan")   # % upsets reales
    upset_detect_rate: float = float("nan")   # % upsets que el modelo "anticipó"
    calibration_ece:   float = float("nan")   # Expected Calibration Error
    n_matches:         int   = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BacktestWindow:
    """Resultado de una ventana de backtesting temporal."""
    train_end:  str           # último día de entrenamiento
    test_year:  int
    metrics:    MetricSet
    n_train:    int


@dataclass
class CalibrationBin:
    bin_range:       str
    pred_mean:       float    # probabilidad media predicha en este bin
    obs_rate:        float    # tasa real observada
    n:               int
    abs_error:       float
    is_overconfident: bool    # pred_mean > obs_rate


@dataclass
class CalibrationResult:
    bins:    List[CalibrationBin]
    ece:     float    # Expected Calibration Error (promedio ponderado)
    mce:     float    # Maximum Calibration Error


@dataclass
class ComparisonRow:
    model:             str
    brier:             float
    log_loss:          float
    accuracy:          float
    calibration_error: float
    upset_detect_rate: float
    n_matches:         int


# ──────────────────────────────────────────────────────────────────────────────
# 1. LIVE ELO TRACKER
# ──────────────────────────────────────────────────────────────────────────────

class LiveEloTracker:
    """
    Mantiene ratings ELO actualizados partido a partido.
    Garantía principal: la predicción para el partido N se genera
    ANTES de actualizar con el resultado de N → zero leakage.
    """

    def __init__(self, home_adv: int = HOME_ADVANTAGE):
        self._ratings: Dict[str, float] = defaultdict(lambda: ELO_BASE)
        self._history: List[Dict]       = []   # log completo de actualizaciones
        self.home_adv = home_adv

    def get(self, team: str) -> float:
        return self._ratings[team]

    def snapshot(self) -> Dict[str, float]:
        """Copia inmutable de todos los ratings en este instante."""
        return dict(self._ratings)

    def set_ratings(self, ratings: Dict[str, float]) -> None:
        self._ratings.update(ratings)

    def predict(
        self, home: str, away: str, neutral: bool = False
    ) -> Tuple[float, float, float]:
        """
        Predicción ELO pura: (p_home_win, p_draw, p_away_win).
        """
        r_h   = self._ratings[home]
        r_a   = self._ratings[away]
        bonus = 0 if neutral else self.home_adv

        e_h   = 1.0 / (1.0 + 10.0 ** (-(r_h + bonus - r_a) / 400.0))

        p_draw = np.clip(0.28 - 0.30 * abs(e_h - 0.5), 0.08, 0.30)
        p_home = e_h * (1.0 - p_draw)
        p_away = (1.0 - e_h) * (1.0 - p_draw)

        total  = p_home + p_draw + p_away
        return p_home / total, p_draw / total, p_away / total

    def update(
        self,
        home:       str,
        away:       str,
        g_home:     int,
        g_away:     int,
        tournament: str  = "Friendly",
        neutral:    bool = False,
    ) -> Tuple[float, float]:
        r_h   = self._ratings[home]
        r_a   = self._ratings[away]
        bonus = 0 if neutral else self.home_adv

        e_h   = 1.0 / (1.0 + 10.0 ** (-(r_h + bonus - r_a) / 400.0))
        e_a   = 1.0 - e_h

        if   g_home > g_away: s_h, s_a = 1.0, 0.0
        elif g_home < g_away: s_h, s_a = 0.0, 1.0
        else:                 s_h, s_a = 0.5, 0.5

        k      = self._k(tournament, abs(g_home - g_away))
        new_rh = r_h + k * (s_h - e_h)
        new_ra = r_a + k * (s_a - e_a)

        self._ratings[home] = new_rh
        self._ratings[away] = new_ra

        self._history.append({
            "home": home, "away": away,
            "g_home": g_home, "g_away": g_away,
            "tournament": tournament, "neutral": neutral,
            "k": k,
            "delta_home": round(new_rh - r_h, 3),
            "delta_away": round(new_ra - r_a, 3),
        })

        return new_rh, new_ra

    def bulk_train(self, df: pd.DataFrame) -> None:
        assert "date" in df.columns, "El DataFrame debe tener columna 'date'"
        df_sorted = df.sort_values("date")

        for _, row in df_sorted.iterrows():
            h  = str(row["home_team"])
            a  = str(row["away_team"])
            gh = int(row["home_score"])
            ga = int(row["away_score"])
            t  = str(row.get("tournament", "Friendly"))
            ne = bool(row.get("neutral", False))
            self.update(h, a, gh, ga, t, ne)

    def top_ratings(self, n: int = 20) -> List[Tuple[str, float]]:
        return sorted(self._ratings.items(), key=lambda x: x[1], reverse=True)[:n]

    def validate_no_leakage(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> bool:
        max_train = pd.to_datetime(df_train["date"]).max()
        min_test  = pd.to_datetime(df_test["date"]).min()
        assert max_train < min_test, (
            f"LEAKAGE DETECTADO: train tiene datos hasta {max_train.date()}, "
            f"test empieza en {min_test.date()}"
        )
        return True

    @staticmethod
    def validate_probs(p_home: float, p_draw: float, p_away: float,
                       tol: float = 1e-6) -> bool:
        total = p_home + p_draw + p_away
        assert abs(total - 1.0) < tol, f"Probabilidades suman {total:.8f} ≠ 1.0"
        assert all(0 <= p <= 1 for p in [p_home, p_draw, p_away]), \
            f"Probabilidad fuera de [0,1]: {p_home}, {p_draw}, {p_away}"
        return True

    @staticmethod
    def _k(tournament: str, margin: int) -> float:
        t = str(tournament)
        k_base = K_DEFAULT
        for key, k in TOURNAMENT_K.items():
            if key.lower() in t.lower():
                k_base = k
                break
        multiplier = min(1.0 + margin * 0.10, 2.0)
        return k_base * multiplier


# ──────────────────────────────────────────────────────────────────────────────
# 2. PREDICTION EVALUATOR
# ──────────────────────────────────────────────────────────────────────────────

class PredictionEvaluator:
    """
    Evaluación temporal estricta del modelo predictivo y su calibración.
    """

    def __init__(
        self,
        csv_path:  str  = "results.csv",
        predictor        = None,
        cache_dir: str  = ".eval_cache",
    ):
        self.csv_path  = csv_path
        self.predictor = predictor
        self.cache_dir = cache_dir
        self._df: Optional[pd.DataFrame] = None

        os.makedirs(cache_dir, exist_ok=True)

    def _load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"No encontrado: '{self.csv_path}'")

        df = pd.read_csv(self.csv_path, parse_dates=["date"])
        df["home_team"] = df["home_team"].replace(NAME_MAP)
        df["away_team"] = df["away_team"].replace(NAME_MAP)

        df = df.dropna(subset=["home_score", "away_score"])
        df["home_score"] = df["home_score"].astype(int)
        df["away_score"] = df["away_score"].astype(int)
        df["neutral"]    = df.get("neutral", pd.Series(False, index=df.index)).astype(bool)
        df = df.sort_values("date").reset_index(drop=True)

        self._df = df
        return df

    def _build_records(
        self,
        df_eval:   pd.DataFrame,
        tracker:   LiveEloTracker,
        model_tag: str = "elo",
        use_predictor: bool = False,
        calibrator = None
    ) -> List[MatchRecord]:
        records: List[MatchRecord] = []

        for _, row in df_eval.iterrows():
            home  = str(row["home_team"])
            away  = str(row["away_team"])
            g_h   = int(row["home_score"])
            g_a   = int(row["away_score"])
            tourn = str(row.get("tournament", "Friendly"))
            neut  = bool(row.get("neutral", False))
            date  = row["date"]

            actual = 1 if g_h > g_a else (-1 if g_h < g_a else 0)
            elo_diff = tracker.get(home) - tracker.get(away)

            if use_predictor and self.predictor is not None:
                try:
                    pred = self.predictor.predict_match(home, away, "group", neutral_venue=neut)
                    p_h = pred["team1_win"]
                    p_d = pred["draw"]
                    p_a = pred["team2_win"]
                except Exception:
                    p_h, p_d, p_a = tracker.predict(home, away, neut)
            else:
                p_h, p_d, p_a = tracker.predict(home, away, neut)

            # Si hay calibrador, procesar las probabilidades
            if calibrator is not None:
                p_cal = calibrator.calibrate(np.array([[p_h, p_d, p_a]]))[0]
                p_h, p_d, p_a = float(p_cal[0]), float(p_cal[1]), float(p_cal[2])

            LiveEloTracker.validate_probs(p_h, p_d, p_a)

            records.append(MatchRecord(
                date=str(date.date()), home=home, away=away,
                tournament=tourn, year=date.year,
                p_home=p_h, p_draw=p_d, p_away=p_a,
                actual=actual, g_home=g_h, g_away=g_a,
                elo_diff=elo_diff, neutral=neut, model_tag=model_tag,
            ))

            tracker.update(home, away, g_h, g_a, tourn, neut)

        return records

    @staticmethod
    def multiclass_brier_score(records: List[MatchRecord]) -> float:
        if not records:
            return float("nan")
        probs   = np.array([r.probs()   for r in records])
        onehots = np.array([r.one_hot() for r in records])
        return float(np.mean(np.sum((probs - onehots) ** 2, axis=1)))

    @staticmethod
    def multiclass_log_loss(records: List[MatchRecord]) -> float:
        if not records:
            return float("nan")
        probs   = np.clip(np.array([r.probs()   for r in records]), LOG_CLIP, 1.0)
        onehots = np.array([r.one_hot() for r in records])
        return float(-np.mean(np.sum(onehots * np.log(probs), axis=1)))

    @staticmethod
    def accuracy_top_prediction(records: List[MatchRecord]) -> float:
        if not records:
            return float("nan")
        return float(np.mean([r.is_correct() for r in records])) * 100

    @staticmethod
    def upset_detection_rate(records: List[MatchRecord]) -> Tuple[float, float]:
        asymmetric = [r for r in records if r.has_clear_favorite()]
        if not asymmetric:
            return float("nan"), float("nan")

        upsets = [r for r in asymmetric if r.is_upset()]
        upset_rate = len(upsets) / len(asymmetric) * 100

        detected = sum(
            1 for r in upsets
            if min(r.p_home, r.p_away) >= 0.20
        )
        detect_rate = (detected / len(upsets) * 100) if upsets else 0.0

        return round(upset_rate, 2), round(detect_rate, 2)

    @staticmethod
    def calibration_error(records: List[MatchRecord]) -> float:
        cal = CalibrationAnalyzer.analyze(records)
        return cal.ece

    @staticmethod
    def compute_metrics(records: List[MatchRecord]) -> MetricSet:
        if not records:
            return MetricSet()

        brier   = PredictionEvaluator.multiclass_brier_score(records)
        ll      = PredictionEvaluator.multiclass_log_loss(records)
        acc     = PredictionEvaluator.accuracy_top_prediction(records)
        u_rate, u_detect = PredictionEvaluator.upset_detection_rate(records)
        ece     = PredictionEvaluator.calibration_error(records)

        return MetricSet(
            brier             = round(brier,    4),
            log_loss          = round(ll,       4),
            accuracy          = round(acc,      2),
            upset_rate        = round(u_rate,   2),
            upset_detect_rate = round(u_detect, 2),
            calibration_ece   = round(ece,      4),
            n_matches         = len(records),
        )

    # ── Validación Temporal Estricta: Temperature Scaling ────────────────────
    
    def evaluate_calibrated_predictions(
        self,
        train_end: str = "2020-12-31",
        cal_end: str   = "2022-12-31",
        test_end: str  = "2025-12-31",
        export: bool   = True
    ) -> Dict:
        """
        Ejecuta una evaluación temporal estricta de Temperature Scaling:
          1. Entrena ratings ELO con partidos hasta train_end (Entrenamiento ELO)
          2. Genera predicciones para cal_end (Fase de Calibración) para ajustar T
          3. Evalúa y compara ELO Puro vs ELO Calibrado en test_end (Fase de Test)
        """
        df = self._load()
        
        t_end = pd.Timestamp(train_end)
        c_end = pd.Timestamp(cal_end)
        te_end = pd.Timestamp(test_end)
        
        df_train = df[df["date"] <= t_end].copy()
        df_cal   = df[(df["date"] > t_end) & (df["date"] <= c_end)].copy()
        df_test  = df[(df["date"] > c_end) & (df["date"] <= te_end)].copy()
        
        print("\n" + "=" * 60)
        print(" VALIDACION TEMPORAL ESTRICTA: TEMPERATURE SCALING")
        print("=" * 60)
        print(f"  Fase ELO Train (<= {train_end}) : {len(df_train):,} partidos")
        print(f"  Fase Calibrar  (2021-2022)       : {len(df_cal):,} partidos")
        print(f"  Fase Test      (2023-2025)       : {len(df_test):,} partidos")
        print("-" * 60)
        
        # 1. Entrenar ELO Tracker hasta el fin de la fase de entrenamiento
        tracker = LiveEloTracker()
        tracker.bulk_train(df_train)
        
        # 2. Generar predicciones para el set de calibración
        cal_records = self._build_records(df_cal, tracker, model_tag="elo_raw")
        
        cal_probs = np.array([r.probs() for r in cal_records])
        cal_actuals = np.array([r.actual for r in cal_records])
        
        # Ajustar Temperature Scaling
        calibrator = TemperatureScalingCalibrator()
        calibrator.fit(cal_probs, cal_actuals)
        fitted_T = calibrator.temperature
        
        # 3. Evaluar sobre el conjunto de test (2023-2025)
        tracker_test = LiveEloTracker()
        tracker_test.set_ratings(tracker.snapshot())
        
        test_records_raw = []
        test_records_cal = []
        
        for _, row in df_test.iterrows():
            h  = str(row["home_team"])
            a  = str(row["away_team"])
            gh = int(row["home_score"])
            ga = int(row["away_score"])
            t  = str(row.get("tournament", "Friendly"))
            ne = bool(row.get("neutral", False))
            dt = row["date"]
            
            actual = 1 if gh > ga else (-1 if gh < ga else 0)
            elo_diff = tracker_test.get(h) - tracker_test.get(a)
            
            # Predicción Raw ELO
            p_h, p_d, p_a = tracker_test.predict(h, a, ne)
            
            # Calibración
            p_cal = calibrator.calibrate(np.array([[p_h, p_d, p_a]]))[0]
            
            test_records_raw.append(MatchRecord(
                date=str(dt.date()), home=h, away=a, tournament=t, year=dt.year,
                p_home=p_h, p_draw=p_d, p_away=p_a, actual=actual,
                g_home=gh, g_away=ga, elo_diff=elo_diff, neutral=ne, model_tag="elo_raw"
            ))
            
            test_records_cal.append(MatchRecord(
                date=str(dt.date()), home=h, away=a, tournament=t, year=dt.year,
                p_home=p_cal[0], p_draw=p_cal[1], p_away=p_cal[2], actual=actual,
                g_home=gh, g_away=ga, elo_diff=elo_diff, neutral=ne, model_tag="elo_calibrated"
            ))
            
            tracker_test.update(h, a, gh, ga, t, ne)
            
        # 4. Calcular métricas
        metrics_raw = self.compute_metrics(test_records_raw)
        metrics_cal = self.compute_metrics(test_records_cal)
        
        # Ajustar errores de calibración específicos
        cal_analyzer_raw = CalibrationAnalyzer.analyze(test_records_raw, outcome="any_win")
        cal_analyzer_cal = CalibrationAnalyzer.analyze(test_records_cal, outcome="any_win")
        metrics_raw.calibration_ece = cal_analyzer_raw.ece
        metrics_cal.calibration_ece = cal_analyzer_cal.ece
        
        # 5. Análisis de Overconfidence por Buckets
        bucket_stats = self._analyze_overconfidence_buckets(test_records_raw, test_records_cal)
        
        # 6. Análisis de Draws
        draw_stats = self._analyze_draw_calibration(test_records_raw, test_records_cal)
        
        # 7. Tabla Comparativa
        print("\n RESULTADOS DEL BENCHMARK (RAW vs CALIBRATED):")
        print(f"{'Modelo':<20} {'Brier':>7} {'LogLoss':>8} {'ECE':>7} {'MCE':>7} {'Accuracy':>9}")
        print("-" * 63)
        print(f"{'ELO Raw':<20} {metrics_raw.brier:>7.4f} {metrics_raw.log_loss:>8.4f} "
              f"{cal_analyzer_raw.ece:>7.4f} {cal_analyzer_raw.mce:>7.4f} {metrics_raw.accuracy:>8.1f}%")
        print(f"{'ELO + TempScaling':<20} {metrics_cal.brier:>7.4f} {metrics_cal.log_loss:>8.4f} "
              f"{cal_analyzer_cal.ece:>7.4f} {cal_analyzer_cal.mce:>7.4f} {metrics_cal.accuracy:>8.1f}%")
        
        results_summary = {
            "metadata": {
                "train_end": train_end,
                "cal_end": cal_end,
                "test_end": test_end,
                "fitted_temperature": fitted_T,
                "n_test_matches": len(df_test)
            },
            "metrics": {
                "raw": {**asdict(metrics_raw), "mce": cal_analyzer_raw.mce},
                "calibrated": {**asdict(metrics_cal), "mce": cal_analyzer_cal.mce}
            },
            "buckets": bucket_stats,
            "draw_analysis": draw_stats
        }
        
        # Guardar Persistencia
        if export:
            # Guardar modelo calibrador
            calibrator.save("temperature_calibrator.pkl")
            
            # Guardar reporte JSON
            with open("calibration_benchmark_report.json", "w", encoding="utf-8") as f:
                json.dump(results_summary, f, ensure_ascii=False, indent=2)
            print("\n Reporte persistido en 'calibration_benchmark_report.json'")
            print(" Calibrador guardado en 'temperature_calibrator.pkl'")
            
        return results_summary

    @staticmethod
    def _analyze_overconfidence_buckets(
        raw_recs: List[MatchRecord], 
        cal_recs: List[MatchRecord]
    ) -> Dict:
        """
        Analiza predicciones y overconfidence para buckets específicos:
          - Favoritos > 70%
          - Favoritos > 80%
          - Underdogs < 25%
        """
        buckets = {
            "fav_70": {"threshold": 0.70, "type": "favorite"},
            "fav_80": {"threshold": 0.80, "type": "favorite"},
            "under_25": {"threshold": 0.25, "type": "underdog"}
        }
        
        stats = {}
        for name, conf in buckets.items():
            t = conf["threshold"]
            b_type = conf["type"]
            
            raw_probs = []
            cal_probs = []
            outcomes = []
            
            for r_raw, r_cal in zip(raw_recs, cal_recs):
                if b_type == "favorite":
                    if r_raw.p_home > t:
                        raw_probs.append(r_raw.p_home)
                        cal_probs.append(r_cal.p_home)
                        outcomes.append(1 if r_raw.actual == 1 else 0)
                    elif r_raw.p_away > t:
                        raw_probs.append(r_raw.p_away)
                        cal_probs.append(r_cal.p_away)
                        outcomes.append(1 if r_raw.actual == -1 else 0)
                else:  # underdog < 25%
                    if r_raw.p_home < t:
                        raw_probs.append(r_raw.p_home)
                        cal_probs.append(r_cal.p_home)
                        outcomes.append(1 if r_raw.actual == 1 else 0)
                    if r_raw.p_away < t:
                        raw_probs.append(r_raw.p_away)
                        cal_probs.append(r_cal.p_away)
                        outcomes.append(1 if r_raw.actual == -1 else 0)
                        
            n = len(raw_probs)
            if n > 0:
                avg_raw = float(np.mean(raw_probs))
                avg_cal = float(np.mean(cal_probs))
                actual_rate = float(np.mean(outcomes))
                raw_gap = avg_raw - actual_rate
                cal_gap = avg_cal - actual_rate
                
                stats[name] = {
                    "n_matches": n,
                    "avg_raw": round(avg_raw, 4),
                    "avg_cal": round(avg_cal, 4),
                    "actual_rate": round(actual_rate, 4),
                    "raw_overconfidence_gap": round(raw_gap, 4),
                    "cal_overconfidence_gap": round(cal_gap, 4)
                }
            else:
                stats[name] = {"n_matches": 0, "avg_raw": 0, "avg_cal": 0, "actual_rate": 0, "raw_overconfidence_gap": 0, "cal_overconfidence_gap": 0}
                
        return stats

    @staticmethod
    def _analyze_draw_calibration(
        raw_recs: List[MatchRecord], 
        cal_recs: List[MatchRecord]
    ) -> Dict:
        """Analiza la calibración específica de empates."""
        raw_draw = CalibrationAnalyzer.analyze(raw_recs, outcome="draw")
        cal_draw = CalibrationAnalyzer.analyze(cal_recs, outcome="draw")
        
        return {
            "raw_draw_ece": raw_draw.ece,
            "raw_draw_mce": raw_draw.mce,
            "cal_draw_ece": cal_draw.ece,
            "cal_draw_mce": cal_draw.mce,
            "bins_raw": [asdict(b) for b in raw_draw.bins],
            "bins_cal": [asdict(b) for b in cal_draw.bins]
        }

    # ── Métodos de Caché e Históricos ──────────────────────────────────────────

    def rolling_backtest(
        self,
        train_from:    str        = "2010-01-01",
        test_years:    List[int]  = None,
        model_tag:     str        = "elo",
        use_predictor: bool       = False,
        use_cache:     bool       = True,
    ) -> List[BacktestWindow]:
        if test_years is None:
            test_years = list(range(2019, 2026))

        cache_key = self._cache_key(train_from, test_years, model_tag)
        if use_cache:
            cached = self._load_cache(cache_key)
            if cached is not None:
                return cached

        df = self._load()
        df = df[df["date"] >= pd.Timestamp(train_from)].copy()

        tracker   = LiveEloTracker()
        results: List[BacktestWindow] = []
        first_year = min(test_years)

        df_pre = df[df["date"] < pd.Timestamp(f"{first_year}-01-01")]
        tracker.bulk_train(df_pre)

        for year in sorted(test_years):
            df_test = df[
                (df["date"] >= pd.Timestamp(f"{year}-01-01")) &
                (df["date"] <  pd.Timestamp(f"{year+1}-01-01"))
            ].copy()

            df_train_so_far = df[df["date"] < pd.Timestamp(f"{year}-01-01")]

            if len(df_test) < MIN_MATCHES_EVAL:
                continue

            if len(df_train_so_far) > 0:
                tracker.validate_no_leakage(df_train_so_far, df_test)

            records  = self._build_records(df_test, tracker, model_tag, use_predictor)
            metrics  = self.compute_metrics(records)

            results.append(BacktestWindow(
                train_end = f"{year-1}-12-31",
                test_year = year,
                metrics   = metrics,
                n_train   = len(df_train_so_far),
            ))

        if use_cache:
            self._save_cache(cache_key, results)

        return results

    def evaluate_by_tournament(
        self,
        records: List[MatchRecord],
        min_matches: int = MIN_MATCHES_EVAL,
    ) -> Dict[str, MetricSet]:
        by_tourn: Dict[str, List[MatchRecord]] = defaultdict(list)
        for r in records:
            by_tourn[r.tournament[:35]].append(r)

        return {
            tourn: self.compute_metrics(recs)
            for tourn, recs in sorted(by_tourn.items())
            if len(recs) >= min_matches
        }

    def evaluate_by_year(
        self,
        records: List[MatchRecord],
    ) -> Dict[int, MetricSet]:
        by_year: Dict[int, List[MatchRecord]] = defaultdict(list)
        for r in records:
            by_year[r.year].append(r)

        return {
            year: self.compute_metrics(recs)
            for year, recs in sorted(by_year.items())
            if len(recs) >= MIN_MATCHES_EVAL
        }

    def evaluate_match_predictions(
        self,
        from_date:     str  = "2018-01-01",
        model_tag:     str  = "elo",
        use_predictor: bool = False,
        use_cache:     bool = True,
    ) -> List[MatchRecord]:
        cache_key = self._cache_key(from_date, [], model_tag + "_records")
        if use_cache:
            cached = self._load_cache(cache_key)
            if cached is not None:
                return cached

        df       = self._load()
        df_pre   = df[df["date"] < pd.Timestamp(from_date)]
        df_eval  = df[df["date"] >= pd.Timestamp(from_date)].copy()

        tracker  = LiveEloTracker()
        tracker.bulk_train(df_pre)

        records = self._build_records(df_eval, tracker, model_tag, use_predictor)

        if use_cache:
            self._save_cache(cache_key, records)

        return records

    def _cache_key(self, *args) -> str:
        payload = json.dumps(args, default=str)
        return hashlib.md5(payload.encode()).hexdigest()

    def _cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.pkl")

    def _save_cache(self, key: str, data) -> None:
        with open(self._cache_path(key), "wb") as f:
            pickle.dump(data, f)

    def _load_cache(self, key: str):
        path = self._cache_path(key)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def save_parquet(self, records: List[MatchRecord], path: str = "eval_records.parquet") -> str:
        df = pd.DataFrame([asdict(r) for r in records])
        df.to_parquet(path, index=False)
        return path


# ──────────────────────────────────────────────────────────────────────────────
# 3. CALIBRATION ANALYZER
# ──────────────────────────────────────────────────────────────────────────────

class CalibrationAnalyzer:
    """
    Análisis de calibración: expected probability vs observed frequency.
    """

    @staticmethod
    def analyze(
        records:  List[MatchRecord],
        n_bins:   int  = N_CALIBRATION_BINS,
        outcome:  str  = "home_win",
    ) -> CalibrationResult:
        if not records:
            return CalibrationResult(bins=[], ece=float("nan"), mce=float("nan"))

        if outcome == "home_win":
            preds    = np.array([r.p_home   for r in records])
            actuals  = np.array([1 if r.actual ==  1 else 0 for r in records])
        elif outcome == "draw":
            preds    = np.array([r.p_draw   for r in records])
            actuals  = np.array([1 if r.actual ==  0 else 0 for r in records])
        elif outcome == "away_win":
            preds    = np.array([r.p_away   for r in records])
            actuals  = np.array([1 if r.actual == -1 else 0 for r in records])
        else:  # any_win
            preds   = np.concatenate([
                [r.p_home for r in records],
                [r.p_away for r in records],
            ])
            actuals = np.concatenate([
                [1 if r.actual ==  1 else 0 for r in records],
                [1 if r.actual == -1 else 0 for r in records],
            ])

        bin_edges  = np.linspace(0.0, 1.0, n_bins + 1)
        bins_data: List[CalibrationBin] = []
        n_total    = len(preds)
        ece_accum  = 0.0
        mce        = 0.0

        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask   = (preds >= lo) & (preds < hi) if i < n_bins - 1 else (preds >= lo) & (preds <= hi)
            n      = int(mask.sum())

            if n < 3:
                continue

            pred_mean = float(preds[mask].mean())
            obs_rate  = float(actuals[mask].mean())
            err       = abs(pred_mean - obs_rate)

            bins_data.append(CalibrationBin(
                bin_range        = f"{lo:.1f}–{hi:.1f}",
                pred_mean        = round(pred_mean, 4),
                obs_rate         = round(obs_rate,  4),
                n                = n,
                abs_error        = round(err, 4),
                is_overconfident = pred_mean > obs_rate,
            ))

            ece_accum += (n / n_total) * err
            mce        = max(mce, err)

        return CalibrationResult(
            bins = bins_data,
            ece  = round(ece_accum, 4),
            mce  = round(mce, 4),
        )

    @staticmethod
    def print_table(cal: CalibrationResult, title: str = "Calibración") -> None:
        print(f"\n📊 {title}")
        print(f"{'Bin':<12} {'P̄_pred':>8} {'P̄_obs':>8} {'N':>6} {'Error':>7}  Estado")
        print("─" * 55)
        for b in cal.bins:
            status = "✅ OK" if b.abs_error < 0.04 else ("↑ sobreconf." if b.is_overconfident else "↓ subconf.")
            print(f"{b.bin_range:<12} {b.pred_mean:>8.3f} {b.obs_rate:>8.3f} "
                  f"{b.n:>6} {b.abs_error:>7.3f}  {status}")
        grade = "🟢" if cal.ece < 0.04 else ("🟡" if cal.ece < 0.07 else "🔴")
        print(f"\n  ECE={cal.ece:.4f} {grade}  MCE={cal.mce:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. MODEL COMPARISON
# ──────────────────────────────────────────────────────────────────────────────

class ModelComparison:
    """
    Compara múltiples modelos sobre el mismo conjunto de evaluación.
    """

    def __init__(
        self,
        evaluator:  PredictionEvaluator,
        from_date:  str = "2020-01-01",
        test_years: List[int] = None,
    ):
        self.evaluator  = evaluator
        self.from_date  = from_date
        self.test_years = test_years or list(range(2021, 2026))
        self._results:  Dict[str, List[BacktestWindow]] = {}

    def run(
        self,
        models:    List[str] = None,
        use_cache: bool      = True,
    ) -> List[ComparisonRow]:
        models = models or ["elo"]
        rows: List[ComparisonRow] = []

        for model in models:
            tag_label = "ELO puro" if model == "elo" else model
            windows = self.evaluator.rolling_backtest(
                train_from    = self.from_date,
                test_years    = self.test_years,
                model_tag     = model,
                use_predictor = False,
                use_cache     = use_cache,
            )
            self._results[model] = windows

            if not windows:
                continue

            avg = MetricSet(
                brier             = round(np.mean([w.metrics.brier             for w in windows]), 4),
                log_loss          = round(np.mean([w.metrics.log_loss          for w in windows]), 4),
                accuracy          = round(np.mean([w.metrics.accuracy          for w in windows]), 2),
                upset_rate        = round(np.mean([w.metrics.upset_rate        for w in windows]), 2),
                upset_detect_rate = round(np.mean([w.metrics.upset_detect_rate for w in windows]), 2),
                calibration_ece   = round(np.mean([w.metrics.calibration_ece   for w in windows]), 4),
                n_matches         = sum(w.metrics.n_matches for w in windows),
            )

            rows.append(ComparisonRow(
                model             = tag_label,
                brier             = avg.brier,
                log_loss          = avg.log_loss,
                accuracy          = avg.accuracy,
                calibration_error = avg.calibration_ece,
                upset_detect_rate = avg.upset_detect_rate,
                n_matches         = avg.n_matches,
            ))

        rows.sort(key=lambda r: r.brier)
        return rows


# ──────────────────────────────────────────────────────────────────────────────
# REPORTE COMPLETO
# ──────────────────────────────────────────────────────────────────────────────

def run_full_evaluation(
    csv_path:   str       = "results.csv",
    predictor             = None,
    from_date:  str       = "2018-01-01",
    test_years: List[int] = None,
    export:     bool      = True,
    use_cache:  bool      = True,
) -> Dict:
    evaluator = PredictionEvaluator(csv_path, predictor, cache_dir=".eval_cache")

    # 1. Ejecutar la validación temporal de Temperature Scaling
    results = evaluator.evaluate_calibrated_predictions(
        train_end="2020-12-31",
        cal_end="2022-12-31",
        test_end="2025-12-31",
        export=export
    )
    
    return results


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT STANDALONE
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluador científico FIFA 2026")
    parser.add_argument("--csv",       default="results.csv", help="Ruta al CSV")
    parser.add_argument("--from",     dest="from_date", default="2018-01-01")
    args = parser.parse_args()

    run_full_evaluation(
        csv_path   = args.csv,
        from_date  = args.from_date,
        use_cache  = True,
        export     = True,
    )
