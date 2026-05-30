"""
DRAW CALIBRATION BENCHMARK - FIFA World Cup 2026 Predictor
===========================================================

Evalúa diferentes DRAW_CALIBRATION_FACTOR en histórico:
- 1.20
- 1.25
- 1.30
- 1.35
- 1.40

Reporta: Factor, Accuracy, Log Loss, Brier
Selecciona automáticamente el mejor Log Loss.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List

from core.predictor import WorldCupPredictor

print("=" * 80)
print("DRAW CALIBRATION BENCHMARK")
print("=" * 80)
print(f"Inicio: {datetime.now()}")
print()

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
SEED = 42
FACTORS = [1.20, 1.25, 1.30, 1.35, 1.40]

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
# SPLIT TEMPORAL
# =============================================================================
# Usar últimos 2 años como test (2022-2023)
train_cutoff = pd.Timestamp("2022-01-01")
df_train = df[df["date"] < train_cutoff].copy()
df_test = df[df["date"] >= train_cutoff].copy()

print(f"  Train: {len(df_train)} partidos (antes de {train_cutoff})")
print(f"  Test: {len(df_test)} partidos (desde {train_cutoff})")
print()

# =============================================================================
# FUNCIONES DE MÉTRICAS
# =============================================================================
def compute_metrics(probs, actuals):
    """Calcula Accuracy, Log Loss, Brier."""
    probs = np.array(probs, dtype=np.float64)
    actuals = np.array(actuals, dtype=np.float64)
    
    # Clip probabilidades para evitar log(0)
    probs = np.clip(probs, 1e-15, 1 - 1e-15)
    
    # Accuracy
    pred_classes = np.argmax(probs, axis=1)
    actual_classes = np.argmax(actuals, axis=1)
    accuracy = np.mean(pred_classes == actual_classes)
    
    # Log Loss
    log_loss = -np.mean(np.sum(actuals * np.log(probs), axis=1))
    
    # Brier
    brier = np.mean(np.sum((probs - actuals) ** 2, axis=1))
    
    return {
        "accuracy": float(accuracy),
        "log_loss": float(log_loss),
        "brier": float(brier),
    }

# =============================================================================
# BENCHMARK
# =============================================================================
print("Ejecutando benchmark...")
print()
print(f"{'Factor':>8} {'Accuracy':>10} {'Log Loss':>10} {'Brier':>10}")
print("-" * 50)

results = []

for factor in FACTORS:
    print(f"{factor:>8.2f}", end=" ", flush=True)
    
    try:
        # Crear predictor con factor específico
        predictor = WorldCupPredictor(seed=SEED, draw_calibration_factor=factor)
        
        # Entrenar
        predictor.fit(df_train)
        
        # Predecir
        probs = []
        actuals = []
        for _, row in df_test.iterrows():
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
        
        # Calcular métricas
        metrics = compute_metrics(probs, actuals)
        results.append({
            "factor": factor,
            "accuracy": metrics["accuracy"],
            "log_loss": metrics["log_loss"],
            "brier": metrics["brier"],
        })
        
        print(f"{metrics['accuracy']:>10.4f} {metrics['log_loss']:>10.4f} {metrics['brier']:>10.4f}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        continue

print()

# =============================================================================
# SELECCIONAR MEJOR LOG LOSS
# =============================================================================
if results:
    print("=" * 50)
    print("RESULTADOS")
    print("=" * 50)
    print()
    
    for r in results:
        print(f"Factor {r['factor']:.2f}: Accuracy={r['accuracy']:.4f}, Log Loss={r['log_loss']:.4f}, Brier={r['brier']:.4f}")
    
    print()
    
    # Mejor Log Loss
    best = min(results, key=lambda x: x["log_loss"])
    print("=" * 50)
    print("MEJOR LOG LOSS")
    print("=" * 50)
    print(f"Factor: {best['factor']:.2f}")
    print(f"Accuracy: {best['accuracy']:.4f}")
    print(f"Log Loss: {best['log_loss']:.4f}")
    print(f"Brier: {best['brier']:.4f}")
    print()
    
    # Guardar resultados
    df_results = pd.DataFrame(results)
    df_results.to_csv('draw_calibration_results.csv', index=False)
    print("Resultados guardados: draw_calibration_results.csv")
else:
    print("No se obtuvieron resultados.")

print()
print("=" * 80)
print(f"Fin: {datetime.now()}")
print("=" * 80)
