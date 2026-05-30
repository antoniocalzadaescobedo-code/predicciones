"""
PROTOCOLLO DE CORRECCIÓN - FASE 1: ML NATIVO MULTICLASE
========================================================

Entrena directamente en el espacio {away_win: -1, draw: 0, home_win: 1}.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import sys
sys.path.insert(0, 'C:\\Proyecto_FIFA')

print("=" * 80)
print("FASE 1: ML NATIVO MULTICLASE")
print("=" * 80)
print()

# =============================================================================
# CARGAR DATOS
# =============================================================================

print("Cargando datos...")
df = pd.read_csv("results.csv", parse_dates=["date"])
df = df.dropna(subset=["home_score", "away_score"])
df["home_score"] = df["home_score"].astype(int)
df["away_score"] = df["away_score"].astype(int)
df["neutral"] = df.get("neutral", pd.Series(False, index=df.index)).astype(bool)

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

# Filtrar datos recientes
df = df[df["date"] >= "2010-01-01"].copy()
df = df.sort_values("date").reset_index(drop=True)

print(f"Total partidos: {len(df)}")
print()

# =============================================================================
# AGREGAR FEATURES WALK-FORWARD
# =============================================================================

def _form_from_history(points, n_matches=10):
    recent = np.asarray(points[-n_matches:], dtype=float)
    if recent.size == 0:
        return 0.5
    decay = np.exp(-0.1 * np.arange(recent.size - 1, -1, -1))
    return float(np.average(recent, weights=decay))

print("Agregando features walk-forward...")

from core.predictor import EloTracker

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
        outcome = 1  # home_win
    elif g_home < g_away:
        outcome = -1  # away_win
    else:
        outcome = 0  # draw
    
    rows.append({
        "elo_diff": (r_home + bonus) - r_away,
        "is_neutral": int(neutral),
        "form_home": _form_from_history(form_history[home]),
        "form_away": _form_from_history(form_history[away]),
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
df = pd.concat([df, feature_df], axis=1)

print(f"Features agregados: {len(feature_df.columns)}")
print()

# =============================================================================
# SPLIT TEMPORAL WALK-FORWARD
# =============================================================================

print("Split temporal walk-forward...")
df_train = df[df["date"] < pd.Timestamp("2022-12-31")].copy()
df_test = df[df["date"] >= pd.Timestamp("2023-01-01")].copy()

print(f"Train: {len(df_train)} partidos")
print(f"Test: {len(df_test)} partidos")
print()

# =============================================================================
# FASE 1: ML NATIVO MULTICLASE
# =============================================================================

print("=" * 80)
print("FASE 1: ML NATIVO MULTICLASE")
print("=" * 80)
print()

# 1. Target multiclase
print("1. Target multiclase...")
y_multiclass = df_train['outcome'].values
print(f"   y_multiclass shape: {y_multiclass.shape}")
print(f"   Valores únicos: {np.unique(y_multiclass)}")
print()

# 2. Modelo nativo 3-clase
print("2. Modelo nativo 3-clase...")
features = ["elo_diff", "is_neutral", "form_home", "form_away", "h2h"]
X_train = df_train[features].fillna(0.5)
X_test = df_test[features].fillna(0.5)

rf_mc = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight='balanced',  # crucial para draws
    random_state=42,
    n_jobs=-1
)

print("   Entrenando Random Forest multiclase...")
rf_mc.fit(X_train, y_multiclass)
print("   Entrenamiento completado")
print()

# 3. Validación temporal walk-forward (ya aplicada)
print("3. Validación temporal walk-forward...")
print("   [OK] Train < 2022-12-31, Test >= 2023-01-01")
print()

# 4. Verificar salida
print("4. Verificar salida predict_proba()...")
proba = rf_mc.predict_proba(X_test)
print(f"   Shape: {proba.shape}")
print(f"   Clases: {rf_mc.classes_}")
print(f"   Suma probs fila 0: {proba[0].sum():.6f}")
print()

# Validación obligatoria
print("Validación obligatoria:")

try:
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5), "Probabilidades no suman 1"
    print("   [OK] Probabilidades suman 1.0")
except AssertionError as e:
    print(f"   [ERROR] {e}")

try:
    assert set(rf_mc.classes_) == {-1, 0, 1}, "Mapeo de clases incorrecto"
    print("   [OK] Mapeo de clases correcto: {-1, 0, 1}")
except AssertionError as e:
    print(f"   [ERROR] {e}")

print()

# =============================================================================
# EVALUACIÓN
# =============================================================================

print("=" * 80)
print("EVALUACIÓN")
print("=" * 80)
print()

# Predicciones
y_pred = rf_mc.predict(X_test)
y_test = df_test['outcome'].values

# Accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.4f}")
print()

# Distribución de predicciones
print("Distribución de predicciones:")
pred_counts = np.unique(y_pred, return_counts=True)
for label, count in zip(pred_counts[0], pred_counts[1]):
    print(f"   {label}: {count} ({count/len(y_pred)*100:.2f}%)")
print()

print("Distribución de labels reales:")
test_counts = np.unique(y_test, return_counts=True)
for label, count in zip(test_counts[0], test_counts[1]):
    print(f"   {label}: {count} ({count/len(y_test)*100:.2f}%)")
print()

# =============================================================================
# COMPARACIÓN CON OTROS MODELOS
# =============================================================================

print("=" * 80)
print("COMPARACIÓN CON OTROS MODELOS")
print("=" * 80)
print()

# Gradient Boosting
print("Gradient Boosting multiclase...")
gb_mc = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
)
gb_mc.fit(X_train, y_multiclass)
y_pred_gb = gb_mc.predict(X_test)
accuracy_gb = np.mean(y_pred_gb == y_test)
print(f"   Accuracy: {accuracy_gb:.4f}")
print()

# Logistic Regression
print("Logistic Regression multiclase...")
lr_mc = LogisticRegression(
    class_weight='balanced',
    random_state=42,
    max_iter=1000
)
lr_mc.fit(X_train, y_multiclass)
y_pred_lr = lr_mc.predict(X_test)
accuracy_lr = np.mean(y_pred_lr == y_test)
print(f"   Accuracy: {accuracy_lr:.4f}")
print()

print("Resumen:")
print(f"   Random Forest: {accuracy:.4f}")
print(f"   Gradient Boosting: {accuracy_gb:.4f}")
print(f"   Logistic Regression: {accuracy_lr:.4f}")
print()

print("=" * 80)
print("FASE 1 COMPLETADA")
print("=" * 80)
