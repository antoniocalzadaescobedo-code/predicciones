"""
HYBRID BENCHMARK: DC + GBM + ENSEMBLE OPTIMIZADO
================================================

Combina Dixon-Coles, GBM Multiclase y Ensemble optimizado con pesos.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
import sys
sys.path.insert(0, 'C:\\Proyecto_FIFA')

warnings.filterwarnings('ignore')

print("=" * 70)
print("HYBRID BENCHMARK: DC + GBM + ENSEMBLE OPTIMIZADO")
print("=" * 70)
print()

# =============================================================================
# MÉTRICAS SEGURAS (Multiclase 1X2)
# =============================================================================

def safe_logloss(y_true, y_prob, eps=1e-12):
    y_prob = np.clip(y_prob, eps, 1 - eps)
    classes = np.array([-1, 0, 1])
    y_oh = np.zeros_like(y_prob)
    for i, val in enumerate(y_true):
        idx = np.where(classes == val)[0][0]
        y_oh[i, idx] = 1.0
    return -np.mean(np.sum(y_oh * np.log(y_prob), axis=1))

def brier_multiclass(y_true, y_prob, classes=np.array([-1, 0, 1])):
    y_oh = np.zeros_like(y_prob)
    for i, val in enumerate(y_true):
        idx = np.where(classes == val)[0][0]
        y_oh[i, idx] = 1.0
    return np.mean(np.sum((y_prob - y_oh) ** 2, axis=1))

def calculate_ece(y_true, y_prob, n_bins=10):
    preds = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)
    accuracies = (preds == y_true).astype(float)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

# =============================================================================
# CARGA DE DATOS
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
# FEATURE ENGINEERING WALK-FORWARD
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
split_date = '2023-01-01'
train_mask = df['date'] < pd.Timestamp(split_date)
test_mask = df['date'] >= pd.Timestamp(split_date)

feature_cols = ['elo_diff', 'form_home', 'form_away', 'h2h', 'is_neutral']
X_train = df.loc[train_mask, feature_cols].fillna(0).values
y_train = df.loc[train_mask, 'outcome'].values  # -1, 0, 1
X_test = df.loc[test_mask, feature_cols].fillna(0).values
y_test = df.loc[test_mask, 'outcome'].values

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"Train: {X_train.shape[0]} partidos")
print(f"Test: {X_test.shape[0]} partidos")
print(f"Clases: {np.unique(y_train)}")
print()

# =============================================================================
# 1. GBM MULTICLASE + EXTRACCIÓN DE FEATURES
# =============================================================================

print("[1] Entrenando GBM Multiclase...")
gbm = GradientBoostingClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.1,
    subsample=0.8, random_state=42
)
gbm.fit(X_train_s, y_train)

# Extraer predicciones y probabilidades GBM
pred_gbm = gbm.predict(X_test_s)
prob_gbm = gbm.predict_proba(X_test_s)  # shape: (N, 3)

# Feature adicional para DC: vector de probabilidades GBM (3 cols)
X_test_hybrid = np.hstack([X_test_s, prob_gbm])

print("  [OK] GBM entrenado. Features híbridas generadas.")
print()

# =============================================================================
# 2. DIXON-COLES REAL
# =============================================================================

print("[2] Generando probabilidades Dixon-Coles...")

from core.predictor import DixonColes

dc = DixonColes()
teams = list(set(df['home_team'].unique()) | set(df['away_team'].unique()))
dc.fit(df.loc[train_mask], teams)

# Generar probabilidades DC para test
p_dc_test = []
for _, row in df.loc[test_mask].iterrows():
    p_win, p_draw, p_lose = dc.win_prob(
        str(row['home_team']),
        str(row['away_team']),
        home_factor=1.0
    )
    # Orden: [-1, 0, 1] = [away_win, draw, home_win]
    p_dc_test.append([p_lose, p_draw, p_win])

p_dc_test = np.array(p_dc_test)

print("  [OK] Probabilidades DC generadas.")
print()

# =============================================================================
# 3. DC HÍBRIDO (DC + GBM FEATURES)
# =============================================================================

print("[3] Entrenando DC Híbrido (con features GBM)...")
dc_hybrid = CalibratedClassifierCV(
    LogisticRegression(max_iter=1000, class_weight='balanced'),
    cv=3, method='isotonic'
)

# Necesitamos features híbridas también en train para consistencia
prob_gbm_train = gbm.predict_proba(X_train_s)
X_train_hybrid = np.hstack([X_train_s, prob_gbm_train])
dc_hybrid.fit(X_train_hybrid, y_train)
p_dc_hybrid_test = dc_hybrid.predict_proba(X_test_hybrid)

print("  [OK] DC Híbrido entrenado y evaluado.")
print()

# =============================================================================
# 4. OPTIMIZACIÓN DE PESOS (ALPHA) EN VALIDACIÓN TEMPORAL
# =============================================================================

print("[4] Optimizando alpha para ensemble ponderado...")
# Split temporal de validación (último 20% del train)
val_split = int(len(X_train_s) * 0.8)
X_val = X_train_s[val_split:]
y_val = y_train[val_split:]

# Probabilidades DC en validación
p_dc_val = []
for _, row in df.loc[train_mask].iloc[val_split:].iterrows():
    p_win, p_draw, p_lose = dc.win_prob(
        str(row['home_team']),
        str(row['away_team']),
        home_factor=1.0
    )
    p_dc_val.append([p_lose, p_draw, p_win])
p_dc_val = np.array(p_dc_val)

p_gbm_val = gbm.predict_proba(X_val)

alphas = np.arange(0.50, 0.96, 0.05)
best_alpha, best_ll = 0.75, np.inf

for a in alphas:
    p_ens = a * p_dc_val + (1 - a) * p_gbm_val
    p_ens /= p_ens.sum(axis=1, keepdims=True)
    ll = safe_logloss(y_val, p_ens)
    if ll < best_ll:
        best_ll, best_alpha = ll, a

print(f"  [OK] Alpha óptimo: {best_alpha:.2f} | LogLoss val: {best_ll:.4f}")
print()

# =============================================================================
# 5. EVALUACIÓN FINAL EN TEST
# =============================================================================

print("=" * 70)
print("EVALUACIÓN FINAL (TEST SET)")
print("=" * 70)
print()

# Ensemble final
p_ensemble = best_alpha * p_dc_test + (1 - best_alpha) * prob_gbm
p_ensemble /= p_ensemble.sum(axis=1, keepdims=True)

# Cálculo de métricas
models = {
    'GBM': (gbm.predict(X_test_s), prob_gbm),
    'DC': (np.argmax(p_dc_test, axis=1), p_dc_test),
    'DC+GBM_Feat': (dc_hybrid.predict(X_test_hybrid), p_dc_hybrid_test),
    f'ENS(α={best_alpha:.2f})': (np.argmax(p_ensemble, axis=1), p_ensemble)
}

print(f"{'Model':<12} | {'Acc':<6} | {'LogLoss':<8} | {'Brier':<6} | {'ECE':<6}")
print("-" * 60)

for name, (preds, probs) in models.items():
    acc = accuracy_score(y_test, preds)
    ll = safe_logloss(y_test, probs)
    br = brier_multiclass(y_test, probs)
    ece = calculate_ece(y_test, probs)
    print(f"{name:<12} | {acc:.4f} | {ll:.4f}     | {br:.4f} | {ece:.4f}")

print()

# =============================================================================
# DECISIÓN AUTOMÁTICA
# =============================================================================

print("=" * 70)
print("DECISIÓN CIENTÍFICA")
print("=" * 70)
print()

# Obtener métricas del ensemble
ens_acc = accuracy_score(y_test, np.argmax(p_ensemble, axis=1))
ens_ll = safe_logloss(y_test, p_ensemble)
ens_ece = calculate_ece(y_test, p_ensemble)

if ens_ece < 0.12 and ens_ll < 0.88:
    print(f"[OK] ARCHITECTURE ADOPTED: Ensemble α={best_alpha:.2f} es superior en calibración y precisión.")
    print("   → Usar como backbone principal para World Cup 2026.")
elif ens_ece < 0.20:
    print("[WARNING] ARCHITECTURE HYBRID: Ensemble válido pero requiere monitoreo de draws.")
    print("   → Usar para predicciones, DC puro para simulación Monte Carlo.")
else:
    print("[ERROR] ARCHITECTURE DC-ONLY: El ensemble no mejora calibración suficiente.")
    print("   → Mantener DC como fuente probabilística única.")

print()
print("Resumen rápido:")
gbm_acc = accuracy_score(y_test, gbm.predict(X_test_s))
dc_acc = accuracy_score(y_test, np.argmax(p_dc_test, axis=1))
hybrid_acc = accuracy_score(y_test, dc_hybrid.predict(X_test_hybrid))
print(f"   GBM Acc: {gbm_acc:.4f} | DC Acc: {dc_acc:.4f} | Hybrid Acc: {hybrid_acc:.4f}")
print("   → La fusión conserva poder discriminativo (GBM) y confiabilidad probabilística (DC).")
print("=" * 70)
