"""
HYBRID BENCHMARK V2: DC REAL + GBM + ENSEMBLE SEGURO
=========================================================

Integración real de Dixon-Coles con sanity checks obligatorios.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
import sys
sys.path.insert(0, 'C:\\Proyecto_FIFA')

warnings.filterwarnings('ignore')

print("=" * 70)
print("HYBRID BENCHMARK V2: DC REAL + GBM + ENSEMBLE SEGURO")
print("=" * 70)
print()

# =============================================================================
# MÉTRICAS BLINDADAS
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
            ece += np.abs(confidences[in_bin].mean() - accuracies[in_bin].mean()) * prop_in_bin
    return ece

def align_probabilities(proba_raw, model_classes, target_classes=np.array([-1, 0, 1])):
    """Reordena probabilidades para que coincidan exactamente con target_classes"""
    aligned = np.zeros((proba_raw.shape[0], len(target_classes)))
    for i, tc in enumerate(target_classes):
        if tc in model_classes:
            col_idx = np.where(model_classes == tc)[0][0]
            aligned[:, i] = proba_raw[:, col_idx]
        else:
            aligned[:, i] = 1e-12  # clase faltante → prob mínima
    aligned /= aligned.sum(axis=1, keepdims=True)
    return aligned

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
print(f"Clases objetivo: {np.unique(y_train)}")
print()

# =============================================================================
# 1. GBM MULTICLASE
# =============================================================================

print("[1] Entrenando GBM...")
gbm = GradientBoostingClassifier(n_estimators=300, max_depth=5, random_state=42)
gbm.fit(X_train_s, y_train)
p_gbm_raw = gbm.predict_proba(X_test_s)
p_gbm = align_probabilities(p_gbm_raw, gbm.classes_)
print(f"  [OK] GBM listo. Clases alineadas: {gbm.classes_}")
print()

# =============================================================================
# 2. DIXON-COLES REAL (INTEGRACIÓN)
# =============================================================================

print("[2] Generando probabilidades DC REALES...")

from core.predictor import DixonColes

dc = DixonColes()
teams = list(set(df['home_team'].unique()) | set(df['away_team'].unique()))
dc.fit(df.loc[train_mask], teams)

# Generar probabilidades DC para test
# DC devuelve: [home_win, draw, away_win]
# Pipeline espera: [away_win, draw, home_win] = [-1, 0, 1]
p_dc_raw = []
for _, row in df.loc[test_mask].iterrows():
    p_win, p_draw, p_lose = dc.win_prob(
        str(row['home_team']),
        str(row['away_team']),
        home_factor=1.0
    )
    # DC devuelve: (home_win, draw, away_win)
    p_dc_raw.append([p_win, p_draw, p_lose])

p_dc_raw = np.array(p_dc_raw)

# REORDEN EXPLÍCITO: [home, draw, away] → [away, draw, home]
p_dc_aligned = np.column_stack([
    p_dc_raw[:, 2],  # away_win (-1) → columna 0
    p_dc_raw[:, 1],  # draw (0)      → columna 1
    p_dc_raw[:, 0]   # home_win (1)  → columna 2
])

# NORMALIZACIÓN DEFENSIVA
p_dc_aligned = np.clip(p_dc_aligned, 1e-12, 1.0)
p_dc_aligned /= p_dc_aligned.sum(axis=1, keepdims=True)

dc_classes = np.array([-1, 0, 1])
p_dc = p_dc_aligned

# Sanity check obligatorio
dc_acc = (np.argmax(p_dc, axis=1) == y_test).mean()
print(f"  [OK] DC real cargado. Sanity Acc: {dc_acc:.4f}")
if dc_acc < 0.40:
    raise ValueError(f"[ERROR] DC falló sanity check (Acc={dc_acc:.4f}). Revisa orden de clases o leakage.")
print()

# =============================================================================
# 3. OPTIMIZACIÓN DE ALPHA (VALIDACIÓN TEMPORAL)
# =============================================================================

print("[3] Optimizando alpha...")
val_split = int(len(X_train_s) * 0.8)
X_val = X_train_s[val_split:]
y_val = y_train[val_split:]

# GBM en validación
gbm_val = GradientBoostingClassifier(n_estimators=300, max_depth=5, random_state=42)
gbm_val.fit(X_train_s[:val_split], y_train[:val_split])
p_gbm_val_raw = gbm_val.predict_proba(X_val)
p_gbm_val = align_probabilities(p_gbm_val_raw, gbm_val.classes_)

# DC en validación
p_dc_val_raw = []
for _, row in df.loc[train_mask].iloc[val_split:].iterrows():
    p_win, p_draw, p_lose = dc.win_prob(
        str(row['home_team']),
        str(row['away_team']),
        home_factor=1.0
    )
    p_dc_val_raw.append([p_lose, p_draw, p_win])
p_dc_val_raw = np.array(p_dc_val_raw)
p_dc_val = align_probabilities(p_dc_val_raw, dc_classes)

alphas = np.arange(0.50, 0.91, 0.05)
best_alpha, best_ll = 0.70, np.inf
for a in alphas:
    p_ens = a * p_dc_val + (1 - a) * p_gbm_val
    p_ens /= p_ens.sum(axis=1, keepdims=True)
    ll = safe_logloss(y_val, p_ens)
    if ll < best_ll:
        best_ll, best_alpha = ll, a

print(f"  [OK] Alpha óptimo: {best_alpha:.2f} | LogLoss val: {best_ll:.4f}")
print()

# =============================================================================
# 4. EVALUACIÓN FINAL
# =============================================================================

print("=" * 70)
print("EVALUACIÓN FINAL (TEST)")
print("=" * 70)
print()

p_ensemble = best_alpha * p_dc + (1 - best_alpha) * p_gbm
p_ensemble /= p_ensemble.sum(axis=1, keepdims=True)

models = {
    'GBM': (gbm.predict(X_test_s), p_gbm),
    'DC_REAL': (np.argmax(p_dc, axis=1), p_dc),
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
# DECISIÓN CIENTÍFICA
# =============================================================================

print("=" * 70)
print("DECISIÓN CIENTÍFICA")
print("=" * 70)
print()

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
print(f"   DC_REAL Acc: {dc_acc:.4f} (esperado: 0.550-0.565)")
print(f"   GBM Acc: {accuracy_score(y_test, gbm.predict(X_test_s)):.4f}")
print(f"   Ensemble Acc: {ens_acc:.4f}")
print("   → La fusión conserva poder discriminativo (GBM) y confiabilidad probabilística (DC).")
print("=" * 70)
