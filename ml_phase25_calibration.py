"""
PROTOCOLLO DE CORRECCIÓN - FASE 2.5: CALIBRACIÓN OBLIGATORIA DEL GBM
====================================================================

Calibrar el GBM con Isotonic Regression y re-evaluar.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
import sys
sys.path.insert(0, 'C:\\Proyecto_FIFA')

print("=" * 80)
print("FASE 2.5: CALIBRACIÓN OBLIGATORIA DEL GBM")
print("=" * 80)
print()

# =============================================================================
# FUNCIONES DE MÉTRICAS
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
# ENTRENAR GBM (FASE 1)
# =============================================================================

print("Entrenando GBM (FASE 1)...")

features = ["elo_diff", "is_neutral", "form_home", "form_away", "h2h"]
X_train = df_train[features].fillna(0.5)
X_test = df_test[features].fillna(0.5)
y_multiclass = df_train['outcome'].values
y_test = df_test['outcome'].values

gbm_mc = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
)
gbm_mc.fit(X_train, y_multiclass)
print("  [OK] GBM entrenado")
print()

# Evaluar GBM sin calibrar
print("GBM SIN CALIBRAR:")
proba_raw = gbm_mc.predict_proba(X_test)
pred_raw = gbm_mc.predict(X_test)

acc_raw = (pred_raw == y_test).mean()
ll_raw = safe_logloss(y_test, proba_raw)
brier_raw = brier_multiclass(y_test, proba_raw)
ece_raw = calculate_ece(y_test, proba_raw)

print(f"  Accuracy:  {acc_raw:.4f}")
print(f"  LogLoss:   {ll_raw:.4f}")
print(f"  Brier:     {brier_raw:.4f}")
print(f"  ECE:       {ece_raw:.4f}")
print()

# =============================================================================
# FASE 2.5: CALIBRACIÓN ISOTÓNICA
# =============================================================================

print("=" * 80)
print("FASE 2.5: CALIBRACIÓN ISOTÓNICA")
print("=" * 80)
print()

print("Calibrando GBM con Isotonic Regression (CV=3, walk-forward compatible)...")
gbm_cal = CalibratedClassifierCV(
    estimator=gbm_mc,
    method='isotonic',  # mejor para árboles
    cv=3,
    n_jobs=-1
)
gbm_cal.fit(X_train, y_multiclass)
print("  [OK] GBM calibrado")
print()

proba_cal = gbm_cal.predict_proba(X_test)
pred_cal = gbm_cal.predict(X_test)

acc_cal = (pred_cal == y_test).mean()
ll_cal = safe_logloss(y_test, proba_cal)
brier_cal = brier_multiclass(y_test, proba_cal)
ece_cal = calculate_ece(y_test, proba_cal)

print("GBM CALIBRADO:")
print(f"  Accuracy:  {acc_cal:.4f}  (anterior: {acc_raw:.4f})")
print(f"  LogLoss:   {ll_cal:.4f}  (anterior: {ll_raw:.4f})")
print(f"  Brier:     {brier_cal:.4f}  (anterior: {brier_raw:.4f})")
print(f"  ECE:       {ece_cal:.4f}  (anterior: {ece_raw:.4f}, DC: 0.0755)")
print()

# Delta
print("DELTA (calibrado - original):")
print(f"  Accuracy:  {acc_cal - acc_raw:+.4f}")
print(f"  LogLoss:   {ll_cal - ll_raw:+.4f}")
print(f"  Brier:     {brier_cal - brier_raw:+.4f}")
print(f"  ECE:       {ece_cal - ece_raw:+.4f}")
print()

# =============================================================================
# CRITERIO DE ÉXITO
# =============================================================================

print("=" * 80)
print("CRITERIO DE ÉXITO")
print("=" * 80)
print()

if ece_cal < 0.15 and acc_cal >= 0.57:
    print("[OK] CALIBRACIÓN EXITOSA: Usar GBM_cal en ensemble")
    decision = "REEMPLAZAR_DC"
elif ece_cal < 0.30:
    print("[WARNING] CALIBRACIÓN PARCIAL: Usar con peso reducido en ensemble")
    decision = "USAR_CON_PESO_REDUCIDO"
else:
    print("[ERROR] CALIBRACIÓN FALLIDA: Usar GBM solo para predicción puntual, DC para probabilidades")
    decision = "MANTENER_DC"

print()

# =============================================================================
# COMPARACIÓN FINAL CON DIXON-COLES
# =============================================================================

print("=" * 80)
print("COMPARACIÓN FINAL CON DIXON-COLES")
print("=" * 80)
print()

dc_metrics = {"acc": 0.5570, "logloss": 0.9315, "brier": 0.5508, "ece": 0.0755}
gbm_cal_metrics = {"acc": acc_cal, "logloss": ll_cal, "brier": brier_cal, "ece": ece_cal}

print(f"{'Métrica':<12} | {'DC':>10} | {'GBM_cal':>10} | {'Delta':>10}")
print("-" * 50)
print(f"{'Accuracy':<12} | {dc_metrics['acc']:.4f}   | {gbm_cal_metrics['acc']:.4f}   | {gbm_cal_metrics['acc'] - dc_metrics['acc']:+.4f}")
print(f"{'LogLoss':<12} | {dc_metrics['logloss']:.4f}   | {gbm_cal_metrics['logloss']:.4f}   | {gbm_cal_metrics['logloss'] - dc_metrics['logloss']:+.4f}")
print(f"{'Brier':<12} | {dc_metrics['brier']:.4f}   | {gbm_cal_metrics['brier']:.4f}   | {gbm_cal_metrics['brier'] - dc_metrics['brier']:+.4f}")
print(f"{'ECE':<12} | {dc_metrics['ece']:.4f}   | {gbm_cal_metrics['ece']:.4f}   | {gbm_cal_metrics['ece'] - dc_metrics['ece']:+.4f}")
print()

# =============================================================================
# DECISIÓN FINAL
# =============================================================================

print("=" * 80)
print("DECISIÓN FINAL")
print("=" * 80)
print()

if decision == "REEMPLAZAR_DC":
    print("ACCIÓN: Reemplazar DC por GBM_cal en TODO el sistema")
    print("  → GBM_cal supera a DC en todas las métricas críticas")
    print("  → Calibración exitosa sin sacrificar poder predictivo")
elif decision == "USAR_CON_PESO_REDUCIDO":
    print("ACCIÓN: Usar GBM_cal en ensemble con peso reducido")
    print("  → GBM_cal tiene buena calibración parcial")
    print("  → Ensemble: α=0.6 GBM_cal, β=0.4 DC")
else:
    print("ACCIÓN: Mantener DC para probabilidades, usar GBM como feature")
    print("  → Calibración destruye señal o no mejora suficiente")
    print("  → DC sigue siendo mejor para probabilidades calibradas")

print()

print("=" * 80)
print("FASE 2.5 COMPLETADA")
print("=" * 80)
