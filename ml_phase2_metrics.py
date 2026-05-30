"""
PROTOCOLLO DE CORRECCIÓN - FASE 2: MÉTRICAS COMPLETAS + CALIBRACIÓN
================================================================

Necesitamos responder: ¿GBM supera a DC en LogLoss, Brier y ECE, o solo en Accuracy?
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import sys
sys.path.insert(0, 'C:\\Proyecto_FIFA')

print("=" * 80)
print("FASE 2: MÉTRICAS COMPLETAS + CALIBRACIÓN")
print("=" * 80)
print()

# =============================================================================
# FUNCIONES DE MÉTRICAS
# =============================================================================

def safe_logloss(y_true, y_prob, eps=1e-12):
    y_prob = np.clip(y_prob, eps, 1 - eps)
    # one-hot encode y_true
    classes = np.array([-1, 0, 1])
    y_oh = np.zeros_like(y_prob)
    for i, val in enumerate(y_true):
        idx = np.where(classes == val)[0][0]
        y_oh[i, idx] = 1.0
    return -np.mean(np.sum(y_oh * np.log(y_prob), axis=1))

def brier_multiclass(y_true, y_prob, classes=np.array([-1, 0, 1])):
    # Brier score multiclase: mean squared error probabilístico
    y_oh = np.zeros_like(y_prob)
    for i, val in enumerate(y_true):
        idx = np.where(classes == val)[0][0]
        y_oh[i, idx] = 1.0
    return np.mean(np.sum((y_prob - y_oh) ** 2, axis=1))

def calculate_ece(y_true, y_prob, n_bins=10):
    # Expected Calibration Error multiclase (simplificado: clase mayoritaria)
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
# ENTRENAR MODELOS (FASE 1)
# =============================================================================

print("Entrenando modelos (FASE 1)...")

features = ["elo_diff", "is_neutral", "form_home", "form_away", "h2h"]
X_train = df_train[features].fillna(0.5)
X_test = df_test[features].fillna(0.5)
y_multiclass = df_train['outcome'].values
y_test = df_test['outcome'].values

# Random Forest
rf_mc = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_mc.fit(X_train, y_multiclass)
print("  [OK] Random Forest entrenado")

# Gradient Boosting
gbm_mc = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
)
gbm_mc.fit(X_train, y_multiclass)
print("  [OK] Gradient Boosting entrenado")

# Logistic Regression
lr_mc = LogisticRegression(
    class_weight='balanced',
    random_state=42,
    max_iter=1000
)
lr_mc.fit(X_train, y_multiclass)
print("  [OK] Logistic Regression entrenado")
print()

# =============================================================================
# FASE 2: MÉTRICAS COMPLETAS
# =============================================================================

print("=" * 80)
print("FASE 2: MÉTRICAS MULTICLASE COMPLETAS")
print("=" * 80)
print()

models = {
    'GBM': gbm_mc,
    'RF': rf_mc,
    'LR': lr_mc
}

print(f"{'Model':<6} | {'Acc':>8} | {'LogLoss':>10} | {'Brier':>8} | {'ECE':>8}")
print("-" * 60)

for name, model in models.items():
    proba = model.predict_proba(X_test)
    pred = model.predict(X_test)
    
    acc = (pred == y_test).mean()
    ll = safe_logloss(y_test, proba)
    brier = brier_multiclass(y_test, proba)
    ece = calculate_ece(y_test, proba)
    
    print(f"{name:<6} | {acc:.4f}   | {ll:.4f}     | {brier:.4f} | {ece:.4f}")

print()

# Comparación con Dixon-Coles
print("=" * 80)
print("COMPARACIÓN CON DIXON-COLES")
print("=" * 80)
print()

print("Dixon-Coles (referencia del benchmark anterior):")
print("  Acc: 0.5570 | LogLoss: 0.9315 | Brier: 0.5508 | ECE: 0.0755")
print()

# Obtener métricas de GBM
proba_gbm = gbm_mc.predict_proba(X_test)
pred_gbm = gbm_mc.predict(X_test)
acc_gbm = (pred_gbm == y_test).mean()
ll_gbm = safe_logloss(y_test, proba_gbm)
brier_gbm = brier_multiclass(y_test, proba_gbm)
ece_gbm = calculate_ece(y_test, proba_gbm)

print("Gradient Boosting (actual):")
print(f"  Acc: {acc_gbm:.4f} | LogLoss: {ll_gbm:.4f} | Brier: {brier_gbm:.4f} | ECE: {ece_gbm:.4f}")
print()

# =============================================================================
# CRITERIOS DE DECISIÓN CIENTÍFICA
# =============================================================================

print("=" * 80)
print("CRITERIOS DE DECISIÓN CIENTÍFICA")
print("=" * 80)
print()

dc_metrics = {"acc": 0.5570, "logloss": 0.9315, "brier": 0.5508, "ece": 0.0755}
gbm_metrics = {"acc": acc_gbm, "logloss": ll_gbm, "brier": brier_gbm, "ece": ece_gbm}

# Contar cuántas métricas gana GBM
wins = 0
if gbm_metrics["acc"] > dc_metrics["acc"]:
    wins += 1
    print(f"[GBM] Accuracy: {gbm_metrics['acc']:.4f} > {dc_metrics['acc']:.4f}")
else:
    print(f"[DC]  Accuracy: {dc_metrics['acc']:.4f} >= {gbm_metrics['acc']:.4f}")

if gbm_metrics["logloss"] < dc_metrics["logloss"]:
    wins += 1
    print(f"[GBM] LogLoss: {gbm_metrics['logloss']:.4f} < {dc_metrics['logloss']:.4f}")
else:
    print(f"[DC]  LogLoss: {dc_metrics['logloss']:.4f} <= {gbm_metrics['logloss']:.4f}")

if gbm_metrics["brier"] < dc_metrics["brier"]:
    wins += 1
    print(f"[GBM] Brier: {gbm_metrics['brier']:.4f} < {dc_metrics['brier']:.4f}")
else:
    print(f"[DC]  Brier: {dc_metrics['brier']:.4f} <= {gbm_metrics['brier']:.4f}")

if gbm_metrics["ece"] < dc_metrics["ece"]:
    wins += 1
    print(f"[GBM] ECE: {gbm_metrics['ece']:.4f} < {dc_metrics['ece']:.4f}")
else:
    print(f"[DC]  ECE: {dc_metrics['ece']:.4f} <= {gbm_metrics['ece']:.4f}")

print()
print(f"GBM gana en {wins}/4 métricas")
print()

# Decisión
if wins >= 3:
    print("[DECISIÓN] GBM supera a DC en >=3 métricas")
    print("  → ML multiclase es superior real")
    print("  → Reemplazar DC por GBM como backbone")
elif wins == 2 and gbm_metrics["logloss"] < dc_metrics["logloss"]:
    print("[DECISIÓN] GBM gana en 2 métricas incluyendo LogLoss")
    print("  → ML es bueno clasificando y calibrado")
    print("  → Considerar reemplazar DC por GBM")
elif gbm_metrics["acc"] > dc_metrics["acc"] and gbm_metrics["logloss"] > dc_metrics["logloss"]:
    print("[DECISIÓN] GBM gana en Accuracy pero pierde en LogLoss/ECE")
    print("  → ML es bueno clasificando pero mal calibrado")
    print("  → Aplicar CalibratedClassifierCV antes de ensemble")
else:
    print("[DECISIÓN] DC gana en LogLoss/Brier/ECE")
    print("  → DC está mejor calibrado probabilísticamente")
    print("  → Usar DC como base, ML como feature adicional")

print()

# =============================================================================
# CALIBRACIÓN (si es necesario)
# =============================================================================

if gbm_metrics["logloss"] > 1.10:
    print("=" * 80)
    print("CALIBRACIÓN ISOTÓNICA")
    print("=" * 80)
    print()
    print(f"LogLoss GBM ({gbm_metrics['logloss']:.4f}) > 1.10")
    print("Aplicando CalibratedClassifierCV con método='isotonic'...")
    print()
    
    gbm_cal = CalibratedClassifierCV(gbm_mc, cv=3, method='isotonic')
    gbm_cal.fit(X_train, y_multiclass)
    
    proba_cal = gbm_cal.predict_proba(X_test)
    pred_cal = gbm_cal.predict(X_test)
    
    acc_cal = (pred_cal == y_test).mean()
    ll_cal = safe_logloss(y_test, proba_cal)
    brier_cal = brier_multiclass(y_test, proba_cal)
    ece_cal = calculate_ece(y_test, proba_cal)
    
    print("GBM Calibrado:")
    print(f"  Acc: {acc_cal:.4f} | LogLoss: {ll_cal:.4f} | Brier: {brier_cal:.4f} | ECE: {ece_cal:.4f}")
    print()
    
    print("Delta (calibrado - original):")
    print(f"  Acc: {acc_cal - acc_gbm:+.4f}")
    print(f"  LogLoss: {ll_cal - ll_gbm:+.4f}")
    print(f"  Brier: {brier_cal - brier_gbm:+.4f}")
    print(f"  ECE: {ece_cal - ece_gbm:+.4f}")
    print()
else:
    print("=" * 80)
    print("CALIBRACIÓN NO NECESARIA")
    print("=" * 80)
    print()
    print(f"LogLoss GBM ({gbm_metrics['logloss']:.4f}) <= 1.10")
    print("No se requiere calibración isotónica")
    print()

print("=" * 80)
print("FASE 2 COMPLETADA")
print("=" * 80)
