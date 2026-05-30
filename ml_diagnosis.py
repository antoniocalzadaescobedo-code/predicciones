"""
ML DIAGNOSIS - 5 Experimentos para identificar bugs en modelo ML
================================================================

Experimento 1: Sanity check extremo
- Distribución de labels
- Clases
- Shape
- Features
- Probabilidades
- Confusion matrix
- Baseline random
- Baseline majority class

Experimento 2: Verificar target mapping
- home_win = 1
- draw = 0
- away_win = -1

Experimento 3: Revisar probabilidades
- predict_proba()
- Orden de clases
- Columnas
- Mapping

Experimento 4: Comparar contra dummy classifier
- Dummy stratified
- Dummy majority

Experimento 5: Matriz de confusión
- Qué está prediciendo
- Si colapsa a una clase
- Si invierte home/away
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, 'C:\\Proyecto_FIFA')

print("=" * 80)
print("ML DIAGNOSIS - 5 Experimentos para identificar bugs en modelo ML")
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
print(f"Rango temporal: {df['date'].min()} a {df['date'].max()}")
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
# EXPERIMENTO 1: SANITY CHECK EXTREMO
# =============================================================================

print("=" * 80)
print("EXPERIMENTO 1: SANITY CHECK EXTREMO")
print("=" * 80)
print()

# Split temporal
df_train = df[df["date"] < pd.Timestamp("2022-12-31")].copy()
df_test = df[df["date"] >= pd.Timestamp("2023-01-01")].copy()

print(f"Train shape: {df_train.shape}")
print(f"Test shape: {df_test.shape}")
print()

# Distribución de labels
print("Distribución de labels (train):")
label_counts = Counter(df_train["outcome"])
for label, count in sorted(label_counts.items()):
    print(f"  {label}: {count} ({count/len(df_train)*100:.2f}%)")
print()

print("Distribución de labels (test):")
label_counts_test = Counter(df_test["outcome"])
for label, count in sorted(label_counts_test.items()):
    print(f"  {label}: {count} ({count/len(df_test)*100:.2f}%)")
print()

# Features
features = ["elo_diff", "is_neutral", "form_home", "form_away", "h2h"]
print(f"Features: {features}")
print()

print("Estadísticas de features (train):")
for feat in features:
    print(f"  {feat}:")
    print(f"    Mean: {df_train[feat].mean():.4f}")
    print(f"    Std: {df_train[feat].std():.4f}")
    print(f"    Min: {df_train[feat].min():.4f}")
    print(f"    Max: {df_train[feat].max():.4f}")
print()

# Baseline random
print("Baseline random:")
random_acc = 1/3  # 3 clases
print(f"  Accuracy esperada: {random_acc:.4f}")
print()

# Baseline majority class
print("Baseline majority class:")
majority_class = max(label_counts, key=label_counts.get)
majority_acc = label_counts[majority_class] / len(df_train)
print(f"  Clase mayoritaria: {majority_class}")
print(f"  Accuracy: {majority_acc:.4f}")
print()

# =============================================================================
# EXPERIMENTO 2: VERIFICAR TARGET MAPPING
# =============================================================================

print("=" * 80)
print("EXPERIMENTO 2: VERIFICAR TARGET MAPPING")
print("=" * 80)
print()

print("Mapeo de outcome:")
print("  1 = home_win")
print("  0 = draw")
print("  -1 = away_win")
print()

# Verificar consistencia
print("Verificando consistencia de mapeo...")
sample_home_win = df_train[df_train["outcome"] == 1].head(1)
if not sample_home_win.empty:
    row = sample_home_win.iloc[0]
    print(f"  Ejemplo home_win (outcome=1):")
    print(f"    {row['home_team']} {row['home_score']} - {row['away_score']} {row['away_team']}")
    print(f"    home_score > away_score: {row['home_score'] > row['away_score']}")
print()

sample_draw = df_train[df_train["outcome"] == 0].head(1)
if not sample_draw.empty:
    row = sample_draw.iloc[0]
    print(f"  Ejemplo draw (outcome=0):")
    print(f"    {row['home_team']} {row['home_score']} - {row['away_score']} {row['away_team']}")
    print(f"    home_score == away_score: {row['home_score'] == row['away_score']}")
print()

sample_away_win = df_train[df_train["outcome"] == -1].head(1)
if not sample_away_win.empty:
    row = sample_away_win.iloc[0]
    print(f"  Ejemplo away_win (outcome=-1):")
    print(f"    {row['home_team']} {row['home_score']} - {row['away_score']} {row['away_team']}")
    print(f"    home_score < away_score: {row['home_score'] < row['away_score']}")
print()

# =============================================================================
# EXPERIMENTO 3: REVISAR PROBABILIDADES
# =============================================================================

print("=" * 80)
print("EXPERIMENTO 3: REVISAR PROBABILIDADES")
print("=" * 80)
print()

# Convertir a clasificación binaria (home_win vs no_home_win)
df_train["binary_outcome"] = (df_train["outcome"] == 1).astype(int)
df_test["binary_outcome"] = (df_test["outcome"] == 1).astype(int)

X_train = df_train[features].fillna(0.5)
y_train = df_train["binary_outcome"]
X_test = df_test[features].fillna(0.5)
y_test = df_test["binary_outcome"]

print("Entrenando modelo Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print()

print("predict_proba() - Primeras 5 predicciones:")
probs = rf.predict_proba(X_test)[:5]
print(f"  Shape: {probs.shape}")
print(f"  Clases: {rf.classes_}")
print(f"  Probabilidades:")
for i, p in enumerate(probs):
    print(f"    Sample {i}: {p}")
print()

print("Verificando orden de clases:")
print(f"  rf.classes_ = {rf.classes_}")
print(f"  Clase positiva (1) = home_win")
print(f"  Clase negativa (0) = no_home_win (draw + away_win)")
print()

# =============================================================================
# EXPERIMENTO 4: COMPARAR CONTRA DUMMY CLASSIFIER
# =============================================================================

print("=" * 80)
print("EXPERIMENTO 4: COMPARAR CONTRA DUMMY CLASSIFIER")
print("=" * 80)
print()

# Dummy stratified
print("Dummy stratified:")
dummy_strat = DummyClassifier(strategy="stratified", random_state=42)
dummy_strat.fit(X_train, y_train)
dummy_strat_acc = dummy_strat.score(X_test, y_test)
print(f"  Accuracy: {dummy_strat_acc:.4f}")
print()

# Dummy majority
print("Dummy majority:")
dummy_majority = DummyClassifier(strategy="most_frequent")
dummy_majority.fit(X_train, y_train)
dummy_majority_acc = dummy_majority.score(X_test, y_test)
print(f"  Accuracy: {dummy_majority_acc:.4f}")
print()

# Random Forest
print("Random Forest:")
rf_acc = rf.score(X_test, y_test)
print(f"  Accuracy: {rf_acc:.4f}")
print()

print("Comparación:")
print(f"  Dummy stratified: {dummy_strat_acc:.4f}")
print(f"  Dummy majority: {dummy_majority_acc:.4f}")
print(f"  Random Forest: {rf_acc:.4f}")
print()

if rf_acc < dummy_strat_acc:
    print("[WARNING] RF pierde contra dummy stratified - posible bug estructural")
if rf_acc < dummy_majority_acc:
    print("[WARNING] RF pierde contra dummy majority - posible bug estructural")
print()

# =============================================================================
# EXPERIMENTO 5: MATRIZ DE CONFUSIÓN
# =============================================================================

print("=" * 80)
print("EXPERIMENTO 5: MATRIZ DE CONFUSIÓN")
print("=" * 80)
print()

y_pred = rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print("Matriz de confusión (binary: home_win vs no_home_win):")
print(f"  Predicciones: {rf.classes_}")
print(f"  [[TN, FP],")
print(f"   [FN, TP]]")
print(f"  {cm}")
print()

print("Interpretación:")
tn, fp, fn, tp = cm.ravel()
print(f"  TN (True Negative): {tn} - Predijo no_home_win, era no_home_win")
print(f"  FP (False Positive): {fp} - Predijo home_win, era no_home_win")
print(f"  FN (False Negative): {fn} - Predijo no_home_win, era home_win")
print(f"  TP (True Positive): {tp} - Predijo home_win, era home_win")
print()

# Verificar si colapsa a una clase
total = tn + fp + fn + tp
if tp + fp == 0:
    print("[WARNING] Modelo colapsa a clase negativa (nunca predice home_win)")
elif tn + fn == 0:
    print("[WARNING] Modelo colapsa a clase positiva (siempre predice home_win)")
else:
    print("Modelo no colapsa a una sola clase")
print()

# Verificar si invierte
if fp > tp and fn > tn:
    print("[WARNING] Posible inversión de predicciones (FP > TP y FN > TN)")
else:
    print("No hay evidencia de inversión de predicciones")
print()

# Classification report
print("Classification report:")
print(classification_report(y_test, y_pred, target_names=["no_home_win", "home_win"]))
print()

# =============================================================================
# RESUMEN
# =============================================================================

print("=" * 80)
print("RESUMEN")
print("=" * 80)
print()

print("Experimento 1 - Sanity check:")
print(f"  Train shape: {df_train.shape}")
print(f"  Test shape: {df_test.shape}")
print(f"  Distribución labels: {dict(label_counts)}")
print(f"  Baseline random: {random_acc:.4f}")
print(f"  Baseline majority: {majority_acc:.4f}")
print()

print("Experimento 2 - Target mapping:")
print("  Mapeo: 1=home_win, 0=draw, -1=away_win")
print("  Mapeo binario: 1=home_win, 0=no_home_win")
print()

print("Experimento 3 - Probabilidades:")
print(f"  rf.classes_ = {rf.classes_}")
print(f"  predict_proba() shape: {probs.shape}")
print()

print("Experimento 4 - Dummy comparison:")
print(f"  Dummy stratified: {dummy_strat_acc:.4f}")
print(f"  Dummy majority: {dummy_majority_acc:.4f}")
print(f"  Random Forest: {rf_acc:.4f}")
if rf_acc < dummy_strat_acc:
    print("  [WARNING] RF pierde contra dummy stratified")
if rf_acc < dummy_majority_acc:
    print("  [WARNING] RF pierde contra dummy majority")
print()

print("Experimento 5 - Confusion matrix:")
print(f"  TN={tn}, FP={fp}, FN={fn}, TP={tp}")
if tp + fp == 0 or tn + fn == 0:
    print("  [WARNING] Modelo colapsa a una clase")
if fp > tp and fn > tn:
    print("  [WARNING] Posible inversión de predicciones")
print()

print("=" * 80)
