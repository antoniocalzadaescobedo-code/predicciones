"""
DC DEBUG - Diagnóstico de desalineación de clases Dixon-Coles
============================================================

Identifica el mapeo de clases que DC devuelve vs el esperado [-1, 0, 1].
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import sys
sys.path.insert(0, 'C:\\Proyecto_FIFA')

print("=" * 70)
print("DC DEBUG - Diagnóstico de desalineación de clases")
print("=" * 70)
print()

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
test_mask = df['date'] >= pd.Timestamp(split_date)

X_test = df.loc[test_mask, ['elo_diff', 'form_home', 'form_away', 'h2h', 'is_neutral']].fillna(0).values
y_test = df.loc[test_mask, 'outcome'].values

print(f"Test: {X_test.shape[0]} partidos")
print()

# =============================================================================
# DIXON-COLES REAL
# =============================================================================

print("Generando probabilidades DC REALES...")

from core.predictor import DixonColes

dc = DixonColes()
teams = list(set(df['home_team'].unique()) | set(df['away_team'].unique()))
dc.fit(df[df['date'] < pd.Timestamp(split_date)], teams)

# Generar probabilidades DC para test
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

# Clases que DC usa (según implementación)
dc_classes = np.array([1, 0, -1])  # [home_win, draw, away_win]

print("Clases retornadas por DC:", dc_classes)
print("Shape probabilidades:", p_dc_raw.shape)
print()

print("Primeras 5 filas de probabilidades:")
print(p_dc_raw[:5])
print()

print("Primeras 5 predicciones (argmax sobre columnas):", np.argmax(p_dc_raw, axis=1)[:5])
print("Primeras 5 labels reales:", y_test[:5])
print()

# =============================================================================
# ANÁLISIS DE MAPEO
# =============================================================================

print("=" * 70)
print("ANÁLISIS DE MAPEO")
print("=" * 70)
print()

print("Interpretación:")
print("  DC devuelve [home_win, draw, away_win] = [1, 0, -1]")
print("  Nuestro pipeline espera [away_win, draw, home_win] = [-1, 0, 1]")
print()
print("  argmax columna 0 (home_win) → se interpreta como -1 (away_win) → INVERSIÓN")
print("  argmax columna 1 (draw) → se interpreta como 0 (draw) → CORRECTO")
print("  argmax columna 2 (away_win) → se interpreta como 1 (home_win) → INVERSIÓN")
print()

# Calcular accuracy con mapeo incorrecto
preds_incorrect = np.argmax(p_dc_raw, axis=1)
acc_incorrect = np.mean(preds_incorrect == y_test)
print(f"Accuracy con mapeo INCORRECTO: {acc_incorrect:.4f}")
print()

# Calcular accuracy con mapeo correcto
# Reordenar: [home_win, draw, away_win] → [away_win, draw, home_win]
p_dc_corrected = p_dc_raw[:, [2, 1, 0]]  # [away, draw, home]
preds_correct = np.argmax(p_dc_corrected, axis=1)
acc_correct = np.mean(preds_correct == y_test)
print(f"Accuracy con mapeo CORRECTO: {acc_correct:.4f}")
print()

print("MAPEO CORRECTO:")
print("  p_dc_corrected[:, 0] = p_dc_raw[:, 2]  # away_win")
print("  p_dc_corrected[:, 1] = p_dc_raw[:, 1]  # draw")
print("  p_dc_corrected[:, 2] = p_dc_raw[:, 0]  # home_win")
print()

print("=" * 70)
