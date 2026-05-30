"""
Entrenamiento del modelo de predicción de fútbol usando datos históricos.
Este script carga el DataFrame desde nueva_data_partidos_290526.py,
realiza feature engineering, entrena y evalúa clasificadores,
y guarda el mejor modelo en core/trained_model.pkl.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# 1. Cargar datos desde el archivo existente
# Agregar el directorio padre al path para poder importar
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from nueva_data_partidos_290526 import df

print("✅ Datos cargados. Shape inicial:", df.shape)

# 2. Preparación y limpieza
df = df.copy()
# Usamos los nombres de columnas reales encontrados en el dataset
# 'MatchDate', 'HomeTeam', 'AwayTeam', 'FTHome', 'FTAway'
df['Date'] = pd.to_datetime(df['MatchDate'], errors='coerce')
df = df.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTHome', 'FTAway'])
df = df[df['Date'] >= '2010-01-01']
df = df.sort_values('Date').reset_index(drop=True)

# 3. Variable objetivo
def get_result(row):
    if row['FTHome'] > row['FTAway']:
        return 1  # victoria local
    elif row['FTHome'] < row['FTAway']:
        return 2  # victoria visitante
    else:
        return 0  # empate

df['resultado'] = df.apply(get_result, axis=1)
print(f"Partidos después de filtro: {len(df)}")

# 4. Feature engineering: Elo, forma reciente, diferencia de goles, días descanso
# Inicializar diccionarios para Elo y historiales
elo = {}  # elo actual por equipo
all_teams = set(df['HomeTeam']).union(set(df['AwayTeam']))
for team in all_teams:
    elo[team] = 1500.0

# Para almacenar features
elo_local_list = []
elo_visit_list = []
form_local_list = []
form_away_list = []
gd_local_list = []
gd_away_list = []
days_since_last_local_list = []
days_since_last_away_list = []
localia_list = []  # 1 si es local (casi siempre 1, pero respetamos columna Neutral si existe)

# Historial para forma y goles: dict de listas (partidos recientes)
history = {team: [] for team in all_teams}  # cada elemento: (fecha, goles_favor, goles_contra, puntos)

# Recorrer partidos en orden cronológico
print("Calculando features (esto puede tardar)...")
for idx, row in df.iterrows():
    home = row['HomeTeam']
    away = row['AwayTeam']
    date = row['Date']
    fthg = row['FTHome']
    ftag = row['FTAway']

    # Guardar Elos previos
    elo_h = elo[home]
    elo_a = elo[away]
    elo_local_list.append(elo_h)
    elo_visit_list.append(elo_a)
    
    # Calcular forma reciente (puntos últimos 5 partidos) y diferencia de goles promedio
    def get_recent_form_and_gd(team_history, current_date, n=5):
        recent = [m for m in team_history if m[0] < current_date]
        recent = recent[-n:]
        if not recent:
            return 0.5, 0.0  # valor neutro
        puntos = sum(m[3] for m in recent)
        gd = sum(m[4] for m in recent)  # diferencia de goles (favor - contra)
        return puntos / len(recent), gd / len(recent)

    form_h, gd_h = get_recent_form_and_gd(history[home], date)
    form_a, gd_a = get_recent_form_and_gd(history[away], date)
    form_local_list.append(form_h)
    form_away_list.append(form_a)
    gd_local_list.append(gd_h)
    gd_away_list.append(gd_a)

    # Días desde último partido
    def days_since_last(team_history, current_date):
        recent = [m for m in team_history if m[0] < current_date]
        if not recent:
            return 14  # valor por defecto (dos semanas)
        last_date = max(m[0] for m in recent)
        return (current_date - last_date).days

    days_h = days_since_last(history[home], date)
    days_a = days_since_last(history[away], date)
    days_since_last_local_list.append(days_h)
    days_since_last_away_list.append(days_a)

    # Localía: asumimos que casa = 1
    localia = 1 
    localia_list.append(localia)

    # Actualizar historial
    if fthg > ftag:
        puntos_h, puntos_a = 1.0, 0.0
    elif fthg < ftag:
        puntos_h, puntos_a = 0.0, 1.0
    else:
        puntos_h, puntos_a = 0.5, 0.5
    
    history[home].append((date, fthg, ftag, puntos_h, fthg - ftag))
    history[away].append((date, ftag, fthg, puntos_a, ftag - fthg))
    
    # Actualizar Elo
    expected_h = 1 / (1 + 10**((elo_a - elo_h) / 400))
    expected_a = 1 - expected_h
    K = 32
    elo[home] = elo_h + K * (puntos_h - expected_h)
    elo[away] = elo_a + K * (puntos_a - expected_a)

# Construir DataFrame de features
features_df = pd.DataFrame({
    'dif_elo': [elo_local_list[i] - elo_visit_list[i] for i in range(len(elo_local_list))],
    'form_local': form_local_list,
    'form_away': form_away_list,
    'gd_local': gd_local_list,
    'gd_away': gd_away_list,
    'localia': localia_list,
    'days_since_last_local': days_since_last_local_list,
    'days_since_last_away': days_since_last_away_list,
})
features_df['resultado'] = df['resultado'].values
features_df['Date'] = df['Date'].values

features_df = features_df.replace([np.inf, -np.inf], np.nan).dropna()
print(f"Features listas. Shape final: {features_df.shape}")

# 5. División temporal
train_mask = features_df['Date'] <= '2018-12-31'
val_mask = (features_df['Date'] > '2018-12-31') & (features_df['Date'] <= '2022-12-31')
test_mask = features_df['Date'] > '2022-12-31'

feature_cols = ['dif_elo', 'form_local', 'form_away', 'gd_local', 'gd_away', 'localia', 'days_since_last_local', 'days_since_last_away']
X_train, y_train = features_df.loc[train_mask, feature_cols], features_df.loc[train_mask, 'resultado']
X_val, y_val = features_df.loc[val_mask, feature_cols], features_df.loc[val_mask, 'resultado']
X_test, y_test = features_df.loc[test_mask, feature_cols], features_df.loc[test_mask, 'resultado']

# 6. Entrenar y evaluar
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred_proba_test = model.predict_proba(X_test)
ll_test = log_loss(y_test, y_pred_proba_test)
acc_test = accuracy_score(y_test, model.predict(X_test))

print(f"\nRandomForest - Test Log Loss: {ll_test:.4f} | Test Accuracy: {acc_test:.4f}")

# Guardar
joblib.dump(model, os.path.join(PROJECT_ROOT, 'core', 'trained_model.pkl'))
joblib.dump(feature_cols, os.path.join(PROJECT_ROOT, 'core', 'feature_columns.pkl'))

# Guardar estado final de Elo y forma para el predictor
final_state = {team: {'elo': elo[team], 'history': history[team][-5:]} for team in all_teams}
joblib.dump(final_state, os.path.join(PROJECT_ROOT, 'core', 'teams_final_state.pkl'))

print("✅ Todo guardado correctamente.")
