# Script para fusionar datasets del Mundial 2026
# Combina partidos, clima, ranking FIFA y stats de jugadores

import pandas as pd
import numpy as np
from datetime import datetime
import os

def preparar_dataset_entrenamiento():
    """
    Fusiona múltiples fuentes de datos en un DataFrame maestro para entrenamiento.
    
    Returns:
        DataFrame con todas las features combinadas
    """
    print("=" * 60)
    print("🔗 FUSIÓN DE DATASETS - MUNDIAL 2026")
    print("=" * 60)
    
    # 1. Cargar partidos oficiales
    matches_path = "data/oficial/matches.csv"
    if os.path.exists(matches_path):
        matches = pd.read_csv(matches_path)
        print(f"✅ Partidos cargados: {len(matches)} registros")
    else:
        print(f"⚠️ No se encontró {matches_path}")
        matches = None
    
    # 2. Cargar datos climáticos
    clima_path = "data/clima/clima_fusionado.csv"
    if os.path.exists(clima_path):
        clima = pd.read_csv(clima_path)
        print(f"✅ Clima cargado: {len(clima)} registros")
    else:
        print(f"⚠️ No se encontró {clima_path}")
        clima = None
    
    # 3. Cargar ranking FIFA
    ranking_path = "data/historico/fifa_ranking-latest.csv"
    if os.path.exists(ranking_path):
        ranking = pd.read_csv(ranking_path)
        print(f"✅ Ranking FIFA cargado: {len(ranking)} registros")
    else:
        print(f"⚠️ No se encontró {ranking_path}")
        ranking = None
    
    # 4. Cargar stats de jugadores
    players_path = "data/jugadores/players_stats_2024.csv"
    if os.path.exists(players_path):
        players = pd.read_csv(players_path)
        print(f"✅ Stats de jugadores cargados: {len(players)} registros")
    else:
        print(f"⚠️ No se encontró {players_path}")
        players = None
    
    # Si no hay partidos, no podemos continuar
    if matches is None:
        print("❌ Error: No se encontraron datos de partidos. Descarga el dataset de Kaggle primero.")
        return None
    
    # Iniciar con el DataFrame de partidos
    df = matches.copy()
    
    # 5. Fusionar con clima (por fecha + ciudad)
    if clima is not None:
        # Asumimos que matches tiene 'kickoff_at' y 'city_id'
        # Convertir fechas si es necesario
        if 'kickoff_at' in df.columns:
            df['match_date'] = pd.to_datetime(df['kickoff_at']).dt.date
        if 'date' in clima.columns:
            clima['clima_date'] = pd.to_datetime(clima['date']).dt.date
        
        # Cruzar por fecha y ciudad
        df = df.merge(
            clima, 
            left_on=['match_date', 'city_id'], 
            right_on=['clima_date', 'city'],
            how='left'
        )
        print(f"✅ Clima fusionado: {df['temperature_2m_max'].notna().sum()} partidos con datos climáticos")
    
    # 6. Fusionar con ranking FIFA (por equipo + fecha)
    if ranking is not None:
        # Fusionar ranking para equipo local
        df = df.merge(
            ranking,
            left_on=['home_team_id', 'match_date'],
            right_on=['id', 'rank_date'],
            how='left',
            suffixes=('', '_home')
        )
        
        # Fusionar ranking para equipo visitante
        df = df.merge(
            ranking,
            left_on=['away_team_id', 'match_date'],
            right_on=['id', 'rank_date'],
            how='left',
            suffixes=('', '_away')
        )
        print(f"✅ Ranking FIFA fusionado")
    
    # 7. Calcular features agregadas de jugadores
    if players is not None:
        # Calcular poder ofensivo por selección
        def calcular_poder_equipo(nationality):
            team_players = players[players['nationality'] == nationality]
            if len(team_players) == 0:
                return {
                    'xg_promedio': 0,
                    'goles_promedio': 0,
                    'asistencias_promedio': 0,
                    'rating_promedio': 0
                }
            return {
                'xg_promedio': team_players['xG_per_90'].mean() if 'xG_per_90' in team_players.columns else 0,
                'goles_promedio': team_players['goals_per_90'].mean() if 'goals_per_90' in team_players.columns else 0,
                'asistencias_promedio': team_players['assists_per_90'].mean() if 'assists_per_90' in team_players.columns else 0,
                'rating_promedio': team_players['overall_rating'].mean() if 'overall_rating' in team_players.columns else 0
            }
        
        # Obtener lista de equipos únicos
        equipos = pd.concat([df['home_team_id'], df['away_team_id']]).unique()
        
        # Calcular features por equipo
        team_features = {}
        for team in equipos:
            team_features[team] = calcular_poder_equipo(team)
        
        # Agregar features al DataFrame
        df['home_xg'] = df['home_team_id'].map(lambda x: team_features.get(x, {}).get('xg_promedio', 0))
        df['away_xg'] = df['away_team_id'].map(lambda x: team_features.get(x, {}).get('xg_promedio', 0))
        df['home_rating'] = df['home_team_id'].map(lambda x: team_features.get(x, {}).get('rating_promedio', 0))
        df['away_rating'] = df['away_team_id'].map(lambda x: team_features.get(x, {}).get('rating_promedio', 0))
        
        print(f"✅ Features de jugadores calculadas para {len(equipos)} equipos")
    
    # 8. Crear features derivadas
    if 'temperature_2m_max' in df.columns:
        # Feature: temperatura promedio del partido
        df['temp_avg'] = (df['temperature_2m_max'] + df['temperature_2m_min']) / 2
        
        # Feature: condiciones adversas (calor extremo > 30°C o lluvia > 10mm)
        df['condiciones_adversas'] = (
            (df['temperature_2m_max'] > 30) | 
            (df['precipitation_sum'] > 10)
        ).astype(int)
    
    if 'rank_home' in df.columns and 'rank_away' in df.columns:
        # Feature: diferencial de ranking FIFA
        df['rank_diff'] = df['rank_home'] - df['rank_away']
    
    if 'home_xg' in df.columns and 'away_xg' in df.columns:
        # Feature: diferencial de xG esperado
        df['xg_diff'] = df['home_xg'] - df['away_xg']
    
    # 9. Guardar dataset fusionado
    output_path = "data/dataset_maestro.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Dataset maestro guardado en: {output_path}")
    print(f"📊 Total registros: {len(df)}")
    print(f"📏 Total columnas: {len(df.columns)}")
    
    # Mostrar resumen de columnas
    print("\n📋 Columnas del dataset maestro:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    return df

def generar_features_para_prediccion(df):
    """
    Genera features específicas para el modelo de predicción GBM.
    
    Args:
        df: DataFrame fusionado
    
    Returns:
        DataFrame con features listas para el modelo
    """
    features = []
    
    # Features base (siempre disponibles)
    base_features = [
        'home_team_id', 'away_team_id', 'stage_id'
    ]
    
    # Features climáticas (si disponibles)
    clima_features = [
        'temperature_2m_max', 'temperature_2m_min', 
        'precipitation_sum', 'wind_speed_10m', 
        'temp_avg', 'condiciones_adversas'
    ]
    
    # Features de ranking (si disponibles)
    ranking_features = [
        'rank_home', 'rank_away', 'rank_diff',
        'total_points_home', 'total_points_away'
    ]
    
    # Features de jugadores (si disponibles)
    player_features = [
        'home_xg', 'away_xg', 'xg_diff',
        'home_rating', 'away_rating'
    ]
    
    # Seleccionar solo features disponibles
    for feature_list in [base_features, clima_features, ranking_features, player_features]:
        for feature in feature_list:
            if feature in df.columns:
                features.append(feature)
    
    print(f"\n✅ Features seleccionadas para predicción: {len(features)}")
    return features

if __name__ == "__main__":
    # Ejecutar fusión de datos
    df_maestro = preparar_dataset_entrenamiento()
    
    if df_maestro is not None:
        # Generar lista de features para el modelo
        features = generar_features_para_prediccion(df_maestro)
        
        print("\n" + "=" * 60)
        print("🎯 SIGUIENTES PASOS")
        print("=" * 60)
        print("1. Revisar el dataset maestro en: data/dataset_maestro.csv")
        print("2. Re-entrenar el modelo GBM con las nuevas features:")
        print("   python train_model_with_new_features.py")
        print("3. Actualizar gbm_production.py para usar las nuevas features")
        print("4. Validar el modelo con el dataset de prueba")
