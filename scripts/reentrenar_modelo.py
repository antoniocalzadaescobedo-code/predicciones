#!/usr/bin/env python3
"""
Script de Re-entrenamiento - Predictor FIFA 2026 (v2.0)
Entrena el modelo GBM con features reales: Clima, Ranking FIFA, Diferencias H2H, etc.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
import joblib

# Agregar directorio padre para importar gbm_production
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gbm_production import FIFA2026Predictor

def main():
    print("🚀 Iniciando Re-entrenamiento con Dataset Maestro...")
    
    # 1. Cargar datos
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset_maestro.csv')
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: No se encontró {csv_path}")
        print("💡 Ejecuta primero 'scripts/fusionar_datos.py'")
        return

    df = pd.read_csv(csv_path)
    print(f"✅ Cargados {len(df)} partidos históricos.")

    # 2. Selección de Features
    # Definimos qué columnas usará el modelo para aprender
    # (Basado en lo que generó fusionar_datos.py)
    feature_cols = [
        'elo_diff',
        'rank_diff',          # Diferencia de ranking FIFA
        'temp_home',          # Temperatura local
        'precipitation_sum',  # Lluvia (afecta juego)
        'wind_speed',         # Viento
        'h2h_win_rate',       # Historial directo
        'form_home',          # Racha reciente local
        'form_away',          # Racha reciente visitante
        'importance_score'    # Importancia del partido (amistoso vs eliminatoria)
    ]
    
    # Verificar que existen las columnas
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"⚠️ Faltan columnas en el CSV: {missing}")
        print("👉 Verifica que fusionar_datos.py generó todas las features.")
        # Fallback a features básicas si faltan las nuevas
        feature_cols = ['elo_diff', 'form_home', 'form_away', 'h2h_win_rate']

    X = df[feature_cols]
    y = df['result_code']  # 0: Local, 1: Empate, 2: Visitante

    # Manejo de valores nulos (rellenar con medianas)
    X = X.fillna(X.median())

    # 3. División Entrenamiento / Prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"📊 Split: {len(X_train)} train | {len(X_test)} test")

    # 4. Entrenamiento
    print("🧠 Entrenando modelo GBM (esto puede tardar unos segundos)...")
    
    # Instanciamos la clase con calibración isotónica activada
    predictor = FIFA2026Predictor(calibrate=True)
    
    # Método interno para entrenar (asumiendo que tienes train() en tu clase)
    # Si no tienes un método train() público, lo simulamos o lo añadimos.
    # Asumimos que gbm_production tiene la lógica para encajar datos.
    # NOTA: Si tu clase solo tiene load(), necesitarás añadir un método fit().
    # Aquí uso una simulación de llamada para propósitos del script.
    
    try:
        # Intentamos usar la lógica interna de tu clase
        # Si FIFA2026Predictor no tiene .fit(), debes añadirlo o usar el código de abajo
        if hasattr(predictor, 'fit'):
             predictor.fit(X_train, y_train)
        else:
            # Fallback si no existe método fit público
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.calibration import CalibratedClassifierCV
            
            base_model = GradientBoostingClassifier(
                n_estimators=300, 
                learning_rate=0.05, 
                max_depth=4, 
                random_state=42
            )
            # Entrenar calibrado
            calibrated_model = CalibratedClassifierCV(base_model, cv=5, method='isotonic')
            calibrated_model.fit(X_train, y_train)
            
            predictor.model = calibrated_model
            predictor.feature_names = feature_cols
            predictor.is_fitted = True
            
    except Exception as e:
        print(f"❌ Error durante entrenamiento: {e}")
        return

    # 5. Evaluación
    y_pred_prob = predictor.model.predict_proba(X_test)
    y_pred = predictor.model.predict(X_test)
    
    loss = log_loss(y_test, y_pred_prob)
    acc = accuracy_score(y_test, y_pred)
    
    print("-" * 40)
    print("📈 RESULTADOS DE VALIDACIÓN")
    print("-" * 40)
    print(f" Accuracy: {acc:.3f}")
    print(f"📉 Log Loss: {loss:.3f}")
    print("-" * 40)

    # 6. Guardar Modelo
    output_path = os.path.join(os.path.dirname(__file__), '..', 'gbm_wc2026_v2_real_data.joblib')
    joblib.dump({
        'model': predictor.model,
        'feature_names': feature_cols,
        'metrics_history': {'accuracy': acc, 'log_loss': loss}
    }, output_path)
    
    print(f"✅ Modelo guardado en: {output_path}")
    print("💡 Recuerda actualizar app_streamlit.py para cargar este nuevo archivo si quieres usarlo.")

if __name__ == "__main__":
    main()
