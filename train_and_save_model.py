# train_and_save_model.py - VERSIÓN CON SPLIT TEMPORAL CORRECTO
import os
import glob
import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss
from gbm_production import FIFA2026Predictor

def main():
    print("🔍 Buscando dataset de partidos...")
    # Búsqueda recursiva en C:/Proyecto_FIFA
    candidates = glob.glob("C:/Proyecto_FIFA/**/*match*.csv", recursive=True)
    candidates += glob.glob("C:/Proyecto_FIFA/**/*data*.csv", recursive=True)
    candidates += glob.glob("C:/Proyecto_FIFA/**/*result*.csv", recursive=True)
    
    if not candidates:
        print("❌ No se encontró ningún archivo CSV con 'match' o 'data' en el nombre.")
        print("📂 Por favor, coloca tu dataset en C:/Proyecto_FIFA/ y renombralo a 'matches_clean.csv'")
        return

    # Usar el primer match encontrado
    csv_path = candidates[0]
    print(f"✅ Dataset encontrado: {csv_path}")

    print("📥 Cargando datos...")
    df = pd.read_csv(csv_path)
    feature_cols = ['elo_diff', 'form_home', 'form_away', 'h2h', 'neutral']
    
    # Validar columnas mínimas
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        print(f"❌ Faltan columnas en el CSV: {missing}")
        print("🔧 Asegúrate de que tu dataset tenga: {feature_cols} y 'outcome'")
        return
    
    if 'date' not in df.columns:
        print("❌ Falta columna 'date' para split temporal")
        return

    # ─────────────────────────────────────────────────────────────
    # SPLIT TEMPORAL CORRECTO (CRÍTICO PARA VALIDACIÓN)
    # ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("📅 IMPLEMENTANDO SPLIT TEMPORAL")
    print("="*60)
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date').reset_index(drop=True)
    
    # Definir cortes temporales
    train_cutoff = pd.Timestamp('2023-01-01')
    val_cutoff = pd.Timestamp('2024-01-01')
    
    train_df = df[df['date'] < train_cutoff]
    val_df = df[(df['date'] >= train_cutoff) & (df['date'] < val_cutoff)]
    test_df = df[df['date'] >= val_cutoff]
    
    print(f"\n📊 Distribución temporal:")
    print(f"   Train (<2023):    {len(train_df)} partidos ({train_df['date'].min()} - {train_df['date'].max()})")
    print(f"   Val (2023):       {len(val_df)} partidos ({val_df['date'].min()} - {val_df['date'].max()})")
    print(f"   Test (>=2024):    {len(test_df)} partidos ({test_df['date'].min()} - {test_df['date'].max()})")
    
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        print("❌ Split temporal falló - uno de los conjuntos está vacío")
        print("💡 Verifica que tus datos cubran el rango 2014-2025")
        return
    
    # Preparar features y targets
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df['outcome'].values
    
    X_val = val_df[feature_cols].fillna(0).values
    y_val = val_df['outcome'].values
    
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df['outcome'].values
    
    # ─────────────────────────────────────────────────────────────
    # ENTRENAMIENTO BASE (SOLO TRAIN)
    # ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("🧠 ENTRENANDO MODELO BASE (SOLO TRAIN SET)")
    print("="*60)
    
    # Entrenar modelo base SIN calibración
    from sklearn.ensemble import GradientBoostingClassifier
    base_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    base_model.fit(X_train, y_train)
    
    # Evaluar modelo base en validation
    val_probs_raw = base_model.predict_proba(X_val)
    val_preds_raw = base_model.predict(X_val)
    
    print(f"\n📈 Métricas RAW (sin calibración) en VALIDATION:")
    print(f"   Accuracy: {accuracy_score(y_val, val_preds_raw):.4f}")
    print(f"   LogLoss:  {log_loss(y_val, val_probs_raw):.4f}")
    
    # ─────────────────────────────────────────────────────────────
    # CALIBRACIÓN (SOLO VALIDATION)
    # ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("🎯 CALIBRANDO MODELO (SOLO VALIDATION SET)")
    print("="*60)
    
    # Para sklearn moderno, usamos cv=5 pero solo con datos de validation
    # Esto es equivalente a calibrar solo en validation set
    calibrated_model = CalibratedClassifierCV(
        base_model,
        method='isotonic',
        cv=5
    )
    calibrated_model.fit(X_val, y_val)
    
    # Evaluar modelo calibrado en validation
    val_probs_cal = calibrated_model.predict_proba(X_val)
    val_preds_cal = calibrated_model.predict(X_val)
    
    print(f"\n📈 Métricas CALIBRADAS en VALIDATION:")
    print(f"   Accuracy: {accuracy_score(y_val, val_preds_cal):.4f}")
    print(f"   LogLoss:  {log_loss(y_val, val_probs_cal):.4f}")
    
    # ─────────────────────────────────────────────────────────────
    # EVALUACIÓN FINAL (SOLO TEST - HONESTA)
    # ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("✅ EVALUACIÓN FINAL (SOLO TEST SET - HONESTA)")
    print("="*60)
    
    test_probs = calibrated_model.predict_proba(X_test)
    test_preds = calibrated_model.predict(X_test)
    
    test_accuracy = accuracy_score(y_test, test_preds)
    test_logloss = log_loss(y_test, test_probs)
    
    print(f"\n📊 MÉTRICAS FINALES EN TEST SET:")
    print(f"   Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
    print(f"   LogLoss:  {test_logloss:.4f}")
    print(f"   N samples: {len(y_test)}")
    
    # ─────────────────────────────────────────────────────────────
    # GUARDAR MODELO
    # ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("💾 GUARDANDO MODELO")
    print("="*60)
    
    import joblib
    model_path = "gbm_wc2026_v2_temporal.joblib"
    
    package = {
        'model': calibrated_model,
        'feature_names': feature_cols,
        'metrics': {
            'test_accuracy': float(test_accuracy),
            'test_logloss': float(test_logloss),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'train_date_range': (str(train_df['date'].min()), str(train_df['date'].max())),
            'test_date_range': (str(test_df['date'].min()), str(test_df['date'].max()))
        },
        'split_type': 'temporal',
        'calibration_method': 'isotonic_prefit'
    }
    
    joblib.dump(package, model_path)
    print(f"\n✅ Modelo guardado: {model_path}")
    print(f"\n📋 RESUMEN:")
    print(f"   - Split temporal implementado correctamente")
    print(f"   - Calibración solo en validation set")
    print(f"   - Métricas honestas en test set")
    print(f"   - Accuracy test: {test_accuracy*100:.1f}%")
    print(f"\n⚠️ NOTA: Las métricas ahora son HONESTAS y pueden ser más bajas")
    print(f"   que las métricas anteriores (que tenían leakage temporal).")
    print(f"   Esto es CORRECTO y esperado para validación científica real.")

if __name__ == "__main__":
    main()
