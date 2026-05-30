#!/usr/bin/env python3
"""
Purged Time Series Cross-Validation for FIFA 2026 Predictor
===========================================================
Implementa validación temporal rodante con gap/purge para evitar leakage
y detectar drift de calibración de draws en múltiples ventanas temporales.
"""

import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.calibration import calibration_curve
from datetime import datetime, timedelta

print("=" * 80)
print("PURGED TIME SERIES CROSS-VALIDATION")
print("=" * 80)
print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE VENTANAS TEMPORALES
# ─────────────────────────────────────────────────────────────
# Configuración de Purged Time Series CV
N_WINDOWS = 3  # Número de ventanas temporales
TRAIN_MONTHS = 24  # Meses de entrenamiento por ventana
VAL_MONTHS = 6  # Meses de validación por ventana
TEST_MONTHS = 6  # Meses de test por ventana
GAP_DAYS = 7  # Gap/purge entre ventanas (días para evitar leakage)

# Filtrar datos modernos (desde 2010 para tener suficiente densidad)
MODERN_START = pd.Timestamp('2010-01-01')

print(f"📊 CONFIGURACIÓN DE VENTANAS TEMPORALES:")
print(f"   Ventanas: {N_WINDOWS}")
print(f"   Train: {TRAIN_MONTHS} meses por ventana")
print(f"   Val: {VAL_MONTHS} meses por ventana")
print(f"   Test: {TEST_MONTHS} meses por ventana")
print(f"   Gap: {GAP_DAYS} días (purge para evitar leakage)")

# ─────────────────────────────────────────────────────────────
# CARGAR DATOS
# ─────────────────────────────────────────────────────────────
print("\n📦 Cargando datos...")

try:
    df = pd.read_csv("matches_clean.csv")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date').reset_index(drop=True)
    
    # Filtrar datos modernos
    df = df[df['date'] >= MODERN_START].reset_index(drop=True)
    
    print(f"✅ Dataset cargado: {len(df)} partidos (desde {MODERN_START.date()})")
    print(f"   Rango: {df['date'].min()} - {df['date'].max()}")
except Exception as e:
    print(f"❌ Error cargando dataset: {e}")
    exit(1)

feature_cols = ['elo_diff', 'form_home', 'form_away', 'h2h', 'neutral']

# ─────────────────────────────────────────────────────────────
# GENERAR VENTANAS TEMPORALES
# ─────────────────────────────────────────────────────────────
print("\n📅 Generando ventanas temporales purgadas...")

windows = []
min_date = df['date'].min()
max_date = df['date'].max()

for i in range(N_WINDOWS):
    # Calcular fechas para ventana i
    train_start = min_date + timedelta(days=30 * TRAIN_MONTHS * i)
    train_end = train_start + timedelta(days=30 * TRAIN_MONTHS)
    
    # Gap/purge entre train y val
    val_start = train_end + timedelta(days=GAP_DAYS)
    val_end = val_start + timedelta(days=30 * VAL_MONTHS)
    
    # Gap/purge entre val y test
    test_start = val_end + timedelta(days=GAP_DAYS)
    test_end = test_start + timedelta(days=30 * TEST_MONTHS)
    
    # Verificar que no excedemos el rango de datos
    if test_end > max_date:
        print(f"   Ventana {i+1}: Excede rango de datos, deteniendo")
        break
    
    # Extraer datos para esta ventana
    train_df = df[(df['date'] >= train_start) & (df['date'] < train_end)]
    val_df = df[(df['date'] >= val_start) & (df['date'] < val_end)]
    test_df = df[(df['date'] >= test_start) & (df['date'] < test_end)]
    
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        print(f"   Ventana {i+1}: Sin datos suficientes, saltando")
        continue
    
    windows.append({
        'window_id': i + 1,
        'train': (train_start, train_end, train_df),
        'val': (val_start, val_end, val_df),
        'test': (test_start, test_end, test_df)
    })
    
    print(f"   Ventana {i+1}:")
    print(f"      Train: {len(train_df)} ({train_start.date()} - {train_end.date()})")
    print(f"      Val:   {len(val_df)} ({val_start.date()} - {val_end.date()})")
    print(f"      Test:  {len(test_df)} ({test_start.date()} - {test_end.date()})")

print(f"\n✅ {len(windows)} ventanas temporales generadas")

# ─────────────────────────────────────────────────────────────
# ENTRENAMIENTO Y EVALUACIÓN POR VENTANA
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("ENTRENAMIENTO Y EVALUACIÓN POR VENTANA")
print("=" * 80)

window_results = []
draw_calibration_slopes = []

for window in windows:
    print(f"\n🔄 VENTANA {window['window_id']}:")
    
    # Extraer datos
    train_df = window['train'][2]
    val_df = window['val'][2]
    test_df = window['test'][2]
    
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df['outcome'].values
    
    X_val = val_df[feature_cols].fillna(0).values
    y_val = val_df['outcome'].values
    
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df['outcome'].values
    
    # Entrenar modelo base con regularización agresiva
    print(f"   Entrenando modelo base...")
    base_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,  # Reducido de default para evitar overfitting
        min_samples_leaf=20,  # Aumentado para regularización
        subsample=0.8,  # Feature dropout
        random_state=42
    )
    base_model.fit(X_train, y_train)
    
    # Calibrar con validation set
    print(f"   Calibrando...")
    calibrated_model = CalibratedClassifierCV(
        base_model,
        method='isotonic',
        cv=3  # Reducido a 3-fold para manejar conjuntos más pequeños
    )
    calibrated_model.fit(X_val, y_val)
    
    # Evaluar en test
    test_probs = calibrated_model.predict_proba(X_test)
    test_preds = calibrated_model.predict(X_test)
    
    test_accuracy = accuracy_score(y_test, test_preds)
    test_logloss = log_loss(y_test, test_probs)
    
    # Calibración específica de draw
    p_draw_test = test_probs[:, 1]
    y_draw_test = (y_test == 0).astype(int)
    
    try:
        fraction_pos, mean_pred = calibration_curve(y_draw_test, p_draw_test, n_bins=10)
        draw_slope = np.polyfit(mean_pred, fraction_pos, 1)[0]
    except:
        draw_slope = 0.0
    
    draw_calibration_slopes.append(draw_slope)
    
    print(f"   Accuracy: {test_accuracy:.4f}")
    print(f"   LogLoss:  {test_logloss:.4f}")
    print(f"   Draw Calibration Slope: {draw_slope:.4f}")
    
    window_results.append({
        'window_id': window['window_id'],
        'test_start': str(window['test'][0].date()),
        'test_end': str(window['test'][1].date()),
        'test_samples': len(y_test),
        'accuracy': float(test_accuracy),
        'logloss': float(test_logloss),
        'draw_slope': float(draw_slope)
    })

# ─────────────────────────────────────────────────────────────
# ANÁLISIS DE RESULTADOS POR VENTANA
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("ANÁLISIS DE RESULTADOS POR VENTANA")
print("=" * 80)

# Métricas agregadas
avg_accuracy = np.mean([w['accuracy'] for w in window_results])
avg_logloss = np.mean([w['logloss'] for w in window_results])
avg_draw_slope = np.mean(draw_calibration_slopes)

print(f"\n📊 MÉTRICAS AGREGADAS:")
print(f"   Accuracy promedio: {avg_accuracy:.4f}")
print(f"   LogLoss promedio:  {avg_logloss:.4f}")
print(f"   Draw Slope promedio: {avg_draw_slope:.4f}")

# Análisis de estabilidad de calibración de draw
draw_slope_std = np.std(draw_calibration_slopes)
print(f"\n📊 ESTABILIDAD DE CALIBRACIÓN DE DRAW:")
print(f"   Std de Draw Slope: {draw_slope_std:.4f}")

if draw_slope_std > 0.1:
    print(f"   ⚠️ ALTA VARIABILIDAD - Calibración de draw inestable entre ventanas")
    print(f"   Solución: Más regularización + features más robustas temporalmente")
elif draw_slope_std > 0.05:
    print(f"   ⚠️ VARIABILIDAD MODERADA - Monitorear drift temporal")
else:
    print(f"   ✅ ESTABLE - Calibración consistente entre ventanas")

# Detectar ventanas con calibración invertida
inverted_windows = [w for w in window_results if w['draw_slope'] < 0]
if inverted_windows:
    print(f"\n🔴 VENTANAS CON CALIBRACIÓN INVERTIDA:")
    for w in inverted_windows:
        print(f"   Ventana {w['window_id']}: Slope {w['draw_slope']:.4f} ({w['test_start']} - {w['test_end']})")
else:
    print(f"\n✅ Ninguna ventana tiene calibración invertida")

# Plot de Draw Slope por ventana
fig, ax = plt.subplots(figsize=(12, 6))
window_ids = [w['window_id'] for w in window_results]
slopes = [w['draw_slope'] for w in window_results]

ax.bar(window_ids, slopes, color='steelblue', alpha=0.7)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Cero (calibración perfecta)')
ax.axhline(y=avg_draw_slope, color='green', linestyle=':', linewidth=2, 
           label=f'Promedio ({avg_draw_slope:.3f})')
ax.set_xlabel('Ventana Temporal', fontsize=12)
ax.set_ylabel('Pendiente de Calibración Draw', fontsize=12)
ax.set_title('Estabilidad de Calibración de Draw por Ventana Temporal', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('draw_calibration_stability_windows.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Gráfico de estabilidad guardado: draw_calibration_stability_windows.png")

# ─────────────────────────────────────────────────────────────
# RESUMEN FINAL
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("RESUMEN FINAL DE PURGED TIME SERIES CV")
print("=" * 80)

cv_summary = {
    "n_windows": len(window_results),
    "avg_accuracy": float(avg_accuracy),
    "avg_logloss": float(avg_logloss),
    "avg_draw_slope": float(avg_draw_slope),
    "draw_slope_std": float(draw_slope_std),
    "inverted_windows_count": len(inverted_windows),
    "window_results": window_results,
    "configuration": {
        "n_windows": N_WINDOWS,
        "train_months": TRAIN_MONTHS,
        "val_months": VAL_MONTHS,
        "test_months": TEST_MONTHS,
        "gap_days": GAP_DAYS
    },
    "recommendation": "UNKNOWN"
}

# Recomendación basada en resultados
if len(inverted_windows) == 0 and draw_slope_std < 0.05:
    cv_summary["recommendation"] = "STABLE_CALIBRATION"
    print(f"\n✅ RECOMENDACIÓN: CALIBRACIÓN ESTABLE")
    print(f"   La calibración de draw es consistente entre ventanas temporales")
    print(f"   Se puede proceder con reentrenamiento usando esta arquitectura de validación")
elif len(inverted_windows) > len(window_results) / 2:
    cv_summary["recommendation"] = "SYSTEMATIC_INVERSION"
    print(f"\n🔴 RECOMENDACIÓN: INVERSIÓN SISTEMÁTICA")
    print(f"   Más de la mitad de las ventanas tienen calibración invertida")
    print(f"   Requiere: Auditoría profunda de features + posible reingeniería")
elif draw_slope_std > 0.1:
    cv_summary["recommendation"] = "HIGH_INSTABILITY"
    print(f"\n🔴 RECOMENDACIÓN: ALTA INESTABILIDAD")
    print(f"   Alta variabilidad en calibración entre ventanas")
    print(f"   Requiere: Más regularización + features más robustas temporalmente")
else:
    cv_summary["recommendation"] = "MODERATE_INSTABILITY"
    print(f"\n⚠️ RECOMENDACIÓN: INESTABILIDAD MODERADA")
    print(f"   Algunas ventanas tienen problemas de calibración")
    print(f"   Requiere: Regularización + monitoreo continuo")

# Guardar resumen
with open('purged_cv_summary.json', 'w') as f:
    json.dump(cv_summary, f, indent=2)
print(f"\n✅ Resumen guardado: purged_cv_summary.json")

print("\n" + "=" * 80)
print("PURGED TIME SERIES CV COMPLETADO")
print("=" * 80)
