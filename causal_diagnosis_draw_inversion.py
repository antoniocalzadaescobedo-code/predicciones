#!/usr/bin/env python3
"""
Protocolo Forense de Causa Raíz - Inversión de Calibración de Draws
================================================================
3 pasos para determinar POR QUÉ la calibración de draws está invertida
antes de decidir la intervención correcta.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import json
from sklearn.calibration import calibration_curve
from datetime import datetime

print("=" * 80)
print("PROTOCOL FORENSE DE CAUSA RAÍZ - INVERSIÓN DE CALIBRACIÓN DE DRAWS")
print("=" * 80)
print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ─────────────────────────────────────────────────────────────
# CARGAR MODELO Y DATOS
# ─────────────────────────────────────────────────────────────
print("📦 Cargando modelo y datos...")

try:
    package = joblib.load("gbm_wc2026_v2_temporal.joblib")
    model = package['model']
    feature_names = package['feature_names']
    print(f"✅ Modelo cargado")
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    exit(1)

try:
    df = pd.read_csv("matches_clean.csv")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date').reset_index(drop=True)
    print(f"✅ Dataset cargado: {len(df)} partidos")
except Exception as e:
    print(f"❌ Error cargando dataset: {e}")
    exit(1)

# ─────────────────────────────────────────────────────────────
# SPLIT TEMPORAL (TRAIN/VAL/TEST)
# ─────────────────────────────────────────────────────────────
print("\n📅 Aplicando split temporal...")

train_cutoff = pd.Timestamp('2023-01-01')
val_cutoff = pd.Timestamp('2024-01-01')

train_df = df[df['date'] < train_cutoff]
val_df = df[(df['date'] >= train_cutoff) & (df['date'] < val_cutoff)]
test_df = df[df['date'] >= val_cutoff]

print(f"Train: {len(train_df)} partidos")
print(f"Val: {len(val_df)} partidos")
print(f"Test: {len(test_df)} partidos")

feature_cols = ['elo_diff', 'form_home', 'form_away', 'h2h', 'neutral']

X_train = train_df[feature_cols].fillna(0).values
y_train = train_df['outcome'].values

X_val = val_df[feature_cols].fillna(0).values
y_val = val_df['outcome'].values

X_test = test_df[feature_cols].fillna(0).values
y_test = test_df['outcome'].values

# ─────────────────────────────────────────────────────────────
# GENERAR PREDICCIONES
# ─────────────────────────────────────────────────────────────
print("\n🔮 Generando predicciones...")

train_probs = model.predict_proba(X_train)
val_probs = model.predict_proba(X_val)
test_probs = model.predict_proba(X_test)

# Draw probabilities (índice 1 = draw)
p_draw_train = train_probs[:, 1]
p_draw_val = val_probs[:, 1]
p_draw_test = test_probs[:, 1]

y_draw_train = (y_train == 0).astype(int)
y_draw_val = (y_val == 0).astype(int)
y_draw_test = (y_test == 0).astype(int)

# ─────────────────────────────────────────────────────────────
# PASO 1: VERIFICAR SI LA INVERSIÓN EXISTE EN TRAIN
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("PASO 1: VERIFICAR INVERSIÓN EN TRAIN VS TEST")
print("=" * 80)

# Calibración en TRAIN
fraction_pos_train, mean_pred_train = calibration_curve(y_draw_train, p_draw_train, n_bins=15)
slope_train = np.polyfit(mean_pred_train, fraction_pos_train, 1)[0]

# Calibración en TEST
fraction_pos_test, mean_pred_test = calibration_curve(y_draw_test, p_draw_test, n_bins=15)
slope_test = np.polyfit(mean_pred_test, fraction_pos_test, 1)[0]

print(f"\n📊 PENDIENTE DE CALIBRACIÓN (Draw):")
print(f"   TRAIN: {slope_train:.4f}")
print(f"   TEST:  {slope_test:.4f}")
print(f"   VAL:   {np.polyfit(*calibration_curve(y_draw_val, p_draw_val, n_bins=15), 1)[0]:.4f}")

step1_diagnosis = {
    "slope_train": float(slope_train),
    "slope_val": float(np.polyfit(*calibration_curve(y_draw_val, p_draw_val, n_bins=15), 1)[0]),
    "slope_test": float(slope_test),
    "pattern": "UNKNOWN"
}

if slope_train > 0 and slope_test < 0:
    step1_diagnosis["pattern"] = "OVERFITTING_TEMPORAL"
    step1_diagnosis["explanation"] = "Pendiente positiva en train, negativa en test - Overfitting temporal CONFIRMADO"
    print(f"\n🔴 PATRÓN: OVERFITTING TEMPORAL CONFIRMADO")
    print(f"   El modelo memorizó rachas de empates del pasado que no se repiten en test")
    print(f"   Solución: Regularización más fuerte + menos features + validación temporal estricta")
elif slope_train < 0 and slope_test < 0:
    step1_diagnosis["pattern"] = "LABEL_LEAKAGE_OR_SPURIOUS"
    step1_diagnosis["explanation"] = "Pendiente negativa en train Y test - Label leakage o feature espuria"
    print(f"\n🔴 PATRÓN: LABEL LEAKAGE O FEATURE ESPURIA")
    print(f"   La inversión existe tanto en train como en test")
    print(f"   Solución: Eliminar feature espuria + reentrenar desde cero")
elif slope_train > 0 and slope_test > 0:
    step1_diagnosis["pattern"] = "NO_INVERSION"
    step1_diagnosis["explanation"] = "Pendiente positiva en ambos - No hay inversión"
    print(f"\n✅ PATRÓN: NO HAY INVERSIÓN")
    print(f"   La calibración es correcta en ambos conjuntos")
else:
    step1_diagnosis["pattern"] = "COMPLEX"
    step1_diagnosis["explanation"] = "Patrón complejo - Requiere análisis adicional"
    print(f"\n⚠️ PATRÓN: COMPLEJO - Requiere análisis adicional")

# Plot comparativo
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(mean_pred_train, fraction_pos_train, "s-", linewidth=2, markersize=8, 
        label="Train Calibration", color='blue')
ax1.plot([0, 1], [0, 1], "k:", linewidth=2, label="Perfectly calibrated")
ax1.set_xlabel("Mean Predicted Probability (p_draw)", fontsize=12)
ax1.set_ylabel("Fraction of Positives (Actual Draw)", fontsize=12)
ax1.set_title(f"TRAIN Calibration (Slope: {slope_train:.3f})", fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])

ax2.plot(mean_pred_test, fraction_pos_test, "s-", linewidth=2, markersize=8, 
        label="Test Calibration", color='red')
ax2.plot([0, 1], [0, 1], "k:", linewidth=2, label="Perfectly calibrated")
ax2.set_xlabel("Mean Predicted Probability (p_draw)", fontsize=12)
ax2.set_ylabel("Fraction of Positives (Actual Draw)", fontsize=12)
ax2.set_title(f"TEST Calibration (Slope: {slope_test:.3f})", fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('train_vs_test_calibration.png', dpi=150, bbox_inches='tight')
print(f"✅ Gráfico comparativo guardado: train_vs_test_calibration.png")

# ─────────────────────────────────────────────────────────────
# PASO 2: TEST DE ESTABILIDAD TEMPORAL
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("PASO 2: TEST DE ESTABILIDAD TEMPORAL")
print("=" * 80)

# Dividir test set en mitades cronológicas
midpoint = len(y_test) // 2

test_first_half = test_df.iloc[:midpoint]
test_second_half = test_df.iloc[midpoint:]

X_test_first = test_first_half[feature_cols].fillna(0).values
y_test_first = test_first_half['outcome'].values

X_test_second = test_second_half[feature_cols].fillna(0).values
y_test_second = test_second_half['outcome'].values

probs_first = model.predict_proba(X_test_first)
probs_second = model.predict_proba(X_test_second)

p_draw_first = probs_first[:, 1]
p_draw_second = probs_second[:, 1]

y_draw_first = (y_test_first == 0).astype(int)
y_draw_second = (y_test_second == 0).astype(int)

# Calibración en cada mitad
frac_first, pred_first = calibration_curve(y_draw_first, p_draw_first, n_bins=10)
frac_second, pred_second = calibration_curve(y_draw_second, p_draw_second, n_bins=10)

slope_first = np.polyfit(pred_first, frac_first, 1)[0]
slope_second = np.polyfit(pred_second, frac_second, 1)[0]

print(f"\n📊 PENDIENTE POR MITAD TEMPORAL:")
print(f"   Primera mitad: {slope_first:.4f}")
print(f"   Segunda mitad: {slope_second:.4f}")
print(f"   Diferencia: {abs(slope_first - slope_second):.4f}")

step2_diagnosis = {
    "slope_first_half": float(slope_first),
    "slope_second_half": float(slope_second),
    "slope_difference": float(abs(slope_first - slope_second)),
    "pattern": "UNKNOWN"
}

if abs(slope_first - slope_second) > 0.1:
    step2_diagnosis["pattern"] = "TEMPORAL_DRIFT"
    step2_diagnosis["explanation"] = "La inversión empeora con el tiempo - Drift/overfitting temporal"
    print(f"\n🔴 PATRÓN: TEMPORAL DRIFT")
    print(f"   La inversión empeora significativamente con el tiempo")
    print(f"   Solución: Validación temporal más estricta + regularización")
elif abs(slope_first - slope_second) < 0.05:
    step2_diagnosis["pattern"] = "STABLE_INVERSION"
    step2_diagnosis["explanation"] = "Inversión estable en ambas mitades - Problema estructural de features"
    print(f"\n🔴 PATRÓN: INVERSIÓN ESTABLE")
    print(f"   La inversión es estable en el tiempo")
    print(f"   Solución: Problema estructural de features - requiere ingeniería específica")
else:
    step2_diagnosis["pattern"] = "MODERATE_DRIFT"
    step2_diagnosis["explanation"] = "Drift moderado - Requiere monitoreo"
    print(f"\n⚠️ PATRÓN: DRIFT MODERADO")
    print(f"   Drift moderado - Requiere monitoreo")

# Plot temporal stability
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(pred_first, frac_first, "s-", linewidth=2, markersize=8, 
        label=f"First Half (Slope: {slope_first:.3f})", color='blue')
ax.plot(pred_second, frac_second, "o-", linewidth=2, markersize=8, 
        label=f"Second Half (Slope: {slope_second:.3f})", color='red')
ax.plot([0, 1], [0, 1], "k:", linewidth=2, label="Perfectly calibrated")
ax.set_xlabel("Mean Predicted Probability (p_draw)", fontsize=12)
ax.set_ylabel("Fraction of Positives (Actual Draw)", fontsize=12)
ax.set_title("Temporal Stability Test - Draw Calibration", fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.tight_layout()
plt.savefig('temporal_stability_test.png', dpi=150, bbox_inches='tight')
print(f"✅ Gráfico de estabilidad temporal guardado: temporal_stability_test.png")

# ─────────────────────────────────────────────────────────────
# PASO 3: ANÁLISIS DE IMPORTANCIA DE FEATURES PARA DRAW
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("PASO 3: ANÁLISIS DE IMPORTANCIA DE FEATURES PARA DRAW")
print("=" * 80)

# Permutation importance simplificado para clase Draw
from sklearn.metrics import log_loss

baseline_loss = log_loss(y_draw_test, p_draw_test)
print(f"\n📊 BASELINE LOGLOSS (Draw): {baseline_loss:.4f}")

feature_importance_draw = {}

for i, feature in enumerate(feature_cols):
    # Permutar feature
    X_test_permuted = X_test.copy()
    np.random.shuffle(X_test_permuted[:, i])
    
    # Recalcular probabilidades
    probs_permuted = model.predict_proba(X_test_permuted)
    p_draw_permuted = probs_permuted[:, 1]
    
    # Calcular nuevo loss
    permuted_loss = log_loss(y_draw_test, p_draw_permuted)
    
    # Importancia = aumento en loss
    importance = permuted_loss - baseline_loss
    feature_importance_draw[feature] = importance
    
    print(f"   {feature}: {importance:.4f}")

# Ordenar features por importancia
sorted_features = sorted(feature_importance_draw.items(), key=lambda x: -x[1])

print(f"\n📊 RANKING DE IMPORTANCIA PARA DRAW:")
for feature, importance in sorted_features:
    print(f"   {feature}: {importance:.4f}")

step3_diagnosis = {
    "feature_importance_draw": feature_importance_draw,
    "most_important_feature": sorted_features[0][0] if sorted_features else None,
    "pattern": "UNKNOWN"
}

# Diagnóstico de feature insuficiencia
if sorted_features[0][1] < 0.01:
    step3_diagnosis["pattern"] = "FEATURE_INSUFFICIENCY"
    step3_diagnosis["explanation"] = "Ninguna feature tiene impacto significativo en draw - Feature insuficiencia"
    print(f"\n🔴 PATRÓN: FEATURE INSUFFICIENCY")
    print(f"   Ninguna feature tiene impacto significativo en predicción de draws")
    print(f"   Solución: Ingeniería de features específica para draws (xG gap, Elo convergence, H2H draw rate)")
else:
    step3_diagnosis["pattern"] = "ADEQUATE_FEATURES"
    step3_diagnosis["explanation"] = "Features tienen impacto adecuado - El problema no es de insuficiencia"
    print(f"\n✅ PATRÓN: FEATURES ADECUADAS")
    print(f"   Las features tienen impacto adecuado - El problema no es de insuficiencia")

# ─────────────────────────────────────────────────────────────
# RESUMEN FINAL DE DIAGNÓSTICO CAUSAL
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("RESUMEN FINAL DE DIAGNÓSTICO CAUSAL")
print("=" * 80)

causal_diagnosis = {
    "step1_train_vs_test": step1_diagnosis,
    "step2_temporal_stability": step2_diagnosis,
    "step3_feature_importance": step3_diagnosis,
    "root_cause": "UNKNOWN",
    "recommended_action": "UNKNOWN",
    "diagnostic_date": datetime.now().isoformat()
}

# Determinar causa raíz y acción recomendada
if step1_diagnosis["pattern"] == "OVERFITTING_TEMPORAL":
    causal_diagnosis["root_cause"] = "OVERFITTING_TEMPORAL"
    causal_diagnosis["recommended_action"] = "Regularización más fuerte + menos features + validación temporal estricta"
    print(f"\n🔴 CAUSA RAÍZ: OVERFITTING TEMPORAL")
    print(f"   El modelo memorizó patrones temporales que no generalizan")
    print(f"   ACCIÓN: Regularización más fuerte + menos features + validación temporal estricta")
elif step1_diagnosis["pattern"] == "LABEL_LEAKAGE_OR_SPURIOUS":
    causal_diagnosis["root_cause"] = "LABEL_LEAKAGE_OR_SPURIOUS"
    causal_diagnosis["recommended_action"] = "Eliminar feature espuria + reentrenar desde cero"
    print(f"\n🔴 CAUSA RAÍZ: LABEL LEAKAGE O FEATURE ESPURIA")
    print(f"   Hay una feature correlacionada con draws en train pero no en test")
    print(f"   ACCIÓN: Eliminar feature espuria + reentrenar desde cero")
elif step2_diagnosis["pattern"] == "TEMPORAL_DRIFT":
    causal_diagnosis["root_cause"] = "TEMPORAL_DRIFT"
    causal_diagnosis["recommended_action"] = "Validación temporal más estricta + regularización"
    print(f"\n🔴 CAUSA RAÍZ: TEMPORAL DRIFT")
    print(f"   La inversión empeora con el tiempo")
    print(f"   ACCIÓN: Validación temporal más estricta + regularización")
elif step3_diagnosis["pattern"] == "FEATURE_INSUFFICIENCY":
    causal_diagnosis["root_cause"] = "FEATURE_INSUFFICIENCY"
    causal_diagnosis["recommended_action"] = "Ingeniería de features específica para draws"
    print(f"\n🔴 CAUSA RAÍZ: FEATURE INSUFFICIENCY")
    print(f"   No hay señal real para predecir draws")
    print(f"   ACCIÓN: Ingeniería de features específica (xG gap, Elo convergence, H2H draw rate)")
else:
    causal_diagnosis["root_cause"] = "COMPLEX"
    causal_diagnosis["recommended_action"] = "Análisis adicional requerido (SHAP values, feature engineering)"
    print(f"\n⚠️ CAUSA RAÍZ: COMPLEJA")
    print(f"   Requiere análisis adicional con SHAP values")
    print(f"   ACCIÓN: SHAP analysis + feature engineering específica")

# Guardar diagnóstico causal completo
with open('causal_diagnosis_draw_inversion.json', 'w') as f:
    json.dump(causal_diagnosis, f, indent=2)
print(f"\n✅ Diagnóstico causal guardado: causal_diagnosis_draw_inversion.json")

print("\n" + "=" * 80)
print("PROTOCOL FORENSE COMPLETADO")
print("=" * 80)
print(f"\n📋 PRÓXIMOS PASOS:")
print(f"   1. Revisar los 3 gráficos generados:")
print(f"      - train_vs_test_calibration.png")
print(f"      - temporal_stability_test.png")
print(f"   2. Leer el reporte JSON: causal_diagnosis_draw_inversion.json")
print(f"   3. Implementar la acción recomendada según causa raíz identificada")
