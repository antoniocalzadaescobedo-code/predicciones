#!/usr/bin/env python3
"""
Diagnóstico Forense de Compresión de Probabilidad de Empate
============================================================
Protocolo de 3 auditorías para cuantificar y caracterizar la compresión
de p_draw antes de decidir la intervención (calibración vs reentrenamiento).
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import json
from sklearn.calibration import calibration_curve
from datetime import datetime

print("=" * 80)
print("DIAGNÓSTICO FORENSE DE COMPRESIÓN DE PROBABILIDAD DE EMPATE")
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
# SPLIT TEMPORAL (TEST SET)
# ─────────────────────────────────────────────────────────────
print("\n📅 Aplicando split temporal (test set)...")

val_cutoff = pd.Timestamp('2024-01-01')
test_df = df[df['date'] >= val_cutoff]
print(f"Test set: {len(test_df)} partidos")

feature_cols = ['elo_diff', 'form_home', 'form_away', 'h2h', 'neutral']
X_test = test_df[feature_cols].fillna(0).values
y_test = test_df['outcome'].values

# ─────────────────────────────────────────────────────────────
# GENERAR PREDICCIONES
# ─────────────────────────────────────────────────────────────
print("\n🔮 Generando predicciones...")

test_probs = model.predict_proba(X_test)
test_preds = model.predict(X_test)

# Model classes are [-1, 0, 1] = [away, draw, home]
p_draw = test_probs[:, 1]  # Draw probability
y_draw = (y_test == 0).astype(int)  # Draw indicator (0 = draw in model classes)

# ─────────────────────────────────────────────────────────────
# AUDITORÍA 1: AUTOPSIA DE DISTRIBUCIÓN P_DRAW
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("AUDITORÍA 1: AUTOPSIA DE DISTRIBUCIÓN P_DRAW")
print("=" * 80)

print(f"\n📊 ESTADÍSTICAS P_DRAW:")
print(f"   Mean:   {p_draw.mean():.4f}")
print(f"   Std:    {p_draw.std():.4f}")
print(f"   Min:    {p_draw.min():.4f}")
print(f"   Max:    {p_draw.max():.4f}")
print(f"   P5:     {np.percentile(p_draw, 5):.4f}")
print(f"   P25:    {np.percentile(p_draw, 25):.4f}")
print(f"   P50:    {np.percentile(p_draw, 50):.4f}")
print(f"   P75:    {np.percentile(p_draw, 75):.4f}")
print(f"   P95:    {np.percentile(p_draw, 95):.4f}")
print(f"   Real Draw Rate: {y_draw.mean():.4f}")

# Density analysis
density_20_35 = ((p_draw >= 0.20) & (p_draw <= 0.35)).mean()
density_10_20 = ((p_draw >= 0.10) & (p_draw <= 0.20)).mean()
density_35_plus = (p_draw > 0.35).mean()

print(f"\n📊 ANÁLISIS DE DENSIDAD:")
print(f"   Density [0.10-0.20]: {density_10_20:.2%}")
print(f"   Density [0.20-0.35]: {density_20_35:.2%}")
print(f"   Density [0.35+]:     {density_35_plus:.2%}")

# Histogram
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(p_draw, bins=50, density=True, alpha=0.7, label='p_draw model', color='steelblue')
ax.axvline(x=y_draw.mean(), color='red', linestyle='--', linewidth=2, 
           label=f'Real Draw Rate ({y_draw.mean():.2%})')
ax.axvline(x=p_draw.mean(), color='green', linestyle=':', linewidth=2,
           label=f'Model Mean ({p_draw.mean():.2%})')
ax.set_xlabel('Probabilidad de Empate (p_draw)', fontsize=12)
ax.set_ylabel('Densidad', fontsize=12)
ax.set_title('Distribución de Probabilidad de Empate - Autopsia', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('draw_distribution_autopsy.png', dpi=150, bbox_inches='tight')
print(f"✅ Histograma guardado: draw_distribution_autopsy.png")

# Diagnóstico de compresión
compression_diagnosis = {
    "mean": float(p_draw.mean()),
    "std": float(p_draw.std()),
    "min": float(p_draw.min()),
    "max": float(p_draw.max()),
    "p95": float(np.percentile(p_draw, 95)),
    "real_draw_rate": float(y_draw.mean()),
    "density_10_20": float(density_10_20),
    "density_20_35": float(density_20_35),
    "density_35_plus": float(density_35_plus)
}

if p_draw.std() < 0.05:
    compression_diagnosis["pattern"] = "NARROW_BELL"
    compression_diagnosis["severity"] = "HIGH"
    print(f"\n⚠️ PATRÓN: Campana estrecha (std={p_draw.std():.4f}) - Compresión CONFIRMADA")
elif p_draw.max() < 0.30:
    compression_diagnosis["pattern"] = "TRUNCATED_TAIL"
    compression_diagnosis["severity"] = "HIGH"
    print(f"\n⚠️ PATRÓN: Cola truncada (max={p_draw.max():.4f}) - Sin rango dinámico superior")
else:
    compression_diagnosis["pattern"] = "NORMAL"
    compression_diagnosis["severity"] = "LOW"
    print(f"\n✅ PATRÓN: Distribución normal - Sin compresión significativa")

# ─────────────────────────────────────────────────────────────
# AUDITORÍA 2: CURVA DE CALIBRACIÓN ESPECÍFICA DE DRAW
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("AUDITORÍA 2: CURVA DE CALIBRACIÓN ESPECÍFICA DE DRAW")
print("=" * 80)

fraction_pos, mean_predicted = calibration_curve(y_draw, p_draw, n_bins=15)

print(f"\n📊 PUNTOS DE CALIBRACIÓN (Draw):")
print(f"   Bin   | Predicho | Observado | Gap")
print(f"   " + "-" * 40)
for i, (pred, obs) in enumerate(zip(mean_predicted, fraction_pos)):
    gap = obs - pred
    print(f"   {i:2d}    | {pred:.3f}    | {obs:.3f}     | {gap:+.3f}")

# Plot calibration curve
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(mean_predicted, fraction_pos, "s-", linewidth=2, markersize=8, 
        label="Draw Calibration", color='darkorange')
ax.plot([0, 1], [0, 1], "k:", linewidth=2, label="Perfectly calibrated")
ax.set_xlabel("Mean Predicted Probability (p_draw)", fontsize=12)
ax.set_ylabel("Fraction of Positives (Actual Draw)", fontsize=12)
ax.set_title("Calibration Curve: DRAW ONLY", fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.tight_layout()
plt.savefig('draw_calibration_curve.png', dpi=150, bbox_inches='tight')
print(f"✅ Curva de calibración guardada: draw_calibration_curve.png")

# Diagnóstico de patrón de calibración
slope = np.polyfit(mean_predicted, fraction_pos, 1)[0]
calibration_diagnosis = {
    "slope": float(slope),
    "pattern": "UNKNOWN"
}

if abs(slope - 1.0) < 0.2:
    calibration_diagnosis["pattern"] = "LINEAR_COMPRESSION"
    calibration_diagnosis["solution"] = "Isotonic/Platt scaling post-hoc podría bastar"
    print(f"\n⚠️ PATRÓN: Pendiente {slope:.3f} ~ 1.0 - Compresión LINEAL uniforme")
    print(f"   Solución: Calibración post-hoc específica para Draw")
elif slope < 0.5:
    calibration_diagnosis["pattern"] = "S_SHAPED_FLATTENED"
    calibration_diagnosis["solution"] = "Reentrenamiento con class weights"
    print(f"\n🔴 PATRÓN: Pendiente {slope:.3f} < 0.5 - Forma S achatada")
    print(f"   Solución: Reentrenamiento con class_weight=2.0-3.0")
elif slope < 0.2:
    calibration_diagnosis["pattern"] = "COLLAPSE"
    calibration_diagnosis["solution"] = "Auditoría de features o loss rota"
    print(f"\n🔴 PATRÓN: Pendiente {slope:.3f} < 0.2 - COLAPSO TOTAL")
    print(f"   Solución: Auditoría de features específicas de empate")
else:
    calibration_diagnosis["pattern"] = "HIGH_VARIANCE"
    calibration_diagnosis["solution"] = "Ruido puro, no hay aprendizaje real"
    print(f"\n⚠️ PATRÓN: Alta dispersión sin tendencia - Ruido puro")

# ─────────────────────────────────────────────────────────────
# AUDITORÍA 3: ANÁLISIS DE RANKING RELATIVO
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("AUDITORÍA 3: ANÁLISIS DE RANKING RELATIVO")
print("=" * 80)

p_away = test_probs[:, 0]
p_home = test_probs[:, 2]

max_ha = np.maximum(p_away, p_home)
gap_to_max = max_ha - p_draw

print(f"\n📊 GAP ANÁLISIS (max(H,A) - D):")
print(f"   Gap medio:   {gap_to_max.mean():.4f}")
print(f"   Gap mediano: {np.median(gap_to_max):.4f}")
print(f"   Gap std:     {gap_to_max.std():.4f}")
print(f"   Gap min:     {gap_to_max.min():.4f}")
print(f"   Gap max:     {gap_to_max.max():.4f}")

draw_dominates = (p_draw > max_ha).mean()
print(f"\n📊 DOMINANCIA DE DRAW:")
print(f"   % partidos donde D > max(H,A): {draw_dominates:.2%}")
print(f"   % partidos donde D > 0.30:     {(p_draw > 0.30).mean():.2%}")
print(f"   % partidos donde D > 0.25:     {(p_draw > 0.25).mean():.2%}")

ranking_diagnosis = {
    "gap_mean": float(gap_to_max.mean()),
    "gap_median": float(np.median(gap_to_max)),
    "draw_dominates_rate": float(draw_dominates),
    "pattern": "UNKNOWN"
}

if gap_to_max.mean() > 0.15 and draw_dominates < 0.02:
    ranking_diagnosis["pattern"] = "SEVERE_SUPPRESSION"
    ranking_diagnosis["solution"] = "Boost masivo en señal de empate durante entrenamiento"
    print(f"\n🔴 PATRÓN: Gap medio {gap_to_max.mean():.4f} > 0.15 y D domina < 2%")
    print(f"   Solución: Boost masivo en señal de empate durante entrenamiento")
elif gap_to_max.mean() > 0.10:
    ranking_diagnosis["pattern"] = "MODERATE_SUPPRESSION"
    ranking_diagnosis["solution"] = "Boost moderado + class weights"
    print(f"\n⚠️ PATRÓN: Gap medio {gap_to_max.mean():.4f} > 0.10 - Supresión moderada")
    print(f"   Solución: Boost moderado + class weights")
else:
    ranking_diagnosis["pattern"] = "ACCEPTABLE"
    ranking_diagnosis["solution"] = "No requiere intervención mayor"
    print(f"\n✅ PATRÓN: Gap aceptable - No requiere intervención mayor")

# ─────────────────────────────────────────────────────────────
# RESUMEN FINAL DE DIAGNÓSTICO
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("RESUMEN FINAL DE DIAGNÓSTICO")
print("=" * 80)

final_diagnosis = {
    "compression_pattern": compression_diagnosis["pattern"],
    "compression_severity": compression_diagnosis["severity"],
    "calibration_pattern": calibration_diagnosis["pattern"],
    "calibration_solution": calibration_diagnosis["solution"],
    "ranking_pattern": ranking_diagnosis["pattern"],
    "ranking_solution": ranking_diagnosis["solution"],
    "recommended_intervention": "UNKNOWN",
    "diagnostic_date": datetime.now().isoformat()
}

# Determinar intervención recomendada
if (compression_diagnosis["severity"] == "HIGH" and 
    calibration_diagnosis["pattern"] == "COLLAPSE"):
    final_diagnosis["recommended_intervention"] = "FEATURE_ENGINEERING"
    print(f"\n🔴 INTERVENCIÓN RECOMENDADA: FEATURE ENGINEERING")
    print(f"   El modelo tiene colapso estructural en empates.")
    print(f"   Requiere: Auditoría de features específicas de empate")
    print(f"   (diferencia de Elo, xG gap, historial H2H draws, etc.)")
elif calibration_diagnosis["pattern"] == "LINEAR_COMPRESSION":
    final_diagnosis["recommended_intervention"] = "POST_HOC_CALIBRATION"
    print(f"\n⚠️ INTERVENCIÓN RECOMENDADA: CALIBRACIÓN POST-HOC")
    print(f"   Compresión lineal uniforme - escalable.")
    print(f"   Requiere: Isotonic/Platt scaling específico para Draw con renormalización.")
elif calibration_diagnosis["pattern"] == "S_SHAPED_FLATTENED":
    final_diagnosis["recommended_intervention"] = "RETRAIN_WITH_WEIGHTS"
    print(f"\n🔴 INTERVENCIÓN RECOMENDADA: REENTRENAMIENTO CON CLASS WEIGHTS")
    print(f"   Subestimación en medios, sobre en extremos.")
    print(f"   Requiere: Reentrenamiento con class_weight={{'D': 2.0-3.0}} + early stopping.")
elif ranking_diagnosis["pattern"] == "SEVERE_SUPPRESSION":
    final_diagnosis["recommended_intervention"] = "TRAINING_BOOST"
    print(f"\n🔴 INTERVENCIÓN RECOMENDADA: BOOST EN ENTRENAMIENTO")
    print(f"   El modelo necesita boost masivo en señal de empate.")
    print(f"   Requiere: Modificar loss function o agregar features específicos.")
else:
    final_diagnosis["recommended_intervention"] = "MINOR_ADJUSTMENT"
    print(f"\n✅ INTERVENCIÓN RECOMENDADA: AJUSTE MENOR")
    print(f"   La compresión es manejable con ajustes menores.")

# Guardar diagnóstico completo
diagnostic_report = {
    "compression_analysis": compression_diagnosis,
    "calibration_analysis": calibration_diagnosis,
    "ranking_analysis": ranking_diagnosis,
    "final_diagnosis": final_diagnosis
}

with open('draw_compression_diagnostic.json', 'w') as f:
    json.dump(diagnostic_report, f, indent=2)
print(f"\n✅ Reporte de diagnóstico guardado: draw_compression_diagnostic.json")

print("\n" + "=" * 80)
print("DIAGNÓSTICO FORENSE COMPLETADO")
print("=" * 80)
print(f"\n📋 PRÓXIMOS PASOS:")
print(f"   1. Revisar los 3 gráficos generados:")
print(f"      - draw_distribution_autopsy.png")
print(f"      - draw_calibration_curve.png")
print(f"   2. Leer el reporte JSON: draw_compression_diagnostic.json")
print(f"   3. Implementar la intervención recomendada según diagnóstico")
