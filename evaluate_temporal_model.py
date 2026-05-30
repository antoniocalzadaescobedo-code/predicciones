#!/usr/bin/env python3
"""
Evaluación Completa del Modelo Temporal
=======================================
Genera métricas detalladas por clase, matriz de confusión, curvas de calibración,
Brier Score y comparación con baselines.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("EVALUACIÓN COMPLETA DEL MODELO TEMPORAL")
print("=" * 80)
print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ─────────────────────────────────────────────────────────────
# CARGAR DATOS Y MODELO
# ─────────────────────────────────────────────────────────────
print("📦 Cargando modelo y datos...")

try:
    package = joblib.load("gbm_wc2026_v2_temporal.joblib")
    model = package['model']
    feature_names = package['feature_names']
    metrics = package['metrics']
    print(f"✅ Modelo cargado: {metrics['train_samples']} train, {metrics['test_samples']} test")
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
# SPLIT TEMPORAL (MISMO QUE ENTRENAMIENTO)
# ─────────────────────────────────────────────────────────────
print("\n📅 Aplicando split temporal...")

train_cutoff = pd.Timestamp('2023-01-01')
val_cutoff = pd.Timestamp('2024-01-01')

test_df = df[df['date'] >= val_cutoff]
print(f"Test set: {len(test_df)} partidos ({test_df['date'].min()} - {test_df['date'].max()})")

feature_cols = ['elo_diff', 'form_home', 'form_away', 'h2h', 'neutral']
X_test = test_df[feature_cols].fillna(0).values
y_test = test_df['outcome'].values

# ─────────────────────────────────────────────────────────────
# PREDICCIONES
# ─────────────────────────────────────────────────────────────
print("\n🔮 Generando predicciones...")

test_probs = model.predict_proba(X_test)
test_preds = model.predict(X_test)

# Mapeo de clases
class_names = ['Away Win', 'Draw', 'Home Win']
class_map = {-1: 'Away Win', 0: 'Draw', 1: 'Home Win'}

# ─────────────────────────────────────────────────────────────
# 1. MÉTRICAS POR CLASE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("1. MÉTRICAS POR CLASE (Precision/Recall/F1)")
print("=" * 80)

report = classification_report(y_test, test_preds, target_names=class_names, output_dict=True)
print(classification_report(y_test, test_preds, target_names=class_names))

# Guardar métricas por clase
class_metrics = {
    'home_win': {
        'precision': report['Home Win']['precision'],
        'recall': report['Home Win']['recall'],
        'f1': report['Home Win']['f1-score'],
        'support': int(report['Home Win']['support'])
    },
    'draw': {
        'precision': report['Draw']['precision'],
        'recall': report['Draw']['recall'],
        'f1': report['Draw']['f1-score'],
        'support': int(report['Draw']['support'])
    },
    'away_win': {
        'precision': report['Away Win']['precision'],
        'recall': report['Away Win']['recall'],
        'f1': report['Away Win']['f1-score'],
        'support': int(report['Away Win']['support'])
    }
}

print(f"\n📊 Análisis de Balance de Clases:")
print(f"   Home Win: {class_metrics['home_win']['support']} muestras ({class_metrics['home_win']['support']/len(y_test)*100:.1f}%)")
print(f"   Draw:     {class_metrics['draw']['support']} muestras ({class_metrics['draw']['support']/len(y_test)*100:.1f}%)")
print(f"   Away Win: {class_metrics['away_win']['support']} muestras ({class_metrics['away_win']['support']/len(y_test)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────
# 2. MATRIZ DE CONFUSIÓN
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("2. MATRIZ DE CONFUSIÓN")
print("=" * 80)

cm = confusion_matrix(y_test, test_preds)
print("\nMatriz de Confusión:")
print("                Predicho")
print("Real        Away  Draw  Home")
print(f"Away Win    {cm[0,0]:4d}  {cm[0,1]:4d}  {cm[0,2]:4d}")
print(f"Draw        {cm[1,0]:4d}  {cm[1,1]:4d}  {cm[1,2]:4d}")
print(f"Home Win    {cm[2,0]:4d}  {cm[2,1]:4d}  {cm[2,2]:4d}")

# Análisis de errores
print(f"\n📊 Análisis de Errores:")
print(f"   Draws confundidos con Home: {cm[1,2]} ({cm[1,2]/cm[1].sum()*100:.1f}% de draws)")
print(f"   Draws confundidos con Away: {cm[1,0]} ({cm[1,0]/cm[1].sum()*100:.1f}% de draws)")
print(f"   Away Wins confundidos con Draw: {cm[0,1]} ({cm[0,1]/cm[0].sum()*100:.1f}% de away wins)")
print(f"   Home Wins confundidos con Draw: {cm[2,1]} ({cm[2,1]/cm[2].sum()*100:.1f}% de home wins)")

# ─────────────────────────────────────────────────────────────
# 3. BRIER SCORE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("3. BRIER SCORE (Calibración de Probabilidades)")
print("=" * 80)

# One-hot encoding para Brier multiclase
y_test_oh = np.zeros((len(y_test), 3))
for i, val in enumerate(y_test):
    idx = np.where(model.classes_ == val)[0][0]
    y_test_oh[i, idx] = 1.0

brier_scores = []
for i in range(3):
    brier = brier_score_loss(y_test_oh[:, i], test_probs[:, i])
    brier_scores.append(brier)
    print(f"   {class_names[i]}: {brier:.4f}")

brier_mean = np.mean(brier_scores)
print(f"\n   Brier Score Promedio: {brier_mean:.4f}")

# ─────────────────────────────────────────────────────────────
# 4. CURVAS DE CALIBRACIÓN
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("4. CURVAS DE CALIBRACIÓN")
print("=" * 80)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Curvas de Calibración por Clase', fontsize=14, fontweight='bold')

for i, (ax, class_name) in enumerate(zip(axes, class_names)):
    prob_true, prob_pred = calibration_curve(y_test_oh[:, i], test_probs[:, i], n_bins=10)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectamente Calibrado')
    ax.plot(prob_pred, prob_true, 'o-', linewidth=2, label='Modelo')
    ax.set_xlabel('Probabilidad Predicha', fontsize=10)
    ax.set_ylabel('Probabilidad Observada', fontsize=10)
    ax.set_title(class_name, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('calibration_curves.png', dpi=150, bbox_inches='tight')
print("✅ Curvas de calibración guardadas: calibration_curves.png")

# ─────────────────────────────────────────────────────────────
# 5. BASELINE COMPARISON
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("5. COMPARACIÓN CON BASELINES")
print("=" * 80)

# Baseline 1: Siempre Home
baseline_home_preds = np.ones(len(y_test))
baseline_home_acc = (baseline_home_preds == (y_test == 1).astype(int)).mean()
print(f"   Baseline 'Siempre Home': {baseline_home_acc:.4f} ({baseline_home_acc*100:.1f}%)")

# Baseline 2: Siempre Away
baseline_away_preds = np.ones(len(y_test))
baseline_away_acc = (baseline_away_preds == (y_test == -1).astype(int)).mean()
print(f"   Baseline 'Siempre Away': {baseline_away_acc:.4f} ({baseline_away_acc*100:.1f}%)")

# Baseline 3: Aleatorio (proporcional a clases)
class_counts = np.bincount(y_test + 1)  # +1 para convertir -1,0,1 a 0,1,2
class_probs = class_counts / len(y_test)
baseline_random_preds = np.random.choice([-1, 0, 1], size=len(y_test), p=class_probs)
baseline_random_acc = (baseline_random_preds == y_test).mean()
print(f"   Baseline 'Aleatorio': {baseline_random_acc:.4f} ({baseline_random_acc*100:.1f}%)")

# Modelo actual
model_acc = (test_preds == y_test).mean()
print(f"\n   Modelo GBM Temporal: {model_acc:.4f} ({model_acc*100:.1f}%)")

print(f"\n📊 Mejora vs Baseline:")
print(f"   vs 'Siempre Home': {(model_acc - baseline_home_acc)*100:+.1f}%")
print(f"   vs 'Siempre Away': {(model_acc - baseline_away_acc)*100:+.1f}%")
print(f"   vs 'Aleatorio': {(model_acc - baseline_random_acc)*100:+.1f}%")

# ─────────────────────────────────────────────────────────────
# 6. RESUMEN FINAL
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("RESUMEN FINAL DE EVALUACIÓN")
print("=" * 80)

summary = {
    'test_accuracy': float(model_acc),
    'test_brier_score': float(brier_mean),
    'class_metrics': class_metrics,
    'baseline_comparison': {
        'always_home': float(baseline_home_acc),
        'always_away': float(baseline_away_acc),
        'random': float(baseline_random_acc)
    },
    'confusion_matrix': cm.tolist(),
    'test_samples': len(y_test),
    'evaluation_date': datetime.now().isoformat()
}

print(f"\n📊 Métricas Principales:")
print(f"   Accuracy: {model_acc:.4f} ({model_acc*100:.1f}%)")
print(f"   Brier Score: {brier_mean:.4f}")
print(f"   Home Win Recall: {class_metrics['home_win']['recall']:.4f}")
print(f"   Draw Recall: {class_metrics['draw']['recall']:.4f}")
print(f"   Away Win Recall: {class_metrics['away_win']['recall']:.4f}")

# Guardar resumen
import json
with open('evaluation_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\n✅ Resumen guardado: evaluation_summary.json")

print("\n" + "=" * 80)
print("EVALUACIÓN COMPLETADA")
print("=" * 80)
