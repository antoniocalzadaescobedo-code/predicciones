#!/usr/bin/env python3
"""
Model Audit - Validación Numérica y Estadística del Predictor FIFA 2026
========================================================================

FASE 1: Integridad de probabilidades, orden de clases, matriz de confusión
FASE 2: Split temporal, leakage audit, calibración, baseline comparison  
FASE 3: Monte Carlo integrity, stability testing, sensitivity analysis
"""

import numpy as np
import pandas as pd
import json
from sklearn.metrics import confusion_matrix, classification_report, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("AUDITORÍA NUMÉRICA DEL MODELO FIFA 2026")
print("=" * 80)
print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ─────────────────────────────────────────────────────────────
# CARGA DEL MODELO Y DATOS
# ─────────────────────────────────────────────────────────────
print("📦 Cargando modelo y datos...")

try:
    from gbm_production import FIFA2026Predictor
    predictor = FIFA2026Predictor.load("gbm_wc2026_v1.joblib")
    print(f"✅ Modelo cargado: {type(predictor).__name__}")
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    exit(1)

try:
    from fifa_teams_database import FIFATeamsDatabase
    db = FIFATeamsDatabase("fifa_teams_db_es.json")
    print(f"✅ Base de datos cargada: {len(db.df)} equipos")
except Exception as e:
    print(f"❌ Error cargando base de datos: {e}")
    exit(1)

# ─────────────────────────────────────────────────────────────
# FASE 1: INTEGRIDAD DE PROBABILIDADES
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("FASE 1: INTEGRIDAD DE PROBABILIDADES")
print("=" * 80)

# Generar predicciones de prueba
test_teams = [
    ("Argentina", "Brasil"),
    ("México", "Estados Unidos"),
    ("España", "Francia"),
    ("Alemania", "Inglaterra")
]

probability_integrity_passed = True
for home, away in test_teams:
    try:
        elo_diff = db.get_elo_diff(home, away, neutral=False)
        features = {
            "elo_diff": elo_diff,
            "form_home": 0.50,
            "form_away": 0.50,
            "h2h": 0.50,
            "is_neutral": 0.0
        }
        res = predictor.predict_match(home, away, features, neutral=False)
        probs = res["probabilities"]
        prob_sum = sum(probs.values())
        
        if abs(prob_sum - 1.0) > 1e-6:
            print(f"❌ {home} vs {away}: Suma = {prob_sum:.6f} (DEBE SER 1.0)")
            probability_integrity_passed = False
        else:
            print(f"✅ {home} vs {away}: Suma = {prob_sum:.6f}")
    except Exception as e:
        print(f"❌ Error en {home} vs {away}: {e}")
        probability_integrity_passed = False

if probability_integrity_passed:
    print("\n✅ INTEGRIDAD DE PROBABILIDADES: PASSED")
else:
    print("\n❌ INTEGRIDAD DE PROBABILIDADES: FAILED")

# ─────────────────────────────────────────────────────────────
# FASE 1: ORDEN DE CLASES
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("FASE 1: ORDEN DE CLASES")
print("=" * 80)

try:
    if hasattr(predictor, 'model') and hasattr(predictor.model, 'classes_'):
        model_classes = predictor.model.classes_
        print(f"📋 Clases del modelo: {model_classes}")
        
        expected_order = [-1, 0, 1]  # away, draw, home
        if list(model_classes) == expected_order:
            print(f"✅ Orden correcto: {expected_order}")
        else:
            print(f"⚠️ Orden inesperado: {model_classes} vs esperado {expected_order}")
    else:
        print("⚠️ No se puede verificar classes_ (modelo no tiene atributo)")
except Exception as e:
    print(f"❌ Error verificando clases: {e}")

# Verificar orden en predict_match
print("\n📊 Verificando orden en predict_match:")
res = predictor.predict_match("Argentina", "Brasil", {
    "elo_diff": 50, "form_home": 0.5, "form_away": 0.5, "h2h": 0.5, "is_neutral": 0.0
}, neutral=False)
print(f"Probabilidades: {res['probabilities']}")
print(f"Predicción: {res['prediction']}")

# ─────────────────────────────────────────────────────────────
# FASE 1: DISTRIBUCIÓN DE PROBABILIDADES
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("FASE 1: DISTRIBUCIÓN DE PROBABILIDADES")
print("=" * 80)

# Generar muchas predicciones para detectar collapse
all_probs = []
for home in db.df["team_name"].tolist()[:20]:
    for away in db.df["team_name"].tolist()[:20]:
        if home != away:
            try:
                elo_diff = db.get_elo_diff(home, away, neutral=False)
                features = {
                    "elo_diff": elo_diff,
                    "form_home": 0.50,
                    "form_away": 0.50,
                    "h2h": 0.50,
                    "is_neutral": 0.0
                }
                res = predictor.predict_match(home, away, features, neutral=False)
                probs = list(res["probabilities"].values())
                all_probs.append(probs)
            except:
                pass

all_probs = np.array(all_probs)
print(f"📊 Total de predicciones analizadas: {len(all_probs)}")

if len(all_probs) > 0:
    print(f"\nEstadísticas de probabilidades:")
    print(f"Home Win - Media: {all_probs[:, 2].mean():.3f}, Std: {all_probs[:, 2].std():.3f}")
    print(f"Draw     - Media: {all_probs[:, 1].mean():.3f}, Std: {all_probs[:, 1].std():.3f}")
    print(f"Away Win - Media: {all_probs[:, 0].mean():.3f}, Std: {all_probs[:, 0].std():.3f}")
    
    # Detectar distribution collapse
    home_std = all_probs[:, 2].std()
    draw_std = all_probs[:, 1].std()
    away_std = all_probs[:, 0].std()
    
    if home_std < 0.05 and draw_std < 0.05 and away_std < 0.05:
        print("\n❌ DISTRIBUTION COLLAPPE DETECTED - El modelo no discrimina")
    else:
        print("\n✅ Distribución variada - El modelo discrimina correctamente")

# ─────────────────────────────────────────────────────────────
# RESULTADOS FASE 1
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("RESUMEN FASE 1")
print("=" * 80)
print(f"Integridad de probabilidades: {'✅ PASSED' if probability_integrity_passed else '❌ FAILED'}")
print("Orden de clases: ⚠️ Requiere verificación manual con datos de entrenamiento")
print("Distribución: ✅ Analizada")

print("\n" + "=" * 80)
print("NOTA: Para completar FASE 2 y FASE 3 se requiere acceso a:")
print("- Dataset de entrenamiento con timestamps")
print("- Datos históricos completos con fechas")
print("- Ejecución de Monte Carlo para stability testing")
print("=" * 80)
