"""
RUN WORLDCUP PREDICTION - End-to-end para FIFA World Cup 2026
==============================================================

Script end-to-end para entrenar y evaluar el predictor GBM.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from gbm_production import FIFA2026Predictor
import json
import sys
sys.path.insert(0, 'C:\\Proyecto_FIFA')

def main():
    print("🏆 FIFA WORLD CUP 2026 — PREDICTOR GBM")
    print("=" * 60)
    print()
    
    # 1. CARGA DE DATOS
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
    
    # 2. FEATURE ENGINEERING WALK-FORWARD
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
    
    # 3. SPLIT TEMPORAL WALK-FORWARD
    print("Split temporal walk-forward...")
    split_date = '2023-01-01'
    train = df[df['date'] < pd.Timestamp(split_date)].copy()
    test = df[df['date'] >= pd.Timestamp(split_date)].copy()
    
    feature_cols = ['elo_diff', 'form_home', 'form_away', 'h2h', 'is_neutral']
    X_train = train[feature_cols].fillna(0)
    y_train = train['outcome'].values
    X_test = test[feature_cols].fillna(0)
    y_test = test['outcome'].values
    
    print(f"Train: {len(X_train)} partidos")
    print(f"Test: {len(X_test)} partidos")
    print()
    
    # 4. ENTRENAMIENTO
    print("🔧 Entrenando GBM Predictor...")
    predictor = FIFA2026Predictor(calibrate=True)  # Mejor calibración para apuestas/Monte Carlo
    print(f"   Modo: calibrate={predictor.calibrate} → Prioridad: Calibración (apuestas/Monte Carlo)")
    predictor.fit(X_train, y_train, feature_names=feature_cols)
    print("  [OK] Entrenamiento completado")
    print()
    
    # 5. EVALUACIÓN
    print("📊 Evaluando en test set...")
    metrics = predictor.evaluate(X_test, y_test)
    print()
    
    # 6. EXPORTAR RESULTADOS
    output = {
        'model_info': {
            'type': 'GradientBoostingClassifier',
            'classes': [-1, 0, 1],
            'class_labels': ['away_win', 'draw', 'home_win'],
            'calibrated': False,
            'feature_names': feature_cols
        },
        'performance': metrics,
        'sample_predictions': []
    }
    
    # Agregar 10 predicciones de ejemplo
    sample_idx = np.random.choice(len(X_test), min(10, len(X_test)), replace=False)
    for idx in sample_idx:
        row = test.iloc[idx]
        features = {col: row[col] for col in feature_cols}
        pred = predictor.predict_match(
            team_home=row['home_team'],
            team_away=row['away_team'],
            features=features,
            neutral=bool(row['is_neutral'])
        )
        pred['actual_outcome'] = int(row['outcome'])
        output['sample_predictions'].append(pred)
    
    # Guardar
    with open('worldcup2026_predictions.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Resultados exportados: worldcup2026_predictions.json")
    print(f"✅ Accuracy test: {metrics['accuracy']:.4f}")
    print()
    
    # 7. DEMO: Predicciones para partidos de interés
    print("🎯 Demo: Predicciones para partidos seleccionados")
    demo_matches = [
        {'home': 'Argentina', 'away': 'Brazil', 'elo_diff': 45.2, 'form_home': 0.75, 'form_away': 0.68, 'h2h': 0.55},
        {'home': 'Spain', 'away': 'Germany', 'elo_diff': -12.3, 'form_home': 0.62, 'form_away': 0.71, 'h2h': 0.48},
    ]
    
    for m in demo_matches:
        features = {k: v for k, v in m.items() if k in feature_cols}
        result = predictor.predict_match(m['home'], m['away'], features)
        p = result['probabilities']
        print(f"\n   {m['home']} vs {m['away']}")
        print(f"   🏠 Home: {p['home_win']*100:5.1f}% | ⚖ Draw: {p['draw']*100:5.1f}% | ✈ Away: {p['away_win']*100:5.1f}%")
        print(f"   → Predicción: {result['prediction']['outcome']} ({result['prediction']['confidence']*100:.1f}% confianza)")
    
    print()
    print("=" * 60)
    print("🚀 Predictor listo para integración con UI/API/Monte Carlo")
    print("=" * 60)

if __name__ == "__main__":
    main()
