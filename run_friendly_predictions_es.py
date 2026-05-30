# run_friendly_predictions_es.py
import os
import json
from datetime import datetime
from fifa_teams_database import FIFATeamsDatabase
from gbm_production import FIFA2026Predictor

def main():
    print("🤝 PREDICTOR DE AMISTOSOS FIFA 2026 (ESPAÑOL)")
    print("="*65)
    
    # 1. Cargar componentes
    db = FIFATeamsDatabase("fifa_teams_db_es.json")
    model_path = "gbm_wc2026_v1.joblib"
    if not os.path.exists(model_path):
        print(f"❌ Modelo no encontrado: {model_path}")
        return
    
    predictor = FIFA2026Predictor.load(model_path)
    print("✅ Modelo calibrado cargado correctamente.")

    # 2. Calendario (motivation: 0.80=experimental, 0.95=preparación seria)
    friendlies = [
        {"date": "2026-05-24", "home": "Portugal", "away": "Siria", "venue": "Lisboa", "neutral": False, "motivation": 0.88},
        {"date": "2026-05-24", "home": "España", "away": "Andorra", "venue": "Madrid", "neutral": False, "motivation": 0.85},
        {"date": "2026-05-25", "home": "Argentina", "away": "Guatemala", "venue": "Buenos Aires", "neutral": False, "motivation": 0.90},
        {"date": "2026-05-25", "home": "México", "away": "Ecuador", "venue": "Ciudad de México", "neutral": False, "motivation": 0.88},
        {"date": "2026-05-26", "home": "Japón", "away": "Paraguay", "venue": "Tokio", "neutral": False, "motivation": 0.87},
        {"date": "2026-05-26", "home": "Marruecos", "away": "Uganda", "venue": "Rabat", "neutral": False, "motivation": 0.86},
        {"date": "2026-05-27", "home": "Brasil", "away": "Corea del Sur", "venue": "Londres", "neutral": True, "motivation": 0.92},
        {"date": "2026-05-27", "home": "Francia", "away": "Canadá", "venue": "Saint-Denis", "neutral": False, "motivation": 0.90}
    ]

    results = []
    print(f"\n{'Fecha':<12} | {'Partido':<36} | {'Local%':<7} | {'Empate%':<7} | {'Visita%':<7} | {'Predicción'}")
    print("-" * 105)

    for m in friendlies:
        try:
            # 1. Calcular features inline usando la base FIFA
            elo_diff = db.get_elo_diff(m["home"], m["away"], neutral=m.get("neutral", False))
            features = {
                "elo_diff": elo_diff,
                "form_home": db.get_elo(m["home"]) / 2000.0,  # Normalizado ~0.8 para top teams
                "form_away": db.get_elo(m["away"]) / 2000.0,
                "h2h": 0.60 if db.get_elo(m["home"]) > db.get_elo(m["away"]) else 0.40,
                "neutral": 1.0 if m.get("neutral", False) else 0.0
            }
            
            # 2. Predicción base con modelo validado
            res = predictor.predict_match(m["home"], m["away"], features, neutral=m.get("neutral", False))
            
            # 3. Ajuste por naturaleza de amistoso (suavizar probabilidades)
            mot = m.get("motivation", 0.88)
            if mot < 1.0:
                p = res["probabilities"]
                p_adj = {k: v*mot + (1-mot)/3 for k, v in p.items()}
                total = sum(p_adj.values())
                res["probabilities"] = {k: v/total for k, v in p_adj.items()}
                res["prediction"]["confidence"] *= mot
                res["match_type"] = "friendly"
                
            p = res["probabilities"]
            conf = res["prediction"]["confidence"]
            out = res["prediction"]["outcome"]
            out_es = {"home_win": "V. Local", "draw": "Empate", "away_win": "V. Visitante"}[out]
            
            results.append({**m, **res, "prediccion_es": out_es})
            print(f"{m['date']:<12} | {m['home']} vs {m['away']:<24} | {p['home_win']*100:5.1f}% | {p['draw']*100:5.1f}% | {p['away_win']*100:5.1f}% | {out_es} ({conf*100:.0f}%)")
            
        except Exception as e:
            print(f"{m['date']:<12} | {m['home']} vs {m['away']:<24} | ❌ Error: {str(e)[:40]}")

    # 4. Exportar
    output = {
        "export_date": datetime.now().isoformat(),
        "modelo": "GBM_Calibrado_v1_ES",
        "total_predicciones": len(results),
        "predicciones": results
    }
    
    out_file = "amistosos_2026_predictions.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ {len(results)} predicciones exportadas: {out_file}")
    print("📅 Actualiza la lista `friendlies` con el calendario oficial.")
    print("🚀 Listo para validación en vivo.")

if __name__ == "__main__":
    main()
