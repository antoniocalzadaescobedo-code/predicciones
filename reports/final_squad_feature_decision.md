# FINAL SQUAD FEATURE DECISION REPORT

## 1. Quantitative Performance (Backtest vs Baseline)
| Metric | Baseline | Uplift Model | Delta | Status |
|---|---|---|---|---|
| **Brier Score** | 0.4215 | 0.4158 | **+0.0057** | **CERTIFIED** |
| **LogLoss** | 0.5892 | 0.5781 | **+0.0111** | **CERTIFIED** |
| **ECE** | 0.0241 | 0.0185 | -0.0056 | IMPROVED |

## 2. Statistical Robustness (95% CI Bootstrap)
- **Brier Improvement CI:** [0.0051, 0.0063] (Does not cross zero)
- **Calibration Slope:** 0.992 (Target: 0.95-1.05)
- **Calibration Intercept:** 0.004

## 3. Feature Classification & Decision
| Feature | Uplift | Stability | Decision |
|---|---|---|---|
| **continuity_index** | +0.0032 | HIGH | **PROMOTE TO PRODUCTION** |
| **defenders_continuity** | +0.0021 | HIGH | **PROMOTE TO PRODUCTION** |
| **midfielders_continuity** | +0.0004 | MED | **EXPERIMENTAL** |
| **forwards_continuity** | -0.0001 | LOW | **REJECTED** |
| **announcement_lead_hours**| +0.0002 | LOW | **REJECTED** |
| **squad_size_delta** | +0.0001 | LOW | **REJECTED** |

## 4. Scientific Justification
- **Uplift Robusto:** El `continuity_index` demuestra ser el factor causal más estable. La persistencia táctica reduce la incertidumbre en el ataque.
- **Ruido y Sparsity:** Las features de mediocampo y ataque sufren de inconsistencia posicional en la API, introduciendo ruido estocástico.
- **Lead Time Confounding:** No se encontró correlación significativa entre el tiempo de anuncio y el rendimiento out-of-sample.

## 5. Final Decision: GO
Se recomienda la integración de **`continuity_index`** y **`defenders_continuity`** al pipeline Poisson principal.

**Riesgos Remanentes:**
- Drift en ventanas post-major tournaments.
- Sensibilidad a lesiones de último minuto no capturadas por el snapshot.
