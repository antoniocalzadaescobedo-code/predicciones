# Production Readiness Report: Squad Temporal Features
**Date:** 2026-05-28
**Phase:** G - Safe Production Integration

## 1. Feature Governance Table
| Feature | Status | Impact (Brier) | Drift Risk | Owner |
|---|---|---|---|---|
| **continuity_index** | APPROVED | +0.0032 | LOW | Principal_ML_Engineer |
| **defenders_continuity** | APPROVED | +0.0021 | MEDIUM | Principal_ML_Engineer |
| **midfielders_continuity** | EXPERIMENTAL | +0.0004 | HIGH | Principal_ML_Engineer |
| **forwards_continuity** | REJECTED | -0.0001 | HIGH | Principal_ML_Engineer |

## 2. Operational Guardrails (Automatic Rollback)
The system will automatically fallback to the **Baseline Model** if any of the following triggers occur:
- **PSI > 0.20:** Significant distribution shift in squad continuity.
- **Calibration Slope Shift > 5%:** Evidence of overconfidence or underconfidence.
- **Missing Data > 10%:** If approved features are missing for more than 10% of live matches.
- **Temporal Leakage:** Any request where `feature_timestamp >= kickoff_timestamp` will trigger a hard-fail for that prediction.

## 3. Shadow Deployment Strategy
The system is currently in **Shadow Mode**. 
- Uplifted expectations are logged but not used for final probability output.
- Disagreement analysis is performed daily to compare Baseline vs Uplifted ROI.

## 4. Post-Mundial 2026 Monitoring Plan
A special monitoring regime will be active from **2026-07-20 to 2026-09-20**.
- **Regime Change Detector:** High sensitivity to Jaccard index drops (expected due to retirements/generational change).
- **Manual Override:** The production owner can disable squad features via the `toggle_squad_features` flag if instability is confirmed.

## 5. Monitoring Checklist
- [ ] Daily PSI report for `continuity_index`.
- [ ] Weekly calibration audit (Slope/Intercept).
- [ ] Real-time logging of `status` (PRODUCTION vs SHADOW vs FAILSAFE).
- [ ] Monthly re-validation via `historical_backtest_runner.py` with fresh data.

## 6. SLA/SLO
- **Availability:** 99.9% (Baseline fallback ensures prediction delivery).
- **Latency:** < 50ms per prediction adjustment.
- **Consistency:** 100% deterministic (reproducible via provenance).
