# Operational Runbook: Squad Temporal Features Go-Live

## 1. Deployment Checklist
- [ ] **Infrastructure Check:** SQLite/DuckDB storage partitions are current.
- [ ] **Ingestion Health:** `BatchIngestionPipeline` has processed snapshots for the current FIFA window.
- [ ] **Shadow Audit:** `shadow_analyzer.py` confirms Shadow ROI >= Baseline ROI.
- [ ] **Operational Stress Test:** `operational_validator.py` results show 0% leakage violations and 0% unhandled exceptions.
- [ ] **Certification:** `final_squad_feature_decision.md` is signed off.

## 2. Go-Live Procedure
To switch from Shadow to Live:
1. Run `python -m src.production.live_rollout`.
2. Monitor `production_integration.log` for status labels changing from `SHADOW` to `PRODUCTION`.
3. Verify `SLA/Latency` remains < 50ms per prediction.

## 3. Rollback Procedure (Instant Disabling)
If calibration drift or systemic failures are detected:
1. **Command:** `python -m src.production.emergency_rollback`.
   *   Alternatively, set `feature_flag_enabled = False` in `SquadUpliftIntegration`.
2. **Result:** The system will immediately ignore squad features and return to the Baseline model.
3. **Verification:** Confirm all prediction responses contain `status: BASELINE_ONLY`.

## 4. Incident Response Guide
### Scenario A: Temporal Leakage Alert
- **Indication:** `FAILSAFE_LEAKAGE` logs detected.
- **Action:**
  1. Trigger Rollback.
  2. Audit `ingestion_timestamp_utc` in SQLite.
  3. Purge corrupted feature partitions from Parquet store.

### Scenario B: High PSI Drift (> 0.20)
- **Indication:** `GUARDRAIL VIOLATION: PSI` in `production_monitor.log`.
- **Action:**
  1. Re-run `backtest_runner.py` with the last 30 days of data.
  2. If Brier Delta < 0, disable squad features.
  3. Re-fit Poisson coefficients ($\beta$) via `feature_batch_processor.py`.

### Scenario C: Post-Tournament Regime Change
- **Indication:** 60-day window following Mundial 2026.
- **Action:**
  1. Set `midfielders_continuity` to `EXPERIMENTAL`.
  2. Enforce higher shrinkage on Jaccard coefficients to prevent over-adjustment during generational turnover.
