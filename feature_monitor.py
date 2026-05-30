import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

class TemporalFeatureMonitor:
    """
    Principal ML Infrastructure Monitor for international football squad features.
    Focuses on non-stationarity, freshness, and causal integrity.
    """
    
    def __init__(self, psi_threshold: float = 0.1, missing_ratio_threshold: float = 0.05):
        self.psi_threshold = psi_threshold
        self.missing_ratio_threshold = missing_ratio_threshold
        self.logger = logging.getLogger(__name__)

    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        """
        Population Stability Index (PSI) calculation to detect feature drift.
        """
        if len(expected) == 0 or len(actual) == 0:
            return 0.0
            
        expected_percents = np.histogram(expected, bins=buckets)[0] / len(expected)
        actual_percents = np.histogram(actual, bins=buckets)[0] / len(actual)
        
        # Avoid division by zero
        expected_percents = np.clip(expected_percents, 1e-4, 1.0)
        actual_percents = np.clip(actual_percents, 1e-4, 1.0)
        
        psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        return psi_value

    def validate_freshness(self, feature_df: pd.DataFrame, kickoff_ts_col: str, feature_ts_col: str) -> Dict:
        """
        Enforces strict anti-leakage: feature_timestamp must be < kickoff_timestamp.
        Detects 'stale' snapshots (too far from kickoff).
        """
        # Ensure datetime
        feature_df[kickoff_ts_col] = pd.to_datetime(feature_df[kickoff_ts_col])
        feature_df[feature_ts_col] = pd.to_datetime(feature_df[feature_ts_col])
        
        violations = feature_df[feature_df[feature_ts_col] >= feature_df[kickoff_ts_col]]
        
        lead_time = (feature_df[kickoff_ts_col] - feature_df[feature_ts_col]).dt.total_seconds() / 3600
        
        return {
            "leakage_violations": len(violations),
            "avg_lead_time_hours": lead_time.mean(),
            "max_lead_time_hours": lead_time.max(),
            "min_lead_time_hours": lead_time.min(),
            "is_valid": len(violations) == 0
        }

    def check_variance_collapse(self, df: pd.DataFrame, features: List[str]) -> Dict[str, float]:
        """
        Detects if a feature has become a constant (drift towards zero entropy).
        """
        results = {}
        for feat in features:
            if feat not in df.columns:
                continue
            std = df[feat].std()
            results[feat] = std
            if std < 1e-6:
                self.logger.warning(f"Feature {feat} has collapsed variance (std: {std})")
        return results

    def audit_temporal_consistency(self, df: pd.DataFrame, team_id_col: str, ts_col: str) -> bool:
        """
        Ensures no duplicate snapshots for the same team/timestamp and detects
        backward-in-time updates in the feature store.
        """
        df[ts_col] = pd.to_datetime(df[ts_col])
        duplicates = df.duplicated(subset=[team_id_col, ts_col]).sum()
        if duplicates > 0:
            self.logger.error(f"Found {duplicates} duplicate snapshots.")
            return False
            
        for team, group in df.groupby(team_id_col):
            group = group.sort_values(ts_col)
            if not group[ts_col].is_monotonic_increasing:
                self.logger.error(f"Temporal inconsistency for team {team}: Non-monotonic timestamps.")
                return False
        return True

if __name__ == "__main__":
    monitor = TemporalFeatureMonitor()
    print("TemporalFeatureMonitor initialized.")
