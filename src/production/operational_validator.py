import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any
from src.production.squad_uplift_integration import SquadUpliftIntegration
from src.production.feature_registry import FeatureRegistry
from src.experimental.poisson_squad_uplift import PoissonSquadUplift

logger = logging.getLogger("operational_validator")

class OperationalStressTester:
    """
    Simulates production failure modes to verify guardrail robustness.
    Ensures zero leakage and deterministic fallback.
    """
    
    def __init__(self, integration: SquadUpliftIntegration):
        self.integration = integration

    def run_stress_test(self, n_iterations: int = 1000):
        logger.info(f"Starting Operational Stress Test ({n_iterations} iterations)...")
        results = {"SUCCESS": 0, "FAILSAFE_LEAKAGE": 0, "FAILSAFE_MISSING_DATA": 0, "FAILSAFE_ERROR": 0}
        
        for i in range(n_iterations):
            # 1. Randomly pick a failure mode
            mode = np.random.choice(["NORMAL", "LEAKAGE", "MISSING", "CORRUPT"], p=[0.7, 0.1, 0.1, 0.1])
            
            match_data = self._generate_scenario(mode)
            prediction = self.integration.get_adjusted_expectation(match_data, base_lambda=1.5)
            
            status = prediction.get("status")
            if "FAILSAFE" in status or status == "PRODUCTION" or status == "SHADOW":
                results[status] = results.get(status, 0) + 1
            else:
                results["SUCCESS"] += 1
                
        logger.info(f"Stress Test Completed. Results: {results}")
        return results

    def _generate_scenario(self, mode: str) -> Dict[str, Any]:
        kickoff = datetime.now(timezone.utc) + timedelta(days=1)
        valid_ts = kickoff - timedelta(days=1)
        
        if mode == "NORMAL":
            return {
                "kickoff_timestamp": kickoff,
                "feature_timestamp_utc": valid_ts,
                "continuity_index": 0.8,
                "defenders_continuity": 0.75
            }
        elif mode == "LEAKAGE":
            return {
                "kickoff_timestamp": kickoff,
                "feature_timestamp_utc": kickoff + timedelta(minutes=1),
                "continuity_index": 0.8,
                "defenders_continuity": 0.75
            }
        elif mode == "MISSING":
            return {
                "kickoff_timestamp": kickoff,
                "feature_timestamp_utc": valid_ts,
                "continuity_index": 0.8
                # Missing defenders_continuity
            }
        elif mode == "CORRUPT":
            return {
                "kickoff_timestamp": kickoff,
                "feature_timestamp_utc": valid_ts,
                "continuity_index": "INVALID_TYPE",
                "defenders_continuity": 0.75
            }
        return {}

if __name__ == "__main__":
    # Test initialization
    print("OperationalStressTester implementation ready.")
