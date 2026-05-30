import pandas as pd
import numpy as np
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from src.production.feature_registry import FeatureRegistry, FeatureStatus
from src.experimental.poisson_squad_uplift import PoissonSquadUplift

logger = logging.getLogger("production_integration")

class SquadUpliftIntegration:
    """
    Production-grade integration layer for squad features.
    Supports Shadow Mode, Feature Flags, and Automatic Fallback.
    """
    
    def __init__(self, 
                 uplift_model: PoissonSquadUplift, 
                 registry: FeatureRegistry,
                 shadow_mode: bool = True):
        self.uplift_model = uplift_model
        self.registry = registry
        self.shadow_mode = shadow_mode
        self._feature_flag_enabled = True

    def toggle_squad_features(self, enabled: bool):
        self._feature_flag_enabled = enabled
        logger.info(f"Squad features toggle: {'ENABLED' if enabled else 'DISABLED'}")

    def get_adjusted_expectation(self, 
                                 match_data: Dict[str, Any], 
                                 base_lambda: float) -> Dict[str, float]:
        """
        Main entry point for live prediction.
        Returns both baseline and uplifted lambdas for shadow monitoring.
        """
        # 1. Zero Leakage Pre-Check
        kickoff_ts = match_data.get("kickoff_timestamp")
        feature_ts = match_data.get("feature_timestamp_utc")
        
        if feature_ts and feature_ts >= kickoff_ts:
            logger.error("CRITICAL: Temporal Leakage detected in production request! Falling back to baseline.")
            return {"baseline": base_lambda, "uplifted": base_lambda, "status": "FAILSAFE_LEAKAGE"}

        # 2. Check Feature Flags & Registry
        if not self._feature_flag_enabled:
            return {"baseline": base_lambda, "uplifted": base_lambda, "status": "BASELINE_ONLY"}

        # 3. Filter only approved features with type validation
        approved_features = self.registry.get_approved_features()
        input_features = {}
        
        for f in approved_features:
            value = match_data.get(f)
            if value is None:
                continue
            
            # Type validation: ensure numeric values
            try:
                if isinstance(value, (int, float)):
                    input_features[f] = float(value)
                elif isinstance(value, str):
                    # Attempt to convert string to float
                    input_features[f] = float(value)
                else:
                    logger.warning(f"Feature {f} has invalid type {type(value)}. Skipping.")
                    continue
            except (ValueError, TypeError) as e:
                logger.warning(f"Feature {f} cannot be converted to float: {value} ({type(value)}). Skipping.")
                continue
        
        if len(input_features) < len(approved_features):
            logger.warning("Missing or invalid approved features. Falling back to baseline to preserve calibration.")
            return {"baseline": base_lambda, "uplifted": base_lambda, "status": "FAILSAFE_MISSING_DATA"}

        # 4. Validate base_lambda type
        try:
            base_lambda = float(base_lambda)
        except (ValueError, TypeError) as e:
            logger.error(f"base_lambda cannot be converted to float: {base_lambda} ({type(base_lambda)}). Fallback to baseline.")
            return {"baseline": base_lambda, "uplifted": base_lambda, "status": "FAILSAFE_ERROR"}

        # 5. Compute Uplift
        try:
            # Convert to DataFrame for model compatibility
            df_input = pd.DataFrame([input_features])
            df_input["lambda_base"] = base_lambda
            
            # Predict using the parametric model from Phase E
            adj_lambda = self.uplift_model.predict_adjusted_lambda(df_input, "lambda_base")[0]
            
            # 5. Shadow Mode Check
            if self.shadow_mode:
                logger.info(f"SHADOW MODE: Base {base_lambda:.4f} -> Uplifted {adj_lambda:.4f} (Diff: {adj_lambda-base_lambda:.4f})")
                return {"baseline": base_lambda, "uplifted": base_lambda, "shadow_uplifted": adj_lambda, "status": "SHADOW"}
            
            return {"baseline": base_lambda, "uplifted": adj_lambda, "status": "PRODUCTION"}

        except Exception as e:
            logger.error(f"Uplift calculation failed: {e}. Safe fallback triggered.")
            return {"baseline": base_lambda, "uplifted": base_lambda, "status": "FAILSAFE_ERROR"}

if __name__ == "__main__":
    logger.info("SquadUpliftIntegration loaded.")
