import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
from src.production.squad_uplift_integration import SquadUpliftIntegration
from src.production.feature_registry import FeatureRegistry
from src.experimental.poisson_squad_uplift import PoissonSquadUplift

def test_production_fallback_on_leakage():
    """
    Critical Production Test: Ensure the system falls back to baseline 
    if a request contains future information.
    """
    registry = FeatureRegistry()
    # Mock model that doesn't need fitting for this test
    uplift_model = PoissonSquadUplift(feature_columns=["continuity_index"])
    integration = SquadUpliftIntegration(uplift_model, registry, shadow_mode=False)
    
    kickoff = datetime(2026, 6, 1, 20, 0, tzinfo=timezone.utc)
    # LEAKAGE: Feature timestamp is AFTER kickoff
    future_feature_ts = kickoff + timedelta(hours=1)
    
    match_data = {
        "kickoff_timestamp": kickoff,
        "feature_timestamp_utc": future_feature_ts,
        "continuity_index": 0.8
    }
    
    result = integration.get_adjusted_expectation(match_data, base_lambda=1.5)
    
    assert result["status"] == "FAILSAFE_LEAKAGE"
    assert result["uplifted"] == 1.5

def test_production_fallback_on_missing_approved_features():
    """
    Ensure the system defaults to baseline if an approved feature is missing 
    from the production request.
    """
    registry = FeatureRegistry()
    uplift_model = PoissonSquadUplift(feature_columns=["continuity_index", "defenders_continuity"])
    integration = SquadUpliftIntegration(uplift_model, registry, shadow_mode=False)
    
    kickoff = datetime(2026, 6, 1, 20, 0, tzinfo=timezone.utc)
    valid_ts = kickoff - timedelta(days=2)
    
    # Missing 'defenders_continuity' which is APPROVED in registry
    incomplete_match_data = {
        "kickoff_timestamp": kickoff,
        "feature_timestamp_utc": valid_ts,
        "continuity_index": 0.8
    }
    
    result = integration.get_adjusted_expectation(incomplete_match_data, base_lambda=1.5)
    
    assert result["status"] == "FAILSAFE_MISSING_DATA"
    assert result["uplifted"] == 1.5

def test_shadow_mode_transparency():
    """
    Verify that in Shadow Mode, the 'uplifted' value returned matches the 'baseline',
    preserving the production environment while logging the 'shadow_uplifted'.
    """
    registry = FeatureRegistry()
    # Mocking betas for a simple calculation
    uplift_model = PoissonSquadUplift(feature_columns=["continuity_index", "defenders_continuity"])
    uplift_model.betas = np.array([0.0, 0.1, 0.05]) # const, continuity, def_cont
    
    integration = SquadUpliftIntegration(uplift_model, registry, shadow_mode=True)
    
    kickoff = datetime(2026, 6, 1, 20, 0, tzinfo=timezone.utc)
    valid_ts = kickoff - timedelta(days=1)
    
    match_data = {
        "kickoff_timestamp": kickoff,
        "feature_timestamp_utc": valid_ts,
        "continuity_index": 0.8,
        "defenders_continuity": 0.7
    }
    
    result = integration.get_adjusted_expectation(match_data, base_lambda=1.2)
    
    assert result["status"] == "SHADOW"
    assert result["uplifted"] == 1.2 # Must be baseline in shadow mode
    assert "shadow_uplifted" in result
    assert result["shadow_uplifted"] > 1.2
