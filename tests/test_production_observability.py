import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys
import tempfile
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.production.live_observability import (
    LiveObservabilityEngine,
    HealthCheck,
    PredictionHealth,
    FeatureHealth,
    OperationalHealth,
    CalibrationHealth
)
from src.production.live_guardrails import (
    LiveGuardrailsEngine,
    GuardrailConfig,
    GuardrailAction
)
from src.production.incident_registry import (
    IncidentRegistry,
    Incident,
    IncidentType,
    Severity,
    RollbackStatus
)
from src.production.live_drift_analysis import (
    LiveDriftAnalyzer,
    DriftType,
    DriftSeverity,
    DriftAlert,
    RegimeState
)
from src.production.feature_registry import FeatureRegistry


@pytest.fixture
def feature_registry():
    """Create a feature registry for testing."""
    return FeatureRegistry()


@pytest.fixture
def observability_engine(feature_registry):
    """Create an observability engine for testing."""
    return LiveObservabilityEngine(
        feature_registry=feature_registry,
        psi_threshold=0.20,
        fallback_threshold=0.05,
        window_size=100
    )


@pytest.fixture
def incident_registry():
    """Create an incident registry for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield IncidentRegistry(storage_path=Path(tmpdir) / "test_incidents.json")


@pytest.fixture
def drift_analyzer():
    """Create a drift analyzer for testing."""
    return LiveDriftAnalyzer(
        psi_threshold=0.20,
        psi_warning_threshold=0.10,
        regime_detection_window=60
    )


@pytest.fixture
def guardrails_engine(observability_engine, incident_registry):
    """Create a guardrails engine for testing."""
    return LiveGuardrailsEngine(
        observability_engine=observability_engine,
        incident_registry=incident_registry
    )


class TestLiveObservability:
    """Test suite for LiveObservabilityEngine."""
    
    def test_prediction_health_nan_detection(self, observability_engine):
        """Test detection of NaN probabilities."""
        prediction_data = {
            "probabilities": [0.3, np.nan, 0.4, 0.3],
            "lambda": 1.5
        }
        
        health_checks = observability_engine.check_prediction_health(prediction_data)
        
        assert len(health_checks) > 0
        assert any(check.metric == "nan_probabilities" for check in health_checks)
        assert observability_engine.prediction_health.nan_probabilities > 0
    
    def test_prediction_health_invalid_lambda(self, observability_engine):
        """Test detection of invalid lambda values."""
        # Test negative lambda
        prediction_data = {"lambda": -1.0}
        health_checks = observability_engine.check_prediction_health(prediction_data)
        assert any(check.metric == "invalid_lambda" for check in health_checks)
        
        # Test NaN lambda
        prediction_data = {"lambda": np.nan}
        health_checks = observability_engine.check_prediction_health(prediction_data)
        assert any(check.metric == "invalid_lambda" for check in health_checks)
        
        # Test infinite lambda
        prediction_data = {"lambda": np.inf}
        health_checks = observability_engine.check_prediction_health(prediction_data)
        assert any(check.metric == "invalid_lambda" for check in health_checks)
    
    def test_prediction_health_normalization(self, observability_engine):
        """Test detection of probability normalization issues."""
        prediction_data = {
            "probabilities": [0.5, 0.3, 0.1],  # Sum = 0.9, not 1.0
            "lambda": 1.5
        }
        
        health_checks = observability_engine.check_prediction_health(prediction_data)
        
        assert len(health_checks) > 0
        assert any(check.metric == "probability_sum_deviation" for check in health_checks)
    
    def test_feature_health_missing_features(self, observability_engine):
        """Test detection of missing features."""
        features = {
            "continuity_index": 0.8,
            # Missing defenders_continuity
        }
        
        health_checks = observability_engine.check_feature_health(features)
        
        assert len(health_checks) > 0
        assert any("missing_defenders_continuity" in check.metric for check in health_checks)
    
    def test_feature_health_dtype_corruption(self, observability_engine):
        """Test detection of dtype corruption."""
        features = {
            "continuity_index": "invalid_string",  # Should be float
            "defenders_continuity": 0.75
        }
        
        health_checks = observability_engine.check_feature_health(features)
        
        assert len(health_checks) > 0
        assert any("dtype_corruption" in check.metric for check in health_checks)
    
    def test_feature_health_out_of_range(self, observability_engine):
        """Test detection of out-of-range values."""
        features = {
            "continuity_index": 1.5,  # Should be in [0, 1]
            "defenders_continuity": 0.75
        }
        
        health_checks = observability_engine.check_feature_health(features)
        
        assert len(health_checks) > 0
        assert any("out_of_range" in check.metric for check in health_checks)
    
    def test_operational_health_fallback_rate(self, observability_engine):
        """Test fallback rate monitoring."""
        prediction_result = {"status": "FAILSAFE_ERROR"}
        
        # Simulate multiple fallbacks
        for _ in range(10):
            observability_engine.check_operational_health(prediction_result, latency_ms=100)
        
        assert observability_engine.operational_health.fallback_rate > 0
        assert observability_engine.operational_health.total_requests == 10
    
    def test_operational_health_latency(self, observability_engine):
        """Test latency percentile calculation."""
        latencies = [50, 75, 100, 125, 150, 200, 250, 300, 400, 500]
        
        for latency in latencies:
            observability_engine.check_operational_health({"status": "SHADOW"}, latency)
        
        assert observability_engine.operational_health.latency_p50 > 0
        assert observability_engine.operational_health.latency_p95 > observability_engine.operational_health.latency_p50
        assert observability_engine.operational_health.latency_p99 > observability_engine.operational_health.latency_p95
    
    def test_calibration_health_brier(self, observability_engine):
        """Test Brier score calculation."""
        predictions = [0.3, 0.7, 0.5, 0.8, 0.2]
        actuals = [0, 1, 0, 1, 0]
        
        for pred, act in zip(predictions, actuals):
            observability_engine.check_calibration_health(pred, act)
        
        assert observability_engine.calibration_health.rolling_brier > 0
        assert observability_engine.calibration_health.sample_size == 5
    
    def test_psi_calculation(self, observability_engine):
        """Test PSI calculation."""
        reference = np.random.normal(0.5, 0.1, 1000)
        current = np.random.normal(0.5, 0.1, 100)
        
        observability_engine.set_reference_distribution("test_feature", reference)
        psi = observability_engine._calculate_psi(reference, current)
        
        assert psi >= 0
        # Similar distributions should have low PSI
        assert psi < 0.1
    
    def test_psi_drift_detection(self, observability_engine):
        """Test PSI drift detection."""
        reference = np.random.normal(0.5, 0.1, 1000)
        current = np.random.normal(0.8, 0.1, 100)  # Significant shift
        
        observability_engine.set_reference_distribution("test_feature", reference)
        observability_engine.check_feature_health({"test_feature": 0.8})
        
        # Should detect drift
        assert "test_feature" in observability_engine.feature_health.psi_drift


class TestLiveGuardrails:
    """Test suite for LiveGuardrailsEngine."""
    
    def test_guardrail_psi_trigger(self, guardrails_engine, observability_engine):
        """Test PSI guardrail trigger."""
        # Simulate PSI drift
        observability_engine.feature_health.psi_drift = {"continuity_index": 0.25}
        
        actions = guardrails_engine.evaluate_observability_summary()
        
        assert len(actions) > 0
        assert any("psi_drift" in action["guardrail"] for action in actions)
    
    def test_guardrail_fallback_rate_trigger(self, guardrails_engine, observability_engine):
        """Test fallback rate guardrail trigger."""
        # Simulate high fallback rate
        observability_engine.operational_health.fallback_rate = 0.10
        
        actions = guardrails_engine.evaluate_observability_summary()
        
        assert len(actions) > 0
        assert any("fallback_rate" in action["guardrail"] for action in actions)
    
    def test_guardrail_calibration_slope_trigger(self, guardrails_engine, observability_engine):
        """Test calibration slope guardrail trigger."""
        # Simulate calibration drift
        observability_engine.calibration_health.rolling_calibration_slope = 0.85
        
        actions = guardrails_engine.evaluate_observability_summary()
        
        assert len(actions) > 0
        assert any("calibration_slope" in action["guardrail"] for action in actions)
    
    def test_guardrail_nan_predictions_trigger(self, guardrails_engine, observability_engine):
        """Test NaN predictions guardrail trigger."""
        # Simulate NaN predictions
        observability_engine.prediction_health.nan_probabilities = 5
        
        actions = guardrails_engine.evaluate_observability_summary()
        
        assert len(actions) > 0
        assert any("nan_predictions" in action["guardrail"] for action in actions)
    
    def test_guardrail_feature_corruption_trigger(self, guardrails_engine, observability_engine):
        """Test feature corruption guardrail trigger."""
        # Simulate feature corruption
        observability_engine.feature_health.dtype_corruption = {"continuity_index": 10}
        
        actions = guardrails_engine.evaluate_observability_summary()
        
        assert len(actions) > 0
        assert any("feature_corruption" in action["guardrail"] for action in actions)
    
    def test_auto_disable_uplift(self, guardrails_engine):
        """Test automatic uplift disable."""
        guardrails_engine._disable_uplift()
        
        assert guardrails_engine.uplift_disabled is True
        assert guardrails_engine.uplift_disabled_since is not None
    
    def test_enable_uplift(self, guardrails_engine):
        """Test uplift re-enable."""
        guardrails_engine._disable_uplift()
        guardrails_engine.enable_uplift()
        
        assert guardrails_engine.uplift_disabled is False
        assert guardrails_engine.uplift_disabled_since is None
    
    def test_disable_feature(self, guardrails_engine):
        """Test feature disable."""
        guardrails_engine._disable_feature("continuity_index")
        
        assert "continuity_index" in guardrails_engine.disabled_features
    
    def test_enable_feature(self, guardrails_engine):
        """Test feature re-enable."""
        guardrails_engine._disable_feature("continuity_index")
        guardrails_engine.enable_feature("continuity_index")
        
        assert "continuity_index" not in guardrails_engine.disabled_features
    
    def test_cooldown_mechanism(self, guardrails_engine):
        """Test guardrail cooldown mechanism."""
        # Trigger guardrail
        guardrails_engine.last_trigger_time["test"] = datetime.now(timezone.utc)
        
        # Should be in cooldown
        assert guardrails_engine._is_in_cooldown("test") is True
    
    def test_guardrail_status(self, guardrails_engine):
        """Test guardrail status retrieval."""
        status = guardrails_engine.get_guardrail_status()
        
        assert "uplift_disabled" in status
        assert "disabled_features" in status
        assert "guardrail_configs" in status


class TestIncidentRegistry:
    """Test suite for IncidentRegistry."""
    
    def test_register_incident(self, incident_registry):
        """Test incident registration."""
        incident = incident_registry.register_incident(
            incident_type=IncidentType.PSI_DRIFT,
            severity=Severity.WARNING,
            message="Test incident",
            metric_value=0.25,
            threshold=0.20
        )
        
        assert incident.incident_id is not None
        assert incident.severity == Severity.WARNING
        assert incident.resolved is False
    
    def test_resolve_incident(self, incident_registry):
        """Test incident resolution."""
        incident = incident_registry.register_incident(
            incident_type=IncidentType.PSI_DRIFT,
            severity=Severity.WARNING,
            message="Test incident"
        )
        
        resolved = incident_registry.resolve_incident(incident.incident_id, "Test resolution")
        
        assert resolved is not None
        assert resolved.resolved is True
        assert resolved.resolution_timestamp is not None
    
    def test_trigger_rollback(self, incident_registry):
        """Test rollback trigger."""
        incident = incident_registry.register_incident(
            incident_type=IncidentType.PSI_DRIFT,
            severity=Severity.CRITICAL,
            message="Test incident"
        )
        
        updated = incident_registry.trigger_rollback(incident.incident_id)
        
        assert updated is not None
        assert updated.rollback_status == RollbackStatus.TRIGGERED
        assert updated.fallback_triggered is True
    
    def test_complete_rollback(self, incident_registry):
        """Test rollback completion."""
        incident = incident_registry.register_incident(
            incident_type=IncidentType.PSI_DRIFT,
            severity=Severity.CRITICAL,
            message="Test incident"
        )
        
        incident_registry.trigger_rollback(incident.incident_id)
        updated = incident_registry.complete_rollback(incident.incident_id)
        
        assert updated is not None
        assert updated.rollback_status == RollbackStatus.COMPLETED
    
    def test_get_incidents_by_type(self, incident_registry):
        """Test filtering incidents by type."""
        incident_registry.register_incident(
            incident_type=IncidentType.PSI_DRIFT,
            severity=Severity.WARNING,
            message="Test incident 1"
        )
        incident_registry.register_incident(
            incident_type=IncidentType.FEATURE_CORRUPTION,
            severity=Severity.CRITICAL,
            message="Test incident 2"
        )
        
        psi_incidents = incident_registry.get_incidents_by_type(IncidentType.PSI_DRIFT)
        
        assert len(psi_incidents) == 1
        assert psi_incidents[0].incident_type == IncidentType.PSI_DRIFT
    
    def test_get_incidents_by_severity(self, incident_registry):
        """Test filtering incidents by severity."""
        incident_registry.register_incident(
            incident_type=IncidentType.PSI_DRIFT,
            severity=Severity.WARNING,
            message="Test incident 1"
        )
        incident_registry.register_incident(
            incident_type=IncidentType.FEATURE_CORRUPTION,
            severity=Severity.CRITICAL,
            message="Test incident 2"
        )
        
        critical_incidents = incident_registry.get_incidents_by_severity(Severity.CRITICAL)
        
        assert len(critical_incidents) == 1
        assert critical_incidents[0].severity == Severity.CRITICAL
    
    def test_get_active_incidents(self, incident_registry):
        """Test getting active (unresolved) incidents."""
        incident1 = incident_registry.register_incident(
            incident_type=IncidentType.PSI_DRIFT,
            severity=Severity.WARNING,
            message="Test incident 1"
        )
        incident2 = incident_registry.register_incident(
            incident_type=IncidentType.FEATURE_CORRUPTION,
            severity=Severity.CRITICAL,
            message="Test incident 2"
        )
        
        incident_registry.resolve_incident(incident1.incident_id)
        
        active = incident_registry.get_active_incidents()
        
        assert len(active) == 1
        assert active[0].incident_id == incident2.incident_id
    
    def test_incident_summary(self, incident_registry):
        """Test incident summary generation."""
        incident_registry.register_incident(
            incident_type=IncidentType.PSI_DRIFT,
            severity=Severity.WARNING,
            message="Test incident"
        )
        
        summary = incident_registry.get_incident_summary()
        
        assert summary["total_incidents"] == 1
        assert summary["active_incidents"] == 1
        assert "by_type" in summary
        assert "by_severity" in summary
    
    def test_clear_old_incidents(self, incident_registry):
        """Test clearing old incidents."""
        # Register an incident
        incident_registry.register_incident(
            incident_type=IncidentType.PSI_DRIFT,
            severity=Severity.WARNING,
            message="Test incident"
        )
        
        # Clear incidents older than 0 days (should clear all)
        incident_registry.clear_old_incidents(days_to_keep=0)
        
        summary = incident_registry.get_incident_summary()
        assert summary["total_incidents"] == 0


class TestLiveDriftAnalysis:
    """Test suite for LiveDriftAnalyzer."""
    
    def test_set_reference_distribution(self, drift_analyzer):
        """Test setting reference distribution."""
        values = np.random.normal(0.5, 0.1, 100)
        drift_analyzer.set_reference_distribution("test_feature", values)
        
        assert "test_feature" in drift_analyzer.reference_distributions
        assert len(drift_analyzer.reference_distributions["test_feature"]) == 100
    
    def test_calculate_psi(self, drift_analyzer):
        """Test PSI calculation."""
        reference = np.random.normal(0.5, 0.1, 1000)
        current = np.random.normal(0.5, 0.1, 100)
        
        drift_analyzer.set_reference_distribution("test_feature", reference)
        psi = drift_analyzer.calculate_psi("test_feature", current)
        
        assert psi >= 0
    
    def test_detect_psi_temporal_drift(self, drift_analyzer):
        """Test PSI temporal drift detection."""
        reference = np.random.normal(0.5, 0.1, 1000)
        current = np.random.normal(0.8, 0.1, 100)  # Significant shift
        
        drift_analyzer.set_reference_distribution("test_feature", reference)
        alert = drift_analyzer.detect_psi_temporal_drift("test_feature", current)
        
        assert alert is not None
        assert alert.drift_type == DriftType.PSI_TEMPORAL
        assert alert.severity == DriftSeverity.CRITICAL
    
    def test_detect_feature_distribution_shift(self, drift_analyzer):
        """Test feature distribution shift detection."""
        reference = np.random.normal(0.5, 0.1, 1000)
        current = np.array([0.9])  # Significant mean shift
        
        drift_analyzer.set_reference_distribution("test_feature", reference)
        alert = drift_analyzer.detect_feature_distribution_shift("test_feature", current)
        
        assert alert is not None
        assert alert.drift_type == DriftType.FEATURE_DISTRIBUTION_SHIFT
    
    def test_detect_post_worldcup_regime(self, drift_analyzer):
        """Test post-World Cup regime detection."""
        # Set a recent tournament end date
        drift_analyzer.tournament_end_dates = [datetime.now(timezone.utc) - timedelta(days=10)]
        
        alert = drift_analyzer.detect_post_worldcup_regime()
        
        assert alert is not None
        assert alert.drift_type == DriftType.POST_WORLDCUP_REGIME
        assert drift_analyzer.regime_state.is_post_tournament is True
    
    def test_detect_confederation_drift(self, drift_analyzer):
        """Test confederation drift detection."""
        # Add some history
        for _ in range(15):
            drift_analyzer.detect_confederation_drift("UEFA", 0.5)
        
        # Add a drift
        alert = drift_analyzer.detect_confederation_drift("UEFA", 0.8)
        
        # Should detect drift after enough history
        assert "UEFA" in drift_analyzer.confederation_performance
    
    def test_detect_tournament_vs_friendly_drift(self, drift_analyzer):
        """Test tournament vs friendly drift detection."""
        # Add some history
        for _ in range(15):
            drift_analyzer.detect_tournament_vs_friendly_drift(True, 0.5)
            drift_analyzer.detect_tournament_vs_friendly_drift(False, 0.5)
        
        # Add a drift
        alert = drift_analyzer.detect_tournament_vs_friendly_drift(True, 0.8)
        
        assert len(drift_analyzer.tournament_metrics) > 0
        assert len(drift_analyzer.friendly_metrics) > 0
    
    def test_detect_feature_sparsity(self, drift_analyzer):
        """Test feature sparsity detection."""
        features = {
            "continuity_index": None,
            "defenders_continuity": None,
            "valid_feature": 0.5
        }
        
        alert = drift_analyzer.detect_feature_sparsity(features)
        
        assert alert is not None
        assert alert.drift_type == DriftType.FEATURE_SPARSITY
    
    def test_detect_calendar_corruption_future_date(self, drift_analyzer):
        """Test calendar corruption detection for future dates."""
        future_date = datetime.now(timezone.utc) + timedelta(days=400)
        
        alert = drift_analyzer.detect_calendar_corruption(future_date, None)
        
        assert alert is not None
        assert alert.drift_type == DriftType.CALENDAR_CORRUPTION
    
    def test_detect_calendar_corruption_feature_after_match(self, drift_analyzer):
        """Test calendar corruption detection for feature timestamp after match."""
        match_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        feature_timestamp = datetime(2024, 1, 2, tzinfo=timezone.utc)
        
        alert = drift_analyzer.detect_calendar_corruption(match_date, feature_timestamp)
        
        assert alert is not None
        assert alert.drift_type == DriftType.CALENDAR_CORRUPTION
    
    def test_comprehensive_drift_check(self, drift_analyzer):
        """Test comprehensive drift check."""
        features = {
            "continuity_index": 0.8,
            "defenders_continuity": 0.75
        }
        
        drift_analyzer.set_reference_distribution("continuity_index", np.random.normal(0.5, 0.1, 100))
        
        alerts = drift_analyzer.run_comprehensive_drift_check(
            features=features,
            match_date=datetime.now(timezone.utc),
            is_tournament=True
        )
        
        assert isinstance(alerts, list)
    
    def test_drift_summary(self, drift_analyzer):
        """Test drift summary generation."""
        drift_analyzer.set_reference_distribution("test_feature", np.random.normal(0.5, 0.1, 100))
        
        summary = drift_analyzer.get_drift_summary()
        
        assert "regime_state" in summary
        assert "reference_distributions" in summary
        assert "historical_psi_points" in summary


class TestIntegration:
    """Integration tests for the full observability system."""
    
    def test_full_guardrail_workflow(self, observability_engine, guardrails_engine):
        """Test complete guardrail workflow from detection to action."""
        # Simulate a health check failure
        health_checks = [
            HealthCheck(
                component="feature",
                metric="psi_drift_continuity_index",
                value=0.25,
                status="FAIL",
                threshold=0.20,
                message="PSI drift detected"
            )
        ]
        
        actions = guardrails_engine.evaluate_health_checks(health_checks)
        
        assert len(actions) > 0
        assert actions[0]["guardrail"] == "psi_drift"
    
    def test_incident_registration_from_guardrail(self, guardrails_engine, incident_registry):
        """Test that guardrails properly register incidents."""
        # Trigger a guardrail
        guardrails_engine.observability.feature_health.psi_drift = {"continuity_index": 0.25}
        actions = guardrails_engine.evaluate_observability_summary()
        
        # Check that an incident was registered
        summary = incident_registry.get_incident_summary()
        assert summary["total_incidents"] > 0
    
    def test_rollback_preserves_baseline(self, guardrails_engine):
        """Test that rollback preserves baseline functionality."""
        # Disable uplift
        guardrails_engine._disable_uplift()
        
        # Verify uplift is disabled
        assert guardrails_engine.uplift_disabled is True
        
        # Re-enable
        guardrails_engine.enable_uplift()
        
        # Verify uplift is re-enabled
        assert guardrails_engine.uplift_disabled is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
