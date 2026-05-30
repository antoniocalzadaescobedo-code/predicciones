import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from enum import Enum

from src.production.live_observability import (
    LiveObservabilityEngine, 
    HealthCheck, 
    Severity
)
from src.production.incident_registry import (
    IncidentRegistry, 
    IncidentType, 
    RollbackStatus
)

logger = logging.getLogger("live_guardrails")

class GuardrailAction(Enum):
    """Actions that can be taken when a guardrail is triggered."""
    DISABLE_FEATURE = "DISABLE_FEATURE"
    DISABLE_UPLIFT = "DISABLE_UPLIFT"
    FALLBACK_BASELINE = "FALLBACK_BASELINE"
    LOG_WARNING = "LOG_WARNING"
    REGISTER_INCIDENT = "REGISTER_INCIDENT"

@dataclass
class GuardrailConfig:
    """Configuration for a single guardrail."""
    name: str
    enabled: bool = True
    threshold: float = 0.0
    action: GuardrailAction = GuardrailAction.LOG_WARNING
    severity: Severity = Severity.WARNING
    cooldown_seconds: int = 300  # 5 minutes default cooldown
    auto_rollback: bool = False

class LiveGuardrailsEngine:
    """
    Production-grade guardrails system with auto-disable capabilities.
    Monitors health checks and triggers automatic actions when thresholds are exceeded.
    
    PRINCIPLE: system stability > prediction uplift
    """
    
    def __init__(self, 
                 observability_engine: LiveObservabilityEngine,
                 incident_registry: IncidentRegistry,
                 uplift_integration: Optional[Any] = None):
        self.observability = observability_engine
        self.incident_registry = incident_registry
        self.uplift_integration = uplift_integration
        
        # Guardrail configurations
        self.guardrail_configs: Dict[str, GuardrailConfig] = {
            "psi_drift": GuardrailConfig(
                name="PSI Drift",
                enabled=True,
                threshold=0.20,
                action=GuardrailAction.DISABLE_FEATURE,
                severity=Severity.CRITICAL,
                cooldown_seconds=600,
                auto_rollback=True
            ),
            "fallback_rate": GuardrailConfig(
                name="High Fallback Rate",
                enabled=True,
                threshold=0.05,
                action=GuardrailAction.DISABLE_UPLIFT,
                severity=Severity.CRITICAL,
                cooldown_seconds=300,
                auto_rollback=True
            ),
            "calibration_slope": GuardrailConfig(
                name="Calibration Slope Drift",
                enabled=True,
                threshold=0.05,  # Deviation from 1.0
                action=GuardrailAction.DISABLE_UPLIFT,
                severity=Severity.CRITICAL,
                cooldown_seconds=600,
                auto_rollback=True
            ),
            "brier_degradation": GuardrailConfig(
                name="Brier Score Degradation",
                enabled=True,
                threshold=0.01,
                action=GuardrailAction.DISABLE_UPLIFT,
                severity=Severity.CRITICAL,
                cooldown_seconds=600,
                auto_rollback=True
            ),
            "nan_predictions": GuardrailConfig(
                name="NaN Predictions",
                enabled=True,
                threshold=0.0,  # Any NaN is unacceptable
                action=GuardrailAction.DISABLE_UPLIFT,
                severity=Severity.EMERGENCY,
                cooldown_seconds=60,
                auto_rollback=True
            ),
            "feature_corruption": GuardrailConfig(
                name="Feature Corruption",
                enabled=True,
                threshold=0.0,  # Any corruption is unacceptable
                action=GuardrailAction.DISABLE_FEATURE,
                severity=Severity.EMERGENCY,
                cooldown_seconds=120,
                auto_rollback=True
            ),
            "temporal_leakage": GuardrailConfig(
                name="Temporal Leakage",
                enabled=True,
                threshold=0.0,
                action=GuardrailAction.REGISTER_INCIDENT,
                severity=Severity.CRITICAL,
                cooldown_seconds=0,  # Always log
                auto_rollback=False
            )
        }
        
        # Cooldown tracking
        self.last_trigger_time: Dict[str, datetime] = {}
        
        # Disabled features tracking
        self.disabled_features: Dict[str, datetime] = {}
        
        # Uplift disabled flag
        self.uplift_disabled = False
        self.uplift_disabled_since: Optional[datetime] = None
        
        logger.info("LiveGuardrailsEngine initialized with auto-disable capabilities.")

    def evaluate_health_checks(self, health_checks: List[HealthCheck]) -> List[Dict[str, Any]]:
        """
        Evaluates health checks and triggers guardrail actions if needed.
        
        Args:
            health_checks: List of health checks from observability engine
            
        Returns:
            List of actions taken
        """
        actions_taken = []
        
        for check in health_checks:
            if check.status != "FAIL":
                continue
            
            # Map health check to guardrail
            guardrail_key = self._map_health_check_to_guardrail(check)
            if guardrail_key is None:
                continue
            
            config = self.guardrail_configs.get(guardrail_key)
            if config is None or not config.enabled:
                continue
            
            # Check cooldown
            if self._is_in_cooldown(guardrail_key):
                logger.debug(f"Guardrail {guardrail_key} is in cooldown, skipping")
                continue
            
            # Trigger guardrail action
            action_result = self._trigger_guardrail(guardrail_key, config, check)
            actions_taken.append(action_result)
        
        return actions_taken

    def evaluate_observability_summary(self) -> List[Dict[str, Any]]:
        """
        Evaluates the full observability summary and triggers guardrails.
        
        Returns:
            List of actions taken
        """
        actions_taken = []
        summary = self.observability.get_health_summary()
        
        # Check PSI drift
        for feature, psi in summary["feature_health"]["psi_drift"].items():
            if psi > self.guardrail_configs["psi_drift"].threshold:
                check = HealthCheck(
                    component="feature",
                    metric=f"psi_drift_{feature}",
                    value=psi,
                    status="FAIL",
                    threshold=self.guardrail_configs["psi_drift"].threshold,
                    message=f"PSI drift in {feature}: {psi:.4f}"
                )
                action_result = self._trigger_guardrail("psi_drift", self.guardrail_configs["psi_drift"], check, affected_feature=feature)
                actions_taken.append(action_result)
        
        # Check fallback rate
        fallback_rate = summary["operational_health"]["fallback_rate"]
        if fallback_rate > self.guardrail_configs["fallback_rate"].threshold:
            check = HealthCheck(
                component="operational",
                metric="fallback_rate",
                value=fallback_rate,
                status="FAIL",
                threshold=self.guardrail_configs["fallback_rate"].threshold,
                message=f"Fallback rate: {fallback_rate:.2%}"
            )
            action_result = self._trigger_guardrail("fallback_rate", self.guardrail_configs["fallback_rate"], check)
            actions_taken.append(action_result)
        
        # Check calibration slope
        calib_slope = summary["calibration_health"]["rolling_calibration_slope"]
        slope_range = self.observability.calibration_slope_range
        if not (slope_range[0] <= calib_slope <= slope_range[1]):
            deviation = max(abs(calib_slope - slope_range[0]), abs(calib_slope - slope_range[1]))
            check = HealthCheck(
                component="calibration",
                metric="calibration_slope",
                value=calib_slope,
                status="FAIL",
                threshold=slope_range,
                message=f"Calibration slope outside range: {calib_slope:.4f}"
            )
            action_result = self._trigger_guardrail("calibration_slope", self.guardrail_configs["calibration_slope"], check)
            actions_taken.append(action_result)
        
        # Check for NaN predictions
        nan_count = summary["prediction_health"]["nan_probabilities"]
        if nan_count > 0:
            check = HealthCheck(
                component="prediction",
                metric="nan_probabilities",
                value=nan_count,
                status="FAIL",
                threshold=0,
                message=f"NaN predictions detected: {nan_count}"
            )
            action_result = self._trigger_guardrail("nan_predictions", self.guardrail_configs["nan_predictions"], check)
            actions_taken.append(action_result)
        
        # Check for feature corruption
        for feature, count in summary["feature_health"]["dtype_corruption"].items():
            if count > 0:
                check = HealthCheck(
                    component="feature",
                    metric=f"dtype_corruption_{feature}",
                    value=count,
                    status="FAIL",
                    threshold=0,
                    message=f"Feature corruption in {feature}: {count} instances"
                )
                action_result = self._trigger_guardrail("feature_corruption", self.guardrail_configs["feature_corruption"], check, affected_feature=feature)
                actions_taken.append(action_result)
        
        return actions_taken

    def _map_health_check_to_guardrail(self, check: HealthCheck) -> Optional[str]:
        """Maps a health check to a guardrail key."""
        metric = check.metric.lower()
        
        if "psi" in metric and "drift" in metric:
            return "psi_drift"
        elif "fallback" in metric:
            return "fallback_rate"
        elif "calibration" in metric and "slope" in metric:
            return "calibration_slope"
        elif "brier" in metric and "degradation" in metric:
            return "brier_degradation"
        elif "nan" in metric:
            return "nan_predictions"
        elif "dtype_corruption" in metric or "corruption" in metric:
            return "feature_corruption"
        elif "leakage" in metric:
            return "temporal_leakage"
        
        return None

    def _is_in_cooldown(self, guardrail_key: str) -> bool:
        """Checks if a guardrail is in cooldown period."""
        if guardrail_key not in self.last_trigger_time:
            return False
        
        config = self.guardrail_configs[guardrail_key]
        elapsed = (datetime.now(timezone.utc) - self.last_trigger_time[guardrail_key]).total_seconds()
        
        return elapsed < config.cooldown_seconds

    def _trigger_guardrail(self, 
                         guardrail_key: str, 
                         config: GuardrailConfig,
                         check: HealthCheck,
                         affected_feature: Optional[str] = None) -> Dict[str, Any]:
        """
        Triggers the appropriate action for a guardrail.
        
        Args:
            guardrail_key: Key of the guardrail
            config: Guardrail configuration
            check: Health check that triggered the guardrail
            affected_feature: Feature that caused the issue (if applicable)
            
        Returns:
            Dictionary describing the action taken
        """
        self.last_trigger_time[guardrail_key] = datetime.now(timezone.utc)
        
        # Register incident
        incident_type = self._map_guardrail_to_incident_type(guardrail_key)
        incident = self.incident_registry.register_incident(
            incident_type=incident_type,
            severity=config.severity,
            message=check.message,
            metric_value=check.value,
            threshold=check.threshold if isinstance(check.threshold, (int, float)) else 0.0,
            affected_feature=affected_feature,
            fallback_triggered=config.auto_rollback
        )
        
        action_taken = {
            "guardrail": guardrail_key,
            "action": config.action.value,
            "incident_id": incident.incident_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": check.message
        }
        
        # Execute action
        if config.action == GuardrailAction.DISABLE_FEATURE and affected_feature:
            self._disable_feature(affected_feature)
            action_taken["disabled_feature"] = affected_feature
            
            if config.auto_rollback:
                self.incident_registry.trigger_rollback(incident.incident_id)
                action_taken["rollback_triggered"] = True
        
        elif config.action == GuardrailAction.DISABLE_UPLIFT:
            self._disable_uplift()
            action_taken["uplift_disabled"] = True
            
            if config.auto_rollback:
                self.incident_registry.trigger_rollback(incident.incident_id)
                action_taken["rollback_triggered"] = True
        
        elif config.action == GuardrailAction.FALLBACK_BASELINE:
            if self.uplift_integration:
                self.uplift_integration.toggle_squad_features(False)
            action_taken["fallback_to_baseline"] = True
            
            if config.auto_rollback:
                self.incident_registry.trigger_rollback(incident.incident_id)
                action_taken["rollback_triggered"] = True
        
        logger.critical(
            f"GUARDRAIL TRIGGERED: {config.name} - {config.action.value} - "
            f"Incident: {incident.incident_id}"
        )
        
        return action_taken

    def _disable_feature(self, feature_name: str):
        """Disables a specific feature."""
        self.disabled_features[feature_name] = datetime.now(timezone.utc)
        logger.critical(f"Feature {feature_name} disabled due to guardrail violation")

    def _disable_uplift(self):
        """Disables the entire uplift system."""
        self.uplift_disabled = True
        self.uplift_disabled_since = datetime.now(timezone.utc)
        
        if self.uplift_integration:
            self.uplift_integration.toggle_squad_features(False)
        
        logger.critical("UPLIFT SYSTEM DISABLED due to guardrail violation")

    def enable_uplift(self):
        """Re-enables the uplift system."""
        self.uplift_disabled = False
        self.uplift_disabled_since = None
        
        if self.uplift_integration:
            self.uplift_integration.toggle_squad_features(True)
        
        logger.info("UPLIFT SYSTEM RE-ENABLED")

    def enable_feature(self, feature_name: str):
        """Re-enables a specific feature."""
        if feature_name in self.disabled_features:
            del self.disabled_features[feature_name]
            logger.info(f"Feature {feature_name} re-enabled")

    def _map_guardrail_to_incident_type(self, guardrail_key: str) -> IncidentType:
        """Maps guardrail key to incident type."""
        mapping = {
            "psi_drift": IncidentType.PSI_DRIFT,
            "fallback_rate": IncidentType.HIGH_FALLBACK_RATE,
            "calibration_slope": IncidentType.CALIBRATION_DRIFT,
            "brier_degradation": IncidentType.BRIER_DEGRADATION,
            "nan_predictions": IncidentType.NAN_PREDICTIONS,
            "feature_corruption": IncidentType.FEATURE_CORRUPTION,
            "temporal_leakage": IncidentType.TEMPORAL_LEAKAGE
        }
        return mapping.get(guardrail_key, IncidentType.FEATURE_CORRUPTION)

    def get_guardrail_status(self) -> Dict[str, Any]:
        """Returns current status of all guardrails."""
        return {
            "uplift_disabled": self.uplift_disabled,
            "uplift_disabled_since": self.uplift_disabled_since.isoformat() if self.uplift_disabled_since else None,
            "disabled_features": self.disabled_features,
            "guardrail_configs": {
                key: {
                    "enabled": config.enabled,
                    "threshold": config.threshold,
                    "action": config.action.value,
                    "in_cooldown": self._is_in_cooldown(key)
                }
                for key, config in self.guardrail_configs.items()
            }
        }

    def reset_guardrails(self):
        """Resets all guardrails to initial state."""
        self.disabled_features.clear()
        self.uplift_disabled = False
        self.uplift_disabled_since = None
        self.last_trigger_time.clear()
        logger.info("All guardrails reset to initial state")

if __name__ == "__main__":
    logger.info("LiveGuardrailsEngine module ready for production deployment.")
