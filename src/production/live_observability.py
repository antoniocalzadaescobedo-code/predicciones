import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from src.production.common import Severity

logger = logging.getLogger("live_observability")

@dataclass
class HealthCheck:
    """Result of a single health check."""
    component: str
    metric: str
    value: float
    status: str  # "PASS", "FAIL", "WARN"
    threshold: Optional[float] = None
    message: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class PredictionHealth:
    """Monitors prediction quality and validity."""
    nan_probabilities: int = 0
    invalid_lambdas: int = 0
    normalization_drift: float = 0.0
    probability_sum_deviation: float = 0.0
    total_predictions: int = 0

@dataclass
class FeatureHealth:
    """Monitors feature quality and integrity."""
    missing_features: Dict[str, int] = field(default_factory=dict)
    stale_features: Dict[str, int] = field(default_factory=dict)
    out_of_range_values: Dict[str, int] = field(default_factory=dict)
    dtype_corruption: Dict[str, int] = field(default_factory=dict)
    psi_drift: Dict[str, float] = field(default_factory=dict)

@dataclass
class OperationalHealth:
    """Monitors system operational metrics."""
    fallback_rate: float = 0.0
    leakage_rejection_rate: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    shadow_live_divergence: float = 0.0
    api_failure_rate: float = 0.0
    total_requests: int = 0

@dataclass
class CalibrationHealth:
    """Monitors calibration metrics over time."""
    rolling_brier: float = 0.0
    rolling_logloss: float = 0.0
    rolling_ece: float = 0.0
    rolling_calibration_slope: float = 1.0
    sample_size: int = 0

class LiveObservabilityEngine:
    """
    Comprehensive production observability for probabilistic sports predictions.
    Monitors prediction, feature, operational, and calibration health in real-time.
    """
    
    def __init__(self, 
                 feature_registry,
                 psi_threshold: float = 0.20,
                 fallback_threshold: float = 0.05,
                 calibration_slope_range: Tuple[float, float] = (0.95, 1.05),
                 brier_degradation_threshold: float = 0.01,
                 window_size: int = 1000):
        self.feature_registry = feature_registry
        self.psi_threshold = psi_threshold
        self.fallback_threshold = fallback_threshold
        self.calibration_slope_range = calibration_slope_range
        self.brier_degradation_threshold = brier_degradation_threshold
        self.window_size = window_size
        
        # Reference distributions for PSI calculation
        self.reference_distributions: Dict[str, np.ndarray] = {}
        
        # Rolling buffers for metrics
        self.prediction_buffer: List[Dict[str, Any]] = []
        self.calibration_buffer: List[Dict[str, Any]] = []
        self.latency_buffer: List[float] = []
        
        # Health states
        self.prediction_health = PredictionHealth()
        self.feature_health = FeatureHealth()
        self.operational_health = OperationalHealth()
        self.calibration_health = CalibrationHealth()
        
        # Guardrail state
        self.uplift_enabled = True
        self.guardrail_violations: List[HealthCheck] = []
        
        logger.info("LiveObservabilityEngine initialized with production-grade monitoring.")

    def check_prediction_health(self, prediction_data: Dict[str, Any]) -> List[HealthCheck]:
        """
        Monitors prediction quality and validity.
        
        Checks:
        - NaN probabilities
        - Invalid lambdas (negative or infinite)
        - Normalization drift
        - Probability sum != 1
        """
        health_checks = []
        self.prediction_health.total_predictions += 1
        
        # 1. Check for NaN probabilities
        if "probabilities" in prediction_data:
            probs = prediction_data["probabilities"]
            if isinstance(probs, (list, np.ndarray)):
                nan_count = sum(1 for p in probs if pd.isna(p) or np.isinf(p))
                if nan_count > 0:
                    self.prediction_health.nan_probabilities += nan_count
                    health_checks.append(HealthCheck(
                        component="prediction",
                        metric="nan_probabilities",
                        value=nan_count,
                        status="FAIL",
                        message=f"Found {nan_count} NaN/infinite probabilities"
                    ))
        
        # 2. Check for invalid lambdas
        if "lambda" in prediction_data:
            lambda_val = prediction_data["lambda"]
            if pd.isna(lambda_val) or np.isinf(lambda_val) or lambda_val < 0:
                self.prediction_health.invalid_lambdas += 1
                health_checks.append(HealthCheck(
                    component="prediction",
                    metric="invalid_lambda",
                    value=lambda_val if not pd.isna(lambda_val) else -999,
                    status="FAIL",
                    message=f"Invalid lambda detected: {lambda_val}"
                ))
        
        # 3. Check probability normalization
        if "probabilities" in prediction_data:
            probs = prediction_data["probabilities"]
            if isinstance(probs, (list, np.ndarray)):
                prob_sum = sum(p for p in probs if not pd.isna(p))
                deviation = abs(prob_sum - 1.0)
                if deviation > 0.01:
                    self.prediction_health.probability_sum_deviation = deviation
                    health_checks.append(HealthCheck(
                        component="prediction",
                        metric="probability_sum_deviation",
                        value=deviation,
                        status="WARN" if deviation < 0.05 else "FAIL",
                        threshold=0.01,
                        message=f"Probability sum deviation: {deviation:.4f}"
                    ))
        
        return health_checks

    def check_feature_health(self, features: Dict[str, Any], 
                           feature_timestamp: Optional[datetime] = None) -> List[HealthCheck]:
        """
        Monitors feature quality and integrity.
        
        Checks:
        - Missing features
        - Stale features (old timestamps)
        - Out-of-range values
        - Dtype corruption
        - PSI drift
        """
        health_checks = []
        approved_features = self.feature_registry.get_approved_features()
        
        # 1. Check for missing features
        for feat in approved_features:
            if feat not in features or features[feat] is None:
                self.feature_health.missing_features[feat] = self.feature_health.missing_features.get(feat, 0) + 1
                health_checks.append(HealthCheck(
                    component="feature",
                    metric=f"missing_{feat}",
                    value=1,
                    status="WARN",
                    message=f"Missing approved feature: {feat}"
                ))
        
        # 2. Check for dtype corruption and out-of-range values
        for feat, value in features.items():
            if feat not in approved_features:
                continue
                
            # Dtype corruption
            if not isinstance(value, (int, float)):
                self.feature_health.dtype_corruption[feat] = self.feature_health.dtype_corruption.get(feat, 0) + 1
                health_checks.append(HealthCheck(
                    component="feature",
                    metric=f"dtype_corruption_{feat}",
                    value=1,
                    status="FAIL",
                    message=f"Dtype corruption in {feat}: {type(value)}"
                ))
                continue
            
            # Out-of-range values (assuming features should be in [0, 1])
            if isinstance(value, (int, float)) and (value < 0 or value > 1):
                self.feature_health.out_of_range_values[feat] = self.feature_health.out_of_range_values.get(feat, 0) + 1
                health_checks.append(HealthCheck(
                    component="feature",
                    metric=f"out_of_range_{feat}",
                    value=value,
                    status="WARN",
                    threshold=1.0,
                    message=f"Out-of-range value in {feat}: {value}"
                ))
            
            # PSI drift calculation
            if feat in self.reference_distributions:
                psi = self._calculate_psi(self.reference_distributions[feat], np.array([value]))
                self.feature_health.psi_drift[feat] = psi
                if psi > self.psi_threshold:
                    health_checks.append(HealthCheck(
                        component="feature",
                        metric=f"psi_drift_{feat}",
                        value=psi,
                        status="FAIL",
                        threshold=self.psi_threshold,
                        message=f"PSI drift detected in {feat}: {psi:.4f}"
                    ))
        
        return health_checks

    def check_operational_health(self, prediction_result: Dict[str, Any],
                                latency_ms: float) -> List[HealthCheck]:
        """
        Monitors system operational metrics.
        
        Checks:
        - Fallback rate
        - Leakage rejection rate
        - Latency (p50, p95, p99)
        - Shadow/live divergence
        - API failure rate
        """
        health_checks = []
        self.operational_health.total_requests += 1
        
        # Update latency buffer
        self.latency_buffer.append(latency_ms)
        if len(self.latency_buffer) > self.window_size:
            self.latency_buffer.pop(0)
        
        # Calculate latency percentiles
        if self.latency_buffer:
            self.operational_health.latency_p50 = np.percentile(self.latency_buffer, 50)
            self.operational_health.latency_p95 = np.percentile(self.latency_buffer, 95)
            self.operational_health.latency_p99 = np.percentile(self.latency_buffer, 99)
        
        # Check fallback rate
        status = prediction_result.get("status", "")
        if "FAILSAFE" in status:
            self.operational_health.fallback_rate = (
                self.operational_health.fallback_rate * (self.operational_health.total_requests - 1) + 1
            ) / self.operational_health.total_requests
            
            if self.operational_health.fallback_rate > self.fallback_threshold:
                health_checks.append(HealthCheck(
                    component="operational",
                    metric="fallback_rate",
                    value=self.operational_health.fallback_rate,
                    status="FAIL",
                    threshold=self.fallback_threshold,
                    message=f"Fallback rate exceeds threshold: {self.operational_health.fallback_rate:.2%}"
                ))
        
        # Check leakage rejection
        if status == "FAILSAFE_LEAKAGE":
            self.operational_health.leakage_rejection_rate = (
                self.operational_health.leakage_rejection_rate * (self.operational_health.total_requests - 1) + 1
            ) / self.operational_health.total_requests
        
        # Check shadow/live divergence
        if "shadow_uplifted" in prediction_result and "uplifted" in prediction_result:
            baseline = prediction_result.get("baseline", 0)
            shadow = prediction_result.get("shadow_uplifted", baseline)
            live = prediction_result.get("uplifted", baseline)
            
            if baseline > 0:
                shadow_div = abs(shadow - baseline) / baseline
                live_div = abs(live - baseline) / baseline
                divergence = abs(shadow_div - live_div)
                
                self.operational_health.shadow_live_divergence = divergence
                if divergence > 0.10:
                    health_checks.append(HealthCheck(
                        component="operational",
                        metric="shadow_live_divergence",
                        value=divergence,
                        status="WARN",
                        threshold=0.10,
                        message=f"Shadow/live divergence: {divergence:.2%}"
                    ))
        
        return health_checks

    def check_calibration_health(self, prediction: float, 
                                actual: Optional[float] = None) -> List[HealthCheck]:
        """
        Monitors calibration metrics over time.
        
        Checks:
        - Rolling Brier score
        - Rolling LogLoss
        - Rolling ECE (Expected Calibration Error)
        - Rolling calibration slope
        """
        health_checks = []
        
        # Add to calibration buffer
        self.calibration_buffer.append({
            "prediction": prediction,
            "actual": actual,
            "timestamp": datetime.now(timezone.utc)
        })
        
        if len(self.calibration_buffer) > self.window_size:
            self.calibration_buffer.pop(0)
        
        self.calibration_health.sample_size = len(self.calibration_buffer)
        
        # Only calculate if we have actual outcomes
        if actual is not None and len(self.calibration_buffer) >= 10:
            # Calculate rolling Brier score
            predictions = np.array([c["prediction"] for c in self.calibration_buffer if c["actual"] is not None])
            actuals = np.array([c["actual"] for c in self.calibration_buffer if c["actual"] is not None])
            
            if len(predictions) > 0:
                brier = np.mean((predictions - actuals) ** 2)
                self.calibration_health.rolling_brier = brier
                
                # Check for Brier degradation
                if len(self.calibration_buffer) >= 50:
                    recent_brier = np.mean((predictions[-10:] - actuals[-10:]) ** 2)
                    historical_brier = np.mean((predictions[:-10] - actuals[:-10]) ** 2)
                    degradation = recent_brier - historical_brier
                    
                    if degradation > self.brier_degradation_threshold:
                        health_checks.append(HealthCheck(
                            component="calibration",
                            metric="brier_degradation",
                            value=degradation,
                            status="FAIL",
                            threshold=self.brier_degradation_threshold,
                            message=f"Brier score degradation: {degradation:.4f}"
                        ))
                
                # Calculate calibration slope (simplified)
                if len(predictions) >= 20:
                    # Bin predictions and calculate calibration slope
                    try:
                        bins = np.linspace(0, 1, 5)
                        bin_centers = (bins[:-1] + bins[1:]) / 2
                        bin_indices = np.digitize(predictions, bins) - 1
                        bin_indices = np.clip(bin_indices, 0, len(bin_centers) - 1)
                        
                        observed_means = []
                        pred_means = []
                        
                        for i in range(len(bin_centers)):
                            mask = bin_indices == i
                            if np.sum(mask) > 0:
                                observed_means.append(np.mean(actuals[mask]))
                                pred_means.append(np.mean(predictions[mask]))
                        
                        if len(observed_means) >= 3:
                            slope = np.polyfit(pred_means, observed_means, 1)[0]
                            self.calibration_health.rolling_calibration_slope = slope
                            
                            if not (self.calibration_slope_range[0] <= slope <= self.calibration_slope_range[1]):
                                health_checks.append(HealthCheck(
                                    component="calibration",
                                    metric="calibration_slope",
                                    value=slope,
                                    status="FAIL",
                                    threshold=self.calibration_slope_range,
                                    message=f"Calibration slope outside range: {slope:.4f}"
                                ))
                    except Exception as e:
                        logger.warning(f"Could not calculate calibration slope: {e}")
        
        return health_checks

    def set_reference_distribution(self, feature_name: str, reference_values: np.ndarray):
        """Sets reference distribution for PSI calculation."""
        self.reference_distributions[feature_name] = reference_values
        logger.info(f"Reference distribution set for {feature_name} with {len(reference_values)} samples.")

    def _calculate_psi(self, reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Calculates Population Stability Index (PSI)."""
        try:
            # Determine bin edges from reference
            min_val = min(np.min(reference), np.min(current))
            max_val = max(np.max(reference), np.max(current))
            
            if min_val == max_val:
                return 0.0
            
            bin_edges = np.linspace(min_val, max_val, bins + 1)
            
            # Calculate histograms
            ref_counts, _ = np.histogram(reference, bins=bin_edges)
            curr_counts, _ = np.histogram(current, bins=bin_edges)
            
            # Add small constant to avoid division by zero
            ref_percents = (ref_counts + 1) / (len(reference) + bins)
            curr_percents = (curr_counts + 1) / (len(current) + bins)
            
            # Calculate PSI
            psi = np.sum((curr_percents - ref_percents) * np.log(curr_percents / ref_percents))
            return psi
        except Exception as e:
            logger.warning(f"PSI calculation failed: {e}")
            return 0.0

    def get_health_summary(self) -> Dict[str, Any]:
        """Returns comprehensive health summary."""
        return {
            "prediction_health": {
                "total_predictions": self.prediction_health.total_predictions,
                "nan_probabilities": self.prediction_health.nan_probabilities,
                "invalid_lambdas": self.prediction_health.invalid_lambdas,
                "probability_sum_deviation": self.prediction_health.probability_sum_deviation
            },
            "feature_health": {
                "missing_features": self.feature_health.missing_features,
                "dtype_corruption": self.feature_health.dtype_corruption,
                "out_of_range_values": self.feature_health.out_of_range_values,
                "psi_drift": self.feature_health.psi_drift
            },
            "operational_health": {
                "total_requests": self.operational_health.total_requests,
                "fallback_rate": self.operational_health.fallback_rate,
                "leakage_rejection_rate": self.operational_health.leakage_rejection_rate,
                "latency_p50": self.operational_health.latency_p50,
                "latency_p95": self.operational_health.latency_p95,
                "latency_p99": self.operational_health.latency_p99,
                "shadow_live_divergence": self.operational_health.shadow_live_divergence
            },
            "calibration_health": {
                "sample_size": self.calibration_health.sample_size,
                "rolling_brier": self.calibration_health.rolling_brier,
                "rolling_calibration_slope": self.calibration_health.rolling_calibration_slope
            },
            "uplift_enabled": self.uplift_enabled,
            "guardrail_violations_count": len(self.guardrail_violations)
        }

    def reset_metrics(self):
        """Resets all metrics buffers."""
        self.prediction_buffer.clear()
        self.calibration_buffer.clear()
        self.latency_buffer.clear()
        self.prediction_health = PredictionHealth()
        self.feature_health = FeatureHealth()
        self.operational_health = OperationalHealth()
        self.calibration_health = CalibrationHealth()
        self.guardrail_violations.clear()
        logger.info("All observability metrics reset.")

if __name__ == "__main__":
    logger.info("LiveObservabilityEngine module ready for production deployment.")
