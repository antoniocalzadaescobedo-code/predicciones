import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json

logger = logging.getLogger("live_drift_analysis")

class DriftType(Enum):
    PSI_TEMPORAL = "PSI_TEMPORAL"
    FEATURE_DISTRIBUTION_SHIFT = "FEATURE_DISTRIBUTION_SHIFT"
    POST_WORLDCUP_REGIME = "POST_WORLDCUP_REGIME"
    CONFEDERATION_DRIFT = "CONFEDERATION_DRIFT"
    TOURNAMENT_VS_FRIENDLY = "TOURNAMENT_VS_FRIENDLY"
    CALENDAR_CORRUPTION = "CALENDAR_CORRUPTION"
    FEATURE_SPARSITY = "FEATURE_SPARSITY"

class DriftSeverity(Enum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class DriftAlert:
    """Represents a detected drift alert."""
    drift_type: DriftType
    severity: DriftSeverity
    feature: Optional[str] = None
    metric_value: float = 0.0
    threshold: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RegimeState:
    """Represents the current operational regime."""
    is_post_tournament: bool = False
    tournament_end_date: Optional[datetime] = None
    days_since_tournament: int = 0
    regime_confidence: float = 0.0

class LiveDriftAnalyzer:
    """
    Advanced drift detection for production sports prediction systems.
    Detects temporal, distributional, and regime shifts that could impact model performance.
    """
    
    def __init__(self, 
                 psi_threshold: float = 0.20,
                 psi_warning_threshold: float = 0.10,
                 regime_detection_window: int = 60,  # days
                 feature_sparsity_threshold: float = 0.30,
                 confederation_drift_threshold: float = 0.15):
        self.psi_threshold = psi_threshold
        self.psi_warning_threshold = psi_warning_threshold
        self.regime_detection_window = regime_detection_window
        self.feature_sparsity_threshold = feature_sparsity_threshold
        self.confederation_drift_threshold = confederation_drift_threshold
        
        # Reference distributions for PSI calculation
        self.reference_distributions: Dict[str, np.ndarray] = {}
        self.reference_timestamps: Dict[str, datetime] = {}
        
        # Historical data for trend analysis
        self.historical_psi: Dict[str, List[Tuple[datetime, float]]] = {}
        self.historical_feature_means: Dict[str, List[Tuple[datetime, float]]] = {}
        
        # Regime detection
        self.regime_state = RegimeState()
        self.tournament_end_dates = [
            datetime(2026, 7, 19, tzinfo=timezone.utc),  # World Cup 2026 Final (Official: 19 July 2026)
            datetime(2024, 7, 14, tzinfo=timezone.utc),  # Euro 2024
            datetime(2024, 6, 30, tzinfo=timezone.utc),  # Copa America 2024
        ]
        
        # Confederation tracking
        self.confederation_performance: Dict[str, List[float]] = {}
        
        # Match type tracking
        self.tournament_metrics: List[float] = []
        self.friendly_metrics: List[float] = []
        
        logger.info("LiveDriftAnalyzer initialized with multi-dimensional drift detection.")

    def set_reference_distribution(self, feature_name: str, values: np.ndarray):
        """Sets reference distribution for a feature."""
        self.reference_distributions[feature_name] = values
        self.reference_timestamps[feature_name] = datetime.now(timezone.utc)
        logger.info(f"Reference distribution set for {feature_name} with {len(values)} samples")

    def calculate_psi(self, feature_name: str, current_values: np.ndarray) -> float:
        """
        Calculates Population Stability Index (PSI) for temporal drift detection.
        
        Args:
            feature_name: Name of the feature
            current_values: Current feature values
            
        Returns:
            PSI value
        """
        if feature_name not in self.reference_distributions:
            logger.warning(f"No reference distribution for {feature_name}")
            return 0.0
        
        reference = self.reference_distributions[feature_name]
        
        try:
            # Determine bin edges from reference
            min_val = min(np.min(reference), np.min(current_values))
            max_val = max(np.max(reference), np.max(current_values))
            
            if min_val == max_val:
                return 0.0
            
            bins = 10
            bin_edges = np.linspace(min_val, max_val, bins + 1)
            
            # Calculate histograms
            ref_counts, _ = np.histogram(reference, bins=bin_edges)
            curr_counts, _ = np.histogram(current_values, bins=bin_edges)
            
            # Add small constant to avoid division by zero
            ref_percents = (ref_counts + 1) / (len(reference) + bins)
            curr_percents = (curr_counts + 1) / (len(current_values) + bins)
            
            # Calculate PSI
            psi = np.sum((curr_percents - ref_percents) * np.log(curr_percents / ref_percents))
            
            # Store historical PSI
            if feature_name not in self.historical_psi:
                self.historical_psi[feature_name] = []
            self.historical_psi[feature_name].append((datetime.now(timezone.utc), psi))
            
            # Keep only last 1000 points
            if len(self.historical_psi[feature_name]) > 1000:
                self.historical_psi[feature_name] = self.historical_psi[feature_name][-1000:]
            
            return psi
        except Exception as e:
            logger.warning(f"PSI calculation failed for {feature_name}: {e}")
            return 0.0

    def detect_psi_temporal_drift(self, feature_name: str, current_values: np.ndarray) -> Optional[DriftAlert]:
        """
        Detects temporal drift using PSI.
        
        Args:
            feature_name: Name of the feature
            current_values: Current feature values
            
        Returns:
            DriftAlert if drift detected, None otherwise
        """
        psi = self.calculate_psi(feature_name, current_values)
        
        if psi > self.psi_threshold:
            return DriftAlert(
                drift_type=DriftType.PSI_TEMPORAL,
                severity=DriftSeverity.CRITICAL,
                feature=feature_name,
                metric_value=psi,
                threshold=self.psi_threshold,
                message=f"Critical PSI drift detected in {feature_name}: {psi:.4f}"
            )
        elif psi > self.psi_warning_threshold:
            return DriftAlert(
                drift_type=DriftType.PSI_TEMPORAL,
                severity=DriftSeverity.MEDIUM,
                feature=feature_name,
                metric_value=psi,
                threshold=self.psi_warning_threshold,
                message=f"Warning: PSI drift in {feature_name}: {psi:.4f}"
            )
        
        return None

    def detect_feature_distribution_shift(self, feature_name: str, 
                                        current_values: np.ndarray) -> Optional[DriftAlert]:
        """
        Detects distribution shift using statistical tests.
        
        Args:
            feature_name: Name of the feature
            current_values: Current feature values
            
        Returns:
            DriftAlert if shift detected, None otherwise
        """
        if feature_name not in self.reference_distributions:
            return None
        
        reference = self.reference_distributions[feature_name]
        
        # Calculate statistical moments
        ref_mean, ref_std = np.mean(reference), np.std(reference)
        curr_mean, curr_std = np.mean(current_values), np.std(current_values)
        
        # Store historical means
        if feature_name not in self.historical_feature_means:
            self.historical_feature_means[feature_name] = []
        self.historical_feature_means[feature_name].append((datetime.now(timezone.utc), curr_mean))
        
        # Keep only last 1000 points
        if len(self.historical_feature_means[feature_name]) > 1000:
            self.historical_feature_means[feature_name] = self.historical_feature_means[feature_name][-1000:]
        
        # Detect mean shift (using 3-sigma rule)
        if ref_std > 0:
            z_score = abs(curr_mean - ref_mean) / ref_std
            if z_score > 3.0:
                return DriftAlert(
                    drift_type=DriftType.FEATURE_DISTRIBUTION_SHIFT,
                    severity=DriftSeverity.HIGH,
                    feature=feature_name,
                    metric_value=z_score,
                    threshold=3.0,
                    message=f"Mean shift detected in {feature_name}: {curr_mean:.4f} vs {ref_mean:.4f} (z={z_score:.2f})"
                )
        
        # Detect variance shift
        if ref_std > 0 and curr_std > 0:
            variance_ratio = curr_std / ref_std
            if variance_ratio > 2.0 or variance_ratio < 0.5:
                return DriftAlert(
                    drift_type=DriftType.FEATURE_DISTRIBUTION_SHIFT,
                    severity=DriftSeverity.MEDIUM,
                    feature=feature_name,
                    metric_value=variance_ratio,
                    threshold=2.0,
                    message=f"Variance shift detected in {feature_name}: {curr_std:.4f} vs {ref_std:.4f}"
                )
        
        return None

    def detect_post_worldcup_regime(self) -> Optional[DriftAlert]:
        """
        Detects post-World Cup regime where continuity features may collapse.
        
        Returns:
            DriftAlert if post-tournament regime detected, None otherwise
        """
        now = datetime.now(timezone.utc)
        
        # Check if we're in post-tournament window
        for wc_end in self.tournament_end_dates:
            if wc_end < now < wc_end + timedelta(days=self.regime_detection_window):
                days_since = (now - wc_end).days
                
                # Calculate regime confidence based on days since tournament
                confidence = 1.0 - (days_since / self.regime_detection_window)
                
                self.regime_state = RegimeState(
                    is_post_tournament=True,
                    tournament_end_date=wc_end,
                    days_since_tournament=days_since,
                    regime_confidence=confidence
                )
                
                severity = DriftSeverity.CRITICAL if days_since < 30 else DriftSeverity.HIGH
                
                return DriftAlert(
                    drift_type=DriftType.POST_WORLDCUP_REGIME,
                    severity=severity,
                    metric_value=confidence,
                    threshold=0.5,
                    message=f"Post-tournament regime detected: {days_since} days since tournament end. "
                            f"Expect continuity feature collapse."
                )
        
        self.regime_state = RegimeState(is_post_tournament=False)
        return None

    def detect_confederation_drift(self, confederation: str, 
                                  performance_metric: float) -> Optional[DriftAlert]:
        """
        Detects drift in performance by confederation.
        
        Args:
            confederation: Confederation name (e.g., UEFA, CONMEBOL)
            performance_metric: Performance metric value
            
        Returns:
            DriftAlert if drift detected, None otherwise
        """
        if confederation not in self.confederation_performance:
            self.confederation_performance[confederation] = []
        
        self.confederation_performance[confederation].append(performance_metric)
        
        # Keep only last 100 points
        if len(self.confederation_performance[confederation]) > 100:
            self.confederation_performance[confederation] = self.confederation_performance[confederation][-100:]
        
        # Detect drift if we have enough history
        if len(self.confederation_performance[confederation]) >= 20:
            recent = np.mean(self.confederation_performance[confederation][-10:])
            historical = np.mean(self.confederation_performance[confederation][-20:-10])
            
            drift = abs(recent - historical)
            if drift > self.confederation_drift_threshold:
                return DriftAlert(
                    drift_type=DriftType.CONFEDERATION_DRIFT,
                    severity=DriftSeverity.MEDIUM,
                    feature=confederation,
                    metric_value=drift,
                    threshold=self.confederation_drift_threshold,
                    message=f"Performance drift detected for {confederation}: {drift:.4f}"
                )
        
        return None

    def detect_tournament_vs_friendly_drift(self, is_tournament: bool, 
                                          performance_metric: float) -> Optional[DriftAlert]:
        """
        Detects drift between tournament and friendly match performance.
        
        Args:
            is_tournament: Whether the match is a tournament match
            performance_metric: Performance metric value
            
        Returns:
            DriftAlert if drift detected, None otherwise
        """
        if is_tournament:
            self.tournament_metrics.append(performance_metric)
        else:
            self.friendly_metrics.append(performance_metric)
        
        # Keep only last 100 points each
        if len(self.tournament_metrics) > 100:
            self.tournament_metrics = self.tournament_metrics[-100:]
        if len(self.friendly_metrics) > 100:
            self.friendly_metrics = self.friendly_metrics[-100:]
        
        # Detect drift if we have enough data
        if len(self.tournament_metrics) >= 10 and len(self.friendly_metrics) >= 10:
            tournament_mean = np.mean(self.tournament_metrics[-10:])
            friendly_mean = np.mean(self.friendly_metrics[-10:])
            
            divergence = abs(tournament_mean - friendly_mean)
            if divergence > self.confederation_drift_threshold:
                return DriftAlert(
                    drift_type=DriftType.TOURNAMENT_VS_FRIENDLY,
                    severity=DriftSeverity.MEDIUM,
                    metric_value=divergence,
                    threshold=self.confederation_drift_threshold,
                    message=f"Tournament vs Friendly divergence: {divergence:.4f} "
                            f"(Tournament: {tournament_mean:.4f}, Friendly: {friendly_mean:.4f})"
                )
        
        return None

    def detect_feature_sparsity(self, features: Dict[str, Any]) -> Optional[DriftAlert]:
        """
        Detects spikes in feature sparsity (missing values).
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            DriftAlert if sparsity spike detected, None otherwise
        """
        total_features = len(features)
        missing_count = sum(1 for v in features.values() if v is None or pd.isna(v))
        
        if total_features > 0:
            sparsity = missing_count / total_features
            if sparsity > self.feature_sparsity_threshold:
                return DriftAlert(
                    drift_type=DriftType.FEATURE_SPARSITY,
                    severity=DriftSeverity.HIGH,
                    metric_value=sparsity,
                    threshold=self.feature_sparsity_threshold,
                    message=f"Feature sparsity spike: {sparsity:.2%} ({missing_count}/{total_features} missing)"
                )
        
        return None

    def detect_calendar_corruption(self, match_date: datetime, 
                                  feature_timestamp: datetime) -> Optional[DriftAlert]:
        """
        Detects calendar corruption (e.g., future dates, impossible dates).
        
        Args:
            match_date: Match date
            feature_timestamp: Feature timestamp
            
        Returns:
            DriftAlert if calendar corruption detected, None otherwise
        """
        now = datetime.now(timezone.utc)
        
        # Check for future dates
        if match_date > now + timedelta(days=365):
            return DriftAlert(
                drift_type=DriftType.CALENDAR_CORRUPTION,
                severity=DriftSeverity.CRITICAL,
                metric_value=(match_date - now).days,
                threshold=365,
                message=f"Future match date detected: {match_date.isoformat()}"
            )
        
        # Check for feature timestamp in the future relative to match
        if feature_timestamp and match_date and feature_timestamp > match_date:
            return DriftAlert(
                drift_type=DriftType.CALENDAR_CORRUPTION,
                severity=DriftSeverity.CRITICAL,
                metric_value=(feature_timestamp - match_date).total_seconds(),
                threshold=0,
                message=f"Feature timestamp after match date: {feature_timestamp.isoformat()} > {match_date.isoformat()}"
            )
        
        # Check for very old dates (before 2020)
        if match_date < datetime(2020, 1, 1, tzinfo=timezone.utc):
            return DriftAlert(
                drift_type=DriftType.CALENDAR_CORRUPTION,
                severity=DriftSeverity.HIGH,
                metric_value=(datetime.now(timezone.utc) - match_date).days,
                threshold=0,
                message=f"Suspiciously old match date: {match_date.isoformat()}"
            )
        
        return None

    def run_comprehensive_drift_check(self, 
                                    features: Dict[str, Any],
                                    match_date: Optional[datetime] = None,
                                    feature_timestamp: Optional[datetime] = None,
                                    is_tournament: bool = False,
                                    confederation: Optional[str] = None,
                                    performance_metric: Optional[float] = None) -> List[DriftAlert]:
        """
        Runs all drift checks and returns detected alerts.
        
        Args:
            features: Dictionary of feature values
            match_date: Match date
            feature_timestamp: Feature timestamp
            is_tournament: Whether match is tournament
            confederation: Confederation name
            performance_metric: Performance metric
            
        Returns:
            List of detected drift alerts
        """
        alerts = []
        
        # 1. Post-World Cup regime detection
        regime_alert = self.detect_post_worldcup_regime()
        if regime_alert:
            alerts.append(regime_alert)
        
        # 2. Feature sparsity detection
        sparsity_alert = self.detect_feature_sparsity(features)
        if sparsity_alert:
            alerts.append(sparsity_alert)
        
        # 3. Calendar corruption detection
        if match_date:
            calendar_alert = self.detect_calendar_corruption(match_date, feature_timestamp)
            if calendar_alert:
                alerts.append(calendar_alert)
        
        # 4. PSI temporal drift for each feature
        for feature_name, value in features.items():
            if value is not None and not pd.isna(value) and isinstance(value, (int, float)):
                current_values = np.array([value])
                
                psi_alert = self.detect_psi_temporal_drift(feature_name, current_values)
                if psi_alert:
                    alerts.append(psi_alert)
                
                dist_alert = self.detect_feature_distribution_shift(feature_name, current_values)
                if dist_alert:
                    alerts.append(dist_alert)
        
        # 5. Confederation drift
        if confederation and performance_metric is not None:
            confed_alert = self.detect_confederation_drift(confederation, performance_metric)
            if confed_alert:
                alerts.append(confed_alert)
        
        # 6. Tournament vs Friendly drift
        if performance_metric is not None:
            tvf_alert = self.detect_tournament_vs_friendly_drift(is_tournament, performance_metric)
            if tvf_alert:
                alerts.append(tvf_alert)
        
        return alerts

    def get_drift_summary(self) -> Dict[str, Any]:
        """Returns comprehensive drift analysis summary."""
        return {
            "regime_state": {
                "is_post_tournament": self.regime_state.is_post_tournament,
                "tournament_end_date": self.regime_state.tournament_end_date.isoformat() if self.regime_state.tournament_end_date else None,
                "days_since_tournament": self.regime_state.days_since_tournament,
                "regime_confidence": self.regime_state.regime_confidence
            },
            "reference_distributions": {
                feat: {
                    "sample_count": len(dist),
                    "set_at": self.reference_timestamps[feat].isoformat()
                }
                for feat, dist in self.reference_distributions.items()
            },
            "historical_psi_points": {
                feat: len(points)
                for feat, points in self.historical_psi.items()
            },
            "confederation_tracking": {
                confed: len(metrics)
                for confed, metrics in self.confederation_performance.items()
            },
            "match_type_tracking": {
                "tournament_samples": len(self.tournament_metrics),
                "friendly_samples": len(self.friendly_metrics)
            }
        }

    def export_drift_report(self, output_path: Path):
        """Exports drift analysis report to file."""
        summary = self.get_drift_summary()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Drift analysis report exported to {output_path}")

if __name__ == "__main__":
    logger.info("LiveDriftAnalyzer module ready for production deployment.")
