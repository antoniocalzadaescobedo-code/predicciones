# PRODUCTION HARDENING - TEMPORAL SQUAD FEATURES
## Operational Resilience, Statistical Robustness, Feature Governance

## 1. RUNTIME MONITORING SYSTEM

### 1.1 Feature Drift Detection

**Mathematical Definition:**

Population Stability Index (PSI):
```
PSI = Σ (expected_i - actual_i) * ln(expected_i / actual_i)
```

Where:
- expected_i: Expected distribution (baseline)
- actual_i: Actual distribution (current)

**Thresholds:**
- PSI < 0.1: No significant drift
- 0.1 ≤ PSI < 0.2: Moderate drift (monitor)
- PSI ≥ 0.2: Significant drift (action required)

**Implementation:**

```python
"""
runtime_monitoring.py
Production-grade runtime monitoring for temporal features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
from scipy import stats
from collections import defaultdict

class DriftSeverity(Enum):
    NONE = "none"
    MODERATE = "moderate"
    SEVERE = "severe"

@dataclass(frozen=True)
class FeatureDriftMetric:
    """Immutable drift metric for auditability."""
    feature_name: str
    psi_value: float
    ks_statistic: float
    ks_p_value: float
    severity: DriftSeverity
    timestamp: str
    baseline_window_start: str
    baseline_window_end: str
    current_window_start: str
    current_window_end: str

@dataclass(frozen=True)
class FeatureFreshnessMetric:
    """Immutable freshness metric."""
    feature_name: str
    latest_timestamp: str
    age_hours: float
    staleness_threshold_hours: float
    is_stale: bool
    timestamp: str

@dataclass(frozen=True)
class MissingRatioMetric:
    """Immutable missing ratio metric."""
    feature_name: str
    total_samples: int
    missing_count: int
    missing_ratio: float
    threshold: float
    exceeds_threshold: bool
    timestamp: str

@dataclass(frozen=True)
class VarianceCollapseMetric:
    """Immutable variance collapse metric."""
    feature_name: str
    current_variance: float
    baseline_variance: float
    variance_ratio: float
    collapse_threshold: float
    is_collapsed: bool
    timestamp: str

class RuntimeMonitor:
    """
    Production-grade runtime monitoring for temporal features.
    
    Monitors:
    - Feature drift (PSI, KS test)
    - Feature freshness
    - Missing ratios
    - Variance collapse
    - Stale snapshots
    - Temporal inconsistencies
    """
    
    def __init__(self, 
                 psi_threshold: float = 0.2,
                 staleness_threshold_hours: float = 24.0,
                 missing_ratio_threshold: float = 0.1,
                 variance_collapse_threshold: float = 0.1):
        self.psi_threshold = psi_threshold
        self.staleness_threshold_hours = staleness_threshold_hours
        self.missing_ratio_threshold = missing_ratio_threshold
        self.variance_collapse_threshold = variance_collapse_threshold
        
        # Baseline distributions (initialized from historical data)
        self.baseline_distributions: Dict[str, np.ndarray] = {}
        self.baseline_windows: Dict[str, Tuple[str, str]] = {}
    
    def initialize_baseline(self, feature_name: str, 
                          values: np.ndarray,
                          window_start: str,
                          window_end: str) -> None:
        """
        Initialize baseline distribution for a feature.
        
        Args:
            feature_name: Name of feature
            values: Historical feature values
            window_start: Start of baseline window
            window_end: End of baseline window
        """
        self.baseline_distributions[feature_name] = values
        self.baseline_windows[feature_name] = (window_start, window_end)
    
    def calculate_psi(self, expected: np.ndarray, 
                     actual: np.ndarray,
                     bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI = Σ (expected_i - actual_i) * ln(expected_i / actual_i)
        
        Args:
            expected: Expected distribution (baseline)
            actual: Actual distribution (current)
            bins: Number of bins for histogram
        
        Returns:
            PSI value
        """
        # Create bins based on expected distribution
        expected_min, expected_max = expected.min(), expected.max()
        bin_edges = np.linspace(expected_min, expected_max, bins + 1)
        
        # Calculate histograms
        expected_hist, _ = np.histogram(expected, bins=bin_edges)
        actual_hist, _ = np.histogram(actual, bins=bin_edges)
        
        # Normalize to percentages
        expected_pct = expected_hist / expected_hist.sum()
        actual_pct = actual_hist / actual_hist.sum()
        
        # Handle zero bins (add small epsilon)
        epsilon = 1e-10
        expected_pct = np.maximum(expected_pct, epsilon)
        actual_pct = np.maximum(actual_pct, epsilon)
        
        # Calculate PSI
        psi = np.sum((expected_pct - actual_pct) * np.log(expected_pct / actual_pct))
        
        return psi
    
    def detect_feature_drift(self, feature_name: str,
                            current_values: np.ndarray,
                            current_window_start: str,
                            current_window_end: str) -> FeatureDriftMetric:
        """
        Detect feature drift using PSI and KS test.
        
        Args:
            feature_name: Name of feature
            current_values: Current feature values
            current_window_start: Start of current window
            current_window_end: End of current window
        
        Returns:
            FeatureDriftMetric with drift analysis
        """
        if feature_name not in self.baseline_distributions:
            raise ValueError(f"No baseline initialized for {feature_name}")
        
        baseline = self.baseline_distributions[feature_name]
        baseline_window = self.baseline_windows[feature_name]
        
        # Calculate PSI
        psi = self.calculate_psi(baseline, current_values)
        
        # Calculate KS statistic
        ks_stat, ks_p_value = stats.ks_2samp(baseline, current_values)
        
        # Determine severity
        if psi < 0.1:
            severity = DriftSeverity.NONE
        elif psi < 0.2:
            severity = DriftSeverity.MODERATE
        else:
            severity = DriftSeverity.SEVERE
        
        return FeatureDriftMetric(
            feature_name=feature_name,
            psi_value=psi,
            ks_statistic=ks_stat,
            ks_p_value=ks_p_value,
            severity=severity,
            timestamp=datetime.utcnow().isoformat(),
            baseline_window_start=baseline_window[0],
            baseline_window_end=baseline_window[1],
            current_window_start=current_window_start,
            current_window_end=current_window_end
        )
    
    def check_feature_freshness(self, feature_name: str,
                               latest_timestamp: str) -> FeatureFreshnessMetric:
        """
        Check if feature is stale (too old).
        
        Args:
            feature_name: Name of feature
            latest_timestamp: Latest timestamp of feature
        
        Returns:
            FeatureFreshnessMetric with freshness analysis
        """
        latest_dt = datetime.fromisoformat(latest_timestamp)
        current_dt = datetime.utcnow()
        age_hours = (current_dt - latest_dt).total_seconds() / 3600
        
        is_stale = age_hours > self.staleness_threshold_hours
        
        return FeatureFreshnessMetric(
            feature_name=feature_name,
            latest_timestamp=latest_timestamp,
            age_hours=age_hours,
            staleness_threshold_hours=self.staleness_threshold_hours,
            is_stale=is_stale,
            timestamp=current_dt.isoformat()
        )
    
    def calculate_missing_ratio(self, feature_name: str,
                               values: np.ndarray) -> MissingRatioMetric:
        """
        Calculate missing ratio for feature.
        
        Args:
            feature_name: Name of feature
            values: Feature values (may contain NaN)
        
        Returns:
            MissingRatioMetric with missing analysis
        """
        total_samples = len(values)
        missing_count = np.sum(pd.isna(values))
        missing_ratio = missing_count / total_samples if total_samples > 0 else 0.0
        
        exceeds_threshold = missing_ratio > self.missing_ratio_threshold
        
        return MissingRatioMetric(
            feature_name=feature_name,
            total_samples=total_samples,
            missing_count=missing_count,
            missing_ratio=missing_ratio,
            threshold=self.missing_ratio_threshold,
            exceeds_threshold=exceeds_threshold,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def detect_variance_collapse(self, feature_name: str,
                                current_values: np.ndarray) -> VarianceCollapseMetric:
        """
        Detect variance collapse (feature lost discriminative power).
        
        Args:
            feature_name: Name of feature
            current_values: Current feature values
        
        Returns:
            VarianceCollapseMetric with variance analysis
        """
        if feature_name not in self.baseline_distributions:
            raise ValueError(f"No baseline initialized for {feature_name}")
        
        baseline = self.baseline_distributions[feature_name]
        baseline_variance = np.var(baseline)
        current_variance = np.var(current_values)
        
        variance_ratio = current_variance / baseline_variance if baseline_variance > 0 else 0.0
        is_collapsed = variance_ratio < self.variance_collapse_threshold
        
        return VarianceCollapseMetric(
            feature_name=feature_name,
            current_variance=current_variance,
            baseline_variance=baseline_variance,
            variance_ratio=variance_ratio,
            collapse_threshold=self.variance_collapse_threshold,
            is_collapsed=is_collapsed,
            timestamp=datetime.utcnow().isoformat()
        )
```

### 1.2 Temporal Inconsistency Detection

**Mathematical Definition:**

Timestamp Consistency Check:
```
∀ snapshots s_i, s_j of same team:
if s_i.fetch_timestamp > s_j.fetch_timestamp:
    s_i must be >= s_j in temporal order
```

**Implementation:**

```python
class TemporalConsistencyValidator:
    """
    Validate temporal consistency of squad snapshots.
    
    Detects:
    - Out-of-order timestamps
    - Duplicate timestamps
    - Gaps in temporal sequence
    - Impossible temporal intervals
    """
    
    def __init__(self):
        self.violations: List[Dict] = []
    
    def validate_team_snapshots(self, team_id: str,
                               snapshots: List[Dict]) -> List[Dict]:
        """
        Validate temporal consistency of team snapshots.
        
        Args:
            team_id: Team identifier
            snapshots: List of snapshot dicts
        
        Returns:
            List of temporal violations
        """
        self.violations = []
        
        if len(snapshots) < 2:
            return self.violations
        
        # Sort by fetch_timestamp
        sorted_snapshots = sorted(snapshots, 
                                  key=lambda x: x['fetch_timestamp_utc'])
        
        # Check for out-of-order timestamps
        for i in range(1, len(sorted_snapshots)):
            prev_ts = datetime.fromisoformat(sorted_snapshots[i-1]['fetch_timestamp_utc'])
            curr_ts = datetime.fromisoformat(sorted_snapshots[i]['fetch_timestamp_utc'])
            
            if curr_ts < prev_ts:
                self.violations.append({
                    'type': 'out_of_order',
                    'team_id': team_id,
                    'prev_timestamp': sorted_snapshots[i-1]['fetch_timestamp_utc'],
                    'curr_timestamp': sorted_snapshots[i]['fetch_timestamp_utc'],
                    'severity': 'severe'
                })
        
        # Check for duplicate timestamps
        timestamps = [s['fetch_timestamp_utc'] for s in sorted_snapshots]
        timestamp_counts = defaultdict(int)
        for ts in timestamps:
            timestamp_counts[ts] += 1
        
        for ts, count in timestamp_counts.items():
            if count > 1:
                self.violations.append({
                    'type': 'duplicate_timestamp',
                    'team_id': team_id,
                    'timestamp': ts,
                    'count': count,
                    'severity': 'moderate'
                })
        
        # Check for temporal gaps
        for i in range(1, len(sorted_snapshots)):
            prev_ts = datetime.fromisoformat(sorted_snapshots[i-1]['fetch_timestamp_utc'])
            curr_ts = datetime.fromisoformat(sorted_snapshots[i]['fetch_timestamp_utc'])
            gap_days = (curr_ts - prev_ts).days
            
            # Gap > 30 days is suspicious for squad announcements
            if gap_days > 30:
                self.violations.append({
                    'type': 'temporal_gap',
                    'team_id': team_id,
                    'prev_timestamp': sorted_snapshots[i-1]['fetch_timestamp_utc'],
                    'curr_timestamp': sorted_snapshots[i]['fetch_timestamp_utc'],
                    'gap_days': gap_days,
                    'severity': 'moderate'
                })
        
        return self.violations
```

## 2. STATISTICAL ROBUSTNESS FRAMEWORK

### 2.1 Block Bootstrap Temporal

**Mathematical Definition:**

Moving Block Bootstrap (MBB):
```
For i = 1 to B:
    Sample blocks of length L from time series
    Concatenate blocks to form bootstrap sample
    Compute statistic on bootstrap sample
```

**Implementation:**

```python
"""
statistical_robustness.py
Statistical robustness framework for temporal features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime
from dataclasses import dataclass
from scipy import stats
from scipy.stats import bootstrap as scipy_bootstrap
import warnings

@dataclass(frozen=True)
class BootstrapResult:
    """Immutable bootstrap result."""
    statistic_name: str
    original_value: float
    bootstrap_mean: float
    bootstrap_std: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    n_bootstrap: int
    block_length: int

class BlockBootstrap:
    """
    Moving Block Bootstrap for temporal data.
    
    Preserves temporal autocorrelation structure.
    """
    
    def __init__(self, block_length: int = 5, n_bootstrap: int = 1000):
        self.block_length = block_length
        self.n_bootstrap = n_bootstrap
    
    def moving_block_bootstrap(self, data: np.ndarray,
                               statistic: Callable[[np.ndarray], float],
                               ci_level: float = 0.95) -> BootstrapResult:
        """
        Perform moving block bootstrap.
        
        Args:
            data: Time series data
            statistic: Function to compute statistic
            ci_level: Confidence interval level
        
        Returns:
            BootstrapResult with bootstrap statistics
        """
        n = len(data)
        
        if n < self.block_length:
            warnings.warn(f"Data length {n} < block length {self.block_length}")
        
        # Compute original statistic
        original_value = statistic(data)
        
        # Bootstrap samples
        bootstrap_values = []
        
        for _ in range(self.n_bootstrap):
            # Sample blocks
            n_blocks = n // self.block_length
            block_starts = np.random.randint(0, n - self.block_length + 1, n_blocks)
            
            # Construct bootstrap sample
            bootstrap_sample = []
            for start in block_starts:
                block = data[start:start + self.block_length]
                bootstrap_sample.extend(block)
            
            # Trim to original length
            bootstrap_sample = np.array(bootstrap_sample[:n])
            
            # Compute statistic
            bootstrap_values.append(statistic(bootstrap_sample))
        
        bootstrap_values = np.array(bootstrap_values)
        
        # Compute bootstrap statistics
        bootstrap_mean = np.mean(bootstrap_values)
        bootstrap_std = np.std(bootstrap_values)
        
        # Compute confidence interval
        alpha = 1 - ci_level
        ci_lower = np.percentile(bootstrap_values, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_values, (1 - alpha / 2) * 100)
        
        return BootstrapResult(
            statistic_name=statistic.__name__,
            original_value=original_value,
            bootstrap_mean=bootstrap_mean,
            bootstrap_std=bootstrap_std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_level=ci_level,
            n_bootstrap=self.n_bootstrap,
            block_length=self.block_length
        )

class AutocorrelationAnalyzer:
    """
    Analyze autocorrelation in temporal features.
    
    High autocorrelation → temporal dependence → invalid standard errors
    """
    
    @staticmethod
    def calculate_autocorrelation(data: np.ndarray, max_lag: int = 10) -> np.ndarray:
        """
        Calculate autocorrelation function.
        
        Args:
            data: Time series data
            max_lag: Maximum lag to compute
        
        Returns:
            Array of autocorrelation coefficients
        """
        n = len(data)
        mean = np.mean(data)
        variance = np.var(data)
        
        if variance == 0:
            return np.zeros(max_lag + 1)
        
        autocorr = np.zeros(max_lag + 1)
        autocorr[0] = 1.0  # Lag 0 is always 1
        
        for lag in range(1, max_lag + 1):
            if lag >= n:
                autocorr[lag] = 0.0
                continue
            
            covariance = np.mean((data[:n-lag] - mean) * (data[lag:] - mean))
            autocorr[lag] = covariance / variance
        
        return autocorr
    
    @staticmethod
    def ljung_box_test(data: np.ndarray, lags: int = 10) -> Tuple[float, float]:
        """
        Ljung-Box test for autocorrelation.
        
        H0: No autocorrelation up to specified lag
        
        Args:
            data: Time series data
            lags: Number of lags to test
        
        Returns:
            (test_statistic, p_value)
        """
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        result = acorr_ljungbox(data, lags=[lags])
        test_statistic = result.iloc[0]['lb_stat']
        p_value = result.iloc[0]['lb_pvalue']
        
        return test_statistic, p_value

class UpliftStabilityAnalyzer:
    """
    Analyze stability of uplift metrics over time.
    
    Detects if uplift is consistent or spurious.
    """
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
    
    def rolling_uplift(self, baseline_metrics: np.ndarray,
                      model_metrics: np.ndarray) -> np.ndarray:
        """
        Calculate rolling uplift.
        
        Args:
            baseline_metrics: Baseline metrics over time
            model_metrics: Model metrics over time
        
        Returns:
            Array of rolling uplift values
        """
        uplift = baseline_metrics - model_metrics
        rolling_uplift = np.convolve(uplift, 
                                    np.ones(self.window_size)/self.window_size, 
                                    mode='valid')
        return rolling_uplift
    
    def uplift_stability_test(self, uplift_series: np.ndarray,
                              significance_level: float = 0.05) -> Dict:
        """
        Test if uplift is stable over time.
        
        Uses Augmented Dickey-Fuller test for stationarity.
        
        Args:
            uplift_series: Uplift values over time
            significance_level: Significance level for test
        
        Returns:
            Dict with test results
        """
        from statsmodels.tsa.stattools import adfuller
        
        result = adfuller(uplift_series)
        
        is_stationary = result[1] < significance_level
        
        return {
            'is_stationary': is_stationary,
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'significance_level': significance_level
        }

class FeatureEntropyAnalyzer:
    """
    Analyze entropy of feature distributions.
    
    Low entropy → feature has little discriminative power
    High entropy → feature has high discriminative power
    """
    
    @staticmethod
    def calculate_shannon_entropy(values: np.ndarray, 
                                 bins: int = 10) -> float:
        """
        Calculate Shannon entropy of feature distribution.
        
        H = -Σ p(x) * log(p(x))
        
        Args:
            values: Feature values
            bins: Number of bins for histogram
        
        Returns:
            Shannon entropy
        """
        hist, _ = np.histogram(values, bins=bins)
        probs = hist / hist.sum()
        
        # Remove zero probabilities
        probs = probs[probs > 0]
        
        entropy = -np.sum(probs * np.log(probs))
        
        return entropy
    
    @staticmethod
    def entropy_drift(baseline_entropy: float,
                    current_entropy: float,
                    threshold: float = 0.1) -> Dict:
        """
        Detect entropy drift.
        
        Args:
            baseline_entropy: Baseline entropy
            current_entropy: Current entropy
            threshold: Drift threshold
        
        Returns:
            Dict with drift analysis
        """
        drift = abs(current_entropy - baseline_entropy)
        is_drifted = drift > threshold
        
        return {
            'baseline_entropy': baseline_entropy,
            'current_entropy': current_entropy,
            'drift': drift,
            'threshold': threshold,
            'is_drifted': is_drifted
        }

class UncertaintyPropagator:
    """
    Propagate uncertainty through feature pipeline.
    
    Quantifies how uncertainty in inputs affects outputs.
    """
    
    @staticmethod
    def monte_carlo_uncertainty(feature_values: np.ndarray,
                               noise_std: float = 0.01,
                               n_samples: int = 1000) -> Dict:
        """
        Monte Carlo uncertainty propagation.
        
        Args:
            feature_values: Feature values
            noise_std: Standard deviation of input noise
            n_samples: Number of Monte Carlo samples
        
        Returns:
            Dict with uncertainty statistics
        """
        results = []
        
        for _ in range(n_samples):
            # Add noise to input
            noisy_values = feature_values + np.random.normal(0, noise_std, len(feature_values))
            
            # Compute statistic (e.g., mean)
            results.append(np.mean(noisy_values))
        
        results = np.array(results)
        
        return {
            'mean': np.mean(results),
            'std': np.std(results),
            'ci_lower': np.percentile(results, 2.5),
            'ci_upper': np.percentile(results, 97.5),
            'coefficient_of_variation': np.std(results) / np.mean(results)
        }
```

## 3. FEATURE GOVERNANCE SYSTEM

### 3.1 Feature Registry

**Schema:**

```sql
CREATE TABLE feature_registry (
    feature_id TEXT PRIMARY KEY,
    feature_name TEXT NOT NULL UNIQUE,
    feature_type TEXT NOT NULL,
    data_type TEXT NOT NULL,
    description TEXT,
    causal_hypothesis TEXT,
    scientific_risk TEXT,
    mathematical_definition TEXT,
    created_at TEXT NOT NULL,
    created_by TEXT NOT NULL,
    status TEXT NOT NULL,  -- experimental, validated, deprecated
    version INTEGER NOT NULL,
    schema_version TEXT NOT NULL,
    INDEX idx_feature_name (feature_name),
    INDEX idx_status (status)
);

CREATE TABLE feature_version_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    schema_version TEXT NOT NULL,
    computation_version TEXT NOT NULL,
    changed_at TEXT NOT NULL,
    changed_by TEXT NOT NULL,
    change_description TEXT,
    FOREIGN KEY (feature_id) REFERENCES feature_registry(feature_id),
    INDEX idx_feature_version (feature_id, version)
);

CREATE TABLE feature_lineage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_id TEXT NOT NULL,
    source_snapshot_id INTEGER NOT NULL,
    transformation_type TEXT NOT NULL,
    parameters_json TEXT,
    output_feature_id TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (feature_id) REFERENCES feature_registry(feature_id),
    FOREIGN KEY (source_snapshot_id) REFERENCES squad_snapshots(id),
    INDEX idx_feature_lineage (feature_id, created_at)
);
```

**Implementation:**

```python
"""
feature_governance.py
Feature governance system with registry, versioning, and lineage.
"""

import sqlite3
import json
import uuid
from datetime import datetime, UTC
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class FeatureStatus(Enum):
    EXPERIMENTAL = "experimental"
    VALIDATED = "validated"
    DEPRECATED = "deprecated"

class FeatureType(Enum):
    CONTINUITY = "continuity"
    STABILITY = "stability"
    TEMPORAL = "temporal"
    DERIVED = "derived"

@dataclass(frozen=True)
class FeatureDefinition:
    """Immutable feature definition."""
    feature_id: str
    feature_name: str
    feature_type: FeatureType
    data_type: str
    description: str
    causal_hypothesis: str
    scientific_risk: str
    mathematical_definition: str
    status: FeatureStatus
    version: int
    schema_version: str
    created_at: str
    created_by: str

@dataclass(frozen=True)
class FeatureVersion:
    """Immutable feature version."""
    feature_id: str
    version: int
    schema_version: str
    computation_version: str
    changed_at: str
    changed_by: str
    change_description: str

@dataclass(frozen=True)
class FeatureLineage:
    """Immutable feature lineage record."""
    feature_id: str
    source_snapshot_id: int
    transformation_type: str
    parameters: Dict
    output_feature_id: Optional[str]
    created_at: str

class FeatureRegistry:
    """
    Feature registry with versioning and lineage tracking.
    """
    
    def __init__(self, db_path: str = "data/feature_governance.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize feature governance schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_registry (
                feature_id TEXT PRIMARY KEY,
                feature_name TEXT NOT NULL UNIQUE,
                feature_type TEXT NOT NULL,
                data_type TEXT NOT NULL,
                description TEXT,
                causal_hypothesis TEXT,
                scientific_risk TEXT,
                mathematical_definition TEXT,
                created_at TEXT NOT NULL,
                created_by TEXT NOT NULL,
                status TEXT NOT NULL,
                version INTEGER NOT NULL,
                schema_version TEXT NOT NULL,
                INDEX idx_feature_name (feature_name),
                INDEX idx_status (status)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_version_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                schema_version TEXT NOT NULL,
                computation_version TEXT NOT NULL,
                changed_at TEXT NOT NULL,
                changed_by TEXT NOT NULL,
                change_description TEXT,
                FOREIGN KEY (feature_id) REFERENCES feature_registry(feature_id),
                INDEX idx_feature_version (feature_id, version)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_lineage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_id TEXT NOT NULL,
                source_snapshot_id INTEGER NOT NULL,
                transformation_type TEXT NOT NULL,
                parameters_json TEXT,
                output_feature_id TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (feature_id) REFERENCES feature_registry(feature_id),
                INDEX idx_feature_lineage (feature_id, created_at)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def register_feature(self, definition: FeatureDefinition) -> None:
        """
        Register a new feature in the registry.
        
        Args:
            definition: Feature definition
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO feature_registry 
                (feature_id, feature_name, feature_type, data_type,
                 description, causal_hypothesis, scientific_risk,
                 mathematical_definition, created_at, created_by,
                 status, version, schema_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                definition.feature_id,
                definition.feature_name,
                definition.feature_type.value,
                definition.data_type,
                definition.description,
                definition.causal_hypothesis,
                definition.scientific_risk,
                definition.mathematical_definition,
                definition.created_at,
                definition.created_by,
                definition.status.value,
                definition.version,
                definition.schema_version
            ))
            
            conn.commit()
            
        except sqlite3.IntegrityError:
            conn.rollback()
            raise ValueError(f"Feature {definition.feature_name} already exists")
        finally:
            conn.close()
    
    def update_feature_status(self, feature_id: str, 
                            new_status: FeatureStatus,
                            changed_by: str) -> None:
        """
        Update feature status.
        
        Args:
            feature_id: Feature identifier
            new_status: New status
            changed_by: User making change
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current version
        cursor.execute("SELECT version FROM feature_registry WHERE feature_id = ?", 
                      (feature_id,))
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            raise ValueError(f"Feature {feature_id} not found")
        
        current_version = result[0]
        new_version = current_version + 1
        
        # Update status and version
        cursor.execute("""
            UPDATE feature_registry
            SET status = ?, version = ?
            WHERE feature_id = ?
        """, (new_status.value, new_version, feature_id))
        
        # Record version history
        cursor.execute("""
            INSERT INTO feature_version_history
            (feature_id, version, schema_version, computation_version,
             changed_at, changed_by, change_description)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            feature_id,
            new_version,
            "1.0",  # Current schema version
            "1.0",  # Current computation version
            datetime.utcnow().isoformat(),
            changed_by,
            f"Status changed to {new_status.value}"
        ))
        
        conn.commit()
        conn.close()
    
    def record_lineage(self, lineage: FeatureLineage) -> None:
        """
        Record feature lineage for auditability.
        
        Args:
            lineage: Feature lineage record
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO feature_lineage
            (feature_id, source_snapshot_id, transformation_type,
             parameters_json, output_feature_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            lineage.feature_id,
            lineage.source_snapshot_id,
            lineage.transformation_type,
            json.dumps(lineage.parameters),
            lineage.output_feature_id,
            lineage.created_at
        ))
        
        conn.commit()
        conn.close()
    
    def get_feature_definition(self, feature_name: str) -> Optional[FeatureDefinition]:
        """
        Get feature definition by name.
        
        Args:
            feature_name: Feature name
        
        Returns:
            FeatureDefinition or None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT feature_id, feature_name, feature_type, data_type,
                   description, causal_hypothesis, scientific_risk,
                   mathematical_definition, created_at, created_by,
                   status, version, schema_version
            FROM feature_registry
            WHERE feature_name = ?
        """, (feature_name,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return FeatureDefinition(
            feature_id=row[0],
            feature_name=row[1],
            feature_type=FeatureType(row[2]),
            data_type=row[3],
            description=row[4],
            causal_hypothesis=row[5],
            scientific_risk=row[6],
            mathematical_definition=row[7],
            created_at=row[8],
            created_by=row[9],
            status=FeatureStatus(row[10]),
            version=row[11],
            schema_version=row[12]
        )
    
    def get_validated_features(self) -> List[FeatureDefinition]:
        """
        Get all validated features.
        
        Returns:
            List of validated feature definitions
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT feature_id, feature_name, feature_type, data_type,
                   description, causal_hypothesis, scientific_risk,
                   mathematical_definition, created_at, created_by,
                   status, version, schema_version
            FROM feature_registry
            WHERE status = 'validated'
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            FeatureDefinition(
                feature_id=row[0],
                feature_name=row[1],
                feature_type=FeatureType(row[2]),
                data_type=row[3],
                description=row[4],
                causal_hypothesis=row[5],
                scientific_risk=row[6],
                mathematical_definition=row[7],
                created_at=row[8],
                created_by=row[9],
                status=FeatureStatus(row[10]),
                version=row[11],
                schema_version=row[12]
            )
            for row in rows
        ]
```

### 3.2 Feature Contracts

**Implementation:**

```python
"""
feature_contracts.py
Feature contracts for runtime validation.
"""

from typing import Protocol, runtime_checkable, Any, Dict
from dataclasses import dataclass
from datetime import datetime
import numpy as np

@runtime_checkable
class FeatureContract(Protocol):
    """Protocol for feature contracts."""
    
    def validate(self, value: Any) -> bool:
        """Validate feature value."""
        ...
    
    def get_schema(self) -> Dict:
        """Get feature schema."""
        ...

@dataclass(frozen=True)
class ContinuityFeatureContract:
    """Contract for continuity features."""
    
    min_value: float = 0.0
    max_value: float = 1.0
    nullable: bool = False
    
    def validate(self, value: Any) -> bool:
        """Validate continuity feature value."""
        if value is None and self.nullable:
            return True
        
        if not isinstance(value, (int, float)):
            return False
        
        return self.min_value <= value <= self.max_value
    
    def get_schema(self) -> Dict:
        """Get feature schema."""
        return {
            'type': 'float',
            'min': self.min_value,
            'max': self.max_value,
            'nullable': self.nullable
        }

@dataclass(frozen=True)
class CountFeatureContract:
    """Contract for count features (e.g., squad size)."""
    
    min_value: int = 0
    max_value: int = 50
    nullable: bool = False
    
    def validate(self, value: Any) -> bool:
        """Validate count feature value."""
        if value is None and self.nullable:
            return True
        
        if not isinstance(value, (int, float)):
            return False
        
        return self.min_value <= value <= self.max_value
    
    def get_schema(self) -> Dict:
        """Get feature schema."""
        return {
            'type': 'int',
            'min': self.min_value,
            'max': self.max_value,
            'nullable': self.nullable
        }

class FeatureContractValidator:
    """Validate features against contracts."""
    
    def __init__(self):
        self.contracts: Dict[str, FeatureContract] = {}
        self._register_default_contracts()
    
    def _register_default_contracts(self) -> None:
        """Register default feature contracts."""
        self.contracts['continuity_index'] = ContinuityFeatureContract()
        self.contracts['continuity_defense'] = ContinuityFeatureContract()
        self.contracts['continuity_midfield'] = ContinuityFeatureContract()
        self.contracts['continuity_forward'] = ContinuityFeatureContract()
        self.contracts['squad_size'] = CountFeatureContract()
        self.contracts['squad_size_delta'] = CountFeatureContract(min_value=-50, max_value=50)
    
    def register_contract(self, feature_name: str, 
                         contract: FeatureContract) -> None:
        """
        Register feature contract.
        
        Args:
            feature_name: Name of feature
            contract: Feature contract
        """
        self.contracts[feature_name] = contract
    
    def validate_feature(self, feature_name: str, value: Any) -> bool:
        """
        Validate feature value against contract.
        
        Args:
            feature_name: Name of feature
            value: Feature value
        
        Returns:
            True if valid, False otherwise
        """
        if feature_name not in self.contracts:
            # No contract registered → accept
            return True
        
        contract = self.contracts[feature_name]
        return contract.validate(value)
    
    def validate_features(self, features: Dict[str, Any]) -> Dict[str, bool]:
        """
        Validate multiple features.
        
        Args:
            features: Dict of feature names to values
        
        Returns:
            Dict of validation results
        """
        return {
            feature_name: self.validate_feature(feature_name, value)
            for feature_name, value in features.items()
        }
```

## 4. STORAGE OPTIMIZATION

### 4.1 Parquet Partitioning Strategy

**Strategy:**

```
data/feature_store/
├── squad_snapshots/
│   ├── team_id=Mexico/
│   │   ├── year=2024/
│   │   │   ├── month=01/
│   │   │   │   ├── part-00000.parquet
│   │   │   │   └── part-00001.parquet
│   │   └── month=02/
│   └── team_id=Argentina/
├── match_features/
│   ├── year=2024/
│   │   ├── month=01/
│   │   │   └── day=01/
│   └── year=2024/
└── temporal_features/
    ├── feature_name=continuity_index/
    │   ├── year=2024/
    │   └── year=2025/
    └── feature_name=squad_size/
```

**Rationale:**
- `team_id`: Most common filter in queries
- `year/month`: Temporal pruning for rolling windows
- `feature_name`: Feature-specific queries
- `day`: Fine-grained temporal queries

**Implementation:**

```python
"""
storage_optimization.py
Storage optimization for Parquet + DuckDB.
"""

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Optional, List
from datetime import datetime

class ParquetPartitioner:
    """
    Intelligent Parquet partitioning for temporal features.
    """
    
    def __init__(self, base_path: str = "data/feature_store"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def partition_squad_snapshots(self, df: pd.DataFrame) -> None:
        """
        Partition squad snapshots by team_id, year, month.
        
        Args:
            df: DataFrame with squad snapshots
        """
        # Add partition columns
        df['year'] = pd.to_datetime(df['fetch_timestamp_utc']).dt.year
        df['month'] = pd.to_datetime(df['fetch_timestamp_utc']).dt.month
        
        # Convert to PyArrow
        table = pa.Table.from_pandas(df)
        
        # Write with partitioning
        output_path = self.base_path / "squad_snapshots"
        
        pq.write_to_dataset(
            table,
            root_path=str(output_path),
            partition_cols=['team_id', 'year', 'month'],
            compression='snappy',
            row_group_size=100000  # Optimize for reads
        )
    
    def partition_match_features(self, df: pd.DataFrame) -> None:
        """
        Partition match features by year, month, day.
        
        Args:
            df: DataFrame with match features
        """
        df['year'] = pd.to_datetime(df['kickoff_timestamp']).dt.year
        df['month'] = pd.to_datetime(df['kickoff_timestamp']).dt.month
        df['day'] = pd.to_datetime(df['kickoff_timestamp']).dt.day
        
        table = pa.Table.from_pandas(df)
        
        output_path = self.base_path / "match_features"
        
        pq.write_to_dataset(
            table,
            root_path=str(output_path),
            partition_cols=['year', 'month', 'day'],
            compression='snappy',
            row_group_size=100000
        )
    
    def partition_temporal_features(self, df: pd.DataFrame) -> None:
        """
        Partition temporal features by feature_name, year.
        
        Args:
            df: DataFrame with temporal features
        """
        df['year'] = pd.to_datetime(df['computation_timestamp']).dt.year
        
        table = pa.Table.from_pandas(df)
        
        output_path = self.base_path / "temporal_features"
        
        pq.write_to_dataset(
            table,
            root_path=str(output_path),
            partition_cols=['feature_name', 'year'],
            compression='snappy',
            row_group_size=100000
        )

class DuckDBOptimizer:
    """
    Optimize DuckDB queries for temporal feature store.
    """
    
    def __init__(self, parquet_path: str = "data/feature_store"):
        self.parquet_path = parquet_path
        self.con = duckdb.connect(database=":memory:")
    
    def create_optimized_views(self) -> None:
        """
        Create optimized DuckDB views for common queries.
        """
        # View for latest squad snapshots per team
        self.con.execute("""
            CREATE OR REPLACE VIEW latest_squad_snapshots AS
            WITH ranked_snapshots AS (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY team_id 
                           ORDER BY fetch_timestamp_utc DESC
                       ) as rn
                FROM read_parquet('{}/squad_snapshots/**/*.parquet')
            )
            SELECT * FROM ranked_snapshots WHERE rn = 1
        """.format(self.parquet_path))
        
        # View for temporal features with anti-leakage validation
        self.con.execute("""
            CREATE OR REPLACE VIEW validated_temporal_features AS
            SELECT 
                f.*,
                s.fetch_timestamp_utc as squad_snapshot_timestamp,
                f.computation_timestamp < m.kickoff_timestamp as anti_leakage_valid
            FROM read_parquet('{}/match_features/**/*.parquet') m
            JOIN read_parquet('{}/temporal_features/**/*.parquet') f
                ON m.match_id = f.match_id
            JOIN read_parquet('{}/squad_snapshots/**/*.parquet') s
                ON f.source_snapshot_id = s.id
            WHERE f.computation_timestamp < m.kickoff_timestamp
        """.format(self.parquet_path, self.parquet_path, self.parquet_path))
    
    def optimize_query(self, query: str) -> duckdb.DuckDBPyRelation:
        """
        Execute optimized query with pruning.
        
        Args:
            query: SQL query
        
        Returns:
            DuckDB relation
        """
        return self.con.execute(query)

class SnapshotCompactor:
    """
    Compact old snapshots to reduce storage.
    """
    
    def __init__(self, retention_days: int = 365):
        self.retention_days = retention_days
    
    def compact_old_snapshots(self, parquet_path: str) -> None:
        """
        Compact snapshots older than retention period.
        
        Args:
            parquet_path: Path to Parquet files
        """
        cutoff_date = datetime.utcnow() - pd.Timedelta(days=self.retention_days)
        
        # Read old snapshots
        con = duckdb.connect()
        
        old_snapshots = con.execute(f"""
            SELECT * FROM read_parquet('{parquet_path}/squad_snapshots/**/*.parquet')
            WHERE fetch_timestamp_utc < '{cutoff_date.isoformat()}'
        """).df()
        
        if old_snapshots.empty:
            return
        
        # Compact to single file per team
        for team_id in old_snapshots['team_id'].unique():
            team_snapshots = old_snapshots[old_snapshots['team_id'] == team_id]
            
            # Write compacted file
            output_path = f"{parquet_path}/squad_snapshots_compacted/{team_id}.parquet"
            team_snapshots.to_parquet(output_path, compression='snappy')
        
        # Delete old partitioned files
        # (Implementation depends on storage backend)
```

## 5. OPERATIONAL RESILIENCE

### 5.1 API Failure Handling

**Implementation:**

```python
"""
operational_resilience.py
Operational resilience patterns for API failures and degraded mode.
"""

import time
import logging
from typing import Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
from functools import wraps
import requests
from datetime import datetime

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit tripped
    HALF_OPEN = "half_open"  # Testing recovery

@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    expected_exception: Exception = requests.RequestException

class CircuitBreaker:
    """
    Circuit breaker pattern for API resilience.
    
    Prevents cascading failures by tripping after repeated failures.
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.logger = logging.getLogger(__name__)
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Function result
        
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        if self.last_failure_time is None:
            return True
        
        elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
        return elapsed >= self.config.recovery_timeout
    
    def _on_success(self) -> None:
        """Handle successful call."""
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.logger.info("Circuit breaker reset to CLOSED")
    
    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self.logger.warning(f"Circuit breaker tripped to OPEN after {self.failure_count} failures")

class RetryWithBackoff:
    """
    Retry with exponential backoff.
    """
    
    def __init__(self, max_retries: int = 3, 
                 base_delay: float = 1.0,
                 max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator for retry with backoff."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == self.max_retries:
                        self.logger.error(f"Max retries ({self.max_retries}) exceeded")
                        raise
                    
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    self.logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    time.sleep(delay)
            
            raise last_exception  # Should never reach here
        
        return wrapper

class DegradedModeManager:
    """
    Manage degraded mode when primary systems fail.
    
    Falls back to cached data or simplified computations.
    """
    
    def __init__(self, cache_ttl_hours: float = 24.0):
        self.cache_ttl_hours = cache_ttl_hours
        self.cache: Dict[str, tuple] = {}  # key -> (value, timestamp)
        self.logger = logging.getLogger(__name__)
    
    def get_cached(self, key: str) -> Optional[Any]:
        """
        Get cached value if not expired.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if expired/missing
        """
        if key not in self.cache:
            return None
        
        value, timestamp = self.cache[key]
        age_hours = (datetime.utcnow() - timestamp).total_seconds() / 3600
        
        if age_hours > self.cache_ttl_hours:
            self.logger.warning(f"Cache entry {key} expired ({age_hours:.1f}h old)")
            del self.cache[key]
            return None
        
        return value
    
    def set_cached(self, key: str, value: Any) -> None:
        """
        Set cached value.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        self.cache[key] = (value, datetime.utcnow())
    
    def execute_with_fallback(self, 
                            primary_func: Callable,
                            fallback_func: Callable,
                            *args, **kwargs) -> Any:
        """
        Execute primary function with fallback to degraded mode.
        
        Args:
            primary_func: Primary function
            fallback_func: Fallback function
            *args: Arguments
            **kwargs: Keyword arguments
        
        Returns:
            Result from primary or fallback function
        """
        try:
            return primary_func(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Primary function failed: {e}, using fallback")
            return fallback_func(*args, **kwargs)

class TimestampValidator:
    """
    Validate timestamps for corruption detection.
    """
    
    @staticmethod
    def validate_timestamp(timestamp_str: str) -> bool:
        """
        Validate timestamp format and reasonable range.
        
        Args:
            timestamp_str: Timestamp string
        
        Returns:
            True if valid, False otherwise
        """
        try:
            dt = datetime.fromisoformat(timestamp_str)
            
            # Reasonable range: 2010-2030
            if dt.year < 2010 or dt.year > 2030:
                return False
            
            return True
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_timestamp_order(timestamps: List[str]) -> bool:
        """
        Validate timestamps are in non-decreasing order.
        
        Args:
            timestamps: List of timestamp strings
        
        Returns:
            True if valid, False otherwise
        """
        try:
            dts = [datetime.fromisoformat(ts) for ts in timestamps]
            return all(dts[i] <= dts[i+1] for i in range(len(dts)-1))
        except (ValueError, TypeError):
            return False
```

## 6. PRODUCTION ML VALIDATION

### 6.1 Online vs Offline Skew Detection

**Mathematical Definition:**

Feature Distribution Skew:
```
skew = KL(P_online || P_offline)
```

Where KL is Kullback-Leibler divergence.

**Implementation:**

```python
"""
production_ml_validation.py
Production ML validation for online/offline skew and calibration drift.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass
from scipy import stats
from scipy.stats import entropy

@dataclass(frozen=True)
class SkewMetric:
    """Immutable skew metric."""
    feature_name: str
    kl_divergence: float
    js_divergence: float
    ks_statistic: float
    ks_p_value: float
    is_significant: bool
    threshold: float
    timestamp: str

@dataclass(frozen=True)
class CalibrationDriftMetric:
    """Immutable calibration drift metric."""
    slope: float
    intercept: float
    slope_drift: float
    intercept_drift: float
    is_drifted: bool
    slope_threshold: float = 0.1
    intercept_threshold: float = 0.1

class OnlineOfflineValidator:
    """
    Validate online vs offline feature skew.
    
    Detects:
    - Distribution shift
    - Feature availability differences
    - Statistical differences
    """
    
    def __init__(self, skew_threshold: float = 0.1):
        self.skew_threshold = skew_threshold
    
    def calculate_kl_divergence(self, p: np.ndarray, q: np.ndarray, 
                                bins: int = 10) -> float:
        """
        Calculate Kullback-Leibler divergence.
        
        KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
        
        Args:
            p: Online distribution
            q: Offline distribution
            bins: Number of bins for histogram
        
        Returns:
            KL divergence
        """
        # Create histograms
        p_hist, _ = np.histogram(p, bins=bins, density=True)
        q_hist, _ = np.histogram(q, bins=bins, density=True)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p_hist = p_hist + epsilon
        q_hist = q_hist + epsilon
        
        # Normalize
        p_hist = p_hist / p_hist.sum()
        q_hist = q_hist / q_hist.sum()
        
        # Calculate KL divergence
        kl_div = np.sum(p_hist * np.log(p_hist / q_hist))
        
        return kl_div
    
    def calculate_js_divergence(self, p: np.ndarray, q: np.ndarray,
                                bins: int = 10) -> float:
        """
        Calculate Jensen-Shannon divergence.
        
        JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
        where M = 0.5 * (P + Q)
        
        Args:
            p: Online distribution
            q: Offline distribution
            bins: Number of bins for histogram
        
        Returns:
            JS divergence
        """
        m = 0.5 * (p + q)
        
        kl_pm = self.calculate_kl_divergence(p, m, bins)
        kl_qm = self.calculate_kl_divergence(q, m, bins)
        
        js_div = 0.5 * (kl_pm + kl_qm)
        
        return js_div
    
    def detect_skew(self, feature_name: str,
                   online_values: np.ndarray,
                   offline_values: np.ndarray) -> SkewMetric:
        """
        Detect online/offline skew for a feature.
        
        Args:
            feature_name: Name of feature
            online_values: Online feature values
            offline_values: Offline feature values
        
        Returns:
            SkewMetric with skew analysis
        """
        # Calculate divergences
        kl_div = self.calculate_kl_divergence(online_values, offline_values)
        js_div = self.calculate_js_divergence(online_values, offline_values)
        
        # KS test
        ks_stat, ks_p_value = stats.ks_2samp(online_values, offline_values)
        
        # Determine significance
        is_significant = js_div > self.skew_threshold
        
        return SkewMetric(
            feature_name=feature_name,
            kl_divergence=kl_div,
            js_divergence=js_div,
            ks_statistic=ks_stat,
            ks_p_value=ks_p_value,
            is_significant=is_significant,
            threshold=self.skew_threshold,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def check_feature_availability(self, online_features: Dict[str, np.ndarray],
                                  offline_features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Check feature availability ratio.
        
        Args:
            online_features: Online feature dict
            offline_features: Offline feature dict
        
        Returns:
            Dict of feature availability ratios
        """
        availability = {}
        
        for feature_name in offline_features.keys():
            if feature_name in online_features:
                online_count = len(online_features[feature_name])
                offline_count = len(offline_features[feature_name])
                ratio = online_count / offline_count if offline_count > 0 else 0.0
                availability[feature_name] = ratio
            else:
                availability[feature_name] = 0.0
        
        return availability

class CalibrationDriftDetector:
    """
    Detect calibration drift in production.
    
    Monitors:
    - Calibration slope
    - Calibration intercept
    - Brier score
    - Reliability diagram
    """
    
    def __init__(self, slope_threshold: float = 0.1,
                 intercept_threshold: float = 0.1):
        self.slope_threshold = slope_threshold
        self.intercept_threshold = intercept_threshold
        self.baseline_slope: Optional[float] = None
        self.baseline_intercept: Optional[float] = None
    
    def set_baseline(self, slope: float, intercept: float) -> None:
        """
        Set baseline calibration parameters.
        
        Args:
            slope: Baseline calibration slope
            intercept: Baseline calibration intercept
        """
        self.baseline_slope = slope
        self.baseline_intercept = intercept
    
    def detect_drift(self, current_slope: float,
                    current_intercept: float) -> CalibrationDriftMetric:
        """
        Detect calibration drift.
        
        Args:
            current_slope: Current calibration slope
            current_intercept: Current calibration intercept
        
        Returns:
            CalibrationDriftMetric with drift analysis
        """
        if self.baseline_slope is None or self.baseline_intercept is None:
            raise ValueError("Baseline not set")
        
        slope_drift = abs(current_slope - self.baseline_slope)
        intercept_drift = abs(current_intercept - self.baseline_intercept)
        
        is_drifted = (slope_drift > self.slope_threshold or 
                     intercept_drift > self.intercept_threshold)
        
        return CalibrationDriftMetric(
            slope=current_slope,
            intercept=current_intercept,
            slope_drift=slope_drift,
            intercept_drift=intercept_drift,
            is_drifted=is_drifted,
            slope_threshold=self.slope_threshold,
            intercept_threshold=self.intercept_threshold
        )

class UpliftPersistenceValidator:
    """
    Validate that uplift persists in production.
    
    Detects:
    - Uplift degradation
    - Uplift reversal
    - Uplift instability
    """
    
    def __init__(self, uplift_threshold: float = 0.005):
        self.uplift_threshold = uplift_threshold
        self.historical_uplifts: List[float] = []
    
    def record_uplift(self, uplift: float) -> None:
        """
        Record uplift measurement.
        
        Args:
            uplift: Uplift value
        """
        self.historical_uplifts.append(uplift)
    
    def check_persistence(self) -> Dict:
        """
        Check if uplift persists over time.
        
        Returns:
            Dict with persistence analysis
        """
        if len(self.historical_uplifts) < 2:
            return {'status': 'insufficient_data'}
        
        recent_uplifts = self.historical_uplifts[-10:]  # Last 10 measurements
        mean_uplift = np.mean(recent_uplifts)
        std_uplift = np.std(recent_uplifts)
        
        # Check if uplift is consistently above threshold
        is_persistent = mean_uplift > self.uplift_threshold
        
        # Check for degradation
        if len(self.historical_uplifts) >= 20:
            early_uplifts = self.historical_uplifts[:10]
            recent_uplifts = self.historical_uplifts[-10:]
            
            early_mean = np.mean(early_uplifts)
            recent_mean = np.mean(recent_uplifts)
            
            is_degraded = recent_mean < early_mean * 0.9  # 10% degradation
        else:
            is_degraded = False
        
        return {
            'status': 'persistent' if is_persistent else 'degraded',
            'mean_uplift': mean_uplift,
            'std_uplift': std_uplift,
            'is_degraded': is_degraded,
            'threshold': self.uplift_threshold
        }

class RollbackCriteria:
    """
    Define criteria for automatic rollback.
    
    Rollback if ANY of:
    - Calibration slope outside [0.9, 1.1]
    - Brier score degrades > 0.01
    - Feature availability < 0.8
    - Significant skew detected
    - Uplift becomes negative
    """
    
    @staticmethod
    def should_rollback(calibration_slope: float,
                       brier_degradation: float,
                       feature_availability: float,
                       skew_significant: bool,
                       uplift: float) -> Tuple[bool, List[str]]:
        """
        Determine if rollback should be triggered.
        
        Args:
            calibration_slope: Current calibration slope
            brier_degradation: Brier score degradation
            feature_availability: Feature availability ratio
            skew_significant: Whether skew is significant
            uplift: Current uplift
        
        Returns:
            (should_rollback, reasons)
        """
        reasons = []
        
        if not (0.9 <= calibration_slope <= 1.1):
            reasons.append(f"Calibration slope {calibration_slope:.3f} outside [0.9, 1.1]")
        
        if brier_degradation > 0.01:
            reasons.append(f"Brier degradation {brier_degradation:.4f} > 0.01")
        
        if feature_availability < 0.8:
            reasons.append(f"Feature availability {feature_availability:.2f} < 0.8")
        
        if skew_significant:
            reasons.append("Significant online/offline skew detected")
        
        if uplift < 0:
            reasons.append(f"Uplift negative: {uplift:.4f}")
        
        return (len(reasons) > 0, reasons)
```

## 7. TESTING STRATEGY

### 7.1 Property-Based Testing

**Implementation:**

```python
"""
property_based_tests.py
Property-based tests for temporal features.
"""

import pytest
import numpy as np
from typing import List, Callable
from hypothesis import given, strategies as st
from hypothesis.strategies import lists, floats, integers, text
from datetime import datetime, timedelta
from temporal_features import TemporalFeatureExtractor
from anti_leakage_validator import AntiLeakageValidator

class TestTemporalFeatureProperties:
    """Property-based tests for temporal features."""
    
    @given(lists(text(min_size=1, max_size=50), min_size=5, max_size=50),
           lists(text(min_size=1, max_size=50), min_size=5, max_size=50))
    def test_jaccard_symmetry(self, set_a: List[str], set_b: List[str]) -> None:
        """
        Property: Jaccard similarity is symmetric.
        
        J(A, B) = J(B, A)
        """
        extractor = TemporalFeatureExtractor()
        
        sim_ab = extractor.jaccard_similarity(set(set_a), set(set_b))
        sim_ba = extractor.jaccard_similarity(set(set_b), set(set_a))
        
        assert abs(sim_ab - sim_ba) < 1e-10
    
    @given(lists(text(min_size=1, max_size=50), min_size=5, max_size=50))
    def test_jaccard_identity(self, set_a: List[str]) -> None:
        """
        Property: Jaccard similarity of set with itself is 1.0.
        
        J(A, A) = 1.0
        """
        extractor = TemporalFeatureExtractor()
        
        sim = extractor.jaccard_similarity(set(set_a), set(set_a))
        
        assert sim == 1.0
    
    @given(lists(text(min_size=1, max_size=50), min_size=5, max_size=50),
           lists(text(min_size=1, max_size=50), min_size=5, max_size=50))
    def test_jaccard_range(self, set_a: List[str], set_b: List[str]) -> None:
        """
        Property: Jaccard similarity is always in [0.0, 1.0].
        
        0.0 ≤ J(A, B) ≤ 1.0
        """
        extractor = TemporalFeatureExtractor()
        
        sim = extractor.jaccard_similarity(set(set_a), set(set_b))
        
        assert 0.0 <= sim <= 1.0
    
    @given(lists(text(min_size=1, max_size=50), min_size=5, max_size=50))
    def test_jaccard_empty_sets(self, set_a: List[str]) -> None:
        """
        Property: Jaccard similarity with empty set is 0.0.
        
        J(A, ∅) = 0.0
        """
        extractor = TemporalFeatureExtractor()
        
        sim_empty = extractor.jaccard_similarity(set(set_a), set())
        
        assert sim_empty == 0.0
    
    @given(floats(min_value=0.0, max_value=1.0),
           floats(min_value=0.0, max_value=1.0))
    def test_feature_value_range(self, continuity_a: float, continuity_b: float) -> None:
        """
        Property: Continuity features are always in [0.0, 1.0].
        """
        assert 0.0 <= continuity_a <= 1.0
        assert 0.0 <= continuity_b <= 1.0

class TestAntiLeakageProperties:
    """Property-based tests for anti-leakage."""
    
    @given(text(min_size=1), text(min_size=1))
    def test_leakage_detection(self, feature_ts: str, kickoff_ts: str) -> None:
        """
        Property: If feature_timestamp >= kickoff_timestamp, leakage is detected.
        """
        validator = AntiLeakageValidator(strict_mode=False)
        
        # Only test if timestamps are valid
        try:
            feature_dt = datetime.fromisoformat(feature_ts)
            kickoff_dt = datetime.fromisoformat(kickoff_ts)
        except ValueError:
            return  # Skip invalid timestamps
        
        if feature_dt >= kickoff_dt:
            violations = validator.validate_feature_timestamp(
                feature_ts, kickoff_ts, "test_feature"
            )
            assert len(validator.violations) > 0
    
    @given(text(min_size=1))
    def test_timestamp_format_validation(self, timestamp: str) -> None:
        """
        Property: Invalid timestamp format raises ValueError.
        """
        validator = AntiLeakageValidator(strict_mode=False)
        
        try:
            datetime.fromisoformat(timestamp)
            # Valid timestamp, should not raise
            pass
        except ValueError:
            # Invalid timestamp, should raise in validation
            with pytest.raises(ValueError):
                validator.validate_feature_timestamp(
                    timestamp, "2024-01-01T15:00:00", "test_feature"
                )

class TestTemporalConsistencyProperties:
    """Property-based tests for temporal consistency."""
    
    @given(lists(integers(min_value=0, max_value=1000000), min_size=2, max_size=10))
    def test_timestamp_ordering(self, timestamps: List[int]) -> None:
        """
        Property: Sorted timestamps are non-decreasing.
        """
        sorted_timestamps = sorted(timestamps)
        
        for i in range(len(sorted_timestamps) - 1):
            assert sorted_timestamps[i] <= sorted_timestamps[i + 1]
    
    @given(integers(min_value=0, max_value=1000000),
           integers(min_value=1, max_value=1000000))
    def test_temporal_gap_positive(self, start_ts: int, end_ts: int) -> None:
        """
        Property: Temporal gap is always non-negative.
        """
        if end_ts > start_ts:
            gap = end_ts - start_ts
            assert gap >= 0

class TestStatisticalInvariants:
    """Property-based tests for statistical invariants."""
    
    @given(lists(floats(min_value=0.0, max_value=1.0), min_size=10, max_size=100))
    def test_variance_non_negative(self, values: List[float]) -> None:
        """
        Property: Variance is always non-negative.
        
        Var(X) ≥ 0
        """
        variance = np.var(values)
        assert variance >= 0
    
    @given(lists(floats(min_value=0.0, max_value=1.0), min_size=10, max_size=100))
    def test_mean_in_range(self, values: List[float]) -> None:
        """
        Property: Mean is within range of values.
        
        min(X) ≤ mean(X) ≤ max(X)
        """
        mean_val = np.mean(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        assert min_val <= mean_val <= max_val
    
    @given(lists(floats(min_value=0.0, max_value=1.0), min_size=10, max_size=100))
    def test_standard_deviation_non_negative(self, values: List[float]) -> None:
        """
        Property: Standard deviation is always non-negative.
        
        std(X) ≥ 0
        """
        std_val = np.std(values)
        assert std_val >= 0
```

### 7.2 Reproducibility Tests

**Implementation:**

```python
"""
test_reproducibility.py
Tests for reproducibility of feature computation.
"""

import pytest
import numpy as np
import json
from datetime import datetime
from temporal_features import TemporalFeatureExtractor
from squad_persistence import SquadSnapshot, PlayerSnapshot, SquadStatus

class TestReproducibility:
    """Tests for reproducibility of feature computation."""
    
    def test_deterministic_jaccard(self):
        """
        Test: Jaccard similarity is deterministic.
        
        Same inputs → same outputs always.
        """
        extractor = TemporalFeatureExtractor()
        
        set_a = {"player1", "player2", "player3"}
        set_b = {"player2", "player3", "player4"}
        
        results = []
        for _ in range(10):
            results.append(extractor.jaccard_similarity(set_a, set_b))
        
        # All results should be identical
        assert all(r == results[0] for r in results)
    
    def test_deterministic_continuity_index(self):
        """
        Test: Continuity index computation is deterministic.
        """
        extractor = TemporalFeatureExtractor()
        
        current = {"player1", "player2", "player3"}
        previous = {"player2", "player3", "player4"}
        
        features = []
        for _ in range(10):
            feature = extractor.extract_continuity_index(
                current, previous,
                "2024-01-01T10:00:00",
                "2024-01-01T09:00:00"
            )
            features.append(feature.value)
        
        # All results should be identical
        assert all(f == features[0] for f in features)
    
    def test_snapshot_immutability(self):
        """
        Test: SquadSnapshot is immutable.
        """
        players = [
            PlayerSnapshot("player1", "Defensa", 5, "Club", 25)
        ]
        
        snapshot = SquadSnapshot(
            team_id="MEX",
            team_name="Mexico",
            fetch_timestamp_utc="2024-01-01T10:00:00",
            players=players,
            squad_status=SquadStatus.FINAL,
            announcement_date="2023-12-01",
            api_version="1.0",
            computation_version="1.0",
            extraction_timestamp="2024-01-01T10:00:00"
        )
        
        # Attempting to modify should fail
        with pytest.raises(AttributeError):
            snapshot.team_id = "ARG"
    
    def test_feature_provenance_tracking(self):
        """
        Test: Feature provenance is tracked completely.
        """
        from squad_persistence import FeatureProvenance
        
        provenance = FeatureProvenance(
            feature_name="continuity_index",
            feature_value=0.5,
            source_snapshot_id=1,
            computation_version="1.0",
            extraction_timestamp="2024-01-01T10:00:00"
        )
        
        # All fields should be present
        assert provenance.feature_name == "continuity_index"
        assert provenance.feature_value == 0.5
        assert provenance.source_snapshot_id == 1
        assert provenance.computation_version == "1.0"
        assert provenance.extraction_timestamp == "2024-01-01T10:00:00"
```

## 8. PRODUCTION DEPLOYMENT CHECKLIST

### Pre-Production Validation

**Must Complete:**

- [ ] All features pass statistical validation (Brier, LogLoss, calibration)
- [ ] All features pass temporal stability tests (KS test, PSI)
- [ ] All features pass anti-leakage validation (zero violations)
- [ ] All features have bootstrap confidence intervals
- [ ] All features have documented causal hypothesis
- [ ] All features have documented scientific risks
- [ ] Feature registry populated with all features
- [ ] Feature contracts defined for all features
- [ ] Property-based tests passing
- [ ] Reproducibility tests passing
- [ ] Anti-leakage tests passing
- [ ] Temporal consistency tests passing
- [ ] Online/offline skew validation passing
- [ ] Calibration drift detection configured
- [ ] Rollback criteria defined
- [ ] Monitoring alerts configured
- [ ] Degraded mode tested
- [ ] Circuit breaker tested
- [ ] Retry logic tested
- [ ] Storage optimization tested
- [ ] Parquet partitioning validated
- [ ] DuckDB query optimization validated

### Production Rollout

**Phased Rollout:**

**Phase 1: Shadow Mode (1 week)**
- Run feature pipeline in parallel with production
- Compare outputs without affecting predictions
- Monitor drift, skew, availability
- Validate calibration

**Phase 2: Canary (1 week)**
- Deploy to 10% of traffic
- Monitor uplift, calibration, latency
- Compare with baseline metrics
- Prepare rollback plan

**Phase 3: Full Rollout (if Phase 2 successful)**
- Deploy to 100% of traffic
- Continuous monitoring
- Automated rollback triggers

### Post-Production Monitoring

**Daily:**
- Feature drift metrics (PSI, KS)
- Feature freshness
- Missing ratios
- Calibration metrics
- Uplift metrics

**Weekly:**
- Temporal stability analysis
- Online/offline skew
- Feature availability trends
- Performance degradation analysis

**Monthly:**
- Full statistical validation
- Feature importance analysis
- Retrospective uplift analysis
- Feature governance review

## CRITICAL ASSESSMENT

### Features Likely to Fail

**Continuity Index:**
- **Risk:** High
- **Reason:** Weak causal hypothesis, likely noise
- **Recommendation:** Reject unless strong statistical evidence

**Positional Continuity:**
- **Risk:** Very High
- **Reason:** Even weaker than overall continuity, small subsets
- **Recommendation:** Reject

**Squad Size:**
- **Risk:** Very High
- **Reason:** Likely pure noise, no causal mechanism
- **Recommendation:** Reject

**Squad Size Delta:**
- **Risk:** Very High
- **Reason:** Same as squad size
- **Recommendation:** Reject

**Announcement Lead Time:**
- **Risk:** Moderate
- **Reason:** Moderately weak, confounding factors
- **Recommendation:** Test but expect low uplift

### Architecture Risks

**Dual Storage (SQLite + DuckDB):**
- **Risk:** Operational complexity
- **Mitigation:** Clear separation of concerns, monitoring

**Parquet Partitioning:**
- **Risk:** Small file problem if over-partitioned
- **Mitigation:** Limit partition depth, monitor file sizes

**Block Bootstrap:**
- **Risk:** Computational overhead
- **Mitigation:** Cache results, parallelize

### Final Recommendation

**Default to Rejection:**

Do not integrate any temporal squad features into production unless:

1. Statistical uplift is robust (Brier ≥ 0.005, LogLoss ≥ 0.01)
2. Temporal stability is proven (KS p-value > 0.05)
3. Calibration is maintained (slope 0.95-1.05)
4. Zero leakage violations
5. Bootstrap CI does not overlap zero
6. Uplift persists over time

**If any criterion fails:**

Reject the feature. The cost of complexity does not justify marginal or negative uplift.

**Principle:**

Features are experimental until proven otherwise. The burden of proof is on the feature, not the baseline.
