from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional
import json

class FeatureStatus(Enum):
    APPROVED = "APPROVED"
    EXPERIMENTAL = "EXPERIMENTAL"
    REJECTED = "REJECTED"
    DISABLED = "DISABLED"

@dataclass(frozen=True)
class FeatureMetadata:
    name: str
    status: FeatureStatus
    validation_date: datetime
    calibration_impact: float
    drift_risk: str
    bootstrap_ci: List[float]
    production_owner: str
    version: str = "1.0.0"

class FeatureRegistry:
    """
    Formal governance for squad features in production.
    Ensures only approved signals reach the live model.
    """
    
    def __init__(self):
        # Hardcoded state representing the Phase F certification
        self._registry: Dict[str, FeatureMetadata] = {
            "continuity_index": FeatureMetadata(
                name="continuity_index",
                status=FeatureStatus.APPROVED,
                validation_date=datetime(2026, 5, 28, tzinfo=timezone.utc),
                calibration_impact=0.0057,
                drift_risk="LOW",
                bootstrap_ci=[0.0051, 0.0063],
                production_owner="Principal_ML_Engineer"
            ),
            "defenders_continuity": FeatureMetadata(
                name="defenders_continuity",
                status=FeatureStatus.APPROVED,
                validation_date=datetime(2026, 5, 28, tzinfo=timezone.utc),
                calibration_impact=0.0021,
                drift_risk="MEDIUM",
                bootstrap_ci=[0.0015, 0.0028],
                production_owner="Principal_ML_Engineer"
            ),
            "midfielders_continuity": FeatureMetadata(
                name="midfielders_continuity",
                status=FeatureStatus.EXPERIMENTAL,
                validation_date=datetime(2026, 5, 28, tzinfo=timezone.utc),
                calibration_impact=0.0004,
                drift_risk="HIGH",
                bootstrap_ci=[-0.0001, 0.0010],
                production_owner="Principal_ML_Engineer"
            ),
            "forwards_continuity": FeatureMetadata(
                name="forwards_continuity",
                status=FeatureStatus.REJECTED,
                validation_date=datetime(2026, 5, 28, tzinfo=timezone.utc),
                calibration_impact=-0.0001,
                drift_risk="HIGH",
                bootstrap_ci=[-0.0005, 0.0002],
                production_owner="Principal_ML_Engineer"
            )
        }

    def is_approved(self, feature_name: str) -> bool:
        meta = self._registry.get(feature_name)
        return meta is not None and meta.status == FeatureStatus.APPROVED

    def get_approved_features(self) -> List[str]:
        return [name for name, meta in self._registry.items() if meta.status == FeatureStatus.APPROVED]

    def get_metadata(self, feature_name: str) -> Optional[FeatureMetadata]:
        return self._registry.get(feature_name)

if __name__ == "__main__":
    registry = FeatureRegistry()
    print(f"Approved for production: {registry.get_approved_features()}")
