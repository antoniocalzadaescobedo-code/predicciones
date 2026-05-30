import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from pathlib import Path
import uuid

from src.production.common import Severity

logger = logging.getLogger("incident_registry")

class IncidentType(Enum):
    PSI_DRIFT = "PSI_DRIFT"
    HIGH_FALLBACK_RATE = "HIGH_FALLBACK_RATE"
    CALIBRATION_DRIFT = "CALIBRATION_DRIFT"
    BRIER_DEGRADATION = "BRIER_DEGRADATION"
    NAN_PREDICTIONS = "NAN_PREDICTIONS"
    FEATURE_CORRUPTION = "FEATURE_CORRUPTION"
    TEMPORAL_LEAKAGE = "TEMPORAL_LEAKAGE"
    LATENCY_SPIKE = "LATENCY_SPIKE"
    API_FAILURE = "API_FAILURE"
    REGIME_CHANGE = "REGIME_CHANGE"

class RollbackStatus(Enum):
    NOT_TRIGGERED = "NOT_TRIGGERED"
    TRIGGERED = "TRIGGERED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

@dataclass
class Incident:
    """Represents a production incident."""
    incident_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    incident_type: IncidentType = IncidentType.FEATURE_CORRUPTION
    affected_feature: Optional[str] = None
    fallback_triggered: bool = False
    prediction_id: Optional[str] = None
    severity: Severity = Severity.WARNING
    rollback_status: RollbackStatus = RollbackStatus.NOT_TRIGGERED
    metric_value: float = 0.0
    threshold: float = 0.0
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None
    resolution_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert incident to dictionary for serialization."""
        data = asdict(self)
        data['incident_type'] = self.incident_type.value
        data['severity'] = self.severity.value
        data['rollback_status'] = self.rollback_status.value
        data['timestamp_utc'] = self.timestamp_utc.isoformat()
        if self.resolution_timestamp:
            data['resolution_timestamp'] = self.resolution_timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Incident':
        """Create incident from dictionary."""
        data['incident_type'] = IncidentType(data['incident_type'])
        data['severity'] = Severity(data['severity'])
        data['rollback_status'] = RollbackStatus(data['rollback_status'])
        data['timestamp_utc'] = datetime.fromisoformat(data['timestamp_utc'])
        if data.get('resolution_timestamp'):
            data['resolution_timestamp'] = datetime.fromisoformat(data['resolution_timestamp'])
        return cls(**data)

class IncidentRegistry:
    """
    Centralized incident tracking and management for production systems.
    Provides persistence, querying, and incident lifecycle management.
    """
    
    def __init__(self, storage_path: Path = Path("data/incidents.json")):
        self.storage_path = storage_path
        self.incidents: List[Incident] = []
        self._ensure_storage_directory()
        self._load_incidents()
        logger.info(f"IncidentRegistry initialized with {len(self.incidents)} historical incidents.")

    def _ensure_storage_directory(self):
        """Ensures storage directory exists."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_incidents(self):
        """Loads incidents from persistent storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.incidents = [Incident.from_dict(inc) for inc in data]
                logger.info(f"Loaded {len(self.incidents)} incidents from {self.storage_path}")
            except Exception as e:
                logger.error(f"Failed to load incidents: {e}")
                self.incidents = []

    def _save_incidents(self):
        """Saves incidents to persistent storage."""
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump([inc.to_dict() for inc in self.incidents], f, indent=2, default=str)
            logger.debug(f"Saved {len(self.incidents)} incidents to {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to save incidents: {e}")

    def register_incident(self, 
                         incident_type: IncidentType,
                         severity: Severity,
                         message: str,
                         metric_value: float = 0.0,
                         threshold: float = 0.0,
                         affected_feature: Optional[str] = None,
                         prediction_id: Optional[str] = None,
                         fallback_triggered: bool = False,
                         metadata: Optional[Dict[str, Any]] = None) -> Incident:
        """
        Registers a new incident.
        
        Args:
            incident_type: Type of incident
            severity: Severity level
            message: Human-readable description
            metric_value: Value that triggered the incident
            threshold: Threshold that was exceeded
            affected_feature: Feature that caused the incident (if applicable)
            prediction_id: ID of the prediction that triggered the incident
            fallback_triggered: Whether fallback was triggered
            metadata: Additional context
            
        Returns:
            The registered incident
        """
        incident = Incident(
            incident_type=incident_type,
            severity=severity,
            message=message,
            metric_value=metric_value,
            threshold=threshold,
            affected_feature=affected_feature,
            prediction_id=prediction_id,
            fallback_triggered=fallback_triggered,
            metadata=metadata or {}
        )
        
        self.incidents.append(incident)
        self._save_incidents()
        
        # Log based on severity
        log_method = {
            Severity.INFO: logger.info,
            Severity.WARNING: logger.warning,
            Severity.CRITICAL: logger.critical,
            Severity.EMERGENCY: logger.critical
        }.get(severity, logger.info)
        
        log_method(
            f"INCIDENT REGISTERED [{severity.value}] {incident_type.value}: {message} "
            f"(value={metric_value:.4f}, threshold={threshold:.4f})"
        )
        
        return incident

    def resolve_incident(self, incident_id: str, resolution_notes: str = "") -> Optional[Incident]:
        """
        Marks an incident as resolved.
        
        Args:
            incident_id: ID of the incident to resolve
            resolution_notes: Notes about the resolution
            
        Returns:
            The resolved incident, or None if not found
        """
        for incident in self.incidents:
            if incident.incident_id == incident_id:
                incident.resolved = True
                incident.resolution_timestamp = datetime.now(timezone.utc)
                incident.resolution_notes = resolution_notes
                self._save_incidents()
                logger.info(f"Incident {incident_id} resolved: {resolution_notes}")
                return incident
        logger.warning(f"Incident {incident_id} not found for resolution")
        return None

    def trigger_rollback(self, incident_id: str) -> Optional[Incident]:
        """
        Marks that a rollback was triggered for an incident.
        
        Args:
            incident_id: ID of the incident
            
        Returns:
            The updated incident, or None if not found
        """
        for incident in self.incidents:
            if incident.incident_id == incident_id:
                incident.rollback_status = RollbackStatus.TRIGGERED
                incident.fallback_triggered = True
                self._save_incidents()
                logger.critical(f"Rollback triggered for incident {incident_id}")
                return incident
        logger.warning(f"Incident {incident_id} not found for rollback")
        return None

    def complete_rollback(self, incident_id: str) -> Optional[Incident]:
        """
        Marks that a rollback was completed successfully.
        
        Args:
            incident_id: ID of the incident
            
        Returns:
            The updated incident, or None if not found
        """
        for incident in self.incidents:
            if incident.incident_id == incident_id:
                incident.rollback_status = RollbackStatus.COMPLETED
                self._save_incidents()
                logger.info(f"Rollback completed for incident {incident_id}")
                return incident
        logger.warning(f"Incident {incident_id} not found for rollback completion")
        return None

    def get_incidents_by_type(self, incident_type: IncidentType) -> List[Incident]:
        """Returns all incidents of a specific type."""
        return [inc for inc in self.incidents if inc.incident_type == incident_type]

    def get_incidents_by_severity(self, severity: Severity) -> List[Incident]:
        """Returns all incidents of a specific severity."""
        return [inc for inc in self.incidents if inc.severity == severity]

    def get_active_incidents(self) -> List[Incident]:
        """Returns all unresolved incidents."""
        return [inc for inc in self.incidents if not inc.resolved]

    def get_incidents_by_feature(self, feature_name: str) -> List[Incident]:
        """Returns all incidents related to a specific feature."""
        return [inc for inc in self.incidents if inc.affected_feature == feature_name]

    def get_incidents_in_window(self, 
                               start: datetime, 
                               end: datetime) -> List[Incident]:
        """Returns all incidents within a time window."""
        return [
            inc for inc in self.incidents 
            if start <= inc.timestamp_utc <= end
        ]

    def get_incident_summary(self) -> Dict[str, Any]:
        """Returns a summary of all incidents."""
        active = self.get_active_incidents()
        
        by_type = {}
        for inc_type in IncidentType:
            by_type[inc_type.value] = len(self.get_incidents_by_type(inc_type))
        
        by_severity = {}
        for sev in Severity:
            by_severity[sev.value] = len(self.get_incidents_by_severity(sev))
        
        return {
            "total_incidents": len(self.incidents),
            "active_incidents": len(active),
            "resolved_incidents": len(self.incidents) - len(active),
            "by_type": by_type,
            "by_severity": by_severity,
            "recent_incidents": [
                inc.to_dict() for inc in sorted(self.incidents, key=lambda x: x.timestamp_utc, reverse=True)[:10]
            ]
        }

    def get_rollback_rate(self, window_hours: int = 24) -> float:
        """
        Calculates the rollback rate in the given time window.
        
        Args:
            window_hours: Time window in hours
            
        Returns:
            Rollback rate as a percentage
        """
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=window_hours)
        
        incidents_in_window = self.get_incidents_in_window(start, end)
        rollback_incidents = [inc for inc in incidents_in_window if inc.rollback_status != RollbackStatus.NOT_TRIGGERED]
        
        if len(incidents_in_window) == 0:
            return 0.0
        
        return len(rollback_incidents) / len(incidents_in_window)

    def clear_old_incidents(self, days_to_keep: int = 30):
        """
        Removes incidents older than the specified number of days.
        
        Args:
            days_to_keep: Number of days to keep incidents
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        original_count = len(self.incidents)
        self.incidents = [inc for inc in self.incidents if inc.timestamp_utc > cutoff]
        removed = original_count - len(self.incidents)
        
        if removed > 0:
            self._save_incidents()
            logger.info(f"Cleared {removed} incidents older than {days_to_keep} days")

if __name__ == "__main__":
    # Test incident registry
    registry = IncidentRegistry()
    
    # Register a test incident
    incident = registry.register_incident(
        incident_type=IncidentType.PSI_DRIFT,
        severity=Severity.WARNING,
        message="PSI drift detected in continuity_index",
        metric_value=0.25,
        threshold=0.20,
        affected_feature="continuity_index"
    )
    
    print(f"Registered incident: {incident.incident_id}")
    print(f"Summary: {registry.get_incident_summary()}")
