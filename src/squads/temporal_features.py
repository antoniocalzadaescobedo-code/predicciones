import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Set
import math

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("temporal_features")

@dataclass(frozen=True)
class SquadTemporalFeatures:
    """
    Immutable representation of calculated squad features.
    Includes provenance to ensure reproducibility and auditability.
    """
    team_id: int
    feature_timestamp: datetime
    continuity_index: float
    defenders_continuity: float
    midfielders_continuity: float
    forwards_continuity: float
    squad_size: int
    squad_size_delta: int
    announcement_lead_hours: float
    source_snapshot_id: int
    feature_version: str = "1.0.0"

    def __post_init__(self):
        """Strict validation of feature values."""
        if self.team_id <= 0:
            raise ValueError("team_id must be > 0")
        
        # Continuity indices must be in [0, 1] or NaN if undefined
        for field_name in ['continuity_index', 'defenders_continuity', 'midfielders_continuity', 'forwards_continuity']:
            val = getattr(self, field_name)
            if not (0.0 <= val <= 1.0) and not math.isnan(val):
                raise ValueError(f"{field_name} must be between 0 and 1 (current: {val})")
        
        if self.feature_timestamp.tzinfo is None:
            raise ValueError("feature_timestamp must be timezone-aware UTC.")

class SquadTemporalExtractor:
    """
    Engine for computing mathematically defined squad features.
    Prevents temporal leakage and handles edge cases for international football.
    """

    @staticmethod
    def _get_player_set(players: List[Dict[str, Any]], position_filter: Optional[str] = None) -> Set[str]:
        """
        Extracts a unique identifier set for players. 
        Uses name + position for matching consistency.
        """
        if position_filter:
            return {f"{p['name']}_{p['position']}" for p in players if p.get('position') == position_filter}
        return {f"{p['name']}_{p['position']}" for p in players}

    @staticmethod
    def compute_jaccard_similarity(current_players: Set[str], previous_players: Set[str]) -> float:
        """
        C_t = |S_t ∩ S_{t-1}| / |S_t ∪ S_{t-1}|
        """
        if not current_players and not previous_players:
            return 1.0  # Stable state (both empty)
        
        intersection = len(current_players.intersection(previous_players))
        union = len(current_players.union(previous_players))
        
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def compute_announcement_lead_hours(announcement_ts: datetime, kickoff_ts: datetime) -> float:
        """
        L_t = (kickoff - announcement) in hours.
        Quantifies the freshness of the squad information.
        """
        if announcement_ts > kickoff_ts:
            logger.error(f"LEAKAGE DETECTED: Announcement ({announcement_ts}) is after Kickoff ({kickoff_ts})")
            raise ValueError("Anti-leakage violation: Announcement timestamp cannot be after kickoff.")
        
        delta = kickoff_ts - announcement_ts
        return delta.total_seconds() / 3600.0

    def build_features_from_snapshots(
        self,
        current_snapshot: Any,
        previous_snapshot: Optional[Any],
        kickoff_timestamp: datetime,
        snapshot_id: int
    ) -> SquadTemporalFeatures:
        """
        Main entry point for feature extraction.
        Enforces strict temporal consistency and positional logic.
        """
        if current_snapshot.announcement_timestamp_utc > kickoff_timestamp:
            raise ValueError("Temporal Leakage: current snapshot is from the future relative to kickoff.")

        curr_players = self._get_player_set(current_snapshot.players)
        prev_players = self._get_player_set(previous_snapshot.players) if previous_snapshot else set()

        continuity = self.compute_jaccard_similarity(curr_players, prev_players)
        
        def_curr = self._get_player_set(current_snapshot.players, "Defender")
        def_prev = self._get_player_set(previous_snapshot.players, "Defender") if previous_snapshot else set()
        def_cont = self.compute_jaccard_similarity(def_curr, def_prev)

        mid_curr = self._get_player_set(current_snapshot.players, "Midfielder")
        mid_prev = self._get_player_set(previous_snapshot.players, "Midfielder") if previous_snapshot else set()
        mid_cont = self.compute_jaccard_similarity(mid_curr, mid_prev)

        fwd_curr = self._get_player_set(current_snapshot.players, "Attacker")
        fwd_prev = self._get_player_set(previous_snapshot.players, "Attacker") if previous_snapshot else set()
        fwd_cont = self.compute_jaccard_similarity(fwd_curr, fwd_prev)

        squad_size = len(curr_players)
        prev_size = len(prev_players)
        squad_size_delta = squad_size - prev_size if previous_snapshot else 0

        lead_hours = self.compute_announcement_lead_hours(current_snapshot.announcement_timestamp_utc, kickoff_timestamp)

        return SquadTemporalFeatures(
            team_id=current_snapshot.team_id,
            feature_timestamp=current_snapshot.announcement_timestamp_utc,
            continuity_index=continuity,
            defenders_continuity=def_cont,
            midfielders_continuity=mid_cont,
            forwards_continuity=fwd_cont,
            squad_size=squad_size,
            squad_size_delta=squad_size_delta,
            announcement_lead_hours=lead_hours,
            source_snapshot_id=snapshot_id
        )

if __name__ == "__main__":
    logger.info("SquadTemporalExtractor module loaded.")
