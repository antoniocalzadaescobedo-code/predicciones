import requests
import logging
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from src.squads.squad_persistence import SquadSnapshot

logger = logging.getLogger("historical_fetcher")

class HistoricalSquadFetcher:
    """
    Defensive fetcher for international football squads.
    Handles rate limiting, retries, and data integrity.
    """
    
    def __init__(self, api_key: str, base_url: str = "https://v3.football.api-sports.io"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": "v3.football.api-sports.io"
        })

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60))
    def fetch_team_snapshots(self, team_id: int) -> List[Dict[str, Any]]:
        """
        Downloads all available snapshots for a team.
        """
        url = f"{self.base_url}/players/squads"
        params = {"team": team_id}
        
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        if not data.get("response"):
            logger.warning(f"No squad data found for team_id {team_id}")
            return []
            
        return data["response"]

    def validate_snapshot_integrity(self, raw_data: Dict[str, Any]) -> bool:
        """Validates that the snapshot contains required player data."""
        if not raw_data.get("players"):
            return False
        for p in raw_data["players"]:
            if not p.get("name") or not p.get("position"):
                return False
        return True

    def transform_to_snapshot(self, raw_response: Dict[str, Any], team_id: int, team_name: str, announcement_ts: datetime) -> SquadSnapshot:
        """Converts raw API response to auditable SquadSnapshot dataclass."""
        players = []
        for p in raw_response.get("players", []):
            players.append({
                "name": p["name"],
                "position": p["position"],
                "number": p.get("number", 0)
            })
            
        return SquadSnapshot(
            team_id=team_id,
            team_name=team_name,
            players=players,
            source="API-Football",
            announcement_timestamp_utc=announcement_ts,
            ingestion_timestamp_utc=datetime.now(timezone.utc)
        )
