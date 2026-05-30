import sqlite3
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Any

# Structured logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("squad_persistence")

@dataclass(frozen=True)
class SquadSnapshot:
    """
    Immutable representation of a team squad at a specific point in time.
    Ensures data integrity and auditability.
    """
    team_id: int
    team_name: str
    players: List[Dict[str, Any]]  # List of dicts with name, position, number
    source: str
    announcement_timestamp_utc: datetime
    ingestion_timestamp_utc: datetime

    def __post_init__(self):
        """Validate snapshot integrity upon creation."""
        if self.team_id <= 0:
            raise ValueError(f"Invalid team_id: {self.team_id}. Must be > 0.")
        if not self.team_name.strip():
            raise ValueError("team_name cannot be empty.")
        if not isinstance(self.players, list):
            raise ValueError("players must be a list.")
        if not self.source.strip():
            raise ValueError("source cannot be empty.")
        
        # Mandatory UTC validation
        self._validate_utc(self.announcement_timestamp_utc, "announcement_timestamp_utc")
        self._validate_utc(self.ingestion_timestamp_utc, "ingestion_timestamp_utc")

    @staticmethod
    def _validate_utc(dt: datetime, field_name: str):
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt).total_seconds() != 0:
            raise ValueError(f"{field_name} must be a timezone-aware UTC datetime.")

class SquadPersistence:
    """
    SQLite repository for auditable squad snapshots.
    Guarantees ACID transactions and anti-leakage temporal queries.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initializes the SQLite schema and indices."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS squad_snapshots (
                        snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        team_id INTEGER NOT NULL,
                        team_name TEXT NOT NULL,
                        announcement_timestamp_utc TEXT NOT NULL,
                        ingestion_timestamp_utc TEXT NOT NULL,
                        source TEXT NOT NULL,
                        players_json TEXT NOT NULL
                    )
                ''')
                # Indices for optimized temporal lookups
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_team_timestamp ON squad_snapshots (team_id, announcement_timestamp_utc)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_announcement_timestamp ON squad_snapshots (announcement_timestamp_utc)')
                conn.commit()
                logger.info(f"Database initialized at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def save_snapshot(self, snapshot: SquadSnapshot):
        """Persists a squad snapshot to the historical store."""
        try:
            players_json = json.dumps(snapshot.players)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO squad_snapshots (
                        team_id, team_name, announcement_timestamp_utc, 
                        ingestion_timestamp_utc, source, players_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    snapshot.team_id,
                    snapshot.team_name,
                    snapshot.announcement_timestamp_utc.isoformat(),
                    snapshot.ingestion_timestamp_utc.isoformat(),
                    snapshot.source,
                    players_json
                ))
                conn.commit()
                logger.info(f"Snapshot saved for team {snapshot.team_name} (ID: {snapshot.team_id}) at {snapshot.announcement_timestamp_utc}")
        except (sqlite3.Error, TypeError) as e:
            logger.error(f"Error saving snapshot: {e}")
            raise

    def get_latest_snapshot_before(self, team_id: int, before_timestamp: datetime) -> Optional[SquadSnapshot]:
        """
        Anti-leakage query: retrieves the most recent snapshot strictly before 
        the provided timestamp.
        """
        if before_timestamp.tzinfo is None:
            raise ValueError("before_timestamp must be timezone-aware UTC.")

        query = '''
            SELECT team_id, team_name, players_json, source, 
                   announcement_timestamp_utc, ingestion_timestamp_utc
            FROM squad_snapshots
            WHERE team_id = ? AND announcement_timestamp_utc < ?
            ORDER BY announcement_timestamp_utc DESC
            LIMIT 1
        '''
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, (team_id, before_timestamp.isoformat()))
                row = cursor.fetchone()

                if row:
                    return SquadSnapshot(
                        team_id=row['team_id'],
                        team_name=row['team_name'],
                        players=json.loads(row['players_json']),
                        source=row['source'],
                        announcement_timestamp_utc=datetime.fromisoformat(row['announcement_timestamp_utc']),
                        ingestion_timestamp_utc=datetime.fromisoformat(row['ingestion_timestamp_utc'])
                    )
                return None
        except sqlite3.Error as e:
            logger.error(f"Error querying snapshot: {e}")
            raise

    def get_snapshot_history(self, team_id: int) -> List[SquadSnapshot]:
        """Retrieves full audit trail for a specific team."""
        query = '''
            SELECT team_id, team_name, players_json, source, 
                   announcement_timestamp_utc, ingestion_timestamp_utc
            FROM squad_snapshots
            WHERE team_id = ?
            ORDER BY announcement_timestamp_utc ASC
        '''
        try:
            history = []
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, (team_id,))
                rows = cursor.fetchall()
                
                for row in rows:
                    history.append(SquadSnapshot(
                        team_id=row['team_id'],
                        team_name=row['team_name'],
                        players=json.loads(row['players_json']),
                        source=row['source'],
                        announcement_timestamp_utc=datetime.fromisoformat(row['announcement_timestamp_utc']),
                        ingestion_timestamp_utc=datetime.fromisoformat(row['ingestion_timestamp_utc'])
                    ))
            return history
        except sqlite3.Error as e:
            logger.error(f"Error retrieving history: {e}")
            raise

if __name__ == "__main__":
    # Internal validation logic
    try:
        TEST_DB = Path("squad_persistence.db")
        repo = SquadPersistence(TEST_DB)
        
        example_players = [
            {"name": "Player 1", "position": "Goalkeeper", "number": 1},
            {"name": "Player 2", "position": "Defender", "number": 4}
        ]
        
        snap = SquadSnapshot(
            team_id=1,
            team_name="Test Team",
            players=example_players,
            source="API-Football",
            announcement_timestamp_utc=datetime.now(timezone.utc),
            ingestion_timestamp_utc=datetime.now(timezone.utc)
        )
        
        repo.save_snapshot(snap)
        logger.info("Self-test completed successfully.")
    except Exception as exc:
        logger.critical(f"Self-test failed: {exc}")
