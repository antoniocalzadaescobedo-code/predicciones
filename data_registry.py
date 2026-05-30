import os
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class DatasetSchema:
    name: str
    path: str
    columns: List[str]
    domain: str  # ui | ml | squads | metrics
    strict: bool = True


class DataRegistry:
    """
    Fuente única de verdad para todos los datasets del sistema.
    Evita KeyErrors y mezcla de esquemas.
    """

    base_dir = os.path.dirname(os.path.abspath(__file__))

    DATASETS: Dict[str, DatasetSchema] = {

        # ─────────────────────────────────────────────
        # UI / CALENDARIO
        # ─────────────────────────────────────────────
        "friendlies": DatasetSchema(
            name="Amistosos 2026",
            path=os.path.join(base_dir, "data", "amistosos_reales.csv"),
            columns=["date", "home_team", "away_team"],
            domain="ui",
        ),

        # ─────────────────────────────────────────────
        # HISTÓRICO / ML / ELO
        # ─────────────────────────────────────────────
        "historical_matches": DatasetSchema(
            name="Histórico internacional",
            path=os.path.join(base_dir, "data", "historico", "matches.csv"),
            columns=[
                "date",
                "home_team",
                "away_team",
                "home_goals",
                "away_goals",
                "tournament",
                "neutral"
            ],
            domain="ml",
        ),

        # ─────────────────────────────────────────────
        # SQUADS
        # ─────────────────────────────────────────────
        "squads": DatasetSchema(
            name="Plantillas Mundial 2026",
            path=os.path.join(base_dir, "wc2026_squads", "data", "world_cup_2026_squads.csv"),
            columns=[
                "player_name",
                "team",
                "position",
                "club",
                "age"
            ],
            domain="squads",
        ),

        # ─────────────────────────────────────────────
        # METRICS
        # ─────────────────────────────────────────────
        "model_metrics": DatasetSchema(
            name="Métricas del modelo",
            path=os.path.join(base_dir, "data", "metrics", "model_metrics.json"),
            columns=[
                "accuracy",
                "log_loss",
                "brier_score",
                "ece"
            ],
            domain="metrics",
        ),
    }

    # ─────────────────────────────────────────────
    # API SIMPLE
    # ─────────────────────────────────────────────

    @classmethod
    def get(cls, key: str) -> DatasetSchema:
        if key not in cls.DATASETS:
            raise KeyError(f"Dataset no registrado: {key}")
        return cls.DATASETS[key]

    @classmethod
    def path(cls, key: str) -> str:
        return cls.get(key).path

    @classmethod
    def columns(cls, key: str) -> List[str]:
        return cls.get(key).columns

    @classmethod
    def validate_columns(cls, key: str, df_columns: List[str]) -> bool:
        required = set(cls.get(key).columns)
        actual = set([str(c).lower().strip() for c in df_columns])

        missing = required - actual

        if missing:
            raise ValueError(
                f"[{key}] Columnas faltantes: {missing}. Encontradas: {actual}"
            )

        return True


