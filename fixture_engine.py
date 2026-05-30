"""
fixture_engine.py
=================
Data-Driven Fixture Engine for FIFA World Cup 2026.

Single source of truth: world_cup_2026_fixture.json
- Loads once, cached in module-level singleton.
- Resolves dynamic slots (1A, 2B, best_3rd_ABCDF) against live group results.
- Drives chronological rendering, Monte Carlo bracket advancement, and UI tabs.

Public API
----------
get_fixture()               -> FixtureEngine singleton (cached)
FixtureEngine.group_matches(group=None)    -> list[dict]
FixtureEngine.knockout_matches(phase=None) -> list[dict]
FixtureEngine.resolve_bracket(group_results) -> dict[match_id -> (team1, team2)]
FixtureEngine.chrono_group_calendar()     -> list[dict] ordered by chrono_order
FixtureEngine.get_match(match_id)         -> dict
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

# ── Constants ─────────────────────────────────────────────────────────────────

_FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "world_cup_2026_fixture.json")

# Berger round-robin table for 4 teams [T0,T1,T2,T3]
# indices into list(combinations(teams, 2)):
#   combos = [(T0,T1),(T0,T2),(T0,T3),(T1,T2),(T1,T3),(T2,T3)]
#   MD1: idx 0 (T0vT1) + idx 5 (T2vT3)
#   MD2: idx 1 (T0vT2) + idx 4 (T1vT3)
#   MD3: idx 2 (T0vT3) + idx 3 (T1vT2)
BERGER_IDX: dict[int, list[int]] = {1: [0, 5], 2: [1, 4], 3: [2, 3]}

# Official slot notation used in the JSON for 3rd-place slots
# Maps internal slot tag → eligible group letters
THIRD_PLACE_ELIGIBLE: dict[str, list[str]] = {
    "best_3rd_ABCDF": ["A", "B", "C", "D", "F"],
    "best_3rd_CDFGH": ["C", "D", "F", "G", "H"],
    "best_3rd_CEFHI": ["C", "E", "F", "H", "I"],
    "best_3rd_EHIJK": ["E", "H", "I", "J", "K"],
    "best_3rd_BEFIJ": ["B", "E", "F", "I", "J"],
    "best_3rd_AEHIJ": ["A", "E", "H", "I", "J"],
    "best_3rd_EFGIJ": ["E", "F", "G", "I", "J"],
    "best_3rd_DEIJL": ["D", "E", "I", "J", "L"],
}

# Stage display names
STAGE_NAMES: dict[str, str] = {
    "group": "Fase de Grupos",
    "r32": "⚔️ Dieciseisavos de Final",
    "r16": "🔥 Octavos de Final",
    "quarter": "💥 Cuartos de Final",
    "semi": "🌟 Semifinales",
    "third_place": "🥉 Tercer Puesto",
    "final": "🏆 Gran Final",
}

KNOCKOUT_PHASES = ["r32", "r16", "quarter", "semi", "third_place", "final"]


# ── FixtureEngine ─────────────────────────────────────────────────────────────


class FixtureEngine:
    """
    Immutable fixture data layer. Loaded once from JSON; all resolution
    is done on-demand without mutating the underlying match records.
    """

    def __init__(self, fixture_path: str = _FIXTURE_PATH) -> None:
        with open(fixture_path, encoding="utf-8") as f:
            raw = json.load(f)

        self.meta: dict[str, Any] = raw["meta"]
        self._matches: list[dict] = raw["matches"]

        # Build fast lookup indices
        self._by_id: dict[str, dict] = {m["match_id"]: m for m in self._matches}
        self._by_phase: dict[str, list[dict]] = {}
        self._by_group: dict[str, list[dict]] = {}

        for m in self._matches:
            self._by_phase.setdefault(m["phase"], []).append(m)
            if m.get("group"):
                self._by_group.setdefault(m["group"], []).append(m)

        self._validate()

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(self) -> None:
        """Structural assertions run at load time."""
        assert len(self._matches) == 104, (
            f"Expected 104 matches, got {len(self._matches)}"
        )
        assert len(self._by_phase.get("group", [])) == 72
        assert len(self._by_phase.get("r32", [])) == 16
        assert len(self._by_phase.get("r16", [])) == 8
        assert len(self._by_phase.get("quarter", [])) == 4
        assert len(self._by_phase.get("semi", [])) == 2
        assert len(self._by_phase.get("final", [])) == 1

        # chrono_order must be unique and sequential
        orders = sorted(m["chrono_order"] for m in self._matches)
        assert orders == list(range(1, 105)), "chrono_order must be 1..104"

        # next_match_winner chains must reference existing match_ids
        valid_ids = set(self._by_id)
        for m in self._matches:
            nw = m.get("next_match_winner")
            if nw and nw not in valid_ids:
                raise ValueError(
                    f"{m['match_id']}.next_match_winner={nw!r} not in fixture"
                )

    # ── Public accessors ──────────────────────────────────────────────────────

    def get_match(self, match_id: str) -> dict:
        """Return a single match record by ID."""
        return self._by_id[match_id]

    def group_matches(self, group: str | None = None) -> list[dict]:
        """
        Return group stage matches sorted by chrono_order.
        If group is given (e.g. 'A'), filter to that group only.
        """
        if group:
            return sorted(
                self._by_group.get(group, []), key=lambda m: m["chrono_order"]
            )
        return sorted(self._by_phase.get("group", []), key=lambda m: m["chrono_order"])

    def knockout_matches(self, phase: str | None = None) -> list[dict]:
        """Return knockout matches sorted by chrono_order."""
        if phase:
            return sorted(
                self._by_phase.get(phase, []), key=lambda m: m["chrono_order"]
            )
        result = []
        for ph in KNOCKOUT_PHASES:
            result.extend(
                sorted(self._by_phase.get(ph, []), key=lambda m: m["chrono_order"])
            )
        return result

    def chrono_group_calendar(self) -> list[dict]:
        """
        Return the 72 group-stage matches in strict FIFA chronological order.
        Each dict has the original match record plus a 'teams' field with
        (home_team, away_team) already resolved (direct team names for group stage).
        """
        matches = self.group_matches()
        result = []
        for m in matches:
            result.append(
                {
                    **m,
                    "team1": m["home_slot"],  # group stage: slots ARE team names
                    "team2": m["away_slot"],
                    "jornada": m["matchday"],
                }
            )
        return result

    # ── Slot resolution ───────────────────────────────────────────────────────

    def resolve_slot(
        self,
        slot: str,
        group_results: dict,
        third_pool: dict[str, str] | None = None,
    ) -> str:
        """
        Resolve a slot string to a concrete team name.

        Slot formats:
          "Mexico"           -> direct team name (group stage)
          "1A"               -> winner of Group A
          "2B"               -> runner-up of Group B
          "best_3rd_ABCDF"   -> best available 3rd-place from groups A/B/C/D/F
          "W_M73"            -> winner of match M73 (requires results dict)
          "L_M101"           -> loser of match M101

        Parameters
        ----------
        slot        : slot string from fixture JSON
        group_results : dict from simulate_group_stage()
        third_pool  : mutable dict mapping group letter -> 3rd-place team.
                      CALLER must pass a COPY to avoid side effects.
        """
        # Direct team name
        if slot and not (
            slot[0].isdigit()
            or slot.startswith("best")
            or slot.startswith("W_")
            or slot.startswith("L_")
        ):
            return slot

        # "1X" → winner of group X
        if len(slot) == 2 and slot[0] == "1":
            letter = slot[1]
            gname = f"Group {letter}"
            if gname in group_results:
                return group_results[gname]["standings"][0][0]
            return f"1{letter}"

        # "2X" → runner-up of group X
        if len(slot) == 2 and slot[0] == "2":
            letter = slot[1]
            gname = f"Group {letter}"
            if gname in group_results:
                return group_results[gname]["standings"][1][0]
            return f"2{letter}"

        # "best_3rd_XXXXX" → best available 3rd from eligible groups
        if slot.startswith("best_3rd_"):
            eligible = list(THIRD_PLACE_ELIGIBLE.get(slot, []))
            pool = third_pool or {}
            for g in eligible:
                if g in pool:
                    return pool.pop(g)
            # Fallback: any remaining 3rd
            if pool:
                return pool.pop(next(iter(pool)))
            return slot  # unresolvable

        # "W_Mxx" / "L_Mxx" → from match results (not directly resolvable here)
        return slot

    def resolve_bracket(
        self,
        group_results: dict,
        match_results: dict[str, str] | None = None,
    ) -> dict[str, tuple[str, str]]:
        """
        Resolve all R32 slots into concrete (team1, team2) pairs.

        Returns a dict mapping match_id -> (team1, team2).

        Parameters
        ----------
        group_results : output of simulate_group_stage()
        match_results : dict of match_id -> winner (for R16+ resolution)
        """
        # Build third-place pool (copy to avoid mutation)
        third_pool = self._build_third_pool(group_results)

        resolved: dict[str, tuple[str, str]] = {}
        for m in self._by_phase.get("r32", []):
            t1 = self.resolve_slot(m["home_slot"], group_results, third_pool)
            t2 = self.resolve_slot(m["away_slot"], group_results, third_pool)
            resolved[m["match_id"]] = (t1, t2)

        return resolved

    def r32_matches_resolved(self, group_results: dict) -> list[tuple[str, str]]:
        """
        Return the 16 R32 matches as (team1, team2) tuples,
        ordered by chrono_order (M73..M88).
        Idempotent — builds a fresh copy of the third pool each call.
        """
        third_pool = self._build_third_pool(group_results)
        result = []
        for m in sorted(self._by_phase.get("r32", []), key=lambda x: x["chrono_order"]):
            t1 = self.resolve_slot(m["home_slot"], group_results, dict(third_pool))
            t2 = self.resolve_slot(m["away_slot"], group_results, dict(third_pool))
            result.append((t1, t2))
        # Re-resolve in one pass to handle pool depletion correctly
        third_pool = self._build_third_pool(group_results)
        result = []
        for m in sorted(self._by_phase.get("r32", []), key=lambda x: x["chrono_order"]):
            t1 = self.resolve_slot(m["home_slot"], group_results, third_pool)
            t2 = self.resolve_slot(m["away_slot"], group_results, third_pool)
            result.append((t1, t2))
        return result

    # ── Third-place helpers ───────────────────────────────────────────────────

    def _build_third_pool(self, group_results: dict) -> dict[str, str]:
        """
        Build {group_letter: team_name} for the top-8 third-place finishers.
        Returns a COPY (caller is responsible for further copying if needed).
        """
        thirds = []
        for gname, result in group_results.items():
            standings = result.get("standings", [])
            if len(standings) >= 3:
                letter = gname.replace("Group ", "")
                team, stats = standings[2]
                thirds.append(
                    {
                        "group": letter,
                        "team": team,
                        "points": stats.get("points", 0),
                        "gd": stats.get("gd", 0),
                        "gf": stats.get("gf", 0),
                    }
                )

        # FIFA tiebreaker for best 8 of 12 third-place teams
        thirds.sort(key=lambda x: (x["points"], x["gd"], x["gf"]), reverse=True)
        best8 = thirds[:8]
        return {t["group"]: t["team"] for t in best8}

    # ── Monte Carlo integration ───────────────────────────────────────────────

    def simulate_knockout_with_fixture(
        self,
        group_results: dict,
        simulate_fn,  # callable(team1, team2, stage, knockout) -> (g1, g2, winner, pens)
    ) -> tuple[dict, str | None]:
        """
        Simulate the complete knockout stage using the official fixture tree.

        Parameters
        ----------
        group_results : output of simulate_group_stage()
        simulate_fn   : predictor.simulate_match_with_score

        Returns
        -------
        results : dict mapping stage -> {matches, scores, winners}
        champion : winning team name, or None
        """
        # Build third pool for slot resolution
        third_pool = self._build_third_pool(group_results)

        # Map match_id -> winner (filled as we simulate)
        match_winners: dict[str, str] = {}

        results: dict[str, dict] = {}

        for phase in ["r32", "r16", "quarter", "semi", "final"]:
            phase_matches = sorted(
                self._by_phase.get(phase, []),
                key=lambda m: m["chrono_order"],
            )
            p_match_tuples = []
            p_scores = []
            p_winners = []

            for m in phase_matches:
                mid = m["match_id"]

                # Resolve slots
                hs = m["home_slot"]
                as_ = m["away_slot"]

                if phase == "r32":
                    t1 = self.resolve_slot(hs, group_results, third_pool)
                    t2 = self.resolve_slot(as_, group_results, third_pool)
                else:
                    # W_Mxx slots
                    t1 = match_winners.get(hs[2:], hs) if hs.startswith("W_") else hs
                    t2 = (
                        match_winners.get(as_[2:], as_) if as_.startswith("W_") else as_
                    )

                # Simulate
                g1, g2, winner, pens = simulate_fn(t1, t2, phase, True)
                match_winners[mid] = winner

                p_match_tuples.append((t1, t2))
                p_scores.append((g1, g2, pens))
                p_winners.append(winner)

            results[phase] = {
                "matches": p_match_tuples,
                "scores": p_scores,
                "winners": p_winners,
            }

        champion = results.get("final", {}).get("winners", [None])[0]

        # Structural assertions
        assert len(results.get("r32", {}).get("matches", [])) == 16
        assert len(results.get("r16", {}).get("matches", [])) == 8
        assert len(results.get("quarter", {}).get("matches", [])) == 4
        assert len(results.get("semi", {}).get("matches", [])) == 2
        assert len(results.get("final", {}).get("matches", [])) == 1

        return results, champion

    # ── Calendar helpers for UI ───────────────────────────────────────────────

    def get_group_for_team(self, team: str) -> str | None:
        """Return the group letter for a given team name."""
        for m in self._by_phase.get("group", []):
            if m["home_slot"] == team or m["away_slot"] == team:
                return m["group"]
        return None

    def get_matchday_matches(self, matchday: int) -> list[dict]:
        """Return all group stage matches for a given matchday (1/2/3)."""
        return sorted(
            [m for m in self._by_phase.get("group", []) if m["matchday"] == matchday],
            key=lambda m: m["chrono_order"],
        )

    def get_r32_match_for_slot(self, slot: str) -> dict | None:
        """Find the R32 match that contains a given slot."""
        for m in self._by_phase.get("r32", []):
            if m["home_slot"] == slot or m["away_slot"] == slot:
                return m
        return None

    def chrono_order_of(self, match_id: str) -> int:
        return self._by_id[match_id]["chrono_order"]


# ── Module-level singleton ────────────────────────────────────────────────────

_engine_instance: FixtureEngine | None = None


def get_fixture() -> FixtureEngine:
    """
    Return the module-level FixtureEngine singleton.
    Loads from disk exactly once per Python process.
    """
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = FixtureEngine()
    return _engine_instance
