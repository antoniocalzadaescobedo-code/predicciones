"""
test_fixture.py — Validation suite for the FIFA 2026 Fixture Engine.
Run: py test_fixture.py
"""

import re
import sys
import time
import types
from collections import Counter, defaultdict

# ── Mock Streamlit ────────────────────────────────────────────────────────────
st_mock = types.ModuleType("streamlit")


class SS(dict):
    def __getattr__(self, k):
        return self.get(k)

    def __setattr__(self, k, v):
        self[k] = v

    def __contains__(self, k):
        return dict.__contains__(self, k)


class CM:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


st_mock.session_state = SS()
st_mock.cache_resource = lambda fn: fn
st_mock.spinner = lambda msg: CM()
st_mock.set_page_config = lambda **kw: None
st_mock.markdown = lambda *a, **kw: None
for _a in [
    "write",
    "info",
    "success",
    "error",
    "warning",
    "subheader",
    "header",
    "metric",
    "dataframe",
    "plotly_chart",
    "selectbox",
    "multiselect",
    "slider",
    "button",
    "checkbox",
    "progress",
    "divider",
    "caption",
    "rerun",
    "balloons",
    "radio",
]:
    setattr(st_mock, _a, lambda *x, **k: None)
st_mock.expander = lambda *a, **kw: CM()
st_mock.columns = lambda n: [CM()] * (n if isinstance(n, int) else len(n))
st_mock.tabs = lambda lst: [CM() for _ in lst]
st_mock.sidebar = types.SimpleNamespace(
    **{
        a: lambda *x, **k: None
        for a in [
            "markdown",
            "button",
            "success",
            "metric",
            "write",
            "checkbox",
            "json",
            "info",
        ]
    }
)
sys.modules["streamlit"] = st_mock

# ── Import fixture engine ─────────────────────────────────────────────────────
from fixture_engine import (  # noqa: E402
    THIRD_PLACE_ELIGIBLE,
    FixtureEngine,
    get_fixture,
)

PASS = "✅ PASS"
FAIL = "❌ FAIL"
errors_total = 0


def check(cond, msg):
    global errors_total
    if cond:
        print(f"  {PASS}: {msg}")
    else:
        print(f"  {FAIL}: {msg}")
        errors_total += 1


print("=" * 60)
print("FIXTURE ENGINE — VALIDATION SUITE")
print("=" * 60)

# ── TEST 1: Load & structural integrity ───────────────────────────
print("\nTEST 1: Load + structural integrity")
fx = get_fixture()
check(len(fx._matches) == 104, "104 total matches")
check(len(fx._by_phase["group"]) == 72, "72 group stage matches")
check(len(fx._by_phase["r32"]) == 16, "16 R32 matches")
check(len(fx._by_phase["r16"]) == 8, "8 R16 matches")
check(len(fx._by_phase["quarter"]) == 4, "4 QF matches")
check(len(fx._by_phase["semi"]) == 2, "2 SF matches")
check(len(fx._by_phase["final"]) == 1, "1 Final match")

# ── TEST 2: Chronological ordering ────────────────────────────────
print("\nTEST 2: Chronological ordering")
all_m = sorted(fx._matches, key=lambda m: m["chrono_order"])
orders = [m["chrono_order"] for m in all_m]
check(orders == list(range(1, 105)), "chrono_order 1..104 sequential")
check(all_m[0]["match_id"] == "M1", "First match is M1")
check(all_m[0]["date"] == "2026-06-11", "M1 date = 2026-06-11")
check(all_m[0]["home_slot"] == "Mexico", "M1 home = Mexico")
check(all_m[-1]["match_id"] == "M104", "Last match is M104 (Final)")
check(all_m[-1]["date"] == "2026-07-19", "M104 date = 2026-07-19")
print(
    f"  First: {all_m[0]['match_id']} {all_m[0]['date']} — {all_m[0]['home_slot']} vs {all_m[0]['away_slot']}"
)
print(
    f"  Last:  {all_m[-1]['match_id']} {all_m[-1]['date']} — {all_m[-1]['home_slot']} vs {all_m[-1]['away_slot']}"
)

# ── TEST 3: No duplicate group matches ────────────────────────────
print("\nTEST 3: No duplicate matches in group stage")
group_pairs: set = set()
dup_found = False
for m in fx._by_phase["group"]:
    pair = tuple(sorted([m["home_slot"], m["away_slot"]]))
    if pair in group_pairs:
        print(f"  DUPLICATE: {pair}")
        dup_found = True
    group_pairs.add(pair)
check(len(group_pairs) == 72, "72 unique group stage match pairs")
check(not dup_found, "No duplicate match pairs")

# ── TEST 4: Each team plays exactly once per matchday per group ────
print("\nTEST 4: Each team plays exactly once per matchday (within group)")
md_errors = 0
for letter in list("ABCDEFGHIJKL"):
    for md in [1, 2, 3]:
        md_m = [m for m in fx._by_group.get(letter, []) if m["matchday"] == md]
        if len(md_m) != 2:
            print(f"  FAIL Group {letter} MD{md}: {len(md_m)} matches (expected 2)")
            md_errors += 1
            continue
        teams_used = [m["home_slot"] for m in md_m] + [m["away_slot"] for m in md_m]
        cnt = Counter(teams_used)
        dupes = [t for t, c in cnt.items() if c > 1]
        if dupes:
            print(f"  FAIL Group {letter} MD{md}: duplicates {dupes}")
            md_errors += 1
check(
    md_errors == 0, "All 12 groups x 3 matchdays — 2 matches each, no team duplicated"
)

# ── TEST 5: Matchday 3 simultaneity ───────────────────────────────
print("\nTEST 5: Matchday 3 simultaneity (same date per group)")
sim_errors = 0
for letter in list("ABCDEFGHIJKL"):
    md3 = [m for m in fx._by_group.get(letter, []) if m["matchday"] == 3]
    if len(md3) != 2 or md3[0]["date"] != md3[1]["date"]:
        print(f"  FAIL Group {letter} MD3 dates: {[m['date'] for m in md3]}")
        sim_errors += 1
check(sim_errors == 0, "All 12 groups have simultaneous MD3 matches")

# ── TEST 6: R32 slot resolution ───────────────────────────────────
print("\nTEST 6: R32 slot resolution (deterministic fake groups)")


def make_fake_groups():
    groups = {}
    for letter in list("ABCDEFGHIJKL"):
        groups[f"Group {letter}"] = {
            "standings": [
                (f"W_{letter}", {"points": 9, "gd": 6, "gf": 8}),
                (f"R_{letter}", {"points": 6, "gd": 2, "gf": 5}),
                (f"T_{letter}", {"points": 3, "gd": -2, "gf": 3}),
                (f"L_{letter}", {"points": 0, "gd": -6, "gf": 1}),
            ],
            "qualified": [f"W_{letter}", f"R_{letter}"],
        }
    return groups


fake_gr = make_fake_groups()
r32 = fx.r32_matches_resolved(fake_gr)
flat = [t for pair in r32 for t in pair]
check(len(r32) == 16, "r32_matches_resolved returns 16 matches")
check(len(set(flat)) == 32, "32 unique teams in R32")
check("TBD" not in flat, "No unresolved TBD slots")
check(all("best_3rd" not in t for t in flat), "All best_3rd slots resolved")

# Official crossings
checks = [
    (0, ("R_A", "R_B"), "M73 = 2A vs 2B"),
    (2, ("W_F", "R_C"), "M75 = 1F vs 2C"),
    (3, ("W_C", "R_F"), "M76 = 1C vs 2F"),
    (5, ("R_E", "R_I"), "M78 = 2E vs 2I"),
    (10, ("R_K", "R_L"), "M83 = 2K vs 2L"),
    (11, ("W_H", "R_J"), "M84 = 1H vs 2J"),
    (13, ("W_J", "R_H"), "M86 = 1J vs 2H"),
    (15, ("R_D", "R_G"), "M88 = 2D vs 2G"),
]
for idx, expected, label in checks:
    check(r32[idx] == expected, f"{label} → got {r32[idx]}")

# ── TEST 7: next_match_winner chain ───────────────────────────────
print("\nTEST 7: next_match_winner chain wiring")
# Official bracket:
#   M90 = W_M73 vs W_M75  => M73 winner goes to M90
#   M89 = W_M74 vs W_M77  => M74 winner goes to M89
m73 = fx.get_match("M73")
m74 = fx.get_match("M74")
m75 = fx.get_match("M75")
m77 = fx.get_match("M77")
m104 = fx.get_match("M104")
check(m73["next_match_winner"] == "M90", "M73 \u2192 M90 (M90 = W73 vs W75)")
check(m74["next_match_winner"] == "M89", "M74 \u2192 M89 (M89 = W74 vs W77)")
check(m75["next_match_winner"] == "M90", "M75 \u2192 M90")
check(m77["next_match_winner"] == "M89", "M77 \u2192 M89")
check(m104.get("next_match_winner") is None, "M104 Final has no next_match_winner")
print(
    f"  M73\u2192{m73['next_match_winner']}  M74\u2192{m74['next_match_winner']}  "
    f"M75\u2192{m75['next_match_winner']}  M77\u2192{m77['next_match_winner']}  M104\u2192{m104.get('next_match_winner')}"
)

# ── TEST 8: simulate_knockout_with_fixture end-to-end ─────────────
print("\nTEST 8: simulate_knockout_with_fixture (20 trials)")
with open("app.py", encoding="utf-8", errors="ignore") as f:
    src = f.read()
src = re.sub(
    r"try:\n    from elo_football import \([\s\S]*?\)\n\n    ELO_MODULE_AVAILABLE = True\nexcept ImportError:\n    ELO_MODULE_AVAILABLE = False",
    "ELO_MODULE_AVAILABLE = False",
    src,
)
cut_m = '\nst.markdown(\n    \'<h1 class="main-header">'
cut_i = src.find(cut_m)
if cut_i == -1:
    cut_i = src.find("\n# Main UI")
g = {}
exec(compile(src[:cut_i], "app.py", "exec"), g)
WorldCupPredictor = g["WorldCupPredictor"]
p = WorldCupPredictor()
groups = p.create_groups()

trial_errors = 0
for trial in range(20):
    gr = p.simulate_group_stage(groups)
    results, champion = fx.simulate_knockout_with_fixture(
        gr, p.simulate_match_with_score
    )
    for stage, exp in [
        ("r32", 32),
        ("r16", 16),
        ("quarter", 8),
        ("semi", 4),
        ("final", 2),
    ]:
        teams = set(t for t1, t2 in results[stage]["matches"] for t in [t1, t2])
        if len(teams) != exp:
            trial_errors += 1
    finalists = [t for t1, t2 in results["final"]["matches"] for t in [t1, t2]]
    if champion not in finalists:
        trial_errors += 1

check(
    trial_errors == 0,
    f"20 trials — R32(32) R16(16) QF(8) SF(4) FINAL(2) champion valid",
)

# ── TEST 9: Monte Carlo via fixture engine ────────────────────────
print("\nTEST 9: Monte Carlo via FixtureEngine (50 sims)")
t0 = time.perf_counter()
champ, fin, semi = p.monte_carlo_simulation(50)
elapsed = time.perf_counter() - t0
s = sum(champ.values())
print(f"  50 sims: {elapsed:.2f}s  sum_probs={s:.3f}")
top5 = sorted(champ.items(), key=lambda x: x[1], reverse=True)[:5]
print(f"  Top 5: {[(t, f'{v:.1%}') for t, v in top5]}")
check(s > 0.8, f"sum(champion_probs) = {s:.3f} > 0.8")

# ── TEST 10: Idempotency ──────────────────────────────────────────
print("\nTEST 10: r32_matches_resolved idempotency")
fake_gr2 = make_fake_groups()
r32_a = fx.r32_matches_resolved(fake_gr2)
r32_b = fx.r32_matches_resolved(fake_gr2)
check(r32_a == r32_b, "r32_matches_resolved returns identical output on repeated calls")

# ── Summary ───────────────────────────────────────────────────────
print()
print("=" * 60)
if errors_total == 0:
    print("ALL TESTS PASSED ✅")
else:
    print(f"FAILED: {errors_total} assertion(s) failed ❌")
print("=" * 60)
sys.exit(0 if errors_total == 0 else 1)
