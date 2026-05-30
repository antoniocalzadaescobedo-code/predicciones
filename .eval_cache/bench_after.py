"""
Performance benchmark - AFTER refactor.
Run: py bench_after.py
"""

import os
import sys
import time
import types

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
st_mock.cache_data = lambda fn: fn
st_mock.spinner = lambda msg: CM()
st_mock.set_page_config = lambda **kw: None
st_mock.markdown = lambda *a, **kw: None
st_mock.write = lambda *a, **kw: None
st_mock.info = lambda *a, **kw: None
st_mock.success = lambda *a, **kw: None
st_mock.error = lambda *a, **kw: None
st_mock.warning = lambda *a, **kw: None
st_mock.subheader = lambda *a, **kw: None
st_mock.header = lambda *a, **kw: None
st_mock.metric = lambda *a, **kw: None
st_mock.dataframe = lambda *a, **kw: None
st_mock.plotly_chart = lambda *a, **kw: None
st_mock.selectbox = lambda *a, **kw: None
st_mock.multiselect = lambda *a, **kw: []
st_mock.slider = lambda *a, **kw: 0
st_mock.button = lambda *a, **kw: False
st_mock.checkbox = lambda *a, **kw: False
st_mock.progress = lambda *a, **kw: None
st_mock.expander = lambda *a, **kw: CM()
st_mock.columns = lambda n: [CM()] * (n if isinstance(n, int) else len(n))
st_mock.tabs = lambda lst: [CM() for _ in lst]
st_mock.divider = lambda: None
st_mock.caption = lambda *a, **kw: None
st_mock.rerun = lambda: None
st_mock.sidebar = types.SimpleNamespace(
    markdown=lambda *a, **kw: None,
    button=lambda *a, **kw: False,
    success=lambda *a, **kw: None,
    metric=lambda *a, **kw: None,
    write=lambda *a, **kw: None,
    checkbox=lambda *a, **kw: False,
    json=lambda *a, **kw: None,
    info=lambda *a, **kw: None,
)

sys.modules["streamlit"] = st_mock

# ── Import the predictor class ────────────────────────────────────────────────
# Patch the elo_football import to avoid ImportError
with open("app.py", encoding="utf-8", errors="ignore") as f:
    src = f.read()

# Disable elo_football import — replace the entire try/except block
import re

# Replace the whole try/except for elo_football with a simple False assignment
src = re.sub(
    r"try:\n    from elo_football import \([\s\S]*?\)\n\n    ELO_MODULE_AVAILABLE = True\nexcept ImportError:\n    ELO_MODULE_AVAILABLE = False",
    "ELO_MODULE_AVAILABLE = False",
    src,
)

# Extract only the class + imports part (up to @st.cache_resource)
cut_marker = "\n@st.cache_resource"
cut_idx = src.find(cut_marker)
if cut_idx == -1:
    cut_idx = src.find("\ndef _init_tournament_state")
src_class = src[:cut_idx]

g = {}
try:
    exec(compile(src_class, "app.py", "exec"), g)
except Exception as e:
    print(f"EXEC ERROR: {e}")
    # Show the problematic lines
    lines = src_class.split("\n")
    raise
WorldCupPredictor = g["WorldCupPredictor"]

# ── Run benchmarks ────────────────────────────────────────────────────────────
print("=" * 60)
print("PERFORMANCE BENCHMARK — AFTER REFACTOR")
print("=" * 60)

print("\n--- WorldCupPredictor.__init__ ---")
t0 = time.perf_counter()
p = WorldCupPredictor()
t_init = time.perf_counter() - t0
print(f"  __init__ total:     {t_init:.3f}s")

print(f"\n  Data shape:         {p.historical_data.shape}")
print(f"  Teams count:        {len(p.teams_2026)}")

print("\n--- Individual method timings (3 runs each) ---")

times = []
for _ in range(3):
    t = time.perf_counter()
    p._fit_dixon_coles()
    times.append(time.perf_counter() - t)
print(
    f"  _fit_dixon_coles:   {sum(times) / len(times) * 1000:.1f}ms avg  (min={min(times) * 1000:.1f}ms)"
)

times = []
for _ in range(3):
    t = time.perf_counter()
    p._compute_h2h()
    times.append(time.perf_counter() - t)
print(
    f"  _compute_h2h:       {sum(times) / len(times) * 1000:.1f}ms avg  (min={min(times) * 1000:.1f}ms)"
)

times = []
for _ in range(3):
    t = time.perf_counter()
    p._compute_form()
    times.append(time.perf_counter() - t)
print(
    f"  _compute_form:      {sum(times) / len(times) * 1000:.1f}ms avg  (min={min(times) * 1000:.1f}ms)"
)

print("\n--- Monte Carlo ---")
t = time.perf_counter()
p.monte_carlo_simulation(100)
t_mc100 = time.perf_counter() - t
print(f"  monte_carlo(100):   {t_mc100:.3f}s")
print(f"  monte_carlo(1000) estimate: ~{t_mc100 * 10:.1f}s")

print("\n--- Disk cache test ---")
import os

t = time.perf_counter()
r1 = p.get_cached_monte_carlo(100, force_refresh=True)
t_miss = time.perf_counter() - t
print(f"  cache MISS (force): {t_miss:.3f}s")

t = time.perf_counter()
p.cached_monte_carlo = None  # reset RAM cache to test disk
r2 = p.get_cached_monte_carlo(100)
t_hit = time.perf_counter() - t
print(f"  cache HIT (disk):   {t_hit:.4f}s")

cache_file = ".eval_cache/mc_cache_100.pkl"
if os.path.exists(cache_file):
    size = os.path.getsize(cache_file)
    print(f"  cache file size:    {size / 1024:.1f}KB  ({cache_file})")

print("\n--- DC params validation ---")
alpha = p.dc_params["alpha"]
beta = p.dc_params["beta"]
print(
    f"  alpha Brazil={alpha.get('Brazil', 'N/A'):.3f}  France={alpha.get('France', 'N/A'):.3f}"
)
print(
    f"  beta  Brazil={beta.get('Brazil', 'N/A'):.3f}  France={beta.get('France', 'N/A'):.3f}"
)
print(f"  mu={p.dc_params['mu']:.3f}")

print("\n" + "=" * 60)
print("BASELINE (BEFORE) — from profiling session:")
print("  _fit_dixon_coles:   18,200ms  (iterrows x4 x10 EM x48 teams)")
print("  _compute_h2h:       2,000ms   (iterrows x1128 pairs)")
print("  __init__ total:     ~30s+     (synthetic data path)")
print("=" * 60)
print("\nSPEEDUP SUMMARY:")
baseline_dc = 18.2
baseline_h2h = 2.0
dc_actual = sum(times) / len(times) if times else 0.001
print(
    f"  _fit_dixon_coles:   {baseline_dc / (dc_actual if dc_actual > 0 else 0.001):.0f}x faster"
)
print(
    f"  _compute_h2h:       {baseline_h2h / max(0.001, sum(times) / len(times)):.0f}x faster"
)
