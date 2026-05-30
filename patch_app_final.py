
import os

path = "C:/Proyecto_FIFA/app_streamlit.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Eliminar capa anterior si existe y agregar la nueva
import re

# Buscar el bloque de la capa de tendencia 12m anterior
old_layer_pattern = r"# -+.*?# CAPA DE TENDENCIA 12 MESES.*?TEAM_TRENDS = load_team_trends\(\).*?return adj"
# Since regex for long blocks can be tricky, I will use known markers

insertion_point = "def calculate_prediction(home, away, db, predictor, neutral, match_type):"

new_layer_logic = """
# ------------------------------------------------------------------------------
# CAPA DE FORMA RECIENTE (ÚLTIMOS 12 PARTIDOS) - FIFA 2026
# ------------------------------------------------------------------------------
@st.cache_data
def load_wc_form_data():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, "data/world_cup_48_form.json")
        with open(path, "r", encoding="utf-8") as f:
            import json
            return json.load(f)
    except:
        return {}

WC_FORM_DATA = load_wc_form_data()

def get_trend_adjustment(team_name_es):
    \"\"\"Calcula el ajuste de Elo basado en la forma real de los últimos 12 partidos.\"\"\"
    english_name = None
    for eng, esp in TEAM_NAME_TRANSLATION.items():
        if esp == team_name_es:
            english_name = eng
            break
    
    if not english_name or english_name not in WC_FORM_DATA:
        return 0.0
    
    f = WC_FORM_DATA[english_name]
    # Fórmula Final: Balance entre consistencia (PPG) y momento (Form 5)
    adj = (f["ppg12"] * 20.0) + (f["gd_avg"] * 15.0) + (f["form_last_5"] * 30.0)
    return round(adj, 1)

"""

# Clean up existing trend logic if I added it before
content = re.sub(r"# -+.*?CAPA DE TENDENCIA 12 MESES.*?return adj", "", content, flags=re.DOTALL)

if "CAPA DE FORMA RECIENTE" not in content:
    content = content.replace(insertion_point, new_layer_logic + insertion_point)

# 2. Asegurar que calculate_prediction use la nueva función
old_ranks = """        # Aplicar Ajuste de Tendencia 12m
        adj_h = get_team_trend_adjustment(home)
        adj_a = get_team_trend_adjustment(away)
        rank_home = float(row_home["elo_rating"]) + adj_h
        rank_away = float(row_away["elo_rating"]) + adj_a"""

# Fallback if the above was not found (maybe the user reverted or it failed)
fallback_ranks = """        rank_home = float(row_home["elo_rating"])
        rank_away = float(row_away["elo_rating"])"""

final_ranks = """        # Aplicar Ajuste de Forma Real (Últimos 12 partidos)
        adj_h = get_trend_adjustment(home)
        adj_a = get_trend_adjustment(away)
        rank_home = float(row_home["elo_rating"]) + adj_h
        rank_away = float(row_away["elo_rating"]) + adj_a"""

if old_ranks in content:
    content = content.replace(old_ranks, final_ranks)
elif fallback_ranks in content:
    content = content.replace(fallback_ranks, final_ranks)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("✅ app_streamlit.py actualizado con la Capa de Forma Final")

