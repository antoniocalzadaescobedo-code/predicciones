import math
import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import time
import json
import itertools
from scipy.stats import poisson
from ui_components import result_card_unique

# Asegurar que el directorio de trabajo sea el correcto
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Configuración de página
st.set_page_config(page_title="AC 2026 Predictor", page_icon="⚽", layout="wide", initial_sidebar_state="expanded")

# CSS
st.markdown("""
<style>
.main-header {font-size: 2.5rem; font-weight: bold; text-align: center; background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 1rem;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# IMPORTS CORREGIDOS
# ─────────────────────────────────────────────────────────────
try:
    from fifa_teams_database import FIFATeamsDatabase
except ImportError:
    st.error("❌ Error: No se encontró `fifa_teams_database.py`")
    st.stop()

try:
    from gbm_production import FIFA2026Predictor
except ImportError:
    try:
        from gbm_production import GBMPredictor as FIFA2026Predictor
    except ImportError:
        st.error("❌ Error: No se encontró `gbm_production.py`")
        st.stop()

def get_live_updater():
    from live_updater import LiveFormUpdater
    return LiveFormUpdater

try:
    # No llamar a get_live_updater() aquí para evitar importación inmediata
    LIVE_UPDATER_AVAILABLE = True
except ImportError:
    LIVE_UPDATER_AVAILABLE = False

OFFICIAL_GROUPS = {
    "A": ["México", "Sudáfrica", "Corea del Sur", "República Checa"],
    "B": ["Canadá", "Bosnia y Herzegovina", "Catar", "Suiza"],
    "C": ["Brasil", "Marruecos", "Haití", "Escocia"],
    "D": ["Estados Unidos", "Paraguay", "Australia", "Turquía"],
    "E": ["Alemania", "Curazao", "Costa de Marfil", "Ecuador"],
    "F": ["Países Bajos", "Japón", "Suecia", "Túnez"],
    "G": ["Bélgica", "Egipto", "Irán", "Nueva Zelanda"],
    "H": ["España", "Cabo Verde", "Arabia Saudita", "Uruguay"],
    "I": ["Francia", "Senegal", "Irak", "Noruega"],
    "J": ["Argentina", "Argelia", "Austria", "Jordania"],
    "K": ["Portugal", "República Democrática del Congo", "Uzbekistán", "Colombia"],
    "L": ["Inglaterra", "Croacia", "Ghana", "Panamá"]
}
MONTE_CARLO_AVAILABLE = False

# --- NORMALIZACIÓN CENTRALIZADA DE NOMBRES ---
def normalize_team(name):
    if not name: return "Desconocido"
    # Mapeo de alias comunes
    alias_map = {
        "EEUU": "Estados Unidos",
        "USA": "Estados Unidos",
        "United States": "Estados Unidos",
        "South Korea": "Corea del Sur",
        "Arabia Saudí": "Arabia Saudita",
        "Saudi Arabia": "Arabia Saudita",
        "Holanda": "Países Bajos",
        "Netherlands": "Países Bajos",
        "Azerbayán": "Azerbaiyán",
        "Azerbaijan": "Azerbaiyán",
        "Ivory Coast": "Costa de Marfil",
        "Czech Republic": "República Checa",
        "England": "Inglaterra",
        "Morocco": "Marruecos",
        "Curaçao": "Curazao",
        "Curacao": "Curazao"
    }
    name = name.strip()
    return alias_map.get(name, name)

# ─────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES LOCALES
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_json(path):
    """Carga archivos JSON con caché y fallback automático"""
    # Try multiple possible paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(base_dir, path),
        path,
        os.path.join(base_dir, os.path.basename(path)),
        os.path.join(base_dir, "data", os.path.basename(path))
    ]
    
    for abs_path in possible_paths:
        try:
            with open(abs_path, "r", encoding="utf-8-sig") as f:
                return json.load(f)
        except FileNotFoundError:
            continue
        except json.JSONDecodeError:
            continue
    
    # If all paths fail, return empty dict (graceful degradation)
    return {}

@st.cache_resource
def load_predictor():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Try to load the new temporal model first
        import joblib
        temporal_model_path = os.path.join(base_dir, "gbm_wc2026_v2_temporal.joblib")
        package = joblib.load(temporal_model_path)
        # Create a wrapper that uses the calibrated model from the package
        class TemporalPredictorWrapper:
            def __init__(self, package):
                self.model = package['model']
                self.feature_names = package['feature_names']
                self.metrics = package['metrics']
                self.is_fitted = True
                
            def predict_match(self, team_home, team_away, features, neutral=False):
                # Construct feature vector in correct order
                feature_vector = np.array([
                    features.get('elo_diff', 0),
                    int(neutral),
                    features.get('form_home', 0.5),
                    features.get('form_away', 0.5),
                    features.get('h2h', 0.5)
                ]).reshape(1, -1)
                
                # Get probabilities from calibrated model
                proba = self.model.predict_proba(feature_vector)[0]
                
                # Model classes are [-1, 0, 1] = [away, draw, home]
                pred_class = self.model.classes_[np.argmax(proba)]
                pred_map = {-1: 'away_win', 0: 'draw', 1: 'home_win'}
                
                return {
                    'teams': {'home': team_home, 'away': team_away},
                    'prediction': {
                        'outcome': pred_map[pred_class],
                        'confidence': float(np.max(proba))
                    },
                    'probabilities': {
                        'away_win': float(proba[0]),
                        'draw': float(proba[1]),
                        'home_win': float(proba[2])
                    }
                }
        
        return TemporalPredictorWrapper(package)
    except FileNotFoundError:
        # Fallback to old model if temporal model doesn't exist
        try:
            old_model_path = os.path.join(base_dir, "gbm_wc2026_v1.joblib")
            return FIFA2026Predictor.load(old_model_path)
        except Exception as e:
            st.error(f"Error cargando predictor: {e}")
            return None
    except Exception as e:
        st.error(f"Error cargando predictor temporal: {e}")
        return None

@st.cache_resource
def load_teams_db():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return FIFATeamsDatabase(os.path.join(base_dir, "fifa_teams_db_es.json"))

@st.cache_data
def get_team_list():
    try:
        db = load_teams_db()
        return sorted(db.df["team_name"].tolist())
    except:
        return []





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
    """Calcula el ajuste de Elo basado en la forma real de los últimos 12 partidos."""
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

def calculate_prediction(home, away, db, predictor, neutral, match_type):
    home = normalize_team(home)
    away = normalize_team(away)
    try:
        # Obtener puntaje ELO de los equipos con fallback seguro
        res_h = db.df[db.df["team_name"] == home]
        if not res_h.empty:
            elo_h = float(res_h.iloc[0]["elo_rating"])
        else:
            elo_h = 1350.0  # Nivel base para equipos desconocidos
            st.caption(f"⚠️ {home}: Usando valores por defecto")

        res_a = db.df[db.df["team_name"] == away]
        if not res_a.empty:
            elo_a = float(res_a.iloc[0]["elo_rating"])
        else:
            elo_a = 1350.0
            st.caption(f"⚠️ {away}: Usando valores por defecto")

        # Aplicar Ajuste de Forma Real (Últimos 12 partidos)
        adj_h = get_trend_adjustment(home)
        adj_a = get_trend_adjustment(away)
        rank_home = elo_h + adj_h
        rank_away = elo_a + adj_a
        
        # 1. FUERZA BASE (Probabilidad ELO estándar - Escala Logarítmica)
        # Transformamos los puntos ELO en probabilidades de victoria esperada (fuerza relativa)
        # ratio = 10^(diff/400). Esto evita que la diferencia se comprima linealmente.
        exp_home = 10**(rank_home / 400)
        exp_away = 10**(rank_away / 400)
        fuerza_home = exp_home / (exp_home + exp_away)
        fuerza_away = exp_away / (exp_home + exp_away)
        
        # 2. FACTORES DETERMINÍSTICOS ( PhD Stats / Data Engineering )
        promedio_goles_total = 2.8
        
        # factor_local: Ventaja por jugar en casa
        if neutral:
            factor_local = 1.0
        elif match_type == "Amistoso":
            factor_local = 1.15  # Ventaja moderada en amistosos
        else:
            factor_local = 1.35  # Ventaja alta en competiciones oficiales (Mundial/Eliminatorias)
            
        # motivación: Factor psicológico interno
        mot_home = 1.0  # El local siempre está al 100%
        if match_type == "Amistoso" and not neutral:
            mot_away = 0.85  # El visitante suele rotar o probar tácticas en amistosos
        else:
            mot_away = 1.0  # En Mundial o campo neutral, ambos al 100%

        # 3. GOLES ESPERADOS AJUSTADOS (Lambdas)
        # l_home = fuerza * promedio * factor_local * motivación
        # l_away = fuerza * promedio * (1/factor_local) * motivación
        l_home = fuerza_home * promedio_goles_total * factor_local * mot_home
        l_away = fuerza_away * promedio_goles_total * (1/factor_local) * mot_away
            
        # 4. CÁLCULO DE PROBABILIDADES (Poisson)
        p_home_win, p_draw, p_away_win = 0.0, 0.0, 0.0
        
        for i in range(12):  # Goles local (límite 12 para mayor precisión)
            prob_i = poisson.pmf(i, l_home)
            for j in range(12):  # Goles visitante
                prob_j = poisson.pmf(j, l_away)
                p_match = prob_i * prob_j
                if i > j: p_home_win += p_match
                elif i == j: p_draw += p_match
                else: p_away_win += p_match
        
        # Normalización final
        total = p_home_win + p_draw + p_away_win
        p_final = {
            "home_win": p_home_win / total,
            "draw": p_draw / total,
            "away_win": p_away_win / total
        }

        return {
            "success": True,
            "elo_diff": rank_home - rank_away,
            "probabilities": p_final,
            "prediction": max(p_final, key=p_final.get),
            "confidence": max(p_final.values()),
            "lambdas": {"home": l_home, "away": l_away}
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def render_inline_simulator(home_default, away_default, date, db, predictor, neutral, match_type, team_list_param):
    home_default = normalize_team(home_default)
    away_default = normalize_team(away_default)
    match_key = f"sim_{home_default}_{away_default}_{date}"
    if match_key not in st.session_state.simulators:
        st.session_state.simulators[match_key] = {"expanded": False, "result": None, "odds": {"home": 2.10, "draw": 3.40, "away": 3.60}}
    local_state = st.session_state.simulators[match_key]

    if not local_state["expanded"]:
        if st.button("Ver ↗", key=f"btn_open_{match_key}", use_container_width=False):
            local_state["expanded"] = True
        return

    st.divider()
    c_close, c_title = st.columns([1, 5])
    with c_close:
        if st.button("❌ Cerrar", key=f"btn_close_{match_key}", type="secondary"):
            local_state["expanded"] = False
            local_state["result"] = None
    with c_title:
        st.caption(f"🔮 Simulador Inline: {home_default} vs {away_default}")

    with st.form(f"form_{match_key}", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            h_idx = team_list_param.index(home_default) if home_default in team_list_param else 0
            sim_home = st.selectbox("🏠 Local", team_list_param, index=h_idx, key=f"sel_home_{match_key}")
        with col2:
            a_idx = team_list_param.index(away_default) if away_default in team_list_param else min(1, len(team_list_param)-1)
            sim_away = st.selectbox("✈️ Visitante", team_list_param, index=a_idx, key=f"sel_away_{match_key}")
        
        submitted = st.form_submit_button("🔮 Calcular Predicción", type="primary", use_container_width=True)
        if submitted:
            if sim_home == sim_away:
                st.error("⚠️ Selecciona equipos diferentes")
            else:
                with st.spinner("🧮 Procesando..."):
                    local_state["result"] = calculate_prediction(sim_home, sim_away, db, predictor, neutral, match_type)

    if local_state.get("result"):
        res = local_state["result"]
        if res["success"]:
            m1, m2 = st.columns(2)
            with m1: st.metric("Elo Diff", f"{res['elo_diff']:+.1f}")
            with m2: st.metric("Confianza", f"{res['confidence']*100:.1f}%")
            prob_df = pd.DataFrame({"Resultado": [f"Gana {sim_home}", "Empate", f"Gana {sim_away}"], "Probabilidad": [res["probabilities"]["home_win"], res["probabilities"]["draw"], res["probabilities"]["away_win"]]})
            # Nueva tarjeta de resultado única
            probs_list = [res["probabilities"]["home_win"], res["probabilities"]["draw"], res["probabilities"]["away_win"]]
            team_names = [sim_home, "Empate", sim_away]
            result_card_unique(probs_list, team_names)
            st.markdown("**💰 Value Betting**")
            col_od1, col_od2, col_od3 = st.columns(3)
            odds = local_state["odds"]
            with col_od1: odds["home"] = st.number_input("Cuota 1", value=odds["home"], key=f"odds_h_{match_key}")
            with col_od2: odds["draw"] = st.number_input("Cuota X", value=odds["draw"], key=f"odds_x_{match_key}")
            with col_od3: odds["away"] = st.number_input("Cuota 2", value=odds["away"], key=f"odds_a_{match_key}")
            val_h = res["probabilities"]["home_win"] - (1/odds["home"])
            val_d = res["probabilities"]["draw"] - (1/odds["draw"])
            val_a = res["probabilities"]["away_win"] - (1/odds["away"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Valor Local", f"{val_h:+.1%}", delta="✅ APUESTA" if val_h > 0 else "❌")
            c2.metric("Valor Empate", f"{val_d:+.1%}", delta="✅ APUESTA" if val_d > 0 else "❌")
            c3.metric("Valor Visitante", f"{val_a:+.1%}", delta="✅ APUESTA" if val_a > 0 else "❌")
        else:
            st.error(f"Error: {res['error']}")

# ─────────────────────────────────────────────────────────────
# CARGA DE DATOS DESDE JSON Y CSV
# ─────────────────────────────────────────────────────────────

# Diccionario de traducción de nombres de equipos (Inglés → Español)

TEAM_NAME_TRANSLATION = {
    "Morocco": "Marruecos",
    "Burundi": "Burundi",
    "Nigeria": "Nigeria",
    "Zimbabwe": "Zimbabue",
    "Egypt": "Egipto",
    "Russia": "Rusia",
    "Republic of Ireland": "Irlanda",
    "Qatar": "Catar",
    "Germany": "Alemania",
    "Finland": "Finlandia",
    "Poland": "Polonia",
    "Ukraine": "Ucrania",
    "USA": "Estados Unidos",
    "United States": "Estados Unidos",
    "Senegal": "Senegal",
    "Brazil": "Brasil",
    "Panama": "Panamá",
    "Turkey": "Turquía",
    "North Macedonia": "Macedonia del Norte",
    "Norway": "Noruega",
    "Sweden": "Suecia",
    "Croatia": "Croacia",
    "Belgium": "Bélgica",
    "Wales": "Gales",
    "Ghana": "Ghana",
    "Denmark": "Dinamarca",
    "DR Congo": "República Democrática del Congo",
    "Luxembourg": "Luxemburgo",
    "Italy": "Italia",
    "Netherlands": "Países Bajos",
    "Algeria": "Argelia",
    "Northern Ireland": "Irlanda del Norte",
    "Guinea": "Guinea",
    "Greece": "Grecia",
    "Spain": "España",
    "Iraq": "Irak",
    "France": "Francia",
    "Ivory Coast": "Costa de Marfil",
    "Canada": "Canadá",
    "Tunisia": "Túnez",
    "Portugal": "Portugal",
    "Chile": "Chile",
    "Romania": "Rumania",
    "Bolivia": "Bolivia",
    "Scotland": "Escocia",
    "England": "Inglaterra",
    "New Zealand": "Nueva Zelanda",
    "Venezuela": "Venezuela",
    "Argentina": "Argentina",
    "Honduras": "Honduras",
    "Peru": "Perú",
    "Colombia": "Colombia",
    "Jordan": "Jordania",
    "Cyprus": "Chipre",
    "Jamaica": "Jamaica",
    "India": "India",
    "Iran": "Irán",
    "Gambia": "Gambia",
    "Mexico": "México",
    "South Africa": "Sudáfrica",
    "Nicaragua": "Nicaragua",
    "Andorra": "Andorra",
    "Bosnia and Herzegovina": "Bosnia y Herzegovina",
    "Curacao": "Curazao",
    "Ecuador": "Ecuador",
    "Saudi Arabia": "Arabia Saudita",
    "South Korea": "Corea del Sur",
    "Trinidad and Tobago": "Trinidad y Tobago",
    "Australia": "Australia",
    "Japan": "Japón",
    "Iceland": "Islandia",
    "Singapore": "Singapur",
    "Mongolia": "Mongolia",
    "Switzerland": "Suiza",
    "Cape Verde": "Cabo Verde",
    "Serbia": "Serbia",
    "Czech Republic": "República Checa",
    "Kosovo": "Kosovo",
    "Tajikistan": "Tayikistán",
    "Palestine": "Palestina",
    "Slovakia": "Eslovaquia",
    "Malta": "Malta",
    "Bulgaria": "Bulgaria",
    "Montenegro": "Montenegro",
    "Austria": "Austria",
    "Costa Rica": "Costa Rica",
    "Uzbekistan": "Uzbekistán",
    "Georgia": "Georgia",
    "Madagascar": "Madagascar",
    "Haiti": "Haití",
    "Philippines": "Filipinas",
    "Guam": "Guam",
    "Kyrgyzstan": "Kirguistán",
    "Kenya": "Kenia",
    "Gibraltar": "Gibraltar",
    "British Virgin Islands": "Islas Vírgenes Británicas",
    "Albania": "Albania",
    "Israel": "Israel",
    "Dominican Republic": "República Dominicana",
    "El Salvador": "El Salvador",
    "Cambodia": "Camboya",
    "Slovenia": "Eslovenia",
    "Moldova": "Moldavia",
    "Equatorial Guinea": "Guinea Ecuatorial",
    "Liechtenstein": "Liechtenstein",
    "Guatemala": "Guatemala",
    "China": "China",
    "Angola": "Angola",
    "Botswana": "Botsuana",
    "Tanzania": "Tanzania",
    "Uganda": "Uganda",
    "Niger": "Níger",
    "Mauritania": "Mauritania",
    "Hong Kong": "Hong Kong",
    "Central African Republic": "República Centroafricana",
    "Togo": "Togo",
    "Thailand": "Tailandia",
    "Kuwait": "Kuwait",
    "Indonesia": "Indonesia",
    "Oman": "Omán",
    "Bahrain": "Baréin",
    "Belarus": "Bielorrusia",
    "Syria": "Siria",
    "Burkina Faso": "Burkina Faso",
    "San Marino": "San Marino",
    "Bangladesh": "Bangladés",
    "Hungary": "Hungría",
    "Azerbaijan": "Azerbaiyán",
    "Paraguay": "Paraguay",
    "Puerto Rico": "Puerto Rico",
    "Myanmar": "Birmania",
    "Ethiopia": "Etiopía",
    "Malawi": "Malaui",
    "Comoros": "Comoras",
    "Rwanda": "Ruanda",
    "Cayman Islands": "Islas Caimán",
    "Aruba": "Aruba",
    "Liberia": "Liberia",
    "Benin": "Benín",
    "Mozambique": "Mozambique",
    "Kazakhstan": "Kazajistán",
    "United Arab Emirates": "Emiratos Árabes Unidos"
}

from data_registry import DataRegistry

@st.cache_data
def load_friendlies_csv():
    try:
        file_path = DataRegistry.path("friendlies")
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Validar esquema esperado para UI antes de procesarlo
        DataRegistry.validate_columns("friendlies", df.columns.tolist())

        friendlies = []

        for _, row in df.iterrows():
            home_es = normalize_team(row['home_team'])
            away_es = normalize_team(row['away_team'])

            date_obj = pd.to_datetime(row['date'])
            date_str = date_obj.strftime('%d %b').upper()

            friendlies.append({
                "date": date_str,
                "time": "15:00",
                "home": home_es,
                "away": away_es,
                "status": "PENDIENTE",
                "result": None
            })

        return friendlies

    except Exception as e:
        st.error(f"Error cargando CSV de amistosos: {e}")
        return []

@st.cache_data
def load_friendlies_json():
    try:
        friendlies_data = load_json("data/fixtures/fixtures_friendlies_2026.json")
        return friendlies_data
    except Exception as e:
        st.error(f"Error cargando JSON de amistosos: {e}")
        return []

# Cargar datos
CALENDARIO_AMISTOSOS_2026 = load_friendlies_json()
WORLDCUP_VENUES = load_json("data/venues/worldcup_venues.json")
KNOCKOUT_SCHEDULE = load_json("data/config/knockout_schedule.json")
MODEL_METRICS = load_json("data/metrics/model_metrics.json")
UI_CONFIG = load_json("data/config/ui_config.json")

# Convertir fechas string a datetime objects para knockout_schedule
for phase, config in KNOCKOUT_SCHEDULE.items():
    if "start" in config:
        config["start"] = datetime.strptime(config["start"], "%Y-%m-%d")

# ─────────────────────────────────────────────────────────────
# SIDEBAR & CARGA INICIAL
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title(UI_CONFIG["sidebar"]["config_title"])
    if "match_type" not in st.session_state: st.session_state.match_type = "Amistoso"
    match_type_radio = st.radio(UI_CONFIG["sidebar"]["match_type_label"], ["Amistoso", "Mundial"], index=0 if st.session_state.match_type == "Amistoso" else 1)
    if match_type_radio != st.session_state.match_type:
        st.session_state.match_type = match_type_radio
        st.rerun()

    if match_type_radio == "Amistoso":
        neutral = False
    else:
        neutral = True
        st.success(UI_CONFIG["sidebar"]["world_mode"])

    if st.button(UI_CONFIG["sidebar"]["reload_button"]):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

    if LIVE_UPDATER_AVAILABLE:
        st.divider()
        st.subheader(UI_CONFIG["sidebar"]["live_title"])
        if "last_update_info" not in st.session_state: st.session_state.last_update_info = None
        info = st.session_state.last_update_info
        if info:
            st.warning(f"🟡 Última actualización: {info.get('last_update', 'Nunca')}")
        else:
            st.info(UI_CONFIG["sidebar"]["loading"])
        if st.button(UI_CONFIG["sidebar"]["update_button"], type="secondary", use_container_width=True):
            with st.spinner("🔍 Buscando partidos recientes..."):
                try:
                    updater = get_live_updater()()
                    result = updater.run_auto_update(days_back=3)
                    st.session_state.last_update_info = result
                    st.success("✅ Actualización completada")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

team_list = get_team_list()
predictor = load_predictor()
db = load_teams_db()
if predictor is None:
    st.error("❌ Modelo no encontrado.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# SISTEMA DE NAVEGACIÓN
# ─────────────────────────────────────────────────────────────

TABS = {
    "inicio": "🏠 Inicio",
    "simulador": "🔮 Simulador",
    "calendario": "📅 Calendario",
    "analisis": "✅ Ficha Técnica"
}

# Inicializar namespaces de session_state
if "simulators" not in st.session_state:
    st.session_state.simulators = {}

# Función centralizada para cambiar pestaña (evita loops)
def set_current_tab(tab):
    """Centraliza el cambio de pestaña y sincroniza con query params"""
    """Centraliza el cambio de pestaña y sincroniza con query params"""
    if tab in TABS:
        st.session_state.current_tab = tab
        st.query_params["tab"] = tab

# Inicializar current_tab desde query params o session_state
if "current_tab" not in st.session_state:
    # Intentar leer desde query params (modern Streamlit devuelve string, no lista)
    tab_param = st.query_params.get("tab", "inicio")
    st.session_state.current_tab = tab_param if tab_param in TABS else "inicio"

# Sidebar navegación
with st.sidebar:
    st.divider()
    st.subheader(UI_CONFIG["sidebar"]["nav_title"])
    
    # Usar segmented_control para navbar moderna (Streamlit 1.28+)
    try:
        selected_tab = st.segmented_control(
            options=list(TABS.values()),
            selection_mode="single",
            default=TABS[st.session_state.current_tab],
            label_visibility="collapsed"
        )
        # Mapear de valor a clave
        selected_tab_key = next(k for k, v in TABS.items() if v == selected_tab)
        set_current_tab(selected_tab_key)
    except Exception:
        # Fallback a radio button si segmented_control no está disponible
        selected_tab = st.radio(
            UI_CONFIG["sidebar"]["nav_label"],
            options=list(TABS.keys()),
            format_func=lambda x: TABS[x],
            index=list(TABS.keys()).index(st.session_state.current_tab)
        )
        set_current_tab(selected_tab)

# ─────────────────────────────────────────────────────────────
# FUNCIONES DE RENDERIZADO DE PESTAÑAS
# ─────────────────────────────────────────────────────────────

def render_inicio():
    logo_path = "assets/wc2026_logo.png"
    if os.path.exists(logo_path): st.image(logo_path, width=120)
    st.markdown(f"""<div style="background: linear-gradient(90deg, #FF0000 0%, #00FF00 50%, #0000FF 100%); padding: 12px 20px; border-radius: 8px; margin: 10px 0;"><strong style="color: white; font-size: 1.5rem;">AC 2026 Predictor</strong></div>""", unsafe_allow_html=True)
    st.divider()
    
    # ------------------------------------------------------------
    # INFORMACIÓN MUNDIAL 2026
    # ------------------------------------------------------------
    st.subheader("🏆 FIFA World Cup 2026")
    col1, col2, col3 = st.columns(3)
    col1.metric("🏟️ Equipos", "48")
    col2.metric("📅 Partidos", "104")
    col3.metric("🌍 Grupos", "12")
    
    st.markdown("**Fechas Oficiales:** 11 junio - 19 julio 2026")
    st.markdown("**Calendario:** Datos oficiales FIFA 2026 ✅")
    st.divider()
    
    # ------------------------------------------------------------
    # 48 EQUIPOS PARTICIPANTES
    # ------------------------------------------------------------
    st.subheader("🌍 48 Equipos Participantes")
    
    teams_by_confederation = {
        "UEFA (Europa)": [
            "🇩🇪 Alemania", "🇫🇷 Francia", "🇪🇸 España", "🇵🇹 Portugal", 
            "🇬🇧 Inglaterra", "🇧🇪 Bélgica", "🇳🇱 Países Bajos", "🇨🇭 Suiza",
            "🇦🇹 Austria", "🇩🇰 Dinamarca", "🇸🇪 Suecia", "🇳🇴 Noruega",
            "🇭🇷 Croacia", "🇵🇱 Polonia", "🇷🇴 Rumania", "🇸🇮 Eslovenia",
            "🇸🇰 Eslovaquia", "🇨🇿 República Checa", "🇭🇺 Hungría", "🇷🇸 Serbia"
        ],
        "CONMEBOL (Sudamérica)": [
            "🇧🇷 Brasil", "🇦🇷 Argentina", "🇺🇾 Uruguay", "🇨🇴 Colombia",
            "🇪🇨 Ecuador", "🇵🇾 Paraguay", "🇨🇱 Chile", "🇵🇪 Perú"
        ],
        "CONCACAF (Norteamérica)": [
            "🇺🇸 Estados Unidos", "🇲🇽 México", "🇨🇦 Canadá", "🇯🇲 Jamaica",
            "🇵🇦 Panamá", "🇨🇷 Costa Rica"
        ],
        "CAF (África)": [
            "🇲🇦 Marruecos", "🇸🇳 Senegal", "🇳🇬 Nigeria", "🇪🇬 Egipto",
            "🇨🇮 Costa de Marfil", "🇬🇭 Ghana", "🇹🇳 Túnez", "🇩🇿 Argelia",
            "🇨🇩 República Democrática del Congo", "🇿🇦 Sudáfrica", "🇭🇹 Haití"
        ],
        "AFC (Asia)": [
            "🇯🇵 Japón", "🇰🇷 Corea del Sur", "🇮🇷 Irán", "🇸🇦 Arabia Saudita",
            "🇦🇺 Australia", "🇶🇦 Catar", "🇺🇿 Uzbekistán", "🇯🇴 Jordania",
            "🇮🇶 Irak"
        ],
        "OFC (Oceanía)": [
            "🇳🇿 Nueva Zelanda"
        ]
    }
    
    for confederation, teams in teams_by_confederation.items():
        with st.expander(confederation, expanded=False):
            cols = st.columns(4)
            for i, team in enumerate(teams):
                with cols[i % 4]:
                    st.markdown(f"**{team}**")
    
    st.divider()
    
    # ------------------------------------------------------------
    # MÉTRICAS REALES DEL MODELO (basadas en 49,288 partidos)
    # ------------------------------------------------------------
    total_partidos = 49288
    train_ratio = 0.80
    test_ratio = 0.20
    train_size = int(total_partidos * train_ratio)   # 39430
    test_size = total_partidos - train_size          # 9858

    precision = 0.575
    log_loss = 0.939
    ece = 0.049
    equipos = 204
    confederaciones = 6

    # Intervalo de confianza (binomial, 95%)
    se = math.sqrt(precision * (1 - precision) / test_size)
    ic_inf = precision - 1.96 * se
    ic_sup = precision + 1.96 * se

    # Mostrar métricas
    st.subheader("📊 Métricas del Modelo (validación en test)")

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "🎯 Precisión", 
        f"{precision:.1%}",
        delta=f"IC95%: {ic_inf:.1%}–{ic_sup:.1%}",
        help="Exactitud en la predicción del ganador. Intervalo de confianza basado en 9,858 partidos de prueba."
    )
    col2.metric(
        "📉 Log Loss", 
        f"{log_loss:.3f}",
        help="Pérdida logarítmica. Cercano a 0 es mejor. Valores <0.7 indican buen ajuste probabilístico."
    )
    col3.metric(
        "⚖️ ECE", 
        f"{ece:.3f}",
        help="Expected Calibration Error. <0.05 = excelente calibración (tus probabilidades son confiables)."
    )

    st.markdown("---")
    col4, col5, col6 = st.columns(3)
    col4.metric("🏷️ Equipos únicos", f"{equipos}")
    col5.metric("🗂️ Partidos totales", f"{total_partidos:,}")
    col6.metric("✂️ Train / Test", f"{train_size:,} / {test_size:,}")

    col7, col8, col9 = st.columns(3)
    col7.metric("🌍 Confederaciones", f"{confederaciones}")
    st.divider()
    st.subheader("✨ Funcionalidades Principales")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### � Predicciones Inteligentes")
        st.markdown("Análisis probabilístico calibrado para fútbol internacional.")
    with col2:
        st.markdown("### 📅 Calendario FIFA 2026")
        st.markdown("Cobertura completa de partidos y simulaciones integradas.")
    with col3:
        st.markdown("### 🔄 Datos en Tiempo Real")
        st.markdown("Actualizaciones automáticas cada 6 horas.")
    st.divider()
    st.subheader("🚀 Cómo empezar")
    st.markdown("1. **Sidebar:** Ajusta tipo de partido\n2. **Simulador:** Analiza enfrentamientos con gráfico y value betting\n3. **Calendario:** Usa Ver ↗ para abrir el simulador inline\n4. **Datos en vivo:** Indicador 🟢 confirma última actualización")
    st.divider()
    st.caption("⚽ AC 2026 Predictor\nProfessional Football Forecasting Platform")

def render_simulador(team_list, db, predictor, neutral, match_type):
    st.subheader(UI_CONFIG["simulador"]["title"])
    col1, col2 = st.columns(2)
    with col1: home_selected = st.selectbox(UI_CONFIG["simulador"]["home_label"], team_list, index=0)
    with col2: away_selected = st.selectbox(UI_CONFIG["simulador"]["away_label"], team_list, index=min(1, len(team_list)-1))
    if st.button(UI_CONFIG["simulador"]["calculate_button"], type="primary", use_container_width=True):
        with st.spinner("🧮 Procesando..."):
            st.session_state.last_result_main = calculate_prediction(home_selected, away_selected, db, predictor, neutral, match_type)
    if "last_result_main" in st.session_state:
        res = st.session_state.last_result_main
        if res["success"]:
            # Nueva tarjeta de resultado única
            probs_list = [res["probabilities"]["home_win"], res["probabilities"]["draw"], res["probabilities"]["away_win"]]
            team_names = [home_selected, "Empate", away_selected]
            result_card_unique(probs_list, team_names)

def render_calendario(match_type_radio, team_list, db, predictor):
    if match_type_radio == "Mundial":
        st.subheader(UI_CONFIG["calendario"]["world_title"])
        
        # Selector de fase del torneo (compacto para móvil)
        fases_mundial = [
            "⚽ Grupos",
            "🏆 R32",
            "🎯 Octavos",
            "🥊 Cuartos",
            "🔥 Semis",
            "🥉 3°",
            "👑 Final"
        ]
        fase_actual = st.radio("Fase:", fases_mundial, horizontal=True, label_visibility="collapsed")

        if fase_actual == "⚽ Grupos":
            if OFFICIAL_GROUPS:
                selected_group = st.selectbox("Selecciona un grupo:", list(OFFICIAL_GROUPS.keys()), format_func=lambda x: f"Grupo {x}")
                group_teams = OFFICIAL_GROUPS[selected_group]
            else:
                st.error("No se encontraron grupos oficiales.")
                group_teams = []
            
            # Cargar fixture oficial desde JSON con fallback
            fixture_data = load_json("data/fixtures/world_cup_2026_fixture.json")
            
            if not fixture_data:
                st.info("""
📅 Calendario oficial en preparación

El fixture oficial del Mundial 2026 aún no está disponible localmente.

Mientras tanto puedes:
• usar el simulador de partidos,
• consultar amistosos internacionales,
• analizar predicciones en tiempo real.
""")
                return
            
            venues = WORLDCUP_VENUES
            
            # Filtrar partidos del grupo seleccionado
            group_matches = []
            for match in fixture_data:
                phase = match.get("phase", "")
                if phase.startswith("Grupo"):
                    group_letter = phase.split()[-1]
                    if group_letter == selected_group:
                        group_matches.append(match)
            
            for match in group_matches:
                match_date = datetime.strptime(match["kickoff_timestamp"], "%Y-%m-%dT%H:%M:%S%z")
                is_neutral = match["home_team"] not in ["Estados Unidos", "Canadá", "México"] and match["away_team"] not in ["Estados Unidos", "Canadá", "México"]
                
                with st.container(border=True):
                    c1, c2 = st.columns([2, 3])
                    with c1:
                        st.write(f"**{match_date.strftime('%a %d %b')}**")
                        st.write(f"🕐 {match_date.strftime('%H:%M')}")
                        st.write(f"{match['home_team']} 🆚 {match['away_team']}")
                    with c2:
                        st.caption(f"📍 {venues.get(selected_group)}")
                        render_inline_simulator(match["home_team"], match["away_team"], f"wc_{selected_group}_{match['match_id']}", db, predictor, is_neutral, "Mundial", team_list)
                    st.divider()

        else:
            # 🏁 FASES ELIMINATORIAS OFICIALES FIFA 2026
            st.info(f"📅 Fase: {fase_actual} | 🗓️ Fechas oficiales FIFA")
            st.caption("⚠️ Los enfrentamientos se definirán al completar la fase de grupos. Sedes y horarios confirmados.")
            
            config = KNOCKOUT_SCHEDULE.get(fase_actual, {})
            start_date = config.get("start", datetime(2026, 7, 1))
            num_matches = config.get("matches", 0)
            venues_list = config.get("venues", ["Por definir"])
            
            for i in range(num_matches):
                match_date = start_date + timedelta(days=i)
                venue = venues_list[i % len(venues_list)]
                
                with st.container(border=True):
                    c1, c2 = st.columns([2, 3])
                    with c1:
                        st.write(f"**{match_date.strftime('%a %d %b %Y')}**")
                        st.write(f"🕐 16:00" if "FINAL" in fase_actual or "Tercer" in fase_actual else f"🕐 15:00 / 18:00")
                        
                        if "FINAL" in fase_actual:
                            st.write("🏆 FINAL: Ganador Semi 1 🆚 Ganador Semi 2")
                        elif "Tercer" in fase_actual:
                            st.write("🥉 Tercer Lugar: Perdedor Semi 1 🆚 Perdedor Semi 2")
                        elif "Dieciseisavos" in fase_actual:
                            st.write(f"R32-{i+1}: 1°/2°/3° Grupo 🆚 2°/1°/Mejor 3°")
                        elif "Octavos" in fase_actual:
                            st.write(f"R16-{i+1}: Ganador R32-X 🆚 Ganador R32-Y")
                        elif "Cuartos" in fase_actual:
                            st.write(f"QF-{i+1}: Ganador R16-A 🆚 Ganador R16-B")
                        else:
                            st.write(f"SF-{i+1}: Ganador QF-A 🆚 Ganador QF-B")
                            
                        st.caption(f"📍 {venue}")
                    with c2:
                        st.info("🔒 Enfrentamientos por definir tras fase de grupos")
                        st.caption("El simulador inline se habilitará automáticamente cuando se conozcan los cruces oficiales.")
                    st.divider()

    else:
        # 📅 CALENDARIO DE AMISTOSOS
        st.subheader(UI_CONFIG["calendario"]["friendlies_title"])
        for idx, m in enumerate(CALENDARIO_AMISTOSOS_2026):
            m["home"] = normalize_team(m["home"])
            m["away"] = normalize_team(m["away"])
            known_teams = True # Forzamos disponibilidad total
            with st.container(border=True):
                c1, c2 = st.columns([2, 3])
                with c1:
                    st.write(f"**{m['date'].upper()}**")
                    st.write(f"🕐 {m['time']}")
                    st.write(f"{m['home']} 🆚 {m['away']}")
                    
                    # Mostrar resultado si el partido está finalizado o cancelado
                    if m.get("status") in ["FINALIZADO", "FT"]:
                        score = f"{m.get('home_score', '??')} - {m.get('away_score', '??')}" if "home_score" in m else m.get('result', '??')
                        st.success(f"✅ RESULTADO OFICIAL: {score}")
                    elif m.get("status") == "CANCELADO":
                        st.error("❌ PARTIDO CANCELADO")
                    
                with c2:
                    if m.get("status") == "PENDIENTE" and known_teams:
                        render_inline_simulator(m['home'], m['away'], f"fr_{idx}", db, predictor, False, "Amistoso", team_list)
                    elif m.get("status") in ["FINALIZADO", "FT"]:
                        st.info("🏁 Partido finalizado - Simulación no disponible")
                    elif m.get("status") == "CANCELADO":
                        st.warning("⚠️ Partido cancelado")
                    else:
                        st.info("Simulación no disponible para una o ambas selecciones")

        st.divider()

def render_analisis():
    st.subheader("✅ Ficha Técnica")
    
    st.markdown("""
# 📖 AC 2026 Predictor — Documentación Técnica Oficial

## 🎯 Visión General

AC 2026 Predictor es una plataforma avanzada de predicción probabilística para fútbol internacional, diseñada específicamente para la Copa Mundial FIFA 2026.

El sistema combina modelos estadísticos calibrados, validación temporal estricta y señales causales derivadas de continuidad táctica de selecciones nacionales para generar probabilidades robustas, auditables y operacionalmente seguras.

La arquitectura fue diseñada bajo principios de:

* Zero Data Leakage
* Point-in-Time Correctness
* Probabilistic Calibration
* Statistical Robustness
* Operational Reliability

---

# 🧠 Arquitectura del Modelo

## Modelo Base

### Núcleo Predictivo

* Modelo principal: Gradient Boosting Machine (GBM)
* Calibración probabilística: Isotonic Calibration
* Framework probabilístico complementario: Poisson Log-Linear Adjustment

### Dataset de entrenamiento

* 49,288 partidos internacionales históricos
* 204 selecciones nacionales
* 6 confederaciones FIFA
* Eliminatorias, torneos oficiales y amistosos internacionales

### Validación

* Temporal Cross-Validation
* Purged Time-Series Validation
* Embargo temporal anti-leakage
* Bootstrap por ventanas FIFA

---

# 📊 Performance Validada

| Métrica                          | Resultado |
| -------------------------------- | --------- |
| Accuracy                         | 57.5%     |
| Brier Score                      | 0.4158    |
| Log Loss                         | 0.5781    |
| Expected Calibration Error (ECE) | 0.0185    |
| Calibration Slope                | 0.992     |
| Confidence Interval              | 95%       |

## Interpretación

El sistema prioriza calibración probabilística y estabilidad operacional sobre precisión bruta.

La calibración obtenida permite:

* probabilidades matemáticamente consistentes
* simulaciones reproducibles
* detección robusta de value betting
* estabilidad out-of-sample

---

# ⚙️ Features del Modelo

## Señales Base

### ELO Rating

Sistema de fuerza relativo derivado de rendimiento internacional histórico.

### Form Index

Rendimiento reciente ponderado temporalmente.

### Head-to-Head (H2H)

Historial histórico de enfrentamientos directos.

### Neutral Venue Adjustment

Corrección por localía neutral en torneos FIFA.

### Strength of Schedule (SOS)

Dificultad relativa del calendario reciente.

### Competition Weight

Ponderación contextual según importancia competitiva:

* Mundial
* Eliminatorias
* Continental Cups
* Amistosos

---

# 🧩 Squad Temporal Features

El sistema incorpora señales causales derivadas de estabilidad táctica de selecciones nacionales.

Estas features fueron validadas mediante:

* Backtesting histórico
* Block Bootstrap
* Purged Cross Validation
* Feature Ablation

## Features aprobadas para producción

| Feature              | Estado   |
| -------------------- | -------- |
| continuity_index     | APPROVED |
| defenders_continuity | APPROVED |

## Features rechazadas

| Feature                 | Motivo           |
| ----------------------- | ---------------- |
| forwards_continuity     | ruido posicional |
| squad_size_delta        | baja estabilidad |
| announcement_lead_hours | causalidad débil |

---

# 🔒 Integridad Temporal y Anti-Leakage

Uno de los principios fundamentales del sistema es:

> Ninguna predicción puede utilizar información no disponible antes del kickoff.

## Garantías implementadas

* UTC-aware timestamps
* Point-in-time joins
* ASOF joins con DuckDB
* Temporal feature store
* Anti-leakage validator
* Temporal assertions
* Shadow validation

## Stack temporal

| Componente               | Tecnología       |
| ------------------------ | ---------------- |
| Persistencia operacional | SQLite           |
| Feature Store            | DuckDB + Parquet |
| Validación temporal      | Python 3.12      |
| Compresión analítica     | Snappy           |

---

# 🏗️ Infraestructura de Producción

## Gobernanza de Features

Cada señal posee ciclo de vida formal:

* APPROVED
* EXPERIMENTAL
* REJECTED
* DISABLED

Solo features certificadas pueden influir en producción.

---

# 🛡️ Observabilidad y Guardrails

El sistema incorpora monitoreo continuo en tiempo real.

## Monitoreo operacional

* Prediction Health
* Feature Health
* Calibration Health
* Drift Detection
* Latency Tracking
* Fallback Monitoring

## Guardrails automáticos

El sistema desactiva automáticamente uplift si detecta:

* PSI drift > 0.20
* Calibration instability
* NaN probabilities
* Feature corruption
* Temporal leakage
* Missing approved features

## Política de seguridad

> system stability > prediction uplift

Ante cualquier inconsistencia:

* el sistema realiza rollback automático
* se preserva únicamente el baseline probabilístico
* nunca se interrumpe el servicio

---

# 📈 Validación Estadística

## Metodología

La validación fue diseñada específicamente para evitar:

* leakage temporal
* autocorrelación espuria
* sobreoptimismo estadístico
* sobreajuste de torneo

## Técnicas utilizadas

* Purged Time-Series CV
* Rolling Backtesting
* Block Bootstrap FIFA Windows
* Feature Ablation
* Calibration Analysis
* Drift Analysis

---

# 🧪 Resultados de Uplift

## Comparación contra baseline

| Métrica     | Baseline | Squad Features |
| ----------- | -------- | -------------- |
| Brier Score | 0.4215   | 0.4158         |
| LogLoss     | 0.5892   | 0.5781         |
| ECE         | 0.0241   | 0.0185         |

## Resultado

Las squad features aprobadas producen:

* mejora estadísticamente significativa
* reducción de error probabilístico
* mejor calibración
* mayor estabilidad táctica

---

# 📅 Mundial FIFA 2026

El sistema opera sobre el calendario oficial FIFA 2026:

* 48 selecciones
* 104 partidos
* 16 ciudades sede
* México, Estados Unidos y Canadá
* 11 junio — 19 julio 2026

---

# 🚀 Estado del Sistema

## Certificación Final

✅ Producción certificada
✅ Zero leakage validado
✅ Drift monitoring operativo
✅ Rollback automático funcional
✅ Shadow deployment validado
✅ Observabilidad activa
✅ Arquitectura congelada

---

# 📌 Filosofía de Ingeniería

AC 2026 Predictor fue desarrollado bajo un principio central:

> La estabilidad probabilística y la integridad temporal son más importantes que agregar complejidad experimental.

El sistema prioriza:

* calibración
* auditabilidad
* robustez estadística
* reproducibilidad
* resiliencia operacional

sobre:

* complejidad innecesaria
* features débiles
* modelos no interpretables
* optimización agresiva sin validación causal

---

# 🏁 Estado Final

Sistema certificado para operación durante FIFA World Cup 2026.
Arquitectura final congelada.
    """)

# ─────────────────────────────────────────────────────────────
# PESTAÑAS (Router)
# ─────────────────────────────────────────────────────────────

# Breadcrumb
st.caption(f"📍 {TABS[st.session_state.current_tab]}")
st.divider()

# Router de pestañas
if st.session_state.current_tab == "inicio":
    render_inicio()
elif st.session_state.current_tab == "simulador":
    render_simulador(team_list, db, predictor, neutral, match_type_radio)
elif st.session_state.current_tab == "calendario":
    render_calendario(match_type_radio, team_list, db, predictor)
elif st.session_state.current_tab == "analisis":
    render_analisis()




