# monitor.py
import hashlib
import json
import os
import time
import logging
logger = logging.getLogger(__name__)  # ← Agregar esta línea
from datetime import datetime
from scraper import scrape_all_squads
from cleaner import clean_and_normalize
from exporter import export_datasets, export_summary_report
from config import MONITOR_CHECK_INTERVAL_HOURS, MONITOR_HASH_FILE, OUTPUT_DIR

def compute_data_hash(df) -> str:
    """Calcula hash del DataFrame para detectar cambios"""
    if df.empty:
        return ""
    csv_str = df.to_csv(index=False).encode()
    return hashlib.sha256(csv_str).hexdigest()

def load_last_hash() -> str:
    """Carga hash de la última ejecución"""
    try:
        if os.path.exists(MONITOR_HASH_FILE):
            with open(MONITOR_HASH_FILE, "r") as f:
                return json.load(f).get("hash", "")
    except:
        pass
    return ""

def save_current_hash(hash_val: str):
    """Guarda hash actual para próxima comparación"""
    os.makedirs(os.path.dirname(MONITOR_HASH_FILE), exist_ok=True)
    with open(MONITOR_HASH_FILE, "w") as f:
        json.dump({"hash": hash_val, "timestamp": datetime.now().isoformat()}, f)

def run_monitor_cycle() -> bool:
    """Ejecuta un ciclo de monitoreo: scrape → comparar → exportar si hay cambios"""
    logger.info("🔍 Iniciando ciclo de monitoreo...")
    
    # 1. Scrapear datos actuales
    df_raw = scrape_all_squads()
    if df_raw.empty:
        logger.warning("⚠️ No se encontraron datos. Verificar fuentes.")
        return False
    
    # 2. Limpiar y normalizar
    df_clean = clean_and_normalize(df_raw)
    
    # 3. Calcular hash y comparar con último
    current_hash = compute_data_hash(df_clean)
    last_hash = load_last_hash()
    
    if current_hash == last_hash and last_hash:
        logger.info("✅ Sin cambios detectados. Datos actualizados.")
        return True
    
    # 4. ¡Hay cambios! Exportar nuevos datos
    logger.info("🆕 ¡Nuevos datos detectados! Exportando...")
    export_datasets(df_clean)
    
    # 5. Guardar nuevo hash
    save_current_hash(current_hash)
    
    # 6. Log resumen
    summary = export_summary_report(df_clean)
    logger.info(f"📋 {summary}")
    
    return True

def start_monitor(check_interval_hours: int = None):
    """Inicia monitor en bucle infinito"""
    interval = check_interval_hours or MONITOR_CHECK_INTERVAL_HOURS
    logger.info(f"🔄 Monitor activo. Revisando cada {interval}h...")
    
    while True:
        try:
            run_monitor_cycle()
        except Exception as e:
            logger.error(f"❌ Error en ciclo de monitor: {e}")
        
        logger.info(f"⏳ Esperando {interval}h para próximo ciclo...")
        time.sleep(interval * 3600)

if __name__ == "__main__":
    import sys
    if "--monitor" in sys.argv:
        start_monitor()
    else:
        # Ejecución única
        run_monitor_cycle()
