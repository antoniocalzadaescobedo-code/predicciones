# scraper.py
import requests
import re
import logging
logger = logging.getLogger(__name__)  # ← Agregar esta línea
from bs4 import BeautifulSoup
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config import HEADERS, SQUAD_SOURCES, FIFA_2026_GROUPS, MAX_RETRIES, RETRY_MIN_WAIT, RETRY_MAX_WAIT, REQUEST_TIMEOUT

@retry(stop=stop_after_attempt(MAX_RETRIES), 
       wait=wait_exponential(multiplier=1, min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
       retry=retry_if_exception_type(requests.RequestException))
def fetch_page(url: str) -> str:
    """Descarga página con retries automáticos y headers realistas"""
    logger.info(f"Descargando: {url}")
    resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.text

def parse_wikipedia_squads(html: str) -> pd.DataFrame:
    """
    Parsea convocatorias desde Wikipedia (estructura wikitable estable)
    Retorna DataFrame con columnas crudas para limpieza posterior
    """
    soup = BeautifulSoup(html, "lxml")
    records = []
    
    # Wikipedia usa tablas por grupo/país con clase "wikitable"
    tables = soup.find_all("table", class_="wikitable")
    
    for table in tables:
        # Detectar si es tabla de jugadores (buscar "Player" o "Jugador" en headers)
        headers = [th.text.strip().lower() for th in table.find_all("th")]
        if not any("player" in h or "jugador" in h or "no." in h for h in headers):
            continue
            
        # Procesar filas de datos
        rows = table.find_all("tr")[1:]  # Saltar header
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 3:  # Mínimo: número, posición, jugador
                continue
                
            try:
                raw_data = [c.get_text(" ", strip=True) for c in cells]
                
                # Heurística robusta para extraer campos
                player, position, club, age, number = None, None, None, None, None
                
                for txt in raw_data:
                    txt_clean = txt.strip()
                    if not txt_clean:
                        continue
                    # Número de dorsal (1-99)
                    if txt_clean.isdigit() and 1 <= int(txt_clean) <= 99 and number is None:
                        number = int(txt_clean)
                    # Posición (GK, DF, MF, FW o español)
                    elif any(p in txt_clean.upper() for p in ["GK", "DF", "MF", "FW", "PORT", "DEF", "MED", "DEL"]):
                        position = txt_clean.replace(" ", "").upper()
                    # Club (contiene ".", "FC", "United", etc.)
                    elif any(kw in txt_clean for kw in [".", "FC", "United", "City", "Racing", "Club", "Sporting"]):
                        club = txt_clean
                    # Jugador (texto largo, no numérico, sin @)
                    elif len(txt_clean) > 3 and not txt_clean.isdigit() and "@" not in txt_clean and player is None:
                        player = txt_clean.split("[")[0].strip()  # Remover referencias [1]
                
                # Extraer edad si existe en formato "(age XX)"
                age_match = re.search(r"\(age\s*(\d{1,2})\)", row.text)
                if age_match:
                    age = int(age_match.group(1))
                
                if player and position:
                    records.append({
                        "player_raw": player,
                        "position_raw": position,
                        "club_raw": club or "Unknown",
                        "age_raw": age,
                        "jersey_number_raw": number,
                        "source": "wikipedia"
                    })
            except Exception as e:
                logger.warning(f"Error parseando fila: {e}")
                continue
    
    logger.info(f"Wikipedia: {len(records)} jugadores extraidos")
    return pd.DataFrame(records) if records else pd.DataFrame()

def detect_squad_status(country: str, html: str) -> str:
    """Detecta si la convocatoria es preliminar o final"""
    text_lower = html.lower()
    if any(kw in text_lower for kw in ["final squad", "lista final", "convocatoria definitiva"]):
        return "Final"
    elif any(kw in text_lower for kw in ["preliminary", "preliminar", "lista preliminar"]):
        return "Preliminar"
    return "Final"  # Default seguro

def scrape_all_squads() -> pd.DataFrame:
    """Scrapea convocatorias de todos los países clasificados"""
    all_records = []
    
    # Fuente principal: Wikipedia (estructura consolidada)
    try:
        html = fetch_page(SQUAD_SOURCES["wikipedia"])
        df_raw = parse_wikipedia_squads(html)
        
        if not df_raw.empty:
            # Asignar metadatos por país (heurística por sección de HTML)
            # En producción, iterar por URLs específicas por país
            df_raw["country"] = "México"  # Ejemplo, en prod: parsear por sección
            df_raw["squad_status"] = detect_squad_status("México", html)
            df_raw["announcement_date"] = pd.Timestamp.now().strftime("%Y-%m-%d")
            all_records.append(df_raw)
    except Exception as e:
        logger.error(f"Error scraping Wikipedia: {e}")
    
    # Fallback: CSV local si scraping falla
    if not all_records:
        import os
        csv_path = "data/squads_fallback.csv"
        if os.path.exists(csv_path):
            logger.info(f"Usando fallback: {csv_path}")
            df_fallback = pd.read_csv(csv_path)
            all_records.append(df_fallback)
    
    return pd.concat(all_records, ignore_index=True) if all_records else pd.DataFrame()
