# Script para descargar datos climáticos de las 16 sedes del Mundial 2026
# Usa Open-Meteo API (gratuito, sin API key)

import requests
import pandas as pd
from datetime import datetime
import time

# Coordenadas de las 16 sedes del Mundial 2026
VENUES = {
    "Atlanta": {"lat": 33.7490, "lon": -84.3880, "timezone": "America/New_York", "country": "USA"},
    "Boston": {"lat": 42.3601, "lon": -71.0589, "timezone": "America/New_York", "country": "USA"},
    "Dallas": {"lat": 32.7767, "lon": -96.7970, "timezone": "America/Chicago", "country": "USA"},
    "Houston": {"lat": 29.7604, "lon": -95.3698, "timezone": "America/Chicago", "country": "USA"},
    "Kansas City": {"lat": 39.0997, "lon": -94.5786, "timezone": "America/Chicago", "country": "USA"},
    "Los Angeles": {"lat": 34.0522, "lon": -118.2437, "timezone": "America/Los_Angeles", "country": "USA"},
    "Miami": {"lat": 25.7617, "lon": -80.1918, "timezone": "America/New_York", "country": "USA"},
    "New York/New Jersey": {"lat": 40.7128, "lon": -74.0060, "timezone": "America/New_York", "country": "USA"},
    "Philadelphia": {"lat": 39.9526, "lon": -75.1652, "timezone": "America/New_York", "country": "USA"},
    "San Francisco": {"lat": 37.7749, "lon": -122.4194, "timezone": "America/Los_Angeles", "country": "USA"},
    "Seattle": {"lat": 47.6062, "lon": -122.3321, "timezone": "America/Los_Angeles", "country": "USA"},
    "Toronto": {"lat": 43.6510, "lon": -79.3470, "timezone": "America/Toronto", "country": "Canada"},
    "Vancouver": {"lat": 49.2827, "lon": -123.1207, "timezone": "America/Vancouver", "country": "Canada"},
    "Guadalajara": {"lat": 20.6597, "lon": -103.3496, "timezone": "America/Mexico_City", "country": "Mexico"},
    "Mexico City": {"lat": 19.4326, "lon": -99.1332, "timezone": "America/Mexico_City", "country": "Mexico"},
    "Monterrey": {"lat": 25.6866, "lon": -100.3161, "timezone": "America/Monterrey", "country": "Mexico"}
}

def descargar_clima_sede(city_name, venue_data, start_date, end_date):
    """
    Descarga datos climáticos para una sede específica.
    
    Args:
        city_name: Nombre de la ciudad
        venue_data: Diccionario con lat, lon, timezone, country
        start_date: Fecha inicio (YYYY-MM-DD)
        end_date: Fecha fin (YYYY-MM-DD)
    
    Returns:
        DataFrame con datos climáticos
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": venue_data["lat"],
        "longitude": venue_data["lon"],
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m,sunshine_duration",
        "timezone": venue_data["timezone"]
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Crear DataFrame
        df = pd.DataFrame(data['daily'])
        df['date'] = pd.to_datetime(data['daily']['time'])
        df['city'] = city_name
        df['country'] = venue_data["country"]
        df['latitude'] = venue_data["lat"]
        df['longitude'] = venue_data["lon"]
        
        print(f"✅ {city_name}: {len(df)} días descargados")
        return df
        
    except Exception as e:
        print(f"❌ Error descargando {city_name}: {e}")
        return None

def descargar_todas_sedes(start_date="2024-06-01", end_date="2024-08-31"):
    """
    Descarga datos climáticos para todas las sedes del Mundial 2026.
    
    Args:
        start_date: Fecha inicio (YYYY-MM-DD)
        end_date: Fecha fin (YYYY-MM-DD)
    
    Returns:
        DataFrame combinado con todas las sedes
    """
    all_data = []
    
    print(f"🌦️ Descargando clima para {len(VENUES)} sedes...")
    print(f"📅 Período: {start_date} a {end_date}\n")
    
    for city_name, venue_data in VENUES.items():
        df = descargar_clima_sede(city_name, venue_data, start_date, end_date)
        if df is not None:
            all_data.append(df)
        # Pausa para no sobrecargar la API
        time.sleep(0.5)
    
    if all_data:
        df_combined = pd.concat(all_data, ignore_index=True)
        
        # Guardar CSV fusionado
        output_path = "data/clima/clima_fusionado.csv"
        df_combined.to_csv(output_path, index=False)
        print(f"\n✅ Datos guardados en: {output_path}")
        print(f"📊 Total registros: {len(df_combined)}")
        
        # Guardar CSV individuales por sede
        for city in VENUES.keys():
            df_city = df_combined[df_combined['city'] == city]
            city_filename = city.lower().replace(" ", "_").replace("/", "_")
            df_city.to_csv(f"data/clima/clima_{city_filename}_{start_date}_{end_date}.csv", index=False)
        
        print(f"📁 CSVs individuales guardados en data/clima/")
        
        return df_combined
    else:
        print("❌ No se pudo descargar ningún dato")
        return None

if __name__ == "__main__":
    # Descargar clima para el período del Mundial 2026 (junio-agosto)
    print("=" * 60)
    print("🌍 DESCARGA DE CLIMA - MUNDIAL 2026")
    print("=" * 60)
    
    df_clima = descargar_todas_sedes("2024-06-01", "2024-08-31")
    
    if df_clima is not None:
        print("\n📊 Resumen por sede:")
        print(df_clima.groupby('city').size())
