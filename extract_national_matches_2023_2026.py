import pandas as pd
import requests
import io
from datetime import datetime

def download_historical_dataset():
    """
    Download the historical international matches dataset from GitHub.
    Source: https://github.com/martj42/international_matches
    """
    print("Descargando dataset histórico de partidos internacionales...")
    
    # URL del dataset histórico de GitHub
    url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Leer el CSV directamente desde la respuesta
        df = pd.read_csv(io.StringIO(response.text))
        print(f"Dataset descargado: {df.shape[0]} partidos totales")
        return df
    except Exception as e:
        print(f"Error al descargar el dataset: {e}")
        return None

def filter_matches_2023_2026(df):
    """
    Filtrar partidos de selecciones nacionales desde 2023 hasta 2026.
    """
    print("Filtrando partidos de 2023-2026...")
    
    # Convertir fecha a datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Filtrar por fecha (2023-2026)
    start_date = pd.to_datetime('2023-01-01')
    end_date = pd.to_datetime('2026-12-31')
    
    df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    
    print(f"Partidos encontrados en rango 2023-2026: {df_filtered.shape[0]}")
    return df_filtered

def add_world_cup_cycle(df):
    """
    Agregar columna world_cup_cycle basada en el año del partido.
    """
    def get_world_cup_cycle(date):
        year = date.year
        if 2023 <= year <= 2026:
            return '2026'
        else:
            return 'unknown'
    
    df['world_cup_cycle'] = df['date'].apply(get_world_cup_cycle)
    return df

def standardize_columns(df):
    """
    Estandarizar columnas al formato solicitado.
    """
    print("Estandarizando columnas...")
    
    # Mapeo de columnas del dataset original al formato solicitado
    column_mapping = {
        'date': 'date',
        'home_team': 'home_team',
        'away_team': 'away_team',
        'home_score': 'home_score',
        'away_score': 'away_score',
        'tournament': 'tournament',
        'country': 'country',
        'city': 'city',
        'neutral': 'neutral'
    }
    
    # Seleccionar solo las columnas necesarias
    columns_needed = ['date', 'home_team', 'away_team', 'home_score', 'away_score', 
                     'tournament', 'country', 'city', 'neutral']
    
    df_standardized = df[columns_needed].copy()
    
    return df_standardized

def add_confederation(df):
    """
    Agregar columna confederation basada en el país o torneo.
    """
    def get_confederation(row):
        country = row['country']
        tournament = row['tournament']
        
        # Mapeo simplificado de confederaciones por países principales
        conmebol_countries = ['Argentina', 'Brazil', 'Uruguay', 'Chile', 'Paraguay', 
                             'Bolivia', 'Peru', 'Ecuador', 'Colombia', 'Venezuela']
        
        uefa_countries = ['Germany', 'France', 'Spain', 'Italy', 'England', 'Netherlands',
                         'Portugal', 'Belgium', 'Croatia', 'Switzerland', 'Austria',
                         'Poland', 'Ukraine', 'Russia', 'Turkey', 'Czech Republic']
        
        concacaf_countries = ['Mexico', 'United States', 'Canada', 'Costa Rica', 'Panama',
                             'Jamaica', 'Honduras', 'El Salvador', 'Guatemala']
        
        afc_countries = ['Japan', 'South Korea', 'Australia', 'Iran', 'Saudi Arabia',
                        'UAE', 'Qatar', 'China', 'Iraq', 'Oman']
        
        caf_countries = ['Egypt', 'Nigeria', 'South Africa', 'Morocco', 'Algeria',
                        'Ivory Coast', 'Ghana', 'Senegal', 'Tunisia', 'Cameroon']
        
        ofc_countries = ['New Zealand', 'Fiji', 'Papua New Guinea', 'Solomon Islands']
        
        if country in conmebol_countries:
            return 'CONMEBOL'
        elif country in uefa_countries:
            return 'UEFA'
        elif country in concacaf_countries:
            return 'CONCACAF'
        elif country in afc_countries:
            return 'AFC'
        elif country in ofc_countries:
            return 'OFC'
        elif country in caf_countries:
            return 'CAF'
        else:
            # Intentar inferir por el torneo
            if 'Copa América' in tournament or 'CONMEBOL' in tournament:
                return 'CONMEBOL'
            elif 'UEFA' in tournament or 'Euro' in tournament:
                return 'UEFA'
            elif 'CONCACAF' in tournament or 'Gold Cup' in tournament:
                return 'CONCACAF'
            elif 'AFC' in tournament or 'Asian Cup' in tournament:
                return 'AFC'
            elif 'CAF' in tournament or 'Africa Cup' in tournament:
                return 'CAF'
            else:
                return 'Unknown'
    
    df['confederation'] = df.apply(get_confederation, axis=1)
    return df

def finalize_dataset(df):
    """
    Ordenar columnas y preparar dataset final.
    """
    print("Preparando dataset final...")
    
    # Ordenar por fecha
    df = df.sort_values('date')
    
    # Columnas en el orden solicitado
    final_columns = ['date', 'home_team', 'away_team', 'home_score', 'away_score',
                     'tournament', 'country', 'city', 'neutral', 'confederation',
                     'world_cup_cycle']
    
    df_final = df[final_columns].copy()
    
    return df_final

def main():
    """
    Función principal para extraer y procesar los datos.
    """
    print("=" * 60)
    print("Extracción de partidos de selecciones nacionales 2023-2026")
    print("=" * 60)
    
    # 1. Descargar dataset histórico
    df = download_historical_dataset()
    if df is None:
        print("No se pudo descargar el dataset. Abortando.")
        return
    
    # 2. Filtrar por fecha (2023-2026)
    df_filtered = filter_matches_2023_2026(df)
    
    # 3. Estandarizar columnas
    df_standardized = standardize_columns(df_filtered)
    
    # 4. Agregar columna world_cup_cycle
    df_with_cycle = add_world_cup_cycle(df_standardized)
    
    # 5. Agregar columna confederation
    df_with_confed = add_confederation(df_with_cycle)
    
    # 6. Preparar dataset final
    df_final = finalize_dataset(df_with_confed)
    
    # 7. Guardar CSV
    output_file = "national_matches_2023_2026.csv"
    df_final.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print("=" * 60)
    print(f"CSV generado exitosamente: {output_file}")
    print(f"Total de partidos: {df_final.shape[0]}")
    print(f"Columnas: {list(df_final.columns)}")
    print("=" * 60)
    
    # Mostrar estadísticas básicas
    print("\nEstadísticas del dataset:")
    print(f"- Rango de fechas: {df_final['date'].min()} a {df_final['date'].max()}")
    print(f"- Torneos únicos: {df_final['tournament'].nunique()}")
    print(f"- Países únicos: {df_final['country'].nunique()}")
    print(f"- Selecciones únicas: {df_final['home_team'].nunique()}")
    
    print("\nDistribución por confederación:")
    print(df_final['confederation'].value_counts())
    
    print("\nDistribución por torneo (top 10):")
    print(df_final['tournament'].value_counts().head(10))
    
    # Mostrar primeras filas
    print("\nPrimeras 5 filas del dataset:")
    print(df_final.head())

if __name__ == "__main__":
    main()
