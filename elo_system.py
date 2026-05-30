"""
Sistema de Rating Elo para Selecciones de Fútbol
Implementación modular para predecir resultados del Mundial 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


# Configuración de factores K según tipo de partido
K_FACTORS = {
    'Mundial': 60,
    'Eliminatoria': 50,
    'CopaContinental': 45,
    'Amistoso': 30
}

# Rating inicial para todos los equipos
RATING_INICIAL = 1500

# Ventaja de local en puntos de rating
VENTAJA_LOCAL = 100


def calcular_resultado_esperado(rating_a: float, rating_b: float, ventaja_local: bool = False) -> float:
    """
    Calcula el resultado esperado (probabilidad de victoria) para el equipo A.
    
    Fórmula clásica de Elo: We = 1 / (1 + 10^((rating_rival - rating_equipo)/400))
    
    Args:
        rating_a: Rating del equipo A
        rating_b: Rating del equipo B
        ventaja_local: Si True, suma 100 puntos al rating del equipo A (ventaja de local)
    
    Returns:
        Probabilidad esperada de victoria del equipo A (0 a 1)
    """
    # Aplicar ventaja de local si corresponde
    if ventaja_local:
        rating_a += VENTAJA_LOCAL
    
    # Calcular diferencia de ratings
    diferencia = rating_b - rating_a
    
    # Aplicar fórmula de Elo
    we = 1 / (1 + 10 ** (diferencia / 400))
    
    return we


def ajustar_k_por_margen(k_base: float, diferencia_goles: int) -> float:
    """
    Ajusta el factor K según el margen de victoria.
    
    Multiplica K por (1 + (diferencia_goles * 0.1)), con un máximo de 2.0.
    
    Args:
        k_base: Factor K base según tipo de partido
        diferencia_goles: Diferencia de goles (valor absoluto)
    
    Returns:
        Factor K ajustado por margen de victoria
    """
    # Calcular multiplicador
    multiplicador = 1 + (abs(diferencia_goles) * 0.1)
    
    # Limitar a máximo de 2.0
    multiplicador = min(multiplicador, 2.0)
    
    k_ajustado = k_base * multiplicador
    
    return k_ajustado


def actualizar_ratings(df_ratings: pd.DataFrame, partido: pd.Series) -> pd.DataFrame:
    """
    Actualiza los ratings de ambos equipos después de un partido.
    
    Args:
        df_ratings: DataFrame con columnas 'equipo' y 'rating'
        partido: Serie con columnas: fecha, equipo_local, equipo_visitante, 
                 goles_local, goles_visitante, tipo_partido, sede
    
    Returns:
        DataFrame actualizado con los nuevos ratings
    """
    # Copiar el DataFrame para no modificar el original
    df_ratings = df_ratings.copy()
    
    # Extraer información del partido
    equipo_local = partido['equipo_local']
    equipo_visitante = partido['equipo_visitante']
    goles_local = partido['goles_local']
    goles_visitante = partido['goles_visitante']
    tipo_partido = partido['tipo_partido']
    sede = partido['sede']
    
    # Obtener ratings actuales
    rating_local = df_ratings[df_ratings['equipo'] == equipo_local]['rating'].values[0]
    rating_visitante = df_ratings[df_ratings['equipo'] == equipo_visitante]['rating'].values[0]
    
    # Determinar si es sede neutral
    es_neutral = sede.lower() == 'neutral'
    
    # Calcular resultado esperado para el equipo local
    we_local = calcular_resultado_esperado(rating_local, rating_visitante, ventaja_local=not es_neutral)
    we_visitante = 1 - we_local
    
    # Determinar resultado real
    if goles_local > goles_visitante:
        resultado_local = 1.0
        resultado_visitante = 0.0
        diferencia_goles = goles_local - goles_visitante
    elif goles_local < goles_visitante:
        resultado_local = 0.0
        resultado_visitante = 1.0
        diferencia_goles = goles_visitante - goles_local
    else:
        resultado_local = 0.5
        resultado_visitante = 0.5
        diferencia_goles = 0
    
    # Obtener factor K base según tipo de partido
    k_base = K_FACTORS.get(tipo_partido, 30)  # Default a Amistoso si no está definido
    
    # Ajustar K por margen de victoria
    k_ajustado = ajustar_k_por_margen(k_base, diferencia_goles)
    
    # Actualizar ratings
    nuevo_rating_local = rating_local + k_ajustado * (resultado_local - we_local)
    nuevo_rating_visitante = rating_visitante + k_ajustado * (resultado_visitante - we_visitante)
    
    # Actualizar en el DataFrame
    df_ratings.loc[df_ratings['equipo'] == equipo_local, 'rating'] = nuevo_rating_local
    df_ratings.loc[df_ratings['equipo'] == equipo_visitante, 'rating'] = nuevo_rating_visitante
    
    return df_ratings


def predecir_partido(equipo_a: str, equipo_b: str, rating_a: float, rating_b: float, 
                     neutral: bool = True) -> Dict[str, float]:
    """
    Predice las probabilidades de victoria, empate y derrota para un partido.
    
    Para partidos neutrales (como en Mundial), no se aplica ventaja de local.
    La probabilidad de empate se estima usando una distribución basada en la diferencia de ratings.
    
    Args:
        equipo_a: Nombre del equipo A
        equipo_b: Nombre del equipo B
        rating_a: Rating del equipo A
        rating_b: Rating del equipo B
        neutral: Si True, partido en sede neutral (sin ventaja de local)
    
    Returns:
        Diccionario con probabilidades: {'victoria_a': X, 'empate': Y, 'victoria_b': Z}
    """
    # Calcular probabilidad esperada de victoria para equipo A
    prob_a = calcular_resultado_esperado(rating_a, rating_b, ventaja_local=not neutral)
    prob_b = 1 - prob_a
    
    # Estimar probabilidad de empate
    # Usamos una fórmula basada en la cercanía de los ratings
    # Si los ratings son similares, mayor probabilidad de empate
    diferencia_ratings = abs(rating_a - rating_b)
    
    # Probabilidad base de empate disminuye con la diferencia de ratings
    prob_empate_base = 0.30  # 30% base
    factor_reduccion = min(diferencia_ratings / 200, 0.25)  # Máxima reducción de 25%
    prob_empate = prob_empate_base - factor_reduccion
    
    # Ajustar probabilidades para que sumen 1
    prob_a = prob_a * (1 - prob_empate)
    prob_b = prob_b * (1 - prob_empate)
    
    return {
        'victoria_a': prob_a,
        'empate': prob_empate,
        'victoria_b': prob_b
    }


def procesar_historial_partidos(df_partidos: pd.DataFrame) -> pd.DataFrame:
    """
    Procesa un historial completo de partidos y calcula los ratings finales.
    
    Args:
        df_partidos: DataFrame con columnas: fecha, equipo_local, equipo_visitante,
                     goles_local, goles_visitante, tipo_partido, sede
    
    Returns:
        DataFrame con ratings finales de todos los equipos
    """
    # Obtener todos los equipos únicos
    equipos_local = df_partidos['equipo_local'].unique()
    equipos_visitante = df_partidos['equipo_visitante'].unique()
    todos_equipos = set(equipos_local) | set(equipos_visitante)
    
    # Inicializar ratings
    df_ratings = pd.DataFrame({
        'equipo': list(todos_equipos),
        'rating': float(RATING_INICIAL)
    })
    
    # Ordenar partidos por fecha
    df_partidos_ordenados = df_partidos.sort_values('fecha')
    
    # Procesar cada partido
    for idx, partido in df_partidos_ordenados.iterrows():
        df_ratings = actualizar_ratings(df_ratings, partido)
    
    # Ordenar por rating descendente
    df_ratings = df_ratings.sort_values('rating', ascending=False).reset_index(drop=True)
    
    return df_ratings


def crear_datos_simulados() -> pd.DataFrame:
    """
    Crea un DataFrame con datos de partidos simulados para ejemplo.
    
    Returns:
        DataFrame con 10 partidos ficticios
    """
    datos = {
        'fecha': pd.date_range('2024-01-01', periods=10, freq='7D'),
        'equipo_local': ['Argentina', 'Brasil', 'Germany', 'France', 'Spain',
                        'England', 'Netherlands', 'Portugal', 'Italy', 'Belgium'],
        'equipo_visitante': ['Brazil', 'Argentina', 'France', 'Germany', 'Italy',
                            'Spain', 'Portugal', 'Netherlands', 'Spain', 'France'],
        'goles_local': [2, 1, 3, 2, 1, 2, 1, 2, 0, 2],
        'goles_visitante': [1, 2, 1, 2, 1, 1, 2, 1, 2, 1],
        'tipo_partido': ['Amistoso', 'Amistoso', 'Amistoso', 'Amistoso', 'Amistoso',
                        'Amistoso', 'Amistoso', 'Amistoso', 'Amistoso', 'Amistoso'],
        'sede': ['neutral', 'neutral', 'neutral', 'neutral', 'neutral',
                'neutral', 'neutral', 'neutral', 'neutral', 'neutral']
    }
    
    return pd.DataFrame(datos)


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SISTEMA DE RATING ELO PARA SELECCIONES DE FÚTBOL")
    print("=" * 70)
    print()
    
    # Crear datos simulados
    print("1. Creando datos simulados de partidos...")
    df_partidos = crear_datos_simulados()
    print(f"   Total de partidos: {len(df_partidos)}")
    print()
    
    # Mostrar primeros partidos
    print("   Primeros 5 partidos:")
    print(df_partidos.head())
    print()
    
    # Procesar historial
    print("2. Procesando historial de partidos...")
    df_ratings_finales = procesar_historial_partidos(df_partidos)
    print()
    
    # Mostrar ratings finales
    print("3. Ratings finales después de procesar todos los partidos:")
    print(df_ratings_finales)
    print()
    
    # Ejemplo de predicción para un partido del Mundial 2026
    print("4. Predicción para un partido del Mundial 2026 (sede neutral):")
    equipo_a = "Argentina"
    equipo_b = "Brazil"
    rating_a = df_ratings_finales[df_ratings_finales['equipo'] == equipo_a]['rating'].values[0]
    rating_b = df_ratings_finales[df_ratings_finales['equipo'] == equipo_b]['rating'].values[0]
    
    prediccion = predecir_partido(equipo_a, equipo_b, rating_a, rating_b, neutral=True)
    
    print(f"   {equipo_a} vs {equipo_b}")
    print(f"   Rating {equipo_a}: {rating_a:.1f}")
    print(f"   Rating {equipo_b}: {rating_b:.1f}")
    print(f"   Probabilidad victoria {equipo_a}: {prediccion['victoria_a']*100:.1f}%")
    print(f"   Probabilidad empate: {prediccion['empate']*100:.1f}%")
    print(f"   Probabilidad victoria {equipo_b}: {prediccion['victoria_b']*100:.1f}%")
    print()
    
    # Ejemplo con ventaja de local
    print("5. Predicción para el mismo partido con ventaja de local:")
    prediccion_local = predecir_partido(equipo_a, equipo_b, rating_a, rating_b, neutral=False)
    print(f"   {equipo_a} (local) vs {equipo_b} (visitante)")
    print(f"   Probabilidad victoria {equipo_a}: {prediccion_local['victoria_a']*100:.1f}%")
    print(f"   Probabilidad empate: {prediccion_local['empate']*100:.1f}%")
    print(f"   Probabilidad victoria {equipo_b}: {prediccion_local['victoria_b']*100:.1f}%")
    print()
    
    print("=" * 70)
    print("SISTEMA ELO IMPLEMENTADO CORRECTAMENTE")
    print("=" * 70)
