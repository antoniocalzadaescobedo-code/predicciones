#!/usr/bin/env python3
"""
Módulo de Métricas Avanzadas - Fase 2
Incluye: xG simulado, ROI, Yield, Brier Score, actualización de ELO
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

# ------------------------------------------------------------
# 1. MÉTRICAS AVANZADAS DE FÚTBOL (Simuladas/Proxy)
# ------------------------------------------------------------
def calcular_xg_proxy(elo_ataque: float, elo_defensa: float, neutral: bool = True) -> float:
    """
    Calcula Goles Esperados (xG) como proxy basado en ELO.
    Fórmula simplificada: lambda = base * (elo_ataque/elo_defensa)^factor
    """
    base_xg = 1.2  # Promedio histórico de goles por equipo en Mundial
    factor = 0.002  # Sensibilidad al diferencial de ELO
    
    # Ajuste por ventaja de local (no aplica en Mundial neutral)
    local_bonus = 1.0 if neutral else 1.15
    
    # Cálculo del xG
    xg = base_xg * local_bonus * np.exp(factor * (elo_ataque - elo_defensa))
    return min(3.5, max(0.3, xg))  # Clamp realista

def calcular_duel_power(elo: float) -> float:
    """
    Poder de Duelo: métrica de dominio físico/táctico.
    Escala 0-100, donde 50 es promedio mundial (~1500 ELO).
    """
    return np.clip(50 + (elo - 1500) / 30, 0, 100)

def calcular_pressure_index(elo_defensa: float, elo_rival: float) -> float:
    """
    Índice de Presión: capacidad defensiva para incomodar al rival.
    Mayor cuando tu defensa es superior al ataque rival.
    """
    diff = elo_defensa - elo_rival
    # Función sigmoide para saturación realista
    return 1 / (1 + np.exp(-diff / 150)) * 100

def calcular_xgc(prob_home: float, prob_draw: float, prob_away: float, elo_diff: float) -> Dict[str, float]:
    """
    Contribuciones de Gol Esperadas (xGc): métrica combinada ofensiva.
    Integra eficiencia de tiro, conversión y volumen.
    """
    # Proxy: probabilidad de victoria ajustada por diferencial de ELO
    xgc_home = prob_home * (1 + elo_diff / 2000)
    xgc_away = prob_away * (1 - elo_diff / 2000)
    
    return {
        "xgc_home": min(1.0, max(0.0, xgc_home)),
        "xgc_away": min(1.0, max(0.0, xgc_away)),
        "xgc_total": (xgc_home + xgc_away) / 2
    }

# ------------------------------------------------------------
# 2. MÉTRICAS DE NEGOCIO (Value Betting)
# ------------------------------------------------------------
def calcular_valor_apuesta(prob_modelo: float, cuota_mercado: float) -> Dict[str, float]:
    """
    Calcula el valor esperado de una apuesta.
    Valor > 0 indica oportunidad de apuesta (+EV).
    """
    prob_implicita = 1 / cuota_mercado
    valor = prob_modelo - prob_implicita
    roi_esperado = (prob_modelo * (cuota_mercado - 1)) - (1 - prob_modelo)
    
    return {
        "prob_implicita": prob_implicita,
        "valor": valor,
        "roi_esperado": roi_esperado,
        "recomendacion": "APUESTA" if valor > 0.02 else "EVITAR"  # Threshold del 2%
    }

def calcular_yield(apuestas: List[Dict]) -> Dict[str, float]:
    """
    Calcula Yield y ROI acumulado de un historial de apuestas.
    apuestas: lista de dicts con {cuota, stake, resultado, prob_modelo}
    """
    if not apuestas:
        return {"yield": 0.0, "roi": 0.0, "total_stake": 0, "profit": 0}
    
    total_stake = sum(a["stake"] for a in apuestas)
    total_retorno = sum(
        a["stake"] * a["cuota"] if a["resultado"] == "ganada" else 0 
        for a in apuestas
    )
    profit = total_retorno - total_stake
    
    return {
        "yield": (profit / total_stake * 100) if total_stake > 0 else 0,
        "roi": (profit / total_stake * 100) if total_stake > 0 else 0,
        "total_stake": total_stake,
        "profit": profit,
        "apuestas_ganadas": sum(1 for a in apuestas if a["resultado"] == "ganada"),
        "total_apuestas": len(apuestas)
    }

def calcular_brier_score(predicciones: List[Tuple[float, float, float]], 
                         resultados_reales: List[str]) -> float:
    """
    Brier Score: mide la precisión de probabilidades pronosticadas.
    Menor es mejor (0 = perfecto, 0.33 = aleatorio para 3 clases).
    """
    if len(predicciones) != len(resultados_reales):
        raise ValueError("Longitudes no coinciden")
    
    brier_sum = 0
    for (p_home, p_draw, p_away), resultado in zip(predicciones, resultados_reales):
        if resultado == "home_win":
            brier_sum += (p_home - 1)**2 + p_draw**2 + p_away**2
        elif resultado == "draw":
            brier_sum += p_home**2 + (p_draw - 1)**2 + p_away**2
        else:  # away_win
            brier_sum += p_home**2 + p_draw**2 + (p_away - 1)**2
    
    return brier_sum / len(predicciones)

# ------------------------------------------------------------
# 3. ACTUALIZACIÓN DINÁMICA DE ELO (Post-Simulación)
# ------------------------------------------------------------
def actualizar_elo_post_partido(elo_ganador: float, elo_perdedor: float, 
                                 goles_ganador: int, goles_perdedor: int,
                                 neutral: bool = True, k_factor: float = 30) -> Tuple[float, float]:
    """
    Actualiza ratings ELO después de un partido simulado.
    Incluye ajuste por diferencia de goles.
    """
    # Resultado esperado
    prob_ganador = 1 / (1 + 10**((elo_perdedor - elo_ganador) / 400))
    
    # Resultado real (1 = gana, 0.5 = empate, 0 = pierde)
    resultado_real = 1.0
    
    # Ajuste por diferencia de goles (fórmula FIFA simplificada)
    diff_goles = abs(goles_ganador - goles_perdedor)
    gol_bonus = np.log(diff_goles + 1) * (2.2 / ((elo_ganador - elo_perdedor) * 0.001 + 2.2)) if diff_goles > 0 else 0
    
    # Actualización
    delta = k_factor * (resultado_real - prob_ganador) * (1 + gol_bonus * 0.1)
    
    nuevo_elo_ganador = elo_ganador + delta
    nuevo_elo_perdedor = elo_perdedor - delta
    
    return round(nuevo_elo_ganador, 1), round(nuevo_elo_perdedor, 1)

def simular_actualizacion_grupo(grupo: str, partidos: List[Dict], db) -> Dict[str, float]:
    """
    Simula actualización de ELO para todos los equipos de un grupo
    después de la fase de grupos.
    """
    actualizaciones = {}
    
    for partido in partidos:
        t1, t2 = partido["home"], partido["away"]
        elo1, elo2 = db.get_elo(t1), db.get_elo(t2)
        
        # Determinar ganador simulado
        if partido["goles_home"] > partido["goles_away"]:
            ganador, perdedor, g_gan, g_perd = t1, t2, partido["goles_home"], partido["goles_away"]
        elif partido["goles_away"] > partido["goles_home"]:
            ganador, perdedor, g_gan, g_perd = t2, t1, partido["goles_away"], partido["goles_home"]
        else:
            # Empate: actualizar ambos con ajuste menor
            actualizaciones[t1] = db.get_elo(t1)  # Sin cambio significativo
            actualizaciones[t2] = db.get_elo(t2)
            continue
        
        # Actualizar ELO
        nuevo_gan, nuevo_perd = actualizar_elo_post_partido(
            elo_ganador=db.get_elo(ganador),
            elo_perdedor=db.get_elo(perdedor),
            goles_ganador=g_gan,
            goles_perdedor=g_perd,
            neutral=True  # Mundial = campo neutral
        )
        
        actualizaciones[ganador] = nuevo_gan
        actualizaciones[perdedor] = nuevo_perd
    
    return actualizaciones
