# prepare_dataset.py (VERSIÓN BLINDADA)
import pandas as pd
import numpy as np
import sys

def main():
    print("🔧 Preparando dataset para entrenamiento...")
    try:
        df = pd.read_csv("results.csv")
    except FileNotFoundError:
        print("❌ No se encuentra 'results.csv' en el directorio actual.")
        sys.exit(1)

    # 1. Validación y limpieza estricta
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    initial_len = len(df)
    df = df.dropna(subset=["date", "home_score", "away_score"])
    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"⚠️ Eliminados {dropped} partidos con marcadores/fechas faltantes.")

    df = df.sort_values("date").reset_index(drop=True)
    df["neutral"] = df["neutral"].fillna(0).astype(int)
    
    # Outcome seguro
    df["outcome"] = np.sign(df["home_score"] - df["away_score"]).astype(int)

    # 2. Inicializar features
    df["elo_diff"] = 0.0
    df["form_home"] = 0.5
    df["form_away"] = 0.5
    df["h2h"] = 0.5

    # 3. Cálculo Walk-Forward
    elo = {}
    K = 30.0
    team_history = {}  # {team: [outcomes...]}
    h2h_cache = {}     # {(home, away): [outcomes...]}

    total = len(df)
    print(f"📊 Procesando {total} partidos (cálculo ELO/Forma/H2H en tiempo real)...")

    for i, row in df.iterrows():
        h, a, n, res = row["home_team"], row["away_team"], row["neutral"], row["outcome"]

        # ELO
        r_h = elo.get(h, 1500)
        r_a = elo.get(a, 1500)
        adj = 100.0 if not n else 0.0
        exp_h = 1 / (1 + 10**((r_a - r_h - adj)/400))
        exp_a = 1 - exp_h
        res_h = 1.0 if res == 1 else (0.5 if res == 0 else 0.0)
        elo[h] = r_h + K * (res_h - exp_h)
        elo[a] = r_a + K * ((1-res_h) - exp_a)
        df.at[i, "elo_diff"] = (r_h + adj) - r_a

        # FORMA (últimos 5 partidos, normalizado 0-1)
        h_hist = team_history.get(h, [])[-5:]
        a_hist = team_history.get(a, [])[-5:]
        h_pts = sum(1 if x==1 else 0.5 if x==0 else 0 for x in h_hist)
        a_pts = sum(1 if x==-1 else 0.5 if x==0 else 0 for x in a_hist)
        df.at[i, "form_home"] = h_pts / max(len(h_hist), 1)
        df.at[i, "form_away"] = a_pts / max(len(a_hist), 1)

        team_history.setdefault(h, []).append(res)
        team_history.setdefault(a, []).append(-res)

        # H2H (últimos 5 enfrentamientos directos, perspectiva local)
        direct = []
        if (h, a) in h2h_cache: direct.extend(h2h_cache[(h, a)])
        if (a, h) in h2h_cache: direct.extend([-x for x in h2h_cache[(a, h)]])
        direct = direct[-5:]
        if direct:
            df.at[i, "h2h"] = sum(1 for x in direct if x == 1) / len(direct)
        
        h2h_cache.setdefault((h, a), []).append(res)

        if (i + 1) % 2000 == 0:
            print(f"   ⏳ {i+1}/{total} ({(i+1)/total*100:.1f}%)")

    # 4. Exportar
    out_cols = ["date", "home_team", "away_team", "elo_diff", "form_home", "form_away", "h2h", "neutral", "outcome"]
    df[out_cols].to_csv("matches_clean.csv", index=False, encoding="utf-8-sig")
    print(f"✅ Dataset listo: matches_clean.csv ({len(df)} partidos)")
    print("🚀 Siguiente: py train_and_save_model.py")

if __name__ == "__main__":
    main()
