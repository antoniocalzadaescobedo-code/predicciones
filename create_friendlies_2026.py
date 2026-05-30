import pandas as pd

matches = [
    ["2026-05-26","Morocco","Burundi"],
    ["2026-05-26","Nigeria","Zimbabwe"],
    ["2026-05-28","Egypt","Russia"],
    ["2026-05-28","Republic of Ireland","Qatar"],
    ["2026-05-31","Germany","Finland"],
    ["2026-05-31","Poland","Ukraine"],
    ["2026-05-31","USA","Senegal"],
    ["2026-05-31","Brazil","Panama"],
    ["2026-06-01","Turkey","North Macedonia"],
    ["2026-06-01","Norway","Sweden"],
    ["2026-06-02","Croatia","Belgium"],
    ["2026-06-02","Wales","Ghana"],
    ["2026-06-03","Denmark","DR Congo"],
    ["2026-06-03","Luxembourg","Italy"],
    ["2026-06-03","Netherlands","Algeria"],
    ["2026-06-03","Poland","Nigeria"],
    ["2026-06-04","Northern Ireland","Guinea"],
    ["2026-06-04","Sweden","Greece"],
    ["2026-06-04","Spain","Iraq"],
    ["2026-06-04","France","Ivory Coast"],
    ["2026-06-05","Canada","Republic of Ireland"],
    ["2026-06-06","Belgium","Tunisia"],
    ["2026-06-06","Portugal","Chile"],
    ["2026-06-06","Romania","Wales"],
    ["2026-06-06","USA","Germany"],
    ["2026-06-06","Bolivia","Scotland"],
    ["2026-06-06","Brazil","Egypt"],
    ["2026-06-06","England","New Zealand"],
    ["2026-06-06","Venezuela","Turkey"],
    ["2026-06-06","Argentina","Honduras"],
    ["2026-06-07","Denmark","Ukraine"],
    ["2026-06-07","Greece","Italy"],
    ["2026-06-07","Morocco","Norway"],
    ["2026-06-08","France","Northern Ireland"],
    ["2026-06-08","Peru","Spain"],
    ["2026-06-08","Colombia","Jordan"],
]
df = pd.DataFrame(matches, columns=["date","home_team","away_team"])
df["competition"] = "International Friendly"
import os
file_path = os.path.join("data", "amistosos_reales.csv")
df.to_csv(file_path, index=False, encoding="utf-8-sig")
print(df.head())
