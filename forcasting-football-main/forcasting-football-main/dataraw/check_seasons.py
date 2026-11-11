import pandas as pd
from pathlib import Path

for f in sorted(Path(__file__).parent.rglob("*.csv")):
    df = pd.read_csv(f, low_memory=False)
    print("=" * 50)
    print(f.name, df.shape)
    print(list(df.columns))
    if "season" in df.columns and "games" in df.columns:
        df["games"] = pd.to_numeric(df["games"], errors="coerce")
        print(df.groupby("season")["games"].agg(["count", "median", "max"]))