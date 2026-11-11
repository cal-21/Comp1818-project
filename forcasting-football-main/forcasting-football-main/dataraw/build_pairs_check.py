import pandas as pd
from pathlib import Path

p = pd.read_csv(Path(__file__).parent / "player_stats.csv", low_memory=False)
for c in ["season", "time", "npxG", "xA"]:
    p[c] = pd.to_numeric(p[c], errors="coerce")

p = p[p["season"] <= 2023]          # 2024 is a partial season

print("Position values (top 15):")
print(p["position"].value_counts().head(15))

dup = p.groupby(["player_id", "season"]).size()
print("\nPlayer-seasons with more than one row:", (dup > 1).sum(), "of", len(dup))

# one row per player per season
agg = p.groupby(["player_id", "season"], as_index=False).agg(
    position=("position", "first"), time=("time", "sum"),
    npxG=("npxG", "sum"), xA=("xA", "sum"))

# pair each season with the same player's next season
nxt = agg[["player_id", "season", "time", "npxG", "xA"]].copy()
nxt["season"] = nxt["season"] - 1
pairs = agg.merge(nxt, on=["player_id", "season"], suffixes=("", "_next"))
pairs = pairs[(pairs["time"] >= 900) & (pairs["time_next"] >= 900)]

print("\nPairs surviving the 900-minute filter (season = INPUT season):")
for label, mask in {"position contains F": pairs["position"].str.contains("F", na=False),
                    "position contains F or M": pairs["position"].str.contains("F|M", na=False)}.items():
    print(f"\n{label}")
    print(pairs[mask].groupby("season").size().to_string())