import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE.parent / "processed"
OUT.mkdir(exist_ok=True)

p = pd.read_csv(HERE / "player_stats.csv", low_memory=False)

sums = ["games", "goals", "shots", "time", "xG", "assists", "xA", "key_passes",
        "yellow", "red", "npg", "npxG", "xGChain", "xGBuildup"]
for c in ["season"] + sums:
    p[c] = pd.to_numeric(p[c], errors="coerce")

# 2024 is a partial season (max 10 games), so it is excluded
p = p.dropna(subset=["player_id", "season", "time"])
p = p[(p["season"] <= 2023) & (p["time"] > 0)]
p["season"] = p["season"].astype(int)

# One row per player per season: players who moved club mid-season have two rows.
# Counting stats are summed; position and team come from the row with most minutes.
p = p.sort_values("time", ascending=False)
agg = p.groupby(["player_id", "season"]).agg(
    {**{c: "sum" for c in sums}, "position": "first", "name": "first", "team": "first"}
).reset_index()

# Position groups
def group(pos):
    pos = str(pos)
    if "F" in pos:
        return "FW"
    if "AM" in pos or pos in ("ML", "MR"):
        return "AM"
    return None
agg["group"] = agg["position"].map(group)

# Per-90 features (input season) and the target measure
agg["n90"] = agg["time"] / 90
for c in ["goals", "shots", "xG", "assists", "xA", "key_passes", "npxG", "xGChain", "xGBuildup", "npg"]:
    agg[c + "_p90"] = agg[c] / agg["n90"]
agg["npxG_xA_p90"] = (agg["npxG"] + agg["xA"]) / agg["n90"]

# Experience proxy: how many earlier seasons this player appears in (past data only)
agg = agg.sort_values(["player_id", "season"])
agg["seasons_seen_before"] = agg.groupby("player_id").cumcount()
# Previous season's value (for the trend experiment)
prev = agg[["player_id", "season", "time", "npxG_xA_p90"]].copy()
prev["season"] = prev["season"] + 1
prev = prev.rename(columns={"time": "prev_time", "npxG_xA_p90": "prev_npxG_xA_p90"})
agg = agg.merge(prev, on=["player_id", "season"], how="left")
agg["has_prev"] = agg["prev_time"] >= 900
agg["delta_npxG_xA_p90"] = agg["npxG_xA_p90"] - agg["prev_npxG_xA_p90"]

# Pair each player-season with the same player's NEXT season
nxt = agg[["player_id", "season", "time", "npxG_xA_p90"]].copy()
nxt["season"] = nxt["season"] - 1
nxt = nxt.rename(columns={"time": "next_time", "npxG_xA_p90": "target"})
pairs = agg.merge(nxt, on=["player_id", "season"], how="inner")

# Keep attackers with 900+ minutes in BOTH seasons
pairs = pairs[(pairs["time"] >= 900) & (pairs["next_time"] >= 900) & pairs["group"].notna()].copy()
pairs = pairs.rename(columns={"season": "input_season"})
pairs["baseline"] = pairs["npxG_xA_p90"]          # "no change" prediction

# Chronological split by input season
pairs["split"] = pairs["input_season"].map(
    lambda s: "train" if s <= 2020 else ("val" if s == 2021 else "test"))

pairs.to_csv(OUT / "pairs.csv", index=False)

print("Pairs per split and group:")
print(pairs.groupby(["split", "group"]).size().unstack(fill_value=0))

# Test set is deliberately NOT inspected here
dev = pairs[pairs["split"] != "test"]
print("\nNo-change baseline MAE (train and validation only):")
print((dev["target"] - dev["baseline"]).abs().groupby(dev["split"]).mean())