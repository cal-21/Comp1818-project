import sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

PROC = Path(__file__).parent.parent / "processed"
pairs = pd.read_csv(PROC / "pairs.csv")
pairs["is_AM"] = (pairs["group"] == "AM").astype(int)

FEATURES = ["goals_p90", "shots_p90", "xG_p90", "assists_p90", "xA_p90",
            "key_passes_p90", "npxG_p90", "xGChain_p90", "xGBuildup_p90",
            "npxG_xA_p90", "time", "seasons_seen_before"]

GRIDS = {
    "linear": [{}],
    "rf": [dict(n_estimators=n, max_depth=d, min_samples_leaf=l)
           for n in (200, 500) for d in (None, 5, 10) for l in (1, 5)],
    "xgb": [dict(n_estimators=n, max_depth=d, learning_rate=lr, subsample=0.8)
            for n in (200, 400) for d in (2, 3, 4) for lr in (0.03, 0.1)],
}

def make_model(name, params):
    if name == "linear":
        return make_pipeline(StandardScaler(), LinearRegression())
    if name == "rf":
        return RandomForestRegressor(random_state=42, n_jobs=-1, **params)
    return XGBRegressor(random_state=42, objective="reg:squarederror", **params)

def fit_predict(setup, name, params, tr, ev):
    if setup == "global":
        cols = FEATURES + ["is_AM"]
        m = make_model(name, params).fit(tr[cols], tr["target"])
        return pd.Series(m.predict(ev[cols]), index=ev.index)
    preds = pd.Series(index=ev.index, dtype=float)
    for g in ("FW", "AM"):
        trg, evg = tr[tr.group == g], ev[ev.group == g]
        m = make_model(name, params).fit(trg[FEATURES], trg["target"])
        preds.loc[evg.index] = m.predict(evg[FEATURES])
    return preds

def boot_ci(err, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    means = [err[rng.integers(0, len(err), len(err))].mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])

mode = sys.argv[1] if len(sys.argv) > 1 else "val"
train = pairs[pairs.split == "train"]
val = pairs[pairs.split == "val"]
test = pairs[pairs.split == "test"]

if mode == "val":
    print("Target (train): mean %.3f, std %.3f" % (train.target.mean(), train.target.std()))
    best = {}
    for name, grid in GRIDS.items():
        for setup in ("global", "positional"):
            scored = []
            for params in grid:
                p = fit_predict(setup, name, params, train, val)
                scored.append((mean_absolute_error(val.target, p), params))
            mae, params = min(scored, key=lambda t: t[0])
            best[f"{name}|{setup}"] = params
            print(f"{name:7s} {setup:11s} val MAE {mae:.4f}  best params {params}")
    base = mean_absolute_error(val.target, val.baseline)
    print(f"{'baseline':7s} {'-':11s} val MAE {base:.4f}")
    (PROC / "best_params.json").write_text(json.dumps(best, indent=2))
    print("\nSaved best_params.json. Do NOT run test mode until you are finished tuning.")

elif mode == "test":  # test mode: run once, at the very end
    best = json.loads((PROC / "best_params.json").read_text())
    tr = pd.concat([train, val])           # refit on train + validation
    rows = []
    def record(name, setup, pred):
        err = (test.target - pred).abs().to_numpy()
        lo, hi = boot_ci(err)
        row = dict(model=name, setup=setup, MAE=err.mean(), MAE_lo=lo, MAE_hi=hi,
                   RMSE=np.sqrt(mean_squared_error(test.target, pred)),
                   R2=r2_score(test.target, pred))
        for g in ("FW", "AM"):
            mk = (test.group == g).to_numpy()
            row["MAE_" + g] = err[mk].mean()
        rows.append(row)
    record("baseline", "-", test.baseline)
    for key, params in best.items():
        name, setup = key.split("|")
        record(name, setup, fit_predict(setup, name, params, tr, test))
    res = pd.DataFrame(rows).round(4)
    print(res.to_string(index=False))
    res.to_csv(PROC / "results_test.csv", index=False)
elif mode == "analysis":
    from sklearn.inspection import permutation_importance
    best = json.loads((PROC / "best_params.json").read_text())
    rng = np.random.default_rng(1)
    base_err = (val.target - val.baseline).abs().to_numpy()
    preds = {}
    for key, params in best.items():
        name, setup = key.split("|")
        preds[key] = fit_predict(setup, name, params, train, val)

    def paired(e_a, e_b, n=2000):
        d = e_a - e_b
        idx = rng.integers(0, len(d), (n, len(d)))
        m = d[idx].mean(axis=1)
        return d.mean(), np.percentile(m, [2.5, 97.5])

    print("MAE difference vs baseline (negative = model better), 95% CI, validation:")
    for key, p in preds.items():
        e = (val.target - p).abs().to_numpy()
        d, (lo, hi) = paired(e, base_err)
        print(f"{key:18s} {d:+.4f}  [{lo:+.4f}, {hi:+.4f}]")

    print("\nPositional minus global (negative = positional better):")
    for name in GRIDS:
        eg = (val.target - preds[f"{name}|global"]).abs().to_numpy()
        ep = (val.target - preds[f"{name}|positional"]).abs().to_numpy()
        d, (lo, hi) = paired(ep, eg)
        print(f"{name:8s} {d:+.4f}  [{lo:+.4f}, {hi:+.4f}]")

    cols = FEATURES + ["is_AM"]
    rf = make_model("rf", best["rf|global"]).fit(train[cols], train["target"])
    imp = permutation_importance(rf, val[cols], val["target"], n_repeats=20,
                                 random_state=0, scoring="neg_mean_absolute_error")
    print("\nPermutation importance (RF global, validation; bigger = more important):")
    for i in np.argsort(-imp.importances_mean):
        print(f"{cols[i]:22s} {imp.importances_mean[i]:+.4f}")    
elif mode == "trend":
    which = sys.argv[2] if len(sys.argv) > 2 else "val"
    best = json.loads((PROC / "best_params.json").read_text())
    tr_all = train if which == "val" else pd.concat([train, val])
    ev_all = val if which == "val" else test
    tr = tr_all[tr_all.has_prev]
    ev = ev_all[ev_all.has_prev]
    print(f"Trend subset ({which}): train {len(tr)}, eval {len(ev)} pairs")
    base_err = (ev.target - ev.baseline).abs().to_numpy()
    print(f"baseline MAE on this subset: {base_err.mean():.4f}")
    rng = np.random.default_rng(2)

    def paired2(a, b, n=2000):
        d = a - b
        idx = rng.integers(0, len(d), (n, len(d)))
        m = d[idx].mean(axis=1)
        return d.mean(), np.percentile(m, [2.5, 97.5])

    base_cols = FEATURES + ["is_AM"]
    trend_cols = base_cols + ["prev_npxG_xA_p90", "delta_npxG_xA_p90"]
    for name in GRIDS:
        params = best[f"{name}|global"]
        e = {}
        for label, cols in (("without", base_cols), ("with", trend_cols)):
            m = make_model(name, params).fit(tr[cols], tr["target"])
            e[label] = (ev.target - m.predict(ev[cols])).abs().to_numpy()
        d, (lo, hi) = paired2(e["with"], e["without"])
        print(f"{name:7s} MAE without {e['without'].mean():.4f} | with {e['with'].mean():.4f} | diff {d:+.4f} [{lo:+.4f}, {hi:+.4f}]") 
        dB, (loB, hiB) = paired2(e["with"], base_err)
        print(f"        with-trend vs baseline: {dB:+.4f} [{loB:+.4f}, {hiB:+.4f}]")
elif mode == "testpaired":
    best = json.loads((PROC / "best_params.json").read_text())
    tr = pd.concat([train, val])
    rng = np.random.default_rng(3)

    def paired3(a, b, n=2000):
        d = a - b
        idx = rng.integers(0, len(d), (n, len(d)))
        m = d[idx].mean(axis=1)
        return d.mean(), np.percentile(m, [2.5, 97.5])

    base_err = (test.target - test.baseline).abs().to_numpy()
    errs = {}
    for key, params in best.items():
        name, setup = key.split("|")
        errs[key] = (test.target - fit_predict(setup, name, params, tr, test)).abs().to_numpy()

    print("Model minus baseline MAE (negative = model better), 95% CI, test:")
    for key, e in errs.items():
        d, (lo, hi) = paired3(e, base_err)
        print(f"{key:18s} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    print("\nPositional minus global (negative = positional better):")
    for name in GRIDS:
        d, (lo, hi) = paired3(errs[f"{name}|positional"], errs[f"{name}|global"])
        print(f"{name:8s} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")          
elif mode == "save":
    import joblib
    best = json.loads((PROC / "best_params.json").read_text())
    tr = pd.concat([train, val])
    out = PROC / "models"
    out.mkdir(exist_ok=True)
    for key, params in best.items():
        name, setup = key.split("|")
        if setup == "global":
            cols = FEATURES + ["is_AM"]
            m = make_model(name, params).fit(tr[cols], tr["target"])
            joblib.dump(m, out / f"{name}_global.joblib")
        else:
            for g in ("FW", "AM"):
                trg = tr[tr.group == g]
                m = make_model(name, params).fit(trg[FEATURES], trg["target"])
                joblib.dump(m, out / f"{name}_positional_{g}.joblib")
    print("Saved", len(list(out.glob("*.joblib"))), "models to", out)        