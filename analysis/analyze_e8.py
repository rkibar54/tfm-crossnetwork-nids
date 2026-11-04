"""Analysis of Experiment 2 (raw features, TabPFN-3.5 and TabICLv2, 20k test flows).

Usage: python analysis/analyze_e8.py [results/e8_runs.csv]

Reproduces Tables 2-4 of the paper: target-only comparison (RQ1), the effect of
source flows in the context (RQ2), the context-size ablation (RQ3) and zero-shot.
"""
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")
PATH = sys.argv[1] if len(sys.argv) > 1 else "results/e8_runs.csv"
CLASSIC, TFM = ["RF", "GBT", "MLP"], ["TabPFN", "TabICL"]


def wx(d):
    d = np.asarray(d)
    return wilcoxon(d).pvalue if (d != 0).any() else 1.0


def verdict(d, p):
    return "help" if p < 0.05 and d > 0 else ("hurt" if p < 0.05 else "no diff")


def pooled_ci(pair_means, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    boots = [rng.choice(pair_means, len(pair_means)).mean() for _ in range(n_boot)]
    return np.mean(pair_means), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


r = pd.read_csv(PATH)
r["source"] = r.source.fillna("")
to = r[r.method == "target-only"]

print("== Table 2: target-only BA / F1 by k (mean over 4 targets x 10 seeds)")
for met in ["bacc", "f1"]:
    print(met)
    print(to.pivot_table(index="k", columns="model", values=met)[CLASSIC + TFM].round(3).to_string())

print("\n== RQ1: TFMs vs classical target-only")
pv = {met: to.pivot_table(index=["target", "k", "seed"], columns="model", values=met) for met in ["bacc", "f1"]}
for t in TFM:
    for met in ["bacc", "f1"]:
        p = pv[met].copy()
        best = p.groupby(level=["target", "k"]).mean()[CLASSIC].idxmax(axis=1)
        p = p.join(best.rename("best"), on=["target", "k"])
        d_avg = p[t] - p[CLASSIC].mean(axis=1)
        d_orc = p.apply(lambda x: x[t] - x[x.best], axis=1)
        print(f"{t} {met}: vs average classical {d_avg.groupby(level='k').mean().round(3).to_dict()} (p={wx(d_avg):.2g})")
        print(f"{t} {met}: vs oracle-best {d_orc.groupby(level='k').mean().round(3).to_dict()} (mean {d_orc.mean():.4f}, p={wx(d_orc):.2g})")
    cells = [verdict((g[t] - g[c]).mean(), wx(g[t] - g[c]))
             for _, g in pv["bacc"].groupby(level=["target", "k"]) for c in CLASSIC]
    print(f"{t} vs each classical (BA, 60 tests): {pd.Series(cells).value_counts().to_dict()}")
d = pv["bacc"].TabPFN - pv["bacc"].TabICL
print(f"TabPFN - TabICL target-only BA: {d.groupby(level='k').mean().round(3).to_dict()} (p={wx(d):.2g})")

print("\n== Table 3 / RQ2: pooled-context (S=500) minus target-only, same model")
base = to[to.model.isin(TFM)][["target", "seed", "model", "k", "f1", "bacc"]]
pc = r[r.method == "pooled-context"].merge(base, on=["target", "seed", "model", "k"], suffixes=("", "_to"))
pc["df1"], pc["dbacc"] = pc.f1 - pc.f1_to, pc.bacc - pc.bacc_to
pc["direction"] = pc.source + "->" + pc.target
for t in TFM:
    s = pc[(pc.model == t) & (pc.S == 500)]
    for met in ["df1", "dbacc"]:
        cells = [verdict(g[met].mean(), wx(g[met])) for _, g in s.groupby(["direction", "k"])]
        mean, lo, hi = pooled_ci(s.groupby("direction")[met].mean().values)
        print(f"{t} {met}: {mean:.4f} [{lo:.4f}, {hi:.4f}]  verdicts {pd.Series(cells).value_counts().to_dict()}")
        print(f"    by k {s.groupby('k')[met].mean().round(4).to_dict()}  by target {s.groupby('target')[met].mean().round(3).to_dict()}")

print("\n== Table 4 / RQ3: TabPFN context-size ablation (k = 1, 10)")
s = pc[(pc.model == "TabPFN") & pc.k.isin([1, 10])]
print(s.groupby(["k", "S"])[["df1", "dbacc"]].mean().round(4).to_string())
w = s.pivot_table(index=["direction", "seed", "k"], columns="S", values="f1")
print(f"F1 S=2000 minus S=100: {(w[2000] - w[100]).mean():.4f} (p={wx(w[2000] - w[100]):.2g})")

print("\n== Zero-shot (source-only, S=500), mean BA")
z = r[r.method == "source-only"].assign(direction=lambda x: x.source + "->" + x.target)
print(z.pivot_table(index="direction", columns="model", values="bacc").round(3).to_string())
