"""Analysis of Experiment 1 (shared scaling, TabPFN-3.5, 50k test flows).

Usage: python analysis/analyze_e7.py [results/e7_runs.csv]

Reproduces: reproduction check against Rayamajhi (2026), TabPFN vs classical
target-only models (RQ1), and TabPFN pooled-context vs target-only (RQ2).
"""
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")
PATH = sys.argv[1] if len(sys.argv) > 1 else "results/e7_runs.csv"
CLASSIC = ["RF", "GBT", "MLP"]

# Target-only BA (mean over RF/GBT/MLP), k = 1, 5, 10, from Rayamajhi (2026), shared scaling
REFERENCE = {
    "cic": (0.610, 0.731, 0.823), "unsw": (0.645, 0.900, 0.938),
    "ton": (0.570, 0.773, 0.781), "bot": (0.636, 0.821, 0.857),
}


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
r["tgt"] = r.direction.str.split("->").str[1]
to = r[r.method == "target-only"]

print("== Reproduction of original target-only baselines (BA, mean over classical models)")
m = to[to.model.isin(CLASSIC) & to.k.isin([1, 5, 10])].groupby(["direction", "k"]).bacc.mean()
diffs = [v - REFERENCE[d.split("->")[1]][[1, 5, 10].index(k)] for (d, k), v in m.items()]
print(f"cells={len(diffs)}  mean |diff|={np.mean(np.abs(diffs)):.4f}  max |diff|={np.max(np.abs(diffs)):.4f}")

print("\n== RQ1: TabPFN vs classical target-only (averaged over the 3 sources per target)")
a = to.groupby(["tgt", "k", "seed", "model"])[["bacc", "f1"]].mean().reset_index()
for met in ["bacc", "f1"]:
    p = a.pivot_table(index=["tgt", "k", "seed"], columns="model", values=met)
    best = p.groupby(level=["tgt", "k"]).mean()[CLASSIC].idxmax(axis=1)
    p = p.join(best.rename("best"), on=["tgt", "k"])
    d_avg = p.TabPFN - p[CLASSIC].mean(axis=1)
    d_orc = p.apply(lambda x: x.TabPFN - x[x.best], axis=1)
    print(f"{met}: vs average classical by k {d_avg.groupby(level='k').mean().round(4).to_dict()} (p={wx(d_avg):.2g})")
    print(f"{met}: vs oracle-best classical by k {d_orc.groupby(level='k').mean().round(4).to_dict()} (p={wx(d_orc):.2g})")

print("\n== RQ2: TabPFN pooled-context (S=500) minus TabPFN target-only")
tp = r[(r.model == "TabPFN") & (r.k > 0)]
for met in ["f1", "bacc"]:
    rows = []
    for (d, k), g in tp.groupby(["direction", "k"]):
        w = g.pivot(index="seed", columns="method", values=met)
        diff = w["pooled-context"] - w["target-only"]
        rows.append((d, k, diff.mean(), verdict(diff.mean(), wx(diff))))
    res = pd.DataFrame(rows, columns=["direction", "k", "d", "verdict"])
    mean, lo, hi = pooled_ci(res.groupby("direction").d.mean().values)
    print(f"{met}: mean {mean:.4f} 95% CI [{lo:.4f}, {hi:.4f}]  verdicts {res.verdict.value_counts().to_dict()}")

print("\n== Zero-shot TabPFN (source-only), mean BA per pair")
print(r[r.method == "source-only"].groupby("direction").bacc.mean().round(3).to_string())
