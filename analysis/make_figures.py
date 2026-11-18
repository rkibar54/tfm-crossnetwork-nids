"""Figures 1 and 2 of the paper from Experiment 2 results.

Usage: python analysis/make_figures.py [results/e8_runs.csv] [output_dir]

Figure 1: target-only balanced accuracy against k for all five models.
Figure 2: effect of source flows in the TFM context, (a) by k and (b) by context size S.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PATH = sys.argv[1] if len(sys.argv) > 1 else "results/e8_runs.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})
# Colour-blind safe (Okabe-Ito); markers keep lines distinguishable in grey scale
STYLE = {
    "RF": dict(color="#999999", marker="s", ls="--"),
    "GBT": dict(color="#E69F00", marker="^", ls="--"),
    "MLP": dict(color="#56B4E9", marker="v", ls="--"),
    "TabPFN": dict(color="#0072B2", marker="o", ls="-"),
    "TabICL": dict(color="#D55E00", marker="D", ls="-"),
}
K = [1, 5, 10, 50, 100]


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    half = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return values.mean(), half


def boot_ci(values, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    boots = [rng.choice(values, len(values)).mean() for _ in range(n_boot)]
    return values.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


r = pd.read_csv(PATH)
r["source"] = r.source.fillna("")

# ---------------- Figure 1: target-only learning curves ----------------
to = r[r.method == "target-only"]
CLASSIC = ["RF", "GBT", "MLP"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.8), gridspec_kw={"width_ratios": [1.2, 1]})

# (a) mean balanced accuracy over 4 targets x 10 seeds
mean_ba = to.groupby(["model", "k"]).bacc.mean()
for model in ["RF", "GBT", "MLP", "TabPFN", "TabICL"]:
    st = STYLE[model]
    a1.plot(K, [mean_ba[(model, k)] for k in K], label=model, color=st["color"], marker=st["marker"],
            ls=st["ls"], ms=4, lw=1.2)
a1.set_xscale("log")
a1.set_xticks(K)
a1.set_xticklabels([str(k) for k in K])
a1.set_xlabel("Labelled target flows per class ($k$)")
a1.set_ylabel("Balanced accuracy")
a1.set_title("(a) Target-only learning curves", fontsize=9)
a1.legend(frameon=False, ncol=2, loc="lower right")
a1.grid(axis="y", lw=0.4, alpha=0.5)

# (b) paired difference to the average classical model and to the test-label oracle
pv = to.pivot_table(index=["target", "k", "seed"], columns="model", values="bacc")
best = pv.groupby(level=["target", "k"]).mean()[CLASSIC].idxmax(axis=1)
pv = pv.join(best.rename("best"), on=["target", "k"])
oracle = pv.apply(lambda x: x[x.best], axis=1)
avg = pv[CLASSIC].mean(axis=1)
offsets = {"TabPFN": 0.96, "TabICL": 1.04}
for model in ["TabPFN", "TabICL"]:
    st = STYLE[model]
    for ref, ref_name, ls, fill in [(avg, "avg. classical", "-", True), (oracle, "oracle", ":", False)]:
        d = (pv[model] - ref).groupby(level=["k", "seed"]).mean()  # average over targets per seed
        m, h = map(np.array, zip(*[mean_ci(d.loc[k]) for k in K]))
        a2.errorbar(np.array(K) * offsets[model], m, yerr=h, color=st["color"], marker=st["marker"],
                    ls=ls, ms=4 if fill else 3.5, lw=1.1, capsize=2,
                    mfc=st["color"] if fill else "white", label=f"{model} $-$ {ref_name}")
a2.axhline(0, color="black", lw=0.8)
a2.set_xscale("log")
a2.set_xticks(K)
a2.set_xticklabels([str(k) for k in K])
a2.set_xlabel("Labelled target flows per class ($k$)")
a2.set_ylabel(r"$\Delta$ balanced accuracy")
a2.set_title("(b) TFM minus classical models", fontsize=9)
a2.legend(frameon=False, loc="lower right", fontsize=7)
a2.grid(axis="y", lw=0.4, alpha=0.5)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_target_only.pdf")
fig.savefig(f"{OUT}/fig1_target_only.png", dpi=300)

# ---------------- Figure 2: source effect ----------------
base = to[to.model.isin(["TabPFN", "TabICL"])][["target", "seed", "model", "k", "bacc"]]
pc = r[r.method == "pooled-context"].merge(base, on=["target", "seed", "model", "k"], suffixes=("", "_to"))
pc["d"] = pc.bacc - pc.bacc_to
pc["pair"] = pc.source + "->" + pc.target

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.7), gridspec_kw={"width_ratios": [1.35, 1]})
offsets = {"TabPFN": 0.97, "TabICL": 1.03}
for model in ["TabPFN", "TabICL"]:
    s = pc[(pc.model == model) & (pc.S == 500)]
    rows = [boot_ci(s[s.k == k].groupby("pair").d.mean().values) for k in K]
    m, lo, hi = map(np.array, zip(*rows))
    x = np.array(K) * offsets[model]
    st = STYLE[model]
    a1.errorbar(x, m, yerr=[m - lo, hi - m], label=f"{model} ($S$ = 500)", color=st["color"],
                marker=st["marker"], ls=st["ls"], ms=4, lw=1.2, capsize=2)
a1.axhline(0, color="black", lw=0.8)
a1.set_xscale("log")
a1.set_xticks(K)
a1.set_xticklabels([str(k) for k in K])
a1.set_xlabel("Labelled target flows per class ($k$)")
a1.set_ylabel(r"$\Delta$ BA (pooled $-$ target-only)")
a1.set_title("(a) Effect by label budget", fontsize=9)
a1.legend(frameon=False, loc="lower right")
a1.grid(axis="y", lw=0.4, alpha=0.5)

S_VALUES = [100, 500, 2000]
for k, st in [(1, dict(color="#0072B2", marker="o", ls="-")), (10, dict(color="#009E73", marker="s", ls="--"))]:
    s = pc[(pc.model == "TabPFN") & (pc.k == k)]
    rows = [boot_ci(s[s.S == S].groupby("pair").d.mean().values) for S in S_VALUES]
    m, lo, hi = map(np.array, zip(*rows))
    a2.errorbar(S_VALUES, m, yerr=[m - lo, hi - m], label=f"$k$ = {k}", capsize=2, ms=4, lw=1.2, **st)
a2.axhline(0, color="black", lw=0.8)
a2.set_xscale("log")
a2.set_xticks(S_VALUES)
a2.set_xticklabels([str(s) for s in S_VALUES])
a2.set_xlabel("Source flows per class in context ($S$)")
a2.set_title("(b) TabPFN, effect by context size", fontsize=9)
a2.legend(frameon=False, loc="lower left")
a2.grid(axis="y", lw=0.4, alpha=0.5)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_source_effect.pdf")
fig.savefig(f"{OUT}/fig2_source_effect.png", dpi=300)
print(f"Figures written to {OUT}/")
