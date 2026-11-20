"""Data figures of the revised paper (Experiments 2 and 3).

Usage: python analysis/make_figures_v2.py [results_dir] [output_dir]

fig1_target_only    target-only BA vs k for eight learners; TFM minus average / oracle (six classical)
fig2_source_effect  source effect by k, by context size, and with vs without TTL
fig3_heatmap        per-target BA at k = 1, 10, 100, coloured by gap to the best learner of the row
fig4_cost           accuracy (PR-AUC, BA) vs prediction throughput at k = 10
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np, pandas as pd
from scipy import stats

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "axes.edgecolor": "#555555",
                     "xtick.color": "#555555", "ytick.color": "#555555"})
INK, GRID = "#222222", "#e4e4e4"
CL = ["RF", "GBT", "MLP", "LightGBM", "XGBoost", "CatBoost"]
TF = ["TabPFN", "TabICL"]
ORDER = CL + TF
# fixed categorical order (validated: CVD-safe adjacent pairs); markers give a second encoding
STY = {"TabPFN": ("#2a78d6", "o", "-"), "TabICL": ("#eb6834", "D", "-"), "CatBoost": ("#1baf7a", "s", "--"),
       "RF": ("#eda100", "^", "--"), "GBT": ("#e87ba4", "v", "--"), "LightGBM": ("#008300", "P", "--"),
       "XGBoost": ("#6250d6", "X", "--"), "MLP": ("#e34948", "<", "--")}
K = [1, 5, 10, 50, 100]
NAMES = {"bot": "NF-BoT-IoT", "cic": "NF-CICIDS2018", "ton": "NF-ToN-IoT", "unsw": "NF-UNSW-NB15"}


def logx(ax, ticks, label):
    ax.set_xscale("log"); ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off(); ax.set_xlabel(label); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)


def boot(v, n=10_000, seed=0):
    rng = np.random.default_rng(seed); v = np.asarray(v, float)
    b = rng.choice(v, (n, len(v))).mean(axis=1)
    return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def tci(tm):
    """Mean of target means with a 95% t-interval (targets are the independent units)."""
    tm = np.asarray(tm, float); m = tm.mean()
    h = stats.t.ppf(0.975, len(tm) - 1) * tm.std(ddof=1) / np.sqrt(len(tm))
    return m, m - h, m + h


def hboot(d, n=4000, seed=0):
    """Hierarchical bootstrap: resample targets, then seeds within each target (d indexed by target, seed)."""
    rng = np.random.default_rng(seed)
    g = {t: d.xs(t, level="target").values for t in d.index.get_level_values("target").unique()}
    tg = list(g); out = np.empty(n)
    for i in range(n):
        out[i] = np.mean([rng.choice(g[t], len(g[t])).mean() for t in rng.choice(tg, len(tg))])
    return d.mean(), np.percentile(out, 2.5), np.percentile(out, 97.5)


e8 = pd.read_csv(f"{RES}/e8_runs.csv"); e8["source"] = e8.source.fillna("")
e9 = pd.read_csv(f"{RES}/e9_runs.csv")
A = e9[(e9.method == "target-only") & (e9.features == "all")]

# ---------------- Figure 1 ----------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.9), gridspec_kw={"width_ratios": [1.25, 1]})
m = A.groupby(["model", "k"]).bacc.mean()
for mod in ORDER:
    c, mk, ls = STY[mod]
    tf = mod in TF
    a1.plot(K, [m[(mod, k)] for k in K], color=c, marker=mk, ls=ls, lw=1.6 if tf else 0.9,
            ms=4.5 if tf else 3.5, label=mod, zorder=3 if tf else 2, mec="white", mew=0.5)
logx(a1, K, "Labelled target flows per class ($k$)")
a1.set_ylabel("Balanced accuracy"); a1.set_title("(a) Target-only learning curves", fontsize=9)
a1.legend(frameon=False, ncol=2, loc="lower right", handlelength=2.2)

pv = A.pivot_table(index=["target", "k", "seed"], columns="model", values="bacc")
best = pv.groupby(level=["target", "k"]).mean()[CL].idxmax(axis=1)
orc = pd.Series([pv.loc[i, best[(i[0], i[1])]] for i in pv.index], index=pv.index)
avg = pv[CL].mean(axis=1)
off = {"TabPFN": 0.95, "TabICL": 1.05}
for mod in TF:
    c, mk, _ = STY[mod]
    for ref, name, ls, filled in [(avg, "avg. classical", "-", True), (orc, "oracle", ":", False)]:
        d = pv[mod] - ref
        r = [tci(d.xs(k, level="k").groupby(level="target").mean().values) for k in K]
        mu, lo, hi = map(np.array, zip(*r))
        a2.errorbar(np.array(K) * off[mod], mu, yerr=[mu - lo, hi - mu], color=c, marker=mk, ls=ls, lw=1.1,
                    ms=4, capsize=2, mfc=c if filled else "white", label=f"{mod} $-$ {name}")
a2.axhline(0, color=INK, lw=0.8)
logx(a2, K, "Labelled target flows per class ($k$)")
a2.set_ylabel(r"$\Delta$ balanced accuracy"); a2.set_title("(b) TFM minus six classical learners", fontsize=9)
a2.legend(frameon=False, loc="upper right", fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_target_only.pdf"); fig.savefig(f"{OUT}/fig1_target_only.png", dpi=300)

# ---------------- Figure 2 ----------------
to8 = e8[(e8.method == "target-only") & e8.model.isin(TF)].drop_duplicates(["target", "seed", "model", "k"])
pc = e8[e8.method == "pooled-context"].merge(to8[["target", "seed", "model", "k", "bacc"]],
                                              on=["target", "seed", "model", "k"], suffixes=("", "_to"))
pc["d"] = pc.bacc - pc.bacc_to; pc["pair"] = pc.source + ">" + pc.target
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True, gridspec_kw={"width_ratios": [1.3, 1, 1]})
off = {"TabPFN": 0.96, "TabICL": 1.04}
for mod in TF:
    c, mk, ls = STY[mod]; s = pc[(pc.model == mod) & (pc.S == 500)]
    mu, lo, hi = map(np.array, zip(*[boot(s[s.k == k].groupby("pair").d.mean().values) for k in K]))
    a1.errorbar(np.array(K) * off[mod], mu, yerr=[mu - lo, hi - mu], color=c, marker=mk, ls=ls, lw=1.2, ms=4,
                capsize=2, label=f"{mod}")
a1.axhline(0, color=INK, lw=0.8); logx(a1, K, "$k$ (labelled flows per class)")
a1.set_ylabel(r"$\Delta$ BA (pooled $-$ target-only)"); a1.set_title("(a) By $k$, $S$ = 500", fontsize=9)
a1.legend(frameon=False, loc="lower right")
SV = [100, 500, 2000]
for k, c, mk, ls in [(1, "#2a78d6", "o", "-"), (10, "#1baf7a", "s", "--")]:
    s = pc[(pc.model == "TabPFN") & (pc.k == k)]
    mu, lo, hi = map(np.array, zip(*[boot(s[s.S == S].groupby("pair").d.mean().values) for S in SV]))
    a2.errorbar(SV, mu, yerr=[mu - lo, hi - mu], color=c, marker=mk, ls=ls, lw=1.2, ms=4, capsize=2, label=f"$k$ = {k}")
a2.axhline(0, color=INK, lw=0.8); logx(a2, SV, "$S$ (source flows per class)")
a2.set_title("(b) TabPFN, by $S$", fontsize=9); a2.legend(frameon=False, loc="lower left")
# (c) with vs without TTL, TabPFN, S=500, k in {1,10,100}
K3 = [1, 10, 100]
n9 = e9[(e9.method == "target-only") & (e9.features == "no-ttl") & (e9.model == "TabPFN")][["target", "seed", "k", "bacc"]]
p9 = e9[e9.method == "pooled-context"].merge(n9, on=["target", "seed", "k"], suffixes=("", "_to"))
p9["d"] = p9.bacc - p9.bacc_to; p9["pair"] = p9.source + ">" + p9.target
sa = pc[(pc.model == "TabPFN") & (pc.S == 500)]
for df, lab, c, mk, ls, o in [(sa, "all 45 features", "#2a78d6", "o", "-", 0.95), (p9, "without TTL", "#6250d6", "X", "--", 1.05)]:
    mu, lo, hi = map(np.array, zip(*[boot(df[df.k == k].groupby("pair").d.mean().values) for k in K3]))
    a3.errorbar(np.array(K3) * o, mu, yerr=[mu - lo, hi - mu], color=c, marker=mk, ls=ls, lw=1.2, ms=4, capsize=2, label=lab)
a3.axhline(0, color=INK, lw=0.8); logx(a3, K3, "$k$ (labelled flows per class)")
a3.set_title("(c) TabPFN, with/without TTL", fontsize=9); a3.legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_source_effect.pdf"); fig.savefig(f"{OUT}/fig2_source_effect.png", dpi=300)

# ---------------- Figure 3: heatmap ----------------
mt = A.groupby(["target", "k", "model"]).bacc.mean().unstack()[ORDER]
rows = [(t, k) for t in ["bot", "cic", "ton", "unsw"] for k in [1, 10, 100]]
V = mt.loc[rows].values
gap = V.max(axis=1, keepdims=True) - V           # 0 = best learner of the row
cmap = LinearSegmentedColormap.from_list("blue", ["#0d366b", "#2a78d6", "#86b6ef", "#e8f1fc", "#ffffff"])
fig, ax = plt.subplots(figsize=(6.6, 4.1))
im = ax.imshow(np.clip(gap, 0, 0.10), cmap=cmap, vmin=0, vmax=0.10, aspect="auto")
for i in range(V.shape[0]):
    for j in range(V.shape[1]):
        g = gap[i, j]
        ax.text(j, i, f"{V[i, j]:.3f}", ha="center", va="center", fontsize=7,
                color="white" if g < 0.02 else INK, weight="bold" if g < 1e-9 else "normal")
ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(ORDER, rotation=30, ha="right")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{NAMES[t]}, $k$ = {k}" for t, k in rows])
for y in [2.5, 5.5, 8.5]: ax.axhline(y, color="white", lw=2.5)
ax.axvline(5.5, color="white", lw=2.5)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.set_label("Gap to best learner in row (BA)"); cb.set_ticks([0, 0.02, 0.05, 0.10]); cb.set_ticklabels(["0", "0.02", "0.05", "$\\geq$0.10"])
cb.outline.set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_heatmap.pdf"); fig.savefig(f"{OUT}/fig3_heatmap.png", dpi=300)

# ---------------- Figure 4: cost vs accuracy ----------------
k10 = A[A.k == 10].assign(thr=lambda d: d.n_test / d.pred_sec)
agg = k10.groupby("model").agg(thr=("thr", "median"), pr=("pr_auc", "mean"), ba=("bacc", "mean"))
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7), sharex=True)
for ax, col, lab in [(axs[0], "pr", "PR-AUC"), (axs[1], "ba", "Balanced accuracy")]:
    for mod in ORDER:
        c, mk, _ = STY[mod]
        ax.scatter(agg.thr[mod], agg[col][mod], color=c, marker=mk, s=42, edgecolor="white", linewidth=0.8, zorder=3)
        o = {"pr": {"TabICL": (6, 5), "TabPFN": (6, -7), "XGBoost": (-6, 7), "MLP": (6, 6)},
             "ba": {"GBT": (-6, -9), "LightGBM": (6, 4), "XGBoost": (6, -2)}}[col].get(mod, (6, 0))
        ax.annotate(mod, (agg.thr[mod], agg[col][mod]), xytext=o, textcoords="offset points",
                    fontsize=7, color=INK, va="center", ha="right" if o[0] < 0 else "left")
    ax.set_xscale("log"); ax.set_xlim(500, 4e6); ax.set_ylabel(lab)
    ax.set_xlabel("Prediction throughput (flows/s, log scale)"); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
axs[0].set_title("(a) PR-AUC vs throughput, $k$ = 10", fontsize=9)
axs[1].set_title("(b) Balanced accuracy vs throughput, $k$ = 10", fontsize=9)
axs[0].text(0.02, 0.03, "TFMs: one T4 GPU\nclassical: 4 CPU cores", transform=axs[0].transAxes, fontsize=7, color="#555555")
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_cost.pdf"); fig.savefig(f"{OUT}/fig4_cost.png", dpi=300)
print("figures written to", OUT)
