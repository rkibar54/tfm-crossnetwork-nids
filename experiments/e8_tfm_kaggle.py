# E8 — E7'nin üç açığını kapatan ek deney (TRACE-IDS protokolü, Rayamajhi 2026, MIT lisansı)
#
#  1) Ölçek bağımsızlığı: TFM'ler HAM özelliklerle çalışır (kendi ön işlemelerini yaparlar).
#     Böylece target-only sonucu artık kaynağa bağlı değil; her (hedef, tohum) için bir kez hesaplanır.
#  2) İkinci TFM: TabICL (lisans onayı gerektirmez).
#  3) Kaynak bağlam boyutu ablasyonu: TabPFN pooled-context, sınıf başına S ∈ {100, 500, 2000} kaynak akışı
#     (S=500 tüm k'larda, 100 ve 2000 yalnızca k ∈ {1, 10}).
#  Klasik target-only (RF/GBT/MLP) aynı test setinde yeniden koşulur (CPU, hızlı). Ölçekleme yalnızca hedef
#  havuzunun istatistikleriyle yapılır, yani target-only hiçbir koşulda kaynağa bağlı değildir.
#
# Bölme protokolü E7/TRACE-IDS ile aynı (aynı tohumlarla aynı havuz/test bölmesi ve aynı iç içe k çekimleri);
# test seti süre nedeniyle 20k satır.
#
# Kaggle: GPU T4 x2, Internet On, Secrets'ta TABPFN_TOKEN bağlı, NetFlow v3 veri seti ekli.
# "Save Version -> Save & Run All". Yarım kalırsa çıktısını Input olarak ekleyip tekrar çalıştırın.
import glob, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

# ------------------------------- ayarlar -------------------------------
QUICK = os.environ.get("E8_QUICK") == "1"
WORK = Path(os.environ.get("E8_WORK", "/kaggle/working"))
REPO = WORK / "TRACE-IDS"
DATASETS = ["cic", "unsw", "ton", "bot"]
SEEDS = [0] if QUICK else list(range(10))
K_VALUES = [1, 5, 10, 50, 100]
K_ABL = [1, 10]                                     # S ablasyonunun yapıldığı k değerleri
S_MAIN, S_ABL = (50, [20, 100]) if QUICK else (500, [100, 2000])
SOURCE_N = 2_000 if QUICK else 100_000
TEST_N = 2_000 if QUICK else 20_000
MAX_HOURS = float(os.environ.get("E8_MAX_HOURS", 0.2 if QUICK else 9.5))  # KALAN KOTAYA GÖRE AYARLAYIN
OUT = "e8_runs.csv"
T0 = float(os.environ.get("E8_T0", time.time()))


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f} dk]", *a, flush=True)


# ------------------------------- kurulum -------------------------------
def setup():
    if not QUICK:
        try:
            from kaggle_secrets import UserSecretsClient
            os.environ["TABPFN_TOKEN"] = UserSecretsClient().get_secret("TABPFN_TOKEN")
        except Exception as e:
            sys.exit(f"TABPFN_TOKEN okunamadı — Add-ons > Secrets'ta bu notebook'a bağlayın: {e}")
        for pkg in ["tabpfn", "tabicl"]:
            try:
                __import__(pkg)
            except ImportError:
                log(f"{pkg} kuruluyor")
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
    if not REPO.exists():
        subprocess.run(["git", "clone", "-q", "https://github.com/rijan-rayamajhi/TRACE-IDS.git", str(REPO)],
                       check=True)
    os.chdir(REPO)
    sys.path.insert(0, str(REPO / "src"))
    if QUICK:
        return
    proc = REPO / "data/processed"
    proc.mkdir(parents=True, exist_ok=True)
    missing = [n for n in DATASETS if not (proc / f"{n}.parquet").exists()]
    for n in list(missing):
        prev = glob.glob(f"/kaggle/input/**/processed/{n}.parquet", recursive=True)
        if prev:
            shutil.copy(prev[0], proc / f"{n}.parquet")
            missing.remove(n)
            log(f"{n}: önceki çıktıdan alındı")
    if missing:
        raw = REPO / "data/raw"
        raw.mkdir(parents=True, exist_ok=True)
        csvs = glob.glob("/kaggle/input/**/*.csv", recursive=True)
        words = {"cic": ["cic", "2018"], "unsw": ["unsw"], "ton": ["ton"], "bot": ["bot"]}
        for n in missing:
            hits = [c for c in csvs if "v3" in c.lower() and any(w in Path(c).name.lower() for w in words[n])]
            if n == "ton":
                hits = [c for c in hits if "bot" not in Path(c).name.lower()]
            if len(hits) != 1:
                found = sorted(glob.glob("/kaggle/input/**/*", recursive=True))[:50]
                sys.exit(f"{n}: {len(hits)} CSV eşleşmesi {hits}\n/kaggle/input içeriği: {found}\n"
                         "NetFlow v3 veri setini Add Input ile ekleyin.")
            link = raw / f"{n}.csv"
            if not link.exists():
                link.symlink_to(hits[0])
        log("parquet üretiliyor:", missing)
        subprocess.run([sys.executable, "src/data_loader.py", "build", *missing], check=True)
    out = WORK / OUT
    prev = glob.glob("/kaggle/input/**/e8_runs*.csv", recursive=True)
    if prev and not out.exists():
        pd.concat([pd.read_csv(p) for p in prev], ignore_index=True).to_csv(out, index=False)
        log("önceki sonuçlar alındı:", prev)


# --------- yazarın klasik modelleri (TRACE-IDS src/e2_transfer.py'den birebir, MIT lisansı) ---------
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


def fit_classic(model, X, y, seed):
    if model == "RF":
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1).fit(X, y)
    if model == "GBT":
        leaf = max(1, min(20, len(y) // 10))
        return HistGradientBoostingClassifier(max_iter=200, random_state=seed, early_stopping=False,
                                              min_samples_leaf=leaf).fit(X, y)
    idx = np.random.default_rng(seed).choice(len(y), size=2_000, replace=True)  # yazarın küçük-n yeniden örneklemesi
    return MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=256, learning_rate_init=1e-3,
                         max_iter=15, random_state=seed).fit(X.iloc[idx], y.iloc[idx])


# ------------------------------- TFM'ler -------------------------------
def make_tfms():
    if QUICK:
        from sklearn.linear_model import LogisticRegression
        return {"TabPFN": LogisticRegression(max_iter=300), "TabICL": LogisticRegression(max_iter=300, C=0.5)}
    import tabpfn, tabicl
    from tabpfn import TabPFNClassifier
    from tabicl import TabICLClassifier
    log(f"tabpfn {getattr(tabpfn, '__version__', '?')} | tabicl {getattr(tabicl, '__version__', '?')}")
    try:
        icl = TabICLClassifier(device="cuda")
    except TypeError:
        icl = TabICLClassifier()
    return {"TabPFN": TabPFNClassifier(device="cuda"), "TabICL": icl}


def metrics(y_true, y_pred):
    from common import scores
    s = scores(y_true, y_pred)
    s["bacc"] = (s["recall"] + 1 - s["fpr"]) / 2
    return s


# ------------------------------- bir iş: (hedef, tohum) -------------------------------
def run_job(data, tgt, seed, tfms):
    from common import stats, z, stratified_sample
    Xt, yt = data[tgt]
    rng = np.random.default_rng(seed)                                   # TRACE-IDS ile aynı çekimler
    Xpool, Xtest, ypool, ytest = train_test_split(Xt, yt, test_size=0.5, random_state=seed, stratify=yt)
    test_idx = stratified_sample(ytest, TEST_N, seed)
    Xtest, ytest = Xtest.loc[test_idx], ytest.loc[test_idx]
    atk_order = rng.permutation(ypool.index[ypool == 1].to_numpy())
    ben_order = rng.permutation(ypool.index[ypool == 0].to_numpy())
    ms, ss = stats(Xpool)                                               # yalnızca hedef havuzu (klasikler için)
    Xte_raw = Xtest.to_numpy("float32")
    Xte_z = z(Xtest, ms, ss)

    rows = []

    def add(model, method, source, S, k, pred, sec):
        rows.append({"target": tgt, "source": source, "seed": seed, "model": model, "method": method,
                     "S": S, "k": k, "sec": round(sec, 1), **metrics(ytest, pred)})

    def tfm(name, Xc, yc, method, source, S, k):
        m, t = tfms[name], time.time()
        try:
            if hasattr(m, "random_state"):
                m.set_params(random_state=seed)
            m.fit(np.asarray(Xc, dtype="float32"), np.asarray(yc))
            add(name, method, source, S, k, m.predict(Xte_raw), time.time() - t)
        except Exception as e:
            log(f"  {name} {method} {source} S={S} k={k} HATA: {e}")

    shots = {k: np.concatenate([atk_order[:k], ben_order[:k]]) for k in K_VALUES}
    for k, shot in shots.items():                                       # target-only: kaynaktan bağımsız
        Xk, yk = Xpool.loc[shot], ypool.loc[shot]
        for model in ["RF", "GBT", "MLP"]:
            t = time.time()
            add(model, "target-only", "", 0, k, fit_classic(model, z(Xk, ms, ss), yk, seed).predict(Xte_z),
                time.time() - t)
        for name in ["TabPFN", "TabICL"]:
            tfm(name, Xk, yk, "target-only", "", 0, k)

    for si, src in enumerate(DATASETS):
        if src == tgt:
            continue
        Xs, ys = data[src]
        src_idx = stratified_sample(ys, SOURCE_N, seed)                 # TRACE-IDS ile aynı kaynak örneği
        ysrc = ys.loc[src_idx]
        srng = np.random.default_rng([seed, si, 8])
        perm = {c: srng.permutation(ysrc.index[ysrc == c].to_numpy()) for c in (0, 1)}

        def ctx(S):                                                     # iç içe: S=100 ⊂ S=500 ⊂ S=2000
            idx = np.concatenate([perm[c][:S] for c in (0, 1)])
            return Xs.loc[idx], ys.loc[idx]

        Xs5, ys5 = ctx(S_MAIN)
        for name in ["TabPFN", "TabICL"]:
            tfm(name, Xs5, ys5, "source-only", src, S_MAIN, 0)
        for k, shot in shots.items():
            Xk, yk = Xpool.loc[shot], ypool.loc[shot]
            for name in ["TabPFN", "TabICL"]:
                tfm(name, pd.concat([Xs5, Xk]), pd.concat([ys5, yk]), "pooled-context", src, S_MAIN, k)
            if k in K_ABL:
                for S in S_ABL:
                    Xsa, ysa = ctx(S)
                    tfm("TabPFN", pd.concat([Xsa, Xk]), pd.concat([ysa, yk]), "pooled-context", src, S, k)
    return rows


# ------------------------------- iş dağıtımı -------------------------------
def load_done():
    files = sorted(WORK.glob("e8_runs*.csv"))
    runs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()
    if len(runs):
        runs["source"] = runs["source"].fillna("")
        runs = runs.drop_duplicates(["target", "source", "seed", "model", "method", "S", "k"])
    return runs


def worker(wid, n_workers):
    os.chdir(REPO)
    sys.path.insert(0, str(REPO / "src"))
    from common import load
    data = {n: load(n) for n in DATASETS}
    done = load_done()
    done_keys = set(zip(done.target, done.seed)) if len(done) else set()
    tfms = make_tfms()
    jobs = [(seed, t) for seed in SEEDS for t in DATASETS]
    mine = [j for i, j in enumerate(jobs) if i % n_workers == wid and (j[1], j[0]) not in done_keys]
    out = WORK / f"e8_runs_w{wid}.csv"
    log(f"GPU {wid}: {len(mine)} iş")
    for n, (seed, tgt) in enumerate(mine, 1):
        if (time.time() - T0) / 3600 > MAX_HOURS:
            log(f"GPU {wid}: süre sınırı, duruyorum")
            return
        t = time.time()
        new = pd.DataFrame(run_job(data, tgt, seed, tfms))
        new.to_csv(out, mode="a", header=not out.exists(), index=False)
        log(f"GPU {wid}: hedef {tgt} tohum {seed}: {time.time() - t:.0f} sn | {n}/{len(mine)}")


def main():
    if "E8_WORKER" in os.environ:
        return worker(int(os.environ["E8_WORKER"]), int(os.environ["E8_NW"]))
    setup()
    runs = load_done()
    if len(runs):
        runs.to_csv(WORK / OUT, index=False)
    for f in WORK.glob("e8_runs_w*.csv"):
        f.unlink()
    n_done = runs[["target", "seed"]].drop_duplicates().shape[0] if len(runs) else 0
    log(f"önceden biten iş: {n_done}/{len(SEEDS) * len(DATASETS)}")
    try:
        import torch
        n_gpu = max(1, torch.cuda.device_count())
    except ImportError:
        n_gpu = 1
    n_workers = 2 if QUICK else n_gpu
    script = globals().get("__file__")
    if n_workers == 1 or script is None:
        worker(0, 1)
    else:
        log(f"{n_workers} GPU ile paralel çalışılıyor")
        procs = [subprocess.Popen([sys.executable, script], cwd=str(REPO),
                                  env=dict(os.environ, E8_WORKER=str(i), E8_NW=str(n_workers), E8_T0=str(T0),
                                           CUDA_VISIBLE_DEVICES=str(i)))
                 for i in range(n_workers)]
        for p in procs:
            p.wait()
    runs = load_done()
    runs.to_csv(WORK / OUT, index=False)
    for f in WORK.glob("e8_runs_w*.csv"):
        f.unlink()
    log(f"toplam biten iş: {runs[['target', 'seed']].drop_duplicates().shape[0]}/{len(SEEDS) * len(DATASETS)}")
    finish(runs)


def finish(runs):
    if runs.empty:
        return
    g = runs.groupby(["target", "source", "model", "method", "S", "k"])[["f1", "bacc", "recall", "fpr"]]
    summ = g.mean().round(4).join(g.size().rename("n_seeds")).reset_index()
    summ.to_csv(WORK / "e8_summary.csv", index=False)
    to = summ[(summ.method == "target-only") & summ.k.isin([1, 5, 10])]
    log("Target-only dengeli doğruluk:\n" + to.pivot_table(index=["target", "k"], columns="model",
                                                             values="bacc").round(3).to_string())


if __name__ == "__main__":
    main()
