# E7 — Tablo temel modeli (TabPFN) Rayamajhi'nin sızıntısız protokolünde kaynak etkisini değiştirir mi?
#
# Protokol, TRACE-IDS (Rayamajhi 2026, MIT lisansı) src/e2_transfer.py içindeki run_one ile birebir aynı:
# hedef %50 etiketsiz havuz / %50 test, kaynaktan 100k sınıf dengeli örnek, iç içe k çekimleri,
# shared scaling. Tek sapma: test seti 200k yerine TEST_N (varsayılan 50k) satır.
#
# Kollar (her (çift, tohum, k) için):
#   RF / GBT / MLP target-only   -> yazarın target-only kolunun yeniden üretimi (referans)
#   TabPFN target-only           -> yalnızca k hedef akışı bağlamda
#   TabPFN pooled-context        -> SRC_PER_CLASS kaynak akışı/sınıf + k hedef akışı bağlamda
#   TabPFN source-only (k=0)     -> sıfır atış
#
# Kaggle'da kullanım: GPU T4, Internet On, Secrets'ta TABPFN_TOKEN bağlı, NetFlow v3 veri seti ekli.
# "Save Version -> Save & Run All" ile arka planda çalıştırın. Süre dolmadan kendini durdurur;
# önceki sürümün çıktısını Input olarak ekleyip tekrar çalıştırırsanız kaldığı yerden devam eder.
import glob, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

# ------------------------------- ayarlar -------------------------------
QUICK = os.environ.get("E7_QUICK") == "1"           # yerel duman testi için
WORK = Path(os.environ.get("E7_WORK", "/kaggle/working"))
REPO = WORK / "TRACE-IDS"
DATASETS = ["cic", "unsw", "ton", "bot"]
SEEDS = [0] if QUICK else list(range(10))
K_VALUES = [1, 5, 10, 50, 100]
SOURCE_N = 2_000 if QUICK else 100_000              # yazarınkiyle aynı
TEST_N = 2_000 if QUICK else 50_000                 # yazar: 200k (süre nedeniyle 50k)
SRC_PER_CLASS = 50 if QUICK else 500                # pooled-context kolunda kaynak bağlamı
MAX_HOURS = float(os.environ.get("E7_MAX_HOURS", 0.2 if QUICK else 5.5))  # KALAN KOTANIZA GÖRE AYARLAYIN
OUT_NAME = "e7_runs.csv"
T0 = float(os.environ.get("E7_T0", time.time()))   # alt süreçler ana sürecin saatini kullanır


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f} dk]", *a, flush=True)


# ------------------------------- kurulum -------------------------------
def setup():
    if not QUICK:                                   # önce token ve paket: eksikse 7 dk veri üretmeden dur
        try:
            from kaggle_secrets import UserSecretsClient
            os.environ["TABPFN_TOKEN"] = UserSecretsClient().get_secret("TABPFN_TOKEN")
        except Exception as e:
            sys.exit(f"TABPFN_TOKEN okunamadı — Add-ons > Secrets'ta bu notebook'a bağlayın: {e}")
        try:
            import tabpfn  # noqa: F401
        except ImportError:
            log("tabpfn kuruluyor")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabpfn"], check=True)
    if not REPO.exists():
        subprocess.run(["git", "clone", "-q", "https://github.com/rijan-rayamajhi/TRACE-IDS.git", str(REPO)],
                       check=True)
    os.chdir(REPO)
    sys.path.insert(0, str(REPO / "src"))
    if QUICK:                                       # yerel testte veri hazırlığı yok
        return
    proc = REPO / "data/processed"
    proc.mkdir(parents=True, exist_ok=True)
    missing = [n for n in DATASETS if not (proc / f"{n}.parquet").exists()]
    for n in list(missing):                         # önceki sürümün çıktısında hazır parquet var mı?
        prev = glob.glob(f"/kaggle/input/**/processed/{n}.parquet", recursive=True)
        if prev:
            shutil.copy(prev[0], proc / f"{n}.parquet")
            missing.remove(n)
            log(f"{n}: önceki çıktıdan alındı")
    if missing:                                     # yoksa ham CSV'den yazarın koduyla üret
        raw = REPO / "data/raw"
        raw.mkdir(parents=True, exist_ok=True)
        csvs = glob.glob("/kaggle/input/**/*.csv", recursive=True)
        words = {"cic": ["cic", "2018"], "unsw": ["unsw"], "ton": ["ton"], "bot": ["bot"]}
        for n in missing:
            hits = [c for c in csvs if "v3" in c.lower() and any(w in Path(c).name.lower() for w in words[n])]
            if n == "ton":
                hits = [c for c in hits if "bot" not in Path(c).name.lower()]
            if len(hits) != 1:
                sys.exit(f"{n}: {len(hits)} CSV eşleşmesi {hits}")
            link = raw / f"{n}.csv"
            if not link.exists():
                link.symlink_to(hits[0])
        log("parquet üretiliyor:", missing)
        subprocess.run([sys.executable, "src/data_loader.py", "build", *missing], check=True)
    # önceki sonuçları al (kaldığı yerden devam)
    out = WORK / OUT_NAME
    prev = glob.glob("/kaggle/input/**/e7_runs*.csv", recursive=True)
    if prev and not out.exists():
        pd.concat([pd.read_csv(p) for p in prev], ignore_index=True).to_csv(out, index=False)
        log("önceki sonuçlar alındı:", prev)


# --------- yazarın modelleri (TRACE-IDS src/e2_transfer.py'den birebir, MIT lisansı) ---------
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


def rf(seed, n=100):
    return RandomForestClassifier(n_estimators=n, random_state=seed, n_jobs=-1)


def gbt(seed, n=None):
    leaf = 20 if n is None else max(1, min(20, n // 10))
    return HistGradientBoostingClassifier(max_iter=200, random_state=seed, early_stopping=False,
                                          min_samples_leaf=leaf)


def mlp(seed):
    return MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=256,
                         learning_rate_init=1e-3, max_iter=15, random_state=seed)


def resample(X, y, w, n, seed):
    idx = np.random.default_rng(seed).choice(len(y), size=n, replace=True, p=w / w.sum())
    return X.iloc[idx], y.iloc[idx]


def fit_weighted(model, X, y, w, seed):
    if model == "RF":
        return rf(seed).fit(X, y, sample_weight=w)
    if model == "GBT":
        return gbt(seed, len(y)).fit(X, y, sample_weight=w)
    Xr, yr = resample(X, y, w, min(len(y), 100_000) if len(y) > 1_000 else 2_000, seed)
    return mlp(seed).fit(Xr, yr)


# ------------------------------- deney -------------------------------
def make_tabpfn():
    if QUICK:                                       # yerel testte TabPFN yerine hafif bir yedek
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=500)
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier(device="cuda")


def metrics(y_true, y_pred):
    from common import scores
    s = scores(y_true, y_pred)
    s["bacc"] = (s["recall"] + 1 - s["fpr"]) / 2
    return s


def run_one(Xs, ys, Xt, yt, seed, tfm):
    from common import stats, z, stratified_sample
    rng = np.random.default_rng(seed)                               # yazarınkiyle aynı sıra
    src_idx = stratified_sample(ys, SOURCE_N, seed)
    Xpool, Xtest, ypool, ytest = train_test_split(Xt, yt, test_size=0.5, random_state=seed, stratify=yt)
    test_idx = stratified_sample(ytest, TEST_N, seed)
    Xtest, ytest = Xtest.loc[test_idx], ytest.loc[test_idx]
    ms, ss = stats(pd.concat([Xs.loc[src_idx], Xpool]))             # shared scaling
    Xsrc, ysrc = z(Xs.loc[src_idx], ms, ss), ys.loc[src_idx]
    Xpool, Xtest = z(Xpool, ms, ss), z(Xtest, ms, ss)
    atk_order = rng.permutation(ypool.index[ypool == 1].to_numpy())  # yazarda importance_weights rng
    ben_order = rng.permutation(ypool.index[ypool == 0].to_numpy())  # kullanmaz; sıra korunur

    srng = np.random.default_rng(10_000 + seed)                      # TFM kaynak bağlamı (ayrı rng)
    src_ctx = np.concatenate([srng.choice(ysrc.index[ysrc == c].to_numpy(),
                                          min(SRC_PER_CLASS, int((ysrc == c).sum())), replace=False)
                              for c in (0, 1)])
    Xsc, ysc = Xsrc.loc[src_ctx], ysrc.loc[src_ctx]
    Xte = Xtest.to_numpy("float32")

    rows = []

    def add(model, method, k, pred, sec):
        rows.append({"seed": seed, "model": model, "method": method, "k": k, "sec": round(sec, 1),
                     **metrics(ytest, pred)})

    def tfm_run(Xc, yc, k, method):
        t = time.time()
        try:
            tfm.set_params(random_state=seed) if hasattr(tfm, "random_state") else None
            tfm.fit(Xc.to_numpy("float32"), yc.to_numpy())
            add("TabPFN", method, k, tfm.predict(Xte), time.time() - t)
        except Exception as e:
            log(f"  TabPFN {method} k={k} HATA: {e}")

    tfm_run(Xsc, ysc, 0, "source-only")
    for k in K_VALUES:
        shot = np.concatenate([atk_order[:k], ben_order[:k]])
        Xk, yk = Xpool.loc[shot], ypool.loc[shot]
        for model in ["RF", "GBT", "MLP"]:
            t = time.time()
            m = fit_weighted(model, Xk, yk, np.ones(len(yk)), seed)
            add(model, "target-only", k, m.predict(Xtest), time.time() - t)
        tfm_run(Xk, yk, k, "target-only")
        tfm_run(pd.concat([Xsc, Xk]), pd.concat([ysc, yk]), k, "pooled-context")
    return rows


def load_done():
    files = sorted(WORK.glob("e7_runs*.csv"))
    runs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()
    if len(runs):
        runs = runs.drop_duplicates(["direction", "seed", "model", "method", "k"])
    return runs


def worker(wid, n_workers):
    os.chdir(REPO)
    sys.path.insert(0, str(REPO / "src"))
    from common import load
    data = {n: load(n) for n in DATASETS}
    done = load_done()
    done_keys = set(zip(done.direction, done.seed)) if len(done) else set()
    tfm = make_tabpfn()
    pairs = [(s, t) for s in DATASETS for t in DATASETS if s != t]
    jobs = [(seed, s, t) for seed in SEEDS for s, t in pairs]       # önce tohum: ara sonuçlar tüm çiftleri kapsar
    mine = [j for i, j in enumerate(jobs) if i % n_workers == wid and (f"{j[1]}->{j[2]}", j[0]) not in done_keys]
    out = WORK / f"e7_runs_w{wid}.csv"
    log(f"GPU {wid}: {len(mine)} iş")
    for n, (seed, src, tgt) in enumerate(mine, 1):
        if (time.time() - T0) / 3600 > MAX_HOURS:
            log(f"GPU {wid}: süre sınırı, duruyorum")
            return
        t = time.time()
        d = f"{src}->{tgt}"
        new = pd.DataFrame([{"direction": d, **r} for r in run_one(*data[src], *data[tgt], seed, tfm)])
        new.to_csv(out, mode="a", header=not out.exists(), index=False)   # her işten sonra diske yaz
        log(f"GPU {wid}: {d} tohum {seed}: {time.time() - t:.0f} sn | {n}/{len(mine)}")


def main():
    if "E7_WORKER" in os.environ:                                       # alt süreç
        return worker(int(os.environ["E7_WORKER"]), int(os.environ["E7_NW"]))
    setup()
    runs = load_done()
    runs.to_csv(WORK / OUT_NAME, index=False) if len(runs) else None
    for f in WORK.glob("e7_runs_w*.csv"):                               # birleşti, eski parçaları sil
        f.unlink()
    done = runs[["direction", "seed"]].drop_duplicates().shape[0] if len(runs) else 0
    log(f"önceden biten iş: {done}/120")
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
        procs = []
        for i in range(n_workers):
            env = dict(os.environ, E7_WORKER=str(i), E7_NW=str(n_workers), E7_T0=str(T0),
                       CUDA_VISIBLE_DEVICES=str(i))
            procs.append(subprocess.Popen([sys.executable, script], env=env, cwd=str(REPO)))
        for p in procs:
            p.wait()
    runs = load_done()
    runs.to_csv(WORK / OUT_NAME, index=False)
    for f in WORK.glob("e7_runs_w*.csv"):
        f.unlink()
    log(f"toplam biten iş: {runs[['direction', 'seed']].drop_duplicates().shape[0]}/120")
    finish(runs)


def finish(runs):
    if runs.empty:
        return
    g = runs.groupby(["direction", "model", "method", "k"])[["f1", "bacc", "recall", "fpr"]]
    summ = g.mean().round(4).join(g.size().rename("n_seeds")).reset_index()
    summ.to_csv(WORK / "e7_summary.csv", index=False)
    piv = summ[summ.k.isin([1, 5, 10])].pivot_table(index=["direction", "k"], columns=["model", "method"],
                                                    values="bacc")
    log("Dengeli doğruluk (ortalama), k=1/5/10:\n" + piv.round(3).to_string())


if __name__ == "__main__":
    main()
