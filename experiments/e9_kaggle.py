# E9 — Hakem eleştirileri için ek deney (TRACE-IDS protokolü, Rayamajhi 2026, MIT lisansı)
#
# E8 ile birebir aynı bölmeler, tohumlar ve k çekimleri (test seti 20.000 akış). Yalnızca şunları ekler:
#   1) Güçlü GBDT kütüphaneleri: LightGBM, XGBoost, CatBoost (target-only kolu).
#   2) Eşik bağımsız metrikler: tüm target-only modelleri için olasılık tabanlı ROC-AUC ve PR-AUC,
#      ayrıca tam karmaşıklık matrisi (TP, FP, TN, FN).
#   3) Süre ölçümleri: eğitim/bağlam süresi ile tahmin süresi ayrı ayrı; tohum 0, k=10'da tek akış ve
#      1.000 akışlık toplu tahmin gecikmesi; donanım bilgisi (e9_hardware.json).
#   4) TTL ablasyonu: MIN_TTL ve MAX_TTL çıkarılarak (orijinal çalışmanın tanımı) target-only kolunun
#      tamamı ve TabPFN pooled-context (S=500; k = 1, 10, 100) yeniden koşulur.
#
# Kaggle: GPU T4 x2, Internet On, Secrets'ta TABPFN_TOKEN bağlı, NetFlow v3 veri seti ekli.
# "Save Version -> Save & Run All". Yarım kalırsa çıktısını Input olarak ekleyip tekrar çalıştırın.
# İki GPU, notebook ve script modunda da iş parçacıklarıyla (thread) kullanılır.
import glob, json, os, platform, shutil, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

# ------------------------------- ayarlar -------------------------------
QUICK = os.environ.get("E9_QUICK") == "1"
WORK = Path(os.environ.get("E9_WORK", "/kaggle/working"))
REPO = WORK / "TRACE-IDS"
DATASETS = ["cic", "unsw", "ton", "bot"]
SEEDS = [0] if QUICK else list(range(10))
K_VALUES = [1, 5, 10, 50, 100]
K_POOLED_NOTTL = [1, 10, 100]                      # TTL'siz pooled-context k değerleri
S_MAIN = 50 if QUICK else 500
SOURCE_N = 2_000 if QUICK else 100_000
TEST_N = 2_000 if QUICK else 20_000
LAT_K, LAT_SEED, LAT_REPS = 10, 0, 5               # gecikme ölçümü yapılan k ve tohum
MAX_HOURS = float(os.environ.get("E9_MAX_HOURS", 0.3 if QUICK else 9.5))  # KALAN KOTAYA GÖRE AYARLAYIN
N_THREADS_CPU = 4                                  # Kaggle CPU çekirdek sayısı
MODELS_TO = ["RF", "GBT", "MLP", "LightGBM", "XGBoost", "CatBoost", "TabPFN", "TabICL"]
T0 = time.time()
LOCK = threading.Lock()


def log(*a):
    with LOCK:
        print(f"[{(time.time() - T0) / 60:6.1f} dk]", *a, flush=True)


# ------------------------------- kurulum -------------------------------
def setup():
    if not QUICK:
        try:
            from kaggle_secrets import UserSecretsClient
            os.environ["TABPFN_TOKEN"] = UserSecretsClient().get_secret("TABPFN_TOKEN")
        except Exception as e:
            sys.exit(f"TABPFN_TOKEN okunamadı — Add-ons > Secrets'ta bu notebook'a bağlayın: {e}")
        for pkg in ["tabpfn", "tabicl", "lightgbm", "xgboost", "catboost"]:
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
    out = WORK / "e9_runs.csv"
    prev = glob.glob("/kaggle/input/**/e9_runs*.csv", recursive=True)
    if prev and not out.exists():
        pd.concat([pd.read_csv(p) for p in prev], ignore_index=True).to_csv(out, index=False)
        log("önceki sonuçlar alındı:", prev)


def hardware_info():
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "cpu_threads_used_by_boosters": N_THREADS_CPU}
    try:
        import torch
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info["torch"] = torch.__version__
    except Exception:
        info["gpus"] = []
    for pkg in ["sklearn", "lightgbm", "xgboost", "catboost", "tabpfn", "tabicl"]:
        try:
            info[pkg] = getattr(__import__(pkg), "__version__", "?")
        except Exception:
            info[pkg] = None
    try:
        info["cpu_model"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo")
                                 if l.startswith("model name"))
    except Exception:
        pass
    return info


# ------------------------------- modeller -------------------------------
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split


def leaf_size(n):
    return max(1, min(20, n // 10))        # TRACE-IDS GBT kuralı; GBDT kütüphanelerine de uygulanır


def make_classic(model, n, seed):
    """Klasik ve GBDT modelleri. RF/GBT/MLP ayarları TRACE-IDS ile birebir aynı."""
    if model == "RF":
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=N_THREADS_CPU)
    if model == "GBT":
        return HistGradientBoostingClassifier(max_iter=200, random_state=seed, early_stopping=False,
                                              min_samples_leaf=leaf_size(n))
    if model == "MLP":
        return MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=256, learning_rate_init=1e-3,
                             max_iter=15, random_state=seed)
    if QUICK:                                # yerel testte GBDT kütüphaneleri yok
        return LogisticRegression(max_iter=300)
    if model == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=200, min_child_samples=leaf_size(n), min_data_in_bin=1,
                              random_state=seed, n_jobs=N_THREADS_CPU, verbose=-1)
    if model == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=200, tree_method="hist", random_state=seed, n_jobs=N_THREADS_CPU,
                             eval_metric="logloss")
    if model == "CatBoost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=500, random_seed=seed, thread_count=N_THREADS_CPU, verbose=0,
                                  allow_writing_files=False)
    raise ValueError(model)


def make_tfms(device):
    if QUICK:
        return {"TabPFN": LogisticRegression(max_iter=300), "TabICL": LogisticRegression(max_iter=300, C=0.5)}
    from tabpfn import TabPFNClassifier
    from tabicl import TabICLClassifier
    try:
        icl = TabICLClassifier(device=device)
    except TypeError:
        icl = TabICLClassifier()
    return {"TabPFN": TabPFNClassifier(device=device), "TabICL": icl}


def evaluate(y_true, proba, classes):
    y_true = np.asarray(y_true)
    pred = np.asarray(classes)[np.argmax(proba, axis=1)]
    p1 = proba[:, list(classes).index(1)]
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn), "recall": rec, "precision": prec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "fpr": fpr, "bacc": (rec + 1 - fpr) / 2,
            "roc_auc": roc_auc_score(y_true, p1), "pr_auc": average_precision_score(y_true, p1)}


def sync_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


# ------------------------------- bir iş: (hedef, tohum) -------------------------------
def run_job(data, tgt, seed, tfms, device):
    from common import stats, z, stratified_sample, TTL
    Xt, yt = data[tgt]
    rng = np.random.default_rng(seed)                                   # E8/TRACE-IDS ile aynı çekimler
    Xpool, Xtest, ypool, ytest = train_test_split(Xt, yt, test_size=0.5, random_state=seed, stratify=yt)
    test_idx = stratified_sample(ytest, TEST_N, seed)
    Xtest, ytest = Xtest.loc[test_idx], ytest.loc[test_idx]
    atk_order = rng.permutation(ypool.index[ypool == 1].to_numpy())
    ben_order = rng.permutation(ypool.index[ypool == 0].to_numpy())
    shots = {k: np.concatenate([atk_order[:k], ben_order[:k]]) for k in K_VALUES}
    feats = {"all": list(Xt.columns), "no-ttl": [c for c in Xt.columns if c not in TTL]}
    rows = []

    def run(model, method, fset, source, S, k, Xtr, ytr, Xte, is_tfm):
        try:
            m = tfms[model] if is_tfm else make_classic(model, len(ytr), seed)
            if is_tfm and hasattr(m, "random_state"):
                m.set_params(random_state=seed)
            Xtr_fit, ytr_fit = Xtr, ytr
            if model == "MLP":                                          # TRACE-IDS küçük-n yeniden örneklemesi
                idx = np.random.default_rng(seed).choice(len(ytr), size=2_000, replace=True)
                Xtr_fit, ytr_fit = Xtr.iloc[idx], ytr.iloc[idx]
            Xa = np.asarray(Xtr_fit, dtype="float32")
            t = time.perf_counter()
            m.fit(Xa, np.asarray(ytr_fit))
            sync_cuda()
            fit_sec = time.perf_counter() - t
            t = time.perf_counter()
            proba = m.predict_proba(Xte)
            sync_cuda()
            pred_sec = time.perf_counter() - t
            row = {"target": tgt, "source": source, "seed": seed, "model": model, "method": method,
                   "features": fset, "S": S, "k": k, "n_context": len(ytr), "n_test": len(ytest),
                   "device": device if is_tfm else "cpu", "fit_sec": round(fit_sec, 4),
                   "pred_sec": round(pred_sec, 4), **evaluate(ytest, proba, m.classes_)}
            if method == "target-only" and fset == "all" and k == LAT_K and seed == LAT_SEED:
                for n_rows, key in [(1, "lat1_ms"), (1000, "lat1000_ms")]:
                    xs = Xte[:n_rows]
                    m.predict_proba(xs)                                 # ısınma
                    times = []
                    for _ in range(LAT_REPS):
                        t = time.perf_counter()
                        m.predict_proba(xs)
                        sync_cuda()
                        times.append(time.perf_counter() - t)
                    row[key] = round(1000 * float(np.median(times)), 3)
            rows.append(row)
        except Exception as e:
            log(f"  {model} {method} {fset} {source} k={k} HATA: {e}")

    for fset, cols in feats.items():
        Xp, Xte_df = Xpool[cols], Xtest[cols]
        ms, ss = stats(Xp)                                              # yalnızca hedef havuzu (klasikler için)
        Xte_raw, Xte_z = Xte_df.to_numpy("float32"), z(Xte_df, ms, ss).to_numpy("float32")
        for k, shot in shots.items():
            Xk, yk = Xp.loc[shot], ypool.loc[shot]
            for model in MODELS_TO:
                tfm = model in ("TabPFN", "TabICL")
                run(model, "target-only", fset, "", 0, k, Xk if tfm else z(Xk, ms, ss), yk,
                    Xte_raw if tfm else Xte_z, tfm)

    # TTL'siz TabPFN pooled-context (TTL'li karşılığı E8'de, aynı bağlamla)
    cols = feats["no-ttl"]
    Xte_raw = Xtest[cols].to_numpy("float32")
    for si, src in enumerate(DATASETS):
        if src == tgt:
            continue
        Xs, ys = data[src]
        src_idx = stratified_sample(ys, SOURCE_N, seed)                 # E8 ile aynı kaynak örneği
        ysrc = ys.loc[src_idx]
        srng = np.random.default_rng([seed, si, 8])                     # E8 ile aynı bağlam
        perm = {c: srng.permutation(ysrc.index[ysrc == c].to_numpy()) for c in (0, 1)}
        cidx = np.concatenate([perm[c][:S_MAIN] for c in (0, 1)])
        Xs5, ys5 = Xs.loc[cidx, cols], ys.loc[cidx]
        for k in K_POOLED_NOTTL:
            Xk, yk = Xpool.loc[shots[k], cols], ypool.loc[shots[k]]
            run("TabPFN", "pooled-context", "no-ttl", src, S_MAIN, k, pd.concat([Xs5, Xk]),
                pd.concat([ys5, yk]), Xte_raw, True)
    return rows


# ------------------------------- iş dağıtımı -------------------------------
def load_done():
    files = sorted(WORK.glob("e9_runs*.csv"))
    runs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()
    if len(runs):
        runs["source"] = runs["source"].fillna("")
        runs = runs.drop_duplicates(["target", "source", "seed", "model", "method", "features", "S", "k"])
    return runs


def worker(wid, jobs, data, device):
    try:
        tfms = make_tfms(device)
    except Exception as e:
        log(f"işçi {wid} ({device}) başlatılamadı: {e}")
        return
    out = WORK / f"e9_runs_w{wid}.csv"
    log(f"işçi {wid} ({device}): {len(jobs)} iş")
    for n, (seed, tgt) in enumerate(jobs, 1):
        if (time.time() - T0) / 3600 > MAX_HOURS:
            log(f"işçi {wid}: süre sınırı, duruyorum")
            return
        t = time.time()
        rows = run_job(data, tgt, seed, tfms, device)
        with LOCK:
            pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
        log(f"işçi {wid}: hedef {tgt} tohum {seed}: {time.time() - t:.0f} sn | {n}/{len(jobs)}")


def remaining_jobs():
    done = load_done()
    keys = set(zip(done.target, done.seed)) if len(done) else set()
    return [(s, t) for s in SEEDS for t in DATASETS if (t, s) not in keys]


def main():
    setup()
    hw = hardware_info()
    (WORK / "e9_hardware.json").write_text(json.dumps(hw, indent=2))
    log("donanım:", hw.get("gpus"), "| CPU:", hw.get("cpu_model"), "|", hw.get("cpu_count"), "çekirdek")
    from common import load
    data = {n: load(n) for n in DATASETS}
    jobs = remaining_jobs()
    log(f"kalan iş: {len(jobs)}/{len(SEEDS) * len(DATASETS)}")
    try:
        import torch
        n_gpu = torch.cuda.device_count()
    except ImportError:
        n_gpu = 0
    devices = ["cpu", "cpu"] if QUICK else ([f"cuda:{i}" for i in range(n_gpu)] or ["cpu"])
    if len(devices) > 1:
        log(f"{len(devices)} cihazla paralel çalışılıyor: {devices}")
        with ThreadPoolExecutor(len(devices)) as ex:
            for f in [ex.submit(worker, i, jobs[i::len(devices)], data, d) for i, d in enumerate(devices)]:
                f.result()
    left = remaining_jobs()                                             # başlatılamayan işçinin işleri
    if left and (time.time() - T0) / 3600 < MAX_HOURS:
        log(f"kalan {len(left)} iş tek cihazda tamamlanıyor")
        worker(len(devices), left, data, devices[0])
    runs = load_done()
    runs.to_csv(WORK / "e9_runs.csv", index=False)
    for f in WORK.glob("e9_runs_w*.csv"):
        f.unlink()
    log(f"toplam biten iş: {runs[['target', 'seed']].drop_duplicates().shape[0]}/{len(SEEDS) * len(DATASETS)}")
    if len(runs):
        to = runs[(runs.method == "target-only") & (runs.features == "all")]
        log("Target-only BA (tüm özellikler):\n" + to.pivot_table(index="k", columns="model", values="bacc")
            .round(3).to_string())
        log("Target-only ROC-AUC (tüm özellikler):\n" + to.pivot_table(index="k", columns="model",
                                                                    values="roc_auc").round(3).to_string())


if __name__ == "__main__":
    main()
