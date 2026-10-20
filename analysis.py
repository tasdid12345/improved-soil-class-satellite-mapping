"""Baseline (standard random forest on raw 36 features) vs improved (class-balanced
LightGBM on raw features + 3x3 neighbourhood summaries) on OpenML 182 (Statlog Landsat)."""
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless plotting on CI
import matplotlib.pyplot as plt
import lightgbm as lgb
import openml
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

SEED = 42
SEEDS = [42, 43, 44]          # model seeds (the train/test split is fixed at SEED)
DATA_ID = 182                 # OpenML Statlog Landsat Satellite
SOURCE_URL = "https://www.openml.org/d/182"
MAX_ROWS = 200_000            # subsample cap for runtime safety (dataset is ~6.4k rows)
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def fetch_openml(tries=5):
    """Download via the openml package, retrying with exponential backoff on errors."""
    for attempt in range(tries):
        try:
            ds = openml.datasets.get_dataset(DATA_ID)
            X, y, _, _ = ds.get_data(dataset_format="dataframe", target=ds.default_target_attribute)
            return X, y
        except Exception as e:  # network/HTTP errors (429/5xx surface here)
            wait = 2 ** attempt
            log(f"download failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    sys.exit("ERROR: could not download OpenML dataset 182")


def neighbourhood_features(X):
    """X: (n, 36) = 9 pixels of a 3x3 window x 4 bands (pixel-major order).
    Returns raw features + per-band window mean, window std, and normalised band differences of means."""
    n = X.shape[0]
    P = X.reshape(n, 9, 4)
    mean = P.mean(axis=1)                       # (n, 4) per-band window mean
    std = P.std(axis=1)                         # (n, 4) per-band window std
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    ratio = np.column_stack([(mean[:, i] - mean[:, j]) / (mean[:, i] + mean[:, j] + 1e-6)
                             for i, j in pairs])  # (n, 6) band ratios
    return np.hstack([X, mean, std, ratio])


def evaluate(model, Xa, ya, Xb, yb):
    """Shared evaluation: fit on train, score on the held-out test set."""
    p = model.fit(Xa, ya).predict(Xb)
    return {"f1": f1_score(yb, p, average="macro"),
            "acc": accuracy_score(yb, p),
            "bacc": balanced_accuracy_score(yb, p)}


def main():
    X_df, y_raw = fetch_openml()
    X = X_df.to_numpy(dtype=float)
    y = LabelEncoder().fit_transform(y_raw.astype(str))
    if len(y) > MAX_ROWS:
        idx = np.random.RandomState(SEED).choice(len(y), MAX_ROWS, replace=False)
        X, y = X[idx], y[idx]
    log(f"loaded {X.shape[0]} rows, {X.shape[1]} features, {len(np.unique(y))} classes")

    # Single stratified 75/25 split, fixed seed; test set used only at the end
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    F_tr, F_te = neighbourhood_features(X_tr), neighbourhood_features(X_te)
    log(f"train={len(y_tr)} test={len(y_te)}; improved feature count={F_tr.shape[1]}")

    # Tune the improved model ONLY with 3-fold CV on the training split
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    search = RandomizedSearchCV(
        lgb.LGBMClassifier(class_weight="balanced", n_estimators=300, subsample_freq=1,
                           random_state=SEED, verbose=-1, n_jobs=2),
        param_distributions={"learning_rate": [0.03, 0.1], "num_leaves": [15, 31, 63],
                             "min_child_samples": [10, 20, 40], "subsample": [0.8, 1.0],
                             "colsample_bytree": [0.5, 0.8]},
        n_iter=8, scoring="f1_macro", cv=cv, random_state=SEED, n_jobs=1)
    search.fit(F_tr, y_tr)
    log(f"LightGBM CV macro-F1={search.best_score_:.4f} params={search.best_params_}")

    # Final evaluation: same split, same metrics, models run with seeds 42/43/44
    res = {"baseline": [], "improved": []}
    for s in SEEDS:
        base = RandomForestClassifier(n_estimators=500, random_state=s, n_jobs=-1)
        imp = clone(search.best_estimator_).set_params(random_state=s)
        res["baseline"].append(evaluate(base, X_tr, y_tr, X_te, y_te))
        res["improved"].append(evaluate(imp, F_tr, y_tr, F_te, y_te))
        log(f"seed {s}: RF macro-F1={res['baseline'][-1]['f1']:.4f} "
            f"LGBM macro-F1={res['improved'][-1]['f1']:.4f}")

    def agg(name, key):
        v = np.array([r[key] for r in res[name]])
        return float(v.mean()), float(v.std())

    b_f1, b_f1_sd = agg("baseline", "f1")
    i_f1, i_f1_sd = agg("improved", "f1")
    if not np.all(np.isfinite([b_f1, i_f1])):
        sys.exit("ERROR: non-finite metrics")

    metrics = {
        "baseline_score": b_f1, "improved_score": i_f1, "improvement": i_f1 - b_f1,
        "baseline_score_std": b_f1_sd, "improved_score_std": i_f1_sd,
        "baseline_accuracy": agg("baseline", "acc")[0], "improved_accuracy": agg("improved", "acc")[0],
        "baseline_balanced_accuracy": agg("baseline", "bacc")[0],
        "improved_balanced_accuracy": agg("improved", "bacc")[0],
        "lgbm_cv_macro_f1": float(search.best_score_),
    }

    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Random forest\n(baseline)", "LightGBM +\nneighbourhood"], [b_f1, i_f1],
           yerr=[b_f1_sd, i_f1_sd], capsize=6, color=["grey", "steelblue"])
    ax.set_ylabel("Test macro-F1 (mean ± sd over 3 seeds)")
    ax.set_title("OpenML 182 land/soil classes")
    fig.tight_layout()
    fig.savefig("figures/macro_f1_comparison.png", dpi=120)

    out = {
        "question": "Can a class-balanced LightGBM model with spectral-neighbourhood features beat a standard random forest at classifying land and soil surface classes from multispectral Sentinel-2-like pixels?",
        "method": "Stratified 75/25 split (seed 42). Baseline: random forest on raw 36 features. Improved: class-balanced LightGBM on raw + 3x3 neighbourhood mean/std/band-ratio features, hyper-parameters tuned by 3-fold CV on the training split only. Test set used once; models run with seeds 42-44.",
        "dataset": {"name": "OpenML 182 Statlog Landsat Satellite", "source_url": SOURCE_URL,
                    "n_rows": int(len(y)), "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
                    "same_as_paper": False, "baseline_is_papers_method": False},
        "metric_name": "macro_f1",
        "higher_is_better": True,
        "baseline": "Standard RandomForestClassifier (500 trees, default settings, no class weighting) on the raw 36 spectral features.",
        "improved": "Class-balanced LightGBM (class_weight=balanced) on raw features plus per-band 3x3 window mean, std and normalised band-ratio features; learning rate, leaves, min child samples, subsampling and column sampling tuned by 8-candidate randomized search with 3-fold CV on the training set.",
        "metrics": metrics,
        "notes": "Stand-in for Sentinel-2 soil mapping, not the paper's HorusApp pipeline or data. Assumes OpenML features are pixel-major (9 pixels x 4 bands) for the neighbourhood features. Only the improved model is tuned; the baseline uses default RF settings, so the comparison is not budget-matched in tuning. Single fixed split, so test estimates have sampling uncertainty; std reflects model seed only. The openml package's own retry/timeout settings are used for download.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log(f"done: baseline macro-F1={b_f1:.4f}, improved macro-F1={i_f1:.4f}; wrote results.json")


if __name__ == "__main__":
    main()
