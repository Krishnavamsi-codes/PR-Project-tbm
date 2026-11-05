"""Shared harness: data, leakage-free preprocessing, CV runner, metrics.

Everything learned from data (target encoding, scaling, one-hot, PCA, LDA,
feature selection, class weights) lives inside sklearn Pipelines that are
re-fit on the TRAINING part of every CV fold only.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one thread per worker, parallelism is across folds

import json
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     StratifiedShuffleSplit, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------- config
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
QUICK = os.environ.get("PR_QUICK") == "1"
FULL = os.environ.get("PR_FULL") == "1"          # all 260k rows, 5-fold x 1
RES = ROOT / ("results_quick" if QUICK else "results_full" if FULL else "results")
FIG = ROOT / ("figures_quick" if QUICK else "figures_full" if FULL else "figures")
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

SEED = 42
N_SUB = 4000 if QUICK else 20000
N_SPLITS = 3 if QUICK else 5
N_REPEATS = 1 if (QUICK or FULL) else 3
N_JOBS = max(1, min(N_SPLITS * N_REPEATS, os.cpu_count() or 1))
CLASSES = [1, 2, 3]                      # damage_grade; internally 0,1,2

GEO = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]


# ------------------------------------------------------------------- data
def load_data():
    X = pd.read_csv(DATA / "train_values.csv")
    y = pd.read_csv(DATA / "train_labels.csv")
    assert (X.building_id.values == y.building_id.values).all()
    return X.drop(columns="building_id"), y.damage_grade.values - 1


def get_subsample():
    """Fixed-seed class-proportional subsample (kNN/SVM are too slow on 260k)."""
    X, y = load_data()
    if FULL:
        return X.reset_index(drop=True), y
    _, Xs, _, ys = train_test_split(X, y, test_size=N_SUB, stratify=y, random_state=SEED)
    return Xs.reset_index(drop=True), ys


def get_cv():
    return RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)


# ------------------------------------------------- feature engineering
class Engineer(BaseEstimator, TransformerMixin):
    """Stateless engineered features (no fitting -> no leakage possible)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        sup = [c for c in X.columns if c.startswith("has_superstructure_")]
        sec = [c for c in X.columns if c.startswith("has_secondary_use_")]
        X["log_age"] = np.log1p(X["age"])
        X["age_unknown"] = (X["age"] >= 995).astype(int)
        X["area_x_height"] = X["area_percentage"] * X["height_percentage"]
        X["area_per_height"] = X["area_percentage"] / X["height_percentage"]
        X["floors_x_height"] = X["count_floors_pre_eq"] * X["height_percentage"]
        X["height_per_floor"] = X["height_percentage"] / X["count_floors_pre_eq"]
        X["n_superstructure"] = X[sup].sum(axis=1)
        X["n_secondary_use"] = X[sec].sum(axis=1)
        X["is_rc"] = ((X["has_superstructure_rc_engineered"] + X["has_superstructure_rc_non_engineered"]) > 0).astype(int)
        X["is_mud_stone"] = (X[["has_superstructure_adobe_mud", "has_superstructure_mud_mortar_stone",
                                "has_superstructure_stone_flag", "has_superstructure_mud_mortar_brick"]].sum(axis=1) > 0).astype(int)
        return X


class GeoTargetEncoder(BaseEstimator, TransformerMixin):
    """Hierarchical smoothed target encoding of geo_level_1/2/3 ids.

    For each level l, the class distribution of the region is shrunk towards
    the parent level estimate:  p = (n_k + alpha * p_parent) / (n + alpha).
    fit_transform (used on the training fold) is OUT-OF-FOLD (inner k-fold), so
    a training row never sees its own label; transform (test fold) uses maps
    fit on the whole training fold. Output: P(grade1) and P(grade3) per level.
    """

    def __init__(self, alpha=20.0, n_inner=5, seed=0):
        self.alpha, self.n_inner, self.seed = alpha, n_inner, seed

    @staticmethod
    def _keys(X):
        g1 = X["geo_level_1_id"].values.astype(np.int64)
        g2 = g1 * 10_000 + X["geo_level_2_id"].values.astype(np.int64)
        g3 = g2 * 20_000 + X["geo_level_3_id"].values.astype(np.int64)
        return [g1, g2, g3]

    @staticmethod
    def _lookup(m, keys, fallback):
        p = m.reindex(keys).values
        miss = np.isnan(p[:, 0])
        p[miss] = fallback[miss]
        return p

    def _fit_maps(self, X, y):
        Y = np.eye(3)[y]
        prior = Y.mean(0)
        parent = np.tile(prior, (len(y), 1))
        maps = []
        for k in self._keys(X):
            d = pd.DataFrame(Y)
            d["k"] = k
            s = d.groupby("k").sum()
            n = s.sum(axis=1).values[:, None]
            pp = pd.DataFrame(parent)
            pp["k"] = k
            pp = pp.groupby("k").first().loc[s.index].values
            m = pd.DataFrame((s.values + self.alpha * pp) / (n + self.alpha), index=s.index)
            maps.append(m)
            parent = self._lookup(m, k, parent)
        return prior, maps

    def _apply(self, X, prior, maps):
        parent = np.tile(prior, (len(X), 1))
        cols = []
        for m, k in zip(maps, self._keys(X)):
            parent = self._lookup(m, k, parent)
            cols += [parent[:, 0], parent[:, 2]]
        return np.column_stack(cols)

    def _out(self, X, enc):
        names = [f"te_g{l}_c{c}" for l in (1, 2, 3) for c in (1, 3)]
        out = X.drop(columns=GEO).copy()
        for i, nme in enumerate(names):
            out[nme] = enc[:, i]
        return out

    def fit(self, X, y):
        self.prior_, self.maps_ = self._fit_maps(X, np.asarray(y))
        return self

    def transform(self, X):
        return self._out(X, self._apply(X, self.prior_, self.maps_))

    def fit_transform(self, X, y):
        y = np.asarray(y)
        self.fit(X, y)
        enc = np.zeros((len(X), 6))
        for tr, va in StratifiedKFold(self.n_inner, shuffle=True, random_state=self.seed).split(X, y):
            pr, mp = self._fit_maps(X.iloc[tr], y[tr])
            enc[va] = self._apply(X.iloc[va], pr, mp)
        return self._out(X, enc)


def prep_steps(te=False, eng=False):
    """Leakage-free preprocessing: [engineer] -> [target-encode] -> one-hot + scale."""
    steps = []
    if eng:
        steps.append(("eng", Engineer()))
    if te:
        steps.append(("te", GeoTargetEncoder()))
    steps.append(("cols", ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         make_column_selector(dtype_include=object)),
        ("num", StandardScaler(), make_column_selector(dtype_exclude=object)),
    ])))
    return steps


def build(clf, te=False, eng=False, extra=()):
    return Pipeline(prep_steps(te, eng) + list(extra) + [("clf", clf)])


# --------------------------------------------------------------- metrics
def metrics(y, p):
    P, R, F, _ = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
    return dict(acc=accuracy_score(y, p), macro_f1=f1_score(y, p, average="macro"),
                prec=P.tolist(), rec=R.tolist(), f1=F.tolist(),
                cm=confusion_matrix(y, p, labels=[0, 1, 2]).tolist())


def _one_fold(est, X, y, tr, te, fid):
    t0 = time.time()
    try:
        m = clone(est)
        m.fit(X.iloc[tr], y[tr])
        r = metrics(y[te], m.predict(X.iloc[te]))
        r.update(fold=fid, seconds=time.time() - t0)
        return r
    except Exception:
        return dict(error=traceback.format_exc(), fold=fid)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(RES / "progress.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_cv(name, est, X, y, group, label, force=False):
    """Run repeated stratified CV for one pipeline; cache result to results/<name>.json."""
    path = RES / f"{name}.json"
    if path.exists() and not force:
        return json.loads(path.read_text())
    t0 = time.time()
    splits = list(get_cv().split(X, y))
    folds = Parallel(n_jobs=N_JOBS)(delayed(_one_fold)(est, X, y, tr, te, i)
                                    for i, (tr, te) in enumerate(splits))
    errs = [f["error"] for f in folds if "error" in f]
    if errs:
        (RES / "errors.log").open("a", encoding="utf-8").write(f"=== {name} ===\n{errs[0]}\n")
        log(f"ERROR {name}: {errs[0].strip().splitlines()[-1]}")
        raise RuntimeError(name)
    acc = np.array([f["acc"] for f in folds])
    f1 = np.array([f["macro_f1"] for f in folds])
    out = dict(name=name, group=group, label=label, n_runs=len(folds),
               acc_mean=acc.mean(), acc_std=acc.std(ddof=1),
               f1_mean=f1.mean(), f1_std=f1.std(ddof=1),
               acc=acc.tolist(), macro_f1=f1.tolist(),
               prec=np.mean([f["prec"] for f in folds], 0).tolist(),
               rec=np.mean([f["rec"] for f in folds], 0).tolist(),
               cm=np.sum([f["cm"] for f in folds], 0).tolist(),
               seconds=time.time() - t0)
    path.write_text(json.dumps(out))
    log(f"OK {name:<28} acc={out['acc_mean']:.4f}+-{out['acc_std']:.4f} "
        f"F1={out['f1_mean']:.4f}+-{out['f1_std']:.4f} ({out['seconds']:.0f}s)")
    return out


def _too_slow(name):
    """Methods that do not scale to 260k rows (kept on the 20k subsample only)."""
    n = name.lower()
    return "svm" in n or "ensemble" in n or ("knn" in n and "lda_knn" not in n)


def run_all(jobs, X, y):
    """jobs: list of (name, estimator, group, label). A failure is logged, not fatal."""
    res = []
    for name, est, group, label in jobs:
        if FULL and _too_slow(name):
            continue
        try:
            res.append(run_cv(name, est, X, y, group, label))
        except Exception as e:
            log(f"SKIPPED {name}: {e}")
    return res


# ------------------------------------------- helpers for track 3 methods
def _take(X, idx):
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


class ClassWeightTuner(BaseEstimator, ClassifierMixin):
    """Per-class decision weights (cost-sensitive / threshold tuning).

    Inside the training data only: hold out `val_size`, fit the base model on
    the rest, grid-search weights w so that argmax(w * proba) maximises
    macro-F1 on the hold-out, then refit the base model on all training data.
    """

    def __init__(self, base, val_size=0.25, seed=0):
        self.base, self.val_size, self.seed = base, val_size, seed

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.array([0, 1, 2])
        tr, va = next(StratifiedShuffleSplit(1, test_size=self.val_size, random_state=self.seed).split(X, y))
        m = clone(self.base).fit(_take(X, tr), y[tr])
        P = m.predict_proba(_take(X, va))
        best, bw = -1, np.ones(3)
        for w0 in np.geomspace(0.5, 8, 16):
            for w2 in np.geomspace(0.5, 4, 12):
                w = np.array([w0, 1.0, w2])
                s = f1_score(y[va], np.argmax(P * w, 1), average="macro")
                if s > best:
                    best, bw = s, w
        self.w_ = bw
        self.model_ = clone(self.base).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X) * self.w_, 1)


class WithinClassWhitener(BaseEstimator, TransformerMixin):
    """Mahalanobis metric: whiten with the pooled WITHIN-class covariance so that
    Euclidean distance afterwards equals the Mahalanobis distance of LDA."""

    def __init__(self, reg=1e-2):
        self.reg = reg

    def fit(self, X, y):
        y = np.asarray(y)
        self.mu_ = X.mean(0)
        Xc = np.vstack([X[y == c] - X[y == c].mean(0) for c in np.unique(y)])
        S = Xc.T @ Xc / len(Xc)
        S += self.reg * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
        w, V = np.linalg.eigh(S)
        self.W_ = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
        return self

    def transform(self, X):
        return (X - self.mu_) @ self.W_


class BalancedKNN(BaseEstimator, ClassifierMixin):
    """Distance-weighted kNN whose posterior is divided by the class prior,
    i.e. a class-balanced decision rule (helps the rare grade-1 class)."""

    def __init__(self, n_neighbors=25, power=1.0):
        self.n_neighbors, self.power = n_neighbors, power

    def fit(self, X, y):
        from sklearn.neighbors import KNeighborsClassifier
        self.classes_ = np.array([0, 1, 2])
        self.knn_ = KNeighborsClassifier(self.n_neighbors, weights="distance").fit(X, y)
        self.prior_ = np.bincount(np.asarray(y), minlength=3) / len(y)
        return self

    def predict_proba(self, X):
        P = self.knn_.predict_proba(X) / self.prior_ ** self.power
        return P / P.sum(1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), 1)
