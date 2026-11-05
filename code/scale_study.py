"""Scale study: (a) is the 20k subsample representative of the 260k rows?  (b) learning curve.

Part A compares class shares, numeric / categorical / region distributions of the subsample and the
full data.  Part B trains four models on nested-size training sets (2k ... 80 % of the data) and scores
them on one fixed stratified 20 % hold-out of the full data (no CV; one split, so differences between
nearby points are indicative).  All learned steps sit inside the pipelines (training data only).
Writes results/scale_study.json, figures/scale_*.png.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIG, RES, SEED, GEO, N_SUB, load_data  # noqa: E402
from zoo import zoo  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES.mkdir(exist_ok=True, parents=True)
FIG.mkdir(exist_ok=True, parents=True)
X, y = load_data()
idx_all = np.arange(len(X))
_, sub_idx = train_test_split(idx_all, test_size=N_SUB, stratify=y, random_state=SEED)  # same as get_subsample
Xs, ys = X.iloc[sub_idx], y[sub_idx]
out = {}

# ------------------------------------------------------------ Part A: distributions
cls = {"full": np.bincount(y) / len(y), "sub": np.bincount(ys) / len(ys)}
num = ["count_floors_pre_eq", "age", "area_percentage", "height_percentage", "count_families"]
rows = []
for c in num:
    ks = ks_2samp(X[c], Xs[c])
    rows.append(dict(feature=c, mean_full=X[c].mean(), mean_sub=Xs[c].mean(), med_full=X[c].median(), med_sub=Xs[c].median(),
                     std_full=X[c].std(), std_sub=Xs[c].std(), ks=ks.statistic, ks_p=ks.pvalue))
cat = [c for c in X.columns if X[c].dtype == object]
catdiff = {}
for c in cat:
    a, b = X[c].value_counts(normalize=True), Xs[c].value_counts(normalize=True)
    catdiff[c] = float((a - b.reindex(a.index).fillna(0)).abs().max())
flags = [c for c in X.columns if c.startswith("has_")]
flagdiff = float(max(abs(X[c].mean() - Xs[c].mean()) for c in flags))
geo = {}
for g in GEO:
    cf = X[g].value_counts()
    cs = Xs[g].value_counts()
    geo[g] = dict(n_full=int(len(cf)), n_sub=int(len(cs)), median_full=float(cf.median()), median_sub=float(cs.median()),
                  le5_full=float((cf <= 5).mean()), le5_sub=float((cs <= 5).mean()),
                  rows_in_regions_le5_full=float(cf[cf <= 5].sum() / len(X)), rows_in_regions_le5_sub=float(cs[cs <= 5].sum() / len(Xs)),
                  singletons_sub=float((cs == 1).mean()))
# share of held-out rows whose level-3 region never appears in training: subsample CV vs full CV (80 % train)
rng = np.random.RandomState(SEED)
def unseen(Xd):
    tr, te = train_test_split(np.arange(len(Xd)), test_size=0.2, random_state=SEED)
    seen = set(Xd.geo_level_3_id.values[tr])
    return float(np.mean([v not in seen for v in Xd.geo_level_3_id.values[te]]))
geo["unseen_l3_test_share_full"] = unseen(X)
geo["unseen_l3_test_share_sub"] = unseen(Xs)
# grade-3 share per level-1 region: correlation between subsample and full
g1f = pd.Series(y == 2).groupby(X.geo_level_1_id.values).mean()
g1s = pd.Series(ys == 2).groupby(Xs.geo_level_1_id.values).mean().reindex(g1f.index)
geo["l1_grade3_corr"] = float(np.corrcoef(g1f.fillna(0), g1s.fillna(g1f.mean()))[0, 1])
geo["l1_grade3_max_abs_diff"] = float((g1f - g1s).abs().max())
out["dist"] = dict(class_full=cls["full"].tolist(), class_sub=cls["sub"].tolist(), numeric=rows, cat_maxdiff=catdiff,
                   flag_maxdiff=flagdiff, geo=geo)

fig, ax = plt.subplots(1, 4, figsize=(15, 3.3))
w = 0.38
ax[0].bar(np.arange(3) - w / 2, cls["full"], w, label="all 260,601", color="#2A7AB0")
ax[0].bar(np.arange(3) + w / 2, cls["sub"], w, label="20k subsample", color="#E69F00")
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(["grade 1", "grade 2", "grade 3"]); ax[0].set_title("Class shares"); ax[0].legend(fontsize=8)
bins = np.linspace(0, 100, 26)
ax[1].hist(np.clip(X.age, 0, 100), bins, density=True, alpha=.6, color="#2A7AB0", label="all")
ax[1].hist(np.clip(Xs.age, 0, 100), bins, density=True, alpha=.6, color="#E69F00", label="20k"); ax[1].set_title("age (clipped at 100)")
ax[2].hist(np.clip(X.area_percentage, 0, 20), 30, density=True, alpha=.6, color="#2A7AB0")
ax[2].hist(np.clip(Xs.area_percentage, 0, 20), 30, density=True, alpha=.6, color="#E69F00"); ax[2].set_title("area_percentage (clipped at 20)")
cf, cs = X.geo_level_3_id.value_counts(), Xs.geo_level_3_id.value_counts()
mx = 40
ax[3].hist(np.clip(cf.values, 0, mx), np.arange(0.5, mx + 1.5), density=True, alpha=.6, color="#2A7AB0", label="all")
ax[3].hist(np.clip(cs.values, 0, 10), np.arange(0.5, 11.5), density=True, alpha=.6, color="#E69F00", label="20k")
ax[3].set_title("buildings per level-3 region"); ax[3].set_xlim(0, 20)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(FIG / "scale_distributions.png", dpi=150); plt.close()

# ------------------------------------------------------------ Part B: learning curve
Z = zoo()
models = {"Decision tree (TE+eng)": "Tree", "Random forest, balanced": "RF_bal", "Linear SVM, balanced": "LinSVC_bal",
          "Logistic regression": "LogReg"}
tr_idx, te_idx = train_test_split(idx_all, test_size=0.2, stratify=y, random_state=SEED)
Xte, yte = X.iloc[te_idx], y[te_idx]
sizes = [2000, 5000, 10000, 20000, 50000, 100000, len(tr_idx)]
curve = []
for name, key in models.items():
    est0 = Z[key][0]
    for n in sizes:
        reps = 3 if n <= 50000 else 1
        accs, f1s, r1s, secs = [], [], [], []
        for r in range(reps):
            if n == len(tr_idx):
                sel = tr_idx
            else:
                sel, _ = train_test_split(tr_idx, train_size=n, stratify=y[tr_idx], random_state=SEED + r)
            est = clone(est0)
            nj = {k: 16 for k in est.get_params() if k.endswith("n_jobs")}
            est.set_params(**nj)
            t0 = time.time()
            est.fit(X.iloc[sel], y[sel])
            p = est.predict(Xte)
            secs.append(time.time() - t0)
            accs.append(accuracy_score(yte, p)); f1s.append(f1_score(yte, p, average="macro"))
            r1s.append(recall_score(yte, p, average=None)[0])
        curve.append(dict(model=name, n=int(n), acc=float(np.mean(accs)), f1=float(np.mean(f1s)), f1_std=float(np.std(f1s)),
                          rec1=float(np.mean(r1s)), sec=float(np.mean(secs)), reps=reps))
        print(name, n, round(curve[-1]["f1"], 4), round(curve[-1]["sec"], 1), flush=True)
out["curve"] = curve
json.dump(out, open(RES / "scale_study.json", "w"), indent=1)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
col = ["#7A7A7A", "#2A7AB0", "#D55E00", "#009E73"]
for (name, _), c in zip(models.items(), col):
    d = [r for r in curve if r["model"] == name]
    ax[0].errorbar([r["n"] for r in d], [r["f1"] for r in d], [r["f1_std"] for r in d], marker="o", color=c, label=name, capsize=2)
    ax[1].plot([r["n"] for r in d], [r["rec1"] for r in d], marker="o", color=c, label=name)
for a, t in zip(ax, ["Macro-F1 on fixed 20 % hold-out", "Grade-1 recall"]):
    a.set_xscale("log"); a.set_xlabel("training rows"); a.set_title(t); a.axvline(20000, ls=":", color="k", lw=.8)
    a.spines[["top", "right"]].set_visible(False)
ax[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "scale_learning_curve.png", dpi=150); plt.close()
print("done")
