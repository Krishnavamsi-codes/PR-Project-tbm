"""Error analysis of the balanced random forest trained on 20k vs on 80 % of all rows, scored on the
same fixed 20 % hold-out: confusion matrices, predicted class shares, and accuracy / macro-F1 per
level-1 region (is the gain uniform across regions?).  Writes results/error_study.json and
figures/error_study.png."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIG, RES, SEED, N_SUB, load_data  # noqa: E402
from zoo import zoo  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

X, y = load_data()
tr, te = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=SEED)
sub, _ = train_test_split(tr, train_size=N_SUB, stratify=y[tr], random_state=SEED)
base = zoo()["RF_bal"][0]
Xte, yte = X.iloc[te], y[te]
reg = Xte.geo_level_1_id.values
res, pred = {}, {}
for tag, idx in [("20k", sub), ("full", tr)]:
    est = clone(base)
    est.set_params(**{k: 16 for k in est.get_params() if k.endswith("n_jobs")})
    est.fit(X.iloc[idx], y[idx])
    p = est.predict(Xte)
    pred[tag] = p
    res[tag] = dict(acc=float(accuracy_score(yte, p)), f1=float(f1_score(yte, p, average="macro")),
                    cm=confusion_matrix(yte, p, normalize="true").tolist(), pred_share=(np.bincount(p, minlength=3) / len(p)).tolist())
    print(tag, res[tag]["acc"], res[tag]["f1"], flush=True)
res["true_share"] = (np.bincount(yte, minlength=3) / len(yte)).tolist()
rows = []
for r in np.unique(reg):
    m = reg == r
    row = dict(region=int(r), n_test=int(m.sum()), grade3=float((yte[m] == 2).mean()))
    for tag in ("20k", "full"):
        row[f"acc_{tag}"] = float(accuracy_score(yte[m], pred[tag][m]))
        row[f"f1_{tag}"] = float(f1_score(yte[m], pred[tag][m], average="macro")) if m.sum() >= 50 else None
    rows.append(row)
res["regions"] = rows
a20 = np.array([r["acc_20k"] for r in rows]); af = np.array([r["acc_full"] for r in rows])
res["regions_improved_acc"] = int((af > a20).sum())
res["n_regions"] = len(rows)
res["region_acc_gain_median"] = float(np.median(af - a20))
big = [r for r in rows if r["n_test"] >= 1000]
res["big_regions"] = len(big)
res["big_gain_median"] = float(np.median([r["acc_full"] - r["acc_20k"] for r in big]))
small = [r for r in rows if r["n_test"] < 1000]
res["small_regions"] = len(small)
res["small_gain_median"] = float(np.median([r["acc_full"] - r["acc_20k"] for r in small])) if small else None
json.dump(res, open(RES / "error_study.json", "w"), indent=1)

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for a, tag, t in zip(ax[:2], ("20k", "full"), ("trained on 20k rows", "trained on 208k rows")):
    cm = np.array(res[tag]["cm"])
    a.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            a.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="white" if cm[i, j] > .5 else "black")
    a.set_xticks(range(3)); a.set_yticks(range(3)); a.set_xticklabels(["1", "2", "3"]); a.set_yticklabels(["1", "2", "3"])
    a.set_xlabel("predicted"); a.set_ylabel("true"); a.set_title(f"Balanced RF {t}\nacc {res[tag]['acc']:.3f}, MF1 {res[tag]['f1']:.3f}", fontsize=9)
ax[2].scatter([r["n_test"] for r in rows], af - a20, c=[r["grade3"] for r in rows], cmap="viridis", s=28)
ax[2].axhline(0, color="k", lw=.8); ax[2].set_xscale("log"); ax[2].set_xlabel("test buildings in level-1 region")
ax[2].set_ylabel("accuracy gain (208k vs 20k)"); ax[2].set_title("Gain per level-1 region (colour = grade-3 share)", fontsize=9)
ax[2].spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(FIG / "error_study.png", dpi=150)
print({k: v for k, v in res.items() if k not in ("regions", "20k", "full")}, res["20k"]["pred_share"], res["full"]["pred_share"])
