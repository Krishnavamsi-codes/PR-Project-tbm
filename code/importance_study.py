"""Random-forest (balanced) impurity importances, grouped by feature family, fitted on the 20k
subsample vs on 80 % of all rows (same recipe as the comparison models).  Writes results/importance.json."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RES, SEED, N_SUB, load_data  # noqa: E402
from zoo import zoo  # noqa: E402

X, y = load_data()
_, sub = train_test_split(np.arange(len(X)), test_size=N_SUB, stratify=y, random_state=SEED)
tr, _ = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=SEED)
base = zoo()["RF_bal"][0]


def groups(est):
    names = est.named_steps["cols"].get_feature_names_out()
    imp = est.named_steps["clf"].feature_importances_
    g = {}
    for n, v in zip(names, imp):
        n = n.split("__", 1)[1]
        if n.startswith("te_") or "geo" in n or n.startswith("p"):
            key = "target-encoded geography"
        elif n.startswith("has_secondary"):
            key = "secondary-use flags"
        elif n.startswith("has_superstructure"):
            key = "superstructure flags"
        elif n in ("count_floors_pre_eq", "age", "area_percentage", "height_percentage", "count_families") or n.startswith("log_") or n in ("age_unknown",):
            key = "numeric (floors, age, area, height, families)"
        elif "_x_" in n or "per" in n or n.startswith("n_") or n in ("is_rc", "is_mud_stone", "area_x_height"):
            key = "engineered"
        else:
            key = "categorical (one-hot)"
        g[key] = g.get(key, 0) + float(v)
    top = sorted(zip(names, imp), key=lambda t: -t[1])[:10]
    return g, [(n.split("__", 1)[1], float(v)) for n, v in top]


out = {}
for tag, idx in [("20k", sub), ("full", tr)]:
    est = clone(base)
    est.set_params(**{k: 16 for k in est.get_params() if k.endswith("n_jobs")})
    est.fit(X.iloc[idx], y[idx])
    g, top = groups(est.named_steps["clf"] and est)
    out[tag] = dict(groups=g, top=top, n=int(len(idx)))
    print(tag, {k: round(v, 3) for k, v in g.items()}, top[:6], flush=True)
json.dump(out, open(RES / "importance.json", "w"), indent=1)
