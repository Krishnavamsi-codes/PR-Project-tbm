"""Track 5: hyper-parameter tuning with NESTED cross-validation.

Every outer CV fold (same 5x3 splits as everything else) runs an inner 3-fold
GridSearchCV scored on macro-F1 over the TRAINING part only; the winning
configuration is refit on the whole training part and evaluated on the outer
test fold. So the reported numbers are an honest estimate of "tune, then deploy".
Every inner search is appended to tuning_log.jsonl (best params + the whole grid).
"""
import json
import time

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import zoo as Z
from common import RES, run_all

TUNED = "T Tuned (nested CV)"


class TunedModel(BaseEstimator, ClassifierMixin):
    def __init__(self, pipe, grid, name, log_path, inner=3):
        self.pipe, self.grid, self.name, self.log_path, self.inner = pipe, grid, name, log_path, inner

    def fit(self, X, y):
        t0 = time.time()
        gs = GridSearchCV(clone(self.pipe), self.grid, scoring="f1_macro", n_jobs=1, refit=True,
                          cv=StratifiedKFold(self.inner, shuffle=True, random_state=0))
        gs.fit(X, y)
        self.gs_ = gs
        self.classes_ = np.array([0, 1, 2])
        rec = dict(model=self.name, best_params={k: _j(v) for k, v in gs.best_params_.items()},
                   best_inner_f1=float(gs.best_score_), seconds=round(time.time() - t0, 1),
                   grid=[dict(params={k: _j(v) for k, v in p.items()}, f1=float(s))
                         for p, s in zip(gs.cv_results_["params"], gs.cv_results_["mean_test_score"])])
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return self

    def predict(self, X):
        return self.gs_.predict(X)


def _j(v):
    return v if isinstance(v, (int, float, str, type(None))) else str(v)


# key in zoo -> parameter grid (pipeline step names: clf__, pca__, nys__)
GRIDS = {
    "LDA_shrink": {"clf__shrinkage": [0.0, 0.05, 0.2, 0.5, "auto"]},
    "QDA_r05": {"clf__reg_param": [0.05, 0.2, 0.4, 0.6, 0.8]},
    "NB": {"clf__var_smoothing": [1e-9, 1e-6, 1e-4, 1e-2, 1e-1]},
    "GMM_3": {"clf__n_components": [1, 2, 3, 4], "clf__reg_covar": [1e-3, 1e-1]},
    "Parzen": {"clf__bandwidth": [0.2, 0.4, 0.6, 1.0]},
    "LogReg": {"clf__C": [0.003, 0.03, 0.3, 3, 30], "clf__class_weight": [None, "balanced"]},
    "LinSVC": {"clf__C": [0.003, 0.01, 0.05, 0.3, 1.0], "clf__class_weight": [None, "balanced"]},
    "SVM_rbf": {"clf__C": [1, 3, 10], "clf__gamma": [0.003, 0.01, 0.03]},
    "SVM_poly2": {"clf__C": [0.3, 1, 3], "clf__degree": [2, 3]},
    "Nys_RBF_LogReg": {"nys__gamma": [0.005, 0.02, 0.08], "clf__C": [0.1, 1.0]},
    "kNN15": {"clf__n_neighbors": [5, 11, 21, 41, 81], "clf__weights": ["uniform", "distance"]},
    "Tree": {"clf__max_depth": [4, 6, 8, 12, 16], "clf__min_samples_leaf": [5, 20, 50]},
    "RF": {"clf__max_depth": [8, 16, None], "clf__min_samples_leaf": [1, 3, 8],
           "clf__max_features": ["sqrt", 0.3]},
    "HistGB": {"clf__learning_rate": [0.03, 0.1], "clf__max_leaf_nodes": [15, 31, 63],
               "clf__l2_regularization": [0.0, 1.0]},
    "LightGBM": {"clf__num_leaves": [15, 31, 63], "clf__learning_rate": [0.03, 0.1],
                 "clf__min_child_samples": [20, 50]},
    "MLP_128_64": {"clf__hidden_layer_sizes": [(64,), (128, 64)], "clf__alpha": [1e-4, 1e-2]},
}


def jobs():
    z = Z.zoo()
    log_path = str(RES / "tuning_log.jsonl")
    out = []
    for k, grid in GRIDS.items():
        est, _, lab = z[k]
        out.append((f"t5_{k}", TunedModel(est, grid, k, log_path), TUNED, f"{lab} [tuned]"))
    return out


def run(X, y):
    return run_all(jobs(), X, y)
