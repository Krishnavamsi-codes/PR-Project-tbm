"""Track 2 (Person 2): feature extraction / selection, fold-safe.

Feature sets (all fitted on the training fold only):
  eng        engineered features only (geo ids still raw)
  te         hierarchical out-of-fold target encoding of geo ids
  te_eng     target encoding + engineered features
  te_eng_pca te_eng -> PCA keeping 95 % variance
  te_eng_lda te_eng -> LDA projection (2 discriminants)
  te_eng_mi  te_eng -> top-20 by mutual information
  te_eng_fis te_eng -> top-20 by Fisher score
The same classifiers are run on each set, which also gives the ablation.
"""
from functools import partial

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common import build, run_all


def fisher_score(X, y):
    """Per-feature Fisher score: between-class / within-class variance."""
    mu = X.mean(0)
    num = np.zeros(X.shape[1])
    den = np.zeros(X.shape[1])
    for c in np.unique(y):
        Xc = X[y == c]
        num += len(Xc) * (Xc.mean(0) - mu) ** 2
        den += len(Xc) * Xc.var(0)
    return num / (den + 1e-12)


SETS = {
    "eng": dict(te=False, eng=True, extra=[]),
    "te": dict(te=True, eng=False, extra=[]),
    "te_eng": dict(te=True, eng=True, extra=[]),
    "te_eng_pca": dict(te=True, eng=True, extra=[("pca", PCA(n_components=0.95, random_state=0))]),
    "te_eng_lda": dict(te=True, eng=True, extra=[("lda", LDA())]),
    "te_eng_mi": dict(te=True, eng=True, extra=[("sel", SelectKBest(partial(mutual_info_classif, random_state=0), k=20))]),
    "te_eng_fis": dict(te=True, eng=True, extra=[("sel", SelectKBest(fisher_score, k=20))]),
}

CLFS = {
    "kNN": lambda: KNeighborsClassifier(15),
    "NaiveBayes": lambda: GaussianNB(),
    "SVM-RBF": lambda: SVC(C=1.0, gamma="scale"),
    "DecisionTree": lambda: DecisionTreeClassifier(max_depth=10, min_samples_leaf=20, random_state=0),
    "LDA": lambda: LDA(),
}


def jobs():
    out = []
    for sname, s in SETS.items():
        for cname, mk in CLFS.items():
            if cname == "LDA" and sname == "te_eng_lda":
                continue                      # LDA on an LDA projection is redundant
            if sname == "eng" and cname in ("NaiveBayes", "DecisionTree", "LDA"):
                continue                      # eng is only an ablation step (kNN, SVM)
            out.append((f"t2_{sname}_{cname}", build(mk(), s["te"], s["eng"], s["extra"]),
                        "2 Feature extraction", f"{cname} + {sname}"))
    return out


def run(X, y):
    return run_all(jobs(), X, y)
