"""Track 4: architecture zoo. Every classifier family we can defend for this
problem, all with FIXED sensible hyper-parameters, all on the same leakage-free
target-encoded + engineered feature space (te_eng) and the same CV splits.

Families (group names are used for tables and colours):
  A Generative            LDA, shrinkage-LDA, QDA/RDA, Gaussian NB, GMM-per-class, Parzen/KDE
  B Linear discriminative logistic reg, ridge, perceptron, SGD (hinge/modified huber), linear SVM
  C Kernel methods        SVM poly-2/3, sigmoid, RBF; Nystroem-RBF/poly + linear; poly-feature logreg
  D Instance based        kNN (k = 5/15/45, uniform/distance), nearest centroid
  E Trees and ensembles   tree, bagging, RF, ExtraTrees, AdaBoost, GBM, HistGB, LightGBM, stacking
  F Neural networks       sklearn MLPs: width / depth / activation / regularisation sweep
"""
import numpy as np
from lightgbm import LGBMClassifier
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import (LogisticRegression, Perceptron, RidgeClassifier,
                                  SGDClassifier)
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KernelDensity, KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from common import build

GA, GB, GC, GD, GE, GF = ("A Generative", "B Linear discriminative", "C Kernel methods",
                          "D Instance based", "E Trees and ensembles", "F Neural networks")


# ------------------------------------------------ custom generative models
class _LogPostMixin(ClassifierMixin):
    def predict_proba(self, X):
        L = self._joint(X)
        return np.exp(L - logsumexp(L, axis=1, keepdims=True))

    def predict(self, X):
        return np.argmax(self._joint(X), axis=1)


class GMMClassifier(_LogPostMixin, BaseEstimator):
    """Generative classifier: one Gaussian mixture per class, Bayes rule with class priors."""

    def __init__(self, n_components=2, covariance_type="full", reg_covar=1e-2):
        self.n_components, self.covariance_type, self.reg_covar = n_components, covariance_type, reg_covar

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.array([0, 1, 2])
        self.log_prior_ = np.log(np.bincount(y, minlength=3) / len(y))
        self.models_ = [GaussianMixture(self.n_components, covariance_type=self.covariance_type,
                                        reg_covar=self.reg_covar, random_state=0).fit(X[y == c])
                        for c in self.classes_]
        return self

    def _joint(self, X):
        return np.column_stack([m.score_samples(X) for m in self.models_]) + self.log_prior_


class ParzenClassifier(_LogPostMixin, BaseEstimator):
    """Non-parametric generative classifier: Gaussian-kernel density estimate per class."""

    def __init__(self, bandwidth=0.6):
        self.bandwidth = bandwidth

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.array([0, 1, 2])
        self.log_prior_ = np.log(np.bincount(y, minlength=3) / len(y))
        self.models_ = [KernelDensity(bandwidth=self.bandwidth).fit(X[y == c]) for c in self.classes_]
        return self

    def _joint(self, X):
        return np.column_stack([m.score_samples(X) for m in self.models_]) + self.log_prior_


# ----------------------------------------------------------- zoo definition
def _rf(**kw):
    return RandomForestClassifier(300, min_samples_leaf=2, random_state=0, n_jobs=1, **kw)


def _mlp(hidden, **kw):
    return MLPClassifier(hidden, early_stopping=True, max_iter=300, random_state=0, **kw)


def zoo():
    """Returns {key: (estimator_pipeline, group, label)}; also reused by the tuning script."""
    pca12 = [("pca", PCA(12, whiten=True, random_state=0))]
    pca8 = [("pca", PCA(8, whiten=True, random_state=0))]
    nys = lambda kern, **kw: [("nys", Nystroem(kernel=kern, n_components=500, random_state=0, **kw))]
    polyfe = [("sel", SelectKBest(mutual_info_classif, k=20)),
              ("poly", PolynomialFeatures(2, include_bias=False))]
    B = lambda clf, extra=(): build(clf, te=True, eng=True, extra=extra)

    z = {}

    def add(key, est, group, label):
        z[key] = (est, group, label)

    # A: generative -----------------------------------------------------
    add("LDA", B(LDA()), GA, "LDA (shared covariance)")
    add("LDA_shrink", B(LDA(solver="lsqr", shrinkage="auto")), GA, "LDA, Ledoit-Wolf shrinkage")
    add("QDA_r01", B(QDA(reg_param=0.1)), GA, "QDA reg=0.1")
    add("QDA_r05", B(QDA(reg_param=0.5)), GA, "RDA / QDA reg=0.5")
    add("NB", B(GaussianNB()), GA, "Gaussian Naive Bayes")
    add("GMM_1", B(GMMClassifier(1), pca12), GA, "GMM-Bayes, 1 comp/class (PCA-12)")
    add("GMM_3", B(GMMClassifier(3), pca12), GA, "GMM-Bayes, 3 comp/class (PCA-12)")
    add("GMM_3_diag", B(GMMClassifier(3, "diag"), pca12), GA, "GMM-Bayes, 3 diag comp/class (PCA-12)")
    add("Parzen", B(ParzenClassifier(0.6), pca8), GA, "Parzen window / KDE-Bayes (PCA-8)")

    # B: linear discriminative -----------------------------------------
    add("LogReg", B(LogisticRegression(C=0.3, max_iter=1000)), GB, "Logistic regression (softmax)")
    add("LogReg_bal", B(LogisticRegression(C=0.3, max_iter=1000, class_weight="balanced")), GB, "Logistic regression, balanced")
    add("Ridge", B(RidgeClassifier(alpha=10.0)), GB, "Ridge classifier (least squares)")
    add("Perceptron", B(Perceptron(max_iter=200, random_state=0)), GB, "Perceptron")
    add("SGD_hinge", B(SGDClassifier(loss="hinge", alpha=1e-3, random_state=0)), GB, "SGD linear SVM (hinge)")
    add("SGD_mhuber", B(SGDClassifier(loss="modified_huber", alpha=1e-3, random_state=0)), GB, "SGD modified Huber")
    add("LinSVC", B(LinearSVC(C=0.05, max_iter=3000)), GB, "Linear SVM (OvR)")
    add("LinSVC_bal", B(LinearSVC(C=0.05, max_iter=3000, class_weight="balanced")), GB, "Linear SVM, balanced")

    # C: kernel methods -------------------------------------------------
    add("SVM_poly2", B(SVC(kernel="poly", degree=2, coef0=1.0, C=1.0)), GC, "SVM polynomial deg 2")
    add("SVM_poly3", B(SVC(kernel="poly", degree=3, coef0=1.0, C=1.0)), GC, "SVM polynomial deg 3")
    add("SVM_sigmoid", B(SVC(kernel="sigmoid", C=1.0)), GC, "SVM sigmoid (tanh) kernel")
    add("SVM_rbf", B(SVC(kernel="rbf", C=1.0)), GC, "SVM RBF")
    add("SVM_rbf_bal", B(SVC(kernel="rbf", C=1.0, class_weight="balanced")), GC, "SVM RBF, balanced")
    add("Nys_RBF_LogReg", B(LogisticRegression(C=1.0, max_iter=1000), nys("rbf", gamma=0.02)), GC, "Nystroem RBF + logistic reg")
    add("Nys_poly_LogReg", B(LogisticRegression(C=0.3, max_iter=1000), nys("poly", degree=2, coef0=1.0, gamma=0.02)), GC, "Nystroem poly-2 + logistic reg")
    add("Nys_RBF_LDA", B(LDA(), nys("rbf", gamma=0.02)), GC, "Nystroem RBF + LDA (approx. kernel Fisher)")
    add("PolyFeat_LogReg", B(LogisticRegression(C=0.1, max_iter=2000), polyfe), GC, "MI-20 + degree-2 poly features + logreg")

    # D: instance based -------------------------------------------------
    add("kNN5", B(KNeighborsClassifier(5)), GD, "kNN k=5")
    add("kNN15", B(KNeighborsClassifier(15)), GD, "kNN k=15")
    add("kNN45_dist", B(KNeighborsClassifier(45, weights="distance")), GD, "kNN k=45 distance-weighted")
    add("NearestCentroid", B(NearestCentroid()), GD, "Nearest centroid (prototype)")

    # E: trees and ensembles -------------------------------------------
    add("Tree", B(DecisionTreeClassifier(max_depth=10, min_samples_leaf=20, random_state=0)), GE, "Decision tree depth 10")
    add("Bagging", B(BaggingClassifier(DecisionTreeClassifier(min_samples_leaf=5), n_estimators=50, random_state=0)), GE, "Bagged trees")
    add("RF", B(_rf()), GE, "Random forest")
    add("RF_bal", B(_rf(class_weight="balanced_subsample")), GE, "Random forest, balanced")
    add("ExtraTrees", B(ExtraTreesClassifier(300, min_samples_leaf=2, random_state=0, n_jobs=1)), GE, "Extra-trees")
    add("AdaBoost", B(AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=100, random_state=0)), GE, "AdaBoost (depth-3 stumps)")
    add("GBM", B(GradientBoostingClassifier(n_estimators=150, subsample=0.8, random_state=0)), GE, "Gradient boosting (sklearn)")
    add("HistGB", B(HistGradientBoostingClassifier(max_iter=200, random_state=0)), GE, "Hist gradient boosting")
    add("LightGBM", B(LGBMClassifier(n_estimators=300, learning_rate=0.05, n_jobs=1, verbose=-1, random_state=0)), GE, "LightGBM")
    add("LightGBM_bal", B(LGBMClassifier(n_estimators=300, learning_rate=0.05, n_jobs=1, verbose=-1, random_state=0,
                                         class_weight="balanced")), GE, "LightGBM, balanced")
    add("Stacking", B(StackingClassifier(
        [("lda", LDA()), ("knn", KNeighborsClassifier(25, weights="distance")),
         ("rf", RandomForestClassifier(150, min_samples_leaf=3, random_state=0, n_jobs=1)),
         ("hgb", HistGradientBoostingClassifier(max_iter=100, random_state=0))],
        final_estimator=LogisticRegression(max_iter=500), cv=3)), GE, "Stacking (LDA,kNN,RF,HGB -> logreg)")

    # F: neural networks ------------------------------------------------
    add("MLP_64", B(_mlp((64,))), GF, "MLP 1x64 relu")
    add("MLP_128_64", B(_mlp((128, 64))), GF, "MLP 128-64 relu")
    add("MLP_256_128_64", B(_mlp((256, 128, 64))), GF, "MLP 256-128-64 relu (deep)")
    add("MLP_tanh", B(_mlp((128, 64), activation="tanh")), GF, "MLP 128-64 tanh")
    add("MLP_logistic", B(_mlp((128, 64), activation="logistic")), GF, "MLP 128-64 sigmoid")
    add("MLP_l2", B(_mlp((128, 64), alpha=1e-1)), GF, "MLP 128-64 relu, L2 alpha=0.1")
    return z


def jobs():
    return [(f"t4_{k}", est, g, lab) for k, (est, g, lab) in zoo().items()]
