"""Track 3 (Person 3): modified classical methods + MLP (comparison only).

All run on the target-encoded + engineered feature space (te_eng):
  CostSens-<model>   posterior * per-class weights tuned for macro-F1 on an
                     inner hold-out of the training fold (cost-sensitive Bayes
                     rule / per-class threshold tuning)
  Mahalanobis-kNN    kNN after whitening with pooled within-class covariance
  BalancedKNN        distance-weighted kNN with prior-corrected posterior
  SVM-balanced       SVM with class_weight='balanced'
  Ensemble           soft voting (LDA, NB, kNN, SVM, Tree), plain and tuned
  MLP                reference only (deep learning is not our contribution)
"""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common import BalancedKNN, ClassWeightTuner, WithinClassWhitener, build, run_all

M = "3 Modified methods"


def ensemble():
    return VotingClassifier([
        ("lda", LDA()),
        ("nb", GaussianNB()),
        ("knn", KNeighborsClassifier(25, weights="distance")),
        ("svm", SVC(probability=True, random_state=0)),
        ("tree", DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, random_state=0)),
    ], voting="soft")


def jobs():
    te = dict(te=True, eng=True)
    return [
        ("t3_CostSens_NB", build(ClassWeightTuner(GaussianNB()), **te), M, "CostSens-NaiveBayes"),
        ("t3_CostSens_LDA", build(ClassWeightTuner(LDA()), **te), M, "CostSens-LDA"),
        ("t3_CostSens_QDA", build(ClassWeightTuner(QDA(reg_param=0.5)), **te), M, "CostSens-QDA"),
        ("t3_CostSens_kNN", build(ClassWeightTuner(KNeighborsClassifier(25, weights="distance")), **te), M, "CostSens-kNN"),
        ("t3_Mahalanobis_kNN", build(KNeighborsClassifier(15), extra=[("maha", WithinClassWhitener())], **te), M, "Mahalanobis-kNN"),
        ("t3_Mahalanobis_BalKNN", build(BalancedKNN(25), extra=[("maha", WithinClassWhitener())], **te), M, "Mahalanobis+Balanced kNN"),
        ("t3_BalancedKNN", build(BalancedKNN(25), **te), M, "Balanced kNN"),
        ("t3_SVM_balanced", build(SVC(class_weight="balanced"), **te), M, "SVM-balanced"),
        ("t3_Ensemble", build(ensemble(), **te), M, "Soft-voting ensemble"),
        ("t3_Ensemble_tuned", build(ClassWeightTuner(ensemble()), **te), M, "Ensemble + class-weight tuning"),
        ("t3_MLP", build(MLPClassifier((128, 64), early_stopping=True, max_iter=300, random_state=0), **te),
         "4 Deep learning (reference)", "MLP (te_eng)"),
        ("t3_MLP_raw", build(MLPClassifier((128, 64), early_stopping=True, max_iter=300, random_state=0)),
         "4 Deep learning (reference)", "MLP (raw)"),
    ]


def run(X, y):
    return run_all(jobs(), X, y)
