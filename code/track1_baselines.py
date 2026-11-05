"""Track 1 (Person 1): baselines on RAW features (geo ids left as plain numbers)."""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common import build, run_all

CLFS = {
    "kNN": KNeighborsClassifier(15),
    "NaiveBayes": GaussianNB(),
    "LDA": LDA(),
    "QDA": QDA(reg_param=0.5),
    "SVM-RBF": SVC(C=1.0, gamma="scale"),
    "DecisionTree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=20, random_state=0),
}


def jobs():
    return [(f"t1_{k}", build(c), "1 Baseline (raw)", k) for k, c in CLFS.items()]


def run(X, y):
    return run_all(jobs(), X, y)
