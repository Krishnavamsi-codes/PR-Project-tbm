"""Full-dataset run (all 260,601 rows): best fast models only, plain stratified 5-fold CV.
Exact RBF-SVM / kNN are left out (do not scale); Nystroem-RBF stands in for the kernel method.
    python code/full_data.py
Results: results/full/*.json, results/full/progress.log, results/full/full_vs_subsample.md
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import common

MAIN = common.RES
FULL = MAIN / "full"
FULL.mkdir(exist_ok=True)
common.RES = FULL
common.N_JOBS = 5
common.get_cv = lambda: StratifiedKFold(5, shuffle=True, random_state=common.SEED)

import zoo

KEYS = ["LDA_shrink", "LogReg", "LogReg_bal", "LinSVC_bal", "Nys_RBF_LogReg",
        "LightGBM", "LightGBM_bal", "HistGB", "RF_bal", "MLP_128_64"]


def main():
    X, y = common.load_data()
    common.log(f"FULL START rows={len(X)} 5-fold, jobs={common.N_JOBS}")
    z = zoo.zoo()
    jobs = [(f"full_{k}", z[k][0], "Full data", z[k][2]) for k in KEYS]
    common.run_all(jobs, X, y)
    rows = []
    for k in KEYS:
        f, s = FULL / f"full_{k}.json", MAIN / "ext" / f"t4_{k}.json"
        if f.exists() and s.exists():
            a, b = json.loads(f.read_text()), json.loads(s.read_text())
            rows.append(dict(Model=a["label"], Sub20k_Acc=b["acc_mean"], Full_Acc=a["acc_mean"],
                             Sub20k_F1=b["f1_mean"], Full_F1=a["f1_mean"],
                             F1_gain=a["f1_mean"] - b["f1_mean"], Full_seconds=round(a["seconds"])))
    if rows:
        df = pd.DataFrame(rows).sort_values("Full_F1", ascending=False)
        df.to_csv(FULL / "full_vs_subsample.csv", index=False)
        (FULL / "full_vs_subsample.md").write_text(df.to_markdown(index=False, floatfmt=".4f"), encoding="utf-8")
    common.log("FULL DONE")


if __name__ == "__main__":
    main()
