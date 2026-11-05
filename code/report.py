"""Collect results/*.json -> tables (md/csv), figures, significance tests."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common import FIG, N_SPLITS, RES, log

NAMES = ["Grade 1 (low)", "Grade 2 (medium)", "Grade 3 (high)"]
COL = {"1 Baseline (raw)": "#8c8c8c", "2 Feature extraction": "#2a7ab0",
       "3 Modified methods": "#d9822b", "4 Deep learning (reference)": "#7b5ea7"}


def load():
    rs = [json.loads(p.read_text()) for p in sorted(RES.glob("*.json"))]
    return sorted(rs, key=lambda r: (r["group"], -r["f1_mean"]))


def table(rs):
    df = pd.DataFrame([dict(Group=r["group"], Method=r["label"],
                            Accuracy=f"{r['acc_mean']:.4f} ± {r['acc_std']:.4f}",
                            MacroF1=f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}",
                            acc=r["acc_mean"], f1=r["f1_mean"], name=r["name"],
                            Seconds=round(r["seconds"])) for r in rs])
    return df


def best(rs, group):
    c = [r for r in rs if r["group"] == group]
    return max(c, key=lambda r: r["f1_mean"]) if c else None


def corrected_t(a, b, test_frac):
    """Paired t-test + Nadeau-Bengio corrected resampled t-test (repeated CV)."""
    d = np.array(a) - np.array(b)
    k = len(d)
    t0, p0 = stats.ttest_rel(a, b)
    var = d.var(ddof=1)
    if var == 0:
        return d.mean(), t0, p0, np.nan, np.nan
    t1 = d.mean() / np.sqrt((1 / k + test_frac / (1 - test_frac)) * var)
    p1 = 2 * stats.t.sf(abs(t1), k - 1)
    return d.mean(), t0, p0, t1, p1


def significance(rs):
    bb, bf, bm, bd = (best(rs, g) for g in COL)
    pairs = [("Feature vs Baseline", bf, bb), ("Modified vs Baseline", bm, bb),
             ("Modified vs Feature", bm, bf), ("Modified vs MLP", bm, bd), ("Feature vs MLP", bf, bd)]
    rows = []
    for nm, a, b in pairs:
        if a is None or b is None:
            continue
        for metric, key in (("macro-F1", "macro_f1"), ("accuracy", "acc")):
            d, t0, p0, t1, p1 = corrected_t(a[key], b[key], 1 / N_SPLITS)
            rows.append(dict(Comparison=nm, A=a["label"], B=b["label"], Metric=metric,
                             MeanDiff=round(d, 4), p_paired=f"{p0:.2g}", p_corrected=f"{p1:.2g}"))
    return pd.DataFrame(rows)


def fig_bars(rs):
    top = sorted(rs, key=lambda r: r["f1_mean"])
    fig, ax = plt.subplots(figsize=(9, max(5, 0.28 * len(top))))
    ax.barh([f"{r['label']}" for r in top], [r["f1_mean"] for r in top],
            xerr=[r["f1_std"] for r in top], color=[COL[r["group"]] for r in top], capsize=2)
    ax.set_xlabel("Macro-F1 (mean ± std over runs)")
    ax.set_xlim(max(0, min(r["f1_mean"] for r in top) - 0.05), None)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in COL.values()], labels=list(COL), loc="lower right", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "macro_f1_all_methods.png", dpi=150)
    plt.close(fig)


def fig_cm(rs):
    sel = [b for b in (best(rs, g) for g in COL) if b]
    fig, axs = plt.subplots(1, len(sel), figsize=(4.2 * len(sel), 4))
    for ax, r in zip(np.atleast_1d(axs), sel):
        cm = np.array(r["cm"], float)
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cmn[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center",
                        color="white" if cmn[i, j] > .5 else "black", fontsize=8)
        ax.set_xticks(range(3), ["1", "2", "3"])
        ax.set_yticks(range(3), ["1", "2", "3"])
        ax.set_xlabel("Predicted grade")
        ax.set_ylabel("True grade")
        ax.set_title(f"{r['label']}\nF1={r['f1_mean']:.3f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "confusion_matrices_best.png", dpi=150)
    plt.close(fig)


def fig_perclass(rs):
    sel = [b for b in (best(rs, g) for g in COL) if b]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    w = 0.8 / len(sel)
    for k, (ax, key, t) in enumerate(zip(axs, ("prec", "rec"), ("Per-class precision", "Per-class recall"))):
        for i, r in enumerate(sel):
            ax.bar(np.arange(3) + i * w, r[key], w, label=r["label"], color=list(COL.values())[i])
        ax.set_xticks(np.arange(3) + 0.4 - w / 2, NAMES)
        ax.set_title(t)
        ax.set_ylim(0, 1)
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "per_class_precision_recall.png", dpi=150)
    plt.close(fig)


def fig_ablation(rs):
    by = {r["name"]: r for r in rs}
    order = [("raw", "t1_{c}"), ("+eng", "t2_eng_{c}"), ("+target enc", "t2_te_{c}"),
             ("+te+eng", "t2_te_eng_{c}"), ("+MI top-20", "t2_te_eng_mi_{c}"),
             ("+Fisher top-20", "t2_te_eng_fis_{c}"), ("+PCA 95%", "t2_te_eng_pca_{c}"),
             ("+LDA proj", "t2_te_eng_lda_{c}")]
    fig, ax = plt.subplots(figsize=(9, 4))
    rows = []
    for c, key in (("kNN", "kNN"), ("SVM-RBF", "SVM-RBF"), ("DecisionTree", "DecisionTree"), ("NaiveBayes", "NaiveBayes")):
        xs, ys, es = [], [], []
        for i, (lab, pat) in enumerate(order):
            r = by.get(pat.format(c=key))
            if r:
                xs.append(i); ys.append(r["f1_mean"]); es.append(r["f1_std"])
                rows.append(dict(Classifier=c, Step=lab, MacroF1=round(r["f1_mean"], 4), Acc=round(r["acc_mean"], 4)))
        ax.errorbar(xs, ys, es, marker="o", capsize=2, label=c)
    ax.set_xticks(range(len(order)), [o[0] for o in order], rotation=25)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Ablation: effect of each feature step")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "ablation_feature_steps.png", dpi=150)
    plt.close(fig)
    return pd.DataFrame(rows)


def run():
    rs = load()
    if not rs:
        log("report: no results yet")
        return
    df = table(rs)
    df.drop(columns=["acc", "f1", "name"]).to_csv(RES / "summary.csv", index=False)
    sig = significance(rs)
    sig.to_csv(RES / "significance.csv", index=False)
    abl = fig_ablation(rs)
    abl.to_csv(RES / "ablation.csv", index=False)
    fig_bars(rs); fig_cm(rs); fig_perclass(rs)
    pc = []
    for g in COL:
        b = best(rs, g)
        if b:
            for i, n in enumerate(NAMES):
                pc.append(dict(Group=g, Method=b["label"], Class=n, Precision=round(b["prec"][i], 4), Recall=round(b["rec"][i], 4)))
    pc = pd.DataFrame(pc)
    pc.to_csv(RES / "per_class_best.csv", index=False)
    md = ["# Results (mean ± std over %d runs)\n" % rs[0]["n_runs"],
          "## All methods\n", df.drop(columns=["acc", "f1", "name"]).to_markdown(index=False),
          "\n## Best per group: per-class precision / recall\n", pc.to_markdown(index=False),
          "\n## Ablation\n", abl.to_markdown(index=False),
          "\n## Significance (paired t-test and Nadeau-Bengio corrected t-test)\n", sig.to_markdown(index=False)]
    (RES / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    log("report written")


if __name__ == "__main__":
    run()
