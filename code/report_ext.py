"""Combined tables/figures over results/*.json (tracks 1-3) and results/ext/*.json (zoo + tuned)."""
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIG, N_SPLITS, log
from report import corrected_t

PALETTE = ["#8c8c8c", "#2a7ab0", "#d9822b", "#7b5ea7",
           "#3d9970", "#c0392b", "#16a085", "#b8860b", "#555555", "#e07fb0", "#1f4e79"]


def load(d):
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]


def tidy(rs):
    rows = []
    for r in rs:
        rec = r["rec"]
        rows.append(dict(Group=r["group"], Method=r["label"], Name=r["name"],
                         Accuracy=r["acc_mean"], Acc_std=r["acc_std"],
                         MacroF1=r["f1_mean"], F1_std=r["f1_std"],
                         Rec_g1=rec[0], Rec_g2=rec[1], Rec_g3=rec[2], Seconds=round(r["seconds"])))
    return pd.DataFrame(rows).sort_values("MacroF1", ascending=False).reset_index(drop=True)


def md(df, cols, fmt):
    d = df[cols].copy()
    for c, f in fmt.items():
        d[c] = d[c].map(f.format)
    return d.to_markdown(index=False)


def run(main_res, ext_res):
    rs = load(main_res) + load(ext_res)
    if not rs:
        return
    for r in rs:
        r.setdefault("rec", [np.nan] * 3)
    df = tidy(rs)
    groups = sorted(df.Group.unique())
    col = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(groups)}
    df.to_csv(ext_res / "master_table.csv", index=False)
    df["F1 (mean±std)"] = [f"{a:.4f} ± {b:.4f}" for a, b in zip(df.MacroF1, df.F1_std)]
    df["Acc (mean±std)"] = [f"{a:.4f} ± {b:.4f}" for a, b in zip(df.Accuracy, df.Acc_std)]
    cols = ["Group", "Method", "Acc (mean±std)", "F1 (mean±std)", "Rec_g1", "Rec_g2", "Rec_g3", "Seconds"]
    fmt = dict(Rec_g1="{:.3f}", Rec_g2="{:.3f}", Rec_g3="{:.3f}")
    out = ["# Master results table (all methods, sorted by macro-F1)\n",
           md(df, cols, fmt), "\n\n# Best method per group\n"]
    best = df.sort_values("MacroF1", ascending=False).groupby("Group").head(1).sort_values("Group")
    out.append(md(best, cols, fmt))

    # tuned vs untuned (paired, corrected resampled t-test)
    by = {r["name"]: r for r in rs}
    rows = []
    for n, r in by.items():
        if not n.startswith("t5_"):
            continue
        key = n[3:]
        base = by.get(f"t4_{key}")
        if base is None:
            continue
        d, t0, p0, t1, p1 = corrected_t(r["macro_f1"], base["macro_f1"], 1 / N_SPLITS)
        rows.append(dict(Model=base["label"], Untuned_F1=base["f1_mean"], Tuned_F1=r["f1_mean"],
                         Gain=d, p_corrected=p1, Untuned_Acc=base["acc_mean"], Tuned_Acc=r["acc_mean"]))
    if rows:
        tv = pd.DataFrame(rows).sort_values("Gain", ascending=False)
        tv.to_csv(ext_res / "tuned_vs_untuned.csv", index=False)
        out += ["\n\n# Tuned (nested CV) vs untuned, macro-F1\n",
                md(tv, list(tv.columns), {c: "{:.4f}" for c in tv.columns if c != "Model"})]
        fig, ax = plt.subplots(figsize=(8, 0.35 * len(tv) + 1.5))
        y = np.arange(len(tv))
        ax.barh(y - 0.2, tv.Untuned_F1, 0.4, label="untuned", color="#8c8c8c")
        ax.barh(y + 0.2, tv.Tuned_F1, 0.4, label="tuned (nested CV)", color="#2a7ab0")
        ax.set_yticks(y, tv.Model, fontsize=7)
        ax.set_xlim(max(0, tv[["Untuned_F1", "Tuned_F1"]].min().min() - 0.04), None)
        ax.set_xlabel("Macro-F1")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG / "tuned_vs_untuned.png", dpi=150)
        plt.close(fig)

    # tuning log summary
    lp = ext_res / "tuning_log.jsonl"
    if lp.exists():
        recs = [json.loads(l) for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()]
        rows = []
        for m in sorted({r["model"] for r in recs}):
            rr = [r for r in recs if r["model"] == m]
            cnt = Counter(json.dumps(r["best_params"], sort_keys=True) for r in rr)
            top, n = cnt.most_common(1)[0]
            rows.append(dict(Model=m, Searches=len(rr), Inner_F1=np.mean([r["best_inner_f1"] for r in rr]),
                             Most_chosen=top, Times=f"{n}/{len(rr)}", Secs_per_search=np.mean([r["seconds"] for r in rr])))
        ts = pd.DataFrame(rows)
        ts.to_csv(ext_res / "tuning_summary.csv", index=False)
        out += ["\n\n# What the inner searches picked (most frequent winner over outer folds)\n",
                md(ts, list(ts.columns), {"Inner_F1": "{:.4f}", "Secs_per_search": "{:.0f}"})]

    (ext_res / "master_table.md").write_text("\n".join(out), encoding="utf-8")

    top = df.head(40).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.25 * len(top) + 1.5))
    ax.barh(top.Method, top.MacroF1, xerr=top.F1_std, color=[col[g] for g in top.Group], capsize=2)
    ax.set_xlim(max(0, top.MacroF1.min() - 0.04), None)
    ax.set_xlabel("Macro-F1 (mean ± std, 5-fold x 3 repeats)")
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=col[g]) for g in groups], labels=groups,
              fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "macro_f1_top40_all_families.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for g in groups:
        s = df[df.Group == g]
        ax.scatter(s.Seconds.clip(lower=1), s.MacroF1, label=g, color=col[g], s=28)
    ax.set_xscale("log")
    ax.set_xlabel("Wall-clock seconds for 15 CV runs (log)")
    ax.set_ylabel("Macro-F1")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "f1_vs_cost.png", dpi=150)
    plt.close(fig)
    log(f"report_ext: {len(df)} methods -> master_table.md / csv, 3 figures")
