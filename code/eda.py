"""Person 1: exploratory data analysis figures + stats on the full data and the subsample."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIG, GEO, RES, load_data, get_subsample, log


def run():
    X, y = load_data()
    Xs, ys = get_subsample()
    lines = [f"Full data: {X.shape[0]} rows, {X.shape[1]} features; subsample: {len(Xs)} rows",
             f"Missing values: {int(X.isna().sum().sum())}",
             "Class share full / subsample (grade 1,2,3): "
             + " ".join(f"{(y == c).mean():.3f}/{(ys == c).mean():.3f}" for c in range(3))]
    for g in GEO:
        lines.append(f"{g}: {X[g].nunique()} unique ids")
    (RES / "eda.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    axs[0].bar(["1", "2", "3"], [(y == c).mean() for c in range(3)], color=["#7fb77e", "#e0b44c", "#c9514a"])
    axs[0].set_title("Class balance (damage grade)")
    for g, c in zip((0, 1, 2), ["#7fb77e", "#e0b44c", "#c9514a"]):
        axs[1].hist(np.log1p(X["age"][y == g]), bins=30, alpha=.55, label=f"grade {g+1}", color=c, density=True)
    axs[1].set_title("log(1+age) by grade"); axs[1].legend()
    share = [(y[X["geo_level_1_id"] == k] == 2).mean() for k in sorted(X["geo_level_1_id"].unique())]
    axs[2].bar(range(len(share)), share, color="#2a7ab0")
    axs[2].set_title("Share of grade 3 per geo_level_1 region")
    fig.tight_layout()
    fig.savefig(FIG / "eda_overview.png", dpi=150)
    plt.close(fig)

    cat = [c for c in X.columns if X[c].dtype == object]
    fig, axs = plt.subplots(2, 4, figsize=(14, 6))
    for ax, c in zip(axs.ravel(), cat):
        ct = (X.assign(y=y).groupby([c, "y"]).size().unstack(fill_value=0))
        (ct.div(ct.sum(1), axis=0)).plot.bar(stacked=True, ax=ax, legend=False, color=["#7fb77e", "#e0b44c", "#c9514a"])
        ax.set_title(c, fontsize=9); ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(FIG / "eda_categorical.png", dpi=130)
    plt.close(fig)
    log("eda done")


if __name__ == "__main__":
    run()
