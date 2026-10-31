"""Consistent chart styling. Each model keeps the same colour in every chart."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

# Validated categorical palette (fixed order, never cycled)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MUTED = "#8a8984"
TEXT = "#2b2b29"

MODEL_COLORS = {
    "Logistic Regression": SERIES[0],
    "Random Forest": SERIES[1],
    "XGBoost": SERIES[2],
    "Random Forest + SMOTE (train folds only)": SERIES[3],
}


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "axes.titleweight": "bold",
        "axes.titlesize": 12, "axes.titlelocation": "left",
        "axes.grid": True, "grid.color": "#e4e3de", "grid.linewidth": 0.8,
        "xtick.color": TEXT, "ytick.color": TEXT, "lines.linewidth": 2,
        "legend.frameon": False, "font.size": 10,
    })


def color_for(name, i=0):
    return MODEL_COLORS.get(name, SERIES[i % len(SERIES)])


def roc_pr_panel(y, preds, title, path=None):
    """ROC and precision-recall side by side; PR shows the no-skill baseline."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, (name, p) in enumerate(preds.items()):
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        a.plot(fpr, tpr, color=color_for(name, i), label=name)
        b.plot(rec, prec, color=color_for(name, i), label=name)
    a.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1, label="Random guess")
    b.axhline(np.mean(y), ls="--", color=MUTED, lw=1, label=f"Random guess ({np.mean(y):.1%})")
    a.set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="ROC curve")
    b.set(xlabel="Recall (injuries caught)", ylabel="Precision (flags that were injuries)",
          title="Precision-recall curve")
    a.legend(loc="lower right"); b.legend(loc="upper right")
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout()
    if path:
        fig.savefig(path)
    return fig


def calibration_plot(y, p, title, path=None, n_bins=10):
    frac, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    lim = max(mean_pred.max(), frac.max()) * 1.1
    ax.plot([0, lim], [0, lim], ls="--", color=MUTED, lw=1, label="Perfectly calibrated")
    ax.plot(mean_pred, frac, marker="o", ms=8, color=SERIES[0], label="Model")
    ax.set(xlabel="Predicted injury probability", ylabel="Observed injury rate", title=title,
           xlim=(0, lim), ylim=(0, lim))
    ax.legend(loc="upper left")
    fig.tight_layout()
    if path:
        fig.savefig(path)
    return fig


def lift_plot(y, p, title, path=None):
    """Share of injuries caught when flagging the top-k% riskiest athlete-days."""
    order = np.argsort(-p)
    caught = np.cumsum(np.asarray(y)[order]) / np.sum(y)
    flagged = np.arange(1, len(p) + 1) / len(p)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(flagged * 100, caught * 100, color=SERIES[0], label="Model")
    ax.plot([0, 100], [0, 100], ls="--", color=MUTED, lw=1, label="Random flagging")
    for k in (0.1, 0.2):
        c = caught[int(k * len(p)) - 1]
        ax.plot(k * 100, c * 100, "o", ms=8, color=SERIES[0], mec="white", mew=2)
        ax.annotate(f"top {k:.0%} → {c:.0%} of injuries", (k * 100, c * 100),
                    xytext=(8, -14), textcoords="offset points", color=TEXT)
    ax.set(xlabel="% of athlete-days flagged (highest risk first)", ylabel="% of injuries caught",
           title=title, xlim=(0, 100), ylim=(0, 100))
    ax.legend(loc="lower right")
    fig.tight_layout()
    if path:
        fig.savefig(path)
    return fig


def metric_bars(df, metric, title, path=None, baseline=None, xlabel=None):
    """Horizontal bars, one per row of df (index = label)."""
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(df) + 1.4))
    vals = df[metric].values
    labels = list(df.index)
    colors = [color_for(l, i) for i, l in enumerate(labels)]
    ax.barh(labels, vals, color=colors, height=0.6, edgecolor="white", linewidth=2)
    for i, v in enumerate(vals):
        ax.text(v, i, f"  {v:.3f}", va="center", color=TEXT)
    if baseline is not None:
        ax.axvline(baseline, ls="--", color=MUTED, lw=1)
        ax.text(baseline, len(df) - 0.4, f" random = {baseline:.2f}", color=MUTED, va="bottom")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set(xlabel=xlabel or metric, title=title, xlim=(0, max(vals.max() * 1.18, 1e-9)))
    fig.tight_layout()
    if path:
        fig.savefig(path)
    return fig
