"""Plots of the final test results."""

from typing import Dict, Optional


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLOURS = ["grey", "tab:red", "tab:blue"]


def plot_test_results(test: pd.DataFrame, preds: Dict[str, np.ndarray], y_te: pd.Series,
                      path: Optional[str] = None, show: bool = False):
    """Bar chart of test MAE plus observed-vs-predicted for the first test week.

    Args:
        test: Output of :func:`smart_infrastructure.evaluation.final_test`.
        preds: Predictions per model (same order as ``y_te``).
        y_te: Observed test headcount, indexed by timestamp.
        path: If given, the figure is saved here (PNG).
        show: Call ``plt.show()`` (interactive use).

    Returns:
        The matplotlib figure.
    """
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    test["MAE"].plot.bar(ax=ax[0], color=COLOURS[:len(test)], rot=15)
    ax[0].set(title="Final test MAE (people)", ylabel="MAE")
    n = min(7 * 96, len(y_te))          # first test week (96 x 15 min per day)
    ax[1].plot(y_te.index[:n], y_te.values[:n], "k", lw=1, label="observed")
    for name, c in zip(preds, COLOURS):
        ax[1].plot(y_te.index[:n], preds[name][:n], color=c, lw=1, label=name)
    ax[1].set(title="First test week", ylabel="People")
    ax[1].legend()
    ax[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    if show:
        plt.show()
    return fig
