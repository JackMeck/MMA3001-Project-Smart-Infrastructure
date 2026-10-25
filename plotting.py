"""Figures for the held-out test period."""

from __future__ import annotations


import matplotlib.pyplot as plt
import pandas as pd

COLOURS = ["grey", "tab:red", "tab:blue"]


def plot_test_results(test: pd.DataFrame, y_te: pd.Series, preds: dict, path: str | None = None,
                      days: int = 7):
    """Bar chart of test MAE and a time series of the first test week.

    Args:
        test: Metrics table from :func:`occupancy.experiments.final_test`.
        y_te: Observed test headcounts (DatetimeIndex, 15-minute steps).
        preds: Mapping model name -> predictions aligned with ``y_te``.
        path: If given, the figure is saved there (PNG); otherwise it is only returned.
        days: Number of test days to show (96 intervals per day).

    Returns:
        The matplotlib ``Figure``.
    """
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    test["MAE"].plot.bar(ax=ax[0], color=COLOURS[:len(test)], rot=15)
    ax[0].set(title="Final test MAE (people)", ylabel="MAE")
    n = days * 96  # 96 fifteen-minute intervals per day
    ax[1].plot(y_te.index[:n], y_te.values[:n], "k", lw=1, label="observed")
    for (name, p), c in zip(preds.items(), COLOURS):
        ax[1].plot(y_te.index[:n], p[:n], color=c, lw=1, label=name)
    ax[1].set(title=f"First test {'week' if days == 7 else f'{days} days'}", ylabel="People")
    ax[1].legend()
    ax[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig
