import numpy as np
import pandas as pd

from smart_infrastructure.plots import plot_test_results


def test_plot_saves_png_and_has_two_panels(tmp_path):
    idx = pd.date_range("2024-05-24", periods=800, freq="15min")
    y = pd.Series(np.arange(800.0) % 10, index=idx)
    test = pd.DataFrame({"MAE": [1.0, 0.8, 0.6]}, index=["a", "b", "c"])
    preds = {k: y.to_numpy() + i for i, k in enumerate("abc")}
    p = tmp_path / "f.png"
    fig = plot_test_results(test, preds, y, path=str(p))
    assert p.exists() and len(fig.axes) == 2
    assert len(fig.axes[1].lines) == 4                        # observed + 3 models
