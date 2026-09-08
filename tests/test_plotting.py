"""Tests for plotting observed maturities and fitted NSS curves."""

import matplotlib
import numpy as np

matplotlib.use("Agg")

from fixed_income.curves import NSSParameters, calibrate_nss, nss_yields
from fixed_income.plotting import plot_nss_curve


def test_plot_nss_curve_labels_observations_and_fit() -> None:
    """The plot displays observed nodes and a delegated NSS evaluation."""
    maturities = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
    parameters = NSSParameters(0.04, -0.02, 0.01, -0.005, 1.5, 5.0)
    result = calibrate_nss(maturities, nss_yields(maturities, parameters))

    figure, axes = plot_nss_curve(result)

    assert axes.get_xlabel() == "Maturity (years)"
    assert axes.get_ylabel() == "Annual yield (%)"
    assert len(axes.collections) == 1
    assert len(axes.lines) == 1
    figure.clear()
