"""Principal components of historical government yield changes.

Yield levels and changes are decimal annual rates internally. PCA is fitted to
daily changes, never to yield levels. The default decomposition uses the sample
covariance matrix; correlation PCA is available only through an explicit
method choice. One basis point is exactly ``0.0001`` in decimal yield terms.
"""

from __future__ import annotations

import math
import re
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

BASIS_POINT_DECIMAL = 0.0001
"""One basis point expressed as a decimal annual yield change."""

_MATURITY_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)\s*([MY])$", re.IGNORECASE)


class PCAMethod(StrEnum):
    """Matrix used for principal-component estimation."""

    COVARIANCE = "covariance"
    CORRELATION = "correlation"


class PCAMissingPolicy(StrEnum):
    """Explicit policy for changes affected by missing yield levels."""

    RAISE = "raise"
    DROP = "drop"


@dataclass(frozen=True)
class FactorShapeDiagnostic:
    """Quantitative interpretation of one dimensionless loading vector.

    ``suggested_label`` is ``None`` when the strongest template match is too
    weak or insufficiently distinct from the second strongest match.
    """

    component_number: int
    suggested_label: str | None
    level_similarity: float
    slope_similarity: float
    curvature_similarity: float
    strongest_similarity: float
    confidence_margin: float


@dataclass(frozen=True)
class PCAResult:
    """Historical yield-change PCA result with explicit units.

    ``yield_changes_decimal`` and ``means_decimal`` are decimal annual yield
    changes. ``yield_changes_bp`` contains those same changes divided by
    exactly ``0.0001``. ``covariance_matrix`` is in squared decimal changes.
    Loadings and explained-variance ratios are dimensionless. Covariance-PCA
    eigenvalues and scores are squared decimal changes and decimal changes,
    respectively. Correlation-PCA eigenvalues and scores are dimensionless
    because the centered changes are explicitly standardized first.
    """

    method: PCAMethod
    missing_policy: PCAMissingPolicy
    maturity_labels: tuple[Hashable, ...]
    maturities_years: np.ndarray
    yield_changes_decimal: pd.DataFrame
    yield_changes_bp: pd.DataFrame
    means_decimal: np.ndarray
    scales_decimal: np.ndarray
    covariance_matrix: np.ndarray
    decomposition_matrix: np.ndarray
    eigenvalues: np.ndarray
    loadings: np.ndarray
    explained_variance_ratios: np.ndarray
    factor_scores: np.ndarray

    @property
    def eigenvectors(self) -> np.ndarray:
        """Return dimensionless loadings, provided as an eigenvector alias."""
        return self.loadings

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Return dimensionless variance ratios using the common singular name."""
        return self.explained_variance_ratios

    @property
    def means_bp(self) -> np.ndarray:
        """Return per-maturity mean daily yield changes in basis points."""
        return self.means_decimal / BASIS_POINT_DECIMAL

    @property
    def score_dates(self) -> pd.DatetimeIndex:
        """Return dates attached to factor-score observations; units are dates."""
        return self.yield_changes_decimal.index.copy()

    def reconstruct(
        self, n_components: int | None = None, *, include_mean: bool = True
    ) -> pd.DataFrame:
        """Reconstruct retained daily yield changes in decimal annual units.

        Args:
            n_components: Number of leading components to retain. ``None`` uses
                every component and reproduces the complete input changes up to
                floating-point error.
            include_mean: Add the fitted decimal mean changes when true.

        Returns:
            A date-by-maturity DataFrame of reconstructed decimal daily yield
            changes. Correlation scores are first returned to decimal units
            using the fitted sample standard deviations.
        """
        retained = _validate_component_count(n_components, self.loadings.shape[1])
        standardized = self.factor_scores[:, :retained] @ self.loadings[:, :retained].T
        reconstructed = standardized * self.scales_decimal
        if include_mean:
            reconstructed = reconstructed + self.means_decimal
        return pd.DataFrame(
            reconstructed,
            index=self.yield_changes_decimal.index.copy(),
            columns=self.maturity_labels,
        )


def maturity_in_years(label: Hashable) -> float:
    """Convert a maturity label to years.

    Numeric labels are interpreted directly as years. String labels may be a
    positive number of years (``"2"``), months (``"6M"``), or years
    (``"10Y"``). Boolean values and non-positive maturities are rejected.
    """
    if isinstance(label, bool):
        raise TypeError("maturity labels must be numeric years or strings such as '5Y'")
    if isinstance(label, (int, float, np.integer, np.floating)):
        maturity = float(label)
    elif isinstance(label, str):
        stripped = label.strip()
        match = _MATURITY_PATTERN.fullmatch(stripped)
        if match is not None:
            quantity = float(match.group(1))
            maturity = quantity / 12.0 if match.group(2).upper() == "M" else quantity
        else:
            try:
                maturity = float(stripped)
            except ValueError as exc:
                raise ValueError(
                    f"could not interpret maturity label {label!r} as years"
                ) from exc
    else:
        raise TypeError("maturity labels must be numeric years or strings such as '5Y'")
    if not math.isfinite(maturity) or maturity <= 0.0:
        raise ValueError("maturities must be finite and strictly positive years")
    return maturity


def calculate_yield_changes(
    yield_levels: pd.DataFrame,
    *,
    missing: PCAMissingPolicy | str = PCAMissingPolicy.DROP,
) -> pd.DataFrame:
    """Calculate adjacent-date government yield changes in decimal units.

    Args:
        yield_levels: Increasing, unique date index by maturity panel. Values
            are decimal annual yields, so ``0.0425`` means 4.25%. Maturity
            columns are sorted by numerical tenor before differencing.
        missing: ``"drop"`` removes any change row lacking a maturity;
            ``"raise"`` reports such rows. Missing levels are never filled, so
            a gap invalidates both changes that would otherwise cross it.

    Returns:
        Complete adjacent-date yield changes in decimal annual yield units.
        The first level row has no prior observation and is omitted.
    """
    policy = _normalise_missing_policy(missing)
    levels, _, _ = _validate_and_order_levels(yield_levels)
    changes = levels - levels.shift(1)
    changes = changes.iloc[1:]
    missing_rows = changes.isna().any(axis="columns")
    if bool(missing_rows.any()) and policy is PCAMissingPolicy.RAISE:
        dates = [
            timestamp.date().isoformat() for timestamp in changes.index[missing_rows]
        ]
        raise ValueError(
            "yield changes contain missing values caused by missing adjacent "
            f"levels on dates: {dates}"
        )
    if policy is PCAMissingPolicy.DROP:
        changes = changes.loc[~missing_rows]
    if changes.empty:
        raise ValueError("no complete daily yield changes remain")
    return changes.astype(float)


def fit_yield_change_pca(
    yield_levels: pd.DataFrame,
    *,
    method: PCAMethod | str = PCAMethod.COVARIANCE,
    missing: PCAMissingPolicy | str = PCAMissingPolicy.DROP,
) -> PCAResult:
    """Fit PCA to historical daily yield changes using NumPy eigendecomposition.

    Args:
        yield_levels: Increasing, unique date-by-maturity panel of decimal
            annual government yields. Columns are numerically tenor-sorted.
        method: ``"covariance"`` (default) uses decimal yield changes directly.
            ``"correlation"`` explicitly standardizes each maturity by its
            sample change volatility before decomposition.
        missing: Explicitly ``"drop"`` complete-case change rows or ``"raise"``.
            No forward fill, backward fill, or interpolation is performed.

    Returns:
        All components, descending eigenvalues, deterministic loading signs,
        scores, decimal and bp change panels, and numerical maturities. Sample
        covariance uses denominator ``n-1``.

    Notes:
        Loadings describe empirical movements of the input yield series. If the
        inputs are constant-maturity Treasury yields, this does not turn them
        into a bootstrapped zero-coupon curve.
    """
    normalized_method = _normalise_method(method)
    policy = _normalise_missing_policy(missing)
    ordered_levels, labels, maturities = _validate_and_order_levels(yield_levels)
    changes = calculate_yield_changes(ordered_levels, missing=policy)
    if len(changes) < 2:
        raise ValueError("PCA requires at least two complete daily yield changes")

    observations = changes.to_numpy(dtype=float)
    means = observations.mean(axis=0)
    centered = observations - means
    covariance = centered.T @ centered / (len(centered) - 1)
    if normalized_method is PCAMethod.COVARIANCE:
        scales = np.ones(centered.shape[1], dtype=float)
        analysis_observations = centered
        decomposition_matrix = covariance
    else:
        scales = centered.std(axis=0, ddof=1)
        near_zero = scales == 0.0
        if np.any(near_zero):
            zero_columns = [str(labels[index]) for index in np.flatnonzero(near_zero)]
            raise ValueError(
                "correlation PCA requires non-zero change variance at every "
                f"maturity; zero variance: {zero_columns}"
            )
        analysis_observations = centered / scales
        decomposition_matrix = (
            analysis_observations.T @ analysis_observations / (len(centered) - 1)
        )

    eigenvalues, loadings = np.linalg.eigh(decomposition_matrix)
    order = np.argsort(eigenvalues, kind="stable")[::-1]
    eigenvalues = eigenvalues[order]
    loadings = loadings[:, order]
    # Variance is in squared decimal yields; an absolute unit-sized floor would
    # reject valid small-scale samples and break PCA's scaling identity.
    tolerance = np.finfo(float).eps * float(np.max(np.abs(eigenvalues))) * 100
    if np.any(eigenvalues < -tolerance):
        raise ArithmeticError("PCA decomposition produced materially negative variance")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total_variance = float(eigenvalues.sum())
    if total_variance <= tolerance:
        raise ValueError("yield changes have no measurable cross-sectional variance")

    loadings = _sign_normalize(loadings)
    scores = analysis_observations @ loadings
    changes_bp = changes / BASIS_POINT_DECIMAL
    return PCAResult(
        method=normalized_method,
        missing_policy=policy,
        maturity_labels=labels,
        maturities_years=maturities,
        yield_changes_decimal=changes,
        yield_changes_bp=changes_bp,
        means_decimal=means,
        scales_decimal=scales,
        covariance_matrix=covariance,
        decomposition_matrix=decomposition_matrix,
        eigenvalues=eigenvalues,
        loadings=loadings,
        explained_variance_ratios=eigenvalues / total_variance,
        factor_scores=scores,
    )


def fit_yield_curve_pca(
    yield_levels: pd.DataFrame,
    *,
    method: PCAMethod | str = PCAMethod.COVARIANCE,
    missing: PCAMissingPolicy | str = PCAMissingPolicy.DROP,
) -> PCAResult:
    """Fit yield-change PCA; inputs and outputs use the units of the main API.

    This descriptive alias delegates to :func:`fit_yield_change_pca`. Input
    levels and covariance-PCA scores are decimal annual yield values.
    """
    return fit_yield_change_pca(yield_levels, method=method, missing=missing)


def normalize_loading_signs(
    loadings: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Apply deterministic signs to dimensionless PCA loading columns.

    PC1 is oriented to have a positive mean loading. PC2 is oriented so its
    long-end loading exceeds its short-end loading. PC3 is oriented so its
    quadratic belly-versus-wings contrast is positive. Later components, and
    any near-zero diagnostic contrast, are oriented so the first
    largest-absolute loading is positive. Sign changes do not alter
    reconstruction because PCA scores must receive the same sign when a fitted
    result is constructed.
    """
    array = _loading_matrix(loadings)
    return _sign_normalize(array)


def interpret_loading_shapes(
    maturities_years: Sequence[float] | np.ndarray,
    loadings: Sequence[Sequence[float]] | np.ndarray,
    *,
    max_components: int = 3,
    minimum_similarity: float = 0.65,
    minimum_margin: float = 0.10,
) -> tuple[FactorShapeDiagnostic, ...]:
    """Assess whether leading loadings resemble level, slope, or curvature.

    Args:
        maturities_years: Strictly increasing positive maturities in years.
        loadings: Dimensionless loading matrix with maturities in rows and
            components in columns.
        max_components: Number of leading components to assess, at most three.
        minimum_similarity: Minimum absolute cosine similarity for a label.
        minimum_margin: Required gap between the strongest and next template.

    Returns:
        Diagnostics with dimensionless absolute cosine similarities. Templates
        are orthonormal polynomials over ordered tenor ranks: constant for
        level, monotonic short-to-long for slope, and belly-versus-wings for
        curvature. Rank spacing prevents a sparse long end from dominating.
        A label is withheld unless both thresholds are satisfied; component
        number alone never determines the label.
    """
    maturities = _maturity_vector(maturities_years)
    loading_matrix = _loading_matrix(loadings)
    if loading_matrix.shape[0] != maturities.size:
        raise ValueError("loadings rows must match maturities_years")
    if isinstance(max_components, bool) or not isinstance(max_components, int):
        raise TypeError("max_components must be an integer")
    if max_components < 1 or max_components > 3:
        raise ValueError("max_components must be between one and three")
    count = min(max_components, loading_matrix.shape[1])
    _validate_similarity_threshold(minimum_similarity, "minimum_similarity")
    _validate_similarity_threshold(minimum_margin, "minimum_margin")
    templates = _shape_templates(maturities.size)
    names = ("level", "slope", "curvature")
    diagnostics: list[FactorShapeDiagnostic] = []
    for component in range(count):
        loading = loading_matrix[:, component]
        norm = float(np.linalg.norm(loading))
        if norm <= np.finfo(float).eps:
            raise ValueError("loading columns must have non-zero norm")
        similarities = np.abs(templates.T @ (loading / norm))
        ranked = np.argsort(similarities)[::-1]
        strongest = int(ranked[0])
        best = float(similarities[strongest])
        margin = best - float(similarities[ranked[1]])
        label = (
            names[strongest]
            if best >= minimum_similarity and margin >= minimum_margin
            else None
        )
        diagnostics.append(
            FactorShapeDiagnostic(
                component_number=component + 1,
                suggested_label=label,
                level_similarity=float(similarities[0]),
                slope_similarity=float(similarities[1]),
                curvature_similarity=float(similarities[2]),
                strongest_similarity=best,
                confidence_margin=margin,
            )
        )
    return tuple(diagnostics)


def interpret_pca_factors(
    result: PCAResult,
    *,
    max_components: int = 3,
    minimum_similarity: float = 0.65,
    minimum_margin: float = 0.10,
) -> tuple[FactorShapeDiagnostic, ...]:
    """Interpret a PCA result's leading dimensionless loading shapes.

    Maturities are measured in years. The returned similarity scores are
    dimensionless and labels may be ``None`` when the evidence is ambiguous.
    """
    if not isinstance(result, PCAResult):
        raise TypeError("result must be a PCAResult")
    return interpret_loading_shapes(
        result.maturities_years,
        result.loadings,
        max_components=max_components,
        minimum_similarity=minimum_similarity,
        minimum_margin=minimum_margin,
    )


def _validate_and_order_levels(
    yield_levels: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[Hashable, ...], np.ndarray]:
    if not isinstance(yield_levels, pd.DataFrame):
        raise TypeError("yield_levels must be a pandas DataFrame")
    if yield_levels.empty or len(yield_levels.columns) == 0:
        raise ValueError("yield_levels must contain dates and maturity columns")
    if not isinstance(yield_levels.index, pd.DatetimeIndex):
        raise TypeError("yield_levels must have a DatetimeIndex")
    if yield_levels.index.hasnans or yield_levels.index.has_duplicates:
        raise ValueError("yield_levels dates must be valid and unique")
    if not yield_levels.index.is_monotonic_increasing:
        raise ValueError("yield_levels dates must be strictly increasing")
    if yield_levels.columns.has_duplicates:
        raise ValueError("yield_levels maturity labels must be unique")

    maturities = np.asarray(
        [maturity_in_years(label) for label in yield_levels.columns]
    )
    if len(np.unique(maturities)) != len(maturities):
        raise ValueError("yield_levels contains duplicate numerical maturities")
    order = np.argsort(maturities, kind="stable")
    labels = tuple(yield_levels.columns[index] for index in order)
    ordered_maturities = maturities[order]
    selected = yield_levels.loc[:, list(labels)]
    numeric = selected.apply(pd.to_numeric, errors="coerce")
    malformed = numeric.isna() & ~selected.isna()
    if bool(malformed.to_numpy().any()):
        raise ValueError("yield_levels contains non-numeric yield observations")
    values = numeric.to_numpy(dtype=float)
    finite_or_missing = np.isfinite(values) | np.isnan(values)
    if not bool(finite_or_missing.all()):
        raise ValueError("yield_levels contains infinite yield observations")
    present = values[~np.isnan(values)]
    if np.any((present < -0.2) | (present > 1.0)):
        raise ValueError(
            "yield_levels must be decimal annual rates between -0.2 and 1.0"
        )
    return numeric.astype(float).copy(), labels, ordered_maturities


def _normalise_method(method: PCAMethod | str) -> PCAMethod:
    try:
        return PCAMethod(method)
    except ValueError as exc:
        raise ValueError(
            "method must be explicitly 'covariance' or 'correlation'"
        ) from exc


def _normalise_missing_policy(
    missing: PCAMissingPolicy | str,
) -> PCAMissingPolicy:
    try:
        return PCAMissingPolicy(missing)
    except ValueError as exc:
        raise ValueError("missing must be explicitly 'drop' or 'raise'") from exc


def _loading_matrix(
    loadings: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    try:
        matrix = np.asarray(loadings, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("loadings must be a numeric two-dimensional matrix") from exc
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError("loadings must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("loadings must contain only finite values")
    return matrix.copy()


def _sign_normalize(loadings: np.ndarray) -> np.ndarray:
    oriented = loadings.copy()
    for component in range(oriented.shape[1]):
        vector = oriented[:, component]
        if component == 0:
            contrast = float(vector.mean())
        elif component == 1:
            contrast = float(vector[-1] - vector[0])
        elif component == 2:
            template = _shape_templates(len(vector))[:, 2]
            contrast = float(vector @ template)
        else:
            contrast = 0.0
        if abs(contrast) <= np.finfo(float).eps * 100:
            contrast = float(vector[int(np.argmax(np.abs(vector)))])
        if contrast < 0.0:
            oriented[:, component] *= -1.0
    return oriented


def _shape_templates(maturity_count: int) -> np.ndarray:
    if maturity_count < 3:
        raise ValueError("level/slope/curvature diagnostics require three maturities")
    ranks = np.linspace(-1.0, 1.0, maturity_count)
    candidates = np.column_stack((np.ones(maturity_count), ranks, -(ranks**2)))
    templates, _ = np.linalg.qr(candidates)
    if templates[:, 0].mean() < 0.0:
        templates[:, 0] *= -1.0
    if templates[-1, 1] - templates[0, 1] < 0.0:
        templates[:, 1] *= -1.0
    if templates[:, 2] @ (-(ranks**2)) < 0.0:
        templates[:, 2] *= -1.0
    return templates


def _maturity_vector(maturities_years: Sequence[float] | np.ndarray) -> np.ndarray:
    try:
        maturities = np.asarray(maturities_years, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("maturities_years must be a numeric sequence") from exc
    if maturities.ndim != 1 or maturities.size < 3:
        raise ValueError("at least three one-dimensional maturities are required")
    if not np.all(np.isfinite(maturities)) or np.any(maturities <= 0.0):
        raise ValueError("maturities_years must be finite and positive")
    if np.any(np.diff(maturities) <= 0.0):
        raise ValueError("maturities_years must be strictly increasing")
    return maturities


def _validate_component_count(value: int | None, available: int) -> int:
    if value is None:
        return available
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("n_components must be an integer or None")
    if value < 1 or value > available:
        raise ValueError(f"n_components must be between one and {available}")
    return value


def _validate_similarity_threshold(value: float, name: str) -> None:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a number between zero and one")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a number between zero and one") from exc
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be between zero and one")


__all__ = [
    "BASIS_POINT_DECIMAL",
    "FactorShapeDiagnostic",
    "PCAMethod",
    "PCAMissingPolicy",
    "PCAResult",
    "calculate_yield_changes",
    "fit_yield_change_pca",
    "fit_yield_curve_pca",
    "interpret_loading_shapes",
    "interpret_pca_factors",
    "maturity_in_years",
    "normalize_loading_signs",
]
