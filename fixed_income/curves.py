"""Typed curve representations, NSS fitting, and explicit zero-rate discounting.

Treasury constant-maturity yields (CMTs), par yields, zero rates, discount
factors, and fitted curves are distinct representations in this module.  In
particular, fitting Nelson-Siegel-Svensson (NSS) to CMT observations smooths
those observations; it does not bootstrap zero rates from Treasury cash flows.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum

import numpy as np
from scipy.optimize import least_squares


class CurveRepresentation(StrEnum):
    """Meaning of curve values; rate-like values use decimal annual units."""

    TREASURY_CMT = "treasury_constant_maturity_yield"
    PAR_YIELD = "par_yield"
    ZERO_RATE = "zero_rate"
    DISCOUNT_FACTOR = "discount_factor"
    FITTED_TREASURY_CMT = "fitted_treasury_constant_maturity_yield"
    FITTED_PAR_YIELD = "fitted_par_yield"
    FITTED_ZERO_RATE = "fitted_zero_rate"


class CompoundingConvention(StrEnum):
    """Supported annual-rate compounding conventions."""

    CONTINUOUS = "continuous"
    PERIODIC = "periodic"
    SIMPLE = "simple"


class ExtrapolationPolicy(StrEnum):
    """Behaviour requested outside the supplied maturity-node range."""

    RAISE = "raise"
    FLAT = "flat"


@dataclass(frozen=True)
class YieldCurve:
    """Immutable, tagged set of curve nodes.

    Args:
        maturities_years: Strictly increasing positive maturities in years.
        values: Decimal annual rates for rate representations, or
            dimensionless values for ``DISCOUNT_FACTOR``.
        representation: Financial meaning of ``values``.
        valuation_date: Optional calendar date on which the curve is observed.
        compounding: Required only for zero-rate representations. It states how
            decimal annual zero rates map to discount factors.
        periodic_frequency: Compounding periods per year when compounding is
            ``PERIODIC``; otherwise it must be omitted.
        source: Human-readable data or modelling provenance.

    Notes:
        Construction does not transform one financial representation into
        another. In particular, Treasury CMT nodes cannot be discounted with
        :meth:`discount_factors`.
    """

    maturities_years: tuple[float, ...]
    values: tuple[float, ...]
    representation: CurveRepresentation
    valuation_date: date | None = None
    compounding: CompoundingConvention | None = None
    periodic_frequency: int | None = None
    source: str = ""

    def __post_init__(self) -> None:
        """Validate units, node ordering, and representation metadata."""
        maturities = _as_finite_1d(self.maturities_years, "maturities_years")
        values = _as_finite_1d(self.values, "values")
        if maturities.size != values.size:
            raise ValueError("maturities_years and values must have equal length")
        if maturities.size == 0:
            raise ValueError("a curve must contain at least one node")
        if self.valuation_date is not None and (
            isinstance(self.valuation_date, datetime)
            or not isinstance(self.valuation_date, date)
        ):
            raise TypeError("valuation_date must be a date without a time component")
        if np.any(maturities <= 0.0):
            raise ValueError("maturities_years must be strictly positive")
        if np.any(np.diff(maturities) <= 0.0):
            raise ValueError("maturities_years must be strictly increasing and unique")
        try:
            representation = CurveRepresentation(self.representation)
        except ValueError as exc:
            raise ValueError(
                f"unsupported curve representation: {self.representation!r}"
            ) from exc
        object.__setattr__(self, "representation", representation)
        object.__setattr__(
            self, "maturities_years", tuple(float(x) for x in maturities)
        )
        object.__setattr__(self, "values", tuple(float(x) for x in values))

        zero_representation = representation in {
            CurveRepresentation.ZERO_RATE,
            CurveRepresentation.FITTED_ZERO_RATE,
        }
        if zero_representation:
            if self.compounding is None:
                raise ValueError(
                    "zero-rate curves require an explicit compounding convention"
                )
            convention = _validate_compounding(
                self.compounding, self.periodic_frequency
            )
            object.__setattr__(self, "compounding", convention)
        elif self.compounding is not None or self.periodic_frequency is not None:
            raise ValueError(
                "compounding metadata is only valid for a zero-rate representation"
            )
        if representation is CurveRepresentation.DISCOUNT_FACTOR and np.any(
            values <= 0.0
        ):
            raise ValueError("discount factors must be strictly positive")
        if representation is not CurveRepresentation.DISCOUNT_FACTOR:
            if np.any(np.abs(values) > 1.0):
                raise ValueError(
                    "curve values must be decimal annual rates within [-1, 1]"
                )
        if zero_representation:
            zero_rates_to_discount_factors(
                maturities,
                values,
                compounding=self.compounding,
                periodic_frequency=self.periodic_frequency,
            )

    def discount_factors(
        self,
        target_maturities_years: Sequence[float] | np.ndarray,
        *,
        extrapolation: ExtrapolationPolicy | str = ExtrapolationPolicy.RAISE,
    ) -> np.ndarray:
        """Return dimensionless discount factors at target times in years.

        Only a curve explicitly tagged as a zero-rate representation is
        accepted. Zero rates are linearly interpolated in decimal annual rate
        space before the curve's declared compounding formula is applied.
        """
        if self.representation not in {
            CurveRepresentation.ZERO_RATE,
            CurveRepresentation.FITTED_ZERO_RATE,
        }:
            raise ValueError(
                "discounting requires an explicitly tagged zero-rate curve; "
                f"received {self.representation.value!r}"
            )
        targets = _as_finite_1d(target_maturities_years, "target_maturities_years")
        rates = interpolate_zero_rates(
            self.maturities_years,
            self.values,
            targets,
            extrapolation=extrapolation,
        )
        return zero_rates_to_discount_factors(
            targets,
            rates,
            compounding=self.compounding,
            periodic_frequency=self.periodic_frequency,
        )


@dataclass(frozen=True)
class NSSParameters:
    """NSS parameters, with betas in decimal rates and taus in years."""

    beta0: float
    beta1: float
    beta2: float
    beta3: float
    tau1: float
    tau2: float

    def as_array(self) -> np.ndarray:
        """Return parameters ordered as four decimal betas then two year taus."""
        return np.array(
            [self.beta0, self.beta1, self.beta2, self.beta3, self.tau1, self.tau2],
            dtype=float,
        )


@dataclass(frozen=True)
class NSSCalibrationDiagnostics:
    """Optimizer diagnostics; conditioning is dimensionless, bounds are names.

    ``loading_condition_number`` measures identification of the conditional beta
    regression. ``jacobian_condition_number`` uses unit-norm Jacobian columns
    so decimal-beta and year-tau scales do not alone drive the comparison.
    Negligible derivative columns give infinite condition before normalization;
    rescaling numerical noise cannot identify the taus of a flat curve.
    Convergence does not imply parameter identification or a global optimum.
    """

    success: bool
    status: int
    message: str
    evaluations: int
    jacobian_evaluations: int | None
    optimality: float
    starts_attempted: int
    successful_starts: int
    best_start_index: int
    loading_condition_number: float
    jacobian_condition_number: float
    active_bounds: tuple[str, ...]
    failed_start_messages: tuple[str, ...]


@dataclass(frozen=True)
class NSSCalibrationResult:
    """Structured NSS fit result.

    Fitted yields and residuals are decimal annual rates. ``rmse`` and
    ``weighted_rmse`` are decimal annual rates as well. Residuals are defined
    as observed minus fitted.
    """

    parameters: NSSParameters
    maturities_years: np.ndarray
    observed_yields: np.ndarray
    fitted_yields: np.ndarray
    residuals: np.ndarray
    rmse: float
    weighted_rmse: float
    objective_value: float
    diagnostics: NSSCalibrationDiagnostics
    input_representation: CurveRepresentation

    @property
    def success(self) -> bool:
        """Whether SciPy reported convergence for the selected best start."""
        return self.diagnostics.success


@dataclass(frozen=True)
class DiscountedCashFlows:
    """Auditable spot-curve cash-flow valuation in currency units."""

    times_years: np.ndarray
    cash_flows: np.ndarray
    interpolated_zero_rates: np.ndarray
    discount_factors: np.ndarray
    present_values: np.ndarray
    total_present_value: float
    compounding: CompoundingConvention
    periodic_frequency: int | None


def validate_nss_parameters(
    parameters: NSSParameters | Sequence[float],
) -> NSSParameters:
    """Validate NSS betas (decimal rates) and positive taus (years).

    Args:
        parameters: An :class:`NSSParameters` instance or six values ordered
            ``beta0, beta1, beta2, beta3, tau1, tau2``.

    Returns:
        A validated :class:`NSSParameters` instance. Betas must be finite and
        both decay constants must be finite and strictly positive in years.
    """
    if isinstance(parameters, NSSParameters):
        candidate = parameters
    else:
        values = _as_finite_1d(parameters, "parameters")
        if values.size != 6:
            raise ValueError("NSS parameters must contain exactly six values")
        candidate = NSSParameters(*map(float, values))
    values = candidate.as_array()
    if not np.all(np.isfinite(values)):
        raise ValueError("NSS parameters must all be finite")
    if candidate.tau1 <= 0.0 or candidate.tau2 <= 0.0:
        raise ValueError("NSS tau1 and tau2 must be strictly positive years")
    return candidate


def nss_loadings(
    maturities_years: Sequence[float] | np.ndarray,
    tau1: float,
    tau2: float,
) -> np.ndarray:
    """Calculate standard NSS factor loadings at maturities in years.

    Returns an ``(n, 4)`` dimensionless matrix for level, slope, first
    curvature, and second curvature. At exactly zero maturity, the analytic
    limits are ``(1, 1, 0, 0)``. Stable Taylor expansions are used near zero.
    """
    maturities = _as_finite_1d(maturities_years, "maturities_years")
    if np.any(maturities < 0.0):
        raise ValueError("maturities_years must be non-negative")
    for value, name in ((tau1, "tau1"), (tau2, "tau2")):
        if isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite positive number of years")
        if float(value) <= 0.0:
            raise ValueError(f"{name} must be a finite positive number of years")
    x1 = maturities / float(tau1)
    x2 = maturities / float(tau2)
    slope1, curvature1 = _nss_loading_pair(x1)
    _, curvature2 = _nss_loading_pair(x2)
    return np.column_stack((np.ones_like(maturities), slope1, curvature1, curvature2))


def nss_yields(
    maturities_years: Sequence[float] | np.ndarray,
    parameters: NSSParameters | Sequence[float],
) -> np.ndarray:
    """Evaluate NSS decimal annual yields at maturities in years.

    The returned one-dimensional array has the same number of elements as the
    supplied maturity vector. The meaning of these fitted yields is inherited
    from the observations used for calibration; this function does not itself
    designate them as par yields or zero rates.
    """
    validated = validate_nss_parameters(parameters)
    loadings = nss_loadings(maturities_years, validated.tau1, validated.tau2)
    return loadings @ validated.as_array()[:4]


def nelson_siegel_svensson(
    maturities_years: Sequence[float] | np.ndarray,
    beta0: float,
    beta1: float,
    beta2: float,
    beta3: float,
    tau1: float,
    tau2: float,
) -> np.ndarray:
    """Evaluate standard NSS yields in decimal annual units.

    ``maturities_years``, ``tau1``, and ``tau2`` are years; beta coefficients
    and returned values are decimal annual yields (``0.04`` means 4%).
    """
    return nss_yields(
        maturities_years,
        NSSParameters(beta0, beta1, beta2, beta3, tau1, tau2),
    )


def nss_yield_jacobian(
    maturities_years: Sequence[float] | np.ndarray,
    parameters: NSSParameters | Sequence[float],
) -> np.ndarray:
    """Differentiate decimal NSS yields with respect to all six parameters.

    Maturities and taus are years; betas are decimal annual rates. Returns an
    (n, 6) matrix in beta0, beta1, beta2, beta3, tau1, tau2 order. Beta columns
    are dimensionless and tau columns are decimal annual yield per year of tau.
    Writing x=t/tau, s=(1-exp(-x))/x and c=s-exp(-x), the derivatives are
    ds/dtau=c/tau and dc/dtau=(c-x*exp(-x))/tau, with zero limits at t=0.
    """
    validated = validate_nss_parameters(parameters)
    maturities = _as_finite_1d(maturities_years, "maturities_years")
    loadings = nss_loadings(maturities, validated.tau1, validated.tau2)
    x1 = maturities / validated.tau1
    x2 = maturities / validated.tau2
    tau1_derivative = (
        validated.beta1 * loadings[:, 2]
        + validated.beta2 * (loadings[:, 2] - x1 * np.exp(-x1))
    ) / validated.tau1
    tau2_derivative = (
        validated.beta3 * (loadings[:, 3] - x2 * np.exp(-x2)) / validated.tau2
    )
    return np.column_stack((loadings, tau1_derivative, tau2_derivative))


def calibrate_nss(
    maturities_years: Sequence[float] | np.ndarray,
    observed_yields: Sequence[float] | np.ndarray,
    *,
    weights: Sequence[float] | np.ndarray | None = None,
    initial_guesses: Sequence[NSSParameters | Sequence[float]] | None = None,
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    input_representation: CurveRepresentation | str = CurveRepresentation.TREASURY_CMT,
    max_evaluations: int = 10_000,
) -> NSSCalibrationResult:
    """Calibrate NSS by deterministic bounded nonlinear least squares.

    Args:
        maturities_years: At least six unique, positive maturities in years.
        observed_yields: Finite decimal annual yields corresponding one-for-one
            with maturities. Published percentages must be converted before
            calling this function.
        weights: Optional strictly positive dimensionless least-squares weights.
            Relative weights are normalized internally; multiplying all weights
            by a common positive constant leaves the fit unchanged. Reported
            objective_value is half the sum of original weighted squared decimal
            residuals (not bp squared).
        initial_guesses: Optional NSS starts, with betas in decimal rates and
            taus in years. Defaults use several deterministic tau pairs and
            conditional linear estimates for the betas.
        bounds: Optional lower and upper six-vectors. Defaults are ``-0.5`` to
            ``0.5`` for decimal betas and ``0.02`` to ``30`` years for taus.
        input_representation: Meaning retained by the fitted observations.
            Treasury CMT, par-yield, and zero-rate inputs are accepted; no
            conversion between them is performed.
        max_evaluations: Positive per-start cap on model evaluations.

    Returns:
        Parameters, decimal fitted yields and residuals, decimal RMSE, weighted
        objective, and optimizer diagnostics. The lowest-cost finite result is
        selected across converged starts, or across finite starts if none converge.

    Notes:
        SciPy performs the numerical optimization, but the NSS loadings and
        residual function and analytical Jacobian are implemented explicitly here.
        Fixed optimizer scales are 0.05 decimal for betas and 2/5 years for taus;
        these affect search steps, not parameter bounds or reported units.
        Parameter recovery
        can be weak even with low curve RMSE because NSS decay terms may be
        poorly identified by a small maturity cross-section.
    """
    maturities = _as_finite_1d(maturities_years, "maturities_years")
    yields = _as_finite_1d(observed_yields, "observed_yields")
    if maturities.size != yields.size:
        raise ValueError("maturities_years and observed_yields must have equal length")
    if maturities.size < 6:
        raise ValueError("NSS calibration requires at least six observations")
    if np.any(maturities <= 0.0):
        raise ValueError("calibration maturities_years must be strictly positive")
    if np.unique(maturities).size != maturities.size:
        raise ValueError("calibration maturities_years must be unique")
    order = np.argsort(maturities)
    maturities = maturities[order]
    yields = yields[order]
    if np.any(np.abs(yields) > 1.0):
        raise ValueError(
            "observed_yields appear outside decimal-rate bounds; use 0.04 for 4%"
        )
    if weights is None:
        fit_weights = np.ones_like(yields)
    else:
        fit_weights = _as_finite_1d(weights, "weights")
        if fit_weights.size != yields.size:
            raise ValueError("weights and observed_yields must have equal length")
        fit_weights = fit_weights[order]
        if np.any(fit_weights <= 0.0):
            raise ValueError("weights must be strictly positive")
    if isinstance(max_evaluations, bool) or not isinstance(max_evaluations, int):
        raise TypeError("max_evaluations must be an integer")
    if max_evaluations <= 0:
        raise ValueError("max_evaluations must be positive")

    representation = _validate_fit_representation(input_representation)
    lower, upper = _validate_nss_bounds(bounds)
    normalized_weights = fit_weights / fit_weights.max()
    normalized_weights /= normalized_weights.mean()
    starts = _prepare_nss_starts(
        maturities, yields, normalized_weights, initial_guesses, lower, upper
    )
    sqrt_weights = np.sqrt(normalized_weights)

    def weighted_residuals(vector: np.ndarray) -> np.ndarray:
        params = NSSParameters(*map(float, vector))
        # Solve in bp so the gradient tolerance is meaningful for yield errors.
        return sqrt_weights * (nss_yields(maturities, params) - yields) / 0.0001

    def weighted_jacobian(vector: np.ndarray) -> np.ndarray:
        return sqrt_weights[:, None] * nss_yield_jacobian(maturities, vector) / 0.0001

    candidates = []
    failed_start_messages: list[str] = []
    for start_index, start in enumerate(starts):
        try:
            candidate = least_squares(
                weighted_residuals,
                start,
                jac=weighted_jacobian,
                bounds=(lower, upper),
                method="trf",
                # Fixed economic scales avoid exploding steps in weakly
                # identified tau directions when beta curvature terms vanish.
                x_scale=np.array([0.05, 0.05, 0.05, 0.05, 2.0, 5.0]),
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
                max_nfev=max_evaluations,
            )
        except (FloatingPointError, ValueError) as exc:
            failed_start_messages.append(f"start {start_index}: {exc}")
            continue
        if np.all(np.isfinite(candidate.x)) and math.isfinite(float(candidate.cost)):
            candidates.append((start_index, candidate))
    if not candidates:
        raise RuntimeError(
            "NSS calibration failed to produce a finite optimizer result"
        )
    converged = [item for item in candidates if item[1].success]
    best_index, best = min(
        converged or candidates, key=lambda item: float(item[1].cost)
    )
    parameters = validate_nss_parameters(best.x)
    fitted = nss_yields(maturities, parameters)
    residuals = yields - fitted
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    weighted_rmse = float(np.sqrt(np.mean(normalized_weights * np.square(residuals))))
    jacobian_norms = np.linalg.norm(best.jac, axis=0)
    negligible_column = np.finfo(float).eps * float(jacobian_norms.max()) * 100.0
    jacobian_condition = (
        float(np.linalg.cond(best.jac / jacobian_norms))
        if np.all(jacobian_norms > negligible_column)
        else math.inf
    )
    names = ("beta0", "beta1", "beta2", "beta3", "tau1", "tau2")
    at_bound = (best.x - lower <= (upper - lower) * 1e-6) | (
        upper - best.x <= (upper - lower) * 1e-6
    )
    diagnostics = NSSCalibrationDiagnostics(
        success=bool(best.success),
        status=int(best.status),
        message=str(best.message),
        evaluations=int(best.nfev),
        jacobian_evaluations=None if best.njev is None else int(best.njev),
        optimality=float(best.optimality),
        starts_attempted=len(starts),
        successful_starts=sum(bool(result.success) for _, result in candidates),
        best_start_index=best_index,
        loading_condition_number=float(
            np.linalg.cond(nss_loadings(maturities, parameters.tau1, parameters.tau2))
        ),
        jacobian_condition_number=jacobian_condition,
        active_bounds=tuple(
            name for name, active in zip(names, at_bound, strict=True) if active
        ),
        failed_start_messages=tuple(failed_start_messages),
    )
    return NSSCalibrationResult(
        parameters=parameters,
        maturities_years=maturities.copy(),
        observed_yields=yields.copy(),
        fitted_yields=fitted,
        residuals=residuals,
        rmse=rmse,
        weighted_rmse=weighted_rmse,
        objective_value=float(0.5 * np.sum(fit_weights * residuals**2)),
        diagnostics=diagnostics,
        input_representation=representation,
    )


def interpolate_zero_rates(
    node_maturities_years: Sequence[float] | np.ndarray,
    node_zero_rates: Sequence[float] | np.ndarray,
    target_maturities_years: Sequence[float] | np.ndarray,
    *,
    extrapolation: ExtrapolationPolicy | str = ExtrapolationPolicy.RAISE,
) -> np.ndarray:
    """Linearly interpolate decimal annual zero rates across maturity in years.

    The inputs must already be zero rates; observed CMT or par yields must not
    be passed without a separately documented conversion or bootstrap. With
    ``extrapolation='flat'``, the nearest endpoint zero rate is used outside
    the node range. The default rejects extrapolation at positive times.
    At time zero the first node rate is a reporting convention only: every
    compounding convention gives D(0)=1, so no zero-time rate is inferred.
    A single node is valid at that node or under explicit flat extrapolation.
    """
    nodes = _as_finite_1d(node_maturities_years, "node_maturities_years")
    rates = _as_finite_1d(node_zero_rates, "node_zero_rates")
    targets = _as_finite_1d(target_maturities_years, "target_maturities_years")
    if nodes.size != rates.size:
        raise ValueError("node maturities and zero rates must have equal length")
    if nodes.size < 1:
        raise ValueError("at least one zero-rate node is required")
    if np.any(nodes <= 0.0) or np.any(np.diff(nodes) <= 0.0):
        raise ValueError("zero-rate node maturities must be positive and increasing")
    if np.any(targets < 0.0):
        raise ValueError("target maturities must be non-negative years")
    if np.any(np.abs(rates) > 1.0):
        raise ValueError("zero rates must be decimal annual rates; use 0.04 for 4%")
    try:
        policy = ExtrapolationPolicy(extrapolation)
    except ValueError as exc:
        raise ValueError(
            f"unsupported extrapolation policy: {extrapolation!r}"
        ) from exc
    outside = (targets != 0.0) & ((targets < nodes[0]) | (targets > nodes[-1]))
    if policy is ExtrapolationPolicy.RAISE and np.any(outside):
        raise ValueError("target maturity lies outside the zero-rate node range")
    return np.interp(targets, nodes, rates)


def zero_rates_to_discount_factors(
    maturities_years: Sequence[float] | np.ndarray,
    zero_rates: Sequence[float] | np.ndarray,
    *,
    compounding: CompoundingConvention | str,
    periodic_frequency: int | None = None,
) -> np.ndarray:
    """Convert decimal annual zero rates to dimensionless discount factors.

    For maturity ``t`` years and zero rate ``z``, continuous compounding uses
    ``exp(-z*t)``, simple compounding uses ``1/(1+z*t)``, and periodic
    compounding at ``m`` periods/year uses ``(1+z/m)**(-m*t)``. A maturity of
    zero always has discount factor one.
    """
    maturities = _as_finite_1d(maturities_years, "maturities_years")
    rates = _as_finite_1d(zero_rates, "zero_rates")
    if maturities.size != rates.size:
        raise ValueError("maturities_years and zero_rates must have equal length")
    if np.any(maturities < 0.0):
        raise ValueError("maturities_years must be non-negative")
    if np.any(np.abs(rates) > 1.0):
        raise ValueError("zero_rates must be decimal annual rates; use 0.04 for 4%")
    convention = _validate_compounding(compounding, periodic_frequency)
    if convention is CompoundingConvention.CONTINUOUS:
        factors = np.exp(-rates * maturities)
    elif convention is CompoundingConvention.SIMPLE:
        bases = 1.0 + rates * maturities
        if np.any(bases <= 0.0):
            raise ValueError("simple-compounding discount base must be positive")
        factors = 1.0 / bases
    else:
        assert periodic_frequency is not None
        bases = 1.0 + rates / periodic_frequency
        if np.any(bases <= 0.0):
            raise ValueError("periodic-compounding discount base must be positive")
        factors = bases ** (-periodic_frequency * maturities)
    if not np.all(np.isfinite(factors)) or np.any(factors <= 0.0):
        raise ArithmeticError("discount-factor calculation produced invalid values")
    return factors


def discount_cash_flows(
    times_years: Sequence[float] | np.ndarray,
    cash_flows: Sequence[float] | np.ndarray,
    zero_curve: YieldCurve,
    *,
    extrapolation: ExtrapolationPolicy | str = ExtrapolationPolicy.RAISE,
) -> DiscountedCashFlows:
    """Discount currency cash flows with an explicitly tagged zero-rate curve.

    Args:
        times_years: Non-negative payment times in years from the curve date.
        cash_flows: Currency amounts paid at the corresponding times.
        zero_curve: Curve explicitly represented as decimal annual zero rates.
        extrapolation: Node-range policy used during zero-rate interpolation.

    Returns:
        Interpolated decimal zero rates, dimensionless discount factors,
        currency present values, and their currency total. This is spot-curve
        discounting and does not use a bond's conventional YTM.
    """
    times = _as_finite_1d(times_years, "times_years")
    amounts = _as_finite_1d(cash_flows, "cash_flows")
    if times.size != amounts.size:
        raise ValueError("times_years and cash_flows must have equal length")
    if np.any(times < 0.0):
        raise ValueError("cash-flow times must be non-negative years")
    factors = zero_curve.discount_factors(times, extrapolation=extrapolation)
    rates = interpolate_zero_rates(
        zero_curve.maturities_years,
        zero_curve.values,
        times,
        extrapolation=extrapolation,
    )
    present_values = amounts * factors
    return DiscountedCashFlows(
        times_years=times.copy(),
        cash_flows=amounts.copy(),
        interpolated_zero_rates=rates,
        discount_factors=factors,
        present_values=present_values,
        total_present_value=float(math.fsum(map(float, present_values))),
        compounding=zero_curve.compounding,  # type: ignore[arg-type]
        periodic_frequency=zero_curve.periodic_frequency,
    )


def _nss_loading_pair(scaled_maturity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    small = np.abs(scaled_maturity) < 1e-5
    slope = np.empty_like(scaled_maturity)
    curvature = np.empty_like(scaled_maturity)
    x = scaled_maturity[small]
    slope[small] = 1.0 - x / 2.0 + x**2 / 6.0 - x**3 / 24.0 + x**4 / 120.0
    curvature[small] = x / 2.0 - x**2 / 3.0 + x**3 / 8.0 - x**4 / 30.0
    x = scaled_maturity[~small]
    slope[~small] = -np.expm1(-x) / x
    curvature[~small] = slope[~small] - np.exp(-x)
    return slope, curvature


def _as_finite_1d(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a one-dimensional numeric sequence") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_compounding(
    compounding: CompoundingConvention | str | None,
    periodic_frequency: int | None,
) -> CompoundingConvention:
    try:
        convention = CompoundingConvention(compounding)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"unsupported compounding convention: {compounding!r}"
        ) from exc
    if convention is CompoundingConvention.PERIODIC:
        if (
            isinstance(periodic_frequency, bool)
            or not isinstance(periodic_frequency, int)
            or periodic_frequency <= 0
        ):
            raise ValueError(
                "periodic_frequency must be a positive integer periods/year"
            )
    elif periodic_frequency is not None:
        raise ValueError("periodic_frequency is only valid for periodic compounding")
    return convention


def _validate_fit_representation(
    representation: CurveRepresentation | str,
) -> CurveRepresentation:
    try:
        normalized = CurveRepresentation(representation)
    except ValueError as exc:
        raise ValueError(
            f"unsupported input representation: {representation!r}"
        ) from exc
    allowed = {
        CurveRepresentation.TREASURY_CMT,
        CurveRepresentation.PAR_YIELD,
        CurveRepresentation.ZERO_RATE,
    }
    if normalized not in allowed:
        raise ValueError(
            "input_representation must identify observed CMT, par, or zero rates"
        )
    return normalized


def _validate_nss_bounds(
    bounds: tuple[Sequence[float], Sequence[float]] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if bounds is None:
        lower = np.array([-0.5, -0.5, -0.5, -0.5, 0.02, 0.02])
        upper = np.array([0.5, 0.5, 0.5, 0.5, 30.0, 30.0])
    else:
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            raise TypeError("bounds must be a (lower, upper) tuple")
        lower = _as_finite_1d(bounds[0], "lower bounds")
        upper = _as_finite_1d(bounds[1], "upper bounds")
        if lower.size != 6 or upper.size != 6:
            raise ValueError("NSS lower and upper bounds must each contain six values")
    if np.any(lower >= upper):
        raise ValueError("every NSS lower bound must be below its upper bound")
    if lower[4] <= 0.0 or lower[5] <= 0.0:
        raise ValueError("NSS tau lower bounds must be strictly positive years")
    return lower, upper


def _prepare_nss_starts(
    maturities: np.ndarray,
    yields: np.ndarray,
    weights: np.ndarray,
    initial_guesses: Sequence[NSSParameters | Sequence[float]] | None,
    lower: np.ndarray,
    upper: np.ndarray,
) -> list[np.ndarray]:
    if initial_guesses is not None:
        if len(initial_guesses) == 0:
            raise ValueError("initial_guesses must not be empty")
        starts = [
            validate_nss_parameters(guess).as_array() for guess in initial_guesses
        ]
    else:
        tau_pairs = (
            (0.25, 1.0),
            (0.5, 2.0),
            (1.0, 4.0),
            (1.5, 7.0),
            (2.5, 12.0),
            (5.0, 20.0),
            (1.0, 10.0),
            (4.0, 1.0),
        )
        starts = []
        weighted_sqrt = np.sqrt(weights)
        for tau1, tau2 in tau_pairs:
            design = nss_loadings(maturities, tau1, tau2)
            betas, *_ = np.linalg.lstsq(
                design * weighted_sqrt[:, None], yields * weighted_sqrt, rcond=None
            )
            starts.append(np.r_[betas, tau1, tau2])
    validated_starts = []
    interior_margin = (upper - lower) * 1e-8
    for start in starts:
        if start.size != 6 or not np.all(np.isfinite(start)):
            raise ValueError("every NSS initial guess must contain six finite values")
        validated_starts.append(
            np.clip(start, lower + interior_margin, upper - interior_margin)
        )
    return validated_starts
