# Fixed-Income Yield Curve & Relative-Value Engine: Project Specification

## 1. Scope and design principles

This project will implement a transparent research engine for fixed-rate bond
analytics, curve modelling, interest-rate risk, and historical relative-value
analysis. Important formulas and numerical methods will be implemented in the
project's own Python code using NumPy, pandas, and SciPy. QuantLib will not be
used to supply the core calculations.

The first instrument scope is nominal, fixed-rate, bullet bonds. Callable,
putable, inflation-linked, floating-rate, defaultable, and mortgage instruments
are outside the first release. Calendar rules, settlement lags, day-count
conventions, coupon frequencies, compounding conventions, interpolation
choices, and data transformations must be visible in APIs and output metadata.

Internal rates and yields are decimals: `0.0425` means 4.25%. One basis point
is exactly `0.0001` in decimal yield terms. Currency prices will normally be
quoted per 100 units of face value, but functions must accept face value
explicitly and document whether they return a quoted price or a currency
amount. Time is measured in years unless a function says otherwise.

Constant-maturity Treasury yields are par-yield-like statistical series. The
engine will retain their identity and will not label them zero rates or discount
factors. Any transformation from those observations to a fitted or bootstrapped
curve must state its assumptions.

## 2. Package responsibilities

- `fixed_income.bonds`: schedules, cash flows, accrued interest, quoted prices,
  yield solving, duration, convexity, and instrument-level DV01.
- `fixed_income.curves`: curve data structures, interpolation, discount factors,
  zero-curve valuation, and Nelson-Siegel-Svensson fitting.
- `fixed_income.data`: schema validation, online adapters, and a bundled offline
  data path with identical output schemas.
- `fixed_income.risk`: parallel and shaped shocks, repricing, and key-rate DV01.
- `fixed_income.pca`: historical yield changes, PCA estimation, transformations,
  and factor interpretation.
- `fixed_income.relative_value`: fitted residuals, rolling normalization,
  butterfly measures, and DV01-neutral weights.
- `fixed_income.backtest`: dated signals, lagged positions, P&L, turnover,
  transaction costs, and performance summaries.
- `fixed_income.plotting`: deterministic figures saved below `outputs/`.
- `fixed_income_engine.py`: a thin CLI that validates arguments and delegates to
  package workflows.

## 3. Analytical components

### A. Fixed-rate bond cash flows

- **Objective:** Generate an auditable payment schedule and contractual cash
  flows for a nominal fixed-rate bullet bond.
- **Mathematical definition:** For face value \(F\), annual coupon rate \(c\),
  and \(m\) regular payments per year, a regular coupon is \(C=Fc/m\). Each
  coupon date pays \(C\), and maturity pays \(C+F\). Stub coupons, when
  supported, use an explicitly selected accrual fraction rather than this
  regular-period shortcut.
- **Units:** \(c\) is a decimal annual rate; \(F\), \(C\), and cash flows are
  currency amounts; dates are calendar dates; frequency is payments per year.
- **Inputs:** Issue or accrual-start date, maturity date, coupon rate, face
  value, frequency, calendar, business-day rule, and day-count convention.
- **Outputs:** An ordered table of unadjusted dates, payment dates, accrual
  boundaries, accrual fractions, coupon cash flows, principal cash flows, and
  total cash flows.
- **Assumptions:** The initial implementation supports bullet principal and
  regular semi-annual or annual coupons. Any end-of-month and holiday handling
  will be stated explicitly.
- **Likely numerical issues:** Date-generation drift, leap years, end-of-month
  dates, duplicate adjusted dates, odd first or last coupons, and floating-point
  residue at maturity.
- **Tests:** Known hand-built schedules; leap-year and month-end cases; exact
  coupon totals; one principal repayment; monotonic payment dates; rejection of
  invalid maturities, frequencies, and rates.

### B. Accrued interest

- **Objective:** Measure coupon earned between the previous coupon date and the
  settlement date without conflating it with market value.
- **Mathematical definition:** For a regular coupon \(C\), accrued interest is
  \(AI=C\,a\), where \(a\) is the convention-specific fraction of the current
  coupon period elapsed. The exact numerator and denominator depend on the
  chosen day-count convention.
- **Units:** Accrued interest is currency, or price points per 100 face when
  explicitly requested; \(a\) is dimensionless.
- **Inputs:** Settlement date, surrounding coupon dates, coupon amount,
  day-count convention, and any ex-coupon rule.
- **Outputs:** Accrued interest plus the dates, day counts, and fraction used.
- **Assumptions:** Settlement is handled separately from trade date. Initial
  support will document behavior on coupon dates and will reject unsupported
  ex-coupon periods.
- **Likely numerical issues:** Boundary inclusivity, negative accrual after a
  schedule error, leap-day treatment, and settlement exactly at maturity.
- **Tests:** Zero accrued interest on defined coupon boundaries; known Treasury
  examples; monotonic accrual within a period; convention-specific leap-year
  cases; clean-plus-accrued reconciliation.

### C. Clean and dirty pricing

- **Objective:** Keep quoted clean price, full dirty price, and accrued interest
  distinct and exactly reconcilable.
- **Mathematical definition:** \(P_{dirty}=\sum_i CF_iD(t_i)\), where \(D(t_i)\)
  comes from either a stated YTM convention or a spot curve. Then
  \(P_{clean}=P_{dirty}-AI\).
- **Units:** Prices are currency amounts or points per 100 face as explicitly
  selected; discount factors are dimensionless; time is years.
- **Inputs:** Future cash flows, settlement date, accrued interest, and exactly
  one valuation method: YTM or spot-curve discount factors.
- **Outputs:** A structured result carrying clean price, dirty price, accrued
  interest, valuation method, and convention metadata.
- **Assumptions:** Settlement-date cash-flow entitlement is defined by the
  schedule convention. YTM pricing and curve pricing use separate entry points.
- **Likely numerical issues:** Near-zero time to a cash flow, scaling by face
  value, settlement on coupon dates, and accidental double subtraction of
  accrued interest.
- **Tests:** `clean + accrued == dirty`; zero-coupon special case; manual
  discounted-cash-flow examples; equivalence when a flat spot curve matches the
  stated compounding assumptions.

### D. Yield-to-maturity solving

- **Objective:** Infer the single conventional yield that reproduces a dirty
  price for a specified cash-flow schedule.
- **Mathematical definition:** Solve
  \(f(y)=\sum_i CF_i(1+y/m)^{-m t_i}-P_{dirty}=0\) for nominal annual YTM \(y\)
  with payment frequency \(m\), using schedule-consistent fractional periods.
- **Units:** YTM is a decimal annual rate; price and cash flows share currency
  units; time is years.
- **Inputs:** Dirty price, dated cash flows, settlement date, compounding
  frequency, tolerance, iteration cap, and a valid search bracket.
- **Outputs:** YTM, convergence status, residual, iterations, and solver method.
- **Assumptions:** Cash flows are positive for the initial bullet-bond scope, so
  price is monotone in yield on the admissible domain \(y>-m\).
- **Likely numerical issues:** Invalid negative-yield brackets, very short
  maturities, extreme premiums or discounts, flat derivatives, and ambiguous
  roots for later non-standard cash flows.
- **Tests:** Recover yields from prices generated at known positive, zero, and
  negative yields; compare bracketing and Newton-style results; verify failure
  diagnostics for impossible prices and unbracketed roots.

### E. Macaulay duration

- **Objective:** Measure the present-value-weighted average time to receipt of a
  bond's YTM-discounted cash flows.
- **Mathematical definition:**
  \(D_{Mac}=\sum_i t_iPV(CF_i)/P_{dirty}\).
- **Units:** Years.
- **Inputs:** Dated cash flows, settlement date, YTM, frequency, and dirty price
  or enough information to calculate it.
- **Outputs:** Macaulay duration and the cash-flow weights used.
- **Assumptions:** This is a YTM-based statistic, not a spot-curve sensitivity.
  Compounding matches the price calculation.
- **Likely numerical issues:** Price near zero, inconsistent time fractions,
  stale accrued-interest treatment, and loss of precision at extreme yields.
- **Tests:** Zero-coupon duration equals time to maturity; weighted cash-flow
  weights sum to one; hand calculations; comparison to a derivative-based
  duration only under matching flat-yield assumptions.

### F. Modified duration

- **Objective:** Approximate the proportional dirty-price sensitivity to a small
  change in conventional YTM.
- **Mathematical definition:** For nominal yield compounded \(m\) times a year,
  \(D_{mod}=D_{Mac}/(1+y/m)\), and \(\Delta P/P\approx-D_{mod}\Delta y\).
- **Units:** Years with respect to a decimal annual yield; the product with a
  decimal yield change is dimensionless.
- **Inputs:** Macaulay duration, decimal YTM, and compounding frequency.
- **Outputs:** Modified duration and optional first-order price-change estimate.
- **Assumptions:** A small parallel change in the bond's YTM and unchanged cash
  flows; this does not claim key-rate or spot-curve sensitivity.
- **Likely numerical issues:** Denominator near zero and misuse with yields in
  percentage points or basis points.
- **Tests:** Analytic result versus central finite differences of dirty price;
  zero-coupon cases; explicit 1 bp perturbation using `0.0001`.

### G. Convexity

- **Objective:** Capture the second-order curvature of dirty price with respect
  to YTM.
- **Mathematical definition:** For cash flow \(CF_k\) at coupon-period index
  \(k\), conventional convexity is
  \(C=(1/P)\sum_k CF_k k(k+1)m^{-2}(1+y/m)^{-(k+2)}\). Dated fractional-period
  generalization must be derived and documented before implementation.
- **Units:** Years squared with respect to decimal annual yield.
- **Inputs:** Cash flows, payment times, YTM, compounding frequency, and dirty
  price.
- **Outputs:** Convexity and the second-order estimate
  \(\Delta P/P\approx-D_{mod}\Delta y+\tfrac12C(\Delta y)^2\).
- **Assumptions:** Fixed cash flows and the same YTM convention as bond pricing.
- **Likely numerical issues:** Formula mismatch for fractional periods,
  cancellation in finite differences, and scaling mistakes between bps and
  decimals.
- **Tests:** Analytic convexity versus a central second difference over several
  bump sizes; zero-coupon formula; second-order approximation improves on the
  duration-only estimate for moderate shocks.

### H. DV01

- **Objective:** Express the currency price change for a one-basis-point yield
  increase using a clear sign convention.
- **Mathematical definition:** Report positive risk magnitude
  \(DV01=-\partial P/\partial y\times0.0001\) for an ordinary long bond. A
  central check is \([P(y-0.0001)-P(y+0.0001)]/2\).
- **Units:** Currency per basis point, or price points per 100 face per basis
  point when explicitly requested.
- **Inputs:** Bond, YTM or curve, valuation date, face/notional, bump size, and
  pricing convention.
- **Outputs:** DV01, signed price changes for up and down bumps, and method
  metadata.
- **Assumptions:** Cash flows do not change under the bump. YTM DV01 and
  curve-based DV01 are separate measures.
- **Likely numerical issues:** Sign errors, face-value scaling, bump expressed as
  `1` instead of `0.0001`, and one-sided approximation bias.
- **Tests:** Modified-duration approximation against repricing; symmetry of
  small up/down moves; linear scaling with notional; positive DV01 for standard
  long fixed-rate bonds.

### I. Spot/zero-curve discounting

- **Objective:** Value each cash flow using a maturity-specific discount factor
  rather than a single YTM.
- **Mathematical definition:** \(P=\sum_i CF_iD(t_i)\). For continuously
  compounded zero rate \(z(t)\), \(D(t)=e^{-z(t)t}\); other compounding choices
  require their own explicit formula.
- **Units:** Zero rates are decimal annual rates, discount factors are
  dimensionless, times are years, and present values are currency.
- **Inputs:** Curve valuation date, nodes, node representation, interpolation
  rule, compounding convention, and dated cash flows.
- **Outputs:** Discount factors and cash-flow present values with a curve-pricing
  total.
- **Assumptions:** Initial interpolation should operate on log discount factors
  where valid. Extrapolation policy is explicit and conservative.
- **Likely numerical issues:** Non-positive discount factors, duplicated nodes,
  unstable extrapolation, inconsistent day counts, and interpolation that
  creates implausible forward rates.
- **Tests:** Exact recovery at nodes; monotone discount factors for positive
  curves; flat-curve identities; analytic continuous-compounding examples;
  rejection of invalid curves.

### J. Yield-curve representations

- **Objective:** Prevent accidental interchange of observed yields, par yields,
  zero rates, forward rates, and discount factors.
- **Mathematical definition:** A curve is a valuation date plus maturities and
  values with a tagged representation. Transformations obey
  \(D(t)=e^{-z(t)t}\) for continuous zeros and
  \(f(t_1,t_2)=-\log[D(t_2)/D(t_1)]/(t_2-t_1)\) for continuously compounded
  forwards.
- **Units:** Maturity is years or dated; rates are decimals per annum; discount
  factors are dimensionless.
- **Inputs:** Nodes, representation tag, compounding, day count, interpolation,
  and source metadata.
- **Outputs:** Validated immutable curve objects and explicit conversion results.
- **Assumptions:** Conversion from par instruments to zeros requires instrument
  cash flows and is not inferred from labels alone.
- **Likely numerical issues:** Mixed maturities, hidden percent inputs,
  inconsistent compounding, duplicate tenors, and unsafe extrapolation.
- **Tests:** Round-trip zero/discount transformations; schema validation;
  representation-specific type or runtime guards; no implicit conversion of
  constant-maturity observations.

### K. Nelson-Siegel-Svensson

- **Objective:** Fit a smooth, interpretable cross-sectional curve to maturity
  and yield observations.
- **Mathematical definition:** With \(x_1=(1-e^{-t/\tau_1})/(t/\tau_1)\),
  \(x_2=x_1-e^{-t/\tau_1}\), and
  \(x_3=(1-e^{-t/\tau_2})/(t/\tau_2)-e^{-t/\tau_2}\), fit
  \(y(t)=\beta_0+\beta_1x_1+\beta_2x_2+\beta_3x_3\).
- **Units:** Fitted yields and beta coefficients are decimal annual rates;
  \(t,\tau_1,\tau_2\) are years.
- **Inputs:** Positive maturities, observed yields, optional weights, parameter
  bounds, initial guesses, and optimizer controls.
- **Outputs:** Parameters, fitted yields, residuals, objective value, convergence
  diagnostics, and fit metadata.
- **Assumptions:** The fitted quantity retains the input series' meaning. Fitting
  Treasury constant-maturity yields produces a smooth fitted yield curve, not
  an automatically bootstrapped zero curve.
- **Likely numerical issues:** Near-zero maturity limits, local minima,
  non-identifiability of decay parameters, parameter swapping, poor scaling,
  and sensitivity to starting values.
- **Tests:** Stable factor-load limits at zero; recover synthetic parameters
  within tolerance; deterministic multi-start results; bounded positive taus;
  graceful diagnostics on sparse or degenerate inputs.

### L. Historical Treasury yield data

- **Objective:** Provide reproducible historical maturity panels for empirical
  curve analysis, with online retrieval and an offline sample fallback.
- **Mathematical definition:** Store a date-by-tenor matrix \(Y_{t,m}\) of
  published constant-maturity annual yields, converted once from percent values
  to internal decimal rates.
- **Units:** Stored yields are decimal annual rates; source observations may be
  percent and must carry conversion metadata; tenor is years.
- **Inputs:** Date range, selected maturities, data source, cache location, and
  offline flag.
- **Outputs:** Sorted validated panel plus source, retrieval date, raw units,
  missing-data flags, and transformation metadata.
- **Assumptions:** Publication calendars and missing values are retained before
  an explicit alignment policy. These observations are not zero-coupon rates.
- **Likely numerical issues:** Source revisions, schema changes, missing tenors,
  holiday gaps, percent/decimal errors, and silent interpolation across missing
  dates.
- **Tests:** Offline fixtures only; known unit conversions; unique sorted dates;
  schema parity between online and offline loaders; deterministic missing-data
  handling; mocked network failure falls back as documented.

### M. Parallel curve shocks

- **Objective:** Reprice instruments and portfolios after the same yield or zero
  rate move at every curve node.
- **Mathematical definition:** \(y^{shock}(t)=y(t)+s\), where shock \(s\) is a
  decimal rate change such as `0.0001` for +1 bp, followed by full repricing.
- **Units:** Shock inputs are decimals and reports also label their bp
  equivalent; value changes are currency.
- **Inputs:** Tagged base curve, shock size and direction, instruments,
  notionals, and valuation conventions.
- **Outputs:** Shocked curve, base and shocked values, absolute P&L, percentage
  changes, and portfolio aggregation.
- **Assumptions:** Cash flows and spreads remain fixed unless a scenario says
  otherwise. The exact curve representation being shocked is reported.
- **Likely numerical issues:** Bumping the wrong representation, double unit
  conversion, invalid discount factors after extreme shocks, and aggregation
  sign errors.
- **Tests:** Every node moves by exactly the requested shock; +1 bp finite
  difference reconciles with curve DV01 locally; zero shock gives zero P&L;
  portfolio P&L equals instrument sum.

### N. Steepener/flattener scenarios

- **Objective:** Examine non-parallel exposure to changes in short-versus-long
  rates through reproducible shaped shocks.
- **Mathematical definition:** Define shock values at named maturity anchors and
  interpolate them: \(y^{shock}(t)=y(t)+s(t)\). A steepener lowers short rates
  relative to long rates; a flattener does the reverse, with orientation and
  anchor changes printed explicitly.
- **Units:** Node shocks are decimal rate changes and displayed in bp; value
  changes are currency.
- **Inputs:** Base curve, scenario name or explicit anchor shocks, interpolation
  rule, instruments, and notionals.
- **Outputs:** Shock vector, shocked curve, instrument P&L, portfolio P&L, and
  maturity-bucket attribution.
- **Assumptions:** Named scenarios have versioned definitions; no universal
  market convention is implied by the words steepener and flattener.
- **Likely numerical issues:** Sign ambiguity, discontinuities at anchors,
  unintended overall level shift, and extrapolation outside the anchors.
- **Tests:** Exact anchor shocks; expected slope-direction change; mirrored
  scenarios give approximately opposite first-order P&L; explicit expected
  behavior beyond the outer anchors.

### O. Key-rate DV01

- **Objective:** Allocate price sensitivity across selected maturity points by
  localized curve perturbations.
- **Mathematical definition:** For key rate \(k\), apply a piecewise-linear hat
  bump \(b_k(t)\) peaking at `0.0001` at key \(k\) and zero at adjacent keys,
  then report \(KRDV01_k=[P(y-b_k)-P(y+b_k)]/2\).
- **Units:** Currency per basis point at each key rate; bumps are decimals.
- **Inputs:** Curve, key maturities, bump construction, bond or portfolio,
  notionals, and pricing conventions.
- **Outputs:** Key-rate vector, bump functions, up/down revaluations, and total
  sensitivity.
- **Assumptions:** Key placement and interpolation determine the allocation.
  Cash flows remain fixed under shocks.
- **Likely numerical issues:** Gaps or overlaps in bump functions, interpolation
  dependence, edge-key behavior, bump-size sensitivity, and sign consistency.
- **Tests:** Central finite-difference construction; sum of key-rate DV01s is
  close to parallel DV01 when hat bumps partition a parallel move; linear
  notional scaling; localized synthetic exposure peaks at the expected key.

### P. Historical yield-change PCA

- **Objective:** Estimate dominant empirical modes of co-movement across curve
  maturities.
- **Mathematical definition:** From aligned yields, form
  \(\Delta Y_t=Y_t-Y_{t-1}\), center using training-sample means, estimate the
  covariance matrix, and eigendecompose it. Sort eigenpairs by descending
  eigenvalue and compute scores by projection.
- **Units:** Yield changes are stored as decimal changes; loadings are
  dimensionless; scores are decimal yield changes; eigenvalues are squared
  decimal changes. Reports may convert changes to bp explicitly.
- **Inputs:** Historical yield panel, maturity set, date window, missing-data
  policy, covariance or correlation choice, and fit window.
- **Outputs:** Means, covariance matrix, eigenvalues, explained-variance ratios,
  normalized loadings, scores, and fitted date range.
- **Assumptions:** Standard covariance PCA is the default. The model is fitted
  only on observations available by the relevant evaluation date.
- **Likely numerical issues:** Missing data, eigenvector sign indeterminacy,
  nearly repeated eigenvalues, changing maturity sets, and numerical negative
  eigenvalues close to zero.
- **Tests:** Recover seeded synthetic factors; orthonormal loadings; descending
  non-negative eigenvalues within tolerance; explained variance sums to one;
  inverse transform reconstructs retained components; no future observations in
  rolling fits.

### Q. Level/slope/curvature interpretation

- **Objective:** Attach transparent descriptive labels to leading PCA loadings
  without claiming those labels are guaranteed.
- **Mathematical definition:** Orient PC1 so its average loading is positive;
  orient PC2 by a documented long-minus-short contrast; orient PC3 by a
  documented belly-versus-wings contrast. Label components from quantitative
  shape diagnostics and retain the raw component number.
- **Units:** Loadings and shape scores are dimensionless; factor scores retain
  decimal yield-change units.
- **Inputs:** Ordered maturities, PCA loadings, sign-normalization rules, and
  diagnostic thresholds.
- **Outputs:** Oriented loadings, factor labels, shape diagnostics, and plots.
- **Assumptions:** Level, slope, and curvature are interpretations of a sample,
  not structural identities. Labels may be withheld when shapes are ambiguous.
- **Likely numerical issues:** Arbitrary eigenvector signs, unstable ordering of
  close eigenvalues, irregular tenor spacing, and overconfident auto-labelling.
- **Tests:** Sign normalization is deterministic; synthetic level/slope/curvature
  shapes receive expected labels; ambiguous shapes remain explicitly unlabeled;
  factor reconstruction is unchanged by sign orientation.

### R. Fitted-curve residuals

- **Objective:** Measure how far each observed maturity lies from a stated
  cross-sectional fitted curve.
- **Mathematical definition:** \(r_{t,m}=y^{obs}_{t,m}-y^{fit}_{t,m}\). A positive
  yield residual indicates the observation is high in yield relative to the
  fit; it does not alone establish mispricing or arbitrage.
- **Units:** Residuals are decimal yields internally and displayed in bp through
  multiplication by 10,000.
- **Inputs:** Dated observed yields, fitted model values, fit diagnostics, and
  matching maturities.
- **Outputs:** Residual panel, cross-sectional error summaries, and provenance
  linking each residual to its fit.
- **Assumptions:** The interpretation depends on whether inputs are Treasury
  constant-maturity yields, par yields, or zero rates. Liquidity and measurement
  differences are not automatically removed.
- **Likely numerical issues:** Maturity misalignment, failed fits, outliers,
  rounding, stale observations, and accidental in-sample performance claims.
- **Tests:** Residual identity at every cell; exact zero on synthetic on-curve
  data; correct bp conversion; failed fits propagate explicit missing status;
  date and tenor alignment checks.

### S. Rolling z-scores

- **Objective:** Normalize a residual or spread relative to information available
  in a trailing historical window.
- **Mathematical definition:**
  \(z_t=(x_t-\mu_{t,w})/\sigma_{t,w}\), with a documented window, degrees of
  freedom, minimum observations, and whether the current observation enters the
  estimates. Trading signals will use only values available by decision time.
- **Units:** Input series may be decimal yield or bp; z-score is dimensionless.
- **Inputs:** Dated series, window length, minimum observations, `ddof`, lag
  convention, winsorization policy, and zero-volatility behavior.
- **Outputs:** Rolling mean, rolling standard deviation, z-score, valid-observation
  flag, and window endpoints.
- **Assumptions:** No implicit backfill. Expanding and rolling windows are
  distinct configurations.
- **Likely numerical issues:** Zero variance, insufficient history, missing
  dates, leakage from centered windows, and inconsistent `ddof`.
- **Tests:** Hand-computed windows; initial values remain unavailable until the
  minimum sample; constant series follows defined zero-volatility policy;
  modifying future data cannot alter earlier z-scores.

### T. 2Y/5Y/10Y butterfly analytics

- **Objective:** Track belly-versus-wings curvature and distinguish raw yield
  butterflies from risk-weighted trade structures.
- **Mathematical definition:** The default quoted yield butterfly is
  \(B_t=2y_{5,t}-y_{2,t}-y_{10,t}\). Alternative sign conventions or regression
  weights must be named. Trade weights are separately determined from DV01 or
  factor-neutral constraints.
- **Units:** Butterfly level is a decimal yield internally and displayed in bp;
  trade weights are notionals or dimensionless ratios; P&L is currency.
- **Inputs:** Aligned 2Y, 5Y, and 10Y observations, convention name, optional
  instrument DV01s, and hedge constraints.
- **Outputs:** Raw butterfly, leg contributions, rolling statistics, and optional
  risk-weighted weights.
- **Assumptions:** Constant-maturity series are curve observations rather than
  directly tradable bonds. A backtest must map signals to specified tradable
  proxies or clearly remain a yield-change study.
- **Likely numerical issues:** Sign convention confusion, missing tenor dates,
  roll and maturity mismatch, and assuming `[-1, 2, -1]` notionals are
  DV01-neutral.
- **Tests:** Hand calculations and sign checks; common parallel yield moves
  cancel in the raw butterfly; units convert exactly to bp; DV01-weighted
  structures satisfy their stated exposure equations.

### U. DV01-neutral hedge construction

- **Objective:** Choose hedge notionals that offset a target instrument's
  first-order parallel-rate exposure and, where requested, additional curve
  exposures.
- **Mathematical definition:** Solve \(Aw=b\), where columns contain hedge-leg
  DV01 or factor exposures per unit notional, \(w\) contains signed notionals,
  and \(b\) is the negative target exposure. Use direct solves when determined
  and documented least-squares or constrained solutions otherwise.
- **Units:** DV01 entries are currency per bp per unit notional; notionals are
  currency face amounts or normalized ratios; residual exposures are currency
  per bp.
- **Inputs:** Target exposures, hedge-instrument exposures, constraint set,
  normalization, optional bounds, and conditioning tolerance.
- **Outputs:** Signed hedge notionals, achieved exposures, residual exposures,
  matrix rank, condition number, and solver diagnostics.
- **Assumptions:** Sensitivities are local and evaluated on one valuation date.
  Neutrality does not remove convexity, carry, roll, liquidity, or basis risk.
- **Likely numerical issues:** Singular or ill-conditioned exposure matrices,
  excessive notionals, inconsistent sign conventions, and unit mismatches.
- **Tests:** Substitution verifies constraints; scale invariance; known two-leg
  examples; rank-deficient inputs produce diagnostics; independently bumped
  portfolio has approximately zero targeted first-order exposure.

### V. Historical relative-value backtest

- **Objective:** Evaluate a precisely specified, reproducible historical strategy
  based on curve residuals or butterflies without look-ahead bias.
- **Mathematical definition:** At decision time \(t\), estimate parameters only
  from data available by \(t\), form signal \(s_t\), set position for the next
  tradable interval, and calculate
  \(PnL_{t+1}=w_t^\top\Delta V_{t+1}+carry_{t+1}-cost_{t+1}\). If only yield
  changes are available, label the result a sensitivity-based P&L approximation.
- **Units:** Signals may be dimensionless z-scores; positions are notionals or
  risk units; gross and net P&L are currency; returns require a defined capital
  denominator.
- **Inputs:** Point-in-time data, estimation windows, entry/exit rules, position
  lag, instruments or proxies, pricing method, sizing rule, costs, and start/end
  dates.
- **Outputs:** Dated signals, positions, leg values, gross P&L, costs, net P&L,
  turnover, exposure diagnostics, drawdowns, and summary statistics.
- **Assumptions:** Signal observation, trade execution, and P&L intervals are
  distinct and explicit. Missing data cannot be filled with future information.
  Re-estimation frequency and universe selection are recorded.
- **Likely numerical issues:** Off-by-one lags, survivorship or revision bias,
  overlapping signals, false annualization, stale prices, unstable hedge
  weights, and mixing yield P&L with total returns.
- **Tests:** Future-data mutation leaves past positions unchanged; toy paths
  produce exact expected signals and P&L; positions are lagged; zero position has
  zero P&L; gross-minus-cost equals net; exposure and accounting identities hold.

### W. Transaction-cost sensitivity

- **Objective:** Show how strategy conclusions change across transparent trading
  cost assumptions.
- **Mathematical definition:** For position change \(\Delta w_{i,t}\), cost may
  be \(\sum_i|\Delta w_{i,t}|c_i\) in currency-per-notional terms or half-spread
  times traded risk, depending on the declared model. Run a grid of cost
  multipliers and recompute net results.
- **Units:** Costs are explicitly currency, price points, bp of yield, or bp of
  notional according to the model; aggregate cost and net P&L are currency.
- **Inputs:** Trades or turnover, leg-specific cost assumptions, bid/ask or
  slippage model, scenario grid, and rebalance schedule.
- **Outputs:** Per-trade costs, cumulative costs, net P&L paths, and performance
  metrics by cost scenario.
- **Assumptions:** The simple model is an approximation and does not claim to
  model market impact or executable historical liquidity unless supported by
  data.
- **Likely numerical issues:** Charging costs on held rather than traded
  positions, double-counting entry/exit, wrong notional basis, negative costs,
  and inconsistent annualization.
- **Tests:** Zero-cost net equals gross; costs are non-negative; doubling a linear
  cost rate doubles costs; unchanged positions incur zero turnover cost; manual
  multi-leg trade accounting.

### X. CLI and generated output

- **Objective:** Provide one discoverable root entry point for reproducible
  analytics while keeping financial logic in importable modules.
- **Mathematical definition:** No new financial formula is introduced. Each CLI
  workflow maps validated arguments to package functions and records all
  conventions needed to reproduce results.
- **Units:** Rate and shock inputs must declare decimal, percent, or bp at the
  boundary and be converted once to internal decimals. Output columns include
  units in names or metadata.
- **Inputs:** Subcommand, data mode, dates, instrument parameters, curve and risk
  conventions, deterministic seed where relevant, and output path below
  `outputs/`.
- **Outputs:** Human-readable summaries plus machine-readable CSV/JSON data and
  figures, stored in deterministic run directories or explicitly named files.
- **Assumptions:** `demo --offline` uses bundled data and no network. Commands
  return nonzero status on validation or calculation failure and never silently
  substitute a different analytical method.
- **Likely numerical issues:** CLI strings parsed in the wrong units, locale date
  ambiguity, output overwrite, non-deterministic ordering, and incomplete
  provenance.
- **Tests:** Parser smoke tests for `demo`, `curve`, `bond`, `risk`, `pca`,
  `relative-value`, and `backtest`; offline end-to-end test; invalid-unit tests;
  deterministic artifact names and content; all generated files stay beneath
  `outputs/`.

## 4. Cross-cutting validation

Financial identities will be tested independently whenever practical. Analytic
duration, convexity, and DV01 will be compared with repriced finite differences.
Curve transformations will be checked through round trips. Portfolio totals
will be reconciled to leg totals. Historical tests will use synthetic or bundled
fixtures and will deliberately modify future data to detect leakage.

Property-style parameter grids can be implemented with ordinary parametrized
pytest tests, avoiding an additional runtime dependency. Random synthetic data
will use a recorded deterministic seed. Numerical tolerances will be selected
from the scale and conditioning of each calculation rather than copied across
unrelated tests.

## 5. Delivery sequence

1. Implement date conventions, schedules, fixed cash flows, accrued interest,
   YTM pricing, solving, and instrument risk.
2. Add typed curve representations, interpolation, spot discounting, and
   finite-difference curve risk.
3. Add offline data and an online adapter with common validation and schemas.
4. Add NSS fitting, residual analytics, historical changes, and PCA.
5. Add butterfly and hedge construction with explicit exposure diagnostics.
6. Add the lagged relative-value backtest and cost scenarios.
7. Connect workflows to the CLI and generate documented artifacts.

Each phase must leave the repository importable, offline-testable, lint-clean,
and honest about which planned capabilities remain unimplemented.
