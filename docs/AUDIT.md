# Independent repository audit

Audit date: 9 September 2026. Scope: AGENTS.md, PROJECT_SPEC.md, README.md,
pyproject.toml, every package/CLI Python source and test, the full 124-row sample,
data documentation, CI, ignore rules, and every artifact-writing path. The
initial worktree was clean. No commits, pushes, resets or history changes were
performed. This report describes a code and methodology audit, not a
certification of market realism or trading performance.

## Important findings and corrections

| Area | Finding | Correction and evidence |
|---|---|---|
| Data integrity | Decimal CSV parsing coerced malformed strings into missing values; drop/keep could conceal corruption. Validation after dropping could also conceal bad dates. | Validate numeric cells, chronology and units before the missing policy; regressions cover every policy. |
| Dates | Normalizing two intraday timestamps could produce duplicated daily dates after uniqueness validation. | Daily inputs require timezone-naive midnight dates. |
| Data windows | The online adapter applied missing-value policy before the requested date range, so unrelated old missing tenors could invalidate a clean window. Empty payloads missed the download-failure path. | Bound the request and filter before missing handling; record missing dates; classify empty downloads for the fallback. |
| Discounting | D(0) and immediate cash flows failed the default extrapolation check. A one-node curve could be constructed but not used. | D(0)=1 with a documented zero-time reporting rate; allow exact single-node evaluation and explicit flat extension. Validate zero-curve compounding bases at construction. |
| Curve scenarios | Shocks sampled only at existing nodes distorted a steepener/flattener when an anchor or pivot was absent. | Insert in-range scenario anchors before bumping. A two-node base curve now reproduces a 2Y/10Y/30Y shock exactly between its nodes. Include shaped scenarios in CLI output. |
| Near-coupon workflow | The risk demo failed when a payment preceded the first 3M CMT node, which the bundled June 28 example did not expose. | The illustrative workflow now explicitly selects and reports flat endpoint extrapolation. The generic API keeps its conservative rejection default. A next-day coupon regression reproduced the failure before correction. |
| YTM solving | Trial prices could overflow while bracketing an otherwise valid root for a very long bond. Underflow could return a zero price for positive cash flows. | Bracket with an explicit log-sum-exp price residual, cache cash flows during inversion, check the final currency residual, reject non-positive/non-finite prices. Test a 1,000-year par identity and independent single-payment inversions. |
| NSS scaling | A common rescaling of least-squares weights could affect numerical stopping. Tight valid bounds could invert the clipped start interval. | Normalize relative weights, optimize residuals in bp, retain objective/RMSE in documented original units, and clip starts relative to the bounds' width. |
| NSS identification | Convergence/RMSE did not expose weak tau/beta identification. Historical calibration used only the preceding successful fit as its start. | Independent deterministic multistart fits on every date, explicit analytical Jacobian checked with central differences, loading/Jacobian conditioning and active-bound diagnostics. Negligible derivative columns remain unidentified before column normalization. No global-optimum guarantee is made. |
| Calibration failures | Historical nan-policy failures had no dated explanation in generated reports; latest-curve workflows did not enforce convergence. | Export dated diagnostics and failed-fit messages; require successful latest fits; propagate failure dates and data provenance into metadata. |
| PCA scaling | An absolute variance floor based on a unit-sized matrix rejected valid small decimal changes. | Use a variance-relative numerical tolerance and verify quadratic eigenvalue scaling. One/two-tenor plotting also works without forcing three-factor interpretation. |
| Mean-reversion exits | abs(z)<=exit_z missed paths jumping entirely across the exit band, leaving the old trade open until an opposite entry threshold was reached. | Directional exits: long closes at z<=exit_z, short at z>=-exit_z; opposite entry crossings still reverse. This is an explicit strategy-rule correction and changes some historical trades. |
| Signal interpretation | Positive z was described as automatically cheap to the current fit. A residual above its trailing mean can still be negative. | Distinguish raw observed-minus-fitted rich/cheap labels from historical z-score direction in code, methodology and README. |
| Proxy risk | Two separate implementations disagreed on negative yields and February schedule handling. The historical variant still failed for some leap-year accrual origins. | Reuse one proxy-DV01 implementation and anchor accrual start to the maturity roll. Test February 2024/2025/2028/2029 and negative yields. |
| P&L validation | Blanket filling of missing leg P&L and permissive performance inputs could hide invalid calculations; metric drawdown trusted supplied columns. | Initialize only the no-prior-interval row, reject invalid sizing/P&L, require finite metric inputs and gross-cost=net, and independently reconstruct drawdown from cumulative P&L. |
| Packaging | The offline CSV lived outside the installed package and was omitted from wheels. | Move the unchanged CSV to fixed_income/sample_data and declare package data. Build/extract an isolated wheel and load all 124 x 10 values without the original repository data directory. |
| Verification | Existing tests passed despite the above issues; several risk tests repeated the implementation; CLI subprocess execution was not explicitly included in coverage. | Add analytical, derivative, SVD, invariance and exact-ledger tests; enable subprocess coverage; block network sockets in unit tests. CLI tests explicitly use --offline and have a bounded 300-second allowance for multistart fitting under coverage. Numerical tolerances were not relaxed. |

The first eight targeted regression cases were run against the original code
and all failed, while the original suite passed 131 tests. Subsequent additions
also check identities that the original implementation already satisfied; a
passing identity is evidence, not automatically a defect discovery.

## Financial mathematics assessment

The ordinary bond formulas were sound under their stated generic convention.
For face F, annual decimal coupon c and frequency m, regular coupon currency
is F*c/m. Remaining fractional period counts n give price
sum(CF*(1+y/m)^(-n)). Times in Macaulay duration are n/m; modified duration is
Macaulay/(1+y/m). Convexity differentiates the fractional exponent, giving the
n*(n+1)/m^2 coefficient. Positive long DV01 is -dP/dy*0.0001 currency/bp.
Independent central differences now check annual/semiannual bonds at coupon and
fractional settlements under negative, zero and positive yields, as well as
notional invariance of duration and linear scaling of currency DV01.

No ordinary percentage/decimal, coupon/payment, annualization, DV01 factor-of-100
or factor-of-10,000, sign, or convexity scaling error was found. Clean plus
accrued equals dirty, and settlement-date payments are excluded. Root solving
requires an explicit price basis and a positive periodic base y>-m. Accrual and
coupon schedules retain leap-year/month-end behavior; stubs and unsupported
frequencies are rejected.

Spot pricing remains a separate API with explicit zero-curve tagging and
compounding. Continuous, simple and periodic discount factors pass independent
algebraic inverse checks. A flat periodic spot curve agrees with YTM pricing
only with matching time and compounding conventions. Discount times are the
bond's coupon-period years, not an implied ACT/365 curve convention.

NSS loading formulas agree with numerical integrals of the defining exponential
terms, including the zero limit; all six analytical parameter derivatives agree
with central finite differences. Maturities/taus are years and betas are decimal
annual yields. Unweighted/weighted RMSE is decimal yield; conversion to bp
divides by 0.0001. The optimizer works in scaled bp residuals, while the reported
objective is one half the original weighted sum of squared decimal residuals.
Bounds remain finite and positive for taus; no parameter economic interpretation
is warranted solely from a successful status.

Positive shocks raise rates; long price moves inversely for a parallel move.
The default steepener lowers 2Y by 25 bp, leaves 10Y unchanged, and raises 30Y by
25 bp; the flattener reverses these signs. Interior key bumps are linear hats
with outer shoulders. They partition unity, including beyond the outer keys.
Key curves are augmented at missing key nodes and DV01 is normalized to one bp
even when the numerical bump differs. A 7.5Y zero-coupon example independently
checks half-risk at 5Y/10Y using the exponential's exact hyperbolic-sine response.
Summed key DV01 equals parallel DV01 only in the first-derivative limit; finite
central-bump discrepancies scale quadratically with bump size, which is tested.

## PCA assessment

PCA defaults to adjacent-row decimal yield changes, not levels, and tenors are
sorted numerically (months divided by 12). Covariance uses n-1. Correlation mode
explicitly standardizes each tenor with sample volatility, producing
dimensionless scores/eigenvalues. Covariance scores are decimal changes and its
eigenvalues are squared decimal changes. Eigenpairs are descending; explained
variance divides each eigenvalue by the total. Direct SVD verifies eigenvalues,
score variance and retained-component reconstruction error independently.

Sign normalization is applied before score projection, preserving reconstruction.
Shape diagnostics use rank-spaced orthonormal templates and may withhold labels;
permuting component order does not force PC1/PC2/PC3 to mean level/slope/curvature.
Missing yield cells invalidate both adjacent changes. Workflows now retain
missing level rows for PCA instead of first dropping them and bridging gaps.
The API fits the supplied sample: it does not implement an automatic rolling
out-of-sample PCA engine, and historical descriptive PCA is not a trading signal
in this backtest.

## Relative value and backtest assessment

Residuals consistently equal observed minus fitted: positive is cheap in yield
space, negative rich. Raw butterfly is 2*y5-y2-y10; a parallel move cancels.
Z-score history excludes the current row and uses explicit lookback,
min-observations and ddof. Constant history has an explicit nan/zero/raise policy
and no future filling. Tests check known sample/population variances, window
endpoints, future-data mutation and missing signals.

Hedge notionals use positive actual unit DV01s: each wing offsets half the signed
belly DV01. This is parallel DV01 neutrality, not cash/notional neutrality, slope
neutrality, or a promise that the 5Y NSS residual is exactly hedged. Independent
repricing of an actual three-bond portfolio confirms local first-order neutrality.

The dated ledger now has this explicit order:

1. At t, apply yield changes from t-1 to t to positions and unit DV01 set at t-1.
2. Observe the t residual; normalize with history through t-1 and decide target.
3. Resize wing notionals using t unit DV01; charge costs on absolute target-minus-
   prior-target face. A reversal pays both closing and opening turnover.
4. Hold the new target for t to t+1. On the final date, earn the old interval's
   P&L before liquidation and its cost; do not open a terminal trade by default.

The initial lag and long/short P&L signs were already correct. They were not
rewritten as a supposed leakage fix. New tests establish exact entry, reversal,
rebalance, missing-signal close and terminal-liquidation currency ledgers;
current-date sensitivity changes cannot reprice the previous holding interval.
Independent date-by-date NSS fits and exact prefix comparisons exclude future
calibration rows. The final-date liquidation is an explicit known-horizon
policy, so prefix comparisons at a truncated terminal date require liquidation
to be disabled or the terminal decision to be excluded.

Gross/net totals, transaction costs, turnover, initial-zero high-water mark,
negative drawdowns, sample P&L volatility, sqrt(252) P&L ratio annualization,
active-interval hit rate and zero-volatility undefined ratios were inspected.
No percentage return or capital denominator is fabricated. Cost sensitivity
holds signals/positions/gross P&L fixed. Thresholds and lookback are not optimized
against this sample's P&L. Net active-interval hit rate remains a dated metric,
not a per-trade win rate: same-date costs can belong to the next holding interval.

## Software and test assessment

NumPy, pandas, SciPy and Matplotlib are all used; no opaque finance dependency
was introduced. No circular imports, secret credentials, mutable random-number
state, nondeterministic random tests, or dangerous Git operations were found.
The duplicated proxy-bond construction was removed. The longest
accounting/report functions remain readable straight-line calculations; a broad
API rewrite was not needed for this audit. Compatibility aliases delegate to
one implementation rather than duplicate mathematics.

Computational public APIs carry type hints and financial units in docstrings.
Numerical/schema failures propagate or enter an explicit reported policy.
Calculation APIs remain separate from artifact writing. Paths are based on
pathlib and generated charts, tables, coverage data, metadata and audit packaging
scratch work stay under outputs/. Python caches and pytest temporary files are
development artifacts; compileall was expressly requested. Ignore rules exclude
environments, caches, keys and generated outputs. The source sample was moved,
not modified; line-by-line comparison with its original tracked version agreed.

## Verification

All commands below exited with status zero on the final implementation.

| Command | Final result |
|---|---|
| `pytest -q` | 199 passed in 173.76 seconds |
| `pytest --cov=fixed_income` | 199 passed in 229.13 seconds; 84.94% statement coverage, including CLI subprocesses |
| `ruff check .` | All checks passed |
| `ruff format --check .` | All files formatted |
| `python fixed_income_engine.py demo --offline` | Completed; charts and reports generated under outputs/ |
| `python fixed_income_engine.py backtest --offline` | Completed; 124 NSS fits converged, zero failed fits |
| `python -m compileall fixed_income fixed_income_engine.py` | Completed without errors |

The suite increased from 131 to 199 tests (68 additional cases). Coverage
measured 2,523 executable statements, of which 2,143 were exercised and 380
were not. This is statement coverage, not exhaustive branch or scenario
coverage. Module coverage: bonds 87.72%, curves 84.57%, risk 85.14%, PCA 81.07%,
relative value 78.40%, backtest 86.03%, data 84.39%, plotting 74.04%, workflows
98.01%, package initialization 100%. Remaining unexecuted paths include error
guards, aliases and optional plotting paths; no 100% assurance is claimed.

Environment: Windows, Python 3.12.10, pytest 9.1.1, pytest-cov 7.1.0, coverage
7.16.0, NumPy 2.5.3, pandas 3.0.5, SciPy 1.18.1 and Matplotlib 3.11.1.
The configured Python 3.11/3.13 CI environments were inspected but not executed
in this Windows audit session. The wheel/sample loading check also passed
offline from an isolated extraction under outputs/.

The reproducible default backtest reports 4 entries, gross approximate P&L
1.26465 currency, cost 19.63538 currency, and net approximate P&L -18.37072
currency on 1,000,000 currency belly face normalization. These figures are
accounting outputs for this short sensitivity study, not returns or alpha.
The latest fit has 1.2943 bp RMSE but a scaled Jacobian condition number near
3.07e8, illustrating why fit error alone is not evidence of parameter stability.

The exact final command transcripts are saved in
`outputs/audit_verification.json`; source-data comparison results are in
`outputs/sample_audit.json`. Generated reports include calibration diagnostics,
dated P&L, performance metrics, transaction-cost sensitivity and metadata.

## Remaining methodological limits and portfolio-review suitability

Suitable for public portfolio review as a transparent educational fixed-income
research project once presented with these limits. It does not establish a
production system, executable strategy, arbitrage-free curve, or profitable edge.

- Treasury CMTs are statistical par-curve observations, not tradable security
  prices or automatically bootstrapped zeros. The curve-risk demo explicitly
  copies CMTs into a continuously compounded zero proxy as an illustration.
- Published yields and publication times do not supply executable prices after
  signal availability. Same-observation target execution remains hypothetical.
- The sample covers only 124 complete dates in one six-month 2024 regime and was
  retrieved later. Real-time vintages, revision history, liquidity selection,
  security survivorship and publication timestamps are unavailable. Full online
  history may introduce complete-case selection and unequal gaps.
- P&L is a first-order sensitivity approximation. Coupons/carry, aging/roll,
  convexity, changing securities, futures basis, funding, financing, tax, bid/ask
  and market impact are absent. Daily proxy hedge resizing is not a self-financing
  executable security portfolio. Currency normalization supplies no invested capital.
- Generic regular-bond date and YTM conventions are not the full set of market
  Treasury pricing rules. No holiday calendars, settlement lag, stubs, ex-coupon
  handling or final-period simple-yield market exception is implemented.
- Linear zero interpolation differs from the specification's initial log-discount
  preference. It need not preserve positive forwards or decreasing discount
  factors. Discount-factor-tagged objects are storage representations, not a
  separate discount-factor-interpolation/pricing engine.
- NSS has local minima and weak parameter identification. Multiple starts,
  explicit bounds and diagnostics reduce numerical risk but do not prove a global
  solution or justify trading extrapolated tails or interpreting each beta/tau.
- Annualization assumes daily observation intervals and includes warm-up zeros;
  serial correlation, overlapping risk, irregular gaps and selection uncertainty
  are not statistically corrected. No holdout evidence, parameter robustness
  study or evidence of alpha is claimed.
- The library's offline sample works from a wheel. Root-script report workflows
  remain designed for a repository checkout/editable install and its outputs tree.

Treasury's own documentation confirms that CMTs come from its par yield curve
and describes publication timing: [Treasury methodology](https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics/treasury-yield-curve-methodology),
[Treasury rate FAQs](https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics/interest-rates-frequently-asked-questions).
All 1,240 observations (124 dates x 10 tenors) were compared against the
[official 2024 table](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?field_tdr_date_value=2024&type=daily_treasury_yield_curve);
there were no missing dates or differing published percent values. This does
not verify the originally claimed retrieval timestamp or
point-in-time vintage. Packaging and subprocess coverage configuration follow
[setuptools data-file guidance](https://setuptools.pypa.io/en/latest/userguide/datafiles.html)
and [pytest-cov subprocess guidance](https://pytest-cov.readthedocs.io/en/latest/subprocess-support.html).
