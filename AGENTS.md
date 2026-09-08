# Repository Instructions

These instructions apply to all future work in this repository.

1. Implement important fixed-income mathematics explicitly rather than delegating it to opaque finance libraries.
2. All yields and rates must be represented internally as decimal values. Example: 4.25% = 0.0425.
3. Basis-point conversions must always be explicit: 1 bp = 0.0001 in decimal yield terms.
4. Every public function must document units.
5. Never silently mix clean price, dirty price and accrued interest.
6. Never silently mix YTM-based bond pricing with spot-curve discounting.
7. Constant-maturity Treasury yields must not be presented as if they are automatically a fully bootstrapped zero-coupon curve.
8. Financial conventions, assumptions and approximations must be documented.
9. Tests must work without internet access.
10. Online market-data functionality must have an offline sample-data fallback.
11. Numerical risk measures should be checked independently using finite differences where possible.
12. Historical analysis must not contain look-ahead bias.
13. Signals, position sizing and P&L timing must be explicit.
14. Set deterministic random seeds wherever randomness is used.
15. Generated program output belongs in `outputs/`.
16. Do not commit virtual environments, caches, API keys or secrets.
17. Do not run `git commit`, `git push`, `git reset --hard` or destructive Git commands. The repository owner controls version history.
18. Prefer readable mathematical code over clever abstraction.
19. Public functions require useful type hints and docstrings.
20. README claims must remain reproducible and must not exaggerate realism.
21. Do not describe the system as production-grade, institutional-grade, profitable or arbitrage-free unless those statements are genuinely established.
22. Before completing implementation work, run the relevant tests and Ruff.
