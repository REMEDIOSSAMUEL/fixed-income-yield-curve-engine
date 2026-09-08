# Bundled Treasury yield data

`sample_treasury_yields.csv` is a reproducible offline snapshot of daily U.S.
Treasury constant-maturity series from the Federal Reserve H.15 release,
retrieved through FRED on 8 September 2026. It contains 124 complete business-
day rows from 2 January through 28 June 2024.

The CSV stores the source's published **percent per annum** values. The data
loader divides them by 100 once so all in-memory rates are decimal annual rates.
The included FRED series are:

| Label | Years | FRED series |
|---|---:|---|
| 3M | 0.25 | DGS3MO |
| 6M | 0.50 | DGS6MO |
| 1Y | 1.00 | DGS1 |
| 2Y | 2.00 | DGS2 |
| 3Y | 3.00 | DGS3 |
| 5Y | 5.00 | DGS5 |
| 7Y | 7.00 | DGS7 |
| 10Y | 10.00 | DGS10 |
| 20Y | 20.00 | DGS20 |
| 30Y | 30.00 | DGS30 |

Source: Board of Governors of the Federal Reserve System, H.15 Selected
Interest Rates, distributed through the Federal Reserve Bank of St. Louis:
<https://fred.stlouisfed.org/release?rid=18>.

These are constant-maturity yield observations. They are par-yield-like
statistical series and are not a bootstrapped zero-coupon curve. NSS fits to
these values are fitted CMT curves, not exact discount curves.
