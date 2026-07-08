# Asset Universe — Anchored to Vanguard's 2026 VCMM Forecasts

**What's real vs. assumed (be ready to explain this distinction to judges):**
- **Expected returns** are anchored to Vanguard's own March 31, 2026 Capital
  Markets Model (VCMM) 10-year nominal return ranges, as reported publicly
  (Vanguard corporate site, Morningstar's 2026 capital-market-assumptions survey).
  We use the midpoint of each published range.
- **Volatilities and correlations** are NOT fully public at the sub-asset-class
  level, so these are our own reasonable assumptions, set to be broadly
  consistent with well-known historical asset-class risk characteristics.
  This is explicitly disclosed in our submission per the brief's transparency
  requirement ("clear notes on tools used, assumptions made").

## Asset universe (12 assets, 5 classes)

| ID | Asset | Class | Vanguard VCMM 10yr return range | Return anchor (midpoint) | Assumed volatility | Assumed liquidity tier |
|----|-------|-------|----------------------------------|---------------------------|----|----|
| A01 | US Large-Cap Value Equity | Equity | 5.8%–7.8% | 6.8% | 15% | High |
| A02 | US Large-Cap Growth Equity | Equity | 2.3%–4.3% | 3.3% | 19% | High |
| A03 | US Small-Cap Equity | Equity | 5.1%–7.1% | 6.1% | 22% | Medium |
| A04 | Non-US Developed Equity (unhedged) | Equity | ~6.0%–7.5%* | 6.75% | 17% | High |
| A05 | Emerging Markets Equity (unhedged) | Equity | ~5.0%–7.0%* | 6.0% | 23% | Medium |
| A06 | US Aggregate Bonds | Fixed Income | 3.8%–4.8% | 4.3% | 5% | High |
| A07 | US Long-Term Treasuries | Fixed Income | ~3.5%–4.5%* | 4.0% | 11% | High |
| A08 | US High-Yield Bonds | Fixed Income | 4.3%–5.3% | 4.8% | 9% | Medium |
| A09 | EM Sovereign Bonds (hedged) | Fixed Income | 5.1%–6.1% | 5.6% | 10% | Low |
| A10 | Broad Commodities | Commodities | ~4.0%–5.5%* | 4.75% | 18% | Medium |
| A11 | REITs (US) | Alternatives | ~5.5%–7.0%* | 6.25% | 20% | Medium |
| A12 | Cash / Short-Term Reserves | Cash | ~3.0%–4.0%* | 3.5% | 1% | High |

`*` = not individually confirmed in the sources pulled for this project; interpolated
from Vanguard's general 2026 outlook commentary (e.g., favoring high-quality fixed
income, value equities, and non-US developed markets as strongest risk-return
profiles over 5–10 years) and cross-checked against typical industry ranges
(Morningstar 2026 capital-market-assumptions survey). Treat these as reasonable
placeholders, not verified VCMM outputs — flag this explicitly in your write-up.

## Sector/class groupings (for constraint testing)
- **Equity** (A01–A05): typical guardrail, e.g. max 60% combined
- **Fixed Income** (A06–A09): typical guardrail, e.g. min 25% combined
- **Commodities + Alternatives** (A10–A11): typical guardrail, e.g. max 20% combined
- **Cash** (A12): typical guardrail, e.g. min 2%, max 15%

## Correlation assumptions (documented, not fitted to real data)
- Within-class correlations: high (0.6–0.85) for equities, moderate (0.4–0.6) for
  fixed income
- Cross-class equity/bond correlation: low to slightly negative (-0.1 to 0.2),
  consistent with typical diversification benefit
- Commodities: low correlation to both (0.0–0.2)
- Cash: near-zero correlation to everything

These feed directly into the covariance matrix used for Σ in the Markowitz/QUBO
objective — see `generate_synthetic_data.py`.
