"""
Synthetic Asset Data Generator — Vanguard x WISER Quantum Challenge 2026
Multi-Asset Portfolio Construction track

Generates:
  - expected_returns.csv   : mu vector, anchored to real Vanguard VCMM 2026 forecasts
  - covariance_matrix.csv  : Sigma, built from assumed vol + correlation structure
  - asset_metadata.csv     : sector/class tags, liquidity tier, transaction cost bps
  - synthetic_returns_history.csv : simulated daily return history for backtesting /
                                     bootstrap risk estimation (NOT real market data)

All outputs are synthetic / for research use only, consistent with the challenge's
data privacy rules (no real Vanguard operational data used or published).
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# ---------------------------------------------------------------------------
# 1. Asset universe: real Vanguard VCMM return anchors (see asset_universe.md)
# ---------------------------------------------------------------------------
assets = [
    # id,   name,                                  class,          mu_annual, vol_annual, liquidity, txn_cost_bps
    ("A01", "US Large-Cap Value Equity",            "Equity",       0.068, 0.15, "High",   5),
    ("A02", "US Large-Cap Growth Equity",           "Equity",       0.033, 0.19, "High",   5),
    ("A03", "US Small-Cap Equity",                  "Equity",       0.061, 0.22, "Medium",10),
    ("A04", "Non-US Developed Equity (unhedged)",   "Equity",       0.0675,0.17, "High",   8),
    ("A05", "Emerging Markets Equity (unhedged)",   "Equity",       0.060, 0.23, "Medium",15),
    ("A06", "US Aggregate Bonds",                   "FixedIncome",  0.043, 0.05, "High",   3),
    ("A07", "US Long-Term Treasuries",              "FixedIncome",  0.040, 0.11, "High",   3),
    ("A08", "US High-Yield Bonds",                  "FixedIncome",  0.048, 0.09, "Medium", 8),
    ("A09", "EM Sovereign Bonds (hedged)",          "FixedIncome",  0.056, 0.10, "Low",   12),
    ("A10", "Broad Commodities",                    "Commodities",  0.0475,0.18, "Medium",10),
    ("A11", "REITs (US)",                           "Alternatives", 0.0625,0.20, "Medium", 8),
    ("A12", "Cash / Short-Term Reserves",           "Cash",         0.035, 0.01, "High",   1),
]

ids        = [a[0] for a in assets]
names      = [a[1] for a in assets]
classes    = [a[2] for a in assets]
mu         = np.array([a[3] for a in assets])
vol        = np.array([a[4] for a in assets])
liquidity  = [a[5] for a in assets]
txn_cost   = [a[6] for a in assets]
n = len(assets)

# ---------------------------------------------------------------------------
# 2. Build correlation matrix from documented assumptions (asset_universe.md)
# ---------------------------------------------------------------------------
corr = np.eye(n)


def set_corr(i, j, val):
    corr[i, j] = val
    corr[j, i] = val


class_idx = {c: [i for i, cc in enumerate(classes) if cc == c] for c in set(classes)}

# within-equity: high correlation
for i in class_idx["Equity"]:
    for j in class_idx["Equity"]:
        if i < j:
            set_corr(i, j, rng.uniform(0.60, 0.85))

# within-fixed-income: moderate correlation
for i in class_idx["FixedIncome"]:
    for j in class_idx["FixedIncome"]:
        if i < j:
            set_corr(i, j, rng.uniform(0.40, 0.60))

# equity vs fixed income: low / slightly negative
for i in class_idx["Equity"]:
    for j in class_idx["FixedIncome"]:
        set_corr(i, j, rng.uniform(-0.10, 0.20))

# commodities vs everything: low correlation
for i in class_idx["Commodities"]:
    for j in range(n):
        if i != j and j not in class_idx["Commodities"]:
            set_corr(i, j, rng.uniform(0.0, 0.20))

# alternatives (REITs) behave partly like equity, partly independent
for i in class_idx["Alternatives"]:
    for j in class_idx["Equity"]:
        set_corr(i, j, rng.uniform(0.30, 0.55))
    for j in class_idx["FixedIncome"]:
        set_corr(i, j, rng.uniform(-0.05, 0.15))

# cash: near-zero correlation to everything
for i in class_idx["Cash"]:
    for j in range(n):
        if i != j:
            set_corr(i, j, rng.uniform(-0.02, 0.02))

np.fill_diagonal(corr, 1.0)

# ---------------------------------------------------------------------------
# 3. Ensure positive semi-definite (required for a valid covariance matrix)
# ---------------------------------------------------------------------------
def nearest_psd(matrix):
    eigval, eigvec = np.linalg.eigh(matrix)
    eigval_clipped = np.clip(eigval, 1e-6, None)
    return eigvec @ np.diag(eigval_clipped) @ eigvec.T


corr_psd = nearest_psd(corr)
# re-normalize diagonal back to exactly 1.0 after PSD projection
d = np.sqrt(np.diag(corr_psd))
corr_psd = corr_psd / np.outer(d, d)
np.fill_diagonal(corr_psd, 1.0)

# Covariance = D * Corr * D, where D = diag(volatilities)
D = np.diag(vol)
cov = D @ corr_psd @ D

# ---------------------------------------------------------------------------
# 4. Simulate synthetic daily return history (for bootstrap risk validation)
# ---------------------------------------------------------------------------
N_DAYS = 750  # ~3 trading years
daily_mu = mu / 252
daily_cov = cov / 252

synthetic_returns = rng.multivariate_normal(daily_mu, daily_cov, size=N_DAYS)
dates = pd.bdate_range(end="2026-07-03", periods=N_DAYS)

# ---------------------------------------------------------------------------
# 5. Save all outputs
# ---------------------------------------------------------------------------
pd.DataFrame({"asset_id": ids, "name": names, "expected_return_annual": mu}).to_csv(
    "expected_returns.csv", index=False
)

pd.DataFrame(cov, index=ids, columns=ids).to_csv("covariance_matrix.csv")

pd.DataFrame(
    {
        "asset_id": ids,
        "name": names,
        "asset_class": classes,
        "volatility_annual": vol,
        "liquidity_tier": liquidity,
        "transaction_cost_bps": txn_cost,
    }
).to_csv("asset_metadata.csv", index=False)

hist_df = pd.DataFrame(synthetic_returns, columns=ids, index=dates)
hist_df.to_csv("synthetic_returns_history.csv")

print("Generated files:")
print(" - expected_returns.csv")
print(" - covariance_matrix.csv")
print(" - asset_metadata.csv")
print(" - synthetic_returns_history.csv")
print()
print("Sanity checks:")
print(f" - Covariance matrix is symmetric: {np.allclose(cov, cov.T)}")
eigvals = np.linalg.eigvalsh(cov)
print(f" - Covariance matrix is PSD (min eigenvalue >= 0): {eigvals.min():.6f}")
print(f" - Simulated {N_DAYS} days across {n} assets")
print(f" - Annualized mean of simulated returns (sample vs target):")
sample_mu_annual = hist_df.mean().values * 252
for i, aid in enumerate(ids):
    print(f"     {aid}: target={mu[i]:.4f}  sampled={sample_mu_annual[i]:.4f}")
